"""レポート出力の本体（GUI 非依存。ui/report_worker.py と cli.py の両方から呼ぶ）。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..features import get_extractor
from ..features.adaface import apply_clahe
from .gallery import GalleryEntry, entry_seconds, load_entry, scan_gallery, scan_track_folder
from ..common.imageio import imread
from ..landmarks import get_aligner
from .merge import assign_groups, merge_embeddings
from ..common.phase_log import PhaseLogger
from ..pose import get_pose_estimator
from .cache import EmbeddingCache
from .gate import GateFilter, apply_gate_filter
from .quality import QualityScorer, QueryFilter, apply_query_filter
from ..features.base import WeightsUnavailable
from .excel import MODE_IMAGE, ReportConditions, build_report
from .similarity import aggregate_by_track, rank_pairs, similarity_matrix

CHUNK_SIZE = 256


def _pose_input(estimator, e: GalleryEntry) -> np.ndarray:
    """頭部姿勢モデルに渡す画像。余白の広い切り出しを好むモデル（6DRepNet）には、全身画像と顔枠から作り直した切り出しを渡す。"""
    if getattr(estimator, "prefers_loose_crop", False) and e.body_path is not None and e.person_bbox_xyxy and e.face_bbox_xyxy:
        from ..pose.sixdrepnet import loose_face_crop
        crop = loose_face_crop(imread(e.body_path), e.person_bbox_xyxy, e.face_bbox_xyxy)
        if crop is not None:
            return crop
    return _read(e)


def estimate_poses(estimator, entries: list[GalleryEntry]) -> None:
    """各エントリの顔クロップ(_face.jpg そのまま。モデルによっては余白広めの切り出し)から頭部姿勢を推定し、entry.pose に入れる。"""
    for i in range(0, len(entries), CHUNK_SIZE):
        chunk = entries[i:i + CHUNK_SIZE]
        angles = estimator.angles([_pose_input(estimator, e) for e in chunk])
        for e, a in zip(chunk, angles):
            e.pose = (float(a[0]), float(a[1]), float(a[2]))


@dataclass
class EmbedStats:
    align_failed: int = 0      # アラインに失敗しリサイズで代替した件数
    landmark_missing: int = 0  # ランドマークが得られずテンプレート座標で代替した件数（needs_landmarks のモデルのみ）


def _read(entry: GalleryEntry) -> np.ndarray:
    img = imread(entry.face_path)  # 日本語パス対応
    if img is None:
        img = np.full((112, 112, 3), 114, np.uint8)  # 読めない画像は無地で代替
    return img


def embed_entries(extractor, aligner, entries: list[GalleryEntry], progress=None,
                  cache: EmbeddingCache | None = None, clahe: bool = False) -> tuple[np.ndarray, EmbedStats]:
    """CHUNK_SIZE 枚ずつ読み込み、ランドマークでアラインしてから特徴量化する（メモリ抑制）。
    アラインに失敗した画像は単純リサイズで代替する。cache があれば、保存済みの画像は計算せずに再利用し、
    新しく計算した分をキャッシュへ入れる（save は呼び出し側）。結果はキャッシュの有無で変わらない。
    clahe=True: アライン後の顔画像に CLAHE（LAB の L チャネル, clip 2.0。premises-security の AdaFace 前処理と同じ）を
    掛けてから特徴量化する。cache は variant="clahe" のものを渡すこと（特徴量の値が変わる）。"""
    emb = np.zeros((len(entries), extractor.embed_dim), np.float32)
    stats = EmbedStats()
    size = extractor.input_size
    todo = []  # 計算が必要な (entries の添字)
    for i, e in enumerate(entries):
        hit = cache.get(e.face_path) if cache is not None else None
        if hit is None:
            todo.append(i)
        else:
            emb[i], failed, missing = hit[0], hit[1], hit[2]
            stats.align_failed += int(failed)
            stats.landmark_missing += int(missing and extractor.needs_landmarks)
    for s in range(0, len(todo), CHUNK_SIZE):
        idx = todo[s:s + CHUNK_SIZE]
        faces, landmarks, failed_flags = [], [], []
        for i in idx:
            img = _read(entries[i])
            res = aligner.align_with_landmarks(img, size)
            if res is None:
                stats.align_failed += 1
                face = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                landmarks.append(None)
                failed_flags.append(True)
            else:
                face = res.image
                landmarks.append(res.landmarks)
                failed_flags.append(False)
            faces.append(apply_clahe(face) if clahe else face)
        if extractor.needs_landmarks:
            stats.landmark_missing += sum(1 for lm in landmarks if lm is None)
            chunk = extractor.embed_batch(faces, landmarks=landmarks)
        else:
            chunk = extractor.embed_batch(faces)
        for j, i in enumerate(idx):
            emb[i] = chunk[j]
            if cache is not None:
                cache.put(entries[i].face_path, chunk[j], failed_flags[j], landmarks[j] is None)
        if progress:
            progress(min(s + CHUNK_SIZE, len(todo)), len(todo))
    return emb, stats


def _stats_note(stats: EmbedStats, needs_landmarks: bool) -> str:
    note = f"アライン失敗 {stats.align_failed} 件（リサイズで代替）"
    if needs_landmarks:
        note += f" / ランドマーク未取得 {stats.landmark_missing} 件（テンプレート座標で代替）"
    return note


def run_report(model_name: str, landmark_name: str, mode: str, query_path: Path, gallery_dir: Path,
               output_dir: Path, show_body: bool = True, limit: int = 500, targets=None,
               merge_sec: float = 0.0, pose_name: str = "なし", progress=lambda s: None,
               echo=print, use_cache: bool = True, include_excluded: bool = False,
               query_filter: QueryFilter | None = None, exclude_dirs=None, clahe: bool = False,
               gate_filter: GateFilter | None = None) -> tuple[Path, PhaseLogger]:
    """include_excluded=True: 追跡IDフォルダ内の old フォルダに移した画像も、クエリ・比較対象の両方に元の追跡IDの画像として含める。
    clahe=True: アライン後の顔画像に CLAHE を掛けてから特徴量化する（クエリ・比較対象とも。キャッシュは別ファイル）。
    gate_filter: 顔の採用基準（抽出時の顔ゲート・品質ガードの再判定）。クエリ・比較対象の両方に適用する。
    None は抽出時と同じ基準（抽出時に採用された画像だけを使う = 従来どおり）。
    query_filter: クエリ画像の自動選別（顔信頼度・品質ノルム・顔幅のしきい値）。比較対象には適用しない。
    exclude_dirs: クエリ自身に加えて比較対象から外す追跡IDフォルダ（クエリ画像を別フォルダに集めたときの元の追跡IDなど）。
    「条件」シートに記録され、「探したい人物」に指定されていれば検索結果シートに照合対象外と表示する。"""
    log = PhaseLogger(echo=lambda m: (echo(m), progress(m)))
    generated = datetime.now()
    with log.phase(f"初期化・モデルロード（特徴量: {model_name} / ランドマーク: {landmark_name} / 姿勢: {pose_name}"
                   f"{' / 前処理: CLAHE' if clahe else ''}）"):
        extractor = get_extractor(model_name)
        aligner = get_aligner(landmark_name)
        pose_estimator = get_pose_estimator(pose_name)
        for obj in (extractor, aligner, pose_estimator):
            if obj is not None and not obj.weights_ready():
                obj.download_weights(log=lambda m: (echo(m), progress(m)))
        extractor.load()
        aligner.load()
        if pose_estimator is not None:
            pose_estimator.load()
    with log.phase("クエリ読込・顔特徴量抽出") as rec:
        if mode == MODE_IMAGE:
            queries = [load_entry(Path(query_path))]
            exclude = dict(exclude_files={Path(query_path)})
        else:
            queries = scan_track_folder(Path(query_path), include_excluded=include_excluded)
            exclude = dict(exclude_dirs={Path(query_path)})
        extra_exclude = [Path(p) for p in (exclude_dirs or [])]
        if extra_exclude:
            exclude["exclude_dirs"] = set(exclude.get("exclude_dirs", set())) | set(extra_exclude)
        if not queries:
            raise ValueError("クエリの顔画像(_face.jpg)が見つかりません")
        if mode != MODE_IMAGE and merge_sec > 0:
            queries = sorted(queries, key=entry_seconds)  # 時刻の早い順
        gate_filter = gate_filter or GateFilter()
        n_gate_q = apply_gate_filter(queries, gate_filter)
        n_excluded_q = 0
        if query_filter is not None and query_filter.enabled():
            scorer = QualityScorer()
            norms = None
            cand = [i for i, q in enumerate(queries) if not q.invalid and not q.gate_excluded_reason]  # 使わない画像は計算しない
            if scorer.weights_ready():
                norms = [None] * len(queries)
                for i, v in zip(cand, scorer.norms([_read(queries[i]) for i in cand], aligner)):
                    norms[i] = v
            elif query_filter.min_norm > 0:
                raise WeightsUnavailable(scorer.manual_instructions())
            n_excluded_q = apply_query_filter(queries, query_filter, norms)
        used_queries = [q for q in queries if q.used]  # 無効・採用基準外・自動選別除外は使わない
        n_invalid_q = sum(1 for q in queries if q.invalid)
        if not used_queries:
            q0 = queries[0]
            detail = (f"（理由: {q0.invalid_reason or q0.gate_excluded_reason or q0.excluded_reason}）"
                      if mode == MODE_IMAGE else "")
            raise ValueError(f"クエリ画像が全て無効（または採用基準外・自動選別で除外）のため比較できません{detail}")
        q_emb, q_stats = embed_entries(extractor, aligner, used_queries, clahe=clahe)
        rec.note = (f"クエリ画像 {len(queries)} 件（うち無効 {n_invalid_q} 件、採用基準外 {n_gate_q} 件、"
                    f"自動選別で除外 {n_excluded_q} 件は特徴量比較に不使用）"
                    f" / {_stats_note(q_stats, extractor.needs_landmarks)}")
    group_ids = None
    if mode != MODE_IMAGE and merge_sec > 0:
        with log.phase(f"クエリ特徴量マージ（{merge_sec:g}秒単位）") as rec:
            used_gids = assign_groups([entry_seconds(q) for q in used_queries], merge_sec)
            q_emb = merge_embeddings(q_emb, used_gids)  # 行 = グループ(ID-1)
            it = iter(used_gids)
            group_ids = [next(it) if q.used else None for q in queries]  # 不使用画像はグループなし
            rec.note = f"有効画像 {len(used_queries)} 件 -> グループ {len(q_emb)} 件"
    with log.phase("比較対象スキャン") as rec:
        gallery_all = scan_gallery(Path(gallery_dir), include_excluded=include_excluded, **exclude)
        n_gate_g = apply_gate_filter(gallery_all, gate_filter)
        gallery = [g for g in gallery_all if g.used]  # 無効画像・採用基準外は比較対象から外す
        if not gallery:
            raise ValueError("比較対象フォルダに有効な顔画像(_face.jpg)が見つかりません（無効・採用基準外を除く）")
        rec.note = (f"画像 {len(gallery)} 件 / 追跡ID {len({g.track_key for g in gallery})} 件"
                    f" / 無効画像 {sum(1 for g in gallery_all if g.invalid)} 件・採用基準外 {n_gate_g} 件を除外")
    with log.phase("比較対象 顔特徴量抽出") as rec:
        cache = (EmbeddingCache(Path(gallery_dir), model_name, landmark_name, variant="clahe" if clahe else "")
                 if use_cache else None)
        if cache is not None:
            cache.load()
        g_emb, g_stats = embed_entries(extractor, aligner, gallery,
                                       lambda i, n: progress(f"比較対象の顔特徴量を計算中… ({i}/{n})"), cache,
                                       clahe=clahe)
        rec.note = f"{len(gallery)} 枚 / {_stats_note(g_stats, extractor.needs_landmarks)}"
        if cache is not None:
            cache.save()
            rec.note += f" / 特徴量キャッシュ: 再利用 {cache.hits} 件, 新規計算 {cache.misses} 件（{cache.path.name}）"
    with log.phase("類似度計算・順位付け") as rec:
        sim = similarity_matrix(q_emb, g_emb)
        ranked = rank_pairs(sim, limit)
        aggregated = aggregate_by_track(sim, [g.track_key for g in gallery], limit)
        rec.note = f"全ペア {sim.size} 件 / 出力 {len(ranked)} 件 / 追跡ID集約 {len(aggregated)} 件"
    if pose_estimator is not None:
        with log.phase(f"頭部姿勢推定（{pose_name}）") as rec:
            estimate_poses(pose_estimator, queries)
            estimate_poses(pose_estimator, gallery)
            rec.note = f"クエリ {len(queries)} 件 / 比較対象 {len(gallery)} 件（顔クロップから yaw/pitch/roll を推定）"
    cond = ReportConditions(model_name, landmark_name, mode, str(query_path), str(gallery_dir), str(output_dir),
                            show_body, generated, limit, targets, merge_sec, pose_name, include_excluded,
                            query_filter.describe() if query_filter is not None else "指定なし",
                            tuple(str(p) for p in extra_exclude), clahe, gate_filter.describe())
    out = Path(output_dir) / f"compared_face_{generated:%Y%m%d_%H%M%S}.xlsx"
    build_report(out, cond, queries, gallery, ranked, aggregated, log, group_ids, used_queries,
                 n_align_failed=q_stats.align_failed + g_stats.align_failed,
                 n_landmark_missing=(q_stats.landmark_missing + g_stats.landmark_missing
                                     if extractor.needs_landmarks else None),
                 n_gate_gallery=n_gate_g)
    return out, log
