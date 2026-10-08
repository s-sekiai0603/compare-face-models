from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core.features import WeightsUnavailable, get_extractor
from ..core.gallery import GalleryEntry, entry_seconds, load_entry, scan_gallery, scan_track_folder
from ..core.merge import assign_groups, merge_embeddings
from ..core.phase_log import PhaseLogger
from ..core.report_excel import MODE_IMAGE, ReportConditions, build_report
from ..core.similarity import aggregate_by_track, rank_pairs, similarity_matrix

CHUNK_SIZE = 256


def _read(entry: GalleryEntry) -> np.ndarray:
    img = cv2.imread(str(entry.embed_path))
    if img is None:
        img = np.full((256, 128, 3), 114, np.uint8)  # 読めない画像は無地で代替
    return img


def embed_entries(extractor, entries: list[GalleryEntry], progress=None) -> np.ndarray:
    """CHUNK_SIZE 枚ずつ読み込んで特徴量化（メモリ抑制）。"""
    out = []
    for i in range(0, len(entries), CHUNK_SIZE):
        out.append(extractor.embed_batch([_read(e) for e in entries[i:i + CHUNK_SIZE]]))
        if progress:
            progress(min(i + CHUNK_SIZE, len(entries)), len(entries))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def run_report(model_name: str, mode: str, query_path: Path, gallery_dir: Path, output_dir: Path,
               skeleton: bool, limit: int = 500, targets=None, merge_sec: float = 0.0, progress=lambda s: None, echo=print) -> tuple[Path, PhaseLogger]:
    log = PhaseLogger(echo=lambda m: (echo(m), progress(m)))
    generated = datetime.now()
    with log.phase(f"初期化・モデルロード（{model_name}）"):
        extractor = get_extractor(model_name)
        extractor.load()
    with log.phase("クエリ読込・特徴量抽出") as rec:
        if mode == MODE_IMAGE:
            queries = [load_entry(query_path)]
            exclude = dict(exclude_files={query_path})
        else:
            queries = scan_track_folder(query_path)
            exclude = dict(exclude_dirs={query_path})
        if not queries:
            raise ValueError("クエリの外見画像(_body.jpg)が見つかりません")
        if mode != MODE_IMAGE and merge_sec > 0:
            queries = sorted(queries, key=entry_seconds)  # 時刻の早い順
        used_queries = [q for q in queries if not q.invalid]  # 無効画像は比較・グループ化に使わない
        n_invalid_q = len(queries) - len(used_queries)
        if not used_queries:
            detail = f"（無効理由: {queries[0].invalid_reason}）" if mode == MODE_IMAGE else ""
            raise ValueError(f"クエリ画像が全て無効のため比較できません{detail}")
        q_emb = embed_entries(extractor, used_queries)
        rec.note = f"クエリ画像 {len(queries)} 件（うち無効 {n_invalid_q} 件は特徴量比較に不使用）"
    group_ids = None
    if mode != MODE_IMAGE and merge_sec > 0:
        with log.phase(f"クエリ特徴量マージ（{merge_sec:g}秒単位）") as rec:
            used_gids = assign_groups([entry_seconds(q) for q in used_queries], merge_sec)
            q_emb = merge_embeddings(q_emb, used_gids)  # 行 = グループ(ID-1)
            it = iter(used_gids)
            group_ids = [None if q.invalid else next(it) for q in queries]  # 無効画像はグループなし
            rec.note = f"有効画像 {len(used_queries)} 件 -> グループ {len(q_emb)} 件"
    with log.phase("比較対象スキャン") as rec:
        gallery_all = scan_gallery(gallery_dir, **exclude)
        gallery = [g for g in gallery_all if not g.invalid]  # 無効画像は比較対象から外す
        if not gallery:
            raise ValueError("比較対象フォルダに有効な外見画像(_body.jpg)が見つかりません")
        rec.note = (f"画像 {len(gallery)} 件 / 追跡ID {len({g.track_key for g in gallery})} 件"
                    f" / 無効画像 {len(gallery_all) - len(gallery)} 件を除外")
    with log.phase("比較対象 特徴量抽出") as rec:
        g_emb = embed_entries(extractor, gallery,
                              lambda i, n: progress(f"比較対象の特徴量を計算中… ({i}/{n})"))
        rec.note = f"{len(gallery)} 枚"
    with log.phase("類似度計算・順位付け") as rec:
        sim = similarity_matrix(q_emb, g_emb)
        ranked = rank_pairs(sim, limit)
        aggregated = aggregate_by_track(sim, [g.track_key for g in gallery], limit)
        rec.note = f"全ペア {sim.size} 件 / 出力 {len(ranked)} 件 / 追跡ID集約 {len(aggregated)} 件"
    cond = ReportConditions(model_name, mode, str(query_path), str(gallery_dir), str(output_dir),
                            skeleton, generated, limit, targets, merge_sec)
    out = output_dir / f"compared_feature_{generated:%Y%m%d_%H%M%S}.xlsx"
    build_report(out, cond, queries, gallery, ranked, aggregated, log, group_ids, used_queries)
    return out, log


class WeightsWorker(QThread):
    """画面起動時の重み自動取得。"""
    status = Signal(str)
    ready = Signal()
    unavailable = Signal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self) -> None:
        try:
            ex = get_extractor(self.model_name)
            if not ex.weights_ready():
                ex.download_weights(log=self.status.emit)
            self.ready.emit()
        except WeightsUnavailable as e:
            self.unavailable.emit(str(e))
        except Exception as e:
            self.unavailable.emit(f"重みの準備に失敗しました: {type(e).__name__}: {e}")


class ReportWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            out, _ = run_report(progress=self.progress.emit, **self.kwargs)
            self.finished_ok.emit(str(out))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
