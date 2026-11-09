r"""顔サイズ別精度検証（同一人物の画像同士の類似度レポート / 誤識別となる画像の枚数レポート）。

検証対象人物 = 追跡IDフォルダ（最大 10 件）。各フォルダ直下の顔画像（*_face.jpg）を、検索元セット検索と同じ処理
（アライン → 特徴量 → コサイン類似度）で扱い、顔 bbox のサイズ順に並べた Excel を出す（verify_face_size_excel.py）。
追跡IDフォルダの構成: <比較対象フォルダ>\<カメラID>\<映像ファイル名(拡張子なし)>\<追跡ID>（gallery.scan_track_folder と同じ）。

誤識別となる画像の枚数レポート: 対象者（検証対象人物の画像）の各画像 i を基準に、比較対象フォルダ全体の非対象者（指定した追跡IDフォルダに
入っていない画像）のうち、対象者の画像 j との類似度より i との類似度が高い画像の数を、i × j のセルに出す（対象者どうしは数えない）。
どの非対象者の画像が高かったかは、レポートと同名のテキストファイルに出す。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from ..common.phase_log import PhaseLogger
from ..features import get_extractor
from ..landmarks import get_aligner
from .cache import EmbeddingCache
from .gallery import GalleryEntry, scan_gallery, scan_track_folder
from .gate import GateFilter, apply_gate_filter
from .pipeline import _stats_note, embed_entries
from .similarity import similarity_matrix

OUTPUT_MODES = {
    "similarity": "同一人物の画像同士の類似度レポート出力",
    "misidentify": "誤識別となる画像の枚数レポート出力",
    "rank": "対象画像の類似度順位レポート出力",
    "both": "すべて",
}
ALIGN_SIZE_WIDE = 168  # 「168×168 アライメント」: アラインをこのサイズで行ってから特徴量モデルの入力サイズへ縮小する
ROUND_DIGITS = 4  # 類似度を小数 4 桁に丸めて比べる（検索元セット検索の順位と同じ。同じ値は「高い」に数えない）


def runs_similarity_report(mode: str) -> bool:
    """類似度レポートを出すモードか。"""
    return mode in ("similarity", "both")


def runs_misidentify_report(mode: str) -> bool:
    """誤識別となる画像の枚数レポートを出すモードか。"""
    return mode in ("misidentify", "both")


def runs_rank_report(mode: str) -> bool:
    """対象画像の類似度順位レポートを出すモードか。"""
    return mode in ("rank", "both")


def needs_gallery(mode: str) -> bool:
    """比較対象フォルダ（非対象者の特徴量）が要るモードか。"""
    return runs_misidentify_report(mode) or runs_rank_report(mode)


def report_filename(now: datetime) -> str:
    return f"顔サイズごとの精度検証_{now:%Y%m%d}_{now:%H%M%S}.xlsx"


def _natural(text: str) -> tuple[int, int, str]:
    """数字だけの追跡IDは数値順、それ以外は文字列順（数字が先）。"""
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def sort_entries(entries: list[GalleryEntry]) -> list[GalleryEntry]:
    """顔 bbox の高さ → 幅（昇順。値なしは末尾）→ カメラID → 映像ファイル名 → 追跡ID → 画像ファイル名。"""
    def key(e: GalleryEntry):
        h, w = e.face_height, e.face_width
        return ((h is None, h or 0.0), (w is None, w or 0.0), e.camera_id, e.video_file, _natural(e.track_id), e.face_path.name)

    return sorted(entries, key=key)


def sort_info_entries(entries: list[GalleryEntry]) -> list[GalleryEntry]:
    """対象画像情報シートの並び: カメラID → 追跡ID → 映像ファイル名 → 画像ファイル名（いずれも昇順。数字だけの追跡IDは数値順）。"""
    return sorted(entries, key=lambda e: (e.camera_id, _natural(e.track_id), e.video_file, e.face_path.name))


def collect_entries(track_dirs: list[Path | str]) -> list[GalleryEntry]:
    """追跡IDフォルダ直下の顔画像（old フォルダを除く）を集める。同じフォルダの重複指定は 1 回だけ数える。"""
    seen: set[Path] = set()
    out: list[GalleryEntry] = []
    for d in track_dirs:
        if not str(d).strip():
            continue
        p = Path(d)
        if not p.is_dir():
            raise ValueError(f"追跡IDフォルダが見つかりません: {p}")
        r = p.resolve()
        if r in seen:
            continue
        seen.add(r)
        out += scan_track_folder(p)
    return out


# ---- 誤識別となる画像の枚数 ----
def count_misidentified(target_emb: np.ndarray, nontarget_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """誤識別数 counts[i, j] = 非対象者のうち sim(i, 非対象者) > sim(i, j) の数（対角は 0）と、対象者どうしの類似度行列を返す。
    行 i を基準にするので counts は非対称。対象者どうしの類似度が高いものは数えない。特徴量は L2 正規化済み。"""
    t = np.asarray(target_emb, np.float64)
    g = np.asarray(nontarget_emb, np.float64)
    sim = t @ t.T
    counts = np.zeros(sim.shape, np.int64)
    if len(g) == 0:
        return counts, sim
    ng = g @ t.T                                   # (非対象者数, 対象者数)
    for i in range(len(t)):
        col = np.sort(np.round(ng[:, i], ROUND_DIGITS))
        th = np.round(sim[i], ROUND_DIGITS)
        counts[i] = len(col) - np.searchsorted(col, th, side="right")  # 丸め後に th より大きいものの数
        counts[i, i] = 0
    return counts, sim


def rank_matrix(target_emb: np.ndarray, nontarget_emb: np.ndarray, nontarget_track_keys: list) -> np.ndarray:
    """順位 ranks[i, j] = 1 + 「非対象者の追跡IDのうち、追跡ID内の最大類似度 sim(i, ·) が sim(i, j) より高いものの数」（対角は 0）。
    同じ追跡IDの画像が複数高くても 1 件。対象者どうしは数えない。行 i 基準なので非対称。特徴量は L2 正規化済み。"""
    t = np.asarray(target_emb, np.float64)
    g = np.asarray(nontarget_emb, np.float64)
    sim = t @ t.T
    ranks = np.ones(sim.shape, np.int64)
    if len(g):
        ids: dict = {}
        grp = np.array([ids.setdefault(k, len(ids)) for k in nontarget_track_keys])
        order = np.argsort(grp, kind="stable")
        starts = np.nonzero(np.r_[True, np.diff(grp[order]) != 0])[0]
        ng = np.round(g[order] @ t.T, ROUND_DIGITS)            # (非対象者数, 対象者数)
        tmax = np.maximum.reduceat(ng, starts, axis=0)         # (非対象者の追跡ID数, 対象者数)
        for i in range(len(t)):
            col = np.sort(tmax[:, i])
            th = np.round(sim[i], ROUND_DIGITS)
            ranks[i] = 1 + len(col) - np.searchsorted(col, th, side="right")
    np.fill_diagonal(ranks, 0)
    return ranks


def _label(e: GalleryEntry) -> str:
    return f"{e.camera_id}/{e.video_file}/{e.track_id}/{e.face_path.name}"


def write_misidentify_log(path: Path, entries: list[GalleryEntry], nontargets: list[GalleryEntry], target_emb: np.ndarray,
                          nontarget_emb: np.ndarray, counts: np.ndarray, sim: np.ndarray) -> int:
    """誤識別数が 1 以上の (行 i, 列 j) の組ごとに、i との類似度が j より高かった非対象者の画像を類似度の高い順に 1 行ずつ出す。
    巨大になりうるので 1 組ずつ書き出す。書いた組の数を返す。"""
    t = np.asarray(target_emb, np.float64)
    g = np.asarray(nontarget_emb, np.float64)
    n_pairs = 0
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("誤識別となる画像の一覧: 行の画像との類似度が、列の画像との類似度より高かった非対象者の画像\n"
                f"（非対象者の画像 {len(nontargets)} 枚 / 類似度は小数 {ROUND_DIGITS} 桁で比較）\n")
        for i, row in enumerate(entries):
            s_i = r4 = None
            for j, col in enumerate(entries):
                if i == j or counts[i, j] <= 0:
                    continue
                if s_i is None:
                    s_i = g @ t[i]
                    r4 = np.round(s_i, ROUND_DIGITS)
                hit = np.nonzero(r4 > round(float(sim[i, j]), ROUND_DIGITS))[0]
                hit = hit[np.argsort(-s_i[hit], kind="stable")]
                n_pairs += 1
                f.write(f"\n行: {_label(row)}\n列: {_label(col)}\n"
                        f"  対象者どうしの類似度 {sim[i, j]:.4f} / 誤認した非対象者の画像 {len(hit)} 件\n")
                for k in hit:
                    e = nontargets[k]
                    f.write(f"  {s_i[k]:.4f}  {e.camera_id}/{e.video_file}/{e.track_id}  {e.face_path}\n")
    return n_pairs


# ---- 実行 ----
def run_verify_face_size(track_dirs: list[Path | str], output_dir: Path, model_name: str, landmark_name: str,
                         gate_filter: GateFilter | None = None, mode: str = "similarity", gallery_dir: Path | str | None = None,
                         progress=lambda s: None, echo=print, grayscale: bool = False,
                         align168: bool = False) -> tuple[Path, PhaseLogger]:
    """顔サイズ別精度検証のレポートを出力する（GUI 非依存）。mode は OUTPUT_MODES のキー。
    grayscale=True: アライン後の顔画像に明るさ補正（CLAHE）を掛け、グレースケール化してから特徴量化する（比較対象・検証対象とも。
    キャッシュは別ファイル）。
    align168=True: アラインを 112×112 ではなく 168×168 で行い、特徴量モデルへ渡す直前に 112×112 へ縮小する（キャッシュは別ファイル）。
    誤識別の枚数レポート（misidentify / both）には比較対象フォルダ gallery_dir が必要。"""
    from .verify_face_size_excel import VerifyConditions, build_verify_face_size_report

    want_sim, want_mis, want_rank = runs_similarity_report(mode), runs_misidentify_report(mode), runs_rank_report(mode)
    want_gal = want_mis or want_rank
    if not (want_sim or want_gal):
        raise ValueError(f"出力内容が不明です: {mode}")
    if want_gal and not (gallery_dir and str(gallery_dir).strip()):
        raise ValueError("誤識別となる画像の枚数レポート・類似度順位レポートには比較対象フォルダの指定が必要です")
    log = PhaseLogger(echo=lambda m: (echo(m), progress(m)))
    generated = datetime.now()
    gate_filter = gate_filter or GateFilter()
    align_size = ALIGN_SIZE_WIDE if align168 else None
    variant = "_".join(v for v in ("gray_clahe" if grayscale else "", f"align{align_size}" if align_size else "") if v)
    output_dir = Path(output_dir)
    gallery_path = Path(gallery_dir) if want_gal else None
    with log.phase(f"初期化・モデルロード（特徴量: {model_name} / ランドマーク: {landmark_name}）"):
        extractor = get_extractor(model_name)
        aligner = get_aligner(landmark_name)
        for obj in (extractor, aligner):
            if not obj.weights_ready():
                obj.download_weights(log=lambda m: (echo(m), progress(m)))
        extractor.load()
        aligner.load()
    with log.phase("検証対象人物の顔画像スキャン") as rec:
        all_entries = collect_entries(track_dirs)
        n_gate = apply_gate_filter(all_entries, gate_filter)
        entries = sort_entries([e for e in all_entries if e.used])
        if not entries:
            raise ValueError("追跡IDフォルダに有効な顔画像(_face.jpg)が見つかりません（無効・採用基準外を除く）")
        rec.note = (f"画像 {len(entries)} 件 / 追跡ID {len({e.track_key for e in entries})} 件"
                    f" / 無効画像 {sum(1 for e in all_entries if e.invalid)} 件・採用基準外 {n_gate} 件を除外")
    nontargets: list[GalleryEntry] = []
    nt_emb = None
    cache = None
    if want_gal:
        target_dirs = {Path(d).resolve() for d in track_dirs if str(d).strip()}
        with log.phase("比較対象スキャン（非対象者の特定）") as rec:
            if not gallery_path.is_dir():
                raise ValueError(f"比較対象フォルダが見つかりません: {gallery_path}")
            gallery_all = scan_gallery(gallery_path)
            n_gate_g = apply_gate_filter(gallery_all, gate_filter)
            resolved: dict[Path, Path] = {}
            for g in gallery_all:
                if g.used and resolved.setdefault(g.track_dir, g.track_dir.resolve()) not in target_dirs:
                    nontargets.append(g)
            if not nontargets:
                raise ValueError("比較対象フォルダに非対象者の有効な顔画像(_face.jpg)が見つかりません"
                                 "（指定した追跡IDフォルダ以外・無効・採用基準外を除く）")
            rec.note = (f"非対象者 {len(nontargets)} 件 / 追跡ID {len({g.track_key for g in nontargets})} 件"
                        f" / 無効画像 {sum(1 for g in gallery_all if g.invalid)} 件・採用基準外 {n_gate_g} 件を除外")
        with log.phase("比較対象 顔特徴量抽出") as rec:
            cache = EmbeddingCache(gallery_path, model_name, landmark_name, variant=variant)
            cache.load()
            nt_emb, g_stats = embed_entries(extractor, aligner, nontargets,
                                            lambda i, n: progress(f"比較対象の顔特徴量を計算中… ({i}/{n})"), cache,
                                            clahe=grayscale, grayscale=grayscale, align_size=align_size)
            rec.note = f"{len(nontargets)} 枚 / {_stats_note(g_stats, extractor.needs_landmarks)}"
    with log.phase("検証対象 顔特徴量抽出") as rec:
        emb, stats = embed_entries(extractor, aligner, entries, lambda i, n: progress(f"顔特徴量を計算中… ({i}/{n})"), cache,
                                    clahe=grayscale, grayscale=grayscale, align_size=align_size)
        rec.note = f"{len(entries)} 枚 / {_stats_note(stats, extractor.needs_landmarks)}"
        if cache is not None:
            cache.save()
            rec.note += f" / 特徴量キャッシュ: 再利用 {cache.hits} 件, 新規計算 {cache.misses} 件"
    sim = counts = ranks = None
    n = len(entries)
    if want_sim:
        with log.phase("類似度計算") as rec:
            sim = similarity_matrix(emb, emb)
            rec.note = f"{n} × {n} のうち左下半分 {n * (n - 1) // 2} 組"
    out = output_dir / report_filename(generated)
    # 誤認した画像の一覧テキストはサイズが大きくなりすぎるため出力しない（復活させるときは下のコメントを外す）
    # log_path = out.with_suffix(".txt")
    if want_mis:
        with log.phase("誤識別数の計算") as rec:
            counts, sim_tt = count_misidentified(emb, nt_emb)
            output_dir.mkdir(parents=True, exist_ok=True)
            # n_pairs = write_misidentify_log(log_path, entries, nontargets, emb, nt_emb, counts, sim_tt)
            # rec.note = f"{n} × {n - 1} 組 / 誤識別あり {n_pairs} 組 / 一覧: {log_path.name}"
            rec.note = f"{n} × {n - 1} 組"
    n_tracks = len({g.track_key for g in nontargets})
    if want_rank:
        with log.phase("類似度順位の計算") as rec:
            ranks = rank_matrix(emb, nt_emb, [g.track_key for g in nontargets])  # 非対象者の特徴量は誤識別数と共用
            rec.note = f"{n} × {n - 1} 組 / 非対象者の追跡ID {n_tracks} 件の中での順位"
    cond = VerifyConditions(model_name, landmark_name, gate_filter.describe(), generated,
                            str(gallery_path) if want_gal else "", len(nontargets) if want_gal else None,
                            n_nontarget_tracks=n_tracks if want_rank else None, grayscale=grayscale, align168=align168)
    build_verify_face_size_report(out, cond, entries, sim, log, counts, info_entries=sort_info_entries(entries), ranks=ranks)
    return out, log
