r"""検索元セット検索（総当たり・全体検索）。

検索元セット = 顔画像出力先\検索元\<セット名>\ に人がコピーした顔画像の集まり。サブフォルダ（グループ）で分けてよい。
各画像を 1 枚ずつ検索元にして次の 2 つを求め、1 つの Excel にまとめる（query_set_excel.py）。
  総当たり: セット内の他の画像との類似度（N × (N-1) 組）
  全体検索: 比較対象フォルダ全体で検索したときの、セット内の他の画像の順位（画像単位）と本人の追跡IDの順位（追跡ID単位。
            全体 / 同じカメラの中 / 他人だけ）
識別方法はレポートの顔画像指定と同じ（アライン → 特徴量 → コサイン類似度。検索元自身の画像 1 枚だけを比較対象から外す。
同じ類似度（小数 4 桁）は同順位）。

検索元の画像は、比較対象フォルダ内の元画像をバイト一致で探してカメラ・追跡ID・JSON の情報を引き継ぐ（コピーしただけの
画像を想定。ファイル名を変えてもよい）。元画像の追跡IDは自動で「本人」になり、本人の他の追跡IDは own_specs で足す。
元画像が無い画像（縮小版など）も検索元には使えるが、カメラ等は「-」になり、本人の追跡IDにも数えない。
検索元セットの置き場所は比較対象フォルダの中になることが多いので、gallery.scan_gallery は QUERY_SETS_DIR_NAME を常に飛ばす。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..common.phase_log import PhaseLogger
from ..features import get_extractor
from ..landmarks import get_aligner
from ..pose import get_pose_estimator
from .cache import EmbeddingCache
from .excel import is_target
from .gallery import QUERY_SETS_DIR_NAME, GalleryEntry, in_excluded_dir, scan_gallery
from .gate import GateFilter, apply_gate_filter
from .pipeline import _stats_note, embed_entries, estimate_poses
from .similarity import similarity_matrix

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_BAD_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


@dataclass
class QueryImage:
    no: int                           # 1 始まり（グループ名 → ファイル名の順）
    group: str                        # セット直下のサブフォルダ名（セット直下に置いた画像は ""）
    path: Path                        # セット内の画像
    entry: GalleryEntry               # 元画像のエントリ（比較対象の情報を引き継ぐ）。元画像が無ければ画像そのものから作った最小限のエントリ
    original: Path | None = None      # 比較対象フォルダ内の元画像（バイト一致）。無ければ None
    gallery_index: int | None = None  # 順位付けに使う比較対象一覧での添字。比較対象に無い（無効・old・採用基準外・元画像なし）なら None
    treatment: str = ""               # 比較対象での扱い（「含む」/ 含まない理由）

    @property
    def label(self) -> str:
        return f"{self.group}/{self.path.stem}" if self.group else self.path.stem


# ---- セットの置き場所 ----
def default_query_sets_root(output_root: Path | str) -> Path:
    """検索元セットの置き場所 = 顔画像出力先\\検索元。"""
    return Path(output_root) / QUERY_SETS_DIR_NAME


def scan_query_images(set_dir: Path) -> list[tuple[str, Path]]:
    """セット内の画像を (グループ, パス) で返す。グループ = セット直下のサブフォルダ名（直下の画像は ""）。
    サブフォルダのさらに下の画像もそのサブフォルダのグループとして拾う。並びはグループ名 → 相対パス。"""
    set_dir = Path(set_dir)
    if not set_dir.is_dir():
        return []
    out = []
    for p in set_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES or p.name.startswith("."):
            continue
        rel = p.relative_to(set_dir)
        out.append((rel.parts[0] if len(rel.parts) > 1 else "", p))
    return sorted(out, key=lambda t: (t[0], t[1].relative_to(set_dir).as_posix()))


def list_query_sets(root: Path | str) -> list[tuple[str, int]]:
    """置き場所直下のセット（フォルダ）を (セット名, 画像数) で返す。"""
    root = Path(root)
    if not root.is_dir():
        return []
    return [(d.name, len(scan_query_images(d))) for d in sorted(root.iterdir())
            if d.is_dir() and not d.name.startswith(".")]


def create_query_set(root: Path | str, name: str) -> Path:
    """セットのフォルダを作る（既にあればそのまま）。"""
    name = name.strip()
    if not name or name in (".", "..") or _BAD_NAME_CHARS.search(name):
        raise ValueError('セット名が空か、フォルダ名に使えない文字（\\ / : * ? " < > |）を含んでいます')
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- 元画像の特定 ----
def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def match_originals(paths: list[Path], candidates: list[GalleryEntry]) -> dict[Path, GalleryEntry]:
    """画像の内容（ファイルサイズ + MD5）が一致する比較対象のエントリを探す。見つかった画像だけを返す。"""
    sizes: set[int] = set()
    for p in paths:
        try:
            sizes.add(Path(p).stat().st_size)
        except OSError:
            pass
    by_hash: dict[str, GalleryEntry] = {}
    for e in candidates:
        try:
            if e.face_path.stat().st_size in sizes:
                by_hash.setdefault(_md5(e.face_path), e)
        except OSError:
            continue
    out: dict[Path, GalleryEntry] = {}
    for p in paths:
        try:
            e = by_hash.get(_md5(p))
        except OSError:
            continue
        if e is not None:
            out[p] = e
    return out


def parse_track_lines(text: str) -> list[tuple[str, str, str]]:
    """「カメラID|映像ファイル名|追跡ID」を 1 行 1 件で読む（空行・# 始まりの行は無視。空欄の項目は不問）。"""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.split("|")] + ["", "", ""]
        if any(parts[:3]):
            out.append((parts[0], parts[1], parts[2]))
    return out


# ---- 順位付け ----
@dataclass
class ImageRank:
    sim: float
    rank_camera: int   # 検索対象と同じカメラの画像の中での順位
    rank_all: int      # 比較対象の全画像の中での順位


@dataclass
class TrackRank:
    key: Path          # 追跡IDフォルダ
    camera_id: str
    video_file: str
    track_id: str
    sim: float         # その追跡IDで一番似た画像の類似度（検索元自身の画像は除く）
    rank_camera: int   # 同じカメラの追跡IDの中での順位
    rank_all: int      # 全追跡IDの中での順位
    rank_strangers: int  # 本人の追跡IDを数えない順位（他人の追跡IDの中で何位か）


@dataclass
class SearchResult:
    round_robin: np.ndarray                          # (N, N) 検索元どうしの類似度
    image_ranks: dict[tuple[int, int], ImageRank]    # (検索元 i, 検索対象 j) -> 画像の順位。j の元画像が比較対象に無ければ無し
    track_ranks: dict[tuple[int, Path], TrackRank]   # (検索元 i, 本人の追跡IDフォルダ) -> 追跡IDの順位
    own_tracks: list[tuple[Path, str, str, str]]     # 本人の追跡ID (フォルダ, カメラID, 映像ファイル, 追跡ID)。検索元の順 → own_specs の順
    n_tracks: int = 0
    unmatched_specs: list[tuple[str, str, str]] = field(default_factory=list)  # 比較対象に無かった own_specs


def compute_search(q_emb: np.ndarray, g_emb: np.ndarray, gallery: list[GalleryEntry], queries: list[QueryImage],
                   own_specs=()) -> SearchResult:
    """総当たりの類似度と、比較対象全体での順位を求める（モデル非依存。特徴量は L2 正規化済み）。
    検索元ごとに自身の元画像 1 枚だけを比較対象から外す。本人の追跡ID = 検索元の元画像の追跡ID + own_specs に合う追跡ID。"""
    rr = similarity_matrix(q_emb, q_emb)
    keys: list[Path] = []
    key_index: dict[Path, int] = {}
    rep: dict[Path, GalleryEntry] = {}
    for g in gallery:
        if g.track_key not in key_index:
            key_index[g.track_key] = len(keys)
            keys.append(g.track_key)
            rep[g.track_key] = g
    tidx = np.array([key_index[g.track_key] for g in gallery], int)
    cam_img = np.array([g.camera_id for g in gallery]) if gallery else np.zeros(0, str)
    cam_trk = np.array([rep[k].camera_id for k in keys]) if keys else np.zeros(0, str)
    own_keys: list[Path] = []
    for q in queries:
        if q.original is not None and q.entry.track_key in key_index and q.entry.track_key not in own_keys:
            own_keys.append(q.entry.track_key)
    unmatched = []
    for spec in own_specs:
        hit = [k for k in keys if is_target(rep[k], [spec])]
        if not hit:
            unmatched.append(spec)
        own_keys += [k for k in hit if k not in own_keys]
    own_set = set(own_keys)
    own_mask = np.array([k in own_set for k in keys], bool)
    T = len(keys)

    image_ranks: dict[tuple[int, int], ImageRank] = {}
    track_ranks: dict[tuple[int, Path], TrackRank] = {}
    for i, q in enumerate(queries):
        s = (g_emb @ q_emb[i]).astype(np.float64) if len(gallery) else np.zeros(0)
        valid = np.ones(len(gallery), bool)
        if q.gallery_index is not None:
            valid[q.gallery_index] = False  # 検索元自身の画像は比較対象から外す（レポートと同じ）
        r4 = np.round(s, 4)
        for j, g in enumerate(queries):
            if j == i or g.gallery_index is None or not valid[g.gallery_index]:
                continue  # 検索対象が比較対象に無い、または検索元と同じ画像（同じ元画像を 2 か所に置いた）なら順位なし
            gi = g.gallery_index
            image_ranks[(i, j)] = ImageRank(float(s[gi]),
                                            1 + int(np.sum(valid & (cam_img == cam_img[gi]) & (r4 > r4[gi]))),
                                            1 + int(np.sum(valid & (r4 > r4[gi]))))
        if T == 0:
            continue
        tm = np.full(T, -np.inf)
        np.maximum.at(tm, tidx[valid], s[valid])
        has = np.isfinite(tm)  # 検索元自身の画像しか無い追跡IDは順位なし
        t4 = np.round(np.where(has, tm, 0.0), 4)
        for k in own_keys:
            t = key_index[k]
            if not has[t]:
                continue
            better = has & (t4 > t4[t])
            e = rep[k]
            track_ranks[(i, k)] = TrackRank(k, e.camera_id, e.video_file, e.track_id, float(tm[t]),
                                            1 + int(np.sum(better & (cam_trk == cam_trk[t]))), 1 + int(np.sum(better)),
                                            1 + int(np.sum(better & ~own_mask)))
    return SearchResult(rr, image_ranks, track_ranks,
                        [(k, rep[k].camera_id, rep[k].video_file, rep[k].track_id) for k in own_keys], T, unmatched)


# ---- 実行 ----
def _treatment(q: QueryImage, gallery_root: Path, include_excluded: bool) -> str:
    if q.gallery_index is not None:
        return "含む"
    if q.original is None:
        return "元画像なし（比較対象フォルダに同じ画像が無い）"
    e = q.entry
    if e.invalid:
        return f"無効画像のため含まない（{e.invalid_reason}）"
    if not include_excluded and in_excluded_dir(e.face_path, gallery_root):
        return "old フォルダの画像のため含まない"
    if e.gate_excluded_reason:
        return f"採用基準外のため含まない（{e.gate_excluded_reason}）"
    return "含まない"


def run_query_set_search(set_dir: Path, gallery_dir: Path, output_dir: Path, model_name: str, landmark_name: str,
                         pose_name: str = "なし", include_excluded: bool = False, gate_filter: GateFilter | None = None,
                         own_specs=None, use_cache: bool = True, progress=lambda s: None, echo=print
                         ) -> tuple[Path, PhaseLogger]:
    """検索元セットの各画像で総当たり・全体検索をして Excel を出力する（GUI 非依存。ui と cli の両方から呼ぶ）。
    own_specs: 本人の追跡ID「(カメラID, 映像ファイル名, 追跡ID)」の追加分（検索元の元画像の追跡IDは自動で本人）。
    include_excluded / gate_filter はレポートと同じ意味（old フォルダの扱い / 顔の採用基準）。"""
    from .query_set_excel import QuerySetConditions, build_query_set_report

    log = PhaseLogger(echo=lambda m: (echo(m), progress(m)))
    generated = datetime.now()
    gate_filter = gate_filter or GateFilter()
    own_specs = [s for s in (own_specs or []) if any(s)]
    set_dir, gallery_dir, output_dir = Path(set_dir), Path(gallery_dir), Path(output_dir)
    with log.phase(f"初期化・モデルロード（特徴量: {model_name} / ランドマーク: {landmark_name} / 姿勢: {pose_name}）"):
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
    with log.phase("検索元セット読込・元画像の特定") as rec:
        found = scan_query_images(set_dir)
        if not found:
            raise ValueError(f"検索元セットに画像がありません: {set_dir}")
        # 元画像は old フォルダも含めて探す。セット自体が比較対象フォルダの中（検索元 以外の場所）にあっても、コピーは比較対象に数えない
        set_prefix = os.path.normcase(os.path.abspath(set_dir)) + os.sep
        all_incl = [e for e in scan_gallery(gallery_dir, include_excluded=True)
                    if not os.path.normcase(os.path.abspath(e.face_path)).startswith(set_prefix)]
        originals = match_originals([p for _, p in found], all_incl)
        queries: list[QueryImage] = []
        for no, (group, p) in enumerate(found, start=1):
            e = originals.get(p)
            if e is None:
                e = GalleryEntry(face_path=p, body_path=None, track_dir=p.parent, camera_id="-", video_name="-",
                                 track_id="-", ts=p.stem, location="-", video_file="-")
            queries.append(QueryImage(no, group, p, e, None if e.face_path == p else e.face_path))
        n_matched = sum(1 for q in queries if q.original is not None)
        rec.note = f"画像 {len(queries)} 枚（グループ {len({q.group for q in queries})}）/ 元画像を特定 {n_matched} 枚"
    with log.phase("比較対象スキャン") as rec:
        gallery_all = all_incl if include_excluded else [e for e in all_incl if not in_excluded_dir(e.face_path, gallery_dir)]
        n_gate = apply_gate_filter(gallery_all, gate_filter)
        gallery = [g for g in gallery_all if g.used]
        if not gallery:
            raise ValueError("比較対象フォルダに有効な顔画像(_face.jpg)が見つかりません（無効・採用基準外を除く）")
        index = {g.face_path: i for i, g in enumerate(gallery)}
        for q in queries:
            q.gallery_index = index.get(q.original) if q.original is not None else None
            q.treatment = _treatment(q, gallery_dir, include_excluded)
        rec.note = (f"画像 {len(gallery)} 件 / 追跡ID {len({g.track_key for g in gallery})} 件"
                    f" / 無効画像 {sum(1 for g in gallery_all if g.invalid)} 件・採用基準外 {n_gate} 件を除外")
    with log.phase("比較対象 顔特徴量抽出") as rec:
        cache = EmbeddingCache(gallery_dir, model_name, landmark_name) if use_cache else None
        if cache is not None:
            cache.load()
        g_emb, g_stats = embed_entries(extractor, aligner, gallery,
                                       lambda i, n: progress(f"比較対象の顔特徴量を計算中… ({i}/{n})"), cache)
        rec.note = f"{len(gallery)} 枚 / {_stats_note(g_stats, extractor.needs_landmarks)}"
        if cache is not None:
            cache.save()
            rec.note += f" / 特徴量キャッシュ: 再利用 {cache.hits} 件, 新規計算 {cache.misses} 件（{cache.path.name}）"
    with log.phase("検索元 顔特徴量抽出") as rec:
        q_emb = np.zeros((len(queries), extractor.embed_dim), np.float32)  # 行 = queries の並び（compute_search と同じ）
        todo = [i for i, q in enumerate(queries) if q.gallery_index is None]
        for i, q in enumerate(queries):
            if q.gallery_index is not None:
                q_emb[i] = g_emb[q.gallery_index]  # 同じ画像なので比較対象の特徴量をそのまま使う
        note = f"比較対象の特徴量を再利用 {len(queries) - len(todo)} 枚 / 新規計算 {len(todo)} 枚"
        if todo:
            emb, q_stats = embed_entries(extractor, aligner, [queries[i].entry for i in todo])
            q_emb[todo] = emb
            note += f"（{_stats_note(q_stats, extractor.needs_landmarks)}）"
        rec.note = note
    with log.phase("類似度計算・順位付け") as rec:
        result = compute_search(q_emb, g_emb, gallery, queries, own_specs)
        rec.note = (f"総当たり {len(queries)} × {len(queries) - 1} 組 / 本人の追跡ID {len(result.own_tracks)} 件"
                    + (f" / 比較対象に無い指定 {len(result.unmatched_specs)} 件" if result.unmatched_specs else ""))
    if pose_estimator is not None:
        with log.phase(f"頭部姿勢推定（{pose_name}）") as rec:
            estimate_poses(pose_estimator, [q.entry for q in queries])
            rec.note = f"検索元 {len(queries)} 件"
    cond = QuerySetConditions(model_name, landmark_name, pose_name, str(set_dir), str(gallery_dir), str(output_dir),
                              generated, include_excluded, gate_filter.describe(), list(own_specs), use_cache)
    out = output_dir / f"query_set_{generated:%Y%m%d_%H%M%S}.xlsx"
    build_query_set_report(out, cond, queries, gallery, result, log, n_gate_gallery=n_gate)
    return out, log
