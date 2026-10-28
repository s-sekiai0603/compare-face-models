r"""比較対象フォルダの走査。

構成: 出力先ルート\日時\カメラID\映像名(拡張子なし)\追跡ID\{ts}_face.jpg, {ts}_body.jpg, {追跡ID}.json
画像の付帯情報は追跡IDフォルダの json とフォルダパスから識別する。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_FACE_RE = re.compile(r"^(?P<ts>.+)_face\.jpg$")
_TS_RE = re.compile(r"^(\d{8})_(\d{6})(?:_(\d{3}))?$")


def format_ts(ts: str) -> str:
    """20260311_122723_500 -> 2026/03/11 12:27:23.500。解釈できなければそのまま。"""
    m = _TS_RE.match(ts)
    if not m:
        return ts
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return ts
    s = dt.strftime("%Y/%m/%d %H:%M:%S")
    return f"{s}.{m.group(3)}" if m.group(3) else s


@dataclass
class GalleryEntry:
    face_path: Path
    body_path: Path | None
    track_dir: Path
    camera_id: str
    video_name: str          # 映像ファイル名(拡張子なし)＝フォルダ名
    track_id: str
    ts: str
    location: str = "-"
    video_file: str = "-"
    person_conf: float | None = None
    face_conf: float | None = None     # 顔検出の信頼度
    face_width: float | None = None    # 検出顔boxの幅(px)
    eye_ratio: float | None = None     # 両目間隔/顔幅（正面度。片目のみ検出時は None）
    frontal: bool | None = None        # 正面とみなしたか（eye_ratio >= 0.25）
    quality: float | None = None       # 品質スコア（顔信頼度 + min(10, 顔幅/10)）
    face_bbox_xyxy: list | None = None
    person_bbox_xyxy: list | None = None  # 全身画像の切り出し枠（フレーム座標）。頭部姿勢の余白広めクロップに使う
    time_sec: float | None = None      # 映像内経過秒（json）
    gate_ok: bool = True               # 抽出時の採用基準（顔ゲート + 品質ガード）を通ったか = premises-security が出力する画像か
    gate_reason: str = ""              # 採用基準に外れた理由（抽出時の判定）
    face_visible: bool | None = None   # 骨格ルール（鼻+片目が見える / 下向きでない）を通ったか。判定値の無い古い抽出結果は None
    crop_min_side: float | None = None  # 切り出し box の短辺(px)。品質ガードの判定値。古い抽出結果は None
    invalid: bool = False              # 無効画像（別人の骨格を含む等）。比較に使わない
    invalid_reason: str = ""
    pose: tuple[float, float, float] | None = None  # レポート時に推定した頭部姿勢 (yaw, pitch, roll)[度]
    quality_norm: float | None = None  # 品質ノルム（AdaFace の特徴量ノルム。クエリ自動選別時に計算）
    gate_excluded_reason: str = ""     # レポート時の採用基準（gate.GateFilter）で外した理由（空なら使用）
    excluded_reason: str = ""          # クエリ自動選別で外した理由（空なら使用）

    @property
    def used(self) -> bool:
        """特徴量比較に使うか（無効・採用基準外・自動選別除外のいずれでもない）。"""
        return not (self.invalid or self.gate_excluded_reason or self.excluded_reason)

    @property
    def track_key(self) -> Path:
        return self.track_dir

    @property
    def captured_text(self) -> str:
        return format_ts(self.ts)

    @property
    def embed_path(self) -> Path:
        """特徴量抽出に使う画像。"""
        return self.face_path


def entry_seconds(e: GalleryEntry) -> float:
    """画像の時刻(秒)。ファイル名の日時が読めればその時刻、無ければ映像内経過秒。"""
    m = _TS_RE.match(e.ts)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return dt.timestamp() + (int(m.group(3)) / 1000 if m.group(3) else 0.0)
        except ValueError:
            pass
    return float(e.time_sec or 0.0)


class _TrackMetaCache:
    def __init__(self):
        self._cache: dict[Path, tuple[dict, dict[str, dict]]] = {}

    def get(self, track_dir: Path) -> tuple[dict, dict[str, dict]]:
        if track_dir not in self._cache:
            meta: dict = {}
            images: dict[str, dict] = {}
            jp = track_dir / f"{track_dir.name}.json"
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
                meta = {k: data.get(k) for k in ("location", "video_file")}
                images = {im.get("file_face"): im for im in data.get("images", []) if im.get("file_face")}
            except (OSError, ValueError):
                pass
            self._cache[track_dir] = (meta, images)
        return self._cache[track_dir]


def _make_entry(face: Path, cache: _TrackMetaCache) -> GalleryEntry | None:
    m = _FACE_RE.match(face.name)
    if not m:
        return None
    track_dir = face.parent
    if track_dir.name.lower() in EXCLUDED_DIR_NAMES:  # old フォルダの画像は、その親（元の追跡ID）の画像として扱う
        track_dir = track_dir.parent
    video_dir = track_dir.parent
    camera_dir = video_dir.parent
    if track_dir == track_dir.parent or video_dir == camera_dir:
        return None
    meta, images = cache.get(track_dir)
    im = images.get(face.name, {})
    body = face.with_name(im.get("file_body") or f"{m.group('ts')}_body.jpg")
    return GalleryEntry(
        face_path=face,
        body_path=body if body.is_file() else None,
        track_dir=track_dir,
        camera_id=camera_dir.name,
        video_name=video_dir.name,
        track_id=track_dir.name,
        ts=m.group("ts"),
        location=meta.get("location") or "-",
        video_file=meta.get("video_file") or f"{video_dir.name}.mp4",
        person_conf=im.get("person_conf"),
        face_conf=im.get("face_conf"),
        face_width=im.get("face_width"),
        eye_ratio=im.get("eye_ratio"),
        frontal=im.get("frontal"),
        quality=im.get("quality"),
        face_bbox_xyxy=im.get("face_bbox_xyxy"),
        person_bbox_xyxy=im.get("bbox_xyxy"),
        time_sec=im.get("time_sec"),
        gate_ok=bool(im.get("gate_ok", True)),  # 判定値の無い古い抽出結果は基準を通った画像しか無い
        gate_reason=im.get("gate_reason") or "",
        face_visible=im.get("face_visible"),
        crop_min_side=im.get("crop_min_side"),
        invalid=bool(im.get("invalid", False)),
        invalid_reason=im.get("invalid_reason") or "",
    )


EXCLUDED_DIR_NAMES = {"old"}  # 追跡IDフォルダ内のこの名前のフォルダに移した画像は、既定ではクエリ・比較対象とも使わない
QUERY_SETS_DIR_NAME = "検索元"  # 顔画像出力先直下の検索元セット置き場（query_set.py）。比較対象の走査では常に飛ばす


def _dir_parts(face: Path, root: Path) -> tuple[str, ...]:
    try:
        parts = face.relative_to(root).parts[:-1]
    except ValueError:
        parts = face.parts[:-1]
    return tuple(p.lower() for p in parts)


def in_excluded_dir(face: Path, root: Path) -> bool:
    """old フォルダ（EXCLUDED_DIR_NAMES）の中の画像か。"""
    return any(p in EXCLUDED_DIR_NAMES for p in _dir_parts(face, root))


def in_query_sets_dir(face: Path, root: Path) -> bool:
    """走査ルート直下の検索元セット置き場（QUERY_SETS_DIR_NAME）の中の画像か。比較対象のコピーなので比較対象には数えない。
    直下だけを見る（深い階層に同名のカメラ・追跡IDフォルダがあっても対象のまま）。"""
    parts = _dir_parts(face, root)
    return bool(parts) and parts[0] == QUERY_SETS_DIR_NAME.lower()


def scan_gallery(root: Path, exclude_dirs: set[Path] | None = None,
                 exclude_files: set[Path] | None = None, include_excluded: bool = False) -> list[GalleryEntry]:
    """root 配下の全 _face.jpg を走査する。exclude_* はクエリ自身の除外用（追跡IDフォルダ単位・画像単位）。
    old フォルダ（EXCLUDED_DIR_NAMES）に移された画像は既定では対象外（手で選別した画像を外す仕組み）。
    include_excluded=True なら old の画像も元の追跡IDの画像として含める（選別前の状態で比較するため）。
    検索元セット置き場（QUERY_SETS_DIR_NAME）の画像は常に対象外。"""
    cache = _TrackMetaCache()
    ex_dirs = {p.resolve() for p in (exclude_dirs or set())}
    ex_files = {p.resolve() for p in (exclude_files or set())}
    entries: list[GalleryEntry] = []
    for face in sorted(root.rglob("*_face.jpg")):
        if in_query_sets_dir(face, root):
            continue
        if not include_excluded and in_excluded_dir(face, root):
            continue
        e = _make_entry(face, cache)
        if e is None:
            continue
        if face.resolve() in ex_files or e.track_dir.resolve() in ex_dirs:  # old の画像も元の追跡IDで判定
            continue
        entries.append(e)
    return entries


def load_entry(face: Path) -> GalleryEntry:
    """指定された _face.jpg 1枚のエントリ（クエリ用）。"""
    e = _make_entry(face, _TrackMetaCache())
    if e is None:
        raise ValueError(f"顔画像(_face.jpg)として解釈できません: {face}")
    return e


def scan_track_folder(track_dir: Path, include_excluded: bool = False) -> list[GalleryEntry]:
    """追跡IDフォルダ内の顔画像を時刻順に返す（include_excluded=True なら old フォルダの画像も含める）。"""
    entries = scan_gallery(track_dir, include_excluded=include_excluded)
    entries = [e for e in entries if e.track_dir == track_dir]
    return sorted(entries, key=lambda e: e.ts)
