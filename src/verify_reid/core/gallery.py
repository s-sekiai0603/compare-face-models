r"""比較対象フォルダの走査。

構成: 出力先ルート\日時\カメラID\映像名(拡張子なし)\追跡ID\{ts}_body.jpg, {ts}_body_aligned.jpg, {追跡ID}.json
画像の付帯情報は追跡IDフォルダの json とフォルダパスから識別する。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_BODY_RE = re.compile(r"^(?P<ts>.+)_body\.jpg$")
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
    body_path: Path
    aligned_path: Path | None
    track_dir: Path
    camera_id: str
    video_name: str          # 映像ファイル名(拡張子なし)＝フォルダ名
    track_id: str
    ts: str
    location: str = "-"
    video_file: str = "-"
    person_conf: float | None = None
    bbox_xyxy: list | None = None
    time_sec: float | None = None   # 映像内経過秒（json）
    invalid: bool = False           # 無効画像（別人の骨格を含む等）。比較に使わない
    invalid_reason: str = ""
    keypoints: list | None = field(default=None, repr=False)

    @property
    def track_key(self) -> Path:
        return self.track_dir

    @property
    def captured_text(self) -> str:
        return format_ts(self.ts)

    @property
    def embed_path(self) -> Path:
        """特徴量抽出に使う画像（_body_aligned.jpg、無ければ _body.jpg）。"""
        return self.aligned_path or self.body_path


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
                images = {im.get("file_body"): im for im in data.get("images", []) if im.get("file_body")}
            except (OSError, ValueError):
                pass
            self._cache[track_dir] = (meta, images)
        return self._cache[track_dir]


def _make_entry(body: Path, cache: _TrackMetaCache) -> GalleryEntry | None:
    m = _BODY_RE.match(body.name)
    if not m:
        return None
    track_dir = body.parent
    video_dir = track_dir.parent
    camera_dir = video_dir.parent
    if track_dir == track_dir.parent or video_dir == camera_dir:
        return None
    aligned = body.with_name(f"{m.group('ts')}_body_aligned.jpg")
    meta, images = cache.get(track_dir)
    im = images.get(body.name, {})
    return GalleryEntry(
        body_path=body,
        aligned_path=aligned if aligned.is_file() else None,
        track_dir=track_dir,
        camera_id=camera_dir.name,
        video_name=video_dir.name,
        track_id=track_dir.name,
        ts=m.group("ts"),
        location=meta.get("location") or "-",
        video_file=meta.get("video_file") or f"{video_dir.name}.mp4",
        person_conf=im.get("person_conf"),
        bbox_xyxy=im.get("bbox_xyxy"),
        time_sec=im.get("time_sec"),
        invalid=bool(im.get("invalid", False)),
        invalid_reason=im.get("invalid_reason") or "",
        keypoints=im.get("keypoints"),
    )


def scan_gallery(root: Path, exclude_dirs: set[Path] | None = None,
                 exclude_files: set[Path] | None = None) -> list[GalleryEntry]:
    """root 配下の全 _body.jpg を走査する。exclude_* はクエリ自身の除外用。"""
    cache = _TrackMetaCache()
    ex_dirs = {p.resolve() for p in (exclude_dirs or set())}
    ex_files = {p.resolve() for p in (exclude_files or set())}
    entries: list[GalleryEntry] = []
    for body in sorted(root.rglob("*_body.jpg")):
        rb = body.resolve()
        if rb in ex_files or rb.parent in ex_dirs:
            continue
        e = _make_entry(body, cache)
        if e:
            entries.append(e)
    return entries


def load_entry(body: Path) -> GalleryEntry:
    """指定された _body.jpg 1枚のエントリ（クエリ用）。"""
    e = _make_entry(body, _TrackMetaCache())
    if e is None:
        raise ValueError(f"外見画像(_body.jpg)として解釈できません: {body}")
    return e


def scan_track_folder(track_dir: Path) -> list[GalleryEntry]:
    """追跡IDフォルダ内の外見画像を時刻順に返す。"""
    entries = scan_gallery(track_dir)
    entries = [e for e in entries if e.track_dir == track_dir]
    return sorted(entries, key=lambda e: e.ts)
