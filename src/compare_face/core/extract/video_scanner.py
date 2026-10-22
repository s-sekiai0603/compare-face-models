"""ルートフォルダ\\ロケーション\\カメラID\\*.mp4 の走査。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

_TS_RE = re.compile(r"(\d{14})")


def parse_capture_datetime(name: str) -> datetime | None:
    """ファイル名(YYYYMMDDHHMMSS.mp4)から撮影日時を取得。判断できなければ None。"""
    m = _TS_RE.search(Path(name).stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(round(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def get_duration_sec(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and fps > 0 and frames and frames > 0:
            return frames / fps
        return None
    finally:
        cap.release()


@dataclass
class VideoInfo:
    path: Path
    location: str
    camera_id: str
    captured_at: datetime | None
    duration_sec: float | None

    @property
    def file_name(self) -> str:
        return self.path.name

    @property
    def captured_text(self) -> str:
        return self.captured_at.strftime("%Y/%m/%d %H:%M:%S") if self.captured_at else "-"


def scan_videos(root: Path) -> list[VideoInfo]:
    videos: list[VideoInfo] = []
    if not root.is_dir():
        return videos
    for loc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cam_dir in sorted(p for p in loc_dir.iterdir() if p.is_dir()):
            for f in sorted(cam_dir.iterdir()):
                if f.is_file() and f.suffix.lower() == ".mp4":
                    videos.append(VideoInfo(
                        path=f,
                        location=loc_dir.name,
                        camera_id=cam_dir.name,
                        captured_at=parse_capture_datetime(f.name),
                        duration_sec=get_duration_sec(f),
                    ))
    return videos
