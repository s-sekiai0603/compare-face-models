from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.extract.pipeline import run_extraction
from ..core.extract.video_scanner import VideoInfo


class ExtractWorker(QThread):
    status = Signal(str)
    progress = Signal(int, int, int, int)  # 映像index, 映像数, フレーム, 総フレーム
    done = Signal(list, str, bool)         # 結果一覧, 出力先, キャンセル有無
    failed = Signal(str)

    def __init__(self, videos: list[VideoInfo], output_root: Path, interval_sec: float):
        super().__init__()
        self.videos = videos
        self.output_root = output_root
        self.interval_sec = interval_sec
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            results, run_dir = run_extraction(
                self.videos, self.output_root, self.interval_sec,
                status=self.status.emit, progress=self.progress.emit, is_cancelled=lambda: self._cancel)
            self.done.emit(results, str(run_dir), self._cancel)
        except Exception as e:
            self.failed.emit(str(e))
