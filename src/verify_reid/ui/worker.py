from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.extractor import extract_video
from ..core.video_scanner import VideoInfo


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
            from ..core.model_loader import prepare_model_path
            from ..core.pose_detector import PoseDetector
            self.status.emit("モデルを準備しています...")
            detector = PoseDetector(str(prepare_model_path(log=self.status.emit)))
            run_dir = self.output_root / datetime.now().strftime("%Y%m%d%H%M%S")
            results = []
            for i, v in enumerate(self.videos):
                self.status.emit(f"処理中 ({i + 1}/{len(self.videos)}): {v.file_name}")
                out_dir = run_dir / v.camera_id / v.path.stem
                res = extract_video(
                    v.path, out_dir, detector, self.interval_sec,
                    progress=lambda f, t, i=i: self.progress.emit(i, len(self.videos), f, t),
                    is_cancelled=lambda: self._cancel,
                    location=v.location,
                )
                results.append(res)
                if res.cancelled:
                    break
            self.done.emit(results, str(run_dir), self._cancel)
        except Exception as e:
            self.failed.emit(str(e))
