from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..core.features import WeightsUnavailable, get_extractor
from ..core.landmarks import get_aligner
from ..core.pose import get_pose_estimator
from ..core.report.pipeline import run_report


class WeightsWorker(QThread):
    """画面起動時・モデル切替時の重み確認（特徴量抽出・ランドマーク・頭部姿勢の各モデル）。"""
    status = Signal(str)
    ready = Signal()
    unavailable = Signal(str)

    def __init__(self, model_name: str, landmark_name: str, pose_name: str = "なし"):
        super().__init__()
        self.model_name = model_name
        self.landmark_name = landmark_name
        self.pose_name = pose_name

    def run(self) -> None:
        try:
            for obj in (get_extractor(self.model_name), get_aligner(self.landmark_name),
                        get_pose_estimator(self.pose_name)):
                if obj is not None and not obj.weights_ready():
                    obj.download_weights(log=self.status.emit)
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
