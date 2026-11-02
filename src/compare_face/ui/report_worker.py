from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..core.features import WeightsUnavailable, get_extractor
from ..core.landmarks import get_aligner
from ..core.pose import get_pose_estimator
from ..core.report.face_size_map import run_face_size_map
from ..core.report.pipeline import run_report
from ..core.report.query_set import run_query_set_search
from ..core.report.verify_face_size import run_verify_face_size


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


class QuerySetWorker(QThread):
    """検索元セット検索（query_set.run_query_set_search）を別スレッドで実行する。"""
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            out, _ = run_query_set_search(progress=self.progress.emit, **self.kwargs)
            self.finished_ok.emit(str(out))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class VerifyFaceSizeWorker(QThread):
    """顔サイズ別精度検証（verify_face_size.run_verify_face_size）を別スレッドで実行する。"""
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            out, _ = run_verify_face_size(progress=self.progress.emit, **self.kwargs)
            self.finished_ok.emit(str(out))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class FaceSizeMapWorker(QThread):
    """指定顔サイズマッピング（face_size_map.run_face_size_map）を別スレッドで実行する。"""
    progress = Signal(str)
    finished_ok = Signal(list, list)  # 出力ファイル（文字列）, スキップ・注意の説明
    failed = Signal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            outputs, notes = run_face_size_map(progress=self.progress.emit, **self.kwargs)
            self.finished_ok.emit([str(p) for p in outputs], notes)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
