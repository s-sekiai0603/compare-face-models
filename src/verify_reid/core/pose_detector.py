"""YOLOv8-pose による人物検出+骨格推定（人物検出と骨格を兼用）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONF_TH = 0.25
NMS_IOU = 0.50
MAX_DETS = 100
BATCH_SIZE = 16  # バッチ推論の枚数（TensorRT エンジンの最大バッチと一致させる）
IMGSZ = 640
KPT_CONF_MIN = 0.35
UPPER_BODY_IDX = range(5, 13)  # 肩・肘・手首・腰
MAX_UPPER_MISSING = 4


@dataclass
class PoseDetections:
    boxes: np.ndarray   # (N,4) xyxy
    scores: np.ndarray  # (N,)
    kpts: np.ndarray    # (N,17,3) x,y,conf
    # ハードフィルタ前の全検出（別人の骨格が切り出し範囲に入っているかの判定用）
    all_boxes: np.ndarray | None = field(default=None, repr=False)
    all_kpts: np.ndarray | None = field(default=None, repr=False)


def passes_hard_filter(kpt: np.ndarray) -> bool:
    """足首・顔が少なくとも1点見えていること、上半身欠損が4点未満であること。"""
    vis = kpt[:, 2] >= KPT_CONF_MIN
    if not vis[15:17].any() or not vis[0:5].any():
        return False
    return sum(1 for i in UPPER_BODY_IDX if not vis[i]) < MAX_UPPER_MISSING


class PoseDetector:
    def __init__(self, model_path: str):
        from ultralytics import YOLO
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path), task="pose")
        self.model_name = Path(str(model_path)).name
        self.backend = "TensorRT" if str(model_path).lower().endswith(".engine") else "PyTorch"
        self.describe()

    def describe(self) -> str:
        """実行デバイスとフレームワークをコンソールへ出力する。"""
        import torch
        if self.device == "cpu":
            dev = "CPU"
        else:
            dev = f"GPU (cuda:{self.device}, {torch.cuda.get_device_name(self.device)})"
        msg = (f"[推論環境] デバイス: {dev} / フレームワーク: {self.backend}"
               f" / torch {torch.__version__} / モデル: {self.model_name}")
        print(msg, flush=True)
        return msg

    def detect(self, frame_bgr: np.ndarray) -> PoseDetections:
        return self.detect_batch([frame_bgr])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[PoseDetections]:
        """複数フレームを BATCH_SIZE 単位でまとめて推論する。結果は入力と同じ順序。"""
        out: list[PoseDetections] = []
        for i in range(0, len(frames), BATCH_SIZE):
            results = self.model.predict(
                frames[i:i + BATCH_SIZE], conf=CONF_TH, iou=NMS_IOU, imgsz=IMGSZ,
                max_det=MAX_DETS, device=self.device, verbose=False)
            out.extend(self._to_detections(r) for r in results)
        return out

    @staticmethod
    def _to_detections(res) -> PoseDetections:
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return PoseDetections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                                  np.zeros((0, 17, 3), np.float32))
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        kpts = res.keypoints.data.cpu().numpy().astype(np.float32)
        keep = [i for i in range(len(boxes)) if passes_hard_filter(kpts[i])]
        return PoseDetections(boxes[keep], scores[keep], kpts[keep], boxes, kpts)
