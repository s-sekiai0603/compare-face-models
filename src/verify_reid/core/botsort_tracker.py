"""ultralytics 内蔵 BoT-SORT を検出配列に直接適用し、骨格をIoUで紐付ける。"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .pose_detector import PoseDetections

BOTSORT_CFG = dict(
    tracker_type="botsort", track_high_thresh=0.5, track_low_thresh=0.1,
    new_track_thresh=0.6, track_buffer=30, match_thresh=0.8, fuse_score=True,
    proximity_thresh=0.5, appearance_thresh=0.25, with_reid=False,
    gmc_method="None", model="auto",
)
KPT_ATTACH_IOU = 0.2


class _Det:
    """BOTSORT.update が要求する results 互換オブジェクト。"""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Det":
        d = _Det(self.xyxy[idx], self.conf[idx])
        d.cls = self.cls[idx]
        return d


@dataclass
class TrackedPerson:
    track_id: int
    bbox_xyxy: np.ndarray  # 検出boxを使用（骨格と整合）
    score: float
    kpts: np.ndarray       # (17,3)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class PersonTracker:
    def __init__(self, fps: float):
        from ultralytics.trackers.bot_sort import BOTSORT
        args = SimpleNamespace(**BOTSORT_CFG)
        try:
            self.tracker = BOTSORT(args, frame_rate=max(1, int(round(fps or 30))))
        except TypeError:  # ultralytics 新版は frame_rate 引数なし
            self.tracker = BOTSORT(args)

    def update(self, det: PoseDetections, frame_bgr: np.ndarray) -> list[TrackedPerson]:
        tracks = self.tracker.update(_Det(det.boxes, det.scores), frame_bgr)
        if len(tracks) == 0 or len(det.boxes) == 0:
            return []
        tracks = np.asarray(tracks)
        iou = _iou_matrix(tracks[:, :4].astype(np.float32), det.boxes)
        pairs = sorted(((iou[t, d], t, d) for t in range(iou.shape[0]) for d in range(iou.shape[1])),
                       reverse=True)
        used_t, used_d, out = set(), set(), []
        for v, t, d in pairs:
            if v < KPT_ATTACH_IOU:
                break
            if t in used_t or d in used_d:
                continue
            used_t.add(t)
            used_d.add(d)
            out.append(TrackedPerson(int(tracks[t, 4]), det.boxes[d], float(det.scores[d]), det.kpts[d]))
        return out
