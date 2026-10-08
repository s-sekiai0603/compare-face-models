"""人物クロップの体軸回転補正（参照元 align_person_crop 相当）。"""
from __future__ import annotations

import math

import cv2
import numpy as np

MAX_ANGLE = 45.0
MIN_KP_CONF = 0.3
MIN_ANGLE = 1.0
PAD_COLOR = (114, 114, 114)
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _mid(kpts: np.ndarray, idxs: tuple[int, int]) -> np.ndarray | None:
    pts = [kpts[i, :2] for i in idxs if kpts[i, 2] >= MIN_KP_CONF]
    return np.mean(pts, axis=0) if pts else None


def estimate_body_angle(kpts: np.ndarray) -> float | None:
    """肩中点-腰中点の軸の垂直からの傾き(度)。時計回りが正。"""
    shoulder = _mid(kpts, (L_SHOULDER, R_SHOULDER))
    hip = _mid(kpts, (L_HIP, R_HIP))
    if shoulder is None or hip is None:
        return None
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    return math.degrees(math.atan2(dx, -dy))


def align_person_crop(crop: np.ndarray, kpts: np.ndarray | None) -> np.ndarray:
    if kpts is None or crop.size == 0:
        return crop
    angle = estimate_body_angle(np.asarray(kpts, dtype=np.float32))
    if angle is None or abs(angle) < MIN_ANGLE or abs(angle) > MAX_ANGLE:
        return crop
    h, w = crop.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(m[0, 0]), abs(m[0, 1])
    new_w = math.ceil(h * sin_a + w * cos_a)
    new_h = math.ceil(h * cos_a + w * sin_a)
    m[0, 2] += (new_w - w) / 2
    m[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(crop, m, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=PAD_COLOR)
