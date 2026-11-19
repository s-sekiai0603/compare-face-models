"""MediaPipe Face Landmarker による頭部姿勢（顔の角度）推定。

Face Landmarker の facial_transformation_matrixes（正準顔 → 画像中の顔の 4x4 変換）の回転部から yaw / pitch / roll [度] を求める。
顔の検出も Face Landmarker 自身が行うので、顔がほぼ全面の切り出しだと検出できないことがある。呼び出し側で余白を足した画像を渡す
（pipeline._pose_input が prefers_loose_crop で作り直す）。検出できなかった画像の角度は NaN。
モデル（face_landmarker.task）は Google の公式配布から取得する（ログイン不要）。

POSE_OPTIONS には登録しない（検出失敗で NaN を返すため、レポートの姿勢表示と同じ扱いにできない。顔サイズ別精度検証の「顔の角度」専用）。
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.weights import POSE_DIR, WeightsUnavailable

MEDIAPIPE_DIR = POSE_DIR / "mediapipe"
MODEL_FILE = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MIN_SIDE = 256   # これより小さい画像は拡大してから渡す（小さい切り出しでは検出しにくい）


def euler_deg_from_transform(m: np.ndarray) -> tuple[float, float, float]:
    """4x4（または 3x3）の変換行列 → (yaw, pitch, roll) [度]。座標系は MediaPipe の正準顔（x 右・y 上・z 手前）で、順序は Y→X→Z。
    符号は 6DRepNet / HopeNet に合わせて反転している（実画像で 6DRepNet と比べ、反転前は yaw の相関が負だった）。"""
    r = np.asarray(m, np.float64)[:3, :3]
    r = r / np.maximum(np.linalg.norm(r, axis=0, keepdims=True), 1e-8)   # スケール成分を除く
    yaw = np.arctan2(r[0, 2], r[2, 2])
    pitch = np.arctan2(-r[1, 2], np.hypot(r[0, 2], r[2, 2]))
    roll = np.arctan2(r[1, 0], r[1, 1])
    return tuple(float(-v) for v in np.degrees([yaw, pitch, roll]))


class MediaPipePoseEstimator:
    """Face Landmarker の重みを管理して推論する。HeadPoseEstimator と同じインターフェース（angles が NaN を返しうる点だけ違う）。"""

    prefers_loose_crop = True  # 検出が要るので、余白の広い切り出しを渡す（pipeline._pose_input）

    def __init__(self):
        self.landmarker = None

    @property
    def name(self) -> str:
        return "MediaPipe Face Landmarker"

    def weights_path(self) -> Path:
        return MEDIAPIPE_DIR / MODEL_FILE

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} のモデル {MODEL_FILE} が見つかりません。自動取得します（約 4MB）。\n"
            f"手動で用意する場合は {MODEL_URL} からダウンロードし、次の場所に配置してください。\n"
            f"   {self.weights_path()}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if self.weights_ready():
            return
        log(f"{self.name} のモデルをダウンロードしています: {MODEL_FILE}")
        self.weights_path().parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=MODEL_FILE, suffix=".part", dir=str(self.weights_path().parent))
        os.close(fd)
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
            os.replace(tmp, self.weights_path())
        except Exception as e:  # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            raise WeightsUnavailable(f"{self.manual_instructions()}\n（自動取得に失敗: {e}）") from e
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.landmarker is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        from mediapipe.tasks.python import BaseOptions, vision
        # 日本語を含むパスを避けるためモデルはバイト列で渡す
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=self.weights_path().read_bytes()),
            running_mode=vision.RunningMode.IMAGE, num_faces=1, output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        print(f"[推論環境] 頭部姿勢: デバイス: CPU / フレームワーク: MediaPipe / モデル: {self.name}", flush=True)

    def angles(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, 3) の (yaw, pitch, roll) [度]。顔を検出できなかった画像は NaN。"""
        import mediapipe as mp
        self.load()
        out = np.full((len(faces_bgr), 3), np.nan, np.float32)
        for i, bgr in enumerate(faces_bgr):
            h, w = bgr.shape[:2]
            if min(h, w) < MIN_SIDE:
                k = MIN_SIDE / float(min(h, w))
                bgr = cv2.resize(bgr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_CUBIC)
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.facial_transformation_matrixes:
                out[i] = euler_deg_from_transform(res.facial_transformation_matrixes[0])
        return out
