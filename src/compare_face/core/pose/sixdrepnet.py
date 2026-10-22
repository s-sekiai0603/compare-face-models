"""6DRepNet（6D Rotation Representation for Unconstrained Head Pose Estimation, Hempel et al. ICIP 2022）による頭部姿勢推定。

元コード: https://github.com/thohemp/6DRepNet（MIT。pip の sixdrepnet 0.1.6 の model.py / backbone/repvgg.py / utils.py を移植）。
RepVGG-B1g2 を再パラメータ化した推論構造（各ブロックが 3x3 畳み込み 1 本 + ReLU）の末尾に 6 次元回転表現の回帰層を持ち、
6 次元 -> Gram-Schmidt で回転行列 -> オイラー角（度）に変換する。重みは公式配布の 6DRepNet_300W_LP_AFLW2000.pth
（300W-LP 学習、推論用に変換済み）を作者の大学（OVGU）のクラウドから取得する。

  入力: 顔の切り出しを短辺 224 にリサイズして中央 224x224 を切り出し（公式の Resize(224) + CenterCrop(224) と同じ）、
        ImageNet の平均・標準偏差で正規化した RGB
  出力: (yaw, pitch, roll) [度]。符号は 300W-LP の定義で HopeNet と同じ（yaw 負 = 画面右向き）

学習時のクロップ（300W-LP: 顔枠の横 0.6k・上 2k・下 0.6k、k=0.2〜0.4 の余白）は当ツールの顔切り出し（余白 10〜30%）より広いため、
prefers_loose_crop=True とし、レポートでは全身画像と顔枠から余白の広い切り出し（loose_face_crop）を作り直して入力する
（全身画像が無い場合は顔切り出しをそのまま使う）。PoC7 映像での比較では HopeNet との yaw 差の中央値が 29° -> 18° に縮まった。
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

SIXDREPNET_DIR = POSE_DIR / "6drepnet"
WEIGHTS_FILE = "6DRepNet_300W_LP_AFLW2000.pth"
WEIGHTS_URL = "https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth"
INPUT_SIZE = 224
BATCH = 64
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# RepVGG-B1g2: ステージごとのブロック数と出力チャネル（width_multiplier [2,2,2,4]）。偶数番目のブロックは groups=2
NUM_BLOCKS = (4, 6, 16, 1)
WIDTHS = (128, 256, 512, 2048)
G2_LAYERS = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26}


def build_sixdrepnet():
    """公式 SixDRepNet(backbone='RepVGG-B1g2', deploy=True) と同じキー構造（layer0..4 / rbr_reparam / linear_reg）のモジュール。"""
    import torch.nn as nn

    class RepVGGBlock(nn.Module):  # deploy 構造（rbr_dense / rbr_1x1 / identity を 1 本に畳み込み済み）
        def __init__(self, cin: int, cout: int, stride: int, groups: int):
            super().__init__()
            self.rbr_reparam = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)
            self.nonlinearity = nn.ReLU()

        def forward(self, x):
            return self.nonlinearity(self.rbr_reparam(x))

    class SixDRepNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = RepVGGBlock(3, 64, stride=2, groups=1)
            cin, idx = 64, 1
            stages = []
            for planes, n in zip(WIDTHS, NUM_BLOCKS):
                blocks = []
                for i in range(n):
                    blocks.append(RepVGGBlock(cin, planes, stride=2 if i == 0 else 1, groups=2 if idx in G2_LAYERS else 1))
                    cin, idx = planes, idx + 1
                stages.append(nn.Sequential(*blocks))
            self.layer1, self.layer2, self.layer3, self.layer4 = stages
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.linear_reg = nn.Linear(WIDTHS[-1], 6)

        def forward(self, x):
            x = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))
            return self.linear_reg(self.gap(x).flatten(1))  # (N, 6) の 6D 回転表現

    return SixDRepNet()


def preprocess(face_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR 顔画像 -> (3, 224, 224) float32。短辺を size にリサイズして中央を切り出し、RGB・ImageNet 正規化。"""
    h, w = face_bgr.shape[:2]
    scale = size / float(min(h, w))
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    if (nw, nh) != (w, h):
        face_bgr = cv2.resize(face_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    rgb = cv2.cvtColor(face_bgr[y0:y0 + size, x0:x0 + size], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


LOOSE_SIDE, LOOSE_TOP, LOOSE_BOTTOM = 0.25, 0.6, 0.25  # 学習クロップ（横 0.6k, 上 2k, 下 0.6k, k=0.2〜0.4）の中間


def loose_face_crop(body_bgr: np.ndarray | None, person_box, face_box,
                    side: float = LOOSE_SIDE, top: float = LOOSE_TOP, bottom: float = LOOSE_BOTTOM) -> np.ndarray | None:
    """全身画像（フレーム座標 person_box を切り出したもの）と顔枠（フレーム座標）から、余白の広い顔の切り出しを作る。
    全身画像の外に出る分は端の画素を複製する。材料が無い・枠が不正なら None。"""
    if body_bgr is None or body_bgr.size == 0 or person_box is None or face_box is None:
        return None
    try:
        bx1, by1 = float(person_box[0]), float(person_box[1])
        fx1, fy1, fx2, fy2 = (float(face_box[0]) - bx1, float(face_box[1]) - by1,
                              float(face_box[2]) - bx1, float(face_box[3]) - by1)
    except (TypeError, ValueError, IndexError):
        return None
    w, h = fx2 - fx1, fy2 - fy1
    if w <= 1 or h <= 1:
        return None
    x1, x2 = int(round(fx1 - side * w)), int(round(fx2 + side * w))
    y1, y2 = int(round(fy1 - top * h)), int(round(fy2 + bottom * h))
    bh, bw = body_bgr.shape[:2]
    pad = max(0, -x1, -y1, x2 - bw, y2 - bh)
    if pad:
        body_bgr = cv2.copyMakeBorder(body_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, x2, y1, y2 = x1 + pad, x2 + pad, y1 + pad, y2 + pad
    crop = body_bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def rotation_matrix_from_6d(x6: np.ndarray) -> np.ndarray:
    """(N, 6) の 6D 表現 -> (N, 3, 3) 回転行列（元コードの compute_rotation_matrix_from_ortho6d。列が x, y, z 軸）。"""
    x6 = np.asarray(x6, np.float64)
    x = x6[:, :3] / np.maximum(np.linalg.norm(x6[:, :3], axis=1, keepdims=True), 1e-8)
    z = np.cross(x, x6[:, 3:])
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2)


def euler_deg_from_matrix(r: np.ndarray) -> np.ndarray:
    """(N, 3, 3) 回転行列 -> (N, 3) の (yaw, pitch, roll) [度]（元コードの compute_euler_angles_from_rotation_matrices）。"""
    r = np.asarray(r, np.float64)
    sy = np.sqrt(r[:, 0, 0] ** 2 + r[:, 1, 0] ** 2)
    singular = sy < 1e-6
    pitch = np.where(singular, np.arctan2(-r[:, 1, 2], r[:, 1, 1]), np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    yaw = np.arctan2(-r[:, 2, 0], sy)
    roll = np.where(singular, 0.0, np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    return np.degrees(np.stack([yaw, pitch, roll], axis=1)).astype(np.float32)


class SixDRepNetEstimator:
    """6DRepNet の重みを管理して推論する。HeadPoseEstimator（HopeNet）と同じインターフェース。"""

    prefers_loose_crop = True  # レポートでは loose_face_crop で作った余白の広い切り出しを渡す（pipeline.estimate_poses）

    def __init__(self):
        self.model = None
        self.device = "cpu"

    @property
    def name(self) -> str:
        return "6DRepNet (RepVGG-B1g2, 300W-LP)"

    def weights_path(self) -> Path:
        return SIXDREPNET_DIR / WEIGHTS_FILE

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重み {WEIGHTS_FILE} が見つかりません。画面を開いたときに自動取得します（約 150MB）。\n"
            f"手動で用意する場合は {WEIGHTS_URL} （公式 README の配布先）からダウンロードし、次の場所に配置してください。\n"
            f"   {self.weights_path()}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if self.weights_ready():
            return
        log(f"{self.name} の重みをダウンロードしています: {WEIGHTS_FILE}")
        self.weights_path().parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=WEIGHTS_FILE, suffix=".part", dir=str(self.weights_path().parent))
        os.close(fd)
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, tmp)
            os.replace(tmp, self.weights_path())
        except Exception as e:  # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            raise WeightsUnavailable(f"{self.manual_instructions()}\n（自動取得に失敗: {e}）") from e
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        model = build_sixdrepnet()
        model.load_state_dict(torch.load(str(self.weights_path()), map_location="cpu", weights_only=True), strict=True)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.eval().to(self.device)
        dev = f"GPU ({self.device}, {torch.cuda.get_device_name(0)})" if self.device != "cpu" else "CPU"
        print(f"[推論環境] 頭部姿勢: デバイス: {dev} / フレームワーク: PyTorch / モデル: {self.name}", flush=True)

    def angles(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, 3) の (yaw, pitch, roll) [度]。"""
        import torch
        self.load()
        out = []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), BATCH):
                x = torch.from_numpy(np.stack([preprocess(f) for f in faces_bgr[i:i + BATCH]])).to(self.device)
                out.append(euler_deg_from_matrix(rotation_matrix_from_6d(self.model(x).float().cpu().numpy())))
        return np.concatenate(out) if out else np.zeros((0, 3), np.float32)
