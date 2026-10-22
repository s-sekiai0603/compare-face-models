"""HopeNet（Fine-Grained Head Pose Estimation Without Keypoints, Ruiz et al. 2018）による頭部姿勢推定。

元コード: https://github.com/natanielruiz/deep-head-pose（code/hopenet.py）。ResNet-50 の末尾に yaw / pitch / roll
それぞれ 66 ビンの分類層を持ち、softmax の期待値から角度（度）を復元する。重みは公式の Google Drive 配布を
ミラーした Hugging Face の public-data/Hopenet から取得する（ログイン不要）。

  入力: 顔の切り出しを 224x224 にリサイズし ImageNet の平均・標準偏差で正規化した RGB
  出力: (yaw, pitch, roll) [度]。FSPFM 用には分類層の手前の 2048 次元（avgpool 後）も返す。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.weights import POSE_DIR, WeightsUnavailable

HOPENET_DIR = POSE_DIR / "hopenet"
HF_REPO = "public-data/Hopenet"
INPUT_SIZE = 224
NUM_BINS = 66
BATCH = 64
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# 配布重み。alpha はビン分類と回帰の損失の重み（論文参照）。robust は画質劣化・ブレに強く、実映像向けに公式が推奨
VARIANTS = {
    "robust_alpha1": "hopenet_robust_alpha1.pkl",  # 300W-LP, alpha 1, robust to image quality
    "alpha2": "hopenet_alpha2.pkl",                # 300W-LP, alpha 2（FSPFM が使用）
    "alpha1": "hopenet_alpha1.pkl",                # 300W-LP, alpha 1
}


def build_hopenet():
    """公式 Hopenet（ResNet-50 Bottleneck [3,4,6,3] + 66 ビン x 3）と同じキー構造のモジュール。"""
    import torch
    import torch.nn as nn
    from torchvision.models import resnet

    class Hopenet(nn.Module):
        def __init__(self, block=resnet.Bottleneck, layers=(3, 4, 6, 3), num_bins: int = NUM_BINS):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
            self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
            self.fc_roll = nn.Linear(512 * block.expansion, num_bins)
            self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)  # 元コードの未使用層（キー互換のため保持）

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion))
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
            return nn.Sequential(*layers)

        def features(self, x):
            """avgpool 後の 2048 次元（FSPFM の姿勢入力）。"""
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.avgpool(x).flatten(1)

        def forward(self, x):
            f = self.features(x)
            return self.fc_yaw(f), self.fc_pitch(f), self.fc_roll(f), f

    return Hopenet()


def preprocess(face_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR 顔画像 -> (3, 224, 224) float32（RGB, ImageNet 正規化）。"""
    if face_bgr.shape[0] != size or face_bgr.shape[1] != size:
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


def decode_angles(logits: np.ndarray) -> np.ndarray:
    """(N, 66) のビン分類出力 -> 角度[度]（softmax の期待値 x 3 - 99）。"""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return (p * np.arange(NUM_BINS, dtype=np.float32)).sum(axis=1) * 3.0 - 99.0


class HeadPoseEstimator:
    """HopeNet の重みを管理して推論する。variant は VARIANTS のキー。"""

    def __init__(self, variant: str = "robust_alpha1"):
        if variant not in VARIANTS:
            raise KeyError(f"未知の HopeNet 重み: {variant}（{', '.join(VARIANTS)}）")
        self.variant = variant
        self.model = None
        self.device = "cpu"

    @property
    def name(self) -> str:
        return f"HopeNet ({self.variant})"

    def weights_path(self) -> Path:
        return HOPENET_DIR / "models" / VARIANTS[self.variant]

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重み {VARIANTS[self.variant]} が見つかりません。画面を開いたときに自動取得します（約 96MB）。\n"
            f"手動で用意する場合は https://huggingface.co/{HF_REPO}/tree/main/models からダウンロードし、次の場所に配置してください。\n"
            f"   {self.weights_path()}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if self.weights_ready():
            return
        from huggingface_hub import hf_hub_download
        log(f"{self.name} の重みをダウンロードしています: {VARIANTS[self.variant]}")
        hf_hub_download(HF_REPO, f"models/{VARIANTS[self.variant]}", local_dir=str(HOPENET_DIR))
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        model = build_hopenet()
        model.load_state_dict(torch.load(str(self.weights_path()), map_location="cpu", weights_only=True), strict=True)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.eval().to(self.device)
        dev = f"GPU ({self.device}, {torch.cuda.get_device_name(0)})" if self.device != "cpu" else "CPU"
        print(f"[推論環境] 頭部姿勢: デバイス: {dev} / フレームワーク: PyTorch / モデル: {self.name}", flush=True)

    def _run(self, faces_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """-> (角度 (N,3) [yaw, pitch, roll] 度, 埋め込み (N,2048))。"""
        import torch
        self.load()
        angles, embs = [], []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), BATCH):
                x = torch.from_numpy(np.stack([preprocess(f) for f in faces_bgr[i:i + BATCH]])).to(self.device)
                yaw, pitch, roll, f = self.model(x)
                angles.append(np.stack([decode_angles(t.float().cpu().numpy()) for t in (yaw, pitch, roll)], axis=1))
                embs.append(f.float().cpu().numpy().astype(np.float32))
        if not angles:
            return np.zeros((0, 3), np.float32), np.zeros((0, 2048), np.float32)
        return np.concatenate(angles).astype(np.float32), np.concatenate(embs)

    def angles(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, 3) の (yaw, pitch, roll) [度]。"""
        return self._run(faces_bgr)[0]

    def embed(self, faces_bgr: list[np.ndarray]) -> np.ndarray:
        """(N, 2048) の姿勢埋め込み（FSPFM の入力）。"""
        return self._run(faces_bgr)[1]


# 画面の選択肢（POSE_OPTIONS / get_pose_estimator）は core/pose/__init__.py のレジストリにある
