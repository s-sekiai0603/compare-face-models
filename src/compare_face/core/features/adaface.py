"""AdaFace（IR-101, WebFace12M 学習済み .ckpt）による顔特徴量抽出。

論文・公式コード（https://github.com/mk-minchul/AdaFace）どおりの前処理:
  入力サイズ(112x112)へリサイズ(INTER_CUBIC) → BGR のまま → (x/255 - 0.5)/0.5 → IR-101
  → 左右反転 TTA 平均 → L2 正規化（512 次元）。FP32 推論。
チャネル順は公式の inference.py（to_input が RGB を [:, :, ::-1] で BGR にして入力）と、学習データ変換 convert.py
（「validation code uses Cv2 (BGR)」のため意図的にチャネルを入れ替えて保存）に合わせて BGR。CFP-FP 7000 ペアで
BGR 97.66% / RGB 97.47% と差は小さい（学習時の色の拡張で頑健）が、公式どおり BGR にしている。
CLAHE（LAB の L チャネル, clip 2.0）はレポートの `--clahe` オプションがアライン後の画像に掛ける（apply_clahe）。
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ..common.weights import FEATURE_DIR
from .base import FaceFeatureExtractor, WeightsUnavailable, register

ADAFACE_DIR = FEATURE_DIR / "adaface"
INPUT_SIZE = 112
EMBED_BATCH = 128
TTA_HFLIP = True
CLAHE_CLIP_LIMIT = 2.0  # --clahe で使う clip limit（premises-security の ADAFACE_CLAHE_CLIP_LIMIT と同じ）
ADAFACE_URL = "https://github.com/mk-minchul/AdaFace"


def apply_clahe(bgr: np.ndarray, clip_limit: float = CLAHE_CLIP_LIMIT) -> np.ndarray:
    """CLAHE を LAB 色空間の L チャネルに適用する。clip_limit <= 0 で無効。"""
    if clip_limit <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(face_bgr: np.ndarray, size: int = INPUT_SIZE, clahe: bool = False) -> np.ndarray:
    """BGR 顔画像 -> (3, size, size) float32（[-1, 1] 正規化・CHW）。
    BGR のまま入力する（AdaFace 公式 inference.py と同じ）。clahe=True で CLAHE を適用。"""
    if face_bgr.shape[0] != size or face_bgr.shape[1] != size:
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    if clahe:
        face_bgr = apply_clahe(face_bgr)
    return ((face_bgr.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)


def _forward(model, x):
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out  # Backbone は (特徴量, ノルム) を返す


def load_backbone_state_dict(torch, path: Path) -> dict:
    """Lightning 形式の .ckpt から backbone の state_dict を取り出す（"model." プレフィックスを除去）。"""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}


class _AdaFaceBase(FaceFeatureExtractor):
    """AdaFace 系の共通実装。派生クラスは下の属性を定義するだけでよい。"""

    arch = ""           # adaface_net.build_model に渡すアーキテクチャ名
    weights_file = ""   # models/feature/adaface/ に置く重みファイル名
    weights_env = ""    # 重みパスの上書き用環境変数（検証用）
    weights_hint = ""   # 配布元での重みの説明（手動配置の案内用）
    input_size = INPUT_SIZE
    embed_dim = 512

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def weights_path(self) -> Path:
        env = os.environ.get(self.weights_env) if self.weights_env else None
        return Path(env) if env else ADAFACE_DIR / self.weights_file

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重み {self.weights_file} が見つかりません。次の手順で配置してください。\n"
            f"1. {ADAFACE_URL} の Pretrained Models から {self.weights_hint} をダウンロード\n"
            f"2. 次の場所に配置\n   {self.weights_path()}\n"
            "3. 画面を開き直す"
        )

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        from . import adaface_net
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = adaface_net.build_model(self.arch)
        missing, unexpected = model.load_state_dict(load_backbone_state_dict(torch, self.weights_path()),
                                                    strict=False)
        self.model = model.to(self.device).eval()
        dev = (f"GPU ({self.device}, {torch.cuda.get_device_name(0)})" if self.device != "cpu" else "CPU")
        print(f"[推論環境] 顔特徴量抽出: デバイス: {dev} / フレームワーク: PyTorch (FP32)"
              f" / torch {torch.__version__} / モデル: {self.name}({self.arch})"
              f" / 重み: {self.weights_path().name} (missing={len(missing)}, unexpected={len(unexpected)})"
              f" / 入力: BGR", flush=True)

    def embed_batch(self, faces_bgr: list[np.ndarray],
                    landmarks: list[np.ndarray | None] | None = None) -> np.ndarray:
        import torch
        self.load()  # landmarks は使わない（AdaFace IR-101 はキーポイント不要）
        feats = []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), EMBED_BATCH):
                arr = np.stack([preprocess(f, self.input_size) for f in faces_bgr[i:i + EMBED_BATCH]])
                x = torch.from_numpy(arr).to(self.device)
                f = _forward(self.model, x)
                if TTA_HFLIP:
                    f = (f + _forward(self.model, torch.flip(x, dims=[3]))) * 0.5
                f = torch.nn.functional.normalize(f, dim=1)
                feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats) if feats else np.zeros((0, self.embed_dim), np.float32)


@register
class AdaFaceIR101WebFace12M(_AdaFaceBase):
    name = "AdaFace"
    description = "AdaFace IR-101（WebFace12M 学習済み, 512 次元）。公式コードどおりの前処理（BGR 入力, CLAHE なし）"
    arch = "ir_101"
    weights_file = "adaface_ir101_webface12m.ckpt"
    weights_env = "COMPARE_FACE_ADAFACE_WEIGHTS"
    weights_hint = "adaface_ir101_webface12m.ckpt（IR-101 / WebFace12M）"
