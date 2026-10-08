"""OSNet(osnet_x1_0) による外見特徴量抽出（参照元 Osnet / preprocess_osnet_chw 相当）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .base import FeatureExtractor, WeightsUnavailable, register

MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
WEIGHTS_ENV = "VERIFY_REID_OSNET_WEIGHTS"  # 重みパスの上書き（検証用）
WEIGHTS_FILE = "osnet_x1_0.pth"
# deep-person-reid Model Zoo: osnet_x1_0 (Market1501, softmax)
GDRIVE_ID = "1vduhq5DpN2q1g4fYEZfPI17MJeh9qdrh"
MODEL_ZOO_URL = "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO"
INPUT_H, INPUT_W = 256, 128
MIN_SHORT_SIDE = 48
PAD_COLOR = (114, 114, 114)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EMBED_BATCH = 128
TTA_HFLIP = True


def weights_path() -> Path:
    env = os.environ.get(WEIGHTS_ENV)
    return Path(env) if env else MODELS_DIR / WEIGHTS_FILE


def manual_instructions() -> str:
    return (
        "OSNet の重みを自動取得できませんでした。次の手順で手動配置してください。\n"
        f"1. {MODEL_ZOO_URL} を開き、osnet_x1_0 (Market1501 / softmax) の重みをダウンロード\n"
        f"2. ファイル名を {WEIGHTS_FILE} に変更し、次の場所に配置\n"
        f"   {MODELS_DIR / WEIGHTS_FILE}\n"
        "3. 画面を開き直す"
    )


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """短辺補正 -> letterbox(256x128) -> RGB -> 正規化 -> CHW。"""
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    if short < MIN_SHORT_SIDE:
        s = MIN_SHORT_SIDE / short
        img_bgr = cv2.resize(img_bgr, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_LINEAR)
        h, w = img_bgr.shape[:2]
    scale = min(INPUT_H / h, INPUT_W / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_H, INPUT_W, 3), PAD_COLOR, np.uint8)
    top, left = (INPUT_H - nh) // 2, (INPUT_W - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


@register
class OsnetExtractor(FeatureExtractor):
    name = "OSNet"

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def weights_ready(self) -> bool:
        return weights_path().is_file()

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        dst = weights_path()
        if dst.is_file():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"OSNet の重みをダウンロードしています: {dst.name}")
        try:
            import gdown
            tmp = dst.with_suffix(".part")
            out = gdown.download(id=GDRIVE_ID, output=str(tmp), quiet=True)
            if not out or not tmp.is_file() or tmp.stat().st_size < 1_000_000:
                raise RuntimeError("ダウンロード結果が不正です")
            tmp.replace(dst)
        except Exception as e:
            raise WeightsUnavailable(f"{manual_instructions()}\n\n（自動取得エラー: {type(e).__name__}）") from e

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(manual_instructions())
        import torch
        import torchreid
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = torchreid.models.build_model(name="osnet_x1_0", num_classes=751,
                                             loss="softmax", pretrained=False)
        torchreid.utils.load_pretrained_weights(model, str(weights_path()))
        self.model = model.to(self.device).eval()
        dev = (f"GPU ({self.device}, {torch.cuda.get_device_name(0)})"
               if self.device != "cpu" else "CPU")
        print(f"[推論環境] 特徴量抽出: デバイス: {dev} / フレームワーク: PyTorch"
              f" / torch {torch.__version__} / モデル: OSNet(osnet_x1_0)", flush=True)

    def embed_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        import torch
        self.load()
        feats = []
        with torch.no_grad():
            for i in range(0, len(images_bgr), EMBED_BATCH):
                arr = np.stack([preprocess(im) for im in images_bgr[i:i + EMBED_BATCH]])
                x = torch.from_numpy(arr).to(self.device)
                f = self.model(x)
                if TTA_HFLIP:
                    f = (f + self.model(torch.flip(x, dims=[3]))) / 2
                f = torch.nn.functional.normalize(f, dim=1)
                feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)
