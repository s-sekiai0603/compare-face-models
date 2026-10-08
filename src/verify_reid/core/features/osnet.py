"""OSNet 系(osnet_x1_0 / osnet_ain_x1_0) による外見特徴量抽出（参照元 Osnet / preprocess_osnet_chw 相当）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .base import FeatureExtractor, WeightsUnavailable, register

MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
MODEL_ZOO_URL = "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO"
INPUT_H, INPUT_W = 256, 128
MIN_SHORT_SIDE = 48
PAD_COLOR = (114, 114, 114)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EMBED_BATCH = 128
TTA_HFLIP = True


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


class _OsnetBase(FeatureExtractor):
    """OSNet 系の共通実装。派生クラスは下の属性を定義するだけでよい。"""

    arch = ""          # torchreid のモデル名
    weights_file = ""  # models/ に置く重みファイル名
    weights_env = ""   # 重みパスの上書き用環境変数（検証用）
    gdrive_id = ""     # 自動取得に使う Google Drive ID
    weights_hint = ""  # Model Zoo 上での重みの説明（手動配置の案内用）
    num_classes = 751  # 分類層は特徴量抽出に使わない（形状不一致の層は読込時にスキップされる）

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def weights_path(self) -> Path:
        env = os.environ.get(self.weights_env)
        return Path(env) if env else MODELS_DIR / self.weights_file

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重みを自動取得できませんでした。次の手順で手動配置してください。\n"
            f"1. {MODEL_ZOO_URL} を開き、{self.arch} ({self.weights_hint}) の重みをダウンロード\n"
            f"2. ファイル名を {self.weights_file} に変更し、次の場所に配置\n"
            f"   {MODELS_DIR / self.weights_file}\n"
            "3. 画面を開き直す"
        )

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        dst = self.weights_path()
        if dst.is_file():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"{self.name} の重みをダウンロードしています: {dst.name}")
        try:
            import gdown
            tmp = dst.with_suffix(".part")
            out = gdown.download(id=self.gdrive_id, output=str(tmp), quiet=True)
            if not out or not tmp.is_file() or tmp.stat().st_size < 1_000_000:
                raise RuntimeError("ダウンロード結果が不正です")
            tmp.replace(dst)
        except Exception as e:
            raise WeightsUnavailable(f"{self.manual_instructions()}\n\n（自動取得エラー: {type(e).__name__}）") from e

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        import torchreid
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = torchreid.models.build_model(name=self.arch, num_classes=self.num_classes,
                                             loss="softmax", pretrained=False)
        torchreid.utils.load_pretrained_weights(model, str(self.weights_path()))
        self.model = model.to(self.device).eval()
        dev = (f"GPU ({self.device}, {torch.cuda.get_device_name(0)})"
               if self.device != "cpu" else "CPU")
        print(f"[推論環境] 特徴量抽出: デバイス: {dev} / フレームワーク: PyTorch"
              f" / torch {torch.__version__} / モデル: {self.name}({self.arch})", flush=True)

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


@register
class OsnetExtractor(_OsnetBase):
    name = "OSNet"
    arch = "osnet_x1_0"
    weights_file = "osnet_x1_0.pth"
    weights_env = "VERIFY_REID_OSNET_WEIGHTS"
    gdrive_id = "1vduhq5DpN2q1g4fYEZfPI17MJeh9qdrh"  # Model Zoo: osnet_x1_0 (Market1501, softmax)
    weights_hint = "Market1501 / softmax"


@register
class OsnetAinExtractor(_OsnetBase):
    name = "OSNet-AIN"
    arch = "osnet_ain_x1_0"
    weights_file = "osnet_ain_x1_0.pth"
    weights_env = "VERIFY_REID_OSNET_AIN_WEIGHTS"
    # torchreid が osnet_ain_x1_0 の事前学習重みとして参照している ID（ImageNet 事前学習）
    gdrive_id = "1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo"
    weights_hint = "ドメイン汎化向けの学習済み重み（MSMT17 等）"
