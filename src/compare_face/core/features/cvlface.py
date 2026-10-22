"""CVLFace（ViT + KP-RPE + AdaFace）による顔特徴量抽出。

Hugging Face の minchul/cvlface_adaface_vit_base_kprpe_* を models/feature/cvlface/ に取得し、
transformers を使わず直接読み込む（core/common/cvlface_repo.py）。
  入力: 112x112 RGB, (x/255 - 0.5)/0.5 正規化。KP-RPE は 5 点キーポイント（アライン後画像内の
        座標を [0, 1] に正規化した (N, 5, 2)）が必須。
  出力: 512 次元（モデル出力は未正規化なのでここで L2 正規化）。左右反転 TTA 平均あり
        （反転時はキーポイントも鏡像にして左右を入れ替える）。FP32 推論。
ランドマークが無い画像には ArcFace テンプレートの標準 5 点を代用する（精度は落ちる）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.cvlface_repo import (
    describe_device, download_repo, import_package, is_downloaded, load_conf, local_dir,
    manual_instructions, torch_device, without_extension_build,
)
from ..common.face_template import template_5pts
from .base import FaceFeatureExtractor, WeightsUnavailable, register

PACKAGE = "models"
INPUT_SIZE = 112
EMBED_BATCH = 128
TTA_HFLIP = True
_MIRROR_ORDER = [1, 0, 2, 4, 3]  # 左右反転後の [左目, 右目, 鼻, 左口角, 右口角]


def preprocess(face_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR 顔画像 -> (3, size, size) float32（RGB, [-1, 1]）。"""
    if face_bgr.shape[0] != size or face_bgr.shape[1] != size:
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)


def normalize_keypoints(landmarks: list[np.ndarray | None] | None, n: int, size: int = INPUT_SIZE) -> tuple[np.ndarray, int]:
    """画像ごとの (5, 2) ピクセル座標 -> (n, 5, 2) の [0, 1] 正規化座標。None はテンプレートで代用。
    戻り値は (座標, テンプレートで代用した件数)。"""
    out = np.zeros((n, 5, 2), np.float32)
    missing = 0
    tmpl = template_5pts(size)
    for i in range(n):
        lm = None if landmarks is None else landmarks[i]
        if lm is None or not np.all(np.isfinite(lm)):
            lm = tmpl
            missing += 1
        out[i] = np.asarray(lm, np.float32).reshape(5, 2) / float(size)
    return out, missing


def mirror_keypoints(kp: np.ndarray) -> np.ndarray:
    """左右反転した画像に対応するキーポイント（x -> 1-x、左右の目・口角を入れ替え）。"""
    m = np.asarray(kp, np.float32)[:, _MIRROR_ORDER].copy()
    m[..., 0] = 1.0 - m[..., 0]
    return m


class _CVLFaceBase(FaceFeatureExtractor):
    """CVLFace 系の共通実装。派生クラスは repo_id と名前を定義するだけでよい。"""

    repo_id = ""
    needs_landmarks = True
    input_size = INPUT_SIZE
    embed_dim = 512

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def local_dir(self) -> Path:
        return local_dir(self.repo_id)

    def weights_ready(self) -> bool:
        return is_downloaded(self.repo_id, PACKAGE)

    def manual_instructions(self) -> str:
        return manual_instructions(self.name, self.repo_id)

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if not self.weights_ready():
            download_repo(self.repo_id, log)
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        d = self.local_dir()
        with without_extension_build():
            pkg = import_package(d / PACKAGE, f"cvlface_{d.name}_{PACKAGE}")
            conf = load_conf(d / "pretrained_model" / "model.yaml")
            model = pkg.get_model(conf)
        model.load_state_dict_from_path(str(d / "pretrained_model" / "model.pt"))
        self.device = torch_device()
        self.model = model.eval().to(self.device)
        print(f"[推論環境] 顔特徴量抽出: デバイス: {describe_device()} / フレームワーク: PyTorch (FP32)"
              f" / torch {torch.__version__} / モデル: {self.name} ({self.repo_id})"
              f" / TTA: {'左右反転' if TTA_HFLIP else 'なし'}", flush=True)

    def embed_batch(self, faces_bgr: list[np.ndarray],
                    landmarks: list[np.ndarray | None] | None = None) -> np.ndarray:
        import torch
        self.load()
        feats = []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), EMBED_BATCH):
                chunk = faces_bgr[i:i + EMBED_BATCH]
                arr = np.stack([preprocess(f, self.input_size) for f in chunk])
                kp, _ = normalize_keypoints(None if landmarks is None else landmarks[i:i + EMBED_BATCH],
                                            len(chunk), self.input_size)
                x = torch.from_numpy(arr).to(self.device)
                k = torch.from_numpy(kp).to(self.device)
                f = self.model(x, k)
                if TTA_HFLIP:
                    f = (f + self.model(torch.flip(x, dims=[3]), torch.from_numpy(mirror_keypoints(kp)).to(self.device))) * 0.5
                f = torch.nn.functional.normalize(f.float(), dim=1)
                feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats) if feats else np.zeros((0, self.embed_dim), np.float32)


@register
class CVLFaceKPRPEWebFace12M(_CVLFaceBase):
    name = "CVLFace KP-RPE ViT-B (WebFace12M)"
    description = "CVLFace: ViT-Base + KP-RPE + AdaFace（WebFace12M 学習, 512 次元）。5 点ランドマーク必須。初回は Hugging Face から自動取得"
    repo_id = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"


@register
class CVLFaceKPRPEWebFace4M(_CVLFaceBase):
    name = "CVLFace KP-RPE ViT-B (WebFace4M)"
    description = "CVLFace: ViT-Base + KP-RPE + AdaFace（WebFace4M 学習, 512 次元）。5 点ランドマーク必須。初回は Hugging Face から自動取得"
    repo_id = "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
    hidden = True  # WebFace12M 版があるため画面には出さない（CLI の --model では使える）
