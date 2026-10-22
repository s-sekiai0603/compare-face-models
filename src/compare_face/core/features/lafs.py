"""LAFS（Landmark-based Facial Self-supervised learning, CVPR 2024, Queen Mary University of London）の
Part fViT による顔特徴量抽出。

元コード・配布: https://github.com/szlbiubiubiu/LAFS_CVPR2024（重みは Google Drive）。
  入力: ArcFace テンプレートでアラインした 112x112 RGB、x/255 - 0.5（[-0.5, 0.5]。/0.5 はしない）。
        ランドマークはモデル内蔵の CNN が 196 点を予測するため、外部のランドマークは不要。
  出力: 元リポジトリの IJB 評価（use_flip_test）と同じく、元画像と左右反転画像の特徴（各 768 次元）を
        連結した 1536 次元を L2 正規化したもの。FP32 推論。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.weights import FEATURE_DIR
from .base import FaceFeatureExtractor, WeightsUnavailable, register

LAFS_DIR = FEATURE_DIR / "lafs"
INPUT_SIZE = 112
EMBED_BATCH = 64
TTA_FLIP_CONCAT = True  # 元画像と反転画像の特徴を連結（元リポジトリの flip test と同じ）
REPO_URL = "https://github.com/szlbiubiubiu/LAFS_CVPR2024"
MIN_FILE_SIZE = 800_000_000  # 配布 .pth は約 900MB（分類層込み）


def preprocess(face_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR 顔画像 -> (3, size, size) float32（RGB, x/255 - 0.5）。"""
    if face_bgr.shape[0] != size or face_bgr.shape[1] != size:
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - 0.5).transpose(2, 0, 1)


class _LAFSBase(FaceFeatureExtractor):
    """Part fViT 系の共通実装。派生クラスは重みファイル名と Google Drive の ID を定義するだけでよい。"""

    weights_file = ""
    gdrive_id = ""
    weights_env = ""
    input_size = INPUT_SIZE
    embed_dim = 1536 if TTA_FLIP_CONCAT else 768
    needs_landmarks = False

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def weights_path(self) -> Path:
        env = os.environ.get(self.weights_env) if self.weights_env else None
        return Path(env) if env else LAFS_DIR / self.weights_file

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重み {self.weights_file} が見つかりません。画面を開いたときに Google Drive から自動取得します"
            "（約 900MB、ネットワーク接続が必要。Google Drive の制限で失敗することがあります）。\n"
            f"手動で用意する場合は {REPO_URL} の README にある WebFace4M の Part fViT チェックポイント"
            f"（https://drive.google.com/file/d/{self.gdrive_id}/view）をダウンロードし、次の場所に配置してください。\n"
            f"   {self.weights_path()}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        dst = self.weights_path()
        if dst.is_file():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"{self.name} の重みを Google Drive からダウンロードしています: {dst.name}（約 900MB）")
        try:
            import gdown
            tmp = dst.with_suffix(dst.suffix + ".part")
            out = gdown.download(id=self.gdrive_id, output=str(tmp), quiet=True)
            if not out or not tmp.is_file() or tmp.stat().st_size < MIN_FILE_SIZE:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("ダウンロード結果が不正です（サイズ不足）")
            tmp.replace(dst)
        except Exception as e:
            raise WeightsUnavailable(f"{self.manual_instructions()}\n\n（自動取得エラー: {type(e).__name__}: {e}）") from e

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        from .lafs_net import PartFViT, load_partfvit_state_dict
        model = PartFViT()
        model.load_state_dict(load_partfvit_state_dict(str(self.weights_path())), strict=True)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.eval().to(self.device)
        dev = f"GPU ({self.device}, {torch.cuda.get_device_name(0)})" if self.device != "cpu" else "CPU"
        print(f"[推論環境] 顔特徴量抽出: デバイス: {dev} / フレームワーク: PyTorch (FP32) / torch {torch.__version__}"
              f" / モデル: {self.name} / 重み: {self.weights_path().name}"
              f" / TTA: {'左右反転の連結(1536次元)' if TTA_FLIP_CONCAT else 'なし'}", flush=True)

    def embed_batch(self, faces_bgr: list[np.ndarray],
                    landmarks: list[np.ndarray | None] | None = None) -> np.ndarray:
        import torch
        self.load()  # landmarks は使わない（モデル内蔵の CNN が予測する）
        feats = []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), EMBED_BATCH):
                arr = np.stack([preprocess(f, self.input_size) for f in faces_bgr[i:i + EMBED_BATCH]])
                x = torch.from_numpy(arr).to(self.device)
                f = self.model(x)
                if TTA_FLIP_CONCAT:
                    f = torch.cat([f, self.model(torch.flip(x, dims=[3]))], dim=1)
                f = torch.nn.functional.normalize(f.float(), dim=1)
                feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats) if feats else np.zeros((0, self.embed_dim), np.float32)


@register
class LAFSPartFViTWebFace4M(_LAFSBase):
    name = "LAFS Part fViT (WebFace4M)"
    description = ("LAFS（自己教師あり事前学習 + WebFace4M で学習）の Part fViT。ViT-Base、ランドマーク予測 CNN 内蔵。"
                   "IJB-C TAR@FAR=1e-4 97.58。初回は Google Drive から自動取得（約 900MB）")
    weights_file = "lafs_partfvit_webface4m.pth"
    gdrive_id = "1BUYm2Bcgp8ZRlBcwOZxiJtWiQAvK2Ujy"
    weights_env = "COMPARE_FACE_LAFS_WEIGHTS"


@register
class PartFViTWebFace4M(_LAFSBase):
    name = "Part fViT (WebFace4M, 事前学習なし)"
    description = ("LAFS と同じ Part fViT を WebFace4M で教師ありのみ学習したベースライン。IJB-C TAR@FAR=1e-4 97.40。"
                   "初回は Google Drive から自動取得（約 900MB）")
    weights_file = "partfvit_webface4m.pth"
    gdrive_id = "16fsYE-j4v6dh7V-_aM0nnU9VdjjlZ1VX"
    weights_env = "COMPARE_FACE_PARTFVIT_WEIGHTS"
    hidden = True  # LAFS 論文の比較用ベースライン。画面には出さない（CLI の --model では使える）
