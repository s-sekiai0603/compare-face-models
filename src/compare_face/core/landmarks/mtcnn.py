"""MTCNN（AdaFace 公式デモの face_alignment）による 5 点検出とアライン。

AdaFace の公式リポジトリ（inference.py → face_alignment/align.py）は MTCNN で顔と 5 点を検出し、
96x112 の基準 5 点を 112x112 用に +8px ずらしたもの（= ArcFace テンプレートと同じ座標）へ相似変換して
112x112 にアラインする。MS1MV2 など MTCNN で整列された学習データとも手順が揃う。
重み（pnet/rnet/onet.npy、合計約 3MB）は AdaFace リポジトリから自動取得する。
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PIL import Image

from ..common.weights import LANDMARK_DIR
from .base import AlignResult, FaceAligner, WeightsUnavailable, register
from .insightface_scrfd import estimate_norm
from .mtcnn_net import MTCNNDetector

MTCNN_DIR = LANDMARK_DIR / "mtcnn"
WEIGHT_FILES = ("pnet.npy", "rnet.npy", "onet.npy")
WEIGHTS_BASE_URL = "https://raw.githubusercontent.com/mk-minchul/AdaFace/master/face_alignment/mtcnn_pytorch/src/weights/"
PAD_FRACTION = 0.25  # 顔が全面を占める切り出しの周囲に各辺この割合の黒余白を足す（AdaFace の align.add_padding 相当）
# 公式設定（しきい値 0.6/0.7/0.9）で見つからないときの再試行。監視映像の小さくぼやけた顔では公式設定だと 51 枚中 4 枚しか
# 検出できず、しきい値 0.4 とグレー余白 50% で 20 枚まで拾える（2d106det との 5 点のずれは 2.5px で精度は保たれる）。
# それでも MTCNN は小さい顔に弱く、検出できない画像はリサイズで代替される（件数は条件シートに記録）。
RETRY_THRESHOLDS = (0.4, 0.4, 0.4)
RETRY_PAD_FRACTION = 0.5
RETRY_PAD_FILL = 114


def pad_image(img_bgr: np.ndarray, fraction: float = PAD_FRACTION, fill: int = 0) -> tuple[np.ndarray, int, int]:
    h, w = img_bgr.shape[:2]
    pad_x, pad_y = int(w * fraction), int(h * fraction)
    return cv2.copyMakeBorder(img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(fill,) * 3), pad_x, pad_y


@register
class MTCNNAligner(FaceAligner):
    name = "MTCNN (AdaFace 公式)"
    description = ("AdaFace 公式デモと同じ MTCNN（P/R/O-Net, 原著 Caffe 重み）で 5 点を検出し、112x112 の基準 5 点へ相似変換。"
                   "MS1MV2 の整列手順と同じ。初回は AdaFace リポジトリから重みを自動取得（約 3MB）")

    def __init__(self):
        self.detector: MTCNNDetector | None = None

    def weights_dir(self):
        return MTCNN_DIR

    def weights_ready(self) -> bool:
        return all((MTCNN_DIR / f).is_file() for f in WEIGHT_FILES)

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重み（{', '.join(WEIGHT_FILES)}）が見つかりません。画面を開いたときに自動取得します。\n"
            f"手動で用意する場合は {WEIGHTS_BASE_URL} から 3 ファイルをダウンロードし、次の場所に配置してください。\n   {MTCNN_DIR}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if self.weights_ready():
            return
        import urllib.request
        MTCNN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for f in WEIGHT_FILES:
                dst = MTCNN_DIR / f
                if dst.is_file():
                    continue
                log(f"{self.name} の重みをダウンロードしています: {f}")
                tmp = dst.with_suffix(".npy.part")
                urllib.request.urlretrieve(WEIGHTS_BASE_URL + f, str(tmp))
                if tmp.stat().st_size < 10_000:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"{f} のダウンロード結果が不正です")
                tmp.replace(dst)
        except Exception as e:
            raise WeightsUnavailable(f"{self.manual_instructions()}\n\n（自動取得エラー: {type(e).__name__}: {e}）") from e

    def load(self) -> None:
        if self.detector is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        self.detector = MTCNNDetector(MTCNN_DIR)
        dev = f"GPU ({self.detector.device}, {torch.cuda.get_device_name(0)})" if self.detector.device != "cpu" else "CPU"
        print(f"[推論環境] 顔ランドマーク: デバイス: {dev} / フレームワーク: PyTorch / モデル: {self.name}", flush=True)

    def detect_main_face(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """切り出し内で最もスコアの高い顔の 5 点 (5,2)（切り出し座標）。AdaFace の align_multi(limit=1) と同じ選び方。"""
        self.load()
        for fraction, fill, thresholds in ((PAD_FRACTION, 0, None), (RETRY_PAD_FRACTION, RETRY_PAD_FILL, RETRY_THRESHOLDS)):
            padded, pad_x, pad_y = pad_image(face_bgr, fraction, fill)
            boxes, landmarks = self.detector.detect_faces(Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)), thresholds)
            if len(boxes) > 0:
                lm = landmarks[0]
                return np.array([[lm[j], lm[j + 5]] for j in range(5)], np.float32) - np.array([pad_x, pad_y], np.float32)
        return None

    def align_with_landmarks(self, face_bgr: np.ndarray, out_size: int) -> AlignResult | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        pts = self.detect_main_face(face_bgr)
        if pts is None or not np.all(np.isfinite(pts)):
            return None
        m = estimate_norm(pts, out_size)
        img = cv2.warpAffine(face_bgr, m, (out_size, out_size), borderValue=0.0)
        return AlignResult(img, (pts @ m[:, :2].T + m[:, 2]).astype(np.float32))

    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        r = self.align_with_landmarks(face_bgr, out_size)
        return None if r is None else r.image
