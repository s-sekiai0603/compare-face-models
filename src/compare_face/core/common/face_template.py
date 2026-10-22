"""ArcFace 5 点テンプレート（112x112 基準）: 左目, 右目, 鼻先, 左口角, 右口角。
ランドマークモデルのアライン先と、ランドマークが無いときの代用座標に使う（依存なしのモジュール）。"""
from __future__ import annotations

import numpy as np

ARCFACE_TEMPLATE_5PTS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def template_5pts(out_size: int) -> np.ndarray:
    """out_size 正方形に合わせた ArcFace 5 点テンプレート（ピクセル座標）。"""
    return ARCFACE_TEMPLATE_5PTS if out_size == 112 else ARCFACE_TEMPLATE_5PTS * (float(out_size) / 112.0)


MIRROR_ORDER_5PTS = [1, 0, 2, 4, 3]  # 左右反転後の並び（左目<->右目, 左口角<->右口角）


def mirror_5pts(pts: np.ndarray, width: int) -> np.ndarray:
    """幅 width の画像を左右反転（cv2.flip(img, 1)）したときの 5 点ピクセル座標。x -> width-1-x、左右の点を入れ替える。"""
    m = np.asarray(pts, np.float32)[MIRROR_ORDER_5PTS].copy()
    m[:, 0] = (width - 1) - m[:, 0]
    return m
