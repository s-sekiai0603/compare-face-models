"""顔ランドマークモデル（アライン）の共通インターフェースとレジストリ。

特徴量抽出の直前に、保存済みの顔クロップ(_face.jpg)を各モデルの入力サイズの正方形に整える。
「なし」はランドマークを使わずリサイズするだけ（premises-security でランドマークモデルが無いときと同じ）。
ランドマークを持つモデルは align_with_landmarks で、アライン後画像内の 5 点座標も返す
（KP-RPE のようにキーポイントを必要とする特徴量モデルがこれを使う）。

新しいランドマークモデルを追加するには FaceAligner を継承したクラスを作り、@register を付けて
landmarks/__init__.py で import する。レポート画面の「顔ランドマークモデル」に自動で並ぶ。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.face_template import ARCFACE_TEMPLATE_5PTS, template_5pts  # noqa: F401
from ..common.weights import MODELS_DIR, WeightsUnavailable  # noqa: F401  （同じ例外を共有する）


@dataclass
class AlignResult:
    image: np.ndarray                 # (out_size, out_size, 3) BGR
    landmarks: np.ndarray | None = None  # (5, 2) アライン後画像のピクセル座標 [左目, 右目, 鼻先, 左口角, 右口角]。無ければ None


class FaceAligner(ABC):
    name: str = ""         # 画面・レポートに表示する名前
    description: str = ""

    def weights_ready(self) -> bool:
        return True

    def manual_instructions(self) -> str:
        return ""

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        """重みを自動取得する。既定は自動取得なし（配置手順を WeightsUnavailable で通知）。"""
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        """モデルを読み込む（実行環境をコンソールへ出力すること）。"""

    @abstractmethod
    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        """顔クロップ(BGR) -> (out_size, out_size, 3) のアライン済み画像。失敗時は None。"""

    def align_with_landmarks(self, face_bgr: np.ndarray, out_size: int) -> AlignResult | None:
        """アライン済み画像と、その画像内の 5 点座標。ランドマークを持たないモデルは landmarks=None。"""
        img = self.align(face_bgr, out_size)
        return None if img is None else AlignResult(img, None)


ALIGNERS: dict[str, type[FaceAligner]] = {}


def register(cls: type[FaceAligner]) -> type[FaceAligner]:
    ALIGNERS[cls.name] = cls
    return cls


def get_aligner(name: str) -> FaceAligner:
    if name not in ALIGNERS:
        raise KeyError(f"未登録の顔ランドマークモデル: {name}（登録済み: {', '.join(ALIGNERS)}）")
    return ALIGNERS[name]()


@register
class NoAlign(FaceAligner):
    name = "なし"
    description = "ランドマークを使わず、顔クロップをそのままリサイズ(INTER_CUBIC)する"

    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        if face_bgr.shape[0] == out_size and face_bgr.shape[1] == out_size:
            return face_bgr
        return cv2.resize(face_bgr, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
