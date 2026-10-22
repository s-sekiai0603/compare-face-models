"""顔特徴量抽出モデルの共通インターフェースとレジストリ。

新しいモデルを追加するには FaceFeatureExtractor を継承したクラスを作り、@register を付けて
features/__init__.py で import する。レポート画面の「顔特徴量抽出モデル」に自動で並ぶ。
入力は顔ランドマークモデルでアライン済み（または単純リサイズ済み）の input_size 正方形 BGR 画像。
needs_landmarks=True のモデル（KP-RPE 等）には、アライン後画像内の 5 点座標も渡される。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

from ..common.weights import MODELS_DIR, WeightsUnavailable  # noqa: F401  （他モジュールからも参照される）


class FaceFeatureExtractor(ABC):
    name: str = ""              # 画面・レポートに表示するモデル名
    description: str = ""       # 画面のツールチップ・list-models に表示する説明
    input_size: int = 112       # アライン後の入力正方形サイズ(px)
    embed_dim: int = 512        # 特徴量の次元
    needs_landmarks: bool = False  # True: embed_batch に 5 点ランドマーク（アライン後画像のピクセル座標）が必要
    hidden: bool = False        # True: 画面の選択肢に出さない（CLI の --model では使える）

    @abstractmethod
    def weights_ready(self) -> bool:
        """推論に必要な重みが利用可能か。"""

    def manual_instructions(self) -> str:
        return f"{self.name} の重みを {MODELS_DIR} に配置してください。"

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        """重みを自動取得する。既定は自動取得なし（配置手順を WeightsUnavailable で通知）。"""
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    @abstractmethod
    def load(self) -> None:
        """モデルを読み込む（実行環境をコンソールへ出力すること）。"""

    @abstractmethod
    def embed_batch(self, faces_bgr: list[np.ndarray],
                    landmarks: list[np.ndarray | None] | None = None) -> np.ndarray:
        """BGR 顔画像のリスト -> (N, embed_dim) の L2 正規化済み特徴量。
        landmarks は画像ごとの (5, 2) ピクセル座標（無い画像は None）。needs_landmarks=False のモデルは無視してよい。"""


EXTRACTORS: dict[str, type[FaceFeatureExtractor]] = {}


def register(cls: type[FaceFeatureExtractor]) -> type[FaceFeatureExtractor]:
    EXTRACTORS[cls.name] = cls
    return cls


def get_extractor(name: str) -> FaceFeatureExtractor:
    if name not in EXTRACTORS:
        raise KeyError(f"未登録の顔特徴量抽出モデル: {name}（登録済み: {', '.join(EXTRACTORS)}）")
    return EXTRACTORS[name]()


def visible_extractors() -> dict[str, type[FaceFeatureExtractor]]:
    """画面に並べるモデル（hidden=True を除く）。"""
    return {name: cls for name, cls in EXTRACTORS.items() if not cls.hidden}
