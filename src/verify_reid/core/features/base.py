"""特徴量抽出モデルの共通インターフェースとレジストリ。

新しいモデルを追加するには FeatureExtractor を継承したクラスを作り、
register で登録するだけでレポート画面のラジオボタンに現れる。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class WeightsUnavailable(RuntimeError):
    """重みを取得できない。メッセージにユーザー向けの配置手順を含める。"""


class FeatureExtractor(ABC):
    name: str = ""  # 画面・レポートに表示するモデル名

    @abstractmethod
    def weights_ready(self) -> bool:
        """推論に必要な重みが利用可能か。"""

    @abstractmethod
    def download_weights(self, log: Callable[[str], None] = print) -> None:
        """重みを自動取得する。失敗時は WeightsUnavailable（配置手順つき）を送出。"""

    @abstractmethod
    def load(self) -> None:
        """モデルを読み込む（実行環境をコンソールへ出力すること）。"""

    @abstractmethod
    def embed_batch(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """BGR画像のリスト -> (N, D) の L2 正規化済み特徴量。"""


EXTRACTORS: dict[str, type[FeatureExtractor]] = {}


def register(cls: type[FeatureExtractor]) -> type[FeatureExtractor]:
    EXTRACTORS[cls.name] = cls
    return cls


def get_extractor(name: str) -> FeatureExtractor:
    return EXTRACTORS[name]()
