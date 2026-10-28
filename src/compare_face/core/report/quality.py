"""クエリ画像の品質スコアと自動選別。

品質ノルム: AdaFace IR-101 の正規化前の特徴量ノルム。AdaFace 損失で学習したモデルは、ぼけ・横顔・遮蔽など
画質が低いほどノルムが小さくなる（論文がこの性質を適応マージンに使っている）。識別に使うモデルに関係なく常に
AdaFace で測り、しきい値を 1 つに揃える（CVLFace のノルムも同じ傾向だが分布が圧縮されていて値が合わない）。
監視映像の比較対象 2,000 枚の標本では中央値 20.4、下位 13% が 17 未満。横顔は低く出るので、クエリ選別の
しきい値は「明らかに悪い」水準（17 前後）にとどめる。顔の大きさとは無関係（相関 -0.05）なので、小さい顔を
外したいときは顔幅のしきい値を使う。

自動選別はクエリ画像にだけ適用する。比較対象に適用すると、小さい顔しか無い追跡ID（人物）がまるごと消えるため。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..features.adaface import AdaFaceIR101WebFace12M, preprocess
from .gallery import GalleryEntry

NORM_BATCH = 64


@dataclass
class QueryFilter:
    """クエリ画像の自動選別しきい値。0 はその条件を使わない。"""
    min_face_conf: float = 0.0   # 顔検出（YOLOv8-Face）の信頼度
    min_norm: float = 0.0        # 品質ノルム（AdaFace の特徴量ノルム）
    min_face_width: float = 0.0  # 検出した顔枠の幅(px)

    def enabled(self) -> bool:
        return self.min_face_conf > 0 or self.min_norm > 0 or self.min_face_width > 0

    def describe(self) -> str:
        parts = []
        if self.min_face_conf > 0:
            parts.append(f"顔信頼度 >= {self.min_face_conf:g}")
        if self.min_norm > 0:
            parts.append(f"品質ノルム >= {self.min_norm:g}")
        if self.min_face_width > 0:
            parts.append(f"顔幅 >= {self.min_face_width:g}px")
        return " / ".join(parts) if parts else "指定なし"

    def reason(self, e: GalleryEntry) -> str | None:
        """しきい値に外れる理由。None なら使用する。値が無い項目は「外れ」とみなす。"""
        parts = []
        if self.min_face_conf > 0 and (e.face_conf is None or e.face_conf < self.min_face_conf):
            parts.append(f"顔信頼度 {e.face_conf:.3f} < {self.min_face_conf:g}" if e.face_conf is not None else "顔信頼度なし")
        if self.min_norm > 0 and (e.quality_norm is None or e.quality_norm < self.min_norm):
            parts.append(f"品質ノルム {e.quality_norm:.1f} < {self.min_norm:g}" if e.quality_norm is not None else "品質ノルム未計算")
        if self.min_face_width > 0 and (e.face_width is None or e.face_width < self.min_face_width):
            parts.append(f"顔幅 {e.face_width:.0f}px < {self.min_face_width:g}px" if e.face_width is not None else "顔幅なし")
        return " / ".join(parts) if parts else None


class QualityScorer:
    """AdaFace IR-101 の特徴量ノルムを品質スコアとして返す（識別モデルとは独立に、常に AdaFace を使う）。"""

    def __init__(self):
        self.ext = AdaFaceIR101WebFace12M()

    def weights_ready(self) -> bool:
        return self.ext.weights_ready()

    def manual_instructions(self) -> str:
        return "品質ノルムの計算には AdaFace の重みが必要です。\n" + self.ext.manual_instructions()

    def load(self) -> None:
        self.ext.load()

    def norms(self, faces_bgr: list[np.ndarray], aligner) -> list[float]:
        """顔画像ごとの品質ノルム。aligner で 112x112 に整列してから AdaFace に入れる（整列失敗はリサイズ）。"""
        import torch
        self.load()
        size = self.ext.input_size
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), NORM_BATCH):
                chunk = []
                for img in faces_bgr[i:i + NORM_BATCH]:
                    res = aligner.align_with_landmarks(img, size)
                    chunk.append(res.image if res is not None else cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC))
                x = torch.from_numpy(np.stack([preprocess(f, size) for f in chunk])).to(self.ext.device)
                o = self.ext.model(x)
                n = o[1].reshape(-1) if isinstance(o, (tuple, list)) else torch.norm(o, dim=1)
                out += [float(v) for v in n.float().cpu()]
        return out


def apply_query_filter(queries: list[GalleryEntry], flt: QueryFilter, norms: list[float] | None) -> int:
    """全クエリに品質ノルムを入れ、しきい値に外れた画像に excluded_reason を入れる。外した数を返す。
    無効画像（invalid）と採用基準外（gate_excluded_reason）はもともと使わないので数えない。"""
    if norms is not None:
        for q, n in zip(queries, norms):
            q.quality_norm = n
    n_excluded = 0
    for q in queries:
        q.excluded_reason = ""
        if q.invalid or q.gate_excluded_reason:
            continue
        r = flt.reason(q)
        if r:
            q.excluded_reason = r
            n_excluded += 1
    return n_excluded
