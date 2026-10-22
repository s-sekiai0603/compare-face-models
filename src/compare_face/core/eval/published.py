"""各モデルの公表値（論文・公式 README の 1:1 照合精度, %）。eval の結果に並べて表示する参考値。

学習データ・評価コードが各配布元のものなので、当ツールの数値と完全には一致しない（アライン済み .bin をそのまま使い
flip 融合した条件が最も近い）。モデル名は features/ のレジストリ名、データセット名は .bin の stem。
"""
from __future__ import annotations

# (モデル名, データセット名) -> 公表値 (%)
PUBLISHED: dict[tuple[str, str], float] = {}
SOURCES: dict[str, str] = {}


def _add(model: str, source: str, **values: float) -> None:
    SOURCES[model] = source
    for dataset, v in values.items():
        PUBLISHED[(model, dataset)] = v


# AdaFace 公式 README の R100 / WebFace12M の行（https://github.com/mk-minchul/AdaFace）。検証は元画像と反転画像の特徴を融合
_add("AdaFace", "AdaFace README (R100, WebFace12M)", lfw=99.82, cfp_fp=99.26, cplfw=94.57, calfw=96.12, agedb_30=98.00)
# CVLFace README の ViT KPRPE + AdaFace の行（https://github.com/mk-minchul/CVLface）
_add("CVLFace KP-RPE ViT-B (WebFace12M)", "CVLFace README (ViT KPRPE AdaFace, WebFace12M)",
     lfw=99.82, cfp_fp=99.30, cplfw=95.65, calfw=95.93, agedb_30=98.10)
_add("CVLFace KP-RPE ViT-B (WebFace4M)", "CVLFace README (ViT KPRPE AdaFace, WebFace4M)",
     lfw=99.83, cfp_fp=99.01, cplfw=95.40, calfw=96.00, agedb_30=97.67)
# LAFS 論文 Table 4（WebFace4M 学習, CVPR 2024, arXiv:2403.08161）。CPLFW / CALFW は未掲載
_add("LAFS Part fViT (WebFace4M)", "LAFS 論文 Table 4 (Part fViT + LAFS, WebFace4M)", lfw=99.83, cfp_fp=99.18, agedb_30=97.90)
_add("Part fViT (WebFace4M, 事前学習なし)", "LAFS 論文 Table 4 (Part fViT, WebFace4M)", lfw=99.83, cfp_fp=99.13, agedb_30=97.73)
# FSPFM 論文 Table 1（ICIP 2025, arXiv:2505.16412）。公式リポジトリの再現値は CFP-FP 98.87 / 98.76 とやや高い
_add("FSPFM ResNet50 (正面化あり, 微調整済)", "FSPFM 論文 Table 1 (fine-tuned)",
     lfw=99.83, cfp_fp=98.67, cplfw=93.83, calfw=96.05, agedb_30=98.13)
_add("FSPFM ResNet50 (正面化あり, 事前学習のみ)", "FSPFM 論文 Table 1 (pre-training only)",
     lfw=99.82, cfp_fp=98.60, cplfw=93.58, calfw=96.03, agedb_30=98.13)


def published_accuracy(model: str, dataset: str) -> float | None:
    return PUBLISHED.get((model, dataset))
