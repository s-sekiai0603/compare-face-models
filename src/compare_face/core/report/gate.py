"""レポート時に適用する顔の採用基準（抽出時の顔ゲート・品質ガードの再判定）。

抽出は採用基準に外れた顔も保存し（JSON の gate_ok=false）、使うかどうかはここで決める。既定値は premises-security と
同じ（骨格ルールあり / 両目間隔比 0.05 / 切り出し短辺 60px）なので、何も変えなければ抽出時に採用された画像だけを使う。
しきい値を緩めると抽出時に不採用だった画像（横顔・小さい顔など）も比較に使え、厳しくすると採用画像をさらに絞れる。

クエリ自動選別（quality.py）と違い、クエリと比較対象の両方に適用する（抽出時の基準の再現なので、既定で比較対象から
消える画像は無い）。判定値が JSON に無い古い抽出結果は、基準を通った画像しか保存していないので「通った」とみなす。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..extract.face_crop import MIN_CROP_SIDE, MIN_EYE_RATIO
from .gallery import GalleryEntry


@dataclass
class GateFilter:
    face_visible: bool = True               # 骨格ルール（鼻+片目が見える / 下向きでない）を適用する
    min_eye_ratio: float = MIN_EYE_RATIO    # 両目間隔÷顔幅のしきい値（両目が見える画像だけ判定）。0 で判定しない
    min_crop_side: float = MIN_CROP_SIDE    # 切り出し box の短辺(px) のしきい値。0 で判定しない

    @classmethod
    def accept_all(cls) -> "GateFilter":
        """採用基準を適用しない（抽出した顔を全て使う）。"""
        return cls(face_visible=False, min_eye_ratio=0.0, min_crop_side=0.0)

    def is_default(self) -> bool:
        """抽出時（premises-security）と同じ基準か。"""
        return self == GateFilter()

    def enabled(self) -> bool:
        return self.face_visible or self.min_eye_ratio > 0 or self.min_crop_side > 0

    def describe(self) -> str:
        if not self.enabled():
            return "適用しない（抽出した顔を全て使う）"
        parts = ["骨格ルール（鼻+片目 / 下向きでない）" if self.face_visible else "骨格ルールなし",
                 f"両目間隔比 >= {self.min_eye_ratio:g}" if self.min_eye_ratio > 0 else "両目間隔比の判定なし",
                 f"切り出し短辺 >= {self.min_crop_side:g}px" if self.min_crop_side > 0 else "切り出し短辺の判定なし"]
        text = " / ".join(parts)
        return f"抽出時と同じ（既定）: {text}" if self.is_default() else text

    def reason(self, e: GalleryEntry) -> str | None:
        """基準に外れる理由。None なら使用する。判定値の無い項目（古い抽出結果）は通ったとみなす。"""
        parts = []
        if self.face_visible and e.face_visible is False:
            parts.append(e.gate_reason or "鼻または目が見えない / 下向きの顔")  # 骨格ルールは抽出時の最初の判定なので理由が残っている
        if self.min_eye_ratio > 0 and e.eye_ratio is not None and e.eye_ratio < self.min_eye_ratio:
            parts.append(f"両目間隔比 {e.eye_ratio:.2f} < {self.min_eye_ratio:g}（横顔）")
        if self.min_crop_side > 0 and e.crop_min_side is not None and e.crop_min_side < self.min_crop_side:
            parts.append(f"切り出し短辺 {e.crop_min_side:.0f}px < {self.min_crop_side:g}px")
        return " / ".join(parts) if parts else None


def apply_gate_filter(entries: list[GalleryEntry], flt: GateFilter) -> int:
    """基準に外れた画像に gate_excluded_reason を入れ、その数を返す。無効画像（invalid）はもともと使わないので数えない。"""
    n = 0
    for e in entries:
        e.gate_excluded_reason = ""
        if e.invalid:
            continue
        r = flt.reason(e)
        if r:
            e.gate_excluded_reason = r
            n += 1
    return n
