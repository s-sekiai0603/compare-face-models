"""距離・向き・精度分析のレポート（Excel）の点を、指定した距離の範囲で集計する（集計画面 ui/chart_summary_screen.py の処理本体）。

点は read_chart_data が返す素のデータ（顔の向きの左右への振り分け・距離の集計単位の集計を通さない）を使うので、
チャートを「左右に振り分け」で作っていても集計には影響しない。
距離の範囲は最大 2 つまで。縦の画像の距離 y・横の画像の距離 x のそれぞれが 下限 <= 距離 <= 上限 のものが範囲内（縦横を合わせて指定 = 同じ範囲、
別々に指定 = 縦と横で別の範囲）。2 つ指定したときは、どちらかに入るデータを合算して 1 つの集計にする（重なるデータは 1 回だけ数える）。"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Collection

from .distance_pose_chart import (
    COLOR_GROUPS, ChartData, ChartPoint, color_group, icon_choices,
)

EPS = 1e-9      # 距離の境界（範囲の端ちょうど）を含めるための許容


@dataclass
class SummaryRow:
    section: str            # 区分（「全体」/「距離範囲 0〜10 m」）
    kind: str               # 種別（「総数」/「対象の色」/「対象のアイコン」/「色 × アイコン」）
    item: str               # 項目（色名・アイコン名・「色 / アイコン」）
    count: int | None       # 件数（空白の行は None）
    ratio: float | None     # 割合（分母 = その区分の総数。分母 0 は None）
    color: int | None = None    # 項目が順位（色の段階）のとき COLOR_GROUPS の番号（表のセルの背景色に使う）
    icon: str | None = None     # 項目が向きの組合せのときアイコンのファイル名（表のセルの先頭の画像に使う）
    sim: tuple[float, float, float, float] | None = None    # その行の集計単位の類似度（最小, 最大, 平均, 中央値）。類似度が無ければ None


def select_points(data: ChartData, camera: str | None) -> list[ChartPoint]:
    """集計の対象の点。camera = None はすべてのカメラ（全点）、カメラID は縦・横の両方がそのカメラの点だけ。"""
    if camera is None:
        return list(data.points)
    return [p for p in data.points if p.cam_v == camera and p.cam_h == camera]      # chart_targets のカメラ個別（unit_m = 0）と同じ条件


MAX_RANGES = 2


@dataclass(frozen=True)
class DistanceRange:
    """縦の画像の距離 y が v_min〜v_max、横の画像の距離 x が h_min〜h_max（m。端を含む）。"""
    v_min: float
    v_max: float
    h_min: float
    h_max: float

    @classmethod
    def both(cls, lo: float, hi: float) -> "DistanceRange":
        """縦横を合わせて指定（縦・横とも同じ範囲）。"""
        return cls(lo, hi, lo, hi)

    def contains(self, p: ChartPoint) -> bool:
        return self.v_min - EPS <= p.y <= self.v_max + EPS and self.h_min - EPS <= p.x <= self.h_max + EPS

    def label(self) -> str:
        if (self.v_min, self.v_max) == (self.h_min, self.h_max):
            return f"{self.v_min:g}〜{self.v_max:g} m"
        return f"縦 {self.v_min:g}〜{self.v_max:g} m・横 {self.h_min:g}〜{self.h_max:g} m"


def in_range(p: ChartPoint, ranges: Collection[DistanceRange]) -> bool:
    """どれかの範囲に入るか（2 つ指定したときは合算 = いずれかに入ればよい）。"""
    return any(r.contains(p) for r in ranges)


def similarity_stats(points: list[ChartPoint]) -> tuple[float, float, float, float] | None:
    """点の類似度（最小, 最大, 平均, 中央値）。類似度の無い点は除く。1 つも無ければ None。"""
    vals = [p.sim for p in points if p.sim is not None]
    if not vals:
        return None
    return min(vals), max(vals), statistics.fmean(vals), statistics.median(vals)


def _block(section: str, points: list[ChartPoint], colors: Collection[int], icons: Collection[str]) -> list[SummaryRow]:
    """1 区分の行（総数・色別・アイコン別・色 × アイコン別）。colors / icons は並べたい順（上から）。"""
    total = len(points)

    def row(kind: str, item: str, members: list[ChartPoint], color: int | None = None, icon: str | None = None) -> SummaryRow:
        n = len(members)
        return SummaryRow(section, kind, item, n, (n / total) if total else None, color, icon, similarity_stats(members))

    group_of = [color_group(p.rgb) for p in points]      # 段階に当てはまらない色は None（どの色の行にも入らない）
    color_label = {g: label for g, (label, _, _) in enumerate(COLOR_GROUPS)}
    icon_label = dict((fname, label) for label, fname in icon_choices())

    rows = [SummaryRow(section, "総数", "データ件数", total, 1.0 if total else None, sim=similarity_stats(points))]
    for g in colors:
        rows.append(row("対象の色", color_label[g], [p for p, k in zip(points, group_of) if k == g], color=g))
    for f in icons:
        rows.append(row("対象のアイコン", icon_label[f], [p for p in points if p.icon == f], icon=f))
    for g in colors:
        for f in icons:
            rows.append(row("色 × アイコン", f"{color_label[g]} / {icon_label[f]}",
                            [p for p, k in zip(points, group_of) if k == g and p.icon == f], g, f))
    return rows


def summarize(points: list[ChartPoint], colors: Collection[int], icons: Collection[str],
              ranges: Collection[DistanceRange]) -> list[SummaryRow]:
    """全体と、距離の範囲内（ranges のいずれかに入る = 合算）の集計行。colors / icons は集計する色の段階・アイコンのファイル名（表の行の順）。"""
    ranges = list(ranges)
    if not 1 <= len(ranges) <= MAX_RANGES:
        raise ValueError(f"距離の範囲は 1〜{MAX_RANGES} 個で指定してください。")
    scoped = [p for p in points if in_range(p, ranges)]
    section = "距離範囲 " + " + ".join(r.label() for r in ranges)
    # 範囲内ブロックの先頭（総数の行）の上に、全体に対する範囲内の件数と割合の行を置く
    share = SummaryRow(section, "総数", "全体に対するデータ件数・割合", len(scoped), (len(scoped) / len(points)) if points else None,
                      sim=similarity_stats(scoped))
    block = _block(section, scoped, colors, icons)
    # 範囲内の総数の行の下に、範囲内のうち対象の色に合致するデータの割合の行を置く（件数 = 対象の色の件数の合計）
    matched = sum(r.count for r in block if r.kind == "対象の色")
    in_colors = set(colors)
    cond = SummaryRow(section, "総数", "集計距離範囲内で指定条件に合致するデータ", matched, (matched / len(scoped)) if scoped else None,
                      sim=similarity_stats([p for p in scoped if color_group(p.rgb) in in_colors]))
    return _block("全体", points, colors, icons) + [share, block[0], cond] + block[1:]
