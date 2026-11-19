"""距離×距離のチャート（アイコン表示・ヒートマップ表示。チャート確認画面の「距離の組合せ表示」）。GUI 非依存。

距離・向き・精度分析と同じ Excel レポートから、点（横の画像の距離, 縦の画像の距離, 色の段階）を直接読んでヒートマップを作る。
あらかじめ出力した PNG は使わない。ヒートマップは PNG（6000 × 4800 px）を 1 枚ずつ読んでアイコンを検出する方式より桁違いに速い（Excel の読込 + 格子の計算だけ）。
アイコン表示は点をアイコンで描く（render_chart_image）。両方を表示するとヒートマップの上にアイコンを重ねる。重なりは colors / icons の並び順（先頭ほど手前。色を優先し、同じ色の中ではアイコンの順）。
軸・グリッド・凡例は distance_pose_chart.render_chart_image（点なし・透明背景）で描き、全点から軸を決めるので PNG 出力と同じ軸になる。
色 = 周辺の点の色の段階の平均（緑 → 赤）、濃さ = 点の密集具合。
左右振り分け（split_lr）・距離の集計単位（unit_m）は距離・向き・精度分析画面の「PNG 出力」と同じ指定（Excel からは分からないので画面で指定する）。"""
from __future__ import annotations

import io
from collections.abc import Collection
from dataclasses import replace
from pathlib import Path

import numpy as np

from .distance_pose_chart import (
    ALL_CAMERAS_LABEL, CHART_H_PX, CHART_W_PX, COLOR_GROUPS, ICON_DIR, ICON_PX, PLOT, ChartData, ChartPoint, axis_ranges, chart_targets, color_group,
    color_group_rgb, filter_points, plot_points, read_chart_data, render_chart_image,
)

HEAT_DOWNSCALE = 4          # 計算時の縮小率（プロット領域を 1/4 にして密度を求め、最後に拡大する）
HEAT_SIGMA_ICONS = 1.6      # ぼかしの大きさ（アイコン幅に対する倍率）。点の密集具合をなめらかにする
HEAT_MAX_ALPHA = 0.92
HEAT_ALPHA_GAMMA = 0.45     # 小さいほど、点が少ない所も濃く出る


def heat_palette(levels: np.ndarray) -> np.ndarray:
    """色の段階（0〜6 の連続値）→ RGB（float32, N×3）。段階ごとの色（緑 → 黄 → 橙 → 赤 → 濃赤）の間を補間する。"""
    stops = np.array([color_group_rgb(g) for g in range(len(COLOR_GROUPS))], np.float32)
    t = np.clip(levels, 0, len(stops) - 1)
    k = np.minimum(t.astype(np.int32), len(stops) - 2)
    f = (t - k)[..., None]
    return stops[k] * (1 - f) + stops[k + 1] * f


def plot_box(w: int = CHART_W_PX, h: int = CHART_H_PX) -> tuple[int, int, int, int]:
    """チャート画像の中のプロット領域 (x0, y0, x1, y1)（px）。"""
    px, py, pw, ph = PLOT
    return int(px * w), int(py * h), int((px + pw) * w), int((py + ph) * h)


def heat_layers(points: list[tuple[float, float, int]], axes: tuple[float, float, float, float], box: tuple[int, int, int, int],
                ) -> tuple[np.ndarray, np.ndarray]:
    """点（横の位置, 縦の位置, 色の段階）→ 縮小した格子上の (密度, 段階の合計)。axes = (横の最小, 最大, 縦の最小, 最大)（m）。
    格子は plot_box を HEAT_DOWNSCALE で縮めたもの。軸の範囲外の点は除く。"""
    x0, y0, x1, y1 = box
    gw, gh = max((x1 - x0) // HEAT_DOWNSCALE, 1), max((y1 - y0) // HEAT_DOWNSCALE, 1)
    density, weighted = np.zeros(gw * gh, np.float32), np.zeros(gw * gh, np.float32)
    if points:
        a = np.asarray(points, np.float64)
        xlo, xhi, ylo, yhi = axes
        gx = ((a[:, 0] - xlo) / (xhi - xlo) * gw).astype(np.int64)
        gy = ((1 - (a[:, 1] - ylo) / (yhi - ylo)) * gh).astype(np.int64)      # 縦軸は上が大きい
        ok = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        idx = gy[ok] * gw + gx[ok]
        density += np.bincount(idx, minlength=gw * gh).astype(np.float32)
        weighted += np.bincount(idx, weights=a[ok, 2], minlength=gw * gh).astype(np.float32)
    return density.reshape(gh, gw), weighted.reshape(gh, gw)


def render_heat_layer(density: np.ndarray, weighted: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """(密度, 段階の合計) をぼかして、プロット領域の大きさ size = (幅, 高さ) の RGBA（uint8）にする。点が無ければ None。"""
    import cv2
    sigma = max(HEAT_SIGMA_ICONS * ICON_PX / HEAT_DOWNSCALE, 1.0)
    d = cv2.GaussianBlur(density, (0, 0), sigma)
    w = cv2.GaussianBlur(weighted, (0, 0), sigma)
    live = d > 1e-6
    if not live.any():
        return None
    peak = float(np.percentile(d[live], 99))
    mean_level = np.where(live, w / np.maximum(d, 1e-6), 0)
    alpha = np.clip(d / peak, 0, 1) ** HEAT_ALPHA_GAMMA * HEAT_MAX_ALPHA
    alpha[d < peak * 1e-3] = 0
    heat = np.dstack([heat_palette(mean_level), alpha * 255]).astype(np.float32)
    heat = cv2.resize(heat, size, interpolation=cv2.INTER_LINEAR)
    return np.clip(heat, 0, 255).astype(np.uint8)


class DistanceHeatmapOverlay:
    """（アイコン表示にも使う）画面から使う窓口（ChartOverlay と同じ image / compose(paths, progress) を持つ）。
    configure で条件を決め、compose で Excel を読んで（同じパス・更新時刻なら読み直さない）絞り込み、ヒートマップを作る。"""

    def __init__(self):
        self.image = None
        self.data: ChartData | None = None
        self.message = ""
        self._key = None
        self._params: dict = {}
        self._targets: dict = {}      # (unit_m) → 全カメラ・カメラ個別の全チャートの点（軸を決める）
        self._frame_key = None
        self._frame = None

    def configure(self, excel_path: str, camera: str | None, colors: Collection[int], icons: Collection[str],
                  split_lr: bool = False, unit_m: float = 0.0, show_icons: bool = True, show_heat: bool = False) -> None:
        """camera = カメラID 1 つ（None = すべてのカメラ）。colors / icons = 表示する色の段階・アイコン（手前に出したい順）。
        show_icons / show_heat = アイコン・ヒートマップの表示 / 非表示（両方表示するとヒートマップの上にアイコンを重ねる）。"""
        self._params = dict(excel_path=excel_path, camera=camera, colors=list(colors), icons=list(icons), split_lr=bool(split_lr),
                            unit_m=float(unit_m), show_icons=bool(show_icons), show_heat=bool(show_heat))

    def _frame_image(self, label: str, split_lr: bool, unit_m: float, axis_points: list[ChartPoint]):
        """軸・グリッド・凡例だけの透明な画像（点なし）。同じ条件なら作り直さない。"""
        from PIL import Image
        key = (self._key, label, split_lr, unit_m)
        if self._frame_key != key:
            png = render_chart_image(replace(self.data, points=[]), ICON_DIR, split_lr, unit_m, label, True, axis_points)
            self._frame = Image.open(io.BytesIO(png)).convert("RGBA")
            self._frame_key = key
        return self._frame

    def _icon_image(self, pts: list[ChartPoint], label: str, split_lr: bool, unit_m: float, axis_points: list[ChartPoint],
                    colors: list[int], icons: list[str]):
        """点をアイコンで描いたチャート（透明背景）。奥 → 手前の順に描く（colors / icons の先頭ほど手前。色を優先）。"""
        from PIL import Image
        rank_c = {g: k for k, g in enumerate(colors)}
        rank_i = {f: k for k, f in enumerate(icons)}
        ordered = sorted(pts, key=lambda q: (-rank_c.get(color_group(q.rgb), len(rank_c)), -rank_i.get(q.icon, len(rank_i))))
        png = render_chart_image(replace(self.data, points=ordered), ICON_DIR, split_lr, unit_m, label, True, axis_points)
        return Image.open(io.BytesIO(png)).convert("RGBA")

    def compose(self, paths=None, progress=None):
        p = self._params
        path = Path(p["excel_path"])
        st = path.stat()                                   # 無ければ OSError
        key = (str(path), st.st_mtime_ns, st.st_size)
        if progress is not None:
            progress(0, 1)
        if self.data is None or self._key != key:
            self.data = read_chart_data(path)
            self._key = key
            self._targets = {}
            self._frame_key = self._frame = None
        split_lr, unit_m, cam = p["split_lr"], p["unit_m"], p["camera"]
        if unit_m not in self._targets:
            self._targets[unit_m] = [pt for _, pts in chart_targets(self.data, True, True, unit_m) for pt in pts]
        axis_points = self._targets[unit_m]
        if not axis_points:
            raise ValueError("プロットできる点がありません（距離・顔の向きが読める画像の組合せがありません）。")
        if cam is None:
            charts = chart_targets(self.data, True, False, unit_m)
            label = ALL_CAMERAS_LABEL
        else:
            charts = chart_targets(self.data, False, True, unit_m, [cam])
            label = f"カメラ {cam}"
        pts = filter_points([pt for _, c in charts for pt in c], set(p["colors"]), set(p["icons"]))
        suffix = (f"距離 {unit_m:g} m 単位の集計" if unit_m > 0 else "集計なし") + ("・左右振り分け" if split_lr else "")
        show_icons, show_heat = p["show_icons"], p["show_heat"]
        if show_icons:
            top = self._icon_image(pts, label, split_lr, unit_m, axis_points, p["colors"], p["icons"])      # 枠 + アイコン
        else:
            top = self._frame_image(label, split_lr, unit_m, axis_points)                                   # 枠だけ
        levels = [(x, y, g) for x, y, _, rgb in plot_points(pts, split_lr) if (g := color_group(rgb)) is not None]
        layer = None
        if show_heat:
            xlo, xhi, ylo, yhi, _ = axis_ranges(axis_points, split_lr)
            box = plot_box(top.width, top.height)
            density, weighted = heat_layers(levels, (xlo, xhi, ylo, yhi), box)
            layer = render_heat_layer(density, weighted, (box[2] - box[0], box[3] - box[1]))
        if layer is None:
            self.image = top.copy()
        else:
            from PIL import Image
            canvas = np.zeros((top.height, top.width, 4), np.uint8)
            canvas[box[1]:box[3], box[0]:box[2]] = layer
            self.image = Image.fromarray(canvas, "RGBA")
            self.image.alpha_composite(top)                 # ヒートマップ（背面）の上に枠・アイコン
        what = " + ".join(n for n, on in (("アイコン", show_icons), ("ヒートマップ", show_heat)) if on) or "枠のみ"
        self.message = f"{len(pts)} 点を表示（{what}・{label}・{suffix}）"
        if progress is not None:
            progress(1, 1)
        return self.image
