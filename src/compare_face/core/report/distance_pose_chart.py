"""距離・顔の向き・顔識別精度のチャート（顔サイズ別精度検証の Excel レポートに「距離向き精度分析チャート」シートを追加する）。

データは「比較結果誤識別数まとめ」シート。横軸 = M 列から右に並ぶ画像の「距離」、縦軸 = 16 行目から下に並ぶ画像の「距離」で、
縦 i × 横 j のセルの位置（横の距離, 縦の距離）に、2 枚の画像の「顔の向き」の組合せで決まるアイコン（images/chart_icon）を置く。
アイコンの色は、そのセルの背景色（誤識別数の段階の色）に染める。

- 距離・顔の向きは行列シート上では「対象画像情報」シートを引く VLOOKUP の式。openpyxl 出力のファイルは計算値を持たないので、
  計算値（Excel で保存し直した場合）があればそれを、無ければ「対象画像情報」をキー（カメラID・映像ファイル名・追跡ID・画像ファイル名）で引いて求める。
- 対象の xlsx は openpyxl で保存し直さない（セル内画像のリッチデータが失われる）。新シートは別ブックで作り、zip に移植する。
- チャートは Python（Pillow）で軸・グリッド線（5 m 実線 / 1 m 破線）・アイコン・凡例を描いた 1 枚の PNG を貼る（Excel のチャート機能は使わない。
  Excel の散布図は点にアイコン画像を使えず、図形の大きさにも上限があるため）。1 つのオブジェクトなので動かしてもずれないが、Excel 上では編集できない。
  日本語フォント（Meiryo 等）が必要。
GUI に依存しない。"""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.drawing.spreadsheet_drawing import AbsoluteAnchor
from openpyxl.drawing.xdr import XDRPoint2D, XDRPositiveSize2D

from ..common.settings import PROJECT_ROOT
from .verify_face_size_excel import (
    _COUNT_LEGEND, FIRST_COL, FIRST_ROW, INFO_FIRST_ROW, INFO_SHEET_NAME, MISIDENTIFY_SHEET_NAME, count_bin, count_bin_label,
    count_fill)

CHART_SHEET_NAME = "距離向き精度分析チャート"
ALL_CAMERAS_LABEL = "すべてのカメラ"
CHART_GAP_PX = 120      # 複数のチャートを縦に並べるときの間隔
ICON_DIR = PROJECT_ROOT / "images" / "chart_icon"

FRONT, LEFT, RIGHT, L_DIAG, R_DIAG = "正面", "左横", "右横", "左斜め正面", "右斜め正面"
DIRECTIONS = (FRONT, LEFT, RIGHT, L_DIAG, R_DIAG)

# 顔の向きの組合せ → アイコン。縦×横を入れ替えても同じアイコン（仕様の表はどちらの順でも同じ）
_ICON_TABLE = {
    (FRONT, FRONT): "front_front", (FRONT, LEFT): "front_left", (FRONT, RIGHT): "front_right",
    (FRONT, L_DIAG): "front_l_diagonal", (FRONT, R_DIAG): "front_r_diagonal",
    (LEFT, LEFT): "left_left", (LEFT, RIGHT): "right_left", (LEFT, L_DIAG): "left_l_diagonal", (LEFT, R_DIAG): "left_r_diagonal",
    (RIGHT, RIGHT): "right_right", (RIGHT, L_DIAG): "right_l_diagonal", (RIGHT, R_DIAG): "right_r_diagonal",
    (L_DIAG, L_DIAG): "l_diagonal_l_diagonal", (L_DIAG, R_DIAG): "diagonal_diagonal",
    (R_DIAG, R_DIAG): "r_diagonal_r_diagonal",
}
ICON_FILES = {frozenset(k): f"{v}.png" for k, v in _ICON_TABLE.items()}


def icon_file(a: str, b: str) -> str | None:
    """顔の向き 2 つ（縦・横。順不同）のアイコンのファイル名。向きが 5 種に無ければ None。"""
    return ICON_FILES.get(frozenset((a, b)))


# チャートの寸法(px。96dpi)。アイコンの位置はここから計算する
CHART_W_PX, CHART_H_PX = 6000, 4800
CHART_X0_PX, CHART_Y0_PX = 160, 20                        # シート左上からのチャートの位置（A・B 列の数値表の右）
PLOT = (0.08, 0.09, 0.74, 0.80)                            # プロット領域（内側）の チャート幅・高さに対する x, y, w, h。右の余白に凡例を置く
LEGEND_ICON_PX, LEGEND_ROW_PX = 32, 40                      # 凡例のアイコン・1 行の高さ
LEGEND_FONT_PT = 16
LEGEND_ICON_RGB = (0x80, 0x80, 0x80)                        # 顔の向きの凡例のアイコンの色（誤識別数の色とは無関係なので灰色）
ICON_PX = 19
MAJOR_STEP_M = 5      # 5 m ごとの線（実線）。軸の範囲はこの倍数に合わせる
MINOR_STEP_M = 1      # 1 m ごとの線（破線）
AXIS_PAD_M = 0.5      # 最小・最大の距離の外側の余白(m)
EMU_PER_PX = 9525

ICON_BASE_V = 190.0      # アイコンの緑の部分の明るさ（染め直すときの基準）
_GREEN_HUE = (55, 115)   # OpenCV の色相（0〜180）。緑の部分だけを染める
_MIN_SAT = 40


@dataclass
class ChartPoint:
    x: float            # 横の画像の距離(m)
    y: float            # 縦の画像の距離(m)
    icon: str           # アイコンのファイル名
    rgb: tuple[int, int, int]   # 誤識別数シートのセルの背景色
    h_dir: str = ""     # 横の画像の顔の向き（左右の振り分けに使う）
    cam_v: str = ""     # 縦の画像のカメラID
    cam_h: str = ""     # 横の画像のカメラID


@dataclass
class ChartData:
    points: list[ChartPoint] = field(default_factory=list)
    n_images: int = 0
    no_distance: int = 0        # 距離が無い（「-」等）画像の数
    no_direction: int = 0       # 顔の向きが 5 種のどれでもない（空欄等）画像の数
    no_fill: int = 0            # 背景色が無いセルの数（対角以外）
    n_icons: int = 0            # チャートに描いたアイコンの数（左右に振り分けると正面は 2 倍になる）
    ver: list = field(default_factory=list)     # 縦 i の (距離, 顔の向き)（読めなければ None）
    hor: list = field(default_factory=list)     # 横 j の (距離, 顔の向き)
    cells: dict = field(default_factory=dict)   # (i, j) -> 誤識別数シートのセルの数値（対角・数値でないセルは無い）
    ver_cam: list = field(default_factory=list)   # 縦 i のカメラID
    hor_cam: list = field(default_factory=list)   # 横 j のカメラID
    charts: list = field(default_factory=list)    # 描いたチャート [(ラベル, アイコンの数)]
    n_groups: int = 0           # 距離の集計単位で集計したときの、集計後の縦・横のグループの組合せ数


def _norm(v) -> str:
    return re.sub(r"[\s　]+", "", str(v)) if v is not None else ""


def _as_float(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _direction(v) -> str | None:
    s = _norm(v)
    return s if s in DIRECTIONS else None


def _fill_rgb(cell) -> tuple[int, int, int] | None:
    f = cell.fill
    if f is None or f.fill_type != "solid":
        return None
    rgb = f.fgColor.rgb if f.fgColor is not None and f.fgColor.type == "rgb" else None
    if not isinstance(rgb, str) or len(rgb) < 6:
        return None
    try:
        return int(rgb[-6:-4], 16), int(rgb[-4:-2], 16), int(rgb[-2:], 16)
    except ValueError:
        return None


def read_chart_data(xlsx: Path | str) -> ChartData:
    """誤識別数シートから、チャートの点（距離・顔の向き・背景色）を読む。必要なシートが無ければ ValueError。"""
    path = Path(xlsx)
    wb = load_workbook(path)
    wbv = load_workbook(path, data_only=True)
    if MISIDENTIFY_SHEET_NAME not in wb.sheetnames:
        raise ValueError(f"「{MISIDENTIFY_SHEET_NAME}」シートがありません。顔サイズ別精度検証で出力した Excel を指定してください。")
    ws, wsv = wb[MISIDENTIFY_SHEET_NAME], wbv[MISIDENTIFY_SHEET_NAME]
    info = _read_info(wb[INFO_SHEET_NAME]) if INFO_SHEET_NAME in wb.sheetnames else {}

    label_col = FIRST_COL - 1
    head_row = FIRST_ROW - 1
    row_of = {_norm(ws.cell(row=r, column=label_col).value): r for r in range(5, head_row)}   # 横側: 行見出し → 行
    col_of = {_norm(ws.cell(row=head_row, column=c).value): c for c in range(2, label_col)}   # 縦側: 見出し → 列
    for need in ("距離", "顔の向き"):
        if need not in row_of or need not in col_of:
            raise ValueError(f"「{MISIDENTIFY_SHEET_NAME}」シートに「{need}」の行・列が見つかりません（レイアウトが違います）。")
    n_cols = 0
    while ws.cell(row=5, column=FIRST_COL + n_cols).value not in (None, ""):
        n_cols += 1
    n_rows = 0
    while ws.cell(row=FIRST_ROW + n_rows, column=2).value not in (None, ""):
        n_rows += 1
    if n_cols == 0 or n_rows == 0:
        raise ValueError(f"「{MISIDENTIFY_SHEET_NAME}」シートに画像がありません。")

    def resolve(cached, key, what: str):
        """計算値があればそれ、無ければ対象画像情報を引く。"""
        if cached not in (None, ""):
            return cached
        return info.get(key, {}).get(what)

    def key_h(j: int):
        return tuple(str(ws.cell(row=5 + k, column=FIRST_COL + j).value) for k in range(4))   # カメラID・映像・追跡ID・画像名

    def key_v(i: int):
        return tuple(str(ws.cell(row=FIRST_ROW + i, column=2 + k).value) for k in range(4))

    hor = []   # 横 j: (距離, 向き)
    for j in range(n_cols):
        c = FIRST_COL + j
        hor.append((_as_float(resolve(wsv.cell(row=row_of["距離"], column=c).value, key_h(j), "距離")),
                    _direction(resolve(wsv.cell(row=row_of["顔の向き"], column=c).value, key_h(j), "顔の向き"))))
    ver = []   # 縦 i
    for i in range(n_rows):
        r = FIRST_ROW + i
        ver.append((_as_float(resolve(wsv.cell(row=r, column=col_of["距離"]).value, key_v(i), "距離")),
                    _direction(resolve(wsv.cell(row=r, column=col_of["顔の向き"]).value, key_v(i), "顔の向き"))))

    cam_row, cam_col = row_of.get("カメラID"), col_of.get("カメラID")
    hor_cam = [str(ws.cell(row=cam_row, column=FIRST_COL + j).value or "").strip() if cam_row else "" for j in range(n_cols)]
    ver_cam = [str(ws.cell(row=FIRST_ROW + i, column=cam_col).value or "").strip() if cam_col else "" for i in range(n_rows)]
    data = ChartData(n_images=max(n_rows, n_cols), ver=ver, hor=hor, ver_cam=ver_cam, hor_cam=hor_cam)
    data.no_distance = sum(1 for d, _ in ver if d is None)
    data.no_direction = sum(1 for _, o in ver if o is None)
    for i, (dy, oy) in enumerate(ver):
        for j, (dx, ox) in enumerate(hor):
            if i == j:
                continue
            v = ws.cell(row=FIRST_ROW + i, column=FIRST_COL + j).value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                data.cells[(i, j)] = float(v)
            if dy is None or dx is None or oy is None or ox is None:
                continue
            rgb = _fill_rgb(ws.cell(row=FIRST_ROW + i, column=FIRST_COL + j))
            if rgb is None:
                data.no_fill += 1
                continue
            data.points.append(ChartPoint(dx, dy, ICON_FILES[frozenset((oy, ox))], rgb, ox, ver_cam[i], hor_cam[j]))
    return data


def aggregate_points(data: ChartData, unit_m: float, camera: str | None = None) -> list[ChartPoint]:
    """距離の集計単位（unit_m）ごとに点をまとめる。縦・横それぞれの画像を「顔の向き × 距離の区間（unit_m 刻み）」でグループにし、
    縦グループ × 横グループの組合せごとに、
    - 色: 「誤識別数まとめ」のセルの値（対角を除く）の平均から決める（count_fill）
    - 位置: 横グループの距離の平均（横）・縦グループの距離の平均（縦）
    の 1 点にする。顔の向き・距離が読めない画像は除く。camera を指定すると、そのカメラの画像どうしだけで集計する。"""
    if unit_m <= 0:
        raise ValueError("距離の集計単位は 0 より大きい値にしてください。")

    def groups(items: list, cams: list) -> dict[tuple, list[int]]:
        g: dict[tuple, list[int]] = {}
        for i, (dist, direction) in enumerate(items):
            if dist is None or direction is None or (camera is not None and (cams[i] if i < len(cams) else "") != camera):
                continue
            g.setdefault((direction, math.floor(dist / unit_m + 1e-9)), []).append(i)
        return g

    gv, gh = groups(data.ver, data.ver_cam), groups(data.hor, data.hor_cam)
    out: list[ChartPoint] = []
    for (oy, _), vi in gv.items():
        y = sum(data.ver[i][0] for i in vi) / len(vi)
        for (ox, _), hj in gh.items():
            vals = [data.cells[(i, j)] for i in vi for j in hj if (i, j) in data.cells]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            x = sum(data.hor[j][0] for j in hj) / len(hj)
            rgb = count_fill(avg).fgColor.rgb[-6:]
            out.append(ChartPoint(x, y, ICON_FILES[frozenset((oy, ox))], (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16)), ox))
    return out


def cameras_of(data: ChartData) -> list[str]:
    """画像のカメラID（重複なし・昇順）。"""
    return sorted({c for c in data.ver_cam + data.hor_cam if c})


def chart_targets(data: ChartData, all_cameras: bool, per_camera: bool, unit_m: float = 0.0) -> list[tuple[str, list[ChartPoint]]]:
    """描くチャートの一覧 [(ラベル, 点)]。all_cameras = すべてのカメラを 1 チャートに / per_camera = カメラID ごとに 1 チャート。
    カメラ個別は、縦・横の両方の画像がそのカメラのものである組合せだけを使う。点が無いチャートは含めない。"""
    if not all_cameras and not per_camera:
        raise ValueError("チャートの出力対象（すべてのカメラ / カメラ個別）を 1 つ以上選んでください。")
    out: list[tuple[str, list[ChartPoint]]] = []
    if all_cameras:
        out.append((ALL_CAMERAS_LABEL, aggregate_points(data, unit_m) if unit_m > 0 else list(data.points)))
    if per_camera:
        for cam in cameras_of(data):
            pts = aggregate_points(data, unit_m, cam) if unit_m > 0 else [p for p in data.points if p.cam_v == cam and p.cam_h == cam]
            out.append((f"カメラ {cam}", pts))
    return [(label, pts) for label, pts in out if pts]


def _read_info(ws) -> dict[tuple, dict]:
    """対象画像情報シート: (カメラID, 映像ファイル名, 追跡ID, 画像ファイル名) → {顔の向き, 距離}。列は見出し（2 行目）で探す。"""
    head = {_norm(c.value): c.column for c in ws[INFO_FIRST_ROW - 1] if c.value is not None}
    need = ("カメラID", "映像ファイル名", "追跡ID", "画像ファイル名", "顔の向き", "距離")
    if any(k not in head for k in need):
        return {}
    out: dict[tuple, dict] = {}
    for r in range(INFO_FIRST_ROW, ws.max_row + 1):
        key = tuple(str(ws.cell(row=r, column=head[k]).value) for k in need[:4])
        if key[0] in ("None", ""):
            continue
        out[key] = {"顔の向き": ws.cell(row=r, column=head["顔の向き"]).value, "距離": ws.cell(row=r, column=head["距離"]).value}
    return out


def recolor_icon(png_path: Path | str, rgb: tuple[int, int, int]) -> bytes:
    """アイコン（緑の円）の緑の部分を rgb に染め直した PNG（バイト列）。白・濃色の線は残し、外側の白い角は透過にする。"""
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"アイコン画像を読み込めません: {png_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    bgr = img[..., :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)
    green = (h >= _GREEN_HUE[0]) & (h <= _GREEN_HUE[1]) & (s >= _MIN_SAT)
    target = np.array(rgb[::-1], np.float32)                       # BGR
    shade = np.clip(v / ICON_BASE_V, 0.0, 1.2)[..., None]
    out = bgr.astype(np.float32)
    out[green] = np.clip(target * shade[green], 0, 255)
    img[..., :3] = out.astype(np.uint8)
    # 外側の白（角）を透過にする: 四隅から白をたどる
    white = ((bgr >= 235).all(axis=-1)).astype(np.uint8)
    h_, w_ = white.shape
    mask = np.zeros((h_ + 2, w_ + 2), np.uint8)
    flood = white.copy()
    for seed in ((0, 0), (w_ - 1, 0), (0, h_ - 1), (w_ - 1, h_ - 1)):
        if flood[seed[1], seed[0]] == 1:
            cv2.floodFill(flood, mask, seed, 2)
    img[..., 3] = np.where(flood == 2, 0, img[..., 3])
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("アイコン画像を PNG にできません。")
    return buf.tobytes()


def nice_axis(values: list[float]) -> tuple[float, float, float]:
    """(最小, 最大, 目盛の間隔)。値の範囲に余白を付けたキリのよい範囲。"""
    lo, hi = min(values), max(values)
    step = MAJOR_STEP_M
    return (math.floor((lo - AXIS_PAD_M) / step) * step, math.ceil((hi + AXIS_PAD_M) / step) * step, step)


def split_xs(h_dir: str, x: float) -> list[float]:
    """左右に振り分けたときの横軸の位置（中心 = 0 m）。横の画像の顔の向きに「右」を含めば右側(+)、「左」を含めば左側(-)、
    どちらも含まなければ（正面）右側と左側の両方。"""
    if "右" in h_dir:
        return [x]
    if "左" in h_dir:
        return [-x]
    return [x, -x]


def plot_points(points: list[ChartPoint], split_lr: bool) -> list[tuple[float, float, str, tuple[int, int, int]]]:
    """チャートに描く点（横の位置, 縦の位置, アイコン, 色）。split_lr なら横の位置は左右に振り分けた値。"""
    out = []
    for p in points:
        for x in (split_xs(p.h_dir, p.x) if split_lr else [p.x]):
            out.append((x, p.y, p.icon, p.rgb))
    return out


def axis_ranges(points: list[ChartPoint], split_lr: bool) -> tuple[float, float, float, float, float]:
    """(横の最小, 横の最大, 縦の最小, 縦の最大, 目盛の間隔)。左右に振り分けるときは横軸を 0 を中心に左右対称にする。"""
    lo, hi, step = nice_axis([p.x for p in points] + [p.y for p in points])
    return (-hi, hi, lo, hi, step) if split_lr else (lo, hi, lo, hi, step)


def _px_pos(v: float, lo: float, hi: float, start: float, size: float, invert: bool = False) -> float:
    t = (v - lo) / (hi - lo)
    return start + (1 - t if invert else t) * size


def _build_sheet_workbook(data: ChartData, path: Path, icon_dir: Path, split_lr: bool = False, unit_m: float = 0.0,
                          charts: list[tuple[str, list[ChartPoint]]] | None = None) -> None:
    """新シートだけのブックを path に保存する。Python（Pillow）で描いたチャート全体の画像を、charts の 1 つにつき 1 枚、縦に並べて貼る
    （charts が None なら data.points の 1 枚）。セルは数値のみ（A: 横の距離, B: 縦の距離, C: チャートの番号。参照用）で、
    文字列は持たない（zip へ移植するため）。"""
    charts = charts if charts is not None else [("", data.points)]
    wb = Workbook()
    ws = wb.active
    ws.title = CHART_SHEET_NAME
    row = 1
    for k, (label, pts) in enumerate(charts):
        for x, y, _, _ in plot_points(pts, split_lr):
            ws.cell(row=row, column=1, value=round(x, 4))
            ws.cell(row=row, column=2, value=round(y, 4))
            ws.cell(row=row, column=3, value=k + 1)
            row += 1
        img = XlImage(io.BytesIO(render_chart_image(replace(data, points=pts), icon_dir, split_lr, unit_m, label)))
        y0 = CHART_Y0_PX + k * (CHART_H_PX + CHART_GAP_PX)
        img.anchor = AbsoluteAnchor(pos=XDRPoint2D(CHART_X0_PX * EMU_PER_PX, y0 * EMU_PER_PX),
                                    ext=XDRPositiveSize2D(CHART_W_PX * EMU_PER_PX, CHART_H_PX * EMU_PER_PX))
        ws.add_image(img)
    wb.save(path)


# ---- 凡例（チャートの右の余白）----
def legend_items() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(色の凡例 [(ラベル, 色 RRGGBB)], 顔の向きの凡例 [(ラベル, アイコンのファイル名)])。"""
    colors = [(count_bin_label(count_bin(n)), count_fill(n).fgColor.rgb[-6:]) for n in _COUNT_LEGEND]
    dirs = [(f"{a} × {b}", f"{name}.png") for (a, b), name in _ICON_TABLE.items()]
    return colors, dirs


def legend_layout() -> list[tuple[str, float, float, object]]:
    """凡例の要素（種類, x, y, 内容）。座標はシート左上からの px。種類: head（見出し）/ swatch（色見本。内容 = RRGGBB）/
    label・note（文字）/ icon（顔の向きのアイコン。内容 = ファイル名）。チャートの右の余白に置く。"""
    colors, dirs = legend_items()
    x0 = CHART_X0_PX + (PLOT[0] + PLOT[2] + 0.025) * CHART_W_PX
    y = CHART_Y0_PX + PLOT[1] * CHART_H_PX
    tx = x0 + LEGEND_ICON_PX + 16
    out: list[tuple[str, float, float, object]] = [("head", x0, y, "アイコンの色（誤識別数）")]
    y += LEGEND_ROW_PX + 8
    for label, rgb in colors:
        out += [("swatch", x0, y + 4, rgb), ("label", tx, y, label)]
        y += LEGEND_ROW_PX
    y += LEGEND_ROW_PX
    out.append(("head", x0, y, "アイコンの形（顔の向き: 縦 × 横）"))
    y += LEGEND_ROW_PX + 8
    for label, fname in dirs:
        out += [("icon", x0, y + 4, fname), ("label", tx, y, label)]
        y += LEGEND_ROW_PX
    out.append(("note", x0, y + 8, "（縦と横を入れ替えても同じアイコン）"))
    return out


# ---- Python で描画するチャート（画像）----
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def find_japanese_font() -> str | None:
    return next((f for f in _FONT_CANDIDATES if os.path.isfile(f)), None)


def render_chart_image(data: ChartData, icon_dir: Path, split_lr: bool = False, unit_m: float = 0.0, label: str = "") -> bytes:
    """チャート全体（軸・グリッド線・アイコン・凡例）を Pillow で 1 枚の PNG に描く。大きさは CHART_W_PX × CHART_H_PX。
    split_lr=True: 横軸は中心を 0 m として左右に距離をとり、横の画像の顔の向きで左側 / 右側 / 両側に振り分ける（split_xs）。"""
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_japanese_font()
    if font_path is None:
        raise ValueError("日本語フォントが見つからないため、画像のチャートを描画できません（Meiryo / 游ゴシック等が必要です）。")

    def font(px: int):
        return ImageFont.truetype(font_path, px)

    W, H = CHART_W_PX, CHART_H_PX
    ox, oy = CHART_X0_PX, CHART_Y0_PX        # シート座標 → 画像座標
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    xlo, xhi, ylo, yhi, step = axis_ranges(data.points, split_lr)
    px, py, pw, ph = PLOT
    pl, pt, pwid, phei = px * W, py * H, pw * W, ph * H
    grid = (0x80, 0x80, 0x80)

    def X(v: float) -> float:
        return _px_pos(v, xlo, xhi, pl, pwid)

    def Y(v: float) -> float:
        return _px_pos(v, ylo, yhi, pt, phei, invert=True)

    def dashed(p0, p1, dash=14, gap=10, width=2):
        (x0, y0), (x1, y1) = p0, p1
        length = math.hypot(x1 - x0, y1 - y0)
        t = 0.0
        while t < length:
            e = min(t + dash, length)
            d.line([(x0 + (x1 - x0) * t / length, y0 + (y1 - y0) * t / length),
                    (x0 + (x1 - x0) * e / length, y0 + (y1 - y0) * e / length)], fill=grid, width=width)
            t += dash + gap

    tick_font = font(30)

    def grid_lines(lo: float, hi: float, vertical: bool) -> None:
        k = 0
        while lo + k * MINOR_STEP_M <= hi + 1e-9:
            m = lo + k * MINOR_STEP_M
            major = abs(m / step - round(m / step)) < 1e-9
            if vertical:
                a_, b_ = (X(m), pt), (X(m), pt + phei)
            else:
                a_, b_ = (pl, Y(m)), (pl + pwid, Y(m))
            if major:
                d.line([a_, b_], fill=grid, width=6 if (vertical and split_lr and abs(m) < 1e-9) else 3)    # 5 m ごと = 実線（中心 0 m は太線）
            else:
                dashed(a_, b_)                                                                              # 1 m ごと = 破線
            if major:
                label = f"{abs(m):g}"                                                                       # 左側も距離（正の値）で表示
                if vertical:
                    d.text((X(m), pt + phei + 12), label, fill="black", font=tick_font, anchor="ma")
                else:
                    d.text((pl - 14, Y(m)), label, fill="black", font=tick_font, anchor="rm")
            k += 1

    grid_lines(xlo, xhi, True)
    grid_lines(ylo, yhi, False)
    d.rectangle([pl, pt, pl + pwid, pt + phei], outline=(0x59, 0x59, 0x59), width=3)
    title = "距離・顔の向き・誤識別数（アイコンの色 = 誤識別数の段階）"
    if unit_m > 0:
        title += f"　距離 {unit_m:g} m 単位 × 顔の向きごとの平均"
    if label:
        title = f"【{label}】" + title
    d.text((W * (px + pw / 2), H * 0.035), title, fill="black", font=font(48), anchor="mm")
    xtitle = "横の画像の距離 (m)（中心 0 m。顔の向きが 左 = 左側 / 右 = 右側 / 正面 = 両側）" if split_lr else "横の画像の距離 (m)"
    d.text((pl + pwid / 2, pt + phei + 90), xtitle, fill="black", font=font(38), anchor="mm")
    if split_lr:
        d.text((pl + 20, pt + 20), "← 左", fill="black", font=font(40), anchor="la")
        d.text((pl + pwid - 20, pt + 20), "右 →", fill="black", font=font(40), anchor="ra")
    ytitle = Image.new("RGBA", (700, 60), (255, 255, 255, 0))
    ImageDraw.Draw(ytitle).text((350, 30), "縦の画像の距離 (m)", fill="black", font=font(38), anchor="mm")
    ytitle = ytitle.rotate(90, expand=True)
    img.paste(ytitle, (int(pl - 150 - 30), int(pt + phei / 2 - 350)), ytitle)

    cache: dict[tuple, Image.Image] = {}

    def icon(fname: str, rgb, size: int):
        key = (fname, rgb, size)
        if key not in cache:
            im = Image.open(io.BytesIO(recolor_icon(icon_dir / fname, rgb))).convert("RGBA")
            cache[key] = im.resize((size, size), Image.LANCZOS)
        return cache[key]

    text_px = round(LEGEND_FONT_PT * 96 / 72)
    for kind, x, y, v in legend_layout():
        x, y = x - ox, y - oy
        if kind == "swatch":
            d.rectangle([x, y, x + LEGEND_ICON_PX, y + LEGEND_ICON_PX], fill="#" + v, outline=(0x59, 0x59, 0x59), width=2)
        elif kind == "icon":
            im = icon(v, LEGEND_ICON_RGB, LEGEND_ICON_PX)
            img.paste(im, (int(x), int(y)), im)
        else:
            d.text((x, y + LEGEND_ROW_PX / 2), v, fill="black", font=font(text_px + 3 if kind == "head" else text_px), anchor="lm")
    for x, y, fname, rgb in plot_points(data.points, split_lr):
        im = icon(fname, rgb, ICON_PX)
        img.paste(im, (int(X(x) - ICON_PX / 2), int(Y(y) - ICON_PX / 2)), im)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ---- zip への移植 ----
_CT_SHEET = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
_CT_DRAWING = "application/vnd.openxmlformats-officedocument.drawing+xml"
_CT_CHART = "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"   # 旧版のチャート付きシートを置き換えるときの掃除用
_MEDIA_PREFIX = "xl/media/chart_icon_"


def _rels_path(part: str) -> str:
    d, _, f = part.rpartition("/")
    return f"{d}/_rels/{f}.rels"


def _targets(rels_xml: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'Target="([^"]+)"', rels_xml)]


def _resolve(base_part: str, target: str) -> str:
    """rels の Target（絶対 / 相対）を zip 内のパスにする。"""
    if target.startswith("/"):
        return target.lstrip("/")
    parts = base_part.rpartition("/")[0].split("/")
    for seg in target.split("/"):
        if seg == "..":
            parts.pop()
        elif seg and seg != ".":
            parts.append(seg)
    return "/".join(parts)


def _remove_existing(parts: dict[str, bytes]) -> None:
    """同名のシート（前回のチャート）があれば、シート・図形・チャート・アイコン画像と、それらの登録を取り除く。"""
    wbx = parts["xl/workbook.xml"].decode("utf-8")
    m = re.search(r'<sheet\b[^>]*\bname="%s"[^>]*/>' % re.escape(CHART_SHEET_NAME), wbx)
    if not m:
        return
    rid = re.search(r'r:id="([^"]+)"', m.group(0)).group(1)
    rels = parts["xl/_rels/workbook.xml.rels"].decode("utf-8")
    rm = re.search(r'<Relationship\b[^>]*\bId="%s"[^>]*/>' % re.escape(rid), rels)
    sheet_part = _resolve("xl/workbook.xml", re.search(r'Target="([^"]+)"', rm.group(0)).group(1))
    removed = [sheet_part]
    for rel_part in (_rels_path(sheet_part),):
        if rel_part in parts:
            for t in _targets(parts[rel_part].decode("utf-8")):
                drawing = _resolve(sheet_part, t)
                removed.append(drawing)
                drels = _rels_path(drawing)
                if drels in parts:
                    removed += [_resolve(drawing, x) for x in _targets(parts[drels].decode("utf-8")) if "chart" in x]
                    removed.append(drels)
            removed.append(rel_part)
    removed += [k for k in parts if k.startswith(_MEDIA_PREFIX)]
    ct = parts["[Content_Types].xml"].decode("utf-8")
    for p in removed:
        parts.pop(p, None)
        ct = re.sub(r'<Override\b[^>]*PartName="/%s"[^>]*/>' % re.escape(p), "", ct)
    parts["[Content_Types].xml"] = ct.encode("utf-8")
    parts["xl/workbook.xml"] = wbx.replace(m.group(0), "").encode("utf-8")
    parts["xl/_rels/workbook.xml.rels"] = rels.replace(rm.group(0), "").encode("utf-8")


def _next_index(parts: dict[str, bytes], pattern: str) -> int:
    nums = [int(m.group(1)) for k in parts if (m := re.fullmatch(pattern, k))]
    return max(nums, default=0) + 1


def _graft(target: Path, src: Path) -> None:
    """src（新シートだけのブック）のシートを target の xlsx に移植する（target の他の部分は変えない）。"""
    with zipfile.ZipFile(target) as z:
        order = [i.filename for i in z.infolist()]
        parts = {n: z.read(n) for n in order}
    with zipfile.ZipFile(src) as z:
        new = {n: z.read(n) for n in z.namelist()}

    _remove_existing(parts)
    sheet_no = _next_index(parts, r"xl/worksheets/sheet(\d+)\.xml")
    drawing_no = _next_index(parts, r"xl/drawings/drawing(\d+)\.xml")
    chart_no = _next_index(parts, r"xl/charts/chart(\d+)\.xml")
    sheet_part, drawing_part = f"xl/worksheets/sheet{sheet_no}.xml", f"xl/drawings/drawing{drawing_no}.xml"

    src_sheet = "xl/worksheets/sheet1.xml"
    src_drawing = _resolve(src_sheet, _targets(new[_rels_path(src_sheet)].decode("utf-8"))[0])
    drels = new[_rels_path(src_drawing)].decode("utf-8")

    chart_map: dict[str, str] = {}
    media_map: dict[str, str] = {}
    for t in _targets(drels):
        sp = _resolve(src_drawing, t)
        if "/charts/" in sp:
            dst = f"xl/charts/chart{chart_no + len(chart_map)}.xml"
            chart_map[sp] = dst
            parts[dst] = new[sp]
        elif sp not in media_map:
            digest = hashlib.sha1(new[sp]).hexdigest()[:12]
            dst = f"{_MEDIA_PREFIX}{digest}{Path(sp).suffix}"
            media_map[sp] = dst
            parts[dst] = new[sp]       # 同じ内容の画像は 1 つにまとめる

    def retarget(m: re.Match) -> str:
        sp = _resolve(src_drawing, m.group(1))
        dst = chart_map.get(sp) or media_map.get(sp)
        return f'Target="/{dst}"' if dst else m.group(0)

    parts[drawing_part] = new[src_drawing]
    parts[_rels_path(drawing_part)] = re.sub(r'Target="([^"]+)"', retarget, drels).encode("utf-8")
    sheet_xml = new[src_sheet].decode("utf-8")
    sheet_xml = re.sub(r'\s+tabSelected="1"', "", sheet_xml)   # 先頭シートの選択状態を持ち込まない（シートのグループ化になる）
    parts[sheet_part] = sheet_xml.encode("utf-8")
    srels = new[_rels_path(src_sheet)].decode("utf-8")
    parts[_rels_path(sheet_part)] = re.sub(r'Target="([^"]+)"', f'Target="/{drawing_part}"', srels).encode("utf-8")

    ct = parts["[Content_Types].xml"].decode("utf-8")
    add = [f'<Override PartName="/{sheet_part}" ContentType="{_CT_SHEET}"/>',
           f'<Override PartName="/{drawing_part}" ContentType="{_CT_DRAWING}"/>']
    add += [f'<Override PartName="/{p}" ContentType="{_CT_CHART}"/>' for p in chart_map.values()]
    if not re.search(r'<Default\b[^>]*Extension="png"', ct):
        add.append('<Default Extension="png" ContentType="image/png"/>')
    parts["[Content_Types].xml"] = ct.replace("</Types>", "".join(add) + "</Types>").encode("utf-8")

    rels = parts["xl/_rels/workbook.xml.rels"].decode("utf-8")
    used_ids = [int(x) for x in re.findall(r'Id="rId(\d+)"', rels)]
    rid = f"rId{max(used_ids, default=0) + 1}"
    rels = rels.replace("</Relationships>",
                        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                        f'Target="worksheets/sheet{sheet_no}.xml"/></Relationships>')
    parts["xl/_rels/workbook.xml.rels"] = rels.encode("utf-8")
    wbx = parts["xl/workbook.xml"].decode("utf-8")
    sheet_id = max([int(x) for x in re.findall(r'sheetId="(\d+)"', wbx)], default=0) + 1
    wbx = wbx.replace("</sheets>", f'<sheet name="{CHART_SHEET_NAME}" sheetId="{sheet_id}" r:id="{rid}"/></sheets>')
    parts["xl/workbook.xml"] = wbx.encode("utf-8")

    tmp = target.with_name(target.name + ".tmp")
    names = [n for n in order if n in parts] + [n for n in parts if n not in order]
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, parts[n])
    try:
        os.replace(tmp, target)
    except PermissionError as e:
        tmp.unlink(missing_ok=True)
        raise PermissionError(f"Excel ファイルを書き込めません。Excel で開いている場合は閉じてから実行してください: {target}") from e


def add_chart_sheet(xlsx: Path | str, icon_dir: Path | str = ICON_DIR, split_lr: bool = False, unit_m: float = 0.0,
                    all_cameras: bool = True, per_camera: bool = False) -> ChartData:
    """xlsx に「距離向き精度分析チャート」シートを追加する（あれば置き換える）。戻り値は読み取った内容（点の数など）。
    チャートは Python（Pillow）で描いた画像を貼り付ける（Excel のチャート機能は使わない）。
    split_lr=True: 横軸を中心 0 m の左右にとり、横の画像の顔の向きで左側 / 右側 / 両側に振り分ける。
    unit_m > 0: 距離の集計単位。縦・横それぞれ「顔の向き × 距離の区間」ごとに、セルの値の平均で色を、距離の平均で位置を決める（aggregate_points）。
    all_cameras: すべてのカメラを 1 チャートにプロット / per_camera: カメラIDごとに 1 チャートずつプロット（両方 True ならカメラ数 + 1 個。
    同じシートに縦に並べる。どちらも False はエラー）。"""
    path, icons = Path(xlsx), Path(icon_dir)
    if not path.is_file():
        raise ValueError(f"Excel ファイルが見つかりません: {path}")
    if not icons.is_dir():
        raise ValueError(f"アイコン画像フォルダが見つかりません: {icons}")
    data = read_chart_data(path)
    charts = chart_targets(data, all_cameras, per_camera, unit_m)
    if not charts:
        raise ValueError("チャートに出せるデータがありません（距離・顔の向きが入った画像の組合せがありません）。"
                         "「対象画像情報」シートの「顔の向き」（正面 / 左横 / 右横 / 左斜め正面 / 右斜め正面）を入力してください。")
    data.charts = [(label, len(plot_points(pts, split_lr))) for label, pts in charts]
    data.points = charts[0][1] if len(charts) == 1 else [p for _, pts in charts for p in pts]
    data.n_groups = len(data.points) if unit_m > 0 else 0
    tmp = path.with_name(path.stem + ".chart_tmp.xlsx")
    try:
        data.n_icons = sum(n for _, n in data.charts)
        _build_sheet_workbook(data, tmp, icons, split_lr, unit_m, charts)
        _graft(path, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return data
