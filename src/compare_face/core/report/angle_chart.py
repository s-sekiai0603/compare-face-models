"""顔角度の組合せ表示（チャート(PNG)確認画面の表示モード）。GUI 非依存。

距離・向き・精度分析と同じ Excel レポートの「比較結果誤識別数まとめ」シートから、縦 i × 横 j の組合せごとに
  ・基準となる顔画像（縦 i）の顔の角度 = 縦軸
  ・比較対象の顔画像（横 j）の顔の角度 = 横軸
  ・セルの背景色（誤識別数の段階 = 精度。COLOR_GROUPS）
を読み、角度の組合せで交差する座標に精度の色の点をプロットする。点の色の段階から、周辺の点の加重平均（ガウス重み）でヒートマップを作って背面に敷く。
顔の角度は「対象画像情報」シートの「顔の角度」（yaw[度]。正面 = 0、向かって右向き = マイナス、左向き = プラス）。
軸は両方とも 左 / 上 = +（左向き）→ 右 / 下 = −（右向き）の向きに並べ、軸の外側に images/man_direction の顔画像（左横・左斜め・正面・右斜め・右横）を置く。
距離のフィルタ（集計画面の「集計範囲の距離」と同じ DistanceRange）は、縦（基準）・横（比較対象）の画像の距離それぞれにかける。"""
from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from ..common.settings import PROJECT_ROOT
from .chart_png_viewer import _heat_palette
from .chart_summary import DistanceRange, in_range
from .distance_pose_chart import (
    COLOR_GROUPS, ChartPoint, _as_float, _fill_rgb, _matrix_layout, _norm, _read_info, color_group, color_group_rgb, find_japanese_font,
)
from .verify_face_size_excel import INFO_SHEET_NAME, MISIDENTIFY_SHEET_NAME

MAN_DIR = PROJECT_ROOT / "images" / "man_direction"
# 軸の外側に置く顔画像: (角度[度], ファイル名の glob, 表示名)。+ = 向かって左向き、− = 向かって右向き
MAN_FACES = ((90, "man_left.png", "左横"), (45, "man_left_diagonal.png", "左斜め正面"), (0, "man_front.png", "正面"),
             (-45, "man_right_diagonal.png", "右斜め正面"), (-90, "man_right.png", "右横"))

AXIS_LIMIT = 100       # 軸の範囲 -100〜+100 度（推定値は ±90 を少し超えることがある）
GRID_STEP = 15         # グリッド線の間隔（度）
HEAT_SIGMA_DEG = 7.0   # ヒートマップの加重平均の広がり（ガウス重みの標準偏差。度）
HEAT_MAX_ALPHA = 0.75
HEAT_ALPHA_GAMMA = 0.6
DOT_R = 7              # 点の半径(px)

# 画像の寸法(px)
PLOT = 1500                          # プロット領域（正方形）の一辺
M_LEFT, M_TOP, M_RIGHT, M_BOTTOM = 400, 520, 480, 110
FACE_PX = 150
IMG_W, IMG_H = M_LEFT + PLOT + M_RIGHT, M_TOP + PLOT + M_BOTTOM


@dataclass
class AngleData:
    points: list[ChartPoint] = field(default_factory=list)   # 縦・横の顔の角度が両方ある組合せ。x / y = 横 / 縦の画像の距離（無ければ nan）
    n_images: int = 0
    no_angle: int = 0        # 顔の角度が無い画像（縦側）の数
    no_distance: int = 0     # 距離が無い画像（縦側）の数
    no_fill: int = 0         # 背景色が無いセルの数（対角以外）
    cameras: list[str] = field(default_factory=list)


def read_angle_data(xlsx: Path | str) -> AngleData:
    """誤識別数シートから、顔角度の組合せ表示の点（縦・横の顔の角度・距離・背景色）を読む。必要なシート・行・列が無ければ ValueError。"""
    path = Path(xlsx)
    wb = load_workbook(path)
    wbv = load_workbook(path, data_only=True)
    if MISIDENTIFY_SHEET_NAME not in wb.sheetnames:
        raise ValueError(f"「{MISIDENTIFY_SHEET_NAME}」シートがありません。顔サイズ別精度検証で出力した Excel を指定してください。")
    ws, wsv = wb[MISIDENTIFY_SHEET_NAME], wbv[MISIDENTIFY_SHEET_NAME]
    info = _read_info(wb[INFO_SHEET_NAME]) if INFO_SHEET_NAME in wb.sheetnames else {}
    first_row, first_col = _matrix_layout(ws)
    label_col, head_row = first_col - 1, first_row - 1
    row_of = {_norm(ws.cell(row=r, column=label_col).value): r for r in range(5, head_row)}
    col_of = {_norm(ws.cell(row=head_row, column=c).value): c for c in range(2, label_col)}
    for need in ("距離", "顔の角度"):
        if need not in row_of or need not in col_of:
            raise ValueError(f"「{MISIDENTIFY_SHEET_NAME}」シートに「{need}」の行・列が見つかりません。"
                             "顔の角度を出力する版の顔サイズ別精度検証で、レポートを出力し直してください。")
    n_cols = 0
    while ws.cell(row=5, column=first_col + n_cols).value not in (None, ""):
        n_cols += 1
    n_rows = 0
    while ws.cell(row=first_row + n_rows, column=2).value not in (None, ""):
        n_rows += 1
    if n_cols == 0 or n_rows == 0:
        raise ValueError(f"「{MISIDENTIFY_SHEET_NAME}」シートに画像がありません。")

    def resolve(cached, key, what: str):
        return cached if cached not in (None, "") else info.get(key, {}).get(what)

    def key_h(j: int):
        return tuple(str(ws.cell(row=5 + k, column=first_col + j).value) for k in range(4))

    def key_v(i: int):
        return tuple(str(ws.cell(row=first_row + i, column=2 + k).value) for k in range(4))

    hor = []   # 横 j: (角度, 距離)
    for j in range(n_cols):
        c = first_col + j
        hor.append((_as_float(resolve(wsv.cell(row=row_of["顔の角度"], column=c).value, key_h(j), "顔の角度")),
                    _as_float(resolve(wsv.cell(row=row_of["距離"], column=c).value, key_h(j), "距離"))))
    ver = []   # 縦 i
    for i in range(n_rows):
        r = first_row + i
        ver.append((_as_float(resolve(wsv.cell(row=r, column=col_of["顔の角度"]).value, key_v(i), "顔の角度")),
                    _as_float(resolve(wsv.cell(row=r, column=col_of["距離"]).value, key_v(i), "距離"))))
    cam_row, cam_col = row_of.get("カメラID"), col_of.get("カメラID")
    hor_cam = [str(ws.cell(row=cam_row, column=first_col + j).value or "").strip() if cam_row else "" for j in range(n_cols)]
    ver_cam = [str(ws.cell(row=first_row + i, column=cam_col).value or "").strip() if cam_col else "" for i in range(n_rows)]

    data = AngleData(n_images=max(n_rows, n_cols), cameras=sorted({c for c in ver_cam + hor_cam if c}))
    data.no_angle = sum(1 for a, _ in ver if a is None)
    data.no_distance = sum(1 for _, d in ver if d is None)
    nan = float("nan")
    for i, (ay, dy) in enumerate(ver):
        for j, (ax, dx) in enumerate(hor):
            if i == j or ay is None or ax is None:
                continue
            rgb = _fill_rgb(ws.cell(row=first_row + i, column=first_col + j))
            if rgb is None:
                data.no_fill += 1
                continue
            data.points.append(ChartPoint(nan if dx is None else dx, nan if dy is None else dy, "", rgb, "", ver_cam[i], hor_cam[j],
                                          None, ay, ax))
    return data


def select_angle_points(points: list[ChartPoint], camera: str | None = None, colors: Collection[int] | None = None,
                        ranges: Collection[DistanceRange] | None = None) -> list[ChartPoint]:
    """プロットする点。camera = None はすべてのカメラ、カメラID は縦・横の両方がそのカメラの組合せだけ。
    colors = 対象の色の段階（None = すべて。段階に当てはまらない色の点は色では除かない）。
    ranges = 距離のフィルタ（None = かけない。縦の画像の距離 y・横の画像の距離 x が いずれかの範囲に入る点。距離が無い点は除く）。"""
    out = []
    for p in points:
        if camera is not None and not (p.cam_v == camera and p.cam_h == camera):
            continue
        if colors is not None:
            g = color_group(p.rgb)
            if g is not None and g not in colors:
                continue
        if ranges is not None and not in_range(p, ranges):
            continue
        out.append(p)
    return out


def heat_grid(points: list[ChartPoint], sigma_deg: float = HEAT_SIGMA_DEG) -> tuple[np.ndarray, np.ndarray]:
    """加重平均のヒートマップの元（1 度刻みの格子。縦軸 = 基準の角度 -100〜+100 度、横軸 = 比較対象の角度）。
    戻り値 = (色の段階の加重平均（0〜6。点が無い所は 0）, 密度（ガウス重みの合計））。段階に当てはまらない色の点は数えない。"""
    import cv2
    n = 2 * AXIS_LIMIT + 1
    cnt = np.zeros((n, n), np.float32)
    lev = np.zeros((n, n), np.float32)
    for p in points:
        g = color_group(p.rgb)
        if g is None or p.ang_v is None or p.ang_h is None:
            continue
        r, c = round(p.ang_v) + AXIS_LIMIT, round(p.ang_h) + AXIS_LIMIT
        if 0 <= r < n and 0 <= c < n:
            cnt[r, c] += 1
            lev[r, c] += g
    d = cv2.GaussianBlur(cnt, (0, 0), sigma_deg)
    w = cv2.GaussianBlur(lev, (0, 0), sigma_deg)
    return np.where(d > 1e-6, w / np.maximum(d, 1e-6), 0).astype(np.float32), d


def _to_px(angle: float, origin: int) -> float:
    """角度 → 画像の座標。両軸とも +100 度が左 / 上、-100 度が右 / 下。"""
    return origin + (AXIS_LIMIT - angle) / (2 * AXIS_LIMIT) * PLOT


def _man_faces() -> list[tuple[int, str, "object | None"]]:
    from PIL import Image
    out = []
    for ang, pattern, label in MAN_FACES:
        f = next(iter(sorted(MAN_DIR.glob(pattern))), None)
        im = None
        if f is not None:
            im = Image.open(f).convert("RGBA").resize((FACE_PX, FACE_PX), Image.LANCZOS)
        out.append((ang, label, im))
    return out


def render_angle_chart(points: list[ChartPoint], title: str = "", notes: Collection[str] = (),
                       color_order: list[int] | None = None):
    """顔角度の組合せのチャート（Pillow の RGBA 画像）を描く。ヒートマップ（背面）+ グリッド + 点 + 軸の外側の顔画像 + 色の凡例。
    color_order = 点を手前に描く色の順（先頭ほど手前。None = 色の段階の小さい順 = 精度の高い点が手前）。"""
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_japanese_font()
    if font_path is None:
        raise ValueError("日本語フォントが見つからないため、画像のチャートを描画できません（Meiryo / 游ゴシック等が必要です）。")

    def font(px: int):
        return ImageFont.truetype(font_path, px)

    x0, y0 = M_LEFT, M_TOP
    img = Image.new("RGBA", (IMG_W, IMG_H), "white")

    # ヒートマップ（背面）
    level, dens = heat_grid(points)
    peak = float(np.percentile(dens[dens > 1e-6], 99)) if (dens > 1e-6).any() else 0.0
    if peak > 0:
        import cv2
        alpha = np.clip(dens / peak, 0, 1) ** HEAT_ALPHA_GAMMA * HEAT_MAX_ALPHA
        alpha[dens < peak * 1e-3] = 0
        heat = np.dstack([_heat_palette(level), alpha * 255]).astype(np.float32)
        # 格子（-100〜+100 度）は行 = 縦軸の角度、列 = 横軸の角度（どちらも小さい方が先）。軸は +100 度が左 / 上なので反転する
        heat = heat[::-1, ::-1]
        heat = cv2.resize(heat, (PLOT, PLOT), interpolation=cv2.INTER_CUBIC)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.paste(Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8), "RGBA"), (x0, y0))
        img.alpha_composite(layer)

    d = ImageDraw.Draw(img)
    grid, axis_c, text_c = (205, 205, 205), (60, 60, 60), (30, 30, 30)
    f_tick, f_axis, f_title, f_leg = font(26), font(34), font(40), font(28)

    # グリッド線と目盛りの数値
    for a in range(-90, 91, GRID_STEP):
        major = a % 45 == 0
        px, py = _to_px(a, x0), _to_px(a, y0)
        d.line([(px, y0), (px, y0 + PLOT)], fill=axis_c if a == 0 else grid, width=3 if a == 0 else (2 if major else 1))
        d.line([(x0, py), (x0 + PLOT, py)], fill=axis_c if a == 0 else grid, width=3 if a == 0 else (2 if major else 1))
        t = f"{a:+d}°" if a else "0°"
        tw = d.textlength(t, font=f_tick)
        d.text((px - tw / 2, y0 - 38), t, fill=text_c, font=f_tick)
        d.text((x0 - 12 - tw, py - 15), t, fill=text_c, font=f_tick)
    d.rectangle([x0, y0, x0 + PLOT, y0 + PLOT], outline=axis_c, width=3)

    # 点（精度の色）。手前に描く色の順: color_order の先頭ほど手前
    rank = {g: k for k, g in enumerate(color_order if color_order is not None else range(len(COLOR_GROUPS)))}
    for p in sorted(points, key=lambda q: -rank.get(color_group(q.rgb), len(rank))):
        if p.ang_v is None or p.ang_h is None:
            continue
        cx, cy = _to_px(max(-AXIS_LIMIT, min(AXIS_LIMIT, p.ang_h)), x0), _to_px(max(-AXIS_LIMIT, min(AXIS_LIMIT, p.ang_v)), y0)
        d.ellipse([cx - DOT_R, cy - DOT_R, cx + DOT_R, cy + DOT_R], fill=tuple(p.rgb) + (255,), outline=(40, 40, 40, 255), width=1)

    # 軸の外側の顔画像（上 = 比較対象（横軸）、左 = 基準（縦軸））。名前は顔画像の上（横軸側）・下（縦軸側）
    face_top = y0 - 60 - FACE_PX
    face_left = x0 - 100 - FACE_PX
    for ang, label, face in _man_faces():
        if face is None:
            continue
        px, py = _to_px(ang, x0), _to_px(ang, y0)
        img.alpha_composite(face, (int(px - FACE_PX / 2), int(face_top)))
        img.alpha_composite(face, (int(face_left), int(py - FACE_PX / 2)))
        lw = d.textlength(label, font=f_tick)
        d.text((px - lw / 2, face_top - 34), label, fill=text_c, font=f_tick)
        d.text((face_left + FACE_PX / 2 - lw / 2, py + FACE_PX / 2 + 2), label, fill=text_c, font=f_tick)

    # 軸のタイトル
    d.text((x0, face_top - 110), "比較対象の顔角度（横軸）　+: 向かって左向き　−: 向かって右向き", fill=text_c, font=f_axis)
    vt = "基準となる顔画像の顔角度（縦軸）"
    tw = int(d.textlength(vt, font=f_axis)) + 10
    tmp = Image.new("RGBA", (tw, 50), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((0, 4), vt, fill=text_c + (255,), font=f_axis)
    img.alpha_composite(tmp.rotate(90, expand=True), (30, int(y0 + PLOT / 2 - tw / 2)))

    # タイトル・条件
    ty = 20
    if title:
        d.text((40, ty), title, fill=text_c, font=f_title)
    for k, n in enumerate(notes):
        d.text((40, ty + 56 + 32 * k), n, fill=text_c, font=f_leg)

    # 凡例（右の余白）: 精度の色と、背景のヒートマップの説明
    lx, ly = x0 + PLOT + 40, y0
    d.text((lx, ly), "精度の色（誤識別数の順位）", fill=text_c, font=f_leg)
    counts: dict[int | None, int] = {}
    for p in points:
        g = color_group(p.rgb)
        counts[g] = counts.get(g, 0) + 1
    for g, (label, _, _) in enumerate(COLOR_GROUPS):
        yy = ly + 50 + 46 * g
        d.ellipse([lx, yy + 2, lx + 28, yy + 30], fill=color_group_rgb(g) + (255,), outline=(40, 40, 40, 255), width=1)
        d.text((lx + 42, yy), f"{label}（{counts.get(g, 0)}）", fill=text_c, font=f_leg)
    yy = ly + 50 + 46 * len(COLOR_GROUPS) + 20
    d.text((lx, yy), f"点数: {len(points)}", fill=text_c, font=f_leg)
    for k, t in enumerate(("背景のヒートマップ:", "点の色（順位の段階）の", f"加重平均（ガウス重み σ={HEAT_SIGMA_DEG:g}°）", "濃さ = 点の密集具合")):
        d.text((lx, yy + 50 + 34 * k), t, fill=text_c, font=f_leg)
    return img


class AngleOverlay:
    """画面から使う窓口（ChartOverlay / HeatmapOverlay と同じ image / compose(paths, progress) を持つ）。
    configure で条件を決め、compose で Excel を読んで（同じパス・更新時刻なら読み直さない）絞り込み、描く。"""

    def __init__(self):
        self.image = None
        self.data: AngleData | None = None
        self.message = ""
        self._key = None
        self._params: dict = {}

    def configure(self, excel_path: str, camera: str | None, colors: list[int], ranges: list[DistanceRange] | None) -> None:
        self._params = dict(excel_path=excel_path, camera=camera, colors=list(colors), ranges=ranges)

    def compose(self, paths=None, progress=None):
        p = self._params
        path = Path(p["excel_path"])
        st = path.stat()                                   # 無ければ OSError
        key = (str(path), st.st_mtime_ns, st.st_size)
        if progress is not None:
            progress(0, 1)
        if self.data is None or self._key != key:
            self.data = read_angle_data(path)
            self._key = key
        pts = select_angle_points(self.data.points, p["camera"], p["colors"], p["ranges"])
        notes = [f"Excel: {path.name}",
                 f"対象カメラ: {p['camera'] or 'すべてのカメラ'}",
                 "距離: " + (" + ".join(r.label() for r in p["ranges"]) if p["ranges"] else "絞り込みなし")]
        self.image = render_angle_chart(pts, "顔の角度の組合せによる精度分析", notes, p["colors"])
        d = self.data
        self.message = (f"{len(pts)} 点をプロット（顔の角度がある組合せ {len(d.points)} 点中。画像 {d.n_images} 枚"
                        f"・顔の角度なし {d.no_angle} 枚・距離なし {d.no_distance} 枚）")
        if progress is not None:
            progress(1, 1)
        return self.image
