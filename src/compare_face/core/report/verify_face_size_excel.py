"""顔サイズ別精度検証の Excel 出力（dev_work\\比較結果類似度.xlsx「比較結果類似度まとめ【サンプル】」「比較結果誤識別数まとめ【サンプル】」のレイアウト）。

画像 N 枚を顔 bbox サイズ順（呼び出し側で並べ済み）に、5〜15 行目の M 列より右（横）と B〜L 列の 16 行目より下（縦）に同じ並びで置く。
カメラID・映像ファイル名・追跡ID・画像ファイル名・画像サイズは値、顔画像・顔の向き・顔の信頼度・足元の座標・距離は「対象画像情報」シートを引く VLOOKUP の式
（距離 = カメラ設定ファイルから算出した、カメラ直下から足元までの地面上の距離(m)。設定が無い画像は「-」）。
キー（カメラID|映像ファイル名|追跡ID|画像ファイル名）は 4 行目（横）と A 列（縦）の式。
類似度シート: 縦 i × 横 j のうち i > j（左下半分）だけに類似度を出し、類似度の段階（SIM_BINS）で色を付ける（右上半分は左下と同じ値なので出さない）。
誤識別数シート: 同じ枠の対角以外すべてに「i との類似度が j との類似度より高い非対象者の画像の数」を出し、件数の段階（count_bin）で色を付ける。
10 行目 / G 列の「顔の向き」も同じ（対象画像情報の「顔の向き」は人手で入れる）。
全身画像・顔画像は「セルに画像を貼り付け」と同じ形式（リッチデータ）でセルに入れる（openpyxl は非対応なので保存後に xlsx へ書き足す）。
シート全体のフォントは Meiryo UI（対象画像情報と 3 つのまとめシート）。"""
from __future__ import annotations

import math
import re
import zipfile
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from ..common.imageio import imread
from ..common.phase_log import PhaseLogger
from .excel import _build_log_sheet
from .gallery import GalleryEntry
from .query_set_excel import BORDER, CENTER, HEAD_FILL, HEAD_FONT, NA

SHEET_NAME = "比較結果類似度まとめ"
MISIDENTIFY_SHEET_NAME = "比較結果誤識別数まとめ"
RANK_SHEET_NAME = "比較結果順位まとめ"
INFO_SHEET_NAME = "対象画像情報"
FIRST_ROW = 16   # 縦側の最初の画像の行
FIRST_COL = 13   # 横側の最初の画像の列（M）
SHEET_FONT = "Meiryo UI"
ADDED_COL_WIDTH = 11.38   # 追加した列（顔画像・顔の信頼度・足元の座標）の幅
IMAGE_ROW_PT = 80         # 顔画像の行と、データの行（16 行目以降）の高さ

# 類似度の段階（下限, 色）。下限以上で最後に当てはまる段階の色を塗る。上の段階ほど似ている（赤 → 黄 → 緑）
SIM_BINS = (
    (float("-inf"), "FFF8696B"),
    (0.2, "FFFA9A72"),
    (0.3, "FFFDC17B"),
    (0.4, "FFFFEB84"),
    (0.5, "FFD1E183"),
    (0.6, "FFA3D17F"),
    (0.7, "FF63BE7B"),
)

# 誤識別の件数の段階: 0 / 1〜10 / 11〜20 / 21〜50 / 51〜100 / 以降 50 件刻み。MAX_COUNT_BIN 以上は同じ色で打ち止め
MAX_COUNT_BIN = 12   # 451 件以上
_COUNT_COLORS = ((0xFF, 0xEB, 0x84), (0xFD, 0xB0, 0x7A), (0xF8, 0x69, 0x6B), (0xC0, 0x00, 0x00))  # 黄 → 橙 → 赤 → 濃赤
_COUNT_LEGEND = (0, 1, 11, 21, 51, 101, 10_000)  # 凡例に出す段階の代表件数

# 順位の段階（上限, 表示名）。1 位は緑、2 位以降は段階が上がるほど 黄 → 橙 → 赤 → 濃赤。最後の段階は上限なし
RANK_BINS = ((1, "1位"), (2, "2位"), (3, "3位"), (5, "4〜5位"), (10, "6〜10位"), (20, "11〜20位"), (50, "21〜50位"), (None, "51位以降"))

# 5〜15 行目の行見出し（L 列）と、画像 1 枚の値の取り出し
_ROW_LABELS = ("カメラID", "映像ファイル名", "追跡ID", "画像ファイル名", "顔画像", "顔の向き", "顔の信頼度", "足元の座標", "距離",
               "画像サイズ(幅:px)", "画像サイズ(高さ:px)")
# 15 行目の縦側見出し（B〜L 列）。並びは横側（_ROW_LABELS）と同じ
_COL_HEADS = ("カメラ\nID", "映像\nファイル名", "追跡\nID", "画像\nファイル名", "顔画像", "顔の向き", "顔の信頼度", "足元の座標", "距離",
              "画像サイズ\n(幅:px)", "画像サイズ\n(高さ:px)")
_FACE_K = 4   # _ROW_LABELS / _COL_HEADS の「顔画像」の位置
# 「対象画像情報」シートの B〜N 列のうち VLOOKUP で引く列の位置（B 列 = 1）
_LK_DIR, _LK_CONF, _LK_FOOT, _LK_DIST, _LK_FACE = 6, 9, 10, 11, 13
_LOOKUP_LAST_COL = "N"
_LOOKUP_MIN_LAST_ROW = 200   # VLOOKUP の範囲の最終行の下限（画像が多ければ広げる）


@dataclass
class VerifyConditions:
    model_name: str
    landmark_name: str
    gate_filter_text: str
    generated_at: datetime
    gallery_dir: str = ""            # 比較対象フォルダ（誤識別レポートのとき）
    n_nontarget: int | None = None   # 非対象者の画像数（誤識別レポートのとき）
    log_text_name: str = ""          # 誤認した画像の一覧（テキストファイル名。誤識別レポートのとき）
    n_nontarget_tracks: int | None = None   # 非対象者の追跡ID数（順位レポートのとき）
    grayscale: bool = False          # アライン後の顔画像を明るさ補正（CLAHE）+ グレースケール化してから特徴量化したか
    align168: bool = False           # 168×168 でアラインしてから 112×112 に縮小して特徴量化したか


def sim_fill(sim: float) -> PatternFill:
    color = next(c for lo, c in reversed(SIM_BINS) if sim >= lo)
    return PatternFill("solid", fgColor=color)


def bin_label(i: int) -> str:
    lo = SIM_BINS[i][0]
    if i == 0:
        return f"{SIM_BINS[1][0]:.1f} 未満"
    if i == len(SIM_BINS) - 1:
        return f"{lo:.1f} 以上"
    return f"{lo:.1f} 〜 {SIM_BINS[i + 1][0]:.1f}"


def count_bin(n: int) -> int:
    """件数の段階（0: 0 件 / 1: 1〜10 / 2: 11〜20 / 3: 21〜50 / 4: 51〜100 / 5〜: 101 件以降 50 件刻み。MAX_COUNT_BIN で打ち止め）。"""
    if n <= 0:
        return 0
    for b, upper in enumerate((10, 20, 50, 100), start=1):
        if n <= upper:
            return b
    return min(MAX_COUNT_BIN, 4 + math.ceil((n - 100) / 50))


def count_bin_label(b: int) -> str:
    if b <= 4:
        return ("0 件", "1〜10 件", "11〜20 件", "21〜50 件", "51〜100 件")[b]
    if b >= MAX_COUNT_BIN:
        return f"{101 + (MAX_COUNT_BIN - 5) * 50} 件以上"
    start = 101 + (b - 5) * 50
    return f"{start}〜{start + 49} 件"


def count_fill(n: int) -> PatternFill:
    """0 件は緑。1 件以上は段階が上がるほど 黄 → 橙 → 赤 → 濃赤。"""
    b = count_bin(n)
    if b == 0:
        return PatternFill("solid", fgColor="FF63BE7B")
    t = (b - 1) / (MAX_COUNT_BIN - 1) * (len(_COUNT_COLORS) - 1)
    k = min(int(t), len(_COUNT_COLORS) - 2)
    a, c = _COUNT_COLORS[k], _COUNT_COLORS[k + 1]
    rgb = [round(x + (y - x) * (t - k)) for x, y in zip(a, c)]
    return PatternFill("solid", fgColor="FF" + "".join(f"{v:02X}" for v in rgb))


def rank_bin(rank: int) -> int:
    """順位の段階（RANK_BINS の添字）。"""
    for b, (upper, _) in enumerate(RANK_BINS):
        if upper is None or rank <= upper:
            return b
    return len(RANK_BINS) - 1


def rank_fill(rank: int) -> PatternFill:
    """1 位は緑。2 位以降は段階が上がるほど 黄 → 橙 → 赤 → 濃赤。"""
    b = rank_bin(rank)
    if b == 0:
        return PatternFill("solid", fgColor="FF63BE7B")
    t = (b - 1) / (len(RANK_BINS) - 2) * (len(_COUNT_COLORS) - 1)
    k = min(int(t), len(_COUNT_COLORS) - 2)
    a, c = _COUNT_COLORS[k], _COUNT_COLORS[k + 1]
    rgb = [round(x + (y - x) * (t - k)) for x, y in zip(a, c)]
    return PatternFill("solid", fgColor="FF" + "".join(f"{v:02X}" for v in rgb))


def _px(v) -> object:
    return NA if v is None else round(float(v), 1)


def _values(e: GalleryEntry) -> list:
    """画像 1 枚の値（対象画像情報シートの C〜I 列）。カメラID・映像ファイル名・追跡ID・画像ファイル名・顔の向き（空欄）・幅・高さ。"""
    return [e.camera_id, e.video_file, e.track_id, e.face_path.name, None, _px(e.face_width), _px(e.face_height)]


def _foot_text(e: GalleryEntry) -> str:
    """足元の座標を「(x, y)」で。json に足元が無い（古い抽出結果）なら「-」。"""
    f = e.foot_xy
    if not f or len(f) < 2:
        return NA
    return f"({round(float(f[0]))}, {round(float(f[1]))})"


def _dist_value(e: GalleryEntry):
    """カメラからの距離(m)。カメラ設定ファイルから算出できていない（設定フォルダ未指定・設定なし等）なら「-」。"""
    return NA if e.distance_m is None else round(float(e.distance_m), DIST_DECIMALS)


class _Lookup:
    """行列シートが「対象画像情報」シートを引く VLOOKUP の式と、その結果（顔画像）のセル内画像の登録先。"""

    def __init__(self, images: "_InCellImages", n_info: int):
        self.images = images
        self.last_row = max(_LOOKUP_MIN_LAST_ROW, INFO_FIRST_ROW + n_info - 1)

    def formula(self, key_cell: str, index: int) -> str:
        return f"=VLOOKUP({key_cell}, {INFO_SHEET_NAME}!$B${INFO_FIRST_ROW}:${_LOOKUP_LAST_COL}${self.last_row}, {index}, FALSE)"


def _matrix_values(e: GalleryEntry, key_cell: str, lookup: _Lookup | None) -> list:
    """行列シートの画像 1 枚の値（並びは _ROW_LABELS）。顔画像・顔の信頼度・足元の座標は対象画像情報シートを引く式（key_cell = キーのセル）。"""
    f = (lambda idx: lookup.formula(key_cell, idx)) if lookup is not None else (lambda idx: None)
    return [e.camera_id, e.video_file, e.track_id, e.face_path.name, f(_LK_FACE), f(_LK_DIR), f(_LK_CONF), f(_LK_FOOT),
            f(_LK_DIST), _px(e.face_width), _px(e.face_height)]


def _apply_font(ws, name: str = SHEET_FONT) -> None:
    """シートの使用範囲（A1 から右下まで。空欄も含む）の全セルのフォントを name にする（サイズ・太字・色は変えない）。"""
    for row in ws.iter_rows():
        for cell in row:
            f = copy(cell.font)
            f.name = name
            cell.font = f


def _put(ws, r: int, c: int, v, fill: PatternFill | None = None, head: bool = False):
    cell = ws.cell(row=r, column=c, value=v)
    cell.border = BORDER
    cell.alignment = CENTER
    if head:
        cell.fill, cell.font = HEAD_FILL, HEAD_FONT
    elif fill is not None:
        cell.fill = fill
    return cell


def _build_matrix_sheet(ws, entries: list[GalleryEntry], title: str, note: str, legend: list[tuple[str, PatternFill]],
                        cell, lookup: _Lookup | None = None) -> None:
    """行列シートの共通部。cell(i, j) は縦 i × 横 j のセルに入れる (値, 塗り, 表示形式)。出さないなら None（対角は常に空欄の枠）。
    lookup があれば顔画像・顔の信頼度・足元の座標を対象画像情報シートから引く式にし、顔画像のセルにはセル内画像（式の結果）を登録する。"""
    n = len(entries)
    label_col = FIRST_COL - 1   # 横側の行見出し（K 列）
    head_row = FIRST_ROW - 1    # 縦側の見出し行（14 行目）
    ws.sheet_view.showGridLines = False
    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=14)
    ws.cell(row=2, column=2, value=note).font = Font(size=9)
    for i, (label, fill) in enumerate(legend):
        _put(ws, 3, 2 + i, label, fill=fill).font = Font(size=9)
    # 横側（5〜15 行目）。L 列が行見出し、M 列以降が画像 j。4 行目はキーの式（カメラID|映像ファイル名|追跡ID|画像ファイル名）
    for k, label in enumerate(_ROW_LABELS):
        _put(ws, 5 + k, label_col, label, head=True)
    for j, e in enumerate(entries):
        col = FIRST_COL + j
        letter = get_column_letter(col)
        ws.cell(row=4, column=col, value=f'={letter}5&"|"&{letter}6&"|"&{letter}7&"|"&{letter}8')
        for k, v in enumerate(_matrix_values(e, f"{letter}4", lookup)):
            _put(ws, 5 + k, col, v)
        if lookup is not None:
            lookup.images.mark(ws, 5 + _FACE_K, col, e.face_path, ORIGIN_FORMULA)
    # 縦側（15 行目が見出し、16 行目以降が画像 i）。A 列はキーの式
    for k, h in enumerate(_COL_HEADS):
        _put(ws, head_row, 2 + k, h, head=True)
    for i, e in enumerate(entries):
        r = FIRST_ROW + i
        ws.cell(row=r, column=1, value=f'=B{r}&"|"&C{r}&"|"&D{r}&"|"&E{r}')
        for k, v in enumerate(_matrix_values(e, f"A{r}", lookup)):
            _put(ws, r, 2 + k, v)
        if lookup is not None:
            lookup.images.mark(ws, r, 2 + _FACE_K, e.face_path, ORIGIN_FORMULA)
    for i in range(n):
        for j in range(n):
            if i == j:
                _put(ws, FIRST_ROW + i, FIRST_COL + j, None)  # 対角は枠だけ（自分自身）
                continue
            v = cell(i, j)
            if v is not None:
                value, fill, fmt = v
                _put(ws, FIRST_ROW + i, FIRST_COL + j, value, fill=fill).number_format = fmt
    for r in range(5, head_row):
        ws.row_dimensions[r].height = 15.75
    ws.row_dimensions[8].height = 62
    ws.row_dimensions[6].height = 30
    ws.row_dimensions[5 + _FACE_K].height = IMAGE_ROW_PT
    ws.row_dimensions[head_row].height = 32.25
    for i in range(n):
        ws.row_dimensions[FIRST_ROW + i].height = IMAGE_ROW_PT
    # 列幅 A〜L。追加した列（顔画像 F・顔の信頼度 H・足元の座標 I・距離 J）は ADDED_COL_WIDTH
    for c, w in zip("ABCDEFGHIJKL", (3, 9, 18, 7, 28, ADDED_COL_WIDTH, 11, ADDED_COL_WIDTH, ADDED_COL_WIDTH, ADDED_COL_WIDTH, 11, 20)):
        ws.column_dimensions[c].width = w
    for j in range(n):
        ws.column_dimensions[get_column_letter(FIRST_COL + j)].width = 12
    for row in ws.iter_rows(min_row=5, max_row=head_row - 1, min_col=FIRST_COL, max_col=FIRST_COL + n - 1):
        for c in row:
            c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
            c.font = Font(size=8)
    ws.freeze_panes = f"{get_column_letter(FIRST_COL)}{FIRST_ROW}"
    _apply_font(ws)


def _build_summary(ws, entries: list[GalleryEntry], sim: np.ndarray, lookup: _Lookup | None = None) -> None:
    """類似度シート: 縦 i > 横 j（左下半分）だけに類似度を出す（右上は左下と同じ値）。"""
    legend = [(bin_label(i), PatternFill("solid", fgColor=SIM_BINS[i][1])) for i in range(len(SIM_BINS))]

    def cell(i: int, j: int):
        if i < j:
            return None
        v = round(float(sim[i, j]), 3)
        return v, sim_fill(v), "0.000"

    _build_matrix_sheet(ws, entries, "比較結果：類似度まとめ",
                        f"画像 {len(entries)} 枚を顔 bbox のサイズ（高さ → 幅）の昇順に並べた同一人物の画像どうしの類似度（左下半分のみ）。色は類似度の段階。",
                        legend, cell, lookup)


def _build_misidentify(ws, entries: list[GalleryEntry], counts: np.ndarray, n_nontarget: int,
                       lookup: _Lookup | None = None) -> None:
    """誤識別数シート: 縦 i × 横 j に「i との類似度が j との類似度より高い非対象者の画像の数」（行基準なので左下・右上とも出す）。"""
    legend = [(count_bin_label(count_bin(n)), count_fill(n)) for n in _COUNT_LEGEND]

    def cell(i: int, j: int):
        n = int(counts[i, j])
        return n, count_fill(n), "0"

    _build_matrix_sheet(ws, entries, "比較結果：誤識別画像数まとめ",
                        f"縦の画像（行）を基準に、横の画像（列）との類似度より高く出た非対象者の画像の数（非対象者 {n_nontarget} 枚の中）。"
                        "対象者どうしの類似度が高いものは数えない。色は件数の段階（0 / 1〜10 / 11〜20 / 21〜50 / 51〜100 / 以降 50 件刻み）。",
                        legend, cell, lookup)


def _build_rank(ws, entries: list[GalleryEntry], ranks: np.ndarray, n_tracks: int, lookup: _Lookup | None = None) -> None:
    """順位シート: 縦 i × 横 j に「非対象者の追跡IDのうち i との類似度が j より高い追跡IDの数 + 1」（行基準なので左下・右上とも出す）。"""
    legend = [(label, rank_fill(upper if upper is not None else 10_000)) for upper, label in RANK_BINS]

    def cell(i: int, j: int):
        r = int(ranks[i, j])
        return r, rank_fill(r), "0"

    _build_matrix_sheet(ws, entries, "比較結果：順位まとめ",
                        f"縦の画像（行）を基準に、横の画像（列）の類似度が非対象者の追跡ID {n_tracks} 件の中で何位か（同じ追跡IDの複数画像は 1 件。"
                        "対象者どうしは数えない）。色は順位の段階。",
                        legend, cell, lookup)


# ---- 対象画像情報（サンプル「対象画像情報【サンプル】」）----
INFO_HEADS = ("キー情報", "カメラ\nID", "映像ファイル名", "追跡\nID", "画像ファイル名", "顔の向き", "画像サイズ\n(幅:px)", "画像サイズ\n(高さ:px)",
              "顔の信頼度", "足元の座標", "距離", "全身画像", "顔画像")
INFO_FIRST_ROW = 3
INFO_ROW_PT = 74.25
INFO_WIDTHS = (50, 9, 20.625, 9, 28, 9, 11, 11, 11, 11, 11, 16, 16)  # B〜N 列
_INFO_BODY_COL, _INFO_FACE_COL = 13, 14                               # M 列 全身画像 / N 列 顔画像
DIST_DECIMALS = 2                                                      # 距離(m)の小数桁数


def _col_px(width_chars: float) -> int:
    return int(width_chars * 7 + 5)


# ---- セル内画像（右クリック →「セルに画像を貼り付け」と同じ形式。xlsx のリッチデータ）----
ORIGIN_PASTED = 5    # 画像そのものを貼ったセル（CalcOrigin）
ORIGIN_FORMULA = 4   # 式（VLOOKUP）の結果として画像を表示しているセル
_EMBED_SCALE = 3     # 埋め込む画像の大きさ = 対象画像情報シートのセルの大きさの何倍まで（これより大きい画像は縮小）
_RD_NS = "http://schemas.microsoft.com/office/spreadsheetml/2017/richdata"
_REL_BASE = "http://schemas.microsoft.com/office/2017/06/relationships"
_RD_VALUE_TYPES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<rvTypesInfo xmlns="http://schemas.microsoft.com/office/spreadsheetml/2017/richdata2" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" mc:Ignorable="x" '
    'xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><global><keyFlags>'
    '<key name="_Self"><flag name="ExcludeFromFile" value="1"/><flag name="ExcludeFromCalcComparison" value="1"/></key>'
    + "".join(f'<key name="{k}"><flag name="ExcludeFromCalcComparison" value="1"/></key>'
              for k in ("_DisplayString", "_Flags", "_Format", "_SubLabel", "_Attribution", "_Icon", "_Display",
                        "_CanonicalPropertyNames", "_ClassificationId"))
    + '</keyFlags></global></rvTypesInfo>')
_RD_STRUCTURE_XML = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<rvStructures xmlns="{_RD_NS}" count="1">'
    '<s t="_localImage"><k n="_rvRel:LocalImageIdentifier" t="i"/><k n="CalcOrigin" t="i"/></s></rvStructures>')
# (パッケージ内のパス, 内容種別, ワークブックからの関係の種類)
_RD_PARTS = (
    ("xl/metadata.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheetMetadata+xml",
     "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sheetMetadata"),
    ("xl/richData/rdrichvalue.xml", "application/vnd.ms-excel.rdrichvalue+xml", f"{_REL_BASE}/rdRichValue"),
    ("xl/richData/rdrichvaluestructure.xml", "application/vnd.ms-excel.rdrichvaluestructure+xml", f"{_REL_BASE}/rdRichValueStructure"),
    ("xl/richData/rdRichValueTypes.xml", "application/vnd.ms-excel.rdrichvaluetypes+xml", f"{_REL_BASE}/rdRichValueTypes"),
    ("xl/richData/richValueRel.xml", "application/vnd.ms-excel.richvaluerel+xml",
     "http://schemas.microsoft.com/office/2022/10/relationships/richValueRel"),
)


class _InCellImages:
    """セル内画像の登録と、保存済み xlsx への書き足し（inject）。

    mark() でセルを登録しておき、openpyxl で保存した後に inject() で 画像・richData・metadata.xml を足し、登録したセルに
    vm 属性（= 何番目のリッチデータか）を付ける。同じ画像（パス）は 1 回だけ埋め込み、同じ（画像, CalcOrigin）は同じ vm を共有する。"""

    def __init__(self):
        self.box_w = _EMBED_SCALE * _col_px(INFO_WIDTHS[_INFO_FACE_COL - 2])
        self.box_h = _EMBED_SCALE * int(INFO_ROW_PT * 96 / 72)
        self.jpegs: list[bytes] = []                 # 埋め込む画像（添字 = richValueRel の番号）
        self._rel_by_path: dict[Path, int | None] = {}
        self._rvs: list[tuple[int, int]] = []        # リッチデータ（画像の番号, CalcOrigin）。添字 + 1 が vm
        self._vm_by_rv: dict[tuple[int, int], int] = {}
        self.cells: dict[str, dict[str, int]] = {}   # シート名 -> {セル番地: vm}

    def _rel(self, path: Path) -> int | None:
        if path not in self._rel_by_path:
            rel = None
            img = imread(path)
            if img is not None:
                h, w = img.shape[:2]
                k = min(1.0, max(self.box_w / w, self.box_h / h))   # 大きい画像は埋め込みを小さくする（縦横比は変えない）
                if k < 1.0:
                    img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    self.jpegs.append(buf.tobytes())
                    rel = len(self.jpegs) - 1
            self._rel_by_path[path] = rel
        return self._rel_by_path[path]

    def mark(self, ws, row: int, col: int, path: Path | None, origin: int) -> bool:
        """row / col（1 始まり）のセルをセル内画像として登録する。画像が読めなければ False（登録しない）。"""
        rel = None if path is None else self._rel(Path(path))
        if rel is None:
            return False
        rv = (rel, origin)
        if rv not in self._vm_by_rv:
            self._rvs.append(rv)
            self._vm_by_rv[rv] = len(self._rvs)
        self.cells.setdefault(ws.title, {})[f"{get_column_letter(col)}{row}"] = self._vm_by_rv[rv]
        return True

    def inject(self, xlsx: Path, sheet_names: list[str]) -> None:
        """保存済みの xlsx（sheet_names = ブック内のシートの並び）にセル内画像を書き足す。登録が無ければ何もしない。"""
        if not self._rvs:
            return
        n_rv = len(self._rvs)
        rv_xml = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<rvData xmlns="{_RD_NS}" count="{n_rv}">'
                  + "".join(f'<rv s="0"><v>{rel}</v><v>{origin}</v></rv>' for rel, origin in self._rvs) + "</rvData>")
        bk = "".join(f'<bk><extLst><ext uri="{{3e2802c4-a4d2-4d8b-9148-e3be6c30e623}}"><xlrd:rvb i="{i}"/></ext></extLst></bk>'
                     for i in range(n_rv))
        meta_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                    '<metadata xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                    'xmlns:xlrd="http://schemas.microsoft.com/office/spreadsheetml/2017/richdata">'
                    '<metadataTypes count="1"><metadataType name="XLRICHVALUE" minSupportedVersion="120000" copy="1" pasteAll="1" '
                    'pasteValues="1" merge="1" splitFirst="1" rowColShift="1" clearFormats="1" clearComments="1" assign="1" coerce="1"/>'
                    f'</metadataTypes><futureMetadata name="XLRICHVALUE" count="{n_rv}">{bk}</futureMetadata>'
                    f'<valueMetadata count="{n_rv}">'
                    + "".join(f'<bk><rc t="1" v="{i}"/></bk>' for i in range(n_rv)) + '</valueMetadata></metadata>')
        n_img = len(self.jpegs)
        rel_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                   '<richValueRels xmlns="http://schemas.microsoft.com/office/spreadsheetml/2022/richvaluerel" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   + "".join(f'<rel r:id="rId{i + 1}"/>' for i in range(n_img)) + "</richValueRels>")
        rel_rels_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                        + "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                                  f'relationships/image" Target="../media/image{i + 1}.jpeg"/>' for i in range(n_img))
                        + "</Relationships>")
        new_files = {
            "xl/metadata.xml": meta_xml, "xl/richData/rdrichvalue.xml": rv_xml,
            "xl/richData/rdrichvaluestructure.xml": _RD_STRUCTURE_XML, "xl/richData/rdRichValueTypes.xml": _RD_VALUE_TYPES_XML,
            "xl/richData/richValueRel.xml": rel_xml, "xl/richData/_rels/richValueRel.xml.rels": rel_rels_xml,
        }
        sheet_xml = {f"xl/worksheets/sheet{i + 1}.xml": self.cells[name]
                     for i, name in enumerate(sheet_names) if name in self.cells}
        tmp = Path(str(xlsx) + ".tmp")
        with zipfile.ZipFile(xlsx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "[Content_Types].xml":
                    text = data.decode("utf-8")
                    add = "" if 'Extension="jpeg"' in text else '<Default Extension="jpeg" ContentType="image/jpeg"/>'
                    add += "".join(f'<Override PartName="/{p}" ContentType="{ct}"/>' for p, ct, _ in _RD_PARTS)
                    data = text.replace("</Types>", add + "</Types>").encode("utf-8")
                elif item.filename == "xl/_rels/workbook.xml.rels":
                    text = data.decode("utf-8")
                    add = "".join(f'<Relationship Id="rIdRd{i + 1}" Type="{t}" Target="{p[3:]}"/>' for i, (p, _, t) in enumerate(_RD_PARTS))
                    data = text.replace("</Relationships>", add + "</Relationships>").encode("utf-8")
                elif item.filename in sheet_xml:
                    data = _add_vm(data.decode("utf-8"), sheet_xml[item.filename]).encode("utf-8")
                zout.writestr(item, data)
            for name, text in new_files.items():
                zout.writestr(name, text.encode("utf-8"))
            for i, jpeg in enumerate(self.jpegs):
                zout.writestr(f"xl/media/image{i + 1}.jpeg", jpeg)
        tmp.replace(xlsx)


def _add_vm(sheet_xml: str, vms: dict[str, int]) -> str:
    """シート XML の指定セルを「セル内画像」にする（t="e" vm="N" + 値 #VALUE!。式のセルは式を残す。Excel が保存する形と同じ）。"""
    def fix(m: re.Match) -> str:
        vm = vms.get(m.group(1))
        if vm is None:
            return m.group(0)
        attrs = re.sub(r'\s+t="[^"]*"', "", m.group(2))
        f = re.search(r"<f[ >/].*?(?:</f>|(?<=/)>)", m.group(3) or "", re.S)
        return f'<c r="{m.group(1)}"{attrs} t="e" vm="{vm}">{f.group(0) if f else ""}<v>#VALUE!</v></c>'

    return re.sub(r'<c r="([A-Z]+\d+)"((?:\s+[A-Za-z:]+="[^"]*")*)\s*(?:/>|>(.*?)</c>)', fix, sheet_xml, flags=re.S)


def _build_info_sheet(ws, entries: list[GalleryEntry], images: _InCellImages) -> None:
    """検証対象人物の顔画像情報の一覧。キー情報は関数式（カメラID|映像ファイル名|追跡ID|画像ファイル名）。
    画像サイズは顔 bbox の幅・高さ、足元の座標は「(x, y)」（いずれも追跡IDごとの JSON から）。全身画像・顔画像はセル内画像（「セルに画像を貼り付け」と同じ形式）。"""
    ws.sheet_view.showGridLines = False
    for k, h in enumerate(INFO_HEADS):
        _put(ws, INFO_FIRST_ROW - 1, 2 + k, h, head=True)
    ws.row_dimensions[INFO_FIRST_ROW - 1].height = 32.25
    for i, e in enumerate(entries):
        r = INFO_FIRST_ROW + i
        cam, video, track, fname, direction, w, h = _values(e)
        conf = NA if e.face_conf is None else round(float(e.face_conf), 4)
        vals = [f'=C{r}&"|"&D{r}&"|"&E{r}&"|"&F{r}', cam, video, track, fname, direction, w, h, conf, _foot_text(e), _dist_value(e)]
        for k, v in enumerate(vals):
            _put(ws, r, 2 + k, v)
        for col, path in ((_INFO_BODY_COL, e.body_path), (_INFO_FACE_COL, e.face_path)):   # M 列 全身画像 / N 列 顔画像
            _put(ws, r, col, "#VALUE!" if images.mark(ws, r, col, path, ORIGIN_PASTED) else None)
        ws.row_dimensions[r].height = INFO_ROW_PT
    ws.column_dimensions["A"].width = 3
    for k, w in enumerate(INFO_WIDTHS):
        ws.column_dimensions[get_column_letter(2 + k)].width = w
    last = INFO_FIRST_ROW + max(len(entries), 1) - 1
    ws.auto_filter.ref = f"B{INFO_FIRST_ROW - 1}:{get_column_letter(1 + len(INFO_HEADS))}{last}"
    ws.freeze_panes = f"B{INFO_FIRST_ROW}"
    _apply_font(ws)


def _build_conditions(ws, cond: VerifyConditions, entries: list[GalleryEntry]) -> None:
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 80
    rows = [
        ("生成日時", cond.generated_at.strftime("%Y/%m/%d %H:%M:%S")),
        ("顔特徴量抽出モデル", cond.model_name),
        ("顔ランドマークモデル", cond.landmark_name),
        ("顔の採用基準", cond.gate_filter_text),
        ("前処理 グレースケール化 + 明るさ補正（CLAHE）", "あり（アライン後の顔画像に適用）" if cond.grayscale else "なし"),
        ("前処理 168×168 アライメント", "あり（168×168 でアライン → 112×112 に縮小して特徴量化）" if cond.align168 else "なし"),
        ("画像数", len(entries)),
        ("追跡ID数", len({e.track_key for e in entries})),
    ]
    if cond.n_nontarget is not None:
        rows += [("比較対象フォルダ", cond.gallery_dir), ("非対象者の画像数", cond.n_nontarget)]
        # テキスト一覧を出力する場合のみ: rows.append(("誤認した画像の一覧（テキスト）", cond.log_text_name))
    if cond.n_nontarget_tracks is not None:
        rows.append(("非対象者の追跡ID数", cond.n_nontarget_tracks))
    for i, (k, v) in enumerate(rows, start=1):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        ws.cell(row=i, column=2, value=v).alignment = Alignment(wrap_text=True, vertical="top")


def build_verify_face_size_report(path: Path, cond: VerifyConditions, entries: list[GalleryEntry], sim: np.ndarray | None,
                                  log: PhaseLogger, counts: np.ndarray | None = None,
                                  info_entries: list[GalleryEntry] | None = None, ranks: np.ndarray | None = None) -> None:
    """Excel を作成して保存する。先頭に「対象画像情報」（info_entries があれば。並びは呼び出し側で決める）、sim があれば類似度シート、
    counts があれば誤識別数シートを作る。ログシートは全フェーズ終了後に追記して再保存する。
    セル内画像（全身画像・顔画像）は openpyxl では入れられないので、最後の保存の後に xlsx へ書き足す（_InCellImages.inject）。"""
    images = _InCellImages()
    wb = Workbook()
    wb.remove(wb.active)
    wb.calculation.fullCalcOnLoad = True   # 開いたときに式（VLOOKUP・キー）を計算する
    # 対象画像情報シートが無いと VLOOKUP が引けないので、行列シートの顔画像・顔の信頼度・足元の座標は info_entries があるときだけ出す
    lookup = _Lookup(images, len(info_entries)) if info_entries is not None else None
    with log.phase("Excel作成・保存"):
        if info_entries is not None:
            _build_info_sheet(wb.create_sheet(INFO_SHEET_NAME), info_entries, images)
        if sim is not None:
            _build_summary(wb.create_sheet(SHEET_NAME), entries, sim, lookup)
        if counts is not None:
            _build_misidentify(wb.create_sheet(MISIDENTIFY_SHEET_NAME), entries, counts, cond.n_nontarget or 0, lookup)
        if ranks is not None:
            _build_rank(wb.create_sheet(RANK_SHEET_NAME), entries, ranks, cond.n_nontarget_tracks or 0, lookup)
        _build_conditions(wb.create_sheet("条件"), cond, entries)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)
    log.finish()
    _build_log_sheet(wb.create_sheet("ログ"), log, "ログシート追記のための再保存は記録対象外")
    wb.save(path)
    images.inject(Path(path), wb.sheetnames)
