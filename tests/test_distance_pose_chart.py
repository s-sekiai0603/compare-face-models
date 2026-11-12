"""距離・向き・精度分析チャート（顔サイズ別精度検証の Excel に「距離向き精度分析チャート」シートを追加する）。"""
import io
import re
import shutil
import zipfile

import numpy as np
import openpyxl
import pytest

from compare_face.core.report import distance_pose_chart as dpc
from compare_face.core.report.distance_pose_chart import (
    CHART_SHEET_NAME, DIRECTIONS, ICON_DIR, ChartData, ChartPoint, add_chart_sheet, icon_file, nice_axis,
    read_chart_data, recolor_icon)
from compare_face.core.report.verify_face_size import run_verify_face_size
from compare_face.core.report.verify_face_size_excel import INFO_SHEET_NAME, MISIDENTIFY_SHEET_NAME
from test_camera_distance import _camera_dir
from test_query_set import _MeanExtractor
from test_verify_face_size import _track

# 仕様の表（縦×横）。入れ替えても同じ
SPEC = {
    ("正面", "正面"): "front_front", ("正面", "左横"): "front_left", ("正面", "右横"): "front_right",
    ("正面", "左斜め正面"): "front_l_diagonal", ("正面", "右斜め正面"): "front_r_diagonal",
    ("左横", "正面"): "front_left", ("左横", "左横"): "left_left", ("左横", "右横"): "right_left",
    ("左横", "左斜め正面"): "left_l_diagonal", ("左横", "右斜め正面"): "left_r_diagonal",
    ("右横", "正面"): "front_right", ("右横", "左横"): "right_left", ("右横", "右横"): "right_right",
    ("右横", "左斜め正面"): "right_l_diagonal", ("右横", "右斜め正面"): "right_r_diagonal",
    ("左斜め正面", "正面"): "front_l_diagonal", ("左斜め正面", "左横"): "left_l_diagonal",
    ("左斜め正面", "右横"): "right_l_diagonal", ("左斜め正面", "左斜め正面"): "l_diagonal_l_diagonal",
    ("左斜め正面", "右斜め正面"): "diagonal_diagonal",
    ("右斜め正面", "正面"): "front_r_diagonal", ("右斜め正面", "左横"): "left_r_diagonal",
    ("右斜め正面", "右横"): "right_r_diagonal", ("右斜め正面", "左斜め正面"): "diagonal_diagonal",
    ("右斜め正面", "右斜め正面"): "r_diagonal_r_diagonal",
}


def test_icon_table_matches_spec():
    assert len(SPEC) == 25
    for (a, b), name in SPEC.items():
        assert icon_file(a, b) == f"{name}.png", (a, b)
        assert (ICON_DIR / f"{name}.png").is_file()
    assert icon_file("正面", "不明") is None and icon_file("", "正面") is None


def test_recolor_icon_changes_green_only():
    import cv2
    src = cv2.imread(str(ICON_DIR / "front_left.png"), cv2.IMREAD_UNCHANGED)
    out = cv2.imdecode(np.frombuffer(recolor_icon(ICON_DIR / "front_left.png", (0xF8, 0x69, 0x6B)), np.uint8), cv2.IMREAD_UNCHANGED)
    assert out.shape == (38, 38, 4)
    assert src[19, 3, :3].tolist() == [123, 190, 99]            # もとは緑（BGR）
    b, g, r = (int(v) for v in out[19, 3, :3])
    assert r > 200 and g < 150 and b < 150                         # 対象色（赤系）に染まる
    assert out[0, 0, 3] == 0 and out[19, 19, 3] == 255             # 外側の白い角は透過、円の中は不透明


def test_nice_axis_covers_values():
    lo, hi, step = nice_axis([3.2, 9.7])
    assert (lo, hi, step) == (0, 10 + 5, 5)                      # 5 m の倍数に合わせ、最小・最大の外側に余白
    assert nice_axis([5.0, 5.0]) == (0, 10, 5)


def _report(tmp_path, monkeypatch):
    import compare_face.core.report.verify_face_size as vfs
    monkeypatch.setattr(vfs, "get_extractor", lambda name: _MeanExtractor())
    a = _track(tmp_path / "f", "Cam1", "v1", 1, [("20260311_000001_000", (60, 80)), ("20260311_000002_000", (30, 40))])
    b = _track(tmp_path / "f", "Cam2", "v2", 2, [("20260311_000003_000", (45, 60))])   # 設定なし → 距離「-」
    _track(tmp_path / "f", "Cam3", "v3", 3, [("20260311_000005_000", (47, 62)), ("20260311_000006_000", (48, 63))])   # 非対象者
    # 行列シートの並びは顔 bbox のサイズ順: Cam1(30×40) → Cam2(45×60) → Cam1(60×80)
    out, _ = run_verify_face_size([a, b], tmp_path / "rep", "AdaFace", "なし", mode="misidentify", gallery_dir=tmp_path / "f",
                                  camera_dir=_camera_dir(tmp_path), echo=lambda m: None)
    return out


def _expected_dist():
    from test_camera_distance import _expected
    return round(_expected((100.4, 200.6)), 2)


def _set_directions(path, dirs):
    """対象画像情報の「顔の向き」（G 列）に値を入れる（openpyxl で保存し直すのでセル内画像は失われる = データ読取のテスト用）。"""
    wb = openpyxl.load_workbook(path)
    ws = wb[INFO_SHEET_NAME]
    for r, d in enumerate(dirs, start=3):
        ws.cell(row=r, column=7, value=d)
    wb.save(path)


def test_read_chart_data_resolves_via_info_sheet(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    with pytest.raises(ValueError):                 # 顔の向きが空欄 → 点が無い
        add_chart_sheet(out)
    data = read_chart_data(out)
    assert data.points == [] and data.no_direction == 3 and data.no_distance == 1   # Cam2 は距離なし
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    data = read_chart_data(out)
    # 3 枚のうち距離があるのは Cam1 の 2 枚だけ → 縦×横の 2 点（対角を除く）。対象画像情報は入力順（Cam1 の 60×80, 30×40, Cam2）なので
    # Cam1 の 2 枚の向きは 正面 × 左横（行列シートの並びとは別。キーで引いている）
    assert len(data.points) == 2 and data.no_distance == 1 and data.no_direction == 0
    assert {p.icon for p in data.points} == {"front_left.png"}
    assert data.points[0].x == pytest.approx(_expected_dist())         # 足元は同じ座標（テストの画像は同じ json）→ 同じ距離
    # 色 = 誤識別数シートの縦 i × 横 j のセルの背景色。(i=0, j=2) と (i=2, j=0)
    ws = openpyxl.load_workbook(out)[MISIDENTIFY_SHEET_NAME]
    rgb = lambda r, c: tuple(int(ws.cell(row=r, column=c).fill.fgColor.rgb[-6:][k:k + 2], 16) for k in (0, 2, 4))
    assert data.points[0].rgb == rgb(16, 15) and data.points[1].rgb == rgb(18, 13)


def test_settings_roundtrip(tmp_path):
    from compare_face.core.common.settings import load_distance_pose_chart_settings, save_distance_pose_chart_settings
    p = tmp_path / "s.json"
    assert load_distance_pose_chart_settings(p)["excel_path"] == ""
    save_distance_pose_chart_settings({"excel_path": r"C:\a\b.xlsx"}, p)
    assert load_distance_pose_chart_settings(p)["excel_path"] == r"C:\a\b.xlsx"


_FONT = pytest.mark.skipif(dpc.find_japanese_font() is None, reason="日本語フォントが無い環境")


@_FONT
def test_add_chart_sheet_pastes_one_picture_keeps_in_cell_images_and_replaces(tmp_path, monkeypatch):
    """チャートは Python で描いた画像 1 枚を貼る（Excel のチャート機能は使わない）。セル内画像・式は残り、再実行は置き換え。"""
    out = _report(tmp_path, monkeypatch)
    pts = [ChartPoint(5.0, 8.0, "front_left.png", (0x63, 0xBE, 0x7B)), ChartPoint(8.0, 5.0, "front_front.png", (0xF8, 0x69, 0x6B))]
    monkeypatch.setattr(dpc, "read_chart_data", lambda p: ChartData(points=pts, n_images=3))
    before = zipfile.ZipFile(out)
    rich_before = {n: before.read(n) for n in before.namelist() if "richData" in n or n.startswith("xl/media/")}
    sheet2 = before.read("xl/worksheets/sheet2.xml")
    before.close()

    for _ in range(2):                                                               # 2 回目 = 置き換え（重複しない）
        add_chart_sheet(out)
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            assert not any(n.startswith("xl/charts/") for n in names)                # Excel のチャートは使わない
            drawing = z.read("xl/drawings/drawing1.xml").decode("utf-8")
            assert drawing.count("<pic>") == 1 and drawing.count("<absoluteAnchor>") == 1
            pngs = [n for n in names if n.startswith("xl/media/chart_icon_")]
            assert len(pngs) == 1
            from PIL import Image
            assert Image.open(io.BytesIO(z.read(pngs[0]))).size == (dpc.CHART_W_PX, dpc.CHART_H_PX)
            assert all(z.read(n) == v for n, v in rich_before.items())                # セル内画像はそのまま
            assert z.read("xl/worksheets/sheet2.xml") == sheet2                       # 既存シート（VLOOKUP 式）もそのまま
            assert z.read("xl/workbook.xml").decode("utf-8").count(CHART_SHEET_NAME) == 1
            assert z.testzip() is None
    wb = openpyxl.load_workbook(out)                                                  # openpyxl でも読める（壊れていない）
    assert wb.sheetnames[-1] == CHART_SHEET_NAME and wb[CHART_SHEET_NAME].cell(row=1, column=1).value == 5.0


def test_old_chart_sheet_with_excel_chart_is_replaced(tmp_path, monkeypatch):
    """旧版（Excel のチャート + アイコン画像）で作ったシートがあっても、チャートごと取り除いて画像のシートに置き換える。"""
    out = _report(tmp_path, monkeypatch)
    old = tmp_path / "old.xlsx"
    from openpyxl.chart import Reference, ScatterChart, Series
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = CHART_SHEET_NAME
    ws.append([1, 2])
    ch = ScatterChart()
    ch.series.append(Series(Reference(ws, min_col=2, min_row=1, max_row=1), Reference(ws, min_col=1, min_row=1, max_row=1)))
    ws.add_chart(ch)
    wb.save(old)
    dpc._graft(out, old)                                                            # 旧版のシートを作る
    with zipfile.ZipFile(out) as z:
        assert any(n.startswith("xl/charts/chart") for n in z.namelist())
    pts = [ChartPoint(5.0, 8.0, "front_left.png", (0x63, 0xBE, 0x7B))]
    monkeypatch.setattr(dpc, "read_chart_data", lambda p: ChartData(points=pts, n_images=3))
    monkeypatch.setattr(dpc, "render_chart_image", lambda data, icon_dir, split_lr=False, unit_m=0.0, label="", transparent=False, axis_points=None: b"\x89PNG\r\n\x1a\n")   # 画像の中身はここでは問わない
    monkeypatch.setattr(dpc, "XlImage", lambda b: __import__("openpyxl").drawing.image.Image(
        io.BytesIO(_tiny_png())))
    add_chart_sheet(out)
    with zipfile.ZipFile(out) as z:
        assert not any(n.startswith("xl/charts/") for n in z.namelist())
        assert z.read("xl/workbook.xml").decode("utf-8").count(CHART_SHEET_NAME) == 1


def _tiny_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, "PNG")
    return buf.getvalue()


@_FONT
def test_render_chart_image_places_icons_on_axes(tmp_path):
    """アイコンは (横の距離, 縦の距離) の位置に、色は指定どおりに描かれる。5 m の線は実線、1 m の線は破線。"""
    from PIL import Image
    pts = [ChartPoint(2.0, 4.0, "front_front.png", (0xC0, 0x00, 0x00)), ChartPoint(6.0, 8.0, "front_front.png", (0x63, 0xBE, 0x7B))]
    img = Image.open(io.BytesIO(dpc.render_chart_image(ChartData(points=pts, n_images=2), ICON_DIR))).convert("RGB")
    assert img.size == (dpc.CHART_W_PX, dpc.CHART_H_PX)
    lo, hi, _ = nice_axis([2.0, 4.0, 6.0, 8.0])
    px, py, pw, ph = dpc.PLOT
    W, H = dpc.CHART_W_PX, dpc.CHART_H_PX
    X = lambda v: (px + (v - lo) / (hi - lo) * pw) * W
    Y = lambda v: (py + (1 - (v - lo) / (hi - lo)) * ph) * H
    for (x, y), rgb in (((2.0, 4.0), (0xC0, 0, 0)), ((6.0, 8.0), (0x63, 0xBE, 0x7B))):
        r, g, b = img.getpixel((int(X(x)), int(Y(y))))
        assert max(abs(r - rgb[0]), abs(g - rgb[1]), abs(b - rgb[2])) < 60, (x, y)    # 中心付近は対象色（円の地）
    # 5 m の位置の縦線は全域で連続した実線 / 1 m の位置の縦線は途切れる（破線）
    x5, x1 = int(X(5)), int(X(1 if lo <= 1 else lo + 1))
    ys = range(int(py * H) + 5, int((py + ph) * H) - 5)
    solid = sum(img.getpixel((x5, y))[0] < 200 for y in ys) / len(ys)
    dashed = sum(img.getpixel((x1, y))[0] < 200 for y in ys) / len(ys)
    assert solid > 0.95 and 0.3 < dashed < 0.8


def test_image_chart_needs_japanese_font(monkeypatch):
    monkeypatch.setattr(dpc, "find_japanese_font", lambda: None)
    with pytest.raises(ValueError, match="フォント"):
        dpc.render_chart_image(ChartData(points=[ChartPoint(1, 2, "front_front.png", (1, 2, 3))]), ICON_DIR)


def test_split_xs_by_horizontal_direction():
    """横の画像の顔の向きに「右」を含めば右側(+)、「左」を含めば左側(-)、どちらも含まなければ両側。"""
    assert dpc.split_xs("右横", 5.0) == [5.0] and dpc.split_xs("右斜め正面", 5.0) == [5.0]
    assert dpc.split_xs("左横", 5.0) == [-5.0] and dpc.split_xs("左斜め正面", 5.0) == [-5.0]
    assert dpc.split_xs("正面", 5.0) == [5.0, -5.0]
    pts = [ChartPoint(5.0, 8.0, "front_left.png", (1, 2, 3), "正面"), ChartPoint(3.0, 8.0, "left_left.png", (1, 2, 3), "左横"),
           ChartPoint(4.0, 6.0, "right_right.png", (1, 2, 3), "右横")]
    assert [(x, y) for x, y, _, _ in dpc.plot_points(pts, True)] == [(5.0, 8.0), (-5.0, 8.0), (-3.0, 8.0), (4.0, 6.0)]
    assert len(dpc.plot_points(pts, False)) == 3
    lo, hi, ylo, yhi, step = dpc.axis_ranges(pts, True)
    assert (lo, hi) == (-hi, hi) and ylo >= 0 and yhi == hi and step == 5                     # 横軸は 0 を中心に左右対称
    assert dpc.axis_ranges(pts, False)[:2] == dpc.axis_ranges(pts, False)[2:4]


@_FONT
def test_render_split_lr_places_icons_left_and_right(tmp_path):
    from PIL import Image
    pts = [ChartPoint(5.0, 8.0, "front_front.png", (0xC0, 0x00, 0x00), "正面"), ChartPoint(3.0, 4.0, "left_left.png", (0x63, 0xBE, 0x7B), "左横")]
    img = Image.open(io.BytesIO(dpc.render_chart_image(ChartData(points=pts, n_images=2), ICON_DIR, split_lr=True))).convert("RGB")
    xlo, xhi, ylo, yhi, _ = dpc.axis_ranges(pts, True)
    px, py, pw, ph = dpc.PLOT
    W, H = dpc.CHART_W_PX, dpc.CHART_H_PX
    X = lambda v: (px + (v - xlo) / (xhi - xlo) * pw) * W
    Y = lambda v: (py + (1 - (v - ylo) / (yhi - ylo)) * ph) * H

    def near(x, y, rgb):
        c = img.getpixel((int(X(x)), int(Y(y))))
        return max(abs(c[0] - rgb[0]), abs(c[1] - rgb[1]), abs(c[2] - rgb[2])) < 60

    assert near(5.0, 8.0, (0xC0, 0, 0)) and near(-5.0, 8.0, (0xC0, 0, 0))                    # 正面 = 両側
    assert near(-3.0, 4.0, (0x63, 0xBE, 0x7B)) and not near(3.0, 4.0, (0x63, 0xBE, 0x7B))     # 左横 = 左側だけ


def test_split_setting_roundtrip(tmp_path):
    from compare_face.core.common.settings import load_distance_pose_chart_settings, save_distance_pose_chart_settings
    p = tmp_path / "s.json"
    assert load_distance_pose_chart_settings(p)["split_lr"] is False
    save_distance_pose_chart_settings({"excel_path": "", "split_lr": True}, p)
    assert load_distance_pose_chart_settings(p)["split_lr"] is True


def _agg_data():
    """画像 4 枚（縦 = 横の並び）: 0: 1.2 m 正面 / 1: 1.8 m 正面 / 2: 3.5 m 左横 / 3: 7.0 m 正面。セルの値 = 100 * i + j。"""
    imgs = [(1.2, "正面"), (1.8, "正面"), (3.5, "左横"), (7.0, "正面")]
    cells = {(i, j): float(100 * i + j) for i in range(4) for j in range(4) if i != j}
    return ChartData(points=[], n_images=4, ver=list(imgs), hor=list(imgs), cells=cells)


def test_aggregate_points_by_unit_and_direction():
    data = _agg_data()
    pts = dpc.aggregate_points(data, 2.0)          # 区間: 0〜2 m = {0, 1} 正面 / 2〜4 m = {2} 左横 / 6〜8 m = {3} 正面
    assert len(pts) == 3 * 3 - 2                   # 対角だけの組合せ（2 → 2, 3 → 3）は値が無いので点にならない
    by = {(round(p.y, 3), round(p.x, 3)): p for p in pts}
    # 縦 {0,1}(正面, 平均 1.5 m) × 横 {0,1}(正面, 平均 1.5 m): セルは (0,1) と (1,0) = 1 と 100 → 平均 50.5（対角は除く）
    p = by[(1.5, 1.5)]
    assert p.icon == "front_front.png" and p.h_dir == "正面"
    from compare_face.core.report.verify_face_size_excel import count_fill
    rgb = count_fill(50.5).fgColor.rgb[-6:]
    assert p.rgb == (int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))
    # 縦 {2}(左横, 3.5 m) × 横 {0,1}(正面, 平均 1.5 m): セル (2,0)=200, (2,1)=201 → 平均 200.5
    p = by[(3.5, 1.5)]
    assert p.icon == "front_left.png"
    assert p.rgb == tuple(int(count_fill(200.5).fgColor.rgb[-6:][k:k + 2], 16) for k in (0, 2, 4))
    assert (7.0, 7.0) not in by and (3.5, 3.5) not in by and (7.0, 1.5) in by and (1.5, 7.0) in by
    with pytest.raises(ValueError):
        dpc.aggregate_points(data, 0)


def test_aggregate_skips_images_without_distance_or_direction():
    data = _agg_data()
    data.ver[1] = (None, "正面")
    data.hor[2] = (3.5, None)
    pts = dpc.aggregate_points(data, 2.0)
    assert all(p.x != 3.5 for p in pts) and all(p.y != 1.8 for p in pts)
    assert len({(p.x, p.y) for p in pts}) == len(pts)


def test_aggregate_real_workbook_reduces_points(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    data = read_chart_data(out)
    assert data.cells and len(data.ver) == 3
    agg = dpc.aggregate_points(data, 100.0)                        # 距離をすべて 1 区間にまとめる → 向きの組合せ数まで減る
    assert 0 < len(agg) <= len(data.points)


def test_unit_setting_roundtrip(tmp_path):
    from compare_face.core.common.settings import load_distance_pose_chart_settings, save_distance_pose_chart_settings
    p = tmp_path / "s.json"
    assert load_distance_pose_chart_settings(p)["unit_m"] == 0.0
    save_distance_pose_chart_settings({"excel_path": "", "split_lr": False, "unit_m": 2.5}, p)
    assert load_distance_pose_chart_settings(p)["unit_m"] == 2.5


@_FONT
def test_add_chart_sheet_with_unit_and_split(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    data = add_chart_sheet(out, split_lr=True, unit_m=5.0)
    assert data.n_groups == len(data.points) > 0 and data.n_icons >= len(data.points)
    assert openpyxl.load_workbook(out).sheetnames[-1] == CHART_SHEET_NAME


def _cam_data():
    """画像 4 枚: 0,1 = cam1 / 2,3 = cam2。縦 = 横の並び。セルの値 = 10 * i + j。"""
    imgs = [(1.0, "正面"), (2.0, "左横"), (3.0, "正面"), (4.0, "右横")]
    cams = ["cam1", "cam1", "cam2", "cam2"]
    cells = {(i, j): float(10 * i + j) for i in range(4) for j in range(4) if i != j}
    pts = [ChartPoint(imgs[j][0], imgs[i][0], dpc.ICON_FILES[frozenset((imgs[i][1], imgs[j][1]))], (1, 2, 3), imgs[j][1], cams[i], cams[j])
           for i in range(4) for j in range(4) if i != j]
    return ChartData(points=pts, n_images=4, ver=list(imgs), hor=list(imgs), ver_cam=list(cams), hor_cam=list(cams), cells=cells)


def test_chart_targets_all_and_per_camera():
    data = _cam_data()
    assert dpc.cameras_of(data) == ["cam1", "cam2"]
    only_all = dpc.chart_targets(data, True, False)
    assert [l for l, _ in only_all] == ["すべてのカメラ"] and len(only_all[0][1]) == 12
    only_each = dpc.chart_targets(data, False, True)
    assert [l for l, _ in only_each] == ["カメラ cam1", "カメラ cam2"]
    assert [len(p) for _, p in only_each] == [2, 2]                                    # 縦・横とも同じカメラの組合せだけ（各 2 画像 → 2 組）
    assert all(p.cam_v == p.cam_h == "cam1" for p in only_each[0][1])
    both = dpc.chart_targets(data, True, True)                                         # カメラ 2 台 + すべて = 3 個
    assert [l for l, _ in both] == ["すべてのカメラ", "カメラ cam1", "カメラ cam2"]
    with pytest.raises(ValueError):
        dpc.chart_targets(data, False, False)


def test_chart_targets_with_unit_filters_by_camera():
    data = _cam_data()
    both = dict(dpc.chart_targets(data, True, True, unit_m=10.0))                      # 距離がすべて 1 区間 → 向きの組合せごとに 1 点
    cam1 = both["カメラ cam1"]
    assert len(cam1) == 2 and {round(p.x, 2) for p in cam1} == {1.0, 2.0}              # cam1 の画像だけの平均（正面 1.0 / 左横 2.0）
    assert len(both["すべてのカメラ"]) > len(cam1)


def test_chart_targets_limited_to_selected_cameras():
    data = _cam_data()
    both = dpc.chart_targets(data, True, True, cameras=["cam1"])
    assert [l for l, _ in both] == ["すべてのカメラ", "カメラ cam1"]                    # cam2 のチャートは出ない
    assert len(both[0][1]) == 2 and all(p.cam_v == p.cam_h == "cam1" for p in both[0][1])   # すべて = 選んだカメラどうしだけ
    both_u = dpc.chart_targets(data, True, True, unit_m=10.0, cameras={"cam2"})
    assert [l for l, _ in both_u] == ["すべてのカメラ", "カメラ cam2"]
    assert dpc.chart_targets(data, True, False, cameras=None)[0][1] == dpc.chart_targets(data, True, False)[0][1]
    assert dpc.chart_targets(data, True, True, cameras=[]) == []                      # 何も選ばなければ点なし


@_FONT
def test_add_chart_sheet_all_and_each_camera_stacks_charts(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    data = add_chart_sheet(out, all_cameras=True, per_camera=True)
    assert [l for l, _ in data.charts] == ["すべてのカメラ", "カメラ Cam1"]               # Cam2 は距離が無いので点が無い = チャートなし
    with zipfile.ZipFile(out) as z:
        drawing = z.read("xl/drawings/drawing1.xml").decode("utf-8")
        assert drawing.count("<pic>") == 2                                                # チャート 2 個 = 画像 2 枚（縦に並ぶ）
        ys = sorted(int(m) // 9525 for m in re.findall(r'<absoluteAnchor><pos x="\d+" y="(\d+)"', drawing))
        assert ys == [dpc.CHART_Y0_PX, dpc.CHART_Y0_PX + dpc.CHART_H_PX + dpc.CHART_GAP_PX]
    only_each = add_chart_sheet(out, all_cameras=False, per_camera=True)                  # 置き換え。カメラ個別だけ
    assert [l for l, _ in only_each.charts] == ["カメラ Cam1"]
    with zipfile.ZipFile(out) as z:
        assert z.read("xl/drawings/drawing1.xml").decode("utf-8").count("<pic>") == 1
    with pytest.raises(ValueError):
        add_chart_sheet(out, all_cameras=False, per_camera=False)


def test_camera_setting_roundtrip(tmp_path):
    from compare_face.core.common.settings import load_distance_pose_chart_settings, save_distance_pose_chart_settings
    p = tmp_path / "s.json"
    s = load_distance_pose_chart_settings(p)
    assert s["all_cameras"] is True and s["per_camera"] is False
    save_distance_pose_chart_settings({**s, "all_cameras": False, "per_camera": True}, p)
    s = load_distance_pose_chart_settings(p)
    assert s["all_cameras"] is False and s["per_camera"] is True


def _rgb_of(n):
    from compare_face.core.report.verify_face_size_excel import count_fill
    h = count_fill(n).fgColor.rgb[-6:]
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def test_color_groups_cover_all_bins_and_legend_has_seven():
    assert len(dpc.COLOR_GROUPS) == 7
    assert [dpc.color_group(_rgb_of(n)) for n in (0, 1, 9, 10, 20, 50, 100, 300, 449, 450, 10_000)] == [0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6]
    assert dpc.color_group((1, 2, 3)) is None
    colors, dirs = dpc.legend_items()
    assert len(colors) == 7 and len(dirs) == 15 == len(dpc.icon_choices())


def test_filter_points_by_color_and_icon():
    pts = [ChartPoint(1, 1, "front_front.png", _rgb_of(0)), ChartPoint(2, 2, "front_left.png", _rgb_of(30)),
           ChartPoint(3, 3, "front_front.png", _rgb_of(30)), ChartPoint(4, 4, "front_front.png", (1, 2, 3))]
    assert dpc.filter_points(pts) == pts
    assert [p.x for p in dpc.filter_points(pts, colors={0})] == [1, 4]                      # 段階に当てはまらない色は色では除かない
    assert [p.x for p in dpc.filter_points(pts, colors={3})] == [2, 3, 4]
    assert [p.x for p in dpc.filter_points(pts, icons={"front_left.png"})] == [2]
    assert [p.x for p in dpc.filter_points(pts, colors={0, 3}, icons={"front_front.png"})] == [1, 3, 4]
    assert dpc.filter_points(pts, colors=set()) == [p for p in pts if dpc.color_group(p.rgb) is None]


@_FONT
def test_add_chart_sheet_with_color_and_icon_selection(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    all_ = add_chart_sheet(out)
    first = dpc.read_chart_data(out).points[0].icon
    only = add_chart_sheet(out, icon_files={first})
    assert 0 < only.n_icons <= all_.n_icons                                                   # アイコンを絞ると点が減る
    with pytest.raises(ValueError):
        add_chart_sheet(out, icon_files=set())                                                    # 何も残らなければエラー



def test_render_chart_image_transparent_background():
    """transparent=True は背景（プロット領域の内側も含む）が透明な RGBA、既定は不透明な白。"""
    import io
    from PIL import Image
    pts = [ChartPoint(1, 1, "front_front.png", (0x63, 0xBE, 0x7B))]
    data = ChartData(points=pts, n_images=2)
    opaque = Image.open(io.BytesIO(dpc.render_chart_image(data, ICON_DIR)))
    assert opaque.mode == "RGB" and opaque.getpixel((5, 5)) == (255, 255, 255)
    img = Image.open(io.BytesIO(dpc.render_chart_image(data, ICON_DIR, transparent=True)))
    assert img.mode == "RGBA" and img.size == opaque.size
    px, py, pw, ph = dpc.PLOT
    W, H = img.size
    assert img.getpixel((5, 5))[3] == 0                                   # 外側の余白
    assert img.getpixel((int((px + pw * 0.5) * W) + 7, int((py + ph * 0.5) * H) + 7))[3] == 0   # プロット領域の内側（グリッドの間）
    assert any(img.getpixel((x, y))[3] == 255 for x in range(0, W, 7) for y in range(0, H, 7))   # アイコン・線は不透明


def test_chart_png_path_sanitizes_label(tmp_path):
    x = tmp_path / "report.xlsx"
    assert dpc.chart_png_path(x, "").name == "report_チャート.png"
    assert dpc.chart_png_path(x, "Cam1").name == "report_チャート_Cam1.png"
    assert dpc.chart_png_path(x, 'a/b:c d').name == "report_チャート_a_b_c_d.png"


def test_add_chart_sheet_output_modes(tmp_path, monkeypatch):
    from PIL import Image
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    assert add_chart_sheet(out).png_files == []                                    # 既定は Excel のみ
    assert CHART_SHEET_NAME in openpyxl.load_workbook(out).sheetnames
    data = add_chart_sheet(out, output_png=True, transparent=True)                 # 両方
    assert len(data.png_files) == 1 and data.png_files[0].parent == out.parent
    img = Image.open(data.png_files[0])
    assert img.mode == "RGBA" and img.getpixel((5, 5))[3] == 0
    data.png_files[0].unlink()
    data = add_chart_sheet(out, output_png=True)                                   # 透過しない設定なら PNG も白背景
    assert Image.open(data.png_files[0]).getpixel((5, 5))[:3] == (255, 255, 255)


def test_add_chart_sheet_png_only_leaves_excel_untouched(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    before = out.read_bytes()
    data = add_chart_sheet(out, output_excel=False, output_png=True)
    assert len(data.png_files) == 1 and data.png_files[0].is_file()
    assert out.read_bytes() == before                                              # Excel は変更しない
    with pytest.raises(ValueError):
        add_chart_sheet(out, output_excel=False, output_png=False)


def test_combination_charts_split_by_color_and_icon():
    pts = [ChartPoint(1, 1, "front_front.png", _rgb_of(0)), ChartPoint(2, 2, "front_front.png", _rgb_of(0)),
           ChartPoint(3, 3, "front_left.png", _rgb_of(0)), ChartPoint(4, 4, "front_front.png", _rgb_of(30)),
           ChartPoint(5, 5, "front_front.png", (1, 2, 3))]          # 段階に当てはまらない色はどの組合せにも入れない
    out = dpc.combination_charts([("すべてのカメラ", pts)])
    assert [(l, [p.x for p in ps]) for l, ps in out] == [
        ("すべてのカメラ・1 位・正面 × 正面", [1, 2]), ("すべてのカメラ・1 位・正面 × 左横", [3]),
        ("すべてのカメラ・2〜10 位・正面 × 正面", [4])] or len(out) == 3
    assert sum(len(ps) for _, ps in out) == 4
    only = dpc.combination_charts([("c", pts)], colors={0}, icons={"front_front.png"})
    assert [[p.x for p in ps] for _, ps in only] == [[1, 2]]
    assert len(dpc.COLOR_GROUPS) * len(dpc.icon_choices()) == 105


def test_add_chart_sheet_combination_outputs_one_chart_per_combination(tmp_path, monkeypatch):
    """合成データの点を 2 色 × 2 アイコンに分け、組合せごとに 1 チャート（PNG）が出る。OR 条件は従来どおり 1 チャート。"""
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])

    def fake_targets(data, all_cameras, per_camera, unit_m=0.0, cameras=None):
        pts = [ChartPoint(1, 1, "front_front.png", _rgb_of(0)), ChartPoint(2, 2, "front_left.png", _rgb_of(0)),
               ChartPoint(3, 3, "front_front.png", _rgb_of(30)), ChartPoint(4, 4, "front_left.png", _rgb_of(30))]
        return [("すべてのカメラ", pts)]

    monkeypatch.setattr(dpc, "chart_targets", fake_targets)
    base = add_chart_sheet(out, output_excel=False, output_png=True)
    assert len(base.charts) == 1
    comb = add_chart_sheet(out, combination=True, output_excel=False, output_png=True)
    assert len(comb.charts) == 4 and len(comb.png_files) == 4 and all(p.is_file() for p in comb.png_files)
    assert len(set(comb.png_files)) == 4                                           # ファイル名が重ならない
    assert sum(n for _, n in comb.charts) == sum(n for _, n in base.charts)        # 点は分かれるだけで増減しない


def test_png_output_dir_is_created_and_used(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    dest = tmp_path / "pngs" / "sub"
    data = add_chart_sheet(out, output_excel=False, output_png=True, png_dir=dest)
    assert data.png_files and all(p.parent == dest and p.is_file() for p in data.png_files)
    assert dpc.chart_png_path(out, "", None).parent == out.parent and dpc.chart_png_path(out, "", "").parent == out.parent


def test_clip_bytes_for_completion_dialog():
    from compare_face.ui.distance_pose_chart_screen import clip_bytes
    assert clip_bytes("abc") == "abc"
    assert clip_bytes("a" * 600) == "a" * 500
    s = clip_bytes("あ" * 200)                      # 1 文字 3 バイト → 166 文字（498 バイト）。文字の途中では切らない
    assert s == "あ" * 166 and len(s.encode("utf-8")) <= 500


def test_combination_pngs_share_axes_so_they_can_be_overlaid(tmp_path, monkeypatch):
    """組合せごとの PNG は全点の範囲で軸を決めるので、点の範囲が違う組合せでも軸の目盛・グリッドが同じ位置になる。"""
    from PIL import Image
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])

    def fake_targets(data, all_cameras, per_camera, unit_m=0.0, cameras=None):
        pts = [ChartPoint(1, 1, "front_front.png", _rgb_of(0)), ChartPoint(30, 30, "front_left.png", _rgb_of(30))]
        return [("すべてのカメラ", pts)]

    monkeypatch.setattr(dpc, "chart_targets", fake_targets)
    data = add_chart_sheet(out, combination=True, output_excel=False, output_png=True, transparent=True)
    assert len(data.png_files) == 2
    imgs = [Image.open(p) for p in data.png_files]
    W, H = imgs[0].size
    px, py, pw, ph = dpc.PLOT
    box = (int(W * px) - 80, int(H * (py + ph)) + 5, int(W * (px + pw)) + 5, int(H * (py + ph)) + 70)     # 横軸の目盛ラベルの帯
    assert imgs[0].crop(box).tobytes() == imgs[1].crop(box).tobytes()
    assert any(imgs[0].crop(box).getchannel("A").getdata())                                              # 帯に目盛ラベルがある
