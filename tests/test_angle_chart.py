"""顔角度の組合せ表示（core/report/angle_chart.py）。"""
import math

import numpy as np
import openpyxl
import pytest

from compare_face.core.report import distance_pose_chart as dpc
from compare_face.core.report.angle_chart import (
    AXIS_LIMIT, AngleOverlay, heat_grid, read_angle_data, render_angle_chart, select_angle_points)
from compare_face.core.report.chart_summary import DistanceRange
from compare_face.core.report.distance_pose_chart import ChartPoint, color_group_rgb
from compare_face.core.report.verify_face_size_excel import INFO_SHEET_NAME
from test_distance_pose_chart import _FONT, _expected_dist, _report


def _set_angles(path, angles):
    """対象画像情報の「顔の角度」に値を入れる（入力順。openpyxl で保存し直すのでセル内画像は失われる = データ読取のテスト用）。"""
    wb = openpyxl.load_workbook(path)
    ws = wb[INFO_SHEET_NAME]
    col = next(c.column for c in ws[2] if c.value == "顔の角度")
    for r, a in enumerate(angles, start=3):
        ws.cell(row=r, column=col, value=a)
    wb.save(path)


def _pt(ang_v, ang_h, g=0, x=5.0, y=5.0, cam="Cam1"):
    return ChartPoint(x, y, "", color_group_rgb(g), "", cam, cam, None, ang_v, ang_h)


def test_read_angle_data_uses_angles_distances_and_cell_colors(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_angles(out, [10, -20, 30.5])                  # 対象画像情報の入力順 = Cam1 の 60×80, 30×40, Cam2
    d = read_angle_data(out)
    assert d.n_images == 3 and d.no_angle == 0 and d.no_distance == 1       # Cam2 は距離なし
    assert len(d.points) == 6 and d.cameras == ["Cam1", "Cam2"]           # 対角を除く 3 × 2
    both_cam1 = [p for p in d.points if p.cam_v == "Cam1" and p.cam_h == "Cam1"]
    assert {(p.ang_v, p.ang_h) for p in both_cam1} == {(10, -20), (-20, 10)}
    assert all(p.x == pytest.approx(_expected_dist()) and p.y == pytest.approx(_expected_dist()) for p in both_cam1)
    assert all(math.isnan(p.x) or math.isnan(p.y) for p in d.points if "Cam2" in (p.cam_v, p.cam_h))
    assert all(dpc.color_group(p.rgb) is not None for p in d.points)


def test_read_angle_data_old_report_without_angle_raises(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    wb = openpyxl.load_workbook(out)
    ws = wb["比較結果誤識別数まとめ"]
    for r in range(5, 17):                              # 「顔の角度」の行見出しを消す = 角度を足す前のレポートと同じ状態
        if ws.cell(row=r, column=13).value == "顔の角度":
            ws.cell(row=r, column=13).value = "x"
    wb.save(out)
    with pytest.raises(ValueError, match="顔の角度"):
        read_angle_data(out)


def test_select_points_filters_camera_color_and_distance():
    pts = [_pt(0, 0, 0, 2, 2), _pt(10, 10, 1, 6, 6), _pt(20, 20, 2, 6, 2), _pt(5, 5, 0, float("nan"), 1, cam="Cam2")]
    assert len(select_angle_points(pts)) == 4
    assert len(select_angle_points(pts, camera="Cam2")) == 1
    assert [p.ang_v for p in select_angle_points(pts, colors={0})] == [0, 5]
    # 縦（基準の画像）の距離 y と横（比較対象）の距離 x に別々の範囲
    assert [p.ang_v for p in select_angle_points(pts, ranges=[DistanceRange(0, 3, 0, 3)])] == [0]
    assert [p.ang_v for p in select_angle_points(pts, ranges=[DistanceRange(0, 3, 5, 7)])] == [20]
    # 2 つの範囲は合算。距離の無い点（nan）はフィルタをかけると除かれる
    assert [p.ang_v for p in select_angle_points(pts, ranges=[DistanceRange.both(0, 3), DistanceRange.both(5, 7)])] == [0, 10]


def test_heat_grid_is_weighted_average_of_levels():
    pts = [_pt(0, 0, 0)] * 5 + [_pt(0, 10, 6)] * 5           # 同じ場所の近くに 1 位 5 件と 451 位以降 5 件
    level, dens = heat_grid(pts)
    n = AXIS_LIMIT
    assert 0 < level[n, n + 5] < 6                           # 間は両者の加重平均
    assert level[n, n] < level[n, n + 5] < level[n, n + 10]  # 1 位側ほど小さく、451 位以降の側ほど大きい
    assert dens[n, n + 40] < dens[n, n] * 1e-3               # 離れた所は密度ほぼ 0
    assert dens[n, n] > 0 and heat_grid([])[1].max() == 0


@_FONT
def test_render_angle_chart_draws_points_and_faces(tmp_path):
    from compare_face.core.report.angle_chart import M_LEFT, M_TOP, PLOT, _to_px
    pts = [_pt(90, -90, 0), _pt(-90, 90, 6)]
    im = render_angle_chart(pts, "t", ["a"])
    x0, y0 = M_LEFT, M_TOP
    # 点: 縦（基準）= +90、横（比較対象）= -90 → 軸は +が左/上なので、右端付近・上端付近に 1 位（緑）の点
    cx, cy = int(_to_px(-90, x0)), int(_to_px(90, y0))
    assert im.getpixel((cx, cy))[:3] == color_group_rgb(0)
    cx, cy = int(_to_px(90, x0)), int(_to_px(-90, y0))
    assert im.getpixel((cx, cy))[:3] == color_group_rgb(6)
    # 軸の外側（上 = 横軸側・左 = 縦軸側）に顔画像（薄い黄色）が置かれる
    assert any(px[:3] != (255, 255, 255) for px in (im.getpixel((int(_to_px(0, x0)), y0 - 150)), im.getpixel((x0 - 175, int(_to_px(0, y0))))))
    assert im.width > PLOT and render_angle_chart([], "t").size == im.size      # 点が無くても描ける


@_FONT
def test_angle_overlay_reads_excel_once_and_applies_conditions(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_angles(out, [10, -20, 30.5])
    o = AngleOverlay()
    o.configure(str(out), None, list(range(7)), None)
    assert o.compose() is not None and "6 点をプロット" in o.message
    data = o.data
    o.configure(str(out), "Cam1", list(range(7)), [DistanceRange.both(0, 1000)])
    o.compose()
    assert o.data is data                               # 更新されていない Excel は読み直さない
    assert "2 点をプロット" in o.message
    assert o.image.size[0] > 0
