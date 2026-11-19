"""距離×距離のヒートマップ表示（core/report/distance_heatmap.py）。"""
import numpy as np
import pytest

from compare_face.core.report.distance_heatmap import (
    DistanceHeatmapOverlay, heat_layers, heat_palette, plot_box, render_heat_layer)
from compare_face.core.report.distance_pose_chart import COLOR_GROUPS, color_group_rgb
from test_distance_pose_chart import _FONT, _report, _set_directions


def test_heat_palette_interpolates_between_group_colors():
    top = len(COLOR_GROUPS) - 1
    rgb = heat_palette(np.array([0.0, 0.5, float(top), 99.0], np.float32))
    assert tuple(rgb[0]) == pytest.approx(color_group_rgb(0))
    assert tuple(rgb[2]) == pytest.approx(color_group_rgb(top)) and tuple(rgb[3]) == pytest.approx(color_group_rgb(top))
    mid = (np.array(color_group_rgb(0)) + np.array(color_group_rgb(1))) / 2
    assert tuple(rgb[1]) == pytest.approx(tuple(mid))


def test_heat_layers_places_points_by_axis_and_ignores_out_of_range():
    box = plot_box()
    axes = (0.0, 10.0, 0.0, 10.0)
    dens, wtd = heat_layers([(0.0001, 9.9999, 2), (5.0, 5.0, 4), (50.0, 5.0, 6)], axes, box)      # 3 点目は軸の外
    assert dens.sum() == 3 - 1 and wtd.sum() == 6
    assert dens[0, 0] == 1 and wtd[0, 0] == 2                                                # 縦軸は上が大きい: (0, 10) は左上
    assert heat_layers([], axes, box)[0].sum() == 0


def test_render_heat_layer_color_follows_level_and_empty_is_none():
    box = plot_box()
    axes = (0.0, 10.0, 0.0, 10.0)
    levels = [(2.0, 5.0, 0)] * 20 + [(8.0, 5.0, 6)] * 20
    layer = render_heat_layer(*heat_layers(levels, axes, box), (box[2] - box[0], box[3] - box[1]))
    h, w = layer.shape[:2]
    left, right = layer[h // 2, int(w * 0.2)], layer[h // 2, int(w * 0.8)]
    assert left[3] > 0 and right[3] > 0 and left[0] < right[0]                               # 1 位（緑）側 → 451 位以降（赤）側は R が大きい
    assert layer[0, 0, 3] == 0
    assert render_heat_layer(*heat_layers([], axes, box), (10, 10)) is None


@_FONT
def test_overlay_builds_from_excel_and_applies_conditions(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    o = DistanceHeatmapOverlay()
    from compare_face.core.report.distance_pose_chart import icon_choices
    icons = [f for _, f in icon_choices()]
    o.configure(str(out), None, range(len(COLOR_GROUPS)), icons, False, 0.0, False, True)
    im = o.compose()
    assert im is not None and im.mode == "RGBA" and im.size[0] > 0 and "ヒートマップ" in o.message and "アイコン" not in o.message
    data = o.data
    n_all = int(o.message.split(" 点")[0])
    assert n_all == 2                                           # 距離のある Cam1 の 2 枚の組合せ（対角を除く 2 点）
    # 更新されていない Excel は読み直さない / 色を全部外すと点 0（枠だけ）
    o.configure(str(out), None, [], icons, False, 0.0, False, True)
    assert o.compose() is not None and o.data is data and o.message.startswith("0 点")
    # 左右振り分け・集計単位・カメラ個別でも描ける
    o.configure(str(out), "Cam1", range(len(COLOR_GROUPS)), icons, True, 1.0, False, True)
    assert o.compose() is not None and "左右振り分け" in o.message and "距離 1 m 単位" in o.message
    assert n_all >= 1


def test_overlay_without_plottable_points_raises(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch)
    import openpyxl
    wb = openpyxl.load_workbook(out)
    ws = wb["比較結果誤識別数まとめ"]
    for row in ws.iter_rows(min_row=5):                        # 距離・向きを消す = どの点も描けない
        for c in row:
            if c.value in ("距離", "顔の向き"):
                c.value = "x"
    wb.save(out)
    o = DistanceHeatmapOverlay()
    o.configure(str(out), None, [0], [])
    with pytest.raises(ValueError):
        o.compose()


@_FONT
def test_icons_and_heatmap_can_be_shown_separately_or_together(tmp_path, monkeypatch):
    from compare_face.core.report.distance_pose_chart import icon_choices
    out = _report(tmp_path, monkeypatch)
    _set_directions(out, ["正面", "左横", "右斜め正面"])
    icons = [f for _, f in icon_choices()]
    cols = range(len(COLOR_GROUPS))
    imgs = {}
    for name, (ic, ht) in {"icon": (True, False), "heat": (False, True), "both": (True, True), "none": (False, False)}.items():
        o = DistanceHeatmapOverlay()
        o.configure(str(out), None, cols, icons, False, 0.0, ic, ht)
        im = o.compose()
        assert im.mode == "RGBA" and im.size[0] > 0
        imgs[name] = im.tobytes()
        if ic and ht:
            assert "アイコン + ヒートマップ" in o.message
    assert len(set(imgs.values())) == 4                          # 4 通りとも別の絵
    o.configure(str(out), None, [], icons, False, 0.0, True, False)    # 色を全部外す → 点 0（枠だけ）
    assert o.compose() is not None and o.message.startswith("0 点")
