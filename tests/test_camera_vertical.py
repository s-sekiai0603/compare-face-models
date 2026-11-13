"""建物の垂直線による垂直補正（core/camera/vertical.py）と、垂直補正後の距離計算。GUI・モデル重みなしで通る。"""
import numpy as np
import pytest

from compare_face.core.camera.projective import ProjectionError, compute_projection, load_config, save_config
from compare_face.core.camera.vertical import (
    VerticalRectification, line_tilt_deg, vertical_vanishing_point,
)

W, H = 1920, 1080
SIZE = (W, H)
F = 1100.0
HEIGHT = 3.0
PITCH = np.radians(35.0)


def _project(pt):
    """ワールド（X 右, Y 前, Z 上）の点 → 画像。カメラは (0, 0, HEIGHT) から PITCH だけ下向き。"""
    fwd = np.array([0.0, np.cos(PITCH), -np.sin(PITCH)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(fwd, right)
    d = np.asarray(pt, dtype=float) - np.array([0.0, 0.0, HEIGHT])
    x, y, z = right @ d, down @ d, fwd @ d
    return (F * x / z + W / 2, F * y / z + H / 2)


def _vertical_lines():
    """地面（Z=0）から高さ 4m までの垂直線（建物の角）。"""
    return [(_project((x, y, 0.0)), _project((x, y, 4.0))) for x, y in [(-6, 8), (-2, 12), (3, 7), (7, 15), (10, 9)]]


def test_vanishing_point_matches_nadir():
    vp = vertical_vanishing_point(_vertical_lines(), SIZE)
    expected = _project((0.0, 0.0, -1e9))  # 真下の無限遠 = カメラ直下の消失点
    assert vp[:2] / vp[2] == pytest.approx(expected, abs=0.5)


def test_rectification_makes_lines_vertical_and_roundtrips():
    lines = _vertical_lines()
    rect = VerticalRectification(vertical_vanishing_point(lines, SIZE), F, SIZE, min_scale=0.0)
    assert 0 < rect.scale <= 1.0 and not rect.cropped
    assert rect.focal_px == pytest.approx(F * rect.scale)
    for a, b in lines:
        assert line_tilt_deg(rect.to_rectified(a), rect.to_rectified(b)) < 1e-6
        assert line_tilt_deg(a, b) > 5  # 補正前は傾いている
    pts = np.array([[100.0, 200.0], [960.0, 540.0], [1800.0, 1000.0]])
    assert rect.to_unrectified(rect.to_rectified(pts)) == pytest.approx(pts, abs=1e-6)
    out = rect.rectify_image(np.zeros((H, W, 3), dtype=np.uint8))
    assert out.shape == (H, W, 3)
    # 補正後の画像の内側に全体が収まる
    corners = rect.to_rectified(np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=float))
    assert corners.min() >= -1e-6 and corners[:, 0].max() <= W + 1e-6 and corners[:, 1].max() <= H + 1e-6


def test_scale_floor_crops_edges():
    rect = VerticalRectification(vertical_vanishing_point(_vertical_lines(), SIZE), F, SIZE)
    assert rect.scale == pytest.approx(0.5) and rect.cropped  # このカメラは全体を収めると 0.5 未満になる


def test_distance_after_rectification_matches_truth():
    rect = VerticalRectification(vertical_vanishing_point(_vertical_lines(), SIZE), F, SIZE)
    # 地面上の長方形（4 x 2 m、少し回転させて対辺が平行に写らないようにする）
    cx, cy, yaw = 1.0, 6.0, 0.3
    local = [(-2, -1), (2, -1), (2, 1), (-2, 1)]
    ground = [(cx + x * np.cos(yaw) - y * np.sin(yaw), cy + x * np.sin(yaw) + y * np.cos(yaw)) for x, y in local]
    pts = [tuple(rect.to_rectified(_project((x, y, 0.0)))) for x, y in ground]
    proj = compute_projection(pts, 4.0, 2.0, SIZE, focal_px=rect.focal_px, vertical=rect.vertical)
    assert proj.camera_xy is None                      # カメラ直下は画像の無限遠
    assert proj.height_m == pytest.approx(HEIGHT, abs=0.01)
    for gx, gy in [(2.0, 10.0), (-5.0, 4.0), (0.5, 20.0)]:
        got = proj.distance_m(tuple(rect.to_rectified(_project((gx, gy, 0.0)))))
        assert got == pytest.approx(np.hypot(gx, gy), rel=1e-3)
        assert proj.slant_distance_m(tuple(rect.to_rectified(_project((gx, gy, 0.0))))) == pytest.approx(
            np.hypot(np.hypot(gx, gy), HEIGHT), rel=1e-3)


def test_vertical_requires_points_below_horizon():
    rect = VerticalRectification(vertical_vanishing_point(_vertical_lines(), SIZE), F, SIZE)
    above = [(300.0, 100.0), (700.0, 100.0), (700.0, 20.0), (300.0, 20.0)]  # 画像の上の方 = 地平線より上
    with pytest.raises(ProjectionError):
        compute_projection(above, 4.0, 2.0, SIZE, focal_px=rect.focal_px, vertical=rect.vertical)


def test_errors():
    one = _vertical_lines()[:1]
    with pytest.raises(ProjectionError):
        vertical_vanishing_point(one, SIZE)
    same = [((100.0, 100.0), (200.0, 900.0))] * 2
    with pytest.raises(ProjectionError):
        vertical_vanishing_point(same, SIZE)
    with pytest.raises(ProjectionError):
        vertical_vanishing_point([((5.0, 5.0), (5.0, 5.0)), ((1.0, 1.0), (3.0, 9.0))], SIZE)
    # 消失点が画像の横方向の遠くにある（鉛直が画像の縦方向から大きく外れる）
    sideways = [((0.0, 500.0), (400.0, 520.0)), ((0.0, 600.0), (400.0, 560.0))]
    with pytest.raises(ProjectionError):
        VerticalRectification(vertical_vanishing_point(sideways, SIZE), F, SIZE)
    with pytest.raises(ProjectionError):
        VerticalRectification((960.0, 2000.0, 1.0), 0.0, SIZE)


def test_parallel_vertical_lines_are_allowed():
    """画像上で平行な垂直線（消失点が無限遠）でもよい（カメラが水平で傾きが無い）。"""
    lines = [((300.0, 100.0), (300.0, 900.0)), ((1500.0, 120.0), (1500.0, 800.0))]
    vp = vertical_vanishing_point(lines, SIZE)
    assert abs(vp[2]) < 1e-6 * abs(vp[1])
    rect = VerticalRectification(vp, F, SIZE)
    assert rect.scale == pytest.approx(1.0)
    assert rect.to_rectified([123.0, 456.0]) == pytest.approx([123.0, 456.0], abs=1e-6)


def test_config_roundtrip_with_rectification(tmp_path):
    p = tmp_path / "cam1.json"
    pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    rect = {"lines": [((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 9.0))], "focal_px": 1000.0}
    save_config(p, "cam1.png", pts, 4.0, 2.0, "long", None, rectification=rect)
    cfg = load_config(p)
    assert cfg["camera_xy"] is None
    assert cfg["rectification"] == rect
    save_config(p, "cam1.png", pts, 4.0, 2.0, "long", (1.0, 2.0))
    cfg = load_config(p)
    assert cfg["camera_xy"] == (1.0, 2.0) and cfg["rectification"] is None
