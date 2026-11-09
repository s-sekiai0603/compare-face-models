"""カメラ射影変換設定（core/camera/projective.py）。GUI・モデル重みなしで通る。"""
import numpy as np
import pytest

from compare_face.core.camera.projective import (
    ProjectionError, calibrate_focal, clip_line_to_rect, compute_projection, load_config, resolve_sides, save_config,
)

W, H = 1920, 1080
F = 1100.0
HEIGHT = 3.0
PITCH = np.deg2rad(35)


def _project(xw, yw, zw=0.0, yaw=0.0):
    """世界(X 右, Y 前, Z 上)、カメラは (0,0,HEIGHT) で PITCH だけ下向き。画像座標を返す。"""
    fwd = np.array([np.sin(yaw) * np.cos(PITCH), np.cos(yaw) * np.cos(PITCH), -np.sin(PITCH)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd])
    p = r @ (np.array([xw, yw, zw]) - np.array([0, 0, HEIGHT]))
    return F * p[0] / p[2] + W / 2, F * p[1] / p[2] + H / 2


def _rect(x0, y0, a, b, yaw=0.0):
    return [_project(x, y, yaw=yaw) for x, y in [(x0, y0), (x0 + a, y0), (x0 + a, y0 + b), (x0, y0 + b)]]


def test_camera_xy_and_distance_recovered():
    a, b = 4.0, 2.0
    pts = _rect(-1.0, 3.0, a, b, yaw=0.2)
    proj = compute_projection(pts, a, b, (W, H))
    true_nadir = _project(0.0, 0.0, -HEIGHT, yaw=0.2)
    assert proj.camera_xy == pytest.approx(true_nadir, abs=0.5)
    assert proj.focal_px == pytest.approx(F, rel=0.01)
    assert proj.height_m == pytest.approx(HEIGHT, rel=0.01)
    # 地面上の (2, 8) までの距離 = hypot(2, 8)
    assert proj.distance_m(_project(2.0, 8.0, yaw=0.2)) == pytest.approx(np.hypot(2.0, 8.0), rel=0.01)


def test_slant_distance_includes_height():
    a, b = 4.0, 2.0
    proj = compute_projection(_rect(-1.0, 3.0, a, b, yaw=0.2), a, b, (W, H))
    xy = _project(2.0, 8.0, yaw=0.2)
    assert proj.slant_distance_m(xy) == pytest.approx(np.sqrt(2.0**2 + 8.0**2 + HEIGHT**2), rel=0.01)


def test_calibrate_focal_with_measured_distance():
    a, b = 4.0, 2.0
    pts = _rect(-1.0, 3.0, a, b, yaw=0.2)
    # 4 点にクリックずれがあり、自動推定のカメラ位置が真値からずれている状況
    noisy = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, [(4, -3), (-3, 5), (2, 2), (-5, -4)])]
    ref = _project(2.0, 8.0, yaw=0.2)
    true_d = np.hypot(2.0, 8.0)
    auto = compute_projection(noisy, a, b, (W, H))
    fixed = calibrate_focal(noisy, a, b, (W, H), ref, true_d)
    assert fixed.distance_m(ref) == pytest.approx(true_d, abs=1e-3)
    assert abs(fixed.distance_m(ref) - true_d) <= abs(auto.distance_m(ref) - true_d)
    with pytest.raises(ProjectionError):
        calibrate_focal(noisy, a, b, (W, H), ref, 0.0)
    with pytest.raises(ProjectionError):
        calibrate_focal(noisy, a, b, (W, H), ref, 1e6)


def test_config_roundtrip_with_calibration(tmp_path):
    p = tmp_path / "cam1.json"
    pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    save_config(p, "cam1.png", pts, 4.0, 2.0, "long", (1.0, 2.0), focal_px=900.0, ref_xy=(10.0, 20.0), measured_m=5.5)
    assert load_config(p)["calibration"] == {"focal_px": 900.0, "ref_xy": (10.0, 20.0), "measured_m": 5.5}
    save_config(p, "cam1.png", pts, 4.0, 2.0, "long", (1.0, 2.0))
    assert load_config(p)["calibration"] is None


def test_horizon_from_vanishing_points(monkeypatch):
    a, b = 4.0, 2.0
    # 傾き 35° では地平線は画像の上に外れる。浅い 15° なら画像を横切る
    proj = compute_projection(_rect(-1.0, 3.0, a, b, yaw=0.2), a, b, (W, H))
    assert clip_line_to_rect(*proj.vanishing_points, W, H) is None
    monkeypatch.setitem(globals(), "PITCH", np.deg2rad(15))
    proj = compute_projection(_rect(-1.0, 6.0, a, b, yaw=0.2), a, b, (W, H))
    seg = clip_line_to_rect(*proj.vanishing_points, W, H)
    assert seg is not None
    for x, y in seg:
        assert -1e-6 <= x <= W + 1e-6 and -1e-6 <= y <= H + 1e-6
    # 地面上の遠方点（真の地平線上の点）が、算出した地平線の直線上にある
    far = _project(0.0, 1e6, yaw=0.2)
    (x1, y1), (x2, y2) = seg
    y_on_line = y1 + (far[0] - x1) * (y2 - y1) / (x2 - x1)
    assert far[1] == pytest.approx(y_on_line, abs=1.0)


def test_clip_line_to_rect():
    assert clip_line_to_rect((-10, 50), (-5, 50), 100, 100) == ((0.0, 50.0), (100.0, 50.0))  # 水平線は両端が枠に当たる
    assert clip_line_to_rect((0, 200), (100, 200), 100, 100) is None                          # 枠の外を通る
    assert clip_line_to_rect((5, 5), (5, 5), 100, 100) is None                                # 同一点
    (x1, y1), (x2, y2) = clip_line_to_rect((0, 0), (1, 1), 100, 50)
    assert (x1, y1, x2, y2) == (0.0, 0.0, 50.0, 50.0)


def test_config_roundtrip_with_distortion(tmp_path):
    p = tmp_path / "cam1.json"
    pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    dist = {"lambda": -0.2, "scale": 0.8, "lines": [[(1.0, 2.0), (3.0, 4.0), (5.0, 7.0)]]}
    save_config(p, "cam1.png", pts, 4.0, 2.0, "long", (1.0, 2.0), distortion=dist)
    assert load_config(p)["distortion"] == dist
    save_config(p, "cam1.png", pts, 4.0, 2.0, "long", (1.0, 2.0))
    assert load_config(p)["distortion"] is None


def test_rotated_camera():
    a, b = 2.0, 5.0
    pts = [_project(x, y, yaw=0.3) for x, y in [(-1, 4), (1, 4), (1, 9), (-1, 9)]]
    proj = compute_projection(pts, a, b, (W, H))
    # 画像→地面の座標系は 4 点の長方形基準だが、カメラ直下からの距離は座標系に依存しない
    assert proj.distance_m(_project(1.0, 4.0, yaw=0.3)) == pytest.approx(np.hypot(1.0, 4.0), rel=0.01)


def test_invalid_inputs():
    pts = _rect(-1.0, 3.0, 4.0, 2.0)
    rot = _rect(-1.0, 3.0, 4.0, 2.0, yaw=0.2)
    with pytest.raises(ProjectionError):
        compute_projection([rot[0], rot[2], rot[1], rot[3]], 4.0, 2.0, (W, H))  # 蝶ネクタイ
    with pytest.raises(ProjectionError):  # カメラ正対で長方形が軸に揃うと横の辺が平行（焦点距離が決まらない）
        compute_projection(pts, 4.0, 2.0, (W, H))
    with pytest.raises(ProjectionError):
        compute_projection(rot, 0.0, 2.0, (W, H))
    # 正対して撮った長方形（対辺が平行）
    flat = [(100, 100), (500, 100), (500, 300), (100, 300)]
    with pytest.raises(ProjectionError):
        compute_projection(flat, 4.0, 2.0, (W, H))


def test_resolve_sides_and_config_roundtrip(tmp_path):
    assert resolve_sides(4.0, 2.0, "long") == (4.0, 2.0)
    assert resolve_sides(4.0, 2.0, "short") == (2.0, 4.0)
    p = tmp_path / "out" / "cam1.json"
    pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    save_config(p, "cam1.png", pts, 4.0, 2.0, "short", (-10.5, 1500.0))
    d = load_config(p)
    assert d["points"] == pts and d["first_side"] == "short" and d["camera_xy"] == (-10.5, 1500.0)
    assert load_config(tmp_path / "none.json") is None
