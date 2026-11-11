"""距離範囲マッピング（core/report/distance_range_map.py）。GPU・モデル重みなしで通る。"""
import numpy as np
import pytest

from compare_face.core.camera.distance import CameraDistance
from compare_face.core.common.imageio import imread, imwrite
from compare_face.core.common.settings import load_face_size_map_settings, save_face_size_map_settings
from compare_face.core.report.distance_range_map import run_distance_range_map
from test_camera_distance import SIZE, _camera_dir, _expected


def _filled_ratio(path):
    img = imread(path)
    gray = np.full(img.shape, 128, np.uint8)
    return float(np.mean(np.any(img != gray, axis=-1)))


def _make(tmp_path):
    cams = _camera_dir(tmp_path)
    gray = np.full((SIZE[1], SIZE[0], 3), 128, np.uint8)
    assert imwrite(cams / "Cam1.png", gray)
    return cams


def test_distance_map_matches_distance_m(tmp_path):
    cams = _make(tmp_path)
    cd = CameraDistance(cams)
    _, dmap = cd.corrected_distance_map("Cam1")
    assert dmap.shape == (SIZE[1], SIZE[0])
    for x, y in ((100, 400), (320, 300), (500, 450)):
        assert dmap[y, x] == pytest.approx(_expected((x, y)))


def test_filled_area_grows_with_distance(tmp_path):
    cams = _make(tmp_path)
    ratios = []
    for d in (3.0, 8.0, 20.0):
        outs, notes = run_distance_range_map(cams, cams, tmp_path / f"out{d}", d)
        assert len(outs) == 1 and outs[0].name == "Cam1_distance.png"
        ratios.append(_filled_ratio(outs[0]))
    assert ratios[0] < ratios[1] < ratios[2]


def test_near_pixel_filled_far_pixel_not(tmp_path):
    cams = _make(tmp_path)
    cd = CameraDistance(cams)
    _, dmap = cd.corrected_distance_map("Cam1")
    ys, xs = np.where(np.isfinite(dmap))
    near, far = np.argmin(dmap[ys, xs]), np.argmax(dmap[ys, xs])
    limit = float((dmap[ys[near], xs[near]] + dmap[ys[far], xs[far]]) / 2)
    outs, _ = run_distance_range_map(cams, cams, tmp_path / "out", limit)
    img = imread(outs[0])
    assert not np.array_equal(img[ys[near], xs[near]], [128, 128, 128])   # 塗られている
    assert np.array_equal(img[ys[far], xs[far]], [128, 128, 128])         # 範囲外はそのまま


def test_missing_config_is_skipped(tmp_path):
    cams = _make(tmp_path)
    assert imwrite(cams / "Cam2.png", np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    outs, notes = run_distance_range_map(cams, cams, tmp_path / "out", 5.0)
    assert [p.name for p in outs] == ["Cam1_distance.png"]
    assert any("Cam2" in n for n in notes)


def test_rejects_non_positive_distance(tmp_path):
    cams = _make(tmp_path)
    with pytest.raises(ValueError):
        run_distance_range_map(cams, cams, tmp_path / "out", 0.0)


def test_settings_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    s = load_face_size_map_settings(p)
    assert s["mode"] == "face_size"
    s.update(mode="distance_range", config_dir="C:/cfg", max_distance_m=12.5)
    save_face_size_map_settings(s, p)
    got = load_face_size_map_settings(p)
    assert (got["mode"], got["config_dir"], got["max_distance_m"]) == ("distance_range", "C:/cfg", 12.5)
