"""指定顔サイズマッピング（core/report/face_size_map.py）。GPU・モデル重みなしで通る。"""
import json

import numpy as np

from compare_face.core.common.imageio import imread, imwrite
from compare_face.core.common.settings import load_face_size_map_settings, save_face_size_map_settings
from compare_face.core.report.face_size_map import collect_foot_points, render_heatmap, run_face_size_map


def _track(root, cam, tid, images, sub=()):
    d = root.joinpath("20260101000000", cam, "video1", tid, *sub)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tid}.json").write_text(json.dumps({"images": images}), encoding="utf-8")


def _im(h, foot, bbox=False):
    d = {"foot_xy": foot}
    d["face_bbox_xyxy" if bbox else "face_height"] = [0, 0, 10, h] if bbox else h
    return d


def test_collect_filters_by_height_and_camera(tmp_path):
    _track(tmp_path, "Cam1", "1", [_im(80, [10, 20]), _im(79.9, [30, 40]), _im(100, [50, 60], bbox=True)])
    _track(tmp_path, "Cam2", "2", [_im(200, [1, 2]), {"face_height": 200}])  # foot_xy 無しは除く
    pts = collect_foot_points(tmp_path, 80)
    assert pts == {"cam1": [(10.0, 20.0), (50.0, 60.0)], "cam2": [(1.0, 2.0)]}


def test_collect_skips_old_and_query_sets(tmp_path):
    _track(tmp_path, "Cam1", "1", [_im(100, [1, 1])], sub=("old",))
    (tmp_path / "検索元" / "set" / "Cam1" / "v" / "1").mkdir(parents=True)
    (tmp_path / "検索元" / "set" / "Cam1" / "v" / "1" / "1.json").write_text(
        json.dumps({"images": [_im(100, [2, 2])]}), encoding="utf-8")
    assert collect_foot_points(tmp_path, 1) == {}


def test_render_heatmap_overlay():
    img = np.full((600, 800, 3), 128, np.uint8)
    assert np.array_equal(render_heatmap(img, []), img)
    out = render_heatmap(img, [(100.0, 60.0)] * 3)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out[60, 100], img[60, 100])   # 点の位置は色が乗る
    assert np.array_equal(out[-1, -1], img[-1, -1])           # 遠い所は元画像のまま
    assert out[60, 100].astype(int).sum() != 128 * 3        # 完全な塗りつぶしではなく元画像が混ざる


def test_run_writes_per_camera(tmp_path):
    gal, cams, out = tmp_path / "g", tmp_path / "cams", tmp_path / "out"
    _track(gal, "Cam1", "1", [_im(100, [50, 50])])
    cams.mkdir()
    imwrite(cams / "cam1.png", np.full((100, 100, 3), 100, np.uint8))
    imwrite(cams / "cam9.png", np.full((100, 100, 3), 100, np.uint8))
    outputs, notes = run_face_size_map(gal, cams, out, 80)
    assert [p.name for p in outputs] == ["cam1_heatmap.png"]
    assert imread(outputs[0]).shape == (100, 100, 3)
    assert len(notes) == 1 and "cam9.png" in notes[0]


def test_settings_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    s = load_face_size_map_settings(p)
    assert s["face_height"] == 80 and s["camera_dir"].endswith("org")
    s["face_height"] = 120
    save_face_size_map_settings(s, p)
    assert load_face_size_map_settings(p)["face_height"] == 120
