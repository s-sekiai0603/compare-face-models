"""魚眼補正して抽出した結果の扱い: 補正部品・抽出履歴・カメラ設定の検証・後段（距離・足元ヒートマップ）の補正済み判定。"""
import json

import cv2
import numpy as np
import pytest

from compare_face.core.camera.distance import CameraDistance
from compare_face.core.camera.distortion import Distortion, Remapper
from compare_face.core.camera.projective import save_config
from compare_face.core.common.imageio import imread, imwrite
from compare_face.core.extract.history import (
    HISTORY_FILE_NAME, FisheyeConfigError, fisheye_lambda_of_track, load_fisheye_lambdas, write_history)
from compare_face.core.extract.pipeline import _Decoder
from compare_face.core.report.face_size_map import run_face_size_map
from compare_face.core.report.gallery import scan_gallery
from test_camera_distance import POINTS, SIZE, _expected

LAM = -0.1


def _cams(tmp_path, with_distortion=True, cams=("Cam1",)):
    d = tmp_path / "cams"
    d.mkdir(parents=True, exist_ok=True)
    for c in cams:
        save_config(d / f"{c}.json", f"{c}.png", POINTS, 4.0, 3.0, "long", None,
                    distortion={"lambda": LAM, "scale": 1.0, "lines": []} if with_distortion else None)
    return d


def _run_dir(tmp_path, corrected, cam="Cam1", images=None):
    """出力先/日時/カメラID/映像名/追跡ID/<追跡ID>.json と抽出履歴を作る。"""
    run = tmp_path / "out" / "20260101000000"
    tdir = run / cam / "v1" / "7"
    tdir.mkdir(parents=True)
    imwrite(tdir / "20260101_000001_000_face.jpg", np.zeros((20, 20, 3), np.uint8))
    (tdir / "7.json").write_text(json.dumps({"track_id": 7, "images": images or [
        {"file_face": "20260101_000001_000_face.jpg", "foot_xy": [100.0, 200.0], "face_height": 100}]}), encoding="utf-8")
    write_history(run, {"fisheye_correct": corrected, "fisheye_lambdas": {cam: LAM} if corrected else {}})
    return tdir


def test_remapper_matches_correct_image():
    img = np.random.default_rng(0).integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
    d = Distortion(LAM, SIZE)
    r = Remapper(d)
    assert np.array_equal(r.apply(img), d.correct_image(img))
    assert not np.array_equal(r.apply(img), img)
    ident = Remapper(Distortion(0.0, SIZE))
    assert ident.apply(img) is img


def test_decoder_applies_transform(tmp_path):
    path = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    for _ in range(3):
        vw.write(np.full((48, 64, 3), 100, np.uint8))
    vw.release()
    cap = cv2.VideoCapture(str(path))
    dec = _Decoder(cap, lambda f: np.zeros_like(f))
    dec.start()
    items = []
    while (it := dec.q.get(timeout=5)) is not None:
        items.append(it)
    cap.release()
    assert [i for i, _ in items] == [0, 1, 2] and all(f.max() == 0 for _, f in items)


def test_load_fisheye_lambdas(tmp_path):
    d = _cams(tmp_path, cams=("Cam1", "Cam2"))
    assert load_fisheye_lambdas(d, ["cam1", "Cam2"]) == {"cam1": LAM, "Cam2": LAM}   # 大文字小文字は無視
    with pytest.raises(FisheyeConfigError, match="Cam9"):
        load_fisheye_lambdas(d, ["Cam1", "Cam9"])                                   # 設定ファイルなし
    with pytest.raises(FisheyeConfigError, match="魚眼補正の情報"):
        load_fisheye_lambdas(_cams(tmp_path / "nd", with_distortion=False), ["Cam1"])  # distortion なし
    with pytest.raises(FisheyeConfigError, match="フォルダ"):
        load_fisheye_lambdas(tmp_path / "none", ["Cam1"])
    with pytest.raises(FisheyeConfigError, match="フォルダ"):
        load_fisheye_lambdas("", ["Cam1"])


def test_fisheye_lambda_from_history_and_track_json(tmp_path):
    assert fisheye_lambda_of_track(_run_dir(tmp_path / "a", True)) == pytest.approx(LAM)
    assert fisheye_lambda_of_track(_run_dir(tmp_path / "b", False)) is None
    # 履歴が無い（フォルダを移した等）→ 追跡ID json の記録
    t = _run_dir(tmp_path / "c", False)
    (t.parents[2] / HISTORY_FILE_NAME).unlink()
    assert fisheye_lambda_of_track(t) is None   # 古い抽出結果 = 未補正
    j = json.loads((t / "7.json").read_text(encoding="utf-8"))
    j.update(fisheye_corrected=True, fisheye_lambda=LAM)
    (t / "7.json").write_text(json.dumps(j), encoding="utf-8")
    assert fisheye_lambda_of_track(t) == pytest.approx(LAM)


def test_history_file_not_picked_as_track_or_image(tmp_path):
    t = _run_dir(tmp_path, True)
    entries = scan_gallery(tmp_path / "out")
    assert [e.track_id for e in entries] == ["7"] and entries[0].fisheye_lambda == pytest.approx(LAM)


def test_distance_skips_distortion_when_corrected(tmp_path):
    from compare_face.core.common.imageio import imwrite as _w
    cams = _cams(tmp_path)
    _w(cams / "Cam1.png", np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    cd = CameraDistance(cams)
    xy = (100.0, 400.0)
    # 補正済みの座標はそのまま射影変換へ（二重補正しない）。未補正は歪み補正してから
    assert cd.distance_m("Cam1", xy, corrected=True) == pytest.approx(_expected(xy))
    c = Distortion(LAM, SIZE).to_corrected(xy)
    assert cd.distance_m("Cam1", xy) == pytest.approx(_expected((float(c[0]), float(c[1]))))
    # 元画像の点 p を補正して抽出した結果の座標 = to_corrected(p)。どちらの経路でも同じ距離になる
    assert cd.distance_m("Cam1", [float(c[0]), float(c[1])], corrected=True) == pytest.approx(cd.distance_m("Cam1", xy))


def test_face_size_map_corrects_image_for_corrected_tracks(tmp_path):
    t = _run_dir(tmp_path, True)
    cams = tmp_path / "pics"
    cams.mkdir()
    img = np.random.default_rng(1).integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
    imwrite(cams / "Cam1.png", img)
    outputs, notes = run_face_size_map(tmp_path / "out", cams, tmp_path / "res", 80)
    assert len(outputs) == 1 and any("魚眼補正" in n for n in notes)
    # 点の周り以外は補正後の画像と一致する
    got, want = imread(outputs[0]), Distortion(LAM, SIZE).correct_image(img)
    assert np.array_equal(got[-1, -1], want[-1, -1]) and not np.array_equal(got[-1, -1], img[-1, -1])
