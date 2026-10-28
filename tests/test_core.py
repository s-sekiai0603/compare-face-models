from datetime import datetime
from pathlib import Path

import numpy as np

from compare_face.core.extract.pipeline import (
    OUTPUT_ACCEPTED, OUTPUT_REJECTED, decide_output, make_timestamp, should_output,
)
from compare_face.core.extract.face_crop import (
    MIN_CROP_SIDE, SMALL_FACE_TARGET_LONG_SIDE, adaptive_margin, attach_faces, crop_face, crop_guard_reason,
    crop_min_side, evaluate_face_gate, inflate_box, make_face_crop, prepare_small_face_roi, quality_score,
)
from compare_face.core.extract.face_detector import FACE_MODEL_CANDIDATES, find_face_model
from compare_face.core.extract.pose_detector import passes_hard_filter
from compare_face.core.extract.video_scanner import format_duration, parse_capture_datetime


def test_parse_capture_datetime():
    assert parse_capture_datetime("20260311122723.mp4") == datetime(2026, 3, 11, 12, 27, 23)
    assert parse_capture_datetime("abc.mp4") is None


def test_format_duration():
    assert format_duration(3725) == "01:02:05"
    assert format_duration(None) == "-"


def test_should_output():
    assert should_output(None, 0.0, 0.5)
    assert not should_output(1.0, 1.4, 0.5)
    assert should_output(1.0, 1.5, 0.5)


def test_decide_output_keeps_accepted_cadence():
    # 採用基準を通った顔: 従来どおり採用の間隔だけで判定（基準外の顔を出していても影響しない）
    assert decide_output(None, 0.0, 0.1, 0.5, gate_ok=True) == OUTPUT_ACCEPTED
    assert decide_output(0.0, None, 0.4, 0.5, gate_ok=True) is None
    assert decide_output(0.0, None, 0.5, 0.5, gate_ok=True) == OUTPUT_ACCEPTED
    assert decide_output(0.0, 0.4, 0.5, 0.5, gate_ok=True) == OUTPUT_ACCEPTED
    # 基準外の顔: 採用の間隔内（採用できる顔を出した直後）は出さない。採用できる顔が無い間は基準外の間隔で出す
    assert decide_output(0.0, None, 0.4, 0.5, gate_ok=False) is None
    assert decide_output(None, None, 0.0, 0.5, gate_ok=False) == OUTPUT_REJECTED
    assert decide_output(None, 0.0, 0.4, 0.5, gate_ok=False) is None
    assert decide_output(None, 0.0, 0.5, 0.5, gate_ok=False) == OUTPUT_REJECTED


def test_make_timestamp():
    assert make_timestamp(datetime(2026, 3, 11, 12, 27, 23), 15, 30.0) == "20260311_122723_500"
    assert make_timestamp(None, 7, 30.0) == "frame00000007_000"


def test_hard_filter_function_kept():
    k = np.full((17, 3), 0.9, np.float32)
    assert passes_hard_filter(k)
    k2 = k.copy(); k2[15:17, 2] = 0
    assert not passes_hard_filter(k2)


# ---- 顔ゲート ----
FACE = (40.0, 5.0, 60.0, 30.0)  # 幅 20px


def _kpts(nose=(50, 20, 0.9), l_eye=(45, 15, 0.9), r_eye=(55, 15, 0.9), l_ear=(40, 18, 0.9), r_ear=(60, 18, 0.9)):
    k = np.zeros((17, 3), np.float32)
    for i, v in enumerate((nose, l_eye, r_eye, l_ear, r_ear)):
        k[i] = v
    return k


def test_face_gate_ok_and_frontal():
    g = evaluate_face_gate(FACE, _kpts())
    assert g.ok and g.face_visible and g.frontal and g.reason == ""
    assert g.face_width == 20 and abs(g.eye_ratio - 0.5) < 1e-6
    assert np.allclose([g.nose_conf, g.l_eye_conf, g.r_eye_conf], 0.9)


def test_face_gate_rejects_without_nose():
    g = evaluate_face_gate(FACE, _kpts(nose=(50, 20, 0.1)))
    assert not g.ok and not g.face_visible and "鼻" in g.reason


def test_face_gate_one_eye_is_enough_but_not_frontal():
    g = evaluate_face_gate(FACE, _kpts(r_eye=(55, 15, 0.0)))
    assert g.ok and g.eye_ratio is None and not g.frontal


def test_face_gate_rejects_face_down():
    g = evaluate_face_gate(FACE, _kpts(l_ear=(40, 10, 0.9)))  # 耳(y=10)が目(y=15)より上
    assert not g.ok and not g.face_visible and "下向き" in g.reason


def test_face_gate_rejects_narrow_eyes():
    g = evaluate_face_gate(FACE, _kpts(l_eye=(50, 15, 0.9), r_eye=(50.5, 15, 0.9)))
    assert not g.ok and "横顔" in g.reason
    assert g.face_visible and g.eye_ratio < 0.05  # 骨格ルールは通っている（レポート時に両目間隔比だけ緩められる）


def test_face_gate_without_keypoints():
    assert not evaluate_face_gate(FACE, None).ok
    assert not evaluate_face_gate(FACE, np.zeros((3, 3), np.float32)).ok
    assert not evaluate_face_gate(FACE, None).face_visible


# ---- マージン・ROI・クロップ ----
def test_adaptive_margin():
    assert adaptive_margin(39) == 0.30 and adaptive_margin(40) == 0.20
    assert adaptive_margin(79) == 0.20 and adaptive_margin(80) == 0.10


def test_inflate_box_expands_and_clips():
    assert inflate_box((10, 10, 20, 20), (100, 100), 0.5).tolist() == [5, 5, 25, 25]
    b = inflate_box((95, 95, 100, 100), (100, 100), 0.5)
    assert b[0] >= 0 and b[1] >= 0 and b[2] <= 100 and b[3] <= 100 and b[2] > b[0] and b[3] > b[1]


def test_small_face_roi_upscales_only_small_faces():
    frame = np.zeros((400, 400, 3), np.uint8)
    big = (100.0, 100.0, 200.0, 220.0)
    f, b = prepare_small_face_roi(frame, big)
    assert f is frame and b.tolist() == list(big)
    small = (100.0, 100.0, 130.0, 140.0)  # 30x40
    f2, b2 = prepare_small_face_roi(frame, small)
    assert max(f2.shape[:2]) == SMALL_FACE_TARGET_LONG_SIDE
    assert 0 <= b2[0] < b2[2] <= f2.shape[1] and 0 <= b2[1] < b2[3] <= f2.shape[0]
    assert (b2[2] - b2[0]) > 30 and (b2[3] - b2[1]) > 40  # 拡大されている


def test_crop_face_shape_and_guard_value():
    frame = np.zeros((300, 300, 3), np.uint8)
    small = (0, 0, MIN_CROP_SIDE - 10, MIN_CROP_SIDE - 10)
    assert crop_face(frame, small).shape == (MIN_CROP_SIDE - 9, MIN_CROP_SIDE - 9, 3)  # 小さくても切り出す（ガードは別判定）
    assert crop_min_side(small) == MIN_CROP_SIDE - 10
    assert "切り出しが小さい" in crop_guard_reason(crop_min_side(small)) and crop_guard_reason(MIN_CROP_SIDE) == ""
    assert crop_face(frame, (10, 10, 80, 90)).shape == (81, 71, 3) and crop_min_side((10, 10, 80, 90)) == 70


def test_make_face_crop_pipeline():
    frame = np.random.randint(0, 255, (300, 300, 3), np.uint8)
    fc = make_face_crop(frame, (100, 100, 180, 190))
    assert fc.image is not None and fc.image.ndim == 3 and fc.inflated.shape == (4,) and fc.guard_ok
    assert fc.inflated[0] < 100 and fc.inflated[2] > 180  # マージンで拡張されている
    # 小顔は ROI 拡大後の座標で短辺ガード(60px)を判定する（premises-security と同じ）。
    # 幅 2px の極端に細い顔は拡大しても 60px に届かずガードに外れるが、画像は返す（使うかはレポート時に決める）
    tiny = make_face_crop(frame, (10, 10, 12, 20))
    assert tiny.image is not None and not tiny.guard_ok and tiny.min_side < MIN_CROP_SIDE


def test_quality_score():
    assert quality_score(0.9, 50) == 0.9 + 5.0
    assert quality_score(0.5, 200) == 10.5


def test_attach_faces_center_in_smallest_box_and_best_score():
    persons = [(0, 0, 100, 200), (50, 0, 120, 200)]  # 2人目の方が小さい
    faces = [(60, 10, 80, 30), (10, 10, 20, 20), (12, 30, 22, 40), (300, 300, 310, 310)]
    scores = [0.9, 0.5, 0.8, 0.99]
    got = attach_faces(persons, faces, scores)
    assert got == [2, 0]  # 人物0: 顔1と顔2のうち高スコアの顔2 / 人物1: 重なり領域の顔0 / 顔3は誰にも付かない
    assert attach_faces([], faces, scores) == []
    assert attach_faces(persons, [], []) == [None, None]


def test_find_face_model_candidate_order(tmp_path):
    assert find_face_model(tmp_path) is None
    (tmp_path / FACE_MODEL_CANDIDATES[-1]).write_bytes(b"x")
    assert find_face_model(tmp_path).name == FACE_MODEL_CANDIDATES[-1]
    (tmp_path / FACE_MODEL_CANDIDATES[0]).write_bytes(b"x")
    assert find_face_model(tmp_path).name == FACE_MODEL_CANDIDATES[0]


# ---- 顔検出モデルのフォールバック ----
def test_prepare_face_detector_falls_back_in_order(tmp_path, monkeypatch):
    from compare_face.core.extract import face_detector as fd
    tried = []

    class FakeDetector:
        def __init__(self, path):
            tried.append(Path(path).name)
            if path.endswith(".engine") and "auto" not in path:
                raise RuntimeError("engine version mismatch")
            self.path = path

    monkeypatch.setattr(fd, "FaceDetector", FakeDetector)
    monkeypatch.setattr(fd, "export_face_engine", lambda pt, dst, log: (dst.write_bytes(b"e"), dst)[1])
    monkeypatch.setattr("compare_face.core.extract.pose_model.tensorrt_available", lambda: True)
    (tmp_path / fd.FACE_ENGINE_CANDIDATES[0]).write_bytes(b"x")
    (tmp_path / fd.FACE_PT_NAME).write_bytes(b"x")
    det = fd.prepare_face_detector(log=lambda m: None, models_dir=tmp_path)
    # 既存エンジンは失敗 → .pt から本ツール用エンジンへ変換して使う（既存エンジンは上書きしない）
    assert tried == [fd.FACE_ENGINE_CANDIDATES[0], fd.FACE_ENGINE_AUTO]
    assert Path(det.path).name == fd.FACE_ENGINE_AUTO
    assert (tmp_path / fd.FACE_ENGINE_CANDIDATES[0]).read_bytes() == b"x"


def test_prepare_face_detector_downloads_pt_without_tensorrt(tmp_path, monkeypatch):
    from compare_face.core.extract import face_detector as fd

    class FakeDetector:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(fd, "FaceDetector", FakeDetector)
    monkeypatch.setattr(fd, "download_face_pt", lambda dst, log, url=None: (dst.write_bytes(b"pt"), dst)[1])
    monkeypatch.setattr("compare_face.core.extract.pose_model.tensorrt_available", lambda: False)
    det = fd.prepare_face_detector(log=lambda m: None, models_dir=tmp_path)
    assert Path(det.path).name == fd.FACE_PT_NAME and (tmp_path / fd.FACE_PT_NAME).is_file()


def test_face_model_candidates_and_message(tmp_path):
    from compare_face.core.extract.face_detector import FACE_ENGINE_CANDIDATES, FACE_PT_NAME, face_model_missing_message
    assert FACE_MODEL_CANDIDATES == FACE_ENGINE_CANDIDATES + (FACE_PT_NAME,)
    assert FACE_PT_NAME in face_model_missing_message(tmp_path) and "https://" in face_model_missing_message(tmp_path)


# ---- 人物検出モデルの再変換 ----
def test_prepare_pose_detector_rebuilds_stale_engine(tmp_path, monkeypatch):
    from compare_face.core.extract import pose_model as ml
    engine = tmp_path / f"{ml.MODEL_NAME}-b{ml.BATCH_SIZE}.engine"
    engine.write_bytes(b"stale")
    (tmp_path / f"{ml.MODEL_NAME}.pt").write_bytes(b"pt")
    calls = []

    class FakeDetector:
        def __init__(self, path):
            calls.append(Path(path).name)
            if len(calls) == 1:
                raise RuntimeError("engine version mismatch")
            self.path = path

    def fake_export(pt, dst, log):
        dst.write_bytes(b"fresh")
        return dst

    monkeypatch.setattr(ml, "PoseDetector", FakeDetector)
    monkeypatch.setattr(ml, "export_pose_engine", fake_export)
    monkeypatch.setattr(ml, "tensorrt_available", lambda: True)
    det = ml.prepare_pose_detector(log=lambda m: None, models_dir=tmp_path)
    assert calls == [engine.name, engine.name]           # 失敗 → 作り直して再読込
    assert engine.read_bytes() == b"fresh" and Path(det.path) == engine


def test_prepare_pose_detector_falls_back_to_pt(tmp_path, monkeypatch):
    from compare_face.core.extract import pose_model as ml
    engine = tmp_path / f"{ml.MODEL_NAME}-b{ml.BATCH_SIZE}.engine"
    engine.write_bytes(b"stale")
    pt = tmp_path / f"{ml.MODEL_NAME}.pt"
    pt.write_bytes(b"pt")

    class FakeDetector:
        def __init__(self, path):
            if path.endswith(".engine"):
                raise RuntimeError("broken")
            self.path = path

    def failing_export(pt_path, dst, log):
        raise RuntimeError("no tensorrt builder")

    monkeypatch.setattr(ml, "PoseDetector", FakeDetector)
    monkeypatch.setattr(ml, "export_pose_engine", failing_export)
    monkeypatch.setattr(ml, "tensorrt_available", lambda: True)
    det = ml.prepare_pose_detector(log=lambda m: None, models_dir=tmp_path)
    assert Path(det.path) == pt and not engine.exists()  # 壊れたエンジンは削除し .pt を使う


def test_imageio_handles_japanese_paths(tmp_path):
    from compare_face.core.common.imageio import imread, imwrite
    d = tmp_path / "出力先" / "カメラ１"
    d.mkdir(parents=True)
    img = np.random.randint(0, 255, (20, 30, 3), np.uint8)
    p = d / "顔_face.png"
    assert imwrite(p, img) and p.is_file()
    back = imread(p)
    assert back is not None and back.shape == img.shape and np.array_equal(back, img)
    assert imread(d / "ない.jpg") is None
