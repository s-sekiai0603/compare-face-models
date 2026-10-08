from datetime import datetime

import numpy as np

from verify_reid.core.body_align import align_person_crop, estimate_body_angle
from verify_reid.core.extractor import make_timestamp, should_output
from verify_reid.core.pose_detector import passes_hard_filter
from verify_reid.core.video_scanner import format_duration, parse_capture_datetime


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


def test_make_timestamp():
    assert make_timestamp(datetime(2026, 3, 11, 12, 27, 23), 15, 30.0) == "20260311_122723_500"
    assert make_timestamp(None, 7, 30.0) == "frame00000007_000"


def _kpts(shoulder_x):
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in {5: (shoulder_x, 10), 6: (shoulder_x, 10), 11: (50, 90), 12: (50, 90)}.items():
        k[i] = (x, y, 0.9)
    return k


def test_body_angle_and_align():
    assert abs(estimate_body_angle(_kpts(50))) < 1e-6
    assert estimate_body_angle(_kpts(80)) > 0
    crop = np.zeros((100, 40, 3), np.uint8)
    assert align_person_crop(crop, _kpts(50)).shape == crop.shape  # 補正不要
    assert align_person_crop(crop, _kpts(60)).shape[1] > 40         # キャンバス拡張


def test_hard_filter():
    k = np.full((17, 3), 0.9, np.float32)
    assert passes_hard_filter(k)
    k2 = k.copy(); k2[15:17, 2] = 0
    assert not passes_hard_filter(k2)
