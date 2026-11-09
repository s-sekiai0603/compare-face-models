from compare_face.core.extract.pipeline import (
    FOOT_MAX_EXTRAPOLATE_SEC, _apply_foot_estimates, estimate_foot_by_motion,
)


def test_estimate_from_past_moves_along_velocity():
    obs = [(0.0, 100.0, 200.0), (0.5, 110.0, 200.0), (1.0, 120.0, 200.0)]  # +20px/秒
    xy, src = estimate_foot_by_motion(obs, 1.5)
    assert src == "estimated_past"
    assert xy == [130.0, 200.0]


def test_estimate_past_extrapolation_is_capped():
    obs = [(0.0, 0.0, 100.0), (1.0, 10.0, 100.0)]  # +10px/秒
    xy, _ = estimate_foot_by_motion(obs, 100.0)
    assert xy == [round(10.0 + 10.0 * FOOT_MAX_EXTRAPOLATE_SEC, 2), 100.0]


def test_estimate_backfills_from_future_when_no_past():
    obs = [(2.0, 120.0, 300.0), (2.5, 130.0, 300.0)]  # +20px/秒
    xy, src = estimate_foot_by_motion(obs, 1.5)
    assert src == "estimated_future"
    assert xy == [110.0, 300.0]


def test_estimate_none_without_observations():
    assert estimate_foot_by_motion([], 1.0) is None


def test_estimate_clamped_to_frame():
    obs = [(0.0, 10.0, 50.0), (1.0, 0.0, 50.0)]
    xy, _ = estimate_foot_by_motion(obs, 2.0, frame_size=(640, 480))
    assert xy[0] == 0.0


def test_apply_replaces_only_bbox_bottom_records():
    records = {1: [
        {"time_sec": 0.0, "foot_xy": [5.0, 5.0], "foot_source": "bbox_bottom", "face_bbox_xyxy": [40.0, 0.0, 60.0, 10.0]},
        {"time_sec": 1.0, "foot_xy": [100.0, 100.0], "foot_source": "ankle", "face_bbox_xyxy": [0.0, 0.0, 1.0, 1.0]},
        {"time_sec": 1.5, "foot_xy": [9.0, 9.0], "foot_source": "bbox_bottom", "face_bbox_xyxy": [70.0, 0.0, 90.0, 10.0]},
    ], 2: [{"time_sec": 0.0, "foot_xy": [7.0, 7.0], "foot_source": "bbox_bottom", "face_bbox_xyxy": [0.0, 0.0, 1.0, 1.0]}]}
    obs = {1: [(0.5, 300.0, 100.0), (1.0, 300.0, 100.0)]}
    _apply_foot_estimates(records, obs, None)
    # X は顔 bbox の左右の中央、Y は足首の遷移からの推測
    assert records[1][0]["foot_source"] == "estimated_future" and records[1][0]["foot_xy"] == [50.0, 100.0]
    assert records[1][1]["foot_source"] == "ankle"
    assert records[1][2]["foot_source"] == "estimated_past" and records[1][2]["foot_xy"] == [80.0, 100.0]
    assert records[2][0]["foot_source"] == "bbox_bottom" and records[2][0]["foot_xy"] == [7.0, 7.0]
