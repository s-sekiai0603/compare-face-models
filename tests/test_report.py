import json
from datetime import datetime

import cv2
import numpy as np
import openpyxl

from verify_reid.core.features import EXTRACTORS
from verify_reid.core.features.osnet import preprocess
from verify_reid.core.gallery import format_ts, load_entry, scan_gallery, scan_track_folder
from verify_reid.core.phase_log import PhaseLogger
from verify_reid.core.report_excel import MODE_IMAGE, ReportConditions, build_report
from verify_reid.core.similarity import aggregate_by_track, competition_ranks, rank_pairs, similarity_matrix


def _make_track(root, run, cam, video, tid, stamps, location="本社"):
    d = root / run / cam / video / str(tid)
    d.mkdir(parents=True)
    images = []
    for ts in stamps:
        for kind in ("body", "body_aligned"):
            cv2.imwrite(str(d / f"{ts}_{kind}.jpg"), np.full((80, 40, 3), 100, np.uint8))
        images.append({"file_body": f"{ts}_body.jpg", "person_conf": 0.9, "bbox_xyxy": [0, 0, 40, 80],
                       "keypoints": [{"x": 10, "y": 10, "conf": 0.9}] * 17})
    (d / f"{tid}.json").write_text(json.dumps(
        {"track_id": tid, "video_file": f"{video}.mp4", "location": location, "images": images},
        ensure_ascii=False), encoding="utf-8")
    return d


def test_registry_has_osnet():
    assert "OSNet" in EXTRACTORS


def test_preprocess_shape():
    assert preprocess(np.zeros((30, 10, 3), np.uint8)).shape == (3, 256, 128)


def test_format_ts():
    assert format_ts("20260311_122723_500") == "2026/03/11 12:27:23.500"
    assert format_ts("frame00000007_000") == "frame00000007_000"


def test_scan_gallery_reads_json_and_path(tmp_path):
    _make_track(tmp_path, "20261008120000", "Cam1", "20260311122723", 3,
                ["20260311_122723_000", "20260311_122723_500"])
    entries = scan_gallery(tmp_path)
    assert len(entries) == 2
    e = entries[0]
    assert (e.camera_id, e.video_name, e.track_id) == ("Cam1", "20260311122723", "3")
    assert (e.location, e.video_file, e.person_conf) == ("本社", "20260311122723.mp4", 0.9)
    assert e.embed_path.name.endswith("_body_aligned.jpg")


def test_scan_excludes_query(tmp_path):
    d = _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000"])
    _make_track(tmp_path, "r", "C", "v", 2, ["20260311_122724_000"])
    assert len(scan_gallery(tmp_path, exclude_dirs={d})) == 1
    q = load_entry(next(d.glob("*_body.jpg")))
    assert len(scan_gallery(tmp_path, exclude_files={q.body_path})) == 1
    assert len(scan_track_folder(d)) == 1


def test_missing_json_falls_back(tmp_path):
    d = _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000"])
    (d / "1.json").unlink()
    e = scan_gallery(tmp_path)[0]
    assert (e.location, e.person_conf, e.video_file) == ("-", None, "v.mp4")


def test_similarity_and_aggregation():
    q = np.array([[1, 0]], np.float32)
    g = np.array([[1, 0], [0.6, 0.8], [0, 1], [0.8, 0.6]], np.float32)
    sim = similarity_matrix(q, g)
    pairs = rank_pairs(sim)
    assert [p.g_idx for p in pairs] == [0, 3, 1, 2]
    agg = aggregate_by_track(sim, ["a", "a", "b", "b"])
    assert [(p.g_idx, round(p.sim, 2)) for p in agg] == [(0, 1.0), (3, 0.8)]


def test_competition_ranks():
    assert competition_ranks([0.9, 0.9, 0.8, 0.70001, 0.70004]) == [1, 1, 3, 4, 4]


def test_phase_logger_records():
    log = PhaseLogger(echo=lambda m: None)
    with log.phase("a") as r:
        r.note = "n"
    try:
        with log.phase("b"):
            raise ValueError("x")
    except ValueError:
        pass
    assert [r.name for r in log.records] == ["a", "b"]
    assert log.records[0].end >= log.records[0].start
    assert "失敗" in log.records[1].note and log.records[1].end


def test_build_report_sheets(tmp_path):
    _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000"])
    _make_track(tmp_path, "r", "C", "v", 2, ["20260311_122724_000", "20260311_122725_000"])
    gallery = scan_gallery(tmp_path)
    queries = gallery[:1]
    sim = np.array([[1.0, 0.5, 0.4]], np.float32)
    log = PhaseLogger(echo=lambda m: None)
    cond = ReportConditions("OSNet", MODE_IMAGE, "q", str(tmp_path), str(tmp_path), True, datetime.now())
    out = tmp_path / "rep" / "r.xlsx"
    build_report(out, cond, queries, gallery, rank_pairs(sim),
                 aggregate_by_track(sim, [g.track_key for g in gallery]), log)
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["条件", "外見比較", "外見比較(追跡ID集約)", "ログ"]
    text = wb["外見比較"]["A4"].value
    assert "Loc:本社" in text and "信頼度:0.900" in text and "映像:v.mp4" in text
    names = [r[1] for r in wb["ログ"].iter_rows(min_row=3, values_only=True)]
    assert any("Excel保存" in n for n in names)
