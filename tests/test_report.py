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


def test_limit_applies_and_setting_default():
    from verify_reid.core.settings import load_settings
    sim = np.arange(10, dtype=np.float32).reshape(1, 10) / 10
    assert len(rank_pairs(sim, 3)) == 3
    assert len(aggregate_by_track(sim, list(range(10)), 4)) == 4
    assert load_settings(path=__import__("pathlib").Path("none.json"))["report_limit"] == 500


def test_is_target_and_pink(tmp_path):
    from verify_reid.core.report_excel import TARGET_FILL, is_target
    _make_track(tmp_path, "r", "Cam1", "v", 1, ["20260311_122723_000"])
    _make_track(tmp_path, "r", "Cam2", "v", 2, ["20260311_122724_000"])
    g = scan_gallery(tmp_path)
    assert is_target(g[0], [("Cam1", "v.mp4", "1")])
    assert is_target(g[0], [("Cam1", "v", "")])            # 拡張子なし・空欄は不問
    assert not is_target(g[0], [("Cam2", "v.mp4", "1")])
    assert not is_target(g[0], [("", "", "")]) and not is_target(g[0], [])
    sim = np.array([[1.0, 0.5]], np.float32)
    cond = ReportConditions("OSNet", MODE_IMAGE, "q", "g", "o", False, datetime.now(), 500, [("Cam2", "", "2")])
    out = tmp_path / "t.xlsx"
    build_report(out, cond, g[:1], g, rank_pairs(sim), aggregate_by_track(sim, [x.track_key for x in g]),
                 PhaseLogger(echo=lambda m: None))
    ws = openpyxl.load_workbook(out)["外見比較"]
    # 行1-2: クエリ。行4(情報)/5(画像)に順位1(Cam1)・列B=順位2(Cam2)
    assert ws["A4"].fill.fgColor.rgb != TARGET_FILL.fgColor.rgb
    assert ws["B4"].fill.fgColor.rgb == TARGET_FILL.fgColor.rgb
    assert ws["B5"].fill.fgColor.rgb == TARGET_FILL.fgColor.rgb
    cond_rows = {r[0]: r[1] for r in openpyxl.load_workbook(out)["条件"].iter_rows(values_only=True)}
    assert "Cam2" in cond_rows["探したい人物(ピンク表示)"]


def test_registry_has_osnet_ain():
    assert {"OSNet", "OSNet-AIN"} <= set(EXTRACTORS)
    ain = EXTRACTORS["OSNet-AIN"]()
    assert ain.arch == "osnet_ain_x1_0" and ain.weights_path().name == "osnet_ain_x1_0.pth"


def test_assign_groups_fixed_windows_and_skip_empty():
    from verify_reid.core.merge import assign_groups
    # 先頭(0.0)起点で 2 秒ごと: [0,2) [2,4) [4,6) [6,8)。区間[2,4)は空なので飛ばして連番
    assert assign_groups([0.0, 0.5, 1.9, 4.0, 5.9, 6.0], 2.0) == [1, 1, 1, 2, 2, 3]
    assert assign_groups([], 1.0) == []


def test_merge_embeddings_mean_and_normalized():
    from verify_reid.core.merge import merge_embeddings
    emb = np.array([[1, 0], [0, 1], [0, 1]], np.float32)
    m = merge_embeddings(emb, [1, 1, 2])
    assert m.shape == (2, 2)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    assert np.allclose(m[0], [2 ** -0.5, 2 ** -0.5]) and np.allclose(m[1], [0, 1])


def test_entry_seconds(tmp_path):
    from verify_reid.core.gallery import entry_seconds
    _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000", "20260311_122724_500"])
    e = scan_gallery(tmp_path)
    assert entry_seconds(e[1]) - entry_seconds(e[0]) == 1.5


def test_report_shows_group_ids(tmp_path):
    _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000", "20260311_122730_000"])
    _make_track(tmp_path, "r", "C", "v", 2, ["20260311_122800_000"])
    allx = scan_gallery(tmp_path)
    queries, gallery = allx[:2], allx[2:]
    sim = np.array([[0.9], [0.1]], np.float32)  # 本来はグループ数分の行。ここでは2グループ
    cond = ReportConditions("OSNet", MODE_IMAGE, "q", "g", "o", False, datetime.now(), 500, None, 5.0)
    out = tmp_path / "g.xlsx"
    build_report(out, cond, queries, gallery, rank_pairs(sim), aggregate_by_track(sim, ["k"]),
                 PhaseLogger(echo=lambda m: None), group_ids=[1, 2])
    ws = openpyxl.load_workbook(out)["外見比較"]
    assert "クエリグループID:G1" in ws["A1"].value and "クエリグループID:G2" in ws["B1"].value
    assert "ヒットしたクエリグループID:G1" in ws["A4"].value  # 最も類似したのは G1
    rows = {r[0]: r[1] for r in openpyxl.load_workbook(out)["条件"].iter_rows(values_only=True)}
    assert "5" in rows["特徴量マージ単位(秒)"] and "2" in rows["特徴量マージ単位(秒)"]


def test_foreign_skeleton_reason():
    from verify_reid.core.validity import foreign_skeleton_reason
    def person(x0):  # 17点を x0..x0+16, y=10 に並べる
        k = np.zeros((17, 3), np.float32)
        k[:, 0] = np.arange(17) + x0; k[:, 1] = 10; k[:, 2] = 0.9
        return k
    own = np.array([0, 0, 20, 50], np.float32)
    boxes = np.array([own, [100, 0, 120, 50]], np.float32)
    kp = np.stack([person(0), person(100)])
    assert foreign_skeleton_reason(own, boxes, kp, (0, 0, 20, 50)) is None        # 自分自身のみ
    kp2 = np.stack([person(0), person(10)])                                         # 他人物の点が範囲内に入る
    r = foreign_skeleton_reason(own, np.array([own, [10, 0, 30, 50]], np.float32), kp2, (0, 0, 20, 50))
    assert r and "別人の骨格" in r
    assert foreign_skeleton_reason(own, None, None, (0, 0, 20, 50)) is None


def _write_invalid(d, ts, reason="別人の骨格を含む（テスト）"):
    import json
    jp = next(d.glob("*.json")); data = json.loads(jp.read_text(encoding="utf-8"))
    for im in data["images"]:
        if im["file_body"].startswith(ts):
            im["invalid"] = True; im["invalid_reason"] = reason
    jp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_gallery_reads_invalid_flag(tmp_path):
    d = _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000", "20260311_122724_000"])
    _write_invalid(d, "20260311_122724_000")
    e = scan_gallery(tmp_path)
    assert [x.invalid for x in e] == [False, True] and e[1].invalid_reason.startswith("別人")


def test_invalid_query_gray_and_excluded(tmp_path):
    from verify_reid.core.report_excel import INVALID_FILL
    d = _make_track(tmp_path, "r", "C", "v", 1, ["20260311_122723_000", "20260311_122724_000"])
    _write_invalid(d, "20260311_122723_000")
    _make_track(tmp_path, "r", "C", "v", 2, ["20260311_122800_000"])
    allx = scan_gallery(tmp_path)
    queries, gallery = allx[:2], allx[2:]
    used = [q for q in queries if not q.invalid]
    sim = np.array([[0.9]], np.float32)
    cond = ReportConditions("OSNet", MODE_IMAGE, "q", "g", "o", False, datetime.now(), 500, None, 5.0)
    out = tmp_path / "i.xlsx"
    build_report(out, cond, queries, gallery, rank_pairs(sim), aggregate_by_track(sim, ["k"]),
                 PhaseLogger(echo=lambda m: None), group_ids=[None, 1], used_queries=used)
    wb = openpyxl.load_workbook(out); ws = wb["外見比較"]
    assert "【無効画像】" in ws["A1"].value and "別人の骨格を含む（テスト）" in ws["A1"].value
    assert "グループID" not in ws["A1"].value and "クエリグループID:G1" in ws["B1"].value
    assert ws["A1"].fill.fgColor.rgb == INVALID_FILL.fgColor.rgb
    assert ws["A2"].fill.fgColor.rgb == INVALID_FILL.fgColor.rgb       # 画像セルも
    assert ws["B1"].fill.fgColor.rgb != INVALID_FILL.fgColor.rgb
    rows = {r[0]: r[1] for r in wb["条件"].iter_rows(values_only=True)}
    assert rows["クエリ無効画像数（比較に不使用）"] == 1
