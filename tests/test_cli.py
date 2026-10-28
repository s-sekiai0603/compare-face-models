import pytest

from compare_face.cli import build_parser, main, parse_target


def test_list_models_prints_registry(capsys):
    assert main(["list-models"]) == 0
    out = capsys.readouterr().out
    assert "AdaFace" in out and "なし" in out and "InsightFace 2d106det" in out
    assert "顔検出モデル" in out


def test_report_requires_arguments():
    with pytest.raises(SystemExit) as ex:
        main(["report"])
    assert ex.value.code == 2


def test_report_defaults_and_targets():
    args = build_parser().parse_args(["report", "--query", "q", "--gallery", "g", "--out", "o",
                                      "--target", "Cam1|20260311122723.mp4|3", "--target", "Cam2"])
    assert (args.model, args.landmark, args.limit, args.merge_sec, args.no_body) == ("AdaFace", "なし", 500, 0.0, False)
    assert (args.no_cache, args.include_old) == (False, False)
    assert (args.min_face_conf, args.min_norm, args.min_face_width) == (0.0, 0.0, 0.0)
    a2 = build_parser().parse_args(["report", "--query", "q", "--gallery", "g", "--out", "o",
                                    "--min-face-conf", "0.65", "--min-norm", "17", "--min-face-width", "30"])
    assert (a2.min_face_conf, a2.min_norm, a2.min_face_width) == (0.65, 17.0, 30.0)
    # 顔の採用基準の既定は抽出時と同じ（骨格ルールあり / 両目間隔比 0.05 / 切り出し短辺 60px）
    assert (args.no_face_gate, args.min_eye_ratio, args.min_crop_side, args.accept_all) == (False, 0.05, 60, False)
    a3 = build_parser().parse_args(["report", "--query", "q", "--gallery", "g", "--out", "o",
                                    "--no-face-gate", "--min-eye-ratio", "0", "--min-crop-side", "30"])
    assert (a3.no_face_gate, a3.min_eye_ratio, a3.min_crop_side) == (True, 0.0, 30.0)
    assert build_parser().parse_args(["report", "--query", "q", "--gallery", "g", "--out", "o", "--accept-all"]).accept_all
    assert build_parser().parse_args(["revalidate", "--faces", "f"]).faces == "f"
    # 検索元セット検索: 採用基準の引数は report と同じ。--person は複数指定
    q = build_parser().parse_args(["query-set", "--set", "s", "--gallery", "g", "--out", "o", "--person", "Cam1|v.mp4|3",
                                   "--person", "Cam2", "--min-crop-side", "0"])
    assert (q.set, q.model, q.landmark, q.pose, q.include_old, q.no_cache) == ("s", "AdaFace", "なし", "なし", False, False)
    assert (q.no_face_gate, q.min_eye_ratio, q.min_crop_side, q.accept_all) == (False, 0.05, 0.0, False)
    assert [parse_target(t) for t in q.person] == [("Cam1", "v.mp4", "3"), ("Cam2", "", "")]
    assert build_parser().parse_args(["report", "--query", "q", "--gallery", "g", "--out", "o", "--include-old"]).include_old
    assert build_parser().parse_args(["report", "--query", "q", "--gallery", "g", "--out", "o",
                                      "--exclude-dir", "d1", "--exclude-dir", "d2"]).exclude_dir == ["d1", "d2"]
    assert [parse_target(t) for t in args.target] == [("Cam1", "20260311122723.mp4", "3"), ("Cam2", "", "")]


def test_report_with_missing_query_returns_error(tmp_path, capsys):
    rc = main(["report", "--query", str(tmp_path / "none_face.jpg"), "--gallery", str(tmp_path),
               "--out", str(tmp_path), "--model", "存在しないモデル"])
    assert rc == 1
    assert "エラー" in capsys.readouterr().err


def test_revalidate_command(tmp_path, capsys):
    import json
    p = tmp_path / "run" / "Cam1" / "v" / "3" / "3.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"images": [{"file_face": "a_face.jpg", "face_bbox_xyxy": [0, 0, 10, 10], "invalid": False,
                                          "invalid_reason": "", "keypoints": [{"x": 50, "y": 50, "conf": 0.9}]}]}),
                 encoding="utf-8")
    assert main(["revalidate", "--faces", str(tmp_path)]) == 0
    assert "無効の増減 +1 枚" in capsys.readouterr().out
    assert json.loads(p.read_text(encoding="utf-8"))["images"][0]["invalid"] is True


def test_extract_without_videos_returns_error(tmp_path, capsys):
    assert main(["extract", "--video-root", str(tmp_path), "--out", str(tmp_path)]) == 1
    assert "映像がありません" in capsys.readouterr().err
