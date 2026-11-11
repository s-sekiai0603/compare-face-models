"""カメラ設定ファイル（<カメラID>.json）から足元座標 → カメラ直下からの距離(m)を出す処理と、顔サイズ別精度検証レポートの「距離」。"""
import re
import zipfile

import numpy as np
import openpyxl
import pytest

from compare_face.core.camera.distance import CameraDistance
from compare_face.core.camera.projective import compute_projection, save_config
from compare_face.core.common.imageio import imwrite
from compare_face.core.report.verify_face_size import run_verify_face_size
from compare_face.core.report.verify_face_size_excel import FIRST_COL, FIRST_ROW, INFO_SHEET_NAME, SHEET_NAME
from test_query_set import _MeanExtractor
from test_verify_face_size import _track

SIZE = (640, 480)


def _ground_rect_points(f=500.0, height=3.0, pitch_deg=30.0):
    """高さ 3m・俯角 30° のピンホールカメラに写る、地面の長方形（幅 4m × 奥行 3m。近い辺から一周）の画像上の 4 点。"""
    th, yaw = np.radians(pitch_deg), np.radians(20.0)   # 長方形を水平に 20° 回す（正面向きだと対辺が平行に写り消失点が出ない）
    pts = []
    for x0, z0 in ((-2, 4), (2, 4), (2, 7), (-2, 7)):
        x, z = x0 * np.cos(yaw) + z0 * np.sin(yaw), -x0 * np.sin(yaw) + z0 * np.cos(yaw)
        cx, cy, cz = x, height * np.cos(th) - z * np.sin(th), height * np.sin(th) + z * np.cos(th)
        pts.append((f * cx / cz + SIZE[0] / 2, f * cy / cz + SIZE[1] / 2))
    return pts


POINTS = _ground_rect_points()


def _camera_dir(tmp_path, with_image=True, cam="Cam1"):
    d = tmp_path / "cams"
    d.mkdir(parents=True, exist_ok=True)
    save_config(d / f"{cam}.json", f"{cam}.png", POINTS, 4.0, 3.0, "long", None)
    if with_image:
        assert imwrite(d / f"{cam}.png", np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    return d


def _expected(xy):
    return compute_projection(POINTS, 4.0, 3.0, SIZE).distance_m(xy)


def test_distance_from_config(tmp_path):
    cd = CameraDistance(_camera_dir(tmp_path))
    assert cd.distance_m("Cam1", [100.4, 200.6]) == pytest.approx(_expected((100.4, 200.6)))
    assert cd.distance_m("cam1", [300, 400]) == pytest.approx(_expected((300, 400)))   # カメラIDの大文字小文字は無視
    assert cd.distance_m("Cam1", None) is None and cd.distance_m("Cam1", []) is None   # 足元なし
    assert cd.errors == {}


def test_distance_with_lens_distortion(tmp_path):
    """4 点は歪み補正後の画像の座標で保存されるので、足元（元画像の座標）は補正後の座標に直してから距離を出す。"""
    from compare_face.core.camera.distortion import Distortion
    d = _camera_dir(tmp_path)
    save_config(d / "Cam1.json", "Cam1.png", POINTS, 4.0, 3.0, "long", None,
                distortion={"lambda": -0.1, "scale": 1.0, "lines": []})
    xy = (100.0, 400.0)
    corrected = Distortion(-0.1, SIZE).to_corrected(xy)
    got = CameraDistance(d).distance_m("Cam1", xy)
    assert got == pytest.approx(_expected((float(corrected[0]), float(corrected[1]))))
    assert got != pytest.approx(_expected(xy))


def test_image_size_from_separate_image_folder(tmp_path):
    """画像サイズはカメラ画像フォルダの <カメラID>.png から取る（設定フォルダに画像が無くてもよい。大文字小文字は無視）。"""
    cfg = _camera_dir(tmp_path, with_image=False)
    pics = tmp_path / "pics"
    pics.mkdir()
    assert imwrite(pics / "CAM1.png", np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    cd = CameraDistance(cfg, pics)
    assert cd.distance_m("Cam1", [100, 200]) == pytest.approx(_expected((100, 200)))
    # 別のサイズの画像なら主点が変わるので距離も変わる = 画像サイズは画像フォルダの画像から取っている
    other = tmp_path / "pics2"
    other.mkdir()
    assert imwrite(other / "Cam1.png", np.zeros((SIZE[1] + 100, SIZE[0] + 200, 3), np.uint8))
    assert CameraDistance(cfg, other).distance_m("Cam1", [100, 200]) != pytest.approx(cd.distance_m("Cam1", [100, 200]))
    # 画像フォルダを指定したら設定フォルダの画像は見ない
    assert CameraDistance(_camera_dir(tmp_path / "z"), tmp_path / "none").distance_m("Cam1", [1, 2]) is None


def test_distance_unavailable(tmp_path):
    assert CameraDistance(_camera_dir(tmp_path)).distance_m("Cam9", [1, 2]) is None     # 設定ファイルなし
    cd = CameraDistance(_camera_dir(tmp_path / "x", with_image=False))
    assert cd.distance_m("Cam1", [1, 2]) is None and "画像" in cd.errors["Cam1"]        # 画像サイズが取れない
    bad = _camera_dir(tmp_path / "y")
    (bad / "Cam1.json").write_text("{", encoding="utf-8")
    assert CameraDistance(bad).distance_m("Cam1", [1, 2]) is None
    assert CameraDistance(tmp_path / "none").distance_m("Cam1", [1, 2]) is None          # フォルダなし


def test_report_has_distance_row_and_column(tmp_path, monkeypatch):
    import compare_face.core.report.verify_face_size as vfs
    monkeypatch.setattr(vfs, "get_extractor", lambda name: _MeanExtractor())
    a = _track(tmp_path / "f", "Cam1", "v1", 1, [("20260311_000001_000", (60, 80)), ("20260311_000002_000", (30, 40))])
    b = _track(tmp_path / "f", "Cam2", "v2", 2, [("20260311_000003_000", (45, 60))])   # Cam2 は設定なし → 「-」
    out, log = run_verify_face_size([a, b], tmp_path / "rep", "AdaFace", "なし", camera_dir=_camera_dir(tmp_path),
                                    echo=lambda m: None)
    wb = openpyxl.load_workbook(out)
    info = wb[INFO_SHEET_NAME]
    exp = round(_expected((100.4, 200.6)), 2)
    # 並び: Cam1 の 2 枚（距離あり）→ Cam2（設定なし）。L 列 = 足元の座標（K 列）の右の「距離」
    assert info["L2"].value == "距離" and info["K2"].value == "足元の座標"
    assert [info.cell(row=r, column=12).value for r in (3, 4, 5)] == [exp, exp, "-"]
    ws = wb[SHEET_NAME]
    assert ws.cell(row=13, column=12).value == "距離" and ws.cell(row=12, column=12).value == "足元の座標"
    assert ws.cell(row=15, column=10).value == "距離"
    lk = "=VLOOKUP({}, 対象画像情報!$B$3:$N$200, 11, FALSE)"
    assert ws.cell(row=13, column=FIRST_COL).value == lk.format("M4")
    assert ws.cell(row=FIRST_ROW, column=10).value == lk.format("A16")
    with zipfile.ZipFile(out) as z:
        assert re.search(r"<f>VLOOKUP\(M4, 対象画像情報!\$B\$3:\$N\$200, 11, FALSE\)</f>", z.read("xl/worksheets/sheet2.xml").decode("utf-8"))


def test_legacy_calibration_focal_is_still_used(tmp_path):
    """廃止した実測補正で保存された設定（calibration.focal_px）も、距離の算出では従来どおり焦点距離として使う。"""
    import json
    d = _camera_dir(tmp_path)
    plain = CameraDistance(d).distance_m("Cam1", [100, 200])
    cfg = json.loads((d / "Cam1.json").read_text(encoding="utf-8"))
    cfg["calibration"] = {"focal_px": 800.0, "ref_xy": [1.0, 2.0], "measured_m": 3.0}
    (d / "Cam1.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert CameraDistance(d).distance_m("Cam1", [100, 200]) != pytest.approx(plain)
