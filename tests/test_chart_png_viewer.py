"""チャート確認画面の PNG 出力まわり（core/report/chart_png_viewer.py）。GPU・モデル重みなしで通る。"""
from datetime import datetime

from PIL import Image

from compare_face.core.report import chart_png_viewer as cv


def test_overlay_png_path_names(tmp_path):
    now = datetime(2026, 11, 12, 9, 8, 7)
    assert cv.overlay_png_path(tmp_path, "rep", "カメラ_Cam1", now).name == "rep_重ね合わせ_カメラ_Cam1_20261112_090807.png"
    assert cv.overlay_png_path(tmp_path, "rep", "すべてのカメラ", now).name == "rep_重ね合わせ_すべてのカメラ_20261112_090807.png"
    assert cv.camera_display("カメラ_Cam1") == "Cam1" and cv.camera_display("Cam1") == "Cam1"


def test_on_white_composites_transparent_image():
    im = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    assert cv.on_white(im).getpixel((0, 0)) == (255, 255, 255, 255) and im.getpixel((0, 0))[3] == 0


def test_write_png_adds_suffix_and_creates_folder(tmp_path):
    im = Image.new("RGBA", (8, 6), (9, 8, 7, 255))
    p = cv.write_png(im, tmp_path / "新しい" / "重ね")                     # 拡張子なし → .png を付ける・無いフォルダは作る
    assert p.name == "重ね.png" and Image.open(p).getpixel((0, 0)) == (9, 8, 7, 255)
    assert cv.write_png(im, tmp_path / "a.PNG").name == "a.PNG"             # 大文字の .PNG はそのまま
