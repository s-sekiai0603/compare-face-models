"""チャート PNG ビューアの処理（core/report/chart_png_viewer.py）。GPU・モデル重みなしで通る。"""
from pathlib import Path

from PIL import Image

from compare_face.core.report import chart_png_viewer as cv
from compare_face.core.report.distance_pose_chart import COLOR_GROUPS, chart_png_path, icon_choices


def _name(excel: Path, camera_label: str, g: int, icon_label: str) -> str:
    return chart_png_path(excel, f"{camera_label}・{COLOR_GROUPS[g][0]}・{icon_label}").name


def _touch(folder: Path, name: str, size=(40, 30), color=(0, 0, 0, 0)) -> Path:
    p = folder / name
    Image.new("RGBA", size, color).save(p)
    return p


def test_parse_matches_chart_png_path_rule(tmp_path):
    xlsx = tmp_path / "report.xlsx"
    icon_label, icon_file = icon_choices()[1]
    for cam, expect in (("すべてのカメラ", "すべてのカメラ"), ("カメラ Cam1", "カメラ_Cam1")):
        for g in range(len(COLOR_GROUPS)):
            name = _name(xlsx, cam, g, icon_label)
            assert cv.parse_chart_png_name(name) == ("report", (expect, g, icon_file))


def test_parse_rejects_other_files():
    assert cv.parse_chart_png_name("report_チャート.png") is None                       # OR 条件の PNG
    assert cv.parse_chart_png_name("report_チャート_すべてのカメラ.png") is None
    assert cv.parse_chart_png_name("memo.png") is None and cv.parse_chart_png_name("a_チャート_x・y・z.txt") is None
    assert cv.parse_chart_png_name("report_チャート_カメラ_C・不明な色・正面_×_正面.png") is None


def test_scan_and_select(tmp_path):
    xlsx = tmp_path / "rep.xlsx"
    il = icon_choices()
    (tmp_path / "other.txt").write_text("x")
    _touch(tmp_path, "rep_チャート.png")
    for cam in ("すべてのカメラ", "カメラ Cam1", "カメラ Cam2"):
        for g in (0, 1):
            _touch(tmp_path, _name(xlsx, cam, g, il[0][0]))
    idx = cv.scan_chart_pngs(tmp_path)
    assert idx.excel_names() == ["rep"] and idx.cameras("rep") == ["カメラ_Cam1", "カメラ_Cam2"] and idx.has_all_cameras("rep")
    assert cv.camera_display("カメラ_Cam1") == "Cam1"
    paths, missing = cv.select_paths(idx, "rep", [cv.ALL_CAMERAS], [0, 1, 2], [il[0][1], il[1][1]])
    assert len(paths) == 2 and missing == 4        # 色 3 × アイコン 2 = 6 のうち 2 つだけ存在
    assert cv.select_paths(idx, "rep", ["カメラ_Cam1"], [0], [il[0][1]])[1] == 0
    assert cv.scan_chart_pngs(tmp_path / "none").excel_names() == []


def test_overlay_composes_and_clears_title_band_after_first(tmp_path):
    a, b = _touch(tmp_path, "a.png", (100, 100)), _touch(tmp_path, "b.png", (100, 100))
    Image.new("RGBA", (100, 100), (255, 0, 0, 255)).save(a)                # 1 枚目: 全面赤
    Image.new("RGBA", (100, 100), (0, 0, 255, 255)).save(b)                # 2 枚目: 全面青（上端のタイトル帯は消える）
    ov = cv.ChartOverlay()
    assert ov.compose([]) is None
    first = ov.compose([a])
    assert first.getpixel((50, 5)) == (255, 0, 0, 255)
    both = ov.compose([a, b])
    assert both.size == (100, 100)
    assert both.getpixel((50, 5)) == (255, 0, 0, 255)                      # タイトル帯は 1 枚目のまま
    assert both.getpixel((50, 90)) == (0, 0, 255, 255)                     # それ以外は 2 枚目が上に重なる
    only_b = ov.compose([b])                                               # 外したら最初から重ね直す（b が先頭になりタイトルも残る）
    assert only_b.getpixel((50, 5)) == (0, 0, 255, 255)


def test_save_overlay_png_names_and_writes(tmp_path):
    from datetime import datetime
    now = datetime(2026, 11, 12, 9, 8, 7)
    assert cv.overlay_png_path(tmp_path, "rep", "カメラ_Cam1", now).name == "rep_重ね合わせ_カメラ_Cam1_20261112_090807.png"
    assert cv.overlay_png_path(tmp_path, "rep", "すべてのカメラ", now).name == "rep_重ね合わせ_すべてのカメラ_20261112_090807.png"
    out = tmp_path / "出力" / "sub"                                              # 無いフォルダは作る・日本語パス可
    p = cv.save_overlay_png(Image.new("RGBA", (30, 20), (1, 2, 3, 100)), out, "rep", "すべてのカメラ", now)
    assert p.parent == out and Image.open(p).size == (30, 20) and Image.open(p).getpixel((0, 0)) == (1, 2, 3, 100)


def test_select_paths_orders_back_to_front_with_color_priority(tmp_path):
    """先頭ほど手前。重ねる順（奥 → 手前）で返り、色の順が優先、同じ色の中ではアイコンの順。"""
    xlsx = tmp_path / "rep.xlsx"
    il = icon_choices()
    names = {}
    for g in (0, 1):
        for k in (0, 1):
            names[(g, k)] = _touch(tmp_path, _name(xlsx, "すべてのカメラ", g, il[k][0]))
    idx = cv.scan_chart_pngs(tmp_path)
    paths, _ = cv.select_paths(idx, "rep", [cv.ALL_CAMERAS], [0, 1], [il[0][1], il[1][1]])
    assert paths == [names[(1, 1)], names[(1, 0)], names[(0, 1)], names[(0, 0)]]       # 最後 = 最も手前 = 色 0・アイコン 0
    paths, _ = cv.select_paths(idx, "rep", [cv.ALL_CAMERAS], [1, 0], [il[1][1], il[0][1]])   # 並べ替えると手前が入れ替わる
    assert paths == [names[(0, 0)], names[(0, 1)], names[(1, 0)], names[(1, 1)]]


def test_overlay_reorders_and_extends_incrementally(tmp_path):
    red, blue = _touch(tmp_path, "r.png", (10, 10)), _touch(tmp_path, "b.png", (10, 10))
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(red)
    Image.new("RGBA", (10, 10), (0, 0, 255, 255)).save(blue)
    loads = []
    ov = cv.ChartOverlay(lambda p: (loads.append(p), cv.load_rgba(p))[1])
    assert ov.compose([red, blue]).getpixel((5, 9)) == (0, 0, 255, 255)      # 後ろに置いたものが手前
    assert ov.compose([blue, red]).getpixel((5, 9)) == (255, 0, 0, 255)      # 順序が変わったら重ね直し
    n = len(loads)
    assert ov.compose([blue, red]).getpixel((5, 9)) == (255, 0, 0, 255) and len(loads) == n   # 同じ並びなら読み直さない


def test_overlay_reports_progress(tmp_path):
    ps = [_touch(tmp_path, f"{k}.png", (10, 10), (k, 0, 0, 255)) for k in range(3)]
    calls = []
    ov = cv.ChartOverlay()
    ov.compose(ps, lambda done, total: calls.append((done, total)))
    assert calls == [(0, 3), (1, 3), (2, 3), (3, 3)]
    calls.clear()
    ov.compose(ps, lambda done, total: calls.append((done, total)))              # 変わらなければ重ねるものは無い
    assert calls == [(0, 0)]


def test_write_png_adds_suffix_and_creates_folder(tmp_path):
    im = Image.new("RGBA", (8, 6), (9, 8, 7, 255))
    p = cv.write_png(im, tmp_path / "新しい" / "重ね")                     # 拡張子なし → .png を付ける・無いフォルダは作る
    assert p.name == "重ね.png" and Image.open(p).getpixel((0, 0)) == (9, 8, 7, 255)
    assert cv.write_png(im, tmp_path / "a.PNG").name == "a.PNG"             # 大文字の .PNG はそのまま
