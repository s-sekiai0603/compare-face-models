from compare_face.core.report.chart_summary import DistanceRange, in_range, select_points, summarize
from compare_face.core.report.distance_pose_chart import COLOR_GROUPS, ChartData, ChartPoint, color_group_rgb, icon_choices

ICON_A = icon_choices()[0][1]
ICON_B = icon_choices()[1][1]


def pt(x, y, g, icon=ICON_A, cam="1"):
    return ChartPoint(x, y, icon, color_group_rgb(g), "", cam, cam)


def rows_of(rows, section_prefix):
    return {(r.kind, r.item): r for r in rows if r.section.startswith(section_prefix)}


def test_in_range_requires_both_and_includes_edges():
    assert in_range(pt(10.0, 0.0, 0), [DistanceRange.both(0.0, 10.0)])
    assert not in_range(pt(10.5, 5.0, 0), [DistanceRange.both(0.0, 10.0)])
    assert not in_range(pt(5.0, 10.5, 0), [DistanceRange.both(0.0, 10.0)])


def test_summarize_counts_and_ratios():
    points = [pt(1, 1, 0), pt(2, 2, 0, ICON_B), pt(3, 3, 1), pt(20, 1, 1)]
    rows = summarize(points, [0, 1], [ICON_A, ICON_B], [DistanceRange.both(0.0, 10.0)])
    whole, scoped = rows_of(rows, "全体"), rows_of(rows, "距離範囲")
    assert whole[("総数", "データ件数")].count == 4
    assert whole[("対象の色", COLOR_GROUPS[0][0])].count == 2
    assert whole[("対象の色", COLOR_GROUPS[1][0])].ratio == 0.5
    assert scoped[("総数", "データ件数")].count == 3          # x=20 の点は範囲外
    share = scoped[("総数", "全体に対するデータ件数・割合")]
    assert (share.count, share.ratio) == (3, 0.75)
    assert [r.item for r in rows if r.section.startswith("距離範囲")][:2] == ["全体に対するデータ件数・割合", "データ件数"]
    assert scoped[("対象の色", COLOR_GROUPS[1][0])].count == 1
    assert abs(scoped[("対象の色", COLOR_GROUPS[1][0])].ratio - 1 / 3) < 1e-9   # 分母は範囲内の件数
    label_a, label_b = dict((f, l) for l, f in icon_choices())[ICON_A], dict((f, l) for l, f in icon_choices())[ICON_B]
    assert whole[("色 × アイコン", f"{COLOR_GROUPS[0][0]} / {label_b}")].count == 1


def test_summarize_empty_range_has_no_ratio():
    rows = summarize([pt(50, 50, 0)], [0], [ICON_A], [DistanceRange.both(0.0, 10.0)])
    scoped = rows_of(rows, "距離範囲")
    assert scoped[("総数", "データ件数")].count == 0
    assert scoped[("対象の色", COLOR_GROUPS[0][0])].ratio is None


def test_select_points_by_camera():
    data = ChartData(points=[pt(1, 1, 0, cam="1"), pt(1, 1, 0, cam="2")])
    assert len(select_points(data, None)) == 2
    assert len(select_points(data, "1")) == 1
    assert select_points(data, "9") == []


def test_rows_carry_color_and_icon():
    rows = summarize([pt(1, 1, 0)], [0], [ICON_A], [DistanceRange.both(0.0, 10.0)])
    by_kind = {r.kind: r for r in rows if r.section == "全体"}
    assert by_kind["対象の色"].color == 0 and by_kind["対象の色"].icon is None
    assert by_kind["対象のアイコン"].icon == ICON_A and by_kind["対象のアイコン"].color is None
    assert (by_kind["色 × アイコン"].color, by_kind["色 × アイコン"].icon) == (0, ICON_A)


def test_condition_share_row_below_scoped_total():
    points = [pt(1, 1, 0), pt(2, 2, 1), pt(3, 3, 2), pt(20, 1, 0)]
    rows = [r for r in summarize(points, [0, 1], [ICON_A], [DistanceRange.both(0.0, 10.0)]) if r.section.startswith("距離範囲")]
    assert [r.item for r in rows[:3]] == ["全体に対するデータ件数・割合", "データ件数", "集計距離範囲内で指定条件に合致するデータ"]
    assert rows[2].count == 2 and abs(rows[2].ratio - 2 / 3) < 1e-9     # 範囲内 3 件のうち色 0・1 の 2 件


def test_similarity_stats_columns():
    a, b, c = pt(1, 1, 0), pt(2, 2, 0), pt(3, 3, 1)
    a.sim, b.sim, c.sim = 0.2, 0.6, 0.9
    rows = {(r.section, r.kind, r.item): r for r in summarize([a, b, c], [0, 1], [ICON_A], [DistanceRange.both(0.0, 10.0)])}
    s = rows[("全体", "対象の色", COLOR_GROUPS[0][0])].sim
    assert s[0] == 0.2 and s[1] == 0.6 and abs(s[2] - 0.4) < 1e-9 and abs(s[3] - 0.4) < 1e-9
    assert rows[("全体", "総数", "データ件数")].sim[3] == 0.6      # 3 件の中央値
    assert pt(1, 1, 2).sim is None


def test_read_chart_data_fills_upper_triangle_similarity(tmp_path):
    from openpyxl import Workbook
    from compare_face.core.report import distance_pose_chart as dpc
    from compare_face.core.report.verify_face_size_excel import FIRST_COL, FIRST_ROW, MISIDENTIFY_SHEET_NAME, SHEET_NAME
    wb = Workbook()
    ws = wb.active
    ws.title = MISIDENTIFY_SHEET_NAME
    ss = wb.create_sheet(SHEET_NAME)
    labels = {5: "カメラID", 6: "映像", 7: "追跡ID", 8: "画像", 9: "距離", 10: "顔の向き"}
    for r, name in labels.items():
        ws.cell(row=r, column=FIRST_COL - 1, value=name)
    for k, name in enumerate(["カメラID", "映像", "追跡ID", "画像", "距離", "顔の向き"]):
        ws.cell(row=FIRST_ROW - 1, column=2 + k, value=name)
    for j in range(2):
        for r, v in zip(range(5, 11), ["c1", "v", "1", f"i{j}", 1.0 + j, "正面"]):
            ws.cell(row=r, column=FIRST_COL + j, value=v)
    for i in range(2):
        for k, v in enumerate(["c1", "v", "1", f"i{i}", 1.0 + i, "正面"]):
            ws.cell(row=FIRST_ROW + i, column=2 + k, value=v)
    from openpyxl.styles import PatternFill
    for (i, j) in ((1, 0), (0, 1)):
        c = ws.cell(row=FIRST_ROW + i, column=FIRST_COL + j, value=0)
        c.fill = PatternFill("solid", fgColor="%02X%02X%02X" % dpc.color_group_rgb(0))
    ss.cell(row=FIRST_ROW + 1, column=FIRST_COL + 0, value=0.75)       # 左下（縦 1 × 横 0）だけに値がある
    path = tmp_path / "t.xlsx"
    wb.save(path)
    data = dpc.read_chart_data(path)
    assert len(data.points) == 2 and all(p.sim == 0.75 for p in data.points)


def test_separate_vertical_horizontal_range():
    r = DistanceRange(0.0, 5.0, 10.0, 15.0)            # 縦 0〜5 m・横 10〜15 m
    assert in_range(pt(12, 3, 0), [r])                 # pt(x=横, y=縦)
    assert not in_range(pt(3, 12, 0), [r])
    assert r.label() == "縦 0〜5 m・横 10〜15 m"


def test_two_ranges_are_merged_without_double_count():
    points = [pt(1, 1, 0), pt(12, 12, 0), pt(7, 7, 0), pt(1, 1, 1)]
    ranges = [DistanceRange.both(0.0, 5.0), DistanceRange.both(10.0, 15.0)]
    rows = {(r.section, r.kind, r.item): r for r in summarize(points, [0, 1], [ICON_A], ranges)}
    sec = "距離範囲 0〜5 m + 10〜15 m"
    assert rows[(sec, "総数", "データ件数")].count == 3          # 7 m の点は両方の範囲外
    overlap = [DistanceRange.both(0.0, 5.0), DistanceRange.both(0.0, 3.0)]
    assert next(r for r in summarize(points, [0], [ICON_A], overlap) if r.item == "データ件数" and r.section != "全体").count == 2
