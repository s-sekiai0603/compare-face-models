"""チャートビューア画面。

距離・向き・精度分析画面が「組合せごと」で出力した PNG を、出力対象・対象カメラ・対象の色・対象のアイコンを選んで重ねて表示する
（処理は core/report/chart_png_viewer.py）。表示域はマウスホイールで拡大・縮小、ドラッグで移動できる。
どの表示モードも、あらかじめ出力した PNG は使わず、指定した Excel レポートから直接作る（Excel は距離・向き・精度分析画面の Excel が初期値）。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QGroupBox, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from ..core.report.angle_chart import AngleOverlay
from ..core.report.chart_summary import DistanceRange
from ..core.report.distance_heatmap import DistanceHeatmapOverlay
from ..core.report.chart_png_viewer import ALL_CAMERAS, camera_display, on_white, overlay_png_path, write_png
from ..core.report.distance_pose_chart import COLOR_GROUPS, ICON_DIR, cameras_of, color_group_rgb, icon_choices
from .chart_summary_screen import ChartSummaryScreen, RangeRow
from .widgets import add_check_list, add_path_row, checked_values, fixed_label, item_values, swatch_icon

TITLE = "チャート確認"
ZOOM_STEP = 1.15
ZOOM_RANGE = (0.02, 20.0)


def _ordered(items: list, order) -> list:
    """items（各要素の 2 番目が値）を保存済みの並び順 order に並べ替える（order に無いものは元の順で後ろ）。"""
    pos = {v: k for k, v in enumerate(order or [])}
    return sorted(items, key=lambda it: pos.get(it[1], len(pos)))


class PanZoomView(QGraphicsView):
    """ホイールで拡大・縮小（カーソル位置を中心）、ドラッグで表示位置を移動する。"""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#d9d9d9"))
        self._item: QGraphicsPixmapItem | None = None

    def set_image(self, img: QImage | None, fit: bool = False) -> None:
        """画像を差し替える。拡大率・位置は保つ（fit=True のときは全体表示にする）。"""
        scene = self.scene()
        if self._item is not None:
            scene.removeItem(self._item)
            self._item = None
        if img is None:
            return
        self._item = scene.addPixmap(QPixmap.fromImage(img))
        self._item.setTransformationMode(Qt.SmoothTransformation)
        scene.setSceneRect(self._item.boundingRect())
        if fit:
            self.fit_all()

    def has_image(self) -> bool:
        return self._item is not None

    def fit_all(self) -> None:
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        cur = self.transform().m11()
        factor = min(max(factor, ZOOM_RANGE[0] / cur), ZOOM_RANGE[1] / cur)
        self.scale(factor, factor)

    def wheelEvent(self, e) -> None:
        self.zoom(ZOOM_STEP if e.angleDelta().y() > 0 else 1 / ZOOM_STEP)
        e.accept()


class _ComposeWorker(QThread):
    """Excel を読んでチャートを描く処理（6000 × 4800 px の画像を描くので画面を止めないよう別スレッドで行う）。"""
    done = Signal(object)    # 重ねた PIL 画像（なければ None）
    failed = Signal(str)

    def __init__(self, overlay, paths: list[Path]):
        super().__init__()
        self.overlay, self.paths = overlay, paths

    def run(self) -> None:
        try:
            self.done.emit(self.overlay.compose(self.paths))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def pil_to_qimage(im) -> QImage:
    """RGBA の Pillow 画像 → QImage（バッファのコピーを持つ）。"""
    data = im.tobytes("raw", "RGBA")
    return QImage(data, im.width, im.height, im.width * 4, QImage.Format_RGBA8888).copy()


class ChartViewerScreen(QWidget):
    """独立したウィンドウとして開く。settings はビューアの保存項目（viewer_out_dir など）を持つ辞書、save は保存関数。"""

    def __init__(self, initial_dir: str, settings: dict | None = None, save: Callable[[dict], None] | None = None):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1280, 800)
        self.settings = settings if settings is not None else {}
        self._save = save
        self._initial_dir = initial_dir      # PNG 出力ダイアログの初期フォルダ（前回の保存先が無いとき）
        self.heat_overlay = DistanceHeatmapOverlay()      # アイコン表示・ヒートマップ表示（距離 × 距離）
        self.angle_overlay = AngleOverlay()
        self.overlay = self.heat_overlay      # 表示中のモードの合成器
        self._worker: _ComposeWorker | None = None
        self._pending = False
        self._fit_next = True
        self._busy = False

        root = QVBoxLayout(self)
        hint = QLabel("指定した Excel レポートから、距離・顔の向き・精度のチャートを表示する（ホイール: 拡大・縮小 / ドラッグ: 移動）。"
                      "PNG は使わない。距離の組合せ表示の「左右振り分け」「距離の集計単位」は、距離・向き・精度分析画面と同じ指定をする。")
        hint.setWordWrap(True)
        root.addWidget(hint)

        body = QHBoxLayout()
        root.addLayout(body, 1)
        left = QVBoxLayout()
        body.addLayout(left, 1)
        self.view = PanZoomView()
        left.addWidget(self.view, 1)
        self.btn_export = QPushButton("チャート画像出力")
        self.btn_export.setToolTip("表示中のチャートを PNG の画像で保存する。押すと保存先を指定するダイアログが開く")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        erow = QHBoxLayout()
        erow.addStretch()
        erow.addWidget(self.btn_export)
        left.addLayout(erow)

        side_w = QWidget()
        side_w.setFixedWidth(468)      # 一覧が見やすいよう、右パネルを左へ広げる（360 px の 1.3 倍）
        side = QVBoxLayout(side_w)
        side.setContentsMargins(0, 0, 0, 0)
        body.addWidget(side_w)

        mrow = QHBoxLayout()
        mrow.addWidget(fixed_label("出力対象:", 80))
        self.rb_all = QRadioButton("すべてのカメラ")
        self.rb_each = QRadioButton("カメラ個別")
        group = QButtonGroup(self)
        group.addButton(self.rb_all)
        group.addButton(self.rb_each)
        self.rb_all.setChecked(True)
        mrow.addWidget(self.rb_all)
        mrow.addWidget(self.rb_each)
        mrow.addStretch()
        side.addLayout(mrow)

        crow = QHBoxLayout()
        crow.addWidget(fixed_label("対象カメラ:", 80))
        self.cam_list = QListWidget()
        self.cam_list.setMaximumHeight(100)
        self.cam_list.setToolTip("「カメラ個別」のときに表示するカメラを 1 つ選ぶ（「すべてのカメラ」では選べない）")
        crow.addWidget(self.cam_list, 1)
        side.addLayout(crow)

        excluded_colors = set(self.settings.get("viewer_excluded_colors") or [])
        excluded_icons = set(self.settings.get("viewer_excluded_icons") or [])
        color_items = _ordered([(label, g, swatch_icon(color_group_rgb(g)), g not in excluded_colors)
                                for g, (label, _, _) in enumerate(COLOR_GROUPS)], self.settings.get("viewer_color_order"))
        icon_items = _ordered([(label, fname, QIcon(str(ICON_DIR / fname)), fname not in excluded_icons) for label, fname in icon_choices()],
                              self.settings.get("viewer_icon_order"))
        tip = "上にあるほど手前に表示する。ドラッグ＆ドロップか「上へ」「下へ」で並び順を変えられる。色とアイコンの順が衝突するときは色を優先する"
        self.color_list = add_check_list(side, "対象の色:", "重ねる色（誤識別数の段階）。" + tip, 140, color_items, self._mark_dirty, reorderable=True)
        self.icon_list = add_check_list(side, "対象のアイコン:", "重ねるアイコン（顔の向きの組合せ: 縦 × 横）。" + tip, 220, icon_items,
                                        self._mark_dirty, reorderable=True)
        order_hint = QLabel("※ 一覧の上にあるものほど手前に表示（色を優先し、同じ色の中ではアイコンの順）")
        order_hint.setWordWrap(True)
        side.addWidget(order_hint)

        mode_box = QGroupBox("表示モード")
        mode_lay = QHBoxLayout(mode_box)
        self.rb_dist = QRadioButton("距離の組合せ表示")
        self.rb_dist.setToolTip("指定した Excel レポートの点（横の画像の距離 × 縦の画像の距離）を、アイコン・ヒートマップで表示する")
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.rb_dist)
        mode_lay.addWidget(self.rb_dist)
        self.cb_angle = QRadioButton("顔角度の組合せ表示")
        self.cb_angle.setToolTip("指定した Excel レポートから、基準の顔画像の顔角度（縦軸）× 比較対象の顔角度（横軸）の組合せに、精度の色の点と"
                                 "加重平均のヒートマップを表示する。ONのときは出力対象のカメラ・対象の色・並び順を使う")
        mode_group.addButton(self.cb_angle)
        self.cb_angle.setChecked(bool(self.settings.get("angle_mode")))
        self.rb_dist.setChecked(not self.cb_angle.isChecked())
        mode_lay.addWidget(self.cb_angle)
        mode_lay.addStretch()
        side.addWidget(mode_box)
        self._build_heat_box(side)
        self._build_angle_box(side)
        self.cb_angle.toggled.connect(self._on_angle_toggled)

        self.btn_apply = QPushButton("条件確定（表示を更新）")
        self.btn_apply.setToolTip("上の条件（Excel・出力対象・対象カメラ・対象の色・対象のアイコン・並び順・表示モード）で表示を更新する。条件を変えただけでは表示は変わらない")
        self.btn_apply.clicked.connect(self._refresh)
        self.btn_summary = QPushButton("集計画面")
        self.btn_summary.setToolTip("距離・向き・精度分析の Excel を、指定した距離の範囲で集計する画面を開く")
        self.btn_summary.clicked.connect(self._open_summary)
        self._summary = None
        arow = QHBoxLayout()
        arow.addWidget(self.btn_apply, 1)
        arow.addWidget(self.btn_summary)
        side.addLayout(arow)
        self.busy_bar = QProgressBar()      # 処理中だけ出す。枚数が分かるまでは動き続ける表示（範囲 0〜0）
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(True)
        self.busy_bar.setVisible(False)
        side.addWidget(self.busy_bar)
        self.dirty_label = QLabel("")
        self.dirty_label.setStyleSheet("color: #c0392b;")
        self.dirty_label.setWordWrap(True)
        side.addWidget(self.dirty_label)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        brow = QHBoxLayout()
        btn_fit = QPushButton("全体表示")
        btn_fit.clicked.connect(self.view.fit_all)
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        brow.addWidget(btn_fit)
        brow.addWidget(btn_close)
        side.addLayout(brow)
        side.addStretch()

        self.rb_all.toggled.connect(self._on_target_changed)
        self.cam_list.currentRowChanged.connect(lambda _row: self._mark_dirty())
        self._update_angle_enabled()
        self._refresh()      # 開いたときは、指定された Excel の内容を表示する（条件の変更とは別）

    # ---- ヒートマップ表示（距離 × 距離。Excel から作る）----
    def _build_heat_box(self, side: QVBoxLayout) -> None:
        """アイコン表示・ヒートマップ表示の入力: Excel と、距離・向き・精度分析画面と同じ指定（顔の向きの左右振り分け・距離の集計単位。Excel からは分からないので指定する）。"""
        self.heat_box = QGroupBox("距離の組合せ表示の条件")
        lay = QVBoxLayout(self.heat_box)
        self.heat_excel_edit = QLineEdit(self.settings.get("heat_excel_path") or self.settings.get("excel_path") or self.settings.get("summary_excel_path") or "")
        self.heat_excel_edit.setReadOnly(True)
        add_path_row(lay, "Excel:", self.heat_excel_edit, self._pick_heat_excel)
        srow = QHBoxLayout()
        srow.addWidget(fixed_label("表示する内容:", 100))
        self.cb_dist_icon = QCheckBox("アイコン表示")
        self.cb_dist_heat = QCheckBox("ヒートマップ表示")
        self.cb_dist_icon.setToolTip("点を色・顔の向きのアイコンで表示する。重なりは対象の色・アイコンの並び順（上ほど手前）")
        self.cb_dist_heat.setToolTip("色 = 周辺の点の色の段階の平均、濃さ = 密集具合のヒートマップを表示する。アイコンと両方 ON のときはヒートマップの上にアイコンを重ねる")
        self.cb_dist_icon.setChecked(self.settings.get("dist_show_icons") is not False)
        self.cb_dist_heat.setChecked(self.settings.get("dist_show_heat") is True)
        srow.addWidget(self.cb_dist_icon)
        srow.addWidget(self.cb_dist_heat)
        srow.addStretch()
        lay.addLayout(srow)
        for w in (self.cb_dist_icon, self.cb_dist_heat):
            w.toggled.connect(lambda _c: self._mark_dirty())
        self.cb_heat_split = QCheckBox("顔の向きをチャートの左右に振り分け")
        self.cb_heat_split.setToolTip("距離・向き・精度分析画面の「顔の向きをチャートの左右に振り分け」と同じ。PNG と同じ見た目にしたいときはその指定に合わせる")
        self.cb_heat_split.setChecked(bool(self.settings.get("heat_split_lr")))
        lay.addWidget(self.cb_heat_split)
        urow = QHBoxLayout()
        urow.addWidget(fixed_label("距離の集計単位:", 100))
        self.heat_unit_spin = QDoubleSpinBox()
        self.heat_unit_spin.setRange(0.0, 100.0)
        self.heat_unit_spin.setDecimals(1)
        self.heat_unit_spin.setSingleStep(0.5)
        self.heat_unit_spin.setSuffix(" m（0 = 集計しない）")
        try:
            self.heat_unit_spin.setValue(max(0.0, float(self.settings.get("heat_unit_m") or 0.0)))
        except (TypeError, ValueError):
            pass
        self.heat_unit_spin.setToolTip("距離・向き・精度分析画面の「距離の集計単位」と同じ。0 = 集計せず、すべての組合せの点を使う")
        urow.addWidget(self.heat_unit_spin)
        urow.addStretch()
        lay.addLayout(urow)
        side.addWidget(self.heat_box)
        self.cb_heat_split.toggled.connect(lambda _c: self._mark_dirty())
        self.heat_unit_spin.valueChanged.connect(lambda _v: self._mark_dirty())

    def _pick_heat_excel(self) -> None:
        start = self.heat_excel_edit.text() or ""
        f, _ = QFileDialog.getOpenFileName(self, "距離の組合せ表示に使う Excel を選択", start, "Excel (*.xlsx *.xlsm)")
        if f:
            self.heat_excel_edit.setText(f)
            self.settings["heat_excel_path"] = f
            self._persist()
            self._mark_dirty()

    def _excel_overlay(self):
        return self.angle_overlay if self.cb_angle.isChecked() else self.heat_overlay

    # ---- 顔角度の組合せ表示 ----
    def _build_angle_box(self, side: QVBoxLayout) -> None:
        """顔角度の組合せ表示の入力: Excel と、距離のフィルタ（集計画面の「集計範囲の距離」と同じ指定。縦 = 基準の顔画像、横 = 比較対象）。"""
        self.angle_box = QGroupBox("顔角度の組合せ表示の条件")
        lay = QVBoxLayout(self.angle_box)
        self.angle_excel_edit = QLineEdit(self.settings.get("angle_excel_path") or self.settings.get("summary_excel_path")
                                          or self.settings.get("excel_path") or "")
        self.angle_excel_edit.setReadOnly(True)
        add_path_row(lay, "Excel:", self.angle_excel_edit, self._pick_angle_excel)
        srow = QHBoxLayout()
        srow.addWidget(fixed_label("表示する内容:", 100))
        self.cb_angle_points = QCheckBox("点（アイコン）")
        self.cb_angle_heat = QCheckBox("ヒートマップ")
        self.cb_angle_points.setChecked(self.settings.get("angle_show_points") is not False)
        self.cb_angle_heat.setChecked(self.settings.get("angle_show_heat") is not False)
        srow.addWidget(self.cb_angle_points)
        srow.addWidget(self.cb_angle_heat)
        srow.addStretch()
        lay.addLayout(srow)
        for w in (self.cb_angle_points, self.cb_angle_heat):
            w.toggled.connect(lambda _c: self._mark_dirty())
        self.cb_angle_filter = QCheckBox("カメラからの距離で絞り込む（縦 = 基準の顔画像、横 = 比較対象の顔画像）")
        self.cb_angle_filter.setChecked(bool(self.settings.get("angle_filter")))
        lay.addWidget(self.cb_angle_filter)
        mode_row = QHBoxLayout()
        mode_row.addWidget(fixed_label("集計範囲の距離:", 100))
        self.rb_angle_both = QRadioButton("縦横を合わせて指定")
        self.rb_angle_sep = QRadioButton("縦横を別々に指定")
        self.rb_angle_both.setToolTip("基準の画像・比較対象の画像の距離がどちらも同じ範囲に入る組合せだけを表示する")
        self.rb_angle_sep.setToolTip("基準の画像の距離（縦）と比較対象の画像の距離（横）に、別々の範囲を指定する")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_angle_both)
        grp.addButton(self.rb_angle_sep)
        (self.rb_angle_sep if self.settings.get("angle_separate") else self.rb_angle_both).setChecked(True)
        mode_row.addWidget(self.rb_angle_both)
        mode_row.addWidget(self.rb_angle_sep)
        mode_row.addStretch()
        lay.addLayout(mode_row)
        saved = self.settings.get("angle_ranges") or []
        self.angle_ranges = [RangeRow("範囲 1", saved[0] if len(saved) > 0 else None, False),
                             RangeRow("範囲 2", saved[1] if len(saved) > 1 else None, True)]
        self.angle_ranges[1].use.setChecked(bool(self.settings.get("angle_use_range2")))
        for rr in self.angle_ranges:
            lay.addWidget(rr)
        lay.addWidget(QLabel("※ 範囲を 2 つ指定したときは、どちらかに入る組合せを合算して表示する"))
        side.addWidget(self.angle_box)
        self.rb_angle_sep.toggled.connect(self._update_angle_range_mode)
        self.cb_angle_filter.toggled.connect(self._update_angle_enabled)
        for w in (self.cb_angle_filter, self.rb_angle_sep, self.angle_ranges[0].use, self.angle_ranges[1].use):
            w.toggled.connect(lambda _c: self._mark_dirty())
        for rr in self.angle_ranges:
            for sp in rr.spins:
                sp.valueChanged.connect(lambda _v: self._mark_dirty())
        self._update_angle_range_mode()

    def _update_angle_range_mode(self) -> None:
        for rr in self.angle_ranges:
            rr.set_separate(self.rb_angle_sep.isChecked())

    def _angle_ranges(self) -> list[DistanceRange] | None:
        """距離のフィルタ（かけないときは None）。"""
        if not self.cb_angle_filter.isChecked():
            return None
        sep = self.rb_angle_sep.isChecked()
        return [rr.value(sep) for rr in self.angle_ranges if rr.used()]

    def _update_angle_enabled(self) -> None:
        """顔角度の組合せ表示の ON / OFF と距離フィルタの有無に合わせて、使えない入力を無効にする。"""
        on = self.cb_angle.isChecked()
        self.angle_box.setVisible(on)
        self.heat_box.setVisible(not on)
        filt = self.cb_angle_filter.isChecked()
        for w in (self.rb_angle_both, self.rb_angle_sep, *self.angle_ranges):
            w.setEnabled(filt)
        if filt:
            for rr in self.angle_ranges:
                rr._enable()
        self.icon_list.setEnabled(not on)                                           # アイコンは顔角度の組合せ表示では使わない

    def _on_angle_toggled(self) -> None:
        self._update_angle_enabled()
        self._rebuild_cameras()
        self._mark_dirty()

    def _pick_angle_excel(self) -> None:
        start = self.angle_excel_edit.text() or ""
        f, _ = QFileDialog.getOpenFileName(self, "顔角度の組合せ表示に使う Excel を選択", start, "Excel (*.xlsx *.xlsm)")
        if f:
            self.angle_excel_edit.setText(f)
            self.settings["angle_excel_path"] = f
            self._persist()
            self._mark_dirty()

    def _fill_excel_cameras(self) -> None:
        """読み込み済みの Excel のカメラIDを対象カメラの一覧にする（まだ読んでいなければ空。「条件確定」で読む）。"""
        data = self._excel_overlay().data
        cams = (list(data.cameras) if self.cb_angle.isChecked() else cameras_of(data)) if data is not None else []
        cur = self.cam_list.currentItem().data(Qt.UserRole) if self.cam_list.currentItem() is not None else None
        self.cam_list.blockSignals(True)
        self.cam_list.clear()
        for cam in cams:
            it = QListWidgetItem(camera_display(cam))
            it.setData(Qt.UserRole, cam)
            self.cam_list.addItem(it)
        if self.cam_list.count():
            self.cam_list.setCurrentRow(cams.index(cur) if cur in cams else 0)
        self.cam_list.blockSignals(False)
        self.rb_all.setEnabled(True)
        self.rb_each.setEnabled(self.cam_list.count() > 0)
        if self.rb_each.isChecked() and not self.rb_each.isEnabled():
            self.rb_all.setChecked(True)
        self._update_camera_enabled()

    # ---- 一覧 ----
    def _open_summary(self) -> None:
        """集計画面を開く（Excel の初期値は距離・向き・精度分析画面の Excel。設定は共有）。"""
        self._summary = ChartSummaryScreen(self.settings, self._save)
        self._summary.show()

    def _export(self) -> None:
        """いま表示している重ねた画像を、保存先を指定するダイアログで決めた場所に PNG で保存する。
        ダイアログの初期位置は前回の保存先フォルダ（無ければ画面を開いたときの PNG 出力先）、初期のファイル名は overlay_png_path の規則。"""
        if self._worker is not None or self.overlay.image is None:
            QMessageBox.information(self, TITLE, "描画中、または表示している画像がありません。")
            return
        cams = self._selected_cameras()
        start_dir = self.settings.get("viewer_out_dir") or self._initial_dir
        if self.overlay is self.angle_overlay:
            excel_name = Path(self.angle_excel_edit.text()).stem + "_顔角度"
        else:
            excel_name = Path(self.heat_excel_edit.text()).stem + ("_ヒートマップ" if not self.cb_dist_icon.isChecked() else "_アイコン")
        default = overlay_png_path(start_dir or ".", excel_name, cams[0] if cams else "")
        chosen, _ = QFileDialog.getSaveFileName(self, "重ねた画像の保存先を指定", str(default), "PNG (*.png)")
        if not chosen:
            return
        try:
            path = write_png(on_white(self.overlay.image), chosen)
        except OSError as e:
            QMessageBox.critical(self, TITLE, f"保存できませんでした:\n{e}")
            return
        self.settings["viewer_out_dir"] = str(path.parent)
        self._persist()
        self.status.setText(f"出力しました: {path}")
        QMessageBox.information(self, TITLE, f"PNG を出力しました。\n{path}")

    def _rebuild_cameras(self) -> None:
        """読み込み済みの Excel のカメラで、対象カメラの一覧と出力対象（ラジオ）の選べる状態を作る。表示は更新しない。"""
        self._fill_excel_cameras()

    def _on_target_changed(self) -> None:
        self._update_camera_enabled()
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """条件を変えたことを知らせる（表示は「条件確定」を押すまで変えない）。"""
        self.dirty_label.setText("条件が変更されています。「条件確定」を押すと表示を更新します。")

    def _update_camera_enabled(self) -> None:
        """「すべてのカメラ」では対象カメラを選べない。"""
        self.cam_list.setEnabled(self.rb_each.isChecked())

    # ---- 表示 ----
    def _selected_cameras(self) -> list[str]:
        if self.rb_all.isChecked():
            return [ALL_CAMERAS]
        it = self.cam_list.currentItem()
        return [it.data(Qt.UserRole)] if it is not None else []

    def _refresh(self) -> None:
        """指定した Excel から表示を作る（「条件確定」・開いたとき）。処理中なら終わってからもう一度行う。"""
        self.dirty_label.setText("")
        if self._worker is not None:
            self._pending = True
            return
        colors, _ = checked_values(self.color_list)
        icons, _ = checked_values(self.icon_list)
        self._refresh_excel(colors, icons)

    def _refresh_excel(self, colors: list, icons: list) -> None:
        """アイコン表示 / ヒートマップ表示 / 顔角度の組合せ表示: 指定した Excel から点を読んで描く（別スレッド。Excel は更新されていなければ読み直さない）。"""
        angle = self.cb_angle.isChecked()
        path = (self.angle_excel_edit if angle else self.heat_excel_edit).text()
        if not path or not Path(path).is_file():
            self.status.setText(("顔角度の組合せ表示" if angle else "表示") + "に使う Excel を指定してください（ファイルが見つかりません）。")
            return
        cams = self._selected_cameras()
        camera = None if not cams or cams[0] == ALL_CAMERAS else cams[0]
        if angle:
            ranges = self._angle_ranges()
            if ranges is not None and not ranges:
                self.status.setText("距離の範囲を 1 つ以上指定してください。")
                return
            if not (self.cb_angle_points.isChecked() or self.cb_angle_heat.isChecked()):
                self.status.setText("表示する内容（点・ヒートマップ）を 1 つ以上選んでください。")
                return
            self.angle_overlay.configure(path, camera, colors, ranges, self.cb_angle_points.isChecked(), self.cb_angle_heat.isChecked())
        else:
            if not (self.cb_dist_icon.isChecked() or self.cb_dist_heat.isChecked()):
                self.status.setText("表示する内容（アイコン・ヒートマップ）を 1 つ以上選んでください。")
                return
            self.heat_overlay.configure(path, camera, colors, icons, self.cb_heat_split.isChecked(), round(self.heat_unit_spin.value(), 1),
                                        self.cb_dist_icon.isChecked(), self.cb_dist_heat.isChecked())
        self.overlay = self._excel_overlay()
        self._persist_selection(colors, icons)
        self.btn_export.setEnabled(False)
        self.status.setText("Excel を読み込んで描画中...")
        worker = self._worker = _ComposeWorker(self.overlay, [])
        worker.done.connect(self._on_composed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_thread_finished)
        self._set_busy(True)
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        """処理中の表示: 砂時計（待機カーソル）+ 動き続けるバー。「条件確定」は処理中は押せない。"""
        if busy:
            self.busy_bar.setRange(0, 0)
            self.busy_bar.setFormat("Excel を読み込んで描画しています...")
        if busy != self._busy:
            self._busy = busy
            if busy:
                QApplication.setOverrideCursor(QCursor(Qt.BusyCursor))
            else:
                QApplication.restoreOverrideCursor()
        self.busy_bar.setVisible(busy)
        self.btn_apply.setEnabled(not busy)

    def _on_composed(self, im) -> None:
        if self._pending:
            return      # 選択が変わっているので、この結果は捨てて _on_thread_finished で作り直す
        self.btn_export.setEnabled(im is not None)
        self.view.set_image(pil_to_qimage(on_white(im)) if im is not None else None, fit=self._fit_next and im is not None)
        if im is not None:
            self._fit_next = False
        self.status.setText(self.overlay.message)
        self._fill_excel_cameras()

    def _on_failed(self, msg: str) -> None:
        self.status.setText(f"表示できませんでした: {msg}")

    def _on_thread_finished(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        if self._pending:
            self._pending = False
            self._refresh()
        else:
            self._set_busy(False)

    # ---- 設定 ----
    def _persist_selection(self, colors: list, icons: list) -> None:
        self.settings["viewer_excluded_colors"] = [g for g in range(len(COLOR_GROUPS)) if g not in colors]
        self.settings["viewer_excluded_icons"] = [f for _, f in icon_choices() if f not in icons]
        self.settings["dist_show_icons"] = self.cb_dist_icon.isChecked()
        self.settings["dist_show_heat"] = self.cb_dist_heat.isChecked()
        self.settings["angle_mode"] = self.cb_angle.isChecked()
        self.settings["angle_excel_path"] = self.angle_excel_edit.text()
        self.settings["heat_excel_path"] = self.heat_excel_edit.text()
        self.settings["heat_split_lr"] = self.cb_heat_split.isChecked()
        self.settings["heat_unit_m"] = round(self.heat_unit_spin.value(), 1)
        self.settings["angle_show_points"] = self.cb_angle_points.isChecked()
        self.settings["angle_show_heat"] = self.cb_angle_heat.isChecked()
        self.settings["angle_filter"] = self.cb_angle_filter.isChecked()
        self.settings["angle_separate"] = self.rb_angle_sep.isChecked()
        self.settings["angle_ranges"] = [rr.raw() for rr in self.angle_ranges]
        self.settings["angle_use_range2"] = self.angle_ranges[1].used()
        self.settings["viewer_color_order"] = item_values(self.color_list)
        self.settings["viewer_icon_order"] = item_values(self.icon_list)
        self._persist()

    def _persist(self) -> None:
        if self._save is not None:
            self._save(self.settings)

    def closeEvent(self, e) -> None:
        if self._worker is not None:
            self._worker.wait()
        self._set_busy(False)      # 待機カーソルを戻す
        super().closeEvent(e)
