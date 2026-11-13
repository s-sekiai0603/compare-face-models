"""集計画面。距離・向き・精度分析のレポート（Excel）の点を、指定した距離の範囲で集計して一覧にする（処理は core/report/chart_summary.py）。
チャート(PNG)確認画面の「集計画面」から開く。Excel の初期値は距離・向き・精度分析画面で指定した Excel。
一覧は範囲選択して Ctrl+C（または「一覧をコピー」）で Excel に貼り付けられる。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.report.chart_summary import select_points, summarize
from ..core.report.distance_pose_chart import COLOR_GROUPS, ICON_DIR, ChartData, cameras_of, color_group_rgb, icon_choices, read_chart_data
from .widgets import CopyTable, add_check_list, add_path_row, checked_values, fixed_label, swatch_icon, table_to_tsv

TITLE = "集計"
HIGHLIGHT_RGB = "#dff1fb"      # 距離範囲の総数の行の背景色（薄い水色）
HEADERS = ["区分", "種別", "項目", "件数", "割合", "類似度 最小", "類似度 最大", "類似度 平均", "類似度 中央値"]


class _LoadWorker(QThread):
    """Excel の読み込み（時間がかかるので別スレッドで行う）。"""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            self.done.emit(read_chart_data(self.path))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class ChartSummaryScreen(QWidget):
    """独立したウィンドウとして開く。settings は距離・向き・精度分析画面と共通の設定辞書、save は保存関数。"""

    def __init__(self, settings: dict | None = None, save: Callable[[dict], None] | None = None):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1100, 760)
        self.settings = settings if settings is not None else {}
        self._save = save
        self.data: ChartData | None = None
        self._loaded_key: tuple | None = None      # (パス, 更新時刻)。変わったら読み直す
        self._worker: _LoadWorker | None = None

        root = QVBoxLayout(self)
        self.excel_edit = QLineEdit(self.settings.get("summary_excel_path") or self.settings.get("excel_path") or "")
        self.excel_edit.setReadOnly(True)
        add_path_row(root, "Excel:", self.excel_edit, self._pick_excel)

        body = QHBoxLayout()
        root.addLayout(body, 1)
        side_w = QWidget()
        side_w.setFixedWidth(468)
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
        self.cam_list.setToolTip("「カメラ個別」のときに集計するカメラを 1 つ選ぶ（縦・横の画像の両方がそのカメラの組合せだけを使う）")
        crow.addWidget(self.cam_list, 1)
        side.addLayout(crow)
        self.rb_all.toggled.connect(self._update_camera_enabled)
        self._update_camera_enabled()

        excluded_colors = set(self.settings.get("summary_excluded_colors") or [])
        excluded_icons = set(self.settings.get("summary_excluded_icons") or [])
        color_items = [(label, g, swatch_icon(color_group_rgb(g)), g not in excluded_colors) for g, (label, _, _) in enumerate(COLOR_GROUPS)]
        icon_items = [(label, fname, QIcon(str(ICON_DIR / fname)), fname not in excluded_icons) for label, fname in icon_choices()]
        self.color_list = add_check_list(side, "対象の色:", "集計する色（誤識別数の段階）", 140, color_items)
        self.icon_list = add_check_list(side, "対象のアイコン:", "集計するアイコン（顔の向きの組合せ: 縦 × 横）", 220, icon_items)

        drow = QHBoxLayout()
        drow.addWidget(fixed_label("集計範囲の距離:", 100))
        self.min_spin, self.max_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, key, default in ((self.min_spin, "summary_d_min", 0.0), (self.max_spin, "summary_d_max", 10.0)):
            sp.setRange(0.0, 1000.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.5)
            sp.setSuffix(" m")
            try:
                sp.setValue(float(self.settings.get(key, default)))
            except (TypeError, ValueError):
                sp.setValue(default)
        self.min_spin.setToolTip("下限（この距離以上）")
        self.max_spin.setToolTip("上限（この距離以下）。縦の画像・横の画像の距離の両方が範囲内のデータを数える")
        drow.addWidget(self.min_spin)
        drow.addWidget(QLabel("〜"))
        drow.addWidget(self.max_spin)
        drow.addStretch()
        side.addLayout(drow)

        self.btn_run = QPushButton("集計")
        self.btn_run.clicked.connect(self._run)
        side.addWidget(self.btn_run)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFormat("Excel を読み込んでいます...")
        self.busy_bar.setVisible(False)
        side.addWidget(self.busy_bar)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        brow = QHBoxLayout()
        self.btn_copy = QPushButton("一覧をコピー")
        self.btn_copy.setToolTip("見出しを含めた一覧全体をコピーする（Excel に貼り付けられる）。範囲選択して Ctrl+C なら選択範囲だけ")
        self.btn_copy.clicked.connect(self._copy_all)
        self.btn_copy.setEnabled(False)
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        brow.addWidget(self.btn_copy)
        brow.addWidget(btn_close)
        side.addLayout(brow)
        side.addStretch()

        self.table = CopyTable(HEADERS)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 220)
        self.table.setColumnWidth(3, 70)
        body.addWidget(self.table, 1)

        if self.excel_edit.text():
            self._load(then_run=False)      # カメラの一覧を作るために読んでおく

    # ---- Excel ----
    def _pick_excel(self) -> None:
        start = self.excel_edit.text() or ""
        f, _ = QFileDialog.getOpenFileName(self, "集計する Excel を選択", start, "Excel (*.xlsx *.xlsm)")
        if f:
            self.excel_edit.setText(f)
            self.settings["summary_excel_path"] = f
            self._persist()
            self.data, self._loaded_key = None, None
            self._load(then_run=False)

    @staticmethod
    def _key(path: str) -> tuple | None:
        try:
            return (path, Path(path).stat().st_mtime_ns)
        except OSError:
            return None

    def _load(self, then_run: bool) -> None:
        """Excel を読み込む（別スレッド）。済んでいて更新されていなければ読み直さない。"""
        path = self.excel_edit.text()
        if not path or self._key(path) is None:
            QMessageBox.warning(self, TITLE, "Excel を指定してください（ファイルが見つかりません）。")
            return
        if self.data is not None and self._loaded_key == self._key(path):
            if then_run:
                self._summarize()
            return
        if self._worker is not None:
            return
        self._set_busy(True)
        worker = self._worker = _LoadWorker(path)
        key = self._key(path)
        worker.done.connect(lambda d, k=key, r=then_run: self._on_loaded(d, k, r))
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_thread_finished)
        worker.start()

    def _on_loaded(self, data: ChartData, key: tuple, then_run: bool) -> None:
        self.data, self._loaded_key = data, key
        self._rebuild_cameras()
        if then_run:
            self._summarize()
        else:
            self.status.setText(f"Excel を読み込みました（点 {len(data.points)} 件）。")

    def _on_failed(self, msg: str) -> None:
        self.status.setText(f"Excel を読み込めませんでした: {msg}")

    def _on_thread_finished(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.busy_bar.setVisible(busy)
        self.btn_run.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _rebuild_cameras(self) -> None:
        self.cam_list.clear()
        for cam in cameras_of(self.data) if self.data is not None else []:
            it = QListWidgetItem(f"カメラ {cam}")
            it.setData(Qt.UserRole, cam)
            self.cam_list.addItem(it)
        if self.cam_list.count():
            self.cam_list.setCurrentRow(0)
        self.rb_each.setEnabled(self.cam_list.count() > 0)

    def _update_camera_enabled(self) -> None:
        self.cam_list.setEnabled(self.rb_each.isChecked())

    # ---- 集計 ----
    def _run(self) -> None:
        if self.min_spin.value() > self.max_spin.value():
            QMessageBox.warning(self, TITLE, "距離の下限が上限より大きくなっています。")
            return
        self._load(then_run=True)

    def _summarize(self) -> None:
        camera = None
        if self.rb_each.isChecked():
            it = self.cam_list.currentItem()
            if it is None:
                QMessageBox.warning(self, TITLE, "対象カメラを選んでください。")
                return
            camera = it.data(Qt.UserRole)
        colors, _ = checked_values(self.color_list)
        icons, _ = checked_values(self.icon_list)
        d_min, d_max = self.min_spin.value(), self.max_spin.value()
        points = select_points(self.data, camera)
        rows = summarize(points, colors, icons, d_min, d_max)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            ratio = "" if row.ratio is None else f"{row.ratio * 100:.1f}%"
            sims = ("", "", "", "") if row.sim is None else tuple(f"{v:.3f}" for v in row.sim)
            for c, text in enumerate((row.section, row.kind, row.item, "" if row.count is None else str(row.count), ratio, *sims)):
                it = QTableWidgetItem(text)
                if row.kind == "総数":      # 全体・距離範囲の総数の行（追加行を含む）は薄い水色
                    it.setBackground(QColor(HIGHLIGHT_RGB))
                if c >= 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 2:
                    if row.color is not None:      # 順位のセルは背景を順位の色にする
                        it.setBackground(QColor(*color_group_rgb(row.color)))
                        it.setForeground(QColor("black"))
                    if row.icon is not None:       # 向きの組合せのセルは先頭にアイコンを出す
                        it.setIcon(QIcon(str(ICON_DIR / row.icon)))
                self.table.setItem(r, c, it)
        self.btn_copy.setEnabled(bool(rows))
        self.status.setText(f"集計しました（{'すべてのカメラ' if camera is None else 'カメラ ' + camera}、距離 {d_min:g}〜{d_max:g} m）。")
        self._persist_conditions(colors, icons)

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(table_to_tsv(self.table))
        self.status.setText("一覧をコピーしました。Excel に貼り付けてください。")

    # ---- 設定 ----
    def _persist_conditions(self, colors: list, icons: list) -> None:
        self.settings["summary_excluded_colors"] = [g for g in range(len(COLOR_GROUPS)) if g not in colors]
        self.settings["summary_excluded_icons"] = [f for _, f in icon_choices() if f not in icons]
        self.settings["summary_d_min"] = self.min_spin.value()
        self.settings["summary_d_max"] = self.max_spin.value()
        self._persist()

    def _persist(self) -> None:
        if self._save is not None:
            self._save(self.settings)

    def closeEvent(self, e) -> None:
        if self._worker is not None:
            self._worker.wait()
        super().closeEvent(e)
