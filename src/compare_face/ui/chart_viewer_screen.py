"""チャート(PNG)ビューア画面。

距離・向き・精度分析画面が「組合せごと」で出力した PNG を、出力対象・対象カメラ・対象の色・対象のアイコンを選んで重ねて表示する
（処理は core/report/chart_png_viewer.py）。表示域はマウスホイールで拡大・縮小、ドラッグで移動できる。
PNG の格納先の初期値は距離・向き・精度分析画面の「PNG の出力先」。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from ..core.report.chart_png_viewer import (
    ALL_CAMERAS, ChartOverlay, ChartPngIndex, camera_display, overlay_png_path, scan_chart_pngs, select_paths, write_png,
)
from ..core.report.distance_pose_chart import COLOR_GROUPS, ICON_DIR, color_group_rgb, icon_choices
from .widgets import add_check_list, add_path_row, checked_values, fixed_label, item_values, swatch_icon

TITLE = "チャート(PNG)確認"
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
    """PNG を重ねる処理（6000 × 4800 px の画像を何枚も読むので画面を止めないよう別スレッドで行う）。"""
    done = Signal(object)    # 重ねた PIL 画像（なければ None）
    failed = Signal(str)
    progress = Signal(int, int)    # (済んだ枚数, 今回重ねる枚数)

    def __init__(self, overlay: ChartOverlay, paths: list[Path]):
        super().__init__()
        self.overlay, self.paths = overlay, paths

    def run(self) -> None:
        try:
            self.done.emit(self.overlay.compose(self.paths, self.progress.emit))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


def pil_to_qimage(im) -> QImage:
    """RGBA の Pillow 画像 → QImage（バッファのコピーを持つ）。"""
    data = im.tobytes("raw", "RGBA")
    return QImage(data, im.width, im.height, im.width * 4, QImage.Format_RGBA8888).copy()


class ChartViewerScreen(QWidget):
    """独立したウィンドウとして開く。settings はビューアの保存項目（viewer_dir など）を持つ辞書、save は保存関数。"""

    def __init__(self, initial_dir: str, settings: dict | None = None, save: Callable[[dict], None] | None = None):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1280, 800)
        self.settings = settings if settings is not None else {}
        self._save = save
        self.index = ChartPngIndex({})
        self.overlay = ChartOverlay()
        self._worker: _ComposeWorker | None = None
        self._pending = False
        self._fit_next = True
        self._busy = False

        root = QVBoxLayout(self)
        self.dir_edit = QLineEdit(initial_dir)
        self.dir_edit.setReadOnly(True)
        add_path_row(root, "PNG の格納先:", self.dir_edit, self._pick_dir)
        hint = QLabel("「色・アイコンの条件 = 組合せごと」で出力した PNG を重ねて表示する（ホイール: 拡大・縮小 / ドラッグ: 移動）。"
                      "軸を共通にして出力した PNG が必要なため、古い PNG は再生成する。")
        hint.setWordWrap(True)
        root.addWidget(hint)

        body = QHBoxLayout()
        root.addLayout(body, 1)
        left = QVBoxLayout()
        body.addLayout(left, 1)
        self.view = PanZoomView()
        left.addWidget(self.view, 1)
        self.btn_export = QPushButton("重ねた画像をPNG出力")
        self.btn_export.setToolTip("表示中の重ねた画像を PNG で保存する。押すと保存先を指定するダイアログが開く")
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

        erow = QHBoxLayout()
        erow.addWidget(fixed_label("Excel 名:", 80))
        self.excel_combo = QComboBox()
        erow.addWidget(self.excel_combo, 1)
        side.addLayout(erow)

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

        self.btn_apply = QPushButton("条件確定（表示を更新）")
        self.btn_apply.setToolTip("上の条件（出力対象・対象カメラ・対象の色・対象のアイコン・並び順）で PNG を重ねて表示する。条件を変えただけでは表示は変わらない")
        self.btn_apply.clicked.connect(self._refresh)
        side.addWidget(self.btn_apply)
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

        self.excel_combo.currentIndexChanged.connect(self._on_excel_changed)
        self.rb_all.toggled.connect(self._on_target_changed)
        self.cam_list.currentRowChanged.connect(lambda _row: self._mark_dirty())
        self._rescan()      # 開いたときと格納先を変えたときは、読み込み直した内容を表示する（条件の変更とは別）

    # ---- 格納先・一覧 ----
    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "PNG の格納先フォルダを選択", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)
            self.settings["viewer_dir"] = d
            self._persist()
            self._rescan()

    def _export(self) -> None:
        """いま表示している重ねた画像を、保存先を指定するダイアログで決めた場所に PNG で保存する。
        ダイアログの初期位置は前回の保存先フォルダ（無ければ PNG の格納先）、初期のファイル名は overlay_png_path の規則。"""
        if self._worker is not None or self.overlay.image is None:
            QMessageBox.information(self, TITLE, "合成中、または重ねる PNG がありません。")
            return
        cams = self._selected_cameras()
        start_dir = self.settings.get("viewer_out_dir") or self.dir_edit.text()
        default = overlay_png_path(start_dir or ".", self.excel_combo.currentText(), cams[0] if cams else "")
        chosen, _ = QFileDialog.getSaveFileName(self, "重ねた画像の保存先を指定", str(default), "PNG (*.png)")
        if not chosen:
            return
        try:
            path = write_png(self.overlay.image, chosen)
        except OSError as e:
            QMessageBox.critical(self, TITLE, f"保存できませんでした:\n{e}")
            return
        self.settings["viewer_out_dir"] = str(path.parent)
        self._persist()
        self.status.setText(f"出力しました: {path}")
        QMessageBox.information(self, TITLE, f"PNG を出力しました。\n{path}")

    def _rescan(self) -> None:
        """格納先をスキャンし直して、Excel 名・カメラの一覧を作る。"""
        self.index = scan_chart_pngs(self.dir_edit.text()) if self.dir_edit.text() else ChartPngIndex({})
        self.excel_combo.blockSignals(True)
        self.excel_combo.clear()
        self.excel_combo.addItems(self.index.excel_names())
        self.excel_combo.blockSignals(False)
        self._fit_next = True
        self._rebuild_cameras()
        self._refresh()

    def _on_excel_changed(self) -> None:
        self._rebuild_cameras()
        self._mark_dirty()

    def _rebuild_cameras(self) -> None:
        """選んだ Excel 名に合わせて、対象カメラの一覧と出力対象（ラジオ）の選べる状態を作る。表示は更新しない。"""
        name = self.excel_combo.currentText()
        self.cam_list.blockSignals(True)
        self.cam_list.clear()
        for cam in self.index.cameras(name):
            it = QListWidgetItem(camera_display(cam))
            it.setData(Qt.UserRole, cam)
            self.cam_list.addItem(it)
        if self.cam_list.count():
            self.cam_list.setCurrentRow(0)
        self.cam_list.blockSignals(False)
        self.rb_all.setEnabled(self.index.has_all_cameras(name))
        self.rb_each.setEnabled(self.cam_list.count() > 0)
        if self.rb_all.isChecked() and not self.rb_all.isEnabled() and self.rb_each.isEnabled():
            self.rb_each.setChecked(True)
        elif self.rb_each.isChecked() and not self.rb_each.isEnabled() and self.rb_all.isEnabled():
            self.rb_all.setChecked(True)
        self._update_camera_enabled()

    def _on_target_changed(self) -> None:
        self._update_camera_enabled()
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """条件を変えたことを知らせる（表示は「条件確定」を押すまで変えない）。"""
        self.dirty_label.setText("条件が変更されています。「条件確定」を押すと表示を更新します。")

    def _update_camera_enabled(self) -> None:
        """「すべてのカメラ」では対象カメラを選べない（生成済みの「すべてのカメラ」の PNG を使う）。"""
        self.cam_list.setEnabled(self.rb_each.isChecked())

    # ---- 表示 ----
    def _selected_cameras(self) -> list[str]:
        if self.rb_all.isChecked():
            return [ALL_CAMERAS]
        it = self.cam_list.currentItem()
        return [it.data(Qt.UserRole)] if it is not None else []

    def _refresh(self) -> None:
        """選択に合う PNG を重ねて表示する（「条件確定」・開いたとき・格納先の変更時）。処理中なら終わってからもう一度行う。"""
        self.dirty_label.setText("")
        if self._worker is not None:
            self._pending = True
            return
        name = self.excel_combo.currentText()
        colors, _ = checked_values(self.color_list)
        icons, _ = checked_values(self.icon_list)
        paths, missing = select_paths(self.index, name, self._selected_cameras(), colors, icons)     # colors / icons は上にあるものほど手前
        if not self.index.files:
            self.status.setText("この格納先に、組合せごとに出力したチャートの PNG がありません。")
        else:
            self.status.setText(f"{len(paths)} 枚を重ねて表示" + (f"（点が無く出力されなかった組合せ {missing} 件）" if missing else ""))
        self._persist_selection(colors, icons)
        self.btn_export.setEnabled(False)    # 合成中は保存しない（合成が画像を書き換えるため）
        worker = self._worker = _ComposeWorker(self.overlay, paths)
        worker.done.connect(self._on_composed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_thread_finished)   # スレッドが完全に終わってから参照を手放す（実行中に破棄するとクラッシュする）
        worker.progress.connect(self._on_progress)
        self._set_busy(bool(paths))
        if paths:
            self.status.setText(self.status.text() + "　合成中...")
        self._worker.start()

    def _set_busy(self, busy: bool) -> None:
        """処理中の表示: 砂時計（待機カーソル）+ 動き続けるバー。「条件確定」は処理中は押せない。"""
        if busy:
            self.busy_bar.setRange(0, 0)
            self.busy_bar.setFormat("PNG を読み込んで重ねています...")
        if busy != self._busy:
            self._busy = busy
            if busy:
                QApplication.setOverrideCursor(QCursor(Qt.BusyCursor))
            else:
                QApplication.restoreOverrideCursor()
        self.busy_bar.setVisible(busy)
        self.btn_apply.setEnabled(not busy)

    def _on_progress(self, done: int, total: int) -> None:
        """枚数が分かったら進み具合（n / 枚数）を出す。0 枚目は読み込み中なので動き続ける表示のまま。"""
        if total > 0 and done > 0:
            self.busy_bar.setRange(0, total)
            self.busy_bar.setValue(done)
            self.busy_bar.setFormat(f"重ねています... {done} / {total} 枚")

    def _on_composed(self, im) -> None:
        if self._pending:
            return      # 選択が変わっているので、この結果は捨てて _on_thread_finished で作り直す
        self.btn_export.setEnabled(im is not None)
        self.view.set_image(pil_to_qimage(im) if im is not None else None, fit=self._fit_next and im is not None)
        if im is not None:
            self._fit_next = False
        self.status.setText(self.status.text().replace("　合成中...", ""))

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
