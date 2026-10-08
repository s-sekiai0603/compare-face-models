"""特徴量比較レポート出力画面（参照元 out_report_compared_feature.py の MainWindow 相当）。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Signal

from ..core.features import EXTRACTORS
from ..core.gallery import scan_track_folder
from ..core.report_excel import MODE_IMAGE, MODE_TRACK
from ..core.settings import load_settings, save_settings
from .report_worker import ReportWorker, WeightsWorker

LABEL_WIDTH = 140


class DropLineEdit(QLineEdit):
    """ファイル/フォルダのドラッグ&ドロップを受け付ける読み取り専用の入力欄。"""

    def __init__(self, on_drop, placeholder: str):
        super().__init__()
        self.on_drop = on_drop
        self.setReadOnly(True)
        self.setAcceptDrops(True)
        self.setPlaceholderText(placeholder)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.on_drop(urls[0].toLocalFile())


class ReportScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.worker: ReportWorker | None = None
        self.weights_worker: WeightsWorker | None = None
        self.weights_ok = False

        root = QVBoxLayout(self)
        title = QLabel("特徴量比較レポート出力")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        # クエリ指定方法
        row = QHBoxLayout()
        row.addWidget(self._label("クエリ指定方法:"))
        self.rb_image = QRadioButton(MODE_IMAGE)
        self.rb_track = QRadioButton(MODE_TRACK)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_image)
        grp.addButton(self.rb_track)
        row.addWidget(self.rb_image)
        row.addWidget(self.rb_track)
        row.addStretch()
        root.addLayout(row)

        self.query_edit = DropLineEdit(
            self._set_query_image, "外見画像(_body.jpg)をドラッグ&ドロップ、または「参照...」で選択")
        self._row(root, "クエリ外見画像:", self.query_edit, self._pick_query_image)
        self.track_edit = DropLineEdit(
            self._set_track_dir, "追跡IDフォルダをドラッグ&ドロップ、または「参照...」で選択")
        self.btn_track = self._row(root, "追跡IDフォルダ:", self.track_edit, self._pick_track_dir)
        self.track_info = QLabel("")
        root.addWidget(self.track_info)

        self.gallery_edit = QLineEdit(self.settings["report_gallery_dir"] or self.settings["output_root"])
        self.gallery_edit.setReadOnly(True)
        self._row(root, "比較対象フォルダ:", self.gallery_edit, self._pick_gallery)
        self.out_edit = QLineEdit(self.settings["report_output_dir"] or self.settings["output_root"])
        self.out_edit.setReadOnly(True)
        self._row(root, "レポート出力先フォルダ:", self.out_edit, self._pick_output)

        self.cb_skeleton = QCheckBox("骨格描画")
        self.cb_skeleton.setChecked(bool(self.settings["report_skeleton"]))
        root.addWidget(self.cb_skeleton)

        # 特徴量抽出モデル（将来はレジストリへの登録でラジオが増える）
        box = QGroupBox("特徴量抽出モデル")
        bl = QHBoxLayout(box)
        self.model_group = QButtonGroup(self)
        self.model_buttons: dict[str, QRadioButton] = {}
        saved = self.settings["report_model"]
        for i, name in enumerate(EXTRACTORS):
            rb = QRadioButton(name)
            rb.setChecked(name == saved or (i == 0 and saved not in EXTRACTORS))
            self.model_group.addButton(rb)
            self.model_buttons[name] = rb
            bl.addWidget(rb)
            rb.toggled.connect(self._on_model_changed)
        bl.addStretch()
        root.addWidget(box)

        self.guide = QPlainTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setMaximumHeight(110)
        self.guide.hide()
        root.addWidget(self.guide)

        self.btn_start = QPushButton("レポート作成開始")
        self.btn_back = QPushButton("戻る")
        brow = QHBoxLayout()
        brow.addWidget(self.btn_back)
        brow.addWidget(self.btn_start, 1)
        root.addLayout(brow)
        self.status = QLabel("")
        root.addWidget(self.status)
        root.addStretch()

        self.rb_image.toggled.connect(self._on_mode_toggled)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_back.clicked.connect(self.back_requested.emit)

        # 前回値の復元
        (self.rb_track if self.settings["report_mode"] == MODE_TRACK else self.rb_image).setChecked(True)
        q = self.settings["report_query_path"]
        if self.settings["report_mode"] != MODE_TRACK and q and Path(q).is_file():
            self.query_edit.setText(q)
        t = self.settings["report_track_dir"]
        if t and Path(t).is_dir():
            self._set_track_dir(t, quiet=True)
        self._on_mode_toggled()
        self._start_weights_check()

    # ---- UI部品 ----
    @staticmethod
    def _label(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setFixedWidth(LABEL_WIDTH)
        return lb

    def _row(self, parent: QVBoxLayout, label: str, edit: QLineEdit, slot) -> QPushButton:
        row = QHBoxLayout()
        row.addWidget(self._label(label))
        row.addWidget(edit, 1)
        btn = QPushButton("参照...")
        btn.clicked.connect(slot)
        row.addWidget(btn)
        parent.addLayout(row)
        return btn

    def selected_model(self) -> str:
        return next(n for n, b in self.model_buttons.items() if b.isChecked())

    # ---- モードと入力 ----
    def _on_mode_toggled(self) -> None:
        image_mode = self.rb_image.isChecked()
        self.query_edit.setEnabled(image_mode)
        self.track_edit.setEnabled(not image_mode)
        self.btn_track.setEnabled(not image_mode)
        self.track_info.setVisible(not image_mode)

    def _set_query_image(self, path: str) -> None:
        p = Path(path)
        if p.name.endswith("_body_aligned.jpg") or not p.name.endswith("_body.jpg") or not p.is_file():
            QMessageBox.warning(self, "クエリ外見画像",
                                "外見画像(_body.jpg)を指定してください。\n(_body_aligned.jpg は指定できません)")
            return
        self.query_edit.setText(str(p))

    def _pick_query_image(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "クエリ外見画像を選択", self.query_edit.text(),
                                           "外見画像 (*_body.jpg)")
        if f:
            self._set_query_image(f)

    def _set_track_dir(self, path: str, quiet: bool = False) -> None:
        p = Path(path)
        if not p.is_dir():
            if not quiet:
                QMessageBox.warning(self, "追跡IDフォルダ", "フォルダを指定してください。")
            return
        n = len(scan_track_folder(p))
        if n == 0 and not quiet:
            QMessageBox.warning(self, "追跡IDフォルダ", "外見画像(_body.jpg)が見つかりません。追跡IDフォルダを指定してください。")
            return
        self.track_edit.setText(str(p))
        self.track_info.setText(f"検出件数: 外見 {n}件")

    def _pick_track_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "追跡IDフォルダを選択", self.track_edit.text())
        if d:
            self._set_track_dir(d)

    def _pick_gallery(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "比較対象フォルダを選択", self.gallery_edit.text())
        if d:
            self.gallery_edit.setText(d)

    def _pick_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "レポート出力先フォルダを選択", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    # ---- 重みの自動取得 ----
    def _on_model_changed(self) -> None:
        if self.sender() is None or self.sender().isChecked():
            self._start_weights_check()

    def _start_weights_check(self) -> None:
        self.weights_ok = False
        self.guide.hide()
        self.btn_start.setEnabled(False)
        self.btn_back.setEnabled(False)  # 取得スレッド実行中は戻れない
        self.status.setText("モデルの重みを確認しています...")
        self.weights_worker = WeightsWorker(self.selected_model())
        self.weights_worker.status.connect(self.status.setText)
        self.weights_worker.ready.connect(self._on_weights_ready)
        self.weights_worker.unavailable.connect(self._on_weights_unavailable)
        self.weights_worker.finished.connect(lambda: self.btn_back.setEnabled(self.worker is None))
        self.weights_worker.start()

    def _on_weights_ready(self) -> None:
        self.weights_ok = True
        self.status.setText("")
        self.btn_start.setEnabled(self.worker is None)

    def _on_weights_unavailable(self, msg: str) -> None:
        self.status.setText("モデルの重みを取得できませんでした。下記の手順で配置してください。")
        self.guide.setPlainText(msg)
        self.guide.show()

    # ---- レポート作成 ----
    def _on_start(self) -> None:
        image_mode = self.rb_image.isChecked()
        query = self.query_edit.text() if image_mode else self.track_edit.text()
        if not query:
            QMessageBox.warning(self, "レポート作成", "クエリ外見画像または追跡IDフォルダを指定してください。")
            return
        if not self.gallery_edit.text() or not self.out_edit.text():
            QMessageBox.warning(self, "レポート作成", "比較対象フォルダとレポート出力先フォルダを指定してください。")
            return
        self.settings.update(
            report_mode=MODE_IMAGE if image_mode else MODE_TRACK,
            report_query_path=self.query_edit.text(), report_track_dir=self.track_edit.text(),
            report_gallery_dir=self.gallery_edit.text(), report_output_dir=self.out_edit.text(),
            report_skeleton=self.cb_skeleton.isChecked(), report_model=self.selected_model())
        save_settings(self.settings)
        self.worker = ReportWorker(
            model_name=self.selected_model(), mode=MODE_IMAGE if image_mode else MODE_TRACK,
            query_path=Path(query), gallery_dir=Path(self.gallery_edit.text()),
            output_dir=Path(self.out_edit.text()), skeleton=self.cb_skeleton.isChecked())
        self.worker.progress.connect(self.status.setText)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._set_running(True)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and self.weights_ok)
        self.btn_back.setEnabled(not running)

    def _on_finished(self, path: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText(f"完了: {path}")
        QMessageBox.information(self, "レポート作成", f"レポートを出力しました。\n{path}")

    def _on_failed(self, msg: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText("失敗しました")
        QMessageBox.critical(self, "レポート作成", f"エラーが発生しました:\n{msg}")
