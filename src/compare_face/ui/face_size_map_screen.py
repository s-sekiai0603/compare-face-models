"""指定顔サイズマッピング画面。

顔 bbox の高さが指定サイズ以上で取れた人物の足元位置を、カメラ画像にヒートマップで重ねて保存する（core/report/face_size_map.py）。
画面の入力内容は「プロット」押下時に face_size_map_screen_setting.json へ保存し、次回起動時に復元する。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.common.settings import load_face_size_map_settings, save_face_size_map_settings
from .report_worker import FaceSizeMapWorker
from .widgets import add_path_row, fixed_label


class FaceSizeMapScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_face_size_map_settings()
        self.worker: FaceSizeMapWorker | None = None

        root = QVBoxLayout(self)
        title = QLabel("指定顔サイズマッピング")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)
        root.addWidget(QLabel("顔 bbox の高さが指定サイズ以上の顔が取れた足元の位置を、カメラ画像ごとにヒートマップでプロットする。"))

        row = QHBoxLayout()
        row.addWidget(fixed_label("顔サイズ（bbox の高さ）:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 9999)
        self.size_spin.setSuffix(" px 以上")
        self.size_spin.setValue(int(self.settings["face_height"]))
        row.addWidget(self.size_spin)
        row.addStretch()
        root.addLayout(row)

        self.gallery_edit = QLineEdit(self.settings["gallery_dir"])
        self.gallery_edit.setReadOnly(True)
        add_path_row(root, "比較対象フォルダ:", self.gallery_edit, lambda: self._pick(self.gallery_edit, "比較対象フォルダを選択"))
        self.camera_edit = QLineEdit(self.settings["camera_dir"])
        self.camera_edit.setReadOnly(True)
        add_path_row(root, "カメラ画像フォルダ:", self.camera_edit, lambda: self._pick(self.camera_edit, "カメラ画像フォルダを選択"))
        self.out_edit = QLineEdit(self.settings["output_dir"])
        self.out_edit.setReadOnly(True)
        add_path_row(root, "結果出力先フォルダ:", self.out_edit, lambda: self._pick(self.out_edit, "結果出力先フォルダを選択"))

        self.btn_start = QPushButton("プロット")
        self.btn_back = QPushButton("戻る")
        brow = QHBoxLayout()
        brow.addWidget(self.btn_back)
        brow.addWidget(self.btn_start, 1)
        root.addLayout(brow)
        self.status = QLabel("")
        root.addWidget(self.status)
        root.addStretch()

        self.btn_start.clicked.connect(self._on_start)
        self.btn_back.clicked.connect(self.back_requested.emit)

    def _pick(self, edit: QLineEdit, caption: str) -> None:
        d = QFileDialog.getExistingDirectory(self, caption, edit.text())
        if d:
            edit.setText(d)

    def _on_start(self) -> None:
        for edit, name in ((self.gallery_edit, "比較対象フォルダ"), (self.camera_edit, "カメラ画像フォルダ"),
                           (self.out_edit, "結果出力先フォルダ")):
            if not edit.text():
                QMessageBox.warning(self, "指定顔サイズマッピング", f"{name}を指定してください。")
                return
        if not Path(self.gallery_edit.text()).is_dir() or not Path(self.camera_edit.text()).is_dir():
            QMessageBox.warning(self, "指定顔サイズマッピング", "比較対象フォルダ・カメラ画像フォルダが見つかりません。")
            return
        self.settings.update(face_height=self.size_spin.value(), gallery_dir=self.gallery_edit.text(),
                             camera_dir=self.camera_edit.text(), output_dir=self.out_edit.text())
        save_face_size_map_settings(self.settings)
        self.worker = FaceSizeMapWorker(
            gallery_dir=Path(self.gallery_edit.text()), camera_dir=Path(self.camera_edit.text()),
            output_dir=Path(self.out_edit.text()), min_face_height=self.size_spin.value())
        self.worker.progress.connect(self.status.setText)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._set_running(True)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_back.setEnabled(not running)

    def _on_finished(self, outputs: list, notes: list) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText(f"完了: {len(outputs)} 枚")
        msg = f"{len(outputs)} 枚の画像を出力しました。\n{self.out_edit.text()}"
        if notes:
            msg += "\n\n" + "\n".join(notes)
        QMessageBox.information(self, "指定顔サイズマッピング", msg)

    def _on_failed(self, msg: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText("失敗しました")
        QMessageBox.critical(self, "指定顔サイズマッピング", f"エラーが発生しました:\n{msg}")
