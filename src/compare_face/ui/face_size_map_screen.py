"""指定顔サイズマッピング画面。

顔サイズマッピング: 顔 bbox の高さが指定サイズ以上で取れた人物の足元位置を、カメラ画像にヒートマップで重ねて保存する（core/report/face_size_map.py）。
距離範囲マッピング: カメラ設定（<カメラID>.json）から、カメラから指定距離以内の範囲をカメラ画像に半透明で塗って保存する（core/report/distance_range_map.py）。
モードに関係ない入力欄は非活性にする。
画面の入力内容は「プロット」押下時に face_size_map_screen_setting.json へ保存し、次回起動時に復元する。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QScrollArea,
    QButtonGroup, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QRadioButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from ..core.common.settings import load_face_size_map_settings, save_face_size_map_settings
from .report_worker import DistanceRangeMapWorker, FaceSizeMapWorker
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
        self.description = QLabel("")
        root.addWidget(self.description)

        mrow = QHBoxLayout()
        mrow.addWidget(fixed_label("モード:"))
        self.rb_size = QRadioButton("顔サイズマッピング")
        self.rb_range = QRadioButton("距離範囲マッピング")
        group = QButtonGroup(self)
        group.addButton(self.rb_size)
        group.addButton(self.rb_range)
        (self.rb_range if self.settings["mode"] == "distance_range" else self.rb_size).setChecked(True)
        mrow.addWidget(self.rb_size)
        mrow.addWidget(self.rb_range)
        mrow.addStretch()
        root.addLayout(mrow)

        row = QHBoxLayout()
        self.size_label = fixed_label("顔サイズ（bbox の高さ）:")
        row.addWidget(self.size_label)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 9999)
        self.size_spin.setSuffix(" px 以上")
        self.size_spin.setValue(int(self.settings["face_height"]))
        row.addWidget(self.size_spin)
        row.addStretch()
        root.addLayout(row)

        self.gallery_edit = QLineEdit(self.settings["gallery_dir"])
        self.gallery_edit.setReadOnly(True)
        self.gallery_btn = add_path_row(root, "比較対象フォルダ:", self.gallery_edit,
                                        lambda: self._pick(self.gallery_edit, "比較対象フォルダを選択"))

        row = QHBoxLayout()
        self.dist_label = fixed_label("カメラからの距離:")
        row.addWidget(self.dist_label)
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0.1, 9999.0)
        self.dist_spin.setDecimals(1)
        self.dist_spin.setSingleStep(1.0)
        self.dist_spin.setSuffix(" m 以内")
        self.dist_spin.setValue(float(self.settings["max_distance_m"]))
        row.addWidget(self.dist_spin)
        row.addStretch()
        root.addLayout(row)
        self.config_edit = QLineEdit(self.settings["config_dir"])
        self.config_edit.setReadOnly(True)
        self.config_btn = add_path_row(root, "カメラ設定フォルダ:", self.config_edit,
                                       lambda: self._pick(self.config_edit, "カメラ設定フォルダを選択"))
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

        # 結果画像のプレビュー（完了後に出力した画像を表示。複数枚はコンボボックスで切り替える）
        self._pixmap: QPixmap | None = None
        self.result_combo = QComboBox()
        self.result_combo.setVisible(False)
        self.result_combo.currentIndexChanged.connect(self._show_result)
        root.addWidget(self.result_combo)
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setWidget(self.result_label)
        self.result_scroll.setMinimumHeight(240)
        self.result_scroll.setVisible(False)
        root.addWidget(self.result_scroll, 1)
        self.bottom_spacer = QWidget()  # 結果が無い間は余白を下に集めて、項目を上に寄せる
        root.addWidget(self.bottom_spacer, 1)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_back.clicked.connect(self.back_requested.emit)
        self.rb_size.toggled.connect(self._update_mode)
        self._update_mode()

    def _is_range_mode(self) -> bool:
        return self.rb_range.isChecked()

    def _update_mode(self) -> None:
        """モードに関係ない入力欄を非活性にする。"""
        rng = self._is_range_mode()
        for w in (self.size_label, self.size_spin, self.gallery_edit, self.gallery_btn):
            w.setEnabled(not rng)
        for w in (self.dist_label, self.dist_spin, self.config_edit, self.config_btn):
            w.setEnabled(rng)
        self.btn_start.setText("距離範囲を塗る" if rng else "プロット")
        self.description.setText(
            "カメラ設定（<カメラID>.json）から、カメラから指定した距離以内の範囲を、カメラ画像ごとに半透明で塗りつぶす（魚眼補正後の画像）。"
            if rng else "顔 bbox の高さが指定サイズ以上の顔が取れた足元の位置を、カメラ画像ごとにヒートマップでプロットする。")

    def _pick(self, edit: QLineEdit, caption: str) -> None:
        d = QFileDialog.getExistingDirectory(self, caption, edit.text())
        if d:
            edit.setText(d)

    def _on_start(self) -> None:
        rng = self._is_range_mode()
        required = [(self.camera_edit, "カメラ画像フォルダ"), (self.out_edit, "結果出力先フォルダ")]
        required.append((self.config_edit, "カメラ設定フォルダ") if rng else (self.gallery_edit, "比較対象フォルダ"))
        for edit, name in required:
            if not edit.text():
                QMessageBox.warning(self, "指定顔サイズマッピング", f"{name}を指定してください。")
                return
        src_edit, src_name = required[2]
        if not Path(src_edit.text()).is_dir() or not Path(self.camera_edit.text()).is_dir():
            QMessageBox.warning(self, "指定顔サイズマッピング", f"{src_name}・カメラ画像フォルダが見つかりません。")
            return
        self.settings.update(mode="distance_range" if rng else "face_size", face_height=self.size_spin.value(),
                             gallery_dir=self.gallery_edit.text(), camera_dir=self.camera_edit.text(),
                             output_dir=self.out_edit.text(), config_dir=self.config_edit.text(),
                             max_distance_m=self.dist_spin.value())
        save_face_size_map_settings(self.settings)
        if rng:
            self.worker = DistanceRangeMapWorker(
                config_dir=Path(self.config_edit.text()), camera_dir=Path(self.camera_edit.text()),
                output_dir=Path(self.out_edit.text()), max_distance_m=self.dist_spin.value())
        else:
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
        self.rb_size.setEnabled(not running)
        self.rb_range.setEnabled(not running)

    def _on_finished(self, outputs: list, notes: list) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText(f"完了: {len(outputs)} 枚")
        self._set_results(outputs)
        msg = f"{len(outputs)} 枚の画像を出力しました。\n{self.out_edit.text()}"
        if notes:
            msg += "\n\n" + "\n".join(notes)
        QMessageBox.information(self, "指定顔サイズマッピング", msg)

    def _on_failed(self, msg: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText("失敗しました")
        QMessageBox.critical(self, "指定顔サイズマッピング", f"エラーが発生しました:\n{msg}")

    def _set_results(self, outputs: list) -> None:
        """出力した画像を画面に表示する（先頭の 1 枚を表示。複数あればコンボボックスで選ぶ）。"""
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        for p in outputs:
            self.result_combo.addItem(Path(p).name, str(p))
        self.result_combo.blockSignals(False)
        self.result_combo.setVisible(len(outputs) > 1)
        self.result_scroll.setVisible(bool(outputs))
        self.bottom_spacer.setVisible(not outputs)
        if outputs:
            self._show_result(0)

    def _show_result(self, index: int) -> None:
        path = self.result_combo.itemData(index)
        pm = QPixmap(path) if path else QPixmap()  # 日本語パスも Qt が読める
        self._pixmap = pm if not pm.isNull() else None
        self._fit_result()

    def _fit_result(self) -> None:
        if self._pixmap is None:
            self.result_label.setText("画像を表示できません")
            return
        size = self.result_scroll.viewport().size()
        self.result_label.setPixmap(self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._pixmap is not None:
            self._fit_result()
