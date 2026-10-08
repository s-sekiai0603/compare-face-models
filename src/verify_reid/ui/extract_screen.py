from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.settings import load_settings, save_settings
from ..core.video_scanner import VideoInfo, format_duration, scan_videos
from .worker import ExtractWorker

HEADERS = ["選択", "ロケーション", "カメラID", "撮影日時", "映像の長さ", "ファイル名"]


class ExtractScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.videos: list[VideoInfo] = []
        self.worker: ExtractWorker | None = None

        layout = QVBoxLayout(self)
        self.root_edit = self._folder_row(layout, "映像ルートフォルダ", self.settings["video_root"], self._pick_root)
        self.out_edit = self._folder_row(layout, "外見画像出力先", self.settings["output_root"], self._pick_out)

        row = QHBoxLayout()
        row.addWidget(QLabel("抽出頻度(秒)"))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.1, 3600.0)
        self.interval.setSingleStep(0.1)
        self.interval.setDecimals(1)
        self.interval.setValue(float(self.settings["interval_sec"]))
        row.addWidget(self.interval)
        row.addStretch()
        btn_reload = QPushButton("一覧更新")
        btn_reload.clicked.connect(self.reload_list)
        btn_all = QPushButton("全選択")
        btn_none = QPushButton("全解除")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        row.addWidget(btn_all)
        row.addWidget(btn_none)
        row.addWidget(btn_reload)
        layout.addLayout(row)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.status = QLabel("")
        self.bar = QProgressBar()
        layout.addWidget(self.status)
        layout.addWidget(self.bar)

        btns = QHBoxLayout()
        self.btn_back = QPushButton("戻る")
        self.btn_run = QPushButton("抽出実行")
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        for b in (self.btn_back, self.btn_run, self.btn_cancel):
            btns.addWidget(b)
        layout.addLayout(btns)
        self.btn_back.clicked.connect(self.back_requested.emit)
        self.btn_run.clicked.connect(self.run_extract)
        self.btn_cancel.clicked.connect(lambda: self.worker and self.worker.cancel())

        if self.root_edit.text():
            self.reload_list()

    def _folder_row(self, parent: QVBoxLayout, label: str, value: str, slot) -> QLineEdit:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit(value)
        edit.setReadOnly(True)
        btn = QPushButton("選択...")
        btn.clicked.connect(slot)
        row.addWidget(edit, 1)
        row.addWidget(btn)
        parent.addLayout(row)
        return edit

    def _pick_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "映像ルートフォルダを選択", self.root_edit.text())
        if d:
            self.root_edit.setText(d)
            self._save()
            self.reload_list()

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "外見画像出力先を選択", self.out_edit.text())
        if d:
            self.out_edit.setText(d)
            self._save()

    def _save(self) -> None:
        self.settings.update(video_root=self.root_edit.text(), output_root=self.out_edit.text(),
                             interval_sec=round(self.interval.value(), 1))
        save_settings(self.settings)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(state)

    def reload_list(self) -> None:
        self.videos = scan_videos(Path(self.root_edit.text())) if self.root_edit.text() else []
        self.table.setRowCount(len(self.videos))
        for r, v in enumerate(self.videos):
            vals = [v.location, v.camera_id, v.captured_text, format_duration(v.duration_sec), v.file_name]
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            self.table.setItem(r, 0, chk)
            for c, text in enumerate(vals, start=1):
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.status.setText(f"{len(self.videos)} 件の映像が見つかりました")

    def run_extract(self) -> None:
        rows = [r for r in range(self.table.rowCount()) if self.table.item(r, 0).checkState() == Qt.Checked]
        if not rows:
            QMessageBox.warning(self, "画像抽出", "チェックボックスで映像を選択してください。")
            return
        if not self.out_edit.text():
            QMessageBox.warning(self, "画像抽出", "外見画像出力先を指定してください。")
            return
        self._save()  # 抽出実行時の頻度を次回の既定値にする
        self.worker = ExtractWorker([self.videos[r] for r in rows], Path(self.out_edit.text()),
                                    self.interval.value())
        self.worker.status.connect(self.status.setText)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self._set_running(True)
        self.bar.setValue(0)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_back.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _on_progress(self, idx: int, n: int, frame: int, total: int) -> None:
        if total > 0:
            self.bar.setValue(int((idx + frame / total) / n * 100))

    def _on_done(self, results: list, out_dir: str, cancelled: bool) -> None:
        self._set_running(False)
        tracks = sum(r.track_count for r in results)
        images = sum(r.image_count for r in results)
        self.bar.setValue(100 if not cancelled else self.bar.value())
        QMessageBox.information(
            self, "画像抽出", f"{'キャンセルしました' if cancelled else '完了しました'}\n"
            f"映像: {len(results)} 本 / 追跡ID: {tracks} / 画像: {images} 枚"
            f"（うち無効 {sum(r.invalid_count for r in results)} 枚）\n出力先: {out_dir}")

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        QMessageBox.critical(self, "画像抽出", f"エラーが発生しました:\n{msg}")
