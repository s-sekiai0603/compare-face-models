"""顔サイズ別精度検証画面。

指定した人物（追跡IDフォルダ 1〜10）について、検出した顔 bbox のサイズによる顔識別の精度の違いを検証する。
検索元セット検索画面をベースに、特徴量の抽出・類似度の比較は検索元セット検索と同じ処理を使う（core/report/verify_face_size.py）。
画面の入力内容は「レポート出力」押下時に verify_face_size_screen_setting.json へ保存し、次回起動時に復元する。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..core.common.settings import (
    VERIFY_FACE_SIZE_TRACK_COUNT, load_verify_face_size_settings, save_verify_face_size_settings,
)
from ..core.features import visible_extractors
from ..core.landmarks import ALIGNERS
from ..core.report.verify_face_size import OUTPUT_MODES, runs_misidentify_report
from .report_worker import VerifyFaceSizeWorker
from .widgets import DropLineEdit, GateFilterRow, WeightsGuard, add_model_box, add_path_row, checked_name, fixed_label

VISIBLE_TRACK_ROWS = 5  # 検証対象人物のグループボックスに初期表示で見せる行数


class VerifyFaceSizeScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_verify_face_size_settings()
        self.worker: VerifyFaceSizeWorker | None = None

        root = QVBoxLayout(self)
        title = QLabel("顔サイズ別精度検証")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        # 状態表示・ボタン（モデル選択の toggled から重み確認が走るので先に作り、レイアウトには最後に足す）
        self.guide = QPlainTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setMaximumHeight(110)
        self.guide.hide()
        self.btn_start = QPushButton("レポート出力")
        self.btn_back = QPushButton("戻る")
        self.status = QLabel("")
        self.weights = WeightsGuard(self.status, self.guide, self.btn_start, self.btn_back, lambda: self.worker is not None)

        # 出力内容
        mode_box = QGroupBox("出力内容")
        ml = QHBoxLayout(mode_box)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QRadioButton] = {}
        for key, label in OUTPUT_MODES.items():
            rb = QRadioButton(label)
            rb.setChecked(key == self.settings["output_mode"] or (key == "similarity" and self.settings["output_mode"] not in OUTPUT_MODES))
            self.mode_group.addButton(rb)
            self.mode_buttons[key] = rb
            ml.addWidget(rb)
        ml.addStretch()
        root.addWidget(mode_box)

        # 検証対象人物（追跡IDフォルダ 1〜10。縦スクロール）
        person_box = QGroupBox("検証対象人物")
        pl = QVBoxLayout(person_box)
        pl.addWidget(QLabel("追跡IDフォルダ（<比較対象フォルダ>\\<カメラID>\\<映像ファイル名>\\<追跡ID>）を最大 10 件指定。"
                            "直下の顔画像（*_face.jpg）を検証に使う。"))
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        self.track_edits: list[QLineEdit] = []
        for i in range(VERIFY_FACE_SIZE_TRACK_COUNT):
            edit = DropLineEdit(lambda path, ed=None: None, "フォルダを選択、またはドラッグ&ドロップ")
            edit.on_drop = edit.setText
            edit.setText(self.settings["track_dirs"][i])
            self.track_edits.append(edit)
            add_path_row(il, f"追跡IDフォルダ{i + 1}:", edit, lambda _=False, ed=edit, n=i + 1: self._pick_track(ed, n))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        row_h = inner.sizeHint().height() / VERIFY_FACE_SIZE_TRACK_COUNT
        scroll.setFixedHeight(int(row_h * VISIBLE_TRACK_ROWS) + 6)
        pl.addWidget(scroll)
        root.addWidget(person_box)

        # 比較対象フォルダ（誤識別となる画像の枚数レポートで、指定した追跡ID以外を非対象者として比べる）
        self.gallery_edit = QLineEdit(self.settings["gallery_dir"])
        self.gallery_edit.setReadOnly(True)
        add_path_row(root, "比較対象フォルダ:", self.gallery_edit, self._pick_gallery)
        self.out_edit = QLineEdit(self.settings["output_dir"])
        self.out_edit.setReadOnly(True)
        add_path_row(root, "レポート出力先フォルダ:", self.out_edit, self._pick_output)

        self.gate_row = GateFilterRow({"report_face_gate": self.settings["face_gate"],
                                       "report_min_eye_ratio": self.settings["min_eye_ratio"],
                                       "report_min_crop_side": self.settings["min_crop_side"]})
        root.addWidget(self.gate_row)

        # モデル選択（レジストリへの登録でラジオが増える）
        self.model_group, self.model_buttons = add_model_box(
            self, root, "顔特徴量抽出モデル", visible_extractors(), self.settings["model"], self._on_model_changed)
        self.landmark_group, self.landmark_buttons = add_model_box(
            self, root, "顔ランドマークモデル（特徴量抽出前のアライン）", ALIGNERS, self.settings["landmark"], self._on_model_changed)

        root.addWidget(self.guide)
        brow = QHBoxLayout()
        brow.addWidget(self.btn_back)
        brow.addWidget(self.btn_start, 1)
        root.addLayout(brow)
        root.addWidget(self.status)
        root.addStretch()

        self.btn_start.clicked.connect(self._on_start)
        self.btn_back.clicked.connect(self.back_requested.emit)
        self._start_weights_check()

    # ---- UI部品 ----
    def selected_mode(self) -> str:
        return checked_name(self.mode_buttons)

    def selected_model(self) -> str:
        return checked_name(self.model_buttons)

    def selected_landmark(self) -> str:
        return checked_name(self.landmark_buttons)

    def track_dirs(self) -> list[str]:
        return [e.text().strip() for e in self.track_edits]

    # ---- フォルダ選択 ----
    def _pick_track(self, edit: QLineEdit, n: int) -> None:
        d = QFileDialog.getExistingDirectory(self, f"追跡IDフォルダ{n}を選択", edit.text())
        if d:
            edit.setText(d)

    def _pick_gallery(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "比較対象フォルダを選択", self.gallery_edit.text())
        if d:
            self.gallery_edit.setText(d)

    def _pick_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "レポート出力先フォルダを選択", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    # ---- 重みの確認 ----
    def _on_model_changed(self, checked: bool) -> None:
        if checked:
            self._start_weights_check()

    def _start_weights_check(self) -> None:
        self.weights.check(self.selected_model(), self.selected_landmark(), "なし")

    # ---- 入力内容の保存 ----
    def _save_settings(self) -> None:
        gate = {}
        self.gate_row.save_to(gate)
        self.settings.update(
            output_mode=self.selected_mode(), track_dirs=self.track_dirs(), gallery_dir=self.gallery_edit.text(),
            output_dir=self.out_edit.text(),
            model=self.selected_model(), landmark=self.selected_landmark(), face_gate=gate["report_face_gate"],
            min_eye_ratio=gate["report_min_eye_ratio"], min_crop_side=gate["report_min_crop_side"])
        save_verify_face_size_settings(self.settings)

    # ---- レポート出力 ----
    def _on_start(self) -> None:
        dirs = [d for d in self.track_dirs() if d]
        if not dirs:
            QMessageBox.warning(self, "顔サイズ別精度検証", "検証対象人物の追跡IDフォルダを 1 件以上指定してください。")
            return
        if not self.out_edit.text():
            QMessageBox.warning(self, "顔サイズ別精度検証", "レポート出力先フォルダを指定してください。")
            return
        mode = self.selected_mode()
        if runs_misidentify_report(mode) and not self.gallery_edit.text():
            QMessageBox.warning(self, "顔サイズ別精度検証", "誤識別となる画像の枚数レポートには比較対象フォルダの指定が必要です。")
            return
        self._save_settings()
        self.worker = VerifyFaceSizeWorker(
            track_dirs=[Path(d) for d in dirs], output_dir=Path(self.out_edit.text()),
            model_name=self.selected_model(), landmark_name=self.selected_landmark(), gate_filter=self.gate_row.value(),
            mode=mode, gallery_dir=Path(self.gallery_edit.text()) if self.gallery_edit.text() else None)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._set_running(True)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and self.weights.ok)
        self.btn_back.setEnabled(not running)

    def _on_finished(self, path: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText(f"完了: {path}")
        txt = Path(path).with_suffix(".txt")
        extra = f"\n誤認した画像の一覧: {txt}" if txt.is_file() else ""
        QMessageBox.information(self, "顔サイズ別精度検証", f"Excel を出力しました。\n{path}{extra}")

    def _on_failed(self, msg: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText("失敗しました")
        QMessageBox.critical(self, "顔サイズ別精度検証", f"エラーが発生しました:\n{msg}")
