"""顔特徴量比較レポート出力画面。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDoubleSpinBox, QSpinBox, QGridLayout, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt, Signal

from ..core.features import visible_extractors
from ..core.report.gallery import scan_track_folder
from ..core.landmarks import ALIGNERS
from ..core.report.excel import MAX_TARGETS, MODE_IMAGE, MODE_TRACK
from ..core.report.quality import QueryFilter
from ..core.common.settings import load_settings, save_settings
from .report_worker import ReportWorker
from .widgets import (
    DropLineEdit, GateFilterRow, WeightsGuard, add_model_box, add_path_row, checked_name, fixed_label, pose_registry,
)

TARGET_VISIBLE_ROWS = 5  # 探したい人物欄で最初から表示する行数（残りはスクロール）


class ReportScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.worker: ReportWorker | None = None

        root = QVBoxLayout(self)
        title = QLabel("顔特徴量比較レポート出力")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        # 状態表示・ボタン（モデル選択の toggled から重み確認が走るので先に作り、レイアウトには最後に足す）
        self.guide = QPlainTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setMaximumHeight(110)
        self.guide.hide()
        self.btn_start = QPushButton("レポート作成開始")
        self.btn_back = QPushButton("戻る")
        self.status = QLabel("")
        self.weights = WeightsGuard(self.status, self.guide, self.btn_start, self.btn_back, lambda: self.worker is not None)

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
            self._set_query_image, "顔画像(_face.jpg)をドラッグ&ドロップ、または「参照...」で選択")
        self._row(root, "クエリ顔画像:", self.query_edit, self._pick_query_image)
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

        self.cb_body = QCheckBox("全身画像も表示（顔画像の下に _body.jpg を並べる）")
        self.cb_body.setChecked(bool(self.settings["report_show_body"]))
        root.addWidget(self.cb_body)
        self.cb_include_old = QCheckBox("old フォルダの画像も含める（追跡IDフォルダ内の old に移した画像を、クエリ・比較対象に元の追跡IDとして戻して比較）")
        self.cb_include_old.setChecked(bool(self.settings["report_include_old"]))
        root.addWidget(self.cb_include_old)

        row = QHBoxLayout()
        row.addWidget(self._label("レポート出力件数:"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 100000)
        self.limit_spin.setSuffix(" 件（上位）")
        self.limit_spin.setValue(int(self.settings["report_limit"]))
        row.addWidget(self.limit_spin)
        row.addStretch()
        root.addLayout(row)

        row = QHBoxLayout()
        self.merge_label = self._label("特徴量マージ単位:")
        row.addWidget(self.merge_label)
        self.merge_spin = QDoubleSpinBox()
        self.merge_spin.setRange(0.0, 86400.0)
        self.merge_spin.setDecimals(1)
        self.merge_spin.setSingleStep(0.5)
        self.merge_spin.setSpecialValueText("指定なし")
        self.merge_spin.setSuffix(" 秒")
        self.merge_spin.setValue(float(self.settings["report_merge_sec"]))
        row.addWidget(self.merge_spin)
        row.addWidget(QLabel("（追跡ID指定時のみ。クエリ画像を時刻順に指定秒数ごとにグループ化して特徴量をマージ）"))
        row.addStretch()
        root.addLayout(row)

        # クエリの自動選別（比較対象には適用しない）
        row = QHBoxLayout()
        row.addWidget(self._label("クエリ自動選別:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setSpecialValueText("無効")
        self.conf_spin.setPrefix("顔信頼度 >= ")
        self.conf_spin.setValue(float(self.settings["report_min_face_conf"]))
        row.addWidget(self.conf_spin)
        self.norm_spin = QDoubleSpinBox()
        self.norm_spin.setRange(0.0, 100.0)
        self.norm_spin.setDecimals(1)
        self.norm_spin.setSingleStep(1.0)
        self.norm_spin.setSpecialValueText("無効")
        self.norm_spin.setPrefix("品質ノルム >= ")
        self.norm_spin.setValue(float(self.settings["report_min_norm"]))
        row.addWidget(self.norm_spin)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 1000)
        self.width_spin.setSpecialValueText("無効")
        self.width_spin.setPrefix("顔幅 >= ")
        self.width_spin.setSuffix(" px")
        self.width_spin.setValue(int(self.settings["report_min_face_width"]))
        row.addWidget(self.width_spin)
        row.addWidget(QLabel("（しきい値未満のクエリ画像を比較に使わない。0 で無効。目安 0.65 / 17 / 30）"))
        row.addStretch()
        root.addLayout(row)

        # 顔の採用基準（抽出時の顔ゲート・品質ガードをレポート時に再判定。クエリ・比較対象の両方に適用）
        self.gate_row = GateFilterRow(self.settings)
        root.addWidget(self.gate_row)

        # 探したい人物（該当画像をレポートでピンク表示）。5件分を表示し、残りは縦スクロール
        tbox = QGroupBox(f"探したい人物（最大{MAX_TARGETS}件・該当画像をピンク表示）")
        tlay = QVBoxLayout(tbox)
        head = QHBoxLayout()
        head.addStretch()
        btn_clear_all = QPushButton("全クリア")
        btn_clear_all.clicked.connect(self._clear_all_targets)
        head.addWidget(btn_clear_all)
        tlay.addLayout(head)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 0, 0)
        for c, h in enumerate(("", "カメラID", "映像ファイル名", "追跡ID", "")):
            grid.addWidget(QLabel(h), 0, c)
        saved_targets = list(self.settings["report_targets"])
        self.target_edits: list[tuple[QLineEdit, QLineEdit, QLineEdit]] = []
        for i in range(MAX_TARGETS):
            cam, vid, trk = QLineEdit(), QLineEdit(), QLineEdit()
            cam.setPlaceholderText("例: Cam1")
            vid.setPlaceholderText("例: 20260311122723.mp4")
            trk.setPlaceholderText("例: 3")
            if i < len(saved_targets) and len(saved_targets[i]) == 3:
                cam.setText(saved_targets[i][0]); vid.setText(saved_targets[i][1]); trk.setText(saved_targets[i][2])
            cam.editingFinished.connect(lambda k=i: self._split_pipe_values(k))  # フォーカス移動/Enterで確定
            btn = QPushButton("クリア")
            btn.clicked.connect(lambda _=False, k=i: self._clear_target_row(k))
            grid.addWidget(QLabel(f"{i + 1}"), i + 1, 0)
            for c, ed in enumerate((cam, vid, trk), start=1):
                grid.addWidget(ed, i + 1, c)
            grid.addWidget(btn, i + 1, 4)
            self.target_edits.append((cam, vid, trk))
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        row_h = grid.itemAtPosition(1, 1).widget().sizeHint().height() + grid.verticalSpacing()
        header_h = grid.itemAtPosition(0, 1).widget().sizeHint().height() + grid.verticalSpacing()
        scroll.setFixedHeight(header_h + row_h * TARGET_VISIBLE_ROWS + 2 * scroll.frameWidth() + 2)
        tlay.addWidget(scroll)
        root.addWidget(tbox)

        # モデル選択（レジストリへの登録でラジオが増える）
        self.model_group, self.model_buttons = self._model_box(
            root, "顔特徴量抽出モデル", visible_extractors(), self.settings["report_model"])
        self.landmark_group, self.landmark_buttons = self._model_box(
            root, "顔ランドマークモデル（特徴量抽出前のアライン）", ALIGNERS, self.settings["report_landmark"])
        self.pose_group, self.pose_buttons = self._model_box(
            root, "頭部姿勢（情報欄に yaw/pitch/roll を表示）", pose_registry("情報欄に yaw/pitch/roll を表示"), self.settings["report_pose"])

        root.addWidget(self.guide)
        brow = QHBoxLayout()
        brow.addWidget(self.btn_back)
        brow.addWidget(self.btn_start, 1)
        root.addLayout(brow)
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
        return fixed_label(text)

    def _row(self, parent: QVBoxLayout, label: str, edit: QLineEdit, slot) -> QPushButton:
        return add_path_row(parent, label, edit, slot)

    def _model_box(self, parent: QVBoxLayout, title: str, registry: dict, saved: str
                   ) -> tuple[QButtonGroup, dict[str, QRadioButton]]:
        """レジストリの各モデルをラジオボタンで並べる（widgets.add_model_box）。"""
        return add_model_box(self, parent, title, registry, saved, self._on_model_changed)

    def selected_model(self) -> str:
        return checked_name(self.model_buttons)

    def selected_landmark(self) -> str:
        return checked_name(self.landmark_buttons)

    def selected_pose(self) -> str:
        return checked_name(self.pose_buttons)

    # ---- モードと入力 ----
    def _on_mode_toggled(self) -> None:
        image_mode = self.rb_image.isChecked()
        self.query_edit.setEnabled(image_mode)
        self.track_edit.setEnabled(not image_mode)
        self.btn_track.setEnabled(not image_mode)
        self.track_info.setVisible(not image_mode)
        self.merge_label.setEnabled(not image_mode)
        self.merge_spin.setEnabled(not image_mode)

    def _set_query_image(self, path: str) -> None:
        p = Path(path)
        if not p.name.endswith("_face.jpg") or not p.is_file():
            QMessageBox.warning(self, "クエリ顔画像", "顔画像(_face.jpg)を指定してください。")
            return
        self.query_edit.setText(str(p))

    def _pick_query_image(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "クエリ顔画像を選択", self.query_edit.text(),
                                           "顔画像 (*_face.jpg)")
        if f:
            self._set_query_image(f)

    def _set_track_dir(self, path: str, quiet: bool = False) -> None:
        p = Path(path)
        if not p.is_dir():
            if not quiet:
                QMessageBox.warning(self, "追跡IDフォルダ", "フォルダを指定してください。")
            return
        entries = scan_track_folder(p)
        n = len(entries)
        if n == 0 and not quiet:
            QMessageBox.warning(self, "追跡IDフォルダ", "顔画像(_face.jpg)が見つかりません。追跡IDフォルダを指定してください。")
            return
        self.track_edit.setText(str(p))
        n_rej = sum(1 for e in entries if not e.gate_ok)
        self.track_info.setText(f"検出件数: 顔 {n}件" + (f"（うち抽出時の採用基準外 {n_rej}件）" if n_rej else ""))

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

    # ---- 重みの確認 ----
    def _on_model_changed(self, checked: bool = True) -> None:
        if checked:
            self._start_weights_check()

    def _start_weights_check(self) -> None:
        """選択中のモデルの重みを別スレッドで確認する（widgets.WeightsGuard）。"""
        self.weights.check(self.selected_model(), self.selected_landmark(), self.selected_pose())

    def _split_pipe_values(self, index: int) -> None:
        """カメラID欄に「|」区切りで入力された値を、カメラID|映像ファイル名|追跡ID に振り分ける。"""
        cam, vid, trk = self.target_edits[index]
        text = cam.text()
        if "|" not in text:
            return
        parts = [p.strip() for p in text.split("|")]
        cam.setText(parts[0])
        if len(parts) > 1:
            vid.setText(parts[1])
        if len(parts) > 2:
            trk.setText(parts[2])

    def _clear_target_row(self, index: int) -> None:
        for ed in self.target_edits[index]:
            ed.clear()

    def _clear_all_targets(self) -> None:
        for i in range(len(self.target_edits)):
            self._clear_target_row(i)

    def _targets(self) -> list[tuple[str, str, str]]:
        """入力のある行だけを (カメラID, 映像ファイル名, 追跡ID) で返す。"""
        out = []
        for cam, vid, trk in self.target_edits:
            row = (cam.text().strip(), vid.text().strip(), trk.text().strip())
            if any(row):
                out.append(row)
        return out

    # ---- レポート作成 ----
    def _on_start(self) -> None:
        image_mode = self.rb_image.isChecked()
        query = self.query_edit.text() if image_mode else self.track_edit.text()
        if not query:
            QMessageBox.warning(self, "レポート作成", "クエリ顔画像または追跡IDフォルダを指定してください。")
            return
        if not self.gallery_edit.text() or not self.out_edit.text():
            QMessageBox.warning(self, "レポート作成", "比較対象フォルダとレポート出力先フォルダを指定してください。")
            return
        self.settings.update(
            report_mode=MODE_IMAGE if image_mode else MODE_TRACK,
            report_query_path=self.query_edit.text(), report_track_dir=self.track_edit.text(),
            report_gallery_dir=self.gallery_edit.text(), report_output_dir=self.out_edit.text(),
            report_show_body=self.cb_body.isChecked(), report_include_old=self.cb_include_old.isChecked(),
            report_limit=self.limit_spin.value(), report_min_face_conf=self.conf_spin.value(),
            report_min_norm=self.norm_spin.value(), report_min_face_width=self.width_spin.value(),
            report_merge_sec=self.merge_spin.value(),
            report_targets=[[c.text().strip(), v.text().strip(), t.text().strip()]
                            for c, v, t in self.target_edits],
            report_model=self.selected_model(), report_landmark=self.selected_landmark(),
            report_pose=self.selected_pose())
        self.gate_row.save_to(self.settings)
        save_settings(self.settings)
        self.worker = ReportWorker(
            model_name=self.selected_model(), landmark_name=self.selected_landmark(),
            mode=MODE_IMAGE if image_mode else MODE_TRACK,
            query_path=Path(query), gallery_dir=Path(self.gallery_edit.text()),
            output_dir=Path(self.out_edit.text()), show_body=self.cb_body.isChecked(),
            limit=self.limit_spin.value(), targets=self._targets(),
            merge_sec=0.0 if image_mode else self.merge_spin.value(), pose_name=self.selected_pose(),
            include_excluded=self.cb_include_old.isChecked(),
            query_filter=QueryFilter(self.conf_spin.value(), self.norm_spin.value(), float(self.width_spin.value())),
            gate_filter=self.gate_row.value())
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
        QMessageBox.information(self, "レポート作成", f"レポートを出力しました。\n{path}")

    def _on_failed(self, msg: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText("失敗しました")
        QMessageBox.critical(self, "レポート作成", f"エラーが発生しました:\n{msg}")
