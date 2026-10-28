"""検索元セット検索画面。

顔画像出力先\\検索元\\<セット名> に人がコピーした顔画像（比較対象フォルダの _face.jpg のコピー。サブフォルダで分けてよい）を
1 枚ずつ検索元にして、総当たり（セット内どうしの類似度）と全体検索（比較対象全体での本人の順位）を 1 つの Excel に出す。
画像のコピーは Explorer で行う（「フォルダを開く」で開く）。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from ..core.common.settings import load_settings, save_settings
from ..core.features import visible_extractors
from ..core.landmarks import ALIGNERS
from ..core.report.gallery import QUERY_SETS_DIR_NAME
from ..core.report.query_set import (
    create_query_set, default_query_sets_root, list_query_sets, parse_track_lines, scan_query_images,
)
from .report_worker import QuerySetWorker
from .widgets import GateFilterRow, WeightsGuard, add_model_box, add_path_row, checked_name, fixed_label, pose_registry


class QuerySetScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.worker: QuerySetWorker | None = None

        root = QVBoxLayout(self)
        title = QLabel("検索元セット検索（総当たり・全体検索）")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)
        desc = QLabel(f"検索元セット = 「顔画像出力先\\{QUERY_SETS_DIR_NAME}\\<セット名>」に顔画像（比較対象フォルダの _face.jpg のコピー）を置いたフォルダ。"
                      "サブフォルダで分けられます（例: Cam4_79\\01_正面.jpg）。各画像を 1 枚ずつ検索元にして、セット内の総当たりの類似度と、"
                      "比較対象フォルダ全体で検索したときの本人の順位（画像単位・追跡ID単位）を 1 つの Excel に出します。"
                      "元画像は比較対象フォルダから自動で探し（同じ内容の画像）、その追跡IDを本人として扱います。")
        desc.setWordWrap(True)
        root.addWidget(desc)

        # 状態表示・ボタン（モデル選択の toggled から重み確認が走るので先に作り、レイアウトには最後に足す）
        self.guide = QPlainTextEdit()
        self.guide.setReadOnly(True)
        self.guide.setMaximumHeight(110)
        self.guide.hide()
        self.btn_start = QPushButton("検索開始")
        self.btn_back = QPushButton("戻る")
        self.status = QLabel("")
        self.weights = WeightsGuard(self.status, self.guide, self.btn_start, self.btn_back, lambda: self.worker is not None)

        # 顔画像出力先 → 検索元セットの置き場所
        self.faces_edit = QLineEdit(self.settings["output_root"])
        self.faces_edit.setReadOnly(True)
        add_path_row(root, "顔画像出力先:", self.faces_edit, self._pick_faces)
        self.root_label = QLabel("")
        root.addWidget(self.root_label)

        # セットの選択
        row = QHBoxLayout()
        row.addWidget(fixed_label("検索元セット:"))
        self.set_combo = QComboBox()
        self.set_combo.setMinimumWidth(320)
        row.addWidget(self.set_combo, 1)
        btn_refresh = QPushButton("更新")
        btn_new = QPushButton("新規作成...")
        btn_open = QPushButton("フォルダを開く")
        for b in (btn_refresh, btn_new, btn_open):
            row.addWidget(b)
        root.addLayout(row)
        self.set_info = QLabel("")
        self.set_info.setWordWrap(True)
        root.addWidget(self.set_info)

        self.gallery_edit = QLineEdit(self.settings["query_set_gallery_dir"] or self.settings["report_gallery_dir"]
                                      or self.settings["output_root"])
        self.gallery_edit.setReadOnly(True)
        add_path_row(root, "比較対象フォルダ:", self.gallery_edit, self._pick_gallery)
        self.out_edit = QLineEdit(self.settings["query_set_output_dir"] or self.settings["report_output_dir"]
                                  or self.settings["output_root"])
        self.out_edit.setReadOnly(True)
        add_path_row(root, "レポート出力先フォルダ:", self.out_edit, self._pick_output)

        # 本人の追跡ID（追加）
        row = QHBoxLayout()
        lb = fixed_label("本人の追跡ID（追加）:")
        row.addWidget(lb)
        self.persons_edit = QPlainTextEdit(self.settings["query_set_persons"])
        self.persons_edit.setMaximumHeight(70)
        self.persons_edit.setPlaceholderText("カメラID|映像ファイル名|追跡ID を 1 行 1 件（空欄の項目は不問）。"
                                             "検索元の元画像の追跡IDは自動で本人になるので、それ以外の本人の追跡IDだけ書く")
        row.addWidget(self.persons_edit, 1)
        root.addLayout(row)

        self.cb_include_old = QCheckBox("old フォルダの画像も含める（追跡IDフォルダ内の old に移した画像を、比較対象に元の追跡IDとして戻して比較）")
        self.cb_include_old.setChecked(bool(self.settings["report_include_old"]))
        root.addWidget(self.cb_include_old)
        self.gate_row = GateFilterRow(self.settings)
        root.addWidget(self.gate_row)

        # モデル選択（レジストリへの登録でラジオが増える）
        self.model_group, self.model_buttons = add_model_box(
            self, root, "顔特徴量抽出モデル", visible_extractors(), self.settings["report_model"], self._on_model_changed)
        self.landmark_group, self.landmark_buttons = add_model_box(
            self, root, "顔ランドマークモデル（特徴量抽出前のアライン）", ALIGNERS, self.settings["report_landmark"], self._on_model_changed)
        self.pose_group, self.pose_buttons = add_model_box(
            self, root, "頭部姿勢（検索元の yaw/pitch/roll を表示）", pose_registry("検索元の yaw/pitch/roll を表示"),
            self.settings["report_pose"], self._on_model_changed)

        root.addWidget(self.guide)
        brow = QHBoxLayout()
        brow.addWidget(self.btn_back)
        brow.addWidget(self.btn_start, 1)
        root.addLayout(brow)
        root.addWidget(self.status)
        root.addStretch()

        btn_refresh.clicked.connect(lambda: self._refresh_sets())
        btn_new.clicked.connect(self._new_set)
        btn_open.clicked.connect(self._open_set_dir)
        self.set_combo.currentIndexChanged.connect(self._on_set_changed)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_back.clicked.connect(self.back_requested.emit)

        self._refresh_sets(select=self.settings["query_set_name"])
        self._start_weights_check()

    # ---- UI部品 ----
    def selected_model(self) -> str:
        return checked_name(self.model_buttons)

    def selected_landmark(self) -> str:
        return checked_name(self.landmark_buttons)

    def selected_pose(self) -> str:
        return checked_name(self.pose_buttons)

    # ---- セット ----
    def sets_root(self) -> Path | None:
        return default_query_sets_root(self.faces_edit.text()) if self.faces_edit.text() else None

    def current_set_dir(self) -> Path | None:
        root = self.sets_root()
        name = self.set_combo.currentData()
        return root / name if root is not None and name else None

    def _refresh_sets(self, select: str | None = None) -> None:
        root = self.sets_root()
        self.root_label.setText(f"検索元セットの置き場所: {root}" if root is not None else "顔画像出力先を指定してください（抽出画面と同じフォルダ）")
        if select is None:
            select = self.set_combo.currentData() or ""
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for name, n in (list_query_sets(root) if root is not None else []):
            self.set_combo.addItem(f"{name}（画像 {n} 枚）", name)
        idx = self.set_combo.findData(select) if select else -1
        self.set_combo.setCurrentIndex(idx if idx >= 0 else (0 if self.set_combo.count() else -1))
        self.set_combo.blockSignals(False)
        self._on_set_changed()

    def _on_set_changed(self) -> None:
        d = self.current_set_dir()
        if d is None:
            self.set_info.setText("セットがありません。「新規作成...」でセットを作り、「フォルダを開く」で開いたフォルダに顔画像をコピーしてください。")
            return
        images = scan_query_images(d)
        groups = {g for g, _ in images}
        if not images:
            self.set_info.setText(f"画像がありません。「フォルダを開く」で開いたフォルダ（{d}）に、比較対象フォルダの顔画像（_face.jpg）を"
                                  "コピーしてください。サブフォルダで分けられます。")
        else:
            self.set_info.setText(f"画像 {len(images)} 枚 / グループ（サブフォルダ）{len(groups)} 件: {d}")

    def _new_set(self) -> None:
        root = self.sets_root()
        if root is None:
            QMessageBox.warning(self, "検索元セット", "先に顔画像出力先を指定してください。")
            return
        name, ok = QInputDialog.getText(self, "検索元セットの新規作成", f"セット名（{root} の下にフォルダを作ります）:")
        if not ok or not name.strip():
            return
        try:
            d = create_query_set(root, name)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "検索元セット", str(e))
            return
        self._refresh_sets(select=d.name)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _open_set_dir(self) -> None:
        d = self.current_set_dir() or self.sets_root()
        if d is None:
            QMessageBox.warning(self, "検索元セット", "先に顔画像出力先を指定してください。")
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "検索元セット", f"フォルダを作れません: {d}\n{e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    # ---- フォルダ選択 ----
    def _pick_faces(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "顔画像出力先を選択", self.faces_edit.text())
        if d:
            self.faces_edit.setText(d)
            self._refresh_sets()

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
        self.weights.check(self.selected_model(), self.selected_landmark(), self.selected_pose())

    # ---- 検索 ----
    def _on_start(self) -> None:
        set_dir = self.current_set_dir()
        if set_dir is None or not scan_query_images(set_dir):
            QMessageBox.warning(self, "検索元セット検索", "画像の入った検索元セットを選んでください。")
            return
        if not self.gallery_edit.text() or not self.out_edit.text():
            QMessageBox.warning(self, "検索元セット検索", "比較対象フォルダとレポート出力先フォルダを指定してください。")
            return
        self.settings.update(
            output_root=self.faces_edit.text(), query_set_name=set_dir.name,
            query_set_gallery_dir=self.gallery_edit.text(), query_set_output_dir=self.out_edit.text(),
            query_set_persons=self.persons_edit.toPlainText(), report_include_old=self.cb_include_old.isChecked(),
            report_model=self.selected_model(), report_landmark=self.selected_landmark(), report_pose=self.selected_pose())
        self.gate_row.save_to(self.settings)
        save_settings(self.settings)
        self.worker = QuerySetWorker(
            set_dir=set_dir, gallery_dir=Path(self.gallery_edit.text()), output_dir=Path(self.out_edit.text()),
            model_name=self.selected_model(), landmark_name=self.selected_landmark(), pose_name=self.selected_pose(),
            include_excluded=self.cb_include_old.isChecked(), gate_filter=self.gate_row.value(),
            own_specs=parse_track_lines(self.persons_edit.toPlainText()))
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
        QMessageBox.information(self, "検索元セット検索", f"Excel を出力しました。\n{path}")

    def _on_failed(self, msg: str) -> None:
        self.worker = None
        self._set_running(False)
        self.status.setText("失敗しました")
        QMessageBox.critical(self, "検索元セット検索", f"エラーが発生しました:\n{msg}")
