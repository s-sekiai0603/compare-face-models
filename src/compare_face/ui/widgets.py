"""レポート画面と検索元セット画面で共用する部品（入力欄・顔の採用基準の行・モデル選択・重みの確認）。"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.pose import POSE_OPTIONS, get_pose_estimator
from ..core.report.gate import GateFilter
from .report_worker import WeightsWorker

LABEL_WIDTH = 150


def fixed_label(text: str, width: int = LABEL_WIDTH) -> QLabel:
    lb = QLabel(text)
    lb.setFixedWidth(width)
    return lb


def add_path_row(parent: QVBoxLayout, label: str, edit: QLineEdit, slot) -> QPushButton:
    """「ラベル + 入力欄 + 参照...」の 1 行。戻り値は参照ボタン。"""
    row = QHBoxLayout()
    row.addWidget(fixed_label(label))
    row.addWidget(edit, 1)
    btn = QPushButton("参照...")
    btn.clicked.connect(slot)
    row.addWidget(btn)
    parent.addLayout(row)
    return btn


def pose_registry(note: str) -> dict:
    """頭部姿勢の選択肢を add_model_box に渡すレジストリ形にする（説明はツールチップ。生成だけで重みは読まない）。"""
    def description(name: str) -> str:
        est = get_pose_estimator(name)
        return "頭部姿勢を表示しない" if est is None else f"{est.name} の重み {est.weights_path().name}（{note}）"

    return {name: type("_Pose", (), {"description": description(name)}) for name in POSE_OPTIONS}


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


class GateFilterRow(QWidget):
    """顔の採用基準（抽出時の顔ゲート・品質ガードをレポート時に再判定）の入力行。設定キーはレポート画面と共通。"""

    def __init__(self, settings: dict):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(fixed_label("顔の採用基準:"))
        self.cb_face_gate = QCheckBox("骨格ルール（鼻+片目が見える / 下向きでない）")
        self.cb_face_gate.setChecked(bool(settings["report_face_gate"]))
        row.addWidget(self.cb_face_gate)
        self.eye_ratio_spin = QDoubleSpinBox()
        self.eye_ratio_spin.setRange(0.0, 1.0)
        self.eye_ratio_spin.setDecimals(2)
        self.eye_ratio_spin.setSingleStep(0.01)
        self.eye_ratio_spin.setSpecialValueText("判定なし")
        self.eye_ratio_spin.setPrefix("両目間隔比 >= ")
        self.eye_ratio_spin.setValue(float(settings["report_min_eye_ratio"]))
        row.addWidget(self.eye_ratio_spin)
        self.crop_side_spin = QSpinBox()
        self.crop_side_spin.setRange(0, 1000)
        self.crop_side_spin.setSpecialValueText("判定なし")
        self.crop_side_spin.setPrefix("切り出し短辺 >= ")
        self.crop_side_spin.setSuffix(" px")
        self.crop_side_spin.setValue(int(settings["report_min_crop_side"]))
        row.addWidget(self.crop_side_spin)
        row.addWidget(QLabel("（抽出時の基準をクエリ・比較対象に適用。既定は抽出時と同じ。緩めると抽出時に基準外だった画像も使う）"))
        row.addStretch()

    def value(self) -> GateFilter:
        return GateFilter(self.cb_face_gate.isChecked(), self.eye_ratio_spin.value(), float(self.crop_side_spin.value()))

    def save_to(self, settings: dict) -> None:
        settings.update(report_face_gate=self.cb_face_gate.isChecked(), report_min_eye_ratio=self.eye_ratio_spin.value(),
                        report_min_crop_side=self.crop_side_spin.value())


def add_model_box(owner: QWidget, parent: QVBoxLayout, title: str, registry: dict, saved: str,
                  on_toggled: Callable[[bool], None] | None = None) -> tuple[QButtonGroup, dict[str, QRadioButton]]:
    """レジストリの各モデルをラジオボタンで並べる。説明はツールチップに表示する。"""
    box = QGroupBox(title)
    bl = QHBoxLayout(box)
    group = QButtonGroup(owner)
    buttons: dict[str, QRadioButton] = {}
    for i, (name, cls) in enumerate(registry.items()):
        rb = QRadioButton(name)
        rb.setToolTip(getattr(cls, "description", "") or name)
        rb.setChecked(name == saved or (i == 0 and saved not in registry))
        group.addButton(rb)
        buttons[name] = rb
        bl.addWidget(rb)
        if on_toggled is not None:
            rb.toggled.connect(on_toggled)
    bl.addStretch()
    parent.addWidget(box)
    return group, buttons


def checked_name(buttons: dict[str, QRadioButton]) -> str:
    return next(n for n, b in buttons.items() if b.isChecked())


class WeightsGuard(QObject):
    """選択中のモデルの重みを別スレッドで確認（無ければ自動取得）し、開始/戻るボタンの有効・無効を切り替える。
    確認中にモデルを切り替えた場合は新しいスレッドを起動し、古いスレッドの結果は無視する
    （古いスレッドは終了まで保持し、実行中の QThread が破棄されないようにする）。"""

    def __init__(self, status: QLabel, guide: QPlainTextEdit, btn_start: QPushButton, btn_back: QPushButton,
                 is_running: Callable[[], bool]):
        super().__init__()
        self.status, self.guide, self.btn_start, self.btn_back, self.is_running = status, guide, btn_start, btn_back, is_running
        self.ok = False
        self.current: WeightsWorker | None = None       # 最新の確認スレッド（結果を採用するのはこれだけ）
        self._workers: list[WeightsWorker] = []         # 実行中のスレッドを終了まで保持する

    def check(self, model_name: str, landmark_name: str, pose_name: str) -> None:
        self.ok = False
        self.guide.hide()
        self.btn_start.setEnabled(False)
        self.btn_back.setEnabled(False)  # 確認スレッド実行中は戻れない
        self.status.setText("モデルの重みを確認しています...")
        worker = WeightsWorker(model_name, landmark_name, pose_name)
        worker.status.connect(lambda msg, w=worker: self._on_status(w, msg))
        worker.ready.connect(lambda w=worker: self._on_ready(w))
        worker.unavailable.connect(lambda msg, w=worker: self._on_unavailable(w, msg))
        worker.finished.connect(lambda w=worker: self._on_finished(w))
        self._workers.append(worker)
        self.current = worker
        worker.start()

    def _on_status(self, w: WeightsWorker, msg: str) -> None:
        if w is self.current:
            self.status.setText(msg)

    def _on_ready(self, w: WeightsWorker) -> None:
        if w is not self.current:
            return
        self.ok = True
        self.status.setText("")
        self.btn_start.setEnabled(not self.is_running())

    def _on_unavailable(self, w: WeightsWorker, msg: str) -> None:
        if w is not self.current:
            return
        self.status.setText("モデルの重みが見つかりません。下記の手順で配置してください。")
        self.guide.setPlainText(msg)
        self.guide.show()

    def _on_finished(self, w: WeightsWorker) -> None:
        if w in self._workers:
            self._workers.remove(w)
        w.deleteLater()
        if not self._workers:  # 古い確認スレッド（取得中など）も含めて全て終わったら戻れる
            self.btn_back.setEnabled(not self.is_running())
