"""距離・向き・精度分析画面。

顔サイズ別精度検証の Excel レポートを指定して「チャート生成」を押すと、「比較結果誤識別数まとめ」シートの内容から
「距離向き精度分析チャート」シートを同じ Excel に追加する（core/report/distance_pose_chart.py）。
最後に選んだ Excel は distance_pose_chart_screen_setting.json に保存し、次回起動時に復元する。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ..core.common.settings import load_distance_pose_chart_settings, save_distance_pose_chart_settings
from ..core.report.distance_pose_chart import (
    CHART_SHEET_NAME, COLOR_GROUPS, ICON_DIR, add_chart_sheet, color_group_rgb, icon_choices, list_cameras,
)
from .widgets import add_path_row, fixed_label

TITLE = "距離・向き・精度分析"


class DistancePoseChartScreen(QWidget):
    back_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = load_distance_pose_chart_settings()

        root = QVBoxLayout(self)
        title = QLabel(TITLE)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)
        root.addWidget(QLabel(
            f"顔サイズ別精度検証で出力した Excel の「比較結果誤識別数まとめ」シートから、人物距離・顔の向き・誤識別数のチャートを作り、\n"
            f"同じ Excel に「{CHART_SHEET_NAME}」シートとして追加する（既にあれば置き換える。チャートは Python で描いた画像を貼り付ける）。事前に「対象画像情報」シートの「顔の向き」を入力しておく。"))

        self.excel_edit = QLineEdit(self.settings["excel_path"])
        self.excel_edit.setReadOnly(True)
        add_path_row(root, "Excel ファイル:", self.excel_edit, self._pick)

        crow = QHBoxLayout()
        crow.addWidget(fixed_label("出力対象:"))
        self.cb_all = QCheckBox("すべてのカメラ")
        self.cb_all.setChecked(self.settings["all_cameras"] is not False)
        self.cb_all.setToolTip("すべてのカメラの画像を 1 つのチャートにプロットする")
        self.cb_each = QCheckBox("カメラ個別")
        self.cb_each.setChecked(self.settings["per_camera"] is True)
        self.cb_each.setToolTip("カメラIDごとに 1 つずつチャートをプロットする（縦・横の両方がそのカメラの画像である組合せだけ）。"
                                "両方オンならカメラ数 + 1 個のチャートを同じシートに縦に並べる")
        crow.addWidget(self.cb_all)
        crow.addWidget(self.cb_each)
        crow.addStretch()
        root.addLayout(crow)

        camrow = QHBoxLayout()
        camrow.addWidget(fixed_label("対象カメラ:"))
        self.cam_list = QListWidget()
        self.cam_list.setMaximumHeight(90)
        self.cam_list.setToolTip("チャートに使うカメラID（チェックしたものだけ）。「すべてのカメラ」「カメラ個別」のどちらも、ここで選んだカメラの画像だけを対象にする")
        camrow.addWidget(self.cam_list, 1)
        side = QVBoxLayout()
        self.btn_cam_all = QPushButton("全選択")
        self.btn_cam_none = QPushButton("全解除")
        side.addWidget(self.btn_cam_all)
        side.addWidget(self.btn_cam_none)
        side.addStretch()
        camrow.addLayout(side)
        root.addLayout(camrow)
        self._load_cameras()

        # 出力対象の色（誤識別数の 7 段階）・アイコン（顔の向きの 15 組合せ）: 複数選択、全選択・全解除
        excluded_colors = set(self.settings.get("excluded_colors") or [])
        excluded_icons = set(self.settings.get("excluded_icons") or [])
        self.color_list = self._make_list(
            "対象の色:", "チャートに出す色（誤識別数の段階）。チェックを外した色のアイコンはプロットしない", 130,
            [(label, g, self._swatch(color_group_rgb(g)), g not in excluded_colors) for g, (label, _, _) in enumerate(COLOR_GROUPS)], root)
        self.icon_list = self._make_list(
            "対象のアイコン:", "チャートに出すアイコン（顔の向きの組合せ: 縦 × 横）。チェックを外したアイコンはプロットしない", 130,
            [(label, fname, self._icon(fname), fname not in excluded_icons) for label, fname in icon_choices()], root)

        self.cb_split = QCheckBox("顔の向きをチャートの左右に振り分け")
        self.cb_split.setChecked(self.settings["split_lr"] is True)
        self.cb_split.setToolTip(
            "横軸を中心 0 m として左右に距離をとる。横に並ぶ画像の顔の向きに「右」を含めば中心から右側、「左」を含めば左側、"
            "どちらも含まない（正面）なら右側と左側の両方にプロットする")
        root.addWidget(self.cb_split)

        urow = QHBoxLayout()
        urow.addWidget(fixed_label("距離の集計単位:"))
        self.unit_spin = QDoubleSpinBox()
        self.unit_spin.setRange(0.0, 100.0)
        self.unit_spin.setDecimals(1)
        self.unit_spin.setSingleStep(0.5)
        self.unit_spin.setSuffix(" m（0 = 集計しない）")
        try:
            self.unit_spin.setValue(max(0.0, float(self.settings["unit_m"])))
        except (TypeError, ValueError):
            pass
        self.unit_spin.setToolTip(
            "プロットするアイコンが多いときに、縦・横それぞれ「顔の向き × 距離の区間（この単位）」ごとに集計して 1 つにまとめる。"
            "アイコンの色は誤識別数のセルの値の平均、位置は距離の平均で決める")
        urow.addWidget(self.unit_spin)
        urow.addStretch()
        root.addLayout(urow)

        self.btn_start = QPushButton("チャート生成")
        self.btn_back = QPushButton("戻る")
        brow = QHBoxLayout()
        brow.addWidget(self.btn_back)
        brow.addWidget(self.btn_start, 1)
        root.addLayout(brow)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_back.clicked.connect(self.back_requested.emit)
        self.btn_cam_all.clicked.connect(lambda: self._set_all_cameras(True))
        self.btn_cam_none.clicked.connect(lambda: self._set_all_cameras(False))

    def _pick(self) -> None:
        start = self.excel_edit.text() or ""
        path, _ = QFileDialog.getOpenFileName(self, "Excel ファイルを選択", start, "Excel (*.xlsx)")
        if path:
            self.excel_edit.setText(path)
            self.settings["excel_path"] = path
            save_distance_pose_chart_settings(self.settings)
            self._load_cameras()

    @staticmethod
    def _swatch(rgb) -> QIcon:
        pm = QPixmap(20, 20)
        pm.fill(QColor(*rgb))
        return QIcon(pm)

    @staticmethod
    def _icon(fname: str) -> QIcon:
        return QIcon(str(ICON_DIR / fname))

    def _make_list(self, label: str, tip: str, height: int, items: list, root: QVBoxLayout) -> QListWidget:
        """チェック付きの一覧 + 全選択・全解除ボタンの行を root に足す。items = [(表示名, 値, アイコン, チェック済みか)]。"""
        row = QHBoxLayout()
        row.addWidget(fixed_label(label))
        lw = QListWidget()
        lw.setMaximumHeight(height)
        lw.setToolTip(tip)
        for text, value, ico, on in items:
            it = QListWidgetItem(ico, text)
            it.setData(Qt.UserRole, value)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            lw.addItem(it)
        row.addWidget(lw, 1)
        side = QVBoxLayout()
        b_all, b_none = QPushButton("全選択"), QPushButton("全解除")
        b_all.clicked.connect(lambda: self._set_all(lw, True))
        b_none.clicked.connect(lambda: self._set_all(lw, False))
        side.addWidget(b_all)
        side.addWidget(b_none)
        side.addStretch()
        row.addLayout(side)
        root.addLayout(row)
        return lw

    @staticmethod
    def _set_all(lw: QListWidget, on: bool) -> None:
        for k in range(lw.count()):
            lw.item(k).setCheckState(Qt.Checked if on else Qt.Unchecked)

    @staticmethod
    def _checked(lw: QListWidget) -> tuple[list, list]:
        """(チェックした値, 外した値)"""
        on, off = [], []
        for k in range(lw.count()):
            it = lw.item(k)
            (on if it.checkState() == Qt.Checked else off).append(it.data(Qt.UserRole))
        return on, off

    def _load_cameras(self) -> None:
        """指定した Excel に出てくるカメラIDをリストに並べる（前回チェックを外したカメラ以外はチェック済み）。"""
        self.cam_list.clear()
        path = self.excel_edit.text()
        if not path or not Path(path).is_file():
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cams = list_cameras(path)
        except Exception as e:  # レイアウト違い等。チャート生成時に同じエラーが出るのでここでは空にするだけ
            self.log.appendPlainText(f"カメラIDを読み取れませんでした: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        excluded = set(self.settings.get("excluded_cameras") or [])
        for cam in cams:
            item = QListWidgetItem(cam)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if cam in excluded else Qt.Checked)
            self.cam_list.addItem(item)

    def _set_all_cameras(self, on: bool) -> None:
        for k in range(self.cam_list.count()):
            self.cam_list.item(k).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _camera_selection(self) -> tuple[list[str], list[str]]:
        """(チェックしたカメラID, 外したカメラID)"""
        on, off = [], []
        for k in range(self.cam_list.count()):
            it = self.cam_list.item(k)
            (on if it.checkState() == Qt.Checked else off).append(it.text())
        return on, off

    def _on_start(self) -> None:
        path = self.excel_edit.text()
        if not path:
            QMessageBox.warning(self, TITLE, "Excel ファイルを指定してください。")
            return
        if not Path(path).is_file():
            QMessageBox.warning(self, TITLE, "Excel ファイルが見つかりません。")
            return
        all_cams, each = self.cb_all.isChecked(), self.cb_each.isChecked()
        if not all_cams and not each:
            QMessageBox.warning(self, TITLE, "出力対象（すべてのカメラ / カメラ個別）を 1 つ以上選んでください。")
            return
        cams_on, cams_off = self._camera_selection()
        if self.cam_list.count() and not cams_on:
            QMessageBox.warning(self, TITLE, "対象カメラを 1 つ以上選んでください。")
            return
        colors_on, colors_off = self._checked(self.color_list)
        icons_on, icons_off = self._checked(self.icon_list)
        if not colors_on:
            QMessageBox.warning(self, TITLE, "対象の色を 1 つ以上選んでください。")
            return
        if not icons_on:
            QMessageBox.warning(self, TITLE, "対象のアイコンを 1 つ以上選んでください。")
            return
        split = self.cb_split.isChecked()
        unit = round(self.unit_spin.value(), 1)
        self.settings.update(excel_path=path, split_lr=split, unit_m=unit, all_cameras=all_cams, per_camera=each, excluded_cameras=cams_off,
                             excluded_colors=colors_off, excluded_icons=icons_off)
        save_distance_pose_chart_settings(self.settings)
        self.log.clear()
        self.log.appendPlainText("チャートを生成しています...")
        self.btn_start.setEnabled(False)
        self.btn_back.setEnabled(False)
        try:
            data = add_chart_sheet(path, split_lr=split, unit_m=unit, all_cameras=all_cams, per_camera=each,
                                  cameras=cams_on if self.cam_list.count() else None,
                                  colors=None if not colors_off else colors_on, icon_files=None if not icons_off else icons_on)
        except Exception as e:  # 入力 Excel の不備・書き込み失敗（Excel で開いている等）を画面に出す
            self.log.appendPlainText(f"失敗: {e}")
            QMessageBox.critical(self, TITLE, f"チャートを生成できませんでした:\n{e}")
            return
        finally:
            self.btn_start.setEnabled(True)
            self.btn_back.setEnabled(True)
        lines = [f"完了: 「{CHART_SHEET_NAME}」シートを追加しました（アイコン {data.n_icons} 個 / 画像 {data.n_images} 枚{'。左右に振り分け' if split else ''}{f'。距離 {unit:g} m 単位で集計' if unit > 0 else ''}）。"]
        lines.append(f"チャート {len(data.charts)} 個: " + "、".join(f"{label}（{n} 個）" for label, n in data.charts))
        if data.no_distance:
            lines.append(f"距離が無い画像 {data.no_distance} 枚は対象外にしました。")
        if data.no_direction:
            lines.append(f"顔の向きが未入力・不正な画像 {data.no_direction} 枚は対象外にしました（正面 / 左横 / 右横 / 左斜め正面 / 右斜め正面）。")
        self.log.appendPlainText("\n".join(lines))
        QMessageBox.information(self, TITLE, "\n".join(lines))
