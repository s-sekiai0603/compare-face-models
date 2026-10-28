from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class TopScreen(QWidget):
    extract_requested = Signal()
    report_requested = Signal()
    query_set_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch()
        title = QLabel("顔識別モデル比較ツール")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        btn_extract = QPushButton("顔画像抽出")
        btn_report = QPushButton("レポート出力")
        btn_query_set = QPushButton("検索元セット検索（総当たり・全体検索）")
        for b in (btn_extract, btn_report, btn_query_set):
            b.setMinimumHeight(48)
            layout.addWidget(b)
        layout.addStretch()
        btn_extract.clicked.connect(self.extract_requested.emit)
        btn_report.clicked.connect(self.report_requested.emit)
        btn_query_set.clicked.connect(self.query_set_requested.emit)
