from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class TopScreen(QWidget):
    extract_requested = Signal()
    report_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch()
        title = QLabel("外見識別(reid) 検証ツール")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        btn_extract = QPushButton("画像抽出")
        btn_report = QPushButton("レポート出力")
        for b in (btn_extract, btn_report):
            b.setMinimumHeight(48)
            layout.addWidget(b)
        layout.addStretch()
        btn_extract.clicked.connect(self.extract_requested.emit)
        btn_report.clicked.connect(self.report_requested.emit)
