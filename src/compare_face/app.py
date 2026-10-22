from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from .ui.extract_screen import ExtractScreen
from .ui.report_screen import ReportScreen
from .ui.top_screen import TopScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("顔識別モデル比較ツール (compare-face-models)")
        self.resize(960, 720)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.top = TopScreen()
        self.stack.addWidget(self.top)
        self.top.extract_requested.connect(self.show_extract)
        self.top.report_requested.connect(self.show_report)

    def show_extract(self) -> None:
        # 画面を開くたびに生成し、前回保存した値を既定表示にする
        screen = ExtractScreen()
        screen.back_requested.connect(lambda: self._back(screen))
        self.stack.addWidget(screen)
        self.stack.setCurrentWidget(screen)

    def show_report(self) -> None:
        screen = ReportScreen()
        screen.back_requested.connect(lambda: self._back(screen))
        self.stack.addWidget(screen)
        self.stack.setCurrentWidget(screen)

    def _back(self, screen) -> None:
        self.stack.setCurrentWidget(self.top)
        self.stack.removeWidget(screen)
        screen.deleteLater()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()
