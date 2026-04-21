from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from quantify_trader.app.main_window import MainWindow


def run_app() -> None:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    raise SystemExit(app.exec())

