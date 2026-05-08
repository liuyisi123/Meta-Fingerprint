"""Meta-Fingerprint Monitor entry point."""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure package root is importable
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# Add the model source when the monitor is stored inside this repository or
# when it is kept next to a cloned Meta-Fingerprint repository.
_REPO_SRC_CANDIDATES = [
    _ROOT.parent / "src",
    _ROOT.parent / "meta_fingerprint_repo" / "meta_fingerprint_repo" / "src",
]
for _repo_src in _REPO_SRC_CANDIDATES:
    if _repo_src.exists() and str(_repo_src) not in sys.path:
        sys.path.insert(0, str(_repo_src))
        break

from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient


def _make_splash(app: QApplication) -> QSplashScreen:
    """Create a programmatic splash screen."""
    w, h = 580, 280
    pix = QPixmap(w, h)
    pix.fill(QColor("#060E1A"))

    painter = QPainter(pix)
    # Gradient top bar
    grad = QLinearGradient(0, 0, w, 0)
    grad.setColorAt(0, QColor("#0088CC"))
    grad.setColorAt(1, QColor("#00C8F0"))
    painter.fillRect(0, 0, w, 5, grad)

    # Logo text
    painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
    painter.setPen(QColor("#00C8F0"))
    painter.drawText(40, 90, "Meta-Fingerprint")

    painter.setFont(QFont("Segoe UI", 13))
    painter.setPen(QColor("#7AAFCF"))
    painter.drawText(44, 125, "Hemodynamic Intelligence Platform")

    painter.setFont(QFont("Segoe UI", 10))
    painter.setPen(QColor("#3A6A8A"))
    painter.drawText(44, 175, "Physics-Grounded Vascular Disentanglement")
    painter.drawText(44, 195, "Cross-Domain Hemodynamic Monitoring")

    # Gradient bottom bar
    painter.fillRect(0, h - 5, w, 5, grad)

    # Version
    painter.setFont(QFont("Segoe UI", 9))
    painter.setPen(QColor("#3A6A8A"))
    painter.drawText(w - 120, h - 18, "v1.0  |  Research Preview")

    painter.end()

    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    splash.setFont(QFont("Segoe UI", 9))
    return splash


def main() -> None:
    # High-DPI support
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("Meta-Fingerprint Monitor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("MetaFingerprint Research")

    # Splash
    splash = _make_splash(app)
    splash.show()
    splash.showMessage("  Initialising...", Qt.AlignBottom | Qt.AlignLeft, QColor("#7AAFCF"))
    app.processEvents()

    # Deferred import to allow splash rendering
    from gui.main_window import MainWindow
    splash.showMessage("  Loading interface...", Qt.AlignBottom | Qt.AlignLeft, QColor("#7AAFCF"))
    app.processEvents()

    window = MainWindow()

    def _show():
        splash.finish(window)
        window.show()

    QTimer.singleShot(1200, _show)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
