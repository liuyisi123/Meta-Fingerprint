"""Main application window with animated sidebar."""
from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame,
    QStatusBar, QSizePolicy, QApplication, QFileDialog,
    QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize, pyqtSignal as Signal, QThread, QObject
from PyQt5.QtGui import QFont, QIcon, QColor

from .style import QSS, PALETTE
from .widgets.dashboard    import DashboardWidget
from .widgets.signal_viewer import SignalViewerWidget
from .widgets.analysis     import AnalysisWidget
from .widgets.patient_mgr  import PatientManagerWidget
from .widgets.lan_panel    import LANPanelWidget
from .widgets.report_panel import ReportPanelWidget
from .widgets.settings_panel import SettingsWidget

from core.inference  import InferenceEngine
from core.database   import PatientDB
from core.lan_server import LANServer


NAV_ITEMS = [
    ("⬟",  "Dashboard",    "Overview and live statistics"),
    ("◈",  "Monitor",      "Real-time ECG/PPG/ABP display"),
    ("⊞",  "Analysis",     "Batch inference on NPZ files"),
    ("◉",  "Patients",     "Patient records and history"),
    ("⋄",  "LAN",          "Network data reception"),
    ("▣",  "Reports",      "Generate and export reports"),
    ("⚙",  "Settings",     "Configuration and calibration"),
]


class SidebarButton(QPushButton):
    def __init__(self, icon: str, label: str, tooltip: str, parent=None):
        super().__init__(parent)
        self._icon_text = icon
        self._label = label
        self.setCheckable(True)
        self.setToolTip(tooltip)
        self._expanded = True
        self._update_text()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._update_text()

    def _update_text(self) -> None:
        if self._expanded:
            self.setText(f"  {self._icon_text}  {self._label}")
        else:
            self.setText(f"  {self._icon_text}")


class ModelLoadWorker(QObject):
    done  = Signal(bool, str)

    def __init__(self, engine: InferenceEngine, path: str):
        super().__init__()
        self.engine = engine
        self.path = path

    def run(self):
        ok = self.engine.load_model(self.path)
        self.done.emit(ok, self.path if ok else "")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Meta-Fingerprint Monitor  |  Hemodynamic Intelligence Platform")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)

        # Core services
        self.engine   = InferenceEngine()
        self.db       = PatientDB()
        self.lan_srv  = LANServer()
        self._sidebar_expanded = True
        self._load_thread: QThread | None = None
        self._load_worker: ModelLoadWorker | None = None

        self._build_ui()
        self._connect_signals()
        self._start_clock()
        self._navigate(0)

    # ── UI construction ────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ──
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(230)
        self.sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(8, 16, 8, 16)
        sb_layout.setSpacing(4)

        # Logo
        logo_container = QWidget()
        logo_lyt = QVBoxLayout(logo_container)
        logo_lyt.setContentsMargins(12, 0, 0, 12)
        logo_label = QLabel("◈  META-FP")
        logo_label.setObjectName("logo_label")
        ver_label = QLabel("v1.0  |  Research Preview")
        ver_label.setObjectName("version_label")
        logo_lyt.addWidget(logo_label)
        logo_lyt.addWidget(ver_label)
        sb_layout.addWidget(logo_container)

        sep = QFrame(); sep.setObjectName("separator")
        sep.setMaximumHeight(1); sep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sb_layout.addWidget(sep)
        sb_layout.addSpacing(8)

        # Nav buttons
        self._nav_buttons: list[SidebarButton] = []
        for icon, label, tip in NAV_ITEMS:
            btn = SidebarButton(icon, label, tip)
            sb_layout.addWidget(btn)
            self._nav_buttons.append(btn)

        sb_layout.addStretch()

        sep2 = QFrame(); sep2.setObjectName("separator")
        sep2.setMaximumHeight(1); sep2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sb_layout.addWidget(sep2)
        sb_layout.addSpacing(4)

        # Bottom controls
        self._model_btn = QPushButton("  ⊡  Load Model")
        self._model_btn.setObjectName("btn_secondary")
        self._model_btn.setFixedHeight(36)
        sb_layout.addWidget(self._model_btn)

        self._collapse_btn = QPushButton("  ◁  Collapse")
        self._collapse_btn.setObjectName("btn_icon")
        self._collapse_btn.setFixedHeight(32)
        sb_layout.addWidget(self._collapse_btn)

        root.addWidget(self.sidebar)

        # ── Main area ──
        main_area = QWidget()
        main_lyt = QVBoxLayout(main_area)
        main_lyt.setContentsMargins(0, 0, 0, 0)
        main_lyt.setSpacing(0)

        # Topbar
        self.topbar = QFrame()
        self.topbar.setObjectName("topbar")
        self.topbar.setFixedHeight(52)
        tb_lyt = QHBoxLayout(self.topbar)
        tb_lyt.setContentsMargins(20, 0, 20, 0)

        self._page_title = QLabel("Dashboard")
        self._page_title.setObjectName("page_title")
        tb_lyt.addWidget(self._page_title)
        tb_lyt.addStretch()

        self._model_status = QLabel("● Model: Demo Mode")
        self._model_status.setStyleSheet(f"color: {PALETTE['warning']}; font-size: 11px;")
        tb_lyt.addWidget(self._model_status)

        tb_lyt.addSpacing(16)
        self._lan_status = QLabel("◯ LAN: Offline")
        self._lan_status.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 11px;")
        tb_lyt.addWidget(self._lan_status)

        tb_lyt.addSpacing(16)
        self._clock = QLabel()
        self._clock.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px; font-family: monospace;")
        tb_lyt.addWidget(self._clock)

        main_lyt.addWidget(self.topbar)

        # Pages
        self._pages = QStackedWidget()
        self._page_widgets = [
            DashboardWidget(self.db, self.engine),
            SignalViewerWidget(self.engine),
            AnalysisWidget(self.engine, self.db),
            PatientManagerWidget(self.db),
            LANPanelWidget(self.lan_srv, self.engine),
            ReportPanelWidget(self.db, self.engine),
            SettingsWidget(self.engine),
        ]
        for w in self._page_widgets:
            self._pages.addWidget(w)
        main_lyt.addWidget(self._pages)

        root.addWidget(main_area)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready  |  Meta-Fingerprint Monitor  |  Research Use Only")

        self.setStyleSheet(QSS)

    # ── Signals ────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.clicked.connect(lambda checked, idx=i: self._navigate(idx))
        self._collapse_btn.clicked.connect(self._toggle_sidebar)
        self._model_btn.clicked.connect(self._load_model_dialog)

        # LAN callbacks → status bar update
        self.lan_srv.on_client_connect    = lambda cid, addr: self._on_lan_event(f"Connected: {cid}")
        self.lan_srv.on_client_disconnect = lambda cid: self._on_lan_event(f"Disconnected: {cid}")
        self.lan_srv.on_data              = self._on_lan_data

        # Propagate LAN server to LAN panel
        lan_panel: LANPanelWidget = self._page_widgets[4]
        lan_panel.server_started.connect(self._update_lan_status)
        lan_panel.server_stopped.connect(self._update_lan_status)

    # ── Navigation ─────────────────────────────────────────────
    def _navigate(self, idx: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == idx)
        self._pages.setCurrentIndex(idx)
        self._page_title.setText(NAV_ITEMS[idx][1])
        # Trigger page-specific refresh
        page = self._page_widgets[idx]
        if hasattr(page, "refresh"):
            page.refresh()

    # ── Sidebar collapse ────────────────────────────────────────
    def _toggle_sidebar(self) -> None:
        self._sidebar_expanded = not self._sidebar_expanded
        target_w = 230 if self._sidebar_expanded else 60
        anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
        anim.setDuration(180)
        anim.setStartValue(self.sidebar.width())
        anim.setEndValue(target_w)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.start()
        self._anim = anim  # keep reference
        self.sidebar.setMaximumWidth(target_w)
        for btn in self._nav_buttons:
            btn.set_expanded(self._sidebar_expanded)
        self._collapse_btn.setText("  ◁  Collapse" if self._sidebar_expanded else "  ▷")

    # ── Model loading ───────────────────────────────────────────
    def _load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model Checkpoint", "",
            "Checkpoint (*.pt *.pth *.ckpt);;All Files (*)"
        )
        if not path:
            return
        self._model_status.setText("● Loading...")
        self._model_status.setStyleSheet(f"color: {PALETTE['warning']}; font-size: 11px;")
        self._load_thread = QThread()
        self._load_worker = ModelLoadWorker(self.engine, path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.done.connect(self._on_model_loaded)
        self._load_worker.done.connect(self._load_thread.quit)
        self._load_thread.start()

    def _on_model_loaded(self, ok: bool, path: str) -> None:
        if ok:
            self._model_status.setText(f"● Model: Loaded")
            self._model_status.setStyleSheet(f"color: {PALETTE['success']}; font-size: 11px;")
            self.status_bar.showMessage(f"Model loaded: {path}")
        else:
            self._model_status.setText("● Model: Demo Mode")
            self._model_status.setStyleSheet(f"color: {PALETTE['warning']}; font-size: 11px;")
            QMessageBox.warning(self, "Model Load Failed",
                                "Could not load model checkpoint.\n"
                                "Running in simulation mode.")

    # ── LAN ────────────────────────────────────────────────────
    def _on_lan_event(self, msg: str) -> None:
        self.status_bar.showMessage(f"LAN: {msg}")

    def _on_lan_data(self, frame: dict) -> None:
        viewer: SignalViewerWidget = self._page_widgets[1]
        if hasattr(viewer, "push_lan_frame"):
            viewer.push_lan_frame(frame)

    def _update_lan_status(self) -> None:
        if self.lan_srv.running:
            ip = self.lan_srv.local_ip
            self._lan_status.setText(f"● LAN: {ip}:{self.lan_srv.port}")
            self._lan_status.setStyleSheet(f"color: {PALETTE['success']}; font-size: 11px;")
        else:
            self._lan_status.setText("◯ LAN: Offline")
            self._lan_status.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 11px;")

    # ── Clock ───────────────────────────────────────────────────
    def _start_clock(self) -> None:
        from datetime import datetime
        def tick():
            self._clock.setText(datetime.now().strftime("  %Y-%m-%d  %H:%M:%S"))
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(tick)
        self._clock_timer.start(1000)
        tick()

    def closeEvent(self, event) -> None:
        self.lan_srv.stop()
        for w in self._page_widgets:
            if hasattr(w, "on_close"):
                w.on_close()
        super().closeEvent(event)
