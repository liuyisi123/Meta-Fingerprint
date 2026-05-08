"""Dashboard: overview metrics, activity feed, quick actions."""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from ..style import PALETTE


class MetricCard(QFrame):
    def __init__(self, label: str, value: str, unit: str = "",
                 trend: str = "", trend_up: bool = True, color: str | None = None):
        super().__init__()
        self.setProperty("class", "MetricCard")
        self.setMinimumHeight(110)
        lyt = QVBoxLayout(self)
        lyt.setContentsMargins(16, 14, 16, 14)
        lyt.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setObjectName("card_label")

        val_lyt = QHBoxLayout()
        self._val_label = QLabel(value)
        self._val_label.setObjectName("card_value")
        if color:
            self._val_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 700;")
        val_lyt.addWidget(self._val_label)

        if unit:
            unit_lbl = QLabel(unit)
            unit_lbl.setObjectName("card_unit")
            unit_lbl.setAlignment(Qt.AlignBottom)
            val_lyt.addWidget(unit_lbl)
        val_lyt.addStretch()

        lyt.addWidget(lbl)
        lyt.addLayout(val_lyt)

        if trend:
            trend_lbl = QLabel(trend)
            trend_lbl.setObjectName("card_trend_up" if trend_up else "card_trend_down")
            lyt.addWidget(trend_lbl)

    def update_value(self, val: str, color: str | None = None) -> None:
        self._val_label.setText(val)
        if color:
            self._val_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 700;")


class DashboardWidget(QWidget):
    def __init__(self, db, engine):
        super().__init__()
        self.db = db
        self.engine = engine
        self._build()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._update_stats)
        self._refresh_timer.start(5000)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(16)

        # ── Section title ──
        title = QLabel("System Overview")
        title.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px;")
        root.addWidget(title)

        # ── Metric cards (2 rows × 4 cols) ──
        cards_grid = QGridLayout()
        cards_grid.setSpacing(12)

        self._cards = {
            "n_patients": MetricCard("Patients", "—", "", "in database"),
            "n_sessions":  MetricCard("Sessions", "—", "", "total analyses"),
            "avg_rmse":    MetricCard("Avg SBP RMSE", "—", "mmHg", "cross-domain"),
            "avg_f1":      MetricCard("Avg Macro-F1", "—", "", "phenotyping"),
            "model":       MetricCard("Model", "Demo" if not self.engine.loaded else "Loaded", "",
                                      "", color=PALETTE["warning"]),
            "lan":         MetricCard("LAN Server", "Offline", "", ""),
            "aami":        MetricCard("AAMI SP10", "—", "", "Settings A-B"),
            "shift":       MetricCard("Domain Shift ρ", "—", "", "lower=better"),
        }
        positions = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
        for (card, (r, c)) in zip(self._cards.values(), positions):
            cards_grid.addWidget(card, r, c)
        root.addLayout(cards_grid)

        # ── Lower section: recent sessions + quick actions ──
        lower = QHBoxLayout()
        lower.setSpacing(16)

        # Recent sessions table
        sessions_frame = QFrame()
        sessions_frame.setProperty("class", "MetricCard")
        sf_lyt = QVBoxLayout(sessions_frame)
        sf_lyt.setContentsMargins(12, 12, 12, 12)
        sf_lyt.setSpacing(8)

        sh = QLabel("RECENT SESSIONS")
        sh.setObjectName("card_label")
        sf_lyt.addWidget(sh)

        self._session_table = QTableWidget(0, 5)
        self._session_table.setHorizontalHeaderLabels(
            ["Patient", "Setting", "RMSE SBP", "Macro-F1", "Timestamp"]
        )
        self._session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._session_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.setMinimumHeight(180)
        sf_lyt.addWidget(self._session_table)
        lower.addWidget(sessions_frame, 3)

        # Quick actions
        actions_frame = QFrame()
        actions_frame.setProperty("class", "MetricCard")
        actions_frame.setFixedWidth(220)
        af_lyt = QVBoxLayout(actions_frame)
        af_lyt.setContentsMargins(12, 12, 12, 12)
        af_lyt.setSpacing(10)

        ah = QLabel("QUICK ACTIONS")
        ah.setObjectName("card_label")
        af_lyt.addWidget(ah)

        actions = [
            ("◈  Open Signal Monitor", PALETTE["accent"]),
            ("⊞  Run Batch Analysis",  PALETTE["accent"]),
            ("▣  Generate Report",     PALETTE["accent"]),
            ("⋄  Start LAN Server",    PALETTE["success"]),
        ]
        for label, color in actions:
            btn = QPushButton(label)
            btn.setObjectName("btn_secondary")
            btn.setFixedHeight(38)
            af_lyt.addWidget(btn)

        af_lyt.addStretch()

        # Status strip
        status_text = (
            f"<b>Meta-Fingerprint v1.0</b><br/>"
            f"<font size='2' color='{PALETTE['text_mute']}'>"
            f"Physics-Grounded Hemodynamic Monitoring<br/>"
            f"Research use only — not a clinical device</font>"
        )
        status_lbl = QLabel(status_text)
        status_lbl.setWordWrap(True)
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px; padding: 8px; "
                                  f"border: 1px solid {PALETTE['border']}; border-radius: 6px;")
        af_lyt.addWidget(status_lbl)

        lower.addWidget(actions_frame, 1)
        root.addLayout(lower)

    def refresh(self) -> None:
        self._update_stats()

    def _update_stats(self) -> None:
        try:
            stats = self.db.summary_stats()
            self._cards["n_patients"].update_value(str(stats["n_patients"]))
            self._cards["n_sessions"].update_value(str(stats["n_sessions"]))
            rmse = stats["avg_rmse_sbp"]
            color = PALETTE["success"] if rmse <= 6.0 else PALETTE["warning"] if rmse <= 9.0 else PALETTE["danger"]
            self._cards["avg_rmse"].update_value(f"{rmse:.2f}", color)
            f1 = stats["avg_f1"]
            f1_color = PALETTE["success"] if f1 >= 0.7 else PALETTE["warning"]
            self._cards["avg_f1"].update_value(f"{f1:.3f}", f1_color)

            model_txt = "Loaded" if self.engine.loaded else "Demo"
            model_col = PALETTE["success"] if self.engine.loaded else PALETTE["warning"]
            self._cards["model"].update_value(model_txt, model_col)
        except Exception:
            pass

        self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        try:
            sessions = self.db.list_sessions()[:8]
            self._session_table.setRowCount(len(sessions))
            for row, s in enumerate(sessions):
                rmse = s.get("rmse_sbp", 0)
                f1   = s.get("macro_f1", 0)
                for col, val in enumerate([
                    s.get("patient_name", "—"),
                    s.get("setting", "—"),
                    f"{rmse:.2f}" if rmse else "—",
                    f"{f1:.3f}"   if f1   else "—",
                    (s.get("timestamp") or "")[:16],
                ]):
                    item = QTableWidgetItem(val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self._session_table.setItem(row, col, item)
        except Exception:
            pass
