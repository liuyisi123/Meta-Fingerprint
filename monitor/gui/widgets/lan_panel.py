"""LAN server control panel."""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QSpinBox, QLineEdit, QGroupBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal
from ..style import PALETTE


class LANPanelWidget(QWidget):
    server_started = Signal()
    server_stopped = Signal()

    def __init__(self, lan_srv, engine):
        super().__init__()
        self.lan_srv = lan_srv
        self.engine  = engine
        self._build()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_clients)
        self._refresh_timer.start(2000)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)

        # ── Server config card ──
        cfg = QFrame(); cfg.setProperty("class", "MetricCard")
        cl = QHBoxLayout(cfg); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(16)

        # IP display
        ip_col = QVBoxLayout()
        ip_lbl = QLabel("LOCAL IP"); ip_lbl.setObjectName("card_label")
        self._ip_val = QLabel("—"); self._ip_val.setObjectName("card_value")
        ip_col.addWidget(ip_lbl); ip_col.addWidget(self._ip_val)
        cl.addLayout(ip_col)

        # Port
        port_col = QVBoxLayout()
        port_lbl = QLabel("PORT"); port_lbl.setObjectName("card_label")
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535); self._port_spin.setValue(50505)
        self._port_spin.setFixedWidth(100)
        port_col.addWidget(port_lbl); port_col.addWidget(self._port_spin)
        cl.addLayout(port_col)

        cl.addStretch()

        # Start/Stop button
        btn_col = QVBoxLayout()
        btn_col.addStretch()
        self._toggle_btn = QPushButton("▶  Start Server")
        self._toggle_btn.setObjectName("btn_primary")
        self._toggle_btn.setFixedHeight(42)
        self._toggle_btn.setFixedWidth(160)
        self._toggle_btn.clicked.connect(self._toggle_server)
        btn_col.addWidget(self._toggle_btn)
        cl.addLayout(btn_col)

        root.addWidget(cfg)

        # ── Status bar ──
        self._status_lbl = QLabel("◯  Server offline — click Start Server to begin accepting connections.")
        self._status_lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 12px; padding: 8px; "
                                        f"background: {PALETTE['bg_card']}; border-radius: 6px;")
        root.addWidget(self._status_lbl)

        # ── Connected clients ──
        clients_frame = QFrame(); clients_frame.setProperty("class", "MetricCard")
        cfl = QVBoxLayout(clients_frame); cfl.setContentsMargins(12, 12, 12, 12)

        ch = QHBoxLayout()
        ch_lbl = QLabel("CONNECTED CLIENTS"); ch_lbl.setObjectName("card_label")
        ch.addWidget(ch_lbl); ch.addStretch()
        self._client_count = QLabel("0 clients")
        self._client_count.setStyleSheet(f"color: {PALETTE['accent']}; font-size: 12px; font-weight: 600;")
        ch.addWidget(self._client_count)
        cfl.addLayout(ch)

        self._client_table = QTableWidget(0, 5)
        self._client_table.setHorizontalHeaderLabels(["Client ID", "IP", "Port", "Frames", "Uptime"])
        self._client_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._client_table.setAlternatingRowColors(True)
        self._client_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._client_table.verticalHeader().setVisible(False)
        self._client_table.setMaximumHeight(180)
        cfl.addWidget(self._client_table)
        root.addWidget(clients_frame)

        # ── Protocol reference ──
        proto = QGroupBox("Client Protocol Reference")
        pl = QVBoxLayout(proto)
        proto_text = QTextEdit()
        proto_text.setReadOnly(True)
        proto_text.setMaximumHeight(160)
        proto_text.setStyleSheet(f"font-family: monospace; font-size: 11px; background: {PALETTE['bg_deep']}; border: none; color: {PALETTE['ecg_col']};")
        proto_text.setPlainText(
            "# Python client example\n"
            "import struct, json, socket, numpy as np\n\n"
            "MAGIC = b'MFPX'\n"
            "MSG_DATA = 0x01\n\n"
            "def send_window(sock, ecg, ppg, patient_id='P001', fs=125):\n"
            "    n = len(ecg)\n"
            "    meta = json.dumps({'patient_id': patient_id, 'fs': fs, 'n_samples': n}).encode()\n"
            "    arr = np.stack([ecg, ppg], axis=0).astype('float32').tobytes()\n"
            "    payload = struct.pack('!I', len(meta)) + meta + arr\n"
            "    hdr = struct.pack('!4sIHH', MAGIC, len(payload), 1, MSG_DATA)\n"
            "    sock.sendall(hdr + payload)\n\n"
            f"# Connect to  {self.lan_srv.local_ip}:{self.lan_srv.port}"
        )
        pl.addWidget(proto_text)
        root.addWidget(proto)
        root.addStretch()

    def _toggle_server(self) -> None:
        if self.lan_srv.running:
            self.lan_srv.stop()
            self._toggle_btn.setText("▶  Start Server")
            self._toggle_btn.setObjectName("btn_primary")
            self._status_lbl.setText("◯  Server stopped.")
            self._status_lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 12px; padding: 8px; background: {PALETTE['bg_card']}; border-radius: 6px;")
            self.server_stopped.emit()
        else:
            self.lan_srv.port = self._port_spin.value()
            ok = self.lan_srv.start()
            if ok:
                ip = self.lan_srv.local_ip
                self._ip_val.setText(ip)
                self._ip_val.setStyleSheet(f"color: {PALETTE['success']}; font-size: 28px; font-weight: 700;")
                self._toggle_btn.setText("⏹  Stop Server")
                self._toggle_btn.setObjectName("btn_danger")
                self._status_lbl.setText(f"●  Server running on  {ip}:{self.lan_srv.port}  — waiting for clients.")
                self._status_lbl.setStyleSheet(f"color: {PALETTE['success']}; font-size: 12px; padding: 8px; background: {PALETTE['bg_card']}; border-radius: 6px;")
                self.server_started.emit()
            else:
                self._status_lbl.setText(f"✗  Failed to start server on port {self.lan_srv.port}.")
                self._status_lbl.setStyleSheet(f"color: {PALETTE['danger']}; font-size: 12px; padding: 8px; background: {PALETTE['bg_card']}; border-radius: 6px;")

    def _refresh_clients(self) -> None:
        if not self.lan_srv.running:
            return
        clients = self.lan_srv.client_list()
        self._client_count.setText(f"{len(clients)} client{'s' if len(clients) != 1 else ''}")
        self._client_table.setRowCount(len(clients))
        for row, c in enumerate(clients):
            for col, val in enumerate([c["client_id"], c["ip"], str(c["port"]),
                                        str(c["frames"]), f"{c['uptime_s']:.0f}s"]):
                self._client_table.setItem(row, col, QTableWidgetItem(val))
