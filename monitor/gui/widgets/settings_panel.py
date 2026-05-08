"""Settings and calibration panel."""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QGroupBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QLineEdit, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt
from ..style import PALETTE


class SettingsWidget(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(16)

        # ── Model section ──
        m_group = self._group("MODEL CONFIGURATION")
        ml = QVBoxLayout(m_group)

        ml.addWidget(self._row_label("Loaded checkpoint"))
        self._model_path_lbl = QLabel("None (running in simulation mode)")
        self._model_path_lbl.setStyleSheet(f"color: {PALETTE['warning']}; font-size: 12px; padding: 6px; background: {PALETTE['bg_panel']}; border-radius: 4px;")
        ml.addWidget(self._model_path_lbl)

        dev_row = QHBoxLayout()
        dev_row.addWidget(self._row_label("Inference device"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda:0", "cuda:1"])
        self._device_combo.setFixedWidth(120)
        dev_row.addWidget(self._device_combo); dev_row.addStretch()
        ml.addLayout(dev_row)
        root.addWidget(m_group)

        # ── AAMI thresholds ──
        a_group = self._group("AAMI SP10 TOLERANCE")
        al = QVBoxLayout(a_group)

        for label, attr, val in [("SBP |bias| threshold (mmHg)", "_aami_bias_sbp", 5.0),
                                   ("SBP σd threshold (mmHg)",     "_aami_sd_sbp",   8.0),
                                   ("DBP |bias| threshold (mmHg)", "_aami_bias_dbp", 5.0),
                                   ("DBP σd threshold (mmHg)",     "_aami_sd_dbp",   8.0)]:
            row = QHBoxLayout()
            row.addWidget(self._row_label(label))
            sp = QDoubleSpinBox(); sp.setRange(1.0, 20.0); sp.setValue(val); sp.setSingleStep(0.5)
            sp.setFixedWidth(90); setattr(self, attr, sp)
            row.addWidget(sp); row.addStretch()
            al.addLayout(row)

        al.addWidget(QLabel(
            "AAMI compliance applies to ABP-equipped Settings A-B only.\n"
            "Setting-C (CNAP reference) is excluded from AAMI assessment."
        ).setStyleSheet if False else self._info(
            "AAMI SP10 compliance applies to ABP-equipped Settings A-B only. "
            "Setting-C (CNAP reference) is excluded from AAMI assessment."
        ))
        root.addWidget(a_group)

        # ── Display settings ──
        d_group = self._group("DISPLAY")
        dl = QVBoxLayout(d_group)

        speed_row = QHBoxLayout()
        speed_row.addWidget(self._row_label("Default playback speed"))
        self._speed_combo = QComboBox(); self._speed_combo.addItems(["0.5×", "1×", "2×", "4×"])
        self._speed_combo.setCurrentIndex(1); self._speed_combo.setFixedWidth(80)
        speed_row.addWidget(self._speed_combo); speed_row.addStretch()
        dl.addLayout(speed_row)

        self._dark_chk = QCheckBox("Dark theme (always on)"); self._dark_chk.setChecked(True)
        self._dark_chk.setEnabled(False)
        dl.addWidget(self._dark_chk)
        root.addWidget(d_group)

        # ── LAN settings ──
        l_group = self._group("LAN DEFAULTS")
        ll = QVBoxLayout(l_group)
        port_row = QHBoxLayout()
        port_row.addWidget(self._row_label("Default port"))
        self._port_spin = QSpinBox(); self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(50505); self._port_spin.setFixedWidth(100)
        port_row.addWidget(self._port_spin); port_row.addStretch()
        ll.addLayout(port_row)
        root.addWidget(l_group)

        # ── About ──
        about_frame = QFrame(); about_frame.setProperty("class", "MetricCard")
        abl = QVBoxLayout(about_frame)
        abl.setContentsMargins(16, 14, 16, 14)
        about_text = (
            f"<b style='color:{PALETTE['accent']}'>Meta-Fingerprint Monitor</b>  v1.0<br/>"
            f"<span style='color:{PALETTE['text_sec']}'>Physics-Grounded Vascular Disentanglement for "
            f"Generalizable Cross-Domain Hemodynamic Monitoring</span><br/><br/>"
            f"<span style='color:{PALETTE['text_mute']}'>Research software — not validated for clinical use.</span>"
        )
        about_lbl = QLabel(about_text); about_lbl.setWordWrap(True)
        abl.addWidget(about_lbl)
        root.addWidget(about_frame)
        root.addStretch()

        # Save button
        save_btn = QPushButton("✓  Apply Settings")
        save_btn.setObjectName("btn_primary")
        save_btn.setFixedHeight(40); save_btn.setFixedWidth(180)
        save_btn.clicked.connect(self._apply)
        root.addWidget(save_btn, alignment=Qt.AlignLeft)

    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {PALETTE['border']};
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background: {PALETTE['bg_card']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px; top: -1px;
                background-color: {PALETTE['bg_card']};
                padding: 0 6px;
                color: {PALETTE['accent']};
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
        """)
        return g

    def _row_label(self, text: str) -> QLabel:
        l = QLabel(text); l.setFixedWidth(240)
        l.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 12px;")
        return l

    def _info(self, text: str) -> QLabel:
        l = QLabel(text); l.setWordWrap(True)
        l.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 10px; padding: 4px; "
                         f"border: 1px solid {PALETTE['border']}; border-radius: 4px; margin-top: 4px;")
        return l

    def _apply(self) -> None:
        QMessageBox.information(self, "Settings", "Settings applied.")
        if self.engine.loaded:
            self._model_path_lbl.setText(self.engine.model_path)
            self._model_path_lbl.setStyleSheet(f"color: {PALETTE['success']}; font-size: 12px; padding: 6px; background: {PALETTE['bg_panel']}; border-radius: 4px;")
