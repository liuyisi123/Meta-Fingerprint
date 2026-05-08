"""Report generation panel."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QComboBox, QFileDialog, QMessageBox, QTextEdit,
    QGroupBox, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal as Signal
from ..style import PALETTE
from core.report_gen import generate_report


class ReportWorker(QObject):
    done = Signal(bool, str)
    def __init__(self, out_path, patient, session, waveforms):
        super().__init__()
        self.out_path = out_path
        self.patient  = patient
        self.session  = session
        self.waveforms = waveforms
    def run(self):
        ok = generate_report(self.out_path, self.patient, self.session, **self.waveforms)
        self.done.emit(ok, self.out_path)


class ReportPanelWidget(QWidget):
    def __init__(self, db, engine):
        super().__init__()
        self.db = db
        self.engine = engine
        self._npz_data: dict | None = None
        self._thread: QThread | None = None
        self._worker: ReportWorker | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)

        # ── Configuration ──
        cfg = QFrame(); cfg.setProperty("class", "MetricCard")
        cl = QVBoxLayout(cfg); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(12)
        cl.addWidget(self._sh("REPORT CONFIGURATION"))

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Patient:").setStyleSheet if False else self._lbl("Patient"))
        self._patient_combo = QComboBox(); self._patient_combo.setMinimumWidth(200)
        row1.addWidget(self._patient_combo); row1.addStretch()
        cl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self._lbl("Setting"))
        self._setting_combo = QComboBox()
        self._setting_combo.addItems(["A — MIMIC-III (ICU/ward)", "B — UCI (ward)",
                                       "C — RWW (wearable)", "D — MC-MED (ED)", "Custom"])
        row2.addWidget(self._setting_combo); row2.addStretch()
        cl.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(self._lbl("NPZ Data"))
        self._npz_btn = QPushButton("⊡  Load NPZ"); self._npz_btn.setObjectName("btn_secondary")
        self._npz_btn.setFixedHeight(32); self._npz_btn.clicked.connect(self._load_npz)
        self._npz_lbl = QLabel("None loaded")
        self._npz_lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px;")
        row3.addWidget(self._npz_btn); row3.addWidget(self._npz_lbl); row3.addStretch()
        cl.addLayout(row3)

        # Checkboxes
        chk_row = QHBoxLayout()
        self._chk_waveforms = QCheckBox("Include Waveforms"); self._chk_waveforms.setChecked(True)
        self._chk_metrics   = QCheckBox("Include Metrics");   self._chk_metrics.setChecked(True)
        self._chk_aami      = QCheckBox("AAMI Compliance");   self._chk_aami.setChecked(True)
        for c in [self._chk_waveforms, self._chk_metrics, self._chk_aami]:
            chk_row.addWidget(c)
        chk_row.addStretch()
        cl.addLayout(chk_row)

        root.addWidget(cfg)

        # ── Notes ──
        notes_frame = QFrame(); notes_frame.setProperty("class", "MetricCard")
        nfl = QVBoxLayout(notes_frame); nfl.setContentsMargins(14, 12, 14, 12)
        nfl.addWidget(self._sh("SESSION NOTES"))
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText("Clinical notes, protocol deviations, observations...")
        self._notes_edit.setMaximumHeight(100)
        nfl.addWidget(self._notes_edit)
        root.addWidget(notes_frame)

        # ── Generate ──
        gen_row = QHBoxLayout()
        self._gen_btn = QPushButton("▣  Generate PDF Report")
        self._gen_btn.setObjectName("btn_primary")
        self._gen_btn.setFixedHeight(44)
        self._gen_btn.setMinimumWidth(220)
        self._gen_btn.clicked.connect(self._generate)

        self._csv_btn = QPushButton("▤  Export CSV")
        self._csv_btn.setObjectName("btn_secondary")
        self._csv_btn.setFixedHeight(44)
        self._csv_btn.clicked.connect(self._export_csv)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 12px;")

        gen_row.addWidget(self._gen_btn)
        gen_row.addWidget(self._csv_btn)
        gen_row.addSpacing(16)
        gen_row.addWidget(self._status_lbl)
        gen_row.addStretch()
        root.addLayout(gen_row)

        # ── Instructions ──
        info = QFrame(); info.setProperty("class", "MetricCard")
        il = QVBoxLayout(info); il.setContentsMargins(14, 12, 14, 12)
        il.addWidget(self._sh("ABOUT REPORTS"))
        info_text = QLabel(
            "Reports include: patient demographics, signal waveforms (ECG/PPG/ABP reference and predicted), "
            "scalar BP metrics, AAMI SP10 compliance flags (Settings A-B only), "
            "risk phenotype classification, domain-shift ratio, and PTT delay statistics.\n\n"
            "AAMI SP10 flags are only meaningful for ABP-equipped Settings A-B. "
            "Setting-C (RWW, CNAP reference) is reported as CNAP-referenced wearable transfer.\n\n"
            "Requires: reportlab  (pip install reportlab)"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px; line-height: 1.5;")
        il.addWidget(info_text)
        root.addWidget(info)
        root.addStretch()

        self._refresh_patients()

    def _sh(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 9px; font-weight: 700; "
                         f"letter-spacing: 1.5px; border-bottom: 1px solid {PALETTE['border']}; padding-bottom: 4px;")
        return l

    def _lbl(self, text: str) -> QLabel:
        l = QLabel(text + ":"); l.setFixedWidth(80)
        l.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 12px;")
        return l

    def refresh(self) -> None:
        self._refresh_patients()

    def _refresh_patients(self) -> None:
        self._patient_combo.clear()
        patients = self.db.list_patients()
        self._patient_data = patients
        for p in patients:
            self._patient_combo.addItem(f"{p.name}  (ID {p.id})", p.id)
        if not patients:
            self._patient_combo.addItem("No patients — add one first", -1)

    def _load_npz(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load NPZ", "", "NumPy Archive (*.npz)")
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=True)
            self._npz_data = {k: data[k] for k in data.files}
            self._npz_lbl.setText(Path(path).name)
            self._npz_lbl.setStyleSheet(f"color: {PALETTE['success']}; font-size: 11px;")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _generate(self) -> None:
        pid = self._patient_combo.currentData()
        if pid is None or pid < 0:
            QMessageBox.warning(self, "Warning", "Please select a patient first."); return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", f"report_patient_{pid}.pdf", "PDF (*.pdf)")
        if not out_path:
            return

        p = self.db.get_patient(pid)
        patient_dict = {"id": p.id, "name": p.name, "dob": p.dob, "sex": p.sex,
                        "height_cm": p.height_cm, "weight_kg": p.weight_kg, "bmi": p.bmi}

        # Build session dict from last inference or defaults
        setting_map = {"A":"A", "B":"B", "C":"C", "D":"D", "C":"custom"}
        setting = self._setting_combo.currentText().split("—")[0].strip()
        session_dict = {
            "setting": setting, "notes": self._notes_edit.toPlainText(),
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M"),
            "rmse_sbp": 0.0, "rmse_dbp": 0.0, "macro_f1": 0.0,
            "aami_sbp": False, "aami_dbp": False, "phenotype": "—",
        }
        if hasattr(self.engine, "_last_result") and self.engine._last_result:
            r = self.engine._last_result
            session_dict.update({"rmse_sbp": r.rmse, "macro_f1": r.macro_f1,
                                  "aami_sbp": r.aami_sbp_pass, "aami_dbp": r.aami_dbp_pass,
                                  "phenotype": r.phenotype_name})

        waveforms = {"waveform_ecg": None, "waveform_ppg": None,
                     "waveform_abp": None, "waveform_abp_pred": None}
        if self._npz_data and self._chk_waveforms.isChecked():
            d = self._npz_data
            if "signals" in d:
                waveforms["waveform_ecg"] = d["signals"][0, 0, :500]
                waveforms["waveform_ppg"] = d["signals"][0, 1, :500]
            elif "ecg" in d:
                waveforms["waveform_ecg"] = d["ecg"][0, :500]
                waveforms["waveform_ppg"] = d["ppg"][0, :500]
            if "abp" in d:
                waveforms["waveform_abp"] = d["abp"][0, :500]

        self._status_lbl.setText("Generating report...")
        self._gen_btn.setEnabled(False)

        self._thread = QThread()
        self._worker = ReportWorker(out_path, patient_dict, session_dict, waveforms)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_report_done)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _on_report_done(self, ok: bool, path: str) -> None:
        self._gen_btn.setEnabled(True)
        if ok:
            self._status_lbl.setText(f"✓ Saved: {Path(path).name}")
            self._status_lbl.setStyleSheet(f"color: {PALETTE['success']}; font-size: 12px;")
            if QMessageBox.question(self, "Report Ready", f"Report saved.\nOpen file?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                os.startfile(path) if os.name == "nt" else None
        else:
            self._status_lbl.setText("✗ Failed — install reportlab")
            self._status_lbl.setStyleSheet(f"color: {PALETTE['danger']}; font-size: 12px;")
            QMessageBox.warning(self, "Error", "Could not generate PDF.\n\npip install reportlab")

    def _export_csv(self) -> None:
        QMessageBox.information(self, "CSV Export", "Use the Analysis panel to export CSV from a batch run.")
