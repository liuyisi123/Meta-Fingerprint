"""Patient management panel."""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QFormLayout, QLineEdit, QComboBox, QTextEdit,
    QDialogButtonBox, QMessageBox, QDoubleSpinBox, QSplitter
)
from PyQt5.QtCore import Qt
from ..style import PALETTE
from core.database import Patient, PatientDB


class PatientDialog(QDialog):
    def __init__(self, parent=None, patient: Patient | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Patient" if patient else "New Patient")
        self.setMinimumWidth(400)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        lyt = QVBoxLayout(self)

        form = QFormLayout()
        self._name  = QLineEdit(patient.name if patient else "")
        self._dob   = QLineEdit(patient.dob  if patient else "2000-01-01")
        self._sex   = QComboBox(); self._sex.addItems(["M","F","O"])
        if patient:
            self._sex.setCurrentText(patient.sex)
        self._ht    = QDoubleSpinBox(); self._ht.setRange(50, 250); self._ht.setSuffix(" cm")
        self._wt    = QDoubleSpinBox(); self._wt.setRange(10, 300); self._wt.setSuffix(" kg")
        if patient:
            self._ht.setValue(patient.height_cm); self._wt.setValue(patient.weight_kg)
        self._notes = QTextEdit(patient.notes if patient else "")
        self._notes.setMaximumHeight(80)

        form.addRow("Name*:", self._name)
        form.addRow("Date of Birth:", self._dob)
        form.addRow("Sex:", self._sex)
        form.addRow("Height:", self._ht)
        form.addRow("Weight:", self._wt)
        form.addRow("Notes:", self._notes)
        lyt.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lyt.addWidget(btns)

    def get_patient(self) -> Patient:
        return Patient(
            id=None, name=self._name.text().strip(),
            dob=self._dob.text().strip(), sex=self._sex.currentText(),
            height_cm=self._ht.value(), weight_kg=self._wt.value(),
            notes=self._notes.toPlainText().strip(),
        )


class PatientManagerWidget(QWidget):
    def __init__(self, db: PatientDB):
        super().__init__()
        self.db = db
        self._selected_id: int | None = None
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # ── Left: patient list ──
        left = QFrame()
        left.setProperty("class", "MetricCard")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("PATIENTS").setStyleSheet if False else self._sh("PATIENTS"))
        top.addStretch()
        self._add_btn = QPushButton("+ New")
        self._add_btn.setObjectName("btn_primary")
        self._add_btn.setFixedHeight(30)
        self._add_btn.clicked.connect(self._add_patient)
        top.addWidget(self._add_btn)
        ll.addLayout(top)

        self._search = QLineEdit(); self._search.setPlaceholderText("Search name...")
        self._search.textChanged.connect(self._refresh_list)
        ll.addWidget(self._search)

        self._patient_table = QTableWidget(0, 4)
        self._patient_table.setHorizontalHeaderLabels(["Name", "Age", "Sex", "BMI"])
        self._patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._patient_table.setAlternatingRowColors(True)
        self._patient_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._patient_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._patient_table.verticalHeader().setVisible(False)
        self._patient_table.itemSelectionChanged.connect(self._on_select)
        ll.addWidget(self._patient_table)

        root.addWidget(left, 2)

        # ── Right: patient detail ──
        right = QFrame()
        right.setProperty("class", "MetricCard")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(14, 14, 14, 14)
        rl.setSpacing(10)

        rl.addWidget(self._sh("PATIENT DETAIL"))

        self._detail_name   = self._detail_row("Name")
        self._detail_dob    = self._detail_row("Date of Birth")
        self._detail_sex    = self._detail_row("Sex")
        self._detail_ht     = self._detail_row("Height")
        self._detail_wt     = self._detail_row("Weight")
        self._detail_bmi    = self._detail_row("BMI")
        self._detail_notes  = self._detail_row("Notes")

        for _, (row_widget, _) in [("name", self._detail_name), ("dob", self._detail_dob),
            ("sex", self._detail_sex), ("ht", self._detail_ht), ("wt", self._detail_wt),
            ("bmi", self._detail_bmi), ("notes", self._detail_notes)]:
            rl.addWidget(row_widget)

        rl.addSpacing(8)
        rl.addWidget(self._sh("SESSION HISTORY"))

        self._session_table = QTableWidget(0, 4)
        self._session_table.setHorizontalHeaderLabels(["Date", "Setting", "SBP RMSE", "F1"])
        self._session_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._session_table.setAlternatingRowColors(True)
        self._session_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._session_table.verticalHeader().setVisible(False)
        self._session_table.setMaximumHeight(160)
        rl.addWidget(self._session_table)

        rl.addStretch()

        btn_row = QHBoxLayout()
        self._edit_btn = QPushButton("✏  Edit")
        self._edit_btn.setObjectName("btn_secondary"); self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._edit_patient)
        self._del_btn  = QPushButton("✕  Delete")
        self._del_btn.setObjectName("btn_danger");  self._del_btn.setEnabled(False)
        self._del_btn.clicked.connect(self._delete_patient)
        btn_row.addWidget(self._edit_btn); btn_row.addWidget(self._del_btn); btn_row.addStretch()
        rl.addLayout(btn_row)

        root.addWidget(right, 3)
        self._refresh_list()

    def _sh(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 9px; font-weight: 700; "
                         f"letter-spacing: 1.5px; border-bottom: 1px solid {PALETTE['border']}; padding-bottom: 4px;")
        return l

    def _detail_row(self, label: str) -> tuple:
        w = QWidget(); lyt = QHBoxLayout(w); lyt.setContentsMargins(0, 2, 0, 2)
        lbl = QLabel(label + ":"); lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px;"); lbl.setFixedWidth(100)
        val = QLabel("—");         val.setStyleSheet(f"color: {PALETTE['text_prim']}; font-size: 12px;")
        lyt.addWidget(lbl); lyt.addWidget(val); lyt.addStretch()
        return (w, val)  # return widget to prevent GC

    def refresh(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        query = self._search.text().strip()
        patients = self.db.list_patients(query)
        self._patients = patients
        self._patient_table.setRowCount(len(patients))
        for row, p in enumerate(patients):
            for col, val in enumerate([p.name, str(p.age), p.sex, f"{p.bmi:.1f}"]):
                item = QTableWidgetItem(val)
                item.setData(Qt.UserRole, p.id)
                self._patient_table.setItem(row, col, item)

    def _on_select(self) -> None:
        rows = self._patient_table.selectedItems()
        if not rows:
            self._selected_id = None
            self._edit_btn.setEnabled(False); self._del_btn.setEnabled(False)
            return
        pid = rows[0].data(Qt.UserRole)
        self._selected_id = pid
        self._edit_btn.setEnabled(True); self._del_btn.setEnabled(True)
        p = self.db.get_patient(pid)
        if not p:
            return
        self._detail_name[1].setText(p.name)
        self._detail_dob[1].setText(p.dob)
        self._detail_sex[1].setText(p.sex)
        self._detail_ht[1].setText(f"{p.height_cm:.0f} cm")
        self._detail_wt[1].setText(f"{p.weight_kg:.0f} kg")
        self._detail_bmi[1].setText(str(p.bmi))
        self._detail_notes[1].setText(p.notes or "—")
        sessions = self.db.list_sessions(pid)
        self._session_table.setRowCount(len(sessions))
        for row, s in enumerate(sessions):
            for col, val in enumerate([(s.get("timestamp",""))[:16], s.get("setting",""),
                                        f"{s.get('rmse_sbp',0):.2f}", f"{s.get('macro_f1',0):.3f}"]):
                self._session_table.setItem(row, col, QTableWidgetItem(val))

    def _add_patient(self) -> None:
        dlg = PatientDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            p = dlg.get_patient()
            if not p.name:
                QMessageBox.warning(self, "Validation", "Name is required."); return
            self.db.add_patient(p)
            self._refresh_list()

    def _edit_patient(self) -> None:
        if self._selected_id is None:
            return
        p = self.db.get_patient(self._selected_id)
        if not p:
            return
        dlg = PatientDialog(self, p)
        if dlg.exec_() == QDialog.Accepted:
            np_ = dlg.get_patient(); np_.id = self._selected_id
            self.db.update_patient(np_)
            self._refresh_list()

    def _delete_patient(self) -> None:
        if self._selected_id is None:
            return
        reply = QMessageBox.question(self, "Confirm Delete",
                                     "Delete this patient and all their sessions?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db.delete_patient(self._selected_id)
            self._selected_id = None
            self._refresh_list()
