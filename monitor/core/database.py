"""SQLite patient database."""
from __future__ import annotations
import sqlite3, json, threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path.home() / ".meta_fingerprint" / "patients.db"


@dataclass
class Patient:
    id: int | None
    name: str
    dob: str            # YYYY-MM-DD
    sex: str            # M/F/O
    height_cm: float
    weight_kg: float
    notes: str = ""
    created_at: str = ""

    @property
    def age(self) -> int:
        try:
            dob = datetime.strptime(self.dob, "%Y-%m-%d")
            return (datetime.now() - dob).days // 365
        except Exception:
            return 0

    @property
    def bmi(self) -> float:
        if self.height_cm > 0 and self.weight_kg > 0:
            return round(self.weight_kg / (self.height_cm / 100) ** 2, 1)
        return 0.0


@dataclass
class Session:
    id: int | None
    patient_id: int
    timestamp: str
    setting: str           # A/B/C/D/custom
    rmse_sbp: float
    rmse_dbp: float
    macro_f1: float
    aami_sbp: bool
    aami_dbp: bool
    domain_shift_ratio: float
    notes: str = ""
    artifacts: str = "{}"  # JSON blob


class PatientDB:
    _local = threading.local()

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                dob        TEXT,
                sex        TEXT DEFAULT 'O',
                height_cm  REAL DEFAULT 0,
                weight_kg  REAL DEFAULT 0,
                notes      TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id          INTEGER NOT NULL REFERENCES patients(id),
                timestamp           TEXT DEFAULT (datetime('now')),
                setting             TEXT DEFAULT 'custom',
                rmse_sbp            REAL DEFAULT 0,
                rmse_dbp            REAL DEFAULT 0,
                macro_f1            REAL DEFAULT 0,
                aami_sbp            INTEGER DEFAULT 0,
                aami_dbp            INTEGER DEFAULT 0,
                domain_shift_ratio  REAL DEFAULT 0,
                notes               TEXT DEFAULT '',
                artifacts           TEXT DEFAULT '{}'
            );
        """)
        c.commit()

    # ── Patients ───────────────────────────────────────────────
    def add_patient(self, p: Patient) -> int:
        c = self._conn()
        cur = c.execute(
            "INSERT INTO patients (name,dob,sex,height_cm,weight_kg,notes) VALUES (?,?,?,?,?,?)",
            (p.name, p.dob, p.sex, p.height_cm, p.weight_kg, p.notes),
        )
        c.commit(); return cur.lastrowid

    def update_patient(self, p: Patient) -> None:
        c = self._conn()
        c.execute(
            "UPDATE patients SET name=?,dob=?,sex=?,height_cm=?,weight_kg=?,notes=? WHERE id=?",
            (p.name, p.dob, p.sex, p.height_cm, p.weight_kg, p.notes, p.id),
        ); c.commit()

    def delete_patient(self, patient_id: int) -> None:
        c = self._conn()
        c.execute("DELETE FROM sessions WHERE patient_id=?", (patient_id,))
        c.execute("DELETE FROM patients WHERE id=?", (patient_id,))
        c.commit()

    def get_patient(self, patient_id: int) -> Patient | None:
        row = self._conn().execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        return _row_to_patient(row) if row else None

    def list_patients(self, query: str = "") -> list[Patient]:
        sql = "SELECT * FROM patients WHERE name LIKE ? ORDER BY name" if query else "SELECT * FROM patients ORDER BY name"
        rows = self._conn().execute(sql, (f"%{query}%",) if query else ()).fetchall()
        return [_row_to_patient(r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────
    def add_session(self, s: Session) -> int:
        c = self._conn()
        cur = c.execute(
            """INSERT INTO sessions
               (patient_id,timestamp,setting,rmse_sbp,rmse_dbp,macro_f1,
                aami_sbp,aami_dbp,domain_shift_ratio,notes,artifacts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (s.patient_id, s.timestamp, s.setting, s.rmse_sbp, s.rmse_dbp,
             s.macro_f1, int(s.aami_sbp), int(s.aami_dbp), s.domain_shift_ratio,
             s.notes, s.artifacts),
        ); c.commit(); return cur.lastrowid

    def list_sessions(self, patient_id: int | None = None) -> list[dict[str, Any]]:
        if patient_id:
            rows = self._conn().execute(
                "SELECT s.*, p.name as patient_name FROM sessions s JOIN patients p ON p.id=s.patient_id WHERE s.patient_id=? ORDER BY s.timestamp DESC",
                (patient_id,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT s.*, p.name as patient_name FROM sessions s JOIN patients p ON p.id=s.patient_id ORDER BY s.timestamp DESC LIMIT 100"
            ).fetchall()
        return [dict(r) for r in rows]

    def summary_stats(self) -> dict[str, Any]:
        row = self._conn().execute(
            "SELECT COUNT(*) as n_patients FROM patients"
        ).fetchone()
        row2 = self._conn().execute(
            "SELECT COUNT(*) as n_sessions, AVG(rmse_sbp) as avg_rmse_sbp, "
            "AVG(macro_f1) as avg_f1 FROM sessions"
        ).fetchone()
        return {
            "n_patients": row["n_patients"],
            "n_sessions": row2["n_sessions"],
            "avg_rmse_sbp": round(float(row2["avg_rmse_sbp"] or 0), 2),
            "avg_f1": round(float(row2["avg_f1"] or 0), 3),
        }


def _row_to_patient(row: sqlite3.Row) -> Patient:
    return Patient(
        id=row["id"], name=row["name"], dob=row["dob"] or "",
        sex=row["sex"] or "O", height_cm=row["height_cm"] or 0,
        weight_kg=row["weight_kg"] or 0, notes=row["notes"] or "",
        created_at=row["created_at"] or "",
    )
