"""Batch analysis panel: load NPZ, run inference, show full metrics."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QGroupBox, QComboBox, QMessageBox, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal as Signal
from ..style import PALETTE

try:
    import pyqtgraph as pg
    _PG = True
except ImportError:
    _PG = False


class BatchWorker(QObject):
    progress = Signal(int, int)
    done     = Signal(list)

    def __init__(self, engine, path):
        super().__init__()
        self.engine = engine
        self.path = path

    def run(self):
        results = self.engine.run_batch(self.path, progress_cb=lambda i, n: self.progress.emit(i, n))
        self.done.emit(results)


class AnalysisWidget(QWidget):
    def __init__(self, engine, db):
        super().__init__()
        self.engine  = engine
        self.db      = db
        self._results = []
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(12)

        # ── Toolbar ──
        tb = QHBoxLayout()
        self._open_btn = QPushButton("⊡  Open NPZ File")
        self._open_btn.setObjectName("btn_primary")
        self._open_btn.setFixedHeight(36)
        self._open_btn.clicked.connect(self._open_file)

        self._run_btn = QPushButton("⊞  Run Analysis")
        self._run_btn.setObjectName("btn_secondary")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_batch)

        self._export_btn = QPushButton("▤  Export CSV")
        self._export_btn.setObjectName("btn_secondary")
        self._export_btn.setFixedHeight(36)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)

        self._file_lbl = QLabel("No file loaded")
        self._file_lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px;")

        tb.addWidget(self._open_btn)
        tb.addWidget(self._run_btn)
        tb.addWidget(self._export_btn)
        tb.addSpacing(12)
        tb.addWidget(self._file_lbl)
        tb.addStretch()
        root.addLayout(tb)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        root.addWidget(self._progress)

        # ── Split: table + plots ──
        splitter = QSplitter(Qt.Vertical)

        # Results table
        table_frame = QFrame()
        tf_lyt = QVBoxLayout(table_frame)
        tf_lyt.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            "#", "SBP", "DBP", "MAP", "RMSE", "Macro-F1",
            "Phenotype", "τ (ms)", "AAMI"
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(200)
        tf_lyt.addWidget(self._table)
        splitter.addWidget(table_frame)

        # Stats + plots
        lower = QFrame()
        ll = QHBoxLayout(lower)
        ll.setContentsMargins(0, 4, 0, 0)
        ll.setSpacing(12)

        # Aggregate stats
        stats_frame = QFrame()
        stats_frame.setProperty("class", "MetricCard")
        stats_frame.setFixedWidth(280)
        sf = QVBoxLayout(stats_frame)
        sf.setContentsMargins(14, 14, 14, 14)

        sf.addWidget(self._sh("AGGREGATE METRICS"))
        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setStyleSheet(
            f"background: {PALETTE['bg_panel']}; border: none; "
            f"color: {PALETTE['text_prim']}; font-family: monospace; font-size: 11px;"
        )
        self._stats_text.setText("Run analysis to see metrics.")
        sf.addWidget(self._stats_text)
        ll.addWidget(stats_frame)

        # ABP distribution plot
        if _PG:
            plot_group = QGroupBox("Predicted SBP/DBP Distribution")
            pg_lyt = QVBoxLayout(plot_group)
            self._dist_plot = pg.PlotWidget()
            self._dist_plot.setBackground(PALETTE["bg_card"])
            self._dist_plot.setTitle("SBP / DBP", color=PALETTE["accent"])
            pg_lyt.addWidget(self._dist_plot)
            ll.addWidget(plot_group, 1)

            # Bland-Altman plot
            ba_group = QGroupBox("Bland-Altman (if reference available)")
            ba_lyt = QVBoxLayout(ba_group)
            self._ba_plot = pg.PlotWidget()
            self._ba_plot.setBackground(PALETTE["bg_card"])
            self._ba_plot.setTitle("Bias vs Mean", color=PALETTE["accent"])
            ba_lyt.addWidget(self._ba_plot)
            ll.addWidget(ba_group, 1)
        else:
            ll.addWidget(QLabel("Install pyqtgraph for plots"), 2)

        splitter.addWidget(lower)
        splitter.setSizes([400, 300])
        root.addWidget(splitter)

    def _sh(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 9px; font-weight: 700; "
                         f"letter-spacing: 1.5px; border-bottom: 1px solid {PALETTE['border']}; "
                         f"padding-bottom: 4px; margin-bottom: 6px;")
        return l

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open NPZ", "", "NumPy Archive (*.npz);;All (*)")
        if not path:
            return
        self._npz_path = path
        self._file_lbl.setText(Path(path).name)
        self._run_btn.setEnabled(True)

    def _run_batch(self) -> None:
        if not hasattr(self, "_npz_path"):
            return
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._thread = QThread()
        self._worker = BatchWorker(self.engine, self._npz_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, i: int, n: int) -> None:
        self._progress.setMaximum(n)
        self._progress.setValue(i)

    def _on_done(self, results: list) -> None:
        self._results = results
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(bool(results))
        self._fill_table(results)
        self._update_stats(results)
        self._update_plots(results)

    def _fill_table(self, results) -> None:
        self._table.setRowCount(len(results))
        pheno_colors = {
            "Hypotension": PALETTE["danger"], "Normal": PALETTE["success"],
            "Pre-HTN": PALETTE["warning"],    "Hypertension": PALETTE["danger"],
        }
        for row, r in enumerate(results):
            aami = "✓" if (r.aami_sbp_pass and r.aami_dbp_pass) else "✗"
            vals = [str(row+1), f"{r.sbp:.0f}", f"{r.dbp:.0f}", f"{r.map_val:.0f}",
                    f"{r.rmse:.2f}", f"{r.macro_f1:.3f}", r.phenotype_name,
                    f"{r.tau_ms:.0f}", aami]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 6:
                    item.setForeground(pheno_colors.get(v, PALETTE["text_prim"]))
                self._table.setItem(row, col, item)

    def _update_stats(self, results) -> None:
        if not results:
            return
        rmses  = np.array([r.rmse for r in results])
        sbps   = np.array([r.sbp  for r in results])
        dbps   = np.array([r.dbp  for r in results])
        f1s    = np.array([r.macro_f1 for r in results])
        aami_n = sum(1 for r in results if r.aami_sbp_pass and r.aami_dbp_pass)
        taus   = np.array([r.tau_ms for r in results])

        pheno_counts = {}
        for r in results:
            pheno_counts[r.phenotype_name] = pheno_counts.get(r.phenotype_name, 0) + 1

        text = (
            f"Windows :  {len(results)}\n\n"
            f"── Waveform ───────────────\n"
            f"RMSE mean :  {rmses.mean():.2f} mmHg\n"
            f"RMSE  std :  {rmses.std():.2f} mmHg\n"
            f"SBP  mean :  {sbps.mean():.0f} mmHg\n"
            f"DBP  mean :  {dbps.mean():.0f} mmHg\n\n"
            f"── Phenotype ──────────────\n"
            f"Macro-F1  :  {f1s.mean():.3f}\n"
            + "".join(f"{k:<12}: {v} ({100*v/len(results):.0f}%)\n"
                      for k, v in pheno_counts.items())
            + f"\n── Delay ──────────────────\n"
            f"τ PTT mean:  {taus.mean():.0f} ms\n"
            f"τ PTT  std:  {taus.std():.0f} ms\n\n"
            f"── AAMI SP10 ──────────────\n"
            f"Pass rate :  {100*aami_n/len(results):.0f}% ({aami_n}/{len(results)})\n"
        )
        self._stats_text.setText(text)

    def _update_plots(self, results) -> None:
        if not _PG or not results:
            return
        sbps = [r.sbp for r in results]
        dbps = [r.dbp for r in results]
        hist_s, bins_s = np.histogram(sbps, bins=20)
        hist_d, bins_d = np.histogram(dbps, bins=20)
        self._dist_plot.clear()
        bg_s = pg.BarGraphItem(x=bins_s[:-1], height=hist_s, width=bins_s[1]-bins_s[0],
                               brush=pg.mkBrush(PALETTE["ppg_col"] + "80"))
        bg_d = pg.BarGraphItem(x=bins_d[:-1], height=hist_d, width=bins_d[1]-bins_d[0],
                               brush=pg.mkBrush(PALETTE["ecg_col"] + "80"))
        self._dist_plot.addItem(bg_s)
        self._dist_plot.addItem(bg_d)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "analysis_results.csv", "CSV (*.csv)")
        if not path or not self._results:
            return
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx","sbp","dbp","map","rmse","macro_f1","phenotype","tau_ms","aami_sbp","aami_dbp"])
            for i, r in enumerate(self._results):
                w.writerow([i+1, f"{r.sbp:.1f}", f"{r.dbp:.1f}", f"{r.map_val:.1f}",
                             f"{r.rmse:.3f}", f"{r.macro_f1:.3f}", r.phenotype_name,
                             f"{r.tau_ms:.0f}", r.aami_sbp_pass, r.aami_dbp_pass])
        QMessageBox.information(self, "Export", f"Saved to:\n{path}")
