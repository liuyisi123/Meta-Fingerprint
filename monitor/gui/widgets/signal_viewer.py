"""Real-time scrolling ECG/PPG/ABP signal viewer with live inference."""
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QSlider, QComboBox, QFileDialog, QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QThread, QObject
from ..style import PALETTE

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background=PALETTE["bg_deep"], foreground=PALETTE["text_prim"])
    _PG = True
except ImportError:
    _PG = False


CHANNELS = [("ECG", PALETTE["ecg_col"]), ("PPG", PALETTE["ppg_col"]), ("ABP (Reference)", "#7AAFCF"), ("ABP (Predicted)", PALETTE["abp_col"])]
DISPLAY_SEC = 10


class InferWorker(QObject):
    result = Signal(object)
    def __init__(self, engine, ecg, ppg, abp=None):
        super().__init__()
        self.engine, self.ecg, self.ppg, self.abp = engine, ecg, ppg, abp
    def run(self):
        r = self.engine.run(self.ecg, self.ppg, self.abp)
        self.result.emit(r)


class SignalViewerWidget(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._fs = 125.0
        self._buf_len = int(self._fs * DISPLAY_SEC)
        self._ecg_buf  = np.zeros(self._buf_len, dtype=np.float32)
        self._ppg_buf  = np.zeros(self._buf_len, dtype=np.float32)
        self._abp_ref  = np.zeros(self._buf_len, dtype=np.float32)
        self._abp_pred = np.zeros(self._buf_len, dtype=np.float32)
        self._playing = False
        self._sim_phase = 0.0
        self._rng = np.random.default_rng(0)
        self._last_result = None
        self._infer_thread: QThread | None = None
        self._infer_worker: InferWorker | None = None
        self._build()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(80)
        self._play_timer.timeout.connect(self._sim_tick)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Toolbar ──
        tb = QHBoxLayout()
        self._play_btn = QPushButton("▶  Play Simulation")
        self._play_btn.setObjectName("btn_primary")
        self._play_btn.setFixedHeight(36)
        self._play_btn.clicked.connect(self._toggle_play)

        self._load_btn = QPushButton("⊡  Load NPZ")
        self._load_btn.setObjectName("btn_secondary")
        self._load_btn.setFixedHeight(36)
        self._load_btn.clicked.connect(self._load_npz)

        self._infer_btn = QPushButton("⊞  Run Inference")
        self._infer_btn.setObjectName("btn_secondary")
        self._infer_btn.setFixedHeight(36)
        self._infer_btn.clicked.connect(self._run_inference)

        self._speed_lbl = QLabel("Speed:")
        self._speed_lbl.setStyleSheet(f"color: {PALETTE['text_sec']};")
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.5×", "1×", "2×", "4×"])
        self._speed_combo.setCurrentIndex(1)
        self._speed_combo.setFixedWidth(70)

        tb.addWidget(self._play_btn)
        tb.addSpacing(8)
        tb.addWidget(self._load_btn)
        tb.addWidget(self._infer_btn)
        tb.addStretch()
        tb.addWidget(self._speed_lbl)
        tb.addWidget(self._speed_combo)
        root.addLayout(tb)

        splitter = QSplitter(Qt.Horizontal)

        # ── Plot area ──
        plot_widget = QWidget()
        plot_lyt = QVBoxLayout(plot_widget)
        plot_lyt.setContentsMargins(0, 0, 0, 0)
        plot_lyt.setSpacing(4)

        if _PG:
            self._plots: list[pg.PlotWidget] = []
            self._curves: list[pg.PlotCurveItem] = []
            for (name, col) in CHANNELS:
                pw = pg.PlotWidget()
                pw.setTitle(name, color=col, size="11pt")
                pw.setLabel("left", "Amplitude")
                pw.showGrid(x=False, y=True, alpha=0.15)
                pw.setMouseEnabled(x=False, y=True)
                pw.setMinimumHeight(80)
                pw.setBackground(PALETTE["bg_card"])
                curve = pw.plot(pen=pg.mkPen(col, width=1.4))
                self._plots.append(pw)
                self._curves.append(curve)
                plot_lyt.addWidget(pw)
        else:
            lbl = QLabel("pyqtgraph not installed.\npip install pyqtgraph")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {PALETTE['text_sec']};")
            plot_lyt.addWidget(lbl)

        splitter.addWidget(plot_widget)

        # ── Side panel: metrics ──
        side = QFrame()
        side.setProperty("class", "MetricCard")
        side.setFixedWidth(220)
        side_lyt = QVBoxLayout(side)
        side_lyt.setContentsMargins(14, 14, 14, 14)
        side_lyt.setSpacing(10)

        side_lyt.addWidget(self._section("CURRENT WINDOW"))

        self._sbp_lbl  = self._metric_row("SBP",  "—",  "mmHg")
        self._dbp_lbl  = self._metric_row("DBP",  "—",  "mmHg")
        self._map_lbl  = self._metric_row("MAP",  "—",  "mmHg")
        self._tau_lbl  = self._metric_row("τ PTT","—",  "ms")
        self._rmse_lbl = self._metric_row("RMSE", "—",  "mmHg")
        for row in [self._sbp_lbl, self._dbp_lbl, self._map_lbl, self._tau_lbl, self._rmse_lbl]:
            side_lyt.addWidget(row[0])

        side_lyt.addSpacing(6)
        side_lyt.addWidget(self._section("PHENOTYPE"))

        self._pheno_label = QLabel("—")
        self._pheno_label.setAlignment(Qt.AlignCenter)
        self._pheno_label.setStyleSheet(
            f"background: {PALETTE['bg_panel']}; border-radius: 8px; "
            f"padding: 10px; font-size: 14px; font-weight: 700; color: {PALETTE['text_prim']};"
        )
        side_lyt.addWidget(self._pheno_label)

        side_lyt.addSpacing(6)
        side_lyt.addWidget(self._section("PROB. DISTRIBUTION"))
        self._prob_labels: list[QLabel] = []
        for name in ["Hypo", "Normal", "Pre-HTN", "HTN"]:
            row_w = QWidget()
            rl = QHBoxLayout(row_w); rl.setContentsMargins(0,0,0,0)
            nl = QLabel(name); nl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px;"); nl.setFixedWidth(60)
            pl = QLabel("—"); pl.setStyleSheet(f"color: {PALETTE['accent']}; font-size: 11px;")
            rl.addWidget(nl); rl.addWidget(pl)
            self._prob_labels.append(pl)
            side_lyt.addWidget(row_w)

        side_lyt.addSpacing(6)
        side_lyt.addWidget(self._section("AAMI SP10"))
        self._aami_sbp_lbl = QLabel("SBP: —")
        self._aami_dbp_lbl = QLabel("DBP: —")
        for l in [self._aami_sbp_lbl, self._aami_dbp_lbl]:
            l.setStyleSheet(f"font-size: 11px; color: {PALETTE['text_sec']};")
            side_lyt.addWidget(l)

        side_lyt.addStretch()

        self._infer_time = QLabel("Inference: —")
        self._infer_time.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 10px;")
        side_lyt.addWidget(self._infer_time)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter)

    # ── Helpers ────────────────────────────────────────────────
    def _section(self, text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 9px; font-weight: 700; "
                         f"letter-spacing: 1.5px; border-bottom: 1px solid {PALETTE['border']}; padding-bottom: 4px;")
        return l

    def _metric_row(self, label: str, value: str, unit: str) -> tuple:
        w = QWidget(); lyt = QHBoxLayout(w); lyt.setContentsMargins(0, 2, 0, 2)
        lbl = QLabel(label); lbl.setStyleSheet(f"color: {PALETTE['text_sec']}; font-size: 11px;"); lbl.setFixedWidth(55)
        val = QLabel(value); val.setStyleSheet(f"color: {PALETTE['text_prim']}; font-size: 13px; font-weight: 600;")
        un  = QLabel(unit);  un.setStyleSheet(f"color: {PALETTE['text_mute']}; font-size: 10px;")
        lyt.addWidget(lbl); lyt.addWidget(val); lyt.addWidget(un); lyt.addStretch()
        return (w, val)  # return widget (not layout) to keep alive

    # ── Simulation ─────────────────────────────────────────────
    def _toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            self._play_btn.setText("⏸  Pause")
            speed = [0.5, 1, 2, 4][self._speed_combo.currentIndex()]
            self._play_timer.setInterval(int(80 / speed))
            self._play_timer.start()
        else:
            self._play_btn.setText("▶  Play Simulation")
            self._play_timer.stop()

    def _sim_tick(self) -> None:
        n = 10
        t = np.linspace(self._sim_phase, self._sim_phase + n / self._fs, n)
        self._sim_phase += n / self._fs
        hr = 75 + 10 * np.sin(self._sim_phase * 0.3)
        bf = hr / 60.0
        ecg_new = np.sin(2 * np.pi * bf * t) + 0.05 * self._rng.normal(0, 1, n)
        ppg_new = 0.6 * np.sin(2 * np.pi * bf * t - 0.4) + 0.03 * self._rng.normal(0, 1, n)
        abp_new = 80 + 40 * np.clip(np.sin(2 * np.pi * bf * t), -0.3, 1) + self._rng.normal(0, 1, n)
        self._ecg_buf  = np.roll(self._ecg_buf,  -n); self._ecg_buf[-n:]  = ecg_new
        self._ppg_buf  = np.roll(self._ppg_buf,  -n); self._ppg_buf[-n:]  = ppg_new
        self._abp_ref  = np.roll(self._abp_ref,  -n); self._abp_ref[-n:]  = abp_new
        self._update_plots()

    def _update_plots(self) -> None:
        if not _PG:
            return
        bufs = [self._ecg_buf, self._ppg_buf, self._abp_ref, self._abp_pred]
        x = np.arange(self._buf_len) / self._fs
        for curve, buf in zip(self._curves, bufs):
            curve.setData(x, buf)
        if self._last_result:
            r = self._last_result
            self._sbp_lbl[1].setText(f"{r.sbp:.0f}")
            self._dbp_lbl[1].setText(f"{r.dbp:.0f}")
            self._map_lbl[1].setText(f"{r.map_val:.0f}")
            self._tau_lbl[1].setText(f"{r.tau_ms:.0f}")
            self._rmse_lbl[1].setText(f"{r.rmse:.2f}")
            col = r.risk_color
            self._pheno_label.setText(r.phenotype_name)
            self._pheno_label.setStyleSheet(
                f"background: {PALETTE['bg_panel']}; border: 2px solid {col}; border-radius: 8px; "
                f"padding: 10px; font-size: 14px; font-weight: 700; color: {col};"
            )
            for i, lbl in enumerate(self._prob_labels):
                p = r.phenotype_prob[i] if i < len(r.phenotype_prob) else 0
                lbl.setText(f"{p*100:.1f}%")
            aami_s = "✓ PASS" if r.aami_sbp_pass else "✗ FAIL"
            aami_d = "✓ PASS" if r.aami_dbp_pass else "✗ FAIL"
            sc = PALETTE["success"] if r.aami_sbp_pass else PALETTE["danger"]
            dc = PALETTE["success"] if r.aami_dbp_pass else PALETTE["danger"]
            self._aami_sbp_lbl.setText(f"SBP: {aami_s}")
            self._aami_sbp_lbl.setStyleSheet(f"font-size: 11px; color: {sc};")
            self._aami_dbp_lbl.setText(f"DBP: {aami_d}")
            self._aami_dbp_lbl.setStyleSheet(f"font-size: 11px; color: {dc};")
            self._infer_time.setText(f"Inference: {r.inference_ms:.0f} ms")

    # ── Load NPZ ───────────────────────────────────────────────
    def _load_npz(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load NPZ", "", "NumPy Archive (*.npz);;All (*)")
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=True)
            if "signals" in data.files:
                sig = data["signals"]
                ecg, ppg = sig[0, 0], sig[0, 1]
            elif "ecg" in data.files:
                ecg, ppg = data["ecg"][0], data["ppg"][0]
            else:
                return
            L = min(len(ecg), self._buf_len)
            self._ecg_buf[:L] = ecg[:L]
            self._ppg_buf[:L] = ppg[:L]
            if "abp" in data.files:
                abp = data["abp"][0]
                self._abp_ref[:L] = abp[:L]
            self._update_plots()
        except Exception:
            pass

    # ── Inference ──────────────────────────────────────────────
    def _run_inference(self) -> None:
        ecg = self._ecg_buf.copy()
        ppg = self._ppg_buf.copy()
        abp = self._abp_ref.copy() if self._abp_ref.any() else None
        self._infer_thread = QThread()
        self._infer_worker = InferWorker(self.engine, ecg, ppg, abp)
        self._infer_worker.moveToThread(self._infer_thread)
        self._infer_thread.started.connect(self._infer_worker.run)
        self._infer_worker.result.connect(self._on_inference_done)
        self._infer_worker.result.connect(self._infer_thread.quit)
        self._infer_thread.start()

    def _on_inference_done(self, result) -> None:
        self._last_result = result
        if len(result.abp_waveform) > 0:
            L = min(len(result.abp_waveform), self._buf_len)
            self._abp_pred[:L] = result.abp_waveform[:L]
        self._update_plots()

    # ── LAN data push ──────────────────────────────────────────
    def push_lan_frame(self, frame: dict) -> None:
        ecg = frame.get("ecg"); ppg = frame.get("ppg")
        if ecg is None or ppg is None:
            return
        n = min(len(ecg), self._buf_len)
        self._ecg_buf = np.roll(self._ecg_buf, -n); self._ecg_buf[-n:] = ecg[-n:]
        self._ppg_buf = np.roll(self._ppg_buf, -n); self._ppg_buf[-n:] = ppg[-n:]
        self._update_plots()
        # auto-infer every 5 frames
        if frame.get("frame", 0) % 5 == 0:
            self._run_inference()
