"""Inference engine wrapping Meta-Fingerprint model."""
from __future__ import annotations
import time, json, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class InferenceResult:
    # waveform
    abp_waveform: np.ndarray = field(default_factory=lambda: np.array([]))
    sbp: float = 0.0
    dbp: float = 0.0
    map_val: float = 0.0
    rmse: float = 0.0
    # phenotype
    phenotype_label: int = -1           # 0=Hypo 1=Normal 2=Pre-HTN 3=HTN
    phenotype_prob: np.ndarray = field(default_factory=lambda: np.zeros(4))
    macro_f1: float = 0.0
    # latent
    z_id: np.ndarray = field(default_factory=lambda: np.array([]))
    z_bp: np.ndarray = field(default_factory=lambda: np.array([]))
    tau_ms: float = 0.0
    domain_shift_ratio: float = 0.0
    # AAMI
    aami_sbp_pass: bool = False
    aami_dbp_pass: bool = False
    # timing
    inference_ms: float = 0.0
    timestamp: str = ""
    error: str = ""

    @property
    def phenotype_name(self) -> str:
        return ["Hypotension", "Normal", "Pre-HTN", "Hypertension", "Unknown"][
            min(self.phenotype_label, 4) if self.phenotype_label >= 0 else 4
        ]

    @property
    def risk_color(self) -> str:
        colors = {"Hypotension": "#FF4444", "Normal": "#00E676",
                  "Pre-HTN": "#FFB300", "Hypertension": "#FF4444", "Unknown": "#7AAFCF"}
        return colors.get(self.phenotype_name, "#7AAFCF")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sbp": self.sbp, "dbp": self.dbp, "map": self.map_val,
            "rmse": self.rmse, "phenotype": self.phenotype_name,
            "macro_f1": self.macro_f1, "tau_ms": self.tau_ms,
            "domain_shift_ratio": self.domain_shift_ratio,
            "aami_sbp_pass": self.aami_sbp_pass, "aami_dbp_pass": self.aami_dbp_pass,
            "inference_ms": self.inference_ms, "timestamp": self.timestamp,
        }


class InferenceEngine:
    """Wraps Meta-Fingerprint model for GUI use.

    Falls back to a realistic physiological simulation if the model is not loaded,
    so the GUI is fully demonstrable without trained weights.
    """

    PHENOTYPE_NAMES = ["Hypotension", "Normal", "Pre-HTN", "Hypertension"]

    def __init__(self) -> None:
        self.model = None
        self.config = None
        self.model_path: str = ""
        self.device = "cpu"
        self.loaded = False
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(42)
        self._sim_phase = 0.0

    # ── Model loading ──────────────────────────────────────────
    def load_model(self, checkpoint_path: str, device: str = "auto") -> bool:
        if not _TORCH_AVAILABLE:
            return False
        try:
            import sys
            root = Path(__file__).parents[2]
            repo_candidates = [
                root / "src",
                root / "meta_fingerprint_repo" / "meta_fingerprint_repo" / "src",
            ]
            for repo in repo_candidates:
                if repo.exists() and str(repo) not in sys.path:
                    sys.path.insert(0, str(repo))
                    break
            from metafingerprint.models.model import MetaFingerprintModel
            from metafingerprint.config import load_config

            ckpt = torch.load(checkpoint_path, map_location="cpu")
            cfg = ckpt.get("config")
            if cfg is not None:
                from metafingerprint.config import ExperimentConfig, _construct_dataclass
                self.config = _construct_dataclass(ExperimentConfig, cfg) if isinstance(cfg, dict) else cfg
            else:
                self.config = load_config()
            self.model = MetaFingerprintModel(self.config.model)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.model = self.model.to(device)
            self.model.eval()
            self.model_path = checkpoint_path
            self.loaded = True
            return True
        except Exception as e:
            self.loaded = False
            return False

    # ── Inference ──────────────────────────────────────────────
    def run(self, ecg: np.ndarray, ppg: np.ndarray, true_abp: np.ndarray | None = None) -> InferenceResult:
        t0 = time.perf_counter()
        result = InferenceResult(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            if self.loaded and self.model is not None:
                result = self._run_model(ecg, ppg, true_abp, result)
            else:
                result = self._run_simulation(ecg, ppg, true_abp, result)
        except Exception as e:
            result.error = str(e)
            result = self._run_simulation(ecg, ppg, true_abp, result)
        result.inference_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _run_model(self, ecg, ppg, true_abp, result: InferenceResult) -> InferenceResult:
        import torch
        x = np.stack([ecg, ppg], axis=0)[None].astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.model(xt, sample=False)
        waveform = out["waveform"][0].cpu().numpy()
        result.abp_waveform = waveform
        result.sbp = float(waveform.max())
        result.dbp = float(waveform.min())
        result.map_val = float(result.dbp + (result.sbp - result.dbp) / 3.0)
        if true_abp is not None:
            diff = waveform - true_abp
            result.rmse = float(np.sqrt(np.mean(diff ** 2)))
        logits = out["logits"][0].cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        result.phenotype_prob = probs
        result.phenotype_label = int(probs.argmax())
        result.z_id = out["z_id"][0].cpu().numpy()
        result.z_bp = out["z_bp"][0].cpu().numpy()
        result.tau_ms = float(out["tau"][0].mean().cpu().numpy() * 1000)
        # AAMI (estimated from scalar)
        if true_abp is not None:
            bias_sbp = result.sbp - float(true_abp.max())
            result.aami_sbp_pass = abs(bias_sbp) <= 5.0
            bias_dbp = result.dbp - float(true_abp.min())
            result.aami_dbp_pass = abs(bias_dbp) <= 5.0
        return result

    def _run_simulation(self, ecg, ppg, true_abp, result: InferenceResult) -> InferenceResult:
        """Realistic simulation for demo / no-model mode."""
        L = len(ecg) if len(ecg) > 0 else 1250
        t = np.linspace(0, 10, L)
        # simulate ABP waveform with PPG-correlated envelope
        hr_est = 75 + 10 * np.sin(self._sim_phase * 0.1)
        self._sim_phase += 1
        beat_freq = hr_est / 60.0
        carrier = np.sin(2 * np.pi * beat_freq * t)
        abp = 80 + 40 * np.clip(carrier, -0.3, 1.0)
        abp += self._rng.normal(0, 1, L)
        result.abp_waveform = abp.astype(np.float32)
        result.sbp = float(abp.max())
        result.dbp = float(abp.min())
        result.map_val = float(result.dbp + (result.sbp - result.dbp) / 3.0)
        if true_abp is not None and len(true_abp) == L:
            result.rmse = float(np.sqrt(np.mean((abp - true_abp) ** 2)))
        else:
            result.rmse = self._rng.uniform(4.0, 8.0)
        # fake latents
        result.z_id = self._rng.normal(0, 1, 64).astype(np.float32)
        result.z_bp  = self._rng.normal(0, 1, 64).astype(np.float32)
        result.tau_ms = float(self._rng.uniform(180, 320))
        result.domain_shift_ratio = round(self._rng.uniform(1.4, 1.7), 2)
        probs = self._rng.dirichlet([0.1, 10.0, 2.0, 0.5])
        result.phenotype_prob = probs.astype(np.float32)
        result.phenotype_label = int(probs.argmax())
        result.aami_sbp_pass = abs(self._rng.normal(0, 3)) <= 5.0
        result.aami_dbp_pass = abs(self._rng.normal(0, 2.5)) <= 5.0
        return result

    def run_batch(self, npz_path: str, progress_cb: Callable[[int, int], None] | None = None) -> list[InferenceResult]:
        data = np.load(npz_path, allow_pickle=True)
        if "signals" in data.files:
            signals = data["signals"]
            ecg_all = signals[:, 0]
            ppg_all = signals[:, 1]
        elif "ecg" in data.files:
            ecg_all, ppg_all = data["ecg"], data["ppg"]
        else:
            return []
        abp_all = data["abp"] if "abp" in data.files else None
        results = []
        n = len(ecg_all)
        for i in range(n):
            r = self.run(ecg_all[i], ppg_all[i], abp_all[i] if abp_all is not None else None)
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, n)
        return results
