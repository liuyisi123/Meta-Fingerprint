"""Waveform and phenotype metrics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))))


def pearsonr_flat(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = np.asarray(pred).reshape(-1), np.asarray(target).reshape(-1)
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def spearmanr_flat(pred: np.ndarray, target: np.ndarray) -> float:
    if stats is None:
        return float("nan")
    return float(stats.spearmanr(np.asarray(pred).reshape(-1), np.asarray(target).reshape(-1)).correlation)


def dtw_distance(a: np.ndarray, b: np.ndarray, downsample: int = 5, window: int | None = None) -> float:
    a = np.asarray(a, dtype=np.float64)[:: max(1, downsample)]
    b = np.asarray(b, dtype=np.float64)[:: max(1, downsample)]
    n, m = len(a), len(b)
    window = max(window or max(n, m), abs(n - m))
    inf = float("inf")
    prev = np.full(m + 1, inf)
    cur = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur.fill(inf)
        for j in range(max(1, i - window), min(m, i + window) + 1):
            cur[j] = abs(a[i - 1] - b[j - 1]) + min(cur[j - 1], prev[j], prev[j - 1])
        prev, cur = cur, prev
    return float(prev[m] / max(n + m, 1))


def bp_scalars(wave: np.ndarray) -> dict[str, np.ndarray]:
    wave = np.asarray(wave, dtype=np.float32)
    sbp, dbp = wave.max(axis=1), wave.min(axis=1)
    return {"sbp": sbp, "dbp": dbp, "map": dbp + (sbp - dbp) / 3.0, "mean_pressure": wave.mean(axis=1)}


def bland_altman(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def waveform_metrics(pred: np.ndarray, target: np.ndarray, compute_dtw: bool = True, max_dtw_samples: int = 128) -> dict[str, Any]:
    out: dict[str, Any] = {"rmse": rmse(pred, target), "mae": mae(pred, target), "pearson": pearsonr_flat(pred, target), "spearman": spearmanr_flat(pred, target)}
    if compute_dtw and len(pred) > 0:
        out["dtw"] = float(np.mean([dtw_distance(pred[i], target[i]) for i in range(min(len(pred), max_dtw_samples))]))
    ps, ts = bp_scalars(pred), bp_scalars(target)
    for k in ["sbp", "dbp", "map"]:
        out[f"{k}_mae"] = mae(ps[k], ts[k])
        out[f"{k}_rmse"] = rmse(ps[k], ts[k])
        ba = bland_altman(ps[k], ts[k])
        out[f"{k}_bias"] = ba["bias"]
        out[f"{k}_sd"] = ba["sd"]
        out[f"{k}_aami_pass"] = bool(abs(ba["bias"]) <= 5.0 and ba["sd"] <= 8.0)
    return out


def classification_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int | None = None) -> dict[str, float]:
    """Compute accuracy, macro-F1, and OVR-AUROC.

    Macro-F1 is computed over *all* classes in [0, num_classes), not only
    classes present in ``labels``.  Classes absent from ``labels`` contribute
    F1 = 0.0, which matches scikit-learn's ``average='macro'`` behaviour and
    avoids inflated scores when a rare class (e.g. Hypotension) is missing
    from a small evaluation set.
    """
    labels = np.asarray(labels, dtype=np.int64)
    mask = labels >= 0
    if mask.sum() == 0:
        return {}
    logits = np.asarray(logits)[mask]
    labels = labels[mask]
    pred = logits.argmax(axis=1)
    out = {"accuracy": float((pred == labels).mean())}
    n_cls = int(num_classes) if num_classes is not None else int(logits.shape[1])
    f1s = []
    for cls in range(n_cls):
        tp = float(((pred == cls) & (labels == cls)).sum())
        fp = float(((pred == cls) & (labels != cls)).sum())
        fn = float(((pred != cls) & (labels == cls)).sum())
        if tp + fp + fn == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / max(tp + fp, 1.0), tp / max(tp + fn, 1.0)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob = prob / prob.sum(axis=1, keepdims=True)
        if len(np.unique(labels)) > 1:
            out["ovr_auroc"] = float(roc_auc_score(labels, prob, multi_class="ovr"))
    except Exception:
        out["ovr_auroc"] = math.nan
    return out
