# Meta-Fingerprint

PyTorch implementation of **Meta-Fingerprint: Physics-Grounded Vascular
Disentanglement for Generalizable Cross-Domain Hemodynamic Monitoring**.

This repository is prepared for anonymous review. It contains the model,
training and evaluation code, structural-subspace calibration, preprocessing
utilities, synthetic smoke tests, and documentation that maps the paper protocol
to executable scripts. The clinical RWW cohort is not redistributed because it
is governed by a data-use agreement; the repository provides the exact NPZ
schema and split-manifest format needed to run the same code on approved data.

## What is implemented

- **HTD-SSM encoder**: Neural-ODE morphology front end, ECG-driven bounded
  pulse-transit delay gate, fractional PPG delay interpolation, and a diagonal
  state-space recurrence with linear sequence complexity.
- **Structured TC-CVAE latent model**: `z_id` is the slowly varying structural
  code and `z_bp` is the dynamic hemodynamic code. The loss implements feature
  reconstruction, MI, TC, marginal KL, and same-patient temporal consistency.
- **AdaIN waveform decoder**: `z_bp` supplies beat-level content and `z_id`
  modulates the ABP waveform envelope through channel-wise scale and shift.
- **Risk phenotyping head**: configurable `num_classes`.
  - `configs/default.yaml` / `configs/task_b_3class.yaml` — three-class
    Task-B for ICD-10 cohorts (MC-MED Setting-D): Hypotension / Normal /
    Hypertension.
  - `configs/task_b_4class_abp.yaml` — four-class Task-B for ABP-equipped
    cohorts (Settings A-C, 2017 ACC/AHA thresholds): Hypotension / Normal /
    Pre-HTN / Hypertension.  Use `bp_scalar_labels_4class()` to generate
    labels for these cohorts.
  - `configs/task_a_5class.yaml` — five-class ACC/AHA Task-A chronic staging
    probe.
- **SR-MAML / calibration**: deployment-time adaptation freezes shared modules
  and updates only the structural encoder.

## Repository layout

```text
.
|-- configs/
|   |-- default.yaml              # paper-style 10 s / 125 Hz, Task-B 3-class (ICD-10)
|   |-- task_b_3class.yaml        # Task-B 3-class for ICD-10 cohorts (MC-MED)
|   |-- task_b_4class_abp.yaml    # Task-B 4-class for ABP-equipped cohorts (Settings A-C)
|   |-- task_a_5class.yaml        # Task-A ACC/AHA 5-class phenotype head
|   |-- debug.yaml                # fast CPU smoke-test config
|   `-- synthetic.yaml            # small synthetic example config
|-- docs/
|   |-- data_access.md        # public/private data handling and RWW limits
|   |-- paper_mapping.md      # equations/algorithms -> code mapping
|   |-- result_artifacts.md   # expected artifacts for tables and figures
|   `-- reproducibility.md    # reviewer reproduction checklist
|-- examples/
|   |-- README.md
|   `-- manifests/
|       |-- split_manifest_template.csv
|       `-- setting_protocol_template.csv
|-- scripts/
|   |-- train.py
|   |-- evaluate.py
|   |-- calibrate.py
|   |-- predict.py
|   |-- prepare_npz_from_csv.py
|   |-- make_synthetic_data.py
|   |-- synthetic_smoke.py
|   `-- figures/
|       |-- README.md
|       `-- plot_metric_bar.py
|-- src/metafingerprint/
|   |-- models/
|   |-- data/
|   |-- losses.py
|   |-- adaptation.py
|   |-- evaluation.py
|   `-- train.py
|-- splits/                 # optional review-safe split manifests
|-- model_zoo/              # optional trained checkpoints when releasable
|-- artifacts/              # aggregate metrics and prediction summaries
|-- figures/                # generated figure outputs
`-- tests/
```

## Installation

```bash
git clone <anonymous-repo-url>
cd meta_fingerprint_repo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[dev,csv,metrics]
```

For CUDA training, install the PyTorch build matching your CUDA version before
installing the project.

## Reviewer quick start

Run the synthetic smoke test:

```bash
python scripts/synthetic_smoke.py --out runs/smoke
```

Run the unit tests:

```bash
pytest -q
```

The smoke test creates a tiny synthetic ECG/PPG/ABP dataset, trains one CPU
epoch with `configs/debug.yaml`, writes a checkpoint, and reports waveform,
classification, and delay metrics. It is not intended to reproduce paper
numbers.

## Data format

The loader accepts either a directory with `train.npz`, optional `val.npz` and
`test.npz`, or a single NPZ with a `split` key. The paper protocol uses
non-overlapping **10-second windows at 125 Hz**, so the default sequence length
is `L = 1250`.

Recommended NPZ format:

```python
np.savez_compressed(
    "train.npz",
    signals=signals.astype("float32"),  # [N, 2, L] or [N, L, 2], ECG then PPG
    abp=abp.astype("float32"),          # optional [N, L]
    labels=labels.astype("int64"),      # optional [N], -1 means missing label
    patient_id=patient_ids,             # optional [N]
    domain=domain_ids,                  # optional [N]
    split=split_names,                  # optional train / val / test
)
```

Compatibility format:

```python
np.savez_compressed(
    "windows.npz",
    ecg=ecg_windows.astype("float32"),  # [N, L]
    ppg=ppg_windows.astype("float32"),  # [N, L]
    abp=abp_windows.astype("float32"),  # optional [N, L]
    label=labels.astype("int64"),       # optional [N]
    patient_id=patient_ids,             # optional [N]
    split=split_names,                  # optional train / val / test
)
```

### Generating Task-B labels

For ABP-equipped cohorts (VitalDB, MIMIC-III, UCI, RWW) use the four-class
function with `configs/task_b_4class_abp.yaml`:

```python
from metafingerprint.data import bp_scalar_labels_4class
labels = bp_scalar_labels_4class(abp_windows)  # [N, L] -> [N] in {0,1,2,3}
```

For the ICD-10 MC-MED cohort (Setting-D, no ABP) use the three-class function
with `configs/default.yaml` or `configs/task_b_3class.yaml`:

```python
from metafingerprint.data import bp_scalar_labels
labels = bp_scalar_labels(abp_windows)  # [N, L] -> [N] in {0,1,2}
```

The AAMI SP10 tolerance flags in evaluation output are valid only for
ABP-equipped Settings A-B. Setting-C (RWW, CNAP reference) should be
reported as CNAP-referenced wearable transfer, not as AAMI compliance.

Detailed preprocessing, split, and data-access notes are in
`docs/data_access.md`.

## Training on the paper-style source cohort

```bash
python scripts/train.py \
  --data data/vitaldb_windows \
  --config configs/default.yaml \
  --output runs/vitaldb_metafingerprint \
  --device cuda:0
```

Outputs:

```text
runs/vitaldb_metafingerprint/
|-- best.pt
|-- last.pt
|-- config.yaml
|-- history.json
`-- train_stats.json
```

## Evaluation

```bash
python scripts/evaluate.py \
  --data data/mimic_setting_a_windows \
  --checkpoint runs/vitaldb_metafingerprint/best.pt \
  --split test \
  --device cuda:0
```

The evaluator reports waveform MAE/RMSE/correlation, SBP/DBP/MAP errors,
Bland-Altman/AAMI-style scalar BP tolerance flags, Macro-F1/AUROC when labels
are available, and delay statistics. AAMI-style flags should be interpreted
only for ABP-equipped Settings A-B, matching the manuscript.

## Structural-subspace calibration

For a held-out subject, prepare disjoint support and query NPZ files:

```bash
python scripts/calibrate.py \
  --support data/rww_subject_001_support.npz \
  --query data/rww_subject_001_query.npz \
  --checkpoint runs/vitaldb_metafingerprint/best.pt \
  --device cuda:0 \
  --inner-steps 3 \
  --inner-lr 0.01
```

The calibration routine deep-copies the trained model, freezes every module
except `model.structural_encoder`, and updates only that branch using support
ABP waveform loss plus the structural-anchor regularizer.

## CSV conversion

```bash
python scripts/prepare_npz_from_csv.py \
  --csv data/raw_stream.csv \
  --out data/windows.npz \
  --source-fs 500 \
  --target-fs 125 \
  --window-sec 10
```

Required CSV columns are `ecg` and `ppg`. Optional columns are `abp`,
`patient_id`, `label`, and `domain`. When `source-fs` is higher than
`target-fs`, the converter applies a sixth-order Chebyshev anti-aliasing
filter before resampling. It also applies a basic finite/flat-line/pressure
range quality filter by default; pass `--no-quality-filter` only for debugging.

## Paper reproducibility

- `docs/reproducibility.md` gives the reviewer-facing reproduction checklist.
- `docs/paper_mapping.md` maps paper equations, algorithms, and claims to code.
- `docs/result_artifacts.md` lists expected metric/figure artifacts and which
  paper table or figure they support.
- `examples/manifests/` contains CSV schemas for public split manifests and
  Setting A-D protocol metadata.
- `splits/`, `model_zoo/`, `artifacts/`, and `figures/` contain README files so
  private reviewer packages can add permitted split manifests, checkpoints,
  aggregate metrics, and rendered figures without changing the code layout.

The anonymous review package does not contain private RWW signals or trained
checkpoints derived from restricted data. Checkpoints and aggregate artifacts can
be added under `checkpoints/` and `artifacts/` for a private reviewer upload,
but raw private data should remain outside the repository.

## License

MIT.
