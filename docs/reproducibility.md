# Reproducibility Checklist

This checklist separates commands that can be run immediately from analyses
that require controlled data access.

## 1. Environment check

```bash
pip install -e .[dev,csv,metrics]
pytest -q
python scripts/synthetic_smoke.py --out runs/smoke
```

Expected outcome: all tests pass and the smoke test writes `best.pt` under
`runs/smoke/run/`.

## 2. Public Setting-A reproduction

1. Download VitalDB and MIMIC-III-Ext-PPG under their respective licenses.
2. Apply the manuscript quality filters and write patient-disjoint NPZ files.
3. Train on VitalDB:

```bash
python scripts/train.py \
  --data data/vitaldb_windows \
  --config configs/default.yaml \
  --output runs/vitaldb_metafingerprint \
  --device cuda:0
```

4. Evaluate on MIMIC-III-Ext-PPG:

```bash
python scripts/evaluate.py \
  --data data/mimic_setting_a_windows \
  --checkpoint runs/vitaldb_metafingerprint/best.pt \
  --split test \
  --device cuda:0
```

## 3. Setting-C private wearable reproduction

Setting-C requires approved access to the 32-subject RWW cohort. The code path is
the same as public evaluation, but the data must remain outside the public Git
repository:

```bash
python scripts/calibrate.py \
  --support data/rww_subject_001_support.npz \
  --query data/rww_subject_001_query.npz \
  --checkpoint runs/vitaldb_metafingerprint/best.pt \
  --device cuda:0 \
  --inner-steps 3 \
  --inner-lr 0.01
```

Support and query windows must be disjoint. Setting-C should be reported as
CNAP-referenced wearable transfer.

## 4. Figure and table artifacts

The manuscript figures are generated from aggregate prediction and metric
artifacts, not raw private signals. The expected artifact names and table/figure
links are listed in `docs/result_artifacts.md`. Reviewers with access to
restricted artifacts can regenerate the final plots using scripts under
`scripts/figures/`. A minimal aggregate plotting utility is provided as
`scripts/figures/plot_metric_bar.py`.

## 5. What is intentionally absent from the public repository

- Raw clinical waveforms.
- RWW signals or subject-level metadata.
- Checkpoints trained on restricted RWW-derived support/query files.
- Provider-controlled data download credentials.

These omissions are data-governance constraints rather than missing code paths.
Permitted checkpoints can be placed under `model_zoo/` for a private reviewer
package.
