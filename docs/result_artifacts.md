# Result Artifact Manifest

The repository can train, evaluate, calibrate, and export predictions. Paper
tables and figures should be generated from the following aggregate artifacts.
Raw private signals should not be committed. Permitted aggregate artifacts should
be placed under `artifacts/`, and releasable checkpoints should be placed under
`model_zoo/`.

| Artifact | Producer | Supports |
|---|---|---|
| `artifacts/setting_a_metrics.json` | `scripts/evaluate.py` on MIMIC-III-Ext-PPG | Table 2, Table 3, Fig. 3 |
| `artifacts/setting_b_metrics.json` | `scripts/evaluate.py` on UCI | Table 2, Table 3, Fig. 3 |
| `artifacts/setting_c_loso_metrics.csv` | `scripts/calibrate.py` loop over RWW subjects | Table 2, Table 3, Fig. 3, Fig. 5 |
| `artifacts/setting_d_mcmed_metrics.json` | `scripts/evaluate.py` on MC-MED labels | Table 3, Fig. 5 |
| `artifacts/htdssm_ablation.csv` | repeated `scripts/train.py`/`scripts/evaluate.py` with encoder variants | Table 6, Fig. 7 |
| `artifacts/decoder_ablation.csv` | repeated decoder-variant runs | Table 5, Fig. 6 |
| `artifacts/latent_reliability.csv` | frozen-encoder latent export and analysis | Table 7, Fig. 8 |
| `artifacts/phenotype_clusters.csv` | latent clustering analysis | Table 10, Fig. 10 |
| `artifacts/noise_stress.csv` | RWW corruption and calibration analysis | Fig. 11, Supplementary Table S6 |

## Minimum aggregate schema

Metric files should include:

- `method`
- `setting`
- `track`
- `cohort`
- `seed`
- `patient_id` when patient-level aggregation is possible
- waveform metrics: `rmse`, `mae`, `pearson`, optional `dtw`
- scalar BP metrics: `sbp_bias`, `sbp_sd`, `dbp_bias`, `dbp_sd`, `map_bias`,
  `map_sd`
- phenotyping metrics: `macro_f1`, `auroc`, optional `hypotension_sensitivity`

Patient identifiers should be anonymized before release.
