# Data Access and Split Policy

This repository does not redistribute clinical waveforms. It provides the
preprocessing code, NPZ schema, split-manifest templates, and executable model
pipeline used by the manuscript.

## Cohort roles and access links

| Cohort | Manuscript role | Access |
|---|---|---|
| VitalDB | Sole training source; in-domain test | https://vitaldb.net |
| MIMIC-III-Ext-PPG | Setting-A external ABP-equipped transfer | https://doi.org/10.13026/nmwb-6h34 |
| UCI cuffless BP / PPG-BP | Setting-B external ABP-equipped transfer | https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation |
| MC-MED | Setting-D zero-shot ED phenotyping | https://doi.org/10.13026/jz99-4j81 |
| RWW | Setting-C private wearable/CNAP transfer | Not redistributed; available only under approved data-use agreement |

## Windowing and sampling

The paper protocol uses non-overlapping 10-second ECG/PPG windows sampled at
125 Hz. Signals acquired at 500 Hz are downsampled to 125 Hz after sixth-order
Chebyshev anti-aliasing. `scripts/prepare_npz_from_csv.py` applies this
anti-aliasing step and a basic finite/flat-line/pressure-range quality filter.
Dataset-specific SQI fields and provider-specific rejection rules should be
applied before writing final NPZ files when those fields are available.

## Patient-level splits

All train/validation/test splits must be patient-disjoint. For a single NPZ,
include a `split` key with values `train`, `val`, and `test`, or provide
`patient_id` so the loader can create a patient-level split. For paper
reproduction, prefer explicit split manifests under the schema in
`examples/manifests/split_manifest_template.csv`.

## Label conventions

- Task-A: five-class ACC/AHA chronic staging. Use `configs/task_a_5class.yaml`.
- Task-B (ABP-equipped, Settings A-C): four-class acute phenotype
  (Hypotension / Normal / Pre-HTN / Hypertension; 2017 ACC/AHA SBP thresholds).
  Use `configs/task_b_4class_abp.yaml` and `bp_scalar_labels_4class()` to
  generate labels from continuous ABP waveforms.
- Task-B (ICD-10, Setting-D MC-MED): three-class acute phenotype
  (Hypotension / Normal / Hypertension; pre-HTN cannot be inferred from ICD-10).
  Use `configs/task_b_3class.yaml` or `configs/default.yaml`.
- Missing labels should be encoded as `-1`; they are masked during training and
  evaluation.

## Reference-scope rule

AAMI-style scalar BP tolerance flags are meaningful only for ABP-equipped
Settings A-B. Setting-C uses a CNAP non-invasive reference and should be
reported as CNAP-referenced wearable transfer, not as an AAMI compliance test.
