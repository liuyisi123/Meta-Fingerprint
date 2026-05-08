# Paper-to-code mapping

| Paper component | Code | Notes |
|---|---|---|
| Neural ODE morphology extractor | `src/metafingerprint/models/ode.py::NeuralODEFrontEnd` | fixed-step RK4, optional spectral normalization |
| bounded ECG-driven delay gate | `src/metafingerprint/models/htd_ssm.py::HTDSSMEncoder.estimate_delay` | sigmoid maps delay into `[tau_min, tau_max]` |
| fractional PPG delay interpolation | `src/metafingerprint/models/htd_ssm.py::fractional_delay` | integer lag plus fractional interpolation; invalid early indices zeroed |
| diagonal SSM recurrence | `src/metafingerprint/models/htd_ssm.py::HTDSSMEncoder.forward` | stable diagonal `A`, ZOH factor `expm1(A dt)/A` |
| factored posterior | `src/metafingerprint/models/latents.py::StructuredLatentEncoder` | `z_id` from `mean(H)`, `z_bp` from temporal Conv1d + attention over full `H` |
| feature-space decoder | `src/metafingerprint/models/decoder.py::FeatureDecoder` | reconstructs `h_bar` from `z_id || z_bp` |
| AdaIN waveform decoder | `src/metafingerprint/models/decoder.py::WaveformDecoder` | `z_id` supplies AdaIN scale/shift; `z_bp` generates content |
| risk classifier | `src/metafingerprint/models/decoder.py::RiskClassifier` | configurable phenotype head; `configs/task_b_4class_abp.yaml` for ABP-equipped Task-B (Settings A-C), `configs/task_b_3class.yaml` for ICD-10 Task-B (Setting-D), `configs/task_a_5class.yaml` for Task-A |
| TC-CVAE decomposition | `src/metafingerprint/losses.py::tc_decomposition` | minibatch estimates of MI, TC, marginal KL |
| temporal consistency | `src/metafingerprint/losses.py::temporal_consistency_loss` | margin loss on same-patient `z_id` pairs |
| total objective | `src/metafingerprint/losses.py::compute_total_loss` | waveform + phenotype + warmed disentanglement loss |
| SR-MAML calibration | `src/metafingerprint/adaptation.py::adapt_structural_branch` | updates only structural branch on support windows |
| FO-MAML optional step | `src/metafingerprint/adaptation.py::first_order_meta_step` | episodic first-order structural meta-update |
| training loop | `src/metafingerprint/train.py::Trainer` | AdamW/cosine schedule, early stopping, checkpointing |
| evaluation | `src/metafingerprint/evaluation.py::evaluate_model` | waveform, scalar BP, classification, delay metrics |
| data and split policy | `docs/data_access.md` and `examples/manifests/` | public/private cohort roles, patient-disjoint split schema, Setting A-D metadata |
| reproducibility checklist | `docs/reproducibility.md` | reviewer commands for smoke tests, public Setting-A, and restricted Setting-C |
| result artifacts | `docs/result_artifacts.md` | expected aggregate files for paper tables and figures |

## Suggested validation checklist

1. Run `python scripts/synthetic_smoke.py --out runs/smoke`.
2. Train a small real subset with `configs/debug.yaml`.
3. Check that `tau` stays inside the configured PTT range.
4. Confirm no patient ID overlap across train/val/test splits.
5. Compare zero-shot evaluation and calibrated evaluation; calibration should only update the structural branch.
6. Use `configs/task_b_4class_abp.yaml` for ABP-equipped cohort Task-B (Settings A-C); use `configs/task_b_3class.yaml` or `configs/default.yaml` for ICD-10 Task-B (Setting-D); use `configs/task_a_5class.yaml` for five-class Task-A.
