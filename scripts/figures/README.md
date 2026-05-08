# Figure Scripts

The manuscript figures are generated from aggregate metrics and prediction
artifacts rather than raw clinical signals. This folder contains review-safe
figure-generation utilities.

Expected inputs are listed in `docs/result_artifacts.md`. Scripts should read
CSV/JSON artifacts from `artifacts/` and write PDF/PNG outputs to `figures/`.
Raw RWW data must not be committed.

Include only aggregate artifacts that are permitted by the data-use agreements.
If a figure depends on restricted subject-level RWW data, provide a
de-identified aggregate CSV with patient IDs replaced by stable anonymous
identifiers.

`plot_metric_bar.py` is a minimal example for rendering a metric summary from a
long-form aggregate CSV with columns `method`, `setting`, `metric`, and `value`.
