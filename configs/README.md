# Scientific Configurations

## Finite Tutoring Settings

| File | Purpose |
| --- | --- |
| `tutoring_conditions_baseline.json` | Original five-condition finite task with 20-query cap. |
| `tutoring_conditions.json` | Adopted single C3 prior revision; same capped task. |
| `tutoring_design_revision_v02.json` | Baseline/engine integrity identities, single allowed change and rationale; stage seeds and criteria are fixed in `src/design_validation/revision.py`. |
| `tutoring_identifiability_validation.json` | C2 response-rule/noise cells and fixed synthetic recovery stages. |

## Statistical Settings

| File | Purpose |
| --- | --- |
| `tutoring_calibration_v01.json` | Original Student/Welch synthetic populations, 390,000 attempts, scalar checks and inference policies. |
| `tutoring_power_sensitivity_v01.json` | 28 conditional planning scenarios; no final sample-size selection. |
| `tutoring_bounded_robustness_v01.json` | Known-bound sensitivity, 100,000 attempts and selection diagnostics. |
| `tutoring_identifiability_robustness_v01.json` | Variance-adaptive sensitivity, 120,000 attempts, deterministic schedules, C1 identification and cohort bounds. |

Frozen scientific identifiers and seeds are retained for reproducibility; their
version labels are not interchangeable model claims. Do not replace historical
calibration failures with results from a different estimator.

See [finite stages](../notebooks/finite_tutoring/README.md) and
[statistical output mapping](../notebooks/statistical_methods/README.md).
