# Portable Commands

Run from the repository root after installation. Each command supports `--help`.
Outputs belong under ignored `results/` paths; use fresh destinations unless a
runner explicitly supports compatible resume.

| Runner | Purpose |
| --- | --- |
| `validate_tutoring_design.py` | Five-condition capped task, episodes, initial decisions, behavioral summaries and allocation-grid sensitivity. |
| `validate_tutoring_design_revision.py` | Single C3 prior revision with frozen screening/confirmation recipes and integrity checks. |
| `validate_allocation_identifiability.py` | C2 synthetic rule/noise cells, separate group and individual recovery results. |
| `audit_tutoring_model_bridge.py` | Deterministic finite-prior and GPA mapping audit, not a separate Gaussian simulation. |
| `calibrate_tutoring_methods.py` | Original Student/Welch calibration, fixed batches, scalar parity, compatible checkpoint/resume and complete failure readback. |
| `reproduce_tutoring_statistics.py` | Inference examples, conditional planning, known-bound and empirical-Bernstein calibration with source-generated schedule inputs. |

## Finite Task

```bash
python3 scripts/validate_tutoring_design.py \
  --conditions-file configs/tutoring_conditions_baseline.json \
  --stage smoke --output-dir results/finite_baseline_smoke
python3 scripts/audit_tutoring_model_bridge.py \
  --design configs/tutoring_conditions.json --output-dir results/model_bridge
```

Full baseline reproduction uses `--stage precision --sensitivity`. The
[finite notebook](../notebooks/finite_tutoring/design_validation.ipynb) includes
the separate screening/confirmation loops for revision and identifiability.

## Statistics

```bash
python3 scripts/reproduce_tutoring_statistics.py \
  --analysis inference --mode smoke --output results/inference
python3 scripts/reproduce_tutoring_statistics.py \
  --analysis planning --mode smoke --output results/planning
```

The [statistical notebook](../notebooks/statistical_methods/calibration_and_power.ipynb)
calls all full calibration recipes. The formal sizes are 390,000 Student/Welch,
100,000 known-bound and 120,000 variance-adaptive attempts; smoke uses fixed
batch zero, not independent new evidence. Node.js is required for deterministic
variance-adaptive schedule preparation. Preserve failures and unavailable
endpoints; successful execution does not mean all scientific criteria passed.
