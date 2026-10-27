# Conditional Statistical Validation

This package generates synthetic inputs, fits the declared models and checks
their numerical outputs independently. The
[statistical notebooks](../../notebooks/statistical_methods/README.md) provide
installation, original population sizes, scientific settings and runnable calls.
The [command-line guide](../../scripts/README.md#conditional-mixed-model-statistics)
describes selected and full reproduction.

## Entry Points

| Module | Purpose |
| --- | --- |
| `reproduction.py` | Null, paired-coverage and direct-fit-power orchestration; immutable inputs, compatible resume, independent admission and complete/partial aggregation. |
| `comparison_runner.py` | Original baseline, same-fit diagnostic and fresh cluster-robust comparison; shared baseline fits are reused rather than redrawn. |
| `comparison_percentage.py` | Synthetic H8/H9 percentage reporting using the original ordered-mean Student estimator; no human-data or fit validation claim. |
| `native_inputs.py` | Regenerate and authenticate the original initial and joint covariate panels. |
| `kernel_entry.R`, `comparison_entry.R` | Actual R fit and saved-artifact readback entry points with reference-package checks. |

## Scientific Components

- `mixed_tests.py`, `restart_decision.py` and `restart_summary.py` implement
  directional tests, the strict-restart decision and its diagnostics. PORT is
  a numerical screen, not a fallback inferential method.
- `joint_design.py`, `joint_generation.py` and `finite_means.py` define native
  population construction, original random streams, support and endpoints.
- `null_reference.py`, `paired_reference.py`, `power_reference.py` and
  `fresh_reference.py` define distinct populations and their reductions.
  `power_origins.py` and `effect_shift.py` preserve reused-input origins and
  actual shifted-response fits.
- `comparison_design.py` and `comparison_reduction.py` define the earlier
  comparison rosters and corrected event counts. `cluster_inference.py`
  contains the shared contrast and paired-transition mathematics.
- `mixed_fit.R`, `null_generation.R`, `null_fit.R`, `restart_diagnostics.R`,
  `joint_fit.R`, `cluster_fit.R` and `population_fit.R` perform the declared
  fits and extraction. `source_transport.R` checks actual R matrices and
  responses before JSON conversion.
- `independent_*.py` and `independent_*_rds.R` independently reconstruct input
  generation, numerical inference, decisions and reductions from scientific
  artifacts. `oracle_gls.py` and `independent_gls.py` compare two known-covariance
  GLS implementations; an oracle is not a human-data estimator.
- `panel_engine.cjs`, `panel_transport.cjs`, `panel_normalizer.cjs`,
  `initial_panel.cjs` and `joint_panel.cjs` regenerate native scripted panels.
  `data/` contains scientific designs, reference covariates and environment
  pins, not simulated responses or fitted results.
- `results_format.py` optionally compares computed quantities with externally
  supplied reference tables. Those tables are not required to calculate results.

## Scope

Every family retains its original model, seeds, missingness rules and error
policy. Exact source/runtime/input bindings protect continuation; a changed
scientific setup requires a new output directory. A selected check cannot be
reported as a full population, and guard failures are not converted to zero
effects or successful fits.

These are conditional budget-only statistical references.
They do not transfer validation to the capped task, establish a final human
analysis or select a recruitment sample size. Preserve adverse coverage and
right-censored power findings alongside successful numerical checks.
