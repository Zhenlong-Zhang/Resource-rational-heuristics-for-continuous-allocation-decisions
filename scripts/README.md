# Command-Line Workflows

## Paired Gamma Diagnostic

The Python-only producer and independent scientific reader use public inputs:

```bash
python scripts/reproduce_paired_diagnostic.py produce --batches 0 39 --output results/paired_gamma/producer
python scripts/reproduce_paired_diagnostic.py read --input results/paired_gamma/producer --output results/paired_gamma/reader --require-full
```

This explicitly opts into the fixed500 reproduction. For a selected250 check,
use `--batches 0` or `--batches 39` and omit `--require-full` on readback.
Each selection retains250-shaped full C1-C5 draws before C4 extraction; batch
size and seed are not configurable. Every output directory must be new.
`--config` defaults to the public `tutoring_calibration_v01.json`; changed
Gamma-law parameters are rejected.

The reference runtime is Python3.10.9/NumPy1.23.5/SciPy1.10.0; use the existing
`configs/native_reference_environment.yml` and preserve its versions with
`configs/native_reference_constraints.txt`. It differs from the earlier
calibration environment. Cross-platform bitwise equivalence is not promised.

Producer outputs are `batch_<id>.npz`, `cases_<id>.json`, `plan.json`,
`tests.json`, and `summary.json`. The reader emits `independent_cases.json`,
`independent_tests.json`, and `summary.json`. Summaries are written last;
incomplete, duplicate, malformed or inconsistent scientific records are rejected.
`selectedComplete` does not imply `fixed500Complete`.

The reader also supports original `historical-gamma-diagnostic-v01` producer
packets through the same `read` command. It reconstructs the scientific data,
not operational receipts, and leaves the input untouched. No private wrapper,
source archive or cached result is needed to produce new inputs. Agreement is
not Type-I/coverage acceptance, historical-failure repair or cap20 transfer.

## Native-State Calibration

`reproduce_native_calibration.py` provides three portable stages:

```bash
python scripts/reproduce_native_calibration.py prepare --prepared results/native/prepared
python scripts/reproduce_native_calibration.py run --prepared results/native/prepared --output results/native/smoke --mode smoke
python scripts/reproduce_native_calibration.py summarize --prepared results/native/prepared --output results/native/smoke --mode smoke
```

Preparation generates all source/lookup/truth inputs. Full mode selects all 40
subcases and 40 batches each, preserving 250 attempts per batch. `--workers` runs
independent cells; `--cell` and `--batch-id` select stable resumable units. A
cell has one writer; each completed batch is committed after content validation.
Only one CLI invocation may own an output directory at a time; use `--workers`
for concurrent cells. Repeating
the command reuses compatible completed batches; different code/runtime or
inputs are rejected. After a process crash, inspect its `.writer.lock` before
removing it; never remove a live writer's lock. A fresh preparation directory is
required when preparation was interrupted or source/runtime changed.
`prepare --source-states FILE` may reuse previously generated source states;
their complete finite-state reconstruction is still checked before building
the lookup and truths. It does not accept a cached lookup as a substitute.

The independent summary writes `summary.json` and `targets.csv`. Smoke remains
PROVISIONAL. Full reference dispositions are 89 PASS, 22 UNAVAILABLE, 7
DIAGNOSTIC_ONLY; do not force a different runtime to reproduce them by rounding
truths or changing thresholds. See the native-state notebook for scope.

The scripts call the implementation in `src/`. They should not contain an independent version of the model.

## General Experiments

`generate_results.py` is the main serial runner. It can generate final-choice comparisons, information-acquisition comparisons, behavioral profiles, parameter sweeps, targeted regime grids, DP diagnostics, Gauss-Hermite diagnostics, and SVG summaries.

```bash
python3 scripts/generate_results.py \
  --preset smoke \
  --sections step7 \
  --output-dir results/smoke
```

`run_parallel_experiments.py` runs the same sections as resumable local shards and combines successful outputs.

```bash
python3 scripts/run_parallel_experiments.py \
  --preset serious \
  --sections regime_grid \
  --regime-grid active_search_equal_outcome_focused \
  --regime-grid-chunks 8 \
  --common-observations on \
  --max-workers 4 \
  --output-dir results/active_search_grid
```

`summarize_results.py` creates a compact Markdown summary from existing result folders without rerunning simulations.

`reproduce_gaussian_reference.py` runs the identified historical Gaussian grids
in an isolated extraction of scientific source from a pinned Git commit. Select
one of its six `--recipe` values and `--mode smoke|full`; `--prepare-only` extracts
and prints the command without calculating. Full recipes preserve the original
120/1200 episodes, 500 VOI draws, common streams and grid-specific partitioning.
Use a full Git clone and a fresh output directory. This does not replace the
current Gaussian solver or promise bitwise results across numerical runtimes.

## Scientific Diagnostics

- `check_observation_streams.py`: verifies that observations and realized utility use the same episode-specific hidden state
- `analyze_approximation_active_search.py`: analyzes paired approximation-method results and the constructed active-search benchmark
- `analyze_active_search.py`: builds objective, information-value, confirmation, and solver-comparison analyses
- `generate_active_search_report.py`: validates collected inputs and creates the corresponding HTML report package
- `combine_method_comparison_results.py`: strictly combines episode-level method-comparison tasks

Expected average utility is the performance measure. True-outcome, allocation-distance, and information-acquisition fields are behavioral diagnostics.

## Structured Evaluation Workflows

The following scripts provide reproducible setup, execution, collection, and validation for larger analyses:

- `active_search_evaluation_workflow.py`: oracle, active-search discovery, fixed-budget, confirmation, and held-out solver families
- `run_method_comparison_task.py`: one checkpointed approximation-method comparison task
- `combine_method_comparison_results.py`: combine completed method-comparison task outputs
- `run_scarcity_public.py`: portable object-, development-, and confirmation-stage scarcity evaluation

## Scarcity Analysis

The implementation is in `src/experiments/scarcity.py` and
`src/experiments/heuristic_map_report.py`. `run_scarcity_public.py` is the portable runner
for object, development, and confirmation summaries; the corresponding reproduction
notebook exposes its smoke and serious/full configurations.

The active-search workflow exposes its operations through `--help`. A typical pattern is:

```bash
python3 scripts/active_search_evaluation_workflow.py create --help
python3 scripts/active_search_evaluation_workflow.py run-task --help
python3 scripts/active_search_evaluation_workflow.py progress --help
python3 scripts/active_search_evaluation_workflow.py collect --help
```

Generated outputs belong under `results/` and are ignored by Git.

## Finite Tutoring Analyses

- `validate_tutoring_design.py`: baseline five-condition validation and allocation-grid sensitivity.
- `validate_tutoring_design_revision.py`: the fixed C3 prior revision, independent confirmation and fixed-initial-rule comparison.
- `validate_allocation_identifiability.py`: C2 group responsiveness and individual predictive classification under declared synthetic noise.
- `audit_tutoring_model_bridge.py`: deterministic finite-support, Gaussian-reference and affine-GPA audit.
- `reproduce_finite_tutoring.py`: paired cap20/budget-only episodes, initial
  diagnostics, terminal-time sensitivity and full-affordable-horizon numerical
  bounds. Stages take public scientific settings and regenerate intermediate
  paired data; no private preparation or approval receipt is required. Terminal
  sensitivity is conditional on the supplied old-policy terminals, not a newly
  simulated cost-aware policy. Computational refinement limits are not query caps.

- `reproduce_finite_auxiliary.py`: `source-panel` produces the exact ENGAGEMENT
  schedules, JS parity and geometry; `time-tables` and `bounds-tables` read
  generated finite stages and produce scoped descriptive summaries. Node.js
  is needed only for `source-panel`; use fresh output directories.

## Expected-GPA and Nominal Analysis

`reproduce_expected_gpa.py targets` generates the working budget-only initial
allocation-target panel. `power` generates 24 conditional sample-size references
and a readable summary. `analyze --trials PATH` accepts synthetic raw histories,
reconstructs posteriors and targets, and applies the fixed nine primary tests
plus one nonprimary diagnostic from `expected_gpa_analysis.json`. All subcommands
require `--output` pointing to a new directory. See the expected-GPA notebook
for input format and scope. This is not full native-state calibration, a human
data adapter, or a new cap20 validation.

## Historical Statistical Analyses

- `reproduce_tutoring_statistics.py`: inference examples, conditional planning,
  known-bound calibration and variance-adaptive calibration. Select
  `--analysis inference|planning|bounded|variance-adaptive` and
  `--mode fixture|smoke|full`, with `--output` specifying the result directory.
  Scientific context is generated from source rather than stored results.
- `calibrate_tutoring_methods.py`: the historical 30-cell Student/Welch
  calibration with `--mode smoke|full` and checkpointed `--resume`. It writes
  `all_criterion_failures.csv`; a completed run does not imply all criteria pass.

Fixture mode tests wiring only. Smoke mode uses the first full-run batch rather
than a new independent evidence set. Pure schedule generation requires Node.js.
