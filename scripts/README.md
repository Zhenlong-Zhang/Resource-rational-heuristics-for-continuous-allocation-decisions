# Command-Line Workflows

Run commands from the repository root after [installation](../README.md#installation).
For an interactive route with settings and output readers, use the
[thematic notebook index](../notebooks/README.md). Scientific inputs are listed
in [configs](../configs/README.md); generated outputs belong under `results/`.

## Selected Cost-Aware Trials

Use the [cost-aware notebook](../notebooks/finite_tutoring/cost_aware_computation.ipynb)
or the following portable stages. Each output directory must be new.

```bash
python scripts/reproduce_cost_aware_tutoring.py prepare --config configs/tutoring_cost_aware.json --selection configs/tutoring_cost_aware_selection.json --output results/cost_aware/inputs --node node
python scripts/reproduce_cost_aware_tutoring.py produce --config configs/tutoring_cost_aware.json --selection configs/tutoring_cost_aware_selection.json --inputs results/cost_aware/inputs --output results/cost_aware/records
python scripts/reproduce_cost_aware_tutoring.py check --config configs/tutoring_cost_aware.json --selection configs/tutoring_cost_aware_selection.json --inputs results/cost_aware/inputs --records results/cost_aware/records --output results/cost_aware/checked --node node
```

`--selection` is a JSON array of objects with exactly `conditionId`,
`scheduleVariant`, `mainIndex`, `replicateId` and `scenarioId`. Coordinates retain
their original randomization identity. Example scenarios are `base|rho_100` and
`low|rho_0`; supported scenarios are low/base/high crossed with rho multipliers
0,.01,.1,1,10,100. Only working roots and original replicate IDs0..63 are supported.
The namespace/recipe is fixed; there is no reseeding, query cap or global mode.

Preparation writes `source_states.json` and `inputs.json`. Source generation
supports every main index1..10 but runs only the selected sessions through their
last requested unpaid root. Genuine checks, practice and baseline ES advancement
are retained. Unneeded paid/exhaustion forks are omitted; their action-counter
increments are reserved so later baseline IDs and timestamps stay unchanged.
`source_states.json` explicitly declares `selected_working_roots`, not a complete
source panel or completed sessions. The existing full-panel native generator is
unchanged. Prepare and check share this selected-source adapter; the checker
still reconstructs schedule and prior mathematics independently. Production writes
`<condition>_v<variant>_m<main>_r<replicate>_<profile>_rho_<multiplier>.ndjson`
and then `reference.json`. Each NDJSON file contains a trial header, initial
decision, latent pair, paid/decision pairs and a final history. A decision embeds
both full100-report panels, distinct branch/observed posterior bits, selected/raw
backups and root bound annotations. The reference retains the roots, reference
rate, final histories and SHA256 of every raw public row excluding its newline.
These public transport identities are not historical manifests or RNG inputs.

The checker regenerates source and independently reconstructs the existing
records, without repeating production. It checks both exact recorded-operand
backups and independently reconstructed full100 backups/policies, including
discrete action/tie agreement. Its price comes from its own reference calculation,
not the approximately compared reported reference. `verification.json` reports the exact
selected scope, numerical errors and completion. Missing/extra/duplicate records,
changed maps/settings/keys, nonfinite values and incomplete output are rejected.
Numerical contradictions remain unavailable, not successful STOP outcomes.
Invalid inputs and numerical noncompletion return exit2; no partial directory
constitutes a complete result. Expected utility, combined GPA, time-priced
objective and scalar realized outcomes remain separate fields.

The numerical reference is Python3.10.9/NumPy1.23.5/SciPy1.10.0; use
`native_reference_environment.yml` and `native_reference_constraints.txt`.
The CLI fixes numerical library threading before imports. Exact probability bits
are environment-sensitive; no cross-platform bitwise promise or historical
acceptance claim follows from a successful check. The two mathematical routes
share NumPy and PCG64. This route excludes person/global reductions, human
inference, new question-time charges and validation under the capped task.

## Personal-Time Reference

`reproduce_personal_time.py` regenerates the native source roots, computes
personal-time-priced initial decisions and checks them with a separate evaluator.

```bash
python scripts/reproduce_personal_time.py all --only C3:0:1,C5:0:1 --output results/personal_time/selected
python scripts/reproduce_personal_time.py all --full --output results/personal_time/full
python scripts/reproduce_personal_time.py examples --output results/personal_time/examples
```

The full recipe has 450 roots and 100 acquisition margins from the 50
variant-zero roots. `prepare`, `produce` and `check` can run separately with
the same selection and output directory. Compatible completed outputs are
verified and reused. Use separate directories for different selections.
Only a complete independently checked roster produces `finite_reference.csv`;
selected checks remain explicitly partial. Other outputs include
`independent_check.json`, `aggregate.json` and `acquisition_margins.csv`.
These are expected endpoints, not sampled realized outcomes or Monte Carlo
precision estimates. The four `examples` distinguish expected-GPA equality
from raw-surplus equality. See the [notebook](../notebooks/finite_tutoring/personal_time_reference.ipynb)
for the reference environment and scientific scope.

## Conditional Mixed-Model Statistics

`reproduce_conditional_statistics.py` calls the original null, paired-coverage
and direct-fit-power producers and independent readers. `plan` is read-only;
numerical work requires `--execute` and an explicit selection.

```bash
python scripts/reproduce_conditional_statistics.py plan
python scripts/reproduce_conditional_statistics.py run --execute --family null --ids 0 --output results/conditional_statistics
python scripts/reproduce_conditional_statistics.py run --execute --family paired --ids 0 --output results/conditional_statistics
python scripts/reproduce_conditional_statistics.py run --execute --family power --ids 0,1,2 --output results/conditional_statistics
```

`--full` instead of `--ids` selects the original 24,000, 10,000 or 80,000
inputs respectively. These settings reproduce declared populations; they are
not interchangeable sample-size knobs. Disjoint ordinal selections can be
combined by `aggregate --execute --family FAMILY --output DIRECTORY`.
Full and partial aggregates are named separately. Actual input generation,
R fits, independent RDS/matrix readback, guarded unavailability and original
seeds are preserved. Compatible resumes reuse committed fits without redraws.
See the [statistical-methods guide](../notebooks/statistical_methods/README.md#mixed-models-paired-coverage-and-direct-fit-power)
for Python/R/Node dependencies, installation, output details and limitations.

`reproduce_statistical_comparisons.py` preserves the distinct baseline,
saved-fit diagnostic and fresh cluster-robust comparison populations:

```bash
python scripts/reproduce_statistical_comparisons.py --family baseline
python scripts/reproduce_statistical_comparisons.py --family baseline --stage all --execute --limit 1 --output results/statistical_comparisons
python scripts/reproduce_statistical_comparisons.py --family diagnostic --stage all --execute --limit 1 --output results/statistical_comparisons
python scripts/reproduce_statistical_comparisons.py --family fresh --stage all --execute --limit 1 --output results/statistical_comparisons
```

Use `--full` to select each original 40,000-, 480- or 26,000-input roster.
Baseline and diagnostic share an output root so the diagnostic reuses the
same saved fits; the fresh population has its own cache namespace and seeds.
The [comparison notebook](../notebooks/statistical_methods/inference_comparisons.ipynb)
also calls the no-fit synthetic H8/H9 percentage-reporting interface. These
conditional comparisons do not establish a human-data method or cap20 validity.

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
packets through the same `read` command, checking their scientific records
without modifying them. Its numerical checks do not authenticate the original
runtime. New inputs are generated from the fixed recipe. Agreement does not
establish Type-I/coverage validity, repair historical failures or validate cap20.

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
  bounds. Stages regenerate intermediate paired data from the scientific
  configurations. Terminal sensitivity is conditional on the supplied
  old-policy terminals, not a newly
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
