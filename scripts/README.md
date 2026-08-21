# Command-Line Workflows

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

## Scientific Diagnostics

- `check_observation_streams.py`: verifies that observations and realized utility use the same episode-specific hidden state
- `analyze_approximation_active_search.py`: analyzes paired approximation-method results and the constructed active-search benchmark
- `analyze_active_search.py`: builds objective, information-value, confirmation, and solver-comparison analyses
- `generate_active_search_report.py`: validates collected inputs and creates the corresponding HTML report package
- `combine_method_comparison_results.py`: strictly combines episode-level method-comparison tasks

Expected average utility is the performance measure. True-outcome, allocation-distance, and information-acquisition fields are behavioral diagnostics.

## Frozen And Resumable Workflows

The following scripts use manifests and atomic task outputs so large runs can be inspected, resumed, and collected without silently rerunning failed shards:

- `active_search_evaluation_workflow.py`: oracle, active-search discovery, fixed-budget, confirmation, and held-out solver families
- `diagnostic_active_search_workflow.py`: manual active-search versus equal-split benchmark
- `method_comparison_episode_workflow.py`: paired approximation-method episodes
- `positive_need_workflow.py`: finite-support positive-need analyses and quadrature diagnostics
- `quadrature_validation_array.py`: task-level quadrature validation
- `strategy_mapping_workflow.py`: held-out strategy comparison and controlled boundary diagnostics
- `terminal_validation_array.py`: frozen terminal evidence execution and read-back validation

## Public R6 scarcity entry points

The public R6 implementation is in `src/experiments/scarcity.py` and
`src/experiments/heuristic_map_report.py`. `run_scarcity_public.py` is the scheduler-free
runner for object, development, and confirmation summaries; the Round 6 notebook exposes
its smoke and serious/full configurations. Internal manifest collection, report packaging,
scheduler wrappers, and provenance audits are kept outside this repository's public
reproduction surface.

Each workflow exposes its operations through `--help`. A typical pattern is:

```bash
python3 scripts/active_search_evaluation_workflow.py create --help
python3 scripts/active_search_evaluation_workflow.py run-task --help
python3 scripts/active_search_evaluation_workflow.py progress --help
python3 scripts/active_search_evaluation_workflow.py collect --help
```

## Local execution boundary

The public repository does not require a cluster wrapper or scheduler account. Any local
workflow wrapper that is retained for older analyses should be treated as a local execution
adapter, not as part of the R6 reproduction contract.

## Terminal Validation

The terminal-validation scripts implement a stricter evidence path for the finite-support terminal allocation problem:

- `audit_terminal_manifest_setup.py` and related audit scripts validate plans without scientific execution
- `run_terminal_targeted_concurrent.py` runs targeted concurrent checks
- `terminal_validation_array.py` freezes manifests, executes tasks, records scheduler evidence, and performs independent read-back
- `export_terminal_base_migration.py` and its companion shell files preserve a historical one-time migration trust boundary

The accepted migration artifact is already tracked under `configs/`. The historical migration is intentionally non-rerunnable after its approved tool hashes change.

Generated outputs belong under `results/` and are ignored by Git.
