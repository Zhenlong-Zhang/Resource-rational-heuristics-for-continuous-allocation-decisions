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
