# Scripts

`generate_results.py` is the main command-line runner.

It can generate:

- Step 7 final-choice comparisons
- information-acquisition comparisons
- behavioral-profile diagnostics
- RR approximation-method comparisons
- one-dimensional sweeps across environment parameters
- candidate regime tables for near-50/50 and equal-outcome behavior
- DP sensitivity diagnostics
- Gauss-Hermite diagnostics
- simple SVG heatmaps and a run summary
- Round 3 true-state equal-outcome diagnostics in behavior/final-choice tables

Example:

```bash
python3 scripts/generate_results.py --preset smoke --sections all --output-dir results/round2_smoke
```

Scale-up example:

```bash
python3 scripts/generate_results.py \
  --preset server \
  --episodes 1200 \
  --voi-samples 500 \
  --common-observations on \
  --sections step7,sweeps,dp,gh \
  --output-dir results/round2_server
```

Generated outputs are written under `results/`, which is ignored by Git.

`run_parallel_r2.py` is the preferred runner for larger Round 2 jobs. It splits the run into section/environment or section/feature shards, runs shards in parallel, writes one stdout/stderr log per shard, and combines successful shard outputs.

Example:

```bash
python3 scripts/run_parallel_r2.py \
  --preset serious \
  --sections all \
  --max-workers 7 \
  --output-dir results/r2_parallel_serious
```

If a task fails, inspect `parallel_run_status.csv`, `parallel_summary.md`, and the corresponding file under `logs/`.

`summarize_round2_results.py` reads one or more existing result folders and writes a compact Markdown summary. It does not rerun simulations.

Example:

```bash
python3 scripts/summarize_round2_results.py \
  results/r2_full_near_50_50_serious \
  results/r2_confirm_equal_outcome_distinct_1200 \
  --output results/round2_summary.md
```

## Round 3 Checks

`check_observation_streams.py` verifies that common observation streams are tied to the same hidden true state used for realized utility. It also reports the correlation between true needs and observation-stream averages.

Example:

```bash
python3 scripts/check_observation_streams.py \
  --episodes 80 \
  --observations-per-person 80 \
  --output-json results/r3_observation_stream_check.json
```

To smoke-test Falk's active-information-search equal-outcome search path:

```bash
python3 scripts/generate_results.py \
  --preset smoke \
  --sections regime_grid \
  --regime-grid active_search_equal_outcome_focused \
  --max-regime-grid-points 4 \
  --common-observations on \
  --output-dir results/r3_smoke_active_search
```

The relevant Round 3 fields include `true_equal_outcome_rate`, `mean_realized_outcome_gap`, `mean_outcome_distance_to_true_equal`, and `closer_to_true_equal_outcome_than_equal_split_rate`.

Interpretation notes:

- `true_equal_outcome_rate` is true-state based. It counts episodes where the final choice is close to the best feasible realized equal-outcome/maximin outcome, measured by realized outcome gap rather than only by allocation distance.
- `mean_realized_outcome_gap` is the average absolute difference between the two realized outcome-minus-need values.
- `mean_outcome_distance_to_true_equal` is the excess realized outcome gap above the best feasible true-state equal-outcome gap.
- `true_equal_outcome_allocation_close_rate` is the allocation-distance analogue: it counts episodes where the chosen allocation is close to the true-state equal-outcome allocation.
- `closer_to_true_equal_outcome_than_equal_split_rate` compares whether the policy's realized outcome gap is closer to the feasible true-state equal-outcome gap than a 50/50 allocation would be.

For a larger Round 3 local or cluster run, increase both grid size and episodes, for example:

```bash
python3 scripts/run_parallel_r2.py \
  --preset server \
  --sections regime_grid \
  --regime-grid active_search_equal_outcome_focused \
  --episodes 1200 \
  --voi-samples 500 \
  --common-observations on \
  --max-workers 7 \
  --output-dir results/r3_active_search_equal_outcome_server
```

On Hoffman2, prefer the submission wrapper:

```bash
bash scripts/submit_hoffman2_round3.sh
```

By default this submits the observation-stream diagnostic and the active-search true-equal-outcome server grid. To also submit the deferred 10x approximation-method comparison:

```bash
RUN_METHOD_COMPARISON_10X=1 bash scripts/submit_hoffman2_round3.sh
```
