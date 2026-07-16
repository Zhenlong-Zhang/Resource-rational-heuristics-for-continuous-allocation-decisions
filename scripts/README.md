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
- Round 4 diagnostic active-search manual-baseline comparisons

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

On Hoffman2, the original broad active-search array can be submitted with:

```bash
bash scripts/submit_hoffman2_round3_array.sh
```

For the Round 3 follow-up, use:

```bash
bash scripts/submit_hoffman2_round3_followup.sh
```

By default this submits two jobs: a high-parallelism array for `active_search_equal_outcome_narrow_followup`, and a separate 1200-episode Step 7 approximation-method comparison.

If the approximation-method comparison queues for too long as a shared/PE job, prefer the dedicated array submitter:

```bash
bash scripts/submit_hoffman2_round3_methods_array.sh
```

This submits one Step 7 environment per array task and then runs a dependent combine/package job. It is the preferred workflow for the Round 3 1200-episode approximation-method comparison.

## Round 4 Diagnostic Active-Search Checks

The `r4_diagnostics` section is for Falk's 07/01 diagnostic request. It compares:

- `myopic_voi`: current RR approximation
- `manual_active_search_equal_outcome`: hand-coded active-search baseline that samples both recipients and then uses the terminal belief to choose an equal-outcome allocation
- `manual_equal_split`: no-search 50/50 baseline

Tiny wiring smoke:

```bash
python3 scripts/generate_results.py \
  --preset smoke \
  --sections r4_diagnostics \
  --regime-grid r4_diagnostic_active_search \
  --max-regime-grid-points 1 \
  --episodes 2 \
  --voi-samples 2 \
  --common-observations on \
  --observations-per-person 10 \
  --allocation-grid-size 5 \
  --expected-utility-draws 5 \
  --manual-active-samples-per-person 1 \
  --output-dir results/r4_diagnostic_smoke
```

For the full Hoffman2 run, use the one-slot array submitter:

```bash
bash scripts/submit_hoffman2_round4_array.sh
```

The submitter maps the 972-environment grid to 486 independent one-core SGE
array tasks, preserving the established two-environment modulo shards. It limits
the number running at once with `MAX_CONCURRENT_TASKS`
(default: 160). It preserves the full scientific settings: 1200 episodes, 500
VOI samples, common observation streams, and 500 pre-generated observations per
recipient. A dependent collector validates every shard and its provenance before
combining outputs; it never reruns missing simulations.

To inspect progress without starting computation:

```bash
python3 scripts/r4_array_workflow.py progress \
  --manifest results/r4_diagnostic_active_search_server/r4_array_manifest.json
```

For a small Hoffman2 wiring test, override the submission settings without
changing the full-run defaults:

```bash
OUTPUT_DIR=results/r4_array_smoke \
MAX_GRID_POINTS=2 \
EPISODES=3 \
VOI_SAMPLES=4 \
OBSERVATIONS_PER_PERSON=20 \
MANUAL_ACTIVE_SAMPLES_PER_PERSON=1 \
MAX_CONCURRENT_TASKS=2 \
bash scripts/submit_hoffman2_round4_array.sh
```

Main outputs:

- `r4_diagnostic_policy_profiles.csv`: policy-level RR/manual/equal-split behavior and true-state metrics
- `r4_diagnostic_environment_summary.csv`: environment-level manual-vs-equal-split and RR-vs-manual contrasts
- `r4_diagnostic_manual_advantage_candidates.csv`: environments where the manual active-search baseline clearly beats equal split under the current thresholds

Interpretation note: the current utility family can approximate "quickly flattening after needs are met" through stronger concavity, but it does not implement a literal post-threshold plateau.

## Round 3/4 Analysis Report

After the validated R3 and R4 result directories are available locally, create
the paired method analysis, diagnostic-policy summary, and professor-facing HTML:

```bash
python3 scripts/analyze_round3_round4.py \
  --r3-dir results/r3_approximation_methods_checkpointed_1200ep_20260713 \
  --r4-dir results/r4_diagnostic_active_search_server_20260714_array486 \
  --output-dir results/round3_round4_report
```

The R3 input must include both the 700-row method summary and the 840,000-row
episode file. The script retains only the canonical myopic VOI, blinkered, and
DP episodes in memory, verifies common-randomness fingerprints, and computes
paired utility and sample-count confidence intervals. The R4 analysis uses the
validated environment summary and policy-profile tables. It does not rerun any
policy simulation.
