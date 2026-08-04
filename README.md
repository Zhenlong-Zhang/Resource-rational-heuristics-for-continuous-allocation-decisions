# Resource-Rational Heuristics for Continuous Allocation Decisions

This repository contains a modular simulator for studying when simple fairness heuristics are resource-rational in continuous allocation decisions. The motivating task is a two-recipient allocation problem, such as deciding how to divide a limited amount of tutoring time between two students whose needs are uncertain.

## Research Goal

The project asks when resource-rational utilitarian decision-making produces behavior that resembles:

- equal division, or 50/50 splits
- equal outcome / maximin allocation
- helping the poorest or greatest-need recipient
- proportional allocation to estimated need
- information-acquisition strategies such as one-and-done, threshold stopping, myopic VOI, blinkered policy, and small-horizon dynamic programming

The current code distinguishes two behaviors that should not be conflated:

- final choice: the allocation selected after deliberation stops
- information acquisition: whether, how long, and whom the policy samples before choosing

Expected average utility is the performance criterion. Allocation distance, sample count, and behavioral-profile metrics are diagnostics.

## Repository Structure

- `src/mdp/`: object-level allocation model and metalevel MDP state/action dynamics
- `src/simulator/`: episode execution helpers and belief-action dictionary policy support
- `src/policies/`: hand-coded final-choice and information-acquisition heuristics plus VOI approximations
- `src/solvers/`: small discretized DP and Gauss-Hermite integration utilities
- `src/experiments/`: comparison, sweep, regime-search, randomization, and DP diagnostic code
- `scripts/`: command-line result generation entrypoints
- `notebooks/`: notebook wrapper for running the same result-generation pipeline interactively
- `results/`: generated output folder, ignored by Git by default

## Quick Start

Run a small smoke test:

```bash
python3 scripts/generate_results.py \
  --preset smoke \
  --sections all \
  --output-dir results/round2_smoke
```

Run a more serious local configuration:

```bash
python3 scripts/generate_results.py \
  --preset serious \
  --sections step7,sweeps,dp,gh \
  --output-dir results/round2_serious
```

Run a server-scale configuration:

```bash
python3 scripts/generate_results.py \
  --preset server \
  --sections step7,sweeps,dp,gh \
  --output-dir results/round2_server
```

Run a targeted regime grid, such as the current distinct equal-outcome search:

```bash
python3 scripts/generate_results.py \
  --preset serious \
  --sections regime_grid \
  --regime-grid equal_outcome_distinct_focused \
  --common-observations on \
  --output-dir results/equal_outcome_distinct_serious
```

For larger grids, use the parallel runner:

```bash
python3 scripts/run_parallel_r2.py \
  --preset serious \
  --sections regime_grid \
  --regime-grid equal_outcome_distinct_focused \
  --regime-grid-chunks 8 \
  --common-observations on \
  --max-workers 4 \
  --output-dir results/equal_outcome_distinct_parallel
```

After downloading or generating result folders, summarize them without rerunning simulations:

```bash
python3 scripts/summarize_round2_results.py \
  results/equal_outcome_distinct_parallel \
  --output results/equal_outcome_distinct_summary.md
```

Create the reproducible Round 3/4 paired analysis and professor-facing report:

```bash
python3 scripts/analyze_round3_round4.py \
  --r3-dir results/r3_approximation_methods_checkpointed_1200ep_20260713 \
  --r4-dir results/r4_diagnostic_active_search_server_20260714_array486 \
  --output-dir results/round3_round4_report
```

This analysis uses episode-level pairing for the canonical R3 approximation
methods and aggregate diagnostic comparisons for R4. It writes a self-contained
HTML report and a `supporting_data/` folder without rerunning simulations.

## Round 5 Workflow

Round 5 separates four questions that should not be collapsed into one result:

- whether the full-information utilitarian objective favors true equal outcome
- whether repeated observations are useful enough to justify active search
- whether an RR approximation discovers that active-search behavior
- whether a prespecified non-myopic DP changes the conclusion

On Hoffman2, submit a frozen array workflow from a clean commit. For example:

```bash
FAMILY=six_sample \
OUTPUT_DIR=results/r5_six_sample_discovery \
EPISODES=120 \
EPISODES_PER_TASK=5 \
OBSERVATION_DRAWS=500 \
SEED_NAMESPACE_OFFSET=0 \
MAX_CONCURRENT=300 \
bash scripts/submit_hoffman2_round5_array.sh
```

The submitter writes a manifest containing the commit, environment grid, seeds,
metrics, and computational settings. A held collector validates all shards before
writing combined episode and summary tables. Inspect progress without rerunning
any simulation:

```bash
python3 scripts/r5_array_workflow.py progress \
  --manifest results/r5_six_sample_discovery/r5_manifest.json
```

After the validated oracle, discovery, independent confirmation, and held-out
solver outputs are available locally, generate the Round 5 report:

```bash
python3 scripts/analyze_round5.py \
  --oracle-dir results/r5_oracle_full \
  --oracle-analysis-dir results/r5_oracle_analysis \
  --formal-dir results/r5_formal_summaries \
  --discovery-dir results/r5_six_sample_discovery \
  --confirmation-dir results/r5_six_sample_confirmation \
  --solver-dir results/r5_solver_comparison \
  --output-dir results/round5_report
```

The report generator does not simulate. It rejects incomplete result folders,
requires independent discovery and confirmation seed namespaces, checks the
1,200-episode/500-VOI confirmation settings, and reports point-estimate discovery
separately from the one-sided Wilson confirmation criterion. The generated
professor-facing package also includes a short README and a copy of the Round 5
notebook that calls this same source-controlled workflow.

You can override the main computational knobs:

```bash
python3 scripts/generate_results.py \
  --preset serious \
  --episodes 120 \
  --voi-samples 500 \
  --common-observations on \
  --observations-per-person 200 \
  --sweep-feature total_time \
  --sweep-feature mu_need \
  --terminal-integration gauss_hermite
```

## Notebook Runner

Open `notebooks/run_round2_pipeline.ipynb`. The notebook installs common Python packages in the first cell, exposes the main knobs as variables, and then calls `scripts/generate_results.py`.

For Round 5, open `notebooks/run_round5_pipeline.ipynb`. It exposes the array
settings, prints or submits the Hoffman2 command, checks manifest progress, and
calls the validated report generator after result folders have been collected.

Use the notebook when you want to inspect outputs interactively. Use the script directly when running on a server.

## Tests

Run the lightweight regression tests with the Python standard library:

```bash
python3 -m unittest discover -s tests
```

The current test suite includes an observation-stream regression check. It verifies that common observation streams are tied to the same hidden true state used for realized utility, and that stream averages are highly correlated with the corresponding true needs.

## Earlier Round 2 Coverage

The current pipeline implements the main items from Falk's second-round feedback:

- continuous final-choice distance metrics: tolerance match rate, mean absolute allocation gap, and RMSE allocation gap
- common true initial states across policies
- optional common observation streams for information-gathering operations
- larger configurable VOI sample counts and episode counts
- positive and near-zero average utility environments
- one-dimensional parameter sweeps from low to high values
- candidate searches for near-always 50/50 and near-always equal-outcome regimes
- focused targeted grids for near-50/50, symmetric equal-outcome, and distinct equal-outcome regimes
- DP sensitivity diagnostics over `max_samples`, `mean_grid_size`, and `observation_branches`
- Gauss-Hermite integration utilities and diagnostics

Prior knowledge is represented as pre-deliberation samples in full episode runs. These samples update the initial belief means and variances before metalevel actions begin, but they do not count as deliberation actions or consume deliberation time.

## Interpretation Notes

Smoke runs are for checking that the code path works. Do not interpret them as evidence.

Local serious runs are useful for preliminary inspection. If confidence intervals overlap substantially, do not claim one policy is better.

Server runs should be used for final claims, especially for approximation-method comparisons and parameter sweeps.
