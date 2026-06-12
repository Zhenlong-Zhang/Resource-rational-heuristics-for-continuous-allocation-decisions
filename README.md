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

Use the notebook when you want to inspect outputs interactively. Use the script directly when running on a server.

## Current Round 2 Coverage

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
