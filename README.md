# Resource-Rational Heuristics for Continuous Allocation Decisions

This repository studies when simple allocation heuristics can be resource-rational. The motivating problem is how to divide a continuous resource, such as tutoring time, between two recipients whose needs are uncertain.

The decision-maker may gather information before allocating the resource. Each observation improves a belief about one recipient's need, but consumes time that could otherwise be allocated. The code evaluates the resulting tradeoff between expected utility and the opportunity cost of deliberation.

## Scientific Questions

The project asks when a resource-rational strategy resembles:

- equal division, or a 50/50 split
- equal outcome / maximin allocation
- helping the recipient in greatest need
- proportional allocation to estimated need
- intermediate or information-dependent strategies

Two kinds of behavior are analyzed separately:

- **Final choice:** the allocation made after deliberation stops.
- **Information acquisition:** whether to sample, whom to sample, and when to stop.

Expected average utility is the performance criterion. Allocation distance, realized outcome gaps, sample counts, and policy-behavior rates are diagnostic measures.

## Model

The object-level action is an allocation fraction `a` in `[0, 1]`. Person 1 receives `a * remaining_time`; person 2 receives the rest. Utility is the sum of the recipients' utilities after accounting for latent need, learning efficiency, and asymmetric penalties for unmet need.

The metalevel state contains beliefs about both needs and remaining time. Available metalevel actions are to sample person 1, sample person 2, or terminate and allocate. Sampling cost is represented through elapsed time rather than an additional utility penalty.

The repository supports both Gaussian beliefs and finite-support priors. Gaussian experiments are used for broad heuristic comparisons and parameter sweeps. Finite-support experiments support controlled active-search analyses and independently checked terminal optimization.

## Installation

```bash
git clone https://github.com/Zhenlong-Zhang/Resource-rational-heuristics-for-continuous-allocation-decisions.git
cd Resource-rational-heuristics-for-continuous-allocation-decisions
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Python 3.10 or newer is required.

## Quick Start

Run a small comparison without launching a large parameter sweep:

```bash
python3 scripts/generate_results.py \
  --preset smoke \
  --sections step7 \
  --output-dir results/smoke
```

Run a targeted grid in parallel:

```bash
python3 scripts/run_parallel_experiments.py \
  --preset serious \
  --sections regime_grid \
  --regime-grid equal_outcome_distinct_focused \
  --regime-grid-chunks 8 \
  --common-observations on \
  --max-workers 4 \
  --output-dir results/equal_outcome_distinct
```

The main computational settings are command-line options, including episode count, VOI sample count, observation-stream length, allocation-grid size, integration backend, and DP resolution. Use `--help` on any runner to inspect its full interface.

## Reproducibility Notebooks

Analyses are reproduced by notebooks under `notebooks/`. Each notebook calls the shared source code and exposes its main settings near the top; model logic is not duplicated in notebook cells.

See `notebooks/README.md` for the analysis index. Full configurations are intentionally gated by `RUN = False`, so opening a notebook does not start a large computation.

## Scarcity Analysis

The scarcity analysis adds a fourth heuristic row: prioritizing the recipient with lower
effort-to-goal (the operational lower-need / closer-to-goal rule). The public implementation
keeps the allocation definitions, kink-aware oracle, paired metrics, frozen thresholds, and
pure heuristic-map builders in `src/experiments/`; the corresponding regression tests are
in `tests/`. Start with
`notebooks/round_06/reproduce_round_06.ipynb` for a portable reproduction entry point. The
`scripts/run_scarcity_public.py` runner writes object, development, and
confirmation summaries under `results/`; use `--mode smoke` for a small wiring check or
`--mode serious` for the frozen public episode configuration.

## Repository Structure

- `src/mdp/`: environment configuration, beliefs, utility, Bayesian updates, and metalevel dynamics
- `src/simulator/`: episode execution and dictionary-driven policies
- `src/policies/`: allocation heuristics, information-acquisition policies, myopic VOI, and blinkered policies
- `src/solvers/`: discretized DP, Gauss-Hermite integration, and terminal allocation solvers
- `src/experiments/`: comparisons, common randomization, sweeps, regime searches, metrics, and controlled analyses
- `scripts/`: portable command-line runners, report generation, and validation tools
- `configs/`: scientific configurations and output schemas
- `notebooks/`: interactive reproduction interfaces
- `tests/`: regression, scientific-invariant, workflow, and command-line tests

Generated outputs are written under `results/`, which is ignored by Git. The repository tracks code, prespecified configurations, notebooks, tests, and small scientific input artifacts rather than generated simulation tables.

## Experiment Workflows

The general experiment runner supports:

- final-choice and information-acquisition comparisons
- continuous allocation-distance metrics
- common true states and optional common observation streams
- one-dimensional parameter sweeps
- targeted 50/50 and equal-outcome regime searches
- positive and near-zero utility environments
- myopic VOI, blinkered, discretized-DP, and Gauss-Hermite diagnostics
- manual active-search and equal-split benchmarks

For structured active-search evaluations, use
`scripts/active_search_evaluation_workflow.py`. Portable one-task method comparisons are
available through `scripts/run_method_comparison_task.py`, and the scarcity analysis uses
`scripts/run_scarcity_public.py`. These runners reuse the scientific implementation in
`src/` and write generated artifacts under the ignored `results/` directory.

## Tests

Run the full test suite from the repository root:

```bash
python3 -m unittest discover -s tests
```

Some terminal-solver tests are computationally heavier than the basic model and workflow tests. See `tests/README.md` for the purpose of each test group.

## Interpretation

Smoke runs verify wiring only and should not be treated as scientific evidence. Comparisons should use sufficiently many episodes, common random inputs when possible, and uncertainty intervals appropriate to the claim. Similar utility does not imply similar behavior, and a tolerance-based allocation match should be interpreted alongside continuous allocation and realized-outcome distances.

The RR methods implemented here are approximations unless explicitly identified as independently validated terminal optimization. Results should not be described as exact continuous-state optimal policies without additional proof.

## Documentation

- `src/README.md`: package map
- `scripts/README.md`: command-line workflow map
- `configs/README.md`: configuration and reference-artifact policy
- `notebooks/README.md`: notebook index
- `tests/README.md`: test-purpose index
