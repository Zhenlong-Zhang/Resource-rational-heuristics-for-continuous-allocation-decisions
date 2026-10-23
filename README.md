# Resource-Rational Allocation: 20-Query Model

This branch reproduces the accepted finite tutoring model with at most **20
information-acquisition queries per trial** and its associated statistical
analyses. A decision-maker learns about two recipients' needs before allocating
the remaining tutoring time. Information improves the decision but consumes
time that could otherwise be used for tutoring.

## Scientific Scope

Needs and reports have integer support from 1 to 100. Beliefs use exact discrete
Bayesian updates. The repeated one-step value-of-information policy compares
sampling either recipient with stopping and allocating. It stops no later than
the 20th query, or earlier when sampling is not worthwhile or affordable.
A 1,001-point allocation grid has prespecified 2,001-point sensitivity checks.

Expected utility is the performance criterion. Allocation resemblance,
information acquisition and true-outcome differences are separate diagnostics.
The policy is a myopic approximation, not a globally optimal metalevel policy.

This branch retains the original five-condition validation, the single C3 prior
revision, C2 response-rule recovery and the finite-prior/GPA audit. Its statistical
analyses retain the original estimators, simulation settings and negative
calibration findings. Synthetic validation does not establish human validity.

The [main branch](https://github.com/Zhenlong-Zhang/Resource-rational-heuristics-for-continuous-allocation-decisions/tree/main)
contains the broader research program and later models. Budget-only, terminal-
time and later mixed-model results are not capped-model results and are not
reproduced by this branch. The two branches share organization, not identical
scientific coverage.

## Installation

```bash
git clone --branch 20-limit https://github.com/Zhenlong-Zhang/Resource-rational-heuristics-for-continuous-allocation-decisions.git
cd Resource-rational-heuristics-for-continuous-allocation-decisions
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Python 3.10 or newer is required. The variance-adaptive statistical workflow
also requires Node.js for deterministic schedule generation; no browser or npm
packages are needed. See the [statistical guide](notebooks/statistical_methods/README.md)
for numerical reference versions and interpretation.

## Reproduce Results

| Result family | Notebook |
| --- | --- |
| Five-condition baseline, C3 revision, C2 synthetic identifiability and finite-prior audit | [Finite task validation](notebooks/finite_tutoring/design_validation.ipynb) |
| Inference, 28 conditional planning scenarios, Student/Welch calibration and bounded/variance-adaptive sensitivities | [Statistical calibration and power](notebooks/statistical_methods/calibration_and_power.ipynb) |

Each notebook installs dependencies in its first code cell and calls shared
source code. Execution defaults to `RUN = False`. Inspect the printed commands,
choose smoke or full settings, and enable execution deliberately. Smoke checks
wiring only; it does not reproduce the full reported populations.

Full numerical recipes retain their original seeds, configurations and inference
policies. Generated outputs go under ignored `results/` paths. They are not
stored in Git, and the notebooks do not require cached private results.

A small finite-task check:

```bash
python3 scripts/validate_tutoring_design.py \
  --conditions-file configs/tutoring_conditions_baseline.json \
  --stage smoke --output-dir results/finite_baseline_smoke
```

Use the [notebook index](notebooks/README.md) for complete workflows, the
[CLI guide](scripts/README.md) for commands, and the
[configuration guide](configs/README.md) for exact settings.

## Structure

- `src/design_validation/`: capped task, Bayesian updates, sampling and allocation, task revision, identifiability and finite-prior audit.
- `src/tutoring_analysis/`: participant-level inference, original synthetic calibration, bounded sensitivities and conditional planning.
- `scripts/`: portable result-generating entry points.
- `configs/`: scientific inputs and frozen analysis settings.
- `notebooks/`: thematic source-calling reproduction interfaces.
- `tests/`: numerical, scientific-invariant and reproducibility checks.

## Tests and Limits

```bash
python3 -m unittest discover -s tests
```

See [test purposes](tests/README.md). Keep failed criteria, unavailable endpoints
and informative-selection limitations in the analysis. Similar utility does not
imply identical behavior; synthetic recovery does not identify a human cognitive
rule. Conditional power does not select a final recruitment size, and supporting
sensitivity methods do not replace the original inference policy.

## Documentation

[Source](src/README.md) | [Configurations](configs/README.md) |
[CLI](scripts/README.md) | [Notebooks](notebooks/README.md) | [Tests](tests/README.md)
