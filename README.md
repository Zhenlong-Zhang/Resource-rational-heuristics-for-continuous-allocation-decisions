# Resource-Rational Heuristics for Continuous Allocation Decisions

This repository studies when simple allocation heuristics can be resource-rational. The motivating problem is how to divide a continuous resource, such as tutoring time, between two recipients whose needs are uncertain.

The decision-maker may gather information before allocating the resource. Each observation improves a belief about one recipient's need, but consumes time that could otherwise be allocated. The code evaluates the resulting tradeoff between expected utility and the opportunity cost of deliberation.

Start with [installation](#installation), then choose an analysis in the
[thematic notebook index](notebooks/README.md). Each entry links its scientific
settings and generated outputs; the [CLI guide](scripts/README.md) provides
command-line alternatives.

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

The repository contains distinct scientific models, not interchangeable versions
of one task. Continuous Gaussian experiments support broad heuristic comparisons,
regime searches and solver diagnostics. The finite tutoring task instead uses
integer needs and reports from 1 to 100, exact discrete Bayes updates and at most
20 information-acquisition queries per trial. Its 1,001-point allocation grid is
checked against 2,001 points; repeated one-step sampling is not a globally optimal
adaptive policy. The finite cap does not change the older Gaussian comparisons.

## Scientific Scope

`main` is the only maintained branch and contains the latest adopted and
verified research implementation. Start with its thematic notebooks for the
current analyses. Historical results retain their original model, settings and
source identity; a newer method is not a replacement for reproducing an older
calculation.

The historical 20-question capped release is preserved at the fixed
[cap20 archive tag](https://github.com/Zhenlong-Zhang/Resource-rational-heuristics-for-continuous-allocation-decisions/tree/archive/cap20-2026-10-26)
(`dd48add9788ae9245019cbc91fbce30fb9ac1311`), with its matching code, statistical
methods and thematic reproduction notebooks. This is an immutable historical
reference, not a parallel development branch.

The Gaussian analyses are separate models. Horizon comparisons explicitly
distinguish cap20 from budget-only paths; expected-GPA nominal inference,
conditional sample-size references and native-state calibration are budget-only.
These findings do not transfer to the capped model. Historical calibration
failures remain method-specific evidence. A separate
[cost-aware calculation](notebooks/finite_tutoring/cost_aware_computation.ipynb)
generates selected trials or the complete working-population budget-only study,
independently checks decision records, and reduces 51,840 model persons across
18 timing/price scenarios. The original terminal-cost-only model, candidate
expected-GPA resemblance endpoints and high-cost zero/reversed predictions are
preserved. It does not establish global optimality or capped-model validity.

The [personal-time reference](notebooks/finite_tutoring/personal_time_reference.ipynb)
separately prices both information requests and terminal operations. Its fixed
five-condition reference comprises 450 expected endpoints, not 450 realized
histories. A separate six-price comparison retains the original timing proxies
and follows actual acquisition trajectories where the policy searches. Its
2,700 endpoints distinguish 1,860 expected-only stops from 840 realized histories;
these finite points do not establish a continuous boundary or a human-calibrated
price. The mixed-model validation and conditional-power notebooks below use
their own declared synthetic populations and statistical models. Numerical
validation neither transfers these findings to the capped model nor adopts a
final human-study analysis.

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

The statistical reproduction workflows also require Node.js for deterministic
schedule generation, as do the finite horizon schedule/parity panel and the
selected cost-aware calculation. No browser or npm packages are needed.

For notebooks, use a Jupyter environment with this Python kernel. Each notebook's
first code cell installs its dependencies. Native-state calibration and paired
Gamma reproduction use a [pinned numerical environment](notebooks/statistical_methods/README.md#native-state-calibration)
instead of the generic environment above.

Mixed-model null, paired coverage and direct-fit power analyses additionally
require R 4.2.2 and the package versions documented in their
[notebook guide](notebooks/statistical_methods/README.md). These workflows fit
the models and independently check their numerical outputs; saved report tables
are not required inputs.

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

See the [notebook index](notebooks/README.md) for scientific questions, outputs,
producers and [configuration files](configs/README.md). The four families are
heuristic mapping, allocation mechanisms, finite tutoring and statistical methods. Full configurations are
intentionally gated by `RUN = False`, so opening a notebook does not start a
large computation. The objective/solver notebook regenerates its oracle,
sweep, fixed-budget, independent confirmation and paired solver intermediates
before the validated report; it does not require cached result folders.

The [historical Gaussian reference](notebooks/heuristic_mapping/historical_gaussian_grids.ipynb)
preserves the identified source and settings for the earlier 120-episode broad
grids and 1,200-episode focused confirmations. It is distinct from evaluating
the current Gaussian solver: source identity, not only episode count, matters.

The [expected-GPA analysis](notebooks/statistical_methods/expected_gpa_and_nominal_inference.ipynb)
regenerates belief-based allocation targets and 24 conditional power references.
Its optional synthetic-history interface reconstructs targets before applying
the nine prespecified nominal tests. It is not a reproduction of the full
native-state calibration or an analysis of human data. Expected-GPA equality,
raw-surplus equality and maximum combined GPA are distinct objectives.
The same notebook also reproduces the fixed deterministic 25-person algebra
fixture and the prospective H6/H7 mapping: surplus-primary in C2/C3,
expected-GPA-secondary, with no secondary rescue. The mapping preserves its
input tests rather than recalculating or relabeling historical numerical results.
This is a software/inference fixture, not native-task or human validation.

The [native-state calibration](notebooks/statistical_methods/native_state_calibration.ipynb)
provides the complete 400,000-attempt route, including source/lookup generation,
independent exact truths, resumable fixed batches and independent count readback.
Opening the notebook does not run it; small smoke settings remain provisional.
Use its pinned numerical environment for reference-matched projection arithmetic.

The [personal-time reference](notebooks/finite_tutoring/personal_time_reference.ipynb)
regenerates native inputs, prices search and termination consistently, and checks
the finite expected endpoints and original-state acquisition margins. It also
compares equal expected GPA with equal mean surplus on matched examples and
reproduces the six-price original-proxy sensitivity study, including its
unfavorable heuristic diagnostics and undefined slopes.

The [mixed-model null validation](notebooks/statistical_methods/mixed_model_null_validation.ipynb)
provides the fixed 24,000-input H3-H5 analysis, including unavailable fits,
baseline comparisons and independent numerical verification. The
[paired coverage and conditional power](notebooks/statistical_methods/paired_coverage_and_conditional_power.ipynb)
notebook provides the fixed 10,000-input CR2/GLS comparison and the 40-cell,
80,000-input direct-fit power analysis. Full runs remain opt-in. Small selections
check interfaces, not the full reported populations. Coverage limitations and
failure to establish 80% discovery on the tested grid remain results, not reasons
to change the simulation settings.

[Inference comparisons](notebooks/statistical_methods/inference_comparisons.ipynb)
retains the earlier baseline and cluster-robust validation families needed to
interpret those methods. A favorable targeted comparison does not erase earlier
coverage deficits or justify transferring inference to unmatched populations.

[Magnitude composition and precision](notebooks/statistical_methods/magnitude_composition_and_precision.ipynb)
provides the candidate difference/sum p-value combination and independent
exact-binomial Monte Carlo planning. Valid input tests remain a prerequisite;
software agreement and the planning result do not establish magnitude-test
calibration, power or a human recruitment target.

[Saved coverage and contrasts](notebooks/statistical_methods/saved_coverage_and_contrasts.ipynb)
diagnoses original saved intervals for the fixed 2,000-member target and computes
eight contrasts from its first saved fit, with independent readback. It separates
historical archive admission from new public input production. The historical
CR2 coverage of .9395 remains a warning; one-fit numerical agreement is not
population validation.

[Magnitude validation and conditional power](notebooks/statistical_methods/magnitude_validation_and_power.ipynb)
provides the fixed 10,645-input fresh validation and eight-cell, 16,000-input
saved-fit power analysis of the original difference/sum magnitude candidate.
It includes source generation, input-specific numerical checking and independent
aggregation, preserving partial denominators and unfavorable coverage/power results.
Saved power reuses fitted inputs from the conditional-statistics workflow.
These finite studies do not adopt the candidate for human-data analysis or
establish an 80% discovery threshold or a recruitment sample size.

The same notebook also reproduces joint-Gaussian magnitude comparisons on the
saved power inputs and a separate five-law fresh validation, the original
known-covariance interval diagnostic, and a fresh paired CR2/Kenward-Roger
coverage comparison. Each keeps its own input law, seeds and denominator.
Numerical agreement does not establish uniform error control, successful
coverage repair, or suitability for human-data analysis.

## Scarcity Analysis

The scarcity analysis adds a fourth heuristic row: prioritizing the recipient with lower
effort-to-goal (the operational lower-need / closer-to-goal rule). The public implementation
keeps the allocation definitions, kink-aware oracle, paired metrics, frozen thresholds, and
pure heuristic-map builders in `src/experiments/`; the corresponding regression tests are
in `tests/`. Start with
[scarcity notebook](notebooks/heuristic_mapping/scarcity.ipynb) for a portable reproduction entry point. The
`scripts/run_scarcity_public.py` runner writes object, development, and
confirmation summaries under `results/`; use `--mode smoke` for a small wiring check or
`--mode serious` for the frozen public episode configuration.

## Repository Structure

- `src/mdp/`: environment configuration, beliefs, utility, Bayesian updates, and metalevel dynamics
- `src/simulator/`: episode execution and dictionary-driven policies
- `src/policies/`: allocation heuristics, information-acquisition policies, myopic VOI, and blinkered policies
- `src/solvers/`: discretized DP, Gauss-Hermite integration, and terminal allocation solvers
- `src/experiments/`: comparisons, common randomization, sweeps, regime searches, metrics, and controlled analyses
- `src/design_validation/`: finite tutoring task, exact one-step sampling, task revision and rule-identifiability diagnostics
- `src/tutoring_analysis/`: participant-level inference, synthetic calibration, bounded sensitivity analyses, and conditional power
- `src/tutoring_statistics/`: mixed-model null validation, paired CR2/GLS coverage and conditional direct-fit power with independent readback
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

## Finite Tutoring Task

The [finite-task notebook](notebooks/finite_tutoring/design_validation.ipynb)
reproduces the capped task's baseline and revised C3 configurations, both retained
because they support distinct reported analyses. Full mode preserves separate
screening and independent confirmation for both the C3 revision and C2 synthetic
identifiability, with distinct fixed seeds and output readers. It also runs the
finite/Gaussian reference audit; this audit does not rerun the older Gaussian
solver or establish equivalence of the two models.

```bash
python3 scripts/validate_tutoring_design.py \
  --conditions-file configs/tutoring_conditions_baseline.json \
  --stage smoke --output-dir results/finite_baseline_smoke
```

The [statistical notebook](notebooks/statistical_methods/calibration_and_power.ipynb)
preserves historical inference examples, conditional planning, Student/Welch
calibration failures, known-bound and empirical-Bernstein sensitivities,
unavailable endpoints, and a supplementary fixed500 paired Gamma attribution
producer with an independent reader. Its separate `RUN_PAIRED` switch keeps
full reproduction opt-in; a selected250 batch retains the original RNG shape.
Same-input agreement does not repair historical failures or establish cap20
transfer. These method-specific supporting analyses are not
substitutes for a different primary inference policy. Synthetic error control
and conditional power calculations do not establish human validity or select
a final sample size. See the
[notebook index](notebooks/README.md) for the exact scope and scientific settings
of each reported analysis.

The [horizon, terminal-time and bounds notebook](notebooks/finite_tutoring/horizon_terminal_bounds.ipynb)
adds three separately scoped analyses: 2,400 paired cap20/budget-only scenarios,
assumed terminal-time sensitivity at initial and fixed old-policy terminal
beliefs, and numerical Bellman bounds for the zero-terminal-cost affordable
horizon. It regenerates the paired panel from the scientific settings.
Root-refinement depth/time limits do not truncate the scientific horizon, and
the resulting numerical bounds do not certify a globally optimal myopic policy.
See the [finite-task output map](notebooks/finite_tutoring/README.md#horizon-and-terminal-diagnostics)
for raw records and derived summaries.

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

- [Source package](src/README.md): model and analysis modules
- [Command-line workflows](scripts/README.md): runners and output formats
- [Configurations](configs/README.md): scientific settings and reference environments
- [Notebooks](notebooks/README.md): questions, reproduction settings and outputs
- [Tests](tests/README.md): scientific and reproducibility checks
