# Test Suite

Run all regression tests from the repository root:

```bash
python3 -m unittest discover -s tests
```

Each test module begins with a `Test purpose` docstring. The suite is grouped by the scientific or reproducibility property it protects.

Some solver and evaluation tests are intentionally more computationally demanding than the basic model and workflow tests.

## Model And Policy Semantics

- `test_finite_support.py`: finite-support belief updates, utility, and myopic VOI behavior.
- `test_observation_streams.py`: observations and realized utility use the same episode-specific hidden state.
- `test_positive_need.py`: positive-need environment construction and policy comparison semantics.
- `test_terminal_optimizer.py`: terminal allocation optimization and utility maximization.
- `test_terminal_reference*.py`: independent terminal-solver references, certificates, symmetry, and agreement tolerances.

## Experiment Metrics And Analyses

- `test_active_search_evaluation.py`: objective, information-value, active-search, and solver-comparison metrics.
- `test_active_search_report.py`: report calculations and strict input validation.
- `test_diagnostic_active_search.py`: manual active-search versus equal-split diagnostic criteria.
- `test_strategy_mapping.py`: held-out strategy comparisons, common randomness, and boundary summaries.
- `test_scarcity.py`: lower-need allocation definitions, oracle gates, paired metrics, and
  frozen scarcity classification rules.
- `test_heuristic_map_report.py`: four-row map shape, claim-ledger types, prototype costs,
  and bounded cross-analysis wording.

## Reproducible Interfaces

- `test_gaussian_reference.py`: pinned historical recipe settings, explicit smoke
  scaling, safe scientific-source extraction and refusal to overwrite outputs.
- `test_scarcity_public_runner.py`: direct CLI help and tiny portable smoke outputs.
- Scientific workflow behavior is also covered through the experiment and analysis tests above.

## Finite Tutoring

- `test_tutoring_design_validation.py`: exact finite Bayes, resource accounting, random-stream and independent scalar-oracle checks.
- `test_tutoring_design_revision.py`: single C3 change, fixed stage seeds, configuration/source identity and selection criteria.
- `test_allocation_identifiability.py`: synthetic response generators, group slope and held-out classification semantics.
- `test_tutoring_model_bridge.py`: affine GPA identity and matched continuous-reference audit.
- `test_tutoring_budget_horizon.py`: native arithmetic, paired shared streams,
  scientific budget limits and capped comparison semantics.
- `test_tutoring_terminal_time.py`: terminal-macro costs, reference rate,
  zero-price/exhausted-budget behavior and expected-outcome allocation.
- `test_tutoring_optimality.py`: rational toy-oracle containment, full branching,
  feasible lower bounds and computational-cutoff versus horizon separation.
- `test_finite_reproduction.py`: small real CLI producer-to-consumer chains,
  incompatible or incomplete panel rejection, input guards and checkpoint integrity.

## Statistical Methods

- `test_tutoring_inference.py`: formula oracles, complete-person eligibility and fixed-slot error policies.
- `test_tutoring_calibration.py`: synthetic populations, seeds, checkpoint identity and failure retention.
- `test_tutoring_bounded_inference.py`: known-bound intervals and unavailable endpoint handling.
- `test_tutoring_bounded_robustness.py`: bounded generators and calibration summaries.
- `test_tutoring_empirical_bernstein.py`: variance-adaptive intervals and range assumptions.
- `test_tutoring_identifiability_robustness.py`: raw projection truths, corner bounds and C1 diagnostic cases.
- `test_tutoring_planning.py`: conditional moments, power and sensitivity assumptions.
- `test_tutoring_reproduction.py`: deterministic schedules, fresh-context generation, full diagnostic retention and CLI execution.

The statistical tests require Node.js for schedule generation as well as the
Python dependencies. They do not consume saved results or browser exports.
