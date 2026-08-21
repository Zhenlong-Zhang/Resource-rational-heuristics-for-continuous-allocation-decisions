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

- `test_scarcity_public_runner.py`: direct CLI help and tiny portable smoke outputs.
- Scientific workflow behavior is also covered through the experiment and analysis tests above.
