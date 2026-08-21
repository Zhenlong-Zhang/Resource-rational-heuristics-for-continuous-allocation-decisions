# Test Suite

Run all regression tests from the repository root:

```bash
python3 -m unittest discover -s tests
```

Each test module begins with a `Test purpose` docstring. The suite is grouped by the scientific or reproducibility property it protects.

Some terminal-solver and evidence tests are intentionally more computationally demanding than the basic model and workflow tests.

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
  and bounded R5/R6 wording.
- `test_scarcity_public_runner.py`: direct CLI help and tiny scheduler-free smoke outputs.

## Reproducible Workflows

- `test_*_workflow.py`: manifest creation, deterministic task identity, strict collection, and failure handling.
- `test_method_comparison_episode_workflow.py`: episode-level pairing and resumable method-comparison tasks.
- `test_quadrature_validation_array.py`: quadrature-validation task identity and collection.
- `test_strategy_mapping_submission.py` and `test_terminal_validation_submission.py`: scheduler command construction and fail-closed submission behavior.

## Terminal Validation Evidence

- `test_terminal_base_migration.py` and `test_terminal_migration_evidence.py`: migration provenance and immutable evidence.
- `test_terminal_canonical_provider.py`: accepted canonical input selection.
- `test_terminal_evidence_rows.py`: row hashes, sidecars, and evidence recomputation.
- `test_terminal_execution.py`: execution state, leases, timeouts, and atomic completion.
- `test_terminal_plan_only.py` and `test_terminal_setup_diagnostics.py`: planning and setup diagnostics without scientific execution.
- `test_terminal_validation_suite.py`: frozen validation design, case coverage, and aggregate acceptance rules.
- `test_terminal_targeted_concurrent.py`: targeted concurrent entrypoint behavior.
