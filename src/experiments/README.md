# Experiments

This package contains reusable experiment logic. Command-line parsing, cluster submission, and report packaging remain in `scripts/`.

## General Comparisons

- `compare.py`: basic environment presets and strategy comparisons
- `randomization.py`: common true states and optional common observation streams
- `regimes.py`: final-choice, information-acquisition, behavior-profile, and approximation-method comparisons
- `sweeps.py`: one-dimensional sweeps, utility-regime builders, and targeted regime grids
- `settings.py`: smoke, serious, and server evaluation settings
- `dp_diagnostics.py`: DP resolution and horizon sensitivity checks
- `diagnostics.py`: manual active-search and equal-split benchmark comparisons

Important functions include:

- `compare_rr_to_heuristics_by_final_choice(...)`
- `compare_rr_information_acquisition_to_heuristics(...)`
- `compare_policy_behavior_profiles(...)`
- `compare_rr_approximation_methods(...)`
- `run_one_dimensional_final_choice_sweeps(...)`
- `run_target_regime_search(...)`
- `run_active_search_diagnostic_policy_grid(...)`

## Controlled Active-Search Analyses

- `active_search_evaluation.py`: full-information oracle, fixed-budget information value, frozen RR evaluation, Wilson intervals, and active-search summaries
- `positive_need.py`: finite-support positive-need environments and policy evaluations
- `strategy_mapping.py`: four-way held-out comparisons, controlled `sigma_need` sweeps, and fixed-total-need mechanism diagnostics
- `scarcity.py`: lower-need allocation definitions, the kink-aware full-information oracle,
  paired metrics, and frozen classification rules for the R6 scarcity analysis
- `heuristic_map_report.py`: pure four-row heuristic-map and claim-ledger builders

These modules keep final-choice metrics separate from information-acquisition metrics and support common-random evaluation episodes.

## Evidence boundary

The public package contains model and metric logic only. Generated R6 evidence, historical
aggregate audits, server manifests, and scheduler provenance are not tracked here; the
professor-facing result package is distributed separately.

## Terminal Evidence

The `terminal_*` modules implement independently checked terminal allocation and its evidence pipeline:

- `terminal_validation_suite.py`: frozen validation cases and identities
- `terminal_canonical_provider.py`: accepted canonical base-belief provider
- `terminal_evidence_rows.py`: evidence rows, sidecars, and hashes
- `terminal_execution.py`: task state, leases, and atomic completion
- `terminal_plan_diagnostics.py`: non-scientific plan validation
- `terminal_setup_diagnostics.py`: setup and environment diagnostics
- `terminal_reference_b_process.py`: isolated reference-solver process support
- `terminal_base_migration.py`: historical base-belief migration validation

Reference JSON files under `configs/reference/` are immutable provenance inputs. Historical labels inside those files are part of their accepted hashes and must not be renamed.
