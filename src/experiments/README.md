# Experiments

This package contains reusable experiment logic. Command-line parsing and output orchestration remain in `scripts/`.

## General Comparisons

- `compare.py`: basic environment presets and strategy comparisons
- `randomization.py`: common true states and optional common observation streams
- `regimes.py`: final-choice, information-acquisition, behavior-profile, and approximation-method comparisons
- `sweeps.py`: one-dimensional sweeps, utility-regime builders, and targeted regime grids
- `settings.py`: smoke, serious, and larger evaluation settings
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
  paired metrics, and frozen scarcity classification rules
- `heuristic_map_report.py`: pure four-row heuristic-map and claim-ledger builders

These modules keep final-choice metrics separate from information-acquisition metrics and support common-random evaluation episodes.

## Output Boundary

The package contains model, policy, metric, and reusable experiment logic. Generated outputs
are written under `results/` and are not tracked by Git.
