# Experiments

`experiments/` contains structured comparisons and result-generation helpers.

Main modules:

- `compare.py`: earlier environment presets and basic strategy comparisons
- `regimes.py`: Step 7 comparisons for final choice, information acquisition, behavioral profiles, and RR approximation methods
- `randomization.py`: common true states and optional common observation streams
- `settings.py`: smoke, serious local, and server-scale evaluation presets
- `sweeps.py`: one-dimensional parameter sweeps and targeted candidate-regime searches, including focused near-50/50, symmetric equal-outcome, and distinct equal-outcome grids
- `dp_diagnostics.py`: DP sensitivity analysis for investigating discretized DP underperformance
- `r5.py`: true-state equal-outcome metrics, the full-information oracle, fixed-budget information-value curves, and frozen R5 RR evaluation

Important functions:

- `compare_rr_to_heuristics_by_final_choice(...)`: compares RR final allocation against final-choice heuristics
- `compare_rr_information_acquisition_to_heuristics(...)`: compares sampling behavior and utility against heuristic policies
- `compare_policy_behavior_profiles(...)`: reports diagnostic behavior rates such as immediate termination, near-50/50 choices, equal-outcome choices, and sample counts
- `compare_rr_approximation_methods(...)`: compares myopic VOI, blinkered policy, and discretized DP
- `run_one_dimensional_final_choice_sweeps(...)`: varies one environment feature at a time
- `run_targeted_regime_final_choice_grid(...)`: evaluates a targeted multidimensional regime grid
- `run_targeted_regime_behavior_grid(...)`: records RR behavior diagnostics for a targeted multidimensional regime grid
- `identify_final_choice_regime_candidates(...)`: finds candidate conditions where RR resembles known heuristics
- `full_information_oracle_metrics(...)`: compares the utilitarian full-information allocation with true equal outcome and equal split
- `evaluate_r5_rr_environment(...)`: evaluates a frozen RR approximation one episode at a time with common true-state and observation provenance
- `summarize_r5_rr_environments(...)`: reports rates, one-sided Wilson lower bounds, online sample counts, and active-search diagnostics
- `evaluate_r5_fixed_sampling_budgets(...)`: compares fixed balanced observation budgets on paired episodes

Use `scripts/generate_results.py` for the earlier interactive pipeline and
`scripts/r5_array_workflow.py` for frozen Hoffman2 R5 runs.
