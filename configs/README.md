# Configurations

This directory contains source-controlled scientific settings and output schemas.

- `positive_need_spec.json`: frozen finite-support positive-need environment specification
- `solver_heldout_configs.json`: prespecified held-out solver environments
- `strategy_mapping_environments.json`: selected strategy-mapping environments
- `strategy_mapping_output_schemas.json`: required collected-output schemas
- `scarcity_frozen_settings.json`: scarcity thresholds, numerical settings, and
  episode/stream contracts
- `scarcity_output_schemas.json`: ordered schemas for portable scarcity summaries and
  paired metrics

Generated simulation outputs belong under `results/` and are not source-controlled.

## Finite Tutoring Settings

- `tutoring_conditions_baseline.json` and `tutoring_conditions.json`: baseline
  and revised finite task; the revised C3 centres are [50, 70] rather than [60, 60].
- `tutoring_design_revision_v02.json`: C3-only change, input/source integrity
  checks and rationale. Public hashes bind the scientific-only configuration.
- `tutoring_identifiability_validation.json`: fixed synthetic response-rule,
  motor-noise, lapse and participant-count assumptions for C2 recovery.
- `tutoring_paired_horizon.json`: 400 paired episodes per active template, fixed
  latent/report seed namespaces and the 20-query comparison cap.
- `tutoring_terminal_time_v01.json`: ten terminal macros, assumed timing profiles,
  price multipliers and initial/conditional-terminal sensitivity settings.
- `tutoring_optimality_v01.json`: numerical-envelope tolerances, full affordable
  horizon and computational refinement limits. A depth limit bounds refinement,
  not the number of legal information-acquisition actions.

## Statistical Settings

- `tutoring_calibration_v01.json`: 30 cells, fixed seeds, historical error
  policies and 390,000 simulation attempts.
- `tutoring_bounded_robustness_v01.json`: ten bounded-generator cells; six
  supported means and three unavailable slopes remain in nine test slots.
- `tutoring_identifiability_robustness_v01.json`: 30-state C1 audit, deterministic
  projection schedules and twelve variance-adaptive calibration cells.
- `tutoring_power_sensitivity_v01.json`: 28 conditional moment/power scenarios,
  not a final recruitment recommendation.
