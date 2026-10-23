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
- `tutoring_finite_auxiliary.json`: exact ENGAGEMENT schedule seed, parity/geometry
  counts, complete terminal pooling recipe and numerical-bound display levels.
- `tutoring_terminal_time_v01.json`: ten terminal macros, assumed timing profiles,
  price multipliers and initial/conditional-terminal sensitivity settings.
- `tutoring_cost_aware.json`: frozen repeated selected-macro working-population
  recipe, potential-stream namespace, original decision tolerances and public
  source-generation dependencies. No personal question-time price or query cap.
- `tutoring_cost_aware_selection.json`: explicit example stable coordinates for
  the selected-trial CLI; changing a selection never renumbers or reseeds it.
- `personal_time_reference.json`: separate finite expected-endpoint recipe,
  fixed native source scope and whole-person/equal-schedule aggregation.
- `personal_time_base.json` and `personal_time_instance.json`: explicit request,
  belief-reading, terminal and random-choice operation costs, valuation and
  native random-allocation law. These are reference assumptions, not estimates
  of all human task durations or adoption of a final paper model.
- `personal_time_design.json` and `personal_time_root_identity.json`: native
  source design and exact original scientific root identities. The runner
  regenerates roots; stored identities do not substitute for input generation.
- `personal_time_selection.json`: small original-coordinate selection, distinct
  from the complete 450-endpoint reference.
- `tutoring_optimality_v01.json`: numerical-envelope tolerances, full affordable
  horizon and computational refinement limits. A depth limit bounds refinement,
  not the number of legal information-acquisition actions.

## Statistical Settings

- `conditional_statistics_environment.json`: exact Python, R, R-package and
  Node reference versions for the conditional mixed-model, paired-coverage,
  direct-fit-power and comparison workflows. The same scientific pins are
  packaged with `src/tutoring_statistics/data/`; both must remain consistent.

- `native_design.json`: resolved synthetic budget-only source design used by
  the native calibration; not a human-session or cap20 validation claim.
- `native_calibration.json`: exact 40-subcase roster, seeds, laws, 250-attempt
  batches and 10,000 attempts per subcase (400,000 total). Smoke selects fixed
  batches without changing RNG shapes or truth definitions.
- `native_reference_environment.yml`: Linux x86-64 reference numerical builds.
  Exact native projection bits depend on the numerical environment, not only seeds.
- `native_reference_constraints.txt`: preserve reference NumPy/SciPy versions
  and compatible plotting/table dependencies when installing the native-state
  notebook or the supplementary paired Gamma diagnostic. The paired reference
  is also Python3.10.9/NumPy1.23.5/SciPy1.10.0, distinct from the earlier
  historical calibration environment.

- `expected_gpa_analysis.json`: nine prespecified budget-only nominal tests,
  one nonprimary diagnostic, fixed arm weights, tails, support and eligibility
  rules. This is not the historical multiplicity policy or a calibration output.

- `tutoring_calibration_v01.json`: 30 cells, fixed seeds, historical error
  policies and 390,000 simulation attempts.
- `tutoring_bounded_robustness_v01.json`: ten bounded-generator cells; six
  supported means and three unavailable slopes remain in nine test slots.
- `tutoring_identifiability_robustness_v01.json`: 30-state C1 audit, deterministic
  projection schedules and twelve variance-adaptive calibration cells.
- `tutoring_power_sensitivity_v01.json`: 28 conditional moment/power scenarios,
  not a final recruitment recommendation.
