# Test Suite

`test_tutoring_local_strategy.cjs` and `test_tutoring_strategy_other.cjs` cover
the standalone response-state module: stable ordering, transactional storage,
explicit Skip, required Other text, Unicode preservation and legacy export
reading. Run with Node's built-in `--test`; no deployed experiment is implied.

`test_operation_menu*.py` protects exact pair signs and open cost boundaries,
structural rather than overlap ties, named-policy overhead, complete menu
coverage, twelve semantic refusals and explicit saved-input/reuse boundaries.
Synthetic operands test interfaces and do not stand in for historical results.

`test_operation_costs.py` protects single-use signed interval reduction,
positive-upper tiny report masses, H-only actor access, native projection,
typed fee ownership and immutable missing-only checkpoints. Independent
reference modules verify actual native entries separately; synthetic tests do
not establish five-root numerical reproduction or human timing validity.

`test_literal_operation_costs.py` checks paid-VOC operation counts, reached-
history conditioning, independent pricing and interrupted numerical resume.
`test_coefficient_*.py` checks saved-design geometry, no-refit behavior, exact
compact-input admission and portable CLI/checkpoint boundaries.
`test_c5_moments.py` compares independent strict-CDF finite laws, moment matrices,
signed ranges and the original empirical-Bernstein precision floor.
`test_c5_precision_planning.py` checks independent adjacent-integer certificates,
saved-design admission, zero-slope cases, source-bound reuse and the public CLI.
These design-only checks do not draw observations or establish achieved precision.

`test_paper_reproduction.py` checks explicit full/partial scope, source-bound
independent-checkpoint reuse and invalid-bundle refusal. `test_paper_figures_plot.py`
checks the four figures' metric definitions, rounded minimum ties, complete
rosters and fixed-schedule SDs. Fixtures test software behavior; accepted-table
parity and a small real native entry are separate validation evidence.
`test_paper_action_tables.py` checks genuine state/event counts, displayed/logical
recipient mapping, actual routine identity and roster/refusal behavior.
`test_paper_figure_original.py` checks exact historical-input admission and the
explicit separation between historical redraw and a new policy-output epoch.

`test_true_need_diagnostics.py` checks true-need estimands, exact ties, fixed-roster
and partial/undefined behavior, paired slope decomposition and display-order
rule counts. `test_true_need_reproduction.py` checks the portable entrypoint,
fixed recipe and selected-routine source linkage. These are software fixtures,
not new policy simulations or human statistical validation.

`test_cost_aware_campaign.py` checks the complete fixed person/history roster,
shard independence, original endpoint expansion, ten-main eligibility,
person/variant MCSE, partial-study gates and source-bound missing-only resume.
Its fixtures test software invariants, not new trajectory evidence.

`test_magnitude_reduction.py` checks fixed case/seed identities, sign-crossing
truth, unavailable-versus-unresolved outcomes and independently reduced partial
denominators. `test_magnitude_reproduction.py` checks explicit selectors,
no-execution defaults, immutable-input boundaries, scalar-event tampering and
wrong-member resume refusal. They do not generate a scientific population or
establish calibration of the candidate method.

`test_h6_h7_surplus_mapping.py` checks prospective roles, historical payload
preservation, no-rescue and unavailable-result handling, fixed algebra-fixture
reproduction, independent tamper rejection and the portable CLI chain. These
deterministic software fixtures are not human or native-population validation.

The candidate composition and precision-entry tests use pytest:

```bash
python -m pip install pytest
python -m pytest tests/test_magnitude_composition.py tests/test_monte_carlo_precision.py
```

- `test_magnitude_composition.py`: eight fixed supplied-p-value fixtures for
  both D/S orientations, strict thresholds, fixed Holm4, explicit unavailable
  and unresolved states, invalid inputs and candidate-only claims; no fits.
- `test_monte_carlo_precision.py`: read-only CLI help and early runtime,
  overwrite and missing-reader refusals without running a precision scan.

`test_native_calibration.py` checks fixed cell/batch seeds, exact truth coverage,
independent CP/power arithmetic, scalar/vector Normal parity and immutable output
writes. Native source/lookup and independent-truth tests cover the prerequisite
interfaces. These are small regression tests, not a rerun of the formal 400,000
attempts. The native-state notebook exposes the full scientific configuration.

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

- `test_personal_time_diagnostics.py`: target-specific fixed payoffs, source
  identity/recipient/index rejection, rank ties, equal-schedule weighting,
  incomplete rosters, immutable settings and the portable analysis entry.

- `test_personal_time_reference.py` and `test_personal_time_source.cjs`: original
  coordinate/input identity, immutable output, all-report accounting, charged
  random exhaustion, independent decision checks and full/partial finite
  aggregation. Fixtures test wiring; they do not rerun 450 reference decisions.
- `test_personal_time_sensitivity.py`: six-price recipe and CLI, original
  key-derived streams, paid updates, expected-only endpoints, immutable resume,
  missing/tampered record rejection and whole-ten/equal-nine reductions.
  Synthetic fixtures do not rerun the 2,700-endpoint comparison; use the
  personal-time notebook for source-generated trajectories and their checks.
- `test_cost_aware_tutoring.py`: quick selection/JSON/NDJSON rejection, frozen
  settings, CDF boundary/fallback, stream identity, pathwise-bound and independent
  zero-budget checks, plus affordable sub-tolerance ASK/STOP and ask-tie
  disagreements and a tolerated reference-price perturbation. Controlled-menu
  boundary fixtures are not accepted histories. Run the real
  selected `prepare`/`produce`/`check` chain separately via the CLI; these tests do
  not repeat it or establish historical parity.
- `test_cost_aware_source.cjs`: full supported root-coordinate selection and
  reserved paid/exhaustion-fork counter arithmetic, without generating sessions
  or policy histories. Run `node tests/test_cost_aware_source.cjs`; syntax-check
  the adapter with `node --check src/design_validation/cost_aware_source.cjs`.
  Python interface tests additionally check selected-root deduplication and
  rejection of misleading full-panel/completed-session source claims.
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
- `test_finite_auxiliary.py`: deterministic ENGAGEMENT/parity/geometry, descriptive
  denominators, complete/partial coverage, corrupt-input rejection and real CLI chains.

## Statistical Methods

- `test_conditional_statistics.py`: frozen null/paired/power rosters and seeds,
  origin reuse, inference/guard and independent reduction conventions, public
  CLI and immutable output boundaries. Actual representative R fits are separate
  integration checks; neither is a rerun of the formal populations.
- `test_paired_diagnostic.py`: deterministic fixed-shape producer/independent-reader
  integration, stable case IDs, partial/full distinction, frozen law, strict
  JSON/NPZ transport, malformed/duplicate records, boundary conventions and safe
  output paths. No stochastic Gamma batch or formal campaign is run by these tests.

- `test_expected_gpa.py`: target reconstruction, person eligibility, nominal
  tails and missing arms, exhaustion/ties, conditional N boundaries and actual
  synthetic-input/target/power CLI routes.

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

## Statistical Comparison Tests

`test_projection_reference.py` checks isolated numerical-dispatch settings,
cache reuse and rejection of ineffective reference modes. Its fixtures do not
assert cross-platform historical bitwise parity.

`test_saved_coverage.py` and `test_saved_coverage_readback.py` check immutable
target admission, exact roster/CI semantics, independent reduction, relocation
and semantic tampering. `test_saved_fit_magnitude.py`, `test_saved_fit_inputs.py`
and `test_saved_fit_readback.py` check eight-L mathematics, actual-K/retention
constraints, original versus public input identities and independent comparison
rejections. Their public-schema records are explicitly synthetic fixtures, not
fitted models or scientific population evidence. Run these fixture groups with
`python3 -m pytest tests/test_saved_*.py`; pytest is a test-only dependency.

`test_joint_interval_statistics.py` checks the distinct input rosters, seed
domains, immutable commitments, independent reductions and full-versus-partial
denominators. `test_joint_magnitude.py` checks covariance-aware composition and
strict Holm decisions. `test_sign_congruence.py` checks analytic limits,
deterministic integration and conservative unresolved bounds.
`test_known_covariance.py` compares the known-Omega calculation with an
independent QR route and checks invalid/rank-deficient input rejection.
These tests use portable deterministic fixtures, not private fitted results.

`test_statistical_comparisons.py` checks original roster/seed mappings,
baseline-fit reuse, corrected denominators, independent reductions, unavailable
inference and percentage-reporting invariants. Fixtures test software behavior;
they are not a new full calibration or human-analysis validation.

`test_within_publication.py` checks fixed coverage/midpoint populations and
seeds, independent within-person projection, strict decisions and scalar
denominators. It also checks source selection, logical-to-displayed conversion,
native-library identity and full-versus-partial admission. Run with
`python -m unittest discover -s tests -p test_within_publication.py`.
Deterministic fixtures supplement, but do not replace, source-generating entry
checks or formal scientific results.

`test_common_t.py` checks the exact design-sign certificate, independently
implemented common-t tails and Holm decisions, unknown states and fixed-roster
paired summaries. `test_common_t_reproduction.py` checks source rebasing,
immutable input admission, ordinal selection and nonexecuting CLI defaults.
Run with `python -m unittest discover -s tests -p 'test_common_t*.py'`.
These deterministic checks do not repeat the original12129-input comparison
or establish prospective calibration, coverage repair or human power.

`test_strategy_card_fixed.py` checks the display-based rules, selected-index
payoff arithmetic, fixed-two order/fees, independent readback and ambiguity
handling. `test_strategy_card_reproduction.py` checks original coordinates,
fixed denominators, missing/unresolved records, strict comparisons, immutable
entry defaults and per-root locking. Run with
`python -m unittest discover -s tests -p 'test_strategy_card*.py'` in the
card notebook's numerical environment. Fixtures and mocked production wiring
do not generate new adaptive trajectories or establish sampling precision.
