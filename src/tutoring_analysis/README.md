# Historical Capped Statistical Methods

This package retains the original inference, synthetic calibration and planning
methods for the count-cap-20, ten-main-trial pilot analysis. The row validator
rejects explicit budget-only or newer-profile identities even when a participant
asked no more than 20 questions. Model identity, not observed count alone,
determines applicability. These method studies are not finite-task policy
trajectories or human-study validation.

## Methods and Dependencies

`inference.py` retains person-weighted Student/Welch tests, raw OLS projections,
complete-ten eligibility, nine core slots and three multiplicity policies.
`calibration.py` generates all 30 original cells and their supporting populations:
390,000 attempts with batch size 250 and unchanged cell seeds. Gaussian,
bounded, heavy-tail and Gamma labels describe abstract response laws, not
separate allocation-task models.

`planning.py` provides all 28 declared-assumption scenarios, conditional moment
geometry and power calculations, including infeasible or unavailable cases.
The retained `dose30` sensitivities in calibration/planning mean 30 trials per
person, not 30 queries within a trial. Their original
`hypothetical_future_profile` labels distinguish them from current complete-ten
pilot observations. Early/late and available-history populations remain
supporting sensitivities, not replacements for the primary population or new
task-model studies. Planning assumptions are not estimated from QA or human data;
no final recruitment N is selected.

`bounded_inference.py` and `robustness.py` implement the optional known-bound
six-mean sensitivity, preserving all nine slots and three unavailable slopes.
The original ten-cell study has 100,000 attempts, separate selected/original
targets and same-draw Student/Welch comparisons; it does not replace primary
inference.

`empirical_bernstein.py` and `identifiability_robustness.py` provide the separate
variance-adaptive method, source-known projection domain, finite-pattern
expectations in real arithmetic, raw floating coefficients and cohort bounds.
The original twelve-cell study has 120,000 attempts. Its source context retains
3,072 projection corners, separate raw/ideal null rosters, 30 diagnostic C1
candidate/comparator states without candidate adoption and 135 hypothetical
cohort-bound cases. These bounds are not confidence intervals or an
original-population selection repair. Binary64 calculations are approximations,
not machine certificates; raw truths are not snapped to ideal zeros.

`finite_metrics.py` isolates historical finite-belief metric calculation for
fresh functionality fixtures. `source_projection_panel.cjs` contains pure
deterministic schedule mathematics and requires Node.js. It preserves the
original schedule identities, seed, shuffle and balancing rules. Neither route
needs a UI engine, normalizer, participant export, deployment or network service.
Finite geometry and the C1 audit depend on `design_validation.bounded_task`,
`design_validation.c1_identifiability` and `configs/tutoring_conditions.json`.

## Reproduction and Interpretation

`reproduction.py` orchestrates these historical methods and emits complete
diagnostic summaries. Its producers recompute every consumed scientific context.
Use `scripts/reproduce_tutoring_statistics.py` and
`scripts/calibrate_tutoring_methods.py`; the
[statistical notebook guide](../../notebooks/statistical_methods/README.md)
maps settings, original scales and outputs. Both checkout entrypoints explicitly
prepend `src/` before importing `tutoring_analysis` and `design_validation`.
The statistics tests do the same, without depending on test order or PYTHONPATH.
NumPy/SciPy provide the Python numerical runtime; Node.js supplies deterministic
schedule generation. All outputs belong under ignored `results/`, not in version
control. Smoke overlaps full batch zero; alternate-sized fixtures are not a
prefix of the scientific run.

The historical Student/Welch result remains MIXED with 111 failed criteria,
including 94 selection-diagnostic rows. Failure enumeration preserves every
primary/supporting population and policy, all-attempt rejection denominators,
available-interval coverage denominators and missing counts. Successful
execution does not mean all criteria pass. The bounded and empirical-Bernstein
studies' absence of frozen failures within their scopes is not a universal
guarantee or a repair of those historical failures.

Historical global-Holm and supporting alternatives retain their original
definitions. Complete-ten records need not have every endpoint observed;
informative selection does not become ignorable. Equal raw-surplus resemblance
is not the later expected-GPA construct. These results are neither validation
transfers to budget-only methods nor new capped studies.
