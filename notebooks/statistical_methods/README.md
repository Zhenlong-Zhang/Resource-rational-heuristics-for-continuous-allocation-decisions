# Statistical Calibration and Conditional Power

Open [calibration_and_power.ipynb](calibration_and_power.ipynb) from a repository
checkout. Its first code cell installs the Python package and reports Node.js
availability. See [installation](../../README.md#installation) and the
[statistical settings](../../configs/README.md#statistical-settings). Node.js
must be installed separately for the pure deterministic schedule helper; no UI
engine, normalizer, private input or external prepared context is required.
Computation defaults to `RUN = False`; generated outputs belong under ignored
`results/` paths.

## Scope and Original Populations

These are the original synthetic method studies and conditional planning
references for the historical capped analysis, not capped task-policy
trajectories, human-study evidence or validation of budget-only nominal methods.
The row validator accepts the historical ten-main-trial pilot profile and
count-cap-20 rule, rejecting explicitly different profile/rule identities even
when observed question counts are at most 20. Model identity and endpoint
definitions matter, not observed counts alone.

| Recipe | Original scope |
| --- | --- |
| Student/Welch | 30 cells: 29 x 10,000 plus one 100,000-attempt low-tail cell, totaling 390,000; seeds 2026100601 through 2026100630. |
| Known-bound Hoeffding | Ten cells x 10,000 = 100,000 attempts; seeds 710071 through 710080; six supported means and three unavailable slope slots. |
| Empirical-Bernstein | Twelve cells x 10,000 = 120,000 attempts; seeds 720071 through 720082; source-bound projections and separate raw/ideal truths. |
| Conditional planning | All 28 declared-assumption scenarios; eligibility, covariance and moment checks, marginal power and conditional joint bounds; no final N selected. |

Gaussian, bounded, heavy-tail and Gamma names in the calibration roster identify
abstract response laws, not separate allocation-task models. The original
`gamma_skew_null` cell remains part of the Student/Welch study.

The original `dose30` sensitivities mean **30 trials per person, not 30 queries
within a trial**. Their `hypothetical_future_profile` labels remain: they are
planning/calibration sensitivities, not current complete-ten pilot observations
or a new task-model study. Early/late windows and available-history populations
likewise remain supporting sensitivities. None replaces the primary population;
no original cell or failed criterion is dropped to improve the result.

Full calibration preserves batch size 250, every cell seed, population, policy
and criterion. Smoke takes batch zero of each cell with the same RNG shape and
overlaps full; it is not additional independent evidence. Fixture mode is for
functionality checks only and changes RNG array shape. Inference examples and
the 28 planning scenarios are deterministic, not extra calibration attempts.

The earlier calibration reference used Python 3.12.14, NumPy 2.3.5 and SciPy
1.16.3. Portable installation is not a promise of bitwise equivalence across
numerical environments. Preserve observed discrepancies rather than adjusting
seeds, thresholds, raw coefficients or truths to match a historical count.

## Outputs

The notebook previews commands before execution and reads only outputs from
its producer commands. Use a fresh output destination; the Student/Welch runner
also supports explicitly requested, source/config/runtime-bound compatible
resume. See the [CLI guide](../../scripts/README.md#statistics).

| Producer | Recomputed output |
| --- | --- |
| `reproduce_tutoring_statistics.py --analysis inference` | Synthetic rows, historical three-policy fits, fresh finite-metric fixture and bounded fit. These are functionality examples, not human inference or policy calibration. |
| `reproduce_tutoring_statistics.py --analysis planning` | `planning.json`: all 28 scenarios, including infeasible-moment and unavailable cases. |
| `calibrate_tutoring_methods.py` | All 30 cells and populations, batch summaries, `calibration_summary.json`, core/policy/power CSV tables, `criteria_readback.json` and `all_criterion_failures.csv`. |
| `reproduce_tutoring_statistics.py --analysis bounded` | Ten cells, all nine slots, same-draw Student/Welch comparisons, scalar parity, route summary and failed criteria. |
| `reproduce_tutoring_statistics.py --analysis variance-adaptive` | Recomputed schedules, 3,072 raw projection corners, separate raw/ideal truth and null rosters, 30 diagnostic-only C1 candidate/comparator states, 135 hypothetical cohort-bound cases, twelve calibration cells, three methods and all policy/scope diagnostics. |

The projection helper preserves the original seed 720061, identity strings,
cyrb128/SFC32 stream, shuffle, template multiplicities, side balance and stage
indices. Prepared context is regenerated, not supplied as cached input. C1
diagnostics do not adopt a task revision. Cohort bounds are not confidence
intervals or a superpopulation-selection repair.

## Negative Findings and Limits

The historical 390,000-attempt result is **MIXED**, with **111 failed criteria**,
including **94 selection-diagnostic rows**. These are historical observations,
not cached notebook outputs or pass thresholds. The full command recomputes the
failure register, including supporting populations, rather than loading that
count. Rejection frequencies use all attempts, including unavailable outcomes;
coverage uses available pointwise intervals with missing counts retained.

Full Hoeffding and empirical-Bernstein runs had no frozen criterion failures
within their declared scopes. They can be conservative and do not repair the
historical Student/Welch failures. Known-bound inference retains three
method-unsupported slopes, so it cannot establish all-nine success. Smoke
failures stay visible; a successful process exit is not a blanket scientific
PASS. Raw floating truths and null rosters are not snapped to ideal zeros.

Historical global-Holm, within-family Holm and four-family alpha-budget policies
remain distinct. Complete-ten record eligibility does not imply complete
endpoint observation. Informative selection changes the target; selected-target
inference does not recover the original population. Equal raw-surplus
resemblance is not an expected-GPA endpoint.

No universal Type-I guarantee, human readiness, final recruitment N, new capped
study or theoretical policy validation is claimed. These interfaces do not
reproduce UI replay or prior report graphics. Expected utility remains the
decision-performance metric; endpoint calibration addresses a separate
statistical question.
