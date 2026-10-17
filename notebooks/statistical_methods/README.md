# Statistical Methods and Conditional Power

## Expected-GPA and Nominal Inference

[expected_gpa_and_nominal_inference.ipynb](expected_gpa_and_nominal_inference.ipynb)
provides the budget-only expected-GPA target, nominal-test and conditional-power
route. Full mode generates 27 initial states x three deterministic allocation
rules and all 24 sample-size cases. These are initial geometry and planning
references, not new on-policy trajectories. Smoke selects one initial state.
Optional `RAW_TRIALS` runs the synthetic paid-history interface; its required
fields are described in the notebook. Cached target columns are not trusted.

The target minimizes `abs(E[GPA_A] - E[GPA_B])` on the native 1001-point grid.
The signed allocation-distance difference is positive when the choice is closer
to this target than to equal split. Nine fixed nominal tests use alpha .05,
equal-person complete-ten aggregation and prespecified tails; a tenth contrast
is nonprimary. Endpoint definitions, support and missingness rules are in
`configs/expected_gpa_analysis.json` and `src/tutoring_analysis/README.md`.

Power uses declared effects .2/.3/.4 and marginal targets .8/.9 under IID normal
person summaries, with common variance for independent arms. It checks integer
N and N-1, and distinguishes expected retention from 90% count assurance. It
does not validate selected heterogeneous Welch/slope power or select final N.

The 400000-attempt native-state calibration is not reproduced by this notebook:
its native lookup, independent truths, prepared-calibration producer and reader
are a separate dependency chain. No cap20 transfer, human-data validation or
new cost-aware theory result is claimed. The historical routes below remain
separate rather than being silently converted to nominal expected-GPA tests.

## Historical Calibration

Open [calibration_and_power.ipynb](calibration_and_power.ipynb) from a repository checkout. The first code
cell installs the Python package and checks Node.js. Node.js must be installed
separately and is required for the extracted
pure deterministic schedule; no UI engine, survey adapter or private inputs are
used. Computation defaults to `RUN = False`.

The notebook calls the accepted historical Student/Welch, known-bound Hoeffding,
empirical-Bernstein and conditional planning implementations. Full settings retain
the exact cell seeds and batch size 250: 390,000, 100,000 and 120,000 attempts,
respectively. Smoke takes batch zero of each cell and is not independent evidence
in addition to full. Fixture mode is only for tests and changes RNG array shape.

These explicitly historical methods preserve supporting comparisons and negative
calibration evidence. They are not a claim that every historical inference policy
is the latest adopted primary analysis, and must not substitute for a newer
accepted method's matching calibration or power calculations. Method-specific
assumptions stay separate even when both branches share these notebook paths.

## Outputs

| Producer | Recomputed output |
| --- | --- |
| `reproduce_tutoring_statistics.py --analysis inference` | Synthetic rows, historical three-policy fits, fresh finite-metric fixture and bounded fit. These are functionality examples, not human inference. |
| `reproduce_tutoring_statistics.py --analysis planning` | `planning.json`: 28 declared-assumption scenarios, eligibility/variance geometry, marginal power and conditional joint bounds. |
| `calibrate_tutoring_methods.py` | All 30 cells and populations, batch summaries, `calibration_summary.json`, core/policy/power CSV tables and `all_criterion_failures.csv`. |
| `reproduce_tutoring_statistics.py --analysis bounded` | Ten cells, all nine slots including three unsupported slopes, same-draw Student/Welch comparisons, batch parity, route summary and failed criteria. |
| `reproduce_tutoring_statistics.py --analysis variance-adaptive` | Recomputed schedules, 3,072 raw projection corners, unsnapped raw and ideal truth rosters, 30 C1 candidate/comparator states, 135 cohort-bound cases, twelve calibration cells, three methods and all policy/scope diagnostics. |

The historical 390,000-attempt result is **MIXED**, with 111 failed criteria,
including 94 selection-diagnostic rows. The full command regenerates the register
from calculations rather than loading that number or stored results. Preserve
any numerical discrepancies; do not tune seeds, thresholds or truths to reproduce
a desired count. Full Hoeffding and empirical-Bernstein runs had no frozen
criterion failures within their scopes, but can be conservative and do not repair
the historical Student/Welch failures. Low-precision smoke failures remain visible.

Inference here is the historical capped, ten-main-trial analysis. Its original
validator checks a 20-question bound but predates later profile-aware guards;
this notebook does not use it for newer budget-only data. Historical global-Holm
and alternative error policies are not silently replaced by later nominal .05
decisions. Equal raw-surplus resemblance is not the later expected-GPA construct.

The source schedule extraction keeps the original identity strings, seeded
cyrb128/SFC32, shuffle, template multiplicities, side balance and stage indices.
The empirical-Bernstein context is regenerated, never supplied as cached input.
This notebook does not reproduce UI replay panels, reports/figures, human fits,
or unfinished theory. Expected utility remains the decision-performance metric;
the endpoint calibration checks a different, explicitly statistical question.
