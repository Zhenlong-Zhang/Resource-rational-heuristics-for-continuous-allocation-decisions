# Statistical Methods and Conditional Power

Choose a method below, inspect its settings, then enable computation in the
notebook. [Installation](../../README.md#installation) and the
[configuration index](../../configs/README.md#statistical-settings) describe the
shared dependencies and inputs. Outputs are generated under ignored `results/`.

## Mixed Models, Paired Coverage and Direct-Fit Power

[mixed_model_null_validation.ipynb](mixed_model_null_validation.ipynb) calls
the original twelve null cases x 2,000 inputs, baseline/strict-restart/PORT
fits, independent actual-RDS extraction, strict decision and case reduction.
[paired_coverage_and_conditional_power.ipynb](paired_coverage_and_conditional_power.ipynb)
calls the original 10,000-input fixed154 paired CR2/GLS validation and the
40-cell x 2,000 direct-fit power reference. All full/run switches default off.
Both regenerate the actual 450-row initial and 270-row joint covariate panels
through the native engine and normalizer. Stored covariates authenticate those
generated scientific inputs; they are not responses or fitted results.

The shared implementation is `src/tutoring_statistics/`. Python/R numerical
definitions, independent readers, native CJS generation and scientific JSON
inputs are packaged together. Existing capped/nominal methods are unchanged.
These are budget-only conditional references, not new cap20
validation, a selected human-analysis method or a recruitment recommendation.

### Reference Environment

Use Python **3.10.9**, NumPy **1.23.5**, SciPy **1.10.0**, R **4.2.2**, Node
**22** (the measured reference used 22.14.0) and GNU coreutils **sha256sum** on
PATH. A POSIX environment is required for advisory file locks and atomic hard
links. The runner refuses a missing `sha256sum` before creating output. Exact R package
versions are listed in `configs/conditional_statistics_environment.json` and
the matching packaged `data/numerical_environment.json`; the R entry checks
them before fitting. Use a separate environment/library to avoid upgrading an
unrelated project. R's source builds require compatible C/C++/Fortran tools.

```bash
python -m pip install -e . numpy==1.23.5 scipy==1.10.0 pandas==2.0.3 matplotlib==3.7.5
```

With R 4.2.2 selected, the following explicit setup recipe installs the declared
versions into a separate local library. Obtain R, Node and coreutils separately.
`remotes` is an installation helper, not a numerical dependency. Package-source
availability, compilers and BLAS compatibility still require verification;
this recipe is not a claim that a fresh installation was tested.

```r
lib <- file.path(getwd(), "results", "conditional_statistics_R_library")
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
install.packages("remotes", lib=lib, repos="https://cloud.r-project.org")
remotes::install_version("jsonlite", version="1.8.8", lib=lib,
                         repos="https://cloud.r-project.org", upgrade="never")
versions <- jsonlite::read_json("configs/conditional_statistics_environment.json")$statistical_reference$R_packages
order <- c("Rcpp", "RcppEigen", "Matrix", "minqa", "nloptr", "numDeriv",
           "rlang", "lifecycle", "ggplot2", "lme4", "lmerTest", "zoo",
           "sandwich", "clubSandwich")
for (name in order) remotes::install_version(name, version=versions[[name]],
    lib=lib, repos="https://cloud.r-project.org", dependencies=NA, upgrade="never")
```

Expose that library to subsequent Rscript processes with `R_LIBS_USER`, or use
an already provisioned matching reference library. Versions alone do not prove
identical numerical binaries. Do not widen tolerances or change seeds when a
platform differs. The first notebook code cell installs the Python package and
pins; R/Node/coreutils are explicit prerequisites, not silently installed.

### Source-Calling Commands

```bash
python scripts/reproduce_conditional_statistics.py plan
python scripts/reproduce_conditional_statistics.py run --execute --family null --ids 0 --output results/conditional_statistics
python scripts/reproduce_conditional_statistics.py run --execute --family paired --ids 0 --output results/conditional_statistics
python scripts/reproduce_conditional_statistics.py run --execute --family power --ids 0,1,2 --output results/conditional_statistics
```

Replace the explicit `--ids` selection with `--full` to reproduce the entire
original family. Alternatively use `--first ORDINAL --count COUNT`; selectors
are zero-based, ordered and unique. `--rscript` and `--node` select executables.
`--no-aggregate` defers reduction across disjoint selections; afterwards use
`aggregate --execute --family FAMILY --output results/conditional_statistics`.
Concurrent overlapping members are refused. Compatible resumes validate all
committed inputs/artifacts and reuse them without redraws or additional fits;
partial/conflicting attempts are preserved and refused, never silently retried.

Full results appear as `FAMILY/aggregate_full.json` only when the exact expected
roster is present; selected checks write `aggregate_partial_<roster>.json`.
Artifacts include actual requests, fits, independent RDS extraction and exact
in-R matrix/response transport with a one-ULP refusal check. Source snapshots
and runtime/native identities bind each reproduction. Independent reductions
work without report tables. Optional `--reference-dir DIRECTORY` compares
external CSVs only for full results; not-requested is not a comparison PASS.

### Scientific Boundaries

Retain strict-only candidate inference, mandatory PORT screens, original
optimizer guards and H3/H4 greater versus H5 less tails. The historical null
reference contained 23,992 usable and eight guard-unavailable outcomes. Paired
CR2/GLS uses terminal-amount S2, 154 complete people per arm and distinct
Holm4 family events, Holm2 paired tests and simultaneous Monte Carlo bounds.
Known-Omega GLS is an oracle, not an available human estimator.

Power retains the original 4,000 fixed154 baseline-prefix inputs, sixteen
actual-shift inputs and new whole-person retention streams, not an all-new
80,000-input redraw. Shifts receive actual fits; the failed coefficient-
translation shortcut and withheld 64,000 derived points are not reinstated.
Original all-count pointwise CP precision and independent beta inversion select
B=2,000. Preserve three guard-unavailable power outcomes, both literal and
sign-consistent tails, adverse initial-amount S1 random-retention coverage
(C1-C2:B, 1,879/2,000) and the right-censored discovery grid. No tested multiplier
establishes pointwise-supported 80% discovery. There is no continuous MDE,
minimum N or 193-recruit adequacy conclusion.

Historical accepted results, selected public-entry checks and new full
reproduction are different evidence. Small checks do not rerun formal
campaigns or establish human validity.
Tests in `tests/test_conditional_statistics.py` cover portable dependencies,
rosters/seeds, tail/guard/denominator invariants and refusal paths without fits.

## Magnitude Composition and Precision

[magnitude_composition_and_precision.ipynb](magnitude_composition_and_precision.ipynb)
calls the candidate difference/sum p-composition API and the separate
deterministic Monte Carlo precision calculation. Execution defaults off.
Use Python 3.10.9, NumPy 1.23.5 and SciPy 1.10.0 for precision. Its independent
reader requires R 4.2.2/jsonlite 1.8.8 and a Python executable for standard-library
SHA-256 only; it does not import the producer's numerical functions.
The first code cell installs Python dependencies. R setup is described above.

For initial-need amount slopes, `D=a-b` and `S=a+b` give `abs(a)<abs(b)` iff
`D*S<0`. The pure API combines two fixed intersection-union branches with
Bonferroni2, then Holm over four comparisons. Input p-values must come from
valid one-sided tests; the API does not estimate, authenticate or calibrate
them. Each sum and difference needs its own variance and degrees of freedom.
Statistical unavailability is padded only inside Holm; technical uncertainty
is not treated as nonrejection. The notebook's example values are fixtures,
not empirical findings.

Precision planning scans every positive integer B and every count from 0 to B.
It asks for at least 90% assurance, under Binomial(B,.05), that a central 95%
Clopper-Pearson interval has half-width at most .01. Numerical ambiguities are
reported, never rounded away. The reference result B=2129 is a planning result,
not a calibration tolerance, simultaneous guarantee, power estimate or human N.
Source/input hashes provide self-consistency, not external archive authenticity.
Output directories must be new; no fit or random draw occurs.

This route does not reproduce the separate saved-output coverage diagnosis or
single-saved-fit eight-contrast extraction. Those require matching retained
inputs and distinct entrypoint validation. It does not relabel earlier
signed-effect power as magnitude power or alter its negative findings.

## Magnitude Validation and Conditional Power

[magnitude_validation_and_power.ipynb](magnitude_validation_and_power.ipynb)
provides the complete source-calling route for the original difference/sum
intersection-union magnitude candidate: five fresh templates x 2,129 inputs
(10,645 total), and eight original saved-power cells x 2,000 (16,000 total).
It uses the reference environment above; its first code cell installs the
Python pins. `RUN=False` disables scientific execution, and `FULL=False`
selects one replicate per case without changing the planned denominators.

For a clean checkout, set `RUN=True`, `FULL=True` and
`PRODUCE_POWER_INPUTS=True`. Fresh input generation, fitting and independent
eight-contrast checks use `scripts/reproduce_magnitude_study.py`. Saved-input
production calls `scripts/reproduce_conditional_statistics.py run --family power`
with comma-separated `--ids`, batches of at most 100 and `--no-aggregate`.
The IDs come from `magnitude_design.manifest('saved_power_core')` and preserve
the sparse original cells 2,4,6,8,10,12,14,16 across all 2,000 replicates. It does
not generate the rest of the 80,000-input grid or require its final aggregate.

To reuse compatible existing public power outputs, including a complete 80k
output, set `PRODUCE_POWER_INPUTS=False` and point `POWER_ROOT` to the root
containing `power/manifest.json` and its committed inputs/fits. Saved magnitude
analysis admits these artifacts and does not redraw or refit them. This is
a new public reproduction, not historical archived-byte identity; arbitrary
private result directories are not interchangeable with public commitments.

Magnitude `--all` selects a full fixed mode, `--start/--count` slices roster
positions, and `--ids` accepts a JSON file rather than a comma string.
Separate `--aggregate` calls independently reduce the admitted events into
five fresh and eight saved case reports. Keep M/T/Aj/A4, inapplicable events,
unknown outcomes, full/available-only rates, paired transitions and all linear
coverage diagnostics. Report files and their dependencies stay under ignored
`results/`; filenames alone never establish full completion.

This route retains both the model-based magnitude diagnostic and the original
CR2 candidate, not the separate joint-inference candidate. The adverse S5
multiplier 2 A-sum coverage (.9395 in each of two correlated contrasts) and the
older, distinct S1 B-difference coverage warning remain. Peak historical
candidate discovery 23.75%/15.75% does not establish 80% power, an 80% MDE, minimum
human N, 193-recruit adequacy or human-method adoption. Opening the notebook or
passing a small entry check is not a new full-population result.

## Inference Comparisons and Percentage Reporting

[inference_comparisons.ipynb](inference_comparisons.ipynb) reproduces three
distinct original populations: the 40,000-input baseline, the 480 saved-fit
diagnostic inputs and the 26,000-input fresh CR2/Satterthwaite validation.
The diagnostic uses the first twenty original replicates of all twenty-four
baseline cases. It reuses their fitted models and adds no independent inputs.
The fresh thirteen-case population uses different declared seeds and must not
be conflated with either the baseline or the later paired 10,000-input study.

All execution switches default off. A selected check uses original ordinal
zero; full settings preserve each complete roster. The notebook calls
`reproduce_statistical_comparisons.py`, whose default `plan` stage calculates
nothing. Compatible baseline/diagnostic fits share a cache under one output
root; fresh fits have a separate namespace. Independent readers verify actual
fit artifacts, matrix transport, contrasts, degrees of freedom and corrected
event denominators. Outputs distinguish full rosters from partial checks.

The same notebook includes a synthetic no-fit H8/H9 percentage example.
Effects, SEs and confidence intervals are rescaled from the original ordered
centered-share estimator; its t statistic, degrees of freedom, p value and
decision are unchanged. Recomputing inference from rounded displayed
percentages is not an equivalent replacement. The cancellation example exposes
that distinction. This software interface is not a human-data adapter or a
new calibration result.

The baseline's adverse coverage and family error, the corrected saved-fit
diagnostic and the fresh study's remaining coverage limitations remain
method-specific evidence. They are not overwritten with the later estimator's
outputs. Use the same reference environment specified above.

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
are provided by [native_state_calibration.ipynb](native_state_calibration.ipynb).
No cap20 transfer, human-data validation or
new cost-aware theory result is claimed. The historical routes below remain
separate rather than being silently converted to nominal expected-GPA tests.

## Native-State Calibration

For exact native calibration reproduction, create the Linux x86-64 numerical
environment in `configs/native_reference_environment.yml`, install the package
there, and select that Python kernel. Node.js is also required. The reference
uses NumPy 1.23.5/SciPy 1.10.0 with MKL; newer numerical stacks can produce
different least-squares coefficient bits and therefore different exact native
means. Do not round those differences away or merge batches across environments.
The same numerical stack is the reference for the paired Gamma diagnostic below;
it is not a universal requirement for every historical analysis.

## Saved Coverage And Contrasts

[saved_coverage_and_contrasts.ipynb](saved_coverage_and_contrasts.ipynb) connects
the existing public input producer to two saved-result diagnostics. Input
production is explicitly separate and default off. The coverage route uses
only cell33's2,000 members; its aggregate can remain partial relative to the
full80,000-input grid. The fit route selects ordinal32, the first member of
that same cell. Preparation never silently fits or draws missing inputs.

Coverage is independently recomputed from the saved CI endpoints and exact
membership. The historical target had1879/2000 CR2 coverage versus1881/2000
baseline coverage,109 common misses and CR2 misses60 below/61 above truth.
These are authenticated historical observations, not required success counts
for a new public reproduction or proof of a unique cause.

The single-fit route retains actual cluster counts, all eight contrast-specific
variances/dfs, full covariances, exact R source/frame checks and independent
lmerTest/clubSandwich calculations. Its output does not validate magnitude
population error control or power. Source/runtime bindings belong to the current
execution and do not assert equality with historical binary files.

Both historical-source positive routes and fixed public-schema fixtures are
separate verification targets. Do not interpret a historical saved-input check
as execution of a fresh public population or a new fit. For archive relocation
and CLI options, see [saved diagnostics](../../scripts/README.md#saved-result-diagnostics).

## Joint Magnitude And Interval Comparisons

The final sections of
[magnitude_validation_and_power.ipynb](magnitude_validation_and_power.ipynb)
connect four distinct comparisons to their public input producers: joint
Gaussian-reference inference on 16,000 saved power inputs; a separate
10,645-input fresh joint validation; known-Omega diagnostics on the original
2,000 S1 inputs; and a separate 4,258-input paired CR2/Kenward-Roger interval
study. No saved-result file is required to create a new matching reproduction.
Compatible already-generated public inputs can instead be reused.

`RUN_JOINT_INTERVAL` and `FULL_JOINT_INTERVAL` default off. The notebook exposes
the original rosters and settings, prints source-calling commands and reads
the resulting full JSON summaries. The `coverage` route adds the exact
packages in `src/tutoring_statistics/data/interval_environment.json` to the R
environment above. See the [CLI guide](../../scripts/README.md#joint-magnitude-and-interval-comparisons).

Original IUT, joint, known-covariance and fresh interval results remain separate.
Pointwise finite-law findings, shared KR/negative-correlation numerical cores,
unfavorable coverage and false-rejection tradeoffs do not establish a final
human-analysis method, uniform calibration or a recruitment target.

## Historical Calibration

Open [calibration_and_power.ipynb](calibration_and_power.ipynb) from a repository checkout. The first code
cell installs the Python package and reports Node.js availability. Node.js must be installed
separately and is required for the extracted
pure deterministic schedule; no UI engine, survey adapter or private inputs are
used. Computation defaults to `RUN = False`.

The notebook calls the historical Student/Welch, known-bound Hoeffding,
empirical-Bernstein and conditional planning implementations. Full settings retain
the exact cell seeds and batch size 250: 390,000, 100,000 and 120,000 attempts,
respectively. Smoke takes batch zero of each cell and is not independent evidence
in addition to full. Fixture mode is only for tests and changes RNG array shape.

These methods retain their own endpoint definitions, error policies and negative
calibration evidence. They do not calibrate the budget-only nominal methods
above; method-specific assumptions remain separate.

## Outputs

The notebook also contains a Python-only fixed500 paired Gamma diagnostic,
controlled independently by `RUN_PAIRED = False`. Its default batch selection
is `[0, 39]`,250 cases each; `[0]` or `[39]` checks one original batch without
changing its random stream. Select the reference Python3.10.9 kernel and set
`PRESERVE_PAIRED_REFERENCE = True` in the first cell to retain NumPy1.23.5 and
SciPy1.10.0. This is not the earlier calibration runtime3.12.14/2.3.5/1.16.3.

The producer regenerates the entire C1-C5 draw order before extracting20 C4
people/10 trials. It compares legacy/current Student inference on identical
original person slopes and separately canonical projection. The independent
reader reconstructs raw inputs and derives its own projection/support/Student
oracle, checking four Student outputs per case. See the
[CLI guide](../../scripts/README.md#paired-gamma-diagnostic) for exact commands,
file schema and reading an existing original scientific packet.

The original diagnostic observed40/500 rejections and471/500 zero-covering
intervals in each arm, with zero paired rejection/coverage disagreements and
nonzero floating differences. These are reference observations, not stored
outputs or pass thresholds. Legacy `p<=.05` and current `p<.05` are both
retained. Neither agreement nor successful readback repairs the historical111
failures, validates Type-I error, certifies old-runtime bits, adds independent
historical attempts, or establishes cap20/native-task/human validity. The reader
leaves input records unchanged and does not authenticate their original runtime.

| Producer | Recomputed output |
| --- | --- |
| `reproduce_tutoring_statistics.py --analysis inference` | Synthetic rows, historical three-policy fits, fresh finite-metric fixture and bounded fit. These are functionality examples, not human inference. |
| `reproduce_tutoring_statistics.py --analysis planning` | `planning.json`: 28 declared-assumption scenarios, eligibility/variance geometry, marginal power and conditional joint bounds. |
| `calibrate_tutoring_methods.py` | All 30 cells and populations, batch summaries, `calibration_summary.json`, core/policy/power CSV tables and `all_criterion_failures.csv`. |
| `reproduce_tutoring_statistics.py --analysis bounded` | Ten cells, all nine slots including three unsupported slopes, same-draw Student/Welch comparisons, batch parity, route summary and failed criteria. |
| `reproduce_tutoring_statistics.py --analysis variance-adaptive` | Recomputed schedules, 3,072 raw projection corners, unsnapped raw and ideal truth rosters, 30 diagnostic-only C1 candidate/comparator states (no task revision), 135 cohort-bound cases, twelve calibration cells, three methods and all policy/scope diagnostics. |
| `reproduce_paired_diagnostic.py produce/read` | Fixed500 or selected250 raw arrays, person qualifications/slopes, legacy/current/projection inference, paired decisions and independent numerical readback; no Type-I repair claim. |

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
This notebook provides numerical statistical diagnostics, not UI replay, report
graphics, human fits or cost-aware policy predictions. Expected utility remains
the decision-performance metric; endpoint calibration checks a different,
explicitly statistical question.
