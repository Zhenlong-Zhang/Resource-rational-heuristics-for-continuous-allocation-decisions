# Statistical Methods

The historical inference route rejects budget-only or newer-profile records,
even when a participant asked no more than 20 questions. The model identity,
not the observed sample count alone, determines which analysis is applicable.

`inference.py` retains historical person-weighted Student/Welch tests, raw OLS
projections, complete-ten eligibility and three multiplicity policies.
`calibration.py` generates all declared historical populations and stress cases;
`planning.py` provides conditional moment geometry and power calculations.

`bounded_inference.py` and `robustness.py` implement the optional known-bound
six-mean sensitivity while keeping all nine slots and three unavailable slopes.
`empirical_bernstein.py` and `identifiability_robustness.py` provide the separate
variance-adaptive method, source-known projection domain, exact finite-pattern
expectations in real arithmetic, raw floating coefficients and cohort bounds.
Numerical calculations are binary64 approximations, not machine certificates.

`finite_metrics.py` isolates the historical finite-belief metric calculation.
`source_projection_panel.cjs` contains only deterministic schedule mathematics;
Node.js is required. `makeSchedule` is also exported for shared scientific
callers. This historical schedule-only route does not use an engine, normalizer,
participant export, deployment or network dependency.

`reproduction.py` orchestrates the accepted kernels and emits complete diagnostic
summaries. Its producers recompute all consumed scientific context. Use
`scripts/reproduce_tutoring_statistics.py` and
`scripts/calibrate_tutoring_methods.py`; notebook guides specify full scales.
All generated files belong under ignored `results/`, not in version control.

## Paired Gamma Attribution

`paired_diagnostic.py` compares unchanged `inference.participant_slope` /
`inference.contrast` with `nominal_inference.test_contrast` on original vector
person inputs, then separately with canonical projection. `paired_recipe.py`
fixes the original Gamma law and portable packet I/O. The existing calibration
generator consumes the complete C1-C5 draw order at seed2026100627 and batch
IDs0/39,250 cases each, before extracting20 C4 people with10 trials.

`paired_readback.py` independently reconstructs projections, eligibility,
Student inference and decisions from saved raw inputs. Only the input-law
generator is shared; no producer estimator supplies its numerical oracle.
Raw arrays and fixed identities are exact; independent numerical fields use
the original absolute1e-10 plus relative1e-9 tolerance, with actual errors
reported. Both `p<=.05` and `p<.05`, coverage and exact-boundary counts remain.

Use `scripts/reproduce_paired_diagnostic.py` or the supplementary section in
`calibration_and_power.ipynb`. Existing original scientific packets can be
read without operational wrappers, but operational provenance is not certified.
The diagnostic does not call the native-row validator or fabricate native
observations from unbounded abstract Gamma responses. It is not a Type-I or
historical-failure repair, original-runtime bit replay, or cap20 validation.

## Expected-GPA and Nominal Analysis

`expected_outcomes.py` reconstructs native-grid targets from synthetic paid
histories. The equal-GPA target minimizes `abs(E[GPA_A] - E[GPA_B])`, not
`E[abs(GPA_A - GPA_B)]` or raw-surplus inequality. Ties use combined GPA, then
distance to equal split and the lower index. Its signed resemblance measure is
`R_EGPA_pp = (abs(j-500) - abs(j-jEGPA))/10`; positive means closer to the
belief-based equal-GPA target. This is not equality of hidden realized outcomes.

`nominal_inference.py` retains complete-ten, equal-person aggregation, fixed
arm weights, Student/Welch tests, nominal alpha .05 and two-sided 95% intervals.
H05 uses the less tail; the other eight primary contrasts use greater. D10 is
nonprimary and less. Missing arms are not reweighted; zero/nonfinite standard
error is unavailable. Exhaustion remains eligible for question count but not
allocation metrics. Tied lower-mean choices contribute zero to the primary
endpoint. Supporting alternatives do not replace primary eligibility rules.

`conditional_power.py` gives 24 IID-normal noncentral-t planning references:
three effects, two power targets, two endpoint references and two tails, with
three retention assumptions each. Independent-group references assume common
variance, not heterogeneous Welch. Retention assurance, marginal power and
joint power are different; no final recruitment size is selected.

Use `scripts/reproduce_expected_gpa.py` and the expected-GPA notebook. These
budget-only methods do not transfer validation to cap20.

## Native-State Calibration

`native_source.cjs` constructs deterministic synthetic sessions through the
bundled engine and normalizer. `native_reconstruction.py` independently rebuilds
schedules, reports and finite likelihoods; `native_lookup.py` checks the source
and creates finite target/projection maps. `native_truth.py` separately derives
exact rational native means under declared ideal probability laws.

`native_calibration.py` runs the fixed 40-subcase, 40x250-batch design with prepared
input validation and scalar/vector endpoint audits. `native_readback.py` independently
recomputes count intervals, applicable Normal-reference power and dispositions.
Use `scripts/reproduce_native_calibration.py` and the native-state notebook for
the entire clean-checkout chain, including input generation and compatible resume.
This is synthetic marginal calibration, not human validation or cap20 transfer.

Python runtime dependencies are NumPy and SciPy. Pure schedule generation also
requires Node.js.

The historical checkout entrypoints are the two historical scripts above.
They explicitly prepend the repository's `src/` directory to `sys.path` before
importing `tutoring_analysis` and `design_validation`. Each statistics test does
the same, so standalone discovery does not depend on test order or `PYTHONPATH`.
The new expected-GPA runner instead prepends the repository root and imports
`src.tutoring_analysis`. Use the documented runners to select the correct model
and method rather than mixing these independent historical and nominal routes.
