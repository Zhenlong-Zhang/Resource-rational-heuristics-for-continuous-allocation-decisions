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
callers. No UI engine, normalizer,
participant export, deployment or network dependency is included.

`reproduction.py` orchestrates the accepted kernels and emits complete diagnostic
summaries. Its producers recompute all consumed scientific context. Use
`scripts/reproduce_tutoring_statistics.py` and
`scripts/calibrate_tutoring_methods.py`; notebook guides specify full scales.
All generated files belong under ignored `results/`, not in version control.

Python runtime dependencies are NumPy and SciPy. Pure schedule generation also
requires Node.js. The package covers the completed historical statistical
analysis only; later nominal-power and budget-planning methods are not included.

The supported checkout entrypoints are the two scripts above and the notebook.
They explicitly prepend the repository's `src/` directory to `sys.path` before
importing `tutoring_analysis` and `design_validation`. Each statistics test does
the same, so standalone discovery does not depend on test order or `PYTHONPATH`.
The editable install supplies dependencies; it does not promise an independent
`src.tutoring_analysis` import interface.
