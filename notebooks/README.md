# Capped-Model Reproduction

These notebooks reproduce the finite tutoring task with at most 20 queries per
trial and its associated statistical analyses. They call public source and
portable runners rather than containing duplicate model implementations.

| Notebook | Full reproduction scope |
| --- | --- |
| [Finite task validation](finite_tutoring/design_validation.ipynb) | Five-condition baseline at 1,200 episodes/condition; original C3 screening and independent confirmation; C2 synthetic response-rule recovery; deterministic finite-prior/GPA audit. |
| [Statistical calibration and power](statistical_methods/calibration_and_power.ipynb) | Original inference examples, 28 planning scenarios, 390,000 Student/Welch attempts, 100,000 known-bound attempts, 120,000 empirical-Bernstein attempts, identifiability and cohort-bound diagnostics. |

## Use

1. Open a notebook from a checkout of `20-limit`.
2. Run its dependency-install cell and inspect settings and printed commands.
3. Choose smoke/full and a fresh output path, then enable `RUN` deliberately.
4. Read the generated tables and all failure/unavailability diagnostics.

Formal computations are disabled by default. Smoke settings check wiring only
and may overlap the full recipe. Outputs are generated under ignored `results/`;
no private files or saved report tables are required.

See the [finite settings](finite_tutoring/README.md) and
[statistical settings](statistical_methods/README.md) for seeds, stages and outputs.
The finite-prior audit is a supporting mathematical comparison, not an execution
of the separate continuous Gaussian model. Budget-only and later statistical
models are outside this branch's scope.
