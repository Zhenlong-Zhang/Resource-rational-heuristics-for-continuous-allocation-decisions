# Finite Tutoring Validation

The task has integer needs/reports on 1..100 and at most 20 information requests
per trial. Each request consumes tutoring time and updates one discrete belief.
The repeated one-step policy compares both sampling actions with stopping.

| Module | Purpose |
| --- | --- |
| `bounded_task.py` | Frozen task configuration, finite likelihoods/posteriors, expected utility, sampling decisions, terminal grid allocation, episode simulation, summaries and grid sensitivity. |
| `revision.py` | Prespecified C3 prior revision, source/configuration hash checks, independent screening/confirmation and selection gates; uses the unchanged task kernel. |
| `identifiability.py` | C2 prior-based response-rule simulations, group responsiveness and individual classification. |
| `c1_identifiability.py` | Deterministic C1 prior/slope-support diagnostics used by the statistical sensitivity analysis. |
| `model_bridge.py` | Independent scalar finite-prior moments and affine GPA mapping, with an analytic Gaussian reference; no Gaussian policy simulation. |

Use [the finite notebook](../../notebooks/finite_tutoring/design_validation.ipynb)
or [portable commands](../../scripts/README.md). Full baseline/revised conditions
and negative or ambiguous diagnostics remain available. One-step policy
validation is not a global optimality proof.
