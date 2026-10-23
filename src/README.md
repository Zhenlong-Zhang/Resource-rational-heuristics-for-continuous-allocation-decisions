# Scientific Source

| Package | Purpose |
| --- | --- |
| [design_validation](design_validation/README.md) | Finite positive-support tutoring with a 20-query cap, exact Bayesian updates, repeated one-step VOI, grid allocation, task revision and identifiability checks. |
| [tutoring_analysis](tutoring_analysis/README.md) | Participant-level inference, fixed synthetic calibration, bounded and variance-adaptive sensitivities, conditional power and public reproduction. |

The portable runners add `src/` to the import path for the original isolated
packages. The installed package also exposes the `src` namespace without loading
unrelated models. Scientific kernels and frozen configurations retain their
original identities; notebooks call the runners.

Expected utility measures performance. Behavioral diagnostics, statistical
calibration and human validity are distinct claims.
