# Finite Task Analyses

This package is separate from the Gaussian metalevel MDP. Needs and reports
have support 1 through 100; posterior updates and one-step report expectations
enumerate that support. The native allocation grid has 1,001 shares.

| Module | Purpose |
| --- | --- |
| `bounded_task.py` | Task configuration, finite Bayes, terminal allocation, exact one-step sampling, seeded episodes and summaries. |
| `revision.py` | One fixed prior revision, confirmation stages, selection criteria and grid sensitivity. |
| `identifiability.py` | Synthetic fixed-share/equal-outcome response rules, group slopes and individual held-out classification. |
| `model_bridge.py` | Finite versus Gaussian reference and affine GPA checks. |
| `c1_identifiability.py` | Target coincidence/separation, side symmetry and noise-dependent discrimination, including the earlier symmetric and failed-gap comparators. |

Use the scripts and notebooks rather than importing notebook code. Repeated
one-step sampling is myopic, not a globally optimal metalevel policy. A
terminal-state stopping certificate does not certify the earlier path.
