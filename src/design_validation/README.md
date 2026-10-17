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
| `budget_horizon.py` | Native-arithmetic paired capped/affordable-budget paths, initial information bounds and paired summaries. |
| `terminal_time.py` | Assumed personal-time macro costs, utility/rate sensitivity and conditional old-terminal analysis. |
| `optimality.py` | Numerical Bellman lower/upper envelopes over the full affordable horizon, with feasible lower-policy witnesses. |
| `finite_reproduction.py` | Portable scientific stage orchestration and compatible paired-input reconstruction. |
| `finite_auxiliary.py`, `finite_auxiliary_inputs.py` | ENGAGEMENT schedules/parity/geometry and strict readers for generated stages. |
| `finite_time_tables.py` | Initial, working-stimulus and old-terminal descriptive summaries with explicit denominators. |
| `finite_bound_tables.py` | Compact raw bounds and separately identified symmetry-derived C5 intersections. |

`budget_source_panel.cjs` and `finite_schedule_engine.cjs` provide the pure
deterministic schedule and native payoff/posterior dependency closure. Node.js
is required; no browser, participant UI or network service is used.

Use the scripts and notebooks rather than importing notebook code. Repeated
one-step sampling is myopic, not a globally optimal metalevel policy. A
terminal-state stopping certificate does not certify the earlier path.

The paired comparison explicitly evaluates two horizons. Terminal-time and
envelope diagnostics use the budget-only model; the envelope analysis assumes
zero terminal computation cost. They do not establish capped-model counterparts,
globally optimal repeated-myopic behavior or human timing validity. Float64
envelopes are numerical bounds, not machine-certified interval arithmetic.
