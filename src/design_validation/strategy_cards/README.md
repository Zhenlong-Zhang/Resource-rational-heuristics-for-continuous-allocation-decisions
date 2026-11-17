# Fixed-Design Strategy Cards

Six policies share the original 450 finite-task roots: equal minutes, similar
projected GPA, all time to the higher displayed guess, all time to the lower
guess, ask each recipient once then match projected GPA, and adaptive one-more
search. The first four integrate latent uncertainty without reports; the fifth
integrates report pairs. The adaptive card follows
one fixed replicate per root. See the [notebook](../../../notebooks/finite_tutoring/strategy_card_comparison.ipynb).

| Module | Function |
| --- | --- |
| `inputs.py`, `display.cjs` | Regenerate and bind original task, ordered binary64 priors, physical recipients and native displayed guesses. |
| `fixed.py`, `fixed_reference.py` | Five-card production and separately implemented checks; fixed-two integrates all 100 x 100 report pairs. |
| `adaptive.py`, `adaptive_path.py` | Repeated strict binary64 one-step decisions, physical-keyed draws and paid updates; only available tutoring time bounds search. |
| `adaptive_reference.py` | Reconstruct saved decisions, report cursors, fees and terminal values independently. |
| `external_bounds.py`, `external_reference.py` | Separate outward enclosures for exact-real external report/payoff evaluation, not substitutes for the binary64 decision rule. |
| `reproduction.py` | Immutable prepared inputs, per-root production/checking and source-linked aggregation. |
| `summary.py` | Fixed 1/90 condition weights, explicit missing/unresolved mass and uncertainty labels. |

Fresh no-question scores do not depend on private historical caches. Previously
reused scores retain their historical numerical identity; numerical agreement
uses the stated 2e-11 GPA tolerance, not a claim of identical execution bits.
The literal RNG domain in the recipe is a scientific identity and is unchanged.

Use Python 3.11.5, NumPy 1.24.3, Node 22.14.0 and one numerical thread per
process. The binding records the actual numerical libraries; identical version
numbers alone do not guarantee identical binary64 decisions on arbitrary
hardware. Existing files cannot be silently mixed with changed source/builds.

The five-card support bounds are not tight numerical errors. Adaptive intervals
enclose numerical evaluation, not report-stream sampling error. One adaptive
path per root supplies no within-root MCSE, certified ranking, globally optimal
policy, or human-study validation. Partial collections keep the original
denominators and cannot be labelled complete.
