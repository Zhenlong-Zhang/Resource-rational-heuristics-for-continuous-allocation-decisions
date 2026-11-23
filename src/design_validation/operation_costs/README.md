# Operation Costs

This module reproduces three distinct calculations on the original five
condition roots, variant 0/main 1. `inputs.py` calls the native source and
display generator; no retained reports or private receipts are required.

| Module | Calculation |
| --- | --- |
| `literal.py`, `costs.py` | Repeated one-step binary64 VOC with the literal full-grid operation inventory, at three prices; initial ES comparison. |
| `actors.py`, `noask.py` | Five separately precommitted H-only rules, typed costs, scalar external GPA, eight affine prices and sufficient pairwise price regions. |
| `ask_once.py`, `fees.py`, `scalars.py` | Ask physical A or B once, then use raw-surplus equalization; integrate all 100 reports on C3/C5 with corrected signed joint arithmetic. |
| `*_reference.py`, `reference_math.py` | Separate target, cost, finite-law, signed-payoff and reduction checks. |
| `checkpoint.py`, `reproduction.py` | Immutable missing-only resume, independently checked products and CSV routing. |

## Scientific Identity

The external law is real exp of the captured binary64 log/width operands, with
each latent-specific channel row normalized over reports 1 through 100.
The actor receives only authenticated displayed inputs. Ask-once integration
retains tiny positive-upper report masses, including zero lower bounds, and
never divides by a potentially tiny report mass. Its scalar signed sum is
single-pass, including for generators.

The objective is **combined** GPA minus rho times personal seconds. Tutoring
resource is debited separately for a question, even at zero rho. Operation
durations are assumptions, not measured human times. Unknown coefficients stay
symbolic; price tables additionally evaluate the member where each unknown
operation costs one second. A table winner is not a free metalevel selector.

Literal full-grid search values after observations are reached-history
conditional values, not complete ex-ante values. Only its zero-report endpoints
can be directly compared to initial ES. In contrast, ask-once values integrate
the full predictive support for each fixed direction. No free best-direction
choice or global-optimality assertion follows.

Noask/ask-once physical scalar operands and initial ES use the original
80-bit isqrt method; their independent reference uses 96-bit scalar bounds
and 52-digit exponential enclosures. Literal reached-history evaluation instead
retains its utility grid `2**20` and probability grid `2**30`.
Historical compatible coarser score bounds can therefore differ in bytes.
Do not label fresh work with historical reuse counts, writer identities or the
rejected pre-repair ask-once GPA output. A dyadic representation precision is
not a uniform accuracy guarantee.

## Reproduction

Use `scripts/reproduce_operation_costs.py all --output results/operation_costs`.
Thematic notebook: `notebooks/finite_tutoring/operation_costs.ipynb`.
Python 3.11.5/NumPy 1.24.3 and Node 22.14.0 preserve the native/source64 route.
This is separate from the older Python 3.10/NumPy 1.23.5 coefficient route.

`prepare` generates native roots and displayed inputs. `produce` writes missing
compatible stages only; `check` independently reconstructs scientific operands;
`tables` reduces checked public products without regeneration. The same output
directory cannot silently change source, runtime, scientific settings or root
selection. Interrupted complete checkpoints remain reusable; corrupted or
foreign checkpoints are refused. `--workers` parallelizes disjoint units, not
numerical threads within a unit. No full calculation starts merely by opening
the notebook.

## Output Families

- `noask.csv`: 25 procedures and 200 price rows in the full selection.
- `menu_winners.csv`: 40 restricted-menu root-price cells; overlapping intervals
  remain indeterminate unless action and effective fee are exactly shared.
- `literal.csv`: 15 reached-history paths and assumed personal seconds.
- `ask_once.csv`: four full ex-ante directions and 32 price rows.
- `ask_once_contrasts.csv`: 64 same-root ES/raw comparisons.

JSON products retain operation traces, symbolic coefficients, complete signed
rows, utility operands, and numerical bounds. CSV tables are summaries, not
substitutes for those evidence fields.
