# Finite Task Reproduction

Open [design_validation.ipynb](design_validation.ipynb) from a repository checkout. The first code cell
installs the Python package; execution defaults to `RUN = False`.

The notebook calls shared code for the baseline five-condition task, the single
C3 prior revision, corrected C2 synthetic identifiability and the finite/Gaussian
reference audit. Full confirmation uses 1,200 episodes per condition; C2 recovery
uses 1,000 groups of 80 synthetic participants per cell. Screening and confirmation
use their distinct fixed seeds.

The finite task permits at most 20 queries per trial, with integer latent needs
and reports from 1 to 100 and exact discrete Bayes updates. This is not the older
unbounded Gaussian model. Both baseline and revised conditions use this cap.

| Notebook route | Producer and full settings | Generated result family |
| --- | --- | --- |
| Baseline | `validate_tutoring_design.py`, baseline conditions, precision stage, seed 2026091703, 1200/condition, sensitivity enabled | Episodes, initial diagnostics, summary, 1001/2001-grid sensitivity, completion record |
| C3 revision | `validate_tutoring_design_revision.py`, screening seed 2026091901 (30/condition), confirmation seed 2026091902 (1200/condition) | Separate `revised/screening` and `revised/confirmation` validation, fixed-rule comparisons and selection gates |
| C2 identifiability | `validate_allocation_identifiability.py`, baseline conditions, `tutoring_identifiability_validation.json`; screening seed 2026091731 (50 groups of 80), confirmation seed 2026091732 (1000 groups of 80) | Separate `identifiability/screening` and `identifiability/confirmation` group statistics, classification summaries and manifests |
| Model bridge | `audit_tutoring_model_bridge.py`, revised conditions, deterministic | Affine GPA and finite-prior summary, template CSV and manifest |

The stage loops read each stage's own outputs. Confirmation neither overwrites
screening nor turns screening selection into independent evidence. Smoke uses
the baseline smoke recipe, C3 screening and C2 smoke, not full confirmation.

Integrity hashes bind the scientific configuration and numerical engine. The
prior-moment audit uses the revised design; both baseline and revised task
results remain reproducible. See the [configuration index](../../configs/README.md#finite-tutoring-settings).

Generated outputs go to ignored `results/finite_reproduction/`. Use a new output
directory for each execution. Smoke is only a wiring check. Target matches and
group responsiveness are not individual cognitive-rule identification or human
validation; repeated one-step sampling is not a globally optimal adaptive policy.

## Horizon and Terminal Diagnostics

[horizon_terminal_bounds.ipynb](horizon_terminal_bounds.ipynb) calls
`scripts/reproduce_finite_tutoring.py` for three explicitly separate analyses:

| Route | Full settings | Outputs |
| --- | --- | --- |
| Paired horizons | Seed2026100901;400 pairs per six C3/C5 templates | 2400 paired scenarios/4800 paths, capped versus budget-only summaries and27 initial diagnostics |
| Terminal-time sensitivity | 36 initial states;three timing profiles;price multipliers0,.01,.1,1,10,100 | 1800 initial scenarios, reference rate, orientation/break-even diagnostics,43200 conditional scenarios at2400 old terminals and macro counts |
| Numerical bounds | Zero terminal cost;full affordable horizon;600s/100000states/depth6 refinement | 27 initial bounds, six root lower/upper envelopes with complete witnesses/checkpoints,2400 fixed-terminal stopping diagnostics |

These routes regenerate the paired input and reject mismatched or incomplete
panels. Timing profiles are assumptions. Conditional terminal sensitivity is not
a new duration-aware trajectory distribution; numerical statewise bounds do not
certify the preceding myopic policy. The default capped task above is unchanged.

The companion `reproduce_finite_auxiliary.py` generates 45 ENGAGEMENT schedules
with seed2026100902, 216 belief and 1620 outcome parity cases, and 6804 balanced
geometry cases. Its readers consume the raw stages above to produce initial
ledgers, equal-template and working-stimulus summaries, terminal pooling and
compact bounds. Undefined observations keep explicit denominators. Working C1
uses gap1; template pooling is not human schedule weighting. C5 symmetry-derived
bounds stay separate from raw root envelopes. Smoke produces partial diagnostics,
not the complete descriptive tables; full mode generates the complete chain.
All inputs are regenerated from the scientific settings. Cost-aware adaptive
policies are not evaluated here; budget-only nominal inference and native-state
calibration have [separate notebooks](../statistical_methods/README.md).

## Selected Cost-Aware Computation

[cost_aware_computation.ipynb](cost_aware_computation.ipynb) supplies a different
trajectory calculation: repeated one-query decisions over duration-priced
terminal macros. It generates native working roots, follows explicitly selected
stable trial coordinates, and independently checks all report branches, draws,
paid updates, terminal values and root bound annotations. The full100-report
selected-child backup is retained separately from raw maxima.

Source generation follows genuine native checks/practice and baseline ES only
through each session's last selected main (1..10 supported). It restores unpaid
selected roots without constructing irrelevant paid/exhaustion prefixes, while
preserving those forks' action-counter gaps for later mains. The source packet
is explicitly partial; no full-session or full-panel completion is claimed.

The first cell installs the package with reference constraints; substantial work
defaults to `RUN = False`. Use Python3.10.9/NumPy1.23.5/SciPy1.10.0 and Node.js.
Exact probability bits may depend on the numerical build/CPU; numerical agreement
does not authenticate historical execution. The reader shares NumPy/PCG64 but
does not call producer numerical helpers.

See [CLI and output layout](../../scripts/README.md#selected-cost-aware-trials).
The selection JSON is the complete declared scope, not a person or campaign
roster. Settings remain fixed; unsupported changes are rejected. Personal
question time is unpriced, terminal time affects the objective only, and exact
exhaustion has undefined share and zero duration. No person inference, global
aggregation, cap20 transfer or global-optimality claim is supplied.

## Personal-Time Reference

[personal_time_reference.ipynb](personal_time_reference.ipynb) is a separate
budget-only reference that prices information requests, reading the updated
beliefs and terminal operations. It includes voluntary random termination and
charges forced random choice at exhaustion. Tutoring resource and personal time
are different costs; expected combined GPA and the net objective are reported
separately.

The full recipe generates 45 native schedules and checks 450 expected endpoints:
five conditions, nine equally weighted schedules each, and ten main trials per
schedule. Independent decision reconstruction precedes whole-person aggregation.
It also computes both request margins at the 50 original variant-zero roots.
Adding back only incremental request/read price is not gross information value:
child termination and tutoring opportunity costs remain. Four initial-state
targets compare equal expected GPA with equal mean surplus using the existing
finite mathematical implementation.

The reference does not produce 450 realized histories or a Monte Carlo precision
estimate. Unexpected search/random choices are retained as discrepancies, not
replaced with expected stopping. Full calculations are disabled by default;
small selections check wiring and remain explicitly partial. Source regeneration,
independent checking and outputs do not require a private report or cached
result tables. This reference is not validation under the 20-query cap or final
paper/human-model adoption.
