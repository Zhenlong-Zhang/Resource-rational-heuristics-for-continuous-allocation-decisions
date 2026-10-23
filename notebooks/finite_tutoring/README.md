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
