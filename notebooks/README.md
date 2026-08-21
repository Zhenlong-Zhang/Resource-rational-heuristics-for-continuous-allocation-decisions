# Reproducibility Notebooks

The notebooks are organized by the research update in which an analysis was reported. They call the shared implementation in `src/` and `scripts/`; they do not duplicate model logic.

| Update | Notebook | Purpose |
| --- | --- | --- |
| Round 2 | `round_02/reproduce_round_02.ipynb` | Reproduce the Step 7 comparisons, parameter sweeps, targeted regime searches, and approximation diagnostics. |
| Round 3 | `round_03/reproduce_round_03.ipynb` | Verify the observation-stream invariant and reproduce the active-search true-outcome searches. |
| Round 4 | `round_04/reproduce_round_04.ipynb` | Reproduce the deliberately constructed active-search benchmark comparison. |
| Round 5 | `round_05/reproduce_round_05.ipynb` | Reproduce the objective, information-value, confirmation, and non-myopic solver analyses. |
| Round 6 | `round_06/reproduce_round_06.ipynb` | Run the portable smoke or serious scarcity workflow, reproduce object/development/confirmation summaries, and inspect lower-need allocation metrics. |

Each notebook exposes the main episode, VOI-sampling, grid, seed, and output settings near the top. Large configurations are shown explicitly but are not started until `RUN = True` is set in the notebook.

Generated tables and figures are written under `results/`, which is ignored by Git.

The Round 6 notebook is intentionally portable: it does not submit jobs, access a cluster,
or include the generated serious evidence. The accepted professor-facing evidence package
and its server-side provenance are separate from this public code repository.
