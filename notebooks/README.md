# Notebooks

`run_round2_pipeline.ipynb` is an interactive wrapper around `scripts/generate_results.py`.

Use it when you want to:

- change episodes, VOI samples, sweep features, DP grids, or integration settings in notebook cells
- run the same pipeline as the command-line script
- inspect generated CSV outputs quickly with pandas

The notebook does not duplicate model code. It builds a command and executes the shared runner, so results remain aligned with the repository code.
