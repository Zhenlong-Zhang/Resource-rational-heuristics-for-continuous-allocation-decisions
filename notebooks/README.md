# Notebooks

`run_round2_pipeline.ipynb` is an interactive wrapper around `scripts/generate_results.py`.

Use it when you want to:

- change episodes, VOI samples, sweep features, DP grids, or integration settings in notebook cells
- run the same pipeline as the command-line script
- inspect generated CSV outputs quickly with pandas

The notebook does not duplicate model code. It builds a command and executes the shared runner, so results remain aligned with the repository code.

`run_round5_pipeline.ipynb` is the equivalent wrapper for the frozen Round 5
Hoffman2 array and report workflow. Submission is disabled by default. The
notebook can print the exact command, submit it when running on Hoffman2, inspect
manifest progress, and generate the final report from strictly collected result
folders without duplicating simulation or analysis code.
