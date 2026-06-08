# Source Package

`src/` is the main Python package for the project.

The package is organized by responsibility:

- `mdp/`: defines the allocation task and metalevel MDP
- `simulator/`: provides episode-level execution helpers
- `policies/`: contains hand-coded heuristics and VOI-style policies
- `solvers/`: contains approximation methods such as discretized DP and Gauss-Hermite integration
- `experiments/`: contains result-generation logic, sweeps, diagnostics, and reproducibility helpers

Most users should call the project through `scripts/generate_results.py` or `notebooks/run_round2_pipeline.ipynb` instead of importing each module manually.
