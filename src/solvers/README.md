# Solvers

`solvers/` contains approximation methods used by policies and diagnostics.

`dp.py`:

- `FiniteHorizonDPSolver`: small discretized backward-induction solver using posterior precision/sample-count structure
- `DiscretizedDynamicProgrammingPolicy`: wraps the solver as a metalevel policy

`gauss_hermite.py`:

- `gauss_hermite_nodes_weights(...)`: returns Hermite-Gauss nodes and weights
- `normal_expectation_1d(...)`: computes one-dimensional Gaussian expectations
- `independent_normal_expectation_2d(...)`: computes expectations under two independent Gaussian variables
- `expected_terminal_utility_gauss_hermite(...)`: evaluates terminal expected utility without Monte Carlo sampling

The DP solver is intentionally approximate. Use `src/experiments/dp_diagnostics.py` to test whether increasing `max_samples`, `mean_grid_size`, and `observation_branches` improves performance.
