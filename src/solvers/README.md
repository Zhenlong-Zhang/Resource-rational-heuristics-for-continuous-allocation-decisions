# Solvers

This package contains metalevel approximations and terminal allocation solvers.

## Metalevel Approximation

`dp.py` provides:

- `FiniteHorizonDPSolver`: discretized backward induction over posterior state summaries
- `DiscretizedDynamicProgrammingPolicy`: policy wrapper for the solver

The DP solver is approximate. Its behavior depends on the sampling horizon, posterior-mean grid, and observation branches. Resolution should be checked with `src/experiments/dp_diagnostics.py` before interpreting comparisons.

## Gaussian Integration

`gauss_hermite.py` provides one- and two-dimensional normal expectations, expected terminal utility, and terminal-allocation search using Gauss-Hermite quadrature.

## Terminal Allocation

`terminal.py` contains the production finite-support terminal optimizer, structural symmetry checks, tie classification, certificates, and performance diagnostics.

Independent implementations are kept in:

- `terminal_reference.py`
- `terminal_reference_b.py`
- `terminal_reference_agreement.py`

The production and reference solvers are compared through frozen validation suites. Passing a smoke test is not equivalent to proving exact continuous-state optimality.
