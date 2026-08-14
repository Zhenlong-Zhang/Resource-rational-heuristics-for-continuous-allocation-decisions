# Source Package

`src/` is the source of truth for model and analysis logic.

- `mdp/`: continuous-allocation and finite-support metalevel MDPs
- `simulator/`: episode execution and policy adapters
- `policies/`: hand-coded heuristics and information-acquisition approximations
- `solvers/`: DP, Gauss-Hermite, and terminal allocation solvers
- `experiments/`: randomization, metrics, sweeps, comparisons, and strict evidence workflows

Use the scripts or notebooks for complete analyses. Import modules directly when developing a new policy, solver, metric, or experiment.

```python
from src.mdp.meta_mdp import EnvironmentConfig
from src.policies.heuristic import EqualDivisionPolicy
from src.simulator.simulator import run_single_episode

config = EnvironmentConfig(random_seed=7)
result = run_single_episode(config, EqualDivisionPolicy())
print(result.realized_utility)
```

Subdirectory READMEs describe the public classes and functions in more detail.
