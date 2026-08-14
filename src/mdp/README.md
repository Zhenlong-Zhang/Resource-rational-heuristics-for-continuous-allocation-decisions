# MDP Models

`meta_mdp.py` defines the Gaussian-belief continuous-allocation problem.

- `EnvironmentConfig`: environmental, utility, timing, information, and numerical settings
- `TrueState`: latent needs and remaining time
- `BeliefState`: posterior means, variances, deliberation time, and observation history
- `ContinuousAllocationMetaMDP`: Bayesian updates, metalevel transitions, terminal allocation, and episode execution
- `utility(...)`: asymmetric utility for outcomes above or below need

`finite_support.py` provides a controlled alternative to Gaussian priors.

- `FiniteSupportAtom`: one latent need configuration
- `FiniteSupportPrior`: weighted finite support with deterministic identity hashes
- `FiniteSupportBeliefState`: posterior weights over atoms
- `FiniteSupportMetaMDP`: exact finite-support Bayesian updates with the shared allocation model

Sampling changes beliefs and consumes time. It does not receive a separate utility penalty in the current model.
