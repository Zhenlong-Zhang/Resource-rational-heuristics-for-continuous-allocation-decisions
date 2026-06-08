# MDP Model

`meta_mdp.py` defines the core allocation model.

Key objects:

- `EnvironmentConfig`: all environment parameters, including need distribution, sampling noise, total time, utility exponent, prior knowledge, learning efficiency, and terminal integration method
- `TrueState`: hidden true needs of the two recipients
- `BeliefState`: current posterior means, variances, deliberation time, and sampling history
- `EpisodeResult`: realized trajectory and final allocation outcome
- `ContinuousAllocationMetaMDP`: state transitions, Gaussian belief updates, sampling, terminal allocation, and realized utility

Important methods:

- `initial_belief(...)`: creates the starting belief from environment parameters
- `transition(...)`: applies a sampling action and updates the belief
- `solve_terminal_allocation(...)`: chooses the terminal allocation according to the current belief
- `resolve_final_allocation(...)`: uses a policy-specific final-choice heuristic when available, otherwise solves the terminal allocation
- `run_episode(...)`: executes a metalevel policy until termination

The model supports optional pre-generated observation streams so different policies can be evaluated on the same true states and information-gathering noise.
