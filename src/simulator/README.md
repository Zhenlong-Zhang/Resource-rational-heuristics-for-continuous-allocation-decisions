# Simulator

`simulator.py` provides small adapters around the MDP episode runner.

- `run_single_episode(...)`: execute one policy in one sampled environment state
- `simulate_many_episodes(...)`: repeat episodes and return standardized summaries
- `episode_to_dict(...)`: serialize an `EpisodeResult`
- `BeliefActionDictionaryPolicy`: execute an arbitrary rounded belief-state-to-action dictionary
- `rounded_belief_key(...)`: construct dictionary keys from continuous beliefs

The simulator delegates utility, belief updates, action feasibility, and terminal allocation to the MDP classes. Experiment-level common randomization is implemented in `src/experiments/randomization.py`.
