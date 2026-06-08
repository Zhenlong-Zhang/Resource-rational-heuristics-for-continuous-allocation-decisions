# Simulator Helpers

`simulator.py` contains lightweight wrappers around the MDP.

Key functions/classes:

- `run_single_episode(...)`: runs one episode with a selected policy
- `simulate_many_episodes(...)`: runs repeated episodes and returns summary statistics
- `episode_to_dict(...)`: converts an `EpisodeResult` to a plain dictionary for display/export
- `BeliefActionDictionaryPolicy`: executes an arbitrary meta-level policy represented as a belief-state to action dictionary
- `rounded_belief_key(...)`: default key function for dictionary policies

Use these helpers for quick debugging. Use `src/experiments/` or `scripts/generate_results.py` for structured comparisons and result files.
