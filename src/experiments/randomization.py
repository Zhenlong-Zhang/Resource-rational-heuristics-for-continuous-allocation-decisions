from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import math
import random

try:
    from ..mdp.meta_mdp import Action, ContinuousAllocationMetaMDP, EnvironmentConfig, TrueState
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import Action, ContinuousAllocationMetaMDP, EnvironmentConfig, TrueState


@dataclass(frozen=True)
class EvaluationEpisode:
    """A reproducible episode scaffold shared across policies."""

    episode_index: int
    true_state: TrueState
    observation_streams: Optional[Dict[Action, List[float]]] = None


def build_observation_streams(
    config: EnvironmentConfig,
    true_state: TrueState,
    seed: int,
    observations_per_person: int,
) -> Dict[Action, List[float]]:
    rng = random.Random(seed)
    return {
        ContinuousAllocationMetaMDP.SAMPLE_PERSON_1: [
            float(rng.gauss(true_state.need_1, config.sigma_sample))
            for _ in range(observations_per_person)
        ],
        ContinuousAllocationMetaMDP.SAMPLE_PERSON_2: [
            float(rng.gauss(true_state.need_2, config.sigma_sample))
            for _ in range(observations_per_person)
        ],
    }


def build_evaluation_episodes(
    config: EnvironmentConfig,
    n_episodes: int,
    include_observation_streams: bool = False,
    observations_per_person: int = 100,
    seed_stride: int = 17,
    seed_offset: int = 1,
    observation_seed_offset: int = 100_000,
) -> List[EvaluationEpisode]:
    """Draw common true states and optionally common observation streams."""

    episodes: List[EvaluationEpisode] = []
    base_seed = config.random_seed or 0
    if include_observation_streams:
        max_possible_samples = math.ceil(config.total_time / max(config.sample_time_cost, 1e-9))
        max_prior_samples = max(config.prior_sample_count_1, config.prior_sample_count_2)
        observations_per_person = max(
            observations_per_person,
            max_possible_samples + max_prior_samples + 5,
        )
    for episode_index in range(n_episodes):
        true_state_seed = base_seed + episode_index * seed_stride + seed_offset
        mdp = ContinuousAllocationMetaMDP(replace(config, random_seed=true_state_seed))
        true_state = mdp.sample_true_state()
        streams = None
        if include_observation_streams:
            streams = build_observation_streams(
                config=config,
                true_state=true_state,
                seed=base_seed + observation_seed_offset + episode_index * seed_stride,
                observations_per_person=observations_per_person,
            )
        episodes.append(
            EvaluationEpisode(
                episode_index=episode_index,
                true_state=true_state,
                observation_streams=streams,
            )
        )
    return episodes


def ensure_evaluation_episodes(
    config: EnvironmentConfig,
    n_episodes: int,
    evaluation_episodes: Optional[Sequence[EvaluationEpisode]] = None,
    include_observation_streams: bool = False,
    observations_per_person: int = 100,
) -> List[EvaluationEpisode]:
    if evaluation_episodes is not None:
        return list(evaluation_episodes)
    return build_evaluation_episodes(
        config=config,
        n_episodes=n_episodes,
        include_observation_streams=include_observation_streams,
        observations_per_person=observations_per_person,
    )


def observation_streams_for_mdp(
    episode: EvaluationEpisode,
) -> Optional[Dict[Action, List[float]]]:
    if episode.observation_streams is None:
        return None
    return {
        action: list(observations)
        for action, observations in episode.observation_streams.items()
    }
