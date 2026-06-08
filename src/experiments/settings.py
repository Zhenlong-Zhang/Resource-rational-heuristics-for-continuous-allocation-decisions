from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

try:
    from ..mdp.meta_mdp import MetaPolicy
    from ..policies.voi import BlinkeredPolicy, MyopicValueOfInformationPolicy
    from ..solvers.dp import DiscretizedDynamicProgrammingPolicy
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import MetaPolicy
    from policies.voi import BlinkeredPolicy, MyopicValueOfInformationPolicy
    from solvers.dp import DiscretizedDynamicProgrammingPolicy


@dataclass(frozen=True)
class EvaluationSettings:
    """Shared simulation settings for smoke, local, and server-scale runs."""

    n_episodes: int
    rr_observation_draws: int
    blinkered_horizon: int
    blinkered_observation_draws: int
    dp_max_samples: int
    dp_mean_grid_size: int
    dp_observation_branches: int
    use_common_observation_streams: bool
    observations_per_person: int


SMOKE_EVALUATION_SETTINGS = EvaluationSettings(
    n_episodes=12,
    rr_observation_draws=24,
    blinkered_horizon=2,
    blinkered_observation_draws=6,
    dp_max_samples=2,
    dp_mean_grid_size=7,
    dp_observation_branches=3,
    use_common_observation_streams=False,
    observations_per_person=100,
)

SERIOUS_LOCAL_EVALUATION_SETTINGS = EvaluationSettings(
    n_episodes=120,
    rr_observation_draws=500,
    blinkered_horizon=2,
    blinkered_observation_draws=100,
    dp_max_samples=4,
    dp_mean_grid_size=21,
    dp_observation_branches=5,
    use_common_observation_streams=True,
    observations_per_person=200,
)

SERVER_EVALUATION_SETTINGS = EvaluationSettings(
    n_episodes=1200,
    rr_observation_draws=500,
    blinkered_horizon=2,
    blinkered_observation_draws=250,
    dp_max_samples=10,
    dp_mean_grid_size=50,
    dp_observation_branches=7,
    use_common_observation_streams=True,
    observations_per_person=500,
)


def settings_with_overrides(
    settings: EvaluationSettings,
    n_episodes: int | None = None,
    rr_observation_draws: int | None = None,
    blinkered_observation_draws: int | None = None,
    use_common_observation_streams: bool | None = None,
    observations_per_person: int | None = None,
) -> EvaluationSettings:
    updates = {}
    if n_episodes is not None:
        updates["n_episodes"] = n_episodes
    if rr_observation_draws is not None:
        updates["rr_observation_draws"] = rr_observation_draws
    if blinkered_observation_draws is not None:
        updates["blinkered_observation_draws"] = blinkered_observation_draws
    if use_common_observation_streams is not None:
        updates["use_common_observation_streams"] = use_common_observation_streams
    if observations_per_person is not None:
        updates["observations_per_person"] = observations_per_person
    return replace(settings, **updates)


def build_rr_policy_from_settings(settings: EvaluationSettings) -> MetaPolicy:
    return MyopicValueOfInformationPolicy(observation_draws=settings.rr_observation_draws)


def build_rr_approximation_policies_from_settings(
    settings: EvaluationSettings,
) -> Sequence[MetaPolicy]:
    return [
        MyopicValueOfInformationPolicy(observation_draws=settings.rr_observation_draws),
        BlinkeredPolicy(
            horizon=settings.blinkered_horizon,
            observation_draws=settings.blinkered_observation_draws,
        ),
        DiscretizedDynamicProgrammingPolicy(
            max_samples=settings.dp_max_samples,
            mean_grid_size=settings.dp_mean_grid_size,
            observation_branches=settings.dp_observation_branches,
        ),
    ]
