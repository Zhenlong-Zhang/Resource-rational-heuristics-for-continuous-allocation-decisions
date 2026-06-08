from __future__ import annotations

from typing import List, Optional, Sequence

try:
    from ..mdp.meta_mdp import EnvironmentConfig, MetaPolicy
    from ..policies.voi import MyopicValueOfInformationPolicy
    from ..solvers.dp import DiscretizedDynamicProgrammingPolicy
    from .randomization import EvaluationEpisode
    from .regimes import compare_rr_approximation_methods
    from .settings import EvaluationSettings, SMOKE_EVALUATION_SETTINGS
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import EnvironmentConfig, MetaPolicy
    from policies.voi import MyopicValueOfInformationPolicy
    from solvers.dp import DiscretizedDynamicProgrammingPolicy
    from experiments.randomization import EvaluationEpisode
    from experiments.regimes import compare_rr_approximation_methods
    from experiments.settings import EvaluationSettings, SMOKE_EVALUATION_SETTINGS


def build_dp_sensitivity_policies(
    max_samples_values: Sequence[int] = (2, 4, 6, 10),
    mean_grid_sizes: Sequence[int] = (7, 11, 21, 50),
    observation_branches_values: Sequence[int] = (3, 5),
) -> List[MetaPolicy]:
    policies: List[MetaPolicy] = []
    for max_samples in max_samples_values:
        for mean_grid_size in mean_grid_sizes:
            for observation_branches in observation_branches_values:
                policy = DiscretizedDynamicProgrammingPolicy(
                    max_samples=max_samples,
                    mean_grid_size=mean_grid_size,
                    observation_branches=observation_branches,
                )
                policy.name = (
                    f"discretized_dp_max{max_samples}"
                    f"_grid{mean_grid_size}"
                    f"_branches{observation_branches}"
                )
                policies.append(policy)
    return policies


def run_dp_sensitivity_analysis(
    environment_name: str,
    config: Optional[EnvironmentConfig] = None,
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    max_samples_values: Sequence[int] = (2, 4, 6, 10),
    mean_grid_sizes: Sequence[int] = (7, 11, 21, 50),
    observation_branches_values: Sequence[int] = (3, 5),
    include_myopic_reference: bool = True,
    evaluation_episodes: Optional[Sequence[EvaluationEpisode]] = None,
) -> List[dict[str, float | str]]:
    policies: List[MetaPolicy] = []
    if include_myopic_reference:
        policies.append(MyopicValueOfInformationPolicy(observation_draws=settings.rr_observation_draws))
    policies.extend(
        build_dp_sensitivity_policies(
            max_samples_values=max_samples_values,
            mean_grid_sizes=mean_grid_sizes,
            observation_branches_values=observation_branches_values,
        )
    )
    return compare_rr_approximation_methods(
        environment_name=environment_name,
        n_episodes=settings.n_episodes,
        config=config,
        policies=policies,
        evaluation_episodes=evaluation_episodes,
        use_common_observation_streams=settings.use_common_observation_streams,
        observations_per_person=settings.observations_per_person,
    )
