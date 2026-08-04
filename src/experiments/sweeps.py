from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence

try:
    from ..mdp.meta_mdp import EnvironmentConfig, MetaPolicy
    from .regimes import compare_policy_behavior_profiles, compare_rr_to_heuristics_by_final_choice
    from .settings import EvaluationSettings, SMOKE_EVALUATION_SETTINGS, build_rr_policy_from_settings
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import EnvironmentConfig, MetaPolicy
    from experiments.regimes import compare_policy_behavior_profiles, compare_rr_to_heuristics_by_final_choice
    from experiments.settings import EvaluationSettings, SMOKE_EVALUATION_SETTINGS, build_rr_policy_from_settings


SWEEP_VALUES: Dict[str, Sequence[float]] = {
    "sigma_need": [10.0, 20.0, 40.0],
    "sigma_sample": [5.0, 10.0, 30.0],
    "total_time": [20.0, 60.0, 100.0],
    "lambda_shortfall": [1.0, 2.0, 4.0],
    "sample_time_cost": [1.0, 2.0, 4.0],
    "mu_need": [70.0, 100.0, 130.0],
    "utility_exponent": [0.5, 0.75],
    "learning_per_unit_of_tutoring": [0.75, 1.0, 1.25],
    "delta_learning_per_unit_tutoring": [-0.25, 0.0, 0.25],
    "prior_sample_count": [0, 2, 6],
    "initial_belief_gap": [0.0, 20.0, 40.0],
}

TARGETED_REGIME_GRID_VALUES: Dict[str, Dict[str, Sequence[float]]] = {
    # Designed to find regimes where the RR policy stops early and chooses
    # approximately equal division because information is unreliable or costly.
    "near_50_50": {
        "sample_time_cost": [1.0, 4.0, 8.0, 16.0, 32.0],
        "sigma_sample": [10.0, 30.0, 60.0, 100.0],
        "sigma_need": [2.0, 5.0, 10.0, 20.0],
        "total_time": [5.0, 10.0, 20.0, 40.0],
    },
    # Designed to find regimes where both recipients can often be brought close
    # to their need threshold, making equal-outcome/maximin behavior plausible.
    "equal_outcome": {
        "mu_need": [10.0, 20.0, 25.0, 30.0, 40.0, 50.0],
        "total_time": [60.0, 80.0, 100.0],
        "learning_per_unit_of_tutoring": [1.0, 1.25, 1.5, 2.0],
        "sigma_need": [5.0, 10.0, 15.0, 20.0],
    },
    # Smaller first-pass grid for the same theoretical target as near_50_50.
    # Use this before the larger grid when cluster feedback needs to be fast.
    "near_50_50_focused": {
        "sample_time_cost": [8.0, 16.0, 32.0],
        "sigma_sample": [60.0, 100.0],
        "sigma_need": [2.0, 5.0],
        "total_time": [5.0, 10.0, 20.0],
    },
    # Smaller first-pass grid around the positive-utility/equal-outcome regimes
    # found in the serious Step 7 run.
    "equal_outcome_focused": {
        "mu_need": [10.0, 20.0, 25.0, 30.0],
        "total_time": [80.0, 100.0],
        "learning_per_unit_of_tutoring": [1.0, 1.5, 2.0],
        "sigma_need": [5.0, 10.0],
    },
    # Designed to find equal-outcome/maximin regimes that are not merely
    # symmetric 50/50 allocations. Prior samples can make the initial beliefs
    # asymmetric before any new metalevel sampling occurs.
    "equal_outcome_distinct_focused": {
        "mu_need": [20.0, 40.0],
        "sigma_need": [10.0, 25.0],
        "sigma_sample": [2.0, 5.0],
        "prior_sample_count": [5.0, 20.0],
        "total_time": [80.0, 120.0],
        "learning_per_unit_of_tutoring": [1.0, 1.5],
    },
    # Designed for Falk's Round 3 request: start from symmetric or very
    # limited prior knowledge, actively collect information, choose unequal
    # allocations, and approximate equal outcomes in the hidden true state.
    "active_search_equal_outcome_focused": {
        "mu_need": [25.0, 30.0, 35.0],
        "total_time": [80.0, 100.0, 120.0],
        "learning_per_unit_of_tutoring": [1.0, 1.25],
        "sigma_need": [15.0, 25.0, 40.0],
        "sigma_sample": [2.0, 5.0, 8.0],
        "sample_time_cost": [0.25, 0.5, 1.0],
        "prior_sample_count": [0.0],
    },
    # Narrow follow-up to the first 486-environment Round 3 active-search run.
    # It keeps no prior samples and concentrates around the strongest previous
    # active/unequal regions, with lower observation noise and lower positive
    # sampling time costs. Costs are kept positive so common-observation stream
    # lengths remain well-defined under the current finite-horizon simulator.
    "active_search_equal_outcome_narrow_followup": {
        "mu_need": [20.0, 25.0, 30.0, 35.0],
        "total_time": [100.0, 120.0, 140.0, 160.0],
        "learning_per_unit_of_tutoring": [1.0, 1.25, 1.5],
        "sigma_need": [10.0, 15.0, 25.0, 40.0],
        "sigma_sample": [1.0, 2.0],
        "sample_time_cost": [0.05, 0.1, 0.25],
        "prior_sample_count": [0.0],
    },
    # R4 diagnostic grid from Falk's 07/01 comments. This grid is not meant to
    # be an exhaustive ecological sweep. It deliberately creates larger need
    # separation, reliable/cheap information, no prior samples, and near
    # just-enough-time conditions so manual active-search equal-outcome behavior
    # should have a clear chance to beat equal split.
    #
    # `utility_exponent < 0.5` approximates "quickly flattens after needs are
    # met" with the current concave utility family. It is not a true plateau.
    "r4_diagnostic_active_search": {
        "mu_need": [35.0, 45.0, 55.0],
        "total_time": [80.0, 100.0, 120.0],
        "learning_per_unit_of_tutoring": [1.0, 1.25],
        "sigma_need": [40.0, 60.0, 80.0],
        "sigma_sample": [1.0, 2.0, 4.0],
        "sample_time_cost": [0.02, 0.05, 0.1],
        "utility_exponent": [0.25, 0.35],
        "prior_sample_count": [0.0],
    },
}

ONE_DIMENSIONAL_SWEEP_VALUES: Dict[str, Sequence[float]] = {
    "sigma_need": [2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    "sigma_sample": [1.0, 3.0, 5.0, 10.0, 20.0, 40.0],
    "total_time": [10.0, 20.0, 40.0, 60.0, 90.0, 120.0],
    "lambda_shortfall": [0.5, 1.0, 2.0, 4.0, 8.0],
    "sample_time_cost": [0.25, 0.5, 1.0, 2.0, 4.0],
    "mu_need": [20.0, 40.0, 70.0, 100.0, 130.0, 180.0],
    "utility_exponent": [0.25, 0.5, 0.75, 1.0],
    "learning_per_unit_of_tutoring": [0.5, 0.75, 1.0, 1.25, 1.5],
    "delta_learning_per_unit_tutoring": [-0.5, -0.25, 0.0, 0.25, 0.5],
    "prior_sample_count": [0, 1, 2, 5, 10, 20],
    "initial_belief_gap": [0.0, 5.0, 10.0, 20.0, 40.0, 80.0],
}


# Manual scale-up markers for later server runs.
DEFAULT_SWEEP_EPISODES = 6
SERVER_SWEEP_EPISODES = 200
DEFAULT_MAX_GRID_POINTS = 36
SERVER_MAX_GRID_POINTS = 2000

R5_SAMPLING_COST_PERCENTAGES: Sequence[float] = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)

R5_ORACLE_GRID_VALUES: Dict[str, Sequence[float]] = {
    "mu_need": [40.0, 60.0, 80.0],
    "sigma_need": [10.0, 20.0, 30.0],
    "total_time": [80.0, 120.0, 160.0],
    "utility_exponent": [0.25, 0.5],
    "lambda_shortfall": [2.0, 2.25],
}


def build_environment(name: str = "baseline") -> EnvironmentConfig:
    if name != "baseline":
        raise ValueError("sweeps.build_environment currently provides only the baseline preset.")
    return EnvironmentConfig(
        mu_need=100.0,
        sigma_need=20.0,
        sigma_sample=10.0,
        total_time=60.0,
        lambda_shortfall=2.0,
        expected_utility_draws=100,
        allocation_grid_size=31,
        random_seed=11,
    )


def build_r5_oracle_map_configs(
    grid_values: Mapping[str, Sequence[float]] | None = None,
) -> List[tuple[str, EnvironmentConfig]]:
    """Build the prespecified unchanged-objective oracle discovery grid."""

    values = dict(grid_values or R5_ORACLE_GRID_VALUES)
    keys = list(values)
    configs: List[tuple[str, EnvironmentConfig]] = []
    for index, combo in enumerate(product(*(values[key] for key in keys))):
        config = replace(
            build_environment(),
            sigma_sample=20.0,
            sample_time_cost=0.1,
            prior_sample_count_1=0,
            prior_sample_count_2=0,
            allocation_grid_size=401,
            expected_utility_draws=500,
        )
        for key, value in zip(keys, combo):
            config = replace(config, **{key: value})
        label = f"r5_oracle_{index:04d}_" + "_".join(
            f"{key}={value:g}" for key, value in zip(keys, combo)
        )
        configs.append((label, config))
    return configs


def build_r5_sampling_cost_configs(
    anchors: Sequence[tuple[str, EnvironmentConfig]],
    percentages: Sequence[float] = R5_SAMPLING_COST_PERCENTAGES,
    zero_cost_cap: int = 12,
    positive_cost_cap: int = 40,
) -> List[tuple[str, EnvironmentConfig]]:
    """Vary only sample time while preserving every anchor parameter."""

    configs: List[tuple[str, EnvironmentConfig]] = []
    for anchor_name, anchor in anchors:
        for percentage in percentages:
            sample_time_cost = anchor.total_time * float(percentage) / 100.0
            cap = zero_cost_cap if percentage == 0.0 else positive_cost_cap
            label = f"{anchor_name}_sample_pct={percentage:g}"
            configs.append(
                (
                    label,
                    replace(
                        anchor,
                        sample_time_cost=sample_time_cost,
                        max_meta_samples=cap,
                        prior_sample_count_1=0,
                        prior_sample_count_2=0,
                    ),
                )
            )
    return configs


def build_r5_oat_configs(
    anchor_name: str,
    anchor: EnvironmentConfig,
    feature_values: Mapping[str, Sequence[float]],
) -> List[tuple[str, str, float, EnvironmentConfig]]:
    """Build controlled one-factor-at-a-time slices around one frozen anchor."""

    configs: List[tuple[str, str, float, EnvironmentConfig]] = []
    for feature, values in feature_values.items():
        for value in values:
            if feature == "sample_time_cost_percent":
                config = replace(
                    anchor,
                    sample_time_cost=anchor.total_time * float(value) / 100.0,
                    max_meta_samples=12 if value == 0 else 40,
                )
            else:
                config = replace(anchor, **{feature: value})
            configs.append(
                (
                    feature,
                    float(value),
                    f"{anchor_name}_{feature}={value:g}",
                    config,
                )
            )
    return configs


def build_r5_six_sample_configs() -> List[tuple[str, EnvironmentConfig]]:
    """Discovery grid where repeated observations can remain decision-relevant."""

    values: Dict[str, Sequence[float]] = {
        # Center the discovery grid on the strongest high-separation oracle
        # region from the R5 full-information map (mu=60, T=160).
        "sigma_need": [20.0, 30.0, 40.0],
        "sigma_sample": [10.0, 20.0, 30.0],
        "total_time": [140.0, 160.0, 180.0],
        "utility_exponent": [0.25, 0.5],
        "sample_time_cost_percent": [0.0, 0.001, 0.005, 0.01, 0.025, 0.05],
    }
    keys = list(values)
    configs: List[tuple[str, EnvironmentConfig]] = []
    for index, combo in enumerate(product(*(values[key] for key in keys))):
        parameters = dict(zip(keys, combo))
        total_time = float(parameters.pop("total_time"))
        percentage = float(parameters.pop("sample_time_cost_percent"))
        config = replace(
            build_environment(),
            mu_need=60.0,
            total_time=total_time,
            learning_per_unit_of_tutoring=1.0,
            delta_learning_per_unit_tutoring=0.0,
            lambda_shortfall=2.25,
            sample_time_cost=total_time * percentage / 100.0,
            prior_sample_count_1=0,
            prior_sample_count_2=0,
            max_meta_samples=12 if percentage == 0.0 else 40,
            allocation_grid_size=401,
            expected_utility_draws=500,
            **parameters,
        )
        label = f"r5_six_{index:04d}_" + "_".join(
            f"{key}={value:g}" for key, value in zip(keys, combo)
        )
        configs.append((label, config))
    return configs


def build_positive_and_near_zero_utility_configs() -> List[tuple[str, EnvironmentConfig]]:
    return [
        (
            "near_zero_utility",
            replace(
                build_environment(),
                mu_need=35.0,
                sigma_need=15.0,
                sigma_sample=8.0,
                total_time=80.0,
                random_seed=12,
            ),
        ),
        (
            "positive_utility_low_need",
            replace(
                build_environment(),
                mu_need=25.0,
                sigma_need=10.0,
                sigma_sample=6.0,
                total_time=80.0,
                random_seed=13,
            ),
        ),
        (
            "positive_utility_high_efficiency",
            replace(
                build_environment(),
                mu_need=50.0,
                sigma_need=15.0,
                sigma_sample=8.0,
                total_time=80.0,
                learning_per_unit_of_tutoring=1.5,
                random_seed=14,
            ),
        ),
    ]


def _config_with_sweep_override(
    config: EnvironmentConfig,
    feature: str,
    value: float,
) -> EnvironmentConfig:
    if feature == "prior_sample_count":
        return replace(
            config,
            prior_sample_count_1=int(value),
            prior_sample_count_2=int(value),
        )
    if feature == "initial_belief_gap":
        return replace(
            config,
            initial_mean_1=config.mu_need + float(value) / 2.0 if value else None,
            initial_mean_2=config.mu_need - float(value) / 2.0 if value else None,
        )
    return replace(config, **{feature: value})


def build_sweep_configs(
    sweep_values: Mapping[str, Sequence[float]] | None = None,
    base_environment_name: str = "baseline",
    max_grid_points: int = DEFAULT_MAX_GRID_POINTS,
) -> List[tuple[str, EnvironmentConfig]]:
    values = dict(sweep_values or SWEEP_VALUES)
    keys = list(values.keys())
    configs: List[tuple[str, EnvironmentConfig]] = []

    for index, combo in enumerate(product(*(values[key] for key in keys))):
        if len(configs) >= max_grid_points:
            break
        config = build_environment(base_environment_name)
        for key, value in zip(keys, combo):
            config = _config_with_sweep_override(config, key, value)
        label_parts = [f"{key}={value:g}" for key, value in zip(keys, combo)]
        configs.append((f"sweep_{index:04d}_" + "_".join(label_parts), config))
    return configs


def build_one_dimensional_sweep_configs(
    feature: str,
    values: Sequence[float] | None = None,
    base_environment_name: str = "baseline",
) -> List[tuple[str, EnvironmentConfig]]:
    selected_values = list(values or ONE_DIMENSIONAL_SWEEP_VALUES[feature])
    base_config = build_environment(base_environment_name)
    configs: List[tuple[str, EnvironmentConfig]] = []
    for value in selected_values:
        label = f"{feature}={value:g}"
        configs.append(
            (
                f"one_dimensional_{label}",
                _config_with_sweep_override(base_config, feature, float(value)),
            )
        )
    return configs


def build_all_one_dimensional_sweep_configs(
    sweep_values: Mapping[str, Sequence[float]] | None = None,
    base_environment_name: str = "baseline",
) -> List[tuple[str, float, str, EnvironmentConfig]]:
    values = dict(sweep_values or ONE_DIMENSIONAL_SWEEP_VALUES)
    rows: List[tuple[str, float, str, EnvironmentConfig]] = []
    for feature, feature_values in values.items():
        for environment_name, config in build_one_dimensional_sweep_configs(
            feature=feature,
            values=feature_values,
            base_environment_name=base_environment_name,
        ):
            value = float(environment_name.split("=")[-1])
            rows.append((feature, value, environment_name, config))
    return rows


def _targeted_regime_base_environment(grid_name: str) -> EnvironmentConfig:
    base = build_environment()
    if grid_name in {"near_50_50", "near_50_50_focused"}:
        return replace(
            base,
            mu_need=100.0,
            learning_per_unit_of_tutoring=1.0,
            delta_learning_per_unit_tutoring=0.0,
        )
    if grid_name in {
        "equal_outcome",
        "equal_outcome_focused",
        "equal_outcome_distinct_focused",
        "active_search_equal_outcome_focused",
        "active_search_equal_outcome_narrow_followup",
        "r4_diagnostic_active_search",
    }:
        return replace(
            base,
            sigma_sample=8.0,
            sample_time_cost=1.0,
            delta_learning_per_unit_tutoring=0.0,
        )
    raise ValueError(f"Unknown targeted regime grid: {grid_name}")


def build_targeted_regime_grid_configs(
    grid_names: Sequence[str] | None = None,
    max_grid_points: int | None = None,
) -> List[tuple[str, int, str, Dict[str, float], EnvironmentConfig]]:
    selected_grid_names = list(grid_names or TARGETED_REGIME_GRID_VALUES)
    missing = sorted(set(selected_grid_names) - set(TARGETED_REGIME_GRID_VALUES))
    if missing:
        raise ValueError(f"Unknown targeted regime grids: {missing}")

    configs: List[tuple[str, int, str, Dict[str, float], EnvironmentConfig]] = []
    for grid_name in selected_grid_names:
        grid_values = TARGETED_REGIME_GRID_VALUES[grid_name]
        keys = list(grid_values)
        base_config = _targeted_regime_base_environment(grid_name)
        for grid_index, combo in enumerate(product(*(grid_values[key] for key in keys))):
            if max_grid_points is not None and len(configs) >= max_grid_points:
                return configs
            parameter_values = {key: float(value) for key, value in zip(keys, combo)}
            config = base_config
            for key, value in parameter_values.items():
                config = _config_with_sweep_override(config, key, value)
            label = f"{grid_name}_{grid_index:04d}_" + "_".join(
                f"{key}={value:g}" for key, value in parameter_values.items()
            )
            configs.append((grid_name, grid_index, label, parameter_values, config))
    return configs


def compact_config_row(environment: str, config: EnvironmentConfig) -> Dict[str, float | int | str | None]:
    row = asdict(config)
    row["environment"] = environment
    return row


def run_final_choice_sweep(
    sweep_configs: Iterable[tuple[str, EnvironmentConfig]],
    n_episodes: int = DEFAULT_SWEEP_EPISODES,
    allocation_tolerance: float = 0.05,
    rr_policy: MetaPolicy | None = None,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for environment_name, config in sweep_configs:
        rows.extend(
            compare_rr_to_heuristics_by_final_choice(
                environment_name=environment_name,
                config=config,
                n_episodes=n_episodes,
                allocation_tolerance=allocation_tolerance,
                rr_policy=rr_policy,
            )
        )
    return rows


def _attach_sweep_metadata(
    rows: List[Dict[str, float | str]],
    feature: str,
    value: float,
) -> List[Dict[str, float | str]]:
    return [
        {
            "sweep_feature": feature,
            "sweep_value": value,
            **row,
        }
        for row in rows
    ]


def _attach_targeted_regime_metadata(
    rows: List[Dict[str, float | str]],
    grid_name: str,
    grid_index: int,
    parameter_values: Dict[str, float],
) -> List[Dict[str, float | str]]:
    metadata: Dict[str, float | str] = {
        "regime_grid": grid_name,
        "grid_index": float(grid_index),
    }
    metadata.update(parameter_values)
    return [
        {
            **metadata,
            **row,
        }
        for row in rows
    ]


def run_one_dimensional_final_choice_sweeps(
    sweep_configs: Iterable[tuple[str, float, str, EnvironmentConfig]],
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    allocation_tolerance: float = 0.05,
    rr_policy: MetaPolicy | None = None,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    policy = rr_policy or build_rr_policy_from_settings(settings)
    for feature, value, environment_name, config in sweep_configs:
        feature_rows = compare_rr_to_heuristics_by_final_choice(
            environment_name=environment_name,
            config=config,
            n_episodes=settings.n_episodes,
            allocation_tolerance=allocation_tolerance,
            rr_policy=policy,
            use_common_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        rows.extend(_attach_sweep_metadata(feature_rows, feature, value))
    return rows


def run_one_dimensional_rr_behavior_sweeps(
    sweep_configs: Iterable[tuple[str, float, str, EnvironmentConfig]],
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    allocation_tolerance: float = 0.05,
    rr_policy: MetaPolicy | None = None,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    policy = rr_policy or build_rr_policy_from_settings(settings)
    for feature, value, environment_name, config in sweep_configs:
        feature_rows = compare_policy_behavior_profiles(
            environment_name=environment_name,
            config=config,
            n_episodes=settings.n_episodes,
            allocation_tolerance=allocation_tolerance,
            rr_policy=policy,
            policies=[],
            use_common_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        rows.extend(_attach_sweep_metadata(feature_rows, feature, value))
    return rows


def run_targeted_regime_final_choice_grid(
    regime_configs: Iterable[tuple[str, int, str, Dict[str, float], EnvironmentConfig]],
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    allocation_tolerance: float = 0.05,
    rr_policy: MetaPolicy | None = None,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    policy = rr_policy or build_rr_policy_from_settings(settings)
    for grid_name, grid_index, environment_name, parameter_values, config in regime_configs:
        grid_rows = compare_rr_to_heuristics_by_final_choice(
            environment_name=environment_name,
            config=config,
            n_episodes=settings.n_episodes,
            allocation_tolerance=allocation_tolerance,
            rr_policy=policy,
            use_common_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        rows.extend(_attach_targeted_regime_metadata(grid_rows, grid_name, grid_index, parameter_values))
    return rows


def run_targeted_regime_behavior_grid(
    regime_configs: Iterable[tuple[str, int, str, Dict[str, float], EnvironmentConfig]],
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    allocation_tolerance: float = 0.05,
    rr_policy: MetaPolicy | None = None,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    policy = rr_policy or build_rr_policy_from_settings(settings)
    for grid_name, grid_index, environment_name, parameter_values, config in regime_configs:
        grid_rows = compare_policy_behavior_profiles(
            environment_name=environment_name,
            config=config,
            n_episodes=settings.n_episodes,
            allocation_tolerance=allocation_tolerance,
            rr_policy=policy,
            policies=[],
            use_common_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        rows.extend(_attach_targeted_regime_metadata(grid_rows, grid_name, grid_index, parameter_values))
    return rows


def identify_rr_behavior_regime_candidates(
    behavior_rows: Sequence[Dict[str, float | str]],
    near_equal_rate_threshold: float = 0.9,
    equal_outcome_rate_threshold: float = 0.9,
    true_equal_outcome_rate_threshold: float = 0.9,
    active_search_sample_threshold: float = 1.0,
    unequal_allocation_threshold: float = 0.05,
) -> List[Dict[str, float | str]]:
    candidates: List[Dict[str, float | str]] = []
    for row in behavior_rows:
        if row.get("policy_type") != "rr_approximation":
            continue
        near_equal_rate = float(row["near_equal_allocation_rate"])
        equal_outcome_rate = float(row["equal_outcome_rate"])
        true_equal_outcome_rate = float(row.get("true_equal_outcome_rate", 0.0))
        sample_count = float(row["mean_sample_count"])
        allocation_from_equal = float(row["mean_abs_allocation_from_equal"])
        closer_to_true_equal = float(row.get("closer_to_true_equal_outcome_than_equal_split_rate", 0.0))
        if near_equal_rate >= near_equal_rate_threshold:
            candidates.append({"candidate_type": "near_always_50_50", **row})
        if equal_outcome_rate >= equal_outcome_rate_threshold:
            candidates.append({"candidate_type": "near_always_equal_outcome", **row})
        if true_equal_outcome_rate >= true_equal_outcome_rate_threshold:
            candidates.append({"candidate_type": "near_always_true_equal_outcome", **row})
        if (
            true_equal_outcome_rate >= true_equal_outcome_rate_threshold
            and sample_count > active_search_sample_threshold
            and allocation_from_equal >= unequal_allocation_threshold
            and closer_to_true_equal >= 0.5
        ):
            candidates.append({"candidate_type": "active_search_true_equal_outcome", **row})
    return candidates


def identify_final_choice_regime_candidates(
    final_choice_rows: Sequence[Dict[str, float | str]],
    target_heuristics: Sequence[str] = ("equal_division", "equal_outcome", "maximin_equal_outcome"),
    match_rate_threshold: float = 0.9,
    mean_abs_gap_threshold: float = 0.05,
) -> List[Dict[str, float | str]]:
    candidates: List[Dict[str, float | str]] = []
    for row in final_choice_rows:
        if row.get("heuristic") not in target_heuristics:
            continue
        if (
            float(row["final_choice_match_rate"]) >= match_rate_threshold
            or float(row["mean_abs_allocation_gap"]) <= mean_abs_gap_threshold
        ):
            candidates.append({"candidate_type": f"close_to_{row['heuristic']}", **row})
    return candidates


def run_target_regime_search(
    sweep_configs: Iterable[tuple[str, float, str, EnvironmentConfig]],
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    allocation_tolerance: float = 0.05,
) -> Dict[str, List[Dict[str, float | str]]]:
    final_choice_rows = run_one_dimensional_final_choice_sweeps(
        sweep_configs=sweep_configs,
        settings=settings,
        allocation_tolerance=allocation_tolerance,
    )
    behavior_rows = run_one_dimensional_rr_behavior_sweeps(
        sweep_configs=sweep_configs,
        settings=settings,
        allocation_tolerance=allocation_tolerance,
    )
    return {
        "final_choice_rows": final_choice_rows,
        "behavior_rows": behavior_rows,
        "final_choice_candidates": identify_final_choice_regime_candidates(final_choice_rows),
        "behavior_candidates": identify_rr_behavior_regime_candidates(behavior_rows),
    }


def summarize_sweep_regimes(rows: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    environments = sorted({str(row["environment"]) for row in rows})
    summary: List[Dict[str, float | str]] = []
    for environment in environments:
        env_rows = [row for row in rows if row["environment"] == environment]
        best_match = max(
            env_rows,
            key=lambda row: (
                float(row["final_choice_match_rate"]),
                -float(row["mean_abs_allocation_gap"]),
            ),
        )
        closest_utility = min(
            env_rows,
            key=lambda row: abs(float(row["utility_gap_rr_minus_heuristic"])),
        )
        summary.append(
            {
                "environment": environment,
                "best_match_heuristic": best_match["heuristic"],
                "best_match_rate": best_match["final_choice_match_rate"],
                "best_match_allocation_gap": best_match["mean_abs_allocation_gap"],
                "closest_utility_heuristic": closest_utility["heuristic"],
                "closest_utility_gap": closest_utility["utility_gap_rr_minus_heuristic"],
                "closest_utility_gap_ci95": closest_utility["utility_gap_ci95"],
            }
        )
    return summary
