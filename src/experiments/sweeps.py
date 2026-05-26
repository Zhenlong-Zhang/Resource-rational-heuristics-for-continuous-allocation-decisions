from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence

try:
    from ..mdp.meta_mdp import EnvironmentConfig
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import EnvironmentConfig
from .compare import build_environment
from .regimes import compare_rr_to_heuristics_by_final_choice


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


# Manual scale-up markers for later server runs.
DEFAULT_SWEEP_EPISODES = 6
SERVER_SWEEP_EPISODES = 200
DEFAULT_MAX_GRID_POINTS = 36
SERVER_MAX_GRID_POINTS = 2000


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
        overrides = dict(zip(keys, combo))
        prior_sample_count = int(overrides.pop("prior_sample_count", 0))
        initial_belief_gap = float(overrides.pop("initial_belief_gap", 0.0))
        config = build_environment(base_environment_name)
        config = replace(
            config,
            **overrides,
            prior_sample_count_1=prior_sample_count,
            prior_sample_count_2=prior_sample_count,
            initial_mean_1=config.mu_need + initial_belief_gap / 2.0 if initial_belief_gap else None,
            initial_mean_2=config.mu_need - initial_belief_gap / 2.0 if initial_belief_gap else None,
        )
        label_parts = [f"{key}={value:g}" for key, value in zip(keys, combo)]
        configs.append((f"sweep_{index:04d}_" + "_".join(label_parts), config))
    return configs


def compact_config_row(environment: str, config: EnvironmentConfig) -> Dict[str, float | int | str | None]:
    row = asdict(config)
    row["environment"] = environment
    return row


def run_final_choice_sweep(
    sweep_configs: Iterable[tuple[str, EnvironmentConfig]],
    n_episodes: int = DEFAULT_SWEEP_EPISODES,
    allocation_tolerance: float = 0.05,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for environment_name, config in sweep_configs:
        rows.extend(
            compare_rr_to_heuristics_by_final_choice(
                environment_name=environment_name,
                config=config,
                n_episodes=n_episodes,
                allocation_tolerance=allocation_tolerance,
            )
        )
    return rows


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
