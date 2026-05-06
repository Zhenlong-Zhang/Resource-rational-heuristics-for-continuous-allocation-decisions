from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Sequence, Tuple

import os

from mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig
from policies.heuristic import build_policy_library
from policies.voi import MyopicValueOfInformationPolicy


ENVIRONMENT_LIBRARY: Dict[str, EnvironmentConfig] = {
    "baseline": EnvironmentConfig(
        mu_need=100.0,
        sigma_need=20.0,
        sigma_sample=10.0,
        total_time=60.0,
        lambda_shortfall=2.0,
        expected_utility_draws=100,
        allocation_grid_size=31,
        random_seed=11,
    ),
    "high_need_variability": EnvironmentConfig(
        mu_need=100.0,
        sigma_need=40.0,
        sigma_sample=10.0,
        total_time=60.0,
        lambda_shortfall=2.0,
        expected_utility_draws=100,
        allocation_grid_size=31,
        random_seed=11,
    ),
    "noisy_information": EnvironmentConfig(
        mu_need=100.0,
        sigma_need=20.0,
        sigma_sample=30.0,
        total_time=60.0,
        lambda_shortfall=2.0,
        expected_utility_draws=100,
        allocation_grid_size=31,
        random_seed=11,
    ),
    "scarce_time": EnvironmentConfig(
        mu_need=100.0,
        sigma_need=20.0,
        sigma_sample=10.0,
        total_time=20.0,
        lambda_shortfall=2.0,
        expected_utility_draws=100,
        allocation_grid_size=31,
        random_seed=11,
    ),
}


def build_environment(name: str = "baseline", **overrides: float) -> EnvironmentConfig:
    return replace(ENVIRONMENT_LIBRARY[name], **overrides)


def model_overview(config: Optional[EnvironmentConfig] = None) -> Dict[str, object]:
    config = config or EnvironmentConfig()
    mdp = ContinuousAllocationMetaMDP(config)
    belief = mdp.initial_belief()
    allocation, expected_utility = mdp.solve_terminal_allocation(belief)
    return {
        "config": asdict(config),
        "initial_belief": {
            "mean_1": belief.mean_1,
            "var_1": belief.var_1,
            "mean_2": belief.mean_2,
            "var_2": belief.var_2,
            "deliberation_time": belief.deliberation_time,
        },
        "available_actions": mdp.available_actions(belief),
        "initial_terminal_allocation_to_person1": allocation,
        "initial_expected_terminal_utility": expected_utility,
    }


def evaluate_policy_library(
    environment_name: str = "baseline",
    n_episodes: int = 10,
    config: Optional[EnvironmentConfig] = None,
) -> List[Dict[str, float]]:
    config = config or build_environment(environment_name)
    mdp = ContinuousAllocationMetaMDP(config)
    policies = build_policy_library() + [MyopicValueOfInformationPolicy(observation_draws=24)]
    rows: List[Dict[str, float]] = []
    import statistics

    for policy in policies:
        results = [mdp.run_episode(policy=policy) for _ in range(n_episodes)]
        utilities = [r.realized_utility for r in results]
        deliberations = [r.final_belief.deliberation_time for r in results]
        allocations = [r.final_allocation_to_person1 for r in results]
        remaining_times = [r.remaining_time for r in results]
        rows.append(
            {
                "policy": policy.name,
                "mean_utility": float(statistics.mean(utilities)),
                "std_utility": float(statistics.pstdev(utilities)) if len(utilities) > 1 else 0.0,
                "mean_deliberation_time": float(statistics.mean(deliberations)),
                "mean_allocation_to_person1": float(statistics.mean(allocations)),
                "mean_remaining_time": float(statistics.mean(remaining_times)),
                "termination_rate": 1.0,
            }
        )
    return rows


def default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _evaluate_policy_task(args: Tuple[str, int, EnvironmentConfig]) -> List[Dict[str, float]]:
    environment_name, n_episodes, config = args
    return evaluate_policy_library(
        environment_name=environment_name,
        n_episodes=n_episodes,
        config=config,
    )


def run_strategy_comparison(
    environment_names: Sequence[str] | None = None,
    n_episodes: int = 10,
    max_workers: Optional[int] = None,
    parallel: bool = True,
) -> List[Dict[str, float | str]]:
    environment_names = environment_names or list(ENVIRONMENT_LIBRARY.keys())
    tasks = [
        (environment_name, n_episodes, build_environment(environment_name))
        for environment_name in environment_names
    ]
    rows: List[Dict[str, float | str]] = []

    if parallel and len(tasks) > 1:
        worker_count = max_workers or default_max_workers()
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(_evaluate_policy_task, tasks))
        except (OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(_evaluate_policy_task, tasks))
        for task, summaries in zip(tasks, results):
            environment_name = task[0]
            for summary in summaries:
                rows.append({"environment": environment_name, **summary})
        return rows

    for environment_name, _, config in tasks:
        for summary in evaluate_policy_library(
            environment_name=environment_name,
            n_episodes=n_episodes,
            config=config,
        ):
            rows.append({"environment": environment_name, **summary})
    return rows


def best_policy_by_environment(rows: Sequence[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    grouped: Dict[str, List[Dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["environment"]), []).append(row)
    best_rows: List[Dict[str, float | str]] = []
    for environment_name, candidates in grouped.items():
        best_row = max(candidates, key=lambda row: float(row["mean_utility"]))
        best_rows.append(
            {
                "environment": environment_name,
                "best_policy": best_row["policy"],
                "best_mean_utility": best_row["mean_utility"],
                "mean_deliberation_time": best_row["mean_deliberation_time"],
            }
        )
    return best_rows
