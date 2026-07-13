from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
import zlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_results import PRESETS, apply_environment_overrides  # noqa: E402
from src.experiments.compare import ENVIRONMENT_LIBRARY  # noqa: E402
from src.experiments.randomization import (  # noqa: E402
    EvaluationEpisode,
    build_evaluation_episodes,
    observation_streams_for_mdp,
)
from src.experiments.settings import settings_with_overrides  # noqa: E402
from src.experiments.sweeps import build_positive_and_near_zero_utility_configs  # noqa: E402
from src.mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, MetaPolicy  # noqa: E402
from src.policies.voi import BlinkeredPolicy, MyopicValueOfInformationPolicy  # noqa: E402
from src.solvers.dp import DiscretizedDynamicProgrammingPolicy  # noqa: E402


EPISODE_FIELDNAMES = [
    "environment",
    "episode_index",
    "policy",
    "policy_observation_draws",
    "policy_horizon",
    "policy_max_samples",
    "policy_mean_grid_size",
    "policy_observation_branches",
    "realized_utility",
    "sample_count",
    "allocation_to_person1",
    "elapsed_seconds",
]

CONFIG_SUMMARY_FIELDS = [
    "mu_need",
    "sigma_need",
    "sigma_sample",
    "total_time",
    "lambda_shortfall",
    "utility_exponent",
    "learning_per_unit_of_tutoring",
    "delta_learning_per_unit_tutoring",
    "prior_sample_count_1",
    "prior_sample_count_2",
    "sample_time_cost",
    "allocation_grid_size",
    "expected_utility_draws",
    "expected_utility_method",
    "gauss_hermite_order",
    "random_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one approximation-method comparison policy/config with checkpointed episode output."
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="server")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--voi-samples", type=int, default=None)
    parser.add_argument("--blinkered-samples", type=int, default=None)
    parser.add_argument("--common-observations", choices=["auto", "on", "off"], default="on")
    parser.add_argument("--observations-per-person", type=int, default=None)
    parser.add_argument("--allocation-grid-size", type=int, default=None)
    parser.add_argument("--expected-utility-draws", type=int, default=None)
    parser.add_argument("--terminal-integration", choices=["monte_carlo", "gauss_hermite"], default=None)
    parser.add_argument("--gauss-hermite-order", type=int, default=15)

    parser.add_argument("--policy", choices=["myopic_voi", "blinkered", "discretized_dp"], required=True)
    parser.add_argument("--policy-label", default="")
    parser.add_argument("--dp-max-samples", type=int, default=2)
    parser.add_argument("--dp-mean-grid-size", type=int, default=7)
    parser.add_argument("--dp-observation-branches", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace):
    if args.common_observations == "on":
        common_observations = True
    elif args.common_observations == "off":
        common_observations = False
    else:
        common_observations = None
    return settings_with_overrides(
        PRESETS[args.preset],
        n_episodes=args.episodes,
        rr_observation_draws=args.voi_samples,
        blinkered_observation_draws=args.blinkered_samples,
        use_common_observation_streams=common_observations,
        observations_per_person=args.observations_per_person,
    )


def environment_config(args: argparse.Namespace) -> EnvironmentConfig:
    configs = dict(ENVIRONMENT_LIBRARY)
    configs.update(dict(build_positive_and_near_zero_utility_configs()))
    if args.environment not in configs:
        raise ValueError(f"Unknown environment: {args.environment}")
    return apply_environment_overrides(configs[args.environment], args)


def policy_from_args(args: argparse.Namespace, settings) -> MetaPolicy:
    if args.policy == "myopic_voi":
        policy = MyopicValueOfInformationPolicy(observation_draws=settings.rr_observation_draws)
    elif args.policy == "blinkered":
        policy = BlinkeredPolicy(
            horizon=settings.blinkered_horizon,
            observation_draws=settings.blinkered_observation_draws,
        )
    else:
        policy = DiscretizedDynamicProgrammingPolicy(
            max_samples=args.dp_max_samples,
            mean_grid_size=args.dp_mean_grid_size,
            observation_branches=args.dp_observation_branches,
        )
        policy.name = (
            f"discretized_dp_max{args.dp_max_samples}"
            f"_grid{args.dp_mean_grid_size}"
            f"_branches{args.dp_observation_branches}"
        )
    if args.policy_label:
        policy.name = args.policy_label
    return policy


def policy_parameter(policy: MetaPolicy, attribute: str) -> float | str:
    value = getattr(policy, attribute, "")
    return float(value) if isinstance(value, (int, float)) else value


def episode_seed(config: EnvironmentConfig, episode: EvaluationEpisode, offset: int = 0) -> int:
    return (config.random_seed or 0) + episode.episode_index * 17 + 1 + offset


def stable_policy_seed_offset(policy_name: str) -> int:
    return 1_000 + (zlib.crc32(policy_name.encode("utf-8")) % 900_000)


def mdp_for_episode(
    config: EnvironmentConfig,
    episode: EvaluationEpisode,
    policy_name: str,
    use_common_observation_streams: bool,
) -> ContinuousAllocationMetaMDP:
    streams = observation_streams_for_mdp(episode) if use_common_observation_streams else None
    return ContinuousAllocationMetaMDP(
        replace(config, random_seed=episode_seed(config, episode, stable_policy_seed_offset(policy_name))),
        observation_streams=streams,
    )


def mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def ci95(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def read_completed_episode_indices(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["episode_index"])
            for row in csv.DictReader(handle)
            if row.get("episode_index", "").strip()
        }


def read_episode_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_episode_row(path: Path, row: Dict[str, object], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_summary(path: Path, args: argparse.Namespace, settings, policy: MetaPolicy, episode_rows: List[Dict[str, str]]) -> None:
    utilities = [float(row["realized_utility"]) for row in episode_rows]
    sample_counts = [float(row["sample_count"]) for row in episode_rows]
    allocations = [float(row["allocation_to_person1"]) for row in episode_rows]
    config = environment_config(args)
    config_dict = asdict(config)
    summary = {
        "environment": args.environment,
        "n_episodes": len(episode_rows),
        "target_episodes": settings.n_episodes,
        "policy": policy.name,
        "policy_observation_draws": policy_parameter(policy, "observation_draws"),
        "policy_horizon": policy_parameter(policy, "horizon"),
        "policy_max_samples": policy_parameter(policy, "max_samples"),
        "policy_mean_grid_size": policy_parameter(policy, "mean_grid_size"),
        "policy_observation_branches": policy_parameter(policy, "observation_branches"),
        "common_true_states": 1.0,
        "common_observation_streams": 1.0 if settings.use_common_observation_streams else 0.0,
        "observations_per_person": settings.observations_per_person,
        "mean_utility": mean(utilities),
        "mean_utility_ci95": ci95(utilities),
        "regret_vs_best_rr_approximation": "",
        "mean_sample_count": mean(sample_counts),
        "mean_allocation_to_person1": mean(allocations),
        "completed_episodes": len(episode_rows),
        "complete": 1.0 if len(episode_rows) >= settings.n_episodes else 0.0,
    }
    for field in CONFIG_SUMMARY_FIELDS:
        summary[field] = config_dict.get(field, "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    args = parse_args()
    settings = settings_from_args(args)
    config = environment_config(args)
    policy = policy_from_args(args, settings)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "rr_approximation_method_episode_results.csv"
    summary_path = output_dir / "rr_approximation_methods_comparison.csv"
    completed = read_completed_episode_indices(episode_path) if args.resume else set()
    write_header = not episode_path.exists() or episode_path.stat().st_size == 0

    episodes = build_evaluation_episodes(
        config=config,
        n_episodes=settings.n_episodes,
        include_observation_streams=settings.use_common_observation_streams,
        observations_per_person=settings.observations_per_person,
    )
    pending_since_summary = 0
    for episode in episodes:
        if episode.episode_index in completed:
            continue
        start = time.time()
        mdp = mdp_for_episode(
            config=config,
            episode=episode,
            policy_name=policy.name,
            use_common_observation_streams=settings.use_common_observation_streams,
        )
        result = mdp.run_episode(policy, true_state=episode.true_state)
        append_episode_row(
            episode_path,
            {
                "environment": args.environment,
                "episode_index": episode.episode_index,
                "policy": policy.name,
                "policy_observation_draws": policy_parameter(policy, "observation_draws"),
                "policy_horizon": policy_parameter(policy, "horizon"),
                "policy_max_samples": policy_parameter(policy, "max_samples"),
                "policy_mean_grid_size": policy_parameter(policy, "mean_grid_size"),
                "policy_observation_branches": policy_parameter(policy, "observation_branches"),
                "realized_utility": result.realized_utility,
                "sample_count": len(result.samples),
                "allocation_to_person1": result.final_allocation_to_person1,
                "elapsed_seconds": time.time() - start,
            },
            write_header=write_header,
        )
        write_header = False
        pending_since_summary += 1
        if pending_since_summary >= max(1, args.flush_every):
            write_summary(summary_path, args, settings, policy, read_episode_rows(episode_path))
            pending_since_summary = 0
    write_summary(summary_path, args, settings, policy, read_episode_rows(episode_path))
    print(f"Generated method comparison task output in {output_dir}")


if __name__ == "__main__":
    main()
