from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.randomization import (  # noqa: E402
    build_evaluation_episodes,
    observation_streams_for_mdp,
)
from src.mdp.meta_mdp import BeliefState, ContinuousAllocationMetaMDP, EnvironmentConfig  # noqa: E402


class TwoSampleThenTerminatePolicy:
    name = "two_sample_then_terminate"

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> str:
        if len(belief.history) == 0:
            return mdp.SAMPLE_PERSON_1
        if len(belief.history) == 1:
            return mdp.SAMPLE_PERSON_2
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> None:
        return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return math.nan
    return sum(x * y for x, y in zip(centered_x, centered_y)) / (denom_x * denom_y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that common observation streams are generated from each episode's true state."
    )
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--observations-per-person", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sigma-need", type=float, default=20.0)
    parser.add_argument("--sigma-sample", type=float, default=10.0)
    parser.add_argument("--min-correlation", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EnvironmentConfig(
        sigma_need=args.sigma_need,
        sigma_sample=args.sigma_sample,
        random_seed=args.seed,
        prior_sample_count_1=0,
        prior_sample_count_2=0,
    )
    episodes = build_evaluation_episodes(
        config=config,
        n_episodes=args.episodes,
        include_observation_streams=True,
        observations_per_person=args.observations_per_person,
    )

    true_1: List[float] = []
    true_2: List[float] = []
    stream_mean_1: List[float] = []
    stream_mean_2: List[float] = []
    true_state_mismatches = 0
    first_sample_mismatches = 0

    policy = TwoSampleThenTerminatePolicy()
    for episode in episodes:
        streams = observation_streams_for_mdp(episode)
        if streams is None:
            raise RuntimeError("Expected observation streams but got None")

        person1_stream = streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_1]
        person2_stream = streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_2]
        true_1.append(episode.true_state.need_1)
        true_2.append(episode.true_state.need_2)
        stream_mean_1.append(_mean(person1_stream))
        stream_mean_2.append(_mean(person2_stream))

        mdp = ContinuousAllocationMetaMDP(
            replace(config, random_seed=args.seed + episode.episode_index * 31),
            observation_streams=streams,
        )
        result = mdp.run_episode(policy, true_state=episode.true_state)
        if result.true_state != episode.true_state:
            true_state_mismatches += 1
        if len(result.samples) < 2:
            first_sample_mismatches += 1
            continue
        if abs(result.samples[0]["observation"] - person1_stream[0]) > 1e-12:
            first_sample_mismatches += 1
        if abs(result.samples[1]["observation"] - person2_stream[0]) > 1e-12:
            first_sample_mismatches += 1

    corr_1 = _correlation(true_1, stream_mean_1)
    corr_2 = _correlation(true_2, stream_mean_2)
    summary = {
        "n_episodes": len(episodes),
        "observations_per_person": args.observations_per_person,
        "sigma_need": args.sigma_need,
        "sigma_sample": args.sigma_sample,
        "true_state_mismatches": true_state_mismatches,
        "first_sample_mismatches": first_sample_mismatches,
        "person1_stream_mean_true_need_correlation": corr_1,
        "person2_stream_mean_true_need_correlation": corr_2,
        "person1_mean_abs_stream_mean_error": _mean(
            [abs(stream_mean - true_need) for stream_mean, true_need in zip(stream_mean_1, true_1)]
        ),
        "person2_mean_abs_stream_mean_error": _mean(
            [abs(stream_mean - true_need) for stream_mean, true_need in zip(stream_mean_2, true_2)]
        ),
        "min_correlation_required": args.min_correlation,
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")

    if true_state_mismatches or first_sample_mismatches:
        raise SystemExit(1)
    if corr_1 < args.min_correlation or corr_2 < args.min_correlation:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

