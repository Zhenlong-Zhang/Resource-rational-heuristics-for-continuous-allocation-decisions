from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

try:
    from ..mdp.meta_mdp import EnvironmentConfig, MetaPolicy
    from ..policies.heuristic import EqualSplitBaselinePolicy, ManualActiveSearchEqualOutcomePolicy
    from .randomization import build_evaluation_episodes
    from .regimes import compare_policy_behavior_profiles
    from .settings import EvaluationSettings, SMOKE_EVALUATION_SETTINGS, build_rr_policy_from_settings
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import EnvironmentConfig, MetaPolicy
    from policies.heuristic import EqualSplitBaselinePolicy, ManualActiveSearchEqualOutcomePolicy
    from experiments.randomization import build_evaluation_episodes
    from experiments.regimes import compare_policy_behavior_profiles
    from experiments.settings import EvaluationSettings, SMOKE_EVALUATION_SETTINGS, build_rr_policy_from_settings


MANUAL_ACTIVE_POLICY_NAME = ManualActiveSearchEqualOutcomePolicy.name
EQUAL_SPLIT_BASELINE_NAME = EqualSplitBaselinePolicy.name


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def build_active_search_manual_baseline_policies(samples_per_person: int = 3) -> List[MetaPolicy]:
    """Return the manual baselines used by the active-search diagnostic.

    `manual_equal_split` is the no-search equality baseline. The active-search
    baseline samples both recipients through the ordinary observation process
    before choosing the belief-based equal-outcome allocation.
    """

    return [
        ManualActiveSearchEqualOutcomePolicy(samples_per_person=samples_per_person),
        EqualSplitBaselinePolicy(),
    ]


def _diagnostic_environment_metadata(config: EnvironmentConfig) -> Dict[str, float]:
    expected_need_sum = 2.0 * config.mu_need
    expected_total_learning = config.total_time * (
        2.0 * config.learning_per_unit_of_tutoring - config.delta_learning_per_unit_tutoring
    ) / 2.0
    return {
        "expected_need_sum": expected_need_sum,
        "expected_total_learning_if_split": expected_total_learning,
        "time_to_expected_need_ratio": (
            config.total_time * config.learning_per_unit_of_tutoring / expected_need_sum
            if expected_need_sum > 0
            else math.nan
        ),
        "need_variability_ratio": config.sigma_need / config.mu_need if config.mu_need else math.nan,
    }


def run_active_search_diagnostic_policy_grid(
    regime_configs: Iterable[tuple[str, int, str, Dict[str, float], EnvironmentConfig]],
    settings: EvaluationSettings = SMOKE_EVALUATION_SETTINGS,
    allocation_tolerance: float = 0.05,
    rr_policy: MetaPolicy | None = None,
    manual_samples_per_person: int = 3,
) -> List[Dict[str, float | str]]:
    """Evaluate RR, manual active-search, and equal-split baselines together."""

    rows: List[Dict[str, float | str]] = []
    policy = rr_policy or build_rr_policy_from_settings(settings)
    manual_policies = build_active_search_manual_baseline_policies(
        samples_per_person=manual_samples_per_person,
    )

    for grid_name, grid_index, environment_name, parameter_values, config in regime_configs:
        episodes = build_evaluation_episodes(
            config=config,
            n_episodes=settings.n_episodes,
            include_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        behavior_rows = compare_policy_behavior_profiles(
            environment_name=environment_name,
            config=config,
            n_episodes=settings.n_episodes,
            allocation_tolerance=allocation_tolerance,
            rr_policy=policy,
            policies=manual_policies,
            evaluation_episodes=episodes,
            use_common_observation_streams=settings.use_common_observation_streams,
            observations_per_person=settings.observations_per_person,
        )
        metadata: Dict[str, float | str] = {
            "regime_grid": grid_name,
            "grid_index": float(grid_index),
            "manual_active_samples_per_person": float(manual_samples_per_person),
        }
        metadata.update(parameter_values)
        metadata.update(_diagnostic_environment_metadata(config))
        metadata.update(asdict(config))
        rows.extend({**metadata, **row} for row in behavior_rows)
    return rows


def identify_active_search_manual_advantage_candidates(
    behavior_rows: Sequence[Dict[str, float | str]],
    min_manual_utility_advantage: float = 0.25,
    min_manual_true_gap_reduction: float = 0.25,
    min_manual_sample_count: float = 1.0,
    min_manual_allocation_from_equal: float = 0.03,
) -> List[Dict[str, float | str]]:
    """Find environments where manual active search clearly beats equal split."""

    by_environment: Dict[str, Dict[str, Dict[str, float | str]]] = {}
    for row in behavior_rows:
        environment = str(row["environment"])
        policy = str(row["policy"])
        by_environment.setdefault(environment, {})[policy] = row

    candidates: List[Dict[str, float | str]] = []
    for environment, policy_rows in sorted(by_environment.items()):
        active = policy_rows.get(MANUAL_ACTIVE_POLICY_NAME)
        split = policy_rows.get(EQUAL_SPLIT_BASELINE_NAME)
        rr = next(
            (row for row in policy_rows.values() if row.get("policy_type") == "rr_approximation"),
            None,
        )
        if active is None or split is None:
            continue

        active_utility = float(active["mean_utility"])
        split_utility = float(split["mean_utility"])
        active_gap = float(active["mean_realized_outcome_gap"])
        split_gap = float(split["mean_realized_outcome_gap"])
        utility_advantage = active_utility - split_utility
        true_gap_reduction = split_gap - active_gap
        active_sample_count = float(active["mean_sample_count"])
        active_allocation_from_equal = float(active["mean_abs_allocation_from_equal"])

        candidate_type = "manual_active_search_advantage"
        passes = (
            utility_advantage >= min_manual_utility_advantage
            and true_gap_reduction >= min_manual_true_gap_reduction
            and active_sample_count > min_manual_sample_count
            and active_allocation_from_equal >= min_manual_allocation_from_equal
        )
        if not passes:
            continue

        row: Dict[str, float | str] = {
            "candidate_type": candidate_type,
            "environment": environment,
            "regime_grid": str(active.get("regime_grid", "")),
            "grid_index": float(active.get("grid_index", math.nan)),
            "manual_active_minus_equal_split_utility": utility_advantage,
            "manual_active_true_gap_reduction_vs_equal_split": true_gap_reduction,
            "manual_active_mean_utility": active_utility,
            "equal_split_mean_utility": split_utility,
            "manual_active_mean_sample_count": active_sample_count,
            "manual_active_mean_abs_allocation_from_equal": active_allocation_from_equal,
            "manual_active_true_equal_outcome_rate": float(active["true_equal_outcome_rate"]),
            "manual_active_closer_to_true_equal_rate": float(
                active["closer_to_true_equal_outcome_than_equal_split_rate"]
            ),
            "manual_active_mean_outcome_distance_to_true_equal": float(
                active["mean_outcome_distance_to_true_equal"]
            ),
            "equal_split_mean_outcome_distance_to_true_equal": float(
                split["mean_outcome_distance_to_true_equal"]
            ),
        }
        if rr is not None:
            row.update(
                {
                    "rr_mean_utility": float(rr["mean_utility"]),
                    "rr_minus_manual_active_utility": float(rr["mean_utility"]) - active_utility,
                    "rr_mean_sample_count": float(rr["mean_sample_count"]),
                    "rr_true_equal_outcome_rate": float(rr["true_equal_outcome_rate"]),
                    "rr_closer_to_true_equal_rate": float(
                        rr["closer_to_true_equal_outcome_than_equal_split_rate"]
                    ),
                    "rr_mean_outcome_distance_to_true_equal": float(
                        rr["mean_outcome_distance_to_true_equal"]
                    ),
                }
            )
        for key in (
            "mu_need",
            "sigma_need",
            "sigma_sample",
            "total_time",
            "sample_time_cost",
            "utility_exponent",
            "alpha",
            "learning_per_unit_of_tutoring",
            "delta_learning_per_unit_tutoring",
            "prior_sample_count_1",
            "prior_sample_count_2",
            "time_to_expected_need_ratio",
            "need_variability_ratio",
        ):
            if key in active:
                row[key] = active[key]
        candidates.append(row)
    return candidates


def summarize_active_search_diagnostic_policies(
    behavior_rows: Sequence[Dict[str, float | str]],
) -> List[Dict[str, float | str]]:
    """One-row environment summary for quick inspection and later reports."""

    by_environment: Dict[str, List[Dict[str, float | str]]] = {}
    for row in behavior_rows:
        by_environment.setdefault(str(row["environment"]), []).append(row)

    summaries: List[Dict[str, float | str]] = []
    for environment, rows in sorted(by_environment.items()):
        row_by_policy = {str(row["policy"]): row for row in rows}
        active = row_by_policy.get(MANUAL_ACTIVE_POLICY_NAME)
        split = row_by_policy.get(EQUAL_SPLIT_BASELINE_NAME)
        rr = next((row for row in rows if row.get("policy_type") == "rr_approximation"), None)
        if active is None or split is None:
            continue
        summary: Dict[str, float | str] = {
            "environment": environment,
            "regime_grid": str(active.get("regime_grid", "")),
            "grid_index": float(active.get("grid_index", math.nan)),
            "manual_active_minus_equal_split_utility": (
                float(active["mean_utility"]) - float(split["mean_utility"])
            ),
            "manual_active_true_gap_reduction_vs_equal_split": (
                float(split["mean_realized_outcome_gap"]) - float(active["mean_realized_outcome_gap"])
            ),
            "manual_active_mean_sample_count": float(active["mean_sample_count"]),
            "manual_active_true_equal_outcome_rate": float(active["true_equal_outcome_rate"]),
            "manual_active_closer_to_true_equal_rate": float(
                active["closer_to_true_equal_outcome_than_equal_split_rate"]
            ),
            "equal_split_true_equal_outcome_rate": float(split["true_equal_outcome_rate"]),
            "equal_split_closer_to_true_equal_rate": float(
                split["closer_to_true_equal_outcome_than_equal_split_rate"]
            ),
            "mean_abs_true_equal_allocation_from_equal_split": abs(
                float(active["mean_abs_true_equal_outcome_allocation_from_equal_split"])
            ),
        }
        if rr is not None:
            summary.update(
                {
                    "rr_minus_manual_active_utility": (
                        float(rr["mean_utility"]) - float(active["mean_utility"])
                    ),
                    "rr_mean_sample_count": float(rr["mean_sample_count"]),
                    "rr_true_equal_outcome_rate": float(rr["true_equal_outcome_rate"]),
                    "rr_closer_to_true_equal_rate": float(
                        rr["closer_to_true_equal_outcome_than_equal_split_rate"]
                    ),
                }
            )
        summaries.append(summary)
    return summaries
