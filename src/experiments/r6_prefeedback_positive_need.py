from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import hashlib
import json
import math
import random
import statistics

from ..mdp.finite_support import FiniteSupportAtom, FiniteSupportMetaMDP, FiniteSupportPrior
from ..mdp.meta_mdp import ContinuousAllocationMetaMDP, EnvironmentConfig, MetaPolicy, TrueState
from ..policies.finite_support_voi import FiniteSupportMyopicVOIPolicy
from ..policies.heuristic import EqualSplitBaselinePolicy, ManualActiveSearchEqualOutcomePolicy
from .r5 import full_information_oracle_metrics, full_information_utilitarian_allocation, wilson_interval
from .randomization import EvaluationEpisode
from .regimes import true_outcome_metrics_for_allocation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "configs" / "r6_prefeedback_positive_need.json"

ROUND_LABEL = "r6_pre_feedback_exploration"
POLICY_RR = "matched_prior_myopic_rr"
POLICY_MANUAL = "manual_active_search_equal_outcome"
POLICY_SPLIT = "equal_split"
POLICY_ORACLE = "full_information_oracle"
SERIOUS_POLICY_ORDER = (POLICY_RR, POLICY_MANUAL, POLICY_SPLIT, POLICY_ORACLE)

COMMON_EPISODE_FIELDS = {
    "round_classification", "environment", "environment_hash", "support_hash",
    "gap_class", "sigma_sample", "sample_time_cost", "episode_index",
    "latent_atom_index", "episode_fingerprint", "stage", "seed_namespace",
    "need_1", "need_2", "total_true_need", "realized_true_need_gap",
    "orientation", "observation_stream_hash_1", "observation_stream_hash_2",
    "observation_residual_hash_1", "observation_residual_hash_2",
    "max_observation_reconstruction_error_1", "max_observation_reconstruction_error_2",
    "positive_need",
}
POLICY_EPISODE_FIELDS = {
    "policy", "allocation_to_person1", "remaining_time", "realized_utility",
    "online_sample_count", "sample_count_1", "sample_count_2",
    "sampled_both_recipients", "immediate_termination", "near_equal_allocation",
    "abs_allocation_from_equal", "terminal_belief_hash", "posterior_weight_sum",
    "posterior_weight_min", "posterior_weight_max", "posterior_weights_finite",
}
TRUE_OUTCOME_FIELDS = {
    "realized_outcome_gap", "equal_split_realized_outcome_gap",
    "true_equal_outcome_solution_gap", "unconstrained_true_equal_outcome_allocation",
    "exact_true_equal_outcome_feasible", "true_equal_outcome_allocation",
    "true_equal_outcome_allocation_gap", "true_equal_outcome",
    "true_equal_outcome_allocation_close", "true_outcome_gap_reduction_vs_equal_split",
    "outcome_distance_to_true_equal", "equal_split_outcome_distance_to_true_equal",
    "allocation_distance_to_true_equal_minus_equal_split",
    "outcome_distance_to_true_equal_minus_equal_split",
    "closer_to_true_equal_outcome_than_equal_split",
    "closer_to_equal_split_than_true_equal_outcome", "true_outcome_classification_tie",
    "legacy_tolerance_closer_to_true_equal_outcome_than_equal_split",
    "legacy_tolerance_true_outcome_classification_tie", "outcome_success_tolerance",
    "classification_tie_tolerance", "negative_need_person1", "negative_need_person2",
    "negative_need_either", "negative_need_both",
}
ORACLE_EPISODE_FIELDS = {
    "oracle_grid_optimality_violation", "true_equal_outcome_regret",
    "equal_split_regret",
}
SERIOUS_EXTRA_FIELDS = {
    "initial_oracle_utility", "utility_regret_to_initial_oracle",
    "time_matched_oracle_allocation", "time_matched_oracle_utility",
    "time_matched_oracle_raw_regret",
}
DEVELOPMENT_EPISODE_SCHEMA = tuple(sorted(
    COMMON_EPISODE_FIELDS
    | POLICY_EPISODE_FIELDS
    | TRUE_OUTCOME_FIELDS
    | ORACLE_EPISODE_FIELDS
    | {"manual_samples_per_person"}
))
CONFIRMATION_EPISODE_SCHEMA = tuple(sorted(
    COMMON_EPISODE_FIELDS
    | POLICY_EPISODE_FIELDS
    | TRUE_OUTCOME_FIELDS
    | ORACLE_EPISODE_FIELDS
    | SERIOUS_EXTRA_FIELDS
))


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def _ci95(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * statistics.stdev(values) / math.sqrt(len(values)))


def _rate_fields(values: Sequence[float], prefix: str) -> Dict[str, float]:
    successes = sum(value >= 0.5 for value in values)
    lower, upper = wilson_interval(successes, len(values))
    one_sided, _ = wilson_interval(successes, len(values), one_sided=True)
    return {
        f"{prefix}_rate": successes / len(values) if values else math.nan,
        f"{prefix}_ci95_low": lower,
        f"{prefix}_ci95_high": upper,
        f"{prefix}_one_sided_95_low": one_sided,
    }


def load_positive_need_spec(path: Path = DEFAULT_SPEC_PATH) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("round_classification") != ROUND_LABEL:
        raise ValueError("positive-need spec has the wrong round classification")
    return value


@dataclass(frozen=True)
class PositiveNeedEnvironment:
    name: str
    gap_class: str
    sigma_sample: float
    sample_time_cost: float
    config: EnvironmentConfig
    prior: FiniteSupportPrior

    @property
    def environment_hash(self) -> str:
        return _canonical_hash(
            {
                "name": self.name,
                "gap_class": self.gap_class,
                "sigma_sample": self.sigma_sample,
                "sample_time_cost": self.sample_time_cost,
                "config": asdict(self.config),
                "support_hash": self.prior.support_hash,
            }
        )


@dataclass(frozen=True)
class FiniteSupportEvaluationEpisode:
    episode_index: int
    latent_atom_index: int
    atom: FiniteSupportAtom
    true_state: TrueState
    observation_streams: Dict[str, List[float]]
    residual_streams: Dict[str, List[float]]
    stage: str
    seed_namespace: int

    @property
    def residual_hash_1(self) -> str:
        return _canonical_hash(self.residual_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_1])

    @property
    def residual_hash_2(self) -> str:
        return _canonical_hash(self.residual_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_2])

    @property
    def observation_hash_1(self) -> str:
        return _canonical_hash(self.observation_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_1])

    @property
    def observation_hash_2(self) -> str:
        return _canonical_hash(self.observation_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_2])

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(
            {
                "episode_index": self.episode_index,
                "latent_atom_index": self.latent_atom_index,
                "need_1": self.true_state.need_1,
                "need_2": self.true_state.need_2,
                "residual_hash_1": self.residual_hash_1,
                "residual_hash_2": self.residual_hash_2,
                "stage": self.stage,
                "seed_namespace": self.seed_namespace,
            }
        )


def _base_environment_config(spec: Mapping[str, object]) -> EnvironmentConfig:
    grid = dict(spec["environment_grid"])  # type: ignore[arg-type]
    numerical = dict(spec["numerical_settings"])  # type: ignore[arg-type]
    totals = list(dict(spec["generator"])["total_need_support"])  # type: ignore[index]
    return EnvironmentConfig(
        mu_need=float(statistics.mean(float(value) for value in totals)) / 2.0,
        sigma_need=1.0,
        sigma_sample=1.0,
        total_time=float(grid["total_time"]),
        lambda_shortfall=float(grid["lambda_shortfall"]),
        utility_exponent=float(grid["utility_exponent"]),
        learning_per_unit_of_tutoring=float(grid["learning_per_unit_of_tutoring"]),
        delta_learning_per_unit_tutoring=float(grid["delta_learning_per_unit_tutoring"]),
        terminate_cost=float(grid["terminate_cost"]),
        sample_time_cost=0.02,
        allocation_grid_size=int(numerical["rr_terminal_grid_size"]),
        prior_sample_count_1=int(grid["prior_sample_count_1"]),
        prior_sample_count_2=int(grid["prior_sample_count_2"]),
        max_meta_samples=int(grid["max_meta_samples"]),
        random_seed=0,
    )


def build_prior(spec: Mapping[str, object], gap_class: str) -> FiniteSupportPrior:
    generator = dict(spec["generator"])  # type: ignore[arg-type]
    gap_supports = dict(generator["gap_supports"])
    if gap_class not in gap_supports:
        raise ValueError(f"unknown gap class: {gap_class}")
    return FiniteSupportPrior.from_total_gap_support(
        total_needs=[float(value) for value in generator["total_need_support"]],
        absolute_gaps=[float(value) for value in gap_supports[gap_class]],
        orientations=[int(value) for value in generator["orientation_support"]],
        total_weights=[float(value) for value in generator["total_need_weights"]],
        gap_weights=[float(value) for value in generator["gap_weights"]],
        orientation_weights=[float(value) for value in generator["orientation_weights"]],
    )


def build_development_environments(
    spec: Optional[Mapping[str, object]] = None,
) -> List[PositiveNeedEnvironment]:
    spec = dict(spec or load_positive_need_spec())
    base = _base_environment_config(spec)
    generator = dict(spec["generator"])  # type: ignore[arg-type]
    grid = dict(spec["environment_grid"])  # type: ignore[arg-type]
    environments: List[PositiveNeedEnvironment] = []
    for gap_class in generator["gap_supports"]:
        prior = build_prior(spec, str(gap_class))
        for sigma_sample in grid["sigma_sample"]:
            for sample_time_cost in grid["sample_time_cost"]:
                name = (
                    f"positive_need_gap={gap_class}_sigma={float(sigma_sample):g}"
                    f"_sample_cost={float(sample_time_cost):g}"
                )
                config = replace(
                    base,
                    sigma_sample=float(sigma_sample),
                    sample_time_cost=float(sample_time_cost),
                )
                environments.append(
                    PositiveNeedEnvironment(
                        name=name,
                        gap_class=str(gap_class),
                        sigma_sample=float(sigma_sample),
                        sample_time_cost=float(sample_time_cost),
                        config=config,
                        prior=prior,
                    )
                )
    return environments


def _weighted_atom_index(prior: FiniteSupportPrior, rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, weight in enumerate(prior.weights):
        cumulative += weight
        if draw < cumulative:
            return index
    return len(prior.states) - 1


def build_finite_support_episode(
    environment: PositiveNeedEnvironment,
    episode_index: int,
    stage: str,
    seed_namespace: int,
    observations_per_person: int,
    balanced_atoms: bool,
) -> FiniteSupportEvaluationEpisode:
    if episode_index < 0 or observations_per_person <= 0:
        raise ValueError("episode_index must be nonnegative and streams must be nonempty")
    if balanced_atoms:
        atom_index = episode_index % len(environment.prior.states)
    else:
        atom_rng = random.Random(seed_namespace + 101 + episode_index * 104729)
        atom_index = _weighted_atom_index(environment.prior, atom_rng)
    atom = environment.prior.states[atom_index]
    true_state = TrueState(atom.need_1, atom.need_2)
    residual_rng = random.Random(seed_namespace + 100_003 + episode_index * 104729)
    residuals_1 = [float(residual_rng.gauss(0.0, 1.0)) for _ in range(observations_per_person)]
    residuals_2 = [float(residual_rng.gauss(0.0, 1.0)) for _ in range(observations_per_person)]
    observations_1 = [true_state.need_1 + environment.sigma_sample * value for value in residuals_1]
    observations_2 = [true_state.need_2 + environment.sigma_sample * value for value in residuals_2]
    return FiniteSupportEvaluationEpisode(
        episode_index=episode_index,
        latent_atom_index=atom_index,
        atom=atom,
        true_state=true_state,
        observation_streams={
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_1: observations_1,
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_2: observations_2,
        },
        residual_streams={
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_1: residuals_1,
            ContinuousAllocationMetaMDP.SAMPLE_PERSON_2: residuals_2,
        },
        stage=stage,
        seed_namespace=seed_namespace,
    )


def build_finite_support_episodes(
    environment: PositiveNeedEnvironment,
    n_episodes: int,
    stage: str,
    seed_namespace: int,
    observations_per_person: int,
    balanced_atoms: bool,
    episode_start: int = 0,
) -> List[FiniteSupportEvaluationEpisode]:
    return [
        build_finite_support_episode(
            environment,
            episode_index=index,
            stage=stage,
            seed_namespace=seed_namespace,
            observations_per_person=observations_per_person,
            balanced_atoms=balanced_atoms,
        )
        for index in range(episode_start, episode_start + n_episodes)
    ]


def _sample_counts(samples: Sequence[Mapping[str, float]]) -> Tuple[int, int]:
    count_1 = sum(float(item["action"]) == 1.0 for item in samples)
    count_2 = sum(float(item["action"]) == 2.0 for item in samples)
    return count_1, count_2


def _episode_common_fields(
    environment: PositiveNeedEnvironment,
    episode: FiniteSupportEvaluationEpisode,
) -> Dict[str, object]:
    reconstructed_1 = [
        episode.true_state.need_1 + environment.sigma_sample * residual
        for residual in episode.residual_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_1]
    ]
    reconstructed_2 = [
        episode.true_state.need_2 + environment.sigma_sample * residual
        for residual in episode.residual_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_2]
    ]
    error_1 = max(
        abs(actual - expected)
        for actual, expected in zip(
            episode.observation_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_1],
            reconstructed_1,
        )
    )
    error_2 = max(
        abs(actual - expected)
        for actual, expected in zip(
            episode.observation_streams[ContinuousAllocationMetaMDP.SAMPLE_PERSON_2],
            reconstructed_2,
        )
    )
    return {
        "round_classification": ROUND_LABEL,
        "environment": environment.name,
        "environment_hash": environment.environment_hash,
        "support_hash": environment.prior.support_hash,
        "gap_class": environment.gap_class,
        "sigma_sample": environment.sigma_sample,
        "sample_time_cost": environment.sample_time_cost,
        "episode_index": episode.episode_index,
        "latent_atom_index": episode.latent_atom_index,
        "episode_fingerprint": episode.fingerprint,
        "stage": episode.stage,
        "seed_namespace": episode.seed_namespace,
        "need_1": episode.true_state.need_1,
        "need_2": episode.true_state.need_2,
        "total_true_need": episode.atom.total_need,
        "realized_true_need_gap": episode.atom.absolute_gap,
        "orientation": episode.atom.orientation,
        "observation_stream_hash_1": episode.observation_hash_1,
        "observation_stream_hash_2": episode.observation_hash_2,
        "observation_residual_hash_1": episode.residual_hash_1,
        "observation_residual_hash_2": episode.residual_hash_2,
        "max_observation_reconstruction_error_1": error_1,
        "max_observation_reconstruction_error_2": error_2,
        "positive_need": 1.0 if episode.true_state.need_1 > 0 and episode.true_state.need_2 > 0 else 0.0,
    }


def _run_policy_row(
    environment: PositiveNeedEnvironment,
    episode: FiniteSupportEvaluationEpisode,
    policy: MetaPolicy,
    policy_name: str,
    allocation_tolerance: float,
) -> Dict[str, object]:
    mdp = FiniteSupportMetaMDP(
        environment.config,
        environment.prior,
        observation_streams=episode.observation_streams,
    )
    max_steps = max(100, (environment.config.max_meta_samples or 0) + 2)
    result = mdp.run_episode(policy, true_state=episode.true_state, max_steps=max_steps)
    metrics = true_outcome_metrics_for_allocation(
        mdp,
        episode.true_state,
        result.final_belief,
        result.final_allocation_to_person1,
        allocation_tolerance=allocation_tolerance,
    )
    count_1, count_2 = _sample_counts(result.samples)
    posterior_weights = tuple(float(weight) for weight in result.final_belief.weights)
    row = _episode_common_fields(environment, episode)
    row.update(
        {
            "policy": policy_name,
            "allocation_to_person1": result.final_allocation_to_person1,
            "remaining_time": result.remaining_time,
            "realized_utility": result.realized_utility,
            "online_sample_count": len(result.samples),
            "sample_count_1": count_1,
            "sample_count_2": count_2,
            "sampled_both_recipients": 1.0 if count_1 > 0 and count_2 > 0 else 0.0,
            "immediate_termination": 1.0 if not result.samples else 0.0,
            "near_equal_allocation": (
                1.0 if abs(result.final_allocation_to_person1 - 0.5) <= allocation_tolerance else 0.0
            ),
            "abs_allocation_from_equal": abs(result.final_allocation_to_person1 - 0.5),
            "terminal_belief_hash": _canonical_hash(
                {
                    "weights": list(result.final_belief.weights),
                    "deliberation_time": result.final_belief.deliberation_time,
                }
            ),
            "posterior_weight_sum": math.fsum(posterior_weights),
            "posterior_weight_min": min(posterior_weights),
            "posterior_weight_max": max(posterior_weights),
            "posterior_weights_finite": (
                1.0 if all(math.isfinite(weight) for weight in posterior_weights) else 0.0
            ),
        }
    )
    row.update(metrics)
    return row


def _oracle_row(
    environment: PositiveNeedEnvironment,
    episode: FiniteSupportEvaluationEpisode,
    allocation_tolerance: float,
    oracle_grid_size: int,
) -> Dict[str, object]:
    scaffold = EvaluationEpisode(
        episode_index=episode.episode_index,
        true_state=episode.true_state,
        observation_streams=episode.observation_streams,
    )
    oracle = full_information_oracle_metrics(
        environment.name,
        environment.config,
        scaffold,
        allocation_tolerance=allocation_tolerance,
        grid_size=oracle_grid_size,
    )
    row = _episode_common_fields(environment, episode)
    row.update(
        {
            "policy": POLICY_ORACLE,
            "allocation_to_person1": oracle["oracle_allocation"],
            "remaining_time": oracle["remaining_time"],
            "realized_utility": oracle["oracle_utility"],
            "online_sample_count": 0,
            "sample_count_1": 0,
            "sample_count_2": 0,
            "sampled_both_recipients": 0.0,
            "immediate_termination": 1.0,
            "near_equal_allocation": (
                1.0 if abs(float(oracle["oracle_allocation"]) - 0.5) <= allocation_tolerance else 0.0
            ),
            "abs_allocation_from_equal": abs(float(oracle["oracle_allocation"]) - 0.5),
            "terminal_belief_hash": "full_information",
            "posterior_weight_sum": 1.0,
            "posterior_weight_min": 1.0,
            "posterior_weight_max": 1.0,
            "posterior_weights_finite": 1.0,
            "true_equal_outcome": oracle["oracle_true_equal_outcome"],
            "closer_to_true_equal_outcome_than_equal_split": oracle[
                "oracle_closer_to_true_equal_than_equal_split"
            ],
            "exact_true_equal_outcome_feasible": oracle["exact_true_equal_outcome_feasible"],
            "true_equal_outcome_allocation": oracle["true_equal_outcome_allocation"],
            "true_equal_outcome_allocation_gap": oracle["oracle_allocation_gap_to_true_equal"],
            "realized_outcome_gap": oracle["oracle_realized_outcome_gap"],
            "outcome_distance_to_true_equal": max(
                0.0,
                float(oracle["oracle_realized_outcome_gap"])
                - float(oracle["true_equal_outcome_solution_gap"]),
            ),
            "equal_split_outcome_distance_to_true_equal": max(
                0.0,
                float(oracle["equal_split_realized_outcome_gap"])
                - float(oracle["true_equal_outcome_solution_gap"]),
            ),
            "oracle_grid_optimality_violation": oracle["oracle_grid_optimality_violation"],
            "true_equal_outcome_regret": oracle["true_equal_outcome_regret"],
            "equal_split_regret": oracle["equal_split_regret"],
        }
    )
    return row


def evaluate_fixed_budgets(
    environment: PositiveNeedEnvironment,
    episodes: Sequence[FiniteSupportEvaluationEpisode],
    samples_per_person: Sequence[int],
    allocation_tolerance: float,
    oracle_grid_size: int,
) -> List[Dict[str, object]]:
    budgets = sorted(set(int(value) for value in samples_per_person))
    if not budgets or any(value < 0 for value in budgets):
        raise ValueError("manual sample budgets must be nonnegative")
    rows: List[Dict[str, object]] = []
    for episode in episodes:
        for budget in budgets:
            row = _run_policy_row(
                environment,
                episode,
                ManualActiveSearchEqualOutcomePolicy(samples_per_person=budget),
                f"manual_equal_outcome_{budget}_per_person",
                allocation_tolerance,
            )
            row["manual_samples_per_person"] = budget
            rows.append(row)
        rows.append(_oracle_row(environment, episode, allocation_tolerance, oracle_grid_size))
    return rows


def initial_information_values(
    environment: PositiveNeedEnvironment,
    quadrature_order: int,
) -> Dict[str, object]:
    mdp = FiniteSupportMetaMDP(environment.config, environment.prior)
    belief = mdp.initial_belief()
    policy = FiniteSupportMyopicVOIPolicy(quadrature_order=quadrature_order)
    action_values = policy.action_values(mdp, belief)
    stop = float(action_values[mdp.TERMINATE])

    oracle_values = []
    for atom, weight in zip(environment.prior.states, environment.prior.weights):
        allocation, value = full_information_utilitarian_allocation(
            ContinuousAllocationMetaMDP(environment.config),
            TrueState(atom.need_1, atom.need_2),
            environment.config.total_time - environment.config.terminate_cost,
            grid_size=4001,
        )
        del allocation
        oracle_values.append(weight * value)
    full_information_value = float(math.fsum(oracle_values))
    return {
        "environment": environment.name,
        "environment_hash": environment.environment_hash,
        "support_hash": environment.prior.support_hash,
        "gap_class": environment.gap_class,
        "sigma_sample": environment.sigma_sample,
        "sample_time_cost": environment.sample_time_cost,
        "initial_termination_value": stop,
        "initial_sample_1_value": float(action_values.get(mdp.SAMPLE_PERSON_1, math.nan)),
        "initial_sample_2_value": float(action_values.get(mdp.SAMPLE_PERSON_2, math.nan)),
        "initial_voc_1": float(action_values.get(mdp.SAMPLE_PERSON_1, -math.inf)) - stop,
        "initial_voc_2": float(action_values.get(mdp.SAMPLE_PERSON_2, -math.inf)) - stop,
        "expected_full_information_value": full_information_value,
        "evpi": full_information_value - stop,
        "selected_initial_action": policy.choose_action(mdp, belief),
        "quadrature_order": quadrature_order,
    }


def _paired_contrast(
    rows: Sequence[Mapping[str, object]],
    positive_policy: str,
    negative_policy: str,
) -> Dict[str, float]:
    by_policy = {
        policy: {
            int(row["episode_index"]): float(row["realized_utility"])
            for row in rows
            if row["policy"] == policy
        }
        for policy in (positive_policy, negative_policy)
    }
    shared = sorted(set(by_policy[positive_policy]).intersection(by_policy[negative_policy]))
    differences = [
        by_policy[positive_policy][index] - by_policy[negative_policy][index]
        for index in shared
    ]
    mean = _mean(differences)
    ci = _ci95(differences)
    return {
        "n_pairs": len(differences),
        "mean": mean,
        "ci95": ci,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def summarize_development_environment(
    environment: PositiveNeedEnvironment,
    rows: Sequence[Mapping[str, object]],
    information: Mapping[str, object],
) -> Dict[str, object]:
    policy_4 = "manual_equal_outcome_2_per_person"
    policy_6 = "manual_equal_outcome_3_per_person"
    policy_split = "manual_equal_outcome_0_per_person"
    manual_6 = [row for row in rows if row["policy"] == policy_6]
    oracle = [row for row in rows if row["policy"] == POLICY_ORACLE]
    contrast_6_split = _paired_contrast(rows, policy_6, policy_split)
    contrast_6_4 = _paired_contrast(rows, policy_6, policy_4)
    summary: Dict[str, object] = {
        "environment": environment.name,
        "environment_hash": environment.environment_hash,
        "support_hash": environment.prior.support_hash,
        "gap_class": environment.gap_class,
        "sigma_sample": environment.sigma_sample,
        "sample_time_cost": environment.sample_time_cost,
        "n_episodes": len(manual_6),
        "positive_need_rate": _mean([float(row["positive_need"]) for row in manual_6]),
        "initial_true_equal_feasibility_rate": _mean(
            [float(row["exact_true_equal_outcome_feasible"]) for row in oracle]
        ),
        "manual_time_true_equal_feasibility_rate": _mean(
            [float(row["exact_true_equal_outcome_feasible"]) for row in manual_6]
        ),
        "mean_abs_true_equal_allocation_from_equal": _mean(
            [abs(float(row["true_equal_outcome_allocation"]) - 0.5) for row in oracle]
        ),
        "mean_oracle_true_equal_regret": _mean(
            [float(row["true_equal_outcome_regret"]) for row in oracle]
        ),
        "max_oracle_optimality_violation": max(
            (float(row["oracle_grid_optimality_violation"]) for row in oracle),
            default=math.nan,
        ),
        "manual_6_minus_split_mean": contrast_6_split["mean"],
        "manual_6_minus_split_ci95_low": contrast_6_split["ci95_low"],
        "manual_6_minus_split_ci95_high": contrast_6_split["ci95_high"],
        "manual_6_minus_manual_4_mean": contrast_6_4["mean"],
        "manual_6_minus_manual_4_ci95_low": contrast_6_4["ci95_low"],
        "manual_6_minus_manual_4_ci95_high": contrast_6_4["ci95_high"],
        "manual_6_mean_abs_allocation_from_equal": _mean(
            [float(row["abs_allocation_from_equal"]) for row in manual_6]
        ),
    }
    summary.update(_rate_fields(
        [float(row["true_equal_outcome"]) for row in oracle],
        "oracle_true_equal_outcome",
    ))
    summary.update(_rate_fields(
        [float(row["closer_to_true_equal_outcome_than_equal_split"]) for row in oracle],
        "oracle_closer_to_true_equal_than_equal_split",
    ))
    summary.update(_rate_fields(
        [float(row["true_equal_outcome"]) for row in manual_6],
        "manual_6_true_equal_outcome",
    ))
    summary.update(_rate_fields(
        [float(row["closer_to_true_equal_outcome_than_equal_split"]) for row in manual_6],
        "manual_6_closer_to_true_equal_than_equal_split",
    ))
    summary.update(
        {
            "initial_voc_1": float(information["initial_voc_1"]),
            "initial_voc_2": float(information["initial_voc_2"]),
            "evpi": float(information["evpi"]),
        }
    )
    target_pass = (
        float(summary["positive_need_rate"]) == 1.0
        and float(summary["initial_true_equal_feasibility_rate"]) == 1.0
        and float(summary["manual_time_true_equal_feasibility_rate"]) == 1.0
        and float(summary["mean_abs_true_equal_allocation_from_equal"]) >= 0.10
        and float(summary["oracle_true_equal_outcome_rate"]) >= 0.80
        and float(summary["oracle_closer_to_true_equal_than_equal_split_rate"]) >= 0.80
        and float(summary["mean_oracle_true_equal_regret"]) <= 1e-4
        and float(summary["max_oracle_optimality_violation"]) <= 1e-9
        and float(summary["manual_6_minus_split_ci95_low"]) > 0.0
        and float(summary["manual_6_true_equal_outcome_rate"]) >= 0.80
        and float(summary["manual_6_closer_to_true_equal_than_equal_split_rate"]) >= 0.80
        and float(summary["manual_6_minus_manual_4_ci95_low"]) > 0.0
        and float(summary["evpi"]) > 0.0
        and max(float(summary["initial_voc_1"]), float(summary["initial_voc_2"])) > 1e-6
    )
    summary["target_gate_pass"] = 1.0 if target_pass else 0.0
    summary["control_manual_gate_pass"] = (
        1.0 if float(summary["manual_6_minus_split_ci95_high"]) <= 0.0 else 0.0
    )
    return summary


def select_target_control_pair(
    development_summaries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    grouped: Dict[Tuple[str, float], List[Mapping[str, object]]] = {}
    for row in development_summaries:
        grouped.setdefault((str(row["gap_class"]), float(row["sigma_sample"])), []).append(row)
    candidates: List[Dict[str, object]] = []
    gap_rank = {"high": 0, "medium": 1, "low": 2}
    for (gap_class, sigma_sample), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row["sample_time_cost"]))
        targets = [row for row in ordered if float(row["target_gate_pass"]) >= 0.5]
        if not targets:
            continue
        target = targets[0]
        controls = [
            row
            for row in ordered
            if float(row["sample_time_cost"]) > float(target["sample_time_cost"])
            and float(row["control_manual_gate_pass"]) >= 0.5
        ]
        if not controls:
            continue
        control = controls[0]
        candidates.append(
            {
                "gap_class": gap_class,
                "sigma_sample": sigma_sample,
                "target_environment": str(target["environment"]),
                "target_environment_hash": str(target["environment_hash"]),
                "target_sample_time_cost": float(target["sample_time_cost"]),
                "control_environment": str(control["environment"]),
                "control_environment_hash": str(control["environment_hash"]),
                "control_sample_time_cost": float(control["sample_time_cost"]),
                "target_manual_minus_split_ci95_low": float(
                    target["manual_6_minus_split_ci95_low"]
                ),
                "target_manual_6_minus_manual_4_ci95_low": float(
                    target["manual_6_minus_manual_4_ci95_low"]
                ),
                "target_gate_pass": float(target["target_gate_pass"]),
                "fixed_budget_evidence_favors_6_over_4": (
                    1.0
                    if float(target["manual_6_minus_manual_4_ci95_low"]) > 0.0
                    else 0.0
                ),
                "rank_key": (
                    -float(target["manual_6_minus_split_ci95_low"]),
                    gap_rank[gap_class],
                    sigma_sample,
                    float(target["sample_time_cost"]),
                    float(control["sample_time_cost"]),
                ),
            }
        )
    if not candidates:
        return {
            "selection_status": "no_valid_target_control_pair",
            "candidate_count": 0,
        }
    selected = min(candidates, key=lambda row: row["rank_key"])
    result = dict(selected)
    result.pop("rank_key", None)
    result["selection_status"] = "selected_without_rr_behavior"
    result["candidate_count"] = len(candidates)
    return result


def evaluate_serious_environment(
    environment: PositiveNeedEnvironment,
    episodes: Sequence[FiniteSupportEvaluationEpisode],
    quadrature_order: int,
    manual_samples_per_person: int,
    allocation_tolerance: float,
    oracle_grid_size: int,
) -> List[Dict[str, object]]:
    rr = FiniteSupportMyopicVOIPolicy(quadrature_order=quadrature_order)
    manual = ManualActiveSearchEqualOutcomePolicy(samples_per_person=manual_samples_per_person)
    split = EqualSplitBaselinePolicy()
    rows: List[Dict[str, object]] = []
    for episode in episodes:
        episode_rows = [
            _run_policy_row(environment, episode, rr, POLICY_RR, allocation_tolerance),
            _run_policy_row(environment, episode, manual, POLICY_MANUAL, allocation_tolerance),
            _run_policy_row(environment, episode, split, POLICY_SPLIT, allocation_tolerance),
            _oracle_row(environment, episode, allocation_tolerance, oracle_grid_size),
        ]
        oracle_utility = float(episode_rows[-1]["realized_utility"])
        for row in episode_rows:
            row["initial_oracle_utility"] = oracle_utility
            row["utility_regret_to_initial_oracle"] = max(
                0.0, oracle_utility - float(row["realized_utility"])
            )
        for row in episode_rows[:2]:
            base_mdp = ContinuousAllocationMetaMDP(environment.config)
            allocation, utility = full_information_utilitarian_allocation(
                base_mdp,
                episode.true_state,
                float(row["remaining_time"]),
                grid_size=oracle_grid_size,
            )
            row["time_matched_oracle_allocation"] = allocation
            row["time_matched_oracle_utility"] = utility
            row["time_matched_oracle_raw_regret"] = utility - float(row["realized_utility"])
        rows.extend(episode_rows)
    validate_serious_common_randomness(rows)
    return rows


def validate_serious_common_randomness(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["environment"]), int(row["episode_index"])), []).append(row)
    for key, episode_rows in grouped.items():
        if len(episode_rows) != len(SERIOUS_POLICY_ORDER):
            raise RuntimeError(f"policy row count mismatch for {key}")
        if {str(row["policy"]) for row in episode_rows} != set(SERIOUS_POLICY_ORDER):
            raise RuntimeError(f"policy set mismatch for {key}")
        for field in (
            "episode_fingerprint",
            "support_hash",
            "need_1",
            "need_2",
            "observation_residual_hash_1",
            "observation_residual_hash_2",
        ):
            if len({row[field] for row in episode_rows}) != 1:
                raise RuntimeError(f"common-randomness mismatch for {key}: {field}")
        if max(float(row["max_observation_reconstruction_error_1"]) for row in episode_rows) > 1e-12:
            raise RuntimeError(f"person-1 observation reconstruction failed for {key}")
        if max(float(row["max_observation_reconstruction_error_2"]) for row in episode_rows) > 1e-12:
            raise RuntimeError(f"person-2 observation reconstruction failed for {key}")


def _policy_summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "environment": rows[0]["environment"] if rows else "",
        "policy": rows[0]["policy"] if rows else "",
        "n_episodes": len(rows),
        "mean_utility": _mean([float(row["realized_utility"]) for row in rows]),
        "mean_sample_count": _mean([float(row["online_sample_count"]) for row in rows]),
        "mean_abs_allocation_from_equal": _mean(
            [float(row["abs_allocation_from_equal"]) for row in rows]
        ),
        "mean_realized_outcome_gap": _mean(
            [float(row["realized_outcome_gap"]) for row in rows]
        ),
        "mean_allocation_distance_to_true_equal": _mean(
            [float(row["true_equal_outcome_allocation_gap"]) for row in rows]
        ),
        "mean_utility_regret_to_initial_oracle": _mean(
            [float(row["utility_regret_to_initial_oracle"]) for row in rows]
        ),
    }
    for field, prefix in (
        ("true_equal_outcome", "true_equal_outcome"),
        (
            "closer_to_true_equal_outcome_than_equal_split",
            "closer_to_true_equal_than_equal_split",
        ),
        ("sampled_both_recipients", "sampled_both_recipients"),
        ("immediate_termination", "immediate_termination"),
        ("near_equal_allocation", "near_equal_allocation"),
    ):
        summary.update(_rate_fields([float(row[field]) for row in rows], prefix))
    time_matched_rows = [
        row
        for row in rows
        if row.get("time_matched_oracle_raw_regret") not in (None, "")
    ]
    if time_matched_rows:
        regrets = [float(row["time_matched_oracle_raw_regret"]) for row in time_matched_rows]
        regret_mean = _mean(regrets)
        regret_ci = _ci95(regrets)
        summary.update(
            {
                "mean_time_matched_oracle_utility": _mean(
                    [float(row["time_matched_oracle_utility"]) for row in time_matched_rows]
                ),
                "mean_time_matched_oracle_raw_regret": regret_mean,
                "time_matched_oracle_raw_regret_ci95_low": regret_mean - regret_ci,
                "time_matched_oracle_raw_regret_ci95_high": regret_mean + regret_ci,
            }
        )
    return summary


def solver_diagnosis_trigger(
    *,
    recovery_ci95_high: float,
    time_matched_regret_ci95_low: float,
    mean_time_matched_regret: float,
    mean_time_matched_oracle_utility: float,
    mean_rr_sample_count: float,
    fixed_budget_evidence_favors_6_over_4: bool,
    development_gate_pass: bool,
) -> bool:
    material_regret = 0.01 * max(1.0, abs(mean_time_matched_oracle_utility))
    return (
        development_gate_pass
        and recovery_ci95_high < 0.0
        and time_matched_regret_ci95_low > 0.0
        and mean_time_matched_regret > material_regret
        and mean_rr_sample_count <= 1.0
        and fixed_budget_evidence_favors_6_over_4
    )


def summarize_serious(
    episode_rows: Sequence[Mapping[str, object]],
    target_environment: str,
    control_environment: str,
    spec: Optional[Mapping[str, object]] = None,
    selection: Optional[Mapping[str, object]] = None,
    scientific_confirmation: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    spec = dict(spec or load_positive_need_spec())
    summaries = []
    for environment in sorted({str(row["environment"]) for row in episode_rows}):
        for policy in SERIOUS_POLICY_ORDER:
            rows = [
                row
                for row in episode_rows
                if row["environment"] == environment and row["policy"] == policy
            ]
            summaries.append(_policy_summary(rows))

    comparisons: List[Dict[str, object]] = []
    contrast_pairs = (
        ("rr_minus_manual", POLICY_RR, POLICY_MANUAL),
        ("rr_minus_equal_split", POLICY_RR, POLICY_SPLIT),
        ("manual_minus_equal_split", POLICY_MANUAL, POLICY_SPLIT),
        ("oracle_minus_rr", POLICY_ORACLE, POLICY_RR),
    )
    for environment in sorted({str(row["environment"]) for row in episode_rows}):
        environment_rows = [row for row in episode_rows if row["environment"] == environment]
        for label, positive, negative in contrast_pairs:
            values = _paired_values(environment_rows, positive, negative, "realized_utility")
            comparisons.append(_contrast_row(environment, label, values))
        recovery = _three_policy_recovery(environment_rows)
        comparisons.append(_contrast_row(
            environment,
            "rr_90_percent_manual_improvement_recovery",
            recovery,
        ))

    summary_lookup = {
        (str(row["environment"]), str(row["policy"])): row for row in summaries
    }
    target_rr = summary_lookup[(target_environment, POLICY_RR)]
    control_rr = summary_lookup[(control_environment, POLICY_RR)]
    target_contrasts = {
        str(row["contrast"]): row for row in comparisons if row["environment"] == target_environment
    }
    thresholds = dict(spec["strict_target_thresholds"])  # type: ignore[arg-type]
    control_thresholds = dict(spec["strict_control_thresholds"])  # type: ignore[arg-type]
    selection = dict(selection or {})
    development_gate_pass = (
        selection.get("selection_status", "selected_without_rr_behavior")
        == "selected_without_rr_behavior"
        and float(selection.get("target_gate_pass", 1.0)) >= 0.5
    )

    target_pass = (
        development_gate_pass
        and
        float(target_rr["true_equal_outcome_rate"]) >= float(thresholds["true_equal_outcome_rate"])
        and float(target_rr["closer_to_true_equal_than_equal_split_rate"])
        >= float(thresholds["closer_to_true_equal_than_equal_split_rate"])
        and float(target_rr["true_equal_outcome_one_sided_95_low"])
        > float(thresholds["true_equal_one_sided_95_low"])
        and float(target_rr["closer_to_true_equal_than_equal_split_one_sided_95_low"])
        > float(thresholds["closer_one_sided_95_low"])
        and float(target_rr["mean_sample_count"])
        > float(thresholds["mean_sample_count_min_exclusive"])
        and float(target_rr["sampled_both_recipients_rate"])
        >= float(thresholds["sampled_both_rate"])
        and float(target_rr["mean_abs_allocation_from_equal"])
        >= float(thresholds["mean_abs_allocation_from_equal_min"])
        and float(
            target_contrasts["rr_90_percent_manual_improvement_recovery"]["ci95_low"]
        )
        >= 0.0
    )

    margin = float(control_thresholds["utility_noninferiority_margin_scale"]) * max(
        1.0,
        abs(float(summary_lookup[(control_environment, POLICY_SPLIT)]["mean_utility"])),
    )
    control_utility = next(
        row
        for row in comparisons
        if row["environment"] == control_environment and row["contrast"] == "rr_minus_equal_split"
    )
    control_pass = (
        float(control_rr["immediate_termination_one_sided_95_low"])
        > float(control_thresholds["immediate_termination_one_sided_95_low"])
        and float(control_rr["near_equal_allocation_one_sided_95_low"])
        > float(control_thresholds["near_equal_allocation_one_sided_95_low"])
        and float(control_rr["mean_sample_count"])
        <= float(control_thresholds["mean_sample_count_max"])
        and float(control_utility["ci95_low"]) > -margin
    )

    paired_environment_rows = _paired_target_control_rows(
        episode_rows,
        target_environment,
        control_environment,
    )
    target_control = {
        "sample_count": _paired_metric_contrast(
            paired_environment_rows, "online_sample_count"
        ),
        "closer": _paired_metric_contrast(
            paired_environment_rows, "closer_to_true_equal_outcome_than_equal_split"
        ),
        "allocation_distance": _paired_metric_contrast(
            paired_environment_rows, "abs_allocation_from_equal"
        ),
    }
    contrast_pass = (
        target_control["sample_count"]["ci95_low"] > 0.0
        and target_control["closer"]["ci95_low"] > 0.0
        and target_control["allocation_distance"]["ci95_low"] > 0.0
        and target_control["allocation_distance"]["mean"] > 0.02
    )

    mean_time_matched_oracle = float(target_rr["mean_time_matched_oracle_utility"])
    mean_time_matched_regret = float(target_rr["mean_time_matched_oracle_raw_regret"])
    material_regret = 0.01 * max(1.0, abs(mean_time_matched_oracle))
    fixed_budget_evidence = (
        float(selection.get("fixed_budget_evidence_favors_6_over_4", 0.0)) >= 0.5
    )
    solver_trigger = solver_diagnosis_trigger(
        recovery_ci95_high=float(
            target_contrasts["rr_90_percent_manual_improvement_recovery"]["ci95_high"]
        ),
        time_matched_regret_ci95_low=float(
            target_rr["time_matched_oracle_raw_regret_ci95_low"]
        ),
        mean_time_matched_regret=mean_time_matched_regret,
        mean_time_matched_oracle_utility=mean_time_matched_oracle,
        mean_rr_sample_count=float(target_rr["mean_sample_count"]),
        fixed_budget_evidence_favors_6_over_4=fixed_budget_evidence,
        development_gate_pass=development_gate_pass,
    )
    if target_pass and control_pass and contrast_pass and not solver_trigger:
        readiness = "ready_for_experiment_design_planning"
    elif solver_trigger:
        readiness = "continue_solver_diagnosis"
    else:
        readiness = "continue_environment_development"
    candidate_readiness = readiness
    evidence_status = "held_out_model_evidence" if scientific_confirmation else "smoke_only_not_scientific_evidence"
    if not scientific_confirmation:
        readiness = "invalid_evidence"
    classification = {
        "target_environment": target_environment,
        "control_environment": control_environment,
        "strict_target_pass": 1.0 if target_pass else 0.0,
        "strict_control_pass": 1.0 if control_pass else 0.0,
        "target_control_contrast_pass": 1.0 if contrast_pass else 0.0,
        "solver_trigger": 1.0 if solver_trigger else 0.0,
        "readiness_classification": readiness,
        "candidate_readiness_classification": candidate_readiness,
        "evidence_status": evidence_status,
        "development_gate_pass": 1.0 if development_gate_pass else 0.0,
        "fixed_budget_evidence_favors_6_over_4": 1.0 if fixed_budget_evidence else 0.0,
        "mean_time_matched_oracle_utility": mean_time_matched_oracle,
        "mean_time_matched_oracle_raw_regret": mean_time_matched_regret,
        "time_matched_oracle_raw_regret_ci95_low": target_rr[
            "time_matched_oracle_raw_regret_ci95_low"
        ],
        "material_time_matched_oracle_regret_threshold": material_regret,
        "target_control_sample_count_difference": target_control["sample_count"]["mean"],
        "target_control_sample_count_ci95_low": target_control["sample_count"]["ci95_low"],
        "target_control_closer_difference": target_control["closer"]["mean"],
        "target_control_closer_ci95_low": target_control["closer"]["ci95_low"],
        "target_control_allocation_distance_difference": target_control[
            "allocation_distance"
        ]["mean"],
        "target_control_allocation_distance_ci95_low": target_control[
            "allocation_distance"
        ]["ci95_low"],
    }
    return summaries, comparisons, classification


def _paired_values(
    rows: Sequence[Mapping[str, object]],
    positive_policy: str,
    negative_policy: str,
    field: str,
) -> List[float]:
    by_policy = {
        policy: {
            int(row["episode_index"]): float(row[field])
            for row in rows
            if row["policy"] == policy
        }
        for policy in (positive_policy, negative_policy)
    }
    shared = sorted(set(by_policy[positive_policy]).intersection(by_policy[negative_policy]))
    return [
        by_policy[positive_policy][index] - by_policy[negative_policy][index]
        for index in shared
    ]


def _three_policy_recovery(rows: Sequence[Mapping[str, object]]) -> List[float]:
    by_policy = {
        policy: {
            int(row["episode_index"]): float(row["realized_utility"])
            for row in rows
            if row["policy"] == policy
        }
        for policy in (POLICY_RR, POLICY_MANUAL, POLICY_SPLIT)
    }
    shared = sorted(
        set(by_policy[POLICY_RR]).intersection(
            by_policy[POLICY_MANUAL], by_policy[POLICY_SPLIT]
        )
    )
    return [
        by_policy[POLICY_RR][index]
        - by_policy[POLICY_MANUAL][index]
        + 0.10 * (by_policy[POLICY_MANUAL][index] - by_policy[POLICY_SPLIT][index])
        for index in shared
    ]


def _contrast_row(environment: str, contrast: str, values: Sequence[float]) -> Dict[str, object]:
    mean = _mean(values)
    ci = _ci95(values)
    return {
        "environment": environment,
        "contrast": contrast,
        "n_pairs": len(values),
        "mean": mean,
        "ci95": ci,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def _paired_target_control_rows(
    rows: Sequence[Mapping[str, object]],
    target_environment: str,
    control_environment: str,
) -> List[Tuple[Mapping[str, object], Mapping[str, object]]]:
    target = {
        int(row["episode_index"]): row
        for row in rows
        if row["environment"] == target_environment and row["policy"] == POLICY_RR
    }
    control = {
        int(row["episode_index"]): row
        for row in rows
        if row["environment"] == control_environment and row["policy"] == POLICY_RR
    }
    shared = sorted(set(target).intersection(control))
    pairs = [(target[index], control[index]) for index in shared]
    for target_row, control_row in pairs:
        for field in (
            "latent_atom_index",
            "need_1",
            "need_2",
            "observation_residual_hash_1",
            "observation_residual_hash_2",
        ):
            if target_row[field] != control_row[field]:
                raise RuntimeError(f"target-control common-randomness mismatch: {field}")
    return pairs


def _paired_metric_contrast(
    pairs: Sequence[Tuple[Mapping[str, object], Mapping[str, object]]],
    field: str,
) -> Dict[str, float]:
    values = [float(target[field]) - float(control[field]) for target, control in pairs]
    mean = _mean(values)
    ci = _ci95(values)
    return {"mean": mean, "ci95": ci, "ci95_low": mean - ci, "ci95_high": mean + ci}


def _numerical_belief(environment: PositiveNeedEnvironment, kind: str):
    mdp = FiniteSupportMetaMDP(environment.config, environment.prior)
    belief = mdp.initial_belief()
    predictive_mean = belief.mean_1
    if kind == "uniform_prior":
        return belief
    if kind == "person1_predictive_mean":
        return mdp.posterior_transition(
            belief, mdp.SAMPLE_PERSON_1, predictive_mean, advance_time=False, record=False
        )
    if kind == "both_predictive_means":
        first = mdp.posterior_transition(
            belief, mdp.SAMPLE_PERSON_1, predictive_mean, advance_time=False, record=False
        )
        return mdp.posterior_transition(
            first, mdp.SAMPLE_PERSON_2, belief.mean_2, advance_time=False, record=False
        )
    if kind == "person1_minimum_support":
        observation = min(state.need_1 for state in belief.states)
    elif kind == "person1_maximum_support":
        observation = max(state.need_1 for state in belief.states)
    else:
        raise ValueError(f"unknown numerical belief kind: {kind}")
    return mdp.posterior_transition(
        belief, mdp.SAMPLE_PERSON_1, observation, advance_time=False, record=False
    )


def _belief_hash(belief) -> str:
    return _canonical_hash(
        {
            "weights": list(belief.weights),
            "deliberation_time": belief.deliberation_time,
            "history": belief.history,
        }
    )


def build_numerical_validation_cases(
    spec: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, object]]:
    spec = dict(spec or load_positive_need_spec())
    environments = [
        item
        for item in build_development_environments(spec)
        if item.sample_time_cost in {0.02, 8.0}
    ]
    kinds = (
        "uniform_prior",
        "person1_predictive_mean",
        "both_predictive_means",
        "person1_minimum_support",
        "person1_maximum_support",
    )
    cases = []
    for environment in environments:
        for kind in kinds:
            belief = _numerical_belief(environment, kind)
            cases.append(
                {
                    "case_id": len(cases),
                    "environment": environment.name,
                    "environment_hash": environment.environment_hash,
                    "belief_kind": kind,
                    "belief_hash": _belief_hash(belief),
                }
            )
    if len(cases) != 90:
        raise RuntimeError("the frozen numerical suite must contain exactly 90 beliefs")
    return cases


def build_latent_support_table(
    spec: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, object]]:
    spec = dict(spec or load_positive_need_spec())
    environments = build_development_environments(spec)
    representatives = {}
    for environment in environments:
        representatives.setdefault(environment.gap_class, environment)
    rows = []
    for gap_class in ("low", "medium", "high"):
        environment = representatives[gap_class]
        for atom_index, (atom, weight) in enumerate(
            zip(environment.prior.states, environment.prior.weights)
        ):
            rows.append(
                {
                    "gap_class": gap_class,
                    "atom_index": atom_index,
                    "total_need": atom.total_need,
                    "absolute_gap": atom.absolute_gap,
                    "gap_fraction": atom.gap_fraction,
                    "orientation": atom.orientation,
                    "need_1": atom.need_1,
                    "need_2": atom.need_2,
                    "prior_weight": weight,
                    "support_hash": environment.prior.support_hash,
                }
            )
    if len(rows) != 54:
        raise RuntimeError("the latent-support table must contain exactly 54 rows")
    return rows


def _selected_action_from_values(
    mdp: FiniteSupportMetaMDP,
    values: Mapping[str, float],
    tolerance: float,
) -> str:
    stop = float(values[mdp.TERMINATE])
    samples = [
        action
        for action in (mdp.SAMPLE_PERSON_1, mdp.SAMPLE_PERSON_2)
        if action in values and float(values[action]) > stop + tolerance
    ]
    if not samples:
        return mdp.TERMINATE
    best = max(float(values[action]) for action in samples)
    return next(action for action in samples if float(values[action]) >= best - tolerance)


NUMERICAL_ACTIONS = ("terminate", "sample_1", "sample_2")
DENSE_NUMERICAL_BELIEF_KINDS = frozenset(
    {
        "uniform_prior",
        "person1_predictive_mean",
        "person1_minimum_support",
        "person1_maximum_support",
    }
)


def dense_numerical_validation_case_ids(
    cases: Sequence[Mapping[str, object]],
    spec: Optional[Mapping[str, object]] = None,
) -> List[int]:
    """Return the exact frozen case IDs that require dense integration."""

    spec = dict(spec or load_positive_need_spec())
    environments = {item.name: item for item in build_development_environments(spec)}
    dense_ids = [
        int(case["case_id"])
        for case in cases
        if environments[str(case["environment"])].sample_time_cost == 0.02
        and str(case["belief_kind"]) in DENSE_NUMERICAL_BELIEF_KINDS
    ]
    if len(dense_ids) != 36 or len(set(dense_ids)) != 36:
        raise RuntimeError("the frozen dense-reference subset must contain 36 cases")
    return dense_ids


def _finite_action_value_map(
    values: Mapping[str, object], field: str
) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        raise RuntimeError(f"{field} must be an action-value map")
    if set(values) != set(NUMERICAL_ACTIONS):
        raise RuntimeError(f"{field} must contain terminate, sample_1, and sample_2")
    result = {action: float(values[action]) for action in NUMERICAL_ACTIONS}
    if any(not math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"{field} contains a non-finite action value")
    return result


def validate_numerical_action_value_maps(row: Mapping[str, object]) -> None:
    """Fail closed unless all action-level numerical evidence is self-consistent."""

    required = {
        "primary_action_values",
        "reference_action_values",
        "primary_reference_action_errors",
        "gh_max_action_value_error",
        "dense_reference_performed",
        "dense_reference_error",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise RuntimeError(f"numerical action-value evidence is incomplete: {missing}")
    primary = _finite_action_value_map(
        row["primary_action_values"], "primary_action_values"  # type: ignore[arg-type]
    )
    reference = _finite_action_value_map(
        row["reference_action_values"], "reference_action_values"  # type: ignore[arg-type]
    )
    errors = _finite_action_value_map(
        row["primary_reference_action_errors"],  # type: ignore[arg-type]
        "primary_reference_action_errors",
    )
    expected_errors = {
        action: abs(primary[action] - reference[action]) for action in NUMERICAL_ACTIONS
    }
    for action in NUMERICAL_ACTIONS:
        if not math.isclose(errors[action], expected_errors[action], rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"primary/reference action error mismatch: {action}")
    if not math.isclose(
        float(row["gh_max_action_value_error"]),
        max(expected_errors.values()),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("maximum primary/reference action error mismatch")

    dense_performed = float(row["dense_reference_performed"]) >= 0.5
    dense_values = row.get("dense_action_values")
    dense_errors = row.get("primary_dense_action_errors")
    if not dense_performed:
        if dense_values is not None or dense_errors is not None:
            raise RuntimeError("non-dense cases must store null dense action maps")
        if float(row["dense_reference_error"]) != 0.0:
            raise RuntimeError("non-dense cases must store zero dense-reference error")
        return

    if dense_values is None or dense_errors is None:
        raise RuntimeError("dense cases require explicit dense action maps")
    dense = _finite_action_value_map(
        dense_values, "dense_action_values"  # type: ignore[arg-type]
    )
    observed_dense_errors = _finite_action_value_map(
        dense_errors, "primary_dense_action_errors"  # type: ignore[arg-type]
    )
    expected_dense_errors = {
        action: abs(primary[action] - dense[action]) for action in NUMERICAL_ACTIONS
    }
    for action in NUMERICAL_ACTIONS:
        if not math.isclose(
            observed_dense_errors[action],
            expected_dense_errors[action],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"primary/dense action error mismatch: {action}")
    if not math.isclose(
        float(row["dense_reference_error"]),
        max(expected_dense_errors.values()),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("maximum primary/dense action error mismatch")


def _dense_sample_action_value(
    mdp: FiniteSupportMetaMDP,
    belief,
    action: str,
    grid_points: int = 8001,
) -> float:
    import numpy as np  # type: ignore

    needs = np.asarray(
        [mdp._need_for_action(state, action) for state in belief.states], dtype=float
    )
    lower = float(np.min(needs) - 8.0 * mdp.config.sigma_sample)
    upper = float(np.max(needs) + 8.0 * mdp.config.sigma_sample)
    observations = np.linspace(lower, upper, grid_points)
    posterior_weights = mdp.posterior_weights_for_observations(
        belief, action, observations
    )
    terminal_values = mdp.optimal_terminal_values_for_weights(
        belief,
        posterior_weights,
        deliberation_time=belief.deliberation_time + mdp.sample_cost(action, belief),
    )
    standardized = (
        observations[:, None] - needs[None, :]
    ) / mdp.config.sigma_sample
    component_density = np.exp(-0.5 * standardized * standardized) / (
        mdp.config.sigma_sample * math.sqrt(2.0 * math.pi)
    )
    predictive_density = component_density @ np.asarray(belief.weights, dtype=float)
    integrand = predictive_density * terminal_values
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(integrand, observations))


def validate_numerical_case(
    case: Mapping[str, object],
    spec: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    spec = dict(spec or load_positive_need_spec())
    numerical = dict(spec["numerical_settings"])  # type: ignore[arg-type]
    order = int(numerical["matched_voi_gauss_hermite_order"])
    reference_order = int(numerical["gauss_hermite_reference_order"])
    value_tolerance = float(numerical["action_value_convergence_tolerance"])
    allocation_tolerance = float(numerical["allocation_convergence_tolerance"])
    tie_tolerance = float(numerical["action_tie_tolerance"])
    environments = {item.name: item for item in build_development_environments(spec)}
    environment = environments[str(case["environment"])]
    belief = _numerical_belief(environment, str(case["belief_kind"]))
    if _belief_hash(belief) != case["belief_hash"]:
        raise RuntimeError(f"numerical belief hash mismatch: {case['case_id']}")
    mdp = FiniteSupportMetaMDP(environment.config, environment.prior)
    policy = FiniteSupportMyopicVOIPolicy(order, tie_tolerance=tie_tolerance)
    reference_policy = FiniteSupportMyopicVOIPolicy(
        reference_order, tie_tolerance=tie_tolerance
    )
    values = policy.action_values(mdp, belief)
    reference_values = reference_policy.action_values(mdp, belief)
    primary_action_values = _finite_action_value_map(values, "primary_action_values")
    reference_action_values = _finite_action_value_map(
        reference_values, "reference_action_values"
    )
    primary_reference_action_errors = {
        key: abs(primary_action_values[key] - reference_action_values[key])
        for key in NUMERICAL_ACTIONS
    }
    action = policy.choose_action(mdp, belief)
    reference_action = reference_policy.choose_action(mdp, belief)
    gh_error = max(primary_reference_action_errors.values())

    reference_environment = PositiveNeedEnvironment(
        name=environment.name,
        gap_class=environment.gap_class,
        sigma_sample=environment.sigma_sample,
        sample_time_cost=environment.sample_time_cost,
        config=replace(
            environment.config,
            allocation_grid_size=int(numerical["rr_terminal_reference_grid_size"]),
        ),
        prior=environment.prior,
    )
    reference_mdp = FiniteSupportMetaMDP(
        reference_environment.config, reference_environment.prior
    )
    allocation, terminal_value = mdp.solve_terminal_allocation(belief)
    reference_allocation, reference_terminal_value = reference_mdp.solve_terminal_allocation(
        belief
    )
    grid_allocation_error = abs(allocation - reference_allocation)
    grid_value_error = abs(terminal_value - reference_terminal_value)
    grid_action = FiniteSupportMyopicVOIPolicy(order, tie_tolerance).choose_action(
        reference_mdp, belief
    )

    dense_error = 0.0
    dense_action = action
    dense_action_values = None
    primary_dense_action_errors = None
    dense_reference_performed = (
        environment.sample_time_cost == 0.02
        and case["belief_kind"] in DENSE_NUMERICAL_BELIEF_KINDS
    )
    if dense_reference_performed:
        dense_values = {mdp.TERMINATE: float(values[mdp.TERMINATE])}
        for sample_action in (mdp.SAMPLE_PERSON_1, mdp.SAMPLE_PERSON_2):
            dense_values[sample_action] = _dense_sample_action_value(
                mdp, belief, sample_action
            )
        dense_action_values = _finite_action_value_map(
            dense_values, "dense_action_values"
        )
        primary_dense_action_errors = {
            key: abs(primary_action_values[key] - dense_action_values[key])
            for key in NUMERICAL_ACTIONS
        }
        dense_error = max(primary_dense_action_errors.values())
        dense_action = _selected_action_from_values(mdp, dense_values, tie_tolerance)

    passed = (
        gh_error <= value_tolerance
        and action == reference_action
        and grid_allocation_error <= allocation_tolerance
        and grid_value_error <= value_tolerance
        and action == grid_action
        and dense_error <= value_tolerance
        and action == dense_action
    )
    row = {
        **case,
        "gh_order": order,
        "gh_reference_order": reference_order,
        "primary_action_values": primary_action_values,
        "reference_action_values": reference_action_values,
        "primary_reference_action_errors": primary_reference_action_errors,
        "gh_max_action_value_error": gh_error,
        "gh_action": action,
        "gh_reference_action": reference_action,
        "terminal_grid_allocation_error": grid_allocation_error,
        "terminal_grid_value_error": grid_value_error,
        "terminal_reference_action": grid_action,
        "dense_reference_error": dense_error,
        "dense_action_values": dense_action_values,
        "primary_dense_action_errors": primary_dense_action_errors,
        "dense_reference_action": dense_action,
        "dense_reference_performed": 1.0 if dense_reference_performed else 0.0,
        "passed": 1.0 if passed else 0.0,
    }
    validate_numerical_action_value_maps(row)
    return row


def summarize_numerical_validation(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not rows:
        raise ValueError("numerical validation requires at least one row")
    failures = [int(row["case_id"]) for row in rows if float(row["passed"]) < 0.5]
    summary = {
        "case_count": len(rows),
        "passed_case_count": len(rows) - len(failures),
        "failed_case_count": len(failures),
        "failed_case_ids": failures,
        "max_gh_action_value_error": max(float(row["gh_max_action_value_error"]) for row in rows),
        "max_terminal_grid_allocation_error": max(
            float(row["terminal_grid_allocation_error"]) for row in rows
        ),
        "max_terminal_grid_value_error": max(
            float(row["terminal_grid_value_error"]) for row in rows
        ),
        "max_dense_reference_error": max(float(row["dense_reference_error"]) for row in rows),
        "valid": not failures,
    }
    return summary


def validate_numerical_suite(
    spec: Optional[Mapping[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    spec = dict(spec or load_positive_need_spec())
    rows = [
        validate_numerical_case(case, spec)
        for case in build_numerical_validation_cases(spec)
    ]
    return rows, summarize_numerical_validation(rows)


def normal_quantile_grid(mean: float, sd: float, points: int = 20001) -> List[float]:
    """Dense deterministic integration grid used only by GH validation."""

    if points < 101 or points % 2 == 0 or sd <= 0.0:
        raise ValueError("dense normal grid requires odd points >=101 and positive sd")
    distribution = NormalDist(mean, sd)
    epsilon = 1e-9
    return [
        distribution.inv_cdf(epsilon + (1.0 - 2.0 * epsilon) * index / (points - 1))
        for index in range(points)
    ]
