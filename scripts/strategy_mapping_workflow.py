#!/usr/bin/env python3
"""Create, execute, monitor, and strictly collect StrategyMapping array shards."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.strategy_mapping import (  # noqa: E402
    STRATEGY_MAPPING_DEFAULT_GAP_BIN_EDGES,
    STRATEGY_MAPPING_DEFAULT_TOTAL_NEED_BIN_EDGES,
    STRATEGY_MAPPING_FIXED_TOTAL_NEED_DIFFERENCES,
    STRATEGY_MAPPING_FIXED_TOTAL_NEED_MEAN,
    STRATEGY_MAPPING_POLICY_ORDER,
    STRATEGY_MAPPING_POLICY_EQUAL_SPLIT,
    STRATEGY_MAPPING_POLICY_MANUAL,
    STRATEGY_MAPPING_POLICY_ORACLE,
    STRATEGY_MAPPING_POLICY_RR,
    evaluate_strategy_mapping_fixed_total_need_diagnostic,
    evaluate_strategy_mapping_four_way_environment,
    evaluate_strategy_mapping_sigma_need_sweep,
    select_strategy_mapping_primary_environments,
    summarize_strategy_mapping_fixed_total_need_diagnostic,
    summarize_strategy_mapping_four_way,
    summarize_strategy_mapping_sigma_need_sweep,
)
from src.experiments.randomization import build_evaluation_episode  # noqa: E402
from src.mdp.meta_mdp import EnvironmentConfig  # noqa: E402
from src.policies.voi import MyopicValueOfInformationPolicy  # noqa: E402


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "strategy_mapping_manifest.json"
PROGRESS_FILENAME = "strategy_mapping_progress.json"
VALIDATION_FILENAME = "strategy_mapping_validation.json"
ARTIFACT_INDEX_FILENAME = "strategy_mapping_artifact_index.json"
COMPLETION_FILENAME = "COMPLETED.json"
SCHEDULER_FILENAME = "strategy_mapping_scheduler.json"
COLLECTION_LOCK_FILENAME = ".strategy_mapping_collection.lock"
TASK_LOCK_FILENAME = ".strategy_mapping_task.lock"
RESULT_VERSIONS_DIRNAME = "result_versions"

ANALYSIS_FOUR_WAY = "four_way"
ANALYSIS_SIGMA = "sigma_need"
ANALYSIS_FIXED = "fixed_total_need"
ANALYSES = (ANALYSIS_FOUR_WAY, ANALYSIS_SIGMA, ANALYSIS_FIXED)

EPISODE_OUTPUTS = {
    ANALYSIS_FOUR_WAY: "strategy_mapping_four_way_episodes.csv",
    ANALYSIS_SIGMA: "strategy_mapping_sigma_need_episodes.csv",
    ANALYSIS_FIXED: "strategy_mapping_fixed_total_need_episodes.csv",
}
SUMMARY_OUTPUTS = (
    "strategy_mapping_four_way_policy_summary.csv",
    "strategy_mapping_four_way_paired_comparisons.csv",
    "strategy_mapping_sigma_need_environment_summary.csv",
    "strategy_mapping_sigma_need_gap_strata.csv",
    "strategy_mapping_fixed_total_need_summary.csv",
)
ALL_OUTPUTS = tuple(EPISODE_OUTPUTS.values()) + SUMMARY_OUTPUTS

EPISODE_TEXT_FIELDS = {
    "environment",
    "environment_config_hash",
    "policy",
    "policy_role",
    "episode_fingerprint",
    "observation_stream_hash_1",
    "observation_stream_hash_2",
    "observation_residual_hash_1",
    "observation_residual_hash_2",
    "realized_sign_stratum",
    "oracle_sign_stratum",
    "non_sigma_config_hash",
    "mechanism_diagnostic",
    "orientation_rule",
    "analysis_mode",
    "scientific_status",
}

COMMON_BINARY_FIELDS = {
    "sampled_both_recipients",
    "immediate_termination",
    "exact_true_equal_outcome_feasible",
    "true_equal_outcome",
    "true_equal_outcome_allocation_close",
    "closer_to_true_equal_outcome_than_equal_split",
    "closer_to_equal_split_than_true_equal_outcome",
    "true_outcome_classification_tie",
    "legacy_tolerance_closer_to_true_equal_outcome_than_equal_split",
    "legacy_tolerance_true_outcome_classification_tie",
    "negative_need_person1",
    "negative_need_person2",
    "negative_need_either",
    "negative_need_both",
}

NONNEGATIVE_FIELDS = {
    "realized_true_need_gap",
    "abs_allocation_from_equal",
    "realized_outcome_gap",
    "equal_split_realized_outcome_gap",
    "true_equal_outcome_solution_gap",
    "true_equal_outcome_allocation_gap",
    "outcome_distance_to_true_equal",
    "equal_split_outcome_distance_to_true_equal",
    "outcome_success_tolerance",
    "classification_tie_tolerance",
    "utility_regret_to_initial_oracle",
    "rr_manual_allocation_gap",
    "rr_time_matched_oracle_regret",
    "rr_time_matched_oracle_optimality_violation",
    "rr_time_matched_oracle_realized_outcome_gap",
    "manual_time_matched_oracle_regret",
    "manual_time_matched_oracle_optimality_violation",
    "manual_time_matched_oracle_realized_outcome_gap",
    "oracle_grid_optimality_violation",
    "sigma_need",
    "initial_oracle_regret",
    "initial_oracle_optimality_violation",
    "max_observation_reconstruction_error_1",
    "max_observation_reconstruction_error_2",
    "fixed_total_need_mean",
    "constructed_need_difference",
}

ALLOCATION_FIELDS = {
    "allocation_to_person1",
    "true_equal_outcome_allocation",
    "initial_oracle_allocation",
    "rr_time_matched_oracle_allocation",
    "manual_time_matched_oracle_allocation",
    "oracle_allocation",
}

RR_TIME_MATCHED_FIELDS = {
    "rr_time_matched_oracle_allocation",
    "rr_time_matched_oracle_utility",
    "rr_time_matched_oracle_regret",
    "rr_time_matched_oracle_raw_regret",
    "rr_time_matched_oracle_optimality_violation",
    "rr_time_matched_oracle_realized_outcome_gap",
    "rr_time_matched_oracle_true_equal_outcome",
    "rr_time_matched_oracle_closer_to_true_equal_than_equal_split",
}

MANUAL_TIME_MATCHED_FIELDS = {
    "manual_time_matched_oracle_allocation",
    "manual_time_matched_oracle_utility",
    "manual_time_matched_oracle_regret",
    "manual_time_matched_oracle_raw_regret",
    "manual_time_matched_oracle_optimality_violation",
    "manual_time_matched_oracle_realized_outcome_gap",
    "manual_time_matched_oracle_true_equal_outcome",
    "manual_time_matched_oracle_closer_to_true_equal_than_equal_split",
}

DIAGNOSTIC_ACTIVE_SEARCH_EXPECTED_HASHES = {
    "active_search_diagnostic_environment_summary.csv": (
        "84bb0e5efede45b4453cdfffca037a299a3dc9e212807a236579df5030f3f3a0"
    ),
    "active_search_diagnostic_manual_advantage_candidates.csv": (
        "c668d43b7538481c88983ea42dde3067e5a96182f76d0e689616524df4810ddd"
    ),
    "diagnostic_active_search_array_manifest.json": (
        "7e4e86c40e0656f7e42ee168ccf59587e61d7a56f6ad77ba9810a333c4949aeb"
    ),
}
DIAGNOSTIC_ACTIVE_SEARCH_SOURCE_COMMIT = "4102fe34a525dd816e7939169c8581159551bed9"
SERIOUS_EPISODES = 1200
SERIOUS_EPISODES_PER_TASK = 10
SERIOUS_OBSERVATION_DRAWS = 500
SERIOUS_ORACLE_GRID_SIZE = 4001
SERIOUS_SEED_NAMESPACE_OFFSET = 60_000_000
SMOKE_SEED_NAMESPACE_OFFSET = 70_000_000
SERIOUS_SIGMA_NEED_VALUES = (10.0, 60.0, 100.0)

IMPLEMENTATION_SOURCES = (
    "configs/strategy_mapping_environments.json",
    "configs/strategy_mapping_output_schemas.json",
    "scripts/strategy_mapping_workflow.py",
    "scripts/submit_hoffman2_strategy_mapping.sh",
    "src/experiments/active_search_evaluation.py",
    "src/experiments/strategy_mapping.py",
    "src/experiments/randomization.py",
    "src/mdp/meta_mdp.py",
    "src/policies/heuristic.py",
    "src/policies/voi.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def schema_hash(fields: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(fields), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def git_is_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = set(fields)
    for index, row in enumerate(rows):
        extra = sorted(set(row).difference(expected))
        if extra:
            raise RuntimeError(
                f"Schema mismatch before writing {path.name} row {index}: "
                f"extra={extra}"
            )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv_schema(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def load_schema_contract(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported StrategyMapping output-schema contract")
    schemas = {str(key): list(value) for key, value in payload["schemas"].items()}
    if set(schemas) != set(ALL_OUTPUTS):
        raise RuntimeError("StrategyMapping output-schema contract has missing or extra tables")
    return schemas


def _parse_float_list(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _source_hashes() -> Dict[str, str]:
    result = {}
    for relative in IMPLEMENTATION_SOURCES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing implementation source: {relative}")
        result[relative] = sha256_file(path)
    return result


def _load_frozen_inputs(
    diagnostic_active_search_dir: Path,
    frozen_config_path: Path,
    seed_namespace_offset: int,
) -> Tuple[List[Mapping[str, object]], List[Tuple[str, EnvironmentConfig]], Dict[str, str]]:
    observed_hashes: Dict[str, str] = {}
    for filename, expected_hash in DIAGNOSTIC_ACTIVE_SEARCH_EXPECTED_HASHES.items():
        path = diagnostic_active_search_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen DiagnosticActiveSearch input: {path}")
        observed_hashes[filename] = sha256_file(path)
        if observed_hashes[filename] != expected_hash:
            raise RuntimeError(f"Frozen DiagnosticActiveSearch hash mismatch: {filename}")

    candidate_path = diagnostic_active_search_dir / "active_search_diagnostic_manual_advantage_candidates.csv"
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        selected_rows = list(select_strategy_mapping_primary_environments(csv.DictReader(handle)))

    frozen = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    frozen_rows = list(frozen["environments"])
    expected_names = [str(row["environment"]) for row in selected_rows]
    if [str(row["environment"]) for row in frozen_rows] != expected_names:
        raise RuntimeError("Frozen StrategyMapping environments do not match deterministic DiagnosticActiveSearch selection")
    checked_fields = (
        "mu_need",
        "sigma_need",
        "sigma_sample",
        "total_time",
        "sample_time_cost",
        "utility_exponent",
        "learning_per_unit_of_tutoring",
        "delta_learning_per_unit_tutoring",
        "prior_sample_count_1",
        "prior_sample_count_2",
    )
    selected_configs: List[Tuple[str, EnvironmentConfig]] = []
    for source, frozen_row in zip(selected_rows, frozen_rows):
        if int(float(source["grid_index"])) != int(frozen_row["source_grid_index"]):
            raise RuntimeError("Frozen StrategyMapping grid index does not match its DiagnosticActiveSearch source row")
        for field in checked_fields:
            if float(source[field]) != float(frozen_row["config"][field]):
                raise RuntimeError(f"Frozen StrategyMapping parameter mismatch: {field}")
        config = EnvironmentConfig(**frozen_row["config"])
        selected_configs.append(
            (
                str(frozen_row["environment"]),
                replace(
                    config,
                    random_seed=(config.random_seed or 0) + seed_namespace_offset,
                ),
            )
        )
    return selected_rows, selected_configs, observed_hashes


def _validate_partition(episodes: int, episodes_per_task: int) -> None:
    if episodes <= 0 or episodes_per_task <= 0:
        raise ValueError("episode counts must be positive")
    if episodes % 2 or episodes_per_task % 2:
        raise ValueError("StrategyMapping fixed-total orientation requires even episode partitions")
    if episodes % episodes_per_task:
        raise ValueError("episodes must be divisible by episodes_per_task")


def build_tasks(
    selected_configs: Sequence[Tuple[str, EnvironmentConfig]],
    episodes: int,
    episodes_per_task: int,
    sigma_need_values: Sequence[float],
    seed_namespace_offset: int,
) -> List[Dict[str, object]]:
    _validate_partition(episodes, episodes_per_task)
    tasks: List[Dict[str, object]] = []

    def add_task(
        analysis: str,
        episode_start: int,
        episode_count: int,
        environment_index: int | None,
        condition_index: int,
        condition_label: str,
        condition_value: object,
        config_hash: str,
        base_seed: int,
        row_multiplier: int,
    ) -> None:
        task = {
            "task_index": len(tasks),
            "analysis": analysis,
            "environment_index": environment_index,
            "condition_index": condition_index,
            "condition_label": condition_label,
            "condition_value": condition_value,
            "environment_config_hash": config_hash,
            "episode_start": episode_start,
            "episode_end": episode_start + episode_count - 1,
            "episode_count": episode_count,
            "expected_row_count": episode_count * row_multiplier,
            "seed_range": {
                "namespace_offset": seed_namespace_offset,
                "first_global_episode_index": episode_start,
                "last_global_episode_index": episode_start + episode_count - 1,
                "true_state_seed_first": base_seed + episode_start * 17 + 1,
                "true_state_seed_last": base_seed + (episode_start + episode_count - 1) * 17 + 1,
                "observation_seed_first": base_seed + 100_000 + episode_start * 17,
                "observation_seed_last": base_seed + 100_000 + (episode_start + episode_count - 1) * 17,
                "policy_seed_first": base_seed + 300_000 + episode_start * 17,
                "policy_seed_last": base_seed + 300_000 + (episode_start + episode_count - 1) * 17,
                "fixed_total_observation_seed_first": base_seed + 600_000 + episode_start * 17,
                "fixed_total_observation_seed_last": base_seed + 600_000 + (episode_start + episode_count - 1) * 17,
            },
        }
        task["scientific_command_fingerprint"] = digest(
            {
                key: task[key]
                for key in (
                    "analysis",
                    "environment_index",
                    "condition_index",
                    "condition_label",
                    "condition_value",
                    "environment_config_hash",
                    "episode_start",
                    "episode_end",
                    "episode_count",
                    "expected_row_count",
                    "seed_range",
                )
            }
        )
        tasks.append(task)

    for environment_index, (environment, config) in enumerate(selected_configs):
        for start in range(0, episodes, episodes_per_task):
            add_task(
                ANALYSIS_FOUR_WAY,
                start,
                episodes_per_task,
                environment_index,
                environment_index,
                "environment",
                environment,
                digest(asdict(config)),
                config.random_seed or 0,
                4,
            )
    _, base_config = selected_configs[0]
    for sigma_index, sigma in enumerate(sigma_need_values):
        sigma_config = replace(base_config, sigma_need=float(sigma))
        for start in range(0, episodes, episodes_per_task):
            add_task(
                ANALYSIS_SIGMA,
                start,
                episodes_per_task,
                None,
                sigma_index,
                "sigma_need",
                float(sigma),
                digest(asdict(sigma_config)),
                sigma_config.random_seed or 0,
                1,
            )
    for difference_index, difference in enumerate(STRATEGY_MAPPING_FIXED_TOTAL_NEED_DIFFERENCES):
        for start in range(0, episodes, episodes_per_task):
            add_task(
                ANALYSIS_FIXED,
                start,
                episodes_per_task,
                None,
                difference_index,
                "constructed_need_difference",
                float(difference),
                digest(asdict(base_config)),
                base_config.random_seed or 0,
                1,
            )
    return tasks


def _expected_row_counts(environment_count: int, episodes: int, sigma_count: int) -> Dict[str, int]:
    four_way_strata = 9
    sigma_strata = 2 + 2 + 4 + (len(STRATEGY_MAPPING_DEFAULT_GAP_BIN_EDGES) - 1) + (
        len(STRATEGY_MAPPING_DEFAULT_TOTAL_NEED_BIN_EDGES) - 1
    )
    fixed_strata = 1 + 2 + 2 + 4
    fixed_count = len(STRATEGY_MAPPING_FIXED_TOTAL_NEED_DIFFERENCES)
    return {
        "strategy_mapping_four_way_episodes.csv": environment_count * episodes * 4,
        "strategy_mapping_four_way_policy_summary.csv": environment_count * four_way_strata * 4,
        "strategy_mapping_four_way_paired_comparisons.csv": environment_count * four_way_strata * 7,
        "strategy_mapping_sigma_need_episodes.csv": sigma_count * episodes,
        "strategy_mapping_sigma_need_environment_summary.csv": sigma_count,
        "strategy_mapping_sigma_need_gap_strata.csv": sigma_count * sigma_strata,
        "strategy_mapping_fixed_total_need_episodes.csv": fixed_count * episodes,
        "strategy_mapping_fixed_total_need_summary.csv": fixed_count * fixed_strata + 1,
    }


def build_manifest(
    *,
    output_dir: Path,
    selected_rows: Sequence[Mapping[str, object]],
    selected_configs: Sequence[Tuple[str, EnvironmentConfig]],
    diagnostic_active_search_dir: Path,
    diagnostic_active_search_hashes: Mapping[str, str],
    frozen_config_path: Path,
    schema_contract_path: Path,
    episodes: int,
    episodes_per_task: int,
    observation_draws: int,
    oracle_grid_size: int,
    observations_per_person: int,
    sigma_need_values: Sequence[float],
    seed_namespace_offset: int,
    analysis_mode: str = "serious",
) -> Dict[str, object]:
    if analysis_mode not in {"serious", "smoke", "test"}:
        raise ValueError("analysis_mode must be serious, smoke, or test")
    if analysis_mode == "serious" and (
        episodes != SERIOUS_EPISODES
        or episodes_per_task != SERIOUS_EPISODES_PER_TASK
        or observation_draws != SERIOUS_OBSERVATION_DRAWS
        or oracle_grid_size != SERIOUS_ORACLE_GRID_SIZE
        or seed_namespace_offset != SERIOUS_SEED_NAMESPACE_OFFSET
        or tuple(float(value) for value in sigma_need_values) != SERIOUS_SIGMA_NEED_VALUES
    ):
        raise ValueError(
            "Serious StrategyMapping settings are frozen at 1200 episodes, 10 episodes/task, "
            "500 draws, oracle grid 4001, namespace 60000000, and sigma 10/60/100"
        )
    if analysis_mode == "smoke" and (
        episodes != 2
        or episodes_per_task != 2
        or observation_draws != SERIOUS_OBSERVATION_DRAWS
        or oracle_grid_size != SERIOUS_ORACLE_GRID_SIZE
        or seed_namespace_offset != SMOKE_SEED_NAMESPACE_OFFSET
    ):
        raise ValueError(
            "Scheduled StrategyMapping smoke is frozen at two episodes, 500 draws, grid 4001, "
            "and namespace 70000000"
        )
    if len(selected_configs) != 3 or len(selected_rows) != 3:
        raise ValueError("StrategyMapping requires exactly three frozen primary environments")
    _validate_partition(episodes, episodes_per_task)
    if observation_draws <= 0 or oracle_grid_size < 3 or observations_per_person <= 0:
        raise ValueError("Invalid policy, oracle, or observation setting")

    sigma_values = [float(value) for value in sigma_need_values]
    if (
        len(set(sigma_values)) != len(sigma_values)
        or any(not math.isfinite(value) or value <= 0.0 for value in sigma_values)
    ):
        raise ValueError("sigma_need values must be unique, finite, and positive")
    schemas = load_schema_contract(schema_contract_path)
    tasks = build_tasks(
        selected_configs,
        episodes,
        episodes_per_task,
        sigma_values,
        seed_namespace_offset,
    )
    expected_counts = _expected_row_counts(len(selected_configs), episodes, len(sigma_values))
    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "workflow": "strategy_mapping_serious_hoffman2_array",
        "analysis_mode": analysis_mode,
        "scientific_completion_allowed": analysis_mode == "serious",
        "output_dir": str(output_dir.resolve()),
        "git_commit": git_commit(),
        "git_worktree_clean_at_creation": git_is_clean(),
        "implementation_source_hashes": _source_hashes(),
        "diagnostic_active_search_input_dir": str(diagnostic_active_search_dir.resolve()),
        "diagnostic_active_search_input_hashes": dict(diagnostic_active_search_hashes),
        "diagnostic_active_search_source_commit": DIAGNOSTIC_ACTIVE_SEARCH_SOURCE_COMMIT,
        "frozen_environment_config_path": str(frozen_config_path.resolve()),
        "frozen_environment_config_sha256": sha256_file(frozen_config_path),
        "output_schema_contract_path": str(schema_contract_path.resolve()),
        "output_schema_contract_sha256": sha256_file(schema_contract_path),
        "selection_rule": (
            "DiagnosticActiveSearch manual advantage > 0; manual true-equal and closer rates >= 0.80; "
            "manual samples >= 6; allocation distance >= 0.05; rank utility descending "
            "then environment ascending; retain first three"
        ),
        "selected_environments": [
            {
                "environment_index": index,
                "environment": environment,
                "source_grid_index": int(float(selected_rows[index]["grid_index"])),
                "config": asdict(config),
            }
            for index, (environment, config) in enumerate(selected_configs)
        ],
        "settings": {
            "episodes_per_condition": episodes,
            "episodes_per_task": episodes_per_task,
            "episode_index_range": [0, episodes - 1],
            "rr_policy_class": "MyopicValueOfInformationPolicy",
            "observation_draws": observation_draws,
            "manual_policy_class": "ManualActiveSearchEqualOutcomePolicy",
            "manual_samples_per_person": 3,
            "equal_split_policy_class": "EqualSplitBaselinePolicy",
            "oracle_function": "full_information_utilitarian_allocation",
            "oracle_grid_size": oracle_grid_size,
            "allocation_tolerance": 0.05,
            "allocation_tie_tolerance": 1e-9,
            "observations_per_person_minimum": observations_per_person,
            "sigma_need_values": sigma_values,
            "realized_need_gap_bin_edges": [
                "inf" if math.isinf(value) else value for value in STRATEGY_MAPPING_DEFAULT_GAP_BIN_EDGES
            ],
            "total_need_bin_edges": [
                "-inf" if math.isinf(value) and value < 0.0 else "inf" if math.isinf(value) else value
                for value in STRATEGY_MAPPING_DEFAULT_TOTAL_NEED_BIN_EDGES
            ],
            "fixed_total_need_mean": STRATEGY_MAPPING_FIXED_TOTAL_NEED_MEAN,
            "fixed_total_need_differences": list(STRATEGY_MAPPING_FIXED_TOTAL_NEED_DIFFERENCES),
            "held_out_seed_namespace_offset": seed_namespace_offset,
            "true_state_seed_rule": "config.random_seed + episode_index * 17 + 1",
            "observation_seed_rule": "config.random_seed + 100000 + episode_index * 17",
            "policy_computation_seed_rule": "config.random_seed + 300000 + episode_index * 17",
            "fixed_total_observation_seed_rule": "config.random_seed + 600000 + episode_index * 17",
            "scheduler_metadata_affects_scientific_seed": False,
        },
        "frozen_thresholds": {
            "diagnostic_rate": 0.80,
            "manual_minus_split_paired_ci95_low": "> 0",
            "utility_recovery_fraction": 0.90,
            "rr_mean_online_samples": "> 1",
            "sample_both_recipients_rate": 0.80,
            "mean_abs_allocation_from_equal": 0.05,
        },
        "expected_output_schemas": schemas,
        "expected_output_schema_hashes": {
            filename: schema_hash(fields) for filename, fields in schemas.items()
        },
        "expected_row_counts": expected_counts,
        "tasks": tasks,
    }
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def load_manifest(path: Path) -> Dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = manifest.pop("manifest_hash", None)
    actual_hash = digest(manifest)
    manifest["manifest_hash"] = claimed_hash
    if claimed_hash != actual_hash:
        raise RuntimeError("StrategyMapping manifest hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported StrategyMapping manifest schema")
    if manifest.get("workflow") != "strategy_mapping_serious_hoffman2_array":
        raise RuntimeError("Unexpected StrategyMapping workflow")
    if manifest.get("analysis_mode") == "serious":
        settings = manifest["settings"]
        if (
            int(settings["episodes_per_condition"]) != SERIOUS_EPISODES
            or int(settings["episodes_per_task"]) != SERIOUS_EPISODES_PER_TASK
            or int(settings["observation_draws"]) != SERIOUS_OBSERVATION_DRAWS
            or int(settings["oracle_grid_size"]) != SERIOUS_ORACLE_GRID_SIZE
            or int(settings["held_out_seed_namespace_offset"])
            != SERIOUS_SEED_NAMESPACE_OFFSET
            or tuple(float(value) for value in settings["sigma_need_values"])
            != SERIOUS_SIGMA_NEED_VALUES
        ):
            raise RuntimeError("Serious StrategyMapping manifest settings changed")
    if manifest.get("analysis_mode") == "smoke":
        settings = manifest["settings"]
        if (
            int(settings["episodes_per_condition"]) != 2
            or int(settings["episodes_per_task"]) != 2
            or int(settings["observation_draws"]) != SERIOUS_OBSERVATION_DRAWS
            or int(settings["oracle_grid_size"]) != SERIOUS_ORACLE_GRID_SIZE
            or int(settings["held_out_seed_namespace_offset"])
            != SMOKE_SEED_NAMESPACE_OFFSET
        ):
            raise RuntimeError("Scheduled StrategyMapping smoke settings changed")
    if manifest.get("analysis_mode") in {"serious", "smoke"} and not manifest.get(
        "git_worktree_clean_at_creation"
    ):
        raise RuntimeError("A serious or scheduled-smoke manifest was frozen from a dirty worktree")
    for filename, fields in manifest["expected_output_schemas"].items():
        if schema_hash(fields) != manifest["expected_output_schema_hashes"].get(filename):
            raise RuntimeError(f"Manifest output-schema hash mismatch: {filename}")
    selected_configs = [
        (str(entry["environment"]), EnvironmentConfig(**entry["config"]))
        for entry in manifest["selected_environments"]
    ]
    settings = manifest["settings"]
    expected_tasks = build_tasks(
        selected_configs,
        int(settings["episodes_per_condition"]),
        int(settings["episodes_per_task"]),
        [float(value) for value in settings["sigma_need_values"]],
        int(settings["held_out_seed_namespace_offset"]),
    )
    if manifest["tasks"] != expected_tasks:
        raise RuntimeError("StrategyMapping task map is not the exact frozen Cartesian partition")
    return manifest


def verify_execution_checkout(manifest: Mapping[str, object]) -> None:
    if git_commit() != manifest["git_commit"]:
        raise RuntimeError("Execution checkout does not match the frozen StrategyMapping commit")
    for relative, expected_hash in manifest["implementation_source_hashes"].items():
        path = PROJECT_ROOT / str(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Implementation source mismatch: {relative}")
    if manifest["analysis_mode"] in {"serious", "smoke"} and not git_is_clean():
        raise RuntimeError("Serious and scheduled-smoke task execution requires a clean worktree")


def task_paths(manifest_path: Path, task_index: int) -> Tuple[Path, Path, Path]:
    task_dir = manifest_path.parent / "tasks" / f"task_{task_index:06d}"
    return task_dir / "rows.csv", task_dir / "status.json", task_dir / "failure.json"


def _task_for_index(manifest: Mapping[str, object], task_index: int) -> Mapping[str, object]:
    tasks = manifest["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError("task_index outside manifest")
    task = tasks[task_index]
    if int(task["task_index"]) != task_index:
        raise RuntimeError("Manifest task ordering is inconsistent")
    return task


def _analysis_mode_fields(manifest: Mapping[str, object], analysis: str) -> Dict[str, str]:
    if manifest["analysis_mode"] == "test":
        return {
            "analysis_mode": "test",
            "scientific_status": "test_only_not_scientific_evidence",
        }
    if manifest["analysis_mode"] == "smoke":
        return {
            "analysis_mode": "smoke",
            "scientific_status": "smoke_only_not_scientific_evidence",
        }
    return {
        "analysis_mode": "held_out",
        "scientific_status": "serious_prespecified_evaluation",
    }


def _environment_entry(manifest: Mapping[str, object], index: int = 0) -> Mapping[str, object]:
    entries = manifest["selected_environments"]
    if index < 0 or index >= len(entries):
        raise IndexError("environment index outside manifest")
    return entries[index]


def execute_task_rows(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
) -> List[Dict[str, object]]:
    settings = manifest["settings"]
    analysis = str(task["analysis"])
    episode_start = int(task["episode_start"])
    episode_count = int(task["episode_count"])
    rr_policy = MyopicValueOfInformationPolicy(
        observation_draws=int(settings["observation_draws"])
    )
    if analysis == ANALYSIS_FOUR_WAY:
        entry = _environment_entry(manifest, int(task["environment_index"]))
        config = EnvironmentConfig(**entry["config"])
        episodes = [
            build_evaluation_episode(
                config,
                episode_index=index,
                include_observation_streams=True,
                observations_per_person=int(settings["observations_per_person_minimum"]),
            )
            for index in range(episode_start, episode_start + episode_count)
        ]
        rows = evaluate_strategy_mapping_four_way_environment(
            str(entry["environment"]),
            config,
            episodes,
            rr_policy=rr_policy,
            manual_samples_per_person=int(settings["manual_samples_per_person"]),
            allocation_tolerance=float(settings["allocation_tolerance"]),
            oracle_grid_size=int(settings["oracle_grid_size"]),
        )
    else:
        entry = _environment_entry(manifest)
        config = EnvironmentConfig(**entry["config"])
        if analysis == ANALYSIS_SIGMA:
            rows = evaluate_strategy_mapping_sigma_need_sweep(
                str(entry["environment"]),
                config,
                [float(task["condition_value"])],
                n_episodes=episode_count,
                episode_start=episode_start,
                rr_policy=rr_policy,
                allocation_tolerance=float(settings["allocation_tolerance"]),
                oracle_grid_size=int(settings["oracle_grid_size"]),
                observations_per_person=int(settings["observations_per_person_minimum"]),
            )
        elif analysis == ANALYSIS_FIXED:
            rows = evaluate_strategy_mapping_fixed_total_need_diagnostic(
                str(entry["environment"]),
                config,
                n_episodes_per_difference=episode_count,
                episode_start=episode_start,
                rr_policy=rr_policy,
                total_need_mean=float(settings["fixed_total_need_mean"]),
                need_differences=[float(task["condition_value"])],
                allocation_tolerance=float(settings["allocation_tolerance"]),
                oracle_grid_size=int(settings["oracle_grid_size"]),
                observations_per_person=int(settings["observations_per_person_minimum"]),
            )
        else:
            raise ValueError(f"Unknown StrategyMapping task analysis: {analysis}")
    tags = _analysis_mode_fields(manifest, analysis)
    for row in rows:
        row.update(tags)
    return rows


def _task_expected_keys(
    manifest: Mapping[str, object], task: Mapping[str, object]
) -> set[Tuple[object, ...]]:
    start = int(task["episode_start"])
    stop = start + int(task["episode_count"])
    analysis = str(task["analysis"])
    if analysis == ANALYSIS_FOUR_WAY:
        environment = str(_environment_entry(manifest, int(task["environment_index"]))["environment"])
        return {
            (environment, index, policy)
            for index in range(start, stop)
            for policy in STRATEGY_MAPPING_POLICY_ORDER
        }
    base_environment = str(_environment_entry(manifest)["environment"])
    if analysis == ANALYSIS_SIGMA:
        sigma = float(task["condition_value"])
        return {
            (f"{base_environment}__sigma_need={sigma:g}", index)
            for index in range(start, stop)
        }
    difference = float(task["condition_value"])
    return {
        (difference, index)
        for index in range(start, stop)
    }


def _row_key(analysis: str, row: Mapping[str, object]) -> Tuple[object, ...]:
    if analysis == ANALYSIS_FOUR_WAY:
        return str(row["environment"]), int(float(row["episode_index"])), str(row["policy"])
    if analysis == ANALYSIS_SIGMA:
        return str(row["environment"]), int(float(row["episode_index"]))
    return float(row["constructed_need_difference"]), int(float(row["episode_index"]))


def _require_finite(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str], label: str
) -> None:
    for row_index, row in enumerate(rows):
        for field in fields:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Missing or nonnumeric {label}.{field} row {row_index}") from error
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {label}.{field} row {row_index}")


def _finite_value(row: Mapping[str, object], field: str, label: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Missing or nonnumeric {label}.{field}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {label}.{field}")
    return value


def _require_integer(value: float, label: str) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise RuntimeError(f"Non-integral {label}")
    return int(rounded)


def _task_environment_config(
    manifest: Mapping[str, object], task: Mapping[str, object]
) -> EnvironmentConfig:
    if task["analysis"] == ANALYSIS_FOUR_WAY:
        entry = _environment_entry(manifest, int(task["environment_index"]))
    else:
        entry = _environment_entry(manifest)
    config = EnvironmentConfig(**entry["config"])
    if task["analysis"] == ANALYSIS_SIGMA:
        config = replace(config, sigma_need=float(task["condition_value"]))
    return config


def _optional_numeric_fields(row: Mapping[str, object]) -> set[str]:
    if str(row.get("policy", "")) == STRATEGY_MAPPING_POLICY_RR:
        optional = set(MANUAL_TIME_MATCHED_FIELDS)
    elif str(row.get("policy", "")) == STRATEGY_MAPPING_POLICY_MANUAL:
        optional = set(RR_TIME_MATCHED_FIELDS)
    else:
        optional = set(RR_TIME_MATCHED_FIELDS | MANUAL_TIME_MATCHED_FIELDS)
    if str(row.get("policy", "")) != STRATEGY_MAPPING_POLICY_ORACLE:
        optional.update({"oracle_grid_size", "oracle_grid_optimality_violation"})
    return optional


def _validate_numeric_schema(
    analysis: str,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    allowed = set(fields)
    for row_index, row in enumerate(rows):
        if not set(row).issubset(allowed):
            raise RuntimeError(f"{analysis} shard row schema mismatch")
        optional = _optional_numeric_fields(row) if analysis == ANALYSIS_FOUR_WAY else set()
        for field in fields:
            raw = row.get(field, "")
            if field in EPISODE_TEXT_FIELDS:
                if raw in (None, ""):
                    raise RuntimeError(f"Missing {analysis}.{field} row {row_index}")
                continue
            if raw in (None, ""):
                if field in optional:
                    continue
                raise RuntimeError(f"Missing numeric {analysis}.{field} row {row_index}")
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Nonnumeric {analysis}.{field} row {row_index}"
                ) from error
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {analysis}.{field} row {row_index}")


def _validate_clipped_regret(
    row: Mapping[str, object], raw_field: str, regret_field: str, violation_field: str
) -> None:
    raw = _finite_value(row, raw_field, "regret")
    regret = _finite_value(row, regret_field, "regret")
    violation = _finite_value(row, violation_field, "regret")
    tolerance = 1e-8 * max(1.0, abs(raw))
    if abs(regret - max(0.0, raw)) > tolerance:
        raise RuntimeError(f"Inconsistent clipped regret: {regret_field}")
    if abs(violation - max(0.0, -raw)) > tolerance:
        raise RuntimeError(f"Inconsistent optimality violation: {violation_field}")


def _validate_episode_domains(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    analysis = str(task["analysis"])
    config = _task_environment_config(manifest, task)
    expected_tags = _analysis_mode_fields(manifest, analysis)
    tolerance = 1e-8 * max(1.0, config.total_time)
    manual_samples = int(manifest["settings"]["manual_samples_per_person"])
    required_manual_total = 2 * manual_samples
    manual_feasible = (
        config.terminate_cost
        + required_manual_total * config.sample_time_cost
        <= config.total_time + tolerance
        and (
            config.max_meta_samples is None
            or config.max_meta_samples >= required_manual_total
        )
    )
    if analysis == ANALYSIS_FOUR_WAY and not manual_feasible:
        raise RuntimeError(
            "Frozen environment cannot execute the declared manual sampling benchmark"
        )

    for row_index, row in enumerate(rows):
        label = f"{analysis} row {row_index}"
        if row.get("analysis_mode") != expected_tags["analysis_mode"]:
            raise RuntimeError(f"Incorrect analysis mode in {label}")
        if row.get("scientific_status") != expected_tags["scientific_status"]:
            raise RuntimeError(f"Incorrect scientific status in {label}")

        allocation = _finite_value(row, "allocation_to_person1", label)
        if allocation < -1e-12 or allocation > 1.0 + 1e-12:
            raise RuntimeError(f"Allocation outside [0,1] in {label}")
        for field in ALLOCATION_FIELDS:
            if row.get(field, "") not in (None, ""):
                value = _finite_value(row, field, label)
                if value < -1e-12 or value > 1.0 + 1e-12:
                    raise RuntimeError(f"Allocation field outside [0,1]: {label}.{field}")

        for field in NONNEGATIVE_FIELDS:
            if row.get(field, "") not in (None, "") and _finite_value(row, field, label) < -1e-10:
                raise RuntimeError(f"Negative nonnegative field: {label}.{field}")

        binary_fields = set(COMMON_BINARY_FIELDS)
        binary_fields.update(
            {
                "oracle_true_equal_outcome",
                "oracle_closer_to_true_equal_than_equal_split",
                "ambiguous_close_true_equal_but_closer_equal_split",
                "constructed_true_state",
                "rr_time_matched_oracle_true_equal_outcome",
                "rr_time_matched_oracle_closer_to_true_equal_than_equal_split",
                "manual_time_matched_oracle_true_equal_outcome",
                "manual_time_matched_oracle_closer_to_true_equal_than_equal_split",
            }
        )
        for field in binary_fields:
            if row.get(field, "") not in (None, ""):
                value = _finite_value(row, field, label)
                if value not in (0.0, 1.0):
                    raise RuntimeError(f"Non-binary field: {label}.{field}")

        counts = {
            field: _require_integer(_finite_value(row, field, label), f"{label}.{field}")
            for field in ("online_sample_count", "sample_count_1", "sample_count_2")
        }
        if any(value < 0 for value in counts.values()):
            raise RuntimeError(f"Negative sample count in {label}")
        if counts["online_sample_count"] != counts["sample_count_1"] + counts["sample_count_2"]:
            raise RuntimeError(f"Sample-count total mismatch in {label}")
        expected_sampled_both = float(
            counts["sample_count_1"] > 0 and counts["sample_count_2"] > 0
        )
        if _finite_value(row, "sampled_both_recipients", label) != expected_sampled_both:
            raise RuntimeError(f"Sampled-both indicator mismatch in {label}")
        expected_immediate = float(counts["online_sample_count"] == 0)
        if _finite_value(row, "immediate_termination", label) != expected_immediate:
            raise RuntimeError(f"Immediate-termination indicator mismatch in {label}")

        remaining_time = _finite_value(row, "remaining_time", label)
        expected_remaining = max(
            0.0,
            config.total_time
            - config.terminate_cost
            - counts["online_sample_count"] * config.sample_time_cost,
        )
        if remaining_time < -tolerance or remaining_time > config.total_time + tolerance:
            raise RuntimeError(f"Remaining time outside environment bounds in {label}")
        if abs(remaining_time - expected_remaining) > tolerance:
            raise RuntimeError(f"Remaining-time/sample-cost mismatch in {label}")

        total_need = _finite_value(row, "total_true_need", label)
        need_1 = _finite_value(row, "need_1", label)
        need_2 = _finite_value(row, "need_2", label)
        if abs(total_need - (need_1 + need_2)) > 1e-9 * max(1.0, abs(total_need)):
            raise RuntimeError(f"Total true need mismatch in {label}")
        if abs(_finite_value(row, "realized_true_need_gap", label) - abs(need_1 - need_2)) > 1e-9:
            raise RuntimeError(f"True need gap mismatch in {label}")
        if abs(_finite_value(row, "abs_allocation_from_equal", label) - abs(allocation - 0.5)) > 1e-9:
            raise RuntimeError(f"Equal-allocation distance mismatch in {label}")

        expected_negative_1 = float(need_1 < 0.0)
        expected_negative_2 = float(need_2 < 0.0)
        expected_negative_either = float(expected_negative_1 or expected_negative_2)
        expected_negative_both = float(expected_negative_1 and expected_negative_2)
        for field, expected in (
            ("negative_need_person1", expected_negative_1),
            ("negative_need_person2", expected_negative_2),
            ("negative_need_either", expected_negative_either),
            ("negative_need_both", expected_negative_both),
        ):
            if _finite_value(row, field, label) != expected:
                raise RuntimeError(f"Negative-need indicator mismatch in {label}.{field}")

        policy = str(row["policy"])
        if analysis == ANALYSIS_FOUR_WAY:
            if policy not in STRATEGY_MAPPING_POLICY_ORDER:
                raise RuntimeError(f"Unknown policy in {label}")
            if policy in (STRATEGY_MAPPING_POLICY_EQUAL_SPLIT, STRATEGY_MAPPING_POLICY_ORACLE) and counts[
                "online_sample_count"
            ] != 0:
                raise RuntimeError(f"Zero-information benchmark sampled in {label}")
            if policy == STRATEGY_MAPPING_POLICY_EQUAL_SPLIT and abs(allocation - 0.5) > 1e-12:
                raise RuntimeError(f"Equal-split benchmark is not 50/50 in {label}")
            if policy == STRATEGY_MAPPING_POLICY_MANUAL and (
                counts["sample_count_1"] != manual_samples
                or counts["sample_count_2"] != manual_samples
            ):
                raise RuntimeError(f"Manual benchmark did not complete frozen samples in {label}")
            if policy == STRATEGY_MAPPING_POLICY_ORACLE:
                if _require_integer(
                    _finite_value(row, "oracle_grid_size", label),
                    f"{label}.oracle_grid_size",
                ) != int(manifest["settings"]["oracle_grid_size"]):
                    raise RuntimeError(f"Oracle grid size mismatch in {label}")
                if abs(
                    _finite_value(row, "allocation_to_person1", label)
                    - _finite_value(row, "initial_oracle_allocation", label)
                ) > 1e-10:
                    raise RuntimeError(f"Initial oracle allocation mismatch in {label}")
                if abs(
                    _finite_value(row, "realized_utility", label)
                    - _finite_value(row, "initial_oracle_utility", label)
                ) > 1e-8:
                    raise RuntimeError(f"Initial oracle utility mismatch in {label}")
            raw_initial = _finite_value(row, "raw_utility_regret_to_initial_oracle", label)
            clipped_initial = _finite_value(row, "utility_regret_to_initial_oracle", label)
            if abs(clipped_initial - max(0.0, raw_initial)) > 1e-8 * max(1.0, abs(raw_initial)):
                raise RuntimeError(f"Initial oracle regret mismatch in {label}")
            if policy == STRATEGY_MAPPING_POLICY_RR:
                _validate_clipped_regret(
                    row,
                    "rr_time_matched_oracle_raw_regret",
                    "rr_time_matched_oracle_regret",
                    "rr_time_matched_oracle_optimality_violation",
                )
            if policy == STRATEGY_MAPPING_POLICY_MANUAL:
                _validate_clipped_regret(
                    row,
                    "manual_time_matched_oracle_raw_regret",
                    "manual_time_matched_oracle_regret",
                    "manual_time_matched_oracle_optimality_violation",
                )
        else:
            if policy != STRATEGY_MAPPING_POLICY_RR:
                raise RuntimeError(f"Unexpected controlled-diagnostic policy in {label}")
            _validate_clipped_regret(
                row,
                "initial_oracle_raw_regret",
                "initial_oracle_regret",
                "initial_oracle_optimality_violation",
            )
            if analysis == ANALYSIS_SIGMA:
                if abs(_finite_value(row, "sigma_need", label) - float(task["condition_value"])) > 1e-12:
                    raise RuntimeError(f"Sigma condition mismatch in {label}")
            else:
                mean = _finite_value(row, "fixed_total_need_mean", label)
                difference = _finite_value(row, "constructed_need_difference", label)
                orientation = _finite_value(row, "orientation", label)
                expected_orientation = -1.0 if _require_integer(
                    _finite_value(row, "episode_index", label), f"{label}.episode_index"
                ) % 2 == 0 else 1.0
                if orientation != expected_orientation:
                    raise RuntimeError(f"Fixed-total orientation mismatch in {label}")
                if abs(difference - float(task["condition_value"])) > 1e-12:
                    raise RuntimeError(f"Fixed-total condition mismatch in {label}")
                if abs(need_1 - (mean + orientation * difference / 2.0)) > 1e-9 or abs(
                    need_2 - (mean - orientation * difference / 2.0)
                ) > 1e-9:
                    raise RuntimeError(f"Constructed fixed-total state mismatch in {label}")


def validate_episode_rows(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    analysis = str(task["analysis"])
    filename = EPISODE_OUTPUTS[analysis]
    fields = manifest["expected_output_schemas"][filename]
    _validate_numeric_schema(analysis, rows, fields)
    if len(rows) != int(task["expected_row_count"]):
        raise RuntimeError(f"{analysis} shard row-count mismatch")
    if {str(row["environment_config_hash"]) for row in rows} != {
        str(task["environment_config_hash"])
    }:
        raise RuntimeError(f"{analysis} shard environment-config hash mismatch")
    keys = [_row_key(analysis, row) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != _task_expected_keys(manifest, task):
        raise RuntimeError(f"{analysis} shard Cartesian keys mismatch")
    _validate_episode_domains(manifest, task, rows)
    _require_finite(
        rows,
        (
            "episode_index",
            "need_1",
            "need_2",
            "allocation_to_person1",
            "remaining_time",
            "realized_utility",
            "online_sample_count",
            "policy_computation_seed",
        ),
        analysis,
    )
    if analysis == ANALYSIS_FOUR_WAY:
        grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
        for row in rows:
            grouped.setdefault(
                (str(row["environment"]), int(float(row["episode_index"]))), []
            ).append(row)
        for key, episode_rows in grouped.items():
            if {str(row["policy"]) for row in episode_rows} != set(STRATEGY_MAPPING_POLICY_ORDER):
                raise RuntimeError(f"Four-way policy mismatch: {key}")
            for field in (
                "need_1",
                "need_2",
                "episode_fingerprint",
                "observation_stream_hash_1",
                "observation_stream_hash_2",
                "policy_computation_seed",
            ):
                if len({str(row[field]) for row in episode_rows}) != 1:
                    raise RuntimeError(f"Four-way common-randomness mismatch: {key} {field}")
            by_policy = {str(row["policy"]): row for row in episode_rows}
            split = by_policy["equal_split"]
            oracle = by_policy["full_information_oracle"]
            rr = by_policy["frozen_rr"]
            manual = by_policy["manual_active_search_equal_outcome"]
            if abs(float(split["allocation_to_person1"]) - 0.5) > 1e-12:
                raise RuntimeError("Equal-split benchmark is not 50/50")
            if int(float(split["online_sample_count"])) or int(float(oracle["online_sample_count"])):
                raise RuntimeError("A zero-information benchmark consumed observations")
            if float(rr["rr_time_matched_oracle_optimality_violation"]) > 1e-8:
                raise RuntimeError("RR exceeded its time-matched oracle")
            if float(manual["manual_time_matched_oracle_optimality_violation"]) > 1e-8:
                raise RuntimeError("Manual policy exceeded its time-matched oracle")
            if float(oracle["oracle_grid_optimality_violation"]) > 1e-8:
                raise RuntimeError("Initial oracle failed its optimality check")
            if any(float(row["raw_utility_regret_to_initial_oracle"]) < -1e-8 for row in episode_rows):
                raise RuntimeError("A policy exceeded the initial-budget oracle")
        return

    _require_finite(
        rows,
        (
            "standardized_need_draw_1",
            "standardized_need_draw_2",
            "allocation_closeness_advantage",
            "outcome_closeness_advantage",
            "initial_oracle_optimality_violation",
        ),
        analysis,
    )
    if any(float(row["initial_oracle_optimality_violation"]) > 1e-8 for row in rows):
        raise RuntimeError(f"{analysis} policy exceeded its numerical oracle")
    grouped_by_episode: Dict[int, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped_by_episode.setdefault(int(float(row["episode_index"])), []).append(row)
    for episode_index, episode_rows in grouped_by_episode.items():
        for field in (
            "observation_residual_hash_1",
            "observation_residual_hash_2",
            "policy_computation_seed",
        ):
            if len({str(row[field]) for row in episode_rows}) != 1:
                raise RuntimeError(f"{analysis} common-randomness mismatch: {episode_index} {field}")
        if analysis == ANALYSIS_SIGMA:
            for field in ("standardized_need_draw_1", "standardized_need_draw_2"):
                values = [float(row[field]) for row in episode_rows]
                if max(values) - min(values) > 1e-10:
                    raise RuntimeError(f"Sigma standardized-need mismatch: {episode_index} {field}")
            if len({str(row["non_sigma_config_hash"]) for row in episode_rows}) != 1:
                raise RuntimeError("A non-sigma field changed in the controlled sweep")
        else:
            totals = [float(row["need_1"]) + float(row["need_2"]) for row in episode_rows]
            if max(totals) - min(totals) > 1e-10:
                raise RuntimeError("Fixed-total diagnostic changed total need")
            if any(float(row["need_1"]) < 0.0 or float(row["need_2"]) < 0.0 for row in episode_rows):
                raise RuntimeError("Fixed-total diagnostic produced a negative need")
    if analysis == ANALYSIS_FIXED:
        for difference in manifest["settings"]["fixed_total_need_differences"]:
            orientations = [
                float(row["orientation"])
                for row in rows
                if float(row["constructed_need_difference"]) == float(difference)
            ]
            if orientations.count(-1.0) != orientations.count(1.0):
                raise RuntimeError(f"Unbalanced fixed-total orientations for d={difference}")


def validate_task_artifact(
    manifest_path: Path,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
) -> Tuple[str, List[str]]:
    rows_path, status_path, failure_path = task_paths(manifest_path, int(task["task_index"]))
    if not rows_path.is_file() or not status_path.is_file():
        return ("failed" if failure_path.is_file() else "missing"), []
    errors: List[str] = []
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "ok":
            errors.append("status is not ok")
        for field in ("task_index", "episode_start", "episode_count", "expected_row_count"):
            if int(status.get(field, -1)) != int(task[field]):
                errors.append(f"status {field} mismatch")
        if status.get("analysis") != task["analysis"]:
            errors.append("status analysis mismatch")
        if status.get("condition_label") != task["condition_label"]:
            errors.append("status condition label mismatch")
        if status.get("condition_value") != task["condition_value"]:
            errors.append("status condition value mismatch")
        if status.get("environment_config_hash") != task["environment_config_hash"]:
            errors.append("status environment-config hash mismatch")
        if status.get("scientific_command_fingerprint") != task[
            "scientific_command_fingerprint"
        ]:
            errors.append("status scientific-command fingerprint mismatch")
        if status.get("manifest_hash") != manifest["manifest_hash"]:
            errors.append("status manifest mismatch")
        if status.get("git_commit") != manifest["git_commit"]:
            errors.append("status commit mismatch")
        if sha256_file(rows_path) != status.get("rows_sha256"):
            errors.append("row hash mismatch")
        fields = manifest["expected_output_schemas"][EPISODE_OUTPUTS[str(task["analysis"])]]
        if read_csv_schema(rows_path) != fields:
            errors.append("row schema mismatch")
        if schema_hash(read_csv_schema(rows_path)) != status.get("schema_sha256"):
            errors.append("row schema hash mismatch")
        rows = read_csv(rows_path)
        validate_episode_rows(manifest, task, rows)
    except Exception as error:
        errors.append(str(error))
    return ("invalid", errors) if errors else ("complete", [])


@contextmanager
def exclusive_task_lock(task_dir: Path):
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / TASK_LOCK_FILENAME
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_task(manifest_path: Path, task_index: int) -> None:
    manifest = load_manifest(manifest_path)
    verify_execution_checkout(manifest)
    task = _task_for_index(manifest, task_index)
    rows_path, status_path, failure_path = task_paths(manifest_path, task_index)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_task_lock(rows_path.parent):
        state, _ = validate_task_artifact(manifest_path, manifest, task)
        if state == "complete":
            print(canonical_json({"task_index": task_index, "status": "already_complete"}))
            return
        status_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        try:
            rows = execute_task_rows(manifest, task)
            filename = EPISODE_OUTPUTS[str(task["analysis"])]
            fields = manifest["expected_output_schemas"][filename]
            validate_episode_rows(manifest, task, rows)
            atomic_write_csv(rows_path, rows, fields)
            status = {
                "status": "ok",
                "completed_at_utc": utc_now(),
                "task_index": task_index,
                "analysis": task["analysis"],
                "condition_label": task["condition_label"],
                "condition_value": task["condition_value"],
                "environment_config_hash": task["environment_config_hash"],
                "episode_start": task["episode_start"],
                "episode_count": task["episode_count"],
                "expected_row_count": task["expected_row_count"],
                "row_count": len(rows),
                "rows_sha256": sha256_file(rows_path),
                "schema_sha256": schema_hash(fields),
                "scientific_command_fingerprint": task["scientific_command_fingerprint"],
                "manifest_hash": manifest["manifest_hash"],
                "git_commit": manifest["git_commit"],
                "scheduler_metadata": {
                    "job_id": os.environ.get("JOB_ID", ""),
                    "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
                    "hostname": platform.node(),
                },
            }
            atomic_write_json(status_path, status)
            final_state, final_errors = validate_task_artifact(manifest_path, manifest, task)
            if final_state != "complete":
                raise RuntimeError(f"Post-write shard validation failed: {final_errors}")
            print(canonical_json(status))
        except Exception as error:
            status_path.unlink(missing_ok=True)
            atomic_write_json(
                failure_path,
                {
                    "status": "failed",
                    "failed_at_utc": utc_now(),
                    "task_index": task_index,
                    "analysis": task["analysis"],
                    "condition_label": task["condition_label"],
                    "condition_value": task["condition_value"],
                    "scientific_command_fingerprint": task[
                        "scientific_command_fingerprint"
                    ],
                    "manifest_hash": manifest["manifest_hash"],
                    "git_commit": manifest["git_commit"],
                    "error": str(error),
                },
            )
            raise


def progress_payload(manifest_path: Path, manifest: Mapping[str, object]) -> Dict[str, object]:
    complete: List[int] = []
    missing: List[int] = []
    failed: List[int] = []
    invalid: Dict[str, List[str]] = {}
    by_analysis = {
        analysis: {"total": 0, "complete": 0, "failed": 0, "invalid": 0}
        for analysis in ANALYSES
    }
    for task in manifest["tasks"]:
        index = int(task["task_index"])
        analysis = str(task["analysis"])
        by_analysis[analysis]["total"] += 1
        state, errors = validate_task_artifact(manifest_path, manifest, task)
        if state == "complete":
            complete.append(index)
            by_analysis[analysis]["complete"] += 1
        elif state == "missing":
            missing.append(index)
        elif state == "failed":
            failed.append(index)
            by_analysis[analysis]["failed"] += 1
        else:
            invalid[str(index)] = errors
            by_analysis[analysis]["invalid"] += 1
    scheduler_path = manifest_path.parent / SCHEDULER_FILENAME
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.is_file() else {}
    total = len(manifest["tasks"])
    return {
        "checked_at_utc": utc_now(),
        "manifest_hash": manifest["manifest_hash"],
        "git_commit": manifest["git_commit"],
        "array_job_id": scheduler.get("array_job_id", ""),
        "collector_job_id": scheduler.get("collector_job_id", ""),
        "total_tasks": total,
        "completed_tasks": len(complete),
        "missing_tasks": missing,
        "failed_tasks": failed,
        "invalid_tasks": invalid,
        "percent_complete": 0.0 if not total else 100.0 * len(complete) / total,
        "by_analysis": by_analysis,
        "ready_to_collect": len(complete) == total and not failed and not invalid,
    }


def _sort_episode_rows(analysis: str, rows: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    policy_order = {policy: index for index, policy in enumerate(STRATEGY_MAPPING_POLICY_ORDER)}
    if analysis == ANALYSIS_FOUR_WAY:
        return sorted(
            rows,
            key=lambda row: (
                str(row["environment"]),
                int(float(row["episode_index"])),
                policy_order[str(row["policy"])],
            ),
        )
    if analysis == ANALYSIS_SIGMA:
        return sorted(rows, key=lambda row: (float(row["sigma_need"]), int(float(row["episode_index"]))))
    return sorted(
        rows,
        key=lambda row: (
            float(row["constructed_need_difference"]),
            int(float(row["episode_index"])),
        ),
    )


def _validate_complete_episode_table(
    manifest: Mapping[str, object], analysis: str, rows: Sequence[Mapping[str, object]]
) -> None:
    relevant_tasks = [task for task in manifest["tasks"] if task["analysis"] == analysis]
    expected_keys: set[Tuple[object, ...]] = set()
    for task in relevant_tasks:
        expected_keys.update(_task_expected_keys(manifest, task))
    actual_keys = [_row_key(analysis, row) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise RuntimeError(f"Complete {analysis} Cartesian keys mismatch")
    for task in relevant_tasks:
        start = int(task["episode_start"])
        stop = start + int(task["episode_count"])
        if analysis == ANALYSIS_FOUR_WAY:
            environment = str(_environment_entry(manifest, int(task["environment_index"]))["environment"])
            selected = [
                row
                for row in rows
                if row["environment"] == environment and start <= int(float(row["episode_index"])) < stop
            ]
        elif analysis == ANALYSIS_SIGMA:
            sigma = float(task["condition_value"])
            selected = [
                row
                for row in rows
                if float(row["sigma_need"]) == sigma
                and start <= int(float(row["episode_index"])) < stop
            ]
        else:
            difference = float(task["condition_value"])
            selected = [
                row
                for row in rows
                if float(row["constructed_need_difference"]) == difference
                and start <= int(float(row["episode_index"])) < stop
            ]
        validate_episode_rows(manifest, task, selected)


def validate_global_common_randomness(
    manifest: Mapping[str, object],
    by_analysis: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    sigma_rows = by_analysis[ANALYSIS_SIGMA]
    sigma_values = {float(value) for value in manifest["settings"]["sigma_need_values"]}
    sigma_by_episode: Dict[int, List[Mapping[str, object]]] = {}
    for row in sigma_rows:
        sigma_by_episode.setdefault(int(float(row["episode_index"])), []).append(row)
    for episode_index, rows in sigma_by_episode.items():
        if {float(row["sigma_need"]) for row in rows} != sigma_values:
            raise RuntimeError(f"Global sigma conditions mismatch for episode {episode_index}")
        for field in ("standardized_need_draw_1", "standardized_need_draw_2"):
            values = [float(row[field]) for row in rows]
            if max(values) - min(values) > 1e-10:
                raise RuntimeError(f"Global sigma standardized-need mismatch: {episode_index} {field}")
        for field in (
            "observation_residual_hash_1",
            "observation_residual_hash_2",
            "policy_computation_seed",
            "non_sigma_config_hash",
        ):
            if len({str(row[field]) for row in rows}) != 1:
                raise RuntimeError(f"Global sigma common-randomness mismatch: {episode_index} {field}")

    fixed_rows = by_analysis[ANALYSIS_FIXED]
    fixed_values = {
        float(value) for value in manifest["settings"]["fixed_total_need_differences"]
    }
    fixed_by_episode: Dict[int, List[Mapping[str, object]]] = {}
    for row in fixed_rows:
        fixed_by_episode.setdefault(int(float(row["episode_index"])), []).append(row)
    for episode_index, rows in fixed_by_episode.items():
        if {float(row["constructed_need_difference"]) for row in rows} != fixed_values:
            raise RuntimeError(f"Global fixed-total conditions mismatch for episode {episode_index}")
        for field in (
            "observation_residual_hash_1",
            "observation_residual_hash_2",
            "policy_computation_seed",
            "observation_seed",
            "orientation",
        ):
            if len({str(row[field]) for row in rows}) != 1:
                raise RuntimeError(f"Global fixed-total common-randomness mismatch: {episode_index} {field}")
        if any(abs(float(row["need_1"]) + float(row["need_2"]) - 70.0) > 1e-10 for row in rows):
            raise RuntimeError("Global fixed-total rows do not preserve total need 70")
    for difference in fixed_values:
        orientations = [
            float(row["orientation"])
            for row in fixed_rows
            if float(row["constructed_need_difference"]) == difference
        ]
        if orientations.count(-1.0) != orientations.count(1.0):
            raise RuntimeError(f"Global fixed-total orientations are unbalanced for d={difference}")

    primary_environment = str(_environment_entry(manifest)["environment"])
    four_rr = {
        int(float(row["episode_index"])): row
        for row in by_analysis[ANALYSIS_FOUR_WAY]
        if row["environment"] == primary_environment and row["policy"] == "frozen_rr"
    }
    sigma_anchor = {
        int(float(row["episode_index"])): row
        for row in sigma_rows
        if float(row["sigma_need"]) == 60.0
    }
    if set(four_rr) != set(sigma_anchor):
        raise RuntimeError("Primary four-way and sigma=60 episode indices do not correspond")
    shared_fields = (
        "environment_config_hash",
        "need_1",
        "need_2",
        "total_true_need",
        "realized_true_need_gap",
        "episode_fingerprint",
        "policy_computation_seed",
        "observation_stream_hash_1",
        "observation_stream_hash_2",
        "allocation_to_person1",
        "remaining_time",
        "realized_utility",
        "online_sample_count",
        "sample_count_1",
        "sample_count_2",
        "realized_outcome_gap",
        "true_equal_outcome_allocation",
        "true_equal_outcome",
        "closer_to_true_equal_outcome_than_equal_split",
    )
    for episode_index in sorted(four_rr):
        for field in shared_fields:
            if str(four_rr[episode_index][field]) != str(sigma_anchor[episode_index][field]):
                raise RuntimeError(
                    f"Primary four-way/sigma=60 mismatch: episode {episode_index} {field}"
                )


@contextmanager
def exclusive_collection_lock(output_dir: Path):
    lock_path = output_dir / COLLECTION_LOCK_FILENAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another StrategyMapping collector holds {lock_path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()} created={utc_now()}\n".encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _tag_summary_rows(
    rows: Sequence[Dict[str, object]], manifest: Mapping[str, object], analysis: str
) -> None:
    tags = _analysis_mode_fields(manifest, analysis)
    for row in rows:
        row.update(tags)


def _normalize_missing_estimates(rows: Sequence[Dict[str, object]]) -> None:
    for row in rows:
        for field, value in list(row.items()):
            if isinstance(value, float) and not math.isfinite(value):
                row[field] = ""


def _validate_summary_rows(filename: str, rows: Sequence[Mapping[str, object]]) -> None:
    for row_index, row in enumerate(rows):
        count_field = "n_pairs" if filename == "strategy_mapping_four_way_paired_comparisons.csv" else "n_episodes"
        if count_field not in row:
            continue
        count = int(float(row[count_field]))
        if count <= 0:
            continue
        required = (
            ("mean_paired_utility_difference",)
            if count_field == "n_pairs"
            else ("mean_utility",)
        )
        for field in required:
            if row.get(field, "") in (None, ""):
                continue
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Non-finite {filename}.{field} row {row_index}")


def _promote_result_directory(staging_dir: Path, result_dir: Path) -> Path:
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    staging_dir.replace(result_dir)
    return result_dir


def _validate_result_directory(
    manifest_path: Path,
    manifest: Mapping[str, object],
    result_dir: Path,
) -> Dict[str, object]:
    artifact_path = result_dir / ARTIFACT_INDEX_FILENAME
    validation_path = result_dir / VALIDATION_FILENAME
    if not result_dir.is_dir() or not artifact_path.is_file() or not validation_path.is_file():
        raise RuntimeError("StrategyMapping promoted result directory is incomplete")
    artifact_index = json.loads(artifact_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if artifact_index.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("StrategyMapping artifact index has a stale manifest fingerprint")
    if validation.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("StrategyMapping validation has a stale manifest fingerprint")
    if artifact_index.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("StrategyMapping artifact index has a stale manifest hash")
    if validation.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("StrategyMapping validation has a stale manifest hash")
    if validation.get("artifact_index_sha256") != sha256_file(artifact_path):
        raise RuntimeError("StrategyMapping validation has a stale artifact-index hash")
    if validation.get("output_metadata") != artifact_index.get("outputs"):
        raise RuntimeError("StrategyMapping validation and artifact index disagree")

    tables: Dict[str, List[Dict[str, str]]] = {}
    for filename in ALL_OUTPUTS:
        path = result_dir / filename
        metadata_row = artifact_index.get("outputs", {}).get(filename)
        if not path.is_file() or metadata_row is None:
            raise RuntimeError(f"Missing promoted StrategyMapping output: {filename}")
        if sha256_file(path) != metadata_row["sha256"]:
            raise RuntimeError(f"Promoted StrategyMapping output hash mismatch: {filename}")
        fields = read_csv_schema(path)
        if fields != manifest["expected_output_schemas"][filename]:
            raise RuntimeError(f"Promoted StrategyMapping output schema mismatch: {filename}")
        if schema_hash(fields) != metadata_row["schema_sha256"]:
            raise RuntimeError(f"Promoted StrategyMapping output schema hash mismatch: {filename}")
        rows = read_csv(path)
        if len(rows) != int(manifest["expected_row_counts"][filename]):
            raise RuntimeError(f"Promoted StrategyMapping output row-count mismatch: {filename}")
        _validate_summary_rows(filename, rows)
        tables[filename] = rows

    by_analysis = {
        analysis: tables[filename] for analysis, filename in EPISODE_OUTPUTS.items()
    }
    for analysis, rows in by_analysis.items():
        _validate_complete_episode_table(manifest, analysis, rows)
    validate_global_common_randomness(manifest, by_analysis)
    return {
        "output_metadata": artifact_index["outputs"],
        "validation_sha256": sha256_file(validation_path),
        "artifact_index_sha256": sha256_file(artifact_path),
    }


def _collect_unlocked(manifest_path: Path) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    verify_execution_checkout(manifest)
    output_dir = manifest_path.parent
    completion_path = output_dir / COMPLETION_FILENAME
    completion_path.unlink(missing_ok=True)

    progress = progress_payload(manifest_path, manifest)
    atomic_write_json(output_dir / PROGRESS_FILENAME, progress)
    if not progress["ready_to_collect"]:
        raise RuntimeError(
            "Strict StrategyMapping collection refused: "
            f"missing={len(progress['missing_tasks'])}; failed={progress['failed_tasks']}; "
            f"invalid={progress['invalid_tasks']}"
        )

    by_analysis: Dict[str, List[Mapping[str, object]]] = {analysis: [] for analysis in ANALYSES}
    for task in manifest["tasks"]:
        rows_path, _, _ = task_paths(manifest_path, int(task["task_index"]))
        by_analysis[str(task["analysis"])].extend(read_csv(rows_path))
    for analysis in ANALYSES:
        by_analysis[analysis] = _sort_episode_rows(analysis, by_analysis[analysis])
        _validate_complete_episode_table(manifest, analysis, by_analysis[analysis])
    validate_global_common_randomness(manifest, by_analysis)

    four_policy, four_paired = summarize_strategy_mapping_four_way(by_analysis[ANALYSIS_FOUR_WAY])
    sigma_environment, sigma_strata = summarize_strategy_mapping_sigma_need_sweep(by_analysis[ANALYSIS_SIGMA])
    fixed_summary = summarize_strategy_mapping_fixed_total_need_diagnostic(by_analysis[ANALYSIS_FIXED])
    _tag_summary_rows(four_policy, manifest, ANALYSIS_FOUR_WAY)
    _tag_summary_rows(four_paired, manifest, ANALYSIS_FOUR_WAY)
    _tag_summary_rows(sigma_environment, manifest, ANALYSIS_SIGMA)
    _tag_summary_rows(sigma_strata, manifest, ANALYSIS_SIGMA)
    _tag_summary_rows(fixed_summary, manifest, ANALYSIS_FIXED)
    for summary_rows in (
        four_policy,
        four_paired,
        sigma_environment,
        sigma_strata,
        fixed_summary,
    ):
        _normalize_missing_estimates(summary_rows)

    tables: Dict[str, Sequence[Mapping[str, object]]] = {
        "strategy_mapping_four_way_episodes.csv": by_analysis[ANALYSIS_FOUR_WAY],
        "strategy_mapping_four_way_policy_summary.csv": four_policy,
        "strategy_mapping_four_way_paired_comparisons.csv": four_paired,
        "strategy_mapping_sigma_need_episodes.csv": by_analysis[ANALYSIS_SIGMA],
        "strategy_mapping_sigma_need_environment_summary.csv": sigma_environment,
        "strategy_mapping_sigma_need_gap_strata.csv": sigma_strata,
        "strategy_mapping_fixed_total_need_episodes.csv": by_analysis[ANALYSIS_FIXED],
        "strategy_mapping_fixed_total_need_summary.csv": fixed_summary,
    }
    staging_dir = output_dir / f".strategy_mapping_collection_staging_{str(manifest['manifest_hash'])[:12]}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for filename, rows in tables.items():
        expected_count = int(manifest["expected_row_counts"][filename])
        if len(rows) != expected_count:
            raise RuntimeError(f"Final row-count mismatch for {filename}: {len(rows)} != {expected_count}")
        fields = manifest["expected_output_schemas"][filename]
        atomic_write_csv(staging_dir / filename, rows, fields)
        _validate_summary_rows(filename, rows)

    output_metadata: Dict[str, Dict[str, object]] = {}
    for filename in ALL_OUTPUTS:
        path = staging_dir / filename
        fields = read_csv_schema(path)
        rows = read_csv(path)
        if fields != manifest["expected_output_schemas"][filename]:
            raise RuntimeError(f"Read-back schema mismatch: {filename}")
        if schema_hash(fields) != manifest["expected_output_schema_hashes"][filename]:
            raise RuntimeError(f"Read-back schema-hash mismatch: {filename}")
        if len(rows) != int(manifest["expected_row_counts"][filename]):
            raise RuntimeError(f"Read-back row-count mismatch: {filename}")
        output_metadata[filename] = {
            "sha256": sha256_file(path),
            "schema_sha256": schema_hash(fields),
            "row_count": len(rows),
        }

    validation = {
        "schema_version": 1,
        "status": (
            "passed_strict_serious_validation"
            if manifest["analysis_mode"] == "serious"
            else "passed_strict_smoke_validation"
            if manifest["analysis_mode"] == "smoke"
            else "passed_strict_test_validation"
        ),
        "scientific_completion": manifest["analysis_mode"] == "serious",
        "validated_at_utc": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "git_commit": manifest["git_commit"],
        "task_count": len(manifest["tasks"]),
        "output_metadata": output_metadata,
        "checks": {
            "all_shards_complete_and_valid": True,
            "exact_cartesian_keys": True,
            "common_randomness": True,
            "primary_four_way_sigma_anchor_correspondence": True,
            "frozen_scientific_settings": True,
            "oracle_dominance": True,
            "schemas_and_row_counts": True,
            "source_and_manifest_hashes": True,
        },
    }
    artifact_index = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "manifest_hash": manifest["manifest_hash"],
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": manifest["git_commit"],
        "outputs": output_metadata,
    }
    atomic_write_json(staging_dir / ARTIFACT_INDEX_FILENAME, artifact_index)
    validation["artifact_index_sha256"] = sha256_file(
        staging_dir / ARTIFACT_INDEX_FILENAME
    )
    atomic_write_json(staging_dir / VALIDATION_FILENAME, validation)
    result_version = (
        f"strategy_mapping_result_{str(manifest['manifest_hash'])[:12]}_"
        f"{digest(output_metadata)[:16]}"
    )
    result_relative = Path(RESULT_VERSIONS_DIRNAME) / result_version
    result_dir = _promote_result_directory(
        staging_dir,
        output_dir / result_relative,
    )
    promoted = _validate_result_directory(manifest_path, manifest, result_dir)
    completion = {
        "schema_version": 1,
        "status": (
            "strict_serious_validation_complete"
            if manifest["analysis_mode"] == "serious"
            else "strict_smoke_validation_complete"
            if manifest["analysis_mode"] == "smoke"
            else "strict_test_validation_complete"
        ),
        "scientific_completion": manifest["analysis_mode"] == "serious",
        "completed_at_utc": utc_now(),
        "result_directory": result_relative.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "validation_sha256": promoted["validation_sha256"],
        "artifact_index_sha256": promoted["artifact_index_sha256"],
        "output_metadata": promoted["output_metadata"],
        "git_commit": manifest["git_commit"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
        },
    }
    atomic_write_json(completion_path, completion)
    return completion


def collect(manifest_path: Path) -> Dict[str, object]:
    output_dir = manifest_path.resolve().parent
    completion_path = output_dir / COMPLETION_FILENAME
    with exclusive_collection_lock(output_dir):
        try:
            completion = _collect_unlocked(manifest_path.resolve())
            validate_final_tree(manifest_path.resolve())
            return completion
        except Exception:
            completion_path.unlink(missing_ok=True)
            raise


def validate_final_tree(manifest_path: Path) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    verify_execution_checkout(manifest)
    output_dir = manifest_path.parent
    completion_path = output_dir / COMPLETION_FILENAME
    if not completion_path.is_file():
        raise RuntimeError("StrategyMapping final tree is missing its completion marker")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    relative_result = Path(str(completion.get("result_directory", "")))
    versions_root = (output_dir / RESULT_VERSIONS_DIRNAME).resolve()
    result_dir = (output_dir / relative_result).resolve()
    if (
        not relative_result.parts
        or relative_result.is_absolute()
        or ".." in relative_result.parts
        or result_dir.parent != versions_root
    ):
        raise RuntimeError("StrategyMapping completion points outside the result-version directory")
    validated = _validate_result_directory(manifest_path, manifest, result_dir)
    validation_path = result_dir / VALIDATION_FILENAME
    artifact_path = result_dir / ARTIFACT_INDEX_FILENAME
    expected_scientific = manifest["analysis_mode"] == "serious"
    expected_status = (
        "strict_serious_validation_complete"
        if expected_scientific
        else "strict_smoke_validation_complete"
        if manifest["analysis_mode"] == "smoke"
        else "strict_test_validation_complete"
    )
    if completion.get("status") != expected_status:
        raise RuntimeError("StrategyMapping completion status does not match manifest mode")
    if bool(completion.get("scientific_completion")) != expected_scientific:
        raise RuntimeError("StrategyMapping completion scientific flag does not match manifest mode")
    if completion.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("StrategyMapping completion has a stale manifest hash")
    if completion.get("validation_sha256") != sha256_file(validation_path):
        raise RuntimeError("StrategyMapping completion has a stale validation hash")
    if completion.get("artifact_index_sha256") != sha256_file(artifact_path):
        raise RuntimeError("StrategyMapping completion has a stale artifact-index hash")
    if completion.get("output_metadata") != validated["output_metadata"]:
        raise RuntimeError("StrategyMapping completion output metadata mismatch")
    return {
        "status": "valid",
        "scientific_completion": expected_scientific,
        "manifest_hash": manifest["manifest_hash"],
        "output_count": len(ALL_OUTPUTS),
    }


def create_manifest_from_args(args: argparse.Namespace) -> Path:
    analysis_mode = str(args.analysis_mode)
    seed_namespace_offset = (
        int(args.seed_namespace_offset)
        if args.seed_namespace_offset is not None
        else SERIOUS_SEED_NAMESPACE_OFFSET
        if analysis_mode == "serious"
        else SMOKE_SEED_NAMESPACE_OFFSET
    )
    if not git_is_clean():
        raise RuntimeError("StrategyMapping manifest creation requires a clean committed worktree")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"StrategyMapping output directory is not empty: {output_dir}")
    selected_rows, selected_configs, diagnostic_active_search_hashes = _load_frozen_inputs(
        args.diagnostic_active_search_dir.resolve(),
        args.frozen_config.resolve(),
        seed_namespace_offset,
    )
    manifest = build_manifest(
        output_dir=output_dir,
        selected_rows=selected_rows,
        selected_configs=selected_configs,
        diagnostic_active_search_dir=args.diagnostic_active_search_dir.resolve(),
        diagnostic_active_search_hashes=diagnostic_active_search_hashes,
        frozen_config_path=args.frozen_config.resolve(),
        schema_contract_path=args.schema_contract.resolve(),
        episodes=args.episodes,
        episodes_per_task=args.episodes_per_task,
        observation_draws=args.observation_draws,
        oracle_grid_size=args.oracle_grid_size,
        observations_per_person=args.observations_per_person,
        sigma_need_values=args.sigma_need_values,
        seed_namespace_offset=seed_namespace_offset,
        analysis_mode=analysis_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    atomic_write_json(manifest_path, manifest)
    progress = progress_payload(manifest_path, manifest)
    atomic_write_json(output_dir / PROGRESS_FILENAME, progress)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible StrategyMapping Hoffman2 array workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--analysis-mode", choices=("serious", "smoke"), default="serious")
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument(
        "--diagnostic_active_search-dir",
        type=Path,
        default=Path("results/active_search_benchmark_server_20260714_array486"),
    )
    create.add_argument(
        "--frozen-config", type=Path, default=Path("configs/strategy_mapping_environments.json")
    )
    create.add_argument(
        "--schema-contract", type=Path, default=Path("configs/strategy_mapping_output_schemas.json")
    )
    create.add_argument("--episodes", type=int, default=SERIOUS_EPISODES)
    create.add_argument("--episodes-per-task", type=int, default=SERIOUS_EPISODES_PER_TASK)
    create.add_argument("--observation-draws", type=int, default=SERIOUS_OBSERVATION_DRAWS)
    create.add_argument("--oracle-grid-size", type=int, default=4001)
    create.add_argument("--observations-per-person", type=int, default=500)
    create.add_argument("--seed-namespace-offset", type=int)
    create.add_argument(
        "--sigma-need-values",
        type=_parse_float_list,
        default=list(SERIOUS_SIGMA_NEED_VALUES),
    )

    run = subparsers.add_parser("run-task")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)

    progress = subparsers.add_parser("progress")
    progress.add_argument("--manifest", type=Path, required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--manifest", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)

    jobs = subparsers.add_parser("record-jobs")
    jobs.add_argument("--manifest", type=Path, required=True)
    jobs.add_argument("--array-job-id", required=True)
    jobs.add_argument("--collector-job-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "create":
        manifest_path = create_manifest_from_args(args)
        manifest = load_manifest(manifest_path)
        print(canonical_json({"manifest": str(manifest_path), "tasks": len(manifest["tasks"])}))
        return 0
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if args.command == "run-task":
        run_task(manifest_path, args.task_index)
        return 0
    if args.command == "progress":
        payload = progress_payload(manifest_path, manifest)
        atomic_write_json(manifest_path.parent / PROGRESS_FILENAME, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if not payload["failed_tasks"] and not payload["invalid_tasks"] else 1
    if args.command == "collect":
        completion = collect(manifest_path)
        print(json.dumps(completion, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        result = validate_final_tree(manifest_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "record-jobs":
        atomic_write_json(
            manifest_path.parent / SCHEDULER_FILENAME,
            {
                "recorded_at_utc": utc_now(),
                "array_job_id": str(args.array_job_id),
                "collector_job_id": str(args.collector_job_id),
                "manifest_hash": manifest["manifest_hash"],
            },
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
