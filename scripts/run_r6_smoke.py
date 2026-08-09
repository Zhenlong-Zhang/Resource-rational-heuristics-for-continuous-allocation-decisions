#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import platform
from datetime import datetime, timezone
from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.r6 import (
    R6_DEFAULT_GAP_BIN_EDGES,
    R6_DEFAULT_TOTAL_NEED_BIN_EDGES,
    R6_FIXED_TOTAL_NEED_DIFFERENCES,
    R6_FIXED_TOTAL_NEED_MEAN,
    evaluate_r6_fixed_total_need_diagnostic,
    evaluate_r6_four_way_environment,
    evaluate_r6_sigma_need_sweep,
    select_r6_primary_environments,
    summarize_r6_fixed_total_need_diagnostic,
    summarize_r6_four_way,
    summarize_r6_sigma_need_sweep,
)
from src.experiments.randomization import build_evaluation_episode
from src.mdp.meta_mdp import EnvironmentConfig
from src.policies.voi import MyopicValueOfInformationPolicy


R4_EXPECTED_HASHES = {
    "r4_diagnostic_environment_summary.csv": "84bb0e5efede45b4453cdfffca037a299a3dc9e212807a236579df5030f3f3a0",
    "r4_diagnostic_manual_advantage_candidates.csv": "c668d43b7538481c88983ea42dde3067e5a96182f76d0e689616524df4810ddd",
    "r4_array_manifest.json": "7e4e86c40e0656f7e42ee168ccf59587e61d7a56f6ad77ba9810a333c4949aeb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_finite(rows: Sequence[Mapping[str, str]], fields: Sequence[str], table: str) -> None:
    for row_index, row in enumerate(rows):
        for field in fields:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Missing or nonnumeric {table}.{field} row {row_index}") from error
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {table}.{field} row {row_index}")


def validate_r6_artifacts(output_dir: Path) -> dict[str, object]:
    """Strictly reload and validate a completed smoke artifact set."""

    completion_path = output_dir / "COMPLETED.json"
    previous_completion = completion_path.read_bytes() if completion_path.exists() else None
    completion_path.unlink(missing_ok=True)
    manifest_path = output_dir / "r6_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing r6_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("analysis_mode") != "smoke":
        raise RuntimeError("The smoke validator received a non-smoke manifest")
    if manifest.get("git_commit") != _git_commit():
        raise RuntimeError("Manifest commit does not match the current checkout")
    for relative_path, expected_hash in manifest["implementation_source_hashes"].items():
        path = PROJECT_ROOT / relative_path
        if not path.exists() or _sha256(path) != expected_hash:
            raise RuntimeError(f"Implementation source changed after manifest creation: {relative_path}")

    tables: dict[str, list[dict[str, str]]] = {}
    for filename, expected in manifest["expected_outputs"].items():
        path = output_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing required R6 output: {filename}")
        if _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Output hash mismatch: {filename}")
        rows = _read_csv(path)
        tables[filename] = rows
        if len(rows) != int(expected["row_count"]):
            raise RuntimeError(f"Output row-count mismatch: {filename}")
        schema = list(rows[0]) if rows else []
        if schema != list(expected["schema"]):
            raise RuntimeError(f"Output schema mismatch: {filename}")
        schema_hash = hashlib.sha256(
            json.dumps(schema, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if schema_hash != expected["schema_sha256"]:
            raise RuntimeError(f"Output schema-hash mismatch: {filename}")
        expected_count = int(manifest["expected_row_counts"][filename])
        if len(rows) != expected_count:
            raise RuntimeError(f"Frozen expected row-count mismatch: {filename}")

    four_way = tables["r6_four_way_episodes.csv"]
    _require_finite(
        four_way,
        (
            "episode_index",
            "need_1",
            "need_2",
            "allocation_to_person1",
            "remaining_time",
            "realized_utility",
            "online_sample_count",
            "policy_computation_seed",
            "true_equal_outcome",
            "closer_to_true_equal_outcome_than_equal_split",
        ),
        "r6_four_way_episodes",
    )
    four_keys = [(row["environment"], int(float(row["episode_index"])), row["policy"]) for row in four_way]
    if len(four_keys) != len(set(four_keys)):
        raise RuntimeError("Duplicate four-way episode key")
    grouped_four: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in four_way:
        grouped_four.setdefault((row["environment"], int(float(row["episode_index"]))), []).append(row)
    expected_policies = {"frozen_rr", "manual_active_search_equal_outcome", "equal_split", "full_information_oracle"}
    selected_environments = [row["environment"] for row in manifest["selected_environments"]]
    episode_indices = range(
        int(manifest["settings"]["episode_index_range"][0]),
        int(manifest["settings"]["episode_index_range"][1]) + 1,
    )
    expected_four_keys = {
        (environment, episode_index, policy)
        for environment in selected_environments
        for episode_index in episode_indices
        for policy in expected_policies
    }
    if set(four_keys) != expected_four_keys:
        raise RuntimeError("Missing or extra four-way episode keys")
    for key, rows in grouped_four.items():
        if {row["policy"] for row in rows} != expected_policies or len(rows) != 4:
            raise RuntimeError(f"Wrong policy set for four-way episode {key}")
        for field in (
            "need_1",
            "need_2",
            "episode_fingerprint",
            "observation_stream_hash_1",
            "observation_stream_hash_2",
            "policy_computation_seed",
        ):
            if len({row[field] for row in rows}) != 1:
                raise RuntimeError(f"Four-way common-randomness mismatch for {key}: {field}")
        split = next(row for row in rows if row["policy"] == "equal_split")
        oracle = next(row for row in rows if row["policy"] == "full_information_oracle")
        if abs(float(split["allocation_to_person1"]) - 0.5) > 1e-12:
            raise RuntimeError("Equal-split benchmark did not allocate 50/50")
        if int(float(split["online_sample_count"])) != 0 or int(float(oracle["online_sample_count"])) != 0:
            raise RuntimeError("Zero-information benchmark consumed observations")
        rr = next(row for row in rows if row["policy"] == "frozen_rr")
        manual = next(row for row in rows if row["policy"] == "manual_active_search_equal_outcome")
        oracle_fields = [
            float(rr["rr_time_matched_oracle_raw_regret"]),
            float(rr["rr_time_matched_oracle_optimality_violation"]),
            float(manual["manual_time_matched_oracle_raw_regret"]),
            float(manual["manual_time_matched_oracle_optimality_violation"]),
            float(oracle["oracle_grid_optimality_violation"]),
            *[float(row["raw_utility_regret_to_initial_oracle"]) for row in rows],
        ]
        if not all(math.isfinite(value) for value in oracle_fields):
            raise RuntimeError("Non-finite four-way oracle diagnostic")
        if float(rr["rr_time_matched_oracle_optimality_violation"]) > 1e-8:
            raise RuntimeError("RR exceeded its numerical time-matched oracle")
        if float(manual["manual_time_matched_oracle_optimality_violation"]) > 1e-8:
            raise RuntimeError("Manual policy exceeded its numerical time-matched oracle")
        if float(oracle["oracle_grid_optimality_violation"]) > 1e-8:
            raise RuntimeError("Initial oracle failed its benchmark optimality check")
        if any(float(row["raw_utility_regret_to_initial_oracle"]) < -1e-8 for row in rows):
            raise RuntimeError("A policy exceeded the numerical initial-budget oracle")

    sigma_rows = tables["r6_sigma_need_episodes.csv"]
    _require_finite(
        sigma_rows,
        (
            "episode_index",
            "sigma_need",
            "need_1",
            "need_2",
            "standardized_need_draw_1",
            "standardized_need_draw_2",
            "allocation_closeness_advantage",
            "outcome_closeness_advantage",
            "oracle_utility",
            "oracle_grid_optimality_violation",
            "initial_oracle_raw_regret",
            "initial_oracle_optimality_violation",
        ),
        "r6_sigma_need_episodes",
    )
    sigma_keys = [(row["environment"], int(float(row["episode_index"]))) for row in sigma_rows]
    if len(sigma_keys) != len(set(sigma_keys)):
        raise RuntimeError("Duplicate sigma-sweep episode key")
    sigma_by_episode: dict[int, list[dict[str, str]]] = {}
    for row in sigma_rows:
        sigma_by_episode.setdefault(int(float(row["episode_index"])), []).append(row)
    for episode_index, rows in sigma_by_episode.items():
        for field in ("standardized_need_draw_1", "standardized_need_draw_2"):
            values = [float(row[field]) for row in rows]
            if max(values) - min(values) > 1e-10:
                raise RuntimeError(f"Standardized need mismatch for sigma episode {episode_index}")
        for field in (
            "observation_residual_hash_1",
            "observation_residual_hash_2",
            "non_sigma_config_hash",
            "policy_computation_seed",
        ):
            if len({row[field] for row in rows}) != 1:
                raise RuntimeError(f"Controlled-sweep mismatch for episode {episode_index}: {field}")
    primary_environment = selected_environments[0]
    expected_sigma_keys = {
        (f"{primary_environment}__sigma_need={float(sigma):g}", episode_index)
        for sigma in manifest["settings"]["sigma_need_values"]
        for episode_index in episode_indices
    }
    if set(sigma_keys) != expected_sigma_keys:
        raise RuntimeError("Missing or extra controlled-sigma episode keys")
    if any(float(row["initial_oracle_optimality_violation"]) > 1e-8 for row in sigma_rows):
        raise RuntimeError("A controlled-sigma policy exceeded its numerical oracle")

    fixed_rows = tables["r6_fixed_total_need_episodes.csv"]
    _require_finite(
        fixed_rows,
        (
            "episode_index",
            "constructed_need_difference",
            "need_1",
            "need_2",
            "orientation",
            "allocation_closeness_advantage",
            "outcome_closeness_advantage",
            "oracle_utility",
            "oracle_grid_optimality_violation",
            "initial_oracle_raw_regret",
            "initial_oracle_optimality_violation",
        ),
        "r6_fixed_total_need_episodes",
    )
    fixed_keys = [
        (float(row["constructed_need_difference"]), int(float(row["episode_index"])))
        for row in fixed_rows
    ]
    if len(fixed_keys) != len(set(fixed_keys)):
        raise RuntimeError("Duplicate fixed-total episode key")
    fixed_episode_indices = range(
        int(manifest["settings"]["fixed_total_episode_index_range"][0]),
        int(manifest["settings"]["fixed_total_episode_index_range"][1]) + 1,
    )
    expected_fixed_keys = {
        (float(difference), episode_index)
        for difference in manifest["settings"]["fixed_total_need_differences"]
        for episode_index in fixed_episode_indices
    }
    if set(fixed_keys) != expected_fixed_keys:
        raise RuntimeError("Missing or extra fixed-total episode keys")
    fixed_by_episode: dict[int, list[dict[str, str]]] = {}
    orientations: dict[float, list[float]] = {}
    expected_total_need = 2.0 * float(manifest["settings"]["fixed_total_need_mean"])
    for row in fixed_rows:
        if abs(float(row["need_1"]) + float(row["need_2"]) - expected_total_need) > 1e-10:
            raise RuntimeError("Fixed-total state does not preserve its frozen total need")
        if float(row["need_1"]) < 0.0 or float(row["need_2"]) < 0.0:
            raise RuntimeError("Fixed-total diagnostic produced a negative need")
        fixed_by_episode.setdefault(int(float(row["episode_index"])), []).append(row)
        orientations.setdefault(float(row["constructed_need_difference"]), []).append(float(row["orientation"]))
    for difference, values in orientations.items():
        if values.count(-1.0) != values.count(1.0):
            raise RuntimeError(f"Unbalanced fixed-total orientations for d={difference}")
    for episode_index, rows in fixed_by_episode.items():
        for field in (
            "observation_residual_hash_1",
            "observation_residual_hash_2",
            "policy_computation_seed",
        ):
            if len({row[field] for row in rows}) != 1:
                raise RuntimeError(f"Fixed-total common-randomness mismatch: {episode_index} {field}")
    if any(float(row["initial_oracle_optimality_violation"]) > 1e-8 for row in fixed_rows):
        raise RuntimeError("A fixed-total policy exceeded its numerical oracle")

    validation = {
        "status": "passed_strict_smoke_validation",
        "strict_scientific_completion": False,
        "manifest_sha256": _sha256(manifest_path),
        "selected_environment_count": len({row["environment"] for row in four_way}),
        "four_way_episode_row_count": len(four_way),
        "sigma_episode_row_count": len(sigma_rows),
        "fixed_total_episode_row_count": len(fixed_rows),
        "output_hashes": {
            filename: metadata["sha256"] for filename, metadata in manifest["expected_outputs"].items()
        },
        "checks": {
            "read_back_hashes_and_schemas": True,
            "four_way_common_randomness": True,
            "policy_seed_isolation_recorded": True,
            "controlled_sigma_standardized_randomness": True,
            "fixed_total_standardized_randomness": True,
            "fixed_total_orientation_balance": True,
            "time_matched_oracle_dominance": True,
            "no_primary_model_change": True,
        },
    }
    if previous_completion is not None:
        completion = json.loads(previous_completion)
        validation_path = output_dir / "r6_validation.json"
        if completion.get("schema_version") != 1:
            raise RuntimeError("Completion record has an unsupported schema version")
        if completion.get("status") != "strict_smoke_validation_complete":
            raise RuntimeError("Completion record has an invalid status")
        if completion.get("scientific_completion") is not False:
            raise RuntimeError("Smoke completion record cannot claim scientific completion")
        if completion.get("manifest_sha256") != _sha256(manifest_path):
            raise RuntimeError("Completion record has a stale manifest hash")
        if not validation_path.exists() or completion.get("validation_sha256") != _sha256(
            validation_path
        ):
            raise RuntimeError("Completion record has a stale validation hash")
        temporary = completion_path.with_name(f".{completion_path.name}.tmp")
        temporary.write_bytes(previous_completion)
        temporary.replace(completion_path)
    return validation


def _load_selected_configs(r4_dir: Path, frozen_config_path: Path):
    observed_hashes = {}
    for filename, expected_hash in R4_EXPECTED_HASHES.items():
        path = r4_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen R4 input: {path}")
        observed_hashes[filename] = _sha256(path)
        if observed_hashes[filename] != expected_hash:
            raise RuntimeError(f"Frozen R4 hash mismatch: {filename}")

    candidate_path = r4_dir / "r4_diagnostic_manual_advantage_candidates.csv"
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        selected_rows = list(select_r6_primary_environments(csv.DictReader(handle)))

    frozen = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    frozen_rows = list(frozen["environments"])
    if [row["environment"] for row in frozen_rows] != [row["environment"] for row in selected_rows]:
        raise RuntimeError("Frozen R6 configuration order does not match deterministic R4 selection")
    candidate_fields = (
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
    for source, frozen_row in zip(selected_rows, frozen_rows):
        if int(float(source["grid_index"])) != int(frozen_row["source_grid_index"]):
            raise RuntimeError("Frozen R6 grid index does not match the R4 source row")
        for field in candidate_fields:
            if float(source[field]) != float(frozen_row["config"][field]):
                raise RuntimeError(f"Frozen R6 parameter mismatch for {field}")
    selected_configs = [
        (
            str(row["environment"]),
            replace(
                EnvironmentConfig(**frozen_row["config"]),
                random_seed=(int(frozen_row["config"]["random_seed"]) + 6_000_000),
            ),
        )
        for row, frozen_row in zip(selected_rows, frozen_rows)
    ]
    return selected_rows, selected_configs, observed_hashes, _sha256(frozen_config_path)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "COMPLETED.json"
    legacy_marker = output_dir / "SMOKE_COMPLETED"
    if (completion_path.exists() or legacy_marker.exists()) and not args.overwrite:
        raise FileExistsError(f"Smoke output already exists: {output_dir}; pass --overwrite")
    completion_path.unlink(missing_ok=True)
    legacy_marker.unlink(missing_ok=True)

    selected_rows, selected_configs, observed_hashes, frozen_config_hash = _load_selected_configs(
        args.r4_dir.resolve(), args.frozen_config.resolve()
    )
    rr_policy = MyopicValueOfInformationPolicy(observation_draws=args.observation_draws)

    four_way_rows = []
    for position, (environment, config) in enumerate(selected_configs, start=1):
        print(f"[R6 smoke] four-way environment {position}/{len(selected_configs)}: {environment}")
        episodes = [
            build_evaluation_episode(
                config,
                episode_index=index,
                include_observation_streams=True,
                observations_per_person=args.observations_per_person,
            )
            for index in range(args.n_episodes)
        ]
        four_way_rows.extend(
            evaluate_r6_four_way_environment(
                environment,
                config,
                episodes,
                rr_policy=rr_policy,
                oracle_grid_size=args.oracle_grid_size,
            )
        )
    policy_summary, paired_summary = summarize_r6_four_way(four_way_rows)

    primary_environment, primary_config = selected_configs[0]
    print(f"[R6 smoke] controlled sigma_need sweep around {primary_environment}")
    sigma_rows = evaluate_r6_sigma_need_sweep(
        primary_environment,
        primary_config,
        args.sigma_need_values,
        n_episodes=args.n_episodes,
        rr_policy=rr_policy,
        oracle_grid_size=args.oracle_grid_size,
        observations_per_person=args.observations_per_person,
    )
    sigma_summary, sigma_gap_summary = summarize_r6_sigma_need_sweep(sigma_rows)

    print(f"[R6 smoke] fixed-total-need diagnostic around {primary_environment}")
    fixed_rows = evaluate_r6_fixed_total_need_diagnostic(
        primary_environment,
        primary_config,
        n_episodes_per_difference=args.fixed_total_episodes,
        rr_policy=rr_policy,
        oracle_grid_size=args.oracle_grid_size,
        observations_per_person=args.observations_per_person,
    )
    fixed_summary = summarize_r6_fixed_total_need_diagnostic(fixed_rows)

    tables = {
        "r6_four_way_episodes.csv": four_way_rows,
        "r6_four_way_policy_summary.csv": policy_summary,
        "r6_four_way_paired_comparisons.csv": paired_summary,
        "r6_sigma_need_episodes.csv": sigma_rows,
        "r6_sigma_need_environment_summary.csv": sigma_summary,
        "r6_sigma_need_gap_strata.csv": sigma_gap_summary,
        "r6_fixed_total_need_episodes.csv": fixed_rows,
        "r6_fixed_total_need_summary.csv": fixed_summary,
    }
    for rows in tables.values():
        for row in rows:
            row["analysis_mode"] = "smoke"
            row["scientific_status"] = "smoke_only_not_scientific_evidence"
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    output_metadata = {
        filename: {
            "sha256": _sha256(output_dir / filename),
            "row_count": len(rows),
            "schema": _fieldnames(rows),
            "schema_sha256": hashlib.sha256(
                json.dumps(_fieldnames(rows), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        for filename, rows in tables.items()
    }
    four_way_strata = 1 + 4 + 2 + 2
    sigma_summary_strata = 2 + 2 + 4 + (len(R6_DEFAULT_GAP_BIN_EDGES) - 1) + (
        len(R6_DEFAULT_TOTAL_NEED_BIN_EDGES) - 1
    )
    fixed_summary_strata = 1 + 2 + 2 + 4
    expected_row_counts = {
        "r6_four_way_episodes.csv": len(selected_configs) * args.n_episodes * 4,
        "r6_four_way_policy_summary.csv": len(selected_configs) * four_way_strata * 4,
        "r6_four_way_paired_comparisons.csv": len(selected_configs) * four_way_strata * 7,
        "r6_sigma_need_episodes.csv": len(args.sigma_need_values) * args.n_episodes,
        "r6_sigma_need_environment_summary.csv": len(args.sigma_need_values),
        "r6_sigma_need_gap_strata.csv": len(args.sigma_need_values) * sigma_summary_strata,
        "r6_fixed_total_need_episodes.csv": len(R6_FIXED_TOTAL_NEED_DIFFERENCES)
        * args.fixed_total_episodes,
        "r6_fixed_total_need_summary.csv": len(R6_FIXED_TOTAL_NEED_DIFFERENCES)
        * fixed_summary_strata
        + 1,
    }
    for filename, expected_count in expected_row_counts.items():
        if len(tables[filename]) != expected_count:
            raise RuntimeError(
                f"Generated row count violates the frozen Cartesian design: {filename}"
            )
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    source_files = (
        PROJECT_ROOT / "src/experiments/r6.py",
        PROJECT_ROOT / "scripts/run_r6_smoke.py",
        PROJECT_ROOT / "tests/test_r6.py",
        args.frozen_config.resolve(),
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "round6_pre_feedback_smoke",
        "analysis_mode": "smoke",
        "scientific_status": "smoke_only_not_scientific_evidence",
        "git_commit": _git_commit(),
        "git_worktree_dirty": dirty,
        "implementation_source_hashes": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_files
        },
        "r4_input_dir": str(args.r4_dir.resolve()),
        "r4_input_hashes": observed_hashes,
        "r4_source_commit": "4102fe34a525dd816e7939169c8581159551bed9",
        "frozen_environment_config_path": str(args.frozen_config.resolve()),
        "frozen_environment_config_sha256": frozen_config_hash,
        "selection_rule": (
            "R4 manual advantage > 0; manual true-equal and closer rates >= 0.80; "
            "manual samples >= 6; allocation distance >= 0.05; rank utility descending "
            "then environment ascending; retain first three"
        ),
        "selected_environments": [
            {
                "environment": environment,
                "config": asdict(config),
                "source_grid_index": selected_rows[index]["grid_index"],
            }
            for index, (environment, config) in enumerate(selected_configs)
        ],
        "settings": {
            "n_episodes_per_four_way_environment": args.n_episodes,
            "n_episodes_per_sigma_level": args.n_episodes,
            "n_episodes_per_fixed_total_difference": args.fixed_total_episodes,
            "episode_index_range": [0, args.n_episodes - 1],
            "fixed_total_episode_index_range": [0, args.fixed_total_episodes - 1],
            "rr_policy_class": "MyopicValueOfInformationPolicy",
            "observation_draws": args.observation_draws,
            "manual_policy_class": "ManualActiveSearchEqualOutcomePolicy",
            "manual_samples_per_person": 3,
            "equal_split_policy_class": "EqualSplitBaselinePolicy",
            "oracle_function": "full_information_utilitarian_allocation",
            "oracle_grid_size": args.oracle_grid_size,
            "allocation_tolerance": 0.05,
            "classification_tolerance": "1e-9 * max(1, remaining_time * sum(learning_rates))",
            "allocation_tie_tolerance": 1e-9,
            "observations_per_person_minimum": args.observations_per_person,
            "sigma_need_values": args.sigma_need_values,
            "realized_need_gap_bin_edges": [
                "inf" if math.isinf(value) else value for value in R6_DEFAULT_GAP_BIN_EDGES
            ],
            "total_need_bin_edges": [
                "-inf" if math.isinf(value) and value < 0.0 else "inf" if math.isinf(value) else value
                for value in R6_DEFAULT_TOTAL_NEED_BIN_EDGES
            ],
            "fixed_total_need_mean": R6_FIXED_TOTAL_NEED_MEAN,
            "fixed_total_need_differences": list(R6_FIXED_TOTAL_NEED_DIFFERENCES),
            "held_out_seed_namespace_offset": 6_000_000,
            "true_state_seed_rule": "config.random_seed + episode_index * 17 + 1",
            "observation_seed_rule": "config.random_seed + 100000 + episode_index * 17",
            "policy_computation_seed_rule": "config.random_seed + 300000 + episode_index * 17",
            "fixed_total_observation_seed_rule": "config.random_seed + 600000 + episode_index * 17",
        },
        "frozen_thresholds": {
            "diagnostic_rate": 0.80,
            "manual_minus_split_paired_ci95_low": "> 0",
            "utility_recovery_fraction": 0.90,
            "rr_mean_online_samples": "> 1",
            "sample_both_recipients_rate": 0.80,
            "mean_abs_allocation_from_equal": 0.05,
        },
        "required_serious_settings_not_used_here": {
            "observation_draws": 500,
            "held_out_episodes": "prespecify before Hoffman2 submission",
        },
        "expected_outputs": output_metadata,
        "expected_row_counts": expected_row_counts,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
        },
    }
    _atomic_write_json(output_dir / "r6_manifest.json", manifest)

    validation = validate_r6_artifacts(output_dir)
    _atomic_write_json(output_dir / "r6_validation.json", validation)
    completion = {
        "schema_version": 1,
        "status": "strict_smoke_validation_complete",
        "scientific_completion": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": _sha256(output_dir / "r6_manifest.json"),
        "validation_sha256": _sha256(output_dir / "r6_validation.json"),
    }
    _atomic_write_json(completion_path, completion)
    print(f"[R6 smoke] completed: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the prespecified Round 6 smoke analysis.")
    parser.add_argument(
        "--r4-dir",
        type=Path,
        default=Path("results/r4_diagnostic_active_search_server_20260714_array486"),
    )
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=Path("configs/r6_primary_environments.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/r6_pre_feedback_smoke"))
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--fixed-total-episodes", type=int, default=4)
    parser.add_argument("--observation-draws", type=int, default=8)
    parser.add_argument("--oracle-grid-size", type=int, default=401)
    parser.add_argument("--observations-per-person", type=int, default=100)
    parser.add_argument(
        "--sigma-need-values",
        type=_parse_float_list,
        default=[10.0, 60.0, 100.0],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
