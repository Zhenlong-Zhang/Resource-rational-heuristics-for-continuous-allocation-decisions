#!/usr/bin/env python3
"""Run and strictly collect reproducible Round 5 Hoffman2 array shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.r5 import (  # noqa: E402
    evaluate_r5_fixed_sampling_budgets,
    evaluate_r5_rr_environment,
    full_information_oracle_metrics,
    summarize_r5_fixed_sampling_budgets,
    summarize_r5_oracle_map,
    summarize_r5_rr_environments,
)
from src.experiments.randomization import build_evaluation_episode  # noqa: E402
from src.experiments.sweeps import (  # noqa: E402
    build_r5_oracle_map_configs,
    build_r5_six_sample_configs,
)
from src.mdp.meta_mdp import EnvironmentConfig  # noqa: E402
from src.policies.voi import MyopicValueOfInformationPolicy  # noqa: E402
from src.solvers.dp import DiscretizedDynamicProgrammingPolicy  # noqa: E402


SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = {3, 4}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family_configs(family: str) -> List[tuple[str, EnvironmentConfig]]:
    if family == "oracle":
        return build_r5_oracle_map_configs()
    if family == "six_sample":
        return build_r5_six_sample_configs()
    raise ValueError(f"Unknown R5 family: {family}")


def configs_from_json(path: Path) -> List[tuple[str, EnvironmentConfig]]:
    """Load a frozen set of named RR environments for confirmation runs."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("configs", payload) if isinstance(payload, dict) else payload
    if not isinstance(specs, list) or not specs:
        raise ValueError("configs JSON must contain a non-empty list")
    configs: List[tuple[str, EnvironmentConfig]] = []
    seen = set()
    for spec in specs:
        if not isinstance(spec, dict) or "environment" not in spec or "config" not in spec:
            raise ValueError("each config entry requires environment and config fields")
        environment = str(spec["environment"])
        if environment in seen:
            raise ValueError(f"duplicate environment in configs JSON: {environment}")
        seen.add(environment)
        configs.append((environment, EnvironmentConfig(**spec["config"])))
    return configs


def parse_sample_budgets(value: str) -> List[int]:
    budgets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not budgets or any(item < 0 or item % 2 for item in budgets):
        raise ValueError("sample budgets must be non-negative even integers")
    return budgets


def create_manifest(args) -> None:
    """Freeze environments, seeds, solver settings, and shard boundaries before submission."""

    output_dir = Path(args.output_dir).resolve()
    configs_path = Path(args.configs_json).resolve() if args.configs_json else None
    custom_families = {"custom_rr", "fixed_budget"}
    if args.family in custom_families and configs_path is None:
        raise ValueError(f"{args.family} requires --configs-json")
    if args.family not in custom_families and configs_path is not None:
        raise ValueError("--configs-json is only valid for custom_rr or fixed_budget")
    configs = configs_from_json(configs_path) if configs_path else family_configs(args.family)
    sample_budgets = parse_sample_budgets(args.sample_budgets) if args.family == "fixed_budget" else []
    environments = [
        {
            "environment_index": index,
            "environment": name,
            "config": asdict(config),
        }
        for index, (name, config) in enumerate(configs)
    ]
    tasks = []
    task_index = 0
    for environment in environments:
        for episode_start in range(0, args.episodes, args.episodes_per_task):
            episode_count = min(args.episodes_per_task, args.episodes - episode_start)
            tasks.append(
                {
                    "task_index": task_index,
                    "environment_index": environment["environment_index"],
                    "episode_start": episode_start,
                    "episode_count": episode_count,
                }
            )
            task_index += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "family": args.family,
        "git_commit": git_commit(),
        "episodes_per_environment": args.episodes,
        "episodes_per_task": args.episodes_per_task,
        "observation_draws": args.observation_draws,
        "allocation_tolerance": args.allocation_tolerance,
        "oracle_grid_size": args.oracle_grid_size,
        "seed_namespace_offset": args.seed_namespace_offset,
        "sample_budgets": sample_budgets,
        "rr_policy": {
            "name": args.rr_policy,
            "observation_draws": args.observation_draws,
            "dp_max_samples": args.dp_max_samples,
            "dp_mean_grid_size": args.dp_mean_grid_size,
            "dp_mean_grid_radius_sd": args.dp_mean_grid_radius_sd,
            "dp_observation_branches": args.dp_observation_branches,
            "dp_observation_integration": args.dp_observation_integration,
        },
        "configs_source_name": configs_path.name if configs_path else "built_in",
        "configs_source_hash": (
            hashlib.sha256(configs_path.read_bytes()).hexdigest() if configs_path else "built_in"
        ),
        "environments": environments,
        "tasks": tasks,
    }
    manifest["manifest_hash"] = digest(manifest)
    write_json(output_dir / "r5_manifest.json", manifest)
    write_json(
        output_dir / "r5_progress.json",
        {"created_at": utc_now(), "total_tasks": len(tasks), "completed_tasks": 0},
    )
    print(canonical_json({"manifest": str(output_dir / "r5_manifest.json"), "tasks": len(tasks)}))


def load_manifest(path: Path) -> Dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = manifest.pop("manifest_hash")
    actual_hash = digest(manifest)
    manifest["manifest_hash"] = claimed_hash
    if claimed_hash != actual_hash:
        raise RuntimeError("Manifest hash mismatch")
    if manifest["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise RuntimeError("Unsupported manifest schema")
    return manifest


def rr_policy_from_manifest(manifest: Mapping[str, object]):
    policy = dict(manifest.get("rr_policy", {}))
    name = str(policy.get("name", "myopic_voi"))
    if name == "myopic_voi":
        return MyopicValueOfInformationPolicy(
            observation_draws=int(policy.get("observation_draws", manifest["observation_draws"])),
        )
    if name == "discretized_dp":
        result = DiscretizedDynamicProgrammingPolicy(
            max_samples=int(policy.get("dp_max_samples", 10)),
            mean_grid_size=int(policy.get("dp_mean_grid_size", 50)),
            mean_grid_radius_sd=float(policy.get("dp_mean_grid_radius_sd", 3.0)),
            observation_branches=int(policy.get("dp_observation_branches", 7)),
            observation_integration=str(
                policy.get("dp_observation_integration", "gauss_hermite")
            ),
        )
        result.name = (
            f"discretized_dp_max{result.max_samples}"
            f"_grid{result.mean_grid_size}"
            f"_branches{result.observation_branches}"
            f"_{result.observation_integration}"
        )
        return result
    raise ValueError(f"Unknown frozen RR policy: {name}")


def task_paths(manifest_path: Path, task_index: int) -> tuple[Path, Path]:
    task_dir = manifest_path.parent / "tasks" / f"task_{task_index:06d}"
    return task_dir / "rows.csv", task_dir / "status.json"


def run_task(args) -> None:
    """Evaluate one manifest shard and atomically record its provenance and row hash."""

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    if git_commit() != manifest["git_commit"]:
        raise RuntimeError("Working tree commit does not match frozen manifest commit")
    tasks = manifest["tasks"]
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError("task_index outside manifest")
    task = tasks[args.task_index]
    environment = manifest["environments"][task["environment_index"]]
    config = EnvironmentConfig(**environment["config"])
    family = manifest["family"]
    if family == "oracle":
        rows = []
        seeded_config = EnvironmentConfig(
            **{
                **environment["config"],
                "random_seed": (config.random_seed or 0) + manifest["seed_namespace_offset"],
            }
        )
        for episode_index in range(
            task["episode_start"],
            task["episode_start"] + task["episode_count"],
        ):
            episode = build_evaluation_episode(seeded_config, episode_index=episode_index)
            rows.append(
                full_information_oracle_metrics(
                    environment["environment"],
                    seeded_config,
                    episode,
                    allocation_tolerance=manifest["allocation_tolerance"],
                    grid_size=manifest["oracle_grid_size"],
                )
            )
    elif family == "fixed_budget":
        rows = evaluate_r5_fixed_sampling_budgets(
            environment=environment["environment"],
            config=config,
            n_episodes=task["episode_count"],
            total_sample_budgets=manifest["sample_budgets"],
            allocation_tolerance=manifest["allocation_tolerance"],
            seed_namespace_offset=manifest["seed_namespace_offset"],
            episode_start=task["episode_start"],
        )
    else:
        policy = rr_policy_from_manifest(manifest)
        rows = evaluate_r5_rr_environment(
            environment=environment["environment"],
            config=config,
            n_episodes=task["episode_count"],
            rr_policy=policy,
            observation_draws=manifest["observation_draws"],
            allocation_tolerance=manifest["allocation_tolerance"],
            seed_namespace_offset=manifest["seed_namespace_offset"],
            episode_start=task["episode_start"],
        )
    rows_path, status_path = task_paths(manifest_path, args.task_index)
    write_csv(rows_path, rows)
    row_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    status = {
        "status": "ok",
        "completed_at": utc_now(),
        "task_index": args.task_index,
        "environment_index": task["environment_index"],
        "episode_start": task["episode_start"],
        "episode_count": task["episode_count"],
        "row_count": len(rows),
        "row_hash": row_hash,
        "manifest_hash": manifest["manifest_hash"],
        "git_commit": manifest["git_commit"],
    }
    write_json(status_path, status)
    print(canonical_json(status))


def progress(args) -> None:
    """Report shard completion from status files without starting new computation."""

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    completed = 0
    failed = 0
    for task in manifest["tasks"]:
        _, status_path = task_paths(manifest_path, task["task_index"])
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "ok":
                completed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    value = {
        "checked_at": utc_now(),
        "total_tasks": len(manifest["tasks"]),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "remaining_tasks": len(manifest["tasks"]) - completed - failed,
    }
    write_json(manifest_path.parent / "r5_progress.json", value)
    print(canonical_json(value))


def collect(args) -> None:
    """Reject incomplete or inconsistent shards before combining scientific outputs."""

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    all_rows: List[Dict[str, str]] = []
    seen_keys = set()
    missing = []
    for task in manifest["tasks"]:
        rows_path, status_path = task_paths(manifest_path, task["task_index"])
        if not rows_path.exists() or not status_path.exists():
            missing.append(task["task_index"])
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "ok":
            raise RuntimeError(f"Task {task['task_index']} is not ok")
        if status.get("manifest_hash") != manifest["manifest_hash"]:
            raise RuntimeError(f"Task {task['task_index']} manifest mismatch")
        if status.get("git_commit") != manifest["git_commit"]:
            raise RuntimeError(f"Task {task['task_index']} commit mismatch")
        if hashlib.sha256(rows_path.read_bytes()).hexdigest() != status.get("row_hash"):
            raise RuntimeError(f"Task {task['task_index']} row hash mismatch")
        rows = read_csv(rows_path)
        row_multiplier = len(manifest["sample_budgets"]) if manifest["family"] == "fixed_budget" else 1
        if len(rows) != task["episode_count"] * row_multiplier:
            raise RuntimeError(f"Task {task['task_index']} row count mismatch")
        for row in rows:
            key = (
                row["environment"],
                int(float(row["episode_index"])),
                int(float(row["sampling_budget_total"]))
                if manifest["family"] == "fixed_budget"
                else None,
            )
            if key in seen_keys:
                raise RuntimeError(f"Duplicate episode key: {key}")
            seen_keys.add(key)
            for required in ("need_1", "need_2"):
                if not math.isfinite(float(row[required])):
                    raise RuntimeError(f"Non-finite {required} in {key}")
            all_rows.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} tasks; first indices: {missing[:10]}")

    output_dir = manifest_path.parent
    if manifest["family"] == "oracle":
        rows_name = "r5_oracle_episodes.csv"
        summary_name = "r5_oracle_environment_summary.csv"
        summaries = summarize_r5_oracle_map(all_rows)
    elif manifest["family"] == "fixed_budget":
        rows_name = "r5_fixed_budget_episodes.csv"
        summary_name = "r5_fixed_budget_summary.csv"
        summaries = summarize_r5_fixed_sampling_budgets(all_rows)
    else:
        rows_name = "r5_rr_episodes.csv"
        summary_name = "r5_rr_environment_summary.csv"
        summaries = summarize_r5_rr_environments(all_rows)
    write_csv(output_dir / rows_name, all_rows)
    write_csv(output_dir / summary_name, summaries)
    candidates = [] if manifest["family"] == "fixed_budget" else [
        row
        for row in summaries
        if float(row.get("r5_joint_oracle_candidate", row.get("r5_joint_discovery_candidate", 0.0))) >= 0.5
    ]
    write_csv(output_dir / "r5_joint_candidates.csv", candidates)
    completed = {
        "completed_at": utc_now(),
        "family": manifest["family"],
        "git_commit": manifest["git_commit"],
        "manifest_hash": manifest["manifest_hash"],
        "task_count": len(manifest["tasks"]),
        "episode_row_count": len(all_rows),
        "summary_row_count": len(summaries),
        "candidate_count": len(candidates),
    }
    write_json(output_dir / "COMPLETED.json", completed)
    print(canonical_json(completed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible Hoffman2 workflow for R5 analyses")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument(
        "--family",
        choices=("oracle", "six_sample", "custom_rr", "fixed_budget"),
        required=True,
    )
    create.add_argument("--configs-json")
    create.add_argument("--output-dir", required=True)
    create.add_argument("--episodes", type=int, required=True)
    create.add_argument("--episodes-per-task", type=int, required=True)
    create.add_argument("--observation-draws", type=int, default=500)
    create.add_argument("--allocation-tolerance", type=float, default=0.05)
    create.add_argument("--oracle-grid-size", type=int, default=4001)
    create.add_argument("--seed-namespace-offset", type=int, default=0)
    create.add_argument("--sample-budgets", default="0,2,4,6,8,10,12")
    create.add_argument(
        "--rr-policy",
        choices=("myopic_voi", "discretized_dp"),
        default="myopic_voi",
    )
    create.add_argument("--dp-max-samples", type=int, default=10)
    create.add_argument("--dp-mean-grid-size", type=int, default=50)
    create.add_argument("--dp-mean-grid-radius-sd", type=float, default=3.0)
    create.add_argument("--dp-observation-branches", type=int, default=7)
    create.add_argument(
        "--dp-observation-integration",
        choices=("quantile", "gauss_hermite"),
        default="gauss_hermite",
    )
    create.set_defaults(function=create_manifest)

    run = subparsers.add_parser("run-task")
    run.add_argument("--manifest", required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.set_defaults(function=run_task)

    check = subparsers.add_parser("progress")
    check.add_argument("--manifest", required=True)
    check.set_defaults(function=progress)

    collector = subparsers.add_parser("collect")
    collector.add_argument("--manifest", required=True)
    collector.set_defaults(function=collect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
