from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combine_method_comparison_results import parse_task_manifest  # noqa: E402
from scripts.run_method_comparison_task import (  # noqa: E402
    EPISODE_FIELDNAMES,
    episode_fingerprint,
    observation_hashes,
    run_single_episode_from_task_metadata,
    write_summary_from_task_metadata,
)
from src.experiments.randomization import build_evaluation_episode  # noqa: E402
from src.mdp.meta_mdp import EnvironmentConfig  # noqa: E402


SCIENTIFIC_BASELINE_COMMIT = "4102fe3"
WORKFLOW = "r3_method_comparison_episode_array"
MANIFEST_FILENAME = "r3_episode_array_manifest.json"
MANIFEST_HASH_FILENAME = "r3_episode_array_manifest.sha256"
SCHEDULER_FILENAME = "r3_episode_array_scheduler.json"
SUBMISSION_JOURNAL_FILENAME = "r3_episode_array_submission.json"
PROGRESS_FILENAME = "r3_episode_array_progress.json"
JOURNAL_FILENAME = "promotion_journal.json"
LOCK_DIRNAME = "collector.lock"
STAGE_DIRNAME = "staging/canonical_complete"
SNAPSHOT_DIRNAME = "snapshot/canonical"
BACKUP_DIRNAME = "promotion/original_canonical"
COMPLETION_MARKER = "R3_EPISODE_ARRAY_COMPLETE"
EPISODE_FILENAME = "rr_approximation_method_episode_results.csv"
SUMMARY_FILENAME = "rr_approximation_methods_comparison.csv"
METADATA_FILENAME = "task_metadata.json"
REQUIRED_TASK_FILES = (EPISODE_FILENAME, SUMMARY_FILENAME, METADATA_FILENAME)
NON_SCIENTIFIC_RETRY_FIELDS = {"elapsed_seconds"}
SERIAL_REFERENCE_FIELDNAMES = tuple(
    field for field in EPISODE_FIELDNAMES if field not in NON_SCIENTIFIC_RETRY_FIELDS
)
PRODUCTION_SCIENTIFIC_SETTINGS = {
    "n_episodes": 1200,
    "rr_observation_draws": 500,
    "blinkered_observation_draws": 250,
    "blinkered_horizon": 2,
    "use_common_observation_streams": True,
    "observations_per_person": 500,
}
EXECUTION_CODE_PATHS = tuple(
    sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "src").rglob("*.py"))
) + (
    "scripts/generate_results.py",
    "scripts/run_method_comparison_task.py",
    "scripts/combine_method_comparison_results.py",
    "scripts/r3_episode_array_workflow.py",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def execution_code_hashes() -> Dict[str, str]:
    return {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in EXECUTION_CODE_PATHS
    }


def tracked_checkout_is_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def tree_inventory(root: Path) -> List[Dict[str, object]]:
    if not root.is_dir():
        raise RuntimeError(f"Tree does not exist: {root}")
    rows: List[Dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def tree_fingerprint(root: Path) -> str:
    return fingerprint(tree_inventory(root))


def set_tree_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        mode = path.stat().st_mode
        path.chmod(mode | 0o200)


def set_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def task_key(environment: str, policy: str) -> str:
    return f"{environment}::{policy}"


def task_relative_dir(environment: str, policy: str) -> Path:
    return Path("tasks") / "methods" / environment / policy


def row_fingerprint(row: Mapping[str, str]) -> str:
    return fingerprint(dict(row))


def scientific_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {key: value for key, value in row.items() if key not in NON_SCIENTIFIC_RETRY_FIELDS}


def scientific_row_fingerprint(row: Mapping[str, str]) -> str:
    return fingerprint(scientific_row(row))


def serial_reference_key(row: Mapping[str, str], *, label: str) -> tuple[str, str, int]:
    environment = row.get("environment", "")
    policy = row.get("policy", "")
    try:
        episode_index = int(row.get("episode_index", ""))
    except ValueError as error:
        raise RuntimeError(f"Invalid episode index in {label}") from error
    if not environment or not policy:
        raise RuntimeError(f"Missing environment or policy in {label}")
    return environment, policy, episode_index


def read_serial_reference_rows(
    path: Path,
) -> tuple[List[Dict[str, str]], Dict[tuple[str, str, int], Dict[str, str]]]:
    fieldnames, rows = read_csv(path)
    if fieldnames != EPISODE_FIELDNAMES:
        raise RuntimeError("Serial reference CSV does not have canonical episode fields")
    references: Dict[tuple[str, str, int], Dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise RuntimeError(f"Malformed serial reference row {row_number}")
        key = serial_reference_key(row, label=f"serial reference row {row_number}")
        if key in references:
            raise RuntimeError(f"Duplicate serial reference identity: {key}")
        references[key] = row
    return rows, references


def serial_reference_target_record(
    task: Mapping[str, object], reference_row: Mapping[str, str]
) -> Dict[str, object]:
    scientific = {field: reference_row[field] for field in SERIAL_REFERENCE_FIELDNAMES}
    return {
        "task_id": int(task["task_id"]),
        "environment": str(task["environment"]),
        "policy": str(task["policy"]),
        "episode_index": int(task["episode_index"]),
        "scientific_row_fingerprint": fingerprint(scientific),
    }


def path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def evidence_file_record(path: Path) -> Dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def validate_evidence_file(record: Mapping[str, object], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise RuntimeError(f"{label} evidence file is missing: {path}")
    if int(record.get("size", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} evidence size mismatch: {path}")
    if record.get("sha256") != file_sha256(path):
        raise RuntimeError(f"{label} evidence hash mismatch: {path}")
    return path


def validate_writer_evidence(
    evidence: Mapping[str, object], required_job_ids: Sequence[str]
) -> None:
    payload = {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}
    if evidence.get("evidence_fingerprint") != fingerprint(payload):
        raise RuntimeError("Writer evidence fingerprint mismatch")
    if evidence.get("writers_quiescent") is not True:
        raise RuntimeError("Writer evidence does not certify quiescence")
    accounted = {str(value) for value in evidence.get("accounted_job_ids", [])}
    missing = sorted(set(required_job_ids) - accounted)
    if missing:
        raise RuntimeError(f"Writer evidence does not account for job IDs: {missing}")
    final_qstat = validate_evidence_file(
        dict(evidence.get("qstat_final", {})), "final qstat"
    )
    final_text = final_qstat.read_text(encoding="utf-8")
    pattern = str(evidence.get("writer_job_name_pattern", ""))
    for line in final_text.splitlines()[2:]:
        fields = line.split()
        if len(fields) >= 3 and (fields[0] in accounted or (pattern and re.search(pattern, fields[2]))):
            raise RuntimeError(f"Final qstat still contains a possible writer: {line}")
    for record in [*evidence.get("jobs", []), *evidence.get("successors", [])]:
        if record.get("can_write") is not False:
            raise RuntimeError(f"Writer is not quiescent: {record}")
        if not record.get("scheduler_disposition"):
            raise RuntimeError(f"Writer lacks scheduler disposition: {record}")
        validate_evidence_file(dict(record.get("qacct", {})), f"qacct {record.get('job_id')}")


def validate_episode_rows(
    path: Path,
    *,
    environment: str,
    policy: str,
    target_episodes: int,
) -> List[Dict[str, str]]:
    fieldnames, rows = read_csv(path)
    if fieldnames != EPISODE_FIELDNAMES:
        raise RuntimeError(f"Unexpected episode fields in {path}")
    seen: set[int] = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise RuntimeError(f"Malformed episode row in {path}")
        if row.get("environment") != environment or row.get("policy") != policy:
            raise RuntimeError(f"Task identity mismatch in {path}")
        try:
            episode_index = int(row["episode_index"])
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"Invalid episode index in {path}") from error
        if episode_index < 0 or episode_index >= target_episodes:
            raise RuntimeError(f"Out-of-range episode index {episode_index} in {path}")
        if episode_index in seen:
            raise RuntimeError(f"Duplicate episode index {episode_index} in {path}")
        seen.add(episode_index)
    return rows


def validate_task_metadata(
    metadata: Mapping[str, object], manifest_task: Mapping[str, str], *, run_mode: str
) -> None:
    if metadata.get("environment") != manifest_task["environment"]:
        raise RuntimeError("Task metadata environment mismatch")
    if metadata.get("policy_arg") != manifest_task["policy_arg"]:
        raise RuntimeError("Task metadata policy argument mismatch")
    policy = dict(metadata.get("policy", {}))
    if policy.get("name") != manifest_task["policy"]:
        raise RuntimeError("Task metadata policy label mismatch")
    settings = dict(metadata.get("settings", {}))
    if int(settings.get("n_episodes", 0)) <= 0:
        raise RuntimeError("Task metadata has invalid target episode count")
    if run_mode in {"production", "smoke"}:
        for name, expected in PRODUCTION_SCIENTIFIC_SETTINGS.items():
            if name == "n_episodes" and run_mode == "smoke":
                continue
            if settings.get(name) != expected:
                raise RuntimeError(
                    f"Task metadata {name}={settings.get(name)!r}; expected {expected!r}"
                )
        policy = dict(metadata.get("policy", {}))
        if policy.get("class") == "MyopicValueOfInformationPolicy" and int(
            policy.get("observation_draws", -1)
        ) != 500:
            raise RuntimeError("Myopic VOI policy must use 500 observation draws")
        if policy.get("class") == "BlinkeredPolicy":
            if int(policy.get("observation_draws", -1)) != 250:
                raise RuntimeError("Blinkered policy must use 250 observation draws")
            if int(policy.get("horizon", -1)) != 2:
                raise RuntimeError("Blinkered policy must use horizon 2")
    EnvironmentConfig(**dict(metadata.get("environment_config", {})))


def manifest_payload(manifest: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}


def finalize_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    manifest = dict(manifest)
    manifest["manifest_fingerprint"] = fingerprint(manifest_payload(manifest))
    return manifest


def write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    finalized = finalize_manifest(manifest)
    write_json_atomic(path, finalized)
    sidecar = path.with_name(MANIFEST_HASH_FILENAME)
    sidecar.write_text(
        file_sha256(path) + "\n", encoding="utf-8"
    )
    path.chmod(path.stat().st_mode & ~0o222)
    sidecar.chmod(sidecar.stat().st_mode & ~0o222)


def load_manifest(path: Path) -> Dict[str, object]:
    manifest = read_json(path)
    if fingerprint(manifest_payload(manifest)) != manifest.get("manifest_fingerprint"):
        raise RuntimeError("Manifest payload fingerprint mismatch")
    sidecar = path.with_name(MANIFEST_HASH_FILENAME)
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != file_sha256(path):
        raise RuntimeError("Manifest file hash mismatch")
    validate_manifest_structure(manifest)
    return manifest


def freeze_checkpoint(
    *,
    canonical_run_dir: Path,
    workflow_run_dir: Path,
    task_manifest_path: Path,
    writer_evidence_path: Path,
    required_writer_job_ids: Sequence[str],
    git_commit: str,
    array_lanes: int,
    lane_throttle: int,
    expected_task_count: int = 700,
    run_mode: str = "test",
    reviewed_commit: str = "",
    enforce_clean_checkout: bool | None = None,
    serial_reference_path: Path | None = None,
) -> Dict[str, object]:
    canonical_run_dir = canonical_run_dir.resolve()
    workflow_run_dir = workflow_run_dir.resolve()
    if workflow_run_dir == canonical_run_dir or canonical_run_dir in workflow_run_dir.parents:
        raise RuntimeError("Workflow run directory must be outside the canonical run directory")
    if array_lanes < 1 or array_lanes > 4:
        raise RuntimeError("array_lanes must be in [1, 4]")
    if lane_throttle < 1 or lane_throttle > 100:
        raise RuntimeError("lane_throttle must be in [1, 100]")
    if array_lanes * (lane_throttle + 1) + 1 > 500:
        raise RuntimeError("Lane concurrency plus collector exceeds max_u_jobs=500")
    if run_mode not in {"production", "smoke", "test"}:
        raise RuntimeError(f"Unsupported run mode: {run_mode}")
    serial_reference: Dict[str, object] | None = None
    serial_references: Dict[tuple[str, str, int], Dict[str, str]] = {}
    if run_mode == "smoke":
        if serial_reference_path is None:
            raise RuntimeError("Smoke freeze requires a serial-reference CSV")
        resolved_reference = serial_reference_path.resolve()
        if not resolved_reference.is_file():
            raise RuntimeError(f"Serial reference CSV is missing: {resolved_reference}")
        if path_is_within(resolved_reference, canonical_run_dir) or path_is_within(
            resolved_reference, workflow_run_dir
        ):
            raise RuntimeError(
                "Serial reference CSV must be outside canonical and workflow run directories"
            )
        _, serial_references = read_serial_reference_rows(resolved_reference)
        serial_reference = {
            "path": str(resolved_reference),
            "size": resolved_reference.stat().st_size,
            "sha256": file_sha256(resolved_reference),
        }
    elif serial_reference_path is not None:
        raise RuntimeError("Serial reference is only valid for smoke freeze")
    if workflow_run_dir.exists() and any(workflow_run_dir.iterdir()):
        raise RuntimeError(f"Workflow run directory is not empty: {workflow_run_dir}")
    if enforce_clean_checkout is None:
        enforce_clean_checkout = run_mode in {"production", "smoke"}
    if run_mode in {"production", "smoke"}:
        if not reviewed_commit:
            raise RuntimeError(f"{run_mode} freeze requires an explicit reviewed commit")
        if git_commit != reviewed_commit:
            raise RuntimeError("Executing commit does not match the reviewed commit")
        if current_git_commit() != reviewed_commit:
            raise RuntimeError("Checkout HEAD does not match the reviewed commit")
        if enforce_clean_checkout and not tracked_checkout_is_clean():
            raise RuntimeError("Execution checkout has tracked modifications")
    evidence = read_json(writer_evidence_path)
    validate_writer_evidence(evidence, required_writer_job_ids)
    task_rows = parse_task_manifest(task_manifest_path)
    if len(task_rows) != expected_task_count:
        raise RuntimeError(
            f"Task manifest has {len(task_rows)} rows; expected {expected_task_count}"
        )

    workflow_run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = workflow_run_dir / SNAPSHOT_DIRNAME
    shutil.copytree(canonical_run_dir, snapshot_root)
    input_dir = workflow_run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest_path = input_dir / "r3_approx_methods_tasks.tsv"
    shutil.copy2(task_manifest_path, frozen_manifest_path)
    frozen_writer_evidence = input_dir / "writer_quiescence.json"
    shutil.copy2(writer_evidence_path, frozen_writer_evidence)

    affected: List[Dict[str, object]] = []
    episode_tasks: List[Dict[str, object]] = []
    all_metadata: List[Dict[str, object]] = []
    for row in task_rows:
        relative_dir = task_relative_dir(row["environment"], row["policy"])
        task_dir = snapshot_root / relative_dir
        for filename in REQUIRED_TASK_FILES:
            if not (task_dir / filename).is_file():
                raise RuntimeError(f"Missing canonical task file: {task_dir / filename}")
        metadata = read_json(task_dir / METADATA_FILENAME)
        validate_task_metadata(metadata, row, run_mode=run_mode)
        target_episodes = int(dict(metadata["settings"])["n_episodes"])
        rows = validate_episode_rows(
            task_dir / EPISODE_FILENAME,
            environment=row["environment"],
            policy=row["policy"],
            target_episodes=target_episodes,
        )
        all_metadata.append(metadata)
        completed = sorted(int(value["episode_index"]) for value in rows)
        missing = sorted(set(range(target_episodes)) - set(completed))
        if not missing:
            continue
        metadata_hash = file_sha256(task_dir / METADATA_FILENAME)
        frozen_file_records = {
            filename: {
                "size": (task_dir / filename).stat().st_size,
                "sha256": file_sha256(task_dir / filename),
            }
            for filename in REQUIRED_TASK_FILES
        }
        frozen_row_fingerprints = {
            str(row_value["episode_index"]): row_fingerprint(row_value)
            for row_value in rows
        }
        key = task_key(row["environment"], row["policy"])
        affected_record = {
            "task_key": key,
            "environment": row["environment"],
            "policy_arg": row["policy_arg"],
            "policy": row["policy"],
            "relative_dir": str(relative_dir),
            "target_episodes": target_episodes,
            "completed_indices": completed,
            "missing_indices": missing,
            "task_metadata_fingerprint": fingerprint(metadata),
            "task_metadata_sha256": metadata_hash,
            "frozen_files": frozen_file_records,
            "frozen_row_fingerprints": frozen_row_fingerprints,
        }
        affected.append(affected_record)
        for episode_index in missing:
            task_id = len(episode_tasks) + 1
            command_identity = {
                "entrypoint": "run_single_episode_from_task_metadata",
                "task_key": key,
                "episode_index": episode_index,
                "task_metadata_fingerprint": fingerprint(metadata),
            }
            shard_root = workflow_run_dir / "shards" / f"task_{task_id:06d}"
            episode_tasks.append(
                {
                    "task_id": task_id,
                    "task_key": key,
                    "environment": row["environment"],
                    "policy_arg": row["policy_arg"],
                    "policy": row["policy"],
                    "episode_index": episode_index,
                    "target_episodes": target_episodes,
                    "relative_task_dir": str(relative_dir),
                    "metadata_path": str(task_dir / METADATA_FILENAME),
                    "task_metadata_fingerprint": fingerprint(metadata),
                    "frozen_episode_csv_sha256": frozen_file_records[EPISODE_FILENAME]["sha256"],
                    "attempts_dir": str(shard_root / "attempts"),
                    "scientific_command": command_identity,
                    "scientific_command_fingerprint": fingerprint(command_identity),
                }
            )

    snapshot_inventory = tree_inventory(snapshot_root)
    scientific_config = {
        "target_episode_counts": sorted(
            {int(dict(metadata["settings"])["n_episodes"]) for metadata in all_metadata}
        ),
        "rr_observation_draws": sorted(
            {int(dict(metadata["settings"])["rr_observation_draws"]) for metadata in all_metadata}
        ),
        "blinkered_observation_draws": sorted(
            {int(dict(metadata["settings"])["blinkered_observation_draws"]) for metadata in all_metadata}
        ),
        "blinkered_horizons": sorted(
            {int(dict(metadata["settings"])["blinkered_horizon"]) for metadata in all_metadata}
        ),
        "common_observation_streams": sorted(
            {bool(dict(metadata["settings"])["use_common_observation_streams"]) for metadata in all_metadata}
        ),
        "observations_per_person": sorted(
            {int(dict(metadata["settings"])["observations_per_person"]) for metadata in all_metadata}
        ),
        "method_task_count": len(task_rows),
        "all_task_metadata_fingerprint": fingerprint(all_metadata),
    }
    lanes: List[Dict[str, object]] = []
    for lane_index in range(array_lanes):
        task_ids = [
            int(task["task_id"])
            for task in episode_tasks
            if (int(task["task_id"]) - 1) % array_lanes == lane_index
        ]
        lane_file = input_dir / f"lane_{lane_index + 1:02d}_task_ids.txt"
        lane_file.write_text(
            "".join(f"{task_id}\n" for task_id in task_ids), encoding="utf-8"
        )
        lanes.append(
            {
                "lane_id": lane_index + 1,
                "task_ids": task_ids,
                "task_file": str(lane_file),
                "task_file_sha256": file_sha256(lane_file),
            }
        )
    if serial_reference is not None:
        target_rows: List[Dict[str, object]] = []
        for task in episode_tasks:
            key = (
                str(task["environment"]),
                str(task["policy"]),
                int(task["episode_index"]),
            )
            reference_row = serial_references.get(key)
            if reference_row is None:
                raise RuntimeError(f"Serial reference is missing smoke target identity: {key}")
            target_rows.append(serial_reference_target_record(task, reference_row))
        resolved_reference = Path(str(serial_reference["path"]))
        if (
            resolved_reference.stat().st_size != serial_reference["size"]
            or file_sha256(resolved_reference) != serial_reference["sha256"]
        ):
            raise RuntimeError("Serial reference changed during smoke freeze")
        serial_reference["compared_fields"] = list(SERIAL_REFERENCE_FIELDNAMES)
        serial_reference["target_rows"] = target_rows
        serial_reference["target_rows_fingerprint"] = fingerprint(target_rows)
    manifest = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "project_root": str(PROJECT_ROOT),
        "workflow_run_dir": str(workflow_run_dir),
        "canonical_run_dir": str(canonical_run_dir),
        "snapshot_root": str(snapshot_root),
        "snapshot_inventory": snapshot_inventory,
        "snapshot_tree_fingerprint": fingerprint(snapshot_inventory),
        "task_manifest_path": str(frozen_manifest_path),
        "task_manifest_sha256": file_sha256(frozen_manifest_path),
        "writer_evidence_path": str(frozen_writer_evidence),
        "writer_evidence_sha256": file_sha256(frozen_writer_evidence),
        "required_writer_job_ids": list(required_writer_job_ids),
        "run_mode": run_mode,
        "reviewed_commit": reviewed_commit or git_commit,
        "enforce_clean_checkout": bool(enforce_clean_checkout),
        "git_commit": git_commit,
        "execution_code_hashes": execution_code_hashes(),
        "execution_code_fingerprint": fingerprint(execution_code_hashes()),
        "scientific_baseline_commit": SCIENTIFIC_BASELINE_COMMIT,
        "scientific_config": scientific_config,
        "scientific_config_fingerprint": fingerprint(scientific_config),
        "expected_method_tasks": expected_task_count,
        "array_lanes": array_lanes,
        "lane_throttle": lane_throttle,
        "scheduler_limits": {
            "max_aj_instances": 100,
            "max_aj_tasks": 200000,
            "max_u_jobs": 500,
        },
        "lanes": lanes,
        "affected_tasks": affected,
        "tasks": episode_tasks,
    }
    if serial_reference is not None:
        manifest["serial_reference"] = serial_reference
    manifest_path = workflow_run_dir / MANIFEST_FILENAME
    write_manifest(manifest_path, manifest)
    set_tree_read_only(snapshot_root)
    return load_manifest(manifest_path)


def validate_manifest_structure(manifest: Mapping[str, object]) -> None:
    if fingerprint(manifest_payload(manifest)) != manifest.get("manifest_fingerprint"):
        raise RuntimeError("Manifest payload fingerprint mismatch")
    if manifest.get("workflow") != WORKFLOW:
        raise RuntimeError("Workflow mismatch")
    if manifest.get("scientific_baseline_commit") != SCIENTIFIC_BASELINE_COMMIT:
        raise RuntimeError("Scientific baseline mismatch")
    run_mode = str(manifest.get("run_mode", ""))
    if run_mode not in {"production", "smoke", "test"}:
        raise RuntimeError("Manifest run mode is invalid")
    if run_mode in {"production", "smoke"} and manifest.get("git_commit") != manifest.get(
        "reviewed_commit"
    ):
        raise RuntimeError("Manifest is not bound to its reviewed commit")
    validate_manifest_serial_reference(manifest)
    code_hashes = dict(manifest.get("execution_code_hashes", {}))
    if set(code_hashes) != set(EXECUTION_CODE_PATHS):
        raise RuntimeError("Manifest execution-code coverage is incomplete")
    if fingerprint(code_hashes) != manifest.get("execution_code_fingerprint"):
        raise RuntimeError("Manifest execution-code fingerprint mismatch")
    if fingerprint(manifest.get("scientific_config")) != manifest.get(
        "scientific_config_fingerprint"
    ):
        raise RuntimeError("Scientific configuration fingerprint mismatch")
    tasks = list(manifest.get("tasks", []))
    task_ids = [int(task["task_id"]) for task in tasks]
    if task_ids != list(range(1, len(tasks) + 1)):
        raise RuntimeError("Episode-array task IDs are not a unique complete range")
    keys = [(str(task["task_key"]), int(task["episode_index"])) for task in tasks]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate method-task/episode keys in manifest")
    attempt_paths = [str(task["attempts_dir"]) for task in tasks]
    if len(attempt_paths) != len(set(attempt_paths)):
        raise RuntimeError("Duplicate attempt directories in manifest")
    affected_missing = sorted(
        (str(task["task_key"]), int(index))
        for task in manifest.get("affected_tasks", [])
        for index in task["missing_indices"]
    )
    if sorted(keys) != affected_missing:
        raise RuntimeError("Manifest rows do not exactly cover frozen missing indices")
    run_dir = Path(str(manifest["workflow_run_dir"]))
    snapshot_root = Path(str(manifest["snapshot_root"]))
    affected_records = affected_by_key(manifest)
    for task in tasks:
        affected = affected_records.get(str(task["task_key"]))
        if affected is None:
            raise RuntimeError("Manifest task does not belong to an affected method task")
        expected_attempts = run_dir / "shards" / f"task_{int(task['task_id']):06d}" / "attempts"
        expected_metadata = snapshot_root / str(affected["relative_dir"]) / METADATA_FILENAME
        command_identity = {
            "entrypoint": "run_single_episode_from_task_metadata",
            "task_key": task["task_key"],
            "episode_index": task["episode_index"],
            "task_metadata_fingerprint": task["task_metadata_fingerprint"],
        }
        if Path(str(task["attempts_dir"])) != expected_attempts:
            raise RuntimeError("Manifest attempt path is not canonical and isolated")
        if Path(str(task["metadata_path"])) != expected_metadata:
            raise RuntimeError("Manifest metadata path mismatch")
        if task.get("scientific_command") != command_identity:
            raise RuntimeError("Manifest scientific command identity mismatch")
        if task.get("scientific_command_fingerprint") != fingerprint(command_identity):
            raise RuntimeError("Manifest scientific command fingerprint mismatch")
        if int(task["episode_index"]) not in affected["missing_indices"]:
            raise RuntimeError("Manifest task is not a frozen missing episode")
    lane_records = list(manifest.get("lanes", []))
    if len(lane_records) != int(manifest.get("array_lanes", 0)):
        raise RuntimeError("Lane count mismatch")
    lane_task_ids = [
        int(task_id) for lane in lane_records for task_id in lane.get("task_ids", [])
    ]
    if sorted(lane_task_ids) != task_ids or len(lane_task_ids) != len(set(lane_task_ids)):
        raise RuntimeError("Lane assignments are not disjoint exact manifest coverage")
    lane_throttle = int(manifest.get("lane_throttle", 0))
    if lane_throttle < 1 or lane_throttle > 100:
        raise RuntimeError("Lane throttle exceeds max_aj_instances=100")
    if len(lane_records) > 4 or len(lane_records) * (lane_throttle + 1) + 1 > 500:
        raise RuntimeError("Lane configuration exceeds recorded max_u_jobs=500")
    for lane in lane_records:
        if len(lane.get("task_ids", [])) > 200000:
            raise RuntimeError("Lane task count exceeds max_aj_tasks=200000")
        task_file = Path(str(lane["task_file"]))
        if not task_file.is_file() or file_sha256(task_file) != lane["task_file_sha256"]:
            raise RuntimeError(f"Lane task file mismatch: {task_file}")
        file_task_ids = [
            int(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line
        ]
        if file_task_ids != lane["task_ids"]:
            raise RuntimeError(f"Lane task mapping mismatch: {task_file}")


def validate_manifest_serial_reference(manifest: Mapping[str, object]) -> Dict[str, object] | None:
    run_mode = str(manifest.get("run_mode", ""))
    if run_mode != "smoke":
        if "serial_reference" in manifest:
            raise RuntimeError("Non-smoke manifest unexpectedly binds a serial reference")
        return None
    record = dict(manifest.get("serial_reference", {}))
    path = Path(str(record.get("path", "")))
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise RuntimeError("Smoke manifest serial reference path is missing or not resolved")
    canonical = Path(str(manifest["canonical_run_dir"])).resolve()
    workflow = Path(str(manifest["workflow_run_dir"])).resolve()
    if path_is_within(path, canonical) or path_is_within(path, workflow):
        raise RuntimeError("Smoke manifest serial reference is inside a run directory")
    if (
        path.stat().st_size != int(record.get("size", -1))
        or file_sha256(path) != record.get("sha256")
    ):
        raise RuntimeError("Smoke manifest serial reference changed")
    if record.get("compared_fields") != list(SERIAL_REFERENCE_FIELDNAMES):
        raise RuntimeError("Smoke manifest serial-reference field binding mismatch")
    _, references = read_serial_reference_rows(path)
    expected_targets: List[Dict[str, object]] = []
    for task in manifest.get("tasks", []):
        key = (
            str(task["environment"]),
            str(task["policy"]),
            int(task["episode_index"]),
        )
        reference_row = references.get(key)
        if reference_row is None:
            raise RuntimeError(f"Serial reference is missing smoke target identity: {key}")
        expected_targets.append(serial_reference_target_record(task, reference_row))
    if record.get("target_rows") != expected_targets or record.get(
        "target_rows_fingerprint"
    ) != fingerprint(expected_targets):
        raise RuntimeError("Smoke manifest serial-reference target binding mismatch")
    return record


def require_execution_checkout(manifest: Mapping[str, object]) -> None:
    actual = current_git_commit()
    expected = str(manifest["git_commit"])
    if actual != expected:
        raise RuntimeError(f"Execution checkout {actual} does not match manifest {expected}")
    reviewed = str(manifest.get("reviewed_commit", ""))
    if manifest.get("run_mode") in {"production", "smoke"} and expected != reviewed:
        raise RuntimeError("Execution checkout is not the reviewed commit")
    actual_hashes = execution_code_hashes()
    if actual_hashes != manifest.get("execution_code_hashes"):
        raise RuntimeError("Execution code hashes do not match the frozen manifest")
    if manifest.get("enforce_clean_checkout") and not tracked_checkout_is_clean():
        raise RuntimeError("Execution checkout has tracked modifications")


def manifest_task(manifest: Mapping[str, object], task_id: int) -> Mapping[str, object]:
    matches = [task for task in manifest["tasks"] if int(task["task_id"]) == task_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one manifest task for ID {task_id}")
    return matches[0]


def sanitize_attempt_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError("Invalid attempt ID")
    return sanitized


def expected_identity(metadata: Mapping[str, object], episode_index: int) -> Dict[str, object]:
    settings = dict(metadata["settings"])
    config = EnvironmentConfig(**dict(metadata["environment_config"]))
    episode = build_evaluation_episode(
        config=config,
        episode_index=episode_index,
        include_observation_streams=bool(settings["use_common_observation_streams"]),
        observations_per_person=int(settings["observations_per_person"]),
    )
    observation_hash_1, observation_hash_2 = observation_hashes(episode)
    return {
        "true_need_1": episode.true_state.need_1,
        "true_need_2": episode.true_state.need_2,
        "observation_stream_hash_1": observation_hash_1,
        "observation_stream_hash_2": observation_hash_2,
        "episode_fingerprint": episode_fingerprint(episode),
    }


def attempt_provenance(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    attempt_id: str,
    row: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "git_commit": manifest["git_commit"],
        "scientific_baseline_commit": manifest["scientific_baseline_commit"],
        "scientific_config_fingerprint": manifest["scientific_config_fingerprint"],
        "task_id": task["task_id"],
        "task_key": task["task_key"],
        "episode_index": task["episode_index"],
        "task_metadata_fingerprint": task["task_metadata_fingerprint"],
        "scientific_command_fingerprint": task["scientific_command_fingerprint"],
        "frozen_episode_csv_sha256": task["frozen_episode_csv_sha256"],
        "attempt_id": attempt_id,
        "scientific_row_fingerprint": scientific_row_fingerprint(row),
    }


def run_manifest_task(
    manifest: Mapping[str, object], task_id: int, attempt_id: str
) -> Path:
    validate_manifest_structure(manifest)
    require_execution_checkout(manifest)
    task = manifest_task(manifest, task_id)
    metadata_path = Path(str(task["metadata_path"]))
    metadata = read_json(metadata_path)
    if fingerprint(metadata) != task["task_metadata_fingerprint"]:
        raise RuntimeError("Frozen task metadata fingerprint mismatch")
    attempts_dir = Path(str(task["attempts_dir"]))
    canonical_root = Path(str(manifest["canonical_run_dir"])).resolve()
    if canonical_root == attempts_dir.resolve() or canonical_root in attempts_dir.resolve().parents:
        raise RuntimeError("Episode attempt path is inside the canonical run")

    attempt_id = sanitize_attempt_id(attempt_id)
    final_dir = attempts_dir / attempt_id
    if final_dir.exists():
        state, errors, _ = validate_attempt(manifest, task, final_dir)
        if state == "valid":
            return final_dir
        raise RuntimeError(f"Existing attempt is not valid: {errors}")
    temporary = attempts_dir / f".{attempt_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        row = run_single_episode_from_task_metadata(metadata, int(task["episode_index"]))
        string_row = {field: str(row.get(field, "")) for field in EPISODE_FIELDNAMES}
        write_csv_atomic(temporary / EPISODE_FILENAME, EPISODE_FIELDNAMES, [string_row])
        provenance = attempt_provenance(manifest, task, attempt_id, string_row)
        provenance["episode_csv_sha256"] = file_sha256(temporary / EPISODE_FILENAME)
        write_json_atomic(temporary / "attempt_provenance.json", provenance)
        attempts_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final_dir)
    except Exception as error:
        write_json_atomic(
            temporary / "attempt_failure.json",
            {"task_id": task_id, "attempt_id": attempt_id, "error": str(error)},
        )
        failed_dir = attempts_dir / f"{attempt_id}.failed.{uuid.uuid4().hex}"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, failed_dir)
        raise
    return final_dir


def validate_attempt(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    attempt_dir: Path,
) -> tuple[str, List[str], Dict[str, str] | None]:
    errors: List[str] = []
    episode_path = attempt_dir / EPISODE_FILENAME
    provenance_path = attempt_dir / "attempt_provenance.json"
    if not episode_path.is_file() or not provenance_path.is_file():
        return "failed", ["required attempt files missing"], None
    try:
        fieldnames, rows = read_csv(episode_path)
        provenance = read_json(provenance_path)
    except Exception as error:
        return "invalid", [f"attempt unreadable: {error}"], None
    if fieldnames != EPISODE_FIELDNAMES or len(rows) != 1:
        errors.append("attempt must contain exactly one canonical episode row")
        return "invalid", errors, rows[0] if rows else None
    row = rows[0]
    expected_provenance = {
        "workflow": WORKFLOW,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "git_commit": manifest["git_commit"],
        "scientific_baseline_commit": manifest["scientific_baseline_commit"],
        "scientific_config_fingerprint": manifest["scientific_config_fingerprint"],
        "task_id": task["task_id"],
        "task_key": task["task_key"],
        "episode_index": task["episode_index"],
        "task_metadata_fingerprint": task["task_metadata_fingerprint"],
        "scientific_command_fingerprint": task["scientific_command_fingerprint"],
        "frozen_episode_csv_sha256": task["frozen_episode_csv_sha256"],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            errors.append(f"provenance mismatch: {key}")
    if provenance.get("episode_csv_sha256") != file_sha256(episode_path):
        errors.append("episode CSV hash mismatch")
    if provenance.get("scientific_row_fingerprint") != scientific_row_fingerprint(row):
        errors.append("scientific row fingerprint mismatch")
    if row.get("environment") != task["environment"]:
        errors.append("environment mismatch")
    if row.get("policy") != task["policy"]:
        errors.append("policy mismatch")
    if row.get("episode_index") != str(task["episode_index"]):
        errors.append("episode index mismatch")
    metadata = read_json(Path(str(task["metadata_path"])))
    identity = expected_identity(metadata, int(task["episode_index"]))
    for key in ("observation_stream_hash_1", "observation_stream_hash_2", "episode_fingerprint"):
        if row.get(key) != str(identity[key]):
            errors.append(f"deterministic identity mismatch: {key}")
    for key in ("true_need_1", "true_need_2"):
        try:
            if float(row[key]) != float(identity[key]):
                errors.append(f"deterministic identity mismatch: {key}")
        except (KeyError, ValueError):
            errors.append(f"invalid deterministic identity field: {key}")
    return ("invalid", errors, row) if errors else ("valid", [], row)


@dataclass(frozen=True)
class ShardValidation:
    selected: Dict[int, Dict[str, str]]
    selected_attempts: Dict[int, str]
    missing: List[int]
    failed: List[int]
    invalid: Dict[int, List[str]]
    duplicates: Dict[int, List[str]]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.failed and not self.invalid and not self.duplicates


def validate_shards(manifest: Mapping[str, object]) -> ShardValidation:
    validate_manifest_structure(manifest)
    selected: Dict[int, Dict[str, str]] = {}
    selected_attempts: Dict[int, str] = {}
    missing: List[int] = []
    failed: List[int] = []
    invalid: Dict[int, List[str]] = {}
    duplicates: Dict[int, List[str]] = {}
    for task in manifest["tasks"]:
        task_id = int(task["task_id"])
        attempts_dir = Path(str(task["attempts_dir"]))
        if not attempts_dir.is_dir():
            missing.append(task_id)
            continue
        valid_rows: List[tuple[str, Dict[str, str]]] = []
        errors: List[str] = []
        saw_failed = False
        for attempt_dir in sorted(item for item in attempts_dir.iterdir() if item.is_dir()):
            state, attempt_errors, row = validate_attempt(manifest, task, attempt_dir)
            if state == "valid" and row is not None:
                valid_rows.append((attempt_dir.name, row))
            elif state == "failed":
                saw_failed = True
            else:
                errors.extend(f"{attempt_dir.name}: {value}" for value in attempt_errors)
        if errors:
            invalid[task_id] = errors
            continue
        if not valid_rows:
            (failed if saw_failed else missing).append(task_id)
            continue
        scientific_fingerprints = {
            scientific_row_fingerprint(row) for _, row in valid_rows
        }
        if len(scientific_fingerprints) != 1:
            duplicates[task_id] = [attempt_id for attempt_id, _ in valid_rows]
            continue
        attempt_id, row = sorted(valid_rows, key=lambda value: value[0])[0]
        selected[task_id] = row
        selected_attempts[task_id] = attempt_id
    return ShardValidation(selected, selected_attempts, missing, failed, invalid, duplicates)


def valid_attempt_count(
    manifest: Mapping[str, object], task: Mapping[str, object]
) -> int:
    attempts_dir = Path(str(task["attempts_dir"]))
    if not attempts_dir.is_dir():
        return 0
    return sum(
        1
        for attempt_dir in attempts_dir.iterdir()
        if attempt_dir.is_dir()
        and validate_attempt(manifest, task, attempt_dir)[0] == "valid"
    )


def validate_stage_evidence(
    manifest: Mapping[str, object], *, require_stage_tree: bool = True
) -> Dict[str, object]:
    path = Path(str(manifest["workflow_run_dir"])) / "stage_validation.json"
    if not path.is_file():
        raise RuntimeError("Strict stage-validation evidence is missing")
    evidence = read_json(path)
    payload = {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}
    if evidence.get("evidence_fingerprint") != fingerprint(payload):
        raise RuntimeError("Stage-validation evidence fingerprint mismatch")
    if evidence.get("manifest_fingerprint") != manifest.get("manifest_fingerprint"):
        raise RuntimeError("Stage-validation evidence belongs to another manifest")
    inventory = list(evidence.get("stage_inventory", []))
    if fingerprint(inventory) != evidence.get("stage_tree_fingerprint"):
        raise RuntimeError("Stage-validation inventory fingerprint mismatch")
    if require_stage_tree:
        stage_root = Path(str(manifest["workflow_run_dir"])) / STAGE_DIRNAME
        if not stage_root.is_dir() or tree_inventory(stage_root) != inventory:
            raise RuntimeError("Validated stage tree is missing or changed")
    return evidence


def build_stage_evidence(
    manifest_fingerprint: str,
    command: Sequence[str],
    stage_inventory: Sequence[Mapping[str, object]],
    selected_attempts: Mapping[int, str],
) -> Dict[str, object]:
    """Build JSON-stable evidence for a fully validated staging tree."""

    evidence: Dict[str, object] = {
        "manifest_fingerprint": manifest_fingerprint,
        "global_combiner_command": list(command),
        "stage_tree_fingerprint": fingerprint(stage_inventory),
        "stage_inventory": list(stage_inventory),
        # JSON object keys are strings. Normalize before hashing so task IDs >= 10
        # cannot change sort order during the write/read round trip.
        "selected_attempts": {
            str(task_id): attempt_id
            for task_id, attempt_id in selected_attempts.items()
        },
    }
    evidence["evidence_fingerprint"] = fingerprint(evidence)
    return evidence


def validate_negative_evidence(
    manifest: Mapping[str, object], path: Path
) -> Dict[str, object]:
    negative = read_json(path)
    payload = {key: value for key, value in negative.items() if key != "evidence_fingerprint"}
    if negative.get("evidence_fingerprint") != fingerprint(payload):
        raise RuntimeError("Negative smoke evidence fingerprint mismatch")
    if negative.get("manifest_fingerprint") != manifest.get("manifest_fingerprint"):
        raise RuntimeError("Negative smoke evidence belongs to another manifest")
    required_negative = {
        "collector_failed_nonzero": True,
        "simulation_invocations": 0,
        "promotion_occurred": False,
        "completion_marker_created": False,
        "canonical_hash_unchanged": True,
        "isolated_shard_restored": True,
    }
    for key, expected in required_negative.items():
        if negative.get(key) != expected:
            raise RuntimeError(f"Negative smoke evidence failed requirement: {key}")
    if negative.get("case") not in {"missing_shard", "corrupt_shard"}:
        raise RuntimeError("Negative smoke evidence must name a missing or corrupt shard case")
    return negative


def record_scheduler_smoke_evidence(
    manifest: Mapping[str, object],
    qstat_path: Path,
    qacct_specs: Sequence[str],
    lane_job_ids: Sequence[str],
    collector_job_id: str,
    collector_qacct_path: Path,
    output_path: Path,
) -> Dict[str, object]:
    lane_job_id_set = {str(value) for value in lane_job_ids}
    nonempty_lanes = [lane for lane in manifest["lanes"] if lane.get("task_ids")]
    if len(nonempty_lanes) != len(lane_job_ids):
        raise RuntimeError("Lane job IDs do not cover the non-empty manifest lanes")
    lane_by_job_id = {
        str(job_id): lane for job_id, lane in zip(lane_job_ids, nonempty_lanes)
    }
    qacct_records: List[Dict[str, object]] = []
    for spec in qacct_specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise RuntimeError("qacct specs must be task_id:job_id:path")
        task_id, job_id, raw_path = parts
        path = Path(raw_path)
        text_value = path.read_text(encoding="utf-8")
        claimed_parts = str(job_id).split(".", 1)
        if len(claimed_parts) != 2 or not claimed_parts[1].isdigit():
            raise RuntimeError("Lane qacct job IDs must use jobnumber.taskid")
        claimed_jobnumber, claimed_taskid = claimed_parts
        if claimed_jobnumber not in lane_job_id_set:
            raise RuntimeError("Lane qacct jobnumber is not a submitted lane job")
        jobnumber_match = re.search(r"(?m)^jobnumber\s+(\S+)\s*$", text_value)
        taskid_match = re.search(r"(?m)^taskid\s+(\S+)\s*$", text_value)
        slots_match = re.search(r"(?m)^slots\s+(\d+)\s*$", text_value)
        exit_match = re.search(r"(?m)^exit_status\s+(-?\d+)\s*$", text_value)
        pe_match = re.search(r"(?m)^granted_pe\s+(\S+)\s*$", text_value)
        if None in (jobnumber_match, taskid_match, slots_match, exit_match, pe_match):
            raise RuntimeError(
                f"qacct evidence lacks jobnumber/taskid/slots/exit_status/granted_pe: {path}"
            )
        if (
            jobnumber_match.group(1) != claimed_jobnumber
            or taskid_match.group(1) != claimed_taskid
        ):
            raise RuntimeError("Lane qacct jobnumber/taskid does not match the claimed task")
        lane = lane_by_job_id[claimed_jobnumber]
        slot = int(claimed_taskid)
        lane_task_ids = [int(value) for value in lane["task_ids"]]
        if slot < 1 or slot > len(lane_task_ids) or lane_task_ids[slot - 1] != int(task_id):
            raise RuntimeError("Lane qacct slot does not map to the claimed manifest task")
        qacct_records.append(
            {
                "task_id": int(task_id),
                "job_id": str(job_id),
                "scheduler_jobnumber": jobnumber_match.group(1),
                "scheduler_taskid": taskid_match.group(1),
                "lane_id": int(lane["lane_id"]),
                "slots": int(slots_match.group(1)),
                "exit_status": int(exit_match.group(1)),
                "granted_pe": pe_match.group(1),
                "file": evidence_file_record(path),
            }
        )
    collector_text = collector_qacct_path.read_text(encoding="utf-8")
    collector_jobnumber = re.search(r"(?m)^jobnumber\s+(\S+)\s*$", collector_text)
    collector_taskid = re.search(r"(?m)^taskid\s+(\S+)\s*$", collector_text)
    collector_slots = re.search(r"(?m)^slots\s+(\d+)\s*$", collector_text)
    collector_exit = re.search(r"(?m)^exit_status\s+(-?\d+)\s*$", collector_text)
    collector_pe = re.search(r"(?m)^granted_pe\s+(\S+)\s*$", collector_text)
    if None in (collector_jobnumber, collector_slots, collector_exit, collector_pe):
        raise RuntimeError("Collector qacct evidence lacks required scheduler fields")
    if collector_jobnumber.group(1) != str(collector_job_id):
        raise RuntimeError("Collector qacct jobnumber does not match the claimed collector")
    if collector_taskid is not None and collector_taskid.group(1).lower() not in {
        "undefined",
        "none",
        "0",
    }:
        raise RuntimeError("Collector qacct unexpectedly identifies an array task")
    collector_record = {
        "job_id": str(collector_job_id),
        "scheduler_jobnumber": collector_jobnumber.group(1),
        "scheduler_taskid": collector_taskid.group(1) if collector_taskid else "undefined",
        "slots": int(collector_slots.group(1)),
        "exit_status": int(collector_exit.group(1)),
        "granted_pe": collector_pe.group(1),
        "file": evidence_file_record(collector_qacct_path),
    }
    payload = {
        "schema_version": 1,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "git_commit": manifest["reviewed_commit"],
        "one_slot_tasks": True,
        "requested_slots_per_task": 1,
        "shared_memory_pe": False,
        "lane_throttle": manifest["lane_throttle"],
        "lane_job_ids": list(lane_job_ids),
        "collector_job_id": str(collector_job_id),
        "collector_qacct": collector_record,
        "qstat": evidence_file_record(qstat_path),
        "qacct_records": qacct_records,
    }
    payload["evidence_fingerprint"] = fingerprint(payload)
    write_json_atomic(output_path, payload)
    return payload


def validate_scheduler_smoke_evidence(
    manifest: Mapping[str, object], path: Path
) -> Dict[str, object]:
    evidence = read_json(path)
    payload = {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}
    if evidence.get("evidence_fingerprint") != fingerprint(payload):
        raise RuntimeError("Scheduler smoke evidence fingerprint mismatch")
    if evidence.get("manifest_fingerprint") != manifest.get("manifest_fingerprint"):
        raise RuntimeError("Scheduler smoke evidence belongs to another manifest")
    if evidence.get("git_commit") != manifest.get("reviewed_commit"):
        raise RuntimeError("Scheduler smoke evidence has the wrong reviewed commit")
    if evidence.get("one_slot_tasks") is not True or int(
        evidence.get("requested_slots_per_task", 0)
    ) != 1:
        raise RuntimeError("Scheduler smoke evidence does not prove one-slot tasks")
    if evidence.get("shared_memory_pe") is not False:
        raise RuntimeError("Scheduler smoke evidence unexpectedly used a shared-memory PE")
    if int(evidence.get("lane_throttle", 0)) > 100:
        raise RuntimeError("Scheduler smoke evidence exceeds max_aj_instances")
    if not evidence.get("lane_job_ids") or not evidence.get("collector_job_id"):
        raise RuntimeError("Scheduler smoke evidence lacks lane or collector job IDs")
    lane_job_ids = {str(value) for value in evidence["lane_job_ids"]}
    ordered_lane_job_ids = [str(value) for value in evidence["lane_job_ids"]]
    nonempty_lanes = [lane for lane in manifest["lanes"] if lane.get("task_ids")]
    if len(nonempty_lanes) != len(ordered_lane_job_ids):
        raise RuntimeError("Scheduler evidence does not cover the non-empty manifest lanes")
    lane_by_job_id = {
        job_id: lane for job_id, lane in zip(ordered_lane_job_ids, nonempty_lanes)
    }
    validate_evidence_file(dict(evidence.get("qstat", {})), "smoke qstat")
    qacct_records = list(evidence.get("qacct_records", []))
    if len(qacct_records) < 2:
        raise RuntimeError("Scheduler smoke evidence needs qacct for two blinkered tasks")
    for index, record in enumerate(qacct_records):
        path_value = validate_evidence_file(
            dict(record.get("file", {})), f"smoke qacct {index}"
        )
        if int(record.get("slots", 0)) != 1 or int(record.get("exit_status", -1)) != 0:
            raise RuntimeError("Scheduler qacct does not prove a successful one-slot task")
        if str(record.get("granted_pe", "")).upper() not in {"NONE", "UNDEFINED"}:
            raise RuntimeError("Scheduler qacct shows a shared-memory PE")
        qacct_text = path_value.read_text(encoding="utf-8")
        jobnumber_match = re.search(r"(?m)^jobnumber\s+(\S+)\s*$", qacct_text)
        taskid_match = re.search(r"(?m)^taskid\s+(\S+)\s*$", qacct_text)
        claimed_parts = str(record.get("job_id", "")).split(".", 1)
        if (
            len(claimed_parts) != 2
            or claimed_parts[0] not in lane_job_ids
            or jobnumber_match is None
            or taskid_match is None
            or jobnumber_match.group(1) != claimed_parts[0]
            or taskid_match.group(1) != claimed_parts[1]
            or record.get("scheduler_jobnumber") != claimed_parts[0]
            or str(record.get("scheduler_taskid")) != claimed_parts[1]
        ):
            raise RuntimeError("Scheduler qacct jobnumber/taskid identity mismatch")
        lane = lane_by_job_id[claimed_parts[0]]
        scheduler_slot = int(claimed_parts[1])
        lane_task_ids = [int(value) for value in lane["task_ids"]]
        if (
            scheduler_slot < 1
            or scheduler_slot > len(lane_task_ids)
            or lane_task_ids[scheduler_slot - 1] != int(record.get("task_id", -1))
            or int(record.get("lane_id", -1)) != int(lane["lane_id"])
        ):
            raise RuntimeError("Scheduler qacct slot-to-manifest-task mapping mismatch")
        if not re.search(r"(?m)^slots\s+1\s*$", qacct_text) or not re.search(
            r"(?m)^exit_status\s+0\s*$", qacct_text
        ):
            raise RuntimeError("Scheduler qacct file does not match declared resources")
        if not re.search(r"(?mi)^granted_pe\s+(NONE|UNDEFINED)\s*$", qacct_text):
            raise RuntimeError("Scheduler qacct file does not prove absence of a PE")
    collector = dict(evidence.get("collector_qacct", {}))
    collector_path = validate_evidence_file(
        dict(collector.get("file", {})), "smoke collector qacct"
    )
    collector_text = collector_path.read_text(encoding="utf-8")
    collector_jobnumber = re.search(r"(?m)^jobnumber\s+(\S+)\s*$", collector_text)
    if (
        collector_jobnumber is None
        or collector_jobnumber.group(1) != str(evidence["collector_job_id"])
        or collector.get("scheduler_jobnumber") != str(evidence["collector_job_id"])
        or int(collector.get("slots", 0)) != 1
        or int(collector.get("exit_status", -1)) != 0
        or str(collector.get("granted_pe", "")).upper() not in {"NONE", "UNDEFINED"}
    ):
        raise RuntimeError("Collector qacct identity or resource evidence mismatch")
    blinkered_task_ids = {
        int(task["task_id"])
        for task in manifest["tasks"]
        if dict(read_json(Path(str(task["metadata_path"]))).get("policy", {})).get("class")
        == "BlinkeredPolicy"
    }
    evidenced_ids = {int(record.get("task_id", -1)) for record in qacct_records}
    if len(blinkered_task_ids & evidenced_ids) < 2:
        raise RuntimeError("Scheduler qacct evidence is not tied to two blinkered tasks")
    return evidence


def validate_smoke_serial_reference(
    manifest: Mapping[str, object],
    validation: ShardValidation,
) -> Dict[str, object]:
    serial_reference = validate_manifest_serial_reference(manifest)
    if serial_reference is None:
        raise RuntimeError("Smoke manifest does not bind a serial reference")
    serial_reference_path = Path(str(serial_reference["path"]))
    reference_rows, references = read_serial_reference_rows(serial_reference_path)

    matches: List[Dict[str, object]] = []
    for task in sorted(manifest["tasks"], key=lambda value: int(value["task_id"])):
        task_id = int(task["task_id"])
        shard_row = validation.selected.get(task_id)
        if shard_row is None:
            raise RuntimeError(f"Smoke shard {task_id} is not valid and selected")
        key = (
            str(task["environment"]),
            str(task["policy"]),
            int(task["episode_index"]),
        )
        reference_row = references.get(key)
        if reference_row is None:
            raise RuntimeError(f"Serial reference is missing smoke shard identity: {key}")
        mismatched_fields = [
            field
            for field in SERIAL_REFERENCE_FIELDNAMES
            if shard_row.get(field) != reference_row.get(field)
        ]
        if mismatched_fields:
            raise RuntimeError(
                f"Serial reference mismatch for smoke task {task_id}: {mismatched_fields}"
            )
        shard_scientific = {
            field: shard_row[field] for field in SERIAL_REFERENCE_FIELDNAMES
        }
        reference_scientific = {
            field: reference_row[field] for field in SERIAL_REFERENCE_FIELDNAMES
        }
        matches.append(
            {
                "task_id": task_id,
                "environment": key[0],
                "policy": key[1],
                "episode_index": key[2],
                "selected_attempt": validation.selected_attempts[task_id],
                "matching_field_count": len(SERIAL_REFERENCE_FIELDNAMES),
                "shard_scientific_row_fingerprint": fingerprint(shard_scientific),
                "serial_reference_row_fingerprint": fingerprint(reference_scientific),
            }
        )

    evidence: Dict[str, object] = {
        "compared_fields": list(SERIAL_REFERENCE_FIELDNAMES),
        "reference_row_count": len(reference_rows),
        "matched_shard_count": len(matches),
        "matches": matches,
    }
    evidence["evidence_fingerprint"] = fingerprint(evidence)
    return evidence


def certify_smoke(
    manifest: Mapping[str, object], negative_evidence_path: Path,
    scheduler_evidence_path: Path, output_path: Path
) -> Dict[str, object]:
    require_execution_checkout(manifest)
    if manifest.get("run_mode") != "smoke":
        raise RuntimeError("Smoke certification requires a smoke-mode manifest")
    validation = validate_shards(manifest)
    if not validation.ok or len(manifest["tasks"]) < 2:
        raise RuntimeError("Smoke certification requires at least two valid episode shards")
    blinkered_tasks = []
    for task in manifest["tasks"]:
        metadata = read_json(Path(str(task["metadata_path"])))
        policy = dict(metadata.get("policy", {}))
        if (
            policy.get("class") == "BlinkeredPolicy"
            and int(policy.get("observation_draws", -1)) == 250
            and int(policy.get("horizon", -1)) == 2
        ):
            blinkered_tasks.append(int(task["task_id"]))
    if len(blinkered_tasks) < 2:
        raise RuntimeError("Smoke certification requires at least two blinkered-250 shards")
    retried = [
        int(task["task_id"])
        for task in manifest["tasks"]
        if valid_attempt_count(manifest, task) >= 2
    ]
    if not retried:
        raise RuntimeError("Smoke certification requires an equivalent successful retry")
    stage = validate_stage_evidence(manifest)
    validate_negative_evidence(manifest, negative_evidence_path)
    scheduler = validate_scheduler_smoke_evidence(manifest, scheduler_evidence_path)
    serial_reference = validate_manifest_serial_reference(manifest)
    if serial_reference is None:
        raise RuntimeError("Smoke manifest does not bind a serial reference")
    serial_parity = validate_smoke_serial_reference(manifest, validation)
    payload = {
        "schema_version": 2,
        "smoke_manifest_path": str(
            (Path(str(manifest["workflow_run_dir"])) / MANIFEST_FILENAME).resolve()
        ),
        "smoke_manifest_sha256": file_sha256(
            Path(str(manifest["workflow_run_dir"])) / MANIFEST_FILENAME
        ),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "git_commit": manifest["git_commit"],
        "reviewed_commit": manifest["reviewed_commit"],
        "scientific_baseline_commit": manifest["scientific_baseline_commit"],
        "scientific_config_fingerprint": manifest["scientific_config_fingerprint"],
        "execution_code_fingerprint": manifest["execution_code_fingerprint"],
        "smoke_passed": True,
        "retry_determinism_passed": True,
        "negative_case_passed": True,
        "serial_reference_parity_passed": True,
        "validated_shards": len(validation.selected),
        "validated_blinkered_250_task_ids": blinkered_tasks,
        "retried_task_ids": retried,
        "stage_validation_fingerprint": stage["evidence_fingerprint"],
        "stage_validation_sha256": file_sha256(
            Path(str(manifest["workflow_run_dir"])) / "stage_validation.json"
        ),
        "negative_evidence_sha256": file_sha256(negative_evidence_path),
        "scheduler_evidence_sha256": file_sha256(scheduler_evidence_path),
        "scheduler_evidence_fingerprint": scheduler["evidence_fingerprint"],
        "negative_evidence_path": str(negative_evidence_path.resolve()),
        "scheduler_evidence_path": str(scheduler_evidence_path.resolve()),
        "serial_reference_path": serial_reference["path"],
        "serial_reference_size": serial_reference["size"],
        "serial_reference_sha256": serial_reference["sha256"],
        "serial_reference_target_rows_fingerprint": serial_reference[
            "target_rows_fingerprint"
        ],
        "serial_reference_parity_evidence": serial_parity,
    }
    payload["gate_fingerprint"] = fingerprint(payload)
    write_json_atomic(output_path, payload)
    return payload


def verify_smoke_gate(path: Path, reviewed_commit: str) -> Dict[str, object]:
    gate = read_json(path)
    payload = {key: value for key, value in gate.items() if key != "gate_fingerprint"}
    if gate.get("gate_fingerprint") != fingerprint(payload):
        raise RuntimeError("Smoke gate fingerprint mismatch")
    if gate.get("reviewed_commit") != reviewed_commit or current_git_commit() != reviewed_commit:
        raise RuntimeError("Smoke gate is not bound to the executing reviewed commit")
    if gate.get("scientific_baseline_commit") != SCIENTIFIC_BASELINE_COMMIT:
        raise RuntimeError("Smoke gate scientific baseline mismatch")
    for name in (
        "smoke_passed",
        "retry_determinism_passed",
        "negative_case_passed",
        "serial_reference_parity_passed",
    ):
        if gate.get(name) is not True:
            raise RuntimeError(f"Smoke gate did not pass: {name}")
    if len(gate.get("validated_blinkered_250_task_ids", [])) < 2:
        raise RuntimeError("Smoke gate lacks two blinkered-250 task identities")
    if not gate.get("retried_task_ids"):
        raise RuntimeError("Smoke gate lacks retry-equivalence evidence")
    manifest_path = Path(str(gate.get("smoke_manifest_path", "")))
    if not manifest_path.is_file() or file_sha256(manifest_path) != gate.get("smoke_manifest_sha256"):
        raise RuntimeError("Smoke gate manifest file is missing or changed")
    manifest = load_manifest(manifest_path)
    if manifest.get("manifest_fingerprint") != gate.get("manifest_fingerprint"):
        raise RuntimeError("Smoke gate manifest ownership mismatch")
    if manifest.get("run_mode") != "smoke":
        raise RuntimeError("Smoke gate manifest is not smoke mode")
    if manifest.get("scientific_config_fingerprint") != gate.get(
        "scientific_config_fingerprint"
    ) or manifest.get("execution_code_fingerprint") != gate.get(
        "execution_code_fingerprint"
    ):
        raise RuntimeError("Smoke gate configuration or execution-code binding mismatch")
    require_execution_checkout(manifest)
    serial_reference = validate_manifest_serial_reference(manifest)
    if serial_reference is None:
        raise RuntimeError("Smoke manifest does not bind a serial reference")
    serial_reference_path = Path(str(serial_reference["path"]))
    if (
        gate.get("serial_reference_path") != serial_reference["path"]
        or int(gate.get("serial_reference_size", -1)) != serial_reference["size"]
        or gate.get("serial_reference_sha256") != serial_reference["sha256"]
        or gate.get("serial_reference_target_rows_fingerprint")
        != serial_reference["target_rows_fingerprint"]
    ):
        raise RuntimeError("Smoke gate serial reference does not match the manifest binding")
    validation = validate_shards(manifest)
    if not validation.ok:
        raise RuntimeError("Smoke gate shards are no longer valid")
    serial_parity = validate_smoke_serial_reference(manifest, validation)
    if serial_parity != gate.get("serial_reference_parity_evidence"):
        raise RuntimeError("Smoke gate serial-reference parity evidence mismatch")
    if int(gate.get("validated_shards", -1)) != len(validation.selected):
        raise RuntimeError("Smoke gate validated shard count mismatch")
    stage_path = Path(str(manifest["workflow_run_dir"])) / "stage_validation.json"
    if file_sha256(stage_path) != gate.get("stage_validation_sha256"):
        raise RuntimeError("Smoke gate stage evidence changed")
    negative_path = Path(str(gate["negative_evidence_path"]))
    scheduler_path = Path(str(gate["scheduler_evidence_path"]))
    if file_sha256(negative_path) != gate.get("negative_evidence_sha256"):
        raise RuntimeError("Smoke gate negative evidence changed")
    if file_sha256(scheduler_path) != gate.get("scheduler_evidence_sha256"):
        raise RuntimeError("Smoke gate scheduler evidence changed")
    validate_stage_evidence(manifest)
    validate_negative_evidence(manifest, negative_path)
    validate_scheduler_smoke_evidence(manifest, scheduler_path)
    return gate


def run_missing_shard_negative_check(
    manifest: Mapping[str, object], task_id: int, output_path: Path
) -> Dict[str, object]:
    """Prove strict collection fails on a missing isolated shard without simulation."""

    task = manifest_task(manifest, task_id)
    attempts_dir = Path(str(task["attempts_dir"]))
    if not attempts_dir.is_dir():
        raise RuntimeError("Negative check needs an existing valid shard")
    held_dir = attempts_dir.with_name(f".{attempts_dir.name}.negative_hold.{uuid.uuid4().hex}")
    canonical = Path(str(manifest["canonical_run_dir"]))
    canonical_before = tree_fingerprint(canonical)
    marker_before = completion_marker_path(manifest).exists()
    journal_before = (
        file_sha256(journal_path(manifest)) if journal_path(manifest).is_file() else ""
    )
    error = ""
    os.replace(attempts_dir, held_dir)
    try:
        try:
            build_staged_complete_view(manifest)
        except RuntimeError as caught:
            error = str(caught)
        else:
            raise RuntimeError("Strict collector unexpectedly accepted a missing shard")
    finally:
        os.replace(held_dir, attempts_dir)
    canonical_after = tree_fingerprint(canonical)
    marker_after = completion_marker_path(manifest).exists()
    journal_after = (
        file_sha256(journal_path(manifest)) if journal_path(manifest).is_file() else ""
    )
    payload = {
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "case": "missing_shard",
        "task_id": task_id,
        "collector_failed_nonzero": bool(error),
        "collector_error": error,
        "simulation_invocations": 0,
        "promotion_occurred": journal_after != journal_before,
        "completion_marker_created": marker_after and not marker_before,
        "canonical_hash_unchanged": canonical_before == canonical_after,
        "isolated_shard_restored": attempts_dir.is_dir(),
    }
    if not payload["canonical_hash_unchanged"] or not payload["isolated_shard_restored"]:
        raise RuntimeError("Negative check did not restore isolated state exactly")
    payload["evidence_fingerprint"] = fingerprint(payload)
    write_json_atomic(output_path, payload)
    return payload


def validate_snapshot(manifest: Mapping[str, object]) -> None:
    snapshot_root = Path(str(manifest["snapshot_root"]))
    actual = tree_inventory(snapshot_root)
    if actual != manifest["snapshot_inventory"]:
        raise RuntimeError("Frozen snapshot inventory mismatch")
    if fingerprint(actual) != manifest["snapshot_tree_fingerprint"]:
        raise RuntimeError("Frozen snapshot tree fingerprint mismatch")
    task_manifest_path = Path(str(manifest["task_manifest_path"]))
    if file_sha256(task_manifest_path) != manifest["task_manifest_sha256"]:
        raise RuntimeError("Frozen task manifest hash mismatch")
    writer_path = Path(str(manifest["writer_evidence_path"]))
    if not writer_path.is_file() or file_sha256(writer_path) != manifest["writer_evidence_sha256"]:
        raise RuntimeError("Frozen writer evidence hash mismatch")
    validate_writer_evidence(
        read_json(writer_path), list(manifest.get("required_writer_job_ids", []))
    )


class ExclusiveLock:
    def __init__(self, path: Path, manifest_fingerprint: str = ""):
        self.path = path
        self.manifest_fingerprint = manifest_fingerprint
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "ExclusiveLock":
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(f"Collector lock is already held: {self.path}") from error
        write_json_atomic(
            self.path / "owner.json",
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "scheduler_job_id": os.environ.get("JOB_ID", ""),
                "manifest_fingerprint": self.manifest_fingerprint,
                "created_unix": time.time(),
                "token": self.token,
            },
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        owner_path = self.path / "owner.json"
        try:
            owner = read_json(owner_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if owner.get("token") == self.token:
            shutil.rmtree(self.path, ignore_errors=True)


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reclaim_stale_lock(
    lock_path: Path,
    manifest_fingerprint: str,
    scheduler_evidence_path: Path | None = None,
) -> Dict[str, object]:
    owner_path = lock_path / "owner.json"
    if not owner_path.is_file():
        raise RuntimeError("Collector lock has no verifiable owner record")
    owner = read_json(owner_path)
    if owner.get("manifest_fingerprint") != manifest_fingerprint:
        raise RuntimeError("Collector lock belongs to another manifest")
    owner_host = str(owner.get("host", ""))
    owner_pid = int(owner.get("pid", 0))
    if owner_host == socket.gethostname():
        if owner_pid > 0 and process_is_alive(owner_pid):
            raise RuntimeError("Collector lock owner may still be alive")
    else:
        if scheduler_evidence_path is None:
            raise RuntimeError("Remote collector lock needs scheduler stale-lock evidence")
        evidence = read_json(scheduler_evidence_path)
        payload = {key: value for key, value in evidence.items() if key != "evidence_fingerprint"}
        if evidence.get("evidence_fingerprint") != fingerprint(payload):
            raise RuntimeError("Stale-lock scheduler evidence fingerprint mismatch")
        if evidence.get("lock_token") != owner.get("token"):
            raise RuntimeError("Stale-lock evidence belongs to another lock owner")
        scheduler_job_id = str(owner.get("scheduler_job_id", ""))
        if not scheduler_job_id or evidence.get("scheduler_job_id") != scheduler_job_id:
            raise RuntimeError("Stale-lock evidence does not identify the owner scheduler job")
        if evidence.get("owner_job_absent") is not True:
            raise RuntimeError("Stale-lock evidence does not prove scheduler absence")
        qstat_path = validate_evidence_file(
            dict(evidence.get("qstat", {})), "stale-lock qstat"
        )
        validate_evidence_file(dict(evidence.get("qacct", {})), "stale-lock qacct")
        for line in qstat_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[0] == scheduler_job_id:
                raise RuntimeError("Stale-lock owner job remains in qstat")
    stale_path = lock_path.with_name(f".{lock_path.name}.stale.{uuid.uuid4().hex}")
    try:
        os.replace(lock_path, stale_path)
    except FileNotFoundError as error:
        raise RuntimeError("Collector lock changed during stale reclamation") from error
    shutil.rmtree(stale_path)
    return owner


def affected_by_key(manifest: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    return {str(record["task_key"]): record for record in manifest["affected_tasks"]}


def frozen_rows_for_task(
    manifest: Mapping[str, object], affected: Mapping[str, object]
) -> List[Dict[str, str]]:
    path = Path(str(manifest["snapshot_root"])) / str(affected["relative_dir"]) / EPISODE_FILENAME
    return validate_episode_rows(
        path,
        environment=str(affected["environment"]),
        policy=str(affected["policy"]),
        target_episodes=int(affected["target_episodes"]),
    )


def validate_frozen_row_preservation(
    frozen: Sequence[Mapping[str, str]], staged: Sequence[Mapping[str, str]]
) -> None:
    staged_by_index = {str(row["episode_index"]): row for row in staged}
    for row in frozen:
        episode_index = str(row["episode_index"])
        if episode_index not in staged_by_index or dict(row) != dict(staged_by_index[episode_index]):
            raise RuntimeError(f"Frozen row changed during staging: episode {episode_index}")


def run_global_combiner(manifest: Mapping[str, object], stage_root: Path) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "combine_method_comparison_results.py"),
        "--input-dir",
        str(stage_root),
        "--output-dir",
        str(stage_root),
        "--task-manifest",
        str(manifest["task_manifest_path"]),
        "--require-complete",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return command


def validate_combined_stage(manifest: Mapping[str, object], stage_root: Path) -> None:
    _, status_rows = read_csv(stage_root / "method_comparison_combine_status.csv")
    if len(status_rows) != 1:
        raise RuntimeError("Staged global combine status is missing")
    status = status_rows[0]
    if int(status["manifest_rows"]) != int(manifest["expected_method_tasks"]):
        raise RuntimeError("Staged global combine did not read the complete task manifest")
    if int(status["manifest_failures"]) != 0:
        raise RuntimeError("Staged global combine has manifest failures")
    if int(status["common_randomness_failures"]) != 0:
        raise RuntimeError("Staged global combine has common-randomness failures")
    _, summary_rows = read_csv(stage_root / SUMMARY_FILENAME)
    if len(summary_rows) != int(manifest["expected_method_tasks"]):
        raise RuntimeError("Staged global summary does not contain every method task")


def build_staged_complete_view(manifest: Mapping[str, object]) -> Path:
    require_execution_checkout(manifest)
    validate_snapshot(manifest)
    validation = validate_shards(manifest)
    if not validation.ok:
        raise RuntimeError(
            "Strict episode collection refused: "
            f"missing={validation.missing}; failed={validation.failed}; "
            f"invalid={validation.invalid}; duplicates={validation.duplicates}"
        )
    run_dir = Path(str(manifest["workflow_run_dir"]))
    lock_path = run_dir / LOCK_DIRNAME
    with ExclusiveLock(lock_path, str(manifest["manifest_fingerprint"])):
        stage_root = run_dir / STAGE_DIRNAME
        if stage_root.exists():
            shutil.rmtree(stage_root)
        temporary = stage_root.with_name(f".{stage_root.name}.{uuid.uuid4().hex}.tmp")
        shutil.copytree(Path(str(manifest["snapshot_root"])), temporary)
        set_tree_writable(temporary)
        affected_records = affected_by_key(manifest)
        selected_by_key: Dict[str, List[Dict[str, str]]] = {}
        for task in manifest["tasks"]:
            selected_by_key.setdefault(str(task["task_key"]), []).append(
                validation.selected[int(task["task_id"])]
            )
        for key, affected in affected_records.items():
            frozen_rows = frozen_rows_for_task(manifest, affected)
            shard_rows = selected_by_key.get(key, [])
            merged = [*frozen_rows, *shard_rows]
            indices = [int(row["episode_index"]) for row in merged]
            target = int(affected["target_episodes"])
            if sorted(indices) != list(range(target)) or len(indices) != len(set(indices)):
                raise RuntimeError(f"Merged task does not exactly cover 0..{target - 1}: {key}")
            merged.sort(key=lambda row: int(row["episode_index"]))
            validate_frozen_row_preservation(frozen_rows, merged)
            task_dir = temporary / str(affected["relative_dir"])
            write_csv_atomic(task_dir / EPISODE_FILENAME, EPISODE_FIELDNAMES, merged)
            metadata = read_json(task_dir / METADATA_FILENAME)
            write_summary_from_task_metadata(task_dir / SUMMARY_FILENAME, metadata, merged)
            _, staged_rows = read_csv(task_dir / EPISODE_FILENAME)
            validate_frozen_row_preservation(frozen_rows, staged_rows)
        command = run_global_combiner(manifest, temporary)
        validate_combined_stage(manifest, temporary)
        stage_inventory = tree_inventory(temporary)
        os.replace(temporary, stage_root)
        stage_evidence = build_stage_evidence(
            str(manifest["manifest_fingerprint"]),
            command,
            stage_inventory,
            validation.selected_attempts,
        )
        write_json_atomic(run_dir / "stage_validation.json", stage_evidence)
        return stage_root


def journal_path(manifest: Mapping[str, object]) -> Path:
    return Path(str(manifest["workflow_run_dir"])) / JOURNAL_FILENAME


def completion_marker_path(manifest: Mapping[str, object]) -> Path:
    return Path(str(manifest["canonical_run_dir"])) / COMPLETION_MARKER


def submission_journal_path(manifest: Mapping[str, object]) -> Path:
    return Path(str(manifest["workflow_run_dir"])) / SUBMISSION_JOURNAL_FILENAME


def initialize_submission(manifest: Mapping[str, object]) -> Dict[str, object]:
    path = submission_journal_path(manifest)
    if path.is_file():
        journal = read_json(path)
        if journal.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
            raise RuntimeError("Submission journal belongs to another manifest")
        if journal.get("state") in {"submitting", "submitted"}:
            return journal
        if journal.get("state") == "rollback_incomplete":
            raise RuntimeError("Previous submission rollback is not scheduler-verified")
        generation = int(journal.get("generation", 0)) + 1
    else:
        generation = 1
    journal = {
        "schema_version": 1,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "state": "submitting",
        "generation": generation,
        "lane_jobs": {},
        "collector_job_id": "",
        "events": [{"event": "initialized", "generation": generation}],
    }
    write_json_atomic(path, journal)
    return journal


def record_lane_submission(
    manifest: Mapping[str, object], lane_id: int, job_id: str
) -> Dict[str, object]:
    journal = initialize_submission(manifest)
    if journal.get("state") != "submitting":
        raise RuntimeError("Cannot add a lane to a non-submitting journal")
    valid_lanes = {int(lane["lane_id"]) for lane in manifest["lanes"] if lane["task_ids"]}
    if lane_id not in valid_lanes:
        raise RuntimeError(f"Lane {lane_id} is not a non-empty manifest lane")
    lane_jobs = dict(journal.get("lane_jobs", {}))
    existing = str(lane_jobs.get(str(lane_id), ""))
    if existing and existing != job_id:
        raise RuntimeError(f"Lane {lane_id} is already recorded as job {existing}")
    lane_jobs[str(lane_id)] = str(job_id)
    journal["lane_jobs"] = lane_jobs
    if not existing:
        journal["events"].append(
            {"event": "lane_submitted", "lane_id": lane_id, "job_id": str(job_id)}
        )
    write_json_atomic(submission_journal_path(manifest), journal)
    return journal


def record_collector_submission(
    manifest: Mapping[str, object], job_id: str
) -> Dict[str, object]:
    journal = initialize_submission(manifest)
    nonempty_lanes = {str(lane["lane_id"]) for lane in manifest["lanes"] if lane["task_ids"]}
    lane_jobs = dict(journal.get("lane_jobs", {}))
    if set(lane_jobs) != nonempty_lanes:
        raise RuntimeError("Collector cannot be recorded before every non-empty lane")
    existing = str(journal.get("collector_job_id", ""))
    if existing and existing != job_id:
        raise RuntimeError(f"Collector is already recorded as job {existing}")
    journal["collector_job_id"] = str(job_id)
    journal["state"] = "submitted"
    if not existing:
        journal["events"].append({"event": "collector_submitted", "job_id": str(job_id)})
    write_json_atomic(submission_journal_path(manifest), journal)
    scheduler = {
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "submission_generation": journal["generation"],
        "lane_job_ids": [lane_jobs[key] for key in sorted(lane_jobs, key=int)],
        "collector_job_id": str(job_id),
        "array_lanes": manifest["array_lanes"],
        "lane_throttle": manifest["lane_throttle"],
        "one_slot_tasks": True,
        "shared_memory_pe": False,
    }
    scheduler["evidence_fingerprint"] = fingerprint(scheduler)
    write_json_atomic(Path(str(manifest["workflow_run_dir"])) / SCHEDULER_FILENAME, scheduler)
    return journal


def mark_submission_rollback(
    manifest: Mapping[str, object], rollback_complete: bool, detail: str
) -> Dict[str, object]:
    path = submission_journal_path(manifest)
    if not path.is_file():
        raise RuntimeError("Submission journal is missing")
    journal = read_json(path)
    if journal.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
        raise RuntimeError("Submission journal belongs to another manifest")
    journal["state"] = "rolled_back" if rollback_complete else "rollback_incomplete"
    journal["events"].append(
        {"event": journal["state"], "detail": detail}
    )
    write_json_atomic(path, journal)
    return journal


def verify_original_canonical(manifest: Mapping[str, object]) -> None:
    canonical = Path(str(manifest["canonical_run_dir"]))
    if tree_fingerprint(canonical) != manifest["snapshot_tree_fingerprint"]:
        raise RuntimeError("Canonical run changed after checkpoint freeze")


def validate_committed_result(manifest: Mapping[str, object]) -> Dict[str, object]:
    path = journal_path(manifest)
    if not path.is_file():
        raise RuntimeError("Committed promotion journal is missing")
    journal = read_json(path)
    if journal.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
        raise RuntimeError("Promotion journal belongs to another manifest")
    if journal.get("state") != "committed":
        raise RuntimeError("Promotion journal is not committed")
    canonical = Path(str(manifest["canonical_run_dir"]))
    marker = canonical / COMPLETION_MARKER
    if not marker.is_file() or file_sha256(marker) != journal.get("completion_marker_sha256"):
        raise RuntimeError("Committed promotion marker is missing or invalid")
    stage_validation = validate_stage_evidence(manifest, require_stage_tree=False)
    inventory_without_marker = [
        record for record in tree_inventory(canonical) if record["path"] != COMPLETION_MARKER
    ]
    if fingerprint(inventory_without_marker) != stage_validation["stage_tree_fingerprint"]:
        raise RuntimeError("Committed canonical result no longer matches the validated stage")
    return journal


def _recover_promotion_unlocked(manifest: Mapping[str, object]) -> str:
    path = journal_path(manifest)
    if not path.is_file():
        return "no_journal"
    journal = read_json(path)
    if journal.get("manifest_fingerprint") != manifest["manifest_fingerprint"]:
        raise RuntimeError("Promotion journal belongs to another manifest")
    state = str(journal["state"])
    if state == "committed":
        validate_committed_result(manifest)
        return state
    canonical = Path(str(manifest["canonical_run_dir"]))
    backup = Path(str(journal["backup_root"]))
    marker = canonical / COMPLETION_MARKER
    if backup.exists():
        if canonical.exists():
            abandoned = Path(str(manifest["workflow_run_dir"])) / "promotion" / (
                f"abandoned_{uuid.uuid4().hex}"
            )
            abandoned.parent.mkdir(parents=True, exist_ok=True)
            os.replace(canonical, abandoned)
        os.replace(backup, canonical)
    if marker.exists():
        marker.unlink()
    verify_original_canonical(manifest)
    journal["state"] = "rolled_back"
    write_json_atomic(path, journal)
    return "rolled_back"


def recover_promotion(
    manifest: Mapping[str, object], stale_lock_evidence_path: Path | None = None
) -> str:
    run_dir = Path(str(manifest["workflow_run_dir"]))
    lock_path = run_dir / LOCK_DIRNAME
    if lock_path.exists():
        reclaim_stale_lock(
            lock_path,
            str(manifest["manifest_fingerprint"]),
            stale_lock_evidence_path,
        )
    with ExclusiveLock(lock_path, str(manifest["manifest_fingerprint"])):
        return _recover_promotion_unlocked(manifest)


def promote_staged_view(
    manifest: Mapping[str, object], fail_after: str = ""
) -> Path:
    require_execution_checkout(manifest)
    run_dir = Path(str(manifest["workflow_run_dir"]))
    with ExclusiveLock(run_dir / LOCK_DIRNAME, str(manifest["manifest_fingerprint"])):
        existing_journal = journal_path(manifest)
        if existing_journal.is_file():
            state = str(read_json(existing_journal)["state"])
            if state == "committed":
                _recover_promotion_unlocked(manifest)
                return completion_marker_path(manifest)
            _recover_promotion_unlocked(manifest)
        validate_snapshot(manifest)
        verify_original_canonical(manifest)
        stage_root = run_dir / STAGE_DIRNAME
        validation_path = run_dir / "stage_validation.json"
        if not stage_root.is_dir() or not validation_path.is_file():
            raise RuntimeError("Validated staged complete view is missing")
        stage_validation = validate_stage_evidence(manifest)
        actual_stage_inventory = tree_inventory(stage_root)
        if fingerprint(actual_stage_inventory) != stage_validation.get("stage_tree_fingerprint"):
            raise RuntimeError("Staged complete view fingerprint mismatch")
        canonical = Path(str(manifest["canonical_run_dir"]))
        backup = run_dir / BACKUP_DIRNAME
        if backup.exists():
            raise RuntimeError(f"Promotion backup already exists: {backup}")
        if canonical.parent.stat().st_dev != stage_root.parent.stat().st_dev:
            raise RuntimeError("Canonical and staged paths are not on the same filesystem")
        backup.parent.mkdir(parents=True, exist_ok=True)
        journal = {
            "schema_version": 1,
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "state": "prepared",
            "canonical_root": str(canonical),
            "stage_root": str(stage_root),
            "backup_root": str(backup),
            "original_tree_fingerprint": manifest["snapshot_tree_fingerprint"],
            "stage_tree_fingerprint": stage_validation["stage_tree_fingerprint"],
        }
        write_json_atomic(journal_path(manifest), journal)
        if fail_after == "prepared":
            raise RuntimeError("Injected failure after prepared")
        os.replace(canonical, backup)
        if fail_after == "original_renamed":
            raise RuntimeError("Injected failure after original_renamed")
        journal["state"] = "original_moved"
        write_json_atomic(journal_path(manifest), journal)
        if fail_after == "original_moved":
            raise RuntimeError("Injected failure after original_moved")
        os.replace(stage_root, canonical)
        if fail_after == "stage_renamed":
            raise RuntimeError("Injected failure after stage_renamed")
        journal["state"] = "stage_promoted"
        write_json_atomic(journal_path(manifest), journal)
        if fail_after == "stage_promoted":
            raise RuntimeError("Injected failure after stage_promoted")
        if tree_fingerprint(canonical) != stage_validation["stage_tree_fingerprint"]:
            raise RuntimeError("Post-promotion canonical validation failed")
        marker = completion_marker_path(manifest)
        marker.write_text(
            f"manifest_fingerprint={manifest['manifest_fingerprint']}\n", encoding="utf-8"
        )
        if fail_after == "marker_written":
            raise RuntimeError("Injected failure after marker_written")
        journal["state"] = "committed"
        journal["completion_marker_sha256"] = file_sha256(marker)
        write_json_atomic(journal_path(manifest), journal)
        return marker


def scheduler_states(manifest: Mapping[str, object]) -> Dict[str, object]:
    scheduler_path = Path(str(manifest["workflow_run_dir"])) / SCHEDULER_FILENAME
    scheduler = read_json(scheduler_path) if scheduler_path.is_file() else {}
    if scheduler:
        payload = {key: value for key, value in scheduler.items() if key != "evidence_fingerprint"}
        if scheduler.get("evidence_fingerprint") != fingerprint(payload):
            raise RuntimeError("Scheduler metadata fingerprint mismatch")
        if scheduler.get("manifest_fingerprint") != manifest.get("manifest_fingerprint"):
            raise RuntimeError("Scheduler metadata belongs to another manifest")
    try:
        completed = subprocess.run(
            ["qstat", "-u", os.environ.get("USER", "")],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"status": "qstat unavailable", **scheduler}
    states: Dict[str, object] = {**scheduler}
    job_records = {
        **{
            f"lane_{index + 1}_job_id": str(job_id)
            for index, job_id in enumerate(scheduler.get("lane_job_ids", []))
        },
        "collector_job_id": str(scheduler.get("collector_job_id", "")),
    }
    for label, job_id in job_records.items():
        matches = []
        for line in completed.stdout.splitlines():
            fields = line.split()
            if job_id and len(fields) >= 5 and fields[0] == job_id:
                matches.append(fields[4])
        states[f"{label}_states"] = sorted(set(matches)) if matches else "not in qstat"
    return states


def progress_payload(manifest: Mapping[str, object]) -> Dict[str, object]:
    validation = validate_shards(manifest)
    frozen_count = sum(len(record["completed_indices"]) for record in manifest["affected_tasks"])
    expected_missing = len(manifest["tasks"])
    completed_missing = len(validation.selected)
    expected_total = frozen_count + expected_missing
    run_dir = Path(str(manifest["workflow_run_dir"]))
    journal = read_json(journal_path(manifest)) if journal_path(manifest).is_file() else {}
    errors: List[str] = []
    writer_evidence = read_json(Path(str(manifest["writer_evidence_path"])))
    for label, operation in (
        ("execution checkout", lambda: require_execution_checkout(manifest)),
        ("snapshot", lambda: validate_snapshot(manifest)),
        (
            "writer evidence",
            lambda: validate_writer_evidence(
                writer_evidence, list(manifest.get("required_writer_job_ids", []))
            ),
        ),
    ):
        try:
            operation()
        except (RuntimeError, OSError, ValueError) as error:
            errors.append(f"{label}: {error}")
    stage_validated = False
    if (run_dir / "stage_validation.json").is_file():
        try:
            validate_stage_evidence(
                manifest,
                require_stage_tree=journal.get("state") != "committed",
            )
            stage_validated = True
        except (RuntimeError, OSError, ValueError) as error:
            errors.append(f"stage evidence: {error}")
    committed = False
    if journal.get("state") == "committed" or completion_marker_path(manifest).exists():
        try:
            validate_committed_result(manifest)
            committed = True
        except (RuntimeError, OSError, ValueError) as error:
            errors.append(f"committed result: {error}")
    try:
        scheduler = scheduler_states(manifest)
    except (RuntimeError, OSError, ValueError) as error:
        errors.append(f"scheduler metadata: {error}")
        scheduler = {"status": "invalid"}
    payload = {
        "git_commit": manifest["git_commit"],
        "scientific_config_fingerprint": manifest["scientific_config_fingerprint"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "snapshot_tree_fingerprint": manifest["snapshot_tree_fingerprint"],
        "expected_episode_shards": expected_missing,
        "array_lanes": manifest["array_lanes"],
        "lane_throttle": manifest["lane_throttle"],
        "valid_episode_shards": completed_missing,
        "missing_shards": validation.missing,
        "failed_shards": validation.failed,
        "invalid_shards": {str(key): value for key, value in validation.invalid.items()},
        "duplicate_shards": {str(key): value for key, value in validation.duplicates.items()},
        "validated_percent_complete": 100.0
        if expected_total == 0
        else 100.0 * (frozen_count + completed_missing) / expected_total,
        "remaining_by_method_task": {
            record["task_key"]: sum(
                1 for task in manifest["tasks"]
                if task["task_key"] == record["task_key"]
                and int(task["task_id"]) not in validation.selected
            )
            for record in manifest["affected_tasks"]
        },
        "writers_quiescent": writer_evidence.get("writers_quiescent", False),
        "scheduler": scheduler,
        "stage_validated": stage_validated,
        "promotion_state": journal.get("state", "not_started"),
        "complete": committed,
        "status_errors": errors,
        "status_valid": not errors and not validation.invalid and not validation.duplicates,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze, shard, validate, stage, and promote R3 method episodes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--canonical-run-dir", required=True)
    freeze.add_argument("--workflow-run-dir", required=True)
    freeze.add_argument("--task-manifest", required=True)
    freeze.add_argument("--writer-evidence", required=True)
    freeze.add_argument("--required-writer-job-id", action="append", default=[])
    freeze.add_argument("--git-commit", required=True)
    freeze.add_argument("--array-lanes", type=int, default=2)
    freeze.add_argument("--lane-throttle", type=int, default=80)
    freeze.add_argument("--expected-task-count", type=int, default=700)
    freeze.add_argument("--run-mode", choices=["production", "smoke"], default="production")
    freeze.add_argument("--reviewed-commit", required=True)
    freeze.add_argument("--serial-reference")

    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--manifest", required=True)
    shard.add_argument("--task-id", type=int, required=True)
    shard.add_argument("--attempt-id", required=True)

    for name in ("validate", "stage", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", required=True)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--manifest", required=True)
    recover.add_argument("--stale-lock-evidence")

    promote = subparsers.add_parser("promote")
    promote.add_argument("--manifest", required=True)
    promote.add_argument(
        "--inject-failure-after",
        choices=[
            "",
            "prepared",
            "original_renamed",
            "original_moved",
            "stage_renamed",
            "stage_promoted",
            "marker_written",
        ],
        default="",
    )

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--promote", action="store_true")

    smoke = subparsers.add_parser("certify-smoke")
    smoke.add_argument("--manifest", required=True)
    smoke.add_argument("--negative-evidence", required=True)
    smoke.add_argument("--scheduler-evidence", required=True)
    smoke.add_argument("--output", required=True)

    scheduler_smoke = subparsers.add_parser("record-smoke-scheduler-evidence")
    scheduler_smoke.add_argument("--manifest", required=True)
    scheduler_smoke.add_argument("--qstat", required=True)
    scheduler_smoke.add_argument("--qacct", action="append", required=True)
    scheduler_smoke.add_argument("--lane-job-id", action="append", required=True)
    scheduler_smoke.add_argument("--collector-job-id", required=True)
    scheduler_smoke.add_argument("--collector-qacct", required=True)
    scheduler_smoke.add_argument("--output", required=True)

    verify_gate = subparsers.add_parser("verify-smoke-gate")
    verify_gate.add_argument("--gate", required=True)
    verify_gate.add_argument("--reviewed-commit", required=True)

    negative = subparsers.add_parser("negative-check")
    negative.add_argument("--manifest", required=True)
    negative.add_argument("--task-id", type=int, required=True)
    negative.add_argument("--output", required=True)

    submission = subparsers.add_parser("submission")
    submission.add_argument("--manifest", required=True)
    submission.add_argument(
        "--action",
        required=True,
        choices=["init", "show", "record-lane", "record-collector", "rollback"],
    )
    submission.add_argument("--lane-id", type=int)
    submission.add_argument("--job-id")
    submission.add_argument("--rollback-complete", choices=["yes", "no"])
    submission.add_argument("--detail", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        manifest = freeze_checkpoint(
            canonical_run_dir=Path(args.canonical_run_dir),
            workflow_run_dir=Path(args.workflow_run_dir),
            task_manifest_path=Path(args.task_manifest),
            writer_evidence_path=Path(args.writer_evidence),
            required_writer_job_ids=args.required_writer_job_id,
            git_commit=args.git_commit,
            array_lanes=args.array_lanes,
            lane_throttle=args.lane_throttle,
            expected_task_count=args.expected_task_count,
            run_mode=args.run_mode,
            reviewed_commit=args.reviewed_commit,
            serial_reference_path=(
                Path(args.serial_reference) if args.serial_reference else None
            ),
        )
        print(Path(str(manifest["workflow_run_dir"])) / MANIFEST_FILENAME)
        return 0

    if args.command == "verify-smoke-gate":
        payload = verify_smoke_gate(Path(args.gate), args.reviewed_commit)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    manifest = load_manifest(Path(args.manifest))
    if args.command == "run-shard":
        print(run_manifest_task(manifest, args.task_id, args.attempt_id))
        return 0
    if args.command == "validate":
        validation = validate_shards(manifest)
        payload = {
            "ok": validation.ok,
            "complete": sorted(validation.selected),
            "missing": validation.missing,
            "failed": validation.failed,
            "invalid": validation.invalid,
            "duplicates": validation.duplicates,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if validation.ok else 1
    if args.command == "stage":
        print(build_staged_complete_view(manifest))
        return 0
    if args.command == "promote":
        print(promote_staged_view(manifest, args.inject_failure_after))
        return 0
    if args.command == "recover":
        evidence = Path(args.stale_lock_evidence) if args.stale_lock_evidence else None
        print(recover_promotion(manifest, evidence))
        return 0
    if args.command == "collect":
        build_staged_complete_view(manifest)
        if args.promote:
            promote_staged_view(manifest)
        return 0
    if args.command == "certify-smoke":
        payload = certify_smoke(
            manifest,
            Path(args.negative_evidence),
            Path(args.scheduler_evidence),
            Path(args.output),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "record-smoke-scheduler-evidence":
        payload = record_scheduler_smoke_evidence(
            manifest,
            Path(args.qstat),
            args.qacct,
            args.lane_job_id,
            args.collector_job_id,
            Path(args.collector_qacct),
            Path(args.output),
        )
        validate_scheduler_smoke_evidence(manifest, Path(args.output))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "negative-check":
        payload = run_missing_shard_negative_check(
            manifest,
            args.task_id,
            Path(args.output),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "submission":
        if args.action == "init":
            payload = initialize_submission(manifest)
        elif args.action == "show":
            path = submission_journal_path(manifest)
            payload = read_json(path) if path.is_file() else {}
        elif args.action == "record-lane":
            if args.lane_id is None or not args.job_id:
                raise RuntimeError("record-lane requires --lane-id and --job-id")
            payload = record_lane_submission(manifest, args.lane_id, args.job_id)
        elif args.action == "record-collector":
            if not args.job_id:
                raise RuntimeError("record-collector requires --job-id")
            payload = record_collector_submission(manifest, args.job_id)
        else:
            if args.rollback_complete is None:
                raise RuntimeError("rollback requires --rollback-complete")
            payload = mark_submission_rollback(
                manifest, args.rollback_complete == "yes", args.detail
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        payload = progress_payload(manifest)
        write_json_atomic(
            Path(str(manifest["workflow_run_dir"])) / PROGRESS_FILENAME, payload
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status_valid"] else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
