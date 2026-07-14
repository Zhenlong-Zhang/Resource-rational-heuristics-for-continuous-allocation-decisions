from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_OUTPUTS = (
    "r4_diagnostic_policy_profiles.csv",
    "r4_diagnostic_environment_summary.csv",
    "r4_diagnostic_manual_advantage_candidates.csv",
)
PROVENANCE_FILENAME = "shard_provenance.json"
MANIFEST_FILENAME = "r4_array_manifest.json"
PROGRESS_FILENAME = "r4_array_progress.json"
SCHEDULER_FILENAME = "r4_scheduler_jobs.json"
FAILURE_FILENAME = "task_failure.json"
COMPLETE_MARKER = "R4_ARRAY_COMPLETE"
SCIENTIFIC_BASELINE_COMMIT = "e92d64d"


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


def canonical_shard_name(grid: str, chunk_index: int, chunk_count: int) -> str:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError("chunk_index must be in [0, chunk_count)")
    return f"{grid}_chunk{chunk_index:02d}_of{chunk_count:02d}"


def scientific_config(
    *,
    grid: str,
    grid_size: int,
    chunk_count: int,
    preset: str,
    episodes: int,
    voi_samples: int,
    common_observations: str,
    observations_per_person: int,
    manual_active_samples_per_person: int,
    max_grid_points: int | None,
) -> Dict[str, object]:
    return {
        "preset": preset,
        "section": "r4_diagnostics",
        "regime_grid": grid,
        "grid_size": grid_size,
        "chunk_count": chunk_count,
        "episodes": episodes,
        "voi_samples": voi_samples,
        "common_observations": common_observations,
        "observations_per_person": observations_per_person,
        "manual_active_samples_per_person": manual_active_samples_per_person,
        "max_grid_points": max_grid_points,
    }


def task_command(config: Mapping[str, object], output_dir: str, chunk_index: int) -> List[str]:
    command = [
        "scripts/generate_results.py",
        "--preset",
        str(config["preset"]),
        "--sections",
        "r4_diagnostics",
        "--output-dir",
        output_dir,
        "--regime-grid",
        str(config["regime_grid"]),
        "--regime-grid-chunk-index",
        str(chunk_index),
        "--regime-grid-chunks",
        str(config["chunk_count"]),
        "--episodes",
        str(config["episodes"]),
        "--voi-samples",
        str(config["voi_samples"]),
        "--common-observations",
        str(config["common_observations"]),
        "--observations-per-person",
        str(config["observations_per_person"]),
        "--manual-active-samples-per-person",
        str(config["manual_active_samples_per_person"]),
    ]
    if config.get("max_grid_points") is not None:
        command.extend(["--max-regime-grid-points", str(config["max_grid_points"])])
    return command


def build_manifest(
    *,
    run_dir: Path,
    git_commit: str,
    baseline_commit: str,
    throttle: int,
    config: Mapping[str, object],
) -> Dict[str, object]:
    chunk_count = int(config["chunk_count"])
    grid = str(config["regime_grid"])
    config_fingerprint = fingerprint(config)
    tasks: List[Dict[str, object]] = []
    for chunk_index in range(chunk_count):
        shard = canonical_shard_name(grid, chunk_index, chunk_count)
        output_dir = run_dir / "tasks" / "r4_diagnostics" / shard
        command = task_command(config, str(output_dir), chunk_index)
        tasks.append(
            {
                "task_id": chunk_index + 1,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "shard": shard,
                "output_dir": str(output_dir),
                "scientific_command": command,
                "scientific_command_fingerprint": fingerprint(command),
            }
        )
    return {
        "schema_version": 1,
        "workflow": "r4_diagnostic_active_search_array",
        "run_dir": str(run_dir),
        "git_commit": git_commit,
        "scientific_baseline_commit": baseline_commit,
        "throttle": throttle,
        "scientific_config": dict(config),
        "scientific_config_fingerprint": config_fingerprint,
        "required_outputs": list(REQUIRED_OUTPUTS),
        "tasks": tasks,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_task(manifest: Mapping[str, object], chunk_index: int) -> Mapping[str, object]:
    tasks = manifest.get("tasks", [])
    matches = [task for task in tasks if int(task["chunk_index"]) == chunk_index]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest task for chunk {chunk_index}")
    return matches[0]


def shard_provenance(manifest: Mapping[str, object], chunk_index: int) -> Dict[str, object]:
    task = manifest_task(manifest, chunk_index)
    return {
        "schema_version": 1,
        "workflow": manifest["workflow"],
        "git_commit": manifest["git_commit"],
        "scientific_baseline_commit": manifest["scientific_baseline_commit"],
        "scientific_config_fingerprint": manifest["scientific_config_fingerprint"],
        "chunk_index": task["chunk_index"],
        "chunk_count": task["chunk_count"],
        "shard": task["shard"],
        "scientific_command_fingerprint": task["scientific_command_fingerprint"],
    }


def completed_shard_provenance(
    manifest: Mapping[str, object], chunk_index: int
) -> Dict[str, object]:
    task = manifest_task(manifest, chunk_index)
    output_dir = Path(str(task["output_dir"]))
    provenance = shard_provenance(manifest, chunk_index)
    provenance["output_sha256"] = {
        filename: file_sha256(output_dir / filename)
        for filename in REQUIRED_OUTPUTS
    }
    return provenance


def record_completed_shard(manifest: Mapping[str, object], chunk_index: int) -> None:
    task = manifest_task(manifest, chunk_index)
    write_json(
        expected_provenance_path(task),
        completed_shard_provenance(manifest, chunk_index),
    )


def expected_provenance_path(task: Mapping[str, object]) -> Path:
    return Path(str(task["output_dir"])) / PROVENANCE_FILENAME


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class ValidationResult:
    complete: List[int]
    missing: List[int]
    failed: List[int]
    invalid: Dict[int, List[str]]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.failed and not self.invalid


def validate_manifest_structure(manifest: Mapping[str, object]) -> List[str]:
    errors: List[str] = []
    tasks = list(manifest.get("tasks", []))
    config = manifest.get("scientific_config", {})
    expected_count = int(config.get("chunk_count", 0))
    indices = [int(task["chunk_index"]) for task in tasks]
    task_ids = [int(task["task_id"]) for task in tasks]
    shards = [str(task["shard"]) for task in tasks]
    output_dirs = [str(task["output_dir"]) for task in tasks]
    if len(tasks) != expected_count:
        errors.append(f"manifest has {len(tasks)} tasks; expected {expected_count}")
    if sorted(indices) != list(range(expected_count)):
        errors.append("chunk indices are not a complete unique range")
    if sorted(task_ids) != list(range(1, expected_count + 1)):
        errors.append("task IDs are not a complete unique range")
    if len(set(shards)) != len(shards):
        errors.append("duplicate shard names")
    if len(set(output_dirs)) != len(output_dirs):
        errors.append("duplicate output directories")
    if fingerprint(config) != manifest.get("scientific_config_fingerprint"):
        errors.append("scientific config fingerprint mismatch")
    if manifest.get("workflow") != "r4_diagnostic_active_search_array":
        errors.append("workflow mismatch")
    if manifest.get("required_outputs") != list(REQUIRED_OUTPUTS):
        errors.append("required outputs mismatch")
    if manifest.get("scientific_baseline_commit") != SCIENTIFIC_BASELINE_COMMIT:
        errors.append("scientific baseline commit mismatch")
    expected = build_manifest(
        run_dir=Path(str(manifest.get("run_dir", ""))),
        git_commit=str(manifest.get("git_commit", "")),
        baseline_commit=str(manifest.get("scientific_baseline_commit", "")),
        throttle=int(manifest.get("throttle", 0)),
        config=config,
    )
    if tasks != expected["tasks"]:
        errors.append("manifest task mapping or command mismatch")
    return errors


def require_execution_checkout(manifest: Mapping[str, object]) -> None:
    actual_commit = current_git_commit()
    recorded_commit = str(manifest.get("git_commit", ""))
    if actual_commit != recorded_commit:
        raise RuntimeError(
            f"Execution checkout {actual_commit} does not match manifest {recorded_commit}"
        )


def validate_task_output(
    manifest: Mapping[str, object], task: Mapping[str, object]
) -> tuple[str, List[str]]:
    chunk_index = int(task["chunk_index"])
    output_dir = Path(str(task["output_dir"]))
    errors: List[str] = []
    absent = [
        name
        for name in REQUIRED_OUTPUTS
        if not (output_dir / name).is_file()
        or (output_dir / name).stat().st_size == 0
    ]
    provenance_path = expected_provenance_path(task)
    if absent or not provenance_path.is_file():
        state = "failed" if (output_dir / FAILURE_FILENAME).is_file() else "missing"
        return state, errors

    actual_provenance = read_json(provenance_path)
    expected = completed_shard_provenance(manifest, chunk_index)
    if actual_provenance != expected:
        errors.append("shard provenance does not match manifest")

    config = manifest["scientific_config"]
    chunk_count = int(config["chunk_count"])
    grid_size = int(config["grid_size"])
    grid = str(config["regime_grid"])
    expected_indices = list(range(chunk_index, grid_size, chunk_count))
    summary_rows = read_csv_rows(output_dir / "r4_diagnostic_environment_summary.csv")
    try:
        actual_indices = sorted(int(float(row.get("grid_index", "nan"))) for row in summary_rows)
    except ValueError:
        actual_indices = []
    if actual_indices != expected_indices:
        errors.append(
            f"environment summary grid indices {actual_indices}; expected {expected_indices}"
        )
    if any(row.get("regime_grid") != grid for row in summary_rows):
        errors.append("environment summary regime_grid mismatch")
    return ("invalid", errors) if errors else ("complete", [])


def validate_shards(manifest: Mapping[str, object]) -> ValidationResult:
    structure_errors = validate_manifest_structure(manifest)
    if structure_errors:
        return ValidationResult(
            complete=[], missing=[], failed=[], invalid={-1: structure_errors}
        )

    complete: List[int] = []
    missing: List[int] = []
    failed: List[int] = []
    invalid: Dict[int, List[str]] = {}
    for task in manifest["tasks"]:
        chunk_index = int(task["chunk_index"])
        state, errors = validate_task_output(manifest, task)
        if state == "missing":
            missing.append(chunk_index)
        elif state == "failed":
            failed.append(chunk_index)
        elif state == "invalid":
            invalid[chunk_index] = errors
        else:
            complete.append(chunk_index)
    return ValidationResult(
        complete=complete, missing=missing, failed=failed, invalid=invalid
    )


def progress_payload(manifest: Mapping[str, object]) -> Dict[str, object]:
    result = validate_shards(manifest)
    expected = len(manifest.get("tasks", []))
    complete = len(result.complete)
    run_dir = Path(str(manifest["run_dir"]))
    scheduler_path = run_dir / SCHEDULER_FILENAME
    scheduler = read_json(scheduler_path) if scheduler_path.is_file() else {}
    return {
        "run_dir": manifest["run_dir"],
        "git_commit": manifest["git_commit"],
        "scientific_baseline_commit": manifest["scientific_baseline_commit"],
        "scientific_config_fingerprint": manifest["scientific_config_fingerprint"],
        "throttle": manifest["throttle"],
        "array_job_id": scheduler.get("array_job_id", ""),
        "collector_job_id": scheduler.get("collector_job_id", ""),
        "scheduler_states": query_scheduler_states(scheduler),
        "expected_shards": expected,
        "complete_shards": complete,
        "missing_shards": result.missing,
        "failed_shards": result.failed,
        "invalid_shards": {str(key): value for key, value in result.invalid.items()},
        "percent_complete": 0.0 if expected == 0 else 100.0 * complete / expected,
        "complete": result.ok,
    }


def query_scheduler_states(scheduler: Mapping[str, object]) -> Dict[str, List[str] | str]:
    job_ids = {
        "array": str(scheduler.get("array_job_id", "")),
        "collector": str(scheduler.get("collector_job_id", "")),
    }
    if not any(job_ids.values()):
        return {"status": "job IDs not recorded"}
    try:
        completed = subprocess.run(
            ["qstat", "-u", os.environ.get("USER", "")],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"status": "qstat unavailable"}
    states: Dict[str, List[str] | str] = {}
    for label, job_id in job_ids.items():
        if not job_id:
            states[label] = "not recorded"
            continue
        matches: List[str] = []
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[0] == job_id:
                matches.append(fields[4])
        states[label] = sorted(set(matches)) if matches else "not in qstat"
    return states


def combine_validated_outputs(manifest: Mapping[str, object]) -> None:
    require_execution_checkout(manifest)
    result = validate_shards(manifest)
    if not result.ok:
        raise RuntimeError(
            "Strict R4 collection refused: "
            f"missing={result.missing}; failed={result.failed}; invalid={result.invalid}"
        )

    run_dir = Path(str(manifest["run_dir"]))
    rows_by_file: Dict[str, List[Dict[str, str]]] = {name: [] for name in REQUIRED_OUTPUTS}
    fields_by_file: Dict[str, List[str]] = {name: [] for name in REQUIRED_OUTPUTS}
    for task in manifest["tasks"]:
        output_dir = Path(str(task["output_dir"]))
        for filename in REQUIRED_OUTPUTS:
            path = output_dir / filename
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for field in reader.fieldnames or []:
                    if field not in fields_by_file[filename]:
                        fields_by_file[filename].append(field)
                rows_by_file[filename].extend(reader)

    for filename, rows in rows_by_file.items():
        fieldnames = fields_by_file[filename]
        if not fieldnames:
            raise RuntimeError(f"Cannot combine {filename}: no CSV header")
        with (run_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Figure generation consumes existing combined rows only; it never launches simulations.
    try:
        from .generate_results import write_figures
    except ImportError:  # Direct execution places scripts/ on sys.path.
        from generate_results import write_figures

    result_sets = {
        "r4_diagnostic_policy_profiles": rows_by_file["r4_diagnostic_policy_profiles.csv"],
        "r4_diagnostic_environment_summary": rows_by_file["r4_diagnostic_environment_summary.csv"],
        "r4_diagnostic_manual_advantage_candidates": rows_by_file[
            "r4_diagnostic_manual_advantage_candidates.csv"
        ],
    }
    write_figures(run_dir, result_sets)
    (run_dir / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")


def run_manifest_shard(manifest: Mapping[str, object], chunk_index: int) -> int:
    structure_errors = validate_manifest_structure(manifest)
    if structure_errors:
        raise RuntimeError(f"Invalid R4 manifest: {structure_errors}")
    require_execution_checkout(manifest)
    task = manifest_task(manifest, chunk_index)
    state, _ = validate_task_output(manifest, task)
    if state == "complete":
        print(f"Shard {chunk_index} is already complete; skipping.")
        return 0
    command = [sys.executable, *[str(value) for value in task["scientific_command"]]]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        record_task_failure(manifest, task, completed.returncode, "simulation")
        return int(completed.returncode)
    try:
        record_completed_shard(manifest, chunk_index)
        state, errors = validate_task_output(manifest, task)
        if state != "complete":
            raise RuntimeError(f"post-process validation state={state}; errors={errors}")
    except Exception as error:
        record_task_failure(manifest, task, 70, "post_process", str(error))
        return 70
    failure_path = Path(str(task["output_dir"])) / FAILURE_FILENAME
    if failure_path.exists():
        failure_path.unlink()
    return 0


def record_task_failure(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    returncode: int,
    stage: str,
    error: str = "",
) -> None:
    write_json(
        Path(str(task["output_dir"])) / FAILURE_FILENAME,
        {
            "chunk_index": int(task["chunk_index"]),
            "chunk_count": int(task["chunk_count"]),
            "git_commit": manifest["git_commit"],
            "returncode": returncode,
            "stage": stage,
            "error": error,
            "scientific_command_fingerprint": task[
                "scientific_command_fingerprint"
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest, provenance, progress, and strict collection for R4 arrays.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--run-dir", required=True)
    create.add_argument("--git-commit", required=True)
    create.add_argument("--baseline-commit", default=SCIENTIFIC_BASELINE_COMMIT)
    create.add_argument("--throttle", type=int, required=True)
    create.add_argument("--grid", default="r4_diagnostic_active_search")
    create.add_argument("--grid-size", type=int, required=True)
    create.add_argument("--chunks", type=int, required=True)
    create.add_argument("--preset", default="server")
    create.add_argument("--episodes", type=int, required=True)
    create.add_argument("--voi-samples", type=int, required=True)
    create.add_argument("--common-observations", default="on")
    create.add_argument("--observations-per-person", type=int, required=True)
    create.add_argument("--manual-active-samples-per-person", type=int, required=True)
    create.add_argument("--max-grid-points", type=int)

    record = subparsers.add_parser("record-shard")
    record.add_argument("--manifest", required=True)
    record.add_argument("--chunk-index", type=int, required=True)

    run = subparsers.add_parser("run-shard")
    run.add_argument("--manifest", required=True)
    run.add_argument("--chunk-index", type=int, required=True)

    jobs = subparsers.add_parser("record-jobs")
    jobs.add_argument("--manifest", required=True)
    jobs.add_argument("--array-job-id", required=True)
    jobs.add_argument("--collector-job-id", required=True)

    progress = subparsers.add_parser("progress")
    progress.add_argument("--manifest", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create-manifest":
        run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
        config = scientific_config(
            grid=args.grid,
            grid_size=args.grid_size,
            chunk_count=args.chunks,
            preset=args.preset,
            episodes=args.episodes,
            voi_samples=args.voi_samples,
            common_observations=args.common_observations,
            observations_per_person=args.observations_per_person,
            manual_active_samples_per_person=args.manual_active_samples_per_person,
            max_grid_points=args.max_grid_points,
        )
        manifest = build_manifest(
            run_dir=run_dir,
            git_commit=args.git_commit,
            baseline_commit=args.baseline_commit,
            throttle=args.throttle,
            config=config,
        )
        write_json(run_dir / MANIFEST_FILENAME, manifest)
        print(run_dir / MANIFEST_FILENAME)
        return 0

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    if args.command == "record-shard":
        record_completed_shard(manifest, args.chunk_index)
        return 0
    if args.command == "run-shard":
        return run_manifest_shard(manifest, args.chunk_index)
    if args.command == "record-jobs":
        write_json(
            Path(str(manifest["run_dir"])) / SCHEDULER_FILENAME,
            {
                "array_job_id": args.array_job_id,
                "collector_job_id": args.collector_job_id,
            },
        )
        return 0
    if args.command == "progress":
        payload = progress_payload(manifest)
        write_json(Path(str(manifest["run_dir"])) / PROGRESS_FILENAME, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if not payload["invalid_shards"] else 1
    if args.command == "collect":
        combine_validated_outputs(manifest)
        payload = progress_payload(manifest)
        write_json(Path(str(manifest["run_dir"])) / PROGRESS_FILENAME, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
