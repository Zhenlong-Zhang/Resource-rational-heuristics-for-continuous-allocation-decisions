from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_results import write_figures  # noqa: E402
from src.experiments.compare import ENVIRONMENT_LIBRARY  # noqa: E402
from src.experiments.sweeps import ONE_DIMENSIONAL_SWEEP_VALUES, build_positive_and_near_zero_utility_configs  # noqa: E402


EXPECTED_OUTPUTS: Dict[str, Sequence[str]] = {
    "step7": (
        "step7_final_choice_comparison.csv",
        "step7_information_acquisition_comparison.csv",
        "step7_behavior_profiles.csv",
        "rr_approximation_methods_comparison.csv",
    ),
    "sweeps": (
        "sweep_final_choice_comparison.csv",
        "sweep_rr_behavior_profiles.csv",
        "sweep_final_choice_candidates.csv",
        "sweep_behavior_candidates.csv",
    ),
    "dp": ("dp_sensitivity_analysis.csv",),
    "gh": ("gauss_hermite_diagnostics.csv",),
}

RESULT_SET_BY_FILE = {
    "step7_final_choice_comparison.csv": "final_choice",
    "step7_information_acquisition_comparison.csv": "information_acquisition",
    "step7_behavior_profiles.csv": "behavior_profiles",
    "rr_approximation_methods_comparison.csv": "approximation_methods",
    "sweep_final_choice_comparison.csv": "sweep_final_choice",
    "sweep_rr_behavior_profiles.csv": "sweep_behavior",
    "sweep_final_choice_candidates.csv": "sweep_final_choice_candidates",
    "sweep_behavior_candidates.csv": "sweep_behavior_candidates",
    "dp_sensitivity_analysis.csv": "dp_sensitivity",
    "gauss_hermite_diagnostics.csv": "gauss_hermite",
}


@dataclass
class TaskSpec:
    task_id: str
    section: str
    shard: str
    output_dir: str
    log_stdout: str
    log_stderr: str
    command: List[str]


@dataclass
class TaskStatus:
    task_id: str
    section: str
    shard: str
    status: str
    returncode: int | None
    elapsed_seconds: float
    output_dir: str
    log_stdout: str
    log_stderr: str
    missing_outputs: str
    command: str


def parse_sections(value: str) -> List[str]:
    sections = [section.strip() for section in value.split(",") if section.strip()]
    if not sections or "all" in sections:
        return ["step7", "sweeps", "dp", "gh"]
    unknown = sorted(set(sections) - set(EXPECTED_OUTPUTS))
    if unknown:
        raise ValueError(f"Unknown sections: {unknown}")
    return sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Round 2 result generation as parallel, resumable shards."
    )
    parser.add_argument("--preset", choices=["smoke", "serious", "server"], default="serious")
    parser.add_argument("--sections", default="all", help="Comma-separated: step7,sweeps,dp,gh or all.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--resume", action="store_true", help="Skip shards that already have all expected files.")

    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--voi-samples", type=int, default=None)
    parser.add_argument("--blinkered-samples", type=int, default=None)
    parser.add_argument("--common-observations", choices=["auto", "on", "off"], default="on")
    parser.add_argument("--observations-per-person", type=int, default=None)
    parser.add_argument("--allocation-grid-size", type=int, default=None)
    parser.add_argument("--expected-utility-draws", type=int, default=None)
    parser.add_argument("--terminal-integration", choices=["monte_carlo", "gauss_hermite"], default=None)
    parser.add_argument("--gauss-hermite-order", type=int, default=15)

    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--sweep-feature", action="append", default=[])
    parser.add_argument("--max-sweep-values-per-feature", type=int, default=None)
    parser.add_argument("--dp-max-samples-values", default="2,4,6,10")
    parser.add_argument("--dp-mean-grid-sizes", default="7,11,21,50")
    parser.add_argument("--dp-observation-branches", default="3,5")
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return cleaned.strip("_") or "shard"


def all_environment_names(selected: Sequence[str]) -> List[str]:
    names = list(ENVIRONMENT_LIBRARY)
    names.extend(name for name, _ in build_positive_and_near_zero_utility_configs())
    if selected:
        missing = sorted(set(selected) - set(names))
        if missing:
            raise ValueError(f"Unknown environments: {missing}")
        return list(selected)
    return names


def all_sweep_features(selected: Sequence[str]) -> List[str]:
    names = list(ONE_DIMENSIONAL_SWEEP_VALUES)
    if selected:
        missing = sorted(set(selected) - set(names))
        if missing:
            raise ValueError(f"Unknown sweep features: {missing}")
        return list(selected)
    return names


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def add_optional_args(command: List[str], args: argparse.Namespace) -> None:
    optional = {
        "--episodes": args.episodes,
        "--voi-samples": args.voi_samples,
        "--blinkered-samples": args.blinkered_samples,
        "--observations-per-person": args.observations_per_person,
        "--allocation-grid-size": args.allocation_grid_size,
        "--expected-utility-draws": args.expected_utility_draws,
        "--terminal-integration": args.terminal_integration,
        "--max-sweep-values-per-feature": args.max_sweep_values_per_feature,
    }
    for flag, value in optional.items():
        if value is not None:
            command.extend([flag, str(value)])
    command.extend(["--common-observations", args.common_observations])
    command.extend(["--gauss-hermite-order", str(args.gauss_hermite_order)])
    command.extend(["--dp-max-samples-values", args.dp_max_samples_values])
    command.extend(["--dp-mean-grid-sizes", args.dp_mean_grid_sizes])
    command.extend(["--dp-observation-branches", args.dp_observation_branches])


def build_task(
    args: argparse.Namespace,
    run_dir: Path,
    section: str,
    shard: str,
    extra_args: Sequence[str],
) -> TaskSpec:
    task_id = f"{section}__{slugify(shard)}"
    output_dir = run_dir / "tasks" / section / slugify(shard)
    log_dir = run_dir / "logs" / section
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_results.py"),
        "--preset",
        args.preset,
        "--sections",
        section,
        "--output-dir",
        relative_to_project(output_dir),
        *extra_args,
    ]
    add_optional_args(command, args)
    return TaskSpec(
        task_id=task_id,
        section=section,
        shard=shard,
        output_dir=str(output_dir),
        log_stdout=str(log_dir / f"{task_id}.stdout.log"),
        log_stderr=str(log_dir / f"{task_id}.stderr.log"),
        command=command,
    )


def build_tasks(args: argparse.Namespace, run_dir: Path) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    sections = parse_sections(args.sections)
    environments = all_environment_names(args.environment)
    sweep_features = all_sweep_features(args.sweep_feature)

    if "step7" in sections:
        for environment in environments:
            tasks.append(build_task(args, run_dir, "step7", environment, ["--environment", environment]))
    if "sweeps" in sections:
        for feature in sweep_features:
            tasks.append(build_task(args, run_dir, "sweeps", feature, ["--sweep-feature", feature]))
    if "dp" in sections:
        for environment in environments:
            tasks.append(build_task(args, run_dir, "dp", environment, ["--environment", environment]))
    if "gh" in sections:
        for environment in environments:
            tasks.append(build_task(args, run_dir, "gh", environment, ["--environment", environment]))
    return tasks


def expected_missing_outputs(task: TaskSpec) -> List[str]:
    output_dir = Path(task.output_dir)
    missing: List[str] = []
    for filename in EXPECTED_OUTPUTS[task.section]:
        path = output_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def run_task(task: TaskSpec) -> TaskStatus:
    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(task.log_stdout, "w", encoding="utf-8") as stdout_handle, open(
        task.log_stderr,
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        stdout_handle.write(f"COMMAND: {' '.join(task.command)}\n")
        stdout_handle.flush()
        process = subprocess.run(
            task.command,
            cwd=PROJECT_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    elapsed = time.time() - start
    missing = expected_missing_outputs(task)
    status = "ok" if process.returncode == 0 and not missing else "failed"
    return TaskStatus(
        task_id=task.task_id,
        section=task.section,
        shard=task.shard,
        status=status,
        returncode=process.returncode,
        elapsed_seconds=elapsed,
        output_dir=task.output_dir,
        log_stdout=task.log_stdout,
        log_stderr=task.log_stderr,
        missing_outputs=";".join(missing),
        command=" ".join(task.command),
    )


def skipped_status(task: TaskSpec) -> TaskStatus:
    return TaskStatus(
        task_id=task.task_id,
        section=task.section,
        shard=task.shard,
        status="skipped_existing",
        returncode=0,
        elapsed_seconds=0.0,
        output_dir=task.output_dir,
        log_stdout=task.log_stdout,
        log_stderr=task.log_stderr,
        missing_outputs="",
        command=" ".join(task.command),
    )


def write_status_files(run_dir: Path, statuses: Sequence[TaskStatus]) -> None:
    rows = [asdict(status) for status in statuses]
    status_csv = run_dir / "parallel_run_status.csv"
    status_json = run_dir / "parallel_run_status.json"
    if rows:
        with status_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    status_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_combined_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_outputs(run_dir: Path, tasks: Sequence[TaskSpec]) -> Dict[str, List[Dict[str, str]]]:
    combined_by_file: Dict[str, List[Dict[str, str]]] = {
        filename: []
        for filenames in EXPECTED_OUTPUTS.values()
        for filename in filenames
    }
    for task in tasks:
        output_dir = Path(task.output_dir)
        for filename in EXPECTED_OUTPUTS[task.section]:
            for row in read_csv_rows(output_dir / filename):
                combined_by_file[filename].append(row)

    result_sets: Dict[str, List[Dict[str, str]]] = {}
    for filename, rows in combined_by_file.items():
        if not rows:
            continue
        write_combined_csv(run_dir / filename, rows)
        result_sets[RESULT_SET_BY_FILE[filename]] = rows
    return result_sets


def write_parallel_summary(
    run_dir: Path,
    args: argparse.Namespace,
    tasks: Sequence[TaskSpec],
    statuses: Sequence[TaskStatus],
) -> None:
    failed = [status for status in statuses if status.status == "failed"]
    ok = [status for status in statuses if status.status in {"ok", "skipped_existing"}]
    lines = [
        "# Parallel Round 2 Run Summary",
        "",
        "This run splits Round 2 into section/environment or section/feature shards.",
        "",
        "## Settings",
        "",
        f"- preset: `{args.preset}`",
        f"- sections: `{args.sections}`",
        f"- max_workers: `{args.max_workers}`",
        f"- episodes override: `{args.episodes}`",
        f"- VOI samples override: `{args.voi_samples}`",
        f"- common observations: `{args.common_observations}`",
        "",
        "## Status",
        "",
        f"- total tasks: `{len(tasks)}`",
        f"- ok/skipped tasks: `{len(ok)}`",
        f"- failed tasks: `{len(failed)}`",
        "",
    ]
    if failed:
        lines.append("## Failed Tasks")
        lines.append("")
        for status in failed:
            lines.append(
                f"- `{status.task_id}` returncode=`{status.returncode}` "
                f"missing=`{status.missing_outputs}` stderr=`{status.log_stderr}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `parallel_run_status.csv`: machine-readable task status table",
            "- `parallel_run_status.json`: JSON copy of task status",
            "- `logs/`: stdout/stderr for each shard",
            "- `tasks/`: raw per-shard outputs",
            "- top-level CSV files: combined outputs across successful shards",
        ]
    )
    (run_dir / "parallel_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / (
        args.output_dir
        or f"results/r2_parallel_{args.preset}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args, run_dir)
    manifest_path = run_dir / "parallel_task_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(task) for task in tasks], indent=2),
        encoding="utf-8",
    )

    statuses: List[TaskStatus] = []
    runnable: List[TaskSpec] = []
    for task in tasks:
        if args.resume and not expected_missing_outputs(task):
            statuses.append(skipped_status(task))
        else:
            runnable.append(task)
    write_status_files(run_dir, statuses)

    print(f"Parallel R2 run directory: {run_dir}")
    print(f"Total tasks: {len(tasks)} | Runnable: {len(runnable)} | max_workers: {args.max_workers}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in runnable}
        for future in as_completed(future_to_task):
            status = future.result()
            statuses.append(status)
            write_status_files(run_dir, statuses)
            print(
                f"[{status.status}] {status.task_id} "
                f"elapsed={status.elapsed_seconds:.1f}s returncode={status.returncode}"
            )

    result_sets = combine_outputs(run_dir, tasks)
    write_figures(run_dir, result_sets)
    write_status_files(run_dir, statuses)
    write_parallel_summary(run_dir, args, tasks, statuses)

    failed = [status for status in statuses if status.status == "failed"]
    if failed:
        print(f"Finished with {len(failed)} failed tasks. See {run_dir / 'parallel_summary.md'}")
        return 1
    print(f"Finished successfully. See {run_dir / 'parallel_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
