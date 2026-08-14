#!/usr/bin/env python3
"""Run and validate the StrategyMapping pre-feedback positive-need exploration."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.positive_need import (  # noqa: E402
    CONFIRMATION_EPISODE_SCHEMA,
    DEFAULT_SPEC_PATH,
    DEVELOPMENT_EPISODE_SCHEMA,
    POLICY_MANUAL,
    POLICY_ORACLE,
    POLICY_RR,
    POLICY_SPLIT,
    ANALYSIS_LABEL,
    SERIOUS_POLICY_ORDER,
    PositiveNeedEnvironment,
    build_latent_support_table,
    build_numerical_validation_cases,
    build_development_environments,
    build_finite_support_episodes,
    evaluate_fixed_budgets,
    evaluate_serious_environment,
    initial_information_values,
    load_positive_need_spec,
    select_target_control_pair,
    summarize_development_environment,
    summarize_numerical_validation,
    summarize_serious,
    validate_numerical_action_value_maps,
    validate_numerical_case,
    validate_serious_common_randomness,
)


SCHEMA_VERSION = 2
MANIFEST_NAME = "positive_need_manifest.json"
PROGRESS_NAME = "positive_need_progress.json"
ARTIFACT_INDEX_NAME = "positive_need_artifact_index.json"
VALIDATION_NAME = "positive_need_validation.json"
COMPLETION_NAME = "COMPLETED.json"
CANDIDATE_POINTER_NAME = "CANDIDATE.json"
CURRENT_POINTER_NAME = "CURRENT.json"
IMPLEMENTATION_SOURCES = (
    "configs/positive_need_spec.json",
    "scripts/positive_need_workflow.py",
    "scripts/submit_hoffman2_positive_need.sh",
    "scripts/hoffman2_scheduler.sh",
    "src/experiments/positive_need.py",
    "src/experiments/active_search_evaluation.py",
    "src/experiments/randomization.py",
    "src/experiments/regimes.py",
    "src/mdp/finite_support.py",
    "src/mdp/meta_mdp.py",
    "src/policies/finite_support_voi.py",
    "src/policies/heuristic.py",
    "src/solvers/gauss_hermite.py",
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


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def git_tree_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=PROJECT_ROOT, text=True
    ).strip()


def git_clean() -> bool:
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


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    expected_fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_fields = {field for row in rows for field in row}
    if expected_fields is None:
        fields = sorted(observed_fields)
    else:
        fields = list(expected_fields)
        unexpected = observed_fields.difference(fields)
        if unexpected:
            raise RuntimeError(f"unexpected CSV fields: {sorted(unexpected)}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_fields(path: Path) -> List[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as error:
            raise RuntimeError(f"empty CSV: {path}") from error


def require_exact_csv_schema(path: Path, expected_fields: Sequence[str]) -> None:
    observed = csv_fields(path)
    expected = list(expected_fields)
    if observed != expected:
        raise RuntimeError(
            f"CSV schema mismatch for {path.name}: expected {expected}, observed {observed}"
        )


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def source_hashes() -> Dict[str, str]:
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_SOURCES
        if (PROJECT_ROOT / relative).exists()
    }


def environment_record(environment: PositiveNeedEnvironment) -> Dict[str, object]:
    return {
        "name": environment.name,
        "gap_class": environment.gap_class,
        "sigma_sample": environment.sigma_sample,
        "sample_time_cost": environment.sample_time_cost,
        "environment_hash": environment.environment_hash,
        "support_hash": environment.prior.support_hash,
        "config": asdict(environment.config),
    }


def find_environment(name: str, spec: Mapping[str, object]) -> PositiveNeedEnvironment:
    matches = [item for item in build_development_environments(spec) if item.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"environment lookup failed: {name}")
    return matches[0]


def manifest_hash(manifest: Mapping[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return digest(payload)


def load_manifest(path: Path) -> Dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != manifest_hash(manifest):
        raise RuntimeError("manifest hash mismatch")
    return manifest


def load_version_pointer(run_dir: Path, name: str) -> tuple[Path, Dict[str, object]]:
    pointer_path = run_dir / name
    if not pointer_path.exists():
        raise RuntimeError(f"version pointer is missing: {name}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer_hash = pointer.pop("pointer_hash", None)
    if pointer_hash != digest(pointer):
        raise RuntimeError(f"version pointer hash mismatch: {name}")
    pointer["pointer_hash"] = pointer_hash
    if pointer.get("manifest_hash") != load_manifest(run_dir / MANIFEST_NAME)["manifest_hash"]:
        raise RuntimeError(f"version pointer manifest mismatch: {name}")
    version = run_dir / str(pointer["version_path"])
    if not version.is_dir():
        raise RuntimeError(f"version directory is missing: {version}")
    if pointer.get("artifact_index_sha256") != sha256_file(version / ARTIFACT_INDEX_NAME):
        raise RuntimeError(f"version pointer artifact index mismatch: {name}")
    if pointer.get("validation_sha256") != sha256_file(version / VALIDATION_NAME):
        raise RuntimeError(f"version pointer validation mismatch: {name}")
    return version, pointer


def next_version_path(run_dir: Path, family: str) -> Path:
    parent = run_dir / family
    parent.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.rsplit("_", 1)[-1])
        for path in parent.glob("version_*")
        if path.is_dir() and path.name.rsplit("_", 1)[-1].isdigit()
    ]
    return parent / f"version_{max(existing, default=0) + 1:04d}"


def write_version_pointer(
    run_dir: Path,
    name: str,
    version: Path,
    manifest: Mapping[str, object],
) -> None:
    value = {
        "created_at": utc_now(),
        "manifest_hash": manifest["manifest_hash"],
        "version_path": str(version.relative_to(run_dir)),
        "artifact_index_sha256": sha256_file(version / ARTIFACT_INDEX_NAME),
        "validation_sha256": sha256_file(version / VALIDATION_NAME),
    }
    value["pointer_hash"] = digest(value)
    atomic_write_json(run_dir / name, value)


def make_tree_read_only(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            item.chmod(0o444)
    for item in sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        item.chmod(0o555)
    path.chmod(0o555)


def create_development(output_dir: Path, require_clean: bool = True) -> Path:
    if require_clean and not git_clean():
        raise RuntimeError("development manifest requires a clean committed worktree")
    spec = load_positive_need_spec()
    environments = build_development_environments(spec)
    development = dict(spec["development"])  # type: ignore[arg-type]
    numerical_cases = build_numerical_validation_cases(spec)
    latent_support_table = build_latent_support_table(spec)
    tasks = [
        {
            "task_index": index,
            "task_type": "environment",
            "environment": environment.name,
        }
        for index, environment in enumerate(environments)
    ]
    environment_task_count = len(tasks)
    tasks.extend(
        {
            "task_index": environment_task_count + offset,
            "task_type": "numerical_validation",
            "case_id": int(case["case_id"]),
        }
        for offset, case in enumerate(numerical_cases)
    )
    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_classification": ANALYSIS_LABEL,
        "analysis": "development",
        "scientific_status": "exploratory_environment_selection",
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": git_commit(),
        "git_tree_hash": git_tree_hash(),
        "git_clean_at_creation": git_clean(),
        "spec_path": str(DEFAULT_SPEC_PATH.relative_to(PROJECT_ROOT)),
        "spec_hash": sha256_file(DEFAULT_SPEC_PATH),
        "source_hashes": source_hashes(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "hostname": platform.node(),
        },
        "episodes_per_environment": int(development["episodes_per_environment"]),
        "observations_per_person": 25,
        "seed_namespace": int(development["seed_namespace_offset"]),
        "manual_samples_per_person": list(development["manual_samples_per_person"]),
        "episode_schema": list(DEVELOPMENT_EPISODE_SCHEMA),
        "episode_schema_hash": digest(list(DEVELOPMENT_EPISODE_SCHEMA)),
        "expected_rows_per_task": int(development["episodes_per_environment"])
        * (len(development["manual_samples_per_person"]) + 1),
        "numerical_validation_cases": numerical_cases,
        "numerical_validation_cases_hash": digest(numerical_cases),
        "latent_support_table_hash": digest(latent_support_table),
        "environment_task_count": len(environments),
        "numerical_task_count": len(numerical_cases),
        "task_count": len(tasks),
        "environments": [environment_record(item) for item in environments],
        "tasks": tasks,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / MANIFEST_NAME
    atomic_write_json(path, manifest)
    return path


def validate_development_for_confirmation(development_dir: Path) -> Dict[str, object]:
    manifest_path = development_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    validate_source_identity(manifest)
    if manifest.get("analysis") != "development":
        raise RuntimeError("confirmation requires a development manifest")
    if manifest.get("episode_schema") != list(DEVELOPMENT_EPISODE_SCHEMA):
        raise RuntimeError("development episode schema does not match the implementation")
    if manifest.get("episode_schema_hash") != digest(list(DEVELOPMENT_EPISODE_SCHEMA)):
        raise RuntimeError("development episode schema hash mismatch")
    version, pointer = load_version_pointer(development_dir, CURRENT_POINTER_NAME)
    completion_path = version / COMPLETION_NAME
    validation_path = version / VALIDATION_NAME
    index_path = version / ARTIFACT_INDEX_NAME
    for path in (completion_path, validation_path, index_path):
        if not path.exists():
            raise RuntimeError(f"validated development artifact is missing: {path.name}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not completion.get("pipeline_complete") or not completion.get("scientific_completion"):
        raise RuntimeError("development has not completed qacct and local read-back validation")
    if not completion.get("local_readback_valid"):
        raise RuntimeError("development local read-back validation is absent")
    if completion.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("development completion manifest hash mismatch")
    if not validation.get("valid") or validation.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("development validation evidence is invalid")
    if validation.get("artifact_index_hash") != sha256_file(index_path):
        raise RuntimeError("development artifact index hash mismatch")
    if completion.get("validation_hash") != sha256_file(validation_path):
        raise RuntimeError("development validation hash mismatch")
    qacct_path = version / "qacct_evidence.json"
    if not qacct_path.exists():
        raise RuntimeError("development qacct evidence is missing")
    if completion.get("qacct_hash") != sha256_file(qacct_path):
        raise RuntimeError("development qacct hash mismatch")
    required = {
        "development_summary.csv",
        "initial_information_values.csv",
        "selected_target_control.json",
        "numerical_validation.csv",
        "numerical_validation_summary.json",
        "latent_support.csv",
    }
    if not required.issubset(index):
        raise RuntimeError("development artifact index is incomplete")
    for name, evidence in index.items():
        path = version / name
        if not path.exists() or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"development artifact mismatch: {name}")
    numerical_summary = json.loads(
        (version / "numerical_validation_summary.json").read_text(encoding="utf-8")
    )
    if not numerical_summary.get("valid") or int(numerical_summary.get("case_count", 0)) != 90:
        raise RuntimeError("development numerical convergence suite did not pass")
    summary_path = version / "development_summary.csv"
    summaries = read_csv(summary_path)
    recomputed = select_target_control_pair(summaries)
    selection_path = version / "selected_target_control.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if canonical_json(selection) != canonical_json(recomputed):
        raise RuntimeError("development selection does not match recomputed selection")
    if selection.get("selection_status") != "selected_without_rr_behavior":
        raise RuntimeError("development did not select a valid target/control pair")
    spec = load_positive_need_spec()
    for role in ("target", "control"):
        environment = find_environment(str(selection[f"{role}_environment"]), spec)
        if environment.environment_hash != selection[f"{role}_environment_hash"]:
            raise RuntimeError(f"{role} environment hash mismatch")
    if pointer["artifact_index_sha256"] != sha256_file(index_path):
        raise RuntimeError("development current-version pointer is stale")
    validate_qacct_evidence(manifest_path, version)
    validate_collected_semantics(manifest_path, version, require_pending_candidate=False)
    return selection


def create_confirmation(
    output_dir: Path,
    development_dir: Path,
    mode: str,
    require_clean: bool = True,
) -> Path:
    if mode not in {"smoke", "serious"}:
        raise ValueError("confirmation mode must be smoke or serious")
    if require_clean and not git_clean():
        raise RuntimeError("confirmation manifest requires a clean committed worktree")
    selection = validate_development_for_confirmation(development_dir)
    development_version, development_pointer = load_version_pointer(
        development_dir, CURRENT_POINTER_NAME
    )
    selection_path = development_version / "selected_target_control.json"
    spec = load_positive_need_spec()
    settings = dict(spec[mode])  # type: ignore[arg-type]
    names = [str(selection["target_environment"]), str(selection["control_environment"])]
    environments = [find_environment(name, spec) for name in names]
    target_config = asdict(environments[0].config)
    control_config = asdict(environments[1].config)
    target_cost = target_config.pop("sample_time_cost")
    control_cost = control_config.pop("sample_time_cost")
    if target_config != control_config or environments[0].prior.support_hash != environments[1].prior.support_hash:
        raise RuntimeError("target and control differ in more than sampling time cost")
    if target_cost == control_cost:
        raise RuntimeError("target and control sampling time costs must differ")
    development_manifest = load_manifest(development_dir / MANIFEST_NAME)
    episodes = int(
        settings["episodes_per_environment"]
        if mode == "smoke"
        else settings["episodes_per_condition"]
    )
    tasks = []
    for environment_role, environment in zip(("target", "control"), environments):
        for episode_index in range(episodes):
            tasks.append(
                {
                    "task_index": len(tasks),
                    "environment_role": environment_role,
                    "environment": environment.name,
                    "episode_index": episode_index,
                }
            )
    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_classification": ANALYSIS_LABEL,
        "analysis": "confirmation",
        "mode": mode,
        "scientific_status": "smoke_only" if mode == "smoke" else "held_out_confirmation",
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": git_commit(),
        "git_tree_hash": git_tree_hash(),
        "git_clean_at_creation": git_clean(),
        "spec_path": str(DEFAULT_SPEC_PATH.relative_to(PROJECT_ROOT)),
        "spec_hash": sha256_file(DEFAULT_SPEC_PATH),
        "source_hashes": source_hashes(),
        "development_manifest_hash": development_manifest["manifest_hash"],
        "development_validated_version": development_pointer["version_path"],
        "development_validated_pointer_hash": development_pointer["pointer_hash"],
        "development_seed_namespace": development_manifest["seed_namespace"],
        "selection_hash": sha256_file(selection_path),
        "selection": selection,
        "episode_schema": list(CONFIRMATION_EPISODE_SCHEMA),
        "episode_schema_hash": digest(list(CONFIRMATION_EPISODE_SCHEMA)),
        "expected_rows_per_task": 4,
        "runtime": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "hostname": platform.node(),
        },
        "episodes_per_condition": episodes,
        "observations_per_person": int(settings.get("observations_per_person", 25)),
        "seed_namespace": int(settings["seed_namespace_offset"]),
        "task_count": len(tasks),
        "expected_episode_rows": len(tasks) * 4,
        "environments": [environment_record(item) for item in environments],
        "tasks": tasks,
    }
    if int(manifest["seed_namespace"]) == int(manifest["development_seed_namespace"]):
        raise RuntimeError("development and confirmation seed namespaces overlap")
    manifest["manifest_hash"] = manifest_hash(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / MANIFEST_NAME
    atomic_write_json(path, manifest)
    return path


def task_directory(manifest_path: Path, task_index: int) -> Path:
    return manifest_path.parent / "shards" / f"task_{task_index:06d}"


def validate_source_identity(manifest: Mapping[str, object]) -> None:
    if not git_clean():
        raise RuntimeError("execution requires a clean worktree")
    if git_commit() != manifest["git_commit"]:
        raise RuntimeError("execution commit differs from manifest commit")
    if git_tree_hash() != manifest["git_tree_hash"]:
        raise RuntimeError("execution tree differs from manifest tree")
    if source_hashes() != manifest["source_hashes"]:
        raise RuntimeError("implementation sources differ from manifest")
    if sha256_file(DEFAULT_SPEC_PATH) != manifest["spec_hash"]:
        raise RuntimeError("frozen spec differs from manifest")


def current_sge_task_id() -> str:
    """Return a real array task ID, ignoring Hoffman's non-array sentinel."""
    value = os.environ.get("SGE_TASK_ID", "").strip()
    return value if value.isdigit() and int(value) > 0 else ""


def parse_quadrature_order_pair(value: str) -> tuple[int, int]:
    try:
        primary_text, reference_text = value.split(":", 1)
        primary = int(primary_text)
        reference = int(reference_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("order pairs must use PRIMARY:REFERENCE") from error
    if primary < 3 or reference <= primary or primary % 2 == 0 or reference % 2 == 0:
        raise argparse.ArgumentTypeError(
            "quadrature orders must be odd and REFERENCE must exceed PRIMARY"
        )
    return primary, reference


def diagnose_quadrature_orders(
    case_id: int,
    order_pairs: Sequence[tuple[int, int]],
) -> List[Dict[str, object]]:
    spec = load_positive_need_spec()
    cases = build_numerical_validation_cases(spec)
    if not 0 <= case_id < len(cases):
        raise ValueError("numerical case ID is outside the frozen suite")
    rows = []
    for primary, reference in order_pairs:
        diagnostic_spec = deepcopy(spec)
        numerical = diagnostic_spec["numerical_settings"]
        numerical["matched_voi_gauss_hermite_order"] = primary  # type: ignore[index]
        numerical["gauss_hermite_reference_order"] = reference  # type: ignore[index]
        row = validate_numerical_case(cases[case_id], diagnostic_spec)
        row["diagnostic_only"] = True
        rows.append(row)
    return rows


def diagnose_quadrature_suite(
    order_pairs: Sequence[tuple[int, int]],
) -> Dict[str, object]:
    """Evaluate candidate GH order pairs on all frozen numerical cases."""

    if not order_pairs:
        raise ValueError("at least one quadrature order pair is required")
    spec = load_positive_need_spec()
    cases = build_numerical_validation_cases(spec)
    if len(cases) != 90:
        raise RuntimeError("the frozen numerical suite must contain exactly 90 beliefs")

    numerical = dict(spec["numerical_settings"])  # type: ignore[arg-type]
    per_case: List[Dict[str, object]] = []
    aggregates: List[Dict[str, object]] = []
    for primary, reference in order_pairs:
        diagnostic_spec = deepcopy(spec)
        diagnostic_numerical = diagnostic_spec["numerical_settings"]
        diagnostic_numerical["matched_voi_gauss_hermite_order"] = primary  # type: ignore[index]
        diagnostic_numerical["gauss_hermite_reference_order"] = reference  # type: ignore[index]
        pair_rows = []
        for case in cases:
            row = validate_numerical_case(case, diagnostic_spec)
            row["diagnostic_only"] = True
            pair_rows.append(row)
        dense_count = sum(
            float(row["dense_reference_performed"]) >= 0.5 for row in pair_rows
        )
        if dense_count != 36:
            raise RuntimeError(
                "the frozen numerical suite must contain exactly 36 dense references"
            )
        summary = summarize_numerical_validation(pair_rows)
        aggregates.append(
            {
                "diagnostic_only": True,
                "gh_order": primary,
                "gh_reference_order": reference,
                "dense_reference_case_count": dense_count,
                **summary,
            }
        )
        per_case.extend(pair_rows)

    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "frozen_case_count": len(cases),
        "frozen_dense_reference_case_count": 36,
        "frozen_case_suite_hash": digest(cases),
        "frozen_numerical_settings": numerical,
        "per_case": per_case,
        "aggregate": aggregates,
    }


def run_task(manifest_path: Path, task_index: int) -> None:
    manifest = load_manifest(manifest_path)
    validate_source_identity(manifest)
    tasks = list(manifest["tasks"])  # type: ignore[arg-type]
    if not 0 <= task_index < len(tasks):
        raise ValueError("task index is outside manifest")
    task = dict(tasks[task_index])
    if int(task["task_index"]) != task_index:
        raise RuntimeError("task index mapping mismatch")
    output = task_directory(manifest_path, task_index)
    with exclusive_lock(output.with_suffix(".lock")):
        if (output / "status.json").exists():
            try:
                existing = json.loads(
                    (output / "status.json").read_text(encoding="utf-8")
                )
                if (
                    existing.get("status") == "complete"
                    and existing.get("manifest_hash") == manifest["manifest_hash"]
                ):
                    validate_task(manifest_path, task_index)
                    return
            except Exception:
                if output.exists():
                    shutil.rmtree(output)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        spec = load_positive_need_spec()
        if manifest["analysis"] == "development" and task.get("task_type") == "numerical_validation":
            case_id = int(task["case_id"])
            cases = list(manifest["numerical_validation_cases"])  # type: ignore[arg-type]
            if not 0 <= case_id < len(cases) or int(cases[case_id]["case_id"]) != case_id:
                raise RuntimeError("numerical case mapping mismatch")
            numerical_row = validate_numerical_case(dict(cases[case_id]), spec)
            atomic_write_json(temporary / "numerical_validation.json", numerical_row)
            scientific_files = ("numerical_validation.json",)
        elif manifest["analysis"] == "development":
            environment = find_environment(str(task["environment"]), spec)
            episodes = build_finite_support_episodes(
                environment,
                n_episodes=int(manifest["episodes_per_environment"]),
                stage="development",
                seed_namespace=int(manifest["seed_namespace"]),
                observations_per_person=int(manifest["observations_per_person"]),
                balanced_atoms=True,
            )
            numerical = dict(spec["numerical_settings"])  # type: ignore[arg-type]
            rows = evaluate_fixed_budgets(
                environment,
                episodes,
                samples_per_person=[int(value) for value in manifest["manual_samples_per_person"]],
                allocation_tolerance=float(numerical["allocation_tolerance"]),
                oracle_grid_size=int(numerical["oracle_grid_size"]),
            )
            information = initial_information_values(
                environment,
                quadrature_order=int(numerical["matched_voi_gauss_hermite_order"]),
            )
            summary = summarize_development_environment(environment, rows, information)
            write_csv(
                temporary / "episodes.csv",
                rows,
                expected_fields=DEVELOPMENT_EPISODE_SCHEMA,
            )
            atomic_write_json(temporary / "information.json", information)
            atomic_write_json(temporary / "summary.json", summary)
            scientific_files = ("episodes.csv", "information.json", "summary.json")
        else:
            environment = find_environment(str(task["environment"]), spec)
            episode_index = int(task["episode_index"])
            episodes = build_finite_support_episodes(
                environment,
                n_episodes=1,
                episode_start=episode_index,
                stage=str(manifest["mode"]),
                seed_namespace=int(manifest["seed_namespace"]),
                observations_per_person=int(manifest["observations_per_person"]),
                balanced_atoms=str(manifest["mode"]) == "smoke",
            )
            numerical = dict(spec["numerical_settings"])  # type: ignore[arg-type]
            development = dict(spec["development"])  # type: ignore[arg-type]
            rows = evaluate_serious_environment(
                environment,
                episodes,
                quadrature_order=int(numerical["matched_voi_gauss_hermite_order"]),
                manual_samples_per_person=int(development["confirmation_manual_samples_per_person"]),
                allocation_tolerance=float(numerical["allocation_tolerance"]),
                oracle_grid_size=int(numerical["oracle_grid_size"]),
            )
            write_csv(
                temporary / "episodes.csv",
                rows,
                expected_fields=CONFIRMATION_EPISODE_SCHEMA,
            )
            scientific_files = ("episodes.csv",)
        file_evidence = {
            name: {
                "sha256": sha256_file(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in scientific_files
        }
        status = {
            "status": "complete",
            "completed_at": utc_now(),
            "task_index": task_index,
            "task": task,
            "manifest_hash": manifest["manifest_hash"],
            "git_commit": manifest["git_commit"],
            "scheduler_metadata": {
                "hostname": platform.node(),
                "job_id": os.environ.get("JOB_ID", ""),
                "sge_task_id": current_sge_task_id(),
            },
            "files": file_evidence,
        }
        status["status_hash"] = digest(status)
        atomic_write_json(temporary / "status.json", status)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)


CSV_STRING_FIELDS = {
    "analysis_classification",
    "environment",
    "environment_hash",
    "support_hash",
    "gap_class",
    "episode_fingerprint",
    "stage",
    "observation_stream_hash_1",
    "observation_stream_hash_2",
    "observation_residual_hash_1",
    "observation_residual_hash_2",
    "policy",
    "terminal_belief_hash",
}


def _finite_float(row: Mapping[str, str], field: str, *, allow_blank: bool = False) -> float:
    value = row.get(field, "")
    if value == "" and allow_blank:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"non-numeric field {field}: {value!r}") from error
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite field {field}: {value!r}")
    return number


def _validate_numeric_cells(rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        for field, value in row.items():
            if field in CSV_STRING_FIELDS or value == "":
                continue
            _finite_float(row, field)


def _validate_episode_domains(
    rows: Sequence[Mapping[str, str]],
    environment: PositiveNeedEnvironment,
) -> None:
    _validate_numeric_cells(rows)
    for row in rows:
        if row["analysis_classification"] != ANALYSIS_LABEL:
            raise RuntimeError("analysis classification mismatch")
        if not row["episode_fingerprint"] or not row["terminal_belief_hash"]:
            raise RuntimeError("episode or terminal-belief fingerprint is missing")
        for field in (
            "need_1",
            "need_2",
            "total_true_need",
            "realized_true_need_gap",
            "realized_utility",
            "realized_outcome_gap",
            "true_equal_outcome_allocation",
            "true_equal_outcome_allocation_gap",
            "true_equal_outcome",
            "closer_to_true_equal_outcome_than_equal_split",
        ):
            _finite_float(row, field)
        allocation = _finite_float(row, "allocation_to_person1")
        remaining = _finite_float(row, "remaining_time")
        samples = _finite_float(row, "online_sample_count")
        count_1 = _finite_float(row, "sample_count_1")
        count_2 = _finite_float(row, "sample_count_2")
        if not 0.0 <= allocation <= 1.0:
            raise RuntimeError("allocation is outside [0, 1]")
        if not 0.0 <= remaining <= environment.config.total_time:
            raise RuntimeError("remaining time is outside the environment horizon")
        if any(value < 0.0 or value != int(value) for value in (samples, count_1, count_2)):
            raise RuntimeError("sample counts must be nonnegative integers")
        if samples != count_1 + count_2:
            raise RuntimeError("online sample count does not equal per-person counts")
        if samples > float(environment.config.max_meta_samples or 0):
            raise RuntimeError("sample count exceeds max_meta_samples")
        if _finite_float(row, "need_1") <= 0.0 or _finite_float(row, "need_2") <= 0.0:
            raise RuntimeError("positive-need run contains a nonpositive need")
        if _finite_float(row, "positive_need") != 1.0:
            raise RuntimeError("positive_need flag is invalid")
        if _finite_float(row, "max_observation_reconstruction_error_1") > 1e-12:
            raise RuntimeError("person-1 observation reconstruction error")
        if _finite_float(row, "max_observation_reconstruction_error_2") > 1e-12:
            raise RuntimeError("person-2 observation reconstruction error")
        weight_sum = _finite_float(row, "posterior_weight_sum")
        weight_min = _finite_float(row, "posterior_weight_min")
        weight_max = _finite_float(row, "posterior_weight_max")
        if abs(weight_sum - 1.0) > 1e-10 or weight_min < -1e-12 or weight_max > 1.0 + 1e-12:
            raise RuntimeError("posterior weights are invalid")
        if _finite_float(row, "posterior_weights_finite") != 1.0:
            raise RuntimeError("posterior weights are non-finite")
        if row["policy"] == POLICY_ORACLE:
            for field in (
                "oracle_grid_optimality_violation",
                "true_equal_outcome_regret",
                "equal_split_regret",
            ):
                _finite_float(row, field)
        if row["policy"] in {POLICY_RR, POLICY_MANUAL} and "time_matched_oracle_utility" in row:
            for field in (
                "time_matched_oracle_allocation",
                "time_matched_oracle_utility",
                "time_matched_oracle_raw_regret",
            ):
                _finite_float(row, field)


def _validate_development_task(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    directory: Path,
) -> None:
    episode_path = directory / "episodes.csv"
    require_exact_csv_schema(episode_path, manifest["episode_schema"])  # type: ignore[arg-type]
    rows = read_csv(episode_path)
    if len(rows) != int(manifest["expected_rows_per_task"]):
        raise RuntimeError("development task row count mismatch")
    spec = load_positive_need_spec()
    environment = find_environment(str(task["environment"]), spec)
    expected_policies = {
        *(f"manual_equal_outcome_{int(value)}_per_person" for value in manifest["manual_samples_per_person"]),  # type: ignore[union-attr]
        POLICY_ORACLE,
    }
    grouped: Dict[int, List[Mapping[str, str]]] = {}
    for row in rows:
        if row["environment"] != environment.name:
            raise RuntimeError("development task contains the wrong environment")
        if row["environment_hash"] != environment.environment_hash:
            raise RuntimeError("development environment hash mismatch")
        if row["support_hash"] != environment.prior.support_hash:
            raise RuntimeError("development support hash mismatch")
        if row["stage"] != "development":
            raise RuntimeError("development stage label mismatch")
        if int(_finite_float(row, "seed_namespace")) != int(manifest["seed_namespace"]):
            raise RuntimeError("development seed namespace mismatch")
        grouped.setdefault(int(_finite_float(row, "episode_index")), []).append(row)
    expected_indices = set(range(int(manifest["episodes_per_environment"])))
    if set(grouped) != expected_indices:
        raise RuntimeError("development episode index set is incomplete")
    for episode_rows in grouped.values():
        if {row["policy"] for row in episode_rows} != expected_policies:
            raise RuntimeError("development policy Cartesian set mismatch")
        if len(episode_rows) != len(expected_policies):
            raise RuntimeError("development contains duplicate policy rows")
    _validate_episode_domains(rows, environment)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    information = json.loads((directory / "information.json").read_text(encoding="utf-8"))
    for value in (summary, information):
        if value.get("environment") != environment.name:
            raise RuntimeError("development metadata environment mismatch")
        if value.get("environment_hash") != environment.environment_hash:
            raise RuntimeError("development metadata environment hash mismatch")
        if value.get("support_hash") != environment.prior.support_hash:
            raise RuntimeError("development metadata support hash mismatch")
    recomputed_summary = summarize_development_environment(environment, rows, information)
    if canonical_json(summary) != canonical_json(recomputed_summary):
        raise RuntimeError("development summary does not match validated episode rows")


def _validate_confirmation_task(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    directory: Path,
) -> None:
    episode_path = directory / "episodes.csv"
    require_exact_csv_schema(episode_path, manifest["episode_schema"])  # type: ignore[arg-type]
    rows = read_csv(episode_path)
    if len(rows) != int(manifest["expected_rows_per_task"]):
        raise RuntimeError("confirmation task row count mismatch")
    spec = load_positive_need_spec()
    environment = find_environment(str(task["environment"]), spec)
    expected_episode = int(task["episode_index"])
    if {row["policy"] for row in rows} != set(SERIOUS_POLICY_ORDER):
        raise RuntimeError("confirmation policy set mismatch")
    if len({(row["policy"], row["episode_index"]) for row in rows}) != 4:
        raise RuntimeError("confirmation contains duplicate policy rows")
    for row in rows:
        if row["environment"] != environment.name or int(row["episode_index"]) != expected_episode:
            raise RuntimeError("confirmation task key mismatch")
        if row["environment_hash"] != environment.environment_hash:
            raise RuntimeError("confirmation environment hash mismatch")
        if row["support_hash"] != environment.prior.support_hash:
            raise RuntimeError("confirmation support hash mismatch")
        if row["stage"] != str(manifest["mode"]):
            raise RuntimeError("confirmation stage label mismatch")
        if int(_finite_float(row, "seed_namespace")) != int(manifest["seed_namespace"]):
            raise RuntimeError("confirmation seed namespace mismatch")
        _finite_float(row, "initial_oracle_utility")
        _finite_float(row, "utility_regret_to_initial_oracle")
    _validate_episode_domains(rows, environment)
    oracle = next(row for row in rows if row["policy"] == POLICY_ORACLE)
    oracle_utility = _finite_float(oracle, "realized_utility")
    for row in rows:
        if _finite_float(row, "realized_utility") > oracle_utility + 1e-9:
            raise RuntimeError("full-information oracle dominance violation")
        if row["policy"] in {POLICY_RR, POLICY_MANUAL}:
            matched = _finite_float(row, "time_matched_oracle_utility")
            if _finite_float(row, "realized_utility") > matched + 1e-9:
                raise RuntimeError("time-matched oracle dominance violation")


def _validate_numerical_task(
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    directory: Path,
) -> None:
    case_id = int(task["case_id"])
    cases = list(manifest["numerical_validation_cases"])  # type: ignore[arg-type]
    if not 0 <= case_id < len(cases):
        raise RuntimeError("numerical case ID is outside the frozen suite")
    expected_case = dict(cases[case_id])
    row = json.loads((directory / "numerical_validation.json").read_text(encoding="utf-8"))
    validate_numerical_action_value_maps(row)
    for field, expected in expected_case.items():
        if row.get(field) != expected:
            raise RuntimeError(f"numerical case provenance mismatch: {field}")
    numerical = dict(load_positive_need_spec()["numerical_settings"])  # type: ignore[arg-type]
    for field in (
        "gh_order",
        "gh_reference_order",
        "gh_max_action_value_error",
        "terminal_grid_allocation_error",
        "terminal_grid_value_error",
        "dense_reference_error",
        "dense_reference_performed",
        "passed",
    ):
        if not math.isfinite(float(row[field])):
            raise RuntimeError(f"non-finite numerical validation field: {field}")
    if float(row.get("passed", 0.0)) != 1.0:
        raise RuntimeError("numerical convergence case failed")
    if float(row["gh_max_action_value_error"]) > float(
        numerical["action_value_convergence_tolerance"]
    ):
        raise RuntimeError("GH convergence threshold exceeded")
    if float(row["terminal_grid_allocation_error"]) > float(
        numerical["allocation_convergence_tolerance"]
    ):
        raise RuntimeError("terminal allocation convergence threshold exceeded")
    if float(row["terminal_grid_value_error"]) > float(
        numerical["action_value_convergence_tolerance"]
    ):
        raise RuntimeError("terminal value convergence threshold exceeded")
    if float(row["dense_reference_error"]) > float(
        numerical["action_value_convergence_tolerance"]
    ):
        raise RuntimeError("dense integration convergence threshold exceeded")
    if row["gh_action"] != row["gh_reference_action"]:
        raise RuntimeError("GH orders select different actions")
    if row["gh_action"] != row["terminal_reference_action"]:
        raise RuntimeError("terminal grids select different actions")
    if row["gh_action"] != row["dense_reference_action"]:
        raise RuntimeError("dense reference selects a different action")
    environment = find_environment(str(expected_case["environment"]), load_positive_need_spec())
    expected_dense = (
        environment.sample_time_cost == 0.02
        and expected_case["belief_kind"]
        in {
            "uniform_prior",
            "person1_predictive_mean",
            "person1_minimum_support",
            "person1_maximum_support",
        }
    )
    if (float(row["dense_reference_performed"]) >= 0.5) != expected_dense:
        raise RuntimeError("dense-reference case classification mismatch")


def validate_task(manifest_path: Path, task_index: int) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    directory = task_directory(manifest_path, task_index)
    status_path = directory / "status.json"
    if not status_path.exists():
        raise RuntimeError(f"missing task status: {task_index}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status_hash = status.pop("status_hash", None)
    if status_hash != digest(status):
        raise RuntimeError(f"task status hash mismatch: {task_index}")
    status["status_hash"] = status_hash
    if status.get("status") != "complete":
        raise RuntimeError(f"task not complete: {task_index}")
    if int(status.get("task_index", -1)) != task_index:
        raise RuntimeError(f"task index mismatch: {task_index}")
    if status.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError(f"task manifest mismatch: {task_index}")
    task = dict(manifest["tasks"][task_index])  # type: ignore[index]
    if status.get("task") != task:
        raise RuntimeError(f"task definition mismatch: {task_index}")
    if status.get("git_commit") != manifest["git_commit"]:
        raise RuntimeError(f"task commit mismatch: {task_index}")
    scheduler_metadata = dict(status.get("scheduler_metadata", {}))
    if scheduler_metadata.get("sge_task_id") not in ("", str(task_index + 1)):
        raise RuntimeError(f"scheduler task ID mismatch: {task_index}")
    if manifest["analysis"] == "development" and task.get("task_type") == "numerical_validation":
        expected_files = {"numerical_validation.json"}
    elif manifest["analysis"] == "development":
        expected_files = {"episodes.csv", "information.json", "summary.json"}
    else:
        expected_files = {"episodes.csv"}
    if set(status.get("files", {})) != expected_files:
        raise RuntimeError(f"task scientific file set mismatch: {task_index}")
    for name, evidence in dict(status["files"]).items():
        path = directory / name
        if not path.exists() or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"task file hash mismatch: {task_index}/{name}")
    if manifest["analysis"] == "development" and task.get("task_type") == "numerical_validation":
        _validate_numerical_task(manifest, task, directory)
    elif manifest["analysis"] == "development":
        _validate_development_task(manifest, task, directory)
    else:
        _validate_confirmation_task(manifest, task, directory)
    return status


def progress(manifest_path: Path) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    valid = []
    invalid = []
    missing = []
    for task in manifest["tasks"]:  # type: ignore[union-attr]
        task_index = int(task["task_index"])
        if not (task_directory(manifest_path, task_index) / "status.json").exists():
            missing.append(task_index)
            continue
        try:
            validate_task(manifest_path, task_index)
            valid.append(task_index)
        except Exception as error:
            invalid.append({"task_index": task_index, "error": str(error)})
    value = {
        "checked_at": utc_now(),
        "analysis": manifest["analysis"],
        "mode": manifest.get("mode", "development"),
        "task_count": manifest["task_count"],
        "valid_task_count": len(valid),
        "missing_task_count": len(missing),
        "invalid_task_count": len(invalid),
        "valid_tasks": valid,
        "missing_tasks": missing,
        "invalid_tasks": invalid,
        "complete": len(valid) == int(manifest["task_count"]) and not invalid,
    }
    atomic_write_json(manifest_path.parent / PROGRESS_NAME, value)
    return value


def validate_confirmation_collection(
    manifest: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
) -> None:
    expected_keys = {
        (str(task["environment"]), int(task["episode_index"]), policy)
        for task in manifest["tasks"]  # type: ignore[union-attr]
        for policy in SERIOUS_POLICY_ORDER
    }
    observed_keys = [
        (str(row["environment"]), int(row["episode_index"]), str(row["policy"]))
        for row in rows
    ]
    if len(observed_keys) != len(set(observed_keys)):
        raise RuntimeError("confirmation contains duplicate Cartesian keys")
    if set(observed_keys) != expected_keys:
        raise RuntimeError("confirmation Cartesian key set is incomplete or unexpected")
    validate_serious_common_randomness(rows)
    selection = dict(manifest["selection"])  # type: ignore[arg-type]
    target = str(selection["target_environment"])
    control = str(selection["control_environment"])
    fields = (
        "latent_atom_index",
        "need_1",
        "need_2",
        "total_true_need",
        "realized_true_need_gap",
        "orientation",
        "episode_fingerprint",
        "observation_residual_hash_1",
        "observation_residual_hash_2",
    )
    by_key = {
        (str(row["environment"]), int(row["episode_index"]), str(row["policy"])): row
        for row in rows
    }
    episodes = int(manifest["episodes_per_condition"])
    for episode_index in range(episodes):
        for policy in SERIOUS_POLICY_ORDER:
            target_row = by_key[(target, episode_index, policy)]
            control_row = by_key[(control, episode_index, policy)]
            for field in fields:
                if target_row[field] != control_row[field]:
                    raise RuntimeError(
                        f"target-control common-randomness mismatch at episode "
                        f"{episode_index}, policy {policy}: {field}"
                    )


def collect(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    validate_source_identity(manifest)
    run_dir = manifest_path.parent
    with exclusive_lock(run_dir / ".collect.lock"):
        state = progress(manifest_path)
        if not state["complete"]:
            raise RuntimeError("cannot collect incomplete or invalid tasks")
        scheduler = validate_scheduler_evidence(manifest_path)
        validate_scheduler_task_bindings(manifest_path, scheduler)
        version = next_version_path(run_dir, "candidate_versions")
        stage = version.with_name(f".{version.name}.tmp.{os.getpid()}")
        stage.mkdir(parents=True, exist_ok=False)
        try:
            if manifest["analysis"] == "development":
                summaries = []
                information = []
                numerical_rows = []
                for task in manifest["tasks"]:  # type: ignore[union-attr]
                    directory = task_directory(manifest_path, int(task["task_index"]))
                    if task.get("task_type") == "numerical_validation":
                        numerical_rows.append(
                            json.loads(
                                (directory / "numerical_validation.json").read_text(
                                    encoding="utf-8"
                                )
                            )
                        )
                    else:
                        summaries.append(
                            json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                        )
                        information.append(
                            json.loads(
                                (directory / "information.json").read_text(encoding="utf-8")
                            )
                        )
                expected_environments = {
                    str(task["environment"])
                    for task in manifest["tasks"]  # type: ignore[union-attr]
                    if task.get("task_type") == "environment"
                }
                if (
                    len(summaries) != int(manifest["environment_task_count"])
                    or {str(row["environment"]) for row in summaries} != expected_environments
                    or {str(row["environment"]) for row in information} != expected_environments
                ):
                    raise RuntimeError("development Cartesian environment set is incomplete")
                frozen_numerical_cases = build_numerical_validation_cases()
                if frozen_numerical_cases != manifest["numerical_validation_cases"]:
                    raise RuntimeError("numerical validation case manifest mismatch")
                if digest(frozen_numerical_cases) != manifest["numerical_validation_cases_hash"]:
                    raise RuntimeError("numerical validation case hash mismatch")
                if (
                    len(numerical_rows) != int(manifest["numerical_task_count"])
                    or {int(row["case_id"]) for row in numerical_rows}
                    != set(range(int(manifest["numerical_task_count"])))
                ):
                    raise RuntimeError("numerical validation Cartesian case set is incomplete")
                numerical_rows.sort(key=lambda row: int(row["case_id"]))
                if sum(
                    float(row["dense_reference_performed"]) >= 0.5
                    for row in numerical_rows
                ) != 36:
                    raise RuntimeError(
                        "numerical suite must contain exactly 36 dense-reference cases"
                    )
                numerical_summary = summarize_numerical_validation(numerical_rows)
                if not numerical_summary["valid"]:
                    raise RuntimeError("numerical convergence suite failed")
                selection = select_target_control_pair(summaries)
                latent_support = build_latent_support_table()
                if digest(latent_support) != manifest["latent_support_table_hash"]:
                    raise RuntimeError("latent-support table hash mismatch")
                write_csv(stage / "development_summary.csv", summaries)
                write_csv(stage / "initial_information_values.csv", information)
                atomic_write_json(stage / "selected_target_control.json", selection)
                write_csv(stage / "numerical_validation.csv", numerical_rows)
                atomic_write_json(
                    stage / "numerical_validation_summary.json", numerical_summary
                )
                write_csv(stage / "latent_support.csv", latent_support)
                outputs = (
                    "development_summary.csv",
                    "initial_information_values.csv",
                    "selected_target_control.json",
                    "numerical_validation.csv",
                    "numerical_validation_summary.json",
                    "latent_support.csv",
                )
                scientific_complete = (
                    selection.get("selection_status") == "selected_without_rr_behavior"
                )
            else:
                rows: List[Dict[str, str]] = []
                for task in manifest["tasks"]:  # type: ignore[union-attr]
                    rows.extend(
                        read_csv(
                            task_directory(manifest_path, int(task["task_index"]))
                            / "episodes.csv"
                        )
                    )
                if len(rows) != int(manifest["expected_episode_rows"]):
                    raise RuntimeError("confirmation episode row count mismatch")
                validate_confirmation_collection(manifest, rows)
                selection = dict(manifest["selection"])  # type: ignore[arg-type]
                summaries, comparisons, classification = summarize_serious(
                    rows,
                    target_environment=str(selection["target_environment"]),
                    control_environment=str(selection["control_environment"]),
                    selection=selection,
                    scientific_confirmation=str(manifest.get("mode")) == "serious",
                )
                if str(manifest.get("mode")) == "serious":
                    classification["pre_qacct_readiness_candidate"] = classification[
                        "candidate_readiness_classification"
                    ]
                    classification["readiness_classification"] = "invalid_evidence"
                    classification["evidence_status"] = "pending_qacct_and_local_readback"
                write_csv(stage / "confirmation_episodes.csv", rows)
                write_csv(stage / "policy_summary.csv", summaries)
                write_csv(stage / "paired_comparisons.csv", comparisons)
                atomic_write_json(stage / "readiness_classification.json", classification)
                outputs = (
                    "confirmation_episodes.csv",
                    "policy_summary.csv",
                    "paired_comparisons.csv",
                    "readiness_classification.json",
                )
                scientific_complete = str(manifest.get("mode")) == "serious"
            index = {
                name: {
                    "sha256": sha256_file(stage / name),
                    "bytes": (stage / name).stat().st_size,
                    "rows": len(read_csv(stage / name)) if name.endswith(".csv") else None,
                }
                for name in outputs
            }
            atomic_write_json(stage / ARTIFACT_INDEX_NAME, index)
            validation = {
                "validated_at": utc_now(),
                "valid": True,
                "task_count": manifest["task_count"],
                "manifest_hash": manifest["manifest_hash"],
                "artifact_index_hash": sha256_file(stage / ARTIFACT_INDEX_NAME),
                "scheduler_evidence_hash": sha256_file(
                    run_dir / "scheduler" / "jobs.json"
                ),
                "array_job_file_hash": scheduler["array_job_sha256"],
                "collector_job_file_hash": scheduler["collector_job_sha256"],
                "scientific_completion_pending_qacct": scientific_complete,
            }
            atomic_write_json(stage / VALIDATION_NAME, validation)
            completion = {
                "completed_at": utc_now(),
                "pipeline_complete": True,
                "scientific_completion": False,
                "reason": (
                    "qacct_and_local_readback_pending"
                    if scientific_complete
                    else "development_or_smoke"
                ),
                "manifest_hash": manifest["manifest_hash"],
                "validation_hash": sha256_file(stage / VALIDATION_NAME),
            }
            atomic_write_json(stage / COMPLETION_NAME, completion)
            os.replace(stage, version)
            write_version_pointer(
                run_dir, CANDIDATE_POINTER_NAME, version, manifest
            )
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise


def validate_scheduler_evidence(manifest_path: Path) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    run_dir = manifest_path.parent
    path = run_dir / "scheduler" / "jobs.json"
    if not path.exists():
        raise RuntimeError("scheduler jobs evidence is missing")
    scheduler = json.loads(path.read_text(encoding="utf-8"))
    if int(scheduler["task_count"]) != int(manifest["task_count"]):
        raise RuntimeError("scheduler task count mismatch")
    expected_phase = (
        "development"
        if manifest["analysis"] == "development"
        else str(manifest["mode"])
    )
    if scheduler["phase"] != expected_phase:
        raise RuntimeError("scheduler phase mismatch")
    if not 1 <= int(scheduler["throttle"]) <= int(manifest["task_count"]):
        raise RuntimeError("scheduler throttle is outside the task range")
    if int(scheduler["task_slots"]) != 1 or int(scheduler["collector_slots"]) != 1:
        raise RuntimeError("this workflow requires one-slot array and collector jobs")
    for role in ("array", "collector"):
        job_path = run_dir / str(scheduler[f"{role}_job_path"])
        if not job_path.exists() or sha256_file(job_path) != scheduler[f"{role}_job_sha256"]:
            raise RuntimeError(f"{role} scheduler job file mismatch")
        text = job_path.read_text(encoding="utf-8")
        expected_data = (
            scheduler["task_h_data"] if role == "array" else scheduler["collector_h_data"]
        )
        expected_time = (
            scheduler["task_h_rt"] if role == "array" else scheduler["collector_h_rt"]
        )
        for directive in (
            f"#$ -q {scheduler['queue']}",
            f"#$ -l h_rt={expected_time}",
            f"#$ -l h_data={expected_data}",
        ):
            if directive not in text:
                raise RuntimeError(f"{role} scheduler resource directive mismatch")
    array_text = (run_dir / str(scheduler["array_job_path"])).read_text(encoding="utf-8")
    if f"#$ -t 1-{manifest['task_count']}" not in array_text:
        raise RuntimeError("scheduler array range mismatch")
    if f"#$ -tc {scheduler['throttle']}" not in array_text:
        raise RuntimeError("scheduler array throttle directive mismatch")
    if not str(scheduler["array_job_id"]).isdigit() or not str(
        scheduler["collector_job_id"]
    ).isdigit():
        raise RuntimeError("scheduler job IDs are invalid")
    submission_evidence = dict(scheduler.get("submission_evidence", {}))
    if not submission_evidence:
        raise RuntimeError("raw qsub submission evidence is missing")
    for relative, evidence in submission_evidence.items():
        path = run_dir / relative
        if (
            not path.exists()
            or sha256_file(path) != evidence["sha256"]
            or path.stat().st_size != evidence["bytes"]
        ):
            raise RuntimeError(f"raw qsub evidence mismatch: {relative}")
    return scheduler


def validate_scheduler_task_bindings(
    manifest_path: Path,
    scheduler: Mapping[str, object],
) -> None:
    manifest = load_manifest(manifest_path)
    for task in manifest["tasks"]:  # type: ignore[union-attr]
        task_index = int(task["task_index"])
        status = json.loads(
            (task_directory(manifest_path, task_index) / "status.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = dict(status.get("scheduler_metadata", {}))
        if metadata.get("job_id") != str(scheduler["array_job_id"]):
            raise RuntimeError(f"task {task_index} scheduler job binding mismatch")
        if metadata.get("sge_task_id") != str(task_index + 1):
            raise RuntimeError(f"task {task_index} scheduler task binding mismatch")


def parse_qacct_records(output: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"=+", stripped):
            if current:
                records.append(current)
                current = {}
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        current[key] = value.strip()
    if current:
        records.append(current)
    return records


def audit_qacct(manifest_path: Path, job_ids: Sequence[str]) -> None:
    manifest = load_manifest(manifest_path)
    validate_source_identity(manifest)
    candidate, _ = load_version_pointer(manifest_path.parent, CANDIDATE_POINTER_NAME)
    scheduler = validate_scheduler_evidence(manifest_path)
    expected_job_ids = {
        str(scheduler["array_job_id"]),
        str(scheduler["collector_job_id"]),
    }
    if set(map(str, job_ids)) != expected_job_ids:
        raise RuntimeError("qacct audit must cover the array and collector jobs exactly")
    records = []
    raw_dir = candidate / "qacct_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for job_id in job_ids:
        output = subprocess.check_output(["qacct", "-j", str(job_id)], text=True)
        raw_path = raw_dir / f"job_{job_id}.txt"
        raw_path.write_text(output, encoding="utf-8")
        qacct_records = parse_qacct_records(output)
        if not qacct_records:
            raise RuntimeError(f"qacct record incomplete for job {job_id}")
        required_qacct_fields = {
            "qname", "hostname", "jobname", "jobnumber", "slots", "failed", "exit_status"
        }
        if any(not required_qacct_fields.issubset(record) for record in qacct_records):
            raise RuntimeError(f"qacct required fields are missing for job {job_id}")
        if any(record.get("jobnumber") != str(job_id) for record in qacct_records):
            raise RuntimeError(f"qacct job identity mismatch for job {job_id}")
        failed_values = [record.get("failed", "") for record in qacct_records]
        exit_values = [record.get("exit_status", "") for record in qacct_records]
        task_ids = [
            record["taskid"]
            for record in qacct_records
            if record.get("taskid") not in (None, "undefined", "NONE")
        ]
        if any(not value for value in failed_values + exit_values):
            raise RuntimeError(f"qacct record incomplete for job {job_id}")
        if any(value != "0" for value in failed_values + exit_values):
            raise RuntimeError(f"qacct failure for job {job_id}")
        if any(int(record.get("slots", "0")) != 1 for record in qacct_records):
            raise RuntimeError(f"qacct slot allocation mismatch for job {job_id}")
        if any(
            record.get("qname", "").split("@", 1)[0] != str(scheduler["queue"])
            for record in qacct_records
        ):
            raise RuntimeError(f"qacct queue mismatch for job {job_id}")
        expected_name = (
            scheduler["array_job_name"]
            if str(job_id) == str(scheduler["array_job_id"])
            else scheduler["collector_job_name"]
        )
        if any(record.get("jobname") != expected_name for record in qacct_records):
            raise RuntimeError(f"qacct job name mismatch for job {job_id}")
        if str(job_id) == str(scheduler["array_job_id"]):
            expected_tasks = {
                str(index) for index in range(1, int(manifest["task_count"]) + 1)
            }
            if set(task_ids) != expected_tasks or len(task_ids) != len(expected_tasks):
                raise RuntimeError("qacct array task coverage is incomplete")
        elif len(qacct_records) != 1 or task_ids:
            raise RuntimeError("qacct collector record is not unique")
        records.append(
            {
                "job_id": str(job_id),
                "failed_values": failed_values,
                "exit_status_values": exit_values,
                "task_ids": sorted(set(task_ids), key=lambda value: int(value)),
                "record_count": len(qacct_records),
                "qacct_records": qacct_records,
                "raw_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "raw_path": str(raw_path.relative_to(candidate)),
            }
        )
    evidence = {
        "audited_at": utc_now(),
        "manifest_hash": manifest["manifest_hash"],
        "jobs": records,
    }
    atomic_write_json(candidate / "qacct_evidence.json", evidence)


def _assert_artifact_index(version: Path) -> Dict[str, object]:
    index_path = version / ARTIFACT_INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for name, evidence in index.items():
        path = version / name
        if not path.exists() or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"artifact mismatch: {name}")
        if evidence.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"artifact size mismatch: {name}")
        if name.endswith(".csv") and evidence.get("rows") != len(read_csv(path)):
            raise RuntimeError(f"artifact row-count mismatch: {name}")
    return index


def _assert_table_matches(version: Path, name: str, rows: Sequence[Mapping[str, object]]) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / name
        write_csv(expected, rows)
        if sha256_file(expected) != sha256_file(version / name):
            raise RuntimeError(f"independent aggregate recomputation mismatch: {name}")


def validate_collected_semantics(
    manifest_path: Path,
    version: Path,
    require_pending_candidate: bool = True,
) -> None:
    manifest = load_manifest(manifest_path)
    index = _assert_artifact_index(version)
    validation = json.loads((version / VALIDATION_NAME).read_text(encoding="utf-8"))
    if validation.get("manifest_hash") != manifest["manifest_hash"] or not validation.get("valid"):
        raise RuntimeError("candidate validation manifest mismatch")
    if validation.get("artifact_index_hash") != sha256_file(version / ARTIFACT_INDEX_NAME):
        raise RuntimeError("candidate artifact-index hash mismatch")
    scheduler = validate_scheduler_evidence(manifest_path)
    if validation.get("scheduler_evidence_hash") != sha256_file(
        manifest_path.parent / "scheduler" / "jobs.json"
    ):
        raise RuntimeError("candidate scheduler-evidence hash mismatch")
    if validation.get("array_job_file_hash") != scheduler["array_job_sha256"]:
        raise RuntimeError("candidate array-job hash mismatch")
    if validation.get("collector_job_file_hash") != scheduler["collector_job_sha256"]:
        raise RuntimeError("candidate collector-job hash mismatch")
    completion = json.loads((version / COMPLETION_NAME).read_text(encoding="utf-8"))
    if completion.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("candidate completion manifest mismatch")
    if completion.get("validation_hash") != sha256_file(version / VALIDATION_NAME):
        raise RuntimeError("candidate completion validation hash mismatch")
    if require_pending_candidate and completion.get("scientific_completion"):
        raise RuntimeError("candidate was prematurely marked scientifically complete")
    if manifest["analysis"] == "development":
        summaries = []
        information = []
        numerical_rows = []
        for task in manifest["tasks"]:  # type: ignore[union-attr]
            directory = task_directory(manifest_path, int(task["task_index"]))
            if task.get("task_type") == "numerical_validation":
                numerical_rows.append(
                    json.loads(
                        (directory / "numerical_validation.json").read_text(encoding="utf-8")
                    )
                )
            else:
                summaries.append(
                    json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                )
                information.append(
                    json.loads((directory / "information.json").read_text(encoding="utf-8"))
                )
        numerical_rows.sort(key=lambda row: int(row["case_id"]))
        if len(numerical_rows) != 90 or sum(
            float(row["dense_reference_performed"]) >= 0.5 for row in numerical_rows
        ) != 36:
            raise RuntimeError("independent numerical-suite cardinality check failed")
        numerical_summary = summarize_numerical_validation(numerical_rows)
        if not numerical_summary["valid"]:
            raise RuntimeError("independent numerical-suite validation failed")
        _assert_table_matches(version, "development_summary.csv", summaries)
        _assert_table_matches(version, "initial_information_values.csv", information)
        _assert_table_matches(version, "numerical_validation.csv", numerical_rows)
        latent_support = build_latent_support_table()
        if digest(latent_support) != manifest["latent_support_table_hash"]:
            raise RuntimeError("independent latent-support table hash mismatch")
        _assert_table_matches(version, "latent_support.csv", latent_support)
        selection = select_target_control_pair(summaries)
        observed_selection = json.loads(
            (version / "selected_target_control.json").read_text(encoding="utf-8")
        )
        if canonical_json(selection) != canonical_json(observed_selection):
            raise RuntimeError("independent target/control selection mismatch")
        observed_numerical = json.loads(
            (version / "numerical_validation_summary.json").read_text(encoding="utf-8")
        )
        if canonical_json(numerical_summary) != canonical_json(observed_numerical):
            raise RuntimeError("independent numerical summary mismatch")
    else:
        rows: List[Dict[str, str]] = []
        for task in manifest["tasks"]:  # type: ignore[union-attr]
            rows.extend(
                read_csv(
                    task_directory(manifest_path, int(task["task_index"])) / "episodes.csv"
                )
            )
        if len(rows) != int(manifest["expected_episode_rows"]):
            raise RuntimeError("independent confirmation row-count mismatch")
        validate_confirmation_collection(manifest, rows)
        _assert_table_matches(version, "confirmation_episodes.csv", rows)
        selection = dict(manifest["selection"])  # type: ignore[arg-type]
        summaries, comparisons, classification = summarize_serious(
            rows,
            target_environment=str(selection["target_environment"]),
            control_environment=str(selection["control_environment"]),
            selection=selection,
            scientific_confirmation=str(manifest.get("mode")) == "serious",
        )
        if str(manifest.get("mode")) == "serious":
            classification["pre_qacct_readiness_candidate"] = classification[
                "candidate_readiness_classification"
            ]
            classification["readiness_classification"] = "invalid_evidence"
            classification["evidence_status"] = "pending_qacct_and_local_readback"
        _assert_table_matches(version, "policy_summary.csv", summaries)
        _assert_table_matches(version, "paired_comparisons.csv", comparisons)
        observed = json.loads(
            (version / "readiness_classification.json").read_text(encoding="utf-8")
        )
        if canonical_json(classification) != canonical_json(observed):
            raise RuntimeError("independent readiness recomputation mismatch")


def validate_qacct_evidence(
    manifest_path: Path,
    version: Path,
) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    evidence_path = version / "qacct_evidence.json"
    if not evidence_path.exists():
        raise RuntimeError("qacct evidence is missing")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("qacct evidence manifest hash mismatch")
    scheduler = validate_scheduler_evidence(manifest_path)
    expected_ids = {str(scheduler["array_job_id"]), str(scheduler["collector_job_id"])}
    records = list(evidence.get("jobs", []))
    if {str(record["job_id"]) for record in records} != expected_ids or len(records) != 2:
        raise RuntimeError("qacct job set mismatch")
    for record in records:
        raw_path = version / str(record["raw_path"])
        if not raw_path.exists() or sha256_file(raw_path) != record["raw_sha256"]:
            raise RuntimeError("raw qacct evidence mismatch")
        parsed = parse_qacct_records(raw_path.read_text(encoding="utf-8"))
        if canonical_json(parsed) != canonical_json(record["qacct_records"]):
            raise RuntimeError("parsed qacct records do not match stored evidence")
        required_qacct_fields = {
            "qname", "hostname", "jobname", "jobnumber", "slots", "failed", "exit_status"
        }
        if any(not required_qacct_fields.issubset(item) for item in parsed):
            raise RuntimeError("qacct required fields are missing")
        if any(
            str(value) != "0"
            for value in list(record["failed_values"]) + list(record["exit_status_values"])
        ):
            raise RuntimeError("qacct reports a failed task")
        if any(item.get("jobnumber") != str(record["job_id"]) for item in parsed):
            raise RuntimeError("qacct job identity mismatch")
        expected_name = (
            scheduler["array_job_name"]
            if str(record["job_id"]) == str(scheduler["array_job_id"])
            else scheduler["collector_job_name"]
        )
        if any(item.get("jobname") != expected_name for item in parsed):
            raise RuntimeError("qacct job name mismatch")
        if any(int(item.get("slots", "0")) != 1 for item in parsed):
            raise RuntimeError("qacct slot mismatch")
        if any(
            item.get("qname", "").split("@", 1)[0] != str(scheduler["queue"])
            for item in parsed
        ):
            raise RuntimeError("qacct queue mismatch")
    array = next(
        record for record in records if str(record["job_id"]) == str(scheduler["array_job_id"])
    )
    expected_tasks = {str(index) for index in range(1, int(manifest["task_count"]) + 1)}
    if set(map(str, array["task_ids"])) != expected_tasks:
        raise RuntimeError("qacct array task set mismatch")
    if int(array.get("record_count", 0)) != int(manifest["task_count"]):
        raise RuntimeError("qacct array record count mismatch")
    collector = next(
        record
        for record in records
        if str(record["job_id"]) == str(scheduler["collector_job_id"])
    )
    if int(collector.get("record_count", 0)) != 1 or collector.get("task_ids"):
        raise RuntimeError("qacct collector record mismatch")
    return evidence


def _verify_local_readback_locked(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    validate_source_identity(manifest)
    run_dir = manifest_path.parent
    if platform.system() != "Darwin":
        raise RuntimeError("independent read-back must run on the local Mac")
    if platform.node() == manifest["runtime"]["hostname"]:  # type: ignore[index]
        raise RuntimeError("independent read-back host matches the manifest creation host")
    candidate, _ = load_version_pointer(run_dir, CANDIDATE_POINTER_NAME)
    state = progress(manifest_path)
    if not state["complete"]:
        raise RuntimeError("task validation failed during final read-back")
    validate_scheduler_task_bindings(
        manifest_path, validate_scheduler_evidence(manifest_path)
    )
    validate_collected_semantics(manifest_path, candidate)
    validate_qacct_evidence(manifest_path, candidate)
    version = next_version_path(run_dir, "validated_versions")
    stage = version.with_name(f".{version.name}.tmp.{os.getpid()}")
    shutil.copytree(candidate, stage)
    output = stage
    index = _assert_artifact_index(output)
    if manifest["analysis"] == "confirmation":
        readiness_path = output / "readiness_classification.json"
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        if str(manifest.get("mode")) == "serious":
            readiness["readiness_classification"] = readiness[
                "candidate_readiness_classification"
            ]
            readiness["evidence_status"] = "validated_held_out_scientific_evidence"
        else:
            readiness["readiness_classification"] = "invalid_evidence"
            readiness["evidence_status"] = "smoke_only_not_scientific_evidence"
        atomic_write_json(readiness_path, readiness)
        index["readiness_classification.json"] = {
            "sha256": sha256_file(readiness_path),
            "bytes": readiness_path.stat().st_size,
            "rows": None,
        }
        atomic_write_json(output / ARTIFACT_INDEX_NAME, index)
    development_selected = False
    if manifest["analysis"] == "development":
        development_selected = (
            json.loads(
                (output / "selected_target_control.json").read_text(encoding="utf-8")
            ).get("selection_status")
            == "selected_without_rr_behavior"
        )
    scientific_completion = (
        development_selected
        or (
            manifest["analysis"] == "confirmation"
            and str(manifest.get("mode")) == "serious"
        )
    )
    validation = json.loads((output / VALIDATION_NAME).read_text(encoding="utf-8"))
    validation["artifact_index_hash"] = sha256_file(output / ARTIFACT_INDEX_NAME)
    validation["qacct_valid"] = True
    validation["local_readback_valid"] = True
    validation["scientific_completion_validated"] = scientific_completion
    atomic_write_json(output / VALIDATION_NAME, validation)
    completion = {
        "completed_at": utc_now(),
        "pipeline_complete": True,
        "scientific_completion": scientific_completion,
        "manifest_hash": manifest["manifest_hash"],
        "validation_hash": sha256_file(output / VALIDATION_NAME),
        "qacct_hash": sha256_file(output / "qacct_evidence.json"),
        "local_readback_valid": True,
        "local_readback_host": platform.node(),
        "local_readback_platform": platform.platform(),
        "local_readback_python": platform.python_version(),
    }
    atomic_write_json(output / COMPLETION_NAME, completion)
    os.replace(stage, version)
    make_tree_read_only(version)
    write_version_pointer(run_dir, CURRENT_POINTER_NAME, version, manifest)


def verify_local_readback(manifest_path: Path) -> None:
    with exclusive_lock(manifest_path.parent / ".local-readback.lock"):
        _verify_local_readback_locked(manifest_path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    create_dev = commands.add_parser("create-development")
    create_dev.add_argument("--output-dir", type=Path, required=True)
    create_dev.add_argument("--allow-dirty", action="store_true")
    create_conf = commands.add_parser("create-confirmation")
    create_conf.add_argument("--output-dir", type=Path, required=True)
    create_conf.add_argument("--development-dir", type=Path, required=True)
    create_conf.add_argument("--mode", choices=("smoke", "serious"), required=True)
    create_conf.add_argument("--allow-dirty", action="store_true")
    run = commands.add_parser("run-task")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    for name in ("progress", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
    audit = commands.add_parser("audit-qacct")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--job-id", action="append", required=True)
    readback = commands.add_parser("verify-local-readback")
    readback.add_argument("--manifest", type=Path, required=True)
    diagnose = commands.add_parser("diagnose-quadrature")
    diagnose.add_argument("--case-id", type=int, required=True)
    diagnose.add_argument(
        "--order-pair",
        type=parse_quadrature_order_pair,
        action="append",
        required=True,
    )
    diagnose_suite = commands.add_parser("diagnose-quadrature-suite")
    diagnose_suite.add_argument(
        "--order-pair",
        type=parse_quadrature_order_pair,
        action="append",
        required=True,
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "create-development":
        print(create_development(args.output_dir, require_clean=not args.allow_dirty))
    elif args.command == "create-confirmation":
        print(
            create_confirmation(
                args.output_dir,
                args.development_dir,
                args.mode,
                require_clean=not args.allow_dirty,
            )
        )
    elif args.command == "run-task":
        run_task(args.manifest, args.task_index)
    elif args.command == "progress":
        print(json.dumps(progress(args.manifest), indent=2, sort_keys=True))
    elif args.command == "collect":
        collect(args.manifest)
    elif args.command == "audit-qacct":
        audit_qacct(args.manifest, args.job_id)
    elif args.command == "verify-local-readback":
        verify_local_readback(args.manifest)
    elif args.command == "diagnose-quadrature":
        print(
            json.dumps(
                diagnose_quadrature_orders(args.case_id, args.order_pair),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "diagnose-quadrature-suite":
        print(
            json.dumps(
                diagnose_quadrature_suite(args.order_pair),
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
