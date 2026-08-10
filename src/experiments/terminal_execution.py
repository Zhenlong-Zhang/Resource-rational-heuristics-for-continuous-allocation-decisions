from __future__ import annotations

"""Fail-closed execution and accounting for terminal validation evidence.

This module deliberately contains no numerical method.  It schedules immutable
descriptor assignments, delegates evaluation to the accepted terminal evidence API,
and binds provisional outputs to scheduler accounting before local read-back.
"""

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .terminal_evidence_rows import (
    TERMINAL_METHOD_ORDER,
    TerminalEvidenceBundle,
    TerminalEvidenceRow,
    TerminalSidecarReference,
    evaluate_terminal_evidence_descriptor,
    reconstruct_terminal_evidence_source,
    terminal_evidence_row_hash,
    validate_terminal_evidence_bundle_source,
)
from .terminal_canonical_provider import (
    accepted_canonical_base_provider,
    load_accepted_canonical_base_provider,
)
from .terminal_validation_suite import (
    AUTHORITATIVE_PROVIDER_KIND,
    CanonicalBaseProvider,
    TerminalValidationDescriptor,
    TerminalValidationSuite,
    build_terminal_base_suite,
    build_terminal_one_step_suite,
    build_terminal_reachable_core_suite,
    canonical_base_provider_failures,
    canonical_hash,
    load_terminal_validation_identities,
    validate_terminal_validation_suite,
)


EXECUTION_MANIFEST_SCHEMA = "terminal_validation_execution_manifest_v1"
TASK_ARTIFACT_SCHEMA = "terminal_validation_task_artifact_v1"
PROVISIONAL_SCHEMA = "terminal_validation_provisional_v1"
SCHEDULER_EVIDENCE_SCHEMA = "terminal_validation_scheduler_evidence_v1"
QACCT_AUDIT_SCHEMA = "terminal_validation_qacct_audit_v1"
POST_JOB_SCHEMA = "terminal_validation_post_job_candidate_v1"
READBACK_SCHEMA = "terminal_validation_independent_readback_v1"
COMPUTE_CEILING_SCHEMA = "hoffman2_compute_ceiling_report_v1"
SOURCE_IDENTITY_SCHEMA = "terminal_validation_source_identity_v1"
EXECUTION_AUTHORIZATION_SCHEMA = "terminal_validation_execution_authorization_v1"

SMOKE_CASE_IDS = (1, 20, 72, 85)
SUITE_ORDER = ("base", "one_step", "reachable_core")
TERMINAL_NUMERICAL_RESOURCE_CAPS = {
    "max_h_rt_seconds": 86_400,
    "max_memory_bytes": 8 * 1024**3,
    "max_array_tasks": 2_000,
    "max_throttle": 500,
    "max_storage_bytes": 200 * 1024**3,
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPUTE_HOST_RE = re.compile(r"^(?:n|g|compute)[a-z0-9.-]*$", re.IGNORECASE)
_MANIFEST_FIELDS = {
    "schema", "stage", "artifact_type", "case_owner_count", "case_owners",
    "task_count", "tasks", "expected_descriptor_count", "expected_row_count",
    "expected_sidecar_count", "suite_manifest_hashes",
    "suite_ordered_descriptor_hashes", "scientific_spec_hash",
    "numerical_method_config_hash", "provider_hash",
    "provider_source_identity_hash", "source_identity",
    "compute_ceiling_report_hash", "resources", "full_stage_strata_counts",
    "planned_full_task_count", "planned_full_task_strata",
    "expected_positive_reference_a_count", "expected_positive_reference_b_count",
    "expected_tie_path_row_count", "expected_symmetry_path_row_count",
    "expected_negative_control_count", "numerical_resource_caps",
    "max_descriptors_per_subshard", "one_slot_required", "array_allowed",
    "no_overwrite", "manifest_hash",
}
_TASK_FIELDS = {
    "task_id", "logical_case_owner", "subshard_index", "subshard_count",
    "descriptors", "assignment_hash",
}
_OWNER_FIELDS = {
    "logical_case_owner", "task_ids", "descriptor_count", "descriptor_hash",
}
_REF_FIELDS = {
    "suite_class", "descriptor_index", "descriptor_hash", "expected_methods",
    "expected_tie_row_count", "expected_symmetry_row_count",
}
_TASK_ARTIFACT_FIELDS = {
    "schema", "manifest_hash", "task_id", "assignment_hash",
    "logical_case_owner", "subshard_index", "subshard_count", "job_id",
    "sge_task_id", "slots", "hostname", "source_identity_hash",
    "provider_hash", "scientific_spec_hash", "numerical_method_config_hash",
    "rows_file_hash", "metrics_file_hash", "row_count",
    "sidecar_count", "sidecar_index", "task_cpu_seconds", "task_wall_seconds",
    "logical_record_hash",
}
_SIDECAR_INDEX_FIELDS = {
    "logical_path", "task_relative_path", "sha256", "byte_count",
}
_METRIC_FIELDS = {
    "task_id", "descriptor_hash", "method", "stratum", "evaluation_count", "cpu_seconds",
    "wall_seconds", "row_bytes", "sidecar_bytes", "timing_scope",
}
_SCHEDULER_FIELDS = {"schema", "manifest_hash", "submissions", "logical_record_hash"}
_SUBMISSION_FIELDS = {
    "job_id", "job_name", "queue", "array_job", "manifest_task_ids",
    "qsub_raw_path", "qsub_raw_sha256", "job_script_path", "job_script_sha256",
    "slots", "h_rt_seconds", "memory_bytes", "throttle", "task_id_mode",
    "parallel_environment", "job_script_semantics_hash",
}
_QACCT_FIELDS = {
    "schema", "manifest_hash", "scheduler_evidence_hash", "raw_qacct_hashes",
    "expected_task_count", "observed_task_count", "task_usage_seconds",
    "qacct_audit_pass", "logical_record_hash",
    "task_bindings",
}
_CEILING_FIELDS = {
    "schema", "captured_at_utc", "max_walltime_seconds", "max_array_tasks",
    "max_throttle", "max_memory_bytes", "max_storage_bytes", "cpu_hours_quota",
    "allowed_queues", "report_hash",
}
_PROVISIONAL_FIELDS = {
    "schema", "artifact_type", "artifact_status", "stage_complete", "manifest_hash",
    "source_hash_match", "scientific_spec_hash_match",
    "numerical_method_config_hash_match", "manifest_hash_match",
    "observed_task_count", "observed_row_count", "observed_sidecar_count",
    "positive_reference_a_count", "positive_reference_b_count",
    "reference_a_complete", "reference_b_complete", "tie_path_exercised",
    "tie_path_pass", "symmetry_path_exercised", "symmetry_path_pass",
    "scalar_batch_parity_pass", "fail_closed_path_exercised",
    "negative_control_rejection_pass", "unexpected_reference_unresolved_count",
    "unexpected_validation_failure_count",
    "missing_duplicate_malformed_nonfinite_stale_invalid_count", "coverage_match",
    "failure_reasons", "task_artifact_hashes", "job_hosts", "metrics",
    "qacct_audit_pass", "finalization_hash_bind_pass", "independent_readback_pass",
    "feasibility_gate_pass", "logical_record_hash", "provisional_gate_pass",
}
_POST_JOB_FIELDS = {
    "schema", "artifact_type", "artifact_status", "stage_complete", "manifest_hash",
    "bound_file_hashes", "qacct_audit_pass", "finalization_hash_bind_pass",
    "independent_readback_pass", "feasibility", "feasibility_gate_pass",
    "finalization_overhead_seconds", "final_gate_pass",
    "logical_record_hash",
}
_READBACK_FIELDS = {
    "schema", "artifact_type", "artifact_status", "stage_complete", "manifest_hash",
    "source_hash_match", "scientific_spec_hash_match",
    "numerical_method_config_hash_match", "manifest_hash_match",
    "observed_task_count", "observed_row_count", "observed_sidecar_count",
    "positive_reference_a_count", "positive_reference_b_count", "coverage_match",
    "reference_a_complete", "reference_b_complete", "tie_path_exercised",
    "tie_path_pass", "symmetry_path_exercised", "symmetry_path_pass",
    "scalar_batch_parity_pass", "fail_closed_path_exercised",
    "negative_control_rejection_pass", "unexpected_reference_unresolved_count",
    "unexpected_validation_failure_count",
    "missing_duplicate_malformed_nonfinite_stale_invalid_count", "provisional_gate_pass",
    "qacct_audit_pass",
    "finalization_hash_bind_pass", "independent_readback_pass", "feasibility",
    "feasibility_gate_pass", "final_gate_pass", "readback_host", "post_job_hash",
    "logical_record_hash",
}
_AUTHORIZATION_FIELDS = {
    "schema", "authorization_status", "verdict", "manifest_hash",
    "source_identity_hash", "source_commit", "source_tree", "provider_hash",
    "provider_source_identity_hash", "scientific_spec_hash",
    "numerical_method_config_hash", "compute_ceiling_report_hash",
    "resources_hash", "execution_script_hashes",
}


@dataclass(frozen=True)
class DescriptorRef:
    suite_class: str
    descriptor_index: int
    descriptor_hash: str
    expected_methods: Tuple[str, ...]
    expected_tie_row_count: int
    expected_symmetry_row_count: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    logical_case_owner: int
    subshard_index: int
    subshard_count: int
    descriptors: Tuple[DescriptorRef, ...]
    assignment_hash: str


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in canonical evidence")
        return {"float_hex": value.hex()}
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical evidence keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported canonical evidence type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"float_hex"}:
            token = value["float_hex"]
            if type(token) is not str:
                raise ValueError("float_hex must be a string")
            result = float.fromhex(token)
            if not math.isfinite(result) or result.hex() != token:
                raise ValueError("float_hex is not canonical finite binary64")
            return result
        return {key: _decode(item) for key, item in value.items()}
    raise ValueError("unsupported canonical JSON value")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def logical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value: Any) -> bool:
    return type(value) is str and _HASH_RE.fullmatch(value) is not None


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create a JSON file and refuse every overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_canonical(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link promotion is atomic and fails when the final path exists.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _without_hash(value: Mapping[str, Any], field: str) -> Dict[str, Any]:
    result = dict(value)
    result[field] = ""
    return result


def _validate_self_hash(value: Mapping[str, Any], field: str, context: str) -> None:
    claimed = value.get(field)
    if not _is_hash(claimed) or logical_hash(_without_hash(value, field)) != claimed:
        raise ValueError(f"{context} self-hash mismatch")


def capture_clean_source_identity(project_root: Path, source_paths: Sequence[str]) -> Dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("terminal execution requires a clean committed worktree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=project_root, text=True).strip()
    records = []
    for relative in sorted(set(source_paths)):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
        path = project_root / relative
        records.append((relative, sha256_file(path)))
    payload: Dict[str, Any] = {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "commit": commit,
        "tree": tree,
        "source_hashes": tuple(records),
        "source_hashes_hash": logical_hash(tuple(records)),
        "identity_hash": "",
    }
    payload["identity_hash"] = logical_hash(_without_hash(payload, "identity_hash"))
    return payload


def validate_clean_source_identity(project_root: Path, identity: Mapping[str, Any]) -> None:
    if identity.get("schema") != SOURCE_IDENTITY_SCHEMA:
        raise RuntimeError("source identity schema mismatch")
    _validate_self_hash(identity, "identity_hash", "source identity")
    current = capture_clean_source_identity(
        project_root, tuple(str(item[0]) for item in identity.get("source_hashes", ()))
    )
    if current != identity:
        raise RuntimeError("clean source identity differs from frozen manifest")


def execution_script_hashes(
    project_root: Path, source_identity: Mapping[str, Any]
) -> Tuple[Tuple[str, str], ...]:
    paths = (
        "scripts/submit_hoffman2_terminal_validation.sh",
        "scripts/terminal_validation_array.py",
        "src/experiments/terminal_execution.py",
    )
    frozen = dict(source_identity.get("source_hashes", ()))
    result = []
    for relative in paths:
        path = project_root / relative
        observed = sha256_file(path)
        if frozen.get(relative) != observed:
            raise RuntimeError(f"execution script is absent from clean source identity: {relative}")
        result.append((relative, observed))
    return tuple(result)


def validate_execution_authorization(
    *,
    authorization_path: Path,
    approved_file_hash: str,
    manifest: Mapping[str, Any],
    project_root: Path,
) -> Mapping[str, Any]:
    """Validate a reviewer approval bound to one exact stage and source tree."""

    if not _is_hash(approved_file_hash) or sha256_file(authorization_path) != approved_file_hash:
        raise RuntimeError("execution authorization file hash is not externally approved")
    authorization = _decode(dict(_load_json(authorization_path)))
    if authorization.get("schema") != EXECUTION_AUTHORIZATION_SCHEMA:
        raise RuntimeError("execution authorization schema mismatch")
    if set(authorization) != _AUTHORIZATION_FIELDS:
        raise RuntimeError("execution authorization fields differ from the exact schema")
    expected_verdict = (
        "ACCEPT TERMINAL IMPLEMENTATION FOR SCHEDULED SMOKE"
        if manifest["stage"] == "smoke"
        else "ACCEPT TERMINAL SMOKE / AUTHORIZE TERMINAL VALIDATION ARRAY"
    )
    source = manifest["source_identity"]
    validate_clean_source_identity(project_root, source)
    expected = {
        "schema": EXECUTION_AUTHORIZATION_SCHEMA,
        "authorization_status": "reviewer_approved_for_exact_terminal_stage",
        "verdict": expected_verdict,
        "manifest_hash": manifest["manifest_hash"],
        "source_identity_hash": source["identity_hash"],
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "provider_hash": manifest["provider_hash"],
        "provider_source_identity_hash": manifest["provider_source_identity_hash"],
        "scientific_spec_hash": manifest["scientific_spec_hash"],
        "numerical_method_config_hash": manifest["numerical_method_config_hash"],
        "compute_ceiling_report_hash": manifest["compute_ceiling_report_hash"],
        "resources_hash": logical_hash(manifest["resources"]),
        "execution_script_hashes": execution_script_hashes(project_root, source),
    }
    if authorization != expected:
        raise RuntimeError("execution authorization does not bind the exact manifest/source")
    return authorization


def build_terminal_suites(
    provider: CanonicalBaseProvider,
    acceptance_validator: Optional[Callable[[CanonicalBaseProvider], bool]] = None,
) -> Dict[str, TerminalValidationSuite]:
    if (
        acceptance_validator is not None
        and acceptance_validator is not accepted_canonical_base_provider
    ):
        raise RuntimeError("custom authoritative provider acceptance is disallowed")
    if not accepted_canonical_base_provider(provider):
        raise RuntimeError("terminal suites require the accepted canonical provider")
    identities = load_terminal_validation_identities()
    suites = {
        "base": build_terminal_base_suite(identities, provider),
        "one_step": build_terminal_one_step_suite(identities, provider),
        "reachable_core": build_terminal_reachable_core_suite(identities),
    }
    for suite in suites.values():
        validation = validate_terminal_validation_suite(
            suite,
            identities,
            base_provider=provider,
            require_authoritative=True,
            authoritative_acceptance_validator=accepted_canonical_base_provider,
        )
        if validation.failures or not validation.authoritative_source_accepted:
            raise RuntimeError("terminal suite lacks authoritative source: " + ",".join(validation.failures))
    return suites


def _require_formal_canonical_acceptance(
    provider: CanonicalBaseProvider,
    acceptance_validator: Optional[Callable[[CanonicalBaseProvider], bool]],
) -> None:
    if (
        acceptance_validator is not None
        and acceptance_validator is not accepted_canonical_base_provider
    ):
        raise RuntimeError("custom authoritative provider acceptance is disallowed")
    if not accepted_canonical_base_provider(provider):
        raise RuntimeError("canonical provider acceptance rejected")


def _descriptor_key(descriptor: TerminalValidationDescriptor) -> str:
    return f"{descriptor.suite_class}:{descriptor.descriptor_index}"


def _owner(descriptor: TerminalValidationDescriptor) -> int:
    if descriptor.source_case_id is not None:
        return int(descriptor.source_case_id)
    return int(descriptor.descriptor_index % 90)


def smoke_descriptors(suites: Mapping[str, TerminalValidationSuite]) -> Tuple[TerminalValidationDescriptor, ...]:
    selected = []
    case_set = set(SMOKE_CASE_IDS)
    selected.extend(
        descriptor for descriptor in suites["base"].descriptors
        if descriptor.source_case_id in case_set
    )
    selected.extend(
        descriptor for descriptor in suites["one_step"].descriptors
        if descriptor.source_case_id in case_set
        and descriptor.component_index == 0
        and descriptor.z_offset == 0
    )
    profile_order = (
        "initial_symmetric",
        "concentrated_depth_6_-1",
        "concentrated_depth_6_+1",
        "balanced_late_feasible",
    )
    for profile in profile_order:
        selected.append(next(item for item in suites["reachable_core"].descriptors if item.profile == profile))
    if len(selected) != 16:
        raise RuntimeError("frozen smoke selection must contain exactly 16 descriptors")
    return tuple(selected)


def _expected_descriptor_plan(
    descriptor: TerminalValidationDescriptor,
    provider: CanonicalBaseProvider,
) -> Tuple[Tuple[str, ...], int, int]:
    mdp, belief = reconstruct_terminal_evidence_source(descriptor, provider)
    bundle = evaluate_terminal_evidence_descriptor(descriptor, mdp, belief)
    methods = tuple(row.method for row in bundle.rows)
    if methods != tuple(method for method in TERMINAL_METHOD_ORDER if method in methods):
        raise RuntimeError("terminal evidence methods are not in frozen order")
    tie_count = sum(row.tie_status not in (None, "unique") for row in bundle.rows)
    symmetry_count = sum(row.symmetry_required for row in bundle.rows)
    return methods, tie_count, symmetry_count


def _task_hash(task: Mapping[str, Any]) -> str:
    return logical_hash(_without_hash(task, "assignment_hash"))


def _planned_full_workload(
    descriptors: Sequence[TerminalValidationDescriptor],
    *,
    max_descriptors_per_subshard: int,
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Freeze conservative full-array tasks with worst-case Reference-B escalation."""

    grouped = {owner: [] for owner in range(90)}
    for descriptor in descriptors:
        grouped[_owner(descriptor)].append(descriptor)
    tasks = []
    for owner in range(90):
        ordered = sorted(
            grouped[owner],
            key=lambda item: (SUITE_ORDER.index(item.suite_class), item.descriptor_index),
        )
        if not ordered:
            raise RuntimeError(f"planned full logical owner {owner} has no descriptors")
        for offset in range(0, len(ordered), max_descriptors_per_subshard):
            counts: Dict[str, int] = {}
            for descriptor in ordered[offset:offset + max_descriptors_per_subshard]:
                for method in TERMINAL_METHOD_ORDER:
                    key = _stratum(method, descriptor)
                    counts[key] = counts.get(key, 0) + 1
            tasks.append(tuple(sorted(counts.items())))
    return tuple(tasks)


def _stratum(method: str, descriptor: TerminalValidationDescriptor) -> str:
    if method in ("reference_b", "agreement"):
        path = "escalation"
    elif descriptor.orientation == "symmetric":
        path = "tie"
    else:
        path = "ordinary"
    return f"{method}:{path}"


def create_execution_manifest(
    *,
    stage: str,
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Optional[Callable[[CanonicalBaseProvider], bool]] = None,
    source_identity: Mapping[str, Any],
    max_descriptors_per_subshard: int,
    resources: Mapping[str, Any],
    compute_ceiling_report_hash: str,
    _validate_result: bool = True,
) -> Dict[str, Any]:
    if stage not in ("smoke", "full"):
        raise ValueError("stage must be smoke or full")
    if max_descriptors_per_subshard < 1:
        raise ValueError("subshard bound must be positive")
    if not _is_hash(compute_ceiling_report_hash):
        raise ValueError("compute ceiling report hash is malformed")
    if provider.provider_kind != AUTHORITATIVE_PROVIDER_KIND or provider.diagnostic_only:
        raise RuntimeError("execution manifests require an authoritative canonical provider")
    _require_formal_canonical_acceptance(provider, acceptance_validator)
    provider_failures = canonical_base_provider_failures(provider)
    if provider_failures:
        raise RuntimeError("canonical provider failed source validation: " + ",".join(provider_failures))
    for suite in suites.values():
        validation = validate_terminal_validation_suite(
            suite,
            base_provider=provider,
            require_authoritative=True,
            authoritative_acceptance_validator=accepted_canonical_base_provider,
        )
        if validation.failures:
            raise RuntimeError("suite validation failed before manifest freeze")

    all_descriptors = tuple(
        descriptor for suite_class in SUITE_ORDER
        for descriptor in suites[suite_class].descriptors
    )
    selected = smoke_descriptors(suites) if stage == "smoke" else all_descriptors
    planned_full_tasks = _planned_full_workload(
        all_descriptors,
        max_descriptors_per_subshard=max_descriptors_per_subshard,
    )
    refs: Dict[str, DescriptorRef] = {}
    for descriptor in selected:
        key = _descriptor_key(descriptor)
        methods, tie_count, symmetry_count = _expected_descriptor_plan(
            descriptor, provider
        )
        if not methods or any(method not in TERMINAL_METHOD_ORDER for method in methods):
            raise RuntimeError(f"invalid frozen method plan for {key}")
        refs[key] = DescriptorRef(
            descriptor.suite_class,
            descriptor.descriptor_index,
            descriptor.descriptor_hash,
            methods,
            tie_count,
            symmetry_count,
        )

    owners = SMOKE_CASE_IDS if stage == "smoke" else tuple(range(90))
    grouped: Dict[int, list[DescriptorRef]] = {owner: [] for owner in owners}
    reachable_smoke_owners = dict(zip(
        (item.descriptor_index for item in selected if item.suite_class == "reachable_core"),
        SMOKE_CASE_IDS,
    ))
    descriptor_lookup = {_descriptor_key(item): item for item in selected}
    for key, ref in refs.items():
        descriptor = descriptor_lookup[key]
        owner = (
            reachable_smoke_owners[descriptor.descriptor_index]
            if stage == "smoke" and descriptor.suite_class == "reachable_core"
            else _owner(descriptor)
        )
        if owner not in grouped:
            raise RuntimeError("descriptor was assigned outside frozen smoke owners")
        grouped[owner].append(ref)

    tasks = []
    owner_records = []
    next_task_id = 1
    for owner in owners:
        ordered = sorted(
            grouped[owner],
            key=lambda item: (SUITE_ORDER.index(item.suite_class), item.descriptor_index),
        )
        if not ordered:
            raise RuntimeError(f"logical owner {owner} has no descriptors")
        chunks = [
            tuple(ordered[index:index + max_descriptors_per_subshard])
            for index in range(0, len(ordered), max_descriptors_per_subshard)
        ]
        owner_task_ids = []
        for subshard_index, chunk in enumerate(chunks):
            task: Dict[str, Any] = {
                "task_id": next_task_id,
                "logical_case_owner": owner,
                "subshard_index": subshard_index,
                "subshard_count": len(chunks),
                "descriptors": tuple(asdict(item) for item in chunk),
                "assignment_hash": "",
            }
            task["assignment_hash"] = _task_hash(task)
            tasks.append(task)
            owner_task_ids.append(next_task_id)
            next_task_id += 1
        owner_records.append({
            "logical_case_owner": owner,
            "task_ids": tuple(owner_task_ids),
            "descriptor_count": len(ordered),
            "descriptor_hash": logical_hash(tuple(item.descriptor_hash for item in ordered)),
        })

    strata: Dict[str, int] = {}
    for task_strata in planned_full_tasks:
        for key, count in task_strata:
            strata[key] = strata.get(key, 0) + count
    payload: Dict[str, Any] = {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "stage": stage,
        "artifact_type": "terminal_smoke" if stage == "smoke" else "terminal_validation",
        "case_owner_count": len(owners),
        "case_owners": tuple(owner_records),
        "task_count": len(tasks),
        "tasks": tuple(tasks),
        "expected_descriptor_count": len(selected),
        "expected_row_count": sum(len(ref.expected_methods) for ref in refs.values()),
        "expected_sidecar_count": sum(len(ref.expected_methods) for ref in refs.values()),
        "expected_positive_reference_a_count": sum(
            "reference_a" in ref.expected_methods for ref in refs.values()
        ),
        "expected_positive_reference_b_count": sum(
            "reference_b" in ref.expected_methods for ref in refs.values()
        ),
        "expected_tie_path_row_count": sum(
            ref.expected_tie_row_count for ref in refs.values()
        ),
        "expected_symmetry_path_row_count": sum(
            ref.expected_symmetry_row_count for ref in refs.values()
        ),
        "expected_negative_control_count": 1,
        "suite_manifest_hashes": tuple(
            (suite_class, suites[suite_class].manifest.manifest_hash)
            for suite_class in SUITE_ORDER
        ),
        "suite_ordered_descriptor_hashes": tuple(
            (suite_class, suites[suite_class].manifest.ordered_descriptor_hash)
            for suite_class in SUITE_ORDER
        ),
        "scientific_spec_hash": load_terminal_validation_identities().scientific_spec_hash,
        "numerical_method_config_hash": load_terminal_validation_identities().numerical_method_config_hash,
        "provider_hash": provider.provider_hash,
        "provider_source_identity_hash": provider.source_identity_hash,
        "source_identity": dict(source_identity),
        "compute_ceiling_report_hash": compute_ceiling_report_hash,
        "resources": dict(resources),
        "numerical_resource_caps": dict(TERMINAL_NUMERICAL_RESOURCE_CAPS),
        "full_stage_strata_counts": tuple(sorted(strata.items())),
        "planned_full_task_count": len(planned_full_tasks),
        "planned_full_task_strata": planned_full_tasks,
        "max_descriptors_per_subshard": max_descriptors_per_subshard,
        "one_slot_required": True,
        "array_allowed": stage == "full",
        "no_overwrite": True,
        "manifest_hash": "",
    }
    payload["manifest_hash"] = logical_hash(_without_hash(payload, "manifest_hash"))
    if _validate_result:
        validate_execution_manifest(
            payload, suites, provider, accepted_canonical_base_provider
        )
    return payload


def validate_execution_manifest(
    manifest: Mapping[str, Any],
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Optional[Callable[[CanonicalBaseProvider], bool]] = None,
) -> None:
    if manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA:
        raise RuntimeError("execution manifest schema mismatch")
    if set(manifest) != _MANIFEST_FIELDS:
        raise RuntimeError("execution manifest fields differ from the exact schema")
    _validate_self_hash(manifest, "manifest_hash", "execution manifest")
    source_identity = manifest.get("source_identity", {})
    if not isinstance(source_identity, Mapping) or source_identity.get("schema") != SOURCE_IDENTITY_SCHEMA:
        raise RuntimeError("manifest source identity schema mismatch")
    _validate_self_hash(source_identity, "identity_hash", "manifest source identity")
    if not all((
        manifest.get("one_slot_required") is True,
        manifest.get("no_overwrite") is True,
        type(manifest.get("array_allowed")) is bool
        and manifest.get("array_allowed") == (manifest.get("stage") == "full"),
        _is_hash(manifest.get("compute_ceiling_report_hash")),
    )):
        raise RuntimeError("manifest execution invariants mismatch")
    resources = manifest.get("resources", {})
    if set(resources) != {"queue", "h_rt_seconds", "memory_bytes", "throttle"}:
        raise RuntimeError("manifest resource fields differ from the exact schema")
    if (
        type(resources["queue"]) is not str or not resources["queue"]
        or any(type(resources[key]) is not int or resources[key] < 1
               for key in ("h_rt_seconds", "memory_bytes", "throttle"))
    ):
        raise RuntimeError("manifest resources are malformed")
    if manifest.get("numerical_resource_caps") != TERMINAL_NUMERICAL_RESOURCE_CAPS:
        raise RuntimeError("manifest numerical resource caps changed")
    caps = manifest["numerical_resource_caps"]
    if (
        resources["h_rt_seconds"] > caps["max_h_rt_seconds"]
        or resources["memory_bytes"] > caps["max_memory_bytes"]
        or resources["throttle"] > caps["max_throttle"]
    ):
        raise RuntimeError("manifest request exceeds frozen numerical resource caps")
    stage = manifest.get("stage")
    expected_owners = SMOKE_CASE_IDS if stage == "smoke" else tuple(range(90)) if stage == "full" else ()
    owners = tuple(int(item["logical_case_owner"]) for item in manifest.get("case_owners", ()))
    if any(set(item) != _OWNER_FIELDS for item in manifest.get("case_owners", ())):
        raise RuntimeError("case-owner fields differ from the exact schema")
    if owners != expected_owners or int(manifest.get("case_owner_count", -1)) != len(expected_owners):
        raise RuntimeError("logical case-owner coverage mismatch")
    _require_formal_canonical_acceptance(provider, acceptance_validator)
    if manifest.get("provider_hash") != provider.provider_hash:
        raise RuntimeError("manifest canonical provider is not accepted")
    provider_failures = canonical_base_provider_failures(provider)
    if provider_failures:
        raise RuntimeError("manifest canonical provider failed validation: " + ",".join(provider_failures))
    if manifest.get("provider_source_identity_hash") != provider.source_identity_hash:
        raise RuntimeError("manifest provider source identity mismatch")
    identities = load_terminal_validation_identities()
    if (
        manifest.get("scientific_spec_hash") != identities.scientific_spec_hash
        or manifest.get("numerical_method_config_hash") != identities.numerical_method_config_hash
    ):
        raise RuntimeError("manifest scientific/numerical identity mismatch")
    expected_suite_hashes = tuple(
        (suite_class, suites[suite_class].manifest.manifest_hash) for suite_class in SUITE_ORDER
    )
    if tuple(tuple(item) for item in manifest.get("suite_manifest_hashes", ())) != expected_suite_hashes:
        raise RuntimeError("manifest suite identity mismatch")
    expected_ordered_hashes = tuple(
        (suite_class, suites[suite_class].manifest.ordered_descriptor_hash)
        for suite_class in SUITE_ORDER
    )
    if tuple(tuple(item) for item in manifest.get("suite_ordered_descriptor_hashes", ())) != expected_ordered_hashes:
        raise RuntimeError("manifest ordered descriptor identity mismatch")
    descriptor_lookup = {
        (suite_class, descriptor.descriptor_index): descriptor
        for suite_class, suite in suites.items() for descriptor in suite.descriptors
    }
    task_ids = []
    seen_refs = set()
    owner_task_ids: Dict[int, list[int]] = {owner: [] for owner in expected_owners}
    owner_descriptor_hashes: Dict[int, list[str]] = {owner: [] for owner in expected_owners}
    owner_subshards: Dict[int, list[Tuple[int, int]]] = {owner: [] for owner in expected_owners}
    row_count = 0
    for raw_task in manifest.get("tasks", ()):
        task = dict(raw_task)
        if set(task) != _TASK_FIELDS:
            raise RuntimeError("task fields differ from the exact schema")
        if _task_hash(task) != task.get("assignment_hash"):
            raise RuntimeError("task assignment hash mismatch")
        task_id = int(task["task_id"])
        owner = int(task["logical_case_owner"])
        task_ids.append(task_id)
        if owner not in owner_task_ids:
            raise RuntimeError("task has an unknown logical owner")
        owner_task_ids[owner].append(task_id)
        owner_subshards[owner].append((int(task["subshard_index"]), int(task["subshard_count"])))
        refs = tuple(task.get("descriptors", ()))
        if not refs or len(refs) > int(manifest["max_descriptors_per_subshard"]):
            raise RuntimeError("task descriptor count violates subshard bound")
        for raw_ref in refs:
            ref = dict(raw_ref)
            if set(ref) != _REF_FIELDS:
                raise RuntimeError("descriptor reference fields differ from the exact schema")
            key = (str(ref["suite_class"]), int(ref["descriptor_index"]))
            descriptor = descriptor_lookup.get(key)
            if descriptor is None or descriptor.descriptor_hash != ref.get("descriptor_hash"):
                raise RuntimeError("task descriptor identity mismatch")
            if key in seen_refs:
                raise RuntimeError("descriptor is assigned to multiple tasks")
            seen_refs.add(key)
            owner_descriptor_hashes[owner].append(str(ref["descriptor_hash"]))
            methods = tuple(ref.get("expected_methods", ()))
            if (
                not methods
                or len(set(methods)) != len(methods)
                or methods != tuple(method for method in TERMINAL_METHOD_ORDER if method in methods)
            ):
                raise RuntimeError("task method plan is invalid")
            for count_field in ("expected_tie_row_count", "expected_symmetry_row_count"):
                if type(ref[count_field]) is not int or not 0 <= ref[count_field] <= len(methods):
                    raise RuntimeError("descriptor path-count plan is invalid")
            row_count += len(methods)
    if task_ids != list(range(1, len(task_ids) + 1)) or len(set(task_ids)) != len(task_ids):
        raise RuntimeError("task IDs must be unique contiguous one-based integers")
    if int(manifest.get("task_count", -1)) != len(task_ids):
        raise RuntimeError("task count mismatch")
    if stage == "full" and len(task_ids) > int(caps["max_array_tasks"]):
        raise RuntimeError("full manifest task count exceeds frozen numerical cap")
    if int(manifest.get("expected_descriptor_count", -1)) != len(seen_refs):
        raise RuntimeError("descriptor count mismatch")
    if int(manifest.get("expected_row_count", -1)) != row_count:
        raise RuntimeError("frozen row count mismatch")
    if int(manifest.get("expected_sidecar_count", -1)) != row_count:
        raise RuntimeError("frozen sidecar count mismatch")
    planned_tasks = tuple(manifest.get("planned_full_task_strata", ()))
    if int(manifest.get("planned_full_task_count", -1)) != len(planned_tasks):
        raise RuntimeError("planned full task count mismatch")
    recomputed_strata: Dict[str, int] = {}
    for task_strata in planned_tasks:
        if not task_strata:
            raise RuntimeError("planned full task has no strata")
        for key, count in task_strata:
            if type(key) is not str or type(count) is not int or count < 1:
                raise RuntimeError("planned full task stratum is malformed")
            recomputed_strata[key] = recomputed_strata.get(key, 0) + count
    if tuple(sorted(recomputed_strata.items())) != tuple(tuple(item) for item in manifest["full_stage_strata_counts"]):
        raise RuntimeError("planned full task strata do not match aggregate counts")
    for raw_owner in manifest.get("case_owners", ()):
        owner = int(raw_owner["logical_case_owner"])
        if tuple(raw_owner["task_ids"]) != tuple(owner_task_ids[owner]):
            raise RuntimeError("case-owner task index mismatch")
        expected_subshards = tuple((index, len(owner_subshards[owner])) for index in range(len(owner_subshards[owner])))
        if tuple(owner_subshards[owner]) != expected_subshards:
            raise RuntimeError("case-owner subshard indices are not exact and contiguous")
        if int(raw_owner["descriptor_count"]) != len(owner_descriptor_hashes[owner]):
            raise RuntimeError("case-owner descriptor count mismatch")
        if raw_owner["descriptor_hash"] != logical_hash(tuple(owner_descriptor_hashes[owner])):
            raise RuntimeError("case-owner descriptor hash mismatch")
    expected = create_execution_manifest(
        stage=str(manifest["stage"]),
        suites=suites,
        provider=provider,
        acceptance_validator=acceptance_validator,
        source_identity=source_identity,
        max_descriptors_per_subshard=int(manifest["max_descriptors_per_subshard"]),
        resources=resources,
        compute_ceiling_report_hash=str(manifest["compute_ceiling_report_hash"]),
        _validate_result=False,
    )
    if canonical_bytes(manifest) != canonical_bytes(expected):
        raise RuntimeError("manifest differs from source-reconstructed frozen workload")


def _row_to_payload(row: TerminalEvidenceRow) -> Mapping[str, Any]:
    return _canonical(row)


def _row_from_payload(payload: Mapping[str, Any]) -> TerminalEvidenceRow:
    decoded = _decode(dict(payload))
    sidecar = TerminalSidecarReference(**decoded["sidecar"])
    decoded["sidecar"] = sidecar
    row = TerminalEvidenceRow(**decoded)
    if terminal_evidence_row_hash(row) != row.logical_record_hash:
        raise ValueError("task row logical hash mismatch")
    return row


def _task_by_id(manifest: Mapping[str, Any], task_id: int) -> Mapping[str, Any]:
    matches = [item for item in manifest["tasks"] if int(item["task_id"]) == task_id]
    if len(matches) != 1:
        raise RuntimeError("task ID is absent or duplicated in manifest")
    return matches[0]


def execute_task(
    *,
    manifest: Mapping[str, Any],
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Callable[[CanonicalBaseProvider], bool],
    output_root: Path,
    task_id: int,
    scheduler_environment: Optional[Mapping[str, str]] = None,
) -> Path:
    validate_execution_manifest(manifest, suites, provider, acceptance_validator)
    environment = dict(scheduler_environment or os.environ)
    if environment.get("NSLOTS") != "1":
        raise RuntimeError("terminal validation tasks require exactly one scheduler slot")
    if not environment.get("JOB_ID", "").isdigit():
        raise RuntimeError("terminal validation task lacks a scheduler job ID")
    if manifest["stage"] == "full" and int(environment.get("SGE_TASK_ID", "-1")) != task_id:
        raise RuntimeError("array task ID differs from immutable manifest task")
    if environment.get("PE_HOSTFILE"):
        raise RuntimeError("shared-memory parallel environments are forbidden")
    task = _task_by_id(manifest, task_id)
    target = output_root / "tasks" / f"task_{task_id:05d}"
    target.parent.mkdir(parents=True, exist_ok=True)
    reservation = target.parent / f".{target.name}.exclusive"
    try:
        reservation_descriptor = os.open(
            reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise FileExistsError(f"task output is already reserved: {target}") from error
    with os.fdopen(reservation_descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{manifest['manifest_hash']}:{task_id}\n")
        handle.flush()
        os.fsync(handle.fileno())
    if target.exists():
        reservation.unlink(missing_ok=True)
        raise FileExistsError(f"task output already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".task_{task_id:05d}.", dir=target.parent))
    rows = []
    sidecar_index = []
    metrics = []
    descriptor_lookup = {
        (suite_class, descriptor.descriptor_index): descriptor
        for suite_class, suite in suites.items() for descriptor in suite.descriptors
    }
    process_start = time.process_time()
    wall_start = time.perf_counter()
    try:
        for raw_ref in task["descriptors"]:
            descriptor = descriptor_lookup[(raw_ref["suite_class"], int(raw_ref["descriptor_index"]))]
            mdp, belief = reconstruct_terminal_evidence_source(descriptor, provider)
            bundle = evaluate_terminal_evidence_descriptor(descriptor, mdp, belief)
            methods = tuple(row.method for row in bundle.rows)
            if methods != tuple(raw_ref["expected_methods"]):
                raise RuntimeError("source method plan differs from frozen manifest")
            failures = validate_terminal_evidence_bundle_source(bundle, descriptor, mdp, belief)
            if failures:
                raise RuntimeError("source evidence validation failed: " + ",".join(failures))
            sidecars = dict(bundle.sidecars)
            for row in bundle.rows:
                rows.append(_row_to_payload(row))
                relative = Path("sidecars") / row.sidecar.relative_path
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise RuntimeError("duplicate sidecar path in task")
                path.write_bytes(sidecars[row.sidecar.relative_path])
                sidecar_index.append({
                    "logical_path": row.sidecar.relative_path,
                    "task_relative_path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "byte_count": path.stat().st_size,
                })
                metrics.append({
                    "task_id": task_id,
                    "descriptor_hash": descriptor.descriptor_hash,
                    "method": row.method,
                    "stratum": _stratum(row.method, descriptor),
                    "evaluation_count": row.evaluation_count,
                    "cpu_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "row_bytes": len(canonical_bytes(row)),
                    "sidecar_bytes": row.sidecar.byte_count,
                    "timing_scope": "task_conservative_shared",
                })
        # Every row conservatively receives the complete measured task evaluation time.
        # qacct is bound later and remains the authoritative feasibility timing source.
        evidence_cpu = time.process_time() - process_start
        evidence_wall = time.perf_counter() - wall_start
        if evidence_cpu <= 0.0 or evidence_wall <= 0.0:
            raise RuntimeError("task timers did not advance")
        for metric in metrics:
            metric["cpu_seconds"] = evidence_cpu
            metric["wall_seconds"] = evidence_wall
            metric["timing_scope"] = "task_conservative_shared"
        rows_path = temporary / "rows.json"
        rows_path.write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        metrics_path = temporary / "metrics.json"
        metrics_path.write_text(json.dumps(_canonical(metrics), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        artifact: Dict[str, Any] = {
            "schema": TASK_ARTIFACT_SCHEMA,
            "manifest_hash": manifest["manifest_hash"],
            "task_id": task_id,
            "assignment_hash": task["assignment_hash"],
            "logical_case_owner": task["logical_case_owner"],
            "subshard_index": task["subshard_index"],
            "subshard_count": task["subshard_count"],
            "job_id": environment["JOB_ID"],
            "sge_task_id": environment.get("SGE_TASK_ID"),
            "slots": 1,
            "hostname": platform.node().split(".", 1)[0].lower(),
            "source_identity_hash": manifest["source_identity"]["identity_hash"],
            "provider_hash": provider.provider_hash,
            "scientific_spec_hash": manifest["scientific_spec_hash"],
            "numerical_method_config_hash": manifest["numerical_method_config_hash"],
            "rows_file_hash": sha256_file(rows_path),
            "metrics_file_hash": sha256_file(metrics_path),
            "row_count": len(rows),
            "sidecar_count": len(sidecar_index),
            "sidecar_index": tuple(sidecar_index),
            "task_cpu_seconds": evidence_cpu,
            "task_wall_seconds": evidence_wall,
            "logical_record_hash": "",
        }
        artifact["logical_record_hash"] = logical_hash(_without_hash(artifact, "logical_record_hash"))
        write_new_json(temporary / "task.json", artifact)
        if target.exists():
            raise FileExistsError(f"task output appeared during execution: {target}")
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        reservation.unlink(missing_ok=True)
    return target


def _load_task_artifact(path: Path) -> Mapping[str, Any]:
    artifact = _decode(dict(_load_json(path / "task.json")))
    if artifact.get("schema") != TASK_ARTIFACT_SCHEMA:
        raise ValueError("task artifact schema mismatch")
    if set(artifact) != _TASK_ARTIFACT_FIELDS:
        raise ValueError("task artifact fields differ from the exact schema")
    if any(set(item) != _SIDECAR_INDEX_FIELDS for item in artifact.get("sidecar_index", ())):
        raise ValueError("task sidecar-index fields differ from the exact schema")
    _validate_self_hash(artifact, "logical_record_hash", "task artifact")
    if sha256_file(path / "rows.json") != artifact["rows_file_hash"]:
        raise ValueError("task row-file hash mismatch")
    if sha256_file(path / "metrics.json") != artifact["metrics_file_hash"]:
        raise ValueError("task metrics-file hash mismatch")
    return artifact


def recompute_provisional(
    *,
    manifest: Mapping[str, Any],
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Callable[[CanonicalBaseProvider], bool],
    output_root: Path,
) -> Mapping[str, Any]:
    validate_execution_manifest(manifest, suites, provider, acceptance_validator)
    descriptor_lookup = {
        (suite_class, descriptor.descriptor_index): descriptor
        for suite_class, suite in suites.items() for descriptor in suite.descriptors
    }
    all_rows = []
    all_metrics = []
    all_sidecars: Dict[str, bytes] = {}
    job_hosts = set()
    failures = []
    observed_tasks = []
    for raw_task in manifest["tasks"]:
        task_id = int(raw_task["task_id"])
        task_dir = output_root / "tasks" / f"task_{task_id:05d}"
        try:
            artifact = _load_task_artifact(task_dir)
            if (
                artifact["manifest_hash"] != manifest["manifest_hash"]
                or artifact["assignment_hash"] != raw_task["assignment_hash"]
                or int(artifact["task_id"]) != task_id
                or int(artifact["logical_case_owner"]) != int(raw_task["logical_case_owner"])
                or int(artifact["subshard_index"]) != int(raw_task["subshard_index"])
                or int(artifact["subshard_count"]) != int(raw_task["subshard_count"])
                or int(artifact["slots"]) != 1
                or artifact["source_identity_hash"] != manifest["source_identity"]["identity_hash"]
                or artifact["provider_hash"] != provider.provider_hash
                or artifact["scientific_spec_hash"] != manifest["scientific_spec_hash"]
                or artifact["numerical_method_config_hash"] != manifest["numerical_method_config_hash"]
            ):
                raise ValueError("task identity mismatch")
            rows_payload = json.loads((task_dir / "rows.json").read_text(encoding="utf-8"))
            rows = tuple(_row_from_payload(item) for item in rows_payload)
            metrics = tuple(_decode(item) for item in json.loads((task_dir / "metrics.json").read_text(encoding="utf-8")))
            if len(rows) != artifact["row_count"] or len(metrics) != len(rows):
                raise ValueError("task row/metric count mismatch")
            for metric in metrics:
                if set(metric) != _METRIC_FIELDS:
                    raise ValueError("task metric fields differ from the exact schema")
                numeric = (
                    metric["evaluation_count"], metric["cpu_seconds"], metric["wall_seconds"],
                    metric["row_bytes"], metric["sidecar_bytes"],
                )
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric):
                    raise ValueError("task metric contains a nonnumeric value")
                if any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric):
                    raise ValueError("task metric contains an invalid numeric value")
                if (
                    int(metric["task_id"]) != task_id
                    or metric["timing_scope"] != "task_conservative_shared"
                    or float(metric["cpu_seconds"]) < 1e-6
                    or float(metric["wall_seconds"]) < 1e-6
                    or float(metric["cpu_seconds"]) != float(artifact["task_cpu_seconds"])
                    or float(metric["wall_seconds"]) != float(artifact["task_wall_seconds"])
                ):
                    raise ValueError("task metric timing/identity is not conservatively bound")
            if len({float(item["cpu_seconds"]) for item in metrics}) != 1 or len(
                {float(item["wall_seconds"]) for item in metrics}
            ) != 1:
                raise ValueError("task metrics do not share the frozen task timing")
            sidecar_map = {}
            for item in artifact["sidecar_index"]:
                path = task_dir / item["task_relative_path"]
                if (
                    not path.is_file()
                    or sha256_file(path) != item["sha256"]
                    or path.stat().st_size != int(item["byte_count"])
                ):
                    raise ValueError("task sidecar evidence mismatch")
                logical_path = item["logical_path"]
                if logical_path in all_sidecars:
                    raise ValueError("duplicate global sidecar path")
                sidecar_map[logical_path] = path.read_bytes()
                all_sidecars[logical_path] = sidecar_map[logical_path]
            derived_metrics = []
            for row, metric in zip(rows, metrics):
                sidecar_bytes = sidecar_map.get(row.sidecar.relative_path)
                if sidecar_bytes is None:
                    raise ValueError("row sidecar is absent from the task index")
                expected_metric = {
                    "task_id": task_id,
                    "descriptor_hash": row.descriptor_hash,
                    "method": row.method,
                    "stratum": _stratum(
                        row.method,
                        descriptor_lookup[(row.suite_class, row.descriptor_index)],
                    ),
                    "evaluation_count": row.evaluation_count,
                    "cpu_seconds": float(metric["cpu_seconds"]),
                    "wall_seconds": float(metric["wall_seconds"]),
                    "row_bytes": len(canonical_bytes(row)),
                    "sidecar_bytes": len(sidecar_bytes),
                    "timing_scope": "task_conservative_shared",
                }
                if metric != expected_metric:
                    raise ValueError("task metric differs from source/file reconstruction")
                derived_metrics.append(expected_metric)
            rows_by_descriptor: Dict[Tuple[str, int], list[TerminalEvidenceRow]] = {}
            for row in rows:
                rows_by_descriptor.setdefault((row.suite_class, row.descriptor_index), []).append(row)
            for raw_ref in raw_task["descriptors"]:
                key = (raw_ref["suite_class"], int(raw_ref["descriptor_index"]))
                descriptor = descriptor_lookup[key]
                descriptor_rows = tuple(rows_by_descriptor.pop(key, ()))
                bundle_paths = {row.sidecar.relative_path for row in descriptor_rows}
                bundle = TerminalEvidenceBundle(
                    descriptor.descriptor_hash,
                    descriptor_rows,
                    tuple(sorted((path, sidecar_map[path]) for path in bundle_paths if path in sidecar_map)),
                )
                mdp, belief = reconstruct_terminal_evidence_source(descriptor, provider)
                bundle_failures = validate_terminal_evidence_bundle_source(bundle, descriptor, mdp, belief)
                if bundle_failures:
                    raise ValueError("descriptor evidence mismatch: " + ",".join(bundle_failures))
                if tuple(row.method for row in descriptor_rows) != tuple(raw_ref["expected_methods"]):
                    raise ValueError("descriptor method-plan mismatch")
            if rows_by_descriptor:
                raise ValueError("unexpected descriptor rows in task")
            all_rows.extend(rows)
            all_metrics.extend(derived_metrics)
            job_hosts.add(artifact["hostname"])
            observed_tasks.append(task_id)
        except Exception as error:  # retain all failures in the provisional artifact
            failures.append(f"task_{task_id}:{type(error).__name__}:{error}")

    expected_tasks = tuple(range(1, int(manifest["task_count"]) + 1))
    counts_match = (
        tuple(observed_tasks) == expected_tasks
        and len(all_rows) == int(manifest["expected_row_count"])
        and len(all_sidecars) == int(manifest["expected_sidecar_count"])
    )
    fail_closed_exercised = False
    negative_control_pass = False
    if all_rows:
        first = all_rows[0]
        forged = replace(first, status="forged", logical_record_hash="")
        forged = replace(forged, logical_record_hash=terminal_evidence_row_hash(forged))
        descriptor = descriptor_lookup[(first.suite_class, first.descriptor_index)]
        mdp, belief = reconstruct_terminal_evidence_source(descriptor, provider)
        related = tuple(
            row if row.method != first.method else forged
            for row in all_rows
            if row.suite_class == first.suite_class and row.descriptor_index == first.descriptor_index
        )
        paths = {row.sidecar.relative_path for row in related}
        forged_bundle = TerminalEvidenceBundle(
            descriptor.descriptor_hash,
            related,
            tuple(sorted((path, all_sidecars[path]) for path in paths)),
        )
        fail_closed_exercised = True
        negative_control_pass = bool(
            validate_terminal_evidence_bundle_source(forged_bundle, descriptor, mdp, belief)
        )

    tie_rows = [row for row in all_rows if row.tie_status not in (None, "unique")]
    symmetry_rows = [row for row in all_rows if row.symmetry_required]
    reference_a = [row for row in all_rows if row.method == "reference_a"]
    reference_b = [row for row in all_rows if row.method == "reference_b"]
    unresolved = [row for row in all_rows if row.unresolved]
    failed = [row for row in all_rows if not row.pass_status]
    payload: Dict[str, Any] = {
        "schema": PROVISIONAL_SCHEMA,
        "artifact_type": manifest["artifact_type"],
        "artifact_status": "provisional",
        "stage_complete": False,
        "manifest_hash": manifest["manifest_hash"],
        "source_hash_match": True,
        "scientific_spec_hash_match": True,
        "numerical_method_config_hash_match": True,
        "manifest_hash_match": True,
        "observed_task_count": len(observed_tasks),
        "observed_row_count": len(all_rows),
        "observed_sidecar_count": len(all_sidecars),
        "positive_reference_a_count": len(reference_a),
        "positive_reference_b_count": len(reference_b),
        "reference_a_complete": bool(reference_a) and not any(row.unresolved for row in reference_a),
        "reference_b_complete": bool(reference_b) and not any(row.unresolved for row in reference_b),
        "tie_path_exercised": bool(tie_rows),
        "tie_path_pass": bool(tie_rows) and all(row.pass_status for row in tie_rows),
        "symmetry_path_exercised": bool(symmetry_rows),
        "symmetry_path_pass": bool(symmetry_rows) and all(row.symmetry_pass for row in symmetry_rows),
        "scalar_batch_parity_pass": all(row.scalar_batch_pass for row in all_rows if row.scalar_batch_required),
        "fail_closed_path_exercised": fail_closed_exercised,
        "negative_control_rejection_pass": negative_control_pass,
        "unexpected_reference_unresolved_count": len(unresolved),
        "unexpected_validation_failure_count": len(failed),
        "missing_duplicate_malformed_nonfinite_stale_invalid_count": len(failures),
        "coverage_match": counts_match,
        "failure_reasons": tuple(failures),
        "task_artifact_hashes": tuple(
            (task_id, sha256_file(output_root / "tasks" / f"task_{task_id:05d}" / "task.json"))
            for task_id in observed_tasks
        ),
        "job_hosts": tuple(sorted(job_hosts)),
        "metrics": tuple(all_metrics),
        "qacct_audit_pass": False,
        "finalization_hash_bind_pass": False,
        "independent_readback_pass": False,
        "feasibility_gate_pass": False,
        "logical_record_hash": "",
    }
    payload["provisional_gate_pass"] = all((
        counts_match,
        not failures,
        len(reference_a) == int(manifest["expected_positive_reference_a_count"]),
        len(reference_b) == int(manifest["expected_positive_reference_b_count"]),
        len(tie_rows) == int(manifest["expected_tie_path_row_count"]),
        len(symmetry_rows) == int(manifest["expected_symmetry_path_row_count"]),
        payload["reference_a_complete"],
        payload["reference_b_complete"],
        payload["tie_path_exercised"],
        payload["tie_path_pass"],
        payload["symmetry_path_exercised"],
        payload["symmetry_path_pass"],
        payload["scalar_batch_parity_pass"],
        fail_closed_exercised,
        negative_control_pass,
        not unresolved,
        not failed,
    ))
    payload["logical_record_hash"] = logical_hash(_without_hash(payload, "logical_record_hash"))
    return payload


def collect_provisional(
    *,
    manifest: Mapping[str, Any],
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Callable[[CanonicalBaseProvider], bool],
    output_root: Path,
    provisional_path: Path,
) -> Mapping[str, Any]:
    if provisional_path.exists():
        raise FileExistsError("provisional artifact already exists")
    payload = recompute_provisional(
        manifest=manifest,
        suites=suites,
        provider=provider,
        acceptance_validator=acceptance_validator,
        output_root=output_root,
    )
    write_new_json(provisional_path, payload)
    return payload


def parse_qacct_records(text: str) -> Tuple[Dict[str, str], ...]:
    records = []
    current: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) == {"="}:
            if current:
                records.append(current)
                current = {}
            continue
        parts = stripped.split(None, 1)
        if len(parts) == 2:
            key, value = parts
            if key == "jobnumber" and current:
                records.append(current)
                current = {}
            current[key] = value.strip()
    if current:
        records.append(current)
    return tuple(records)


def _duration_seconds(value: str) -> float:
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 1:
            result = float(parts[0])
        else:
            raise ValueError
    except ValueError as error:
        raise RuntimeError(f"invalid qacct duration: {value}") from error
    if not math.isfinite(result) or result < 0:
        raise RuntimeError("qacct duration is nonfinite or negative")
    return result


def _memory_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value.strip(), re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"invalid qacct memory: {value}")
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3,
             "T": 1024**4, "P": 1024**5}[match.group(2).upper()]
    result = math.ceil(float(match.group(1)) * scale)
    if result < 1:
        raise RuntimeError("qacct memory must be positive")
    return result


def _queue_matches(requested: str, observed: str) -> bool:
    queue = observed.split("@", 1)[0]
    return queue in (requested, f"{requested}.q")


def _h_rt_seconds(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise RuntimeError("job script h_rt is malformed")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _job_script_semantics(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    job_name: str,
    task_ids: Tuple[int, ...],
    array_job: bool,
) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?m)^#\$\s+-pe\b", text) or "PE_HOSTFILE" in text:
        raise RuntimeError("parallel-environment semantics are forbidden")

    def unique(pattern: str, context: str, required: bool = True) -> Optional[str]:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if len(matches) != (1 if required else 0):
            if not required and not matches:
                return None
            raise RuntimeError(f"job script {context} directive mismatch")
        return str(matches[0])

    queue = unique(r"^#\$\s+-q\s+(\S+)\s*$", "queue")
    name = unique(r"^#\$\s+-N\s+(\S+)\s*$", "name")
    h_rt = unique(r"^#\$\s+-l\s+h_rt=(\S+)\s*$", "h_rt")
    memory = unique(r"^#\$\s+-l\s+h_data=(\d+)\s*$", "h_data")
    array_ranges = re.findall(r"^#\$\s+-t\s+(\d+)-(\d+)\s*$", text, flags=re.MULTILINE)
    throttles = re.findall(r"^#\$\s+-tc\s+(\d+)\s*$", text, flags=re.MULTILINE)
    resources = manifest["resources"]
    if queue != resources["queue"] or name != job_name:
        raise RuntimeError("job script queue/name differs from submission contract")
    if _h_rt_seconds(str(h_rt)) != resources["h_rt_seconds"] or int(str(memory)) != resources["memory_bytes"]:
        raise RuntimeError("job script resources differ from manifest")
    if text.count("scripts/terminal_validation_array.py run-task") != 1:
        raise RuntimeError("job script must invoke the exact terminal task command once")
    required_flags = ("--manifest", "--output-root", "--task-id")
    for required_flag in required_flags:
        if len(re.findall(rf"(?<!\S){re.escape(required_flag)}(?=\s|$)", text)) != 1:
            raise RuntimeError(f"job script command flag mismatch: {required_flag}")
    forbidden_legacy_flags = (
        "--migration",
        "--approved-migration-hash",
        "--migration-execution-approval",
        "--approved-migration-execution-approval-hash",
        "--provider-acceptance",
        "--approved-provider-acceptance-hash",
    )
    if any(
        re.search(rf"(?<!\S){re.escape(flag)}(?=\s|$)", text)
        for flag in forbidden_legacy_flags
    ):
        raise RuntimeError("job script contains a legacy provider acceptance flag")
    allowed_directives = (
        r"^#\$\s+-cwd$", r"^#\$\s+-N\s+\S+$", r"^#\$\s+-q\s+\S+$",
        r"^#\$\s+-j\s+y$", r"^#\$\s+-o\s+\S+$",
        r"^#\$\s+-l\s+h_rt=\S+$", r"^#\$\s+-l\s+h_data=\d+$",
        r"^#\$\s+-t\s+\d+-\d+$", r"^#\$\s+-tc\s+\d+$",
    )
    directives = [line.strip() for line in text.splitlines() if line.strip().startswith("#$")]
    if any(not any(re.fullmatch(pattern, line) for pattern in allowed_directives) for line in directives):
        raise RuntimeError("job script contains an unapproved scheduler directive")
    for required_directive in (r"^#\$\s+-cwd$", r"^#\$\s+-j\s+y$", r"^#\$\s+-o\s+\S+$"):
        if sum(bool(re.fullmatch(required_directive, line)) for line in directives) != 1:
            raise RuntimeError("job script fixed scheduler directives mismatch")
    logical_text = text.replace("\\\n", " ")
    commands = []
    for raw_line in logical_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if (
            line == "set -euo pipefail"
            or line in ("export LANG=C", "export LC_ALL=C")
            or line.startswith("cd ")
            or re.fullmatch(r"task_id=(?:\d+|\$\{SGE_TASK_ID\})", line)
        ):
            continue
        commands.append(line)
    if len(commands) != 1:
        raise RuntimeError("job script must contain exactly one executable command")
    tokens = shlex.split(commands[0])
    if len(tokens) != 3 + 2 * len(required_flags):
        raise RuntimeError("job script terminal command shape mismatch")
    if tokens[1:3] != ["scripts/terminal_validation_array.py", "run-task"]:
        raise RuntimeError("job script terminal command target mismatch")
    if tuple(tokens[3::2]) != required_flags or any(not value for value in tokens[4::2]):
        raise RuntimeError("job script terminal command arguments mismatch")
    if array_job:
        expected_range = [("1", str(manifest["task_count"]))]
        if array_ranges != expected_range or throttles != [str(resources["throttle"])]:
            raise RuntimeError("full array range/throttle differs from manifest")
        if task_ids != tuple(range(1, int(manifest["task_count"]) + 1)):
            raise RuntimeError("full array task mapping differs from manifest")
        if not re.search(r"(?m)^task_id=\$\{SGE_TASK_ID\}\s*$", text):
            raise RuntimeError("full array task ID is not sourced from SGE_TASK_ID")
        task_id_mode = "sge_array_exact"
        throttle = resources["throttle"]
    else:
        if array_ranges or throttles or len(task_ids) != 1:
            raise RuntimeError("smoke task must be one non-array submission")
        if not re.search(rf"(?m)^task_id={task_ids[0]}\s*$", text):
            raise RuntimeError("smoke task literal ID differs from manifest")
        task_id_mode = "literal_non_array"
        throttle = 1
    semantics = {
        "queue": queue,
        "job_name": name,
        "slots": 1,
        "h_rt_seconds": resources["h_rt_seconds"],
        "memory_bytes": resources["memory_bytes"],
        "array_job": array_job,
        "task_ids": task_ids,
        "throttle": throttle,
        "task_id_mode": task_id_mode,
        "parallel_environment": None,
        "command": "terminal_validation_array.py run-task",
    }
    return semantics


def create_scheduler_evidence(
    manifest: Mapping[str, Any],
    submissions: Sequence[Mapping[str, Any]],
    *,
    evidence_root: Path,
) -> Dict[str, Any]:
    """Bind raw terse qsub output to the exact manifest task partition."""

    expected = {int(task["task_id"]) for task in manifest["tasks"]}
    if manifest["stage"] == "smoke":
        if len(submissions) != int(manifest["task_count"]):
            raise RuntimeError("smoke requires one non-array submission per manifest task")
    elif manifest["stage"] == "full":
        if len(submissions) != 1:
            raise RuntimeError("full stage requires exactly one manifest array submission")
    else:
        raise RuntimeError("scheduler stage is unknown")
    observed = set()
    normalized = []
    for raw in submissions:
        job_id = str(raw["job_id"])
        job_name = str(raw["job_name"])
        queue = str(raw["queue"])
        array_job = bool(raw["array_job"])
        task_ids = tuple(int(item) for item in raw["manifest_task_ids"])
        qsub_path = Path(str(raw["qsub_raw_path"])).resolve()
        job_script_path = Path(str(raw["job_script_path"])).resolve()
        root = evidence_root.resolve()
        try:
            qsub_relative = qsub_path.relative_to(root).as_posix()
            job_relative = job_script_path.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("scheduler raw evidence must stay inside evidence_root") from error
        if not job_id.isdigit() or not job_name or not queue or not task_ids:
            raise RuntimeError("scheduler submission identity is malformed")
        if queue != manifest["resources"]["queue"]:
            raise RuntimeError("scheduler queue differs from manifest")
        if manifest["stage"] == "smoke" and array_job:
            raise RuntimeError("smoke submissions must be non-array")
        if manifest["stage"] == "full" and not array_job:
            raise RuntimeError("full submission must be an array")
        if observed.intersection(task_ids):
            raise RuntimeError("scheduler submissions overlap manifest tasks")
        if not set(task_ids).issubset(expected):
            raise RuntimeError("scheduler submission contains an unknown task")
        if not array_job and len(task_ids) != 1:
            raise RuntimeError("non-array submissions must own exactly one task")
        if not qsub_path.is_file():
            raise RuntimeError("raw qsub evidence is missing")
        if not job_script_path.is_file():
            raise RuntimeError("submitted job script evidence is missing")
        raw_text = qsub_path.read_text(encoding="utf-8").strip()
        match = re.fullmatch(r"(?:Your job(?:-array)?\s+)?([0-9]+)(?:\.[0-9:-]+)?(?:\s+.*)?", raw_text)
        if match is None or match.group(1) != job_id:
            raise RuntimeError("raw qsub output does not match the claimed job ID")
        semantics = _job_script_semantics(
            job_script_path,
            manifest,
            job_name=job_name,
            task_ids=task_ids,
            array_job=array_job,
        )
        observed.update(task_ids)
        normalized.append({
            "job_id": job_id,
            "job_name": job_name,
            "queue": queue,
            "array_job": array_job,
            "manifest_task_ids": task_ids,
            "qsub_raw_path": qsub_relative,
            "qsub_raw_sha256": sha256_file(qsub_path),
            "job_script_path": job_relative,
            "job_script_sha256": sha256_file(job_script_path),
            "slots": semantics["slots"],
            "h_rt_seconds": semantics["h_rt_seconds"],
            "memory_bytes": semantics["memory_bytes"],
            "throttle": semantics["throttle"],
            "task_id_mode": semantics["task_id_mode"],
            "parallel_environment": semantics["parallel_environment"],
            "job_script_semantics_hash": logical_hash(semantics),
        })
    if observed != expected:
        raise RuntimeError("scheduler submissions do not exactly cover manifest tasks")
    if manifest["stage"] == "smoke" and tuple(
        int(item["manifest_task_ids"][0]) for item in normalized
    ) != tuple(range(1, int(manifest["task_count"]) + 1)):
        raise RuntimeError("smoke submissions do not preserve manifest task order")
    if manifest["stage"] == "full" and tuple(normalized[0]["manifest_task_ids"]) != tuple(
        range(1, int(manifest["task_count"]) + 1)
    ):
        raise RuntimeError("full array does not preserve exact manifest task order")
    payload: Dict[str, Any] = {
        "schema": SCHEDULER_EVIDENCE_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "submissions": tuple(normalized),
        "logical_record_hash": "",
    }
    payload["logical_record_hash"] = logical_hash(_without_hash(payload, "logical_record_hash"))
    return payload


def audit_qacct(
    manifest: Mapping[str, Any],
    scheduler_evidence: Mapping[str, Any],
    qacct_files: Mapping[str, Path],
    *,
    evidence_root: Path,
) -> Dict[str, Any]:
    if scheduler_evidence.get("schema") != SCHEDULER_EVIDENCE_SCHEMA:
        raise RuntimeError("scheduler evidence schema mismatch")
    if set(scheduler_evidence) != _SCHEDULER_FIELDS:
        raise RuntimeError("scheduler evidence fields differ from the exact schema")
    if any(set(item) != _SUBMISSION_FIELDS for item in scheduler_evidence.get("submissions", ())):
        raise RuntimeError("scheduler submission fields differ from the exact schema")
    _validate_self_hash(scheduler_evidence, "logical_record_hash", "scheduler evidence")
    if scheduler_evidence.get("manifest_hash") != manifest.get("manifest_hash"):
        raise RuntimeError("scheduler evidence manifest mismatch")
    reconstructed_scheduler = create_scheduler_evidence(
        manifest,
        tuple({
            "job_id": item["job_id"],
            "job_name": item["job_name"],
            "queue": item["queue"],
            "array_job": item["array_job"],
            "manifest_task_ids": item["manifest_task_ids"],
            "qsub_raw_path": evidence_root / str(item["qsub_raw_path"]),
            "job_script_path": evidence_root / str(item["job_script_path"]),
        } for item in scheduler_evidence["submissions"]),
        evidence_root=evidence_root,
    )
    if reconstructed_scheduler != scheduler_evidence:
        raise RuntimeError("scheduler evidence differs from raw source reconstruction")
    expected_tasks = {int(task["task_id"]) for task in manifest["tasks"]}
    observed_tasks = set()
    raw_hashes = []
    parsed_records = []
    task_usage = []
    task_bindings = []
    root = evidence_root.resolve()
    for submission in scheduler_evidence.get("submissions", ()):
        job_id = str(submission["job_id"])
        if (
            submission["queue"] != manifest["resources"]["queue"]
            or int(submission["slots"]) != 1
            or int(submission["h_rt_seconds"]) != int(manifest["resources"]["h_rt_seconds"])
            or int(submission["memory_bytes"]) != int(manifest["resources"]["memory_bytes"])
            or submission["parallel_environment"] is not None
        ):
            raise RuntimeError("scheduler resources differ from manifest")
        for path_field, hash_field in (
            ("qsub_raw_path", "qsub_raw_sha256"),
            ("job_script_path", "job_script_sha256"),
        ):
            source_path = root / str(submission[path_field])
            if not source_path.is_file() or sha256_file(source_path) != submission[hash_field]:
                raise RuntimeError("scheduler raw evidence hash mismatch")
        semantics = _job_script_semantics(
            root / str(submission["job_script_path"]),
            manifest,
            job_name=str(submission["job_name"]),
            task_ids=tuple(int(item) for item in submission["manifest_task_ids"]),
            array_job=bool(submission["array_job"]),
        )
        if logical_hash(semantics) != submission["job_script_semantics_hash"]:
            raise RuntimeError("scheduler job-script semantics changed")
        path = qacct_files.get(job_id)
        if path is None or not path.is_file():
            raise RuntimeError(f"qacct evidence missing for job {job_id}")
        try:
            relative_path = path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("qacct raw evidence must stay inside evidence_root") from error
        raw_hashes.append((job_id, relative_path, sha256_file(path)))
        records = parse_qacct_records(path.read_text(encoding="utf-8"))
        if not records:
            raise RuntimeError(f"qacct has no records for job {job_id}")
        expected_job_tasks = tuple(int(item) for item in submission["manifest_task_ids"])
        actual_job_tasks = []
        for record in records:
            required = {
                "jobnumber", "jobname", "qname", "hostname", "slots", "failed",
                "exit_status", "cpu", "ru_wallclock", "maxvmem",
            }
            if not required.issubset(record):
                raise RuntimeError("qacct record lacks required fields")
            if record["jobnumber"] != job_id or record["jobname"] != submission["job_name"]:
                raise RuntimeError("qacct job identity mismatch")
            if not _queue_matches(str(submission["queue"]), record["qname"]):
                raise RuntimeError("qacct queue mismatch")
            if record["slots"] != "1" or record["failed"] != "0" or record["exit_status"] != "0":
                raise RuntimeError("qacct does not prove a successful one-slot task")
            host = record["hostname"].split(".", 1)[0].lower()
            if host.startswith("login") or _COMPUTE_HOST_RE.fullmatch(host) is None:
                raise RuntimeError("qacct hostname is not a compute node")
            if submission["array_job"]:
                if not record.get("taskid", "").isdigit():
                    raise RuntimeError("array qacct task ID is absent")
                task_id = int(record["taskid"])
            else:
                if len(expected_job_tasks) != 1 or record.get("taskid") not in (None, "undefined", "NONE"):
                    raise RuntimeError("non-array qacct task shape mismatch")
                task_id = expected_job_tasks[0]
            actual_job_tasks.append(task_id)
            observed_tasks.add(task_id)
            parsed_records.append(record)
            cpu_seconds = _duration_seconds(record["cpu"])
            wall_seconds = _duration_seconds(record["ru_wallclock"])
            memory_bytes = _memory_bytes(record["maxvmem"])
            task_usage.append((
                task_id,
                cpu_seconds,
                wall_seconds,
            ))
            task_bindings.append({
                "task_id": task_id,
                "job_id": job_id,
                "sge_task_id": task_id if submission["array_job"] else None,
                "hostname": host,
                "slots": 1,
                "cpu_seconds": cpu_seconds,
                "wall_seconds": wall_seconds,
                "max_memory_bytes": memory_bytes,
            })
        if tuple(sorted(actual_job_tasks)) != tuple(sorted(expected_job_tasks)):
            raise RuntimeError("qacct task coverage differs from submission evidence")
    if observed_tasks != expected_tasks or len(parsed_records) != len(expected_tasks):
        raise RuntimeError("qacct coverage is not exactly the manifest task set")
    payload: Dict[str, Any] = {
        "schema": QACCT_AUDIT_SCHEMA,
        "manifest_hash": manifest["manifest_hash"],
        "scheduler_evidence_hash": scheduler_evidence["logical_record_hash"],
        "raw_qacct_hashes": tuple(sorted(raw_hashes)),
        "expected_task_count": len(expected_tasks),
        "observed_task_count": len(observed_tasks),
        "task_usage_seconds": tuple(sorted(task_usage)),
        "task_bindings": tuple(sorted(task_bindings, key=lambda item: item["task_id"])),
        "qacct_audit_pass": True,
        "logical_record_hash": "",
    }
    payload["logical_record_hash"] = logical_hash(_without_hash(payload, "logical_record_hash"))
    return payload


def _revalidate_raw_scheduler_evidence(
    scheduler: Mapping[str, Any],
    qacct: Mapping[str, Any],
    evidence_root: Path,
) -> None:
    root = evidence_root.resolve()
    for submission in scheduler.get("submissions", ()):
        for path_field, hash_field in (
            ("qsub_raw_path", "qsub_raw_sha256"),
            ("job_script_path", "job_script_sha256"),
        ):
            path = root / str(submission[path_field])
            if not path.is_file() or sha256_file(path) != submission[hash_field]:
                raise RuntimeError("raw scheduler evidence changed after qacct audit")
    for job_id, relative_path, claimed_hash in qacct.get("raw_qacct_hashes", ()):
        path = root / str(relative_path)
        if not path.is_file() or sha256_file(path) != claimed_hash:
            raise RuntimeError(f"raw qacct evidence changed for job {job_id}")


def validate_task_scheduler_bindings(
    manifest: Mapping[str, Any],
    *,
    task_output_root: Path,
    scheduler: Mapping[str, Any],
    qacct: Mapping[str, Any],
) -> None:
    """Require each retained task to match one qsub and one qacct identity."""

    submission_by_task: Dict[int, Mapping[str, Any]] = {}
    for submission in scheduler["submissions"]:
        for task_id in submission["manifest_task_ids"]:
            key = int(task_id)
            if key in submission_by_task:
                raise RuntimeError("scheduler task binding is duplicated")
            submission_by_task[key] = submission
    binding_by_task: Dict[int, Mapping[str, Any]] = {}
    for binding in qacct["task_bindings"]:
        key = int(binding["task_id"])
        if key in binding_by_task:
            raise RuntimeError("qacct task binding is duplicated")
        binding_by_task[key] = binding
    expected = set(range(1, int(manifest["task_count"]) + 1))
    if set(submission_by_task) != expected or set(binding_by_task) != expected:
        raise RuntimeError("scheduler/qacct task bindings are not exact")
    for raw_task in manifest["tasks"]:
        task_id = int(raw_task["task_id"])
        artifact = _load_task_artifact(
            task_output_root / "tasks" / f"task_{task_id:05d}"
        )
        submission = submission_by_task[task_id]
        binding = binding_by_task[task_id]
        expected_sge = task_id if manifest["stage"] == "full" else None
        observed_sge = artifact["sge_task_id"]
        if observed_sge in ("undefined", "NONE", ""):
            observed_sge = None
        elif observed_sge is not None:
            if not str(observed_sge).isdigit():
                raise RuntimeError("task artifact SGE task ID is malformed")
            observed_sge = int(observed_sge)
        exact = all((
            artifact["job_id"] == submission["job_id"] == binding["job_id"],
            observed_sge == expected_sge == binding["sge_task_id"],
            artifact["hostname"] == binding["hostname"],
            int(artifact["slots"]) == int(binding["slots"]) == 1,
            artifact["manifest_hash"] == manifest["manifest_hash"],
            artifact["source_identity_hash"] == manifest["source_identity"]["identity_hash"],
            artifact["provider_hash"] == manifest["provider_hash"],
            artifact["scientific_spec_hash"] == manifest["scientific_spec_hash"],
            artifact["numerical_method_config_hash"] == manifest["numerical_method_config_hash"],
            artifact["assignment_hash"] == raw_task["assignment_hash"],
            int(artifact["logical_case_owner"]) == int(raw_task["logical_case_owner"]),
            int(artifact["subshard_index"]) == int(raw_task["subshard_index"]),
            int(artifact["subshard_count"]) == int(raw_task["subshard_count"]),
        ))
        if not exact:
            raise RuntimeError(f"task {task_id} is not bound to qsub/qacct/manifest identity")
        task_cpu = float(artifact["task_cpu_seconds"])
        task_wall = float(artifact["task_wall_seconds"])
        if (
            task_cpu < 1e-6 or task_wall < 1e-6
            or task_cpu > float(binding["cpu_seconds"])
            or task_wall > float(binding["wall_seconds"])
        ):
            raise RuntimeError(f"task {task_id} timers are not bounded by qacct")


def _recompute_qacct_audit(
    manifest: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    qacct: Mapping[str, Any],
    evidence_root: Path,
) -> None:
    qacct_paths = {
        str(job_id): evidence_root / str(relative_path)
        for job_id, relative_path, _ in qacct.get("raw_qacct_hashes", ())
    }
    recomputed = audit_qacct(
        manifest, scheduler, qacct_paths, evidence_root=evidence_root
    )
    if recomputed != qacct:
        raise RuntimeError("qacct audit differs from raw-evidence recomputation")


def _fixed_feasibility_paths(
    *,
    provisional_path: Path,
    scheduler_evidence_path: Path,
    qacct_audit_path: Path,
    compute_ceiling_path: Path,
    scheduler: Mapping[str, Any],
    qacct: Mapping[str, Any],
    scheduler_evidence_root: Path,
) -> Tuple[Path, ...]:
    paths = {
        provisional_path.resolve(), scheduler_evidence_path.resolve(),
        qacct_audit_path.resolve(), compute_ceiling_path.resolve(),
    }
    root = scheduler_evidence_root.resolve()
    for submission in scheduler["submissions"]:
        paths.add((root / str(submission["qsub_raw_path"])).resolve())
        paths.add((root / str(submission["job_script_path"])).resolve())
    for _, relative_path, _ in qacct["raw_qacct_hashes"]:
        paths.add((root / str(relative_path)).resolve())
    return tuple(sorted(paths, key=str))


def _validate_provisional_schema(value: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if value.get("schema") != PROVISIONAL_SCHEMA or set(value) != _PROVISIONAL_FIELDS:
        raise RuntimeError("provisional artifact fields differ from the exact schema")
    _validate_self_hash(value, "logical_record_hash", "provisional artifact")
    if (
        value.get("artifact_type") != manifest["artifact_type"]
        or value.get("artifact_status") != "provisional"
        or value.get("stage_complete") is not False
        or value.get("manifest_hash") != manifest["manifest_hash"]
    ):
        raise RuntimeError("provisional artifact identity/status mismatch")


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot estimate an empty feasibility stratum")
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def validate_compute_ceiling_binding(
    manifest: Mapping[str, Any],
    compute_ceiling: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> None:
    if compute_ceiling.get("schema") != COMPUTE_CEILING_SCHEMA:
        raise RuntimeError("compute ceiling report schema mismatch")
    if set(compute_ceiling) != _CEILING_FIELDS:
        raise RuntimeError("compute ceiling report fields differ from the exact schema")
    _validate_self_hash(compute_ceiling, "report_hash", "compute ceiling report")
    if compute_ceiling["report_hash"] != manifest["compute_ceiling_report_hash"]:
        raise RuntimeError("compute ceiling report differs from manifest")
    timestamp = datetime.fromisoformat(
        str(compute_ceiling["captured_at_utc"]).replace("Z", "+00:00")
    )
    age = (now or datetime.now(timezone.utc)) - timestamp
    if age.total_seconds() < 0 or age.total_seconds() > 24 * 3600:
        raise RuntimeError("compute ceiling report is not within 24 hours")
    resources = manifest["resources"]
    caps = manifest["numerical_resource_caps"]
    checks = (
        resources["queue"] in tuple(compute_ceiling["allowed_queues"]),
        int(resources["h_rt_seconds"]) <= min(
            int(compute_ceiling["max_walltime_seconds"]), int(caps["max_h_rt_seconds"])
        ),
        int(resources["memory_bytes"]) <= min(
            int(compute_ceiling["max_memory_bytes"]), int(caps["max_memory_bytes"])
        ),
        int(resources["throttle"]) <= min(
            int(compute_ceiling["max_throttle"]), int(caps["max_throttle"])
        ),
        int(manifest["task_count"]) <= min(
            int(compute_ceiling["max_array_tasks"]), int(caps["max_array_tasks"])
        ),
    )
    if not all(checks):
        raise RuntimeError("manifest resources do not fit the fresh Compute Ceiling Report")


def compute_feasibility(
    manifest: Mapping[str, Any],
    provisional: Mapping[str, Any],
    compute_ceiling: Mapping[str, Any],
    *,
    qacct_audit: Optional[Mapping[str, Any]] = None,
    fixed_artifact_paths: Sequence[Path],
    finalization_overhead_seconds: float,
    finalization_artifact_bytes: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    validate_compute_ceiling_binding(manifest, compute_ceiling, now=now)
    if (
        not math.isfinite(finalization_overhead_seconds)
        or finalization_overhead_seconds < 0
        or type(finalization_artifact_bytes) is not int
        or finalization_artifact_bytes < 1
    ):
        raise RuntimeError("finalization overhead/storage measurement is invalid")
    if qacct_audit is None or not qacct_audit.get("qacct_audit_pass"):
        raise RuntimeError("feasibility requires a passing raw-qacct audit")
    if set(qacct_audit) != _QACCT_FIELDS:
        raise RuntimeError("qacct audit fields differ from the exact schema")
    binding_by_task = {
        int(item["task_id"]): item for item in qacct_audit["task_bindings"]
    }
    if set(binding_by_task) != set(range(1, int(manifest["task_count"]) + 1)):
        raise RuntimeError("qacct usage coverage differs from smoke task count")
    by_stratum: Dict[str, Dict[str, list[float]]] = {}
    for metric in provisional["metrics"]:
        group = by_stratum.setdefault(metric["stratum"], {"cpu": [], "wall": [], "bytes": []})
        binding = binding_by_task.get(int(metric["task_id"]))
        if binding is None:
            raise RuntimeError("metric lacks an exact qacct task binding")
        # Complete qacct task usage is assigned to each row: conservative, not self-reported.
        group["cpu"].append(float(binding["cpu_seconds"]))
        group["wall"].append(float(binding["wall_seconds"]))
        group["bytes"].append(float(metric["row_bytes"]) + float(metric["sidecar_bytes"]))
    planned = dict(manifest["full_stage_strata_counts"])
    missing = sorted(set(planned) - set(by_stratum))
    if missing:
        raise RuntimeError("smoke lacks feasibility strata: " + ",".join(missing))
    upper = {}
    total_cpu = 0.0
    storage = 0.0
    for stratum, count in planned.items():
        values = by_stratum[stratum]
        upper_cpu = max(max(values["cpu"]), 1.5 * _p95(values["cpu"]))
        upper_wall = max(max(values["wall"]), 1.5 * _p95(values["wall"]))
        upper[stratum] = {"cpu_seconds": upper_cpu, "wall_seconds": upper_wall}
        total_cpu += int(count) * upper_cpu
        storage += int(count) * max(values["bytes"])
    task_metric_wall: Dict[int, float] = {}
    for metric in provisional["metrics"]:
        task_id = int(metric["task_id"])
        task_metric_wall[task_id] = max(
            task_metric_wall.get(task_id, 0.0), float(metric["wall_seconds"])
        )
    startup = max(
        max(0.0, float(binding["wall_seconds"]) - task_metric_wall[task_id])
        for task_id, binding in binding_by_task.items()
    )
    measured_overhead = max(startup, finalization_overhead_seconds)
    projected_wall = max(
        sum(int(count) * upper[stratum]["wall_seconds"] for stratum, count in task_strata)
        + measured_overhead
        for task_strata in manifest["planned_full_task_strata"]
    )
    unique_paths = {Path(path).resolve() for path in fixed_artifact_paths}
    if any(not path.is_file() for path in unique_paths):
        raise RuntimeError("fixed feasibility artifact is missing")
    fixed_bytes = len(canonical_bytes(manifest)) + sum(path.stat().st_size for path in unique_paths)
    projected_storage = math.ceil(1.25 * (
        storage + fixed_bytes + finalization_artifact_bytes
    ))
    resources = manifest["resources"]
    caps = manifest["numerical_resource_caps"]
    measured_memory = max(int(item["max_memory_bytes"]) for item in binding_by_task.values())
    comparisons = {
        "walltime": projected_wall <= min(float(resources["h_rt_seconds"]), float(compute_ceiling["max_walltime_seconds"]), float(caps["max_h_rt_seconds"])),
        "task_count": int(manifest["planned_full_task_count"]) <= min(int(compute_ceiling["max_array_tasks"]), int(caps["max_array_tasks"])),
        "throttle": int(resources["throttle"]) <= min(int(compute_ceiling["max_throttle"]), int(caps["max_throttle"])),
        "memory": measured_memory <= int(resources["memory_bytes"]) <= min(int(compute_ceiling["max_memory_bytes"]), int(caps["max_memory_bytes"])),
        "storage": projected_storage <= min(float(compute_ceiling["max_storage_bytes"]), float(caps["max_storage_bytes"])),
        "queue": resources["queue"] in tuple(compute_ceiling["allowed_queues"]),
    }
    quota = compute_ceiling.get("cpu_hours_quota")
    comparisons["cpu_hours"] = True if quota is None else total_cpu / 3600 <= float(quota)
    return {
        "stratum_upper_bounds": upper,
        "projected_total_cpu_hours": total_cpu / 3600,
        "projected_max_task_walltime_seconds": projected_wall,
        "projected_array_task_count": int(manifest["planned_full_task_count"]),
        "projected_storage_bytes": projected_storage,
        "measured_startup_overhead_seconds": startup,
        "measured_finalization_overhead_seconds": finalization_overhead_seconds,
        "applied_overhead_seconds": measured_overhead,
        "measured_fixed_artifact_bytes": fixed_bytes,
        "measured_finalization_artifact_bytes": finalization_artifact_bytes,
        "measured_max_memory_bytes": measured_memory,
        "comparisons": tuple(sorted(comparisons.items())),
        "cpu_quota_comparison": "not_applicable" if quota is None else "applicable",
        "feasibility_gate_pass": all(comparisons.values()),
    }


def finalize_post_job(
    *,
    manifest: Mapping[str, Any],
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Callable[[CanonicalBaseProvider], bool],
    task_output_root: Path,
    provisional_path: Path,
    scheduler_evidence_path: Path,
    qacct_audit_path: Path,
    compute_ceiling_path: Path,
    scheduler_evidence_root: Path,
    output_path: Path,
    project_root: Path,
    now: Optional[datetime] = None,
) -> Mapping[str, Any]:
    if output_path.exists():
        raise FileExistsError("post-job candidate already exists")
    started = time.perf_counter()
    validate_clean_source_identity(project_root, manifest["source_identity"])
    validate_execution_manifest(manifest, suites, provider, acceptance_validator)
    provisional = _decode(dict(_load_json(provisional_path)))
    scheduler = _decode(dict(_load_json(scheduler_evidence_path)))
    qacct = _decode(dict(_load_json(qacct_audit_path)))
    ceiling = _decode(dict(_load_json(compute_ceiling_path)))
    _validate_provisional_schema(provisional, manifest)
    _validate_self_hash(scheduler, "logical_record_hash", "scheduler evidence")
    _validate_self_hash(qacct, "logical_record_hash", "qacct audit")
    _revalidate_raw_scheduler_evidence(scheduler, qacct, scheduler_evidence_root)
    _recompute_qacct_audit(
        manifest, scheduler, qacct, scheduler_evidence_root
    )
    validate_task_scheduler_bindings(
        manifest,
        task_output_root=task_output_root,
        scheduler=scheduler,
        qacct=qacct,
    )
    recomputed_provisional = recompute_provisional(
        manifest=manifest,
        suites=suites,
        provider=provider,
        acceptance_validator=acceptance_validator,
        output_root=task_output_root,
    )
    if recomputed_provisional != provisional:
        raise RuntimeError("post-job provisional evidence differs from source recomputation")
    if not provisional.get("provisional_gate_pass") or not qacct.get("qacct_audit_pass"):
        raise RuntimeError("post-job finalization requires passing provisional and qacct gates")
    hashes = {
        "manifest": logical_hash(manifest),
        "provisional": sha256_file(provisional_path),
        "scheduler": sha256_file(scheduler_evidence_path),
        "qacct": sha256_file(qacct_audit_path),
        "compute_ceiling": sha256_file(compute_ceiling_path),
    }
    fixed_paths = _fixed_feasibility_paths(
        provisional_path=provisional_path,
        scheduler_evidence_path=scheduler_evidence_path,
        qacct_audit_path=qacct_audit_path,
        compute_ceiling_path=compute_ceiling_path,
        scheduler=scheduler,
        qacct=qacct,
        scheduler_evidence_root=scheduler_evidence_root,
    )
    overhead = time.perf_counter() - started
    artifact_bytes = 1
    payload: Dict[str, Any] = {}
    for _ in range(20):
        feasibility = compute_feasibility(
            manifest,
            provisional,
            ceiling,
            qacct_audit=qacct,
            fixed_artifact_paths=fixed_paths,
            finalization_overhead_seconds=overhead,
            finalization_artifact_bytes=artifact_bytes,
            now=now,
        )
        if not feasibility["feasibility_gate_pass"]:
            raise RuntimeError("post-job feasibility gate failed")
        payload = {
            "schema": POST_JOB_SCHEMA,
            "artifact_type": manifest["artifact_type"],
            "artifact_status": "post_job_finalized_candidate",
            "stage_complete": False,
            "manifest_hash": manifest["manifest_hash"],
            "bound_file_hashes": tuple(sorted(hashes.items())),
            "qacct_audit_pass": True,
            "finalization_hash_bind_pass": True,
            "independent_readback_pass": False,
            "feasibility": feasibility,
            "feasibility_gate_pass": True,
            "finalization_overhead_seconds": overhead,
            "final_gate_pass": True,
            "logical_record_hash": "",
        }
        payload["logical_record_hash"] = logical_hash(
            _without_hash(payload, "logical_record_hash")
        )
        encoded_size = len((json.dumps(
            _canonical(payload), indent=2, sort_keys=True, allow_nan=False
        ) + "\n").encode("utf-8"))
        if encoded_size == artifact_bytes:
            break
        artifact_bytes = encoded_size
    else:
        raise RuntimeError("post-job storage-size fixed point did not converge")
    if set(payload) != _POST_JOB_FIELDS:
        raise RuntimeError("post-job candidate fields differ from the exact schema")
    write_new_json(output_path, payload)
    if output_path.stat().st_size != artifact_bytes:
        raise RuntimeError("post-job artifact size differs from feasibility binding")
    return payload


def independent_readback(
    *,
    manifest: Mapping[str, Any],
    suites: Mapping[str, TerminalValidationSuite],
    provider: CanonicalBaseProvider,
    acceptance_validator: Callable[[CanonicalBaseProvider], bool],
    task_output_root: Path,
    provisional_path: Path,
    scheduler_evidence_path: Path,
    qacct_audit_path: Path,
    compute_ceiling_path: Path,
    scheduler_evidence_root: Path,
    post_job_path: Path,
    final_output_path: Path,
    project_root: Path,
    allow_non_darwin_for_tests: bool = False,
) -> Mapping[str, Any]:
    if final_output_path.exists():
        raise FileExistsError("independent read-back output already exists")
    if platform.system() != "Darwin" and not allow_non_darwin_for_tests:
        raise RuntimeError("independent read-back must run on the local Mac")
    validate_clean_source_identity(project_root, manifest["source_identity"])
    validate_execution_manifest(manifest, suites, provider, acceptance_validator)
    provisional = _decode(dict(_load_json(provisional_path)))
    scheduler = _decode(dict(_load_json(scheduler_evidence_path)))
    qacct = _decode(dict(_load_json(qacct_audit_path)))
    ceiling = _decode(dict(_load_json(compute_ceiling_path)))
    post_job = _decode(dict(_load_json(post_job_path)))
    _validate_provisional_schema(provisional, manifest)
    for value, field, context in (
        (scheduler, "logical_record_hash", "scheduler evidence"),
        (qacct, "logical_record_hash", "qacct audit"),
        (post_job, "logical_record_hash", "post-job candidate"),
    ):
        _validate_self_hash(value, field, context)
    if (
        post_job.get("schema") != POST_JOB_SCHEMA
        or set(post_job) != _POST_JOB_FIELDS
        or post_job.get("artifact_type") != manifest["artifact_type"]
        or post_job.get("artifact_status") != "post_job_finalized_candidate"
        or post_job.get("stage_complete") is not False
        or post_job.get("manifest_hash") != manifest["manifest_hash"]
        or post_job.get("qacct_audit_pass") is not True
        or post_job.get("finalization_hash_bind_pass") is not True
        or post_job.get("independent_readback_pass") is not False
        or post_job.get("feasibility_gate_pass") is not True
        or post_job.get("final_gate_pass") is not True
    ):
        raise RuntimeError("post-job candidate identity/status/gates mismatch")
    _revalidate_raw_scheduler_evidence(scheduler, qacct, scheduler_evidence_root)
    _recompute_qacct_audit(
        manifest, scheduler, qacct, scheduler_evidence_root
    )
    validate_task_scheduler_bindings(
        manifest,
        task_output_root=task_output_root,
        scheduler=scheduler,
        qacct=qacct,
    )
    recomputed_provisional = recompute_provisional(
        manifest=manifest,
        suites=suites,
        provider=provider,
        acceptance_validator=acceptance_validator,
        output_root=task_output_root,
    )
    if recomputed_provisional != provisional:
        raise RuntimeError("independent read-back rows/sidecars differ from source recomputation")
    local_host = platform.node().split(".", 1)[0].lower()
    remote_hosts = {str(item["hostname"]) for item in qacct["task_bindings"]}
    if local_host in remote_hosts:
        raise RuntimeError("independent read-back host is not independent")
    expected_hashes = dict(post_job["bound_file_hashes"])
    actual_hashes = {
        "manifest": logical_hash(manifest),
        "provisional": sha256_file(provisional_path),
        "scheduler": sha256_file(scheduler_evidence_path),
        "qacct": sha256_file(qacct_audit_path),
        "compute_ceiling": sha256_file(compute_ceiling_path),
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError("independent read-back file hashes differ from finalization")
    fixed_paths = _fixed_feasibility_paths(
        provisional_path=provisional_path,
        scheduler_evidence_path=scheduler_evidence_path,
        qacct_audit_path=qacct_audit_path,
        compute_ceiling_path=compute_ceiling_path,
        scheduler=scheduler,
        qacct=qacct,
        scheduler_evidence_root=scheduler_evidence_root,
    )
    feasibility = compute_feasibility(
        manifest,
        provisional,
        ceiling,
        qacct_audit=qacct,
        fixed_artifact_paths=fixed_paths,
        finalization_overhead_seconds=float(post_job["finalization_overhead_seconds"]),
        finalization_artifact_bytes=post_job_path.stat().st_size,
    )
    if feasibility != post_job["feasibility"]:
        raise RuntimeError("independent read-back feasibility recomputation differs")
    final_gate = all((
        provisional["provisional_gate_pass"],
        qacct["qacct_audit_pass"],
        post_job["finalization_hash_bind_pass"],
        feasibility["feasibility_gate_pass"],
        actual_hashes == expected_hashes,
    ))
    payload: Dict[str, Any] = {
        "schema": READBACK_SCHEMA,
        "artifact_type": manifest["artifact_type"],
        "artifact_status": "finalized",
        "stage_complete": final_gate,
        "manifest_hash": manifest["manifest_hash"],
        "source_hash_match": bool(provisional["source_hash_match"]),
        "scientific_spec_hash_match": bool(provisional["scientific_spec_hash_match"]),
        "numerical_method_config_hash_match": bool(provisional["numerical_method_config_hash_match"]),
        "manifest_hash_match": bool(provisional["manifest_hash_match"]),
        "observed_task_count": int(provisional["observed_task_count"]),
        "observed_row_count": int(provisional["observed_row_count"]),
        "observed_sidecar_count": int(provisional["observed_sidecar_count"]),
        "positive_reference_a_count": int(provisional["positive_reference_a_count"]),
        "positive_reference_b_count": int(provisional["positive_reference_b_count"]),
        "coverage_match": bool(provisional["coverage_match"]),
        "reference_a_complete": bool(provisional["reference_a_complete"]),
        "reference_b_complete": bool(provisional["reference_b_complete"]),
        "tie_path_exercised": bool(provisional["tie_path_exercised"]),
        "tie_path_pass": bool(provisional["tie_path_pass"]),
        "symmetry_path_exercised": bool(provisional["symmetry_path_exercised"]),
        "symmetry_path_pass": bool(provisional["symmetry_path_pass"]),
        "scalar_batch_parity_pass": bool(provisional["scalar_batch_parity_pass"]),
        "fail_closed_path_exercised": bool(provisional["fail_closed_path_exercised"]),
        "negative_control_rejection_pass": bool(provisional["negative_control_rejection_pass"]),
        "unexpected_reference_unresolved_count": int(provisional["unexpected_reference_unresolved_count"]),
        "unexpected_validation_failure_count": int(provisional["unexpected_validation_failure_count"]),
        "missing_duplicate_malformed_nonfinite_stale_invalid_count": int(
            provisional["missing_duplicate_malformed_nonfinite_stale_invalid_count"]
        ),
        "provisional_gate_pass": bool(provisional["provisional_gate_pass"]),
        "qacct_audit_pass": True,
        "finalization_hash_bind_pass": True,
        "independent_readback_pass": True,
        "feasibility": feasibility,
        "feasibility_gate_pass": bool(feasibility["feasibility_gate_pass"]),
        "final_gate_pass": final_gate,
        "readback_host": local_host,
        "post_job_hash": sha256_file(post_job_path),
        "logical_record_hash": "",
    }
    if set(payload) != _READBACK_FIELDS or not final_gate:
        raise RuntimeError("independent read-back final gate failed")
    payload["logical_record_hash"] = logical_hash(_without_hash(payload, "logical_record_hash"))
    write_new_json(final_output_path, payload)
    return payload


__all__ = [
    "COMPUTE_CEILING_SCHEMA",
    "DescriptorRef",
    "EXECUTION_AUTHORIZATION_SCHEMA",
    "EXECUTION_MANIFEST_SCHEMA",
    "QACCT_AUDIT_SCHEMA",
    "READBACK_SCHEMA",
    "SCHEDULER_EVIDENCE_SCHEMA",
    "SMOKE_CASE_IDS",
    "TaskSpec",
    "audit_qacct",
    "build_terminal_suites",
    "capture_clean_source_identity",
    "collect_provisional",
    "compute_feasibility",
    "create_scheduler_evidence",
    "create_execution_manifest",
    "execute_task",
    "execution_script_hashes",
    "finalize_post_job",
    "independent_readback",
    "load_accepted_canonical_base_provider",
    "logical_hash",
    "parse_qacct_records",
    "recompute_provisional",
    "sha256_file",
    "smoke_descriptors",
    "validate_clean_source_identity",
    "validate_compute_ceiling_binding",
    "validate_execution_authorization",
    "validate_execution_manifest",
    "validate_task_scheduler_bindings",
    "write_new_json",
]
