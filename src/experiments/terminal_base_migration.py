from __future__ import annotations

"""Reviewed one-time migration of the 90 authoritative terminal base beliefs.

The exporter is read-only with respect to project sources and refuses to run unless the
authoritative commit, tree, manifest, spec, cases, and source-file hashes all match. The
authoritative migration is intentionally not executed or checked into this workspace yet.
"""

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportBeliefState,
    FiniteSupportPrior,
)
from .positive_need import (
    DEFAULT_SPEC_PATH,
    _belief_hash as legacy_belief_hash,
    _numerical_belief,
    build_development_environments,
    load_positive_need_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs"
    / "reference"
    / "terminal_quadrature_manifest_v1.json"
)
ORIGINAL_MANIFEST_PROVENANCE_PATH = (
    "results/r6_prefeedback_quadrature_7376c5d_v1/"
    "r6_quadrature_diagnostic_manifest.json"
)
AUTHORITATIVE_STAGED_MANIFEST_PATH = PROJECT_ROOT / ORIGINAL_MANIFEST_PROVENANCE_PATH
DEFAULT_MIGRATION_PATH = (
    PROJECT_ROOT / "configs" / "terminal_base_beliefs_7376c5d_v1.json"
)

AUTHORITATIVE_COMMIT = "7376c5d70cf2520600894853e2a1275e8d0a89e1"
AUTHORITATIVE_TREE = "87100610dfea58375147f7c5064430f5a77d5926"
ORIGINAL_MANIFEST_FILE_HASH = (
    "9215d3e3823c1f01b070d6a575f214e2bff0f1617262a9445b66a451d02753d2"
)
ORIGINAL_MANIFEST_HASH = (
    "c05629107255799ca5f2fe537530c7f892685d2977d2584f1c39393d49b3c197"
)
ORIGINAL_SPEC_HASH = "9f3152419a6c3ef46cd33baefe6a13f00dc80004025f1fb2f1d2b08f9e307e10"
ORIGINAL_CASE_HASH = "90354e48a36283225b221360c91b07c21ad369fb7f6a8f8191d7f5ed85ef132b"

MIGRATION_SCHEMA = "terminal_base_belief_migration_v2"
MIGRATION_RECORD_SCHEMA = "terminal_base_belief_migration_record_v1"
SUPPORT_SCHEMA = "terminal_support_payload_v1"
BELIEF_SCHEMA = "terminal_belief_payload_v1"
EXECUTION_APPROVAL_SCHEMA = "terminal_base_migration_execution_approval_v1"
EXECUTION_APPROVAL_STATUS = "reviewer_approved_for_one_slot_candidate_migration"
AUTHORITATIVE_STATUS = "authoritative_review_migration"
SYNTHETIC_STATUS = "synthetic_test_fixture_not_authoritative"
MIGRATION_CLI_PATH = "scripts/export_terminal_base_migration.py"
MIGRATION_JOB_PATH = "scripts/hoffman2_terminal_base_migration.job"
MIGRATION_MODULE_PATH = "src/experiments/terminal_base_migration.py"
MIGRATION_TOOL_PATHS = tuple(
    sorted((MIGRATION_CLI_PATH, MIGRATION_JOB_PATH, MIGRATION_MODULE_PATH))
)
AUTHORITATIVE_UNTRACKED_PATHS = (
    ORIGINAL_MANIFEST_PROVENANCE_PATH,
    "scripts/export_terminal_base_migration.py",
    "src/experiments/terminal_base_migration.py",
)
AUTHORITATIVE_RUNTIME_KEYS = (
    "byteorder",
    "libc",
    "platform_machine",
    "platform_release",
    "platform_system",
    "python_build",
    "python_executable",
    "python_implementation",
    "python_version",
)
AUTHORITATIVE_DEPENDENCY_KEYS = ("numpy", "scipy")


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("migration identity cannot contain non-finite floats")
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("migration identity keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_value(item) for item in value)
    raise TypeError(f"unsupported migration identity type: {type(value).__name__}")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _decode_hex(value: str, field_name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be canonical float-hex")
    try:
        decoded = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{field_name} is invalid float-hex") from error
    if not math.isfinite(decoded) or decoded.hex() != value:
        raise ValueError(f"{field_name} is not canonical finite float-hex")
    return decoded


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_output(*args: str) -> str:
    return subprocess.run(
        ("git",) + args,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_original_manifest(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"original quadrature manifest is absent: {path}")
    if _file_hash(path) != ORIGINAL_MANIFEST_FILE_HASH:
        raise RuntimeError("original quadrature manifest file hash mismatch")
    manifest = dict(_load_json_without_duplicate_keys(path))
    unhashed = dict(manifest)
    claimed = unhashed.pop("manifest_hash", None)
    if claimed != ORIGINAL_MANIFEST_HASH or _legacy_hash(unhashed) != claimed:
        raise RuntimeError("original quadrature manifest self-hash mismatch")
    if manifest.get("git_commit") != AUTHORITATIVE_COMMIT:
        raise RuntimeError("original quadrature manifest commit mismatch")
    if manifest.get("git_tree_hash") != AUTHORITATIVE_TREE:
        raise RuntimeError("original quadrature manifest tree mismatch")
    if manifest.get("spec_hash") != ORIGINAL_SPEC_HASH:
        raise RuntimeError("original quadrature manifest spec hash mismatch")
    cases = list(manifest.get("numerical_cases", []))
    if (
        len(cases) != 90
        or [int(case["case_id"]) for case in cases] != list(range(90))
        or manifest.get("numerical_cases_hash") != ORIGINAL_CASE_HASH
        or _legacy_hash(cases) != ORIGINAL_CASE_HASH
    ):
        raise RuntimeError("original quadrature case identity mismatch")
    return manifest


def _load_json_without_duplicate_keys(path: Path) -> Mapping[str, object]:
    def reject_duplicates(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    raw = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("migration artifact must be a JSON object")
    return raw


@dataclass(frozen=True)
class OriginalCaseDescriptor:
    case_id: int
    environment: str
    environment_hash: str
    belief_kind: str
    belief_hash: str


@dataclass(frozen=True)
class SupportAtomPayload:
    total_need_hex: str
    gap_fraction_hex: str
    orientation: int


@dataclass(frozen=True)
class SupportPayload:
    schema: str
    states: Tuple[SupportAtomPayload, ...]
    prior_weights_hex: Tuple[str, ...]
    support_hash: str
    payload_hash: str


@dataclass(frozen=True)
class HistoryStepPayload:
    action_hex: str
    observation_hex: str
    cost_hex: str


@dataclass(frozen=True)
class BeliefPayload:
    schema: str
    support: SupportPayload
    posterior_weights_hex: Tuple[str, ...]
    deliberation_time_hex: str
    history: Tuple[HistoryStepPayload, ...]
    original_belief_hash: str
    payload_hash: str


@dataclass(frozen=True)
class MigrationRecord:
    schema: str
    case: OriginalCaseDescriptor
    belief: BeliefPayload
    original_belief_hash_matches_payload: bool
    record_hash: str


@dataclass(frozen=True)
class BaseBeliefMigration:
    schema: str
    migration_status: str
    source_commit: str
    source_tree_hash: str
    original_manifest_path: str
    original_manifest_file_hash: str
    original_manifest_hash: str
    original_spec_hash: str
    original_case_hash: str
    original_case_descriptors: Tuple[OriginalCaseDescriptor, ...]
    source_hashes: Tuple[Tuple[str, str], ...]
    source_hashes_hash: str
    execution_approval_file_hash: str
    migration_tool_hashes: Tuple[Tuple[str, str], ...]
    migration_tool_hashes_hash: str
    runtime_identity: Tuple[Tuple[str, str], ...]
    runtime_identity_hash: str
    dependency_identity: Tuple[Tuple[str, str], ...]
    dependency_identity_hash: str
    records: Tuple[MigrationRecord, ...]
    records_hash: str
    output_hash: str

    @property
    def authoritative(self) -> bool:
        return self.migration_status == AUTHORITATIVE_STATUS


@dataclass(frozen=True)
class MigrationExecutionApproval:
    """External Reviewer trust root for one scheduled candidate migration."""

    schema: str
    approval_status: str
    source_commit: str
    source_tree_hash: str
    original_manifest_file_hash: str
    migration_tool_hashes: Tuple[Tuple[str, str], ...]
    runtime_identity: Tuple[Tuple[str, str], ...]
    dependency_identity: Tuple[Tuple[str, str], ...]
    allowed_untracked_hashes: Tuple[Tuple[str, str], ...]
    slots: int
    array_job: bool
    rerunnable: bool


def _case_descriptor(raw: Mapping[str, object]) -> OriginalCaseDescriptor:
    _require_exact_keys(raw, OriginalCaseDescriptor, "case descriptor")
    return OriginalCaseDescriptor(
        case_id=_expect_int(raw["case_id"], "case_id"),
        environment=_expect_str(raw["environment"], "environment"),
        environment_hash=_expect_str(raw["environment_hash"], "environment_hash"),
        belief_kind=_expect_str(raw["belief_kind"], "belief_kind"),
        belief_hash=_expect_str(raw["belief_hash"], "belief_hash"),
    )


def _require_exact_keys(
    raw: Mapping[str, object],
    record_type: type,
    context: str,
) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping")
    expected = {field.name for field in fields(record_type)}
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _expect_str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _expect_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _expect_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _expect_list(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _runtime_identity() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": "|".join(platform.python_build()),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "libc": "|".join(platform.libc_ver()),
        "byteorder": sys.byteorder,
    }


def _dependency_identity() -> Dict[str, str]:
    dependencies: Dict[str, str] = {}
    for package in AUTHORITATIVE_DEPENDENCY_KEYS:
        try:
            dependencies[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"authoritative dependency is not installed: {package}"
            ) from error
    return dependencies


def _parse_execution_approval(
    raw: Mapping[str, object],
) -> MigrationExecutionApproval:
    _require_exact_keys(raw, MigrationExecutionApproval, "execution approval")
    approval = MigrationExecutionApproval(
        schema=_expect_str(raw["schema"], "execution approval schema"),
        approval_status=_expect_str(
            raw["approval_status"], "execution approval status"
        ),
        source_commit=_expect_str(raw["source_commit"], "approved source commit"),
        source_tree_hash=_expect_str(
            raw["source_tree_hash"], "approved source tree hash"
        ),
        original_manifest_file_hash=_expect_str(
            raw["original_manifest_file_hash"], "approved original manifest hash"
        ),
        migration_tool_hashes=_parse_pairs(
            raw["migration_tool_hashes"], "approved migration_tool_hashes"
        ),
        runtime_identity=_parse_pairs(
            raw["runtime_identity"], "approved runtime_identity"
        ),
        dependency_identity=_parse_pairs(
            raw["dependency_identity"], "approved dependency_identity"
        ),
        allowed_untracked_hashes=_parse_pairs(
            raw["allowed_untracked_hashes"], "approved allowed_untracked_hashes"
        ),
        slots=_expect_int(raw["slots"], "approved slots"),
        array_job=_expect_bool(raw["array_job"], "approved array_job"),
        rerunnable=_expect_bool(raw["rerunnable"], "approved rerunnable"),
    )
    if (
        approval.schema != EXECUTION_APPROVAL_SCHEMA
        or approval.approval_status != EXECUTION_APPROVAL_STATUS
        or approval.source_commit != AUTHORITATIVE_COMMIT
        or approval.source_tree_hash != AUTHORITATIVE_TREE
        or approval.original_manifest_file_hash != ORIGINAL_MANIFEST_FILE_HASH
    ):
        raise RuntimeError("execution approval immutable provenance mismatch")
    if tuple(path for path, _ in approval.migration_tool_hashes) != MIGRATION_TOOL_PATHS:
        raise RuntimeError("execution approval tool paths mismatch")
    if any(not _is_sha256(value) for _, value in approval.migration_tool_hashes):
        raise RuntimeError("execution approval tool hash is malformed")
    if tuple(key for key, _ in approval.runtime_identity) != AUTHORITATIVE_RUNTIME_KEYS:
        raise RuntimeError("execution approval runtime fields mismatch")
    if tuple(key for key, _ in approval.dependency_identity) != AUTHORITATIVE_DEPENDENCY_KEYS:
        raise RuntimeError("execution approval dependency fields mismatch")
    if any(not value for _, value in approval.runtime_identity):
        raise RuntimeError("execution approval runtime contains an empty value")
    if any(not value or value == "not-installed" for _, value in approval.dependency_identity):
        raise RuntimeError("execution approval dependency is missing")
    if (
        tuple(path for path, _ in approval.allowed_untracked_hashes)
        != AUTHORITATIVE_UNTRACKED_PATHS
    ):
        raise RuntimeError("execution approval untracked allowlist mismatch")
    if any(not _is_sha256(value) for _, value in approval.allowed_untracked_hashes):
        raise RuntimeError("execution approval untracked hash is malformed")
    approved_tools = dict(approval.migration_tool_hashes)
    approved_untracked = dict(approval.allowed_untracked_hashes)
    if approved_untracked[AUTHORITATIVE_UNTRACKED_PATHS[0]] != ORIGINAL_MANIFEST_FILE_HASH:
        raise RuntimeError("execution approval original manifest hash mismatch")
    if (
        approved_untracked[AUTHORITATIVE_UNTRACKED_PATHS[1]]
        != approved_tools[MIGRATION_CLI_PATH]
        or approved_untracked[AUTHORITATIVE_UNTRACKED_PATHS[2]]
        != approved_tools[MIGRATION_MODULE_PATH]
    ):
        raise RuntimeError("execution approval staged-tool hash mismatch")
    if approval.slots != 1 or approval.array_job or approval.rerunnable:
        raise RuntimeError("execution approval is not a one-slot non-array non-rerunnable job")
    return approval


def load_execution_approval(
    path: Path,
    *,
    approved_file_hash: str,
) -> MigrationExecutionApproval:
    """Load an approval only when its byte hash is supplied outside the file."""

    if not _is_sha256(approved_file_hash):
        raise RuntimeError("execution approval has no external Reviewer-approved file hash")
    if not path.is_file() or _file_hash(path) != approved_file_hash:
        raise RuntimeError("execution approval file hash is not approved")
    return _parse_execution_approval(_load_json_without_duplicate_keys(path))


def support_payload_hash(payload: SupportPayload) -> str:
    return canonical_hash(replace(payload, payload_hash=""))


def make_support_payload(prior: FiniteSupportPrior) -> SupportPayload:
    payload = SupportPayload(
        schema=SUPPORT_SCHEMA,
        states=tuple(
            SupportAtomPayload(
                total_need_hex=float(atom.total_need).hex(),
                gap_fraction_hex=float(atom.gap_fraction).hex(),
                orientation=int(atom.orientation),
            )
            for atom in prior.states
        ),
        prior_weights_hex=tuple(float(weight).hex() for weight in prior.weights),
        support_hash=str(prior.support_hash),
        payload_hash="",
    )
    return replace(payload, payload_hash=support_payload_hash(payload))


def belief_payload_hash(payload: BeliefPayload) -> str:
    return canonical_hash(replace(payload, payload_hash=""))


def make_belief_payload(
    prior: FiniteSupportPrior,
    belief: FiniteSupportBeliefState,
    original_belief_hash: str,
) -> BeliefPayload:
    if tuple(prior.states) != tuple(belief.states):
        raise ValueError("belief support does not match the migration prior")
    history = []
    for step in belief.history:
        action_value = float(step["action"])
        if action_value == 1.0:
            action = "sample_1"
        elif action_value == 2.0:
            action = "sample_2"
        else:
            raise ValueError("migration history contains an unknown action")
        history.append(
            HistoryStepPayload(
                action_hex=(1.0 if action == "sample_1" else 2.0).hex(),
                observation_hex=float(step["observation"]).hex(),
                cost_hex=float(step["cost"]).hex(),
            )
        )
    payload = BeliefPayload(
        schema=BELIEF_SCHEMA,
        support=make_support_payload(prior),
        posterior_weights_hex=tuple(float(weight).hex() for weight in belief.weights),
        deliberation_time_hex=float(belief.deliberation_time).hex(),
        history=tuple(history),
        original_belief_hash=original_belief_hash,
        payload_hash="",
    )
    return replace(payload, payload_hash=belief_payload_hash(payload))


def reconstruct_exact_belief(
    payload: BeliefPayload,
) -> Tuple[FiniteSupportPrior, FiniteSupportBeliefState]:
    """Reconstruct the exact stored binary64 belief without the current generator."""

    if payload.schema != BELIEF_SCHEMA or belief_payload_hash(payload) != payload.payload_hash:
        raise ValueError("belief payload schema/hash mismatch")
    support = payload.support
    if support.schema != SUPPORT_SCHEMA or support_payload_hash(support) != support.payload_hash:
        raise ValueError("support payload schema/hash mismatch")
    states = tuple(
        FiniteSupportAtom(
            _decode_hex(atom.total_need_hex, "total_need_hex"),
            _decode_hex(atom.gap_fraction_hex, "gap_fraction_hex"),
            int(atom.orientation),
        )
        for atom in support.states
    )
    prior_weights = tuple(
        _decode_hex(value, "prior_weights_hex") for value in support.prior_weights_hex
    )
    posterior_weights = tuple(
        _decode_hex(value, "posterior_weights_hex")
        for value in payload.posterior_weights_hex
    )
    if len(states) != len(prior_weights) or len(states) != len(posterior_weights):
        raise ValueError("support and weight lengths differ")
    if any(weight <= 0.0 for weight in prior_weights):
        raise ValueError("prior weights must be strictly positive")
    if any(weight < 0.0 for weight in posterior_weights):
        raise ValueError("posterior weights must be nonnegative")
    if not math.isclose(math.fsum(prior_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("prior weights do not sum to one")
    if not math.isclose(math.fsum(posterior_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("posterior weights do not sum to one")
    prior = FiniteSupportPrior(states, prior_weights)
    object.__setattr__(prior, "weights", prior_weights)
    if prior.support_hash != support.support_hash:
        raise ValueError("support hash mismatch")
    history = []
    for step in payload.history:
        action = _decode_hex(step.action_hex, "action_hex")
        if action not in (1.0, 2.0):
            raise ValueError("history action is invalid")
        history.append(
            {
                "action": action,
                "observation": _decode_hex(step.observation_hex, "observation_hex"),
                "cost": _decode_hex(step.cost_hex, "cost_hex"),
            }
        )
    belief = FiniteSupportBeliefState(
        states,
        posterior_weights,
        deliberation_time=_decode_hex(
            payload.deliberation_time_hex,
            "deliberation_time_hex",
        ),
        history=history,
    )
    # The constructor defensively normalizes. Restore the audited stored binary64 vector.
    belief.weights = posterior_weights
    if make_belief_payload(prior, belief, payload.original_belief_hash) != payload:
        raise ValueError("belief payload does not round-trip exactly")
    return prior, belief


def migration_record_hash(record: MigrationRecord) -> str:
    return canonical_hash(replace(record, record_hash=""))


def migration_output_hash(value: Mapping[str, object] | BaseBeliefMigration) -> str:
    payload = _plain(value)
    payload.pop("output_hash", None)
    return canonical_hash(payload)


def migration_to_dict(artifact: BaseBeliefMigration) -> Dict[str, object]:
    return _plain(artifact)


def _make_record(
    case: OriginalCaseDescriptor,
    prior: FiniteSupportPrior,
    belief: FiniteSupportBeliefState,
) -> MigrationRecord:
    payload = make_belief_payload(prior, belief, case.belief_hash)
    record = MigrationRecord(
        schema=MIGRATION_RECORD_SCHEMA,
        case=case,
        belief=payload,
        original_belief_hash_matches_payload=(
            legacy_belief_hash(belief) == case.belief_hash
        ),
        record_hash="",
    )
    return replace(record, record_hash=migration_record_hash(record))


def _make_artifact(
    *,
    migration_status: str,
    source_commit: str,
    source_tree_hash: str,
    manifest: Mapping[str, object],
    records: Sequence[MigrationRecord],
    runtime_identity: Mapping[str, str],
    dependency_identity: Mapping[str, str],
    migration_tool_hashes: Mapping[str, str],
    execution_approval_file_hash: str,
) -> BaseBeliefMigration:
    source_hashes = tuple(
        sorted((str(key), str(value)) for key, value in manifest["source_hashes"].items())
    )
    migration_tool_pairs = tuple(
        sorted((str(path), str(value)) for path, value in migration_tool_hashes.items())
    )
    runtime_pairs = tuple(sorted((str(k), str(v)) for k, v in runtime_identity.items()))
    dependency_pairs = tuple(
        sorted((str(k), str(v)) for k, v in dependency_identity.items())
    )
    artifact = BaseBeliefMigration(
        schema=MIGRATION_SCHEMA,
        migration_status=migration_status,
        source_commit=source_commit,
        source_tree_hash=source_tree_hash,
        original_manifest_path=ORIGINAL_MANIFEST_PROVENANCE_PATH,
        original_manifest_file_hash=ORIGINAL_MANIFEST_FILE_HASH,
        original_manifest_hash=ORIGINAL_MANIFEST_HASH,
        original_spec_hash=ORIGINAL_SPEC_HASH,
        original_case_hash=ORIGINAL_CASE_HASH,
        original_case_descriptors=tuple(
            _case_descriptor(case) for case in manifest["numerical_cases"]
        ),
        source_hashes=source_hashes,
        source_hashes_hash=canonical_hash(source_hashes),
        execution_approval_file_hash=execution_approval_file_hash,
        migration_tool_hashes=migration_tool_pairs,
        migration_tool_hashes_hash=canonical_hash(migration_tool_pairs),
        runtime_identity=runtime_pairs,
        runtime_identity_hash=canonical_hash(runtime_pairs),
        dependency_identity=dependency_pairs,
        dependency_identity_hash=canonical_hash(dependency_pairs),
        records=tuple(records),
        records_hash=canonical_hash(tuple(record.record_hash for record in records)),
        output_hash="",
    )
    return replace(artifact, output_hash=migration_output_hash(artifact))


def _git_quiet(project_root: Path, *args: str) -> bool:
    return subprocess.run(
        ("git",) + args,
        cwd=project_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _require_clean_tracked_worktree(project_root: Path = PROJECT_ROOT) -> None:
    if not _git_quiet(project_root, "diff", "--quiet", "HEAD", "--"):
        raise RuntimeError("authoritative tracked worktree differs from HEAD")
    if not _git_quiet(project_root, "diff", "--cached", "--quiet", "HEAD", "--"):
        raise RuntimeError("authoritative tracked index differs from HEAD")


def _untracked_inventory(project_root: Path = PROJECT_ROOT) -> Tuple[str, ...]:
    output = subprocess.run(
        ("git", "ls-files", "--others", "--"),
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return tuple(sorted(line for line in output.splitlines() if line))


def _require_frozen_untracked_inputs(
    approved_hashes: Mapping[str, str],
    project_root: Path = PROJECT_ROOT,
) -> None:
    expected = tuple(sorted(approved_hashes))
    actual = _untracked_inventory(project_root)
    if actual != expected:
        raise RuntimeError(
            f"authoritative untracked inventory mismatch: expected={expected}, actual={actual}"
        )
    for relative, expected_hash in sorted(approved_hashes.items()):
        path = project_root / relative
        if not path.is_file() or _file_hash(path) != expected_hash:
            raise RuntimeError(f"authoritative untracked input mismatch: {relative}")


def validate_authoritative_export_context(
    manifest: Mapping[str, object],
    approval: MigrationExecutionApproval,
) -> None:
    """Require the exact original tree plus a frozen three-file staging inventory."""

    if _git_output("rev-parse", "HEAD") != AUTHORITATIVE_COMMIT:
        raise RuntimeError("migration exporter must run from authoritative commit 7376c5d")
    if _git_output("rev-parse", "HEAD^{tree}") != AUTHORITATIVE_TREE:
        raise RuntimeError("migration exporter authoritative tree mismatch")
    _require_clean_tracked_worktree(PROJECT_ROOT)
    _require_frozen_untracked_inputs(dict(approval.allowed_untracked_hashes), PROJECT_ROOT)
    for relative, expected_hash in manifest["source_hashes"].items():
        path = PROJECT_ROOT / str(relative)
        if not path.is_file() or _file_hash(path) != expected_hash:
            raise RuntimeError(f"authoritative source mismatch: {relative}")
    if _file_hash(DEFAULT_SPEC_PATH) != ORIGINAL_SPEC_HASH:
        raise RuntimeError("authoritative migration spec mismatch")


def export_authoritative_base_migration(
    output_path: Path,
    manifest_path: Path = AUTHORITATIVE_STAGED_MANIFEST_PATH,
    *,
    execution_approval_path: Path,
    approved_execution_approval_file_hash: str,
    scheduled_job_script_path: Path,
) -> str:
    """Create one immutable migration output; never overwrite an existing path."""

    if output_path.exists():
        raise FileExistsError(f"migration output already exists: {output_path}")
    if manifest_path.resolve() != AUTHORITATIVE_STAGED_MANIFEST_PATH.resolve():
        raise RuntimeError("authoritative manifest must use its frozen staged path")
    manifest = _load_original_manifest(manifest_path)
    approval = load_execution_approval(
        execution_approval_path,
        approved_file_hash=approved_execution_approval_file_hash,
    )
    actual_tool_hashes = {
        MIGRATION_CLI_PATH: _file_hash(PROJECT_ROOT / MIGRATION_CLI_PATH),
        MIGRATION_MODULE_PATH: _file_hash(PROJECT_ROOT / MIGRATION_MODULE_PATH),
        MIGRATION_JOB_PATH: _file_hash(scheduled_job_script_path),
    }
    if tuple(sorted(actual_tool_hashes.items())) != approval.migration_tool_hashes:
        raise RuntimeError("executed migration tools differ from Reviewer approval")
    runtime = _runtime_identity()
    dependencies = _dependency_identity()
    if tuple(sorted(runtime.items())) != approval.runtime_identity:
        raise RuntimeError("executed runtime differs from Reviewer approval")
    if tuple(sorted(dependencies.items())) != approval.dependency_identity:
        raise RuntimeError("executed dependencies differ from Reviewer approval")
    validate_authoritative_export_context(manifest, approval)
    spec = load_positive_need_spec(DEFAULT_SPEC_PATH)
    environments = {
        environment.name: environment
        for environment in build_development_environments(spec)
    }
    records = []
    for raw_case in manifest["numerical_cases"]:
        case = _case_descriptor(raw_case)
        environment = environments[case.environment]
        belief = _numerical_belief(environment, case.belief_kind)
        record = _make_record(case, environment.prior, belief)
        if not record.original_belief_hash_matches_payload:
            raise RuntimeError(
                f"authoritative platform belief mismatch for case {case.case_id}"
            )
        records.append(record)
    artifact = _make_artifact(
        migration_status=AUTHORITATIVE_STATUS,
        source_commit=AUTHORITATIVE_COMMIT,
        source_tree_hash=AUTHORITATIVE_TREE,
        manifest=manifest,
        records=records,
        runtime_identity=runtime,
        dependency_identity=dependencies,
        migration_tool_hashes=actual_tool_hashes,
        execution_approval_file_hash=approved_execution_approval_file_hash,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(_plain(artifact), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return artifact.output_hash


def _parse_support(raw: Mapping[str, object]) -> SupportPayload:
    _require_exact_keys(raw, SupportPayload, "support payload")
    return SupportPayload(
        schema=_expect_str(raw["schema"], "support schema"),
        states=tuple(
            _parse_support_atom(atom)
            for atom in _expect_list(raw["states"], "support states")
        ),
        prior_weights_hex=tuple(
            _expect_str(value, "prior weight")
            for value in _expect_list(raw["prior_weights_hex"], "prior weights")
        ),
        support_hash=_expect_str(raw["support_hash"], "support hash"),
        payload_hash=_expect_str(raw["payload_hash"], "support payload hash"),
    )


def _parse_support_atom(raw: Mapping[str, object]) -> SupportAtomPayload:
    _require_exact_keys(raw, SupportAtomPayload, "support atom")
    return SupportAtomPayload(
        total_need_hex=_expect_str(raw["total_need_hex"], "total_need_hex"),
        gap_fraction_hex=_expect_str(raw["gap_fraction_hex"], "gap_fraction_hex"),
        orientation=_expect_int(raw["orientation"], "support orientation"),
    )


def _parse_belief(raw: Mapping[str, object]) -> BeliefPayload:
    _require_exact_keys(raw, BeliefPayload, "belief payload")
    return BeliefPayload(
        schema=_expect_str(raw["schema"], "belief schema"),
        support=_parse_support(raw["support"]),
        posterior_weights_hex=tuple(
            _expect_str(value, "posterior weight")
            for value in _expect_list(
                raw["posterior_weights_hex"], "posterior weights"
            )
        ),
        deliberation_time_hex=_expect_str(
            raw["deliberation_time_hex"], "deliberation_time_hex"
        ),
        history=tuple(
            _parse_history_step(step)
            for step in _expect_list(raw["history"], "belief history")
        ),
        original_belief_hash=_expect_str(
            raw["original_belief_hash"], "original belief hash"
        ),
        payload_hash=_expect_str(raw["payload_hash"], "belief payload hash"),
    )


def _parse_history_step(raw: Mapping[str, object]) -> HistoryStepPayload:
    _require_exact_keys(raw, HistoryStepPayload, "history step")
    return HistoryStepPayload(
        action_hex=_expect_str(raw["action_hex"], "action_hex"),
        observation_hex=_expect_str(raw["observation_hex"], "observation_hex"),
        cost_hex=_expect_str(raw["cost_hex"], "cost_hex"),
    )


def _parse_pairs(raw: object, context: str) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a list of pairs")
    pairs = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{context} contains a malformed pair")
        key, value = item
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{context} keys and values must be strings")
        pairs.append((key, value))
    result = tuple(pairs)
    if tuple(sorted(result)) != result or len(dict(result)) != len(result):
        raise ValueError(f"{context} must be uniquely keyed and sorted")
    return result


def parse_migration(raw: Mapping[str, object]) -> BaseBeliefMigration:
    _require_exact_keys(raw, BaseBeliefMigration, "migration artifact")
    records = tuple(
        _parse_record(record)
        for record in _expect_list(raw["records"], "migration records")
    )
    return BaseBeliefMigration(
        schema=_expect_str(raw["schema"], "migration schema"),
        migration_status=_expect_str(raw["migration_status"], "migration status"),
        source_commit=_expect_str(raw["source_commit"], "source commit"),
        source_tree_hash=_expect_str(raw["source_tree_hash"], "source tree hash"),
        original_manifest_path=_expect_str(
            raw["original_manifest_path"], "original manifest path"
        ),
        original_manifest_file_hash=_expect_str(
            raw["original_manifest_file_hash"], "original manifest file hash"
        ),
        original_manifest_hash=_expect_str(
            raw["original_manifest_hash"], "original manifest hash"
        ),
        original_spec_hash=_expect_str(raw["original_spec_hash"], "original spec hash"),
        original_case_hash=_expect_str(raw["original_case_hash"], "original case hash"),
        original_case_descriptors=tuple(
            _case_descriptor(case)
            for case in _expect_list(
                raw["original_case_descriptors"], "original case descriptors"
            )
        ),
        source_hashes=_parse_pairs(raw["source_hashes"], "source_hashes"),
        source_hashes_hash=_expect_str(raw["source_hashes_hash"], "source hashes hash"),
        execution_approval_file_hash=_expect_str(
            raw["execution_approval_file_hash"], "execution approval file hash"
        ),
        migration_tool_hashes=_parse_pairs(
            raw["migration_tool_hashes"], "migration_tool_hashes"
        ),
        migration_tool_hashes_hash=_expect_str(
            raw["migration_tool_hashes_hash"], "migration tool hashes hash"
        ),
        runtime_identity=_parse_pairs(raw["runtime_identity"], "runtime_identity"),
        runtime_identity_hash=_expect_str(
            raw["runtime_identity_hash"], "runtime identity hash"
        ),
        dependency_identity=_parse_pairs(
            raw["dependency_identity"], "dependency_identity"
        ),
        dependency_identity_hash=_expect_str(
            raw["dependency_identity_hash"], "dependency identity hash"
        ),
        records=records,
        records_hash=_expect_str(raw["records_hash"], "records hash"),
        output_hash=_expect_str(raw["output_hash"], "output hash"),
    )


def _parse_record(raw: Mapping[str, object]) -> MigrationRecord:
    _require_exact_keys(raw, MigrationRecord, "migration record")
    match = raw["original_belief_hash_matches_payload"]
    if not isinstance(match, bool):
        raise ValueError("original belief match flag must be boolean")
    return MigrationRecord(
        schema=_expect_str(raw["schema"], "migration record schema"),
        case=_case_descriptor(raw["case"]),
        belief=_parse_belief(raw["belief"]),
        original_belief_hash_matches_payload=match,
        record_hash=_expect_str(raw["record_hash"], "migration record hash"),
    )


def validate_migration(
    artifact: BaseBeliefMigration,
    *,
    approved_output_hash: Optional[str],
    allow_synthetic_fixture: bool = False,
    execution_approval: Optional[MigrationExecutionApproval] = None,
    approved_execution_approval_file_hash: Optional[str] = None,
) -> None:
    """Validate provenance and every exact payload before exposing migrated beliefs."""

    if not _is_sha256(approved_output_hash):
        raise RuntimeError("migration has no Reviewer-approved output hash")
    if artifact.output_hash != approved_output_hash:
        raise RuntimeError("migration output hash is not approved")
    if migration_output_hash(artifact) != artifact.output_hash:
        raise RuntimeError("migration output self-hash mismatch")
    if artifact.schema != MIGRATION_SCHEMA:
        raise RuntimeError("migration schema mismatch")
    authoritative = artifact.migration_status == AUTHORITATIVE_STATUS
    synthetic = artifact.migration_status == SYNTHETIC_STATUS
    if not authoritative and not (synthetic and allow_synthetic_fixture):
        raise RuntimeError("migration provenance is not authoritative")
    if authoritative and (
        artifact.source_commit != AUTHORITATIVE_COMMIT
        or artifact.source_tree_hash != AUTHORITATIVE_TREE
    ):
        raise RuntimeError("migration source commit/tree mismatch")
    if synthetic and (
        artifact.source_commit != "synthetic-test-only"
        or artifact.source_tree_hash != "synthetic-test-only"
    ):
        raise RuntimeError("synthetic fixture provenance is ambiguous")
    if authoritative and (
        execution_approval is None
        or not _is_sha256(approved_execution_approval_file_hash)
    ):
        raise RuntimeError("authoritative migration lacks an external execution approval")
    if execution_approval is not None:
        _parse_execution_approval(_plain(execution_approval))
        if artifact.execution_approval_file_hash != approved_execution_approval_file_hash:
            raise RuntimeError("migration execution approval hash is not approved")
        if artifact.migration_tool_hashes != execution_approval.migration_tool_hashes:
            raise RuntimeError("migration tool identity differs from execution approval")
        if artifact.runtime_identity != execution_approval.runtime_identity:
            raise RuntimeError("migration runtime identity differs from execution approval")
        if artifact.dependency_identity != execution_approval.dependency_identity:
            raise RuntimeError("migration dependency identity differs from execution approval")
    if (
        artifact.original_manifest_path != ORIGINAL_MANIFEST_PROVENANCE_PATH
        or artifact.original_manifest_file_hash != ORIGINAL_MANIFEST_FILE_HASH
        or artifact.original_manifest_hash != ORIGINAL_MANIFEST_HASH
        or artifact.original_spec_hash != ORIGINAL_SPEC_HASH
        or artifact.original_case_hash != ORIGINAL_CASE_HASH
    ):
        raise RuntimeError("migration original provenance mismatch")
    manifest = _load_original_manifest(DEFAULT_ORIGINAL_MANIFEST_PATH)
    if _file_hash(DEFAULT_SPEC_PATH) != ORIGINAL_SPEC_HASH:
        raise RuntimeError("trusted validation spec file hash mismatch")
    spec = load_positive_need_spec(DEFAULT_SPEC_PATH)
    environments = {
        environment.name: environment
        for environment in build_development_environments(spec)
    }
    expected_cases = tuple(
        _case_descriptor(case) for case in manifest["numerical_cases"]
    )
    cases = artifact.original_case_descriptors
    if cases != expected_cases:
        raise RuntimeError("migration case descriptors differ from the original manifest")
    if len(cases) != 90 or [case.case_id for case in cases] != list(range(90)):
        raise RuntimeError("migration case IDs must be exactly 0..89")
    if _legacy_hash([_plain(case) for case in cases]) != ORIGINAL_CASE_HASH:
        raise RuntimeError("migration case descriptors differ from the original manifest")
    if len(artifact.records) != 90:
        raise RuntimeError("migration must contain exactly 90 records")
    if not artifact.source_hashes or not artifact.runtime_identity or not artifact.dependency_identity:
        raise RuntimeError("migration source/runtime/dependency identity is incomplete")
    expected_source_hashes = tuple(
        sorted(
            (str(key), str(value))
            for key, value in manifest["source_hashes"].items()
        )
    )
    if artifact.source_hashes != expected_source_hashes:
        raise RuntimeError("migration source hashes differ from the original manifest")
    if (
        canonical_hash(artifact.source_hashes) != artifact.source_hashes_hash
        or not _is_sha256(artifact.source_hashes_hash)
    ):
        raise RuntimeError("migration source-hash aggregate mismatch")
    if tuple(path for path, _ in artifact.migration_tool_hashes) != MIGRATION_TOOL_PATHS:
        raise RuntimeError("migration tool source paths mismatch")
    if any(not _is_sha256(value) for _, value in artifact.migration_tool_hashes):
        raise RuntimeError("migration tool source hash is malformed")
    if (
        canonical_hash(artifact.migration_tool_hashes)
        != artifact.migration_tool_hashes_hash
        or not _is_sha256(artifact.migration_tool_hashes_hash)
    ):
        raise RuntimeError("migration tool-source aggregate hash mismatch")
    if (
        canonical_hash(artifact.runtime_identity) != artifact.runtime_identity_hash
        or not _is_sha256(artifact.runtime_identity_hash)
    ):
        raise RuntimeError("migration runtime identity hash mismatch")
    if (
        canonical_hash(artifact.dependency_identity)
        != artifact.dependency_identity_hash
        or not _is_sha256(artifact.dependency_identity_hash)
    ):
        raise RuntimeError("migration dependency identity hash mismatch")
    if authoritative:
        if tuple(key for key, _ in artifact.runtime_identity) != AUTHORITATIVE_RUNTIME_KEYS:
            raise RuntimeError("authoritative runtime identity fields mismatch")
        if tuple(key for key, _ in artifact.dependency_identity) != AUTHORITATIVE_DEPENDENCY_KEYS:
            raise RuntimeError("authoritative dependency identity fields mismatch")
        if any(not value for _, value in artifact.runtime_identity):
            raise RuntimeError("authoritative runtime identity contains an empty value")
        if any(not value for _, value in artifact.dependency_identity):
            raise RuntimeError("authoritative dependency identity contains an empty value")
    if any(not _is_sha256(value) for _, value in artifact.source_hashes):
        raise RuntimeError("migration source hash is malformed")
    if artifact.records_hash != canonical_hash(
        tuple(record.record_hash for record in artifact.records)
    ):
        raise RuntimeError("migration record aggregate hash mismatch")
    if not _is_sha256(artifact.records_hash):
        raise RuntimeError("migration record aggregate hash is malformed")
    for case, record in zip(cases, artifact.records):
        if record.schema != MIGRATION_RECORD_SCHEMA or record.case != case:
            raise RuntimeError("migration record/case mismatch")
        if migration_record_hash(record) != record.record_hash:
            raise RuntimeError("migration record hash mismatch")
        if not _is_sha256(record.record_hash):
            raise RuntimeError("migration record hash is malformed")
        if record.belief.original_belief_hash != case.belief_hash:
            raise RuntimeError("migration belief payload is bound to the wrong case hash")
        environment = environments.get(case.environment)
        if environment is None:
            raise RuntimeError("migration case refers to an unknown frozen environment")
        if environment.environment_hash != case.environment_hash:
            raise RuntimeError("migration case environment hash mismatch")
        expected_support = make_support_payload(environment.prior)
        if record.belief.support != expected_support:
            raise RuntimeError("migration belief support differs from frozen environment")
        _, belief = reconstruct_exact_belief(record.belief)
        actual_match = legacy_belief_hash(belief) == case.belief_hash
        if record.original_belief_hash_matches_payload != actual_match:
            raise RuntimeError("migration original-belief match evidence is inconsistent")
        if authoritative and not actual_match:
            raise RuntimeError("authoritative migration belief differs from original case")


def load_migration(
    path: Path = DEFAULT_MIGRATION_PATH,
    *,
    approved_output_hash: Optional[str],
    allow_synthetic_fixture: bool = False,
    execution_approval: Optional[MigrationExecutionApproval] = None,
    approved_execution_approval_file_hash: Optional[str] = None,
) -> BaseBeliefMigration:
    if not path.is_file():
        raise FileNotFoundError(f"migration artifact is absent: {path}")
    artifact = parse_migration(_load_json_without_duplicate_keys(path))
    validate_migration(
        artifact,
        approved_output_hash=approved_output_hash,
        allow_synthetic_fixture=allow_synthetic_fixture,
        execution_approval=execution_approval,
        approved_execution_approval_file_hash=approved_execution_approval_file_hash,
    )
    return artifact


def _make_execution_approval(
    *,
    migration_tool_hashes: Mapping[str, str],
    runtime_identity: Mapping[str, str],
    dependency_identity: Mapping[str, str],
    allowed_untracked_hashes: Mapping[str, str],
) -> MigrationExecutionApproval:
    return _parse_execution_approval(
        {
            "schema": EXECUTION_APPROVAL_SCHEMA,
            "approval_status": EXECUTION_APPROVAL_STATUS,
            "source_commit": AUTHORITATIVE_COMMIT,
            "source_tree_hash": AUTHORITATIVE_TREE,
            "original_manifest_file_hash": ORIGINAL_MANIFEST_FILE_HASH,
            "migration_tool_hashes": [
                list(item) for item in sorted(migration_tool_hashes.items())
            ],
            "runtime_identity": [
                list(item) for item in sorted(runtime_identity.items())
            ],
            "dependency_identity": [
                list(item) for item in sorted(dependency_identity.items())
            ],
            "allowed_untracked_hashes": [
                list(item) for item in sorted(allowed_untracked_hashes.items())
            ],
            "slots": 1,
            "array_job": False,
            "rerunnable": False,
        }
    )


def synthetic_execution_approval_for_tests(
    artifact: BaseBeliefMigration,
) -> Tuple[MigrationExecutionApproval, str]:
    """Reconstruct the fixed external profile used by the synthetic test fixture."""

    allowed = {
        AUTHORITATIVE_UNTRACKED_PATHS[0]: ORIGINAL_MANIFEST_FILE_HASH,
        AUTHORITATIVE_UNTRACKED_PATHS[1]: dict(artifact.migration_tool_hashes)[
            AUTHORITATIVE_UNTRACKED_PATHS[1]
        ],
        AUTHORITATIVE_UNTRACKED_PATHS[2]: dict(artifact.migration_tool_hashes)[
            AUTHORITATIVE_UNTRACKED_PATHS[2]
        ],
    }
    approval = _make_execution_approval(
        migration_tool_hashes=dict(artifact.migration_tool_hashes),
        runtime_identity=dict(artifact.runtime_identity),
        dependency_identity=dict(artifact.dependency_identity),
        allowed_untracked_hashes=allowed,
    )
    return approval, canonical_hash(_plain(approval))


def build_synthetic_fixture_for_tests(
    manifest_path: Path = DEFAULT_ORIGINAL_MANIFEST_PATH,
) -> BaseBeliefMigration:
    """Create an in-memory fixture permanently labeled as non-authoritative."""

    manifest = _load_original_manifest(manifest_path)
    spec = load_positive_need_spec(DEFAULT_SPEC_PATH)
    environments = {
        environment.name: environment
        for environment in build_development_environments(spec)
    }
    records = []
    for raw_case in manifest["numerical_cases"]:
        case = _case_descriptor(raw_case)
        environment = environments[case.environment]
        belief = _numerical_belief(environment, case.belief_kind)
        records.append(_make_record(case, environment.prior, belief))
    runtime = {key: "synthetic-test-only" for key in AUTHORITATIVE_RUNTIME_KEYS}
    dependencies = {
        key: "synthetic-test-only" for key in AUTHORITATIVE_DEPENDENCY_KEYS
    }
    tool_hashes = {
        relative: _file_hash(PROJECT_ROOT / relative)
        for relative in MIGRATION_TOOL_PATHS
    }
    allowed = {
        AUTHORITATIVE_UNTRACKED_PATHS[0]: ORIGINAL_MANIFEST_FILE_HASH,
        AUTHORITATIVE_UNTRACKED_PATHS[1]: tool_hashes[AUTHORITATIVE_UNTRACKED_PATHS[1]],
        AUTHORITATIVE_UNTRACKED_PATHS[2]: tool_hashes[AUTHORITATIVE_UNTRACKED_PATHS[2]],
    }
    approval = _make_execution_approval(
        migration_tool_hashes=tool_hashes,
        runtime_identity=runtime,
        dependency_identity=dependencies,
        allowed_untracked_hashes=allowed,
    )
    approval_hash = canonical_hash(_plain(approval))
    artifact = _make_artifact(
        migration_status=SYNTHETIC_STATUS,
        source_commit="synthetic-test-only",
        source_tree_hash="synthetic-test-only",
        manifest=manifest,
        records=records,
        runtime_identity=runtime,
        dependency_identity=dependencies,
        migration_tool_hashes=tool_hashes,
        execution_approval_file_hash=approval_hash,
    )
    validate_migration(
        artifact,
        approved_output_hash=artifact.output_hash,
        allow_synthetic_fixture=True,
        execution_approval=approval,
        approved_execution_approval_file_hash=approval_hash,
    )
    return artifact


__all__ = [
    "AUTHORITATIVE_COMMIT",
    "AUTHORITATIVE_STATUS",
    "AUTHORITATIVE_UNTRACKED_PATHS",
    "AUTHORITATIVE_STAGED_MANIFEST_PATH",
    "BaseBeliefMigration",
    "BeliefPayload",
    "DEFAULT_MIGRATION_PATH",
    "DEFAULT_ORIGINAL_MANIFEST_PATH",
    "ORIGINAL_MANIFEST_PROVENANCE_PATH",
    "HistoryStepPayload",
    "MIGRATION_SCHEMA",
    "MIGRATION_TOOL_PATHS",
    "MigrationExecutionApproval",
    "MigrationRecord",
    "ORIGINAL_CASE_HASH",
    "ORIGINAL_MANIFEST_HASH",
    "ORIGINAL_SPEC_HASH",
    "OriginalCaseDescriptor",
    "SUPPORT_SCHEMA",
    "SYNTHETIC_STATUS",
    "SupportAtomPayload",
    "SupportPayload",
    "belief_payload_hash",
    "build_synthetic_fixture_for_tests",
    "canonical_hash",
    "export_authoritative_base_migration",
    "load_execution_approval",
    "load_migration",
    "make_belief_payload",
    "migration_record_hash",
    "migration_output_hash",
    "migration_to_dict",
    "parse_migration",
    "reconstruct_exact_belief",
    "support_payload_hash",
    "synthetic_execution_approval_for_tests",
    "validate_authoritative_export_context",
    "validate_migration",
]
