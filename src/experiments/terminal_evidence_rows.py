from __future__ import annotations

"""Source-backed terminal rows and complete compressed certificate sidecars.

Rows and sidecars are never trust roots.  The immutable validated suite, accepted base
provider, scientific identity, numerical identity, and deterministic solver recomputation
are the trust roots used by the collector.
"""

from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import gzip
import hashlib
import json
import math
import time
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..mdp.finite_support import FiniteSupportMetaMDP
from ..solvers.terminal import (
    PRODUCTION_VALUE_TOLERANCE,
    StructuralSymmetry,
    TerminalOptimizationResult,
    optimal_terminal_results_for_weight_rows,
    optimize_terminal_allocation,
    optimize_terminal_allocation_with_trace,
    production_terminal_numerical_method_config_hash,
    validate_structural_symmetry_proof,
)
from ..solvers.terminal_reference import (
    CandidateIsolationEvidence,
    TerminalReferenceRecord,
    solve_terminal_reference_a,
    solve_terminal_reference_a_with_trace,
    source_validate_terminal_reference_record,
    terminal_belief_identity_hash,
    terminal_reference_a_numerical_method_config_hash,
    terminal_reference_certificate_hash,
    terminal_reference_cross_process_proof_matches,
    terminal_scientific_spec_hash,
    validate_production_against_reference_a,
)
from ..solvers.terminal_reference_agreement import (
    TERMINAL_PRODUCTION_ALLOCATION_TOLERANCE,
    TERMINAL_PRODUCTION_REGRET_TOLERANCE,
    TERMINAL_PRODUCTION_VALUE_TOLERANCE,
    TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE,
    TerminalReferenceAgreementRecord,
    terminal_reference_agreement_certificate_hash,
    terminal_reference_agreement_numerical_method_config_hash,
    validate_terminal_reference_agreement,
)
from ..solvers.terminal_reference_b import (
    solve_terminal_reference_b_with_trace,
    source_validate_terminal_reference_b_record,
    terminal_reference_b_numerical_method_config_hash,
)
from .r6_prefeedback_positive_need import (
    build_development_environments,
)
from .terminal_reference_b_process import solve_terminal_reference_b_concurrently
from .terminal_validation_suite import (
    CanonicalBaseProvider,
    TerminalHistoryStep,
    TerminalValidationDescriptor,
    TerminalValidationIdentities,
    TerminalValidationSuite,
    build_local_diagnostic_base_provider,
    canonical_hash,
    load_frozen_r6_spec,
    load_terminal_validation_identities,
    reconstruct_canonical_base_record,
    terminal_scientific_spec_hash as terminal_suite_scientific_spec_hash,
    terminal_validation_descriptor_hash,
    validate_terminal_validation_suite,
)


TERMINAL_EVIDENCE_ROW_SCHEMA = "terminal_evidence_row_v1"
TERMINAL_SIDECAR_SCHEMA = "terminal_certificate_sidecar_v1"
TERMINAL_SIDECAR_SCHEMA_VERSION = 1
TERMINAL_COLLECTION_SCHEMA = "terminal_evidence_collection_v2"
TERMINAL_METHOD_ORDER = (
    "production_terminal",
    "reference_a",
    "reference_b",
    "agreement",
)
REFERENCE_B_NEAR_TIE_SEPARATION = 1e-6
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class TerminalSidecarReference:
    relative_path: str
    sha256: str
    byte_count: int
    schema_version: int
    logical_record_hash: str


@dataclass(frozen=True)
class TerminalEvidenceRow:
    schema: str
    suite_class: str
    suite_version: str
    descriptor_index: int
    descriptor_hash: str
    source_case_id: Optional[int]
    belief_hash: str
    suite_history_hash: str
    posterior_weight_hash: str
    scientific_spec_hash: str
    numerical_method_config_hash: str
    method: str
    method_numerical_hash: str
    status: str
    global_value_interval: Optional[Tuple[float, float]]
    candidate_allocation_intervals: Tuple[Tuple[float, float], ...]
    candidate_value_intervals: Tuple[Tuple[float, float], ...]
    canonical_allocation_interval: Optional[Tuple[float, float]]
    production_allocation: Optional[float]
    production_value_interval: Optional[Tuple[float, float]]
    production_regret_interval: Optional[Tuple[float, float]]
    symmetry_proof_id: Optional[str]
    symmetry_required: bool
    symmetry_pass: bool
    tie_status: Optional[str]
    precision_level: Optional[float]
    evaluation_count: int
    scalar_batch_required: bool
    scalar_batch_pass: bool
    reference_b_required: bool
    reference_b_trigger_reasons: Tuple[str, ...]
    validation_checks: Tuple[Tuple[str, bool], ...]
    pass_status: bool
    unresolved: bool
    failure_reasons: Tuple[str, ...]
    sidecar: TerminalSidecarReference
    logical_record_hash: str


@dataclass(frozen=True)
class TerminalEvidenceBundle:
    descriptor_hash: str
    rows: Tuple[TerminalEvidenceRow, ...]
    sidecars: Tuple[Tuple[str, bytes], ...]


@dataclass(frozen=True)
class TerminalEvidencePlan:
    """Manifest metadata projected from terminal evidence rows."""

    expected_methods: Tuple[str, ...]
    expected_tie_row_count: int
    expected_symmetry_row_count: int

    def as_tuple(self) -> Tuple[Tuple[str, ...], int, int]:
        return (
            self.expected_methods,
            self.expected_tie_row_count,
            self.expected_symmetry_row_count,
        )


@dataclass(frozen=True)
class TerminalEvidenceCollectionSummary:
    schema: str
    suite_manifest_hash: str
    suite_ordered_descriptor_hash: str
    suite_validation_status: str
    base_provider_hash: str
    authoritative_source_accepted: bool
    scientific_spec_hash: str
    numerical_method_config_hash: str
    expected_descriptor_count: int
    observed_descriptor_count: int
    observed_row_count: int
    observed_sidecar_count: int
    missing_row_keys: Tuple[str, ...]
    duplicate_row_keys: Tuple[str, ...]
    unexpected_row_keys: Tuple[str, ...]
    invalid_row_keys: Tuple[str, ...]
    invalid_sidecar_paths: Tuple[str, ...]
    failed_row_keys: Tuple[str, ...]
    unresolved_row_keys: Tuple[str, ...]
    reference_b_required_keys: Tuple[str, ...]
    reference_b_missing_keys: Tuple[str, ...]
    maximum_evaluation_count: int
    maximum_regret_upper: Optional[float]
    evidence_valid: bool
    stage_complete: bool
    candidate_pass: bool
    failure_reasons: Tuple[str, ...]
    logical_record_hash: str


@dataclass(frozen=True)
class DecodedTerminalSidecar:
    certificate: Any
    complete_trace: Mapping[str, Any]
    validation_evidence: Mapping[str, Any]
    payload: Mapping[str, Any]


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("terminal evidence contains a non-finite float")
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("terminal evidence mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported terminal evidence type: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _logical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _decode_canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise ValueError("raw JSON floats are forbidden in canonical evidence")
    if isinstance(value, list):
        return tuple(_decode_canonical(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"float_hex"}:
            token = value["float_hex"]
            if type(token) is not str:
                raise ValueError("float_hex must be a string")
            parsed = float.fromhex(token)
            if not math.isfinite(parsed) or parsed.hex() != token:
                raise ValueError("float_hex is nonfinite or noncanonical")
            return parsed
        if any(type(key) is not str for key in value):
            raise ValueError("canonical mapping keys must be strings")
        return {key: _decode_canonical(item) for key, item in value.items()}
    raise ValueError("unsupported canonical JSON value")


def _require_exact_keys(payload: Mapping[str, Any], expected: Sequence[str], context: str) -> None:
    if set(payload) != set(expected):
        raise ValueError(f"{context} fields differ from the exact schema")


def _structural_symmetry(payload: Mapping[str, Any]) -> StructuralSymmetry:
    names = tuple(field.name for field in fields(StructuralSymmetry))
    _require_exact_keys(payload, names, "structural symmetry")
    return StructuralSymmetry(**payload)


def _candidate_evidence(payload: Mapping[str, Any]) -> CandidateIsolationEvidence:
    names = tuple(field.name for field in fields(CandidateIsolationEvidence))
    _require_exact_keys(payload, names, "candidate isolation evidence")
    return CandidateIsolationEvidence(**payload)


def _decode_certificate(method: str, payload: Mapping[str, Any], certificate_type: str) -> Any:
    decoded = _decode_canonical(payload)
    if not isinstance(decoded, dict):
        raise ValueError("certificate payload must decode to a mapping")
    if method == "production_terminal":
        if certificate_type != "TerminalOptimizationResult":
            raise ValueError("production certificate type mismatch")
        names = tuple(field.name for field in fields(TerminalOptimizationResult))
        _require_exact_keys(decoded, names, "production certificate")
        decoded["structural_symmetry"] = _structural_symmetry(decoded["structural_symmetry"])
        return TerminalOptimizationResult(**decoded)
    if method in ("reference_a", "reference_b"):
        if certificate_type != "TerminalReferenceRecord":
            raise ValueError("reference certificate type mismatch")
        names = tuple(field.name for field in fields(TerminalReferenceRecord))
        _require_exact_keys(decoded, names, "terminal reference certificate")
        decoded["candidate_isolation_evidence"] = tuple(
            _candidate_evidence(item) for item in decoded["candidate_isolation_evidence"]
        )
        decoded["structural_symmetry"] = _structural_symmetry(decoded["structural_symmetry"])
        record = TerminalReferenceRecord(**decoded)
        if terminal_reference_certificate_hash(record) != record.certificate_hash:
            raise ValueError("terminal reference internal certificate hash mismatch")
        return record
    if method == "agreement":
        if certificate_type != "TerminalReferenceAgreementRecord":
            raise ValueError("agreement certificate type mismatch")
        names = tuple(field.name for field in fields(TerminalReferenceAgreementRecord))
        _require_exact_keys(decoded, names, "agreement certificate")
        record = TerminalReferenceAgreementRecord(**decoded)
        if terminal_reference_agreement_certificate_hash(record) != record.certificate_hash:
            raise ValueError("agreement internal certificate hash mismatch")
        return record
    raise ValueError("unknown terminal evidence method")


def _finite_interval(value: Any) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is float and math.isfinite(item) for item in value)
        and value[0] <= value[1]
    )


def _allocation_interval(value: Any) -> bool:
    return _finite_interval(value) and 0.0 <= value[0] <= value[1] <= 1.0


def _interval_distance(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0.0


def _point_interval_distance(point: float, interval: Tuple[float, float]) -> float:
    if point < interval[0]:
        return interval[0] - point
    if point > interval[1]:
        return point - interval[1]
    return 0.0


def _row_key_parts(row: TerminalEvidenceRow) -> Tuple[str, str, int, str]:
    return row.suite_class, row.suite_version, row.descriptor_index, row.method


def terminal_evidence_row_key(row: TerminalEvidenceRow) -> str:
    return "/".join(str(part) for part in _row_key_parts(row))


def _descriptor_key(descriptor: TerminalValidationDescriptor) -> Tuple[str, str, int]:
    return descriptor.suite_class, descriptor.suite_version, descriptor.descriptor_index


def _descriptor_key_text(descriptor: TerminalValidationDescriptor) -> str:
    return "/".join(str(part) for part in _descriptor_key(descriptor))


def terminal_evidence_row_hash(row: TerminalEvidenceRow) -> str:
    return _logical_hash(replace(row, logical_record_hash=""))


def _history_hash(belief: Any) -> str:
    history = tuple(
        TerminalHistoryStep(
            action="sample_1" if float(step["action"]) == 1.0 else "sample_2",
            observation=float(step["observation"]),
            cost=float(step["cost"]),
        )
        for step in belief.history
    )
    return canonical_hash(history)


def terminal_descriptor_source_failures(
    descriptor: TerminalValidationDescriptor,
    mdp: Any,
    belief: Any,
) -> Tuple[str, ...]:
    failures = []
    try:
        if descriptor.descriptor_hash != terminal_validation_descriptor_hash(descriptor):
            failures.append("descriptor_hash_mismatch")
        if descriptor.support_hash != mdp.prior.support_hash:
            failures.append("support_hash_mismatch")
        if descriptor.sigma_sample.hex() != float(mdp.config.sigma_sample).hex():
            failures.append("sigma_sample_mismatch")
        if descriptor.sample_time_cost.hex() != float(mdp.config.sample_time_cost).hex():
            failures.append("sample_time_cost_mismatch")
        if descriptor.deliberation_time.hex() != float(belief.deliberation_time).hex():
            failures.append("deliberation_time_mismatch")
        if descriptor.posterior_weight_hash != canonical_hash(tuple(belief.weights)):
            failures.append("posterior_weight_hash_mismatch")
        if descriptor.history_hash != _history_hash(belief):
            failures.append("history_hash_mismatch")
        expected_belief_hash = canonical_hash({
            "support_hash": mdp.prior.support_hash,
            "posterior_weights": tuple(belief.weights),
            "deliberation_time": belief.deliberation_time,
            "history_hash": descriptor.history_hash,
        })
        if descriptor.canonical_belief_hash != expected_belief_hash:
            failures.append("canonical_belief_hash_mismatch")
        if descriptor.scientific_spec_hash != terminal_suite_scientific_spec_hash():
            failures.append("scientific_spec_hash_mismatch")
    except (AttributeError, TypeError, ValueError, OverflowError):
        failures.append("descriptor_source_binding_error")
    return tuple(dict.fromkeys(failures))


def reconstruct_terminal_evidence_source(
    descriptor: TerminalValidationDescriptor,
    base_provider: CanonicalBaseProvider,
) -> Tuple[Any, Any]:
    environments = {
        environment.name: environment
        for environment in build_development_environments(load_frozen_r6_spec())
    }
    environment = environments.get(descriptor.environment)
    if environment is None or environment.environment_hash != descriptor.environment_hash:
        raise ValueError("descriptor environment cannot be reconstructed")
    if descriptor.suite_class in ("base", "one_step"):
        if descriptor.source_case_id is None or not 0 <= descriptor.source_case_id < len(base_provider.records):
            raise ValueError("descriptor source case is unavailable")
        record = base_provider.records[descriptor.source_case_id]
        prior, belief = reconstruct_canonical_base_record(record)
        mdp = FiniteSupportMetaMDP(environment.config, prior)
    elif descriptor.suite_class == "reachable_core":
        mdp = FiniteSupportMetaMDP(environment.config, environment.prior)
        belief = mdp.initial_belief()
    else:
        raise ValueError("unknown descriptor suite class")
    if descriptor.suite_class != "base":
        for step in descriptor.history:
            belief = mdp.posterior_transition(
                belief,
                step.action,
                float(step.observation),
                advance_time=True,
                record=True,
            )
    failures = terminal_descriptor_source_failures(descriptor, mdp, belief)
    if failures:
        raise ValueError("descriptor source reconstruction failed: " + ",".join(failures))
    return mdp, belief


def _candidate_separation_lower(reference_a: TerminalReferenceRecord) -> Optional[float]:
    if len(reference_a.candidate_value_intervals) < 2:
        return None
    ordered = sorted(
        reference_a.candidate_value_intervals,
        key=lambda interval: (interval[0], interval[1]),
        reverse=True,
    )
    return max(0.0, ordered[0][0] - ordered[1][1])


def terminal_reference_b_trigger_reasons(
    descriptor: TerminalValidationDescriptor,
    production: TerminalOptimizationResult,
    reference_a: TerminalReferenceRecord,
    *,
    production_reference_a_pass: bool,
    production_checks_pass: bool = True,
    reference_a_source_valid: bool = True,
) -> Tuple[str, ...]:
    """Derive Section 8.1 triggers only from source-validated records."""

    if not reference_a_source_valid:
        return ("reference_a_unresolved", "production_disagreement_or_failure")
    reasons = []
    if descriptor.suite_class == "base":
        reasons.append("all_base_beliefs")
    if descriptor.reference_b_prespecified:
        reasons.append("prespecified_first_component_z0")
    if production.structural_symmetry.valid:
        reasons.append("structural_symmetry")
    separation = _candidate_separation_lower(reference_a)
    if len(reference_a.candidate_allocation_intervals) > 1:
        reasons.append("reference_a_multiple_viable_candidates")
    if separation is not None and separation <= REFERENCE_B_NEAR_TIE_SEPARATION:
        reasons.append("reference_a_near_tie_separation")
    if reference_a.status != "resolved":
        reasons.append("reference_a_unresolved")
    if not production_reference_a_pass or not production_checks_pass:
        reasons.append("production_disagreement_or_failure")
    return tuple(reasons)


def _sidecar_path(descriptor: TerminalValidationDescriptor, method: str) -> str:
    return f"sidecars/{descriptor.suite_class}/{descriptor.descriptor_index:06d}/{method}.json.gz"


def build_terminal_certificate_sidecar(
    *,
    relative_path: str,
    descriptor: TerminalValidationDescriptor,
    method: str,
    method_numerical_hash: str,
    certificate: Any,
    complete_trace: Mapping[str, Any],
    validation_evidence: Mapping[str, Any],
) -> Tuple[TerminalSidecarReference, bytes]:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("sidecar path must be a safe relative .json.gz path")
    if not complete_trace.get("complete"):
        raise ValueError("sidecar requires an explicitly complete trace")
    certificate_payload = asdict(certificate)
    trace_payload = dict(complete_trace)
    payload = {
        "schema": TERMINAL_SIDECAR_SCHEMA,
        "schema_version": TERMINAL_SIDECAR_SCHEMA_VERSION,
        "suite_class": descriptor.suite_class,
        "suite_version": descriptor.suite_version,
        "descriptor_index": descriptor.descriptor_index,
        "descriptor_hash": descriptor.descriptor_hash,
        "scientific_spec_hash": descriptor.scientific_spec_hash,
        "numerical_method_config_hash": descriptor.numerical_method_config_hash,
        "method": method,
        "method_numerical_hash": method_numerical_hash,
        "certificate_type": type(certificate).__name__,
        "certificate_payload": certificate_payload,
        "certificate_logical_hash": _logical_hash(certificate_payload),
        "complete_trace_payload": trace_payload,
        "complete_trace_logical_hash": _logical_hash(trace_payload),
        "validation_evidence": dict(validation_evidence),
    }
    payload["logical_record_hash"] = _logical_hash(payload)
    compressed = gzip.compress(_canonical_json_bytes(payload), compresslevel=9, mtime=0)
    reference = TerminalSidecarReference(
        relative_path=str(path),
        sha256=hashlib.sha256(compressed).hexdigest(),
        byte_count=len(compressed),
        schema_version=TERMINAL_SIDECAR_SCHEMA_VERSION,
        logical_record_hash=payload["logical_record_hash"],
    )
    return reference, compressed


def decode_terminal_certificate_sidecar(
    reference: TerminalSidecarReference,
    compressed: bytes,
    *,
    descriptor: TerminalValidationDescriptor,
    method: str,
    method_numerical_hash: str,
) -> DecodedTerminalSidecar:
    path = PurePosixPath(reference.relative_path)
    if path.is_absolute() or ".." in path.parts or path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("unsafe sidecar path")
    if reference.schema_version != TERMINAL_SIDECAR_SCHEMA_VERSION:
        raise ValueError("sidecar reference schema version mismatch")
    if reference.byte_count != len(compressed):
        raise ValueError("sidecar byte count mismatch")
    if not _is_sha256(reference.sha256) or reference.sha256 != hashlib.sha256(compressed).hexdigest():
        raise ValueError("sidecar SHA-256 mismatch")
    try:
        payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sidecar decode failure") from error
    expected_keys = (
        "schema", "schema_version", "suite_class", "suite_version",
        "descriptor_index", "descriptor_hash", "scientific_spec_hash",
        "numerical_method_config_hash", "method", "method_numerical_hash",
        "certificate_type", "certificate_payload", "certificate_logical_hash",
        "complete_trace_payload", "complete_trace_logical_hash",
        "validation_evidence", "logical_record_hash",
    )
    _require_exact_keys(payload, expected_keys, "sidecar")
    if payload["schema"] != TERMINAL_SIDECAR_SCHEMA or payload["schema_version"] != TERMINAL_SIDECAR_SCHEMA_VERSION:
        raise ValueError("sidecar schema mismatch")
    bindings = {
        "suite_class": descriptor.suite_class,
        "suite_version": descriptor.suite_version,
        "descriptor_index": descriptor.descriptor_index,
        "descriptor_hash": descriptor.descriptor_hash,
        "scientific_spec_hash": descriptor.scientific_spec_hash,
        "numerical_method_config_hash": descriptor.numerical_method_config_hash,
        "method": method,
        "method_numerical_hash": method_numerical_hash,
    }
    if any(payload[key] != expected for key, expected in bindings.items()):
        raise ValueError("sidecar source binding mismatch")
    supplied_logical = payload["logical_record_hash"]
    logical_payload = dict(payload)
    logical_payload.pop("logical_record_hash")
    if not _is_sha256(supplied_logical) or supplied_logical != _logical_hash(logical_payload):
        raise ValueError("sidecar logical record hash mismatch")
    if reference.logical_record_hash != supplied_logical:
        raise ValueError("sidecar reference logical hash mismatch")
    if payload["certificate_logical_hash"] != _logical_hash(payload["certificate_payload"]):
        raise ValueError("sidecar certificate logical hash mismatch")
    if payload["complete_trace_logical_hash"] != _logical_hash(payload["complete_trace_payload"]):
        raise ValueError("sidecar trace logical hash mismatch")
    certificate = _decode_certificate(
        method,
        payload["certificate_payload"],
        payload["certificate_type"],
    )
    trace = _decode_canonical(payload["complete_trace_payload"])
    validation = _decode_canonical(payload["validation_evidence"])
    if not isinstance(trace, dict) or trace.get("complete") is not True:
        raise ValueError("sidecar complete trace is absent or incomplete")
    expected_trace_schemas = {
        "production_terminal": "terminal_production_complete_trace_v1",
        "reference_a": "terminal_reference_a_complete_trace_v1",
        "reference_b": "terminal_reference_b_complete_trace_v1",
        "agreement": "terminal_agreement_complete_trace_v1",
    }
    if trace.get("schema") != expected_trace_schemas[method]:
        raise ValueError("sidecar trace schema mismatch")
    if not isinstance(validation, dict):
        raise ValueError("sidecar validation evidence must be a mapping")
    return DecodedTerminalSidecar(certificate, trace, validation, payload)


def validate_terminal_certificate_sidecar(
    reference: TerminalSidecarReference,
    compressed: bytes,
    *,
    descriptor: TerminalValidationDescriptor,
    method: str,
    method_numerical_hash: str,
) -> Tuple[str, ...]:
    try:
        decode_terminal_certificate_sidecar(
            reference,
            compressed,
            descriptor=descriptor,
            method=method,
            method_numerical_hash=method_numerical_hash,
        )
    except (KeyError, TypeError, ValueError) as error:
        return (f"sidecar_invalid:{error}",)
    return ()


def _reference_row_fields(reference: TerminalReferenceRecord) -> Dict[str, Any]:
    return {
        "status": reference.status,
        "global_value_interval": reference.global_value_interval,
        "candidate_allocation_intervals": reference.candidate_allocation_intervals,
        "candidate_value_intervals": reference.candidate_value_intervals,
        "canonical_allocation_interval": reference.canonical_allocation_interval,
        "production_allocation": reference.production_allocation,
        "production_value_interval": reference.production_value_interval,
        "production_regret_interval": reference.production_regret_interval,
        "symmetry_proof_id": reference.structural_symmetry.proof_hash or None,
        "symmetry_required": reference.structural_symmetry.valid,
        "tie_status": reference.tie_status,
        "precision_level": reference.precision_level,
        "evaluation_count": reference.objective_evaluation_count,
        "unresolved": reference.status == "reference_unresolved",
    }


def _validated_terminal_evidence_plan(
    methods: Sequence[str],
    tie_count: int,
    symmetry_count: int,
) -> TerminalEvidencePlan:
    ordered = tuple(methods)
    if ordered != tuple(method for method in TERMINAL_METHOD_ORDER if method in ordered):
        raise RuntimeError("terminal evidence methods are not in frozen order")
    if not ordered or len(set(ordered)) != len(ordered):
        raise RuntimeError("terminal evidence plan has absent or duplicate methods")
    if not (0 <= tie_count <= len(ordered) and 0 <= symmetry_count <= len(ordered)):
        raise RuntimeError("terminal evidence path counts exceed the method plan")
    return TerminalEvidencePlan(ordered, int(tie_count), int(symmetry_count))


def project_terminal_evidence_plan(
    rows: Sequence[TerminalEvidenceRow],
) -> TerminalEvidencePlan:
    """Project the exact manifest metadata from authoritative full-evidence rows."""

    return _validated_terminal_evidence_plan(
        tuple(row.method for row in rows),
        sum(row.tie_status not in (None, "unique") for row in rows),
        sum(row.symmetry_required for row in rows),
    )


def require_terminal_evidence_plan_parity(
    expected: TerminalEvidencePlan,
    rows: Sequence[TerminalEvidenceRow],
) -> TerminalEvidencePlan:
    """Fail closed unless a full-evidence row projection equals its frozen plan."""

    observed = project_terminal_evidence_plan(rows)
    if observed != expected:
        raise RuntimeError(
            "source evidence plan differs from frozen manifest: "
            f"expected={expected.as_tuple()!r}, observed={observed.as_tuple()!r}"
        )
    return observed


def _record_phase(
    phase_seconds: Optional[Dict[str, float]],
    name: str,
    started: float,
) -> None:
    if phase_seconds is not None:
        phase_seconds[name] = time.perf_counter() - started


def evaluate_terminal_evidence_plan(
    descriptor: TerminalValidationDescriptor,
    mdp: Any,
    belief: Any,
    *,
    phase_seconds: Optional[Dict[str, float]] = None,
) -> TerminalEvidencePlan:
    """Compute manifest metadata without traces, sidecars, or full evidence rows."""

    total_started = time.perf_counter()
    started = time.perf_counter()
    failures = terminal_descriptor_source_failures(descriptor, mdp, belief)
    if failures:
        raise ValueError("descriptor/source binding failed: " + ",".join(failures))
    _record_phase(phase_seconds, "source_binding_validation", started)

    started = time.perf_counter()
    production = optimize_terminal_allocation(mdp, belief)
    _record_phase(phase_seconds, "production_terminal", started)

    started = time.perf_counter()
    batch = optimal_terminal_results_for_weight_rows(
        mdp, belief, (belief.weights,), float(belief.deliberation_time)
    )[0]
    scalar_batch_pass = production == batch
    symmetry_pass = (
        not production.structural_symmetry.valid
        or validate_structural_symmetry_proof(mdp, belief, production.structural_symmetry)
    )
    _record_phase(phase_seconds, "production_checks", started)

    started = time.perf_counter()
    reference_a = solve_terminal_reference_a(mdp, belief, production.allocation)
    _record_phase(phase_seconds, "reference_a", started)

    started = time.perf_counter()
    # Setup metadata describes evidence that can pass the formal gate. Revalidating A
    # here would deterministically rerun the expensive reference twice; the formal worker
    # performs those source and production checks and rejects any projection mismatch.
    trigger_reasons = terminal_reference_b_trigger_reasons(
        descriptor,
        production,
        reference_a,
        production_reference_a_pass=True,
        production_checks_pass=scalar_batch_pass and symmetry_pass,
        reference_a_source_valid=True,
    )
    _record_phase(phase_seconds, "reference_a_accepted_projection_and_escalation", started)

    tie_statuses = [production.tie_status, reference_a.tie_status]
    symmetry_required = [
        production.structural_symmetry.valid,
        reference_a.structural_symmetry.valid,
    ]
    methods = list(TERMINAL_METHOD_ORDER[:2])
    if trigger_reasons:
        started = time.perf_counter()
        # Planning projects the only B/agreement classification that formal evidence can
        # accept. Formal workers still run both independent methods and enforce exact
        # method/tie/symmetry parity before retaining evidence.
        methods.extend(TERMINAL_METHOD_ORDER[2:])
        tie_statuses.extend((reference_a.tie_status, reference_a.tie_status))
        symmetry_required.extend((
            production.structural_symmetry.valid,
            production.structural_symmetry.valid,
        ))
        _record_phase(phase_seconds, "reference_b_agreement_projection", started)

    plan = _validated_terminal_evidence_plan(
        methods,
        sum(status not in (None, "unique") for status in tie_statuses),
        sum(symmetry_required),
    )
    _record_phase(phase_seconds, "plan_computation_total", total_started)
    return plan


def _make_row(
    *, descriptor: TerminalValidationDescriptor, belief_hash: str, method: str,
    method_numerical_hash: str, sidecar: TerminalSidecarReference, status: str,
    global_value_interval: Optional[Tuple[float, float]],
    candidate_allocation_intervals: Sequence[Tuple[float, float]],
    candidate_value_intervals: Sequence[Tuple[float, float]],
    canonical_allocation_interval: Optional[Tuple[float, float]],
    production_allocation: Optional[float],
    production_value_interval: Optional[Tuple[float, float]],
    production_regret_interval: Optional[Tuple[float, float]],
    symmetry_proof_id: Optional[str], symmetry_required: bool, symmetry_pass: bool,
    tie_status: Optional[str], precision_level: Optional[float], evaluation_count: int,
    scalar_batch_required: bool, scalar_batch_pass: bool,
    reference_b_required: bool, reference_b_trigger_reasons: Sequence[str],
    validation_checks: Sequence[Tuple[str, bool]], pass_status: bool,
    unresolved: bool, failure_reasons: Sequence[str],
) -> TerminalEvidenceRow:
    row = TerminalEvidenceRow(
        TERMINAL_EVIDENCE_ROW_SCHEMA, descriptor.suite_class, descriptor.suite_version,
        descriptor.descriptor_index, descriptor.descriptor_hash, descriptor.source_case_id,
        belief_hash, descriptor.history_hash, descriptor.posterior_weight_hash,
        descriptor.scientific_spec_hash, descriptor.numerical_method_config_hash, method,
        method_numerical_hash, status, global_value_interval,
        tuple(candidate_allocation_intervals), tuple(candidate_value_intervals),
        canonical_allocation_interval, production_allocation, production_value_interval,
        production_regret_interval, symmetry_proof_id, symmetry_required, symmetry_pass,
        tie_status, precision_level, evaluation_count, scalar_batch_required,
        scalar_batch_pass, reference_b_required, tuple(reference_b_trigger_reasons),
        tuple((str(name), bool(passed)) for name, passed in validation_checks),
        bool(pass_status), bool(unresolved), tuple(failure_reasons), sidecar, "",
    )
    return replace(row, logical_record_hash=terminal_evidence_row_hash(row))


def evaluate_terminal_evidence_descriptor(
    descriptor: TerminalValidationDescriptor,
    mdp: Any,
    belief: Any,
    *,
    concurrent_reference_b: bool = False,
    reference_b_runtime_evidence: Optional[MutableMapping[str, Any]] = None,
) -> TerminalEvidenceBundle:
    failures = terminal_descriptor_source_failures(descriptor, mdp, belief)
    if failures:
        raise ValueError("descriptor/source binding failed: " + ",".join(failures))
    production, production_trace = optimize_terminal_allocation_with_trace(mdp, belief)
    batch = optimal_terminal_results_for_weight_rows(
        mdp, belief, (belief.weights,), float(belief.deliberation_time)
    )[0]
    scalar_batch_pass = production == batch
    symmetry_pass = (
        not production.structural_symmetry.valid
        or validate_structural_symmetry_proof(mdp, belief, production.structural_symmetry)
    )
    reference_a, reference_a_trace = solve_terminal_reference_a_with_trace(
        mdp, belief, production.allocation
    )
    a_hash = terminal_reference_a_numerical_method_config_hash(reference_a.evaluation_cap)
    solver_scientific_hash = terminal_scientific_spec_hash(mdp)
    a_source_proof = source_validate_terminal_reference_record(
        reference_a, mdp, belief,
        scientific_spec_hash=solver_scientific_hash,
        numerical_method_config_hash=a_hash,
    )
    a_source_pass = a_source_proof.valid
    production_a = validate_production_against_reference_a(
        mdp, belief, production, reference_a,
        scientific_spec_hash=solver_scientific_hash,
        numerical_method_config_hash=a_hash,
        _source_validation_proof=a_source_proof,
    )
    production_a_pass = production_a.status == "accepted"
    trigger_reasons = terminal_reference_b_trigger_reasons(
        descriptor, production, reference_a,
        production_reference_a_pass=production_a_pass,
        production_checks_pass=scalar_batch_pass and symmetry_pass,
        reference_a_source_valid=a_source_pass,
    )
    b_required = bool(trigger_reasons)
    belief_hash = terminal_belief_identity_hash(belief)
    rows = []
    sidecars: Dict[str, bytes] = {}

    production_checks = (
        ("scalar_batch_parity", scalar_batch_pass),
        ("symmetry_proof", symmetry_pass),
        ("production_reference_a", production_a_pass),
        ("production_value_tolerance", production.regret_upper_bound <= PRODUCTION_VALUE_TOLERANCE),
        ("production_allocation_domain", 0.0 <= production.allocation <= 1.0),
    )
    production_failures = tuple(name for name, passed in production_checks if not passed)
    production_hash = production_terminal_numerical_method_config_hash()
    production_sidecar, payload = build_terminal_certificate_sidecar(
        relative_path=_sidecar_path(descriptor, "production_terminal"),
        descriptor=descriptor, method="production_terminal",
        method_numerical_hash=production_hash, certificate=production,
        complete_trace=production_trace,
        validation_evidence={
            "checks": production_checks,
            "scalar_batch_result": asdict(batch),
            "production_reference_a_validation": asdict(production_a),
        },
    )
    sidecars[production_sidecar.relative_path] = payload
    rows.append(_make_row(
        descriptor=descriptor, belief_hash=belief_hash, method="production_terminal",
        method_numerical_hash=production_hash, sidecar=production_sidecar,
        status="accepted" if not production_failures else "failed",
        global_value_interval=(production.value, production.global_upper_bound),
        candidate_allocation_intervals=production.candidate_intervals,
        candidate_value_intervals=(),
        canonical_allocation_interval=(production.allocation, production.allocation),
        production_allocation=production.allocation,
        production_value_interval=(production.value, production.value),
        production_regret_interval=(0.0, production.regret_upper_bound),
        symmetry_proof_id=production.structural_symmetry.proof_hash or None,
        symmetry_required=production.structural_symmetry.valid,
        symmetry_pass=symmetry_pass, tie_status=production.tie_status,
        precision_level=None, evaluation_count=production.objective_evaluations,
        scalar_batch_required=True, scalar_batch_pass=scalar_batch_pass,
        reference_b_required=b_required, reference_b_trigger_reasons=trigger_reasons,
        validation_checks=production_checks, pass_status=not production_failures,
        unresolved=False, failure_reasons=production_failures,
    ))

    a_checks = (
        ("reference_a_source_valid", a_source_pass),
        ("production_reference_a", production_a_pass),
    )
    a_failures = tuple(name for name, passed in a_checks if not passed)
    a_sidecar, payload = build_terminal_certificate_sidecar(
        relative_path=_sidecar_path(descriptor, "reference_a"), descriptor=descriptor,
        method="reference_a", method_numerical_hash=a_hash, certificate=reference_a,
        complete_trace=reference_a_trace,
        validation_evidence={
            "checks": a_checks,
            "production_validation": asdict(production_a),
        },
    )
    sidecars[a_sidecar.relative_path] = payload
    rows.append(_make_row(
        descriptor=descriptor, belief_hash=belief_hash, method="reference_a",
        method_numerical_hash=a_hash, sidecar=a_sidecar,
        symmetry_pass=symmetry_pass, scalar_batch_required=False,
        scalar_batch_pass=True, reference_b_required=b_required,
        reference_b_trigger_reasons=trigger_reasons, validation_checks=a_checks,
        pass_status=a_source_pass and production_a_pass and reference_a.status == "resolved",
        failure_reasons=a_failures, **_reference_row_fields(reference_a),
    ))

    if b_required:
        cross_process_proof = None
        cross_process_source_hash = None
        cross_process_interpreter_hash = None
        if concurrent_reference_b:
            concurrent_result = solve_terminal_reference_b_concurrently(
                descriptor, mdp, belief, production
            )
            reference_b = concurrent_result.record
            reference_b_trace = concurrent_result.complete_trace
            b_hash = terminal_reference_b_numerical_method_config_hash(
                reference_b.evaluation_cap
            )
            cross_process_proof = concurrent_result.source_validation_proof
            cross_process_source_hash = concurrent_result.traced_worker.source_identity_hash
            cross_process_interpreter_hash = (
                concurrent_result.traced_worker.interpreter_identity_hash
            )
            b_source_pass = terminal_reference_cross_process_proof_matches(
                cross_process_proof,
                reference_b,
                mdp,
                belief,
                scientific_spec_hash=solver_scientific_hash,
                numerical_method_config_hash=b_hash,
                source_identity_hash=cross_process_source_hash,
                interpreter_identity_hash=cross_process_interpreter_hash,
                production_allocation=production.allocation,
            )
            if reference_b_runtime_evidence is not None:
                reference_b_runtime_evidence.update({
                    "traced_worker": asdict(concurrent_result.traced_worker),
                    "source_worker": asdict(concurrent_result.source_worker),
                    "coordinator_peak_rss_bytes": (
                        concurrent_result.coordinator_peak_rss_bytes
                    ),
                })
            b_source_proof = None
        else:
            reference_b, reference_b_trace = solve_terminal_reference_b_with_trace(
                mdp, belief, production.allocation
            )
            b_hash = terminal_reference_b_numerical_method_config_hash(
                reference_b.evaluation_cap
            )
            b_source_proof = source_validate_terminal_reference_b_record(
                reference_b, mdp, belief,
                scientific_spec_hash=solver_scientific_hash,
                numerical_method_config_hash=b_hash,
            )
            b_source_pass = b_source_proof.valid
        agreement = validate_terminal_reference_agreement(
            mdp, belief, production, reference_a, reference_b,
            scientific_spec_hash=solver_scientific_hash,
            reference_a_numerical_method_config_hash=a_hash,
            reference_b_numerical_method_config_hash=b_hash,
            _source_validation_proof_a=a_source_proof,
            _source_validation_proof_b=b_source_proof,
            _cross_process_validation_proof_b=cross_process_proof,
            _cross_process_source_identity_hash_b=cross_process_source_hash,
            _cross_process_interpreter_identity_hash_b=(
                cross_process_interpreter_hash
            ),
        )
        b_checks = (("reference_b_source_valid", b_source_pass),)
        b_sidecar, payload = build_terminal_certificate_sidecar(
            relative_path=_sidecar_path(descriptor, "reference_b"), descriptor=descriptor,
            method="reference_b", method_numerical_hash=b_hash, certificate=reference_b,
            complete_trace=reference_b_trace,
            validation_evidence={"checks": b_checks},
        )
        sidecars[b_sidecar.relative_path] = payload
        rows.append(_make_row(
            descriptor=descriptor, belief_hash=belief_hash, method="reference_b",
            method_numerical_hash=b_hash, sidecar=b_sidecar,
            symmetry_pass=symmetry_pass, scalar_batch_required=False,
            scalar_batch_pass=True, reference_b_required=True,
            reference_b_trigger_reasons=trigger_reasons, validation_checks=b_checks,
            pass_status=b_source_pass and reference_b.status == "resolved",
            failure_reasons=tuple(name for name, passed in b_checks if not passed),
            **_reference_row_fields(reference_b),
        ))
        agreement_hash = terminal_reference_agreement_numerical_method_config_hash()
        agreement_trace = {
            "schema": "terminal_agreement_complete_trace_v1",
            "complete": True,
            "reference_a_certificate_hash": reference_a.certificate_hash,
            "reference_b_certificate_hash": reference_b.certificate_hash,
            "ordered_checks": agreement.checks,
            "failure_reasons": agreement.failure_reasons,
            "trigger_reasons": trigger_reasons,
        }
        agreement_sidecar, payload = build_terminal_certificate_sidecar(
            relative_path=_sidecar_path(descriptor, "agreement"), descriptor=descriptor,
            method="agreement", method_numerical_hash=agreement_hash,
            certificate=agreement, complete_trace=agreement_trace,
            validation_evidence={"checks": agreement.checks},
        )
        sidecars[agreement_sidecar.relative_path] = payload
        agreement_unresolved = (
            reference_a.status == "reference_unresolved"
            or reference_b.status == "reference_unresolved"
        )
        rows.append(_make_row(
            descriptor=descriptor, belief_hash=belief_hash, method="agreement",
            method_numerical_hash=agreement_hash, sidecar=agreement_sidecar,
            status=agreement.status,
            global_value_interval=agreement.agreed_global_value_interval,
            candidate_allocation_intervals=(), candidate_value_intervals=(),
            canonical_allocation_interval=agreement.agreed_canonical_allocation_interval,
            production_allocation=agreement.production_allocation,
            production_value_interval=agreement.production_value_interval,
            production_regret_interval=agreement.production_regret_interval,
            symmetry_proof_id=production.structural_symmetry.proof_hash or None,
            symmetry_required=production.structural_symmetry.valid,
            symmetry_pass=symmetry_pass, tie_status=agreement.tie_status,
            precision_level=None,
            evaluation_count=reference_a.objective_evaluation_count + reference_b.objective_evaluation_count,
            scalar_batch_required=False, scalar_batch_pass=True,
            reference_b_required=True, reference_b_trigger_reasons=trigger_reasons,
            validation_checks=agreement.checks,
            pass_status=agreement.status == "accepted" and not agreement_unresolved,
            unresolved=agreement_unresolved, failure_reasons=agreement.failure_reasons,
        ))
    return TerminalEvidenceBundle(
        descriptor_hash=descriptor.descriptor_hash,
        rows=tuple(rows),
        sidecars=tuple(sorted(sidecars.items())),
    )


def validate_terminal_evidence_row(
    row: TerminalEvidenceRow,
    descriptor: TerminalValidationDescriptor,
) -> Tuple[str, ...]:
    failures = []
    if row.schema != TERMINAL_EVIDENCE_ROW_SCHEMA:
        failures.append("row_schema_mismatch")
    bindings = {
        "suite_class": descriptor.suite_class,
        "suite_version": descriptor.suite_version,
        "descriptor_index": descriptor.descriptor_index,
        "descriptor_hash": descriptor.descriptor_hash,
        "source_case_id": descriptor.source_case_id,
        "suite_history_hash": descriptor.history_hash,
        "posterior_weight_hash": descriptor.posterior_weight_hash,
        "scientific_spec_hash": descriptor.scientific_spec_hash,
        "numerical_method_config_hash": descriptor.numerical_method_config_hash,
    }
    failures.extend(
        f"row_binding_mismatch:{name}"
        for name, expected in bindings.items()
        if getattr(row, name) != expected
    )
    expected_method_hashes = {
        "production_terminal": production_terminal_numerical_method_config_hash(),
        "reference_a": terminal_reference_a_numerical_method_config_hash(),
        "reference_b": terminal_reference_b_numerical_method_config_hash(),
        "agreement": terminal_reference_agreement_numerical_method_config_hash(),
    }
    if row.method not in expected_method_hashes:
        failures.append("row_method_invalid")
    elif row.method_numerical_hash != expected_method_hashes[row.method]:
        failures.append("row_method_numerical_hash_source_mismatch")
    if not _is_sha256(row.belief_hash):
        failures.append("row_belief_hash_invalid")
    if row.production_allocation is not None and not (
        type(row.production_allocation) is float
        and math.isfinite(row.production_allocation)
        and 0.0 <= row.production_allocation <= 1.0
    ):
        failures.append("row_production_allocation_invalid")
    if row.canonical_allocation_interval is not None and not _allocation_interval(row.canonical_allocation_interval):
        failures.append("row_canonical_allocation_interval_invalid")
    if any(not _allocation_interval(item) for item in row.candidate_allocation_intervals):
        failures.append("row_candidate_allocation_interval_invalid")
    if any(not _finite_interval(item) for item in row.candidate_value_intervals):
        failures.append("row_candidate_value_interval_invalid")
    for name, interval in (
        ("global", row.global_value_interval),
        ("production_value", row.production_value_interval),
    ):
        if interval is not None and not _finite_interval(interval):
            failures.append(f"row_{name}_interval_invalid")
    if row.production_regret_interval is not None and (
        not _finite_interval(row.production_regret_interval)
        or row.production_regret_interval[0] < 0.0
    ):
        failures.append("row_regret_interval_invalid")
    allowed_status = {
        "production_terminal": {"accepted", "failed"},
        "reference_a": {"resolved", "reference_unresolved"},
        "reference_b": {"resolved", "reference_unresolved"},
        "agreement": {"accepted", "rejected"},
    }
    if row.method in allowed_status and row.status not in allowed_status[row.method]:
        failures.append("row_status_invalid")
    allowed_ties = {None, "unique", "certified_value_tie", "structural_symmetry_tie", "ordinary_tie_provisional", "reference_unresolved"}
    if row.tie_status not in allowed_ties:
        failures.append("row_tie_status_invalid")
    if type(row.evaluation_count) is not int or row.evaluation_count < 0:
        failures.append("row_evaluation_count_invalid")
    if row.precision_level is not None and not (
        type(row.precision_level) is float
        and math.isfinite(row.precision_level)
        and row.precision_level > 0.0
    ):
        failures.append("row_precision_level_invalid")
    if any(
        type(name) is not str or type(passed) is not bool
        for name, passed in row.validation_checks
    ):
        failures.append("row_validation_check_invalid")
    if len({name for name, _ in row.validation_checks}) != len(row.validation_checks):
        failures.append("row_validation_check_duplicate")
    if len(set(row.failure_reasons)) != len(row.failure_reasons):
        failures.append("row_failure_reason_duplicate")
    if row.symmetry_required and not _is_sha256(row.symmetry_proof_id):
        failures.append("row_symmetry_proof_required")
    if row.method == "production_terminal":
        if (
            row.precision_level is not None
            or row.candidate_value_intervals
            or not row.scalar_batch_required
            or row.unresolved
        ):
            failures.append("row_production_required_fields_invalid")
    elif row.method in ("reference_a", "reference_b"):
        if (
            row.precision_level is None
            or row.scalar_batch_required
            or len(row.candidate_allocation_intervals)
            != len(row.candidate_value_intervals)
            or row.unresolved != (row.status == "reference_unresolved")
        ):
            failures.append("row_reference_required_fields_invalid")
        if row.method == "reference_b" and not row.reference_b_required:
            failures.append("row_reference_b_requirement_invalid")
    elif row.method == "agreement":
        if (
            row.precision_level is not None
            or row.scalar_batch_required
            or row.candidate_allocation_intervals
            or row.candidate_value_intervals
            or not row.reference_b_required
            or (row.unresolved and row.status == "accepted")
        ):
            failures.append("row_agreement_required_fields_invalid")
    if row.pass_status:
        if row.unresolved or row.failure_reasons or any(not passed for _, passed in row.validation_checks):
            failures.append("row_pass_status_inconsistent")
        passing_status = {
            "production_terminal": "accepted",
            "reference_a": "resolved",
            "reference_b": "resolved",
            "agreement": "accepted",
        }
        if row.method in passing_status and row.status != passing_status[row.method]:
            failures.append("row_pass_status_method_mismatch")
        if row.production_regret_interval is not None and row.production_regret_interval[1] > TERMINAL_PRODUCTION_REGRET_TOLERANCE:
            failures.append("row_regret_threshold_failed")
        if row.global_value_interval is None:
            failures.append("row_global_value_threshold_unavailable")
        else:
            global_width_tolerance = (
                PRODUCTION_VALUE_TOLERANCE
                if row.method == "production_terminal"
                else TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE
            )
            if (
                row.global_value_interval[1] - row.global_value_interval[0]
                > global_width_tolerance
            ):
                failures.append("row_global_value_width_threshold_failed")
        if row.method in ("production_terminal", "reference_a", "agreement"):
            if row.global_value_interval is None or row.production_value_interval is None:
                failures.append("row_production_value_threshold_unavailable")
            elif _interval_distance(
                row.production_value_interval,
                row.global_value_interval,
            ) > TERMINAL_PRODUCTION_VALUE_TOLERANCE:
                failures.append("row_production_value_threshold_failed")
            if row.production_allocation is None or row.canonical_allocation_interval is None:
                failures.append("row_production_allocation_threshold_unavailable")
            elif _point_interval_distance(
                row.production_allocation,
                row.canonical_allocation_interval,
            ) > TERMINAL_PRODUCTION_ALLOCATION_TOLERANCE:
                failures.append("row_production_allocation_threshold_failed")
            if row.production_regret_interval is None:
                failures.append("row_regret_threshold_unavailable")
    if row.logical_record_hash != terminal_evidence_row_hash(row):
        failures.append("row_logical_record_hash_mismatch")
    return tuple(failures)


def validate_terminal_evidence_bundle_source(
    bundle: TerminalEvidenceBundle,
    descriptor: TerminalValidationDescriptor,
    mdp: Any,
    belief: Any,
) -> Tuple[str, ...]:
    """Recompute all source certificates/traces and compare every row/sidecar byte."""

    try:
        expected = evaluate_terminal_evidence_descriptor(descriptor, mdp, belief)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as error:
        return (f"source_recomputation_failed:{type(error).__name__}",)
    return _compare_terminal_evidence_bundle_to_expected(bundle, expected, descriptor)


def validate_terminal_evidence_bundle_structure(
    bundle: TerminalEvidenceBundle,
    descriptor: TerminalValidationDescriptor,
) -> Tuple[str, ...]:
    """Validate immutable rows and sidecars without launching numerical solvers."""

    failures = []
    if bundle.descriptor_hash != descriptor.descriptor_hash:
        failures.append("bundle_descriptor_hash_mismatch")
    rows_by_key = {_row_key_parts(row): row for row in bundle.rows}
    if len(rows_by_key) != len(bundle.rows):
        failures.append("bundle_duplicate_rows")
    sidecars = dict(bundle.sidecars)
    if len(sidecars) != len(bundle.sidecars):
        failures.append("bundle_duplicate_sidecars")
    expected_paths = set()
    for row in bundle.rows:
        failures.extend(
            f"{terminal_evidence_row_key(row)}:{reason}"
            for reason in validate_terminal_evidence_row(row, descriptor)
        )
        expected_paths.add(row.sidecar.relative_path)
        payload = sidecars.get(row.sidecar.relative_path)
        if payload is None:
            failures.append(f"{row.sidecar.relative_path}:missing")
            continue
        failures.extend(
            f"{row.sidecar.relative_path}:{reason}"
            for reason in validate_terminal_certificate_sidecar(
                row.sidecar,
                payload,
                descriptor=descriptor,
                method=row.method,
                method_numerical_hash=row.method_numerical_hash,
            )
        )
    if set(sidecars) != expected_paths:
        failures.append("bundle_sidecar_set_mismatch")
    return tuple(dict.fromkeys(failures))


def _compare_terminal_evidence_bundle_to_expected(
    bundle: TerminalEvidenceBundle,
    expected: TerminalEvidenceBundle,
    descriptor: TerminalValidationDescriptor,
) -> Tuple[str, ...]:
    failures = []
    if bundle.descriptor_hash != descriptor.descriptor_hash:
        failures.append("bundle_descriptor_hash_mismatch")
    actual_rows = {_row_key_parts(row): row for row in bundle.rows}
    expected_rows = {_row_key_parts(row): row for row in expected.rows}
    if len(actual_rows) != len(bundle.rows):
        failures.append("bundle_duplicate_rows")
    if set(actual_rows) != set(expected_rows):
        failures.append("bundle_method_set_mismatch")
    actual_sidecars = dict(bundle.sidecars)
    expected_sidecars = dict(expected.sidecars)
    if len(actual_sidecars) != len(bundle.sidecars):
        failures.append("bundle_duplicate_sidecars")
    for key in sorted(set(actual_rows) | set(expected_rows)):
        actual = actual_rows.get(key)
        expected_row = expected_rows.get(key)
        if actual is None or expected_row is None:
            continue
        row_failures = validate_terminal_evidence_row(actual, descriptor)
        failures.extend(f"{terminal_evidence_row_key(actual)}:{reason}" for reason in row_failures)
        if actual != expected_row:
            failures.append(f"{terminal_evidence_row_key(actual)}:row_semantic_mismatch")
        payload = actual_sidecars.get(actual.sidecar.relative_path)
        if payload is None:
            failures.append(f"{actual.sidecar.relative_path}:missing")
            continue
        sidecar_failures = validate_terminal_certificate_sidecar(
            actual.sidecar, payload, descriptor=descriptor, method=actual.method,
            method_numerical_hash=actual.method_numerical_hash,
        )
        failures.extend(f"{actual.sidecar.relative_path}:{reason}" for reason in sidecar_failures)
        if payload != expected_sidecars.get(expected_row.sidecar.relative_path):
            failures.append(f"{actual.sidecar.relative_path}:source_recomputation_mismatch")
    if set(actual_sidecars) != set(expected_sidecars):
        failures.append("bundle_sidecar_set_mismatch")
    return tuple(dict.fromkeys(failures))


def validate_terminal_certificate_sidecar_source(
    reference: TerminalSidecarReference,
    compressed: bytes,
    *,
    descriptor: TerminalValidationDescriptor,
    method: str,
    method_numerical_hash: str,
    mdp: Any,
    belief: Any,
) -> Tuple[str, ...]:
    """Recompute one method's certificate and complete trace from source objects."""

    structural = validate_terminal_certificate_sidecar(
        reference,
        compressed,
        descriptor=descriptor,
        method=method,
        method_numerical_hash=method_numerical_hash,
    )
    if structural:
        return structural
    try:
        expected = evaluate_terminal_evidence_descriptor(descriptor, mdp, belief)
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as error:
        return (f"sidecar_source_recomputation_failed:{type(error).__name__}",)
    expected_rows = {row.method: row for row in expected.rows}
    expected_sidecars = dict(expected.sidecars)
    expected_row = expected_rows.get(method)
    if expected_row is None:
        return ("sidecar_method_not_required_by_validated_source",)
    failures = []
    if reference != expected_row.sidecar:
        failures.append("sidecar_reference_source_mismatch")
    if compressed != expected_sidecars[expected_row.sidecar.relative_path]:
        failures.append("sidecar_certificate_or_trace_source_mismatch")
    return tuple(failures)


def _empty_summary(
    suite: TerminalValidationSuite,
    identities: TerminalValidationIdentities,
    provider_hash: str,
    suite_status: str,
    authoritative: bool,
    failures: Sequence[str],
) -> TerminalEvidenceCollectionSummary:
    summary = TerminalEvidenceCollectionSummary(
        schema=TERMINAL_COLLECTION_SCHEMA,
        suite_manifest_hash=suite.manifest.manifest_hash,
        suite_ordered_descriptor_hash=suite.manifest.ordered_descriptor_hash,
        suite_validation_status=suite_status,
        base_provider_hash=provider_hash,
        authoritative_source_accepted=authoritative,
        scientific_spec_hash=identities.scientific_spec_hash,
        numerical_method_config_hash=identities.numerical_method_config_hash,
        expected_descriptor_count=len(suite.descriptors),
        observed_descriptor_count=0,
        observed_row_count=0,
        observed_sidecar_count=0,
        missing_row_keys=(),
        duplicate_row_keys=(),
        unexpected_row_keys=(),
        invalid_row_keys=tuple(failures),
        invalid_sidecar_paths=(),
        failed_row_keys=(),
        unresolved_row_keys=(),
        reference_b_required_keys=(),
        reference_b_missing_keys=(),
        maximum_evaluation_count=0,
        maximum_regret_upper=None,
        evidence_valid=False,
        stage_complete=False,
        candidate_pass=False,
        failure_reasons=tuple(failures),
        logical_record_hash="",
    )
    return replace(summary, logical_record_hash=_logical_hash(replace(summary, logical_record_hash="")))


def recompute_terminal_evidence_summary(
    rows: Sequence[TerminalEvidenceRow],
    sidecars: Mapping[str, bytes],
    suite: TerminalValidationSuite,
    *,
    base_provider: Optional[CanonicalBaseProvider] = None,
    authoritative_acceptance_validator: Optional[Callable[[CanonicalBaseProvider], bool]] = None,
    require_authoritative: bool = True,
) -> TerminalEvidenceCollectionSummary:
    """Validate immutable suite/source identities, then recompute every evidence row."""

    identities = load_terminal_validation_identities()
    provider = base_provider or build_local_diagnostic_base_provider()
    suite_validation = validate_terminal_validation_suite(
        suite, identities, base_provider=provider,
        require_authoritative=require_authoritative,
        authoritative_acceptance_validator=authoritative_acceptance_validator,
    )
    if suite_validation.failures:
        return _empty_summary(
            suite, identities, suite_validation.provider_hash,
            suite_validation.validation_status,
            suite_validation.authoritative_source_accepted,
            tuple(f"suite:{reason}" for reason in suite_validation.failures),
        )
    row_groups: Dict[Tuple[str, str, int, str], list[TerminalEvidenceRow]] = {}
    for row in rows:
        row_groups.setdefault(_row_key_parts(row), []).append(row)
    duplicate_rows = tuple(sorted(
        "/".join(str(part) for part in key)
        for key, group in row_groups.items() if len(group) != 1
    ))
    unexpected = []
    invalid = []
    invalid_sidecars = []
    required_b = []
    missing_b = []
    expected_keys = set()
    expected_sidecar_paths = set()
    validated_source_rows = []
    for descriptor in suite.descriptors:
        try:
            mdp, belief = reconstruct_terminal_evidence_source(descriptor, provider)
            expected_bundle = evaluate_terminal_evidence_descriptor(descriptor, mdp, belief)
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as error:
            invalid.append(f"{_descriptor_key_text(descriptor)}:source_reconstruction:{type(error).__name__}")
            continue
        if not expected_bundle.rows:
            invalid.append(
                f"{_descriptor_key_text(descriptor)}:source_reconstruction:empty_evidence_bundle"
            )
            continue
        bundle_rows = []
        descriptor_sidecar_paths = set()
        for expected_row in expected_bundle.rows:
            key = _row_key_parts(expected_row)
            expected_keys.add(key)
            validated_source_rows.append((expected_row, descriptor))
            group = row_groups.get(key, ())
            if len(group) == 1:
                bundle_rows.append(group[0])
            descriptor_sidecar_paths.add(expected_row.sidecar.relative_path)
            expected_sidecar_paths.add(expected_row.sidecar.relative_path)
        bundle_sidecars = {
            path: sidecars[path]
            for path in descriptor_sidecar_paths
            if path in sidecars
        }
        submitted_bundle = TerminalEvidenceBundle(
            descriptor.descriptor_hash, tuple(bundle_rows), tuple(sorted(bundle_sidecars.items()))
        )
        failures = _compare_terminal_evidence_bundle_to_expected(
            submitted_bundle, expected_bundle, descriptor
        )
        invalid.extend(f"{_descriptor_key_text(descriptor)}:{reason}" for reason in failures)
        triggers = expected_bundle.rows[0].reference_b_trigger_reasons
        if triggers:
            required_b.append(_descriptor_key_text(descriptor))
            for method in ("reference_b", "agreement"):
                key = (*_descriptor_key(descriptor), method)
                if len(row_groups.get(key, ())) != 1:
                    missing_b.append("/".join(str(part) for part in key))

    supplied_keys = set(row_groups)
    missing = tuple(sorted(
        "/".join(str(part) for part in key) for key in expected_keys - supplied_keys
    ))
    unexpected.extend(
        "/".join(str(part) for part in key) for key in supplied_keys - expected_keys
    )
    referenced_paths = [row.sidecar.relative_path for row in rows]
    invalid_sidecars.extend(
        path + ":duplicate_reference"
        for path, count in Counter(referenced_paths).items() if count > 1
    )
    invalid_sidecars.extend(
        path + ":unreferenced" for path in sidecars if path not in expected_sidecar_paths
    )
    invalid_sidecars.extend(
        path + ":missing" for path in expected_sidecar_paths if path not in sidecars
    )
    failed = tuple(sorted(
        terminal_evidence_row_key(row)
        for row, _ in validated_source_rows if not row.pass_status
    ))
    unresolved = tuple(sorted(
        terminal_evidence_row_key(row)
        for row, _ in validated_source_rows if row.unresolved
    ))
    regrets = [
        row.production_regret_interval[1]
        for row, _ in validated_source_rows
        if row.production_regret_interval is not None
    ]
    source_threshold_failures = tuple(
        f"{terminal_evidence_row_key(row)}:{reason}"
        for row, descriptor in validated_source_rows
        for reason in validate_terminal_evidence_row(row, descriptor)
    )
    invalid.extend(source_threshold_failures)
    reasons = []
    for condition, name in (
        (missing, "missing_rows"), (duplicate_rows, "duplicate_rows"),
        (unexpected, "unexpected_rows"), (invalid, "invalid_rows"),
        (invalid_sidecars, "invalid_sidecars"), (failed, "failed_rows"),
        (unresolved, "unresolved_rows"), (missing_b, "missing_reference_b_escalations"),
    ):
        if condition:
            reasons.append(name)
    evidence_valid = not any((missing, duplicate_rows, unexpected, invalid, invalid_sidecars, missing_b))
    stage_complete = evidence_valid and (
        suite_validation.authoritative_source_accepted if require_authoritative else True
    )
    candidate_pass = (
        stage_complete
        and suite_validation.authoritative_source_accepted
        and not failed
        and not unresolved
        and all(regret <= TERMINAL_PRODUCTION_REGRET_TOLERANCE for regret in regrets)
    )
    if require_authoritative and not suite_validation.authoritative_source_accepted:
        reasons.append("authoritative_source_not_accepted")
    summary = TerminalEvidenceCollectionSummary(
        schema=TERMINAL_COLLECTION_SCHEMA,
        suite_manifest_hash=suite.manifest.manifest_hash,
        suite_ordered_descriptor_hash=suite.manifest.ordered_descriptor_hash,
        suite_validation_status=suite_validation.validation_status,
        base_provider_hash=suite_validation.provider_hash,
        authoritative_source_accepted=suite_validation.authoritative_source_accepted,
        scientific_spec_hash=identities.scientific_spec_hash,
        numerical_method_config_hash=identities.numerical_method_config_hash,
        expected_descriptor_count=len(suite.descriptors),
        observed_descriptor_count=len({key[:3] for key in row_groups}),
        observed_row_count=len(rows),
        observed_sidecar_count=len(sidecars),
        missing_row_keys=missing,
        duplicate_row_keys=duplicate_rows,
        unexpected_row_keys=tuple(sorted(set(unexpected))),
        invalid_row_keys=tuple(sorted(set(invalid))),
        invalid_sidecar_paths=tuple(sorted(set(invalid_sidecars))),
        failed_row_keys=failed,
        unresolved_row_keys=unresolved,
        reference_b_required_keys=tuple(sorted(required_b)),
        reference_b_missing_keys=tuple(sorted(set(missing_b))),
        maximum_evaluation_count=max(
            (row.evaluation_count for row, _ in validated_source_rows), default=0
        ),
        maximum_regret_upper=max(regrets) if regrets else None,
        evidence_valid=evidence_valid,
        stage_complete=stage_complete,
        candidate_pass=candidate_pass,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        logical_record_hash="",
    )
    return replace(summary, logical_record_hash=_logical_hash(replace(summary, logical_record_hash="")))


__all__ = [
    "DecodedTerminalSidecar", "REFERENCE_B_NEAR_TIE_SEPARATION",
    "TERMINAL_COLLECTION_SCHEMA", "TERMINAL_EVIDENCE_ROW_SCHEMA",
    "TERMINAL_METHOD_ORDER", "TERMINAL_SIDECAR_SCHEMA",
    "TERMINAL_SIDECAR_SCHEMA_VERSION", "TerminalEvidenceBundle", "TerminalEvidencePlan",
    "TerminalEvidenceCollectionSummary", "TerminalEvidenceRow",
    "TerminalSidecarReference", "build_terminal_certificate_sidecar",
    "decode_terminal_certificate_sidecar", "evaluate_terminal_evidence_descriptor",
    "evaluate_terminal_evidence_plan", "project_terminal_evidence_plan",
    "require_terminal_evidence_plan_parity",
    "recompute_terminal_evidence_summary", "reconstruct_terminal_evidence_source",
    "terminal_descriptor_source_failures", "terminal_evidence_row_hash",
    "terminal_evidence_row_key", "terminal_reference_b_trigger_reasons",
    "validate_terminal_certificate_sidecar", "validate_terminal_certificate_sidecar_source",
    "validate_terminal_evidence_bundle_source",
    "validate_terminal_evidence_row",
]
