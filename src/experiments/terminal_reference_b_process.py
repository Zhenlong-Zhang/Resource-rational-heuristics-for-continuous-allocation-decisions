from __future__ import annotations

"""Isolated two-process orchestration for independent Terminal Reference B solves."""

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportBeliefState,
    FiniteSupportMetaMDP,
    FiniteSupportPrior,
)
from ..mdp.meta_mdp import EnvironmentConfig
from ..solvers.terminal import StructuralSymmetry, TerminalOptimizationResult
from ..solvers.terminal_reference import (
    CandidateIsolationEvidence,
    TerminalReferenceCrossProcessValidationProof,
    TerminalReferenceRecord,
    _mint_terminal_reference_cross_process_validation_proof,
    terminal_belief_identity_hash,
    terminal_mdp_identity_hash,
    terminal_reference_certificate_hash,
    terminal_scientific_spec_hash,
)
from ..solvers.terminal_reference_b import (
    REFERENCE_B_EVALUATION_CAP,
    solve_terminal_reference_b,
    solve_terminal_reference_b_with_trace,
    terminal_reference_b_numerical_method_config_hash,
    validate_terminal_reference_b_record_structure,
)
from .terminal_validation_suite import (
    TerminalHistoryStep,
    TerminalValidationDescriptor,
    canonical_hash,
    terminal_scientific_spec_hash as terminal_suite_scientific_spec_hash,
    terminal_validation_descriptor_hash,
)


REFERENCE_B_WORKER_INPUT_SCHEMA = "terminal_reference_b_worker_input_v1"
REFERENCE_B_TRACED_OUTPUT_SCHEMA = "terminal_reference_b_traced_worker_output_v1"
REFERENCE_B_SOURCE_OUTPUT_SCHEMA = "terminal_reference_b_source_worker_output_v1"
REFERENCE_B_WORKER_ERROR_SCHEMA = "terminal_reference_b_worker_error_v1"
REFERENCE_B_WORKER_ROLES = ("traced", "source_validation")
# Leave time for parent-side comparison and atomic task publication below the
# unchanged 7,200-second scheduler limit.
REFERENCE_B_WORKER_TIMEOUT_SECONDS = 5_900.0
REFERENCE_B_MAX_FRAME_BYTES = 2 * 1024**3
_FRAME_HEADER = struct.Struct(">Q")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKER_MODULE = "src.experiments.terminal_reference_b_process"
_WORKER_BOOTSTRAP = (
    "import sys;"
    "sys.path.insert(0,sys.argv[1]);"
    "from src.experiments.terminal_reference_b_process import _worker_main;"
    "raise SystemExit(_worker_main(sys.argv[2:]))"
)
_WORKER_SOURCE_PATHS = (
    "src/experiments/terminal_reference_b_process.py",
    "src/experiments/terminal_validation_suite.py",
    "src/mdp/finite_support.py",
    "src/mdp/meta_mdp.py",
    "src/solvers/terminal.py",
    "src/solvers/terminal_reference.py",
    "src/solvers/terminal_reference_b.py",
)
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class ReferenceBWorkerEvidence:
    role: str
    command: Tuple[str, ...]
    command_hash: str
    input_hash: str
    output_hash: str
    record_bytes_hash: str
    source_identity_hash: str
    interpreter_identity_hash: str
    peak_rss_bytes: int
    wall_seconds: float


@dataclass(frozen=True)
class ConcurrentReferenceBResult:
    record: TerminalReferenceRecord
    complete_trace: Mapping[str, Any]
    source_validation_proof: TerminalReferenceCrossProcessValidationProof
    traced_worker: ReferenceBWorkerEvidence
    source_worker: ReferenceBWorkerEvidence
    coordinator_peak_rss_bytes: int


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical IPC cannot contain a non-finite float")
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical IPC mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical IPC type: {type(value).__name__}")


def _encoded_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_ipc_bytes(value: Any) -> bytes:
    encoded = _canonical_value(value)
    if not isinstance(encoded, Mapping):
        raise TypeError("canonical IPC root must be a mapping")
    return _encoded_bytes(encoded)


def _decode_canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise ValueError("raw JSON floats are forbidden in canonical IPC")
    if isinstance(value, list):
        return tuple(_decode_canonical(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"float_hex"}:
            token = value["float_hex"]
            if type(token) is not str:
                raise ValueError("float_hex must be a string")
            parsed = float.fromhex(token)
            if not math.isfinite(parsed) or parsed.hex() != token:
                raise ValueError("float_hex is not canonical finite binary64")
            return parsed
        return {key: _decode_canonical(item) for key, item in value.items()}
    raise ValueError("unsupported canonical IPC value")


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate canonical IPC key: {key}")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("worker payload is not valid canonical JSON") from error
    if not isinstance(value, Mapping) or _encoded_bytes(value) != payload:
        raise ValueError("worker payload bytes are not canonical")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unhashed = dict(payload)
    unhashed[field] = ""
    return _sha256_bytes(_encoded_bytes(unhashed))


def _output_authentication(payload: Mapping[str, Any], key: bytes) -> str:
    unsigned = dict(payload)
    unsigned["output_hash"] = ""
    unsigned["authentication_tag"] = ""
    return hmac.new(key, _encoded_bytes(unsigned), hashlib.sha256).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_fields(payload: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    if set(payload) != set(expected):
        raise ValueError(f"{name} fields differ from the exact schema")


def _read_exact(fd: int, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = os.read(fd, min(1024 * 1024, size - len(chunks)))
        if not block:
            raise EOFError("canonical IPC frame is truncated")
        chunks.extend(block)
    return bytes(chunks)


def _read_frame(fd: int) -> bytes:
    header = _read_exact(fd, _FRAME_HEADER.size)
    size = _FRAME_HEADER.unpack(header)[0]
    if size <= 0 or size > REFERENCE_B_MAX_FRAME_BYTES:
        raise ValueError("canonical IPC frame size is invalid")
    payload = _read_exact(fd, size)
    if os.read(fd, 1):
        raise ValueError("canonical IPC contains duplicate or trailing payload")
    return payload


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BrokenPipeError("canonical IPC write did not make progress")
        view = view[written:]


def _write_frame(fd: int, payload: bytes) -> None:
    if not 0 < len(payload) <= REFERENCE_B_MAX_FRAME_BYTES:
        raise ValueError("canonical IPC frame size is invalid")
    _write_all(fd, _FRAME_HEADER.pack(len(payload)) + payload)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_reference_b_source_identity() -> Mapping[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *_WORKER_SOURCE_PATHS],
        cwd=_PROJECT_ROOT,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Reference-B workers require clean committed source files")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=_PROJECT_ROOT, text=True
    ).strip()
    source_hashes = tuple(
        (relative, _file_sha256(_PROJECT_ROOT / relative))
        for relative in _WORKER_SOURCE_PATHS
    )
    payload: Dict[str, Any] = {
        "schema": "terminal_reference_b_worker_source_identity_v1",
        "commit": commit,
        "tree": tree,
        "source_hashes": source_hashes,
        "identity_hash": "",
    }
    encoded = _canonical_value(payload)
    encoded["identity_hash"] = _payload_hash(encoded, "identity_hash")
    return encoded


def capture_reference_b_interpreter_identity() -> Mapping[str, Any]:
    executable = Path(sys.executable).resolve()
    payload: Dict[str, Any] = {
        "schema": "terminal_reference_b_worker_interpreter_identity_v1",
        "executable": str(executable),
        "executable_sha256": _file_sha256(executable),
        "implementation": sys.implementation.name,
        "version": tuple(sys.version_info[:5]),
        "identity_hash": "",
    }
    encoded = _canonical_value(payload)
    encoded["identity_hash"] = _payload_hash(encoded, "identity_hash")
    return encoded


def _worker_environment() -> Mapping[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.update({name: "1" for name in _THREAD_ENVIRONMENT})
    return environment


def _worker_command(role: str, input_fd: int, output_fd: int) -> Tuple[str, ...]:
    if role not in REFERENCE_B_WORKER_ROLES:
        raise ValueError("unknown Reference-B worker role")
    return (
        sys.executable,
        "-I",
        "-B",
        "-c",
        _WORKER_BOOTSTRAP,
        str(_PROJECT_ROOT),
        role,
        str(input_fd),
        str(output_fd),
    )


def _launch_worker(
    command: Sequence[str], input_read_fd: int, output_write_fd: int
) -> subprocess.Popen:
    return subprocess.Popen(
        tuple(command),
        cwd=_PROJECT_ROOT,
        env=dict(_worker_environment()),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(input_read_fd, output_write_fd),
    )


def _command_hash(command: Sequence[str]) -> str:
    return _sha256_bytes(canonical_ipc_bytes({"argv": tuple(command)}))


def _production_certificate_hash(production: TerminalOptimizationResult) -> str:
    return _sha256_bytes(canonical_ipc_bytes({"certificate": production}))


def _record_bytes(record: TerminalReferenceRecord) -> bytes:
    return canonical_ipc_bytes({"record": record})


def _decode_structural_symmetry(payload: Mapping[str, Any]) -> StructuralSymmetry:
    decoded = _decode_canonical(payload)
    if not isinstance(decoded, Mapping):
        raise ValueError("structural symmetry must be a mapping")
    _require_fields(decoded, tuple(field.name for field in fields(StructuralSymmetry)), "structural symmetry")
    return StructuralSymmetry(**decoded)


def _decode_production(payload: Mapping[str, Any]) -> TerminalOptimizationResult:
    decoded = _decode_canonical(payload)
    if not isinstance(decoded, dict):
        raise ValueError("production certificate must be a mapping")
    _require_fields(decoded, tuple(field.name for field in fields(TerminalOptimizationResult)), "production certificate")
    decoded["structural_symmetry"] = _decode_structural_symmetry(payload["structural_symmetry"])
    return TerminalOptimizationResult(**decoded)


def _decode_record(payload: Mapping[str, Any]) -> TerminalReferenceRecord:
    decoded = _decode_canonical(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Reference-B record must be a mapping")
    _require_fields(decoded, tuple(field.name for field in fields(TerminalReferenceRecord)), "Reference-B record")
    decoded["candidate_isolation_evidence"] = tuple(
        CandidateIsolationEvidence(**item)
        for item in decoded["candidate_isolation_evidence"]
    )
    decoded["structural_symmetry"] = _decode_structural_symmetry(payload["structural_symmetry"])
    record = TerminalReferenceRecord(**decoded)
    if terminal_reference_certificate_hash(record) != record.certificate_hash:
        raise ValueError("Reference-B record certificate hash mismatch")
    return record


def _decode_descriptor(payload: Mapping[str, Any]) -> TerminalValidationDescriptor:
    decoded = _decode_canonical(payload)
    if not isinstance(decoded, dict):
        raise ValueError("descriptor must be a mapping")
    _require_fields(decoded, tuple(field.name for field in fields(TerminalValidationDescriptor)), "descriptor")
    decoded["history"] = tuple(TerminalHistoryStep(**item) for item in decoded["history"])
    descriptor = TerminalValidationDescriptor(**decoded)
    if terminal_validation_descriptor_hash(descriptor) != descriptor.descriptor_hash:
        raise ValueError("descriptor hash mismatch")
    return descriptor


def _decode_source(payload: Mapping[str, Any]) -> Tuple[FiniteSupportMetaMDP, FiniteSupportBeliefState]:
    _require_fields(payload, ("config", "prior", "belief"), "Reference-B source")
    config_values = _decode_canonical(payload["config"])
    prior_values = _decode_canonical(payload["prior"])
    belief_values = _decode_canonical(payload["belief"])
    if not all(isinstance(item, Mapping) for item in (config_values, prior_values, belief_values)):
        raise ValueError("Reference-B source components must be mappings")
    _require_fields(config_values, tuple(field.name for field in fields(EnvironmentConfig)), "environment config")
    _require_fields(prior_values, ("states", "weights"), "finite-support prior")
    _require_fields(belief_values, ("weights", "deliberation_time", "history"), "finite-support belief")
    states = tuple(FiniteSupportAtom(**item) for item in prior_values["states"])
    prior = FiniteSupportPrior(states, tuple(prior_values["weights"]))
    mdp = FiniteSupportMetaMDP(EnvironmentConfig(**config_values), prior)
    belief = FiniteSupportBeliefState(
        prior.states,
        tuple(belief_values["weights"]),
        float(belief_values["deliberation_time"]),
        list(belief_values["history"]),
    )
    return mdp, belief


def _source_payload(mdp: Any, belief: Any) -> Mapping[str, Any]:
    if not isinstance(mdp, FiniteSupportMetaMDP) or not isinstance(belief, FiniteSupportBeliefState):
        raise TypeError("concurrent Reference B requires the finite-support terminal source")
    return {
        "config": mdp.config,
        "prior": {"states": mdp.prior.states, "weights": mdp.prior.weights},
        "belief": {
            "weights": belief.weights,
            "deliberation_time": float(belief.deliberation_time),
            "history": tuple(belief.history),
        },
    }


def _descriptor_source_failures(
    descriptor: TerminalValidationDescriptor,
    mdp: FiniteSupportMetaMDP,
    belief: FiniteSupportBeliefState,
) -> Tuple[str, ...]:
    """Independently bind a worker's reconstructed source to its descriptor."""

    failures = []
    try:
        history = tuple(
            TerminalHistoryStep(
                action="sample_1" if float(step["action"]) == 1.0 else "sample_2",
                observation=float(step["observation"]),
                cost=float(step["cost"]),
            )
            for step in belief.history
        )
        history_hash = canonical_hash(history)
        belief_hash = canonical_hash({
            "support_hash": mdp.prior.support_hash,
            "posterior_weights": tuple(belief.weights),
            "deliberation_time": belief.deliberation_time,
            "history_hash": history_hash,
        })
        checks = (
            ("descriptor_hash", descriptor.descriptor_hash == terminal_validation_descriptor_hash(descriptor)),
            ("support_hash", descriptor.support_hash == mdp.prior.support_hash),
            ("sigma_sample", descriptor.sigma_sample.hex() == float(mdp.config.sigma_sample).hex()),
            ("sample_time_cost", descriptor.sample_time_cost.hex() == float(mdp.config.sample_time_cost).hex()),
            ("deliberation_time", descriptor.deliberation_time.hex() == float(belief.deliberation_time).hex()),
            ("posterior_weight_hash", descriptor.posterior_weight_hash == canonical_hash(tuple(belief.weights))),
            ("history_hash", descriptor.history_hash == history_hash),
            ("canonical_belief_hash", descriptor.canonical_belief_hash == belief_hash),
            ("scientific_spec_hash", descriptor.scientific_spec_hash == terminal_suite_scientific_spec_hash()),
        )
        failures.extend(name for name, passed in checks if not passed)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        failures.append("descriptor_source_binding_error")
    return tuple(dict.fromkeys(failures))


def _build_input(
    *,
    role: str,
    command: Sequence[str],
    descriptor: TerminalValidationDescriptor,
    mdp: Any,
    belief: Any,
    production: TerminalOptimizationResult,
    evaluation_cap: int,
    source_identity: Mapping[str, Any],
    interpreter_identity: Mapping[str, Any],
) -> Tuple[bytes, str]:
    payload: Dict[str, Any] = {
        "schema": REFERENCE_B_WORKER_INPUT_SCHEMA,
        "role": role,
        "descriptor": descriptor,
        "source": _source_payload(mdp, belief),
        "mdp_identity_hash": terminal_mdp_identity_hash(mdp),
        "belief_identity_hash": terminal_belief_identity_hash(belief),
        "scientific_spec_hash": terminal_scientific_spec_hash(mdp),
        "numerical_method_config_hash": terminal_reference_b_numerical_method_config_hash(evaluation_cap),
        "evaluation_cap": evaluation_cap,
        "production_certificate": production,
        "production_certificate_hash": _production_certificate_hash(production),
        "production_allocation_hex": production.allocation.hex(),
        "source_identity": source_identity,
        "interpreter_identity": interpreter_identity,
        "worker_command_hash": _command_hash(command),
        "worker_environment": _worker_environment(),
        "authentication_key_hex": os.urandom(32).hex(),
        "input_hash": "",
    }
    encoded = _canonical_value(payload)
    encoded["input_hash"] = _payload_hash(encoded, "input_hash")
    data = _encoded_bytes(encoded)
    return data, encoded["input_hash"]


def _validate_input(payload: Mapping[str, Any], role: str) -> Tuple[
    TerminalValidationDescriptor,
    FiniteSupportMetaMDP,
    FiniteSupportBeliefState,
    TerminalOptimizationResult,
    int,
]:
    expected = (
        "schema", "role", "descriptor", "source", "mdp_identity_hash",
        "belief_identity_hash", "scientific_spec_hash",
        "numerical_method_config_hash", "evaluation_cap",
        "production_certificate", "production_certificate_hash",
        "production_allocation_hex", "source_identity", "interpreter_identity",
        "worker_command_hash", "worker_environment", "authentication_key_hex",
        "input_hash",
    )
    _require_fields(payload, expected, "Reference-B worker input")
    if payload["schema"] != REFERENCE_B_WORKER_INPUT_SCHEMA or payload["role"] != role:
        raise ValueError("Reference-B worker role or input schema mismatch")
    if not _is_sha256(payload["input_hash"]) or _payload_hash(payload, "input_hash") != payload["input_hash"]:
        raise ValueError("Reference-B worker input hash mismatch")
    if payload["worker_environment"] != _canonical_value(_worker_environment()):
        raise ValueError("Reference-B worker environment mismatch")
    if payload["worker_command_hash"] != _command_hash(tuple(sys.orig_argv)):
        raise ValueError("Reference-B worker command mismatch")
    if payload["source_identity"] != capture_reference_b_source_identity():
        raise ValueError("Reference-B worker source identity mismatch")
    if payload["interpreter_identity"] != capture_reference_b_interpreter_identity():
        raise ValueError("Reference-B worker interpreter identity mismatch")
    authentication_key_hex = payload["authentication_key_hex"]
    if (
        type(authentication_key_hex) is not str
        or len(authentication_key_hex) != 64
        or any(character not in "0123456789abcdef" for character in authentication_key_hex)
    ):
        raise ValueError("Reference-B worker authentication key is invalid")
    descriptor = _decode_descriptor(payload["descriptor"])
    mdp, belief = _decode_source(payload["source"])
    descriptor_failures = _descriptor_source_failures(descriptor, mdp, belief)
    if descriptor_failures:
        raise ValueError(
            "Reference-B worker descriptor/source mismatch: "
            + ",".join(descriptor_failures)
        )
    production = _decode_production(payload["production_certificate"])
    if _production_certificate_hash(production) != payload["production_certificate_hash"]:
        raise ValueError("production certificate hash mismatch")
    if production.allocation.hex() != payload["production_allocation_hex"]:
        raise ValueError("production allocation mismatch")
    evaluation_cap = payload["evaluation_cap"]
    if type(evaluation_cap) is not int or not 1 <= evaluation_cap <= REFERENCE_B_EVALUATION_CAP:
        raise ValueError("Reference-B evaluation cap is invalid")
    expected_identities = (
        ("mdp_identity_hash", terminal_mdp_identity_hash(mdp)),
        ("belief_identity_hash", terminal_belief_identity_hash(belief)),
        ("scientific_spec_hash", terminal_scientific_spec_hash(mdp)),
        ("numerical_method_config_hash", terminal_reference_b_numerical_method_config_hash(evaluation_cap)),
    )
    for name, expected_value in expected_identities:
        if payload[name] != expected_value:
            raise ValueError(f"Reference-B worker {name} mismatch")
    return descriptor, mdp, belief, production, evaluation_cap


def _worker_output(
    *,
    role: str,
    input_hash: str,
    record: TerminalReferenceRecord,
    trace: Optional[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
    interpreter_identity: Mapping[str, Any],
    authentication_key: bytes,
    started: float,
) -> bytes:
    schema = (
        REFERENCE_B_TRACED_OUTPUT_SCHEMA
        if role == "traced"
        else REFERENCE_B_SOURCE_OUTPUT_SCHEMA
    )
    payload: Dict[str, Any] = {
        "schema": schema,
        "role": role,
        "input_hash": input_hash,
        "source_identity_hash": source_identity["identity_hash"],
        "interpreter_identity_hash": interpreter_identity["identity_hash"],
        "record": record,
        "record_bytes_hash": _sha256_bytes(_record_bytes(record)),
        "peak_rss_bytes": _peak_rss_bytes(),
        "wall_seconds": time.perf_counter() - started,
        "authentication_tag": "",
        "output_hash": "",
    }
    if role == "traced":
        payload["complete_trace"] = trace
    encoded = _canonical_value(payload)
    encoded["authentication_tag"] = _output_authentication(
        encoded, authentication_key
    )
    encoded["output_hash"] = _payload_hash(encoded, "output_hash")
    return _encoded_bytes(encoded)


def _worker_main(arguments: Sequence[str]) -> int:
    if len(arguments) != 3:
        return 64
    role, input_token, output_token = arguments
    if role not in REFERENCE_B_WORKER_ROLES:
        return 64
    try:
        input_fd = int(input_token)
        output_fd = int(output_token)
    except ValueError:
        return 64
    started = time.perf_counter()
    try:
        input_bytes = _read_frame(input_fd)
        os.close(input_fd)
        payload = _parse_canonical_json(input_bytes)
        _, mdp, belief, production, evaluation_cap = _validate_input(payload, role)
        if role == "traced":
            record, trace = solve_terminal_reference_b_with_trace(
                mdp, belief, production.allocation, evaluation_cap=evaluation_cap
            )
        else:
            record = solve_terminal_reference_b(
                mdp, belief, production.allocation, evaluation_cap=evaluation_cap
            )
            trace = None
        output = _worker_output(
            role=role,
            input_hash=payload["input_hash"],
            record=record,
            trace=trace,
            source_identity=payload["source_identity"],
            interpreter_identity=payload["interpreter_identity"],
            authentication_key=bytes.fromhex(payload["authentication_key_hex"]),
            started=started,
        )
        _write_frame(output_fd, output)
        os.close(output_fd)
        return 0
    except BaseException as error:
        try:
            failure = canonical_ipc_bytes({
                "schema": REFERENCE_B_WORKER_ERROR_SCHEMA,
                "role": role,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            _write_frame(output_fd, failure)
            os.close(output_fd)
        except BaseException:
            pass
        return 70


def _decode_worker_output(
    payload_bytes: bytes,
    *,
    role: str,
    expected_input_hash: str,
    expected_source_identity_hash: str,
    expected_interpreter_identity_hash: str,
    authentication_key: bytes,
    command: Sequence[str],
) -> Tuple[TerminalReferenceRecord, Optional[Mapping[str, Any]], ReferenceBWorkerEvidence]:
    payload = _parse_canonical_json(payload_bytes)
    if payload.get("schema") == REFERENCE_B_WORKER_ERROR_SCHEMA:
        raise RuntimeError(
            f"Reference-B {role} worker failed: {payload.get('error_type')}:{payload.get('error_message')}"
        )
    expected_fields = [
        "schema", "role", "input_hash", "source_identity_hash",
        "interpreter_identity_hash", "record", "record_bytes_hash",
        "peak_rss_bytes", "wall_seconds", "output_hash",
        "authentication_tag",
    ]
    if role == "traced":
        expected_fields.append("complete_trace")
    _require_fields(payload, expected_fields, f"Reference-B {role} worker output")
    expected_schema = REFERENCE_B_TRACED_OUTPUT_SCHEMA if role == "traced" else REFERENCE_B_SOURCE_OUTPUT_SCHEMA
    if payload["schema"] != expected_schema or payload["role"] != role:
        raise ValueError("Reference-B worker output role or schema mismatch")
    if not _is_sha256(payload["output_hash"]) or _payload_hash(payload, "output_hash") != payload["output_hash"]:
        raise ValueError("Reference-B worker output hash mismatch")
    if (
        not _is_sha256(payload["authentication_tag"])
        or not hmac.compare_digest(
            payload["authentication_tag"],
            _output_authentication(payload, authentication_key),
        )
    ):
        raise ValueError("Reference-B worker output authentication mismatch")
    if payload["input_hash"] != expected_input_hash:
        raise ValueError("Reference-B worker output input binding mismatch")
    if payload["source_identity_hash"] != expected_source_identity_hash:
        raise ValueError("Reference-B worker output source identity mismatch")
    if payload["interpreter_identity_hash"] != expected_interpreter_identity_hash:
        raise ValueError("Reference-B worker output interpreter identity mismatch")
    record = _decode_record(payload["record"])
    record_hash = _sha256_bytes(_record_bytes(record))
    if payload["record_bytes_hash"] != record_hash:
        raise ValueError("Reference-B worker record bytes hash mismatch")
    peak_rss = payload["peak_rss_bytes"]
    wall_seconds = _decode_canonical(payload["wall_seconds"])
    if type(peak_rss) is not int or peak_rss <= 0:
        raise ValueError("Reference-B worker peak RSS is invalid")
    if type(wall_seconds) is not float or not math.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise ValueError("Reference-B worker wall time is invalid")
    trace = None
    if role == "traced":
        trace = _decode_canonical(payload["complete_trace"])
        if not isinstance(trace, Mapping):
            raise ValueError("Reference-B traced worker omitted its complete trace")
        _validate_complete_trace(trace, record)
    evidence = ReferenceBWorkerEvidence(
        role=role,
        command=tuple(command),
        command_hash=_command_hash(command),
        input_hash=expected_input_hash,
        output_hash=payload["output_hash"],
        record_bytes_hash=record_hash,
        source_identity_hash=expected_source_identity_hash,
        interpreter_identity_hash=expected_interpreter_identity_hash,
        peak_rss_bytes=peak_rss,
        wall_seconds=wall_seconds,
    )
    return record, trace, evidence


def _validate_complete_trace(
    trace: Mapping[str, Any], record: TerminalReferenceRecord
) -> None:
    """Bind the complete deterministic B trace to its returned certificate."""

    _require_fields(
        trace,
        ("schema", "complete", "evaluation_cap", "precision_levels", "objective_cache"),
        "Reference-B complete trace",
    )
    if (
        trace["schema"] != "terminal_reference_b_complete_trace_v1"
        or trace["complete"] is not True
        or trace["evaluation_cap"] != record.evaluation_cap
    ):
        raise ValueError("Reference-B complete trace header differs from its record")
    levels = trace["precision_levels"]
    cache = trace["objective_cache"]
    if not isinstance(levels, tuple) or not isinstance(cache, tuple):
        raise ValueError("Reference-B complete trace collections are malformed")
    if len(cache) != record.objective_evaluation_count:
        raise ValueError("Reference-B trace objective count differs from its record")
    allocations = []
    for item in cache:
        numeric = tuple(float(value) for value in item) if (
            isinstance(item, tuple)
            and len(item) == 4
            and all(type(value) in (int, float) for value in item)
        ) else ()
        if (
            len(numeric) != 4
            or any(not math.isfinite(value) for value in numeric)
            or not 0.0 <= numeric[0] <= 1.0
            or numeric[1] > numeric[2]
            or not numeric[1] <= numeric[3] <= numeric[2]
        ):
            raise ValueError("Reference-B trace objective cache is malformed")
        allocations.append(numeric[0].hex())
    if len(set(allocations)) != len(allocations):
        raise ValueError("Reference-B trace objective cache contains duplicate allocations")
    if record.production_allocation.hex() not in allocations:
        raise ValueError("Reference-B trace omits the production allocation evaluation")
    if levels:
        final = levels[-1]
        if not isinstance(final, Mapping) or final.get("complete") is not True:
            raise ValueError("Reference-B trace final precision level is incomplete")
        termination = final.get("termination")
        if termination == "evaluation_cap_exhausted":
            if record.stopping_reason != "evaluation_cap_exhausted":
                raise ValueError("Reference-B trace termination differs from its record")
        else:
            _require_fields(
                final,
                ("precision", "created_nodes", "pop_events", "snapshot", "complete"),
                "Reference-B complete precision level",
            )
            snapshot = final["snapshot"]
            if not isinstance(snapshot, Mapping):
                raise ValueError("Reference-B trace final snapshot is malformed")
            global_interval = tuple(snapshot.get("global_value_interval", ()))
            if global_interval != record.global_value_interval:
                raise ValueError("Reference-B trace global interval differs from its record")
            if float(final["precision"]).hex() != record.precision_level.hex():
                raise ValueError("Reference-B trace precision differs from its record")


def _terminate_workers(processes: Sequence[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _wait_for_workers(
    processes: Sequence[subprocess.Popen], timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while any(process.poll() is None for process in processes):
        if time.monotonic() >= deadline:
            raise TimeoutError("concurrent Reference-B workers exceeded the fixed timeout")
        time.sleep(min(0.1, max(0.001, timeout_seconds / 10.0)))
    for process in processes:
        if process.returncode != 0:
            raise RuntimeError(
                f"Reference-B worker exited with status {process.returncode}"
            )


def solve_terminal_reference_b_concurrently(
    descriptor: TerminalValidationDescriptor,
    mdp: Any,
    belief: Any,
    production: TerminalOptimizationResult,
    *,
    evaluation_cap: int = REFERENCE_B_EVALUATION_CAP,
    timeout_seconds: float = REFERENCE_B_WORKER_TIMEOUT_SECONDS,
) -> ConcurrentReferenceBResult:
    """Run unchanged traced/source B solves concurrently and compare exact records."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("Reference-B worker timeout must be finite and positive")
    source_identity = capture_reference_b_source_identity()
    interpreter_identity = capture_reference_b_interpreter_identity()
    source_identity_hash = source_identity["identity_hash"]
    interpreter_identity_hash = interpreter_identity["identity_hash"]
    roles = REFERENCE_B_WORKER_ROLES
    processes = []
    parent_outputs: Dict[str, int] = {}
    commands: Dict[str, Tuple[str, ...]] = {}
    input_hashes: Dict[str, str] = {}
    authentication_keys: Dict[str, bytes] = {}
    reader_results: Dict[str, Any] = {}
    reader_errors: Dict[str, BaseException] = {}
    threads = []

    def reader(role: str, fd: int) -> None:
        try:
            reader_results[role] = _read_frame(fd)
        except BaseException as error:
            reader_errors[role] = error
        finally:
            os.close(fd)

    try:
        for role in roles:
            input_read, input_write = os.pipe()
            output_read, output_write = os.pipe()
            for descriptor_fd in (input_read, input_write, output_read, output_write):
                os.set_inheritable(descriptor_fd, False)
            command = _worker_command(role, input_read, output_write)
            input_bytes, input_hash = _build_input(
                role=role,
                command=command,
                descriptor=descriptor,
                mdp=mdp,
                belief=belief,
                production=production,
                evaluation_cap=evaluation_cap,
                source_identity=source_identity,
                interpreter_identity=interpreter_identity,
            )
            process = _launch_worker(command, input_read, output_write)
            processes.append(process)
            commands[role] = command
            input_hashes[role] = input_hash
            parsed_input = _parse_canonical_json(input_bytes)
            authentication_keys[role] = bytes.fromhex(
                parsed_input["authentication_key_hex"]
            )
            parent_outputs[role] = output_read
            os.close(input_read)
            os.close(output_write)
            try:
                _write_frame(input_write, input_bytes)
            finally:
                os.close(input_write)

        for role in roles:
            thread = threading.Thread(
                target=reader, args=(role, parent_outputs[role]), daemon=True
            )
            threads.append(thread)
            thread.start()

        deadline = time.monotonic() + timeout_seconds
        _wait_for_workers(processes, timeout_seconds)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("concurrent Reference-B IPC did not terminate")
        if reader_errors:
            role = sorted(reader_errors)[0]
            raise RuntimeError(f"Reference-B {role} worker IPC failed") from reader_errors[role]
        if set(reader_results) != set(roles):
            raise RuntimeError("Reference-B worker output is missing or duplicated")

        traced_record, trace, traced_evidence = _decode_worker_output(
            reader_results["traced"],
            role="traced",
            expected_input_hash=input_hashes["traced"],
            expected_source_identity_hash=source_identity_hash,
            expected_interpreter_identity_hash=interpreter_identity_hash,
            authentication_key=authentication_keys["traced"],
            command=commands["traced"],
        )
        source_record, source_trace, source_evidence = _decode_worker_output(
            reader_results["source_validation"],
            role="source_validation",
            expected_input_hash=input_hashes["source_validation"],
            expected_source_identity_hash=source_identity_hash,
            expected_interpreter_identity_hash=interpreter_identity_hash,
            authentication_key=authentication_keys["source_validation"],
            command=commands["source_validation"],
        )
        if trace is None or source_trace is not None:
            raise RuntimeError("Reference-B worker trace roles are invalid")
        numerical_hash = terminal_reference_b_numerical_method_config_hash(evaluation_cap)
        for record in (traced_record, source_record):
            if not validate_terminal_reference_b_record_structure(
                record,
                mdp,
                belief,
                scientific_spec_hash=terminal_scientific_spec_hash(mdp),
                numerical_method_config_hash=numerical_hash,
            ):
                raise RuntimeError("Reference-B worker record failed structural source validation")
            if record.production_allocation.hex() != production.allocation.hex():
                raise RuntimeError("Reference-B worker record production allocation mismatch")
        traced_bytes = _record_bytes(traced_record)
        source_bytes = _record_bytes(source_record)
        if traced_bytes != source_bytes:
            raise RuntimeError("independent Reference-B records differ in canonical bytes")
        proof = _mint_terminal_reference_cross_process_validation_proof(
            traced_record,
            source_record,
            mdp,
            belief,
            traced_record_bytes=traced_bytes,
            source_record_bytes=source_bytes,
            scientific_spec_hash=terminal_scientific_spec_hash(mdp),
            numerical_method_config_hash=numerical_hash,
            source_identity_hash=source_identity_hash,
            interpreter_identity_hash=interpreter_identity_hash,
            production_allocation=production.allocation,
            worker_roles=roles,
            worker_command_hashes=(traced_evidence.command_hash, source_evidence.command_hash),
            worker_input_hashes=(traced_evidence.input_hash, source_evidence.input_hash),
        )
        return ConcurrentReferenceBResult(
            traced_record,
            trace,
            proof,
            traced_evidence,
            source_evidence,
            _peak_rss_bytes(),
        )
    except BaseException:
        _terminate_workers(processes)
        raise


__all__ = [
    "ConcurrentReferenceBResult",
    "ReferenceBWorkerEvidence",
    "REFERENCE_B_WORKER_INPUT_SCHEMA",
    "REFERENCE_B_WORKER_ROLES",
    "capture_reference_b_interpreter_identity",
    "capture_reference_b_source_identity",
    "canonical_ipc_bytes",
    "solve_terminal_reference_b_concurrently",
]
