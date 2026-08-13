from __future__ import annotations

"""Interval-aware Terminal Reference A and production-reference validation."""

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import heapq
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .terminal import (
    FROZEN_UTILITY_EXPONENT,
    StructuralSymmetry,
    prove_recipient_swap_symmetry,
    rational_power_bounds_7_20,
    validate_structural_symmetry_proof,
)


REFERENCE_PRECISION_LADDER = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
REFERENCE_A_EVALUATION_CAP = 200_000
REFERENCE_ALLOCATION_INTERVAL_TOLERANCE = 1e-4
TERMINAL_TIE_SCALE = 1e-12
REFERENCE_A_METHOD_VERSION = "terminal_reference_a_v2"
REFERENCE_RECORD_SCHEMA = "terminal_reference_record_v2"
REFERENCE_A_BRANCH_RULE = "kink_partition_branch_and_bound"
REFERENCE_A_CAP_RULE = "unresolved_full_domain_evaluation_cap"
REFERENCE_A_CONSTANT_RULE = "constant_objective_full_domain"
REFERENCE_A_ISOLATION_RULES = frozenset(
    {
        REFERENCE_A_BRANCH_RULE,
        REFERENCE_A_CAP_RULE,
        REFERENCE_A_CONSTANT_RULE,
    }
)
_SOURCE_VALIDATION_PROOF_SEAL = object()

TERMINAL_SCIENTIFIC_CONFIG_FIELDS = (
    "mu_need",
    "sigma_need",
    "sigma_sample",
    "total_time",
    "lambda_shortfall",
    "utility_exponent",
    "alpha",
    "learning_per_unit_of_tutoring",
    "delta_learning_per_unit_tutoring",
    "need_threshold",
    "terminate_cost",
    "sample_time_cost",
    "equal_perception_tolerance",
    "initial_mean_1",
    "initial_mean_2",
    "initial_var_1",
    "initial_var_2",
    "prior_sample_count_1",
    "prior_sample_count_2",
    "max_meta_samples",
)


@dataclass(frozen=True)
class CandidateIsolationEvidence:
    """Auditable evidence for one still-viable global-maximizer interval."""

    allocation_interval: Tuple[float, float]
    value_interval: Tuple[float, float]
    witness_allocation: float
    witness_value: float
    partition_count: int
    maximum_depth: int
    isolation_rule: str


@dataclass(frozen=True)
class TerminalReferenceRecord:
    """Operational terminal-reference certificate defined by the repair contract."""

    reference_name: str
    mdp_identity_hash: str
    belief_identity_hash: str
    scientific_spec_hash: str
    numerical_method_config_hash: str
    status: str
    global_value_interval: Tuple[float, float]
    candidate_allocation_intervals: Tuple[Tuple[float, float], ...]
    candidate_value_intervals: Tuple[Tuple[float, float], ...]
    candidate_isolation_evidence: Tuple[CandidateIsolationEvidence, ...]
    canonical_allocation_interval: Optional[Tuple[float, float]]
    representative_allocation: Optional[float]
    tie_status: str
    structural_symmetry: StructuralSymmetry
    production_allocation: float
    production_value_interval: Tuple[float, float]
    production_regret_interval: Tuple[float, float]
    precision_level: float
    objective_evaluation_count: int
    evaluation_cap: int
    stopping_reason: str
    certificate_hash: str


@dataclass(frozen=True)
class TerminalReferenceSourceValidationProof:
    """Process-local proof that one exact record was recomputed from one source."""

    record: TerminalReferenceRecord
    mdp_object_id: int
    belief_object_id: int
    scientific_spec_hash: str
    numerical_method_config_hash: str
    evaluation_cap: int
    certificate_hash: str
    valid: bool
    _seal: object


@dataclass(frozen=True)
class TerminalReferenceValidationResult:
    status: str
    checks: Tuple[Tuple[str, bool], ...]
    failures: Tuple[str, ...]
    certificate_hash: str


@dataclass(frozen=True)
class _CandidateSummary:
    allocation_interval: Tuple[float, float]
    value_interval: Tuple[float, float]
    witness_allocation: float
    witness_value: float
    partition_count: int
    maximum_depth: int
    isolation_rule: str


@dataclass(frozen=True)
class _SearchSnapshot:
    global_value_interval: Tuple[float, float]
    candidates: Tuple[_CandidateSummary, ...]


@dataclass(frozen=True)
class _ReferenceIdentities:
    mdp_identity_hash: str
    belief_identity_hash: str
    scientific_spec_hash: str
    numerical_method_config_hash: str


class _ReferenceEvaluationCap(RuntimeError):
    pass


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source identity contains a non-finite float")
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {
            field.name: _canonical_identity_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("source-identity mapping keys must be strings")
        return {
            key: _canonical_identity_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_identity_value(item) for item in value)
    raise TypeError(f"unsupported source-identity value: {type(value).__name__}")


def _prior_identity_payload(mdp: Any) -> Dict[str, Any]:
    prior = getattr(mdp, "prior", None)
    if prior is None or not hasattr(prior, "states") or not hasattr(prior, "weights"):
        raise TypeError("Terminal Reference A requires a finite-support prior")
    return {
        "states": _canonical_identity_value(tuple(prior.states)),
        "weights": _canonical_identity_value(tuple(float(value) for value in prior.weights)),
        "support_hash": str(prior.support_hash),
    }


def terminal_mdp_identity_hash(mdp: Any) -> str:
    """Hash the actual finite-support MDP object relevant to terminal evaluation."""

    config = getattr(mdp, "config", None)
    if config is None or not is_dataclass(config):
        raise TypeError("Terminal Reference A requires a dataclass environment config")
    return _canonical_hash(
        {
            "schema": "terminal_mdp_identity_v1",
            "class": f"{type(mdp).__module__}.{type(mdp).__qualname__}",
            "config": _canonical_identity_value(config),
            "prior": _prior_identity_payload(mdp),
        }
    )


def terminal_belief_identity_hash(belief: Any) -> str:
    """Hash the ordered support, posterior weights, time, and recorded belief history."""

    required = ("states", "weights", "deliberation_time", "history")
    if any(not hasattr(belief, name) for name in required):
        raise TypeError("Terminal Reference A requires a finite-support belief")
    return _canonical_hash(
        {
            "schema": "terminal_belief_identity_v1",
            "class": f"{type(belief).__module__}.{type(belief).__qualname__}",
            "states": _canonical_identity_value(tuple(belief.states)),
            "weights": _canonical_identity_value(
                tuple(float(value) for value in belief.weights)
            ),
            "deliberation_time": _canonical_identity_value(
                float(belief.deliberation_time)
            ),
            "history": _canonical_identity_value(list(belief.history)),
        }
    )


def terminal_scientific_spec_hash(mdp: Any) -> str:
    """Hash the frozen scientific projection, excluding numerical/runtime controls."""

    config = getattr(mdp, "config", None)
    if config is None:
        raise TypeError("Terminal Reference A requires an environment config")
    projection = {}
    for field_name in TERMINAL_SCIENTIFIC_CONFIG_FIELDS:
        if not hasattr(config, field_name):
            raise TypeError(f"scientific config is missing {field_name}")
        projection[field_name] = _canonical_identity_value(
            getattr(config, field_name)
        )
    return _canonical_hash(
        {
            "schema": "terminal_scientific_projection_v1",
            "config": projection,
            "prior": _prior_identity_payload(mdp),
            "actions": ("terminate", "sample_1", "sample_2"),
            "allocation_domain": ("0x0.0p+0", "0x1.0000000000000p+0"),
            "utility_family": "signed_power_sum_v1",
        }
    )


def terminal_reference_a_numerical_method_config_hash(
    evaluation_cap: int = REFERENCE_A_EVALUATION_CAP,
) -> str:
    """Hash every numerical control used by this Reference-A implementation."""

    if not 1 <= evaluation_cap <= REFERENCE_A_EVALUATION_CAP:
        raise ValueError("Reference A evaluation cap exceeds the contract budget")
    return _canonical_hash(
        {
            "schema": "terminal_numerical_method_config_v1",
            "method_version": REFERENCE_A_METHOD_VERSION,
            "record_schema": REFERENCE_RECORD_SCHEMA,
            "precision_ladder": tuple(
                _float_token(value) for value in REFERENCE_PRECISION_LADDER
            ),
            "evaluation_cap": int(evaluation_cap),
            "allocation_interval_tolerance": _float_token(
                REFERENCE_ALLOCATION_INTERVAL_TOLERANCE
            ),
            "terminal_tie_scale": _float_token(TERMINAL_TIE_SCALE),
            "isolation_rules": tuple(sorted(REFERENCE_A_ISOLATION_RULES)),
            "bound_method": "outward_rational_7_over_20_separable_v1",
            "tie_method": "directed_tau_and_difference_intervals_v1",
            "source_validation": "deterministic_full_reference_recompute_v1",
        }
    )


def _reference_identities(
    mdp: Any,
    belief: Any,
    evaluation_cap: int,
) -> _ReferenceIdentities:
    return _ReferenceIdentities(
        mdp_identity_hash=terminal_mdp_identity_hash(mdp),
        belief_identity_hash=terminal_belief_identity_hash(belief),
        scientific_spec_hash=terminal_scientific_spec_hash(mdp),
        numerical_method_config_hash=terminal_reference_a_numerical_method_config_hash(
            evaluation_cap
        ),
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _float_token(value: float) -> str:
    return float(value).hex()


def _interval_payload(interval: Optional[Tuple[float, float]]) -> Any:
    if interval is None:
        return None
    return tuple(_float_token(value) for value in interval)


def _symmetry_payload(symmetry: StructuralSymmetry) -> Dict[str, Any]:
    return {
        "valid": symmetry.valid,
        "permutation": symmetry.permutation,
        "reason": symmetry.reason,
        "invariant_field_hashes": symmetry.invariant_field_hashes,
        "invariant_hash": symmetry.invariant_hash,
        "proof_hash": symmetry.proof_hash,
    }


def terminal_reference_certificate_hash(record: TerminalReferenceRecord) -> str:
    payload = {
        "schema": REFERENCE_RECORD_SCHEMA,
        "reference_name": record.reference_name,
        "mdp_identity_hash": record.mdp_identity_hash,
        "belief_identity_hash": record.belief_identity_hash,
        "scientific_spec_hash": record.scientific_spec_hash,
        "numerical_method_config_hash": record.numerical_method_config_hash,
        "status": record.status,
        "global_value_interval": _interval_payload(record.global_value_interval),
        "candidate_allocation_intervals": tuple(
            _interval_payload(interval) for interval in record.candidate_allocation_intervals
        ),
        "candidate_value_intervals": tuple(
            _interval_payload(interval) for interval in record.candidate_value_intervals
        ),
        "candidate_isolation_evidence": tuple(
            {
                "allocation_interval": _interval_payload(evidence.allocation_interval),
                "value_interval": _interval_payload(evidence.value_interval),
                "witness_allocation": _float_token(evidence.witness_allocation),
                "witness_value": _float_token(evidence.witness_value),
                "partition_count": evidence.partition_count,
                "maximum_depth": evidence.maximum_depth,
                "isolation_rule": evidence.isolation_rule,
            }
            for evidence in record.candidate_isolation_evidence
        ),
        "canonical_allocation_interval": _interval_payload(
            record.canonical_allocation_interval
        ),
        "representative_allocation": (
            None
            if record.representative_allocation is None
            else _float_token(record.representative_allocation)
        ),
        "tie_status": record.tie_status,
        "structural_symmetry": _symmetry_payload(record.structural_symmetry),
        "production_allocation": _float_token(record.production_allocation),
        "production_value_interval": _interval_payload(record.production_value_interval),
        "production_regret_interval": _interval_payload(record.production_regret_interval),
        "precision_level": _float_token(record.precision_level),
        "objective_evaluation_count": record.objective_evaluation_count,
        "evaluation_cap": record.evaluation_cap,
        "stopping_reason": record.stopping_reason,
    }
    return _canonical_hash(payload)


def _with_certificate(record: TerminalReferenceRecord) -> TerminalReferenceRecord:
    return replace(record, certificate_hash=terminal_reference_certificate_hash(record))


def _next_up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def _next_down(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _add_up(left: float, right: float) -> float:
    return _next_up(float(left) + float(right))


def _subtract_down(left: float, right: float) -> float:
    return _next_down(float(left) - float(right))


def _subtract_up(left: float, right: float) -> float:
    return _next_up(float(left) - float(right))


def _multiply_down(left: float, right: float) -> float:
    return _next_down(float(left) * float(right))


def _multiply_up(left: float, right: float) -> float:
    return _next_up(float(left) * float(right))


def _fixed_value_interval(value: float) -> Tuple[float, float]:
    if not math.isfinite(value):
        raise RuntimeError("reference fixed-allocation value is non-finite")
    return _next_down(value), _next_up(value)


def _regret_interval(
    global_interval: Tuple[float, float],
    production_interval: Tuple[float, float],
) -> Tuple[float, float]:
    lower = max(0.0, _subtract_down(global_interval[0], production_interval[1]))
    upper = max(0.0, _subtract_up(global_interval[1], production_interval[0]))
    return lower, upper


def _interval_width(interval: Tuple[float, float]) -> float:
    return max(0.0, _subtract_up(interval[1], interval[0]))


def _point_interval_distance(point: float, interval: Tuple[float, float]) -> float:
    if point < interval[0]:
        return max(0.0, _subtract_up(interval[0], point))
    if point > interval[1]:
        return max(0.0, _subtract_up(point, interval[1]))
    return 0.0


def _tau_bounds(global_interval: Tuple[float, float]) -> Tuple[float, float]:
    lower, upper = global_interval
    if lower <= 0.0 <= upper:
        minimum_absolute = 0.0
    else:
        minimum_absolute = min(abs(lower), abs(upper))
    tau_low = max(
        0.0,
        _multiply_down(
            TERMINAL_TIE_SCALE,
            max(1.0, minimum_absolute),
        ),
    )
    tau_high = _multiply_up(
        TERMINAL_TIE_SCALE,
        max(1.0, abs(lower), abs(upper)),
    )
    return tau_low, tau_high


def _absolute_difference_interval(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[float, float]:
    lower = max(
        0.0,
        _subtract_down(first[0], second[1]),
        _subtract_down(second[0], first[1]),
    )
    endpoint_1 = (
        _subtract_down(first[0], second[1]),
        _subtract_up(first[0], second[1]),
    )
    endpoint_2 = (
        _subtract_down(first[1], second[0]),
        _subtract_up(first[1], second[0]),
    )
    upper = max(
        abs(endpoint_1[0]),
        abs(endpoint_1[1]),
        abs(endpoint_2[0]),
        abs(endpoint_2[1]),
    )
    return lower, upper


def _strictly_dominates(
    first: _CandidateSummary,
    second: _CandidateSummary,
    tau_high: float,
) -> bool:
    return first.value_interval[0] > _add_up(second.value_interval[1], tau_high)


def _canonical_ordinary_candidate_index(
    candidates: Sequence[_CandidateSummary],
) -> int:
    return min(
        range(len(candidates)),
        key=lambda index: (
            0.0
            if candidates[index].allocation_interval[0]
            <= 0.5
            <= candidates[index].allocation_interval[1]
            else min(
                abs(candidates[index].allocation_interval[0] - 0.5),
                abs(candidates[index].allocation_interval[1] - 0.5),
            ),
            candidates[index].allocation_interval[0],
        ),
    )


def _is_structural_interval_pair(
    symmetry: StructuralSymmetry,
    candidates: Sequence[_CandidateSummary],
) -> bool:
    if not symmetry.valid or len(candidates) != 2:
        return False
    left, right = sorted(candidates, key=lambda candidate: candidate.allocation_interval)
    left_interval = left.allocation_interval
    right_interval = right.allocation_interval
    tolerance = REFERENCE_ALLOCATION_INTERVAL_TOLERANCE
    if left_interval[1] >= 0.5 or right_interval[0] <= 0.5:
        return False
    return (
        abs(left_interval[0] - (1.0 - right_interval[1])) <= tolerance
        and abs(left_interval[1] - (1.0 - right_interval[0])) <= tolerance
    )


def _resolve_candidates(
    snapshot: _SearchSnapshot,
    symmetry: StructuralSymmetry,
) -> Tuple[Optional[str], Tuple[_CandidateSummary, ...], Optional[int], str]:
    candidates = snapshot.candidates
    if len(candidates) == 1:
        if (
            _interval_width(candidates[0].allocation_interval)
            <= REFERENCE_ALLOCATION_INTERVAL_TOLERANCE
        ):
            return "unique", candidates, 0, "unique_candidate_isolated"
        return None, candidates, None, "connected_maximizer_region_provisional"
    if _is_structural_interval_pair(symmetry, candidates):
        canonical = min(
            range(2),
            key=lambda index: candidates[index].allocation_interval,
        )
        return (
            "structural_symmetry_tie",
            candidates,
            canonical,
            "single_structural_mirror_orbit_certified",
        )

    tau_low, tau_high = _tau_bounds(snapshot.global_value_interval)
    dominant: List[int] = []
    for index, candidate in enumerate(candidates):
        if all(
            index == other_index
            or _strictly_dominates(candidate, other, tau_high)
            for other_index, other in enumerate(candidates)
        ):
            dominant.append(index)
    if len(dominant) == 1:
        index = dominant[0]
        return (
            "unique",
            (candidates[index],),
            0,
            "ordinary_value_order_certified",
        )

    pairwise = [
        _absolute_difference_interval(first.value_interval, second.value_interval)
        for first_index, first in enumerate(candidates)
        for second in candidates[first_index + 1 :]
    ]
    if pairwise and all(interval[1] <= tau_low for interval in pairwise):
        canonical = _canonical_ordinary_candidate_index(candidates)
        return (
            "certified_value_tie",
            candidates,
            canonical,
            "ordinary_value_tie_certified",
        )
    return None, candidates, None, "ordinary_tie_provisional"


def _candidate_evidence(
    candidates: Sequence[_CandidateSummary],
) -> Tuple[CandidateIsolationEvidence, ...]:
    return tuple(
        CandidateIsolationEvidence(
            allocation_interval=candidate.allocation_interval,
            value_interval=candidate.value_interval,
            witness_allocation=candidate.witness_allocation,
            witness_value=candidate.witness_value,
            partition_count=candidate.partition_count,
            maximum_depth=candidate.maximum_depth,
            isolation_rule=candidate.isolation_rule,
        )
        for candidate in candidates
    )


def _resolved_record(
    *,
    reference_name: str,
    snapshot: _SearchSnapshot,
    candidates: Sequence[_CandidateSummary],
    canonical_index: int,
    tie_status: str,
    identities: _ReferenceIdentities,
    symmetry: StructuralSymmetry,
    production_allocation: float,
    production_value: float,
    precision: float,
    evaluation_count: int,
    evaluation_cap: int,
    stopping_reason: str,
) -> TerminalReferenceRecord:
    canonical = candidates[canonical_index]
    production_interval = _fixed_value_interval(production_value)
    record = TerminalReferenceRecord(
        reference_name=reference_name,
        mdp_identity_hash=identities.mdp_identity_hash,
        belief_identity_hash=identities.belief_identity_hash,
        scientific_spec_hash=identities.scientific_spec_hash,
        numerical_method_config_hash=identities.numerical_method_config_hash,
        status="resolved",
        global_value_interval=snapshot.global_value_interval,
        candidate_allocation_intervals=tuple(
            candidate.allocation_interval for candidate in candidates
        ),
        candidate_value_intervals=tuple(candidate.value_interval for candidate in candidates),
        candidate_isolation_evidence=_candidate_evidence(candidates),
        canonical_allocation_interval=canonical.allocation_interval,
        representative_allocation=canonical.witness_allocation,
        tie_status=tie_status,
        structural_symmetry=symmetry,
        production_allocation=float(production_allocation),
        production_value_interval=production_interval,
        production_regret_interval=_regret_interval(
            snapshot.global_value_interval,
            production_interval,
        ),
        precision_level=float(precision),
        objective_evaluation_count=int(evaluation_count),
        evaluation_cap=int(evaluation_cap),
        stopping_reason=stopping_reason,
        certificate_hash="",
    )
    return _with_certificate(record)


def _unresolved_record(
    *,
    reference_name: str,
    identities: _ReferenceIdentities,
    symmetry: StructuralSymmetry,
    production_allocation: float,
    production_value: float,
    precision: float,
    evaluation_count: int,
    evaluation_cap: int,
    stopping_reason: str,
    snapshot: Optional[_SearchSnapshot],
    fallback_global_upper: float,
) -> TerminalReferenceRecord:
    production_interval = _fixed_value_interval(production_value)
    if snapshot is None:
        if not math.isfinite(fallback_global_upper):
            raise RuntimeError("Reference A unresolved fallback bound is non-finite")
        global_interval = (
            production_interval[0],
            max(production_interval[0], _next_up(fallback_global_upper)),
        )
        candidates = (
            _CandidateSummary(
                (0.0, 1.0),
                global_interval,
                float(production_allocation),
                float(production_value),
                1,
                0,
                REFERENCE_A_CAP_RULE,
            ),
        )
    else:
        global_interval = snapshot.global_value_interval
        candidates = snapshot.candidates
    record = TerminalReferenceRecord(
        reference_name=reference_name,
        mdp_identity_hash=identities.mdp_identity_hash,
        belief_identity_hash=identities.belief_identity_hash,
        scientific_spec_hash=identities.scientific_spec_hash,
        numerical_method_config_hash=identities.numerical_method_config_hash,
        status="reference_unresolved",
        global_value_interval=global_interval,
        candidate_allocation_intervals=tuple(
            candidate.allocation_interval for candidate in candidates
        ),
        candidate_value_intervals=tuple(candidate.value_interval for candidate in candidates),
        candidate_isolation_evidence=_candidate_evidence(candidates),
        canonical_allocation_interval=None,
        representative_allocation=None,
        tie_status="reference_unresolved",
        structural_symmetry=symmetry,
        production_allocation=float(production_allocation),
        production_value_interval=production_interval,
        production_regret_interval=_regret_interval(global_interval, production_interval),
        precision_level=float(precision),
        objective_evaluation_count=int(evaluation_count),
        evaluation_cap=int(evaluation_cap),
        stopping_reason=stopping_reason,
        certificate_hash="",
    )
    return _with_certificate(record)


@dataclass(frozen=True)
class _ANode:
    lower: float
    upper: float
    best_allocation: float
    best_value: float
    upper_bound: float
    depth: int


class _AObjective:
    def __init__(self, mdp: Any, belief: Any, evaluation_cap: int) -> None:
        self.mdp = mdp
        self.belief = belief
        self.evaluation_cap = evaluation_cap
        self.cache: Dict[str, float] = {}
        self.remaining_time = float(mdp.remaining_time_after_termination(belief))
        self.rate_1, self.rate_2 = (float(value) for value in mdp.learning_rates())
        self.resource_1 = self.rate_1 * self.remaining_time
        self.resource_2 = self.rate_2 * self.remaining_time
        self.lambda_shortfall = float(mdp.config.lambda_shortfall)
        self.person_1_kinks: Dict[int, str] = {}
        self.person_2_kinks: Dict[int, str] = {}
        if self.resource_1 > 0.0:
            for index, state in enumerate(belief.states):
                kink = float(state.need_1) / self.resource_1
                if 0.0 <= kink <= 1.0:
                    self.person_1_kinks[index] = kink.hex()
        if self.resource_2 > 0.0:
            for index, state in enumerate(belief.states):
                kink = 1.0 - float(state.need_2) / self.resource_2
                if 0.0 <= kink <= 1.0:
                    self.person_2_kinks[index] = kink.hex()
        if float(mdp.utility_exponent()).hex() != float(FROZEN_UTILITY_EXPONENT).hex():
            raise ValueError("Reference A supports only alpha=0.35=7/20")

    def value(self, allocation: float) -> float:
        allocation = min(1.0, max(0.0, float(allocation)))
        key = allocation.hex()
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.evaluation_cap:
            raise _ReferenceEvaluationCap("Reference A evaluation cap exhausted")
        value = float(self.mdp.expected_terminal_utility(self.belief, allocation))
        if not math.isfinite(value):
            raise RuntimeError("Reference A objective is non-finite")
        self.cache[key] = value
        return value

    def breakpoints(self) -> Tuple[float, ...]:
        points = {0.0, 1.0}
        if self.resource_1 > 0.0:
            points.update(float(state.need_1) / self.resource_1 for state in self.belief.states)
        if self.resource_2 > 0.0:
            points.update(1.0 - float(state.need_2) / self.resource_2 for state in self.belief.states)
        return tuple(sorted(point for point in points if 0.0 <= point <= 1.0))

    @staticmethod
    def _up(value: float) -> float:
        return math.nextafter(float(value), math.inf)

    @staticmethod
    def _down(value: float) -> float:
        return math.nextafter(float(value), -math.inf)

    def _outcome_1_upper(self, index: int, state: Any, allocation: float) -> float:
        if self.person_1_kinks.get(index) == float(allocation).hex():
            return 0.0
        time = self._up(allocation * self.remaining_time)
        learned = self._up(self.rate_1 * time)
        return self._up(learned - float(state.need_1))

    def _outcome_2_upper(self, index: int, state: Any, allocation: float) -> float:
        if self.person_2_kinks.get(index) == float(allocation).hex():
            return 0.0
        fraction = self._up(1.0 - allocation)
        time = self._up(fraction * self.remaining_time)
        learned = self._up(self.rate_2 * time)
        return self._up(learned - float(state.need_2))

    def _utility_upper(self, outcome: float) -> float:
        if outcome < 0.0:
            root_lower, _ = rational_power_bounds_7_20(-outcome)
            magnitude = max(0.0, self._down(self.lambda_shortfall * root_lower))
            return -magnitude
        return rational_power_bounds_7_20(outcome)[1]

    def upper_bound(self, lower: float, upper: float) -> float:
        terms: List[float] = []
        for index, (state, weight_value) in enumerate(
            zip(self.belief.states, self.belief.weights)
        ):
            weight = float(weight_value)
            if weight == 0.0:
                continue
            utility_sum = self._up(
                self._utility_upper(self._outcome_1_upper(index, state, upper))
                + self._utility_upper(self._outcome_2_upper(index, state, lower))
            )
            terms.append(self._up(weight * utility_sum))
        total = math.fsum(terms)
        padding = (
            (96.0 + 24.0 * len(terms))
            * math.ulp(1.0)
            * max(1.0, math.fsum(abs(term) for term in terms))
        )
        return self._up(self._up(total + padding))


def _a_node(objective: _AObjective, lower: float, upper: float, depth: int) -> _ANode:
    midpoint = lower + 0.5 * (upper - lower)
    points = (lower, midpoint, upper)
    values = tuple(objective.value(point) for point in points)
    best_index = max(
        range(3),
        key=lambda index: (values[index], -abs(points[index] - 0.5), -points[index]),
    )
    return _ANode(
        lower,
        upper,
        points[best_index],
        values[best_index],
        max(values[best_index], objective.upper_bound(lower, upper)),
        depth,
    )


def _group_a_nodes(nodes: Sequence[_ANode], global_lower: float) -> Tuple[_CandidateSummary, ...]:
    groups: List[List[_ANode]] = []
    for node in sorted(nodes, key=lambda item: (item.lower, item.upper)):
        if not groups or node.lower > groups[-1][-1].upper + 8.0 * math.ulp(1.0):
            groups.append([node])
        else:
            groups[-1].append(node)
    summaries: List[_CandidateSummary] = []
    for group in groups:
        witness = max(
            group,
            key=lambda node: (
                node.best_value,
                -abs(node.best_allocation - 0.5),
                -node.best_allocation,
            ),
        )
        summaries.append(
            _CandidateSummary(
                (group[0].lower, group[-1].upper),
                (max(node.best_value for node in group), max(node.upper_bound for node in group)),
                witness.best_allocation,
                witness.best_value,
                len(group),
                max(node.depth for node in group),
                REFERENCE_A_BRANCH_RULE,
            )
        )
    return tuple(summaries)


def _run_reference_a_level(
    objective: _AObjective,
    precision: float,
    audit_level: Optional[Dict[str, Any]] = None,
) -> _SearchSnapshot:
    heap: List[Tuple[float, int, _ANode]] = []
    counter = 0
    best = -math.inf
    if audit_level is not None:
        audit_level.update({
            "precision": float(precision),
            "created_nodes": [],
            "pop_events": [],
            "complete": False,
        })

    def record_node(node: _ANode) -> int:
        if audit_level is None:
            return -1
        identifier = len(audit_level["created_nodes"])
        audit_level["created_nodes"].append((
            identifier, node.lower, node.upper, node.best_allocation,
            node.best_value, node.upper_bound, node.depth,
        ))
        return identifier

    def record_event(identifier: int, disposition: str) -> None:
        if audit_level is not None:
            audit_level["pop_events"].append((identifier, disposition))

    for lower, upper in zip(objective.breakpoints()[:-1], objective.breakpoints()[1:]):
        if upper <= lower:
            continue
        node = _a_node(objective, lower, upper, 0)
        node_id = record_node(node)
        best = max(best, node.best_value)
        heapq.heappush(heap, (-node.upper_bound, counter, (node_id, node)))
        counter += 1
    finalized: List[_ANode] = []
    while heap:
        _, _, traced_node = heapq.heappop(heap)
        node_id, node = traced_node
        _, tau_high = _tau_bounds((best, best))
        if node.upper_bound < _subtract_down(best, tau_high):
            record_event(node_id, "pruned_value")
            continue
        if (
            _subtract_up(node.upper_bound, node.best_value) <= precision
            and _interval_width((node.lower, node.upper))
            <= REFERENCE_ALLOCATION_INTERVAL_TOLERANCE
        ):
            finalized.append(node)
            record_event(node_id, "finalized_precision")
            continue
        midpoint = node.lower + 0.5 * (node.upper - node.lower)
        if midpoint in (node.lower, node.upper):
            finalized.append(node)
            record_event(node_id, "finalized_machine_resolution")
            continue
        record_event(node_id, "split")
        for lower, upper in ((node.lower, midpoint), (midpoint, node.upper)):
            child = _a_node(objective, lower, upper, node.depth + 1)
            child_id = record_node(child)
            best = max(best, child.best_value)
            heapq.heappush(heap, (-child.upper_bound, counter, (child_id, child)))
            counter += 1
    _, tau_high = _tau_bounds((best, best))
    viable = [
        node
        for node in finalized
        if node.upper_bound >= _subtract_down(best, tau_high)
    ]
    candidates = _group_a_nodes(viable, best)
    if not candidates:
        raise RuntimeError("Reference A produced no viable candidate interval")
    lower = max(candidate.value_interval[0] for candidate in candidates)
    upper = max(candidate.value_interval[1] for candidate in candidates)
    snapshot = _SearchSnapshot((lower, max(lower, upper)), candidates)
    if audit_level is not None:
        audit_level["created_nodes"] = tuple(audit_level["created_nodes"])
        audit_level["pop_events"] = tuple(audit_level["pop_events"])
        audit_level["snapshot"] = snapshot
        audit_level["complete"] = True
    return snapshot


def solve_terminal_reference_a(
    mdp: Any,
    belief: Any,
    production_allocation: float,
    *,
    evaluation_cap: int = REFERENCE_A_EVALUATION_CAP,
    _audit_trace: Optional[Dict[str, Any]] = None,
) -> TerminalReferenceRecord:
    """Run kink-partitioned Reference A through the frozen precision ladder."""

    if not 1 <= evaluation_cap <= REFERENCE_A_EVALUATION_CAP:
        raise ValueError("Reference A evaluation cap exceeds the contract budget")
    identities = _reference_identities(mdp, belief, evaluation_cap)
    symmetry = prove_recipient_swap_symmetry(mdp, belief)
    objective = _AObjective(mdp, belief, evaluation_cap)
    if _audit_trace is not None:
        _audit_trace.clear()
        _audit_trace.update({
            "schema": "terminal_reference_a_complete_trace_v1",
            "complete": False,
            "evaluation_cap": int(evaluation_cap),
            "precision_levels": [],
            "objective_cache": (),
        })

    def traced_return(record: TerminalReferenceRecord) -> TerminalReferenceRecord:
        if _audit_trace is not None:
            _audit_trace["precision_levels"] = tuple(_audit_trace["precision_levels"])
            _audit_trace["objective_cache"] = tuple(sorted(objective.cache.items()))
            _audit_trace["complete"] = True
        return record
    production_value = objective.value(production_allocation)
    if objective.resource_1 == 0.0 and objective.resource_2 == 0.0:
        constant_interval = _fixed_value_interval(production_value)
        constant_snapshot = _SearchSnapshot(
            global_value_interval=constant_interval,
            candidates=(
                _CandidateSummary(
                    allocation_interval=(0.0, 1.0),
                    value_interval=constant_interval,
                    witness_allocation=float(production_allocation),
                    witness_value=float(production_value),
                    partition_count=1,
                    maximum_depth=0,
                    isolation_rule=REFERENCE_A_CONSTANT_RULE,
                ),
            ),
        )
        return traced_return(_unresolved_record(
            reference_name="terminal_reference_a",
            identities=identities,
            symmetry=symmetry,
            production_allocation=production_allocation,
            production_value=production_value,
            precision=REFERENCE_PRECISION_LADDER[0],
            evaluation_count=len(objective.cache),
            evaluation_cap=evaluation_cap,
            stopping_reason="connected_plateau_requires_multiple_maximizer_rule",
            snapshot=constant_snapshot,
            fallback_global_upper=constant_interval[1],
        ))
    fallback_global_upper = max(
        production_value,
        objective.upper_bound(0.0, 1.0),
    )
    last_snapshot: Optional[_SearchSnapshot] = None
    last_unresolved_reason = "global_value_interval"
    for precision in REFERENCE_PRECISION_LADDER:
        audit_level: Optional[Dict[str, Any]] = {} if _audit_trace is not None else None
        if audit_level is not None:
            _audit_trace["precision_levels"].append(audit_level)
        try:
            snapshot = _run_reference_a_level(objective, precision, audit_level)
        except _ReferenceEvaluationCap:
            if audit_level is not None:
                audit_level["complete"] = True
                audit_level["termination"] = "evaluation_cap_exhausted"
            return traced_return(_unresolved_record(
                reference_name="terminal_reference_a",
                identities=identities,
                symmetry=symmetry,
                production_allocation=production_allocation,
                production_value=production_value,
                precision=precision,
                evaluation_count=len(objective.cache),
                evaluation_cap=evaluation_cap,
                stopping_reason="evaluation_cap_exhausted",
                snapshot=last_snapshot,
                fallback_global_upper=fallback_global_upper,
            ))
        last_snapshot = snapshot
        if _interval_width(snapshot.global_value_interval) > precision:
            last_unresolved_reason = "global_value_interval"
            continue
        tie_status, candidates, canonical_index, reason = _resolve_candidates(snapshot, symmetry)
        if tie_status is not None and canonical_index is not None:
            return traced_return(_resolved_record(
                reference_name="terminal_reference_a",
                snapshot=snapshot,
                candidates=candidates,
                canonical_index=canonical_index,
                tie_status=tie_status,
                identities=identities,
                symmetry=symmetry,
                production_allocation=production_allocation,
                production_value=production_value,
                precision=precision,
                evaluation_count=len(objective.cache),
                evaluation_cap=evaluation_cap,
                stopping_reason=reason,
            ))
        last_unresolved_reason = reason
    return traced_return(_unresolved_record(
        reference_name="terminal_reference_a",
        identities=identities,
        symmetry=symmetry,
        production_allocation=production_allocation,
        production_value=production_value,
        precision=REFERENCE_PRECISION_LADDER[-1],
        evaluation_count=len(objective.cache),
        evaluation_cap=evaluation_cap,
        stopping_reason=f"{last_unresolved_reason}_precision_ladder_exhausted",
        snapshot=last_snapshot,
        fallback_global_upper=fallback_global_upper,
    ))


def solve_terminal_reference_a_with_trace(
    mdp: Any,
    belief: Any,
    production_allocation: float,
    *,
    evaluation_cap: int = REFERENCE_A_EVALUATION_CAP,
) -> Tuple[TerminalReferenceRecord, Dict[str, Any]]:
    """Return Reference A and every deterministic refinement/node event."""

    trace: Dict[str, Any] = {}
    record = solve_terminal_reference_a(
        mdp,
        belief,
        production_allocation,
        evaluation_cap=evaluation_cap,
        _audit_trace=trace,
    )
    return record, trace


def _is_finite_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value)


def _is_finite_interval(value: Any) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(_is_finite_float(item) for item in value)
    )


def _evidence_rule_is_valid(
    record: TerminalReferenceRecord,
    allocation_interval: Tuple[float, float],
    value_interval: Tuple[float, float],
    evidence: CandidateIsolationEvidence,
) -> bool:
    if evidence.isolation_rule not in REFERENCE_A_ISOLATION_RULES:
        return False
    if type(evidence.partition_count) is not int or evidence.partition_count < 1:
        return False
    if type(evidence.maximum_depth) is not int or evidence.maximum_depth < 0:
        return False
    if evidence.partition_count > record.objective_evaluation_count:
        return False
    if evidence.maximum_depth > record.objective_evaluation_count:
        return False

    if evidence.isolation_rule == REFERENCE_A_BRANCH_RULE:
        return (
            float(evidence.witness_value).hex() == float(value_interval[0]).hex()
            and _interval_width(allocation_interval) > 0.0
        )

    return (
        record.status == "reference_unresolved"
        and allocation_interval == (0.0, 1.0)
        and value_interval == record.global_value_interval
        and evidence.partition_count == 1
        and evidence.maximum_depth == 0
        and float(evidence.witness_allocation).hex()
        == float(record.production_allocation).hex()
        and record.production_value_interval[0]
        <= evidence.witness_value
        <= record.production_value_interval[1]
        and (
            (
                evidence.isolation_rule == REFERENCE_A_CAP_RULE
                and record.stopping_reason == "evaluation_cap_exhausted"
            )
            or (
                evidence.isolation_rule == REFERENCE_A_CONSTANT_RULE
                and record.stopping_reason
                == "connected_plateau_requires_multiple_maximizer_rule"
            )
        )
    )


def _validate_terminal_reference_record_shape(record: TerminalReferenceRecord) -> bool:
    if record.reference_name != "terminal_reference_a":
        return False
    if any(
        not _is_sha256(value)
        for value in (
            record.mdp_identity_hash,
            record.belief_identity_hash,
            record.scientific_spec_hash,
            record.numerical_method_config_hash,
            record.certificate_hash,
        )
    ):
        return False
    if record.status not in {"resolved", "reference_unresolved"}:
        return False
    if terminal_reference_certificate_hash(record) != record.certificate_hash:
        return False
    if (
        not _is_finite_float(record.precision_level)
        or record.precision_level not in REFERENCE_PRECISION_LADDER
    ):
        return False
    if type(record.evaluation_cap) is not int:
        return False
    if not 1 <= record.evaluation_cap <= REFERENCE_A_EVALUATION_CAP:
        return False
    if type(record.objective_evaluation_count) is not int:
        return False
    if not 1 <= record.objective_evaluation_count <= record.evaluation_cap:
        return False
    if not _is_finite_float(record.production_allocation):
        return False
    if not 0.0 <= record.production_allocation <= 1.0:
        return False

    top_level_intervals = (
        record.global_value_interval,
        record.production_value_interval,
        record.production_regret_interval,
    )
    if any(
        not _is_finite_interval(interval)
        or interval[0] > interval[1]
        for interval in top_level_intervals
    ):
        return False
    if not (
        len(record.candidate_allocation_intervals)
        == len(record.candidate_value_intervals)
        == len(record.candidate_isolation_evidence)
        > 0
    ):
        return False

    previous_upper = -math.inf
    candidates: List[_CandidateSummary] = []
    for allocation_interval, value_interval, evidence in zip(
        record.candidate_allocation_intervals,
        record.candidate_value_intervals,
        record.candidate_isolation_evidence,
    ):
        if not isinstance(evidence, CandidateIsolationEvidence):
            return False
        if (
            not _is_finite_interval(allocation_interval)
            or not (0.0 <= allocation_interval[0] <= allocation_interval[1] <= 1.0)
        ):
            return False
        if allocation_interval[0] <= previous_upper:
            return False
        if (
            not _is_finite_interval(value_interval)
            or value_interval[0] > value_interval[1]
        ):
            return False
        if not _is_finite_interval(evidence.allocation_interval):
            return False
        if not _is_finite_interval(evidence.value_interval):
            return False
        if not _is_finite_float(evidence.witness_allocation):
            return False
        if not _is_finite_float(evidence.witness_value):
            return False
        if evidence.allocation_interval != allocation_interval:
            return False
        if evidence.value_interval != value_interval:
            return False
        if not allocation_interval[0] <= evidence.witness_allocation <= allocation_interval[1]:
            return False
        if not value_interval[0] <= evidence.witness_value <= value_interval[1]:
            return False
        if not _evidence_rule_is_valid(
            record,
            allocation_interval,
            value_interval,
            evidence,
        ):
            return False
        if value_interval[0] > record.global_value_interval[0]:
            return False
        if value_interval[1] > record.global_value_interval[1]:
            return False
        previous_upper = allocation_interval[1]
        candidates.append(
            _CandidateSummary(
                allocation_interval,
                value_interval,
                evidence.witness_allocation,
                evidence.witness_value,
                evidence.partition_count,
                evidence.maximum_depth,
                evidence.isolation_rule,
            )
        )

    if not any(
        candidate.value_interval[0] == record.global_value_interval[0]
        for candidate in candidates
    ):
        return False
    if not any(
        candidate.value_interval[1] == record.global_value_interval[1]
        for candidate in candidates
    ):
        return False
    expected_regret = _regret_interval(
        record.global_value_interval,
        record.production_value_interval,
    )
    if expected_regret != record.production_regret_interval:
        return False

    if record.status == "reference_unresolved":
        return (
            record.tie_status == "reference_unresolved"
            and record.canonical_allocation_interval is None
            and record.representative_allocation is None
            and record.stopping_reason
            in {
                "evaluation_cap_exhausted",
                "ordinary_tie_provisional_precision_ladder_exhausted",
                "connected_maximizer_region_provisional_precision_ladder_exhausted",
                "global_value_interval_precision_ladder_exhausted",
                "connected_plateau_requires_multiple_maximizer_rule",
            }
        )

    if record.tie_status not in {
        "unique",
        "certified_value_tie",
        "structural_symmetry_tie",
    }:
        return False
    if record.canonical_allocation_interval is None or record.representative_allocation is None:
        return False
    if record.canonical_allocation_interval not in record.candidate_allocation_intervals:
        return False
    canonical_index = record.candidate_allocation_intervals.index(
        record.canonical_allocation_interval
    )
    canonical_evidence = record.candidate_isolation_evidence[canonical_index]
    if (
        float(record.representative_allocation).hex()
        != float(canonical_evidence.witness_allocation).hex()
    ):
        return False
    if not (
        record.canonical_allocation_interval[0]
        <= record.representative_allocation
        <= record.canonical_allocation_interval[1]
    ):
        return False
    if _interval_width(record.global_value_interval) > record.precision_level:
        return False

    if record.tie_status == "unique":
        return (
            len(candidates) == 1
            and _interval_width(candidates[0].allocation_interval)
            <= REFERENCE_ALLOCATION_INTERVAL_TOLERANCE
            and record.stopping_reason
            in {"unique_candidate_isolated", "ordinary_value_order_certified"}
        )

    if len(candidates) < 2:
        return False
    if record.tie_status == "structural_symmetry_tie":
        lower_index = min(
            range(len(candidates)),
            key=lambda index: candidates[index].allocation_interval,
        )
        return (
            record.stopping_reason == "single_structural_mirror_orbit_certified"
            and canonical_index == lower_index
            and _is_structural_interval_pair(record.structural_symmetry, candidates)
        )

    return (
        record.stopping_reason == "ordinary_value_tie_certified"
        and canonical_index == _canonical_ordinary_candidate_index(candidates)
    )


def validate_terminal_reference_record(
    record: TerminalReferenceRecord,
    mdp: Any,
    belief: Any,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
) -> bool:
    """Validate record shape, source identities, and a deterministic A recomputation."""

    try:
        shape_is_valid = _validate_terminal_reference_record_shape(record)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return False
    if not shape_is_valid:
        return False
    try:
        actual_identities = _reference_identities(
            mdp,
            belief,
            record.evaluation_cap,
        )
    except (TypeError, ValueError):
        return False
    supplied_identities = (
        scientific_spec_hash,
        numerical_method_config_hash,
    )
    if any(not _is_sha256(value) for value in supplied_identities):
        return False
    if scientific_spec_hash != actual_identities.scientific_spec_hash:
        return False
    if numerical_method_config_hash != actual_identities.numerical_method_config_hash:
        return False
    if (
        record.mdp_identity_hash != actual_identities.mdp_identity_hash
        or record.belief_identity_hash != actual_identities.belief_identity_hash
        or record.scientific_spec_hash != actual_identities.scientific_spec_hash
        or record.numerical_method_config_hash
        != actual_identities.numerical_method_config_hash
    ):
        return False
    try:
        recomputed = solve_terminal_reference_a(
            mdp,
            belief,
            record.production_allocation,
            evaluation_cap=record.evaluation_cap,
        )
    except (RuntimeError, TypeError, ValueError):
        return False
    return recomputed.certificate_hash == record.certificate_hash


def source_validate_terminal_reference_record(
    record: TerminalReferenceRecord,
    mdp: Any,
    belief: Any,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
) -> TerminalReferenceSourceValidationProof:
    """Recompute once and bind the result to exact process-local source objects."""

    valid = validate_terminal_reference_record(
        record,
        mdp,
        belief,
        scientific_spec_hash=scientific_spec_hash,
        numerical_method_config_hash=numerical_method_config_hash,
    )
    return TerminalReferenceSourceValidationProof(
        record=record,
        mdp_object_id=id(mdp),
        belief_object_id=id(belief),
        scientific_spec_hash=scientific_spec_hash,
        numerical_method_config_hash=numerical_method_config_hash,
        evaluation_cap=record.evaluation_cap,
        certificate_hash=record.certificate_hash,
        valid=bool(valid),
        _seal=_SOURCE_VALIDATION_PROOF_SEAL,
    )


def terminal_reference_source_proof_matches(
    proof: TerminalReferenceSourceValidationProof,
    record: TerminalReferenceRecord,
    mdp: Any,
    belief: Any,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
) -> bool:
    """Reject stale, copied, cross-source, or false internal validation proofs."""

    try:
        current_mdp_hash = terminal_mdp_identity_hash(mdp)
        current_belief_hash = terminal_belief_identity_hash(belief)
        current_scientific_hash = terminal_scientific_spec_hash(mdp)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return (
        type(proof) is TerminalReferenceSourceValidationProof
        and proof._seal is _SOURCE_VALIDATION_PROOF_SEAL
        and proof.record is record
        and proof.mdp_object_id == id(mdp)
        and proof.belief_object_id == id(belief)
        and proof.scientific_spec_hash == scientific_spec_hash
        and proof.numerical_method_config_hash == numerical_method_config_hash
        and proof.evaluation_cap == record.evaluation_cap
        and proof.certificate_hash == record.certificate_hash
        and record.mdp_identity_hash == current_mdp_hash
        and record.belief_identity_hash == current_belief_hash
        and record.scientific_spec_hash == current_scientific_hash
        and current_scientific_hash == scientific_spec_hash
        and record.numerical_method_config_hash == numerical_method_config_hash
        and proof.valid is True
    )


def validate_production_against_reference_a(
    mdp: Any,
    belief: Any,
    production_result: Any,
    reference_a: TerminalReferenceRecord,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
    _source_validation_proof: Optional[TerminalReferenceSourceValidationProof] = None,
) -> TerminalReferenceValidationResult:
    """Validate one production terminal result against resolved Reference A evidence."""

    checks: List[Tuple[str, bool]] = []

    def check(name: str, value: bool) -> None:
        checks.append((name, bool(value)))

    check(
        "reference_a_source_recomputation",
        (
            terminal_reference_source_proof_matches(
                _source_validation_proof,
                reference_a,
                mdp,
                belief,
                scientific_spec_hash=scientific_spec_hash,
                numerical_method_config_hash=numerical_method_config_hash,
            )
            if _source_validation_proof is not None
            else validate_terminal_reference_record(
                reference_a,
                mdp,
                belief,
                scientific_spec_hash=scientific_spec_hash,
                numerical_method_config_hash=numerical_method_config_hash,
            )
        ),
    )
    check(
        "scientific_spec_hash_match",
        reference_a.scientific_spec_hash == scientific_spec_hash,
    )
    check(
        "numerical_method_config_hash_match",
        reference_a.numerical_method_config_hash == numerical_method_config_hash,
    )
    check("reference_a_resolved", reference_a.status == "resolved")
    check(
        "reference_a_global_width",
        _interval_width(reference_a.global_value_interval) <= 1e-6,
    )
    expected_symmetry = prove_recipient_swap_symmetry(mdp, belief)
    check(
        "reference_a_symmetry_source",
        reference_a.structural_symmetry == expected_symmetry,
    )
    independently_evaluated_production_value = float(
        mdp.expected_terminal_utility(
            belief,
            float(production_result.allocation),
        )
    )
    check(
        "production_source_value",
        independently_evaluated_production_value.hex()
        == float(production_result.value).hex(),
    )
    check(
        "production_allocation_identity",
        float(production_result.allocation).hex()
        == float(reference_a.production_allocation).hex(),
    )
    check(
        "production_fixed_value",
        reference_a.production_value_interval[0]
        <= independently_evaluated_production_value
        <= reference_a.production_value_interval[1],
    )
    check(
        "production_value_global_distance",
        _point_interval_distance(
            float(production_result.value),
            reference_a.global_value_interval,
        )
        <= 1e-4,
    )
    check(
        "production_regret",
        reference_a.production_regret_interval[1] <= 1e-4,
    )
    check(
        "production_allocation_bound",
        reference_a.canonical_allocation_interval is not None
        and _point_interval_distance(
            float(production_result.allocation),
            reference_a.canonical_allocation_interval,
        )
        <= 0.0025,
    )
    if reference_a.tie_status == "structural_symmetry_tie":
        check(
            "structural_symmetry_proof",
            validate_structural_symmetry_proof(
                mdp,
                belief,
                reference_a.structural_symmetry,
            ),
        )

    failures = tuple(name for name, passed in checks if not passed)
    status = "accepted" if not failures else "rejected"
    validation_hash = _canonical_hash(
        {
            "schema": "production_reference_a_validation_v2",
            "status": status,
            "checks": checks,
            "reference_a_certificate": reference_a.certificate_hash,
            "scientific_spec_hash": scientific_spec_hash,
            "numerical_method_config_hash": numerical_method_config_hash,
            "production_allocation": _float_token(production_result.allocation),
            "production_value": _float_token(production_result.value),
        }
    )
    return TerminalReferenceValidationResult(
        status=status,
        checks=tuple(checks),
        failures=failures,
        certificate_hash=validation_hash,
    )
