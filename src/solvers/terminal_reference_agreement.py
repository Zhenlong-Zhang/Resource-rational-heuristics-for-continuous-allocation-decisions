from __future__ import annotations

"""Source-backed agreement and production acceptance for Terminal References A/B."""

from collections import Counter
from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, List, Mapping, Optional, Tuple

from .terminal import StructuralSymmetry, validate_structural_symmetry_proof
from .terminal_reference import (
    TerminalReferenceRecord,
    terminal_belief_identity_hash,
    terminal_mdp_identity_hash,
    terminal_reference_a_numerical_method_config_hash,
    terminal_scientific_spec_hash,
    validate_terminal_reference_record,
)
from .terminal_reference_b import (
    terminal_reference_b_numerical_method_config_hash,
    validate_terminal_reference_b_record,
)


TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE = 1e-6
TERMINAL_REFERENCE_GLOBAL_DISTANCE_TOLERANCE = 1e-6
TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE = 2.5e-4
TERMINAL_PRODUCTION_VALUE_TOLERANCE = 1e-4
TERMINAL_PRODUCTION_REGRET_TOLERANCE = 1e-4
TERMINAL_PRODUCTION_ALLOCATION_TOLERANCE = 0.0025
TERMINAL_REFERENCE_AGREEMENT_SCHEMA = "terminal_reference_agreement_v1"
TERMINAL_REFERENCE_AGREEMENT_METHOD_VERSION = "terminal_reference_agreement_v2"


@dataclass(frozen=True)
class TerminalReferenceAgreementRecord:
    """Machine-checkable A/B agreement and production-acceptance result."""

    status: str
    mdp_identity_hash: str
    belief_identity_hash: str
    scientific_spec_hash: str
    reference_a_numerical_method_config_hash: str
    reference_b_numerical_method_config_hash: str
    reference_a_certificate_hash: str
    reference_b_certificate_hash: str
    tie_status: Optional[str]
    agreed_global_value_interval: Optional[Tuple[float, float]]
    agreed_canonical_allocation_interval: Optional[Tuple[float, float]]
    production_allocation: Optional[float]
    production_value_interval: Optional[Tuple[float, float]]
    production_regret_interval: Optional[Tuple[float, float]]
    checks: Tuple[Tuple[str, bool], ...]
    failure_reasons: Tuple[str, ...]
    failure_reason_counts: Tuple[Tuple[str, int], ...]
    failure_count: int
    certificate_hash: str


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("agreement record contains a non-finite float")
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item) for item in value)
    raise TypeError(f"unsupported agreement value: {type(value).__name__}")


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def terminal_reference_agreement_numerical_method_config_payload() -> Mapping[str, Any]:
    """Return the complete reviewed numerical identity of the agreement gate."""

    return {
        "schema": "terminal_reference_agreement_numerical_config_v1",
        "method_version": TERMINAL_REFERENCE_AGREEMENT_METHOD_VERSION,
        "record_schema": TERMINAL_REFERENCE_AGREEMENT_SCHEMA,
        "global_interval_width_tolerance": float(
            TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE
        ).hex(),
        "global_interval_distance_tolerance": float(
            TERMINAL_REFERENCE_GLOBAL_DISTANCE_TOLERANCE
        ).hex(),
        "canonical_interval_distance_tolerance": float(
            TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE
        ).hex(),
        "production_value_tolerance": float(
            TERMINAL_PRODUCTION_VALUE_TOLERANCE
        ).hex(),
        "production_regret_tolerance": float(
            TERMINAL_PRODUCTION_REGRET_TOLERANCE
        ).hex(),
        "production_allocation_tolerance": float(
            TERMINAL_PRODUCTION_ALLOCATION_TOLERANCE
        ).hex(),
        "directed_distance_method": "fraction_ratio_with_outward_nextafter_v1",
        "source_validation": "full_reference_a_and_b_recompute_v1",
        "identity_rule": "same_mdp_belief_scientific_and_separate_numerical_hashes",
        "ordinary_optimum_rule": "canonical_overlap_or_directed_distance_v1",
        "structural_rule": "same_validated_mirror_proof_pair_and_lower_representative_v1",
        "tie_compatibility_rule": "resolved_matching_unique_or_structural_only_v1",
        "production_acceptance_rule": "value_regret_and_allocation_all_required_v1",
        "check_order": "validator_statement_order_then_failure_counter_v1",
    }


def terminal_reference_agreement_numerical_method_config_hash() -> str:
    return _canonical_hash(terminal_reference_agreement_numerical_method_config_payload())


def terminal_reference_agreement_certificate_hash(
    record: TerminalReferenceAgreementRecord,
) -> str:
    """Hash every agreement field except the certificate itself."""

    return _canonical_hash(
        {
            "schema": TERMINAL_REFERENCE_AGREEMENT_SCHEMA,
            **{
                field.name: getattr(record, field.name)
                for field in fields(record)
                if field.name != "certificate_hash"
            },
        }
    )


def _finite_interval(value: Any) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is float and math.isfinite(item) for item in value)
        and value[0] <= value[1]
    )


def _next_down(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _next_up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def _fraction_to_float_down(value: Fraction) -> float:
    candidate = float(value)
    if Fraction.from_float(candidate) > value:
        candidate = _next_down(candidate)
    return candidate


def _fraction_to_float_up(value: Fraction) -> float:
    candidate = float(value)
    if Fraction.from_float(candidate) < value:
        candidate = _next_up(candidate)
    return candidate


def _subtract_up(left: float, right: float) -> float:
    exact = Fraction.from_float(float(left)) - Fraction.from_float(float(right))
    return _fraction_to_float_up(exact)


def _directed_point_distance(left: float, right: float) -> float:
    exact = abs(
        Fraction.from_float(float(left)) - Fraction.from_float(float(right))
    )
    return _fraction_to_float_up(exact)


def _interval_width(interval: Tuple[float, float]) -> float:
    return max(0.0, _subtract_up(interval[1], interval[0]))


def _interval_distance(
    left: Tuple[float, float],
    right: Tuple[float, float],
) -> float:
    if left[1] >= right[0] and right[1] >= left[0]:
        return 0.0
    if left[1] < right[0]:
        return max(0.0, _subtract_up(right[0], left[1]))
    return max(0.0, _subtract_up(left[0], right[1]))


def _point_interval_distance(point: float, interval: Tuple[float, float]) -> float:
    if interval[0] <= point <= interval[1]:
        return 0.0
    if point < interval[0]:
        return max(0.0, _subtract_up(interval[0], point))
    return max(0.0, _subtract_up(point, interval[1]))


def _interval_hull(
    left: Tuple[float, float],
    right: Tuple[float, float],
) -> Tuple[float, float]:
    return min(left[0], right[0]), max(left[1], right[1])


def _fixed_value_interval(value: float) -> Tuple[float, float]:
    return _next_down(value), _next_up(value)


def _regret_interval(
    global_interval: Tuple[float, float],
    production_interval: Tuple[float, float],
) -> Tuple[float, float]:
    lower = max(0.0, math.nextafter(global_interval[0] - production_interval[1], -math.inf))
    upper = max(0.0, _subtract_up(global_interval[1], production_interval[0]))
    return lower, upper


def _mirrored_interval(interval: Tuple[float, float]) -> Tuple[float, float]:
    one = Fraction(1, 1)
    lower = one - Fraction.from_float(interval[1])
    upper = one - Fraction.from_float(interval[0])
    return _fraction_to_float_down(lower), _fraction_to_float_up(upper)


def _safe_attribute(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _finite_allocation(value: Any) -> bool:
    return (
        type(value) is float
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _finite_interval_pair(value: Any) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(_finite_interval(interval) for interval in value)
    )


def validate_terminal_reference_agreement(
    mdp: Any,
    belief: Any,
    production_result: Any,
    reference_a: TerminalReferenceRecord,
    reference_b: TerminalReferenceRecord,
    *,
    scientific_spec_hash: str,
    reference_a_numerical_method_config_hash: str,
    reference_b_numerical_method_config_hash: str,
) -> TerminalReferenceAgreementRecord:
    """Validate source identities, A/B agreement, and production acceptance.

    This function does not trust either reference record's self-hash. Both references are
    deterministically source-validated against the supplied MDP and belief before their
    evidence is compared. All applicable checks are retained in the returned record.
    """

    checks: List[Tuple[str, bool]] = []

    def check(name: str, condition: Any) -> None:
        checks.append((name, bool(condition)))

    try:
        actual_mdp_hash = terminal_mdp_identity_hash(mdp)
        actual_belief_hash = terminal_belief_identity_hash(belief)
        actual_scientific_hash = terminal_scientific_spec_hash(mdp)
        identities_computed = True
    except (AttributeError, TypeError, ValueError, OverflowError):
        actual_mdp_hash = ""
        actual_belief_hash = ""
        actual_scientific_hash = ""
        identities_computed = False
    check("actual_source_identities_computed", identities_computed)
    check(
        "scientific_spec_hash_matches_source",
        identities_computed and scientific_spec_hash == actual_scientific_hash,
    )

    try:
        expected_a_numerical_hash = terminal_reference_a_numerical_method_config_hash(
            reference_a.evaluation_cap
        )
        a_numerical_identity_computed = True
    except (AttributeError, TypeError, ValueError):
        expected_a_numerical_hash = ""
        a_numerical_identity_computed = False
    try:
        expected_b_numerical_hash = terminal_reference_b_numerical_method_config_hash(
            reference_b.evaluation_cap
        )
        b_numerical_identity_computed = True
    except (AttributeError, TypeError, ValueError):
        expected_b_numerical_hash = ""
        b_numerical_identity_computed = False
    check("reference_a_numerical_identity_computed", a_numerical_identity_computed)
    check("reference_b_numerical_identity_computed", b_numerical_identity_computed)
    check(
        "reference_a_numerical_identity_matches_source",
        a_numerical_identity_computed
        and reference_a_numerical_method_config_hash == expected_a_numerical_hash,
    )
    check(
        "reference_b_numerical_identity_matches_source",
        b_numerical_identity_computed
        and reference_b_numerical_method_config_hash == expected_b_numerical_hash,
    )
    check(
        "reference_numerical_identities_are_separate",
        bool(expected_a_numerical_hash)
        and bool(expected_b_numerical_hash)
        and expected_a_numerical_hash != expected_b_numerical_hash,
    )

    try:
        a_source_valid = validate_terminal_reference_record(
            reference_a,
            mdp,
            belief,
            scientific_spec_hash=scientific_spec_hash,
            numerical_method_config_hash=reference_a_numerical_method_config_hash,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
        a_source_valid = False
    try:
        b_source_valid = validate_terminal_reference_b_record(
            reference_b,
            mdp,
            belief,
            scientific_spec_hash=scientific_spec_hash,
            numerical_method_config_hash=reference_b_numerical_method_config_hash,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
        b_source_valid = False
    check("reference_a_source_valid", a_source_valid)
    check("reference_b_source_valid", b_source_valid)
    check("reference_a_name", getattr(reference_a, "reference_name", None) == "terminal_reference_a")
    check("reference_b_name", getattr(reference_b, "reference_name", None) == "terminal_reference_b")

    a_mdp_hash = getattr(reference_a, "mdp_identity_hash", "")
    b_mdp_hash = getattr(reference_b, "mdp_identity_hash", "")
    a_belief_hash = getattr(reference_a, "belief_identity_hash", "")
    b_belief_hash = getattr(reference_b, "belief_identity_hash", "")
    a_scientific_hash = getattr(reference_a, "scientific_spec_hash", "")
    b_scientific_hash = getattr(reference_b, "scientific_spec_hash", "")
    check("reference_a_mdp_identity", a_mdp_hash == actual_mdp_hash)
    check("reference_b_mdp_identity", b_mdp_hash == actual_mdp_hash)
    check("cross_reference_mdp_identity", a_mdp_hash == b_mdp_hash == actual_mdp_hash)
    check("reference_a_belief_identity", a_belief_hash == actual_belief_hash)
    check("reference_b_belief_identity", b_belief_hash == actual_belief_hash)
    check(
        "cross_reference_belief_identity",
        a_belief_hash == b_belief_hash == actual_belief_hash,
    )
    check("reference_a_scientific_identity", a_scientific_hash == actual_scientific_hash)
    check("reference_b_scientific_identity", b_scientific_hash == actual_scientific_hash)
    check(
        "cross_reference_scientific_identity",
        a_scientific_hash == b_scientific_hash == actual_scientific_hash,
    )
    check(
        "reference_a_record_numerical_identity",
        getattr(reference_a, "numerical_method_config_hash", "")
        == expected_a_numerical_hash,
    )
    check(
        "reference_b_record_numerical_identity",
        getattr(reference_b, "numerical_method_config_hash", "")
        == expected_b_numerical_hash,
    )
    check("reference_a_resolved", getattr(reference_a, "status", None) == "resolved")
    check("reference_b_resolved", getattr(reference_b, "status", None) == "resolved")

    a_global = getattr(reference_a, "global_value_interval", None)
    b_global = getattr(reference_b, "global_value_interval", None)
    global_intervals_valid = _finite_interval(a_global) and _finite_interval(b_global)
    check("global_intervals_finite", global_intervals_valid)
    agreed_global: Optional[Tuple[float, float]] = None
    if global_intervals_valid:
        check(
            "reference_a_global_width",
            _interval_width(a_global) <= TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE,
        )
        check(
            "reference_b_global_width",
            _interval_width(b_global) <= TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE,
        )
        check(
            "global_interval_agreement",
            _interval_distance(a_global, b_global)
            <= TERMINAL_REFERENCE_GLOBAL_DISTANCE_TOLERANCE,
        )
        agreed_global = _interval_hull(a_global, b_global)
    else:
        check("reference_a_global_width", False)
        check("reference_b_global_width", False)
        check("global_interval_agreement", False)

    a_tie = getattr(reference_a, "tie_status", None)
    b_tie = getattr(reference_b, "tie_status", None)
    resolved_ties = {"unique", "certified_value_tie", "structural_symmetry_tie"}
    compatible_tie = a_tie == b_tie and a_tie in resolved_ties
    check("tie_classification_compatible", compatible_tie)
    agreed_tie = a_tie if compatible_tie else None
    agreed_canonical: Optional[Tuple[float, float]] = None

    if agreed_tie == "structural_symmetry_tie":
        a_symmetry = _safe_attribute(reference_a, "structural_symmetry")
        b_symmetry = _safe_attribute(reference_b, "structural_symmetry")
        symmetry_shape_valid = (
            isinstance(a_symmetry, StructuralSymmetry)
            and isinstance(b_symmetry, StructuralSymmetry)
        )
        check("structural_symmetry_shape", symmetry_shape_valid)
        same_proof = symmetry_shape_valid and a_symmetry == b_symmetry
        check("structural_symmetry_same_proof", same_proof)
        try:
            symmetry_source_valid = (
                same_proof
                and validate_structural_symmetry_proof(mdp, belief, a_symmetry)
            )
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            symmetry_source_valid = False
        check(
            "structural_symmetry_source_valid",
            symmetry_source_valid,
        )
        a_candidates = _safe_attribute(
            reference_a,
            "candidate_allocation_intervals",
        )
        b_candidates = _safe_attribute(
            reference_b,
            "candidate_allocation_intervals",
        )
        pair_shape = _finite_interval_pair(a_candidates) and _finite_interval_pair(
            b_candidates
        )
        check("structural_candidate_pair_shape", pair_shape)
        if pair_shape:
            check(
                "reference_a_structural_mirror_pair",
                _interval_distance(_mirrored_interval(a_candidates[0]), a_candidates[1])
                <= TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
            )
            check(
                "reference_b_structural_mirror_pair",
                _interval_distance(_mirrored_interval(b_candidates[0]), b_candidates[1])
                <= TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
            )
            check(
                "structural_lower_interval_agreement",
                _interval_distance(a_candidates[0], b_candidates[0])
                <= TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
            )
            check(
                "structural_upper_interval_agreement",
                _interval_distance(a_candidates[1], b_candidates[1])
                <= TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
            )
            a_canonical = _safe_attribute(
                reference_a,
                "canonical_allocation_interval",
            )
            b_canonical = _safe_attribute(
                reference_b,
                "canonical_allocation_interval",
            )
            lower_canonical = (
                _finite_interval(a_canonical)
                and _finite_interval(b_canonical)
                and a_canonical == a_candidates[0]
                and b_canonical == b_candidates[0]
            )
            check("structural_lower_canonical_intervals", lower_canonical)
            a_representative = _safe_attribute(
                reference_a,
                "representative_allocation",
            )
            b_representative = _safe_attribute(
                reference_b,
                "representative_allocation",
            )
            representatives_valid = (
                _finite_allocation(a_representative)
                and _finite_allocation(b_representative)
                and a_candidates[0][0]
                <= a_representative
                <= a_candidates[0][1]
                and b_candidates[0][0]
                <= b_representative
                <= b_candidates[0][1]
                and a_representative < 0.5
                and b_representative < 0.5
            )
            check("structural_lower_representatives", representatives_valid)
            check(
                "structural_representative_agreement",
                representatives_valid
                and _directed_point_distance(
                    a_representative,
                    b_representative,
                )
                <= TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
            )
            agreed_canonical = _interval_hull(a_candidates[0], b_candidates[0])
        else:
            for name in (
                "reference_a_structural_mirror_pair",
                "reference_b_structural_mirror_pair",
                "structural_lower_interval_agreement",
                "structural_upper_interval_agreement",
                "structural_lower_canonical_intervals",
                "structural_lower_representatives",
                "structural_representative_agreement",
            ):
                check(name, False)
    elif agreed_tie in {"unique", "certified_value_tie"}:
        a_canonical = _safe_attribute(reference_a, "canonical_allocation_interval")
        b_canonical = _safe_attribute(reference_b, "canonical_allocation_interval")
        canonical_valid = _finite_interval(a_canonical) and _finite_interval(b_canonical)
        check("ordinary_canonical_intervals_finite", canonical_valid)
        check(
            "ordinary_canonical_interval_agreement",
            canonical_valid
            and _interval_distance(a_canonical, b_canonical)
            <= TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE,
        )
        a_candidates = _safe_attribute(reference_a, "candidate_allocation_intervals")
        b_candidates = _safe_attribute(reference_b, "candidate_allocation_intervals")
        candidate_collections_valid = (
            type(a_candidates) is tuple and type(b_candidates) is tuple
        )
        if agreed_tie == "unique":
            check(
                "ordinary_unique_isolated",
                candidate_collections_valid
                and len(a_candidates) == 1
                and len(b_candidates) == 1,
            )
        else:
            a_stopping_reason = _safe_attribute(reference_a, "stopping_reason")
            b_stopping_reason = _safe_attribute(reference_b, "stopping_reason")
            check(
                "ordinary_value_tie_certified",
                candidate_collections_valid
                and len(a_candidates) >= 2
                and len(b_candidates) >= 2
                and isinstance(a_stopping_reason, str)
                and isinstance(b_stopping_reason, str)
                and "certified" in a_stopping_reason
                and "certified" in b_stopping_reason,
            )
        if canonical_valid:
            agreed_canonical = _interval_hull(a_canonical, b_canonical)
    else:
        check("unresolved_or_incompatible_tie_blocks_acceptance", False)

    production_allocation: Optional[float]
    production_value: Optional[float]
    try:
        production_allocation = float(production_result.allocation)
        production_value = float(production_result.value)
        production_fields_finite = (
            math.isfinite(production_allocation)
            and 0.0 <= production_allocation <= 1.0
            and math.isfinite(production_value)
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        production_allocation = None
        production_value = None
        production_fields_finite = False
    check("production_fields_finite", production_fields_finite)

    independently_evaluated_value: Optional[float] = None
    production_value_interval: Optional[Tuple[float, float]] = None
    production_regret_interval: Optional[Tuple[float, float]] = None
    if production_fields_finite:
        try:
            independently_evaluated_value = float(
                mdp.expected_terminal_utility(belief, production_allocation)
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            independently_evaluated_value = None
        independent_value_valid = (
            independently_evaluated_value is not None
            and math.isfinite(independently_evaluated_value)
        )
        check("production_value_independently_evaluated", independent_value_valid)
        check(
            "production_reported_value_matches_source",
            independent_value_valid
            and production_value.hex() == independently_evaluated_value.hex(),
        )
        a_production_allocation = _safe_attribute(
            reference_a,
            "production_allocation",
        )
        b_production_allocation = _safe_attribute(
            reference_b,
            "production_allocation",
        )
        check(
            "production_allocation_matches_reference_a",
            _finite_allocation(a_production_allocation)
            and production_allocation.hex() == a_production_allocation.hex(),
        )
        check(
            "production_allocation_matches_reference_b",
            _finite_allocation(b_production_allocation)
            and production_allocation.hex() == b_production_allocation.hex(),
        )
        a_production_interval = _safe_attribute(
            reference_a,
            "production_value_interval",
        )
        b_production_interval = _safe_attribute(
            reference_b,
            "production_value_interval",
        )
        check(
            "reference_a_production_value_contains_source",
            independent_value_valid
            and _finite_interval(a_production_interval)
            and a_production_interval[0]
            <= independently_evaluated_value
            <= a_production_interval[1],
        )
        check(
            "reference_b_production_value_contains_source",
            independent_value_valid
            and _finite_interval(b_production_interval)
            and b_production_interval[0]
            <= independently_evaluated_value
            <= b_production_interval[1],
        )
        if independent_value_valid:
            production_value_interval = _fixed_value_interval(
                independently_evaluated_value
            )
    else:
        for name in (
            "production_value_independently_evaluated",
            "production_reported_value_matches_source",
            "production_allocation_matches_reference_a",
            "production_allocation_matches_reference_b",
            "reference_a_production_value_contains_source",
            "reference_b_production_value_contains_source",
        ):
            check(name, False)

    check(
        "production_value_global_distance",
        production_value_interval is not None
        and agreed_global is not None
        and _interval_distance(production_value_interval, agreed_global)
        <= TERMINAL_PRODUCTION_VALUE_TOLERANCE,
    )
    if production_value_interval is not None and agreed_global is not None:
        production_regret_interval = _regret_interval(
            agreed_global,
            production_value_interval,
        )
    check(
        "production_regret_upper_bound",
        production_regret_interval is not None
        and production_regret_interval[1] <= TERMINAL_PRODUCTION_REGRET_TOLERANCE,
    )
    check(
        "production_allocation_canonical_distance",
        production_allocation is not None
        and agreed_canonical is not None
        and _point_interval_distance(production_allocation, agreed_canonical)
        <= TERMINAL_PRODUCTION_ALLOCATION_TOLERANCE,
    )

    failures = tuple(name for name, passed in checks if not passed)
    reason_counts = tuple(sorted(Counter(failures).items()))
    status = "accepted" if not failures else "rejected"
    record = TerminalReferenceAgreementRecord(
        status=status,
        mdp_identity_hash=actual_mdp_hash,
        belief_identity_hash=actual_belief_hash,
        scientific_spec_hash=actual_scientific_hash,
        reference_a_numerical_method_config_hash=expected_a_numerical_hash,
        reference_b_numerical_method_config_hash=expected_b_numerical_hash,
        reference_a_certificate_hash=_safe_attribute(
            reference_a,
            "certificate_hash",
            "",
        ),
        reference_b_certificate_hash=_safe_attribute(
            reference_b,
            "certificate_hash",
            "",
        ),
        tie_status=agreed_tie,
        agreed_global_value_interval=agreed_global,
        agreed_canonical_allocation_interval=agreed_canonical,
        production_allocation=production_allocation,
        production_value_interval=production_value_interval,
        production_regret_interval=production_regret_interval,
        checks=tuple(checks),
        failure_reasons=failures,
        failure_reason_counts=reason_counts,
        failure_count=len(failures),
        certificate_hash="",
    )
    return replace(
        record,
        certificate_hash=terminal_reference_agreement_certificate_hash(record),
    )


__all__ = [
    "TERMINAL_PRODUCTION_ALLOCATION_TOLERANCE",
    "TERMINAL_PRODUCTION_REGRET_TOLERANCE",
    "TERMINAL_PRODUCTION_VALUE_TOLERANCE",
    "TERMINAL_REFERENCE_CANONICAL_DISTANCE_TOLERANCE",
    "TERMINAL_REFERENCE_GLOBAL_DISTANCE_TOLERANCE",
    "TERMINAL_REFERENCE_GLOBAL_WIDTH_TOLERANCE",
    "TERMINAL_REFERENCE_AGREEMENT_METHOD_VERSION",
    "TerminalReferenceAgreementRecord",
    "terminal_reference_agreement_certificate_hash",
    "terminal_reference_agreement_numerical_method_config_hash",
    "terminal_reference_agreement_numerical_method_config_payload",
    "validate_terminal_reference_agreement",
]
