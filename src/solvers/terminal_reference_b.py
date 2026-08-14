from __future__ import annotations

"""Independent dyadic Terminal Reference B.

This module intentionally does not import or call the production terminal optimizer,
Reference A's solver, or either implementation's interval-search helpers.  It shares only
the immutable reference record schema, canonical source identities, certificate hashing,
and the source-backed recipient-swap proof.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
import heapq
import json
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .terminal import StructuralSymmetry, prove_recipient_swap_symmetry
from .terminal_reference import (
    _SOURCE_VALIDATION_PROOF_SEAL,
    CandidateIsolationEvidence,
    TerminalReferenceRecord,
    TerminalReferenceSourceValidationProof,
    terminal_belief_identity_hash,
    terminal_mdp_identity_hash,
    terminal_reference_certificate_hash,
    terminal_scientific_spec_hash,
)


REFERENCE_B_PRECISION_LADDER = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
REFERENCE_B_EVALUATION_CAP = 500_000
REFERENCE_B_ALLOCATION_TOLERANCE = 1e-4
REFERENCE_B_NODE_WIDTH = 2.5e-5
REFERENCE_B_TIE_SCALE = 1e-12
REFERENCE_B_METHOD_VERSION = "terminal_reference_b_v2"
REFERENCE_B_BRANCH_RULE = "nested_dyadic_branch_and_bound"
REFERENCE_B_CAP_RULE = "nested_dyadic_full_domain_evaluation_cap"
REFERENCE_B_CONSTANT_RULE = "nested_dyadic_constant_objective_full_domain"
REFERENCE_B_ISOLATION_RULES = frozenset(
    {REFERENCE_B_BRANCH_RULE, REFERENCE_B_CAP_RULE, REFERENCE_B_CONSTANT_RULE}
)

_ALPHA_NUMERATOR = 7
_ALPHA_DENOMINATOR = 20
_FROZEN_ALPHA = float(Fraction(_ALPHA_NUMERATOR, _ALPHA_DENOMINATOR))
_ROOT_BRACKET_LIMIT = 4096
REFERENCE_B_HEAP_ORDER = "negative_upper_then_fifo_insertion_counter"
REFERENCE_B_CHILD_ORDER = "left_child_then_right_child"
REFERENCE_B_WITNESS_ORDER = (
    "largest_point_lower",
    "largest_point_representative",
    "closest_to_half",
    "lower_allocation",
)
REFERENCE_B_CANDIDATE_ORDER = "ascending_closed_allocation_interval_then_adjacent_merge"
REFERENCE_B_CANONICAL_ORDER = "closest_to_half_then_lower_side"


class _ReferenceBEvaluationCap(RuntimeError):
    pass


@dataclass(frozen=True)
class _BIdentities:
    mdp_identity_hash: str
    belief_identity_hash: str
    scientific_spec_hash: str
    numerical_method_config_hash: str


@dataclass(frozen=True)
class _BPointValue:
    lower: float
    upper: float
    representative: float


@dataclass(frozen=True)
class _BNode:
    lower: float
    upper: float
    witness_allocation: float
    witness_value: _BPointValue
    upper_bound: float
    depth: int


@dataclass(frozen=True)
class _BCandidate:
    allocation_interval: Tuple[float, float]
    value_interval: Tuple[float, float]
    witness_allocation: float
    witness_value: float
    partition_count: int
    maximum_depth: int


@dataclass(frozen=True)
class _BSnapshot:
    global_value_interval: Tuple[float, float]
    candidates: Tuple[_BCandidate, ...]


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_token(value: float) -> str:
    return float(value).hex()


def terminal_reference_b_numerical_method_config_hash(
    evaluation_cap: int = REFERENCE_B_EVALUATION_CAP,
) -> str:
    """Hash every numerical control used by Reference B."""

    if not 1 <= evaluation_cap <= REFERENCE_B_EVALUATION_CAP:
        raise ValueError("Reference B evaluation cap exceeds the contract budget")
    return _canonical_hash(
        {
            "schema": "terminal_reference_b_numerical_config_v2",
            "method_version": REFERENCE_B_METHOD_VERSION,
            "record_schema": "terminal_reference_record_v2",
            "precision_ladder": tuple(
                _float_token(value) for value in REFERENCE_B_PRECISION_LADDER
            ),
            "evaluation_cap": int(evaluation_cap),
            "allocation_tolerance": _float_token(
                REFERENCE_B_ALLOCATION_TOLERANCE
            ),
            "node_width": _float_token(REFERENCE_B_NODE_WIDTH),
            "tie_scale": _float_token(REFERENCE_B_TIE_SCALE),
            "root_bracket_limit": int(_ROOT_BRACKET_LIMIT),
            "isolation_rules": tuple(sorted(REFERENCE_B_ISOLATION_RULES)),
            "search": "rooted_nested_dyadic_global_branch_and_bound_v1",
            "fixed_value": "exact_fraction_plus_stored_binary64_envelope_v2",
            "bound": "stored_binary64_operation_interval_enclosure_v2",
            "heap_order": REFERENCE_B_HEAP_ORDER,
            "child_order": REFERENCE_B_CHILD_ORDER,
            "witness_order": REFERENCE_B_WITNESS_ORDER,
            "candidate_order": REFERENCE_B_CANDIDATE_ORDER,
            "canonical_order": REFERENCE_B_CANONICAL_ORDER,
            "source_validation": "deterministic_full_reference_b_recompute_v1",
        }
    )


def _reference_b_identities(
    mdp: Any,
    belief: Any,
    evaluation_cap: int,
) -> _BIdentities:
    return _BIdentities(
        mdp_identity_hash=terminal_mdp_identity_hash(mdp),
        belief_identity_hash=terminal_belief_identity_hash(belief),
        scientific_spec_hash=terminal_scientific_spec_hash(mdp),
        numerical_method_config_hash=terminal_reference_b_numerical_method_config_hash(
            evaluation_cap
        ),
    )


def _next_down(value: float) -> float:
    return math.nextafter(float(value), -math.inf)


def _next_up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


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


def _fraction_to_float_down(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("Reference B exact lower bound is outside binary64")
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _fraction_to_float_up(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("Reference B exact upper bound is outside binary64")
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _rounded_fraction_interval(
    lower: Fraction,
    upper: Fraction,
) -> Tuple[float, float]:
    """Enclose one correctly rounded binary64 result over an exact interval."""

    if lower > upper:
        raise ValueError("exact interval is reversed")
    return (
        _next_down(_fraction_to_float_down(lower)),
        _next_up(_fraction_to_float_up(upper)),
    )


def _fl_add_interval(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[float, float]:
    return _rounded_fraction_interval(
        Fraction.from_float(first[0]) + Fraction.from_float(second[0]),
        Fraction.from_float(first[1]) + Fraction.from_float(second[1]),
    )


def _fl_subtract_interval(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[float, float]:
    return _rounded_fraction_interval(
        Fraction.from_float(first[0]) - Fraction.from_float(second[1]),
        Fraction.from_float(first[1]) - Fraction.from_float(second[0]),
    )


def _fl_multiply_interval(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[float, float]:
    products = tuple(
        Fraction.from_float(left) * Fraction.from_float(right)
        for left in first
        for right in second
    )
    return _rounded_fraction_interval(min(products), max(products))


def _stored_power_interval(
    nonnegative: Tuple[float, float],
) -> Tuple[float, float]:
    """Enclose the stored 7/20 power in the monotone nonnegative domain."""

    lower = max(0.0, float(nonnegative[0]))
    upper = max(lower, float(nonnegative[1]))
    root_lower = _fractional_power_bounds(Fraction.from_float(lower))[0]
    root_upper = _fractional_power_bounds(Fraction.from_float(upper))[1]
    return max(0.0, _next_down(root_lower)), _next_up(root_upper)


def _stored_utility_scalar_interval(
    outcome: float,
    lambda_shortfall: float,
) -> Tuple[float, float]:
    if outcome < 0.0:
        magnitude = _stored_power_interval((-outcome, -outcome))
        weighted = _fl_multiply_interval(
            (lambda_shortfall, lambda_shortfall),
            magnitude,
        )
        return -weighted[1], -weighted[0]
    return _stored_power_interval((outcome, outcome))


def _stored_utility_interval(
    outcome: Tuple[float, float],
    lambda_shortfall: float,
) -> Tuple[float, float]:
    """Enclose the monotone signed-power utility over stored outcomes."""

    lower_utility = _stored_utility_scalar_interval(
        outcome[0],
        lambda_shortfall,
    )[0]
    upper_utility = _stored_utility_scalar_interval(
        outcome[1],
        lambda_shortfall,
    )[1]
    return lower_utility, upper_utility


def _stored_fsum_interval(
    terms: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    lower = sum((Fraction.from_float(term[0]) for term in terms), Fraction(0))
    upper = sum((Fraction.from_float(term[1]) for term in terms), Fraction(0))
    return _rounded_fraction_interval(lower, upper)


def _compare_float_to_fractional_power(
    candidate: float,
    value: Fraction,
    numerator: int,
) -> int:
    """Compare a binary64 candidate with ``value**(numerator/20)`` exactly."""

    if candidate < 0.0 or not math.isfinite(candidate) or value < 0:
        raise ValueError("power comparison requires finite nonnegative inputs")
    candidate_fraction = Fraction.from_float(candidate)
    left = candidate_fraction ** _ALPHA_DENOMINATOR
    right = value ** numerator
    return (left > right) - (left < right)


def _fractional_power_bounds(
    value: Fraction,
    numerator: int = _ALPHA_NUMERATOR,
) -> Tuple[float, float]:
    """Bracket ``value**(numerator/20)`` using exact rational comparisons."""

    if value < 0:
        raise ValueError("power bounds require a nonnegative value")
    if value == 0:
        return 0.0, 0.0
    seed_input = float(value)
    if not math.isfinite(seed_input):
        raise RuntimeError("Reference B power input is outside binary64")
    if not 1 <= numerator < _ALPHA_DENOMINATOR:
        raise ValueError("Reference B power numerator must be in [1, 19]")
    seed = math.pow(seed_input, numerator / _ALPHA_DENOMINATOR)
    if not math.isfinite(seed) or seed < 0.0:
        raise RuntimeError("Reference B power seed is non-finite")
    relation = _compare_float_to_fractional_power(seed, value, numerator)
    if relation == 0:
        return seed, seed
    if relation < 0:
        lower = seed
        for _ in range(_ROOT_BRACKET_LIMIT):
            upper = math.nextafter(lower, math.inf)
            if not math.isfinite(upper):
                break
            if _compare_float_to_fractional_power(upper, value, numerator) >= 0:
                return lower, upper
            lower = upper
    else:
        upper = seed
        for _ in range(_ROOT_BRACKET_LIMIT):
            lower = math.nextafter(upper, -math.inf)
            if lower < 0.0:
                lower = 0.0
            if _compare_float_to_fractional_power(lower, value, numerator) <= 0:
                return lower, upper
            upper = lower
    raise RuntimeError("Reference B could not certify a 7/20 root bracket")


def _utility_fraction_bounds(
    outcome: Fraction,
    lambda_shortfall: Fraction,
) -> Tuple[Fraction, Fraction]:
    root_lower, root_upper = _fractional_power_bounds(abs(outcome))
    root_lower_fraction = Fraction.from_float(root_lower)
    root_upper_fraction = Fraction.from_float(root_upper)
    if outcome < 0:
        return (
            -lambda_shortfall * root_upper_fraction,
            -lambda_shortfall * root_lower_fraction,
        )
    return root_lower_fraction, root_upper_fraction


def _utility_derivative_fraction_bounds(
    outcome_lower: Fraction,
    outcome_upper: Fraction,
    lambda_shortfall: Fraction,
) -> Tuple[Fraction, Optional[Fraction]]:
    """Bound du/do on an outcome interval; ``None`` denotes +infinity."""

    if outcome_lower > outcome_upper:
        raise ValueError("outcome derivative interval is reversed")
    alpha = Fraction(_ALPHA_NUMERATOR, _ALPHA_DENOMINATOR)

    def derivative_at(outcome: Fraction) -> Tuple[Fraction, Fraction]:
        if outcome == 0:
            raise ValueError("utility derivative is singular at zero")
        root_lower, root_upper = _fractional_power_bounds(
            abs(outcome),
            _ALPHA_DENOMINATOR - _ALPHA_NUMERATOR,
        )
        coefficient = alpha * (lambda_shortfall if outcome < 0 else 1)
        return (
            coefficient / Fraction.from_float(root_upper),
            coefficient / Fraction.from_float(root_lower),
        )

    if outcome_lower <= 0 <= outcome_upper:
        finite_lowers = []
        if outcome_lower < 0:
            finite_lowers.append(derivative_at(outcome_lower)[0])
        if outcome_upper > 0:
            finite_lowers.append(derivative_at(outcome_upper)[0])
        return (min(finite_lowers) if finite_lowers else Fraction(0), None)
    lower_derivative = derivative_at(outcome_lower)
    upper_derivative = derivative_at(outcome_upper)
    if outcome_lower > 0:
        return upper_derivative[0], lower_derivative[1]
    return lower_derivative[0], upper_derivative[1]


def _interval_width(interval: Tuple[float, float]) -> float:
    return max(0.0, _subtract_up(interval[1], interval[0]))


def _tau_bounds(global_interval: Tuple[float, float]) -> Tuple[float, float]:
    lower, upper = global_interval
    if lower <= 0.0 <= upper:
        minimum_absolute = 0.0
    else:
        minimum_absolute = min(abs(lower), abs(upper))
    return (
        max(
            0.0,
            _multiply_down(
                REFERENCE_B_TIE_SCALE,
                max(1.0, minimum_absolute),
            ),
        ),
        _multiply_up(
            REFERENCE_B_TIE_SCALE,
            max(1.0, abs(lower), abs(upper)),
        ),
    )


def _absolute_difference_interval(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[float, float]:
    lower = max(
        0.0,
        _subtract_down(first[0], second[1]),
        _subtract_down(second[0], first[1]),
    )
    differences = (
        _subtract_down(first[0], second[1]),
        _subtract_up(first[0], second[1]),
        _subtract_down(first[1], second[0]),
        _subtract_up(first[1], second[0]),
    )
    return lower, max(abs(value) for value in differences)


def _regret_interval(
    global_interval: Tuple[float, float],
    production_interval: Tuple[float, float],
) -> Tuple[float, float]:
    return (
        max(0.0, _subtract_down(global_interval[0], production_interval[1])),
        max(0.0, _subtract_up(global_interval[1], production_interval[0])),
    )


class _BObjective:
    """Independent finite-support objective and exact-fraction enclosures."""

    def __init__(self, mdp: Any, belief: Any, evaluation_cap: int) -> None:
        config = getattr(mdp, "config", None)
        required_config = (
            "total_time",
            "terminate_cost",
            "learning_per_unit_of_tutoring",
            "delta_learning_per_unit_tutoring",
            "lambda_shortfall",
            "utility_exponent",
            "alpha",
        )
        if config is None or any(not hasattr(config, name) for name in required_config):
            raise TypeError("Reference B requires the finite-support allocation config")
        if not hasattr(belief, "states") or not hasattr(belief, "weights"):
            raise TypeError("Reference B requires a finite-support belief")
        if tuple(belief.states) != tuple(mdp.prior.states):
            raise ValueError("Reference B belief support does not match the MDP prior")
        alpha = config.alpha if config.alpha is not None else config.utility_exponent
        if float(alpha).hex() != _FROZEN_ALPHA.hex():
            raise ValueError("Reference B supports only alpha=0.35=7/20")
        if not 1 <= evaluation_cap <= REFERENCE_B_EVALUATION_CAP:
            raise ValueError("Reference B evaluation cap exceeds the contract budget")

        already_terminated = bool(
            belief.history and belief.history[-1].get("action") == 0.0
        )
        terminal_cost = 0.0 if already_terminated else float(config.terminate_cost)
        remaining = max(
            0.0,
            float(config.total_time) - float(belief.deliberation_time) - terminal_cost,
        )
        rate_1 = float(config.learning_per_unit_of_tutoring)
        rate_2 = rate_1 - float(config.delta_learning_per_unit_tutoring)
        values = (
            remaining,
            rate_1,
            rate_2,
            float(config.lambda_shortfall),
            *(float(weight) for weight in belief.weights),
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Reference B model contains non-finite values")
        if rate_1 < 0.0 or rate_2 < 0.0:
            raise ValueError("Reference B requires nonnegative learning rates")
        if float(config.lambda_shortfall) < 0.0:
            raise ValueError("Reference B requires nonnegative shortfall loss")

        self.states = tuple(belief.states)
        self.weight_floats = tuple(float(value) for value in belief.weights)
        self.weights = tuple(Fraction.from_float(float(value)) for value in belief.weights)
        self.remaining_float = remaining
        self.rate_1_float = rate_1
        self.rate_2_float = rate_2
        self.lambda_shortfall_float = float(config.lambda_shortfall)
        self.remaining = Fraction.from_float(remaining)
        self.resource_1 = Fraction.from_float(rate_1) * self.remaining
        self.resource_2 = Fraction.from_float(rate_2) * self.remaining
        self.lambda_shortfall = Fraction.from_float(float(config.lambda_shortfall))
        self.evaluation_cap = int(evaluation_cap)
        self.cache: Dict[str, _BPointValue] = {}
        self.upper_bound_cache: Dict[Tuple[str, str], float] = {}

    def _stored_float_value(self, allocation: float) -> float:
        """Independently reproduce the stored fixed-allocation binary64 objective."""

        tutoring_1 = allocation * self.remaining_float
        tutoring_2 = (1.0 - allocation) * self.remaining_float
        amount_1 = self.rate_1_float * tutoring_1
        amount_2 = self.rate_2_float * tutoring_2
        allocation_key = allocation.hex()
        terms = []
        for state, weight in zip(self.states, self.weight_floats):
            if (
                self.remaining_float > 0.0
                and self.rate_1_float > 0.0
                and (
                    float(state.need_1)
                    / (self.rate_1_float * self.remaining_float)
                ).hex()
                == allocation_key
            ):
                outcome_1 = 0.0
            else:
                outcome_1 = amount_1 - float(state.need_1)
            if (
                self.remaining_float > 0.0
                and self.rate_2_float > 0.0
                and (
                    1.0
                    - float(state.need_2)
                    / (self.rate_2_float * self.remaining_float)
                ).hex()
                == allocation_key
            ):
                outcome_2 = 0.0
            else:
                outcome_2 = amount_2 - float(state.need_2)

            utility_1 = (
                -self.lambda_shortfall_float
                * (abs(outcome_1) ** _FROZEN_ALPHA)
                if outcome_1 < 0.0
                else outcome_1 ** _FROZEN_ALPHA
            )
            utility_2 = (
                -self.lambda_shortfall_float
                * (abs(outcome_2) ** _FROZEN_ALPHA)
                if outcome_2 < 0.0
                else outcome_2 ** _FROZEN_ALPHA
            )
            terms.append(weight * (utility_1 + utility_2))
        value = float(math.fsum(terms))
        if not math.isfinite(value):
            raise RuntimeError("Reference B stored objective reproduction is non-finite")
        return value

    def _outcomes(self, state: Any, allocation: float) -> Tuple[Fraction, Fraction]:
        allocation_fraction = Fraction.from_float(float(allocation))
        return (
            self.resource_1 * allocation_fraction
            - Fraction.from_float(float(state.need_1)),
            self.resource_2 * (Fraction(1) - allocation_fraction)
            - Fraction.from_float(float(state.need_2)),
        )

    def fixed_value(self, allocation: float) -> _BPointValue:
        allocation = min(1.0, max(0.0, float(allocation)))
        key = allocation.hex()
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.evaluation_cap:
            raise _ReferenceBEvaluationCap("Reference B evaluation cap exhausted")
        lower_total = Fraction(0)
        upper_total = Fraction(0)
        for state, weight in zip(self.states, self.weights):
            if weight == 0:
                continue
            outcome_1, outcome_2 = self._outcomes(state, allocation)
            utility_1 = _utility_fraction_bounds(outcome_1, self.lambda_shortfall)
            utility_2 = _utility_fraction_bounds(outcome_2, self.lambda_shortfall)
            lower_total += weight * (utility_1[0] + utility_2[0])
            upper_total += weight * (utility_1[1] + utility_2[1])
        lower = _fraction_to_float_down(lower_total)
        upper = _fraction_to_float_up(upper_total)
        representative = self._stored_float_value(allocation)
        stored_interval = self.stored_objective_interval(allocation, allocation)
        lower = min(lower, stored_interval[0], _next_down(representative))
        upper = max(upper, stored_interval[1], _next_up(representative))
        result = _BPointValue(lower, upper, representative)
        self.cache[key] = result
        return result

    def _ideal_separable_upper_bound(self, lower: float, upper: float) -> float:
        lower_fraction = Fraction.from_float(float(lower))
        upper_fraction = Fraction.from_float(float(upper))
        total_upper = Fraction(0)
        for state, weight in zip(self.states, self.weights):
            if weight == 0:
                continue
            outcome_1 = (
                self.resource_1 * upper_fraction
                - Fraction.from_float(float(state.need_1))
            )
            outcome_2 = (
                self.resource_2 * (Fraction(1) - lower_fraction)
                - Fraction.from_float(float(state.need_2))
            )
            utility_1_upper = _utility_fraction_bounds(
                outcome_1,
                self.lambda_shortfall,
            )[1]
            utility_2_upper = _utility_fraction_bounds(
                outcome_2,
                self.lambda_shortfall,
            )[1]
            total_upper += weight * (utility_1_upper + utility_2_upper)
        return _fraction_to_float_up(total_upper)

    def stored_objective_interval(
        self,
        lower: float,
        upper: float,
    ) -> Tuple[float, float]:
        """Enclose every stored binary64 objective operation on a dyadic node."""

        allocation = (float(lower), float(upper))
        if not 0.0 <= allocation[0] <= allocation[1] <= 1.0:
            raise ValueError("Reference B allocation interval must lie in [0, 1]")
        one_minus_allocation = _fl_subtract_interval((1.0, 1.0), allocation)
        time_1 = _fl_multiply_interval(
            allocation,
            (self.remaining_float, self.remaining_float),
        )
        time_2 = _fl_multiply_interval(
            one_minus_allocation,
            (self.remaining_float, self.remaining_float),
        )
        amount_1 = _fl_multiply_interval(
            (self.rate_1_float, self.rate_1_float),
            time_1,
        )
        amount_2 = _fl_multiply_interval(
            (self.rate_2_float, self.rate_2_float),
            time_2,
        )

        weighted_terms: List[Tuple[float, float]] = []
        for state, weight in zip(self.states, self.weight_floats):
            outcome_1 = _fl_subtract_interval(
                amount_1,
                (float(state.need_1), float(state.need_1)),
            )
            outcome_2 = _fl_subtract_interval(
                amount_2,
                (float(state.need_2), float(state.need_2)),
            )
            if self.remaining_float > 0.0 and self.rate_1_float > 0.0:
                kink_1 = float(state.need_1) / (
                    self.rate_1_float * self.remaining_float
                )
                if allocation[0] <= kink_1 <= allocation[1]:
                    outcome_1 = min(outcome_1[0], 0.0), max(outcome_1[1], 0.0)
            if self.remaining_float > 0.0 and self.rate_2_float > 0.0:
                kink_2 = 1.0 - float(state.need_2) / (
                    self.rate_2_float * self.remaining_float
                )
                if allocation[0] <= kink_2 <= allocation[1]:
                    outcome_2 = min(outcome_2[0], 0.0), max(outcome_2[1], 0.0)

            utility_1 = _stored_utility_interval(
                outcome_1,
                self.lambda_shortfall_float,
            )
            utility_2 = _stored_utility_interval(
                outcome_2,
                self.lambda_shortfall_float,
            )
            utility_sum = _fl_add_interval(utility_1, utility_2)
            weighted_terms.append(
                _fl_multiply_interval((weight, weight), utility_sum)
            )
        if not weighted_terms:
            raise RuntimeError("Reference B stored objective has no weighted terms")
        return _stored_fsum_interval(weighted_terms)

    def upper_bound(self, lower: float, upper: float) -> float:
        """Bound both the ideal objective and the actual stored operation sequence."""

        key = (float(lower).hex(), float(upper).hex())
        cached = self.upper_bound_cache.get(key)
        if cached is not None:
            return cached
        stored_upper = self.stored_objective_interval(lower, upper)[1]
        ideal_upper = self._ideal_separable_upper_bound(lower, upper)
        result = max(stored_upper, ideal_upper)
        self.upper_bound_cache[key] = result
        return result

    def derivative_bounds(
        self,
        lower: float,
        upper: float,
    ) -> Tuple[Optional[Fraction], Optional[Fraction]]:
        """Return exact-rational bounds on d expected utility / d allocation.

        ``None`` in the lower or upper position denotes negative or positive infinity,
        respectively.  These bounds are used only to certify monotonicity; an interval
        touching any utility kink cannot receive a finite sign certificate.
        """

        lower_allocation = Fraction.from_float(float(lower))
        upper_allocation = Fraction.from_float(float(upper))
        derivative_lower = Fraction(0)
        derivative_upper = Fraction(0)
        lower_unbounded = False
        upper_unbounded = False
        for state, weight in zip(self.states, self.weights):
            if weight == 0:
                continue
            if self.resource_1 == 0:
                person_1 = (Fraction(0), Fraction(0))
            else:
                person_1 = _utility_derivative_fraction_bounds(
                    self.resource_1 * lower_allocation
                    - Fraction.from_float(float(state.need_1)),
                    self.resource_1 * upper_allocation
                    - Fraction.from_float(float(state.need_1)),
                    self.lambda_shortfall,
                )
            if self.resource_2 == 0:
                person_2 = (Fraction(0), Fraction(0))
            else:
                person_2 = _utility_derivative_fraction_bounds(
                    self.resource_2 * (Fraction(1) - upper_allocation)
                    - Fraction.from_float(float(state.need_2)),
                    self.resource_2 * (Fraction(1) - lower_allocation)
                    - Fraction.from_float(float(state.need_2)),
                    self.lambda_shortfall,
                )

            if person_2[1] is None:
                lower_unbounded = True
            elif not lower_unbounded:
                derivative_lower += weight * (
                    self.resource_1 * person_1[0]
                    - self.resource_2 * person_2[1]
                )
            if person_1[1] is None:
                upper_unbounded = True
            elif not upper_unbounded:
                derivative_upper += weight * (
                    self.resource_1 * person_1[1]
                    - self.resource_2 * person_2[0]
                )
        return (
            None if lower_unbounded else derivative_lower,
            None if upper_unbounded else derivative_upper,
        )


def _make_b_node(
    objective: _BObjective,
    lower: float,
    upper: float,
    depth: int,
) -> _BNode:
    midpoint = lower + 0.5 * (upper - lower)
    points = (lower, midpoint, upper)
    point_values = tuple(objective.fixed_value(point) for point in points)
    best_index = max(
        range(3),
        key=lambda index: (
            point_values[index].lower,
            point_values[index].representative,
            -abs(points[index] - 0.5),
            -points[index],
        ),
    )
    best = point_values[best_index]
    # Ideal-objective derivative signs remain diagnostic only. Rounded binary64 operations
    # can create one-ULP discontinuities, so every operational node uses the independently
    # propagated stored-objective enclosure.
    certified_upper = objective.upper_bound(lower, upper)
    return _BNode(
        lower=float(lower),
        upper=float(upper),
        witness_allocation=float(points[best_index]),
        witness_value=best,
        upper_bound=max(best.upper, certified_upper),
        depth=int(depth),
    )


def _group_b_nodes(nodes: Sequence[_BNode]) -> Tuple[_BCandidate, ...]:
    groups: List[List[_BNode]] = []
    for node in sorted(nodes, key=lambda item: (item.lower, item.upper)):
        if not groups or node.lower > math.nextafter(groups[-1][-1].upper, math.inf):
            groups.append([node])
        else:
            groups[-1].append(node)
    candidates: List[_BCandidate] = []
    for group in groups:
        witness = max(
            group,
            key=lambda node: (
                node.witness_value.lower,
                node.witness_value.representative,
                -abs(node.witness_allocation - 0.5),
                -node.witness_allocation,
            ),
        )
        candidates.append(
            _BCandidate(
                allocation_interval=(group[0].lower, group[-1].upper),
                value_interval=(
                    max(node.witness_value.lower for node in group),
                    max(node.upper_bound for node in group),
                ),
                witness_allocation=witness.witness_allocation,
                witness_value=witness.witness_value.representative,
                partition_count=len(group),
                maximum_depth=max(node.depth for node in group),
            )
        )
    return tuple(candidates)


def _run_b_level(
    objective: _BObjective,
    precision: float,
    audit_level: Optional[Dict[str, Any]] = None,
) -> _BSnapshot:
    root = _make_b_node(objective, 0.0, 1.0, 0)
    if audit_level is not None:
        audit_level.update({
            "precision": float(precision),
            "created_nodes": [],
            "pop_events": [],
            "complete": False,
        })

    def record_node(node: _BNode) -> int:
        if audit_level is None:
            return -1
        identifier = len(audit_level["created_nodes"])
        audit_level["created_nodes"].append((
            identifier, node.lower, node.upper, node.witness_allocation,
            node.witness_value.lower, node.witness_value.upper,
            node.witness_value.representative, node.upper_bound, node.depth,
        ))
        return identifier

    def record_event(identifier: int, disposition: str) -> None:
        if audit_level is not None:
            audit_level["pop_events"].append((identifier, disposition))

    root_id = record_node(root)
    heap: List[Tuple[float, int, Any]] = [(-root.upper_bound, 0, (root_id, root))]
    counter = 1
    global_lower = root.witness_value.lower
    finalized: List[_BNode] = []
    while heap:
        _, _, traced_node = heapq.heappop(heap)
        node_id, node = traced_node
        _, tau_high = _tau_bounds((global_lower, global_lower))
        if node.upper_bound < _subtract_down(global_lower, tau_high):
            record_event(node_id, "pruned_value")
            continue
        if (
            _subtract_up(node.upper_bound, node.witness_value.lower) <= precision
            and _interval_width((node.lower, node.upper)) <= REFERENCE_B_NODE_WIDTH
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
            child = _make_b_node(objective, lower, upper, node.depth + 1)
            child_id = record_node(child)
            global_lower = max(global_lower, child.witness_value.lower)
            heapq.heappush(heap, (-child.upper_bound, counter, (child_id, child)))
            counter += 1

    _, tau_high = _tau_bounds((global_lower, global_lower))
    viable = [
        node
        for node in finalized
        if node.upper_bound >= _subtract_down(global_lower, tau_high)
    ]
    candidates = _group_b_nodes(viable)
    if not candidates:
        raise RuntimeError("Reference B produced no viable candidate interval")
    global_upper = max(candidate.value_interval[1] for candidate in candidates)
    snapshot = _BSnapshot(
        global_value_interval=(global_lower, max(global_lower, global_upper)),
        candidates=candidates,
    )
    if audit_level is not None:
        audit_level["created_nodes"] = tuple(audit_level["created_nodes"])
        audit_level["pop_events"] = tuple(audit_level["pop_events"])
        audit_level["snapshot"] = snapshot
        audit_level["complete"] = True
    return snapshot


def _structural_pair(
    symmetry: StructuralSymmetry,
    candidates: Sequence[_BCandidate],
) -> bool:
    if not symmetry.valid or len(candidates) != 2:
        return False
    left, right = sorted(candidates, key=lambda candidate: candidate.allocation_interval)
    if left.allocation_interval[1] >= 0.5 or right.allocation_interval[0] <= 0.5:
        return False
    tolerance = REFERENCE_B_ALLOCATION_TOLERANCE
    return (
        abs(left.allocation_interval[0] - (1.0 - right.allocation_interval[1]))
        <= tolerance
        and abs(left.allocation_interval[1] - (1.0 - right.allocation_interval[0]))
        <= tolerance
    )


def _canonical_ordinary(candidates: Sequence[_BCandidate]) -> int:
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


def _resolve_b_candidates(
    snapshot: _BSnapshot,
    symmetry: StructuralSymmetry,
) -> Tuple[Optional[str], Tuple[_BCandidate, ...], Optional[int], str]:
    candidates = snapshot.candidates
    if len(candidates) == 1:
        if _interval_width(candidates[0].allocation_interval) <= REFERENCE_B_ALLOCATION_TOLERANCE:
            return "unique", candidates, 0, "unique_dyadic_candidate_isolated"
        return None, candidates, None, "connected_dyadic_maximizer_region_provisional"
    if _structural_pair(symmetry, candidates):
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

    _, tau_high = _tau_bounds(snapshot.global_value_interval)
    dominant = []
    for index, candidate in enumerate(candidates):
        if all(
            index == other_index
            or candidate.value_interval[0]
            > _add_up(other.value_interval[1], tau_high)
            for other_index, other in enumerate(candidates)
        ):
            dominant.append(index)
    if len(dominant) == 1:
        return (
            "unique",
            (candidates[dominant[0]],),
            0,
            "ordinary_dyadic_value_order_certified",
        )

    tau_low, _ = _tau_bounds(snapshot.global_value_interval)
    pairwise = [
        _absolute_difference_interval(first.value_interval, second.value_interval)
        for first_index, first in enumerate(candidates)
        for second in candidates[first_index + 1 :]
    ]
    if pairwise and all(interval[1] <= tau_low for interval in pairwise):
        canonical = _canonical_ordinary(candidates)
        return (
            "certified_value_tie",
            candidates,
            canonical,
            "ordinary_dyadic_value_tie_certified",
        )
    return None, candidates, None, "ordinary_dyadic_tie_provisional"


def _candidate_evidence(
    candidates: Sequence[_BCandidate],
    rule: str = REFERENCE_B_BRANCH_RULE,
) -> Tuple[CandidateIsolationEvidence, ...]:
    return tuple(
        CandidateIsolationEvidence(
            allocation_interval=candidate.allocation_interval,
            value_interval=candidate.value_interval,
            witness_allocation=candidate.witness_allocation,
            witness_value=candidate.witness_value,
            partition_count=candidate.partition_count,
            maximum_depth=candidate.maximum_depth,
            isolation_rule=rule,
        )
        for candidate in candidates
    )


def _with_certificate(record: TerminalReferenceRecord) -> TerminalReferenceRecord:
    return replace(record, certificate_hash=terminal_reference_certificate_hash(record))


def _resolved_record(
    *,
    identities: _BIdentities,
    snapshot: _BSnapshot,
    candidates: Sequence[_BCandidate],
    canonical_index: int,
    tie_status: str,
    symmetry: StructuralSymmetry,
    production_allocation: float,
    production_value: _BPointValue,
    precision: float,
    evaluation_count: int,
    evaluation_cap: int,
    stopping_reason: str,
) -> TerminalReferenceRecord:
    canonical = candidates[canonical_index]
    production_interval = (production_value.lower, production_value.upper)
    record = TerminalReferenceRecord(
        reference_name="terminal_reference_b",
        mdp_identity_hash=identities.mdp_identity_hash,
        belief_identity_hash=identities.belief_identity_hash,
        scientific_spec_hash=identities.scientific_spec_hash,
        numerical_method_config_hash=identities.numerical_method_config_hash,
        status="resolved",
        global_value_interval=snapshot.global_value_interval,
        candidate_allocation_intervals=tuple(
            candidate.allocation_interval for candidate in candidates
        ),
        candidate_value_intervals=tuple(
            candidate.value_interval for candidate in candidates
        ),
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
    identities: _BIdentities,
    symmetry: StructuralSymmetry,
    production_allocation: float,
    production_value: _BPointValue,
    precision: float,
    evaluation_count: int,
    evaluation_cap: int,
    stopping_reason: str,
    snapshot: Optional[_BSnapshot],
    fallback_upper: float,
    isolation_rule: str = REFERENCE_B_BRANCH_RULE,
) -> TerminalReferenceRecord:
    production_interval = (production_value.lower, production_value.upper)
    if snapshot is None:
        global_interval = (
            production_interval[0],
            max(production_interval[0], fallback_upper),
        )
        candidates = (
            _BCandidate(
                allocation_interval=(0.0, 1.0),
                value_interval=global_interval,
                witness_allocation=float(production_allocation),
                witness_value=production_value.representative,
                partition_count=1,
                maximum_depth=0,
            ),
        )
    else:
        global_interval = snapshot.global_value_interval
        candidates = snapshot.candidates
    record = TerminalReferenceRecord(
        reference_name="terminal_reference_b",
        mdp_identity_hash=identities.mdp_identity_hash,
        belief_identity_hash=identities.belief_identity_hash,
        scientific_spec_hash=identities.scientific_spec_hash,
        numerical_method_config_hash=identities.numerical_method_config_hash,
        status="reference_unresolved",
        global_value_interval=global_interval,
        candidate_allocation_intervals=tuple(
            candidate.allocation_interval for candidate in candidates
        ),
        candidate_value_intervals=tuple(
            candidate.value_interval for candidate in candidates
        ),
        candidate_isolation_evidence=_candidate_evidence(candidates, isolation_rule),
        canonical_allocation_interval=None,
        representative_allocation=None,
        tie_status="reference_unresolved",
        structural_symmetry=symmetry,
        production_allocation=float(production_allocation),
        production_value_interval=production_interval,
        production_regret_interval=_regret_interval(
            global_interval,
            production_interval,
        ),
        precision_level=float(precision),
        objective_evaluation_count=int(evaluation_count),
        evaluation_cap=int(evaluation_cap),
        stopping_reason=stopping_reason,
        certificate_hash="",
    )
    return _with_certificate(record)


def solve_terminal_reference_b(
    mdp: Any,
    belief: Any,
    production_allocation: float,
    *,
    evaluation_cap: int = REFERENCE_B_EVALUATION_CAP,
    _audit_trace: Optional[Dict[str, Any]] = None,
) -> TerminalReferenceRecord:
    """Run an independent nested-dyadic global search through the frozen ladder."""

    if not 1 <= evaluation_cap <= REFERENCE_B_EVALUATION_CAP:
        raise ValueError("Reference B evaluation cap exceeds the contract budget")
    production_allocation = float(production_allocation)
    if not math.isfinite(production_allocation) or not 0.0 <= production_allocation <= 1.0:
        raise ValueError("production_allocation must be finite and in [0, 1]")
    identities = _reference_b_identities(mdp, belief, evaluation_cap)
    symmetry = prove_recipient_swap_symmetry(mdp, belief)
    objective = _BObjective(mdp, belief, evaluation_cap)
    if _audit_trace is not None:
        _audit_trace.clear()
        _audit_trace.update({
            "schema": "terminal_reference_b_complete_trace_v1",
            "complete": False,
            "evaluation_cap": int(evaluation_cap),
            "precision_levels": [],
            "objective_cache": (),
        })

    def traced_return(record: TerminalReferenceRecord) -> TerminalReferenceRecord:
        if _audit_trace is not None:
            _audit_trace["precision_levels"] = tuple(_audit_trace["precision_levels"])
            _audit_trace["objective_cache"] = tuple(
                sorted(
                    (
                        allocation,
                        value.lower,
                        value.upper,
                        value.representative,
                    )
                    for allocation, value in objective.cache.items()
                )
            )
            _audit_trace["complete"] = True
        return record
    production_value = objective.fixed_value(production_allocation)
    fallback_upper = max(
        production_value.upper,
        objective.upper_bound(0.0, 1.0),
    )
    if objective.resource_1 == 0 and objective.resource_2 == 0:
        global_interval = (
            production_value.lower,
            production_value.upper,
        )
        snapshot = _BSnapshot(
            global_value_interval=global_interval,
            candidates=(
                _BCandidate(
                    (0.0, 1.0),
                    global_interval,
                    production_allocation,
                    production_value.representative,
                    1,
                    0,
                ),
            ),
        )
        return traced_return(_unresolved_record(
            identities=identities,
            symmetry=symmetry,
            production_allocation=production_allocation,
            production_value=production_value,
            precision=REFERENCE_B_PRECISION_LADDER[0],
            evaluation_count=len(objective.cache),
            evaluation_cap=evaluation_cap,
            stopping_reason="connected_plateau_requires_multiple_maximizer_rule",
            snapshot=snapshot,
            fallback_upper=fallback_upper,
            isolation_rule=REFERENCE_B_CONSTANT_RULE,
        ))

    last_snapshot: Optional[_BSnapshot] = None
    last_reason = "global_value_interval"
    for precision in REFERENCE_B_PRECISION_LADDER:
        audit_level: Optional[Dict[str, Any]] = {} if _audit_trace is not None else None
        if audit_level is not None:
            _audit_trace["precision_levels"].append(audit_level)
        try:
            snapshot = _run_b_level(objective, precision, audit_level)
        except _ReferenceBEvaluationCap:
            if audit_level is not None:
                audit_level["complete"] = True
                audit_level["termination"] = "evaluation_cap_exhausted"
            return traced_return(_unresolved_record(
                identities=identities,
                symmetry=symmetry,
                production_allocation=production_allocation,
                production_value=production_value,
                precision=precision,
                evaluation_count=len(objective.cache),
                evaluation_cap=evaluation_cap,
                stopping_reason="evaluation_cap_exhausted",
                snapshot=last_snapshot,
                fallback_upper=fallback_upper,
                isolation_rule=(
                    REFERENCE_B_CAP_RULE
                    if last_snapshot is None
                    else REFERENCE_B_BRANCH_RULE
                ),
            ))
        last_snapshot = snapshot
        if _interval_width(snapshot.global_value_interval) > precision:
            last_reason = "global_value_interval"
            continue
        tie_status, candidates, canonical_index, reason = _resolve_b_candidates(
            snapshot,
            symmetry,
        )
        if tie_status is not None and canonical_index is not None:
            return traced_return(_resolved_record(
                identities=identities,
                snapshot=snapshot,
                candidates=candidates,
                canonical_index=canonical_index,
                tie_status=tie_status,
                symmetry=symmetry,
                production_allocation=production_allocation,
                production_value=production_value,
                precision=precision,
                evaluation_count=len(objective.cache),
                evaluation_cap=evaluation_cap,
                stopping_reason=reason,
            ))
        last_reason = reason

    return traced_return(_unresolved_record(
        identities=identities,
        symmetry=symmetry,
        production_allocation=production_allocation,
        production_value=production_value,
        precision=REFERENCE_B_PRECISION_LADDER[-1],
        evaluation_count=len(objective.cache),
        evaluation_cap=evaluation_cap,
        stopping_reason=f"{last_reason}_precision_ladder_exhausted",
        snapshot=last_snapshot,
        fallback_upper=fallback_upper,
    ))


def solve_terminal_reference_b_with_trace(
    mdp: Any,
    belief: Any,
    production_allocation: float,
    *,
    evaluation_cap: int = REFERENCE_B_EVALUATION_CAP,
) -> Tuple[TerminalReferenceRecord, Dict[str, Any]]:
    """Return Reference B and every deterministic dyadic refinement/node event."""

    trace: Dict[str, Any] = {}
    record = solve_terminal_reference_b(
        mdp,
        belief,
        production_allocation,
        evaluation_cap=evaluation_cap,
        _audit_trace=trace,
    )
    return record, trace


def _finite_interval(value: Any) -> bool:
    return (
        type(value) is tuple
        and len(value) == 2
        and all(type(item) is float and math.isfinite(item) for item in value)
        and value[0] <= value[1]
    )


def _record_shape_is_valid(record: TerminalReferenceRecord) -> bool:
    if record.reference_name != "terminal_reference_b":
        return False
    if record.status not in {"resolved", "reference_unresolved"}:
        return False
    if terminal_reference_certificate_hash(record) != record.certificate_hash:
        return False
    hashes = (
        record.mdp_identity_hash,
        record.belief_identity_hash,
        record.scientific_spec_hash,
        record.numerical_method_config_hash,
        record.certificate_hash,
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        return False
    if record.precision_level not in REFERENCE_B_PRECISION_LADDER:
        return False
    if type(record.evaluation_cap) is not int or not 1 <= record.evaluation_cap <= REFERENCE_B_EVALUATION_CAP:
        return False
    if (
        type(record.objective_evaluation_count) is not int
        or not 1 <= record.objective_evaluation_count <= record.evaluation_cap
    ):
        return False
    if type(record.production_allocation) is not float or not math.isfinite(record.production_allocation):
        return False
    if not 0.0 <= record.production_allocation <= 1.0:
        return False
    if any(
        not _finite_interval(interval)
        for interval in (
            record.global_value_interval,
            record.production_value_interval,
            record.production_regret_interval,
        )
    ):
        return False
    count = len(record.candidate_allocation_intervals)
    if not count or count != len(record.candidate_value_intervals) or count != len(record.candidate_isolation_evidence):
        return False
    previous_upper = -math.inf
    for allocation_interval, value_interval, evidence in zip(
        record.candidate_allocation_intervals,
        record.candidate_value_intervals,
        record.candidate_isolation_evidence,
    ):
        if not _finite_interval(allocation_interval) or not 0.0 <= allocation_interval[0] <= allocation_interval[1] <= 1.0:
            return False
        if allocation_interval[0] <= previous_upper or not _finite_interval(value_interval):
            return False
        if not isinstance(evidence, CandidateIsolationEvidence):
            return False
        if evidence.allocation_interval != allocation_interval or evidence.value_interval != value_interval:
            return False
        if evidence.isolation_rule not in REFERENCE_B_ISOLATION_RULES:
            return False
        if type(evidence.partition_count) is not int or evidence.partition_count < 1:
            return False
        if type(evidence.maximum_depth) is not int or evidence.maximum_depth < 0:
            return False
        if not (
            type(evidence.witness_allocation) is float
            and math.isfinite(evidence.witness_allocation)
            and allocation_interval[0] <= evidence.witness_allocation <= allocation_interval[1]
        ):
            return False
        if not (
            type(evidence.witness_value) is float
            and math.isfinite(evidence.witness_value)
            and value_interval[0] <= evidence.witness_value <= value_interval[1]
        ):
            return False
        previous_upper = allocation_interval[1]
    if record.status == "reference_unresolved":
        return (
            record.tie_status == "reference_unresolved"
            and record.canonical_allocation_interval is None
            and record.representative_allocation is None
        )
    if record.tie_status not in {"unique", "certified_value_tie", "structural_symmetry_tie"}:
        return False
    if record.canonical_allocation_interval not in record.candidate_allocation_intervals:
        return False
    if type(record.representative_allocation) is not float or not math.isfinite(record.representative_allocation):
        return False
    canonical_index = record.candidate_allocation_intervals.index(
        record.canonical_allocation_interval
    )
    if record.representative_allocation.hex() != record.candidate_isolation_evidence[canonical_index].witness_allocation.hex():
        return False
    if _interval_width(record.global_value_interval) > record.precision_level:
        return False
    if record.tie_status == "unique":
        return len(record.candidate_allocation_intervals) == 1 and _interval_width(record.canonical_allocation_interval) <= REFERENCE_B_ALLOCATION_TOLERANCE
    if record.tie_status == "structural_symmetry_tie":
        return (
            len(record.candidate_allocation_intervals) == 2
            and canonical_index == 0
            and record.structural_symmetry.valid
        )
    return len(record.candidate_allocation_intervals) >= 2


def validate_terminal_reference_b_record(
    record: TerminalReferenceRecord,
    mdp: Any,
    belief: Any,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
) -> bool:
    """Source-validate B by identity checks and deterministic full recomputation."""

    try:
        if not _record_shape_is_valid(record):
            return False
        identities = _reference_b_identities(mdp, belief, record.evaluation_cap)
        if scientific_spec_hash != identities.scientific_spec_hash:
            return False
        if numerical_method_config_hash != identities.numerical_method_config_hash:
            return False
        if (
            record.mdp_identity_hash != identities.mdp_identity_hash
            or record.belief_identity_hash != identities.belief_identity_hash
            or record.scientific_spec_hash != identities.scientific_spec_hash
            or record.numerical_method_config_hash
            != identities.numerical_method_config_hash
        ):
            return False
        recomputed = solve_terminal_reference_b(
            mdp,
            belief,
            record.production_allocation,
            evaluation_cap=record.evaluation_cap,
        )
    except (AttributeError, IndexError, OverflowError, RuntimeError, TypeError, ValueError):
        return False
    return recomputed.certificate_hash == record.certificate_hash


def validate_terminal_reference_b_record_structure(
    record: TerminalReferenceRecord,
    mdp: Any,
    belief: Any,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
) -> bool:
    """Validate a complete B record and source identities without recomputation."""

    try:
        if not _record_shape_is_valid(record):
            return False
        identities = _reference_b_identities(mdp, belief, record.evaluation_cap)
    except (AttributeError, IndexError, OverflowError, TypeError, ValueError):
        return False
    return (
        scientific_spec_hash == identities.scientific_spec_hash
        and numerical_method_config_hash == identities.numerical_method_config_hash
        and record.mdp_identity_hash == identities.mdp_identity_hash
        and record.belief_identity_hash == identities.belief_identity_hash
        and record.scientific_spec_hash == identities.scientific_spec_hash
        and record.numerical_method_config_hash
        == identities.numerical_method_config_hash
    )


def source_validate_terminal_reference_b_record(
    record: TerminalReferenceRecord,
    mdp: Any,
    belief: Any,
    *,
    scientific_spec_hash: str,
    numerical_method_config_hash: str,
) -> TerminalReferenceSourceValidationProof:
    """Recompute Reference B once and bind the result to exact source objects."""

    valid = validate_terminal_reference_b_record(
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


__all__ = [
    "REFERENCE_B_ALLOCATION_TOLERANCE",
    "REFERENCE_B_BRANCH_RULE",
    "REFERENCE_B_EVALUATION_CAP",
    "REFERENCE_B_PRECISION_LADDER",
    "solve_terminal_reference_b",
    "terminal_reference_b_numerical_method_config_hash",
    "validate_terminal_reference_b_record",
    "validate_terminal_reference_b_record_structure",
    "source_validate_terminal_reference_b_record",
]
