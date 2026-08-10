from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import heapq
import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRODUCTION_VALUE_TOLERANCE = 2.5e-5
PRODUCTION_ALLOCATION_TOLERANCE = 5e-4
PRODUCTION_MAX_EVALUATIONS = 100_000
TERMINAL_TIE_SCALE = 1e-12
FROZEN_UTILITY_EXPONENT_NUMERATOR = 7
FROZEN_UTILITY_EXPONENT_DENOMINATOR = 20
FROZEN_UTILITY_EXPONENT = (
    FROZEN_UTILITY_EXPONENT_NUMERATOR / FROZEN_UTILITY_EXPONENT_DENOMINATOR
)
POWER_BRACKET_MAX_STEPS = 4096
PRODUCTION_TERMINAL_METHOD_VERSION = "terminal_production_kink_global_v2"
PRODUCTION_TERMINAL_RESULT_SCHEMA = "terminal_optimization_result_v2"
STRUCTURAL_SYMMETRY_INVARIANT_FIELDS = (
    "support_atoms",
    "swap_permutation",
    "prior_weights",
    "posterior_weights",
    "learning_rate_1",
    "learning_rate_2",
    "observation_noise_sigma",
    "sampling_time_cost",
    "termination_cost",
    "remaining_time",
    "utility_lambda_shortfall",
    "utility_exponent",
)


@dataclass(frozen=True)
class StructuralSymmetry:
    valid: bool
    permutation: Tuple[int, ...]
    reason: str
    invariant_field_hashes: Tuple[Tuple[str, str], ...] = ()
    invariant_hash: str = ""
    proof_hash: str = ""


@dataclass(frozen=True)
class TerminalOptimizationResult:
    allocation: float
    value: float
    global_upper_bound: float
    regret_upper_bound: float
    candidate_intervals: Tuple[Tuple[float, float], ...]
    tie_status: str
    structural_symmetry: StructuralSymmetry
    objective_evaluations: int


@dataclass(frozen=True)
class TerminalPerformanceDiagnostic:
    repeats: int
    row_count: int
    total_seconds: float
    mean_seconds_per_repeat: float
    objective_evaluations_per_repeat: Tuple[int, ...]
    allocations: Tuple[float, ...]
    values: Tuple[float, ...]
    deterministic: bool


@dataclass(frozen=True)
class _Node:
    lower: float
    upper: float
    value_upper: float
    best_allocation: float
    best_value: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def _invalid_symmetry(reason: str) -> StructuralSymmetry:
    return StructuralSymmetry(False, (), reason)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def production_terminal_numerical_method_config_payload(
    *,
    value_tolerance: Optional[float] = None,
    allocation_tolerance: Optional[float] = None,
    max_evaluations: Optional[int] = None,
) -> Dict[str, Any]:
    """Return every frozen numerical control used by the production optimizer."""

    value_tolerance = (
        PRODUCTION_VALUE_TOLERANCE if value_tolerance is None else value_tolerance
    )
    allocation_tolerance = (
        PRODUCTION_ALLOCATION_TOLERANCE
        if allocation_tolerance is None
        else allocation_tolerance
    )
    max_evaluations = (
        PRODUCTION_MAX_EVALUATIONS if max_evaluations is None else max_evaluations
    )
    if value_tolerance <= 0.0 or not math.isfinite(value_tolerance):
        raise ValueError("production value tolerance must be finite and positive")
    if allocation_tolerance <= 0.0 or not math.isfinite(allocation_tolerance):
        raise ValueError("production allocation tolerance must be finite and positive")
    if max_evaluations < 3 or max_evaluations > PRODUCTION_MAX_EVALUATIONS:
        raise ValueError("production evaluation cap exceeds the reviewed budget")
    return {
        "schema": "terminal_production_numerical_config_v2",
        "method_version": PRODUCTION_TERMINAL_METHOD_VERSION,
        "result_schema": PRODUCTION_TERMINAL_RESULT_SCHEMA,
        "value_tolerance": float(value_tolerance).hex(),
        "allocation_tolerance": float(allocation_tolerance).hex(),
        "evaluation_cap": int(max_evaluations),
        "tie_scale": float(TERMINAL_TIE_SCALE).hex(),
        "utility_exponent": {
            "numerator": FROZEN_UTILITY_EXPONENT_NUMERATOR,
            "denominator": FROZEN_UTILITY_EXPONENT_DENOMINATOR,
        },
        "power_bracket_max_steps": POWER_BRACKET_MAX_STEPS,
        "objective": "stored_binary64_posterior_expected_utility_v1",
        "bounds": "outward_rational_7_over_20_separable_v1",
        "breakpoint_order": "ascending_domain_boundaries_and_all_recipient_kinks",
        "heap_order": "negative_upper_then_fifo_insertion_counter",
        "child_order": "left_child_then_right_child",
        "candidate_order": "ascending_unique_breakpoints_witnesses_and_viable_nodes",
        "canonical_order": "closest_to_half_then_lower_allocation",
        "tie_classification": "unique_structural_mirror_pair_or_ordinary_provisional_v2",
        "symmetry_proof_schema": "terminal_structural_symmetry_proof_v2",
        "symmetry_invariant_schema": "terminal_structural_symmetry_invariants_v2",
        "symmetry_invariant_fields": STRUCTURAL_SYMMETRY_INVARIANT_FIELDS,
    }


def production_terminal_numerical_method_config_hash(
    *,
    value_tolerance: Optional[float] = None,
    allocation_tolerance: Optional[float] = None,
    max_evaluations: Optional[int] = None,
) -> str:
    return _canonical_hash(
        production_terminal_numerical_method_config_payload(
            value_tolerance=value_tolerance,
            allocation_tolerance=allocation_tolerance,
            max_evaluations=max_evaluations,
        )
    )


def build_structural_symmetry_hashes(
    permutation: Sequence[int],
    invariant_field_hashes: Sequence[Tuple[str, str]],
) -> Tuple[str, str]:
    """Build the two outer hashes from immutable named field hashes."""

    fields = tuple((str(name), str(value)) for name, value in invariant_field_hashes)
    invariant_hash = _canonical_hash(
        {
            "schema": "terminal_structural_symmetry_invariants_v2",
            "named_field_hashes": fields,
        }
    )
    proof_hash = _canonical_hash(
        {
            "schema": "terminal_structural_symmetry_proof_v2",
            "validation_result": True,
            "permutation": tuple(int(index) for index in permutation),
            "named_field_hashes": fields,
            "invariant_hash": invariant_hash,
        }
    )
    return invariant_hash, proof_hash


def _named_invariant_hashes(field_payloads: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    if tuple(field_payloads) != STRUCTURAL_SYMMETRY_INVARIANT_FIELDS:
        raise RuntimeError("structural symmetry invariant fields are incomplete or reordered")
    return tuple(
        (
            field_name,
            _canonical_hash(
                {
                    "field_name": field_name,
                    "field_value": field_payloads[field_name],
                }
            ),
        )
        for field_name in STRUCTURAL_SYMMETRY_INVARIANT_FIELDS
    )


def prove_recipient_swap_symmetry(mdp: Any, belief: Any) -> StructuralSymmetry:
    """Validate exact recipient-swap invariance of the stored objective.

    The proof reads the actual support, prior weights, posterior weights, and model
    fields. It never trusts caller-supplied symmetry metadata. The resulting proof may
    only classify/canonicalize a mirrored tie; it never changes objective values or bounds.
    """

    states = tuple(belief.states)
    posterior_weights = tuple(float(weight) for weight in belief.weights)
    if not states or len(states) != len(posterior_weights):
        return _invalid_symmetry("belief_support_and_weights_not_aligned")
    if not hasattr(mdp, "prior") or tuple(mdp.prior.states) != states:
        return _invalid_symmetry("belief_support_not_identical_to_prior_support")
    prior_weights = tuple(float(weight) for weight in mdp.prior.weights)
    if len(prior_weights) != len(states):
        return _invalid_symmetry("prior_support_and_weights_not_aligned")

    support_keys = tuple(
        (float(state.need_1).hex(), float(state.need_2).hex()) for state in states
    )
    if len(set(support_keys)) != len(support_keys):
        return _invalid_symmetry("duplicate_need_pair_in_support")
    index_by_needs = {key: index for index, key in enumerate(support_keys)}
    permutation: List[int] = []
    for need_1, need_2 in support_keys:
        swapped_key = (need_2, need_1)
        if swapped_key not in index_by_needs:
            return _invalid_symmetry("support_not_closed_under_swap")
        permutation.append(index_by_needs[swapped_key])

    for index, swapped_index in enumerate(permutation):
        if permutation[swapped_index] != index:
            return _invalid_symmetry("swap_map_not_involutive")
        if prior_weights[index].hex() != prior_weights[swapped_index].hex():
            return _invalid_symmetry("prior_weights_not_exactly_swap_invariant")
        if posterior_weights[index].hex() != posterior_weights[swapped_index].hex():
            return _invalid_symmetry("posterior_weights_not_exactly_swap_invariant")

    rate_1, rate_2 = (float(value) for value in mdp.learning_rates())
    if rate_1.hex() != rate_2.hex():
        return _invalid_symmetry("learning_rates_not_symmetric")
    numeric_invariants = {
        "learning_rate_1": rate_1,
        "learning_rate_2": rate_2,
        "observation_noise_sigma": float(mdp.config.sigma_sample),
        "sampling_time_cost": float(mdp.config.sample_time_cost),
        "termination_cost": float(mdp.config.terminate_cost),
        "remaining_time": float(mdp.remaining_time_after_termination(belief)),
        "utility_lambda_shortfall": float(mdp.config.lambda_shortfall),
        "utility_exponent": float(mdp.utility_exponent()),
    }
    if any(not math.isfinite(value) for value in numeric_invariants.values()):
        return _invalid_symmetry("non_finite_model_invariant")
    field_payloads = {
        "support_atoms": support_keys,
        "swap_permutation": tuple(permutation),
        "prior_weights": tuple(weight.hex() for weight in prior_weights),
        "posterior_weights": tuple(weight.hex() for weight in posterior_weights),
        **{name: value.hex() for name, value in numeric_invariants.items()},
    }
    invariant_field_hashes = _named_invariant_hashes(field_payloads)
    invariant_hash, proof_hash = build_structural_symmetry_hashes(
        permutation,
        invariant_field_hashes,
    )
    return StructuralSymmetry(
        True,
        tuple(permutation),
        "exact_stored_objective_swap_invariance",
        invariant_field_hashes,
        invariant_hash,
        proof_hash,
    )


def validate_structural_symmetry_proof(
    mdp: Any,
    belief: Any,
    proof: StructuralSymmetry,
) -> bool:
    """Recompute a proof from source data and reject any tampered hash layer."""

    if not proof.valid:
        return False
    expected = prove_recipient_swap_symmetry(mdp, belief)
    if not expected.valid:
        return False
    rebuilt_invariant_hash, rebuilt_proof_hash = build_structural_symmetry_hashes(
        proof.permutation,
        proof.invariant_field_hashes,
    )
    if rebuilt_invariant_hash != proof.invariant_hash:
        return False
    if rebuilt_proof_hash != proof.proof_hash:
        return False
    return proof == expected


def _next_up(value: float, steps: int = 1) -> float:
    result = float(value)
    if math.isnan(result):
        raise RuntimeError("terminal bound arithmetic produced NaN")
    for _ in range(steps):
        if result == math.inf:
            break
        result = math.nextafter(result, math.inf)
    return result


def _next_down(value: float, steps: int = 1) -> float:
    result = float(value)
    if math.isnan(result):
        raise RuntimeError("terminal bound arithmetic produced NaN")
    for _ in range(steps):
        if result == -math.inf:
            break
        result = math.nextafter(result, -math.inf)
    return result


def _add_up(left: float, right: float) -> float:
    return _next_up(float(left) + float(right))


def _add_down(left: float, right: float) -> float:
    return _next_down(float(left) + float(right))


def _multiply_up(left: float, right: float) -> float:
    return _next_up(float(left) * float(right))


def _multiply_down(left: float, right: float) -> float:
    return _next_down(float(left) * float(right))


def _subtract_up(left: float, right: float) -> float:
    return _next_up(float(left) - float(right))


def _compare_float_to_seven_twentieths_power(candidate: float, value: float) -> int:
    """Compare ``candidate`` with ``value ** (7 / 20)`` using exact integers."""

    if candidate < 0.0 or value < 0.0:
        raise ValueError("power comparison requires nonnegative floats")
    candidate_numerator, candidate_denominator = candidate.as_integer_ratio()
    value_numerator, value_denominator = value.as_integer_ratio()
    left = (
        candidate_numerator ** FROZEN_UTILITY_EXPONENT_DENOMINATOR
        * value_denominator ** FROZEN_UTILITY_EXPONENT_NUMERATOR
    )
    right = (
        value_numerator ** FROZEN_UTILITY_EXPONENT_NUMERATOR
        * candidate_denominator ** FROZEN_UTILITY_EXPONENT_DENOMINATOR
    )
    return (left > right) - (left < right)


@lru_cache(maxsize=131_072)
def rational_power_bounds_7_20(value: float) -> Tuple[float, float]:
    """Return certified float bounds around the exact rational power ``value**(7/20)``.

    A runtime power call supplies only a starting point. Direction is never trusted: each
    candidate is classified by the exact equivalence ``y**20 <= x**7`` using the integer
    ratios of the two floats. Adjacent floats are walked until they bracket the root. The
    function fails closed if a finite bracket is not found within the fixed work cap.
    """

    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("rational power input must be finite and nonnegative")
    if value == 0.0:
        return 0.0, 0.0
    runtime_seed = float(value ** FROZEN_UTILITY_EXPONENT)
    if not math.isfinite(runtime_seed) or runtime_seed <= 0.0:
        raise RuntimeError("failed to seed the certified rational-power bracket")
    comparison = _compare_float_to_seven_twentieths_power(runtime_seed, value)
    if comparison == 0:
        return runtime_seed, runtime_seed

    if comparison < 0:
        lower = runtime_seed
        upper = math.nextafter(lower, math.inf)
        for _ in range(POWER_BRACKET_MAX_STEPS):
            if not math.isfinite(upper):
                break
            if _compare_float_to_seven_twentieths_power(upper, value) >= 0:
                return min(lower, runtime_seed), max(upper, runtime_seed)
            lower = upper
            upper = math.nextafter(upper, math.inf)
    else:
        upper = runtime_seed
        lower = math.nextafter(upper, -math.inf)
        for _ in range(POWER_BRACKET_MAX_STEPS):
            if lower < 0.0:
                break
            if _compare_float_to_seven_twentieths_power(lower, value) <= 0:
                return min(lower, runtime_seed), max(upper, runtime_seed)
            upper = lower
            lower = math.nextafter(lower, -math.inf)
    raise RuntimeError("certified rational-power bracket exceeded its step cap")


class _TerminalObjective:
    def __init__(
        self,
        mdp: Any,
        belief: Any,
        symmetry: StructuralSymmetry,
        max_evaluations: int,
    ) -> None:
        self.mdp = mdp
        self.belief = belief
        self.symmetry = symmetry
        self.max_evaluations = int(max_evaluations)
        self.remaining_time = float(mdp.remaining_time_after_termination(belief))
        self.rate_1, self.rate_2 = (float(value) for value in mdp.learning_rates())
        self.resource_1 = self.rate_1 * self.remaining_time
        self.resource_2 = self.rate_2 * self.remaining_time
        self.alpha = float(mdp.utility_exponent())
        if self.alpha.hex() != float(FROZEN_UTILITY_EXPONENT).hex():
            raise ValueError(
                "certified terminal bounds support only utility exponent 0.35 = 7/20"
            )
        self.lambda_shortfall = float(mdp.config.lambda_shortfall)
        self.cache: Dict[str, float] = {}
        self.person_1_kinks: Dict[int, str] = {}
        self.person_2_kinks: Dict[int, str] = {}
        if self.resource_1 > 0.0:
            for index, state in enumerate(self.belief.states):
                allocation = float(state.need_1) / self.resource_1
                if 0.0 <= allocation <= 1.0:
                    self.person_1_kinks[index] = allocation.hex()
        if self.resource_2 > 0.0:
            for index, state in enumerate(self.belief.states):
                allocation = 1.0 - float(state.need_2) / self.resource_2
                if 0.0 <= allocation <= 1.0:
                    self.person_2_kinks[index] = allocation.hex()

    def _outcomes(self, index: int, state: Any, allocation: float) -> Tuple[float, float]:
        key = float(allocation).hex()
        tutoring_time_1 = allocation * self.remaining_time
        tutoring_time_2 = (1.0 - allocation) * self.remaining_time
        outcome_1 = self.rate_1 * tutoring_time_1 - float(state.need_1)
        outcome_2 = self.rate_2 * tutoring_time_2 - float(state.need_2)
        if self.person_1_kinks.get(index) == key:
            outcome_1 = 0.0
        if self.person_2_kinks.get(index) == key:
            outcome_2 = 0.0
        return outcome_1, outcome_2

    def _utility(self, outcome: float) -> float:
        if outcome < 0.0:
            return -self.lambda_shortfall * ((-outcome) ** self.alpha)
        return outcome ** self.alpha

    def value(self, allocation: float) -> float:
        allocation = min(1.0, max(0.0, float(allocation)))
        key = allocation.hex()
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.max_evaluations:
            raise RuntimeError("terminal optimization exceeded its evaluation cap")
        value = float(self.mdp.expected_terminal_utility(self.belief, allocation))
        if not math.isfinite(value):
            raise RuntimeError("terminal objective produced a non-finite value")
        self.cache[key] = value
        return value

    def breakpoints(self) -> Tuple[float, ...]:
        points = {0.0, 1.0}
        if self.resource_1 > 0.0:
            points.update(float(state.need_1) / self.resource_1 for state in self.belief.states)
        if self.resource_2 > 0.0:
            points.update(1.0 - float(state.need_2) / self.resource_2 for state in self.belief.states)
        return tuple(sorted(point for point in points if 0.0 <= point <= 1.0))

    def _outcome_upper_person_1(self, state: Any, allocation: float) -> float:
        tutoring_time = _multiply_up(allocation, self.remaining_time)
        learned = _multiply_up(self.rate_1, tutoring_time)
        return _subtract_up(learned, float(state.need_1))

    def _outcome_upper_person_2(self, state: Any, allocation: float) -> float:
        fraction = _subtract_up(1.0, allocation)
        tutoring_time = _multiply_up(fraction, self.remaining_time)
        learned = _multiply_up(self.rate_2, tutoring_time)
        return _subtract_up(learned, float(state.need_2))

    def _utility_upper(self, outcome_upper: float) -> float:
        if outcome_upper < 0.0:
            root_lower, _ = rational_power_bounds_7_20(-outcome_upper)
            magnitude_lower = max(
                0.0,
                _multiply_down(self.lambda_shortfall, root_lower),
            )
            return -magnitude_lower
        _, root_upper = rational_power_bounds_7_20(outcome_upper)
        return root_upper

    def upper_bound(self, lower: float, upper: float) -> float:
        """Return an outward-rounded separable upper bound for the stored objective."""

        lower = float(lower)
        upper = float(upper)
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError("terminal bound interval must lie in [0, 1]")
        terms: List[float] = []
        for state, weight_value in zip(self.belief.states, self.belief.weights):
            weight = float(weight_value)
            if weight == 0.0:
                continue
            utility_1 = self._utility_upper(self._outcome_upper_person_1(state, upper))
            utility_2 = self._utility_upper(self._outcome_upper_person_2(state, lower))
            terms.append(_multiply_up(weight, _add_up(utility_1, utility_2)))
        if not terms:
            raise RuntimeError("terminal objective has no positive posterior weight")
        bound = math.fsum(terms)
        absolute_sum = math.fsum(abs(term) for term in terms)
        accumulation_error = (
            (64.0 + 16.0 * len(terms))
            * math.ulp(1.0)
            * max(1.0, absolute_sum)
        )
        return _next_up(_add_up(bound, accumulation_error), 4)

def terminal_breakpoints(mdp: Any, belief: Any) -> Tuple[float, ...]:
    symmetry = prove_recipient_swap_symmetry(mdp, belief)
    objective = _TerminalObjective(mdp, belief, symmetry, PRODUCTION_MAX_EVALUATIONS)
    return objective.breakpoints()


def terminal_objective_upper_bound(
    mdp: Any,
    belief: Any,
    lower: float,
    upper: float,
) -> float:
    symmetry = prove_recipient_swap_symmetry(mdp, belief)
    objective = _TerminalObjective(mdp, belief, symmetry, PRODUCTION_MAX_EVALUATIONS)
    return objective.upper_bound(lower, upper)


def _make_node(objective: _TerminalObjective, lower: float, upper: float) -> _Node:
    midpoint = lower + 0.5 * (upper - lower)
    points = (lower, midpoint, upper)
    values = tuple(objective.value(point) for point in points)
    best_index = max(
        range(len(points)),
        key=lambda index: (values[index], -abs(points[index] - 0.5), -points[index]),
    )
    value_upper = objective.upper_bound(lower, upper)
    value_upper = max(value_upper, values[best_index])
    return _Node(
        lower=lower,
        upper=upper,
        value_upper=value_upper,
        best_allocation=points[best_index],
        best_value=values[best_index],
    )


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    ordered = sorted(intervals)
    merged: List[List[float]] = []
    for lower, upper in ordered:
        if not merged or lower > merged[-1][1] + 8.0 * math.ulp(max(1.0, abs(lower))):
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return tuple((float(lower), float(upper)) for lower, upper in merged)


def structural_mirror_tie_supported(
    symmetry: StructuralSymmetry,
    tied_allocations: Sequence[float],
    candidate_intervals: Sequence[Tuple[float, float]],
    allocation_tolerance: float,
) -> bool:
    """Accept only one isolated off-center two-element recipient-swap orbit."""

    if (
        not symmetry.valid
        or len(tied_allocations) != 2
        or len(candidate_intervals) != 2
    ):
        return False
    lower_allocation, upper_allocation = sorted(float(value) for value in tied_allocations)
    if (
        lower_allocation >= 0.5 - allocation_tolerance
        or upper_allocation <= 0.5 + allocation_tolerance
        or abs((lower_allocation + upper_allocation) - 1.0) > allocation_tolerance
    ):
        return False
    left_interval, right_interval = sorted(
        (float(lower), float(upper)) for lower, upper in candidate_intervals
    )
    if (
        left_interval[1] >= 0.5
        or right_interval[0] <= 0.5
        or not (
            left_interval[0] - allocation_tolerance
            <= lower_allocation
            <= left_interval[1] + allocation_tolerance
        )
        or not (
            right_interval[0] - allocation_tolerance
            <= upper_allocation
            <= right_interval[1] + allocation_tolerance
        )
    ):
        return False
    return (
        abs(left_interval[0] - (1.0 - right_interval[1])) <= allocation_tolerance
        and abs(left_interval[1] - (1.0 - right_interval[0])) <= allocation_tolerance
    )


def classify_terminal_tie(
    symmetry: StructuralSymmetry,
    distinct_tied_allocations: Sequence[float],
    candidate_intervals: Sequence[Tuple[float, float]],
    allocation_tolerance: float,
) -> str:
    if len(distinct_tied_allocations) <= 1:
        return "unique"
    if structural_mirror_tie_supported(
        symmetry,
        distinct_tied_allocations,
        candidate_intervals,
        allocation_tolerance,
    ):
        return "structural_symmetry_tie"
    return "ordinary_tie_provisional"


def _canonical_best(
    objective: _TerminalObjective,
    candidates: Sequence[float],
    candidate_intervals: Sequence[Tuple[float, float]],
    allocation_tolerance: float,
) -> Tuple[float, float, str]:
    unique = sorted(set(min(1.0, max(0.0, float(value))) for value in candidates))
    if objective.symmetry.valid:
        unique = sorted(set(unique + [1.0 - value for value in unique]))
    scored = [(allocation, objective.value(allocation)) for allocation in unique]
    best_value = max(value for _, value in scored)
    tie_tolerance = TERMINAL_TIE_SCALE * max(1.0, abs(best_value))
    tied = [
        (allocation, value)
        for allocation, value in scored
        if value >= best_value - tie_tolerance
    ]
    allocation, value = min(tied, key=lambda item: (abs(item[0] - 0.5), item[0]))

    distinct_tied: List[float] = []
    for candidate, _ in tied:
        if all(abs(candidate - existing) > allocation_tolerance for existing in distinct_tied):
            distinct_tied.append(candidate)
    tie_status = classify_terminal_tie(
        objective.symmetry,
        distinct_tied,
        candidate_intervals,
        allocation_tolerance,
    )
    return float(allocation), float(value), tie_status


def optimize_terminal_allocation(
    mdp: Any,
    belief: Any,
    *,
    value_tolerance: float = PRODUCTION_VALUE_TOLERANCE,
    allocation_tolerance: float = PRODUCTION_ALLOCATION_TOLERANCE,
    max_evaluations: int = PRODUCTION_MAX_EVALUATIONS,
    _audit_trace: Optional[Dict[str, Any]] = None,
) -> TerminalOptimizationResult:
    """Globally bound the stored finite-support objective on ``[0, 1]``."""

    if value_tolerance <= 0.0 or not math.isfinite(value_tolerance):
        raise ValueError("value_tolerance must be finite and positive")
    if allocation_tolerance <= 0.0 or not math.isfinite(allocation_tolerance):
        raise ValueError("allocation_tolerance must be finite and positive")
    if max_evaluations < 3:
        raise ValueError("max_evaluations must be at least three")

    symmetry = prove_recipient_swap_symmetry(mdp, belief)
    objective = _TerminalObjective(mdp, belief, symmetry, max_evaluations)
    if _audit_trace is not None:
        _audit_trace.clear()
        _audit_trace.update({
            "schema": "terminal_production_complete_trace_v1",
            "complete": False,
            "value_tolerance": float(value_tolerance),
            "allocation_tolerance": float(allocation_tolerance),
            "max_evaluations": int(max_evaluations),
            "breakpoints": (),
            "created_nodes": [],
            "pop_events": [],
            "objective_cache": (),
        })

    def record_node(node: _Node) -> int:
        if _audit_trace is None:
            return -1
        identifier = len(_audit_trace["created_nodes"])
        _audit_trace["created_nodes"].append((
            identifier, node.lower, node.upper, node.value_upper,
            node.best_allocation, node.best_value,
        ))
        return identifier

    def record_event(identifier: int, disposition: str) -> None:
        if _audit_trace is not None:
            _audit_trace["pop_events"].append((identifier, disposition))

    def finalize_trace() -> None:
        if _audit_trace is not None:
            _audit_trace["created_nodes"] = tuple(_audit_trace["created_nodes"])
            _audit_trace["pop_events"] = tuple(_audit_trace["pop_events"])
            _audit_trace["objective_cache"] = tuple(sorted(objective.cache.items()))
            _audit_trace["complete"] = True
    if objective.resource_1 == 0.0 and objective.resource_2 == 0.0:
        value = objective.value(0.5)
        global_upper = max(value, objective.upper_bound(0.0, 1.0))
        result = TerminalOptimizationResult(
            allocation=0.5,
            value=value,
            global_upper_bound=global_upper,
            regret_upper_bound=max(0.0, global_upper - value),
            candidate_intervals=((0.0, 1.0),),
            tie_status="ordinary_tie_provisional",
            structural_symmetry=symmetry,
            objective_evaluations=len(objective.cache),
        )
        finalize_trace()
        return result

    breakpoints = objective.breakpoints()
    if _audit_trace is not None:
        _audit_trace["breakpoints"] = tuple(breakpoints)
    heap: List[Tuple[float, int, _Node]] = []
    counter = 0
    best_value = -math.inf
    best_points: List[float] = []
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        if upper <= lower:
            continue
        node = _make_node(objective, lower, upper)
        node_id = record_node(node)
        best_value = max(best_value, node.best_value)
        best_points.append(node.best_allocation)
        heapq.heappush(heap, (-node.value_upper, counter, (node_id, node)))
        counter += 1

    finalized: List[_Node] = []
    while heap:
        _, _, traced_node = heapq.heappop(heap)
        node_id, node = traced_node
        tie_window = TERMINAL_TIE_SCALE * max(1.0, abs(best_value))
        if node.value_upper < best_value - tie_window:
            record_event(node_id, "pruned_value")
            continue
        if (
            node.value_upper <= best_value + value_tolerance
            and node.width <= allocation_tolerance
        ):
            finalized.append(node)
            record_event(node_id, "finalized_tolerance")
            continue
        midpoint = node.lower + 0.5 * node.width
        if midpoint in (node.lower, node.upper):
            finalized.append(node)
            record_event(node_id, "finalized_machine_resolution")
            continue
        record_event(node_id, "split")
        for lower, upper in ((node.lower, midpoint), (midpoint, node.upper)):
            child = _make_node(objective, lower, upper)
            child_id = record_node(child)
            if child.best_value > best_value:
                best_value = child.best_value
            best_points.append(child.best_allocation)
            heapq.heappush(heap, (-child.value_upper, counter, (child_id, child)))
            counter += 1

    if not finalized:
        raise RuntimeError("terminal optimization produced no viable global interval")

    global_upper = max(best_value, max(node.value_upper for node in finalized))
    tie_window = TERMINAL_TIE_SCALE * max(1.0, abs(best_value))
    viable = [node for node in finalized if node.value_upper >= best_value - tie_window]
    candidate_intervals = _merge_intervals((node.lower, node.upper) for node in viable)
    candidates = list(breakpoints)
    candidates.extend(best_points)
    candidates.extend(node.best_allocation for node in viable)
    allocation, value, tie_status = _canonical_best(
        objective,
        candidates,
        candidate_intervals,
        allocation_tolerance,
    )
    regret_upper = max(0.0, global_upper - value)
    numerical_slack = max(1e-12, 128.0 * math.ulp(max(1.0, abs(value))))
    if regret_upper > value_tolerance + numerical_slack:
        raise RuntimeError(
            "terminal optimization did not certify the production regret: "
            f"{regret_upper:.17g} > {value_tolerance + numerical_slack:.17g}"
        )
    result = TerminalOptimizationResult(
        allocation=allocation,
        value=value,
        global_upper_bound=float(global_upper),
        regret_upper_bound=float(regret_upper),
        candidate_intervals=candidate_intervals,
        tie_status=tie_status,
        structural_symmetry=symmetry,
        objective_evaluations=len(objective.cache),
    )
    finalize_trace()
    return result


def optimize_terminal_allocation_with_trace(
    mdp: Any,
    belief: Any,
    *,
    value_tolerance: float = PRODUCTION_VALUE_TOLERANCE,
    allocation_tolerance: float = PRODUCTION_ALLOCATION_TOLERANCE,
    max_evaluations: int = PRODUCTION_MAX_EVALUATIONS,
) -> Tuple[TerminalOptimizationResult, Dict[str, Any]]:
    """Return the production result plus the complete deterministic search trace."""

    trace: Dict[str, Any] = {}
    result = optimize_terminal_allocation(
        mdp,
        belief,
        value_tolerance=value_tolerance,
        allocation_tolerance=allocation_tolerance,
        max_evaluations=max_evaluations,
        _audit_trace=trace,
    )
    return result, trace


def optimal_terminal_results_for_weight_rows(
    mdp: Any,
    belief: Any,
    posterior_weights: Sequence[Sequence[float]],
    deliberation_time: float,
) -> Tuple[TerminalOptimizationResult, ...]:
    """Return scalar-production results for each stored posterior-weight row."""

    results: List[TerminalOptimizationResult] = []
    for row in posterior_weights:
        posterior = type(belief)(
            states=belief.states,
            weights=tuple(float(weight) for weight in row),
            deliberation_time=float(deliberation_time),
            history=list(belief.history),
        )
        results.append(optimize_terminal_allocation(mdp, posterior))
    return tuple(results)


def optimal_terminal_values_for_weight_rows(
    mdp: Any,
    belief: Any,
    posterior_weights: Sequence[Sequence[float]],
    deliberation_time: float,
):
    """Evaluate each posterior row with exact scalar production semantics."""

    values = [
        result.value
        for result in optimal_terminal_results_for_weight_rows(
            mdp,
            belief,
            posterior_weights,
            deliberation_time,
        )
    ]
    try:
        import numpy as np  # type: ignore

        return np.asarray(values, dtype=float)
    except ImportError:
        return values


def diagnose_terminal_performance(
    mdp: Any,
    belief: Any,
    *,
    posterior_weights: Optional[Sequence[Sequence[float]]] = None,
    deliberation_time: Optional[float] = None,
    repeats: int = 3,
) -> TerminalPerformanceDiagnostic:
    """Repeat the exact solver and report timing plus deterministic evaluation counts."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rows = posterior_weights
    if rows is not None and deliberation_time is None:
        raise ValueError("deliberation_time is required for posterior-weight rows")
    signatures: List[Tuple[Tuple[float, float, int], ...]] = []
    evaluation_counts: List[int] = []
    started = time.perf_counter()
    for _ in range(repeats):
        if rows is None:
            results = (optimize_terminal_allocation(mdp, belief),)
        else:
            results = optimal_terminal_results_for_weight_rows(
                mdp,
                belief,
                rows,
                float(deliberation_time),
            )
        signatures.append(
            tuple(
                (result.allocation, result.value, result.objective_evaluations)
                for result in results
            )
        )
        evaluation_counts.append(sum(result.objective_evaluations for result in results))
    elapsed = time.perf_counter() - started
    first = signatures[0]
    return TerminalPerformanceDiagnostic(
        repeats=repeats,
        row_count=len(first),
        total_seconds=float(elapsed),
        mean_seconds_per_repeat=float(elapsed / repeats),
        objective_evaluations_per_repeat=tuple(evaluation_counts),
        allocations=tuple(item[0] for item in first),
        values=tuple(item[1] for item in first),
        deterministic=all(signature == first for signature in signatures[1:]),
    )
