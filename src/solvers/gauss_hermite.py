from __future__ import annotations

from functools import lru_cache
from math import factorial, isclose, pi, sqrt
from typing import Callable, List, Tuple

try:
    from ..mdp.meta_mdp import BeliefState, ContinuousAllocationMetaMDP, utility
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import BeliefState, ContinuousAllocationMetaMDP, utility


def _hermite_pair(order: int, x: float) -> Tuple[float, float]:
    """Return physicists' H_n(x) and H_{n-1}(x)."""

    if order == 0:
        return 1.0, 0.0
    h_prev = 1.0
    h_curr = 2.0 * x
    if order == 1:
        return h_curr, h_prev
    for n in range(1, order):
        h_next = 2.0 * x * h_curr - 2.0 * n * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr, h_prev


def _append_root_if_new(roots: List[float], root: float, tolerance: float = 1e-9) -> None:
    if not any(abs(root - existing) <= tolerance for existing in roots):
        roots.append(root)


def _bisect_root(order: int, low: float, high: float, iterations: int = 80) -> float:
    f_low = _hermite_pair(order, low)[0]
    for _ in range(iterations):
        mid = (low + high) / 2.0
        f_mid = _hermite_pair(order, mid)[0]
        if f_low * f_mid <= 0:
            high = mid
        else:
            low = mid
            f_low = f_mid
    return (low + high) / 2.0


def _find_hermite_roots(order: int) -> List[float]:
    if order <= 0:
        raise ValueError("order must be positive")
    search_radius = sqrt(2.0 * order + 1.0) + 2.0
    scan_steps = max(4000, order * 400)
    step = 2.0 * search_radius / scan_steps
    roots: List[float] = []
    x_prev = -search_radius
    f_prev = _hermite_pair(order, x_prev)[0]
    for index in range(1, scan_steps + 1):
        x_curr = -search_radius + index * step
        f_curr = _hermite_pair(order, x_curr)[0]
        if isclose(f_curr, 0.0, abs_tol=1e-12):
            _append_root_if_new(roots, x_curr)
        elif f_prev * f_curr < 0.0:
            _append_root_if_new(roots, _bisect_root(order, x_prev, x_curr))
        x_prev = x_curr
        f_prev = f_curr
    roots.sort()
    if len(roots) != order:
        raise RuntimeError(f"Could not find all Hermite roots for order={order}; found {len(roots)}")
    return roots


@lru_cache(maxsize=None)
def gauss_hermite_nodes_weights(order: int = 15) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return nodes and weights for integrating exp(-x^2) f(x)."""

    try:
        import numpy as np  # type: ignore

        nodes, weights = np.polynomial.hermite.hermgauss(order)
        return tuple(float(x) for x in nodes), tuple(float(w) for w in weights)
    except Exception:
        pass

    roots = _find_hermite_roots(order)
    weights = []
    coefficient = (2.0 ** (order - 1)) * factorial(order) * sqrt(pi) / (order ** 2)
    for root in roots:
        _, h_previous = _hermite_pair(order, root)
        weights.append(coefficient / (h_previous * h_previous))
    return tuple(roots), tuple(weights)


def normal_expectation_1d(
    mean: float,
    variance: float,
    fn: Callable[[float], float],
    order: int = 15,
) -> float:
    if variance < 0:
        raise ValueError("variance must be non-negative")
    if variance == 0:
        return float(fn(mean))
    nodes, weights = gauss_hermite_nodes_weights(order)
    sd = sqrt(variance)
    return float(
        sum(
            weight * fn(mean + sqrt(2.0) * sd * node)
            for node, weight in zip(nodes, weights)
        )
        / sqrt(pi)
    )


def independent_normal_expectation_2d(
    mean_1: float,
    variance_1: float,
    mean_2: float,
    variance_2: float,
    fn: Callable[[float, float], float],
    order: int = 15,
) -> float:
    nodes, weights = gauss_hermite_nodes_weights(order)
    sd_1 = sqrt(max(0.0, variance_1))
    sd_2 = sqrt(max(0.0, variance_2))
    total = 0.0
    for node_1, weight_1 in zip(nodes, weights):
        value_1 = mean_1 + sqrt(2.0) * sd_1 * node_1
        for node_2, weight_2 in zip(nodes, weights):
            value_2 = mean_2 + sqrt(2.0) * sd_2 * node_2
            total += weight_1 * weight_2 * fn(value_1, value_2)
    return float(total / pi)


def expected_terminal_utility_gauss_hermite(
    mdp: ContinuousAllocationMetaMDP,
    belief: BeliefState,
    allocation_to_person1: float,
    order: int = 15,
) -> float:
    remaining_time = mdp.remaining_time_after_termination(belief)
    amount_1, amount_2 = mdp.allocation_to_learning_outcomes(allocation_to_person1, remaining_time)
    alpha = mdp.utility_exponent()

    def terminal_utility(need_1: float, need_2: float) -> float:
        return (
            utility(amount_1 - need_1, mdp.config.lambda_shortfall, alpha)
            + utility(amount_2 - need_2, mdp.config.lambda_shortfall, alpha)
        )

    return independent_normal_expectation_2d(
        mean_1=belief.mean_1,
        variance_1=belief.var_1,
        mean_2=belief.mean_2,
        variance_2=belief.var_2,
        fn=terminal_utility,
        order=order,
    )
