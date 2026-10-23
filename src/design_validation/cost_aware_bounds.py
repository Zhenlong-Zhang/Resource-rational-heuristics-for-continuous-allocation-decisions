"""Finite pathwise continuation upper bound for duration-priced terminal macros.

The maximum is taken before expectation. These float64 bounds are not validated
interval arithmetic or a proof of global optimality of a repeated-myopic policy.
"""
import math
import numpy as np


def native_latent_matrices(config, budget_after_first_cost):
    if type(budget_after_first_cost) is not int or budget_after_first_cost < 0:
        raise ValueError("Nonnegative integer native budget required")
    D = config["allocation_denominator"]
    if type(D) is not int or D < 2:
        raise ValueError("Native share denominator required")
    alpha, loss = config["alpha"], config["loss_multiplier"]
    baseline, scale = config["gpa_baseline"], config["gpa_scale"]
    if not all(math.isfinite(x) for x in (alpha, loss, baseline, scale)) or min(alpha, loss, scale) <= 0:
        raise ValueError("Positive utility parameters and GPA scale required")
    needs = np.arange(1, 101, dtype=np.int64)
    def transform(surplus):
        return np.where(surplus >= 0, np.maximum(surplus, 0.)**alpha,
                        -loss*np.maximum(-surplus, 0.)**alpha)
    zero_u = transform(-needs.astype(float))
    zero = 2*baseline+scale*(zero_u[:, None]+zero_u[None, :])
    pi = np.full((100, 100), -np.inf)
    need_numerator = needs*(1000*D)
    for j in range(D+1):
        ua = transform((budget_after_first_cost*j-need_numerator)/(1000*D))
        ub = transform((budget_after_first_cost*(D-j)-need_numerator)/(1000*D))
        np.maximum(pi, 2*baseline+scale*(ua[:, None]+ub[None, :]), out=pi)
    if not np.isfinite(pi).all() or np.any(pi < zero-1e-10):
        raise ValueError("Native PI/zero monotonicity failed")
    return zero, pi


def expected_pathwise_max(weights, zero_GPA, pi_GPA, rho, positive_duration_minimum):
    if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
               for x in (rho, positive_duration_minimum)):
        raise ValueError("Finite nonnegative price and duration required")
    weights, zero, pi = (np.asarray(x, dtype=float) for x in (weights, zero_GPA, pi_GPA))
    if weights.shape != zero.shape or weights.shape != pi.shape or not weights.size:
        raise ValueError("Matching nonempty finite supports required")
    if not np.isfinite(weights).all() or np.any(weights < 0) or abs(float(weights.sum())-1.) > 1e-12:
        raise ValueError("Normalized finite nonnegative probabilities required")
    if not np.isfinite(zero).all() or not np.isfinite(pi).all() or np.any(pi < zero-1e-10):
        raise ValueError("Finite monotone PI/zero rewards required")
    bound = float(np.sum(weights*np.maximum(zero, pi-rho*positive_duration_minimum)))
    unpriced, exhausted = float(np.sum(weights*pi)), float(np.sum(weights*zero))
    if bound > unpriced+1e-10 or bound < exhausted-1e-10:
        raise ValueError("Bound ordering failed")
    return {"stronger_GPA_upper": bound, "reconstructed_unpriced_PI_GPA_upper": unpriced,
            "expected_zero_tutoring_GPA": exhausted,
            "unsafe_max_of_expectations_NOT_a_bound": max(exhausted, unpriced-rho*positive_duration_minimum)}
