"""Deterministic model/GPA audit, independent of the legacy Gaussian implementation.

The continuous Gaussian is a matched-location/spread reference, not a solver comparison.
Its tail probabilities are not the mass removed by the finite discrete kernel.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def gaussian_outside(mean: float, sd: float, lower: float, upper: float) -> float:
    """P(X < lower or X > upper) for a continuous Normal(mean, sd**2)."""
    if not all(math.isfinite(x) for x in (mean, sd, lower, upper)):
        raise ValueError("Gaussian arguments must be finite")
    if sd <= 0 or lower >= upper:
        raise ValueError("Positive SD and ordered boundaries required")
    root_two = math.sqrt(2.0)
    return (0.5 * math.erfc((mean - lower) / (sd * root_two))
            + 0.5 * math.erfc((upper - mean) / (sd * root_two)))


def finite_kernel(centre: float, spread: float, grid: list[int]) -> list[float]:
    """Independent scalar normalization for auditing, not the task's posterior engine."""
    if not math.isfinite(centre) or not math.isfinite(spread) or spread <= 0 or not grid:
        raise ValueError("Finite centre, positive spread and nonempty support required")
    logs = [-0.5 * ((n - centre) / spread) ** 2 for n in grid]
    peak = max(logs)
    weights = [math.exp(x - peak) for x in logs]
    norm = math.fsum(weights)
    return [x / norm for x in weights]


def reference_utility(x: float, exponent: float, loss_multiplier: float) -> float:
    """Scalar algebraic reference used only by the audit."""
    return x ** exponent if x >= 0 else -loss_multiplier * (-x) ** exponent


def audit_bridge(design: dict) -> tuple[dict, list[dict]]:
    common = design["common"]
    needs, reports = common["latent_need"], common["observation"]
    support = (needs["support_min"], needs["support_max"], needs["support_step"])
    if support != (1, 100, 1):
        raise ValueError("This audit targets the declared 1..100 unit-spaced support")
    if tuple(reports[k] for k in ("support_min", "support_max", "support_step")) != support:
        raise ValueError("Need/report support must agree for this audit")
    if common["gpa"]["clipping"]:
        raise ValueError("Clipped GPA is not the specified affine task")
    exponent = common["utility"]["gain_exponent"]
    if exponent != common["utility"]["loss_exponent"] or not 0 < exponent < 1:
        raise ValueError("Audit requires a common positive sublinear exponent")
    loss_multiplier = common["utility"]["loss_multiplier"]
    baseline = common["gpa"]["baseline"]
    gpa_scale = common["gpa"]["utility_scale"]
    if loss_multiplier <= 0 or gpa_scale <= 0:
        raise ValueError("Positive loss multiplier and GPA scale required")
    grid = list(range(1, 101))
    rows, budgets = [], []
    maximum_normalization_error = 0.0
    # Each recipient's reference report marginal has variance s**2 + tau**2.
    for condition in design["conditions"]:
        for scale in common["variation"]["scale_factors"]:
            tau = condition["report_kernel_width"] * scale
            likelihood = [finite_kernel(n, tau, grid) for n in grid]
            maximum_normalization_error = max(maximum_normalization_error,
                max(abs(math.fsum(row) - 1) for row in likelihood))
            for gap in condition["mean_gap_offsets"]:
                budget = condition["total_time"] * scale
                cost = condition["sample_time_cost"] * scale
                budgets.append((budget, cost))
                for recipient in (0, 1):
                    m = (condition["prior_kernel_centres"][recipient]
                         + (gap if recipient == 0 else -gap)) * scale
                    s = condition["prior_kernel_spreads"][recipient] * scale
                    prior = finite_kernel(m, s, grid)
                    mu = math.fsum(n * p for n, p in zip(grid, prior))
                    sd = math.sqrt(math.fsum(p * (n - mu) ** 2 for n, p in zip(grid, prior)))
                    report_mass = [math.fsum(prior[i] * likelihood[i][j]
                                            for i in range(100)) for j in range(100)]
                    rmse = math.sqrt(math.fsum(prior[i] * math.fsum(
                        likelihood[i][j] * (grid[j] - grid[i]) ** 2
                        for j in range(100)) for i in range(100)))
                    maximum_normalization_error = max(maximum_normalization_error,
                        abs(math.fsum(prior) - 1), abs(math.fsum(report_mass) - 1))
                    rows.append({
                        "condition_id": condition["id"], "scale": scale,
                        "gap_offset": gap, "recipient": "AB"[recipient],
                        "prior_kernel_centre": m, "prior_kernel_spread": s,
                        "report_kernel_width": tau, "bounded_prior_mean": mu,
                        "bounded_prior_sd": sd, "bounded_report_rmse": rmse,
                        "bounded_report_mass_sum": math.fsum(report_mass),
                        "bounded_report_at_1_probability": report_mass[0],
                        "bounded_report_at_100_probability": report_mass[-1],
                        "continuous_reference_need_outside_1_100": gaussian_outside(m, s, 1, 100),
                        "continuous_reference_report_outside_1_100":
                            gaussian_outside(m, math.hypot(s, tau), 1, 100),
                    })
    largest_budget = max(b[0] for b in budgets)
    min_gpa = baseline + gpa_scale * reference_utility(-100, exponent, loss_multiplier)
    max_gpa = baseline + gpa_scale * reference_utility(largest_budget - 1, exponent, loss_multiplier)
    if not 0 <= min_gpa <= max_gpa <= 4:
        raise ValueError("Candidate GPA exceeds [0,4] without clipping")
    denominator = common["allocation"]["share_grid_denominator"]
    max_residual, audit_cases = 0.0, 0
    for total, cost in budgets:
        for count in (0, 1, common["maximum_online_assessments"]):
            # Integer accounting is canonical, never rounded participant display values.
            b_milli = round(total * 1000) - count * round(cost * 1000)
            if b_milli < 0:
                continue
            for j in (0, 1, denominator // 2 - 1, denominator // 2,
                      denominator // 2 + 1, denominator - 1, denominator):
                a = b_milli * j / (1000 * denominator)
                b = b_milli * (denominator - j) / (1000 * denominator)
                for n_a in (1, 50, 100):
                    for n_b in (1, 50, 100):
                        u_a = reference_utility(a - n_a, exponent, loss_multiplier)
                        u_b = reference_utility(b - n_b, exponent, loss_multiplier)
                        total_gpa = (baseline + gpa_scale * u_a) + (baseline + gpa_scale * u_b)
                        residual = abs(total_gpa - (2 * baseline + gpa_scale * (u_a + u_b)))
                        max_residual = max(max_residual, residual)
                        audit_cases += 1
    summary = {
        "kind": "deterministic_candidate_model_bridge_audit",
        "template_count": len(rows) // 2, "recipient_template_rows": len(rows),
        "gaussian_reference_interval": [1, 100],
        "gaussian_reference_is_legacy_solver": False,
        "gaussian_tails_are_exact_discrete_removed_mass": False,
        "gaussian_reference_behavioral_equivalence_tested": False,
        "maximum_probability_normalization_error": maximum_normalization_error,
        "gpa_bounds": [min_gpa, max_gpa], "gpa_clipping": False,
        "maximum_gpa_affine_identity_residual": max_residual,
        "gpa_identity_test_cases": audit_cases,
        "gpa_identity": "sum_GPA = 2*baseline + utility_scale*sum_utility",
        "scientific_choice_status": "user_choice_not_inferred_from_numerical_checks",
        "psychological_loss_aversion_estimated": False,
        "spontaneous_fairness_identified": False,
        "conditional_composition_example": {
            "task_loss_multiplier": loss_multiplier,
            "hypothetical_additional_linear_GPA_loss_multiplier": 2,
            "resulting_shortfall_multiplier_relative_to_gain": 2 * loss_multiplier,
            "interpretation": "Algebraic illustration only, not an estimated participant preference",
        },
    }
    if maximum_normalization_error > 1e-12 or max_residual > 1e-12:
        raise ArithmeticError("Independent numerical consistency audit failed")
    return summary, rows


def audit_file(path: str | Path) -> tuple[dict, list[dict]]:
    payload = Path(path).read_bytes()
    summary, rows = audit_bridge(json.loads(payload))
    summary["design_sha256"] = hashlib.sha256(payload).hexdigest()
    return summary, rows
