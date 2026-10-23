"""Independent finite mathematics for the selected cost-aware record reader.

No producer, terminal-menu or bound implementation is imported. NumPy/PCG64
remain a shared numerical backend, not an independent arithmetic certificate.
"""
import hashlib
import math
import numpy as np

from .cost_aware_inputs import NAMESPACE, KEY_FIELDS, bits, compact, integer, require

MACROS = ("ES", "EO_expected_GPA", "EO_raw_surplus", "PN", "AL", "GN", "ML",
          "MFN_A", "MFN_B", "OPT_expected_GPA")
OPS = ("read_scalar", "retrieve_rule", "add_subtract", "multiply_divide", "compare", "clip")
COUNTS = ((0, 1, 0, 0, 0, 0), (3, 1, 0, 0, 0, 1), (3, 1, 2, 1, 0, 1),
          (2, 1, 1, 1, 0, 1), (2, 1, 0, 0, 1, 0), (2, 1, 0, 0, 1, 0),
          (3, 1, 1, 1, 1, 1), (2, 1, 1, 1, 0, 1), (2, 1, 1, 1, 0, 1), (3, 1, 0, 0, 0, 1))


def lse(values, axis=None):
    a = np.asarray(values, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    require(np.isfinite(m).all(), "Finite log maximum")
    out = m + np.log(np.sum(np.exp(a-m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


def normalized(logs):
    a = np.asarray(logs, dtype=np.float64)
    return a-lse(a)


def prior(centre, spread):
    require(spread > 0, "Positive spread")
    return normalized(-.5*((np.arange(1, 101, dtype=np.float64)-centre)/spread)**2)


def channel(width):
    require(width > 0, "Positive channel width")
    n = np.arange(1, 101, dtype=np.float64)
    raw = -.5*((n[:, None]-n[None, :])/width)**2
    return raw-lse(raw, axis=1)[:, None]


def actual_posterior(logs, likelihood, report):
    return normalized(np.asarray(logs, dtype=np.float64)+likelihood[:, integer(report, 1, 100)-1])


def predictive(logs, likelihood):
    joint = normalized(logs)[:, None]+likelihood
    logmass = lse(joint, axis=0)
    return np.exp(logmass), (joint-logmass[None, :]).T


def moments(logs):
    p = np.exp(normalized(logs)); needs = np.arange(1, 101, dtype=np.float64)
    mean = float(p @ needs)
    return [mean, float(np.sqrt(p @ (needs-mean)**2))]


def utility(surplus, config):
    a = np.asarray(surplus, dtype=np.float64)
    return np.where(a >= 0, np.maximum(a, 0)**config["alpha"],
                    -config["loss_multiplier"]*np.maximum(-a, 0)**config["alpha"])


def project(share, denominator=1000):
    target = min(1., max(0., float(share)))*denominator
    lo, hi = math.floor(target), math.ceil(target)
    distance = min(abs(lo-target), abs(hi-target))
    return min((j for j in {lo, hi} if abs(j-target) <= distance+1e-12),
               key=lambda j: (abs(j-denominator/2), j))


def terminal(config, logs, budget):
    """Reconstruct all 1001 native choices, including unavailable zero shares."""
    integer(budget, 0, 2**53-1)
    D = config["allocation_denominator"]
    require(type(D) is int and D == 1000, "Frozen native grid")
    numerator = budget*np.arange(D+1, dtype=np.int64)[:, None]-np.arange(1, 101, dtype=np.int64)[None, :]*(1000*D)
    table = utility(numerator/(1000*D), config)
    ps = [np.exp(normalized(row)) for row in logs]
    ua, ub = table @ ps[0], table[::-1] @ ps[1]
    ga, gb = config["gpa_baseline"]+config["gpa_scale"]*ua, config["gpa_baseline"]+config["gpa_scale"]*ub
    total, gap = ga+gb, np.abs(ga-gb)
    best, mingap = float(np.max(total)), float(np.min(gap))
    opt = np.flatnonzero(total >= best-1e-12*max(1., abs(best))).tolist()
    first = np.flatnonzero(gap <= mingap+1e-12*max(1., abs(mingap)))
    second = float(np.max(total[first]))
    eo = first[total[first] >= second-1e-12*max(1., abs(second))].tolist()
    choose = lambda indices: min(indices, key=lambda j: (abs(j-D/2), j))
    stats = [moments(row) for row in logs]; a, b = stats[0][0], stats[1][0]
    tie = abs(a-b) <= 1e-9*max(1., a, b)
    if budget:
        B = budget/1000
        indices = {"ES": 500, "EO_expected_GPA": choose(eo), "OPT_expected_GPA": choose(opt),
                   "EO_raw_surplus": project((B+a-b)/(2*B)), "PN": project(a/(a+b)),
                   "AL": 500 if tie else 1000 if a < b else 0,
                   "GN": 500 if tie else 1000 if a > b else 0,
                   "ML": 500 if tie else project(min(B, a)/B if a < b else 1-min(B, b)/B),
                   "MFN_A": project(min(B, a)/B), "MFN_B": project(1-min(B, b)/B)}
    else:
        indices = dict.fromkeys(MACROS)
    macros = {}
    for m in MACROS:
        j = indices[m]; k = 0 if j is None else j
        macros[m] = {"index": j, "share_A": None if j is None else j/D,
                     "amount_A_minutes": budget*k/(1000*D), "amount_B_minutes": budget*(D-k)/(1000*D),
                     "expected_utility": float(ua[k]+ub[k]), "expected_GPA_A": float(ga[k]),
                     "expected_GPA_B": float(gb[k]), "expected_combined_GPA": float(total[k]),
                     "expected_GPA_gap": float(gap[k]), "regret_vs_raw_grid_GPA_max": max(0., best-float(total[k]))}
    return {"budget_milliminutes": budget, "moments": stats, "macros": macros,
            "raw_grid_max_combined_GPA": best, "raw_EO_minimum_GPA_gap": mingap,
            "EO_selected_gap_excess": max(0., float(gap[choose(eo)])-mingap),
            "OPT_tied_indices": opt, "EO_tied_indices": eo,
            "automatic_zero_budget": budget == 0, "machine_certified_arithmetic": False}


def durations(settings, profile_name, budget):
    p = settings["assumed_profiles"][profile_name]
    if budget == 0:
        return dict.fromkeys(MACROS, 0.)
    out = {m: settings["common_execution_seconds"]+sum(n*p[k] for n, k in zip(c, OPS)) for m, c in zip(MACROS, COUNTS)}
    out["EO_expected_GPA"] += p["EO_inference_seconds"]
    out["OPT_expected_GPA"] += p["OPT_inference_seconds"]
    return out


def select(summary, times, rho):
    require(math.isfinite(rho) and rho >= 0, "Nonnegative finite price")
    values = np.asarray([summary["macros"][m]["expected_combined_GPA"] for m in MACROS], dtype=np.float64)-rho*np.asarray([times[m] for m in MACROS], dtype=np.float64)
    maximum = float(np.max(values))
    tied = np.flatnonzero(values >= maximum-1e-12*max(1., abs(maximum))).tolist()
    j = min(tied, key=lambda k: (times[MACROS[k]], k)); chosen = MACROS[j]
    classes = {}
    for macro in MACROS:
        classes.setdefault(str(summary["macros"][macro]["index"]), []).append(macro)
    return {"rho_GPA_per_second": rho, "selected_macro": chosen, "selected_index": summary["macros"][chosen]["index"],
            "selected_objective": float(values[j]), "raw_max_objective": maximum,
            "objective_selection_regret": max(0., maximum-float(values[j])), "tied_macros": [MACROS[k] for k in tied],
            "allocation_classes": classes, "objective_by_macro": {m: float(v) for m, v in zip(MACROS, values)},
            "duration_by_macro_seconds": {m: float(times[m]) for m in MACROS}}


def policy(stop, asks, budget, cost):
    if budget < cost:
        return {"action": "stop", "bestGain": None, "chosenGain": None, "actionTie": None, "nearThreshold": None}
    require(len(asks) == 2, "Both selected-child actions required")
    best = max(asks)-stop
    action = "stop" if best <= 1e-10 else "ask_A" if asks[0] >= asks[1]-1e-10 else "ask_B"
    chosen = 0. if action == "stop" else asks[0 if action == "ask_A" else 1]-stop
    return {"action": action, "bestGain": best, "chosenGain": chosen, "actionTie": abs(asks[0]-asks[1]) <= 1e-10,
            "nearThreshold": abs(best-1e-10) <= 1e-8}


def stream_uniform(key):
    require(set(key) == KEY_FIELDS and key["namespace"] == NAMESPACE, "Exact potential-stream key")
    require(key["conditionId"] in ("C1", "C2", "C3", "C4", "C5"), "Stream condition")
    integer(key["scheduleVariant"], 0, 8); integer(key["replicateId"], 0, 63); integer(key["mainIndex"], 1, 10)
    integer(key["reportIndex"], 0, 2**53-1)
    require(key["logicalRecipient"] in ("A", "B") and key["streamKind"] in ("latent", "report"), "Logical potential stream")
    require(key["streamKind"] != "latent" or key["reportIndex"] == 0, "Latent index zero")
    hashed = hashlib.sha256(compact(key).encode()).digest()
    u = float(np.random.Generator(np.random.PCG64(int.from_bytes(hashed[:16], "big"))).random())
    return {"keyPayload": key, "digest": hashed.hex(), "seedHex": hashed[:16].hex(), "uniform": u, "uniformBinary64": bits(u)}


def categorical(probabilities, uniform):
    require(math.isfinite(uniform) and 0 <= uniform < 1 and len(probabilities) == 100, "Full-support CDF input")
    p = np.asarray(probabilities, dtype=np.float64)
    require(np.isfinite(p).all() and np.all(p >= 0), "Nonnegative finite CDF weights")
    mass = float(np.sum(p)); require(abs(mass-1.) <= 1e-12, "CDF mass unresolved; no repair")
    total = np.float64(0.); previous = 0.
    for i, value in enumerate(p):
        previous = float(total); total = np.add(total, value)
        if float(total) > uniform:
            return {"category": i+1, "cdfPrevious": previous, "cdfSelected": float(total), "fallback": False,
                    "rawMass": mass, "massResidual": mass-1., "zeroExponentiatedCount": int(np.count_nonzero(p == 0))}
    return {"category": 100, "cdfPrevious": previous, "cdfSelected": float(total), "fallback": True,
            "rawMass": mass, "massResidual": mass-1., "zeroExponentiatedCount": int(np.count_nonzero(p == 0))}


def realized(config, budget, index, needs, rho, seconds):
    require(index is None if budget == 0 else type(index) is int and 0 <= index <= 1000, "Payoff share availability")
    D = config["allocation_denominator"]; j = 0 if index is None else index
    numerators = [budget*j, budget*(D-j)]
    surplus = [(a-integer(n, 1, 100)*(1000*D))/(1000*D) for a, n in zip(numerators, needs)]
    us = [s**config["alpha"] if s >= 0 else -config["loss_multiplier"]*((-s)**config["alpha"]) for s in surplus]
    combined = us[0]+us[1]
    gpa = 2*config["gpa_baseline"]+config["gpa_scale"]*combined
    return {"amountNumerators": numerators, "surplus": surplus, "utilities": us,
            "individualGPA": [config["gpa_baseline"]+config["gpa_scale"]*u for u in us],
            "combinedUtility": combined, "combinedGPA": gpa, "realizedObjective": gpa-rho*seconds,
            "arithmeticMapId": "integer_numerator_then_native_combined_7_plus_0.03_sum_v1"}


def continuation_upper(config, logs, budget_after, settings, profile, rho):
    n = np.arange(1, 101, dtype=np.int64); zero_u = utility(-n.astype(float), config)
    zero = 2*config["gpa_baseline"]+config["gpa_scale"]*(zero_u[:, None]+zero_u[None, :])
    pi = np.full((100, 100), -np.inf, dtype=float); D = config["allocation_denominator"]
    for j in range(D+1):
        ua = utility((budget_after*j-n*(1000*D))/(1000*D), config)
        ub = utility((budget_after*(D-j)-n*(1000*D))/(1000*D), config)
        np.maximum(pi, 2*config["gpa_baseline"]+config["gpa_scale"]*(ua[:, None]+ub[None, :]), out=pi)
    tau = min(durations(settings, profile, 1).values())
    p = [np.exp(normalized(row)) for row in logs]
    return float(np.sum((p[0][:, None]*p[1][None, :])*np.maximum(zero, pi-rho*tau)))


def reference_record(design):
    """Old symmetric C1 reference, not the working C1 gap-one prior."""
    c = next(c for c in design["conditions"] if c["id"] == "C1")
    common = design["common"]
    config = {"allocation_denominator": 1000, "alpha": common["utility"]["gain_exponent"],
              "loss_multiplier": common["utility"]["loss_multiplier"], "gpa_baseline": common["gpa"]["baseline"],
              "gpa_scale": common["gpa"]["utility_scale"]}
    require(c["prior_kernel_centres"] == [50, 50] and c["total_time"] == 120, "Old symmetric reference only")
    logs = [prior(50., s) for s in c["prior_kernel_spreads"]]
    before, after = terminal(config, logs, 120000), terminal(config, logs, 121000)
    benefit = after["raw_grid_max_combined_GPA"]-before["raw_grid_max_combined_GPA"]
    require(benefit > 0, "Positive reference increment")
    probabilities = [np.exp(normalized(row)) for row in logs]
    def named(amounts):
        return sum(config["gpa_baseline"]+config["gpa_scale"]*float(
            utility((amount-np.arange(1, 101, dtype=np.int64)*1000)/1000, config) @ p)
            for amount, p in zip(amounts, probabilities))
    scale_index = next(i for i, s in enumerate(common["variation"]["scale_factors"]) if s == 1.)
    return {"rho0_GPA_per_personal_second": benefit, "reference_template": f"C1_s{scale_index}_g0",
            "resource_before_minutes": 120., "resource_after_minutes": 121.,
            "raw_optimal_GPA_before": before["raw_grid_max_combined_GPA"],
            "raw_optimal_GPA_after": after["raw_grid_max_combined_GPA"],
            "ES_fixed_share_benefit": after["macros"]["ES"]["expected_combined_GPA"]-before["macros"]["ES"]["expected_combined_GPA"],
            "A_only_non_native_benefit": named((61000, 60000))-named((60000, 60000)),
            "B_only_non_native_benefit": named((60000, 61000))-named((60000, 60000)),
            "empirical_personal_valuation": False, "state_dependent": False}
