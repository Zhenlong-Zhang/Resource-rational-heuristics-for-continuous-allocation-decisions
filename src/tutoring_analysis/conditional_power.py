"""Per-hypothesis conditional sample-size references; not final enrollment.

IID-normal noncentral-t references do not validate heterogeneous Welch/slope
procedures or informative-dropout handling.
"""
from __future__ import annotations

import math

from scipy import stats

EFFECTS = (0.2, 0.3, 0.4)
TARGETS = (0.8, 0.9)
RETENTION = (0.8, 0.9, 0.95)
ALPHA = 0.05
ASSURANCE = 0.90
SEARCH_LIMIT = 100_000
KINDS = ("one_sample_or_paired", "balanced_independent_common_sd")
TAILS = ("one_sided_primary", "two_sided_diagnostic")


def conditional_power(n: int, d: float, kind: str, tail: str, alpha=ALPHA) -> float:
    """n is persons for one-sample, persons PER ARM for balanced independent."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >=2")
    if not math.isfinite(d) or d < 0 or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("finite d>=0 and alpha in (0,1) required")
    if kind not in KINDS or tail not in TAILS:
        raise ValueError("unknown kind or tail")
    df = n - 1 if kind == KINDS[0] else 2 * n - 2
    nc = d * math.sqrt(n if kind == KINDS[0] else n / 2)
    critical = float(stats.t.isf(alpha if tail == TAILS[0] else alpha / 2, df))
    power = float(stats.nct.sf(critical, df, nc))
    if tail == TAILS[1]:
        power += float(stats.nct.cdf(-critical, df, nc))
    if not math.isfinite(power) or not -1e-12 <= power <= 1 + 1e-12:
        raise ArithmeticError("invalid noncentral-t probability")
    return min(1.0, max(0.0, power))


def minimum_integer(probability, target: float, lower=2, upper=SEARCH_LIMIT) -> dict:
    """Bounded monotone probability search; retain explicit predecessor evidence."""
    if not 0 < target < 1 or isinstance(lower, bool) or not isinstance(lower, int):
        raise ValueError("invalid target or lower bound")
    if lower < 0 or isinstance(upper, bool) or not isinstance(upper, int) or upper < lower:
        raise ValueError("invalid integer search limits")
    first = float(probability(lower))
    if not math.isfinite(first) or not 0 <= first <= 1:
        raise ArithmeticError("invalid probability")
    if first >= target:
        return {"status": "available", "minimum": lower, "probability": first,
                "previousProbability": None, "searchLimit": upper}
    lo, hi = lower, min(upper, max(lower + 1, 2 * lower))
    while hi < upper and probability(hi) < target:
        lo, hi = hi, min(upper, 2 * hi)
    if probability(hi) < target:
        return {"status": "unavailable", "minimum": None, "probability": None,
                "previousProbability": None, "searchLimit": upper,
                "reason": "target not attained within fixed search limit"}
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if probability(mid) >= target:
            hi = mid
        else:
            lo = mid
    found, previous = float(probability(hi)), float(probability(hi - 1))
    if not (math.isfinite(found) and math.isfinite(previous) and 0 <= previous < target <= found <= 1):
        raise ArithmeticError("minimum boundary validation failed")
    return {"status": "available", "minimum": hi, "probability": found,
            "previousProbability": previous, "searchLimit": upper}


def enrollment(n: int, retention: float) -> dict:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("target usable count must be integer>=2")
    if not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("retention must be in (0,1]")
    expected = math.ceil(n / retention)
    assurance = minimum_integer(lambda m: float(stats.binom.sf(n - 1, m, retention)),
                                ASSURANCE, lower=n)
    return {"usableTarget": n, "retention": retention,
            "expectedCountEnrollment": expected,
            "expectedUsableAtExpectedCountEnrollment": expected * retention,
            "countAssuranceTarget": ASSURANCE, "assuredEnrollment": assurance,
            "units": "persons for the same single arm/population as usableTarget",
            "assumptions": "Known IID person retention; no informative-loss repair"}


def report() -> dict:
    rows = []
    for kind in KINDS:
        for tail in TAILS:
            for d in EFFECTS:
                for target in TARGETS:
                    boundary = minimum_integer(lambda n: conditional_power(n, d, kind, tail), target)
                    n = boundary["minimum"]
                    rows.append({"kind": kind, "tail": tail, "d": d, "alpha": ALPHA,
                                 "targetPower": target, "boundary": boundary,
                                 "nUnits": "persons" if kind == KINDS[0] else "persons PER ARM",
                                 "contrastAnalyzablePersons": n if kind == KINDS[0] or n is None else 2*n,
                                 "hypotheticalEqualFiveArmAnalyzablePersons": None if n is None else 5*n,
                                 "fiveArmQualification": "5*N only if each of five arms uses this same per-arm target; hypothetical, not a selected study N",
                                 "dDefinition": "mean person endpoint / SD of that endpoint (paired: difference / difference SD)"
                                 if kind == KINDS[0] else "difference of person means / common person SD",
                                 "normalReference": "one-sample/paired t" if kind == KINDS[0] else "pooled equal-variance t, NOT heteroscedastic Welch",
                                 "retention": [] if n is None else [enrollment(n, r) for r in RETENTION]})
    if len(rows) != 24:
        raise AssertionError("frozen row count mismatch")
    return {"schema": "conditional-sample-size-v1", "purpose": "declared_assumption_reference_not_final_N",
            "policy": "per_hypothesis_alpha_0.05_no_blanket_correction",
            "primaryTail": "one-sided in each prospectively specified scientific direction",
            "effectsOrigin": "declared, not estimated from QA or a small human pilot",
            "exactOnlyUnder": "Independent normal person summaries; common variance for balanced independent reference",
            "selectedWelchPowerValidated": False, "selectedSlopePowerValidated": False,
            "jointPowerClaimed": False, "globalFWERClaimed": False, "finalNSelected": False,
            "tinyC1Qualification": "Raw resemblance mean <= mean(100*abs(ES-EO)); d alone supplies no attainable effect or human precision",
            "retentionQualification": "Expected count and binomial assurance differ; both assume known IID noninformative retention",
            "fixedGrid": {"d": EFFECTS, "targetPower": TARGETS, "retention": RETENTION,
                          "alpha": ALPHA, "countAssurance": ASSURANCE, "searchLimit": SEARCH_LIMIT},
            "rows": rows}


def markdown(data: dict) -> str:
    out = ["# Conditional sample-size reference", "",
           "Per-hypothesis alpha=.05; no blanket correction. N below is analyzable persons, not rounds.",
           "All numbers assume independent normal person summaries; independent-group rows assume common variance.",
           "They are not calibrated power of heteroscedastic Welch or heterogeneous person-slope tests.", "",
           "| Endpoint reference | Tail | d | Target power | Minimum N | Previous-N power |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in data["rows"]:
        b = r["boundary"]
        prev = b["previousProbability"]
        previous_text = "unavailable/boundary" if prev is None else f"{prev:.6f}"
        n_text = "unavailable" if b["minimum"] is None else str(b["minimum"])
        out.append(f"| {r['normalReference']} ({r['nUnits']}) | {r['tail']} | {r['d']} | {r['targetPower']} | {n_text} | {previous_text} |")
    out.extend(["", "## Illustrative enrollment at d=.3, marginal80% power, one-sided", "",
                "Counts below apply to each population/arm in the reference, not a selected five-arm study N.", "",
                "| Reference | Retention | Analyzable target | Expected-count enrollment | 90%-assured enrollment |",
                "| --- | ---: | ---: | ---: | ---: |"])
    for r in data["rows"]:
        if r["d"] == .3 and r["targetPower"] == .8 and r["tail"] == TAILS[0]:
            for e in r["retention"]:
                out.append(f"| {r['normalReference']} | {e['retention']} | {e['usableTarget']} | {e['expectedCountEnrollment']} | {e['assuredEnrollment']['minimum']} |")
    out.extend(["", "No final N, joint80% power or study-wide .05 error guarantee is claimed.",
                "Retention is assumed known, IID and noninformative. Tiny-C1 raw effects must respect target-separation bounds.",
                "Human variance, selected-test calibration, endpoint feasibility and recruitment resources remain separate inputs.", ""])
    return "\n".join(out)
