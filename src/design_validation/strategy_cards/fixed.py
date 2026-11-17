"""Display-based no-question cards and complete fixed-two report integration.

Inputs are authenticated by the public input adapter before scoring. They carry
id, task, ordered logsHex and native display strings; fixed-two also requires
orderProof and display.singleReport. No cached historical scores are accepted.
"""
import math
import re
from fractions import Fraction

import numpy as np

CARDS = (
    "equal_minutes", "equal_projected_gpa", "all_to_higher_guess",
    "all_to_lower_guess", "ask_each_equal_projected_gpa",
)
GRID = np.arange(1001)
DISPLAY = re.compile(r"(?:[1-9][0-9]?\.[0-9]|100\.0)\Z")
ORDER_PROOF = {
    "orders": [[0, 1], [1, 0]],
    "weights": ["1/2", "1/2"],
    "law": "independent_recipient_latents_and_recipient_local_reports",
    "normalization": "each_latent_row_divided_by_sum_over_reports_1_to_100",
    "posterior": "one_recipient_local_update_each",
    "terminal": "ordered_terminal_visible_means_and_remaining_budget_only",
    "questionCharges": "same_cost_twice_or_skip_both",
}
VALUE_SOURCE = "public_binary64_from_ordered_logs"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def shapes(logs, width):
    """Use independently row-normalized finite Gaussian likelihoods."""
    n = np.arange(1, 101)
    raw = np.exp(-.5 * ((n[:, None] - n[None, :]) / width) ** 2)
    likelihood = raw / raw.sum(axis=1, keepdims=True)
    a = np.exp(logs - max(logs))
    prior = a / a.sum()
    joint = prior[:, None] * likelihood
    mass = joint.sum(axis=0)
    need(np.all(mass > 0) and abs(float(mass.sum()) - 1) <= 1e-12,
         "Strictly positive normalized predictive mass")
    return prior, mass, (joint / mass).T


def scores(surplus, loss):
    return np.where(surplus >= 0, np.sqrt(np.abs(surplus)),
                    -loss * np.sqrt(np.abs(surplus)))


def grid_utility(budget, loss):
    return scores((budget * GRID[:, None] - 1_000_000 * np.arange(1, 101))
                  / 1_000_000, loss)


def guesses(text):
    need(isinstance(text, list) and all(type(x) is str and DISPLAY.fullmatch(x)
                                      for x in text), "Admitted one-decimal display")
    result = np.array([round(float(x) * 10) for x in text], dtype=np.int64)
    need(all(math.isfinite(float(x)) and 1 <= float(x) <= 100
             and int(y) / 10 == float(x) for x, y in zip(text, result)),
         "Admitted one-decimal display")
    return result


def similar(a, b, budget, loss):
    """Locate closest projected scores using monotonicity in native index."""
    a, b = np.broadcast_arrays(a, b)
    numerator = budget + 100 * (a - b)
    denom = max(2 * budget, 1)
    lo = np.clip((1000 * numerator) // denom, 0, 1000)
    hi = np.clip(lo + 1, 0, 1000)

    def gap(j):
        sa = (budget * j - 100_000 * a) / 1_000_000
        sb = (budget * (1000 - j) - 100_000 * b) / 1_000_000
        return np.abs(scores(sa, loss) - scores(sb, loss))

    gl, gh = gap(lo), gap(hi)
    zero = (1000 * numerator % denom) == 0
    reflected = (a == b) & (lo + hi == 1000)
    ambiguous = (np.abs(gl - gh) <= 1e-12) & (lo != hi) & ~reflected & ~zero
    left = (gl < gh) | ((gl == gh) & (np.abs(lo - 500) < np.abs(hi - 500)))
    first = np.where(left, lo, hi)
    mirror = reflected & (np.abs(lo - 500) == np.abs(hi - 500))
    second = np.where(mirror, 1000 - first, first)
    return first, second, ambiguous


def payoff_interval(task, budget):
    """Universal support enclosure, independent of approximate probabilities."""
    baseline = 2 * Fraction(task["gpa_baseline"])
    scale = Fraction(task["gpa_scale"])
    loss = Fraction(task["loss_multiplier"])
    need(scale > 0 and loss >= 0, "Valid payoff coefficients")
    bound = math.isqrt((budget + 999) // 1000) + 1
    return [str(baseline - 20 * scale * loss),
            str(baseline + 2 * scale * bound)]


def evaluate(root, *, fixed=True):
    """Return {id, rows, fixedEvidence}; five rows by default, four if fixed=False.

    A fresh point is computed at each distinct selected native index. The input
    adapter, not this numerical kernel, authenticates source/display identities.
    """
    need("oldScores" not in root, "Cached oldScores are not public scoring inputs")
    t = root["task"]
    budget, cost = t["total_millihours"], t["cost_millihours"]
    loss, base, scale = t["loss_multiplier"], 2 * t["gpa_baseline"], t["gpa_scale"]
    need(type(budget) is int and type(cost) is int and budget > 0 and cost > 0
         and t["allocation_denominator"] == 1000 and t["alpha"] == .5,
         "Supported native budget/payoff")
    need(all(type(t[k]) in (int, float) and math.isfinite(t[k])
             for k in ("loss_multiplier", "gpa_baseline", "gpa_scale"))
         and loss >= 0 and scale > 0, "Finite payoff coefficients")
    prior = [np.array([float.fromhex(x) for x in row]) for row in root["logsHex"]]
    need(len(prior) == 2 and all(row.shape == (100,) and np.isfinite(row).all()
                                for row in prior), "Complete ordered priors")
    display = guesses(root["display"]["root"])
    need(display.shape == (2,), "Both ordered displayed guesses")
    a, b = display
    j, k, unknown = similar(a, b, budget, loss)
    higher = 1000 if a > b else 0 if a < b else 500
    options = [([500], False), (sorted({int(j), int(k)}), bool(unknown)),
               ([higher], False), ([1000 - higher], False)]
    scores_by_index = {}
    for indices, ambiguous in options:
        if ambiguous:
            continue
        for index in indices:
            if index in scores_by_index:
                continue
            amounts = np.arange(1, 101) * 1_000_000
            scores_for_index = []
            for logs, share in zip(prior, (index, 1000 - index)):
                weights = np.exp(logs - max(logs))
                scores_for_index.append(float(
                    scores((budget * share - amounts) / 1_000_000, loss)
                    @ (weights / weights.sum())))
            scores_by_index[index] = (
                base + scale * sum(scores_for_index),
                dict(source=VALUE_SOURCE, nativeIndex=index),
            )
    rows = []
    for card, (indices, ambiguous) in zip(CARDS, options):
        rows.append(dict(
            card=card, indices=indices if not ambiguous else None,
            expectedGPA=None if ambiguous else
            sum(scores_by_index[i][0] for i in indices) / len(indices),
            valueSources=[] if ambiguous else [scores_by_index[i][1] for i in indices],
            expectedGPAInterval=payoff_interval(t, budget), expectedQuestions=0,
            expectedTutoringMinutes=0, status="unresolved" if ambiguous else "scored",
        ))
    result = dict(id=root["id"], rows=rows, fixedEvidence=None)
    if not fixed:
        return result
    need(root.get("orderProof") == ORDER_PROOF, "Fixed symmetric order and local laws")
    if budget < 2 * cost:
        rows.append(dict(rows[1], card=CARDS[4]))
        result["fixedEvidence"] = dict(orderMasses=[], remainingBudget=budget, skippedBoth=True)
        return result
    remaining = budget - 2 * cost
    need(root["display"].get("orderEquivalent") is True, "Admitted final display/order equivalence")
    width = t["report_width"]
    need(type(width) in (int, float) and math.isfinite(width) and width > 0,
         "Positive finite report width")
    laws = [shapes(row, width) for row in prior]
    weight = laws[0][1][:, None] * laws[1][1][None, :]
    if remaining:
        displayed = root["display"]["singleReport"]
        need(len(displayed) == 2, "Both recipient display supports")
        da, db = [guesses(row) for row in displayed]
        need(len(da) == len(db) == 100, "Complete report display support")
        j, k, unknown = similar(da[:, None], db[None, :], remaining, loss)
        v = grid_utility(remaining, loss)
        ca, cb = [law[2] @ v.T for law in laws]
        r, s = np.ogrid[:100, :100]
        gain = (ca[r, j] + cb[s, 1000-j] + ca[r, k] + cb[s, 1000-k]) / 2
    else:
        v = grid_utility(0, loss)
        gain = np.broadcast_to(v[0] @ laws[0][0] + v[0] @ laws[1][0], weight.shape)
        unknown = np.zeros(weight.shape, dtype=bool)
    mass = float(weight.sum())
    need(abs(mass - 1) <= 1e-12, "Joint report mass conservation")
    unresolved_mass = float(weight[unknown].sum())
    ambiguous_count = int(np.count_nonzero(unknown))
    # A mathematically supported ambiguous pair can have an underflowed weight.
    value = None if ambiguous_count else base + scale * float(np.sum(weight * gain))
    rows.append(dict(
        card=CARDS[4], expectedGPA=value, expectedGPAInterval=payoff_interval(t, remaining),
        expectedQuestions=2, expectedTutoringMinutes=2 * cost / 1000,
        status="unresolved" if ambiguous_count else "scored",
    ))
    result["fixedEvidence"] = dict(
        orderMasses=[mass, mass], remainingBudget=remaining,
        unresolvedMass=unresolved_mass, ambiguousPairCount=ambiguous_count, skippedBoth=False,
    )
    return result
