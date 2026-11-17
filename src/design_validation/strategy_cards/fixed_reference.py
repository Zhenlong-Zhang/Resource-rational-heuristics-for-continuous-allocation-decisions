"""Independent binary64 checks of the four root cards and fixed-two card.

This reader does not import the producer. Source authentication belongs to the
input adapter; numerical agreement is not a narrow certified error enclosure.
"""
import math
import re
from fractions import Fraction

import numpy as np

CARDS = (
    "equal_minutes", "equal_projected_gpa", "all_to_higher_guess",
    "all_to_lower_guess", "ask_each_equal_projected_gpa",
)
DISPLAY = re.compile(r"(?:[1-9][0-9]?\.[0-9]|100\.0)\Z")
TOLERANCE = 2e-11
ORDER_PROOF = {
    "orders": [[0, 1], [1, 0]],
    "weights": ["1/2", "1/2"],
    "law": "independent_recipient_latents_and_recipient_local_reports",
    "normalization": "each_latent_row_divided_by_sum_over_reports_1_to_100",
    "posterior": "one_recipient_local_update_each",
    "terminal": "ordered_terminal_visible_means_and_remaining_budget_only",
    "questionCharges": "same_cost_twice_or_skip_both",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def visible(values):
    require(isinstance(values, list) and all(type(x) is str and DISPLAY.fullmatch(x)
                                           for x in values),
            "Expected source-admitted one-decimal labels")
    return np.array([int(x.replace(".", "")) for x in values], dtype=np.int64)


def prior_weights(log_hex):
    values = tuple(map(float.fromhex, log_hex))
    require(len(values) == 100 and all(map(math.isfinite, values)),
            "Complete source-ordered log law")
    weights = [math.exp(x - max(values)) for x in values]
    denominator = math.fsum(weights)
    return np.array([x / denominator for x in weights])


def weights(log_hex, width):
    prior = prior_weights(log_hex)
    atoms = [math.exp(-.5 * (d / width) ** 2) for d in range(100)]
    channels = [[atoms[abs(need - report)] for report in range(100)]
                for need in range(100)]
    channel = np.array([[x / math.fsum(row) for x in row] for row in channels])
    joint = (channel * prior[:, None]).T
    mass = np.array([math.fsum(row) for row in joint])
    require(np.all(mass > 0) and math.isclose(math.fsum(mass), 1, abs_tol=1e-13),
            "Positive normalized report support")
    return prior, mass, joint / mass[:, None]


def gain(surplus, loss):
    return np.sqrt(np.abs(surplus)) * np.where(surplus < 0, -loss, 1)


def targets(a, b, budget, loss):
    """Compare actual GPA gaps on both sides of the monotone signed crossing."""
    a, b = np.broadcast_arrays(a, b)
    left = np.zeros(a.shape, dtype=np.int64)
    right = np.full(a.shape, 1000, dtype=np.int64)
    for _ in range(10):
        mid = (left + right) // 2
        sign = budget * (2 * mid - 1000) - 100_000 * (a - b)
        left = np.where(sign < 0, mid, left)
        right = np.where(sign >= 0, mid, right)

    def gap(index):
        x = (budget * index - 100_000 * a) / 1_000_000
        y = (budget * (1000 - index) - 100_000 * b) / 1_000_000
        return np.abs(gain(x, loss) - gain(y, loss))

    lhs, rhs = gap(left), gap(right)
    unknown = (np.abs(lhs-rhs) <= 1e-12) & (left != right) & (lhs != 0) & (rhs != 0)
    choose_left = (lhs < rhs) | ((lhs == rhs) & (np.abs(left-500) < np.abs(right-500)))
    return np.where(choose_left, left, right), unknown


def grid_utility(budget, loss):
    allocations = np.arange(1001)[:, None] * budget / 1_000_000
    return gain(allocations - np.arange(1, 101), loss)


def close(actual, reference, reason, tolerance=TOLERANCE):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual-reference) <= tolerance, reason)
    return abs(actual-reference)


def verify_envelope(row, task, budget):
    """Check a support envelope with exact inequalities, not probabilities."""
    low, high = map(Fraction, row["expectedGPAInterval"])
    base = 2 * Fraction(task["gpa_baseline"])
    scale = Fraction(task["gpa_scale"])
    loss = Fraction(task["loss_multiplier"])
    require(scale > 0 and loss >= 0, "Valid payoff coefficients")
    require(low <= base - 2 * scale * loss * 10, "Universal loss-side support")
    upper_gain = (high - base) / (2 * scale)
    require(upper_gain >= 0 and upper_gain**2 >= Fraction(budget, 1000),
            "Universal gain-side support")
    if row["expectedGPA"] is not None:
        require(low <= Fraction(row["expectedGPA"]) <= high,
                "Diagnostic point within support enclosure")


def verify(root, result):
    """Check four or five rows; return discrepancy, coverage and scope diagnostics."""
    require("oldScores" not in root, "Cached oldScores are not public scoring inputs")
    task = root["task"]
    initial, price = task["total_millihours"], task["cost_millihours"]
    loss, base, scale = task["loss_multiplier"], 2 * task["gpa_baseline"], task["gpa_scale"]
    require(type(initial) is int and type(price) is int and initial > 0 and price > 0
            and task["allocation_denominator"] == 1000 and task["alpha"] == .5
            and len(root["logsHex"]) == 2, "Supported exact task")
    require(all(type(task[k]) in (int, float) and math.isfinite(task[k])
                for k in ("loss_multiplier", "gpa_baseline", "gpa_scale"))
            and loss >= 0 and scale > 0, "Finite payoff coefficients")
    rows = result["rows"]
    require(result["id"] == root["id"] and len(rows) in (4, 5)
            and tuple(r["card"] for r in rows) == CARDS[:len(rows)], "Exact root/card roster")
    priors = [prior_weights(log) for log in root["logsHex"]]
    display = visible(root["display"]["root"])
    require(display.shape == (2,), "Both ordered displayed guesses")
    a, b = display
    near, uncertain = targets(a, b, initial, loss)
    higher = 500 if a == b else (1000 if a > b else 0)
    indices = (500, int(near), higher, 1000 - higher)
    calculated, disagreements = {}, []
    for row, index, unknown in zip(rows[:4], indices, (False, uncertain, False, False)):
        verify_envelope(row, task, initial)
        require(row["expectedQuestions"] == row["expectedTutoringMinutes"] == 0,
                "No search charges for root cards")
        if unknown:
            require(row["status"] == "unresolved" and row["indices"] is None
                    and row["expectedGPA"] is None and row["valueSources"] == [],
                    "Preserve undecided discrete GPA gap")
            continue
        require(row["status"] == "scored" and row["indices"] == [index],
                "Independently derived visible-input target")
        require(row["valueSources"] == [dict(source="public_binary64_from_ordered_logs",
                                             nativeIndex=index)], "Fresh selected-index provenance")
        if index not in calculated:
            needs = np.arange(1, 101)
            scores = [math.fsum(gain(initial*j/1_000_000-needs, loss) * prior)
                      for j, prior in zip((index, 1000-index), priors)]
            calculated[index] = base + scale * math.fsum(scores)
        disagreements.append(close(row["expectedGPA"], calculated[index],
                                   "Expected GPA at original budget"))

    evidence = result["fixedEvidence"]
    unresolved_mass = None
    ambiguous_count = None
    if len(rows) == 4:
        require(evidence is None, "No fixed-two evidence on the root-only phase")
    else:
        require(root.get("orderProof") == ORDER_PROOF, "Fixed symmetric order and local laws")
        row = rows[4]
        if initial < 2 * price:
            require(row == dict(rows[1], card=CARDS[4]) and evidence == dict(
                orderMasses=[], remainingBudget=initial, skippedBoth=True),
                "Skip both and retain the checked no-ask card")
            unresolved_mass = 0.
            ambiguous_count = 0
        else:
            budget = initial - 2 * price
            require(row["expectedQuestions"] == 2 and row["expectedTutoringMinutes"] == 2 * price / 1000
                    and evidence["remainingBudget"] == budget and evidence["skippedBoth"] is False,
                    "Only two original fictional budget debits")
            require(root["display"].get("orderEquivalent") is True
                    and len(evidence["orderMasses"]) == 2, "Admitted commuting order and both masses")
            width = task["report_width"]
            require(type(width) in (int, float) and math.isfinite(width) and width > 0,
                    "Positive finite report width")
            verify_envelope(row, task, budget)
            laws = [weights(log, width) for log in root["logsHex"]]
            joint = np.outer(laws[0][1], laws[1][1])
            normalizer = math.fsum(joint.flat)
            close(normalizer, 1, "Independently normalized joint report mass", 1e-12)
            for mass in evidence["orderMasses"]:
                close(mass, normalizer, "Both orders cover the same full report mass", 1e-12)
            curves = grid_utility(budget, loss)
            if budget == 0:
                unresolved_mass, ambiguous_count = 0., 0
                utility = math.fsum(math.fsum(curves[0] * law[0]) for law in laws)
            else:
                displayed = root["display"]["singleReport"]
                require(len(displayed) == 2, "Both recipient display supports")
                a, b = map(visible, displayed)
                require(a.shape == b.shape == (100,), "Every report has an admitted JS display")
                target, ambiguous = targets(a[:, None], b[None, :], budget, loss)
                unresolved_mass = math.fsum(joint[ambiguous])
                ambiguous_count = int(np.count_nonzero(ambiguous))
                left, right = np.indices(joint.shape, sparse=True)
                ga, gb = [law[2] @ curves.T for law in laws]
                utility = math.fsum((joint * (ga[left, target] + gb[right, 1000-target])).flat)
            close(evidence["unresolvedMass"], unresolved_mass, "Conserved unresolved report mass")
            require(type(evidence["ambiguousPairCount"]) is int
                    and evidence["ambiguousPairCount"] == ambiguous_count,
                    "All ambiguous support pairs retained")
            require(row["status"] == ("unresolved" if ambiguous_count else "scored"),
                    "Ambiguous reports retain their evidence status")
            if ambiguous_count:
                require(row["expectedGPA"] is None, "Unresolved integration has no point estimate")
            else:
                value = base + scale * utility
                disagreements.append(close(row["expectedGPA"], value, "Independent fixed-two expected GPA"))
    return dict(
        maximumAbsoluteGPADisagreement=max(disagreements, default=0.),
        independentlyRescoredGPARows=len(disagreements), GPADisagreementTolerance=TOLERANCE,
        GPAIntervalScope="payoff_support", unresolvedMass=unresolved_mass,
        ambiguousPairCount=ambiguous_count,
        unresolvedCards=[row["card"] for row in rows if row["status"] == "unresolved"],
    )
