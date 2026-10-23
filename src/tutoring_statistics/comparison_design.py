"""Original populations for baseline, saved-fit and fresh inference comparisons."""
from __future__ import annotations

from copy import deepcopy

TOTAL = {"baseline": 40000, "diagnostic": 480, "fresh": 26000}
FRESH_INDICES = (1, 2, 4, 7, 8, 9, 10, 13, 14, 16, 19, 20, 22)
SCOPE = {
    "baseline": "conditional_Gaussian_reference_not_human_power_or_193_recruit_sufficiency",
    "diagnostic": "fixed_existing_fit_development",
    "fresh": "fresh_conditional_calibration_not_prospective_paired_validation",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def coordinates(family, ordinal):
    require(family in TOTAL and type(ordinal) is int and 0 <= ordinal < TOTAL[family],
            "Ordinal outside original comparison population")
    if family == "fresh":
        return FRESH_INDICES[ordinal % 13], ordinal // 13 + 1
    if family == "diagnostic" or ordinal < 24000:
        return ordinal % 24 + 1, ordinal // 24 + 1
    eligible = tuple(i for i in range(1, 25) if (i - 1) % 6 < 4)
    return eligible[(ordinal - 24000) % 16], (ordinal - 24000) // 16 + 1001


def case_for(family, ordinal):
    from .joint_generation import measurement_plan
    index, rep = coordinates(family, ordinal)
    case = deepcopy(measurement_plan()["cases"][index - 1])
    design, response = (402610190, 502610190) if family == "fresh" else (202610190, 302610190)
    case.update(repID=rep, designSeed=design + 10000 * case["scenarioIndex"] + rep,
                responseSeed=response + 10000 * index + rep)
    return case


def independent_case(family, ordinal):
    """Use the independently authored plan and separate roster mapping."""
    from .independent_generation import expected_plan
    require(family in TOTAL and type(ordinal) is int and 0 <= ordinal < TOTAL[family],
            "Invalid independent comparison ordinal")
    templates = expected_plan()["cases"]
    if family == "fresh":
        from .independent_fresh import case_for_ordinal
        return case_for_ordinal(ordinal, [c for c in templates if c["caseIndex"] in FRESH_INDICES])
    widths = (24, 16)
    if ordinal < 1000 * widths[0]:
        rep, position = divmod(ordinal, widths[0])
        rep += 1
    else:
        rep, position = divmod(ordinal - 24000, widths[1])
        rep += 1001
    eligible = [c for c in templates if rep <= c["futureProposedB"]]
    case = deepcopy(eligible[position])
    case.update(repID=rep, designSeed=202610190 + 10000 * case["scenarioIndex"] + rep,
                responseSeed=302610190 + 10000 * case["caseIndex"] + rep)
    return case


def fit_key(family, case):
    if family != "fresh" and case["repID"] == 1:
        return case["caseID"]
    return f"{case['caseID']}__r{case['repID']:04d}"


def templates(family):
    return [case_for(family, i) for i in range(13 if family == "fresh" else 24)]


def planned(family, case):
    return 20 if family == "diagnostic" else 2000 if family == "fresh" else case["futureProposedB"]


def selection(family, *, full=False, start=0, limit=1):
    require(family in TOTAL, "Unknown comparison family")
    require(type(full) is bool and type(start) is int and type(limit) is int,
            "Invalid selection types")
    require(0 <= start < TOTAL[family] and limit > 0, "Invalid selection range")
    if full:
        require(start == 0 and limit == 1, "Full roster cannot be combined with start/limit")
        return range(TOTAL[family])
    require(start + limit <= TOTAL[family], "Selection exceeds original population")
    return range(start, start + limit)


def validate_row(family, row):
    case = independent_case(family, row["ordinal"])
    require(row["case"] == case and row["caseID"] == case["caseID"] and
            row["repID"] == case["repID"] and row["fitKey"] == fit_key(family, case),
            "Changed comparison scientific identity")
    require(type(row["technicalCompleted"]) is bool, "Missing technical status")
    if not row["technicalCompleted"]:
        require(family != "baseline" and row["candidateBranches"] is None and
                isinstance(row["technicalError"], dict), "Technical failure became a nonrejection")
    else:
        require(row["technicalError"] is None, "Completed row retains technical error")
    from .joint_generation import H10_IDS, TAILS
    methods = ["baselineBranches"]
    if family != "baseline" and row["technicalCompleted"]:
        methods.append("candidateBranches")
    for name in methods:
        branches = row[name]
        require([b["tailMode"] for b in branches] == list(TAILS), "Changed tail roster")
        require(branches[0]["coverage"] == branches[1]["coverage"], "Tails duplicate coverage")
        for branch in branches:
            nulls = case["trueNullIDs"][branch["tailMode"]]
            tests = branch["tests"]
            require(branch["trueNullIDs"] == nulls and set(tests) == set(H10_IDS) and
                    branch["familySize"] == 4, "Changed family/truth membership")
            require(all(t["status"] in ("available_candidate", "unavailable") and
                        type(t["holmReject"]) is bool for t in tests.values()), "Unknown test status")
            available = sum(t["status"] == "available_candidate" for t in tests.values())
            a4 = available == 4
            f = any(tests[k]["holmReject"] for k in nulls)
            r = any(t["holmReject"] for t in tests.values())
            d = any(tests[k]["holmReject"] for k in H10_IDS if k not in nulls)
            expected = dict(M=1, T=1, technicalMissing=0, A4=int(a4), F=int(f), F4=int(a4 and f),
                            R=int(r), R4=int(a4 and r), D=int(d), D4=int(a4 and d))
            require(branch["counts"] == expected and all(type(x) is int for x in branch["counts"].values())
                    and branch["completeFamilyAvailable"] is a4, "Changed event counts or availability")
            require(set(branch["coverage"]) == set(H10_IDS), "Changed coverage roster")
            for key, value in branch["coverage"].items():
                require(value["truth"] == case["delta"][key] and
                        all(type(value[k]) is int and value[k] in (0, 1) for k in ("A", "CIAvailable", "C"))
                        and value["C"] <= value["CIAvailable"] <= value["A"], "Changed coverage counts")
    return case
