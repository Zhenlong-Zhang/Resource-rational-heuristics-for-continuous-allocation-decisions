"""Independent compact-event checks and fixed magnitude report reduction.

This module does not import producer design, composition or reduction code.
Source/fit admission and independent numerical references belong to the caller.
"""
from __future__ import annotations

from collections import Counter
import math


IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
LINEAR_IDS = tuple(key + ":" + kind for key in IDS for kind in ("D", "S"))
ARMS = ("C1", "C2", "C3")
MODES = ("fresh_validation", "saved_power_core", "fresh_joint_validation",
         "fresh_coverage_validation", "saved_coverage_s1")
CORE = (2, 4, 6, 8, 10, 12, 14, 16)


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(actual, expected, label="identity"):
    require(type(actual) is type(expected), "Changed exact type: " + label)
    if type(expected) is dict:
        require(set(actual) == set(expected), "Changed exact fields: " + label)
        for key, value in expected.items():
            exact(actual[key], value, label + "." + key)
    elif type(expected) is list:
        require(len(actual) == len(expected), "Changed exact length: " + label)
        for index, value in enumerate(expected):
            exact(actual[index], value, label + "[" + str(index) + "]")
    else:
        require(actual == expected, "Changed exact value: " + label)


def near(actual, expected, label="number"):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Changed fields: " + label)
        for key, value in expected.items():
            near(actual[key], value, label + "." + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed length: " + label)
        for index, value in enumerate(expected):
            near(actual[index], value, label + "[" + str(index) + "]")
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=2e-9, rel_tol=2e-8), "Numerical disagreement: " + label)
    else:
        exact(actual, expected, label)


def truth(case):
    result = {}
    for key, reference, need in ((IDS[0], "C2", "A"), (IDS[1], "C2", "B"),
                                 (IDS[2], "C3", "A"), (IDS[3], "C3", "B")):
        a, b = case["parameters"]["slopes"]["C1"][need], case["parameters"]["slopes"][reference][need]
        result[key] = dict(C1=a, comparator=b, D=a-b, S=a+b,
                           magnitudeDifference=abs(b)-abs(a), trueNull=abs(a) >= abs(b))
    return result


def expected_ids(mode):
    require(mode in MODES, "Unknown campaign mode")
    if mode == "fresh_joint_validation":
        return list(range(10645))
    if mode == "fresh_coverage_validation":
        return list(range(4258))
    if mode == "saved_coverage_s1":
        return [40*(rep-1)+32 for rep in range(1, 2001)]
    return list(range(10645)) if mode == "fresh_validation" else [
        40*(rep-1)+cell-1 for rep in range(1, 2001) for cell in CORE]


def case_for(ctx, ordinal):
    """Map an admitted scientific manifest without private source-audit imports."""
    manifest = ctx["manifest"]
    require(type(ordinal) is int and ordinal in ctx["expectedSet"], "Unplanned ordinal")
    if manifest["mode"].startswith("fresh_"):
        n = 2 if manifest["mode"] == "fresh_coverage_validation" else 5
        case = dict(manifest["cases"][ordinal % n], repID=ordinal//n+1)
        bases = {"fresh_validation": dict(retention=1102610260, design=1202610260, response=1302610260),
                 "fresh_joint_validation": dict(retention=1402610270, design=1502610270, response=1602610270),
                 "fresh_coverage_validation": dict(retention=1702610280, design=1802610280, response=1902610280)}[manifest["mode"]]
        stride, index = 100000, case["caseIndex"]
    else:
        index, rep = ordinal % 40+1, ordinal//40+1
        case = dict(next(value for value in manifest["cases"] if value["cellIndex"] == index), repID=rep)
        bases, stride = dict(retention=802610200, design=902610200, response=1002610200), 10000
    case.update(**{role+"Seed": base+stride*index+case["repID"] for role, base in bases.items()})
    return case


def compact_event(record, references, compositions):
    methods = {}
    for name in ("baseline", "candidate"):
        source, comp = record["methods"][name], compositions[name]
        completed = comp["status"] == "complete"
        tests = {key: dict(available=comp["tests"][key]["status"] == "available",
            trueNull=record["truth"][key]["trueNull"], pMagnitude=comp["tests"][key]["pMagnitude"],
            holmP=comp["tests"][key]["holmP"], rejected=comp["tests"][key]["holmReject"]) for key in IDS}
        linear = {}
        for key, ref in references[name].items():
            target = record["truth"][key[:-2]][key[-1]]
            available = ref["status"] == "available"
            linear[key] = dict(available=available, truth=target, estimate=ref["estimate"], SE=ref["SE"], df=ref["df"],
                covered=ref["CI"][0] <= target <= ref["CI"][1] if available else None,
                lessP=ref["lessP"], greaterP=ref["greaterP"], lessTrueNull=target >= 0, greaterTrueNull=target <= 0)
        nulls = [key for key in IDS if tests[key]["trueNull"]]
        alternatives = [key for key in IDS if not tests[key]["trueNull"]]
        methods[name] = dict(status=source["status"], technicalCompleted=completed,
            availableCount=sum(value["available"] for value in tests.values()),
            allAvailable=all(value["available"] for value in tests.values()),
            R=any(value["rejected"] for value in tests.values()) if completed else None,
            F=any(tests[key]["rejected"] for key in nulls) if completed and nulls else None,
            D=any(tests[key]["rejected"] for key in alternatives) if completed and alternatives else None,
            tests=tests, references=linear)
    return dict(ordinal=record["ordinal"], caseID=record["caseID"], repID=record["repID"],
                populationCounts=record["populationCounts"], truth=record["truth"], methods=methods)


def interval(k, n):
    require(type(k) is int and type(n) is int and 0 <= k <= n, "Invalid binomial counts")
    if n == 0:
        return None
    from scipy.special import betaincinv
    return [float(betaincinv(k, n-k+1, .025)) if k else 0.,
            float(betaincinv(k+1, n-k, .975)) if k != n else 1.]


def rate(k, completed, planned):
    require(all(type(value) is int for value in (k, completed, planned))
            and 0 <= k <= completed <= planned, "Invalid planned/technical denominator")
    return dict(count=k, M=planned, T=completed,
        fullRate=k/planned if completed == planned and planned else None,
        fullCI=interval(k, planned) if completed == planned else None,
        completedRate=k/completed if completed else None, completedCI=interval(k, completed),
        unknownEventBounds=[k/planned, (k+planned-completed)/planned] if planned else None,
        complete=completed == planned)


def distribution(values):
    clean = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in clean), "Nonfinite independent compact diagnostic")
    count = len(clean)
    mean = math.fsum(clean)/count if count else None
    sd = math.sqrt(math.fsum((value-mean)**2 for value in clean)/(count-1)) if count > 1 else None
    return dict(n=count, missing=len(values)-count, mean=mean, sd=sd,
                minimum=min(clean) if count else None, maximum=max(clean) if count else None)


def measurement(rows, planned):
    complete = {name: sum(row["methods"][name]["technicalCompleted"] for row in rows)
                for name in ("baseline", "candidate")}
    return dict(M=planned, present=len(rows), candidateT=complete["candidate"], baselineT=complete["baseline"],
        jointT=sum(all(row["methods"][name]["technicalCompleted"] for name in complete) for row in rows),
        missing=planned-len(rows), candidateTechnicalUnresolved=len(rows)-complete["candidate"],
        baselineTechnicalUnresolved=len(rows)-complete["baseline"], complete=complete["candidate"] == planned,
        bothMethodsComplete=all(value == planned for value in complete.values()))


def method_report(rows, method, planned, fixed_truth):
    all_events = [row["methods"][method] for row in rows]
    observed = [value for value in all_events if value["technicalCompleted"]]
    a4 = [value for value in observed if value["allAvailable"]]
    T = len(observed)
    result = dict(M=planned, T=T, technicalUnresolved=len(rows)-T, missing=planned-len(rows), A4=len(a4),
        statisticalUnavailableFamilies=sum(value["availableCount"] < 4 for value in observed),
        family={}, contrasts={}, linearReferences={})
    applicability = dict(R=True, F=any(value["trueNull"] for value in fixed_truth.values()),
                         D=any(not value["trueNull"] for value in fixed_truth.values()))
    for event in ("R", "F", "D"):
        if not applicability[event]:
            result["family"][event] = None
            continue
        count = sum(value[event] is True for value in observed)
        conditional = sum(value[event] is True for value in a4)
        result["family"][event] = dict(all=rate(count, T, planned),
            allAvailable=dict(count=conditional, denominator=len(a4),
                rate=conditional/len(a4) if a4 else None, CI=interval(conditional, len(a4))))
    for key in IDS:
        tests = [value["tests"][key] for value in observed]
        available = [value for value in tests if value["available"]]
        result["contrasts"][key] = dict(rejection=rate(sum(value["rejected"] is True for value in tests), T, planned),
            available=len(available), availableRejection=rate(sum(value["rejected"] is True for value in available), len(available), len(available)))
    for key in LINEAR_IDS:
        refs = [value["references"][key] for value in observed]
        available = [value for value in refs if value["available"]]
        true_value = fixed_truth[key[:-2]][key[-1]]
        tails = {}
        for direction in ("less", "greater"):
            thresholds = []
            for threshold in (.05, .025, .0125, .05/6, .00625):
                count = sum(value[direction+"P"] < threshold for value in available)
                thresholds.append(dict(threshold=threshold, all=rate(count, T, planned),
                                       available=rate(count, len(available), len(available))))
            tails[direction] = dict(trueNull=true_value >= 0 if direction == "less" else true_value <= 0,
                                    rejectionByThreshold=thresholds)
        covered = sum(value["covered"] is True for value in available)
        result["linearReferences"][key] = dict(available=len(available), coverageAll=rate(covered, T, planned),
            coverageAvailable=rate(covered, len(available), len(available)), tails=tails,
            estimate=distribution([value["estimate"] for value in refs]),
            SE=distribution([value["SE"] for value in refs]), df=distribution([value["df"] for value in refs]))
    return result


def reduce_rows(ctx, rows):
    seen = set()
    for row in rows:
        ordinal = row["ordinal"]
        require(type(ordinal) is int and ordinal in ctx["expectedSet"] and ordinal not in seen, "Duplicate/unplanned compact ID")
        seen.add(ordinal)
        case = case_for(ctx, ordinal)
        require(row["caseID"] == case["caseID"] and row["repID"] == case["repID"], "Wrong compact scientific mapping")
        exact(row["truth"], truth(case), "compact fixed truth")
    output = []
    for case in ctx["manifest"]["cases"]:
        selected = [row for row in rows if row["caseID"] == case["caseID"]]
        fixed_truth = truth(case)
        joint = [row for row in selected if all(row["methods"][name]["technicalCompleted"] for name in ("baseline", "candidate"))]
        transitions = Counter(("1" if row["methods"]["baseline"]["R"] else "0")+
                              ("1" if row["methods"]["candidate"]["R"] else "0") for row in joint)
        output.append(dict(caseID=case["caseID"], case=case, truth=fixed_truth,
            measurement=measurement(selected, ctx["manifest"]["B"]),
            methods={name: method_report(selected, name, ctx["manifest"]["B"], fixed_truth) for name in ("baseline", "candidate")},
            paired=dict(jointCompleted=len(joint), familyRejectionTransitions={key: transitions[key] for key in ("00", "01", "10", "11")}),
            populationCounts={arm: distribution([row["populationCounts"][arm] for row in selected]) for arm in ARMS}))
    return output


def compare_report(actual, expected, label="report"):
    """Exact categories/counts; only recomputed continuous leaves use tolerance."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Changed report fields: " + label)
        for key, value in expected.items():
            compare_report(actual[key], value, label + "." + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "Changed report shape: " + label)
        for index, value in enumerate(expected):
            compare_report(actual[index], value, label + "[" + str(index) + "]")
    elif type(expected) is float:
        near(actual, expected, label)
    else:
        exact(actual, expected, label)
