"""Fixed designs for magnitude validation and matched saved-input power.

These scientific manifests contain no execution or source-admission evidence.
"""
from __future__ import annotations

import copy


IDS = ("C1-C2:A", "C1-C2:B", "C1-C3:A", "C1-C3:B")
LINEAR_IDS = tuple(k + ":" + v for k in IDS for v in ("D", "S"))
ARMS = ("C1", "C2", "C3")
METHOD = "h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1"
UPSTREAM = "participant_CR2_Satterthwaite_v1"
MODES = ("fresh_validation", "saved_power_core")
ALL_MODES = MODES + ("fresh_joint_validation", "fresh_coverage_validation", "saved_coverage_s1")
FRESH_NAMESPACE = "r11-h10-magnitude-validation-v1-20261026"
SAVED_NAMESPACE = "r11-h10-magnitude-saved-power-v1-20261026"
TEMPLATES = (1, 2, 3, 4, 6)
CORE = (2, 4, 6, 8, 10, 12, 14, 16)
SEEDS = dict(retention=1102610260, design=1202610260, response=1302610260)
SLOPES = {1: (.25, -.25, .25, -.25, .25, -.25),
          2: (.25, -.25, -.25, .25, -.25, .25),
          3: (0., 0., 0., 0., 0., 0.),
          4: (.025, -.025, .025, -.025, -.025, .025),
          6: (.25, -.25, .25, -.25, .5, -.5)}


def require(value, message):
    if not value:
        raise ValueError(message)


def fresh_cases():
    result = []
    for index in TEMPLATES:
        values = SLOPES[index]
        slopes = {arm: dict(A=values[2*i], B=values[2*i+1]) for i, arm in enumerate(ARMS)}
        result.append(dict(caseID=f"h10magv1_initial_amount_v{index}_recruit193_mcar08",
            caseIndex=index, view="initial_amount", N=193, recruitedPerArm=193,
            fixedEligiblePerArm=None, populationRegime="recruit193_mcar08", eligibilityProbability=.8,
            parameters=dict(alpha=60., sdU=8., sigmaByArm={a: 12. for a in ARMS}, slopes=slopes)))
    return result


def expected_ordinals(mode):
    require(mode in ALL_MODES, "Unknown campaign mode")
    if mode not in ("fresh_validation", "saved_power_core"):
        from . import joint_interval_design
        return joint_interval_design.manifest(mode)["expectedOrdinals"]
    return (list(range(5*2129)) if mode == "fresh_validation" else
            [40*(rep-1)+index-1 for rep in range(1, 2001) for index in CORE])


def manifest(mode):
    """Return the full fixed scientific roster, never a runtime acceptance claim."""
    if mode in ("fresh_joint_validation", "fresh_coverage_validation", "saved_coverage_s1"):
        from . import joint_interval_design
        return joint_interval_design.manifest(mode)
    ids = expected_ordinals(mode)
    if mode == "fresh_validation":
        cases = fresh_cases()
    else:
        from . import power_reference
        cases = [case for case in power_reference.cells() if case["cellIndex"] in CORE]
        require([case["cellIndex"] for case in cases] == list(CORE), "Changed saved power roster")
    return dict(mode=mode, B=2129 if mode == "fresh_validation" else 2000,
        cases=cases, expectedOrdinals=ids, expectedTotal=len(ids),
        namespace=FRESH_NAMESPACE if mode == "fresh_validation" else SAVED_NAMESPACE,
        method=METHOD, upstreamMethod=UPSTREAM)


def case_for(manifest, ordinal):
    if manifest["mode"] in ("fresh_joint_validation", "fresh_coverage_validation", "saved_coverage_s1"):
        from . import joint_interval_design
        return joint_interval_design.case_for(manifest, ordinal)
    valid = (type(ordinal) is int and ((manifest["mode"] == "fresh_validation" and 0 <= ordinal < 10645)
             or (manifest["mode"] == "saved_power_core" and 0 <= ordinal < 80000 and ordinal % 40+1 in CORE)))
    require(valid, "Unplanned scientific ID")
    if manifest["mode"] == "fresh_validation":
        rep, position = ordinal // 5 + 1, ordinal % 5
        case = copy.deepcopy(manifest["cases"][position])
        case["repID"] = rep
        for role, base in SEEDS.items():
            case[role + "Seed"] = base + 100000*case["caseIndex"] + rep
    else:
        rep, index = ordinal // 40 + 1, ordinal % 40 + 1
        case = copy.deepcopy(next(v for v in manifest["cases"] if v["cellIndex"] == index))
        case.update(repID=rep, retentionSeed=802610200+10000*index+rep,
                    designSeed=902610200+10000*index+rep, responseSeed=1002610200+10000*index+rep)
    return case


def truth(case):
    slopes = case["parameters"]["slopes"]
    out = {}
    for key in IDS:
        a = slopes["C1"][key[-1]]; b = slopes[key[3:5]][key[-1]]
        out[key] = dict(C1=a, comparator=b, D=a-b, S=a+b,
                        magnitudeDifference=abs(b)-abs(a), trueNull=abs(a) >= abs(b))
    return out
