"""Frozen joint-validation and interval-diagnostic scientific designs.

Historical RNG namespaces are scientific identifiers, not private runtime paths.
No fitted outcomes are used to select an input or its coefficient law.
"""
import copy

from .magnitude_design import ARMS, require

JOINT_G = 0.8288338510706277
MODES = ("fresh_joint_validation", "fresh_coverage_validation", "saved_coverage_s1")
NAMESPACES = {
    "fresh_joint_validation": "r11-h10-joint-null-validation-v1-20261027",
    "fresh_coverage_validation": "r11-h10-kr-coverage-validation-v1-20261028",
    "saved_coverage_s1": "h10-power-design-v1-20261020",
}
SEEDS = {
    "fresh_joint_validation": dict(retention=1402610270, design=1502610270, response=1602610270),
    "fresh_coverage_validation": dict(retention=1702610280, design=1802610280, response=1902610280),
}
COVERAGE_TARGETS = {
    1: {"C1-C2:B:D": dict(L=[0, 0, 0, 0, 0, 0, 0, 1, 0], truth=0.)},
    2: {"C1-C2:A:S": dict(L=[0, 0, 0, 2, 0, 1, 0, 0, 0], truth=.8),
        "C1-C3:A:S": dict(L=[0, 0, 0, 2, 0, 1, 1, 0, 0], truth=.8)},
}


def cases(mode):
    require(mode in MODES, "Unknown closed design")
    if mode == "saved_coverage_s1":
        from .power_reference import cells
        return [case for case in cells() if case["cellIndex"] == 33]
    if mode == "fresh_joint_validation":
        g = JOINT_G
        rows = (("far_d", (g, -g, g, -g, g, -g)),
                ("far_s", (g, -g, -g, g, -g, g)),
                ("intersection", (0., 0., 0., 0., 0., 0.)),
                ("near_zero", (.025, -.025, .025, -.025, -.025, .025)),
                ("far_partial", (g, -g, g, -g, 2*g, -2*g)))
        prefix = "h10jointv1_initial_amount_"
    else:
        rows = (("s1", (.25, -.25, .25, -.25, .25, -.25)),
                ("s5_l20", (.25, -.25, .55, -.55, .55, -.55)))
        prefix = "h10covv1_initial_amount_"
    return [dict(caseID=prefix+suffix+"_recruit193_mcar08", caseIndex=index,
                 view="initial_amount", N=193, recruitedPerArm=193,
                 populationRegime="recruit193_mcar08", eligibilityProbability=.8,
                 fixedEligiblePerArm=None, parameters=dict(alpha=60., sdU=8.,
                 sigmaByArm={arm: 12. for arm in ARMS},
                 slopes={arm: dict(A=values[2*i], B=values[2*i+1]) for i, arm in enumerate(ARMS)}))
            for index, (suffix, values) in enumerate(rows, 1)]


def manifest(mode):
    roster = cases(mode)
    saved = mode == "saved_coverage_s1"
    ids = [40*(rep-1)+32 for rep in range(1, 2001)] if saved else list(range(2129*len(roster)))
    return dict(mode=mode, B=2000 if saved else 2129, cases=roster,
                expectedOrdinals=ids, expectedTotal=len(ids), namespace=NAMESPACES[mode],
                method="h10_magnitude_DS_IUT_Bonferroni2_Holm4_candidate_v1",
                upstreamMethod="participant_CR2_Satterthwaite_v1")


def case_for(plan, ordinal):
    mode = plan["mode"]
    require(type(ordinal) is int and ordinal in plan["expectedOrdinals"], "Unplanned closed-study input")
    if mode == "saved_coverage_s1":
        value = copy.deepcopy(plan["cases"][0])
        rep = ordinal//40+1
        value.update(repID=rep, retentionSeed=802610200+330000+rep,
                     designSeed=902610200+330000+rep, responseSeed=1002610200+330000+rep)
    else:
        rep, position = divmod(ordinal, len(plan["cases"]))
        value = copy.deepcopy(plan["cases"][position])
        value.update(repID=rep+1, **{role+"Seed": base+100000*value["caseIndex"]+rep+1
                                  for role, base in SEEDS[mode].items()})
    return value
