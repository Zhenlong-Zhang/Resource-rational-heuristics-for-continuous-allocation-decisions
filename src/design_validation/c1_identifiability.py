"""One prospectively fixed C1 diagnostic batch, no candidate adoption or search."""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import combinations
import math

import numpy as np

from .bounded_task import BoundedTask, belief_moments, expand_templates, heuristic_shares


def _balanced_information(configs):
    """Sxx extrema over the legal multisets and all 5/5 orientation assignments."""
    records = []
    for extra in range(3):
        x = []
        for i, config in enumerate(configs):
            means = [belief_moments(p)[0] for p in BoundedTask(config).initial_logs]
            value = (means[0]-means[1])/(config.total_millihours/1000)
            x.extend([value]*(4 if i == extra else 3))
        values = []
        for positive in combinations(range(10), 5):
            own = np.asarray(x).copy(); own[list(positive)] *= -1
            values.append(float(np.sum((own-own.mean())**2)))
        records.append({"extraScaleIndex": extra, "orientationAssignments": len(values),
            "minimumSxx": min(values), "maximumSxx": max(values),
            "absolutePredictorRange": [min(abs(v) for v in x), max(abs(v) for v in x)]})
    return records


def audit_candidates(design, spec):
    if spec.get("candidateDeltas") != [6, 8, 12] or spec.get("comparatorDeltas") != [0, 4] or spec.get("scales") != [.95, 1, 1.05] or spec.get("orientations") != [False, True]:
        raise ValueError("Frozen candidate/comparator batch changed")
    if spec.get("unchanged") != {"spreads": [3, 3], "reportWidth": 30, "budget": 120, "cost": 1.2, "cap": 20, "gridDenominator": 1000}:
        raise ValueError("Frozen noncandidate scientific parameters changed")
    expected = {"fivePointShare": .05, "historicalEightPointShare": .08, "strictDisjointShare": .10,
        "comparatorTolerance": 1e-12, "optimalityTolerance": 1e-10, "noSearchMargin": 1e-8}
    if spec.get("criteria") != expected: raise ValueError("Frozen C1 criteria changed")
    templates = [c for c in expand_templates(design) if c.condition_id == "C1"]
    if len(templates) != 3 or [c.scale for c in templates] != spec["scales"]:
        raise ValueError("Current C1 template scales differ")
    for c in templates:
        if (c.prior_centres != (50*c.scale, 50*c.scale) or c.prior_spreads != (3*c.scale, 3*c.scale)
                or c.report_width != 30*c.scale or c.total_millihours != round(120000*c.scale)
                or c.cost_millihours != round(1200*c.scale) or c.max_assessments != 20 or c.allocation_denominator != 1000
                or (c.alpha, c.loss_multiplier, c.gpa_baseline, c.gpa_scale) != (.5, 2, 3.5, .03)):
            raise ValueError("Current C1/model recipe differs")
    rows, pairwise, information = [], [], []
    for delta in [*spec["comparatorDeltas"], *spec["candidateDeltas"]]:
        kind = "symmetric_comparator" if delta == 0 else "old_failed_delta4_comparator" if delta == 4 else "fixed_candidate"
        configurations = []
        for base in templates:
            centres = ((50-delta/2)*base.scale, (50+delta/2)*base.scale)
            configurations.append(replace(base, prior_centres=centres, template_id=f"new_batch_delta{delta}_{base.template_id}"))
            for reverse in spec["orientations"]:
                config = replace(configurations[-1], prior_centres=centres[::-1] if reverse else centres,
                                 template_id=configurations[-1].template_id+f"_reverse{int(reverse)}")
                task = BoundedTask(config); means = [belief_moments(p)[0] for p in task.initial_logs]
                targets = heuristic_shares(*task.initial_logs, config.total_millihours)
                values = task.terminal_values(*task.initial_logs, config.total_millihours)
                share, optimum = task.terminal_solution(*task.initial_logs, config.total_millihours)
                utilities = {key: float(values[round(target*1000)]) for key, target in targets.items()}
                gap = abs(targets["equal_split"]-targets["equal_outcome"])
                regrets = {key: max(0., optimum-value) for key, value in utilities.items()}
                upper = task.perfect_information_after_first_cost(*task.initial_logs, config.total_millihours)
                state_id = f"delta{delta}_scale{base.scale}_reverse{int(reverse)}"
                rows.append({"stateId": state_id, "kind": kind, "delta": delta, "scale": base.scale,
                    "sideReversed": reverse, "config": asdict(config), "finiteInitialMeans": means,
                    "rawInitialGap": means[0]-means[1], "xInitial": (means[0]-means[1])/(config.total_millihours/1000),
                    "targets": targets, "costFreeOptimalShare": share, "costFreeOptimalExpectedUtility": optimum,
                    "heuristicExpectedUtility": utilities, "heuristicRegret": regrets,
                    "heuristicGPARegret": {key: config.gpa_scale*v for key, v in regrets.items()},
                    "EOESGapShare": gap, "EOESGapPP": gap*100, "EOESNativeSteps": round(gap*1000),
                    "nativeGridDistinguishable": round(gap*1000) >= 1,
                    "fivePointGate": gap >= .05-1e-12, "historicalEightPointGate": gap >= .08-1e-12,
                    "strictlyDisjointClosedFivePointMatchNeighborhoods": gap > .10,
                    "equalSplitCostFreeOptimal": regrets["equal_split"] <= 1e-10,
                    "acquisitionOnlyUpperBound": upper, "numericalNoSearchBoundPass": upper["upper_bound_gain"] < -1e-8,
                    "realArithmeticRMeanDomainPP": [-gap*100, gap*100],
                    "RVarianceRegion": {"formula": "G^2-mu^2 for |mu|<=G", "G": gap*100,
                                        "maximumVarianceAtZero": (gap*100)**2},
                    "candidateAdopted": False})
                for first, second in combinations(targets, 2):
                    spacing = abs(targets[first]-targets[second])
                    pairwise.append({"stateId": state_id, "first": first, "second": second,
                        "spacingPP": 100*spacing, "nativeSteps": round(1000*spacing), "coincident": spacing <= 1e-12})
        information.append({"delta": delta, "kind": kind,
            "tenMainBalancedProjectionInformation": _balanced_information(configurations)})
    current_support = {}
    for cid in ("C2", "C3"):
        own = [c for c in expand_templates(design) if c.condition_id == cid]
        magnitudes = []
        for config in own:
            means = [belief_moments(p)[0] for p in BoundedTask(config).initial_logs]
            magnitudes.append(abs((means[0]-means[1])/(config.total_millihours/1000)))
        if cid == "C3": balanced = _balanced_information(own)
        else:
            balanced = []
            for extra in range(9):
                x = [value for i,value in enumerate(magnitudes) for _ in range(2 if i == extra else 1)]
                sxx = []
                for reversed_side in combinations(range(10),5):
                    v=np.asarray(x).copy();v[list(reversed_side)]*=-1
                    sxx.append(float(np.sum((v-v.mean())**2)))
                balanced.append({"extraTemplate":own[extra].template_id,"orientationAssignments":len(sxx),
                    "minimumSxx":min(sxx),"maximumSxx":max(sxx)})
        current_support[cid] = {"sourceAbsolutePredictorRange":[min(magnitudes),max(magnitudes)],
            "legalTenMainBalancedInformation":balanced,
            "claim":"source schedule projection support, not common causal derivative"}
    return {"schema": "fixed-c1-identifiability-batch-v1", "states": rows, "pairwise": pairwise,
        "projectionInformation": information, "candidateStates": 18, "comparatorStates": 12,
        "unchangedC2C3PredictorComparators":current_support,
        "sourceScope": "deterministic finite initial states; no terminal cost, stochastic policies or human labels",
        "candidateAdopted": False, "batchExpandedAfterOutcomes": False,
        "limitations": ["Acquisition bounds do not make ES terminal-optimal.",
            "Finite spacings/Sxx do not establish human sensitivity or common causal slopes.",
            "Raw finite numerical calculations are not universal analytic or machine certificates."]}
