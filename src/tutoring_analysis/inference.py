"""Prospective independent-person tutoring analysis, not a human-study finding.

Consumes previously integrity-checked metrics. Finite posterior/target/payoff math stays
in the existing descriptive pipeline. Functional synthetic tests are not calibration.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

CONDITIONS = tuple(f"C{i}" for i in range(1, 6))
FAMILIES = {"acquisition": 2, "responsiveness": 3, "resemblance": 2, "lower_need": 2}
POLICIES = ("global_holm", "within_family_holm", "four_family_alpha_budget")
CORE = (
    ("acquisition_C3_C2", "acquisition", "anyQuestion", {"C3": 1., "C2": -1.}, "greater"),
    ("acquisition_C5_C4", "acquisition", "anyQuestion", {"C5": 1., "C4": -1.}, "greater"),
    ("initial_C2", "responsiveness", "slope_initial", {"C2": 1.}, "greater"),
    ("initial_C3", "responsiveness", "slope_initial", {"C3": 1.}, "greater"),
    ("initial_C4", "responsiveness", "slope_initial", {"C4": 1.}, "less"),
    ("resemblance_C2", "resemblance", "EOResemblance", {"C2": 1.}, "greater"),
    ("resemblance_C3", "resemblance", "EOResemblance", {"C3": 1.}, "greater"),
    ("lower_C4", "lower_need", "lowerNeedDirectional", {"C4": 1.}, "greater"),
    ("lower_C5", "lower_need", "lowerNeedDirectional", {"C5": 1.}, "greater"),
)
SLOPES = {"slope_initial": "xInitial", "slope_initial_raw": "initialNeedGapRaw",
          "slope_terminal": "xTerminal"}
MEANS = ("anyQuestion", "EOResemblance", "lowerNeedDirectional",
         "distancePP_equal_split", "distancePP_equal_outcome")
NUMERIC = (*MEANS, *SLOPES.values(), "shareA", "questions", "scale", "gapOffset")
BOOLS = ("analysisEligible", "terminalTie")
SXX_TOL = 1e-12


def scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError("SciPy is required for t/nct inference; install the declared dependency in an isolated supported runtime. No normal/R fallback.") from exc
    return stats


def _number(value, key):
    if value is None or value == "": return None
    if isinstance(value, bool): raise ValueError(f"{key}: boolean is not numeric")
    try: out = float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"{key}: expected finite number") from exc
    if not math.isfinite(out): raise ValueError(f"{key}: nonfinite value")
    return out


def _boolean(value, key):
    if value is True or value == "True": return True
    if value is False or value == "False": return False
    raise ValueError(f"{key}: expected explicit True/False")


def validate_rows(rows: list[dict], origin: str, purpose: str) -> list[dict]:
    """No inferred authentication and no override of preserved QA row provenance."""
    if (origin, purpose) not in (("synthetic_design", "synthetic_validation"),
                                ("human_declared", "human_analysis")):
        raise ValueError("Unsupported purpose/origin; software_check and unclassified cannot execute inference")
    keys, participants, sessions = set(), {}, {}
    out = []
    for r in rows:
        t = dict(r)
        if t.get("profileId", "pilot-4p10m-v1") != "pilot-4p10m-v1" or t.get("questionLimitRule", "count_cap_20") != "count_cap_20":
            raise ValueError("This inferential procedure is scoped to the historical capped pilot profile; budget-only records require a prospective specification")
        for key in ("participantId", "sessionId", "trialId", "conditionId", "stage"):
            if not isinstance(t.get(key), str) or not t[key].strip(): raise ValueError(f"Missing identity {key}")
            if t[key] != t[key].strip(): raise ValueError(f"Whitespace identity {key}")
        p, sid, cid = t["participantId"], t["sessionId"], t["conditionId"]
        if cid not in CONDITIONS or t["stage"] not in ("main", "practice"): raise ValueError("Unknown condition/stage")
        if t.get("dataRole") != origin: raise ValueError("Row dataRole disagrees with declared origin; no provenance override")
        index = _number(t.get("stageIndex"), "stageIndex")
        maximum = 10 if t["stage"] == "main" else 4
        if index is None or index != int(index) or not 1 <= index <= maximum: raise ValueError("Invalid pilot stage index")
        t["stageIndex"] = int(index)
        if t["trialId"] != f"{t['stage']}-{int(index):02d}": raise ValueError("Trial identity disagrees with pilot stage index")
        key = (sid, t["trialId"])
        if key in keys: raise ValueError("Duplicate/conflicting session-trial rows")
        keys.add(key)
        if p in participants and participants[p] != (sid, cid): raise ValueError("Participant has multiple sessions/conditions")
        if sid in sessions and sessions[sid] != (p, cid): raise ValueError("Session has conflicting participant/condition owner")
        participants[p], sessions[sid] = (sid, cid), (p, cid)
        for field in BOOLS:
            if field not in t: raise ValueError(f"Missing field {field}")
            t[field] = _boolean(t[field], field)
        for field in NUMERIC:
            if field not in t: raise ValueError(f"Missing field {field}")
            t[field] = _number(t[field], field)
        for field, low, high in (("shareA", 0, 1), ("anyQuestion", 0, 1),
                                  ("lowerNeedDirectional", -.5, .5),
                                  ("distancePP_equal_split", 0, 100), ("distancePP_equal_outcome", 0, 100)):
            if t[field] is not None and not low <= t[field] <= high: raise ValueError(f"{field} outside domain")
        if t["anyQuestion"] not in (0., 1.): raise ValueError("anyQuestion must be recorded trial indicator")
        q = t["questions"]
        if q is None or q != int(q) or not 0 <= q <= 20 or bool(q) != bool(t["anyQuestion"]): raise ValueError("Invalid paid question count/indicator")
        if t["analysisEligible"] and (t["stage"] != "main" or t.get("exclusionReasons", "")): raise ValueError("Eligibility contradicts stage/exclusions")
        if t["terminalTie"] and t["lowerNeedDirectional"] not in (None,0.): raise ValueError("Terminal ties must contribute zero directional L")
        # R is paired within the same trial, never difference of unrelated subsets.
        d0, d1, resemblance = (t[k] for k in ("distancePP_equal_split", "distancePP_equal_outcome", "EOResemblance"))
        if (resemblance is None) != (d0 is None or d1 is None): raise ValueError("Paired resemblance missingness disagrees with trial distances")
        if all(v is not None for v in (d0, d1, resemblance)) and abs(resemblance - (d0-d1)) > 1e-9:
            raise ValueError("Paired EO resemblance disagrees with trial distances")
        out.append(t)
    return out


def participant_slope(rows, predictor, model="unadjusted") -> dict:
    terms = [] if model == "unadjusted" else ["stageIndex"] if model == "order_only" else ["stageIndex", "scale", "gapOffset"] if model == "order_recipe" else None
    if terms is None: raise ValueError("Unknown prespecified slope model")
    valid = [r for r in rows if r["shareA"] is not None and r[predictor] is not None and all(r[k] is not None for k in terms)]
    x = np.asarray([r[predictor] for r in valid], dtype=float)
    n = len(valid); sxx = float(np.sum((x-x.mean())**2)) if n else 0.
    levels = len(set(float(v) for v in x))
    support = {"observations": n, "predictorLevels": levels, "Sxx": sxx,
               "predictorMinimum": float(x.min()) if n else None, "predictorMaximum": float(x.max()) if n else None,
               "predictorSpan": float(np.ptp(x)) if n else None, "model": model}
    if n < 3 or levels < 2 or sxx <= SXX_TOL:
        return {**support, "status": "unavailable", "reason": "insufficient_varying_predictor", "value": None}
    columns = [np.ones(n), x-x.mean()]
    for name in terms:
        z = np.asarray([r[name] for r in valid], dtype=float)
        columns.append(z-z.mean())
    design = np.column_stack(columns); rank = int(np.linalg.matrix_rank(design)); residual_df = n-design.shape[1]
    support.update(rank=rank, columns=design.shape[1], residualDf=residual_df)
    if rank != design.shape[1] or residual_df <= 0:
        return {**support, "status": "unavailable", "reason": "rank_or_residual_df_failure_no_fallback", "value": None}
    coef = np.linalg.lstsq(design, np.asarray([r["shareA"] for r in valid]), rcond=None)[0]
    return {**support, "status": "available", "reason": None, "value": float(coef[1]),
            "allocationChangeOverSpan": float(coef[1]*np.ptp(x))}


def participant_values(rows, endpoint, eligibility="complete_ten", window="all", model="unadjusted", tie_free=False) -> list[dict]:
    if eligibility not in ("complete_ten", "available_history") or window not in ("all", "early", "late"):
        raise ValueError("Unknown eligibility/window")
    people = defaultdict(list)
    for r in rows: people[r["participantId"]].append(r)
    result=[]
    for p, own in sorted(people.items()):
        main = [r for r in own if r["analysisEligible"]]
        complete = {r["stageIndex"] for r in main} == set(range(1, 11)) and len(main) == 10
        selected = [r for r in main if window == "all" or (r["stageIndex"] <= 5 if window == "early" else r["stageIndex"] > 5)]
        if tie_free: selected = [r for r in selected if not r["terminalTie"]]
        pr = {"participantId": p, "sessionId": own[0]["sessionId"], "conditionId": own[0]["conditionId"],
              "mainTrials": len(main), "completeTen": complete, "endpoint": endpoint,
              "eligibility": eligibility, "window": window, "model": model, "tieFree": tie_free}
        if eligibility == "complete_ten" and not complete:
            pr.update(status="unavailable", reason="incomplete_analytical_ten_main_history", value=None, observations=len(selected))
        elif endpoint in SLOPES:
            if endpoint in ("slope_initial", "slope_initial_raw") and pr["conditionId"] in ("C1", "C5"):
                pr.update(status="unavailable", reason="retained_design_initial_predictor_structurally_constant", value=None, observations=len(selected))
            elif endpoint == "slope_terminal" and pr["conditionId"] == "C5" and not any(r["questions"] for r in selected):
                pr.update(status="unavailable", reason="C5_terminal_variation_requires_recorded_paid_questions", value=None, observations=len(selected))
            else: pr.update(participant_slope(selected, SLOPES[endpoint], model))
        else:
            values=[r[endpoint] for r in selected if r[endpoint] is not None]
            pr.update(value=float(np.mean(values)) if values else None, observations=len(values),
                      status="available" if values else "unavailable", reason=None if values else "no_valid_endpoint_observations")
        pr["selectedMainRecords"]=len(selected)
        pr["endpointMissingWithinSelectedRecords"]=len(selected)-pr["observations"]
        pr["endpointMissingWithinCompleteHistory"]=bool(complete and pr["endpointMissingWithinSelectedRecords"]>0)
        result.append(pr)
    return result


def contrast(values: list[dict], weights: dict[str, float], side="greater", confidence=.95) -> dict:
    if side not in ("greater", "less") or not 0 < confidence < 1: raise ValueError("Invalid sidedness/confidence")
    if not weights or any(k not in CONDITIONS or not math.isfinite(v) or v == 0 for k,v in weights.items()): raise ValueError("Invalid independent-condition weights")
    groups={c:[float(r["value"]) for r in values if r["conditionId"]==c and r["status"]=="available" and r["value"] is not None] for c in weights}
    summary={c:{"participants":len(v), "mean":float(np.mean(v)) if v else None,
                "variance":float(np.var(v,ddof=1)) if len(v)>1 else None} for c,v in groups.items()}
    estimate=sum(weights[c]*summary[c]["mean"] for c in weights) if all(v["mean"] is not None for v in summary.values()) else None
    base={"weights":weights,"side":side,"groups":summary,"estimate":estimate,"confidence":confidence,
          "independentUnit":"participant","weighting":"equal participant within each condition; fixed condition weights",
          "method":"one_sample_t" if len(weights)==1 else "independent_weighted_Welch_Satterthwaite",
          "status":"unavailable","reason":None,"standardError":None,"df":None,"statistic":None,"pValue":None,"interval":None}
    if any(v["participants"]<2 for v in summary.values()): return {**base,"reason":"insufficient_participants_in_required_condition"}
    pieces=[weights[c]**2*v["variance"]/v["participants"] for c,v in summary.items()]
    variance=sum(pieces)
    if variance<=0 or not math.isfinite(variance): return {**base,"reason":"zero_or_nonfinite_standard_error"}
    df=variance**2/sum(p*p/(summary[c]["participants"]-1) for c,p in zip(weights,pieces))
    se=math.sqrt(variance); stat=estimate/se; stats=scipy_stats()
    probability=float(stats.t.sf(stat,df) if side=="greater" else stats.t.cdf(stat,df))
    half=float(stats.t.ppf((1+confidence)/2,df))*se
    return {**base,"status":"available","reason":None,"standardError":se,"df":df,"statistic":stat,
            "pValue":probability,"interval":[estimate-half,estimate+half]}


def _holm(items, alpha):
    ordered=sorted(items,key=lambda item:(1. if item["status"]!="available" else item["pValue"], item["id"]))
    adjusted=0.; alive=True; output={}
    for rank,item in enumerate(ordered):
        p=1. if item["status"]!="available" else item["pValue"]
        if not isinstance(p,(float,int)) or not math.isfinite(p) or not 0<=p<=1: raise ValueError("Invalid nominal p")
        threshold=alpha/(len(items)-rank); adjusted=max(adjusted,(len(items)-rank)*p)
        alive=alive and p<=threshold
        available=item["status"]=="available"
        output[item["id"]]={"plannedSlots":len(items),"rank":rank+1,"stepThreshold":threshold,
                            "adjustedP":min(1.,adjusted) if available else None,"reject":bool(alive and available)}
    return output


def multiplicity(core: list[dict], policy="global_holm", alpha=.05) -> dict:
    if policy not in POLICIES or not math.isfinite(alpha) or not 0<alpha<1: raise ValueError("Invalid multiplicity policy/alpha")
    expected={x[0]:x[1] for x in CORE}
    if len(core)!=9 or {r["id"]:r["family"] for r in core}!=expected: raise ValueError("Multiplicity requires the prospectively fixed nine slots")
    adjusted={}
    if policy=="global_holm": adjusted=_holm(core,alpha)
    else:
        for family in FAMILIES:
            adjusted.update(_holm([r for r in core if r["family"]==family],alpha/4 if policy=="four_family_alpha_budget" else alpha))
    rows=[{**r,**adjusted[r["id"]]} for r in core]
    return {"policy":policy,"alpha":alpha,"slots":9,"retainedUnavailableSlots":sum(r["status"]!="available" for r in core),
            "globalAnyGuarantee":policy!="within_family_holm","guaranteeConditionalOnValidUpstreamPValues":True,
            "anyCoreSuccess":any(r["reject"] for r in rows),"allCoreSuccess":all(r["reject"] for r in rows),"tests":rows}


def fit(rows: list[dict], origin="synthetic_design", purpose="synthetic_validation", policy="global_holm", alpha=.05) -> dict:
    rows=validate_rows(rows,origin,purpose)
    cache={}; participant_output=[]; tests=[]
    def run(identifier,family,endpoint,weights,side,role="supporting",eligibility="complete_ten",window="all",model="unadjusted",tie_free=False,force=None):
        key=(endpoint,eligibility,window,model,tie_free)
        if key not in cache:
            cache[key]=participant_values(rows,*key[:3],model=key[3],tie_free=key[4]);participant_output.extend(cache[key])
        result=contrast(cache[key],weights,side)
        if force: result.update(status="unavailable",reason=force,estimate=None,statistic=None,pValue=None,interval=None,standardError=None,df=None)
        result.update(id=identifier,family=family,endpoint=endpoint,role=role,eligibility=eligibility,window=window,model=model,tieFree=tie_free,
                      timing="endogenous_terminal_association" if endpoint=="slope_terminal" else "assigned_initial_projection" if endpoint.startswith("slope_initial") else "terminal_choice_or_acquisition",
                      confirmatoryRescueAllowed=False)
        tests.append(result);return result
    for identifier,family,endpoint,weights,side in CORE:
        run(identifier,family,endpoint,weights,side,"core")
    core=tests[:9]
    for cid in ("C2","C3"):
        run(f"distance_ES_{cid}_C1","support","distancePP_equal_split",{cid:1.,"C1":-1.},"greater")
        run(f"distance_EO_C1_{cid}","support","distancePP_equal_outcome",{"C1":1.,cid:-1.},"greater")
    run("lower_pooled","support","lowerNeedDirectional",{"C4":.5,"C5":.5,"C1":-1/3,"C2":-1/3,"C3":-1/3},"greater")
    for cid in CONDITIONS:
        side="less" if cid in ("C4","C5") else "greater"
        run(f"raw_initial_{cid}","support","slope_initial_raw",{cid:1.},side)
        run(f"terminal_{cid}","support","slope_terminal",{cid:1.},side)
    for identifier,weights in (("initial_C1",{"C1":1.}),("initial_C5",{"C5":1.}),
                                ("initial_C2_C1",{"C2":1.,"C1":-1.}),("initial_C3_C1",{"C3":1.,"C1":-1.}),
                                ("initial_C2C3_C1",{"C2":.5,"C3":.5,"C1":-1.}),
                                ("initial_C4C5_C2C3",{"C4":.5,"C5":.5,"C2":-.5,"C3":-.5})):
        run(identifier,"support","slope_initial",weights,"less" if identifier=="initial_C4C5_C2C3" else "greater",force="retained_design_required_C1_or_C5_initial_slope_unavailable")
    run("terminal_C4C5_C2C3","support","slope_terminal",{"C4":.5,"C5":.5,"C2":-.5,"C3":-.5},"less")
    run("C1_reverse_no_search_resemblance","support","EOResemblance",{"C1":1.},"less",force="coincident_no_search_targets_reverse_comparison_unavailable")
    for eligibility,window in (("available_history","all"),("complete_ten","early"),("complete_ten","late")):
        for identifier,family,endpoint,weights,side in CORE:
            run(f"{identifier}__{eligibility}_{window}","sensitivity",endpoint,weights,side,eligibility=eligibility,window=window)
    for cid in CONDITIONS:
        run(f"lower_{cid}_tie_free","sensitivity","lowerNeedDirectional",{cid:1.},"greater",tie_free=True)
        for model in ("order_only","order_recipe"):
            for endpoint in ("slope_initial","slope_terminal"):
                run(f"{endpoint}_{cid}_{model}","sensitivity",endpoint,{cid:1.},"less" if cid in ("C4","C5") else "greater",model=model)
    correction=multiplicity(core,policy,alpha)
    indexed={r["id"]:r for r in correction["tests"]}
    tests=[indexed.get(r["id"],{**r,"adjustedP":None,"reject":None}) for r in tests]
    return {"schema":"tutoring-inference-v1","origin":origin,"purpose":purpose,
            "humanEvidence":False if origin=="synthetic_design" else "declared_not_authenticated",
            "statisticalCalibrationPerformed":False,"primaryEligibility":"complete_ten_unique_main_1_to_10",
            "inputRows":len(rows),"practiceRowsExcluded":sum(r["stage"]=="practice" for r in rows),
            "participants":len({r["participantId"] for r in rows}),"tests":tests,"participantEstimates":participant_output,
            "multiplicity":{k:v for k,v in correction.items() if k!="tests"},
            "limitations":["Synthetic functionality is not type-I/coverage calibration or human evidence.",
                           "Mean within-schedule participant projection is not a mixed-model coefficient.",
                           "Complete and available-history populations differ; informative missingness is unresolved.",
                           "Terminal associations are endogenous; supporting tests cannot rescue core claims."]}


def synthetic_example_rows(people_per_condition=6) -> list[dict]:
    """Deterministic algebra fixtures; geometry is illustrative, NOT a new finite recipe."""
    if people_per_condition<2: raise ValueError("Example needs at least two people per condition")
    rows=[]
    for ci,cid in enumerate(CONDITIONS):
        for p in range(people_per_condition):
            for index in range(1,11):
                x=0. if cid in ("C1","C5") else (.15 if index%2 else -.15) if cid=="C3" else (index-5.5)*.04
                q=int((index+p)% (4 if cid in ("C1","C2","C4") else 2)==0)
                xt=x if cid!="C5" else ((index-5.5)*.03 if q else 0.)
                beta=(.2+.015*p) * (-1 if cid in ("C4","C5") else 1)
                share=.5+beta*(xt if cid=="C5" else x)+(p-2.5)*.002+((index*3+p)%7-3)*.0007
                target=.5+.18*xt
                dES=100*abs(share-.5);dEO=100*abs(share-target)
                lower=0. if xt==0 else (.5-share if xt>0 else share-.5)
                row={"participantId":f"synthetic-{cid}-{p:02d}","sessionId":f"synthetic-session-{cid}-{p:02d}",
                     "conditionId":cid,"trialId":f"main-{index:02d}","stage":"main","stageIndex":index,
                     "dataRole":"synthetic_design","analysisEligible":True,"exclusionReasons":"",
                     "terminalTie":xt==0.,"anyQuestion":q,"questions":q,"shareA":share,
                     "xInitial":x,"xTerminal":xt,"initialNeedGapRaw":120*x,
                     "scale":(.8,1.,1.2)[(index-1)%3],"gapOffset":0. if cid=="C3" else (-12.,0.,12.)[(index//3)%3],
                     "distancePP_equal_split":dES,"distancePP_equal_outcome":dEO,
                     "EOResemblance":dES-dEO,"lowerNeedDirectional":lower}
                rows.append(row)
            practice={**rows[-1],"trialId":"practice-01","stage":"practice","stageIndex":1,"analysisEligible":False}
            rows.append(practice)
    return rows
