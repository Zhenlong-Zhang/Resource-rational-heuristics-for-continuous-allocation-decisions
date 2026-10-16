"""Finite-belief metrics for synthetic scientific records; no export adapter."""
import numpy as np
from design_validation.bounded_task import BoundedTask, TaskConfig, belief_moments, heuristic_shares

TARGETS = ("equal_split", "equal_outcome", "all_to_lower", "meet_lower_first",
           "greatest_need", "proportional_need")


def metrics(t: dict) -> dict:
    """Metrics at the recorded finite initial/terminal beliefs, not kernel centres."""
    d, r = t["definition"], t["result"]
    initial, terminal = [np.asarray(t[k], dtype=float) for k in ("initialLogs", "terminalLogs")]
    means0 = [belief_moments(x)[0] for x in initial]
    meansT = [belief_moments(x)[0] for x in terminal]
    B0, BT = d["budgetMillihours"] / 1000, r["remainingMillihours"] / 1000
    share = None if BT == 0 else r["shareIndex"] / 1000
    tie0 = abs(means0[0] - means0[1]) <= 1e-9 * max(1, *means0)
    tieT = abs(meansT[0] - meansT[1]) <= 1e-9 * max(1, *meansT)
    # This historical metric tie rule is not calibration-truth recentering.
    gap0, gapT = (0. if tie0 else means0[0]-means0[1]), (0. if tieT else meansT[0]-meansT[1])
    targets = heuristic_shares(*terminal, r["remainingMillihours"])
    lower = None if share is None else (0. if tieT else share-.5 if meansT[0]<meansT[1] else .5-share)
    config = TaskConfig(t["conditionId"], d["templateId"], d["scale"], d["gap"],
                        tuple(d["priorCentres"]), tuple(d["priorSpreads"]), d["reportWidth"],
                        d["budgetMillihours"], d["costMillihours"], 1)
    task = BoundedTask(config)
    values = task.terminal_values(*terminal, r["remainingMillihours"])
    chosen_value = float(values[0 if share is None else r["shareIndex"]])
    _, optimum_value = task.terminal_solution(*terminal, r["remainingMillihours"])
    out = {"participantId": t["participantId"], "sessionId": t["sessionId"],
           "conditionId": t["conditionId"], "trialId": t["trialId"], "stage": d["stage"],
           "stageIndex": d["stageIndex"], "scheduleIndex": d["scheduleIndex"],
           "templateId": d["templateId"], "scale": d["scale"], "gapOffset": d["gap"],
           "sideReversed": d["sideReversed"], "initialBudget": B0, "remainingBudget": BT,
           "shareA": share, "meanAInitial": means0[0], "meanBInitial": means0[1],
           "meanATerminal": meansT[0], "meanBTerminal": meansT[1],
           "initialNeedGap": gap0, "terminalNeedGap": gapT,
           "initialNeedGapRaw": means0[0]-means0[1], "terminalNeedGapRaw": meansT[0]-meansT[1],
           "xInitial": None if B0 <= 0 else gap0 / B0,
           "xTerminal": None if BT <= 0 else gapT / BT,
           "initialTie": tie0, "terminalTie": tieT,
           "initialLowerNeed": "tie" if tie0 else "A" if means0[0] < means0[1] else "B",
           "terminalLowerNeed": "tie" if tieT else "A" if meansT[0] < meansT[1] else "B",
           "questions": len(t["paid"]), "anyQuestion": int(bool(t["paid"])),
           "questionsA": sum(x["student"] == 0 for x in t["paid"]),
           "questionsB": sum(x["student"] == 1 for x in t["paid"]),
           "questionSequence": "".join("A" if x["student"] == 0 else "B" for x in t["paid"]),
           "questionCostTotal": r["assessmentTimeMillihours"] / 1000,
           "capReached": len(t["paid"]) == 20, "stopReason": r["stopReason"],
           "lowerNeedDirectional": lower,
           "lowerNeedShare": None if lower is None else .5 + lower,
           "realizedUtility": r["totalUtility"], "combinedGPA": r["totalGPA"],
           "combinedGPAChange": r["totalGPA"] - 7,
           "terminalExpectedUtility": chosen_value,
           "costFreeTerminalRegret": max(0., optimum_value-chosen_value),
           "committedNotAdvanced": t["committedNotAdvanced"],
           "allocationEditCount": sum(e.get("editCount", 1) for e in t["events"] if e["type"] == "allocation_edit")}
    for name in TARGETS:
        out["target_" + name] = targets[name]
        out["distancePP_" + name] = None if share is None or targets[name] is None else 100 * abs(share-targets[name])
    out["EOResemblance"] = None if share is None else out["distancePP_equal_split"]-out["distancePP_equal_outcome"]
    return out
