"""Independent internal score reconstruction and external path receiving."""
from fractions import Fraction as F
import math

import numpy as np


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def log_normalizer(values, axis=None):
    peak = np.max(values, axis=axis, keepdims=True)
    out = peak + np.log(np.sum(np.exp(values-peak), axis=axis, keepdims=True))
    return out.reshape(()) if axis is None else np.squeeze(out, axis=axis)


def internal(root, budget, history):
    t = root["task"]
    needs = np.arange(1, 101, dtype=float)
    raw = -.5*((needs[:, None]-needs[None, :])/t["report_width"])**2
    channel = raw-log_normalizer(raw, axis=1)[:, None]
    laws = []
    for initial, reports in zip(root["logsHex"], history):
        belief = np.array([float.fromhex(v) for v in initial], dtype=np.float64)
        for report in reports:
            pasted = belief + channel[:, report-1]
            belief = pasted-log_normalizer(pasted)
        laws.append(belief)
    def grid(resource):
        d = t["allocation_denominator"]
        s = (np.arange(d+1, dtype=float)*resource/(1000*d))[:, None]-needs
        return np.where(s >= 0, np.maximum(s,0)**t["alpha"],
                        -t["loss_multiplier"]*np.maximum(-s,0)**t["alpha"])

    a, b = laws
    wa, wb = [np.exp(v-log_normalizer(v)) for v in laws]
    now = grid(budget)
    objective = now @ wa + now[::-1] @ wb
    stop = float(objective[0] if budget == 0 else objective.max())
    ask = {}
    if budget >= t["cost_millihours"]:
        table = grid(budget-t["cost_millihours"])
        for slot, belief in enumerate((a,b)):
            joint = (belief-log_normalizer(belief))[:, None]+channel
            mass = log_normalizer(joint, axis=0)
            posterior = np.exp((joint-mass[None,:]).T)
            if slot == 0:
                scores = posterior @ table.T + (table[::-1] @ wb)[None,:]
            else:
                scores = (table @ wa)[None,:] + posterior @ table[::-1].T
            ask[slot] = float(np.exp(mass) @ scores.max(axis=1))
    need(all(math.isfinite(v) for v in [stop,*ask.values()]), "Finite independent scores")
    return F.from_float(stop), {k:F.from_float(v) for k,v in ask.items()}, (
        np.flatnonzero(objective==stop).tolist() if budget else [])


def verify(root, replicate, result, *, model, reference):
    """Fail closed on numerical, action, CDF or frontier disagreement."""
    need(model["internal"] == "current_runtime_frozen_source_binary64_utility_strict_float_comparison"
         and model["external"] == "normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160",
         "Exact policy and evaluator identities")
    physical = root["orderedPhysical"]
    need(sorted(physical)==["A","B"] and type(replicate) is int and 0<=replicate<32,
         "Owned physical recipient and replicate")
    ext = reference.Reference(root)
    budget = root["task"]["total_millihours"]
    history, latents = [[],[]], {}
    steps = result["steps"]
    need(bool(steps), "Nonempty path")
    status = None
    for i, step in enumerate(steps):
        before = list(map(len,history))
        proof = step["proof"]
        start = budget
        need(step["budgetBefore"]==start and step["cursorBefore"]==before
             and proof["orderedHistory"]==history
             and proof["law"]=="current_runtime_source_binary64_internal_utility", "Owned state")
        stop, ask, targets = internal(root, budget, history)
        need(tuple(map(F,proof["stop"]))==(stop,stop)
             and {int(k):tuple(map(F,v)) for k,v in proof["asks"].items()}
                 == {k:(v,v) for k,v in ask.items()}
             and proof["terminalIndexTieSet"]==targets
             and proof["certifiedTarget"]==(targets[0] if len(targets)==1 else None),
             "Independent binary64 operand and target equality")
        slot = None
        if not ask or max(ask.values()) <= stop:
            need(step["decision"]=="stop", "Strict stop")
            if budget and len(targets)!=1:
                status = "unresolved_terminal_index"
            else:
                status = "terminal"
                j = targets[0] if budget else 0
                need(result["target"]==(j if budget else None)
                     and result["internalBinary64Utility"]==proof["stop"], "Terminal decision")
                payoff = reference.terms(budget,2)
                a,b = [reference.curve([v],payoff) for v in ext.laws]
                lo,hi = (F(int(a[k][0,j])+int(b[k][0,1000-j]),reference.P*reference.U)
                         for k in (0,1))
                reference.contained(result["externalConditionalExpectedGPAInterval"],
                                    ext.gpa((lo,hi)), "External conditional terminal GPA")
        elif ask[0]==ask[1]:
            key = ["R13-one-more-v1","recipient_tie",root["id"],replicate,i]
            u = reference.key_value(key)
            need(step["decision"]=="ask_tie" and step["tieKey"]==key and F(step["tieUniform"])==u,
                 "Independent symmetric recipient tie")
            slot = physical.index("A" if u<F(1,2) else "B")
        else:
            slot = 0 if ask[0]>ask[1] else 1
            need(step["decision"]=="ask", "Strict best ask")
        if slot is None:
            need(not any(k in step for k in ("report", "reportKey", "latentKey", "physicalRecipient")),
                 "Stopping records carry no paid events")
        if slot is not None and slot not in latents:
            key = ["R13-one-more-v1","latent",root["id"],replicate,physical[slot],None]
            n = reference.read_bin(step,"latent",key,ext.laws[slot])
            if n is None:
                status = "unresolved_latent_boundary"
            else:
                need(step["latentNeed"]==n, "One retained physical latent")
                latents[slot] = n
        elif slot is not None:
            need("latentKey" not in step, "No repeated latent draw")
        if slot is not None and status is None:
            need(step["orderedSlot"]==slot and step["physicalRecipient"]==physical[slot], "Physical report recipient")
            key = ["R13-one-more-v1","report",root["id"],replicate,physical[slot],before[slot]]
            y = reference.read_bin(step,"report",key,ext.channel[latents[slot]-1])
            if y is None:
                status = "unresolved_report_or_normalizer"
            else:
                law, _ = reference.update(ext.laws[slot],ext.channel,y)
                need(step.get("report")==y, "Certified paid report")
                ext.laws[slot] = law
                history[slot].append(y)
                budget -= ext.cost
        need(step["budgetAfter"]==budget and step["cursorAfter"]==list(map(len,history)),
             "Actual fee and only selected cursor advance")
        if status is not None:
            need(i==len(steps)-1 and result["status"]==status, "Closing disposition")
            response = dict(status="PASS", disposition=status, paidReports=sum(map(len,history)))
            if status == "terminal":
                response["terminal"] = dict(budget=budget, cursors=list(map(len, history)),
                    target=result["target"], independentlyEnclosedGPA=list(map(str, ext.gpa((lo,hi)))))
            return response
        need(budget < start, "Strict budget descent after every paid action")
    raise ValueError("Missing closing frontier")
