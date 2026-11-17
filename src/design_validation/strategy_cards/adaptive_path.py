"""Run a single manifested path using certified comparison and CDF operands."""
from fractions import Fraction
import hashlib
import json

RNG = "sha256-first64-big-endian-midpoint-v1"


def key_uniform(namespace, stable_id, replicate, physical, cursor=None):
    if physical not in ("A", "B") or type(replicate) is not int or replicate < 0:
        raise ValueError("Exact replicate and physical recipient")
    if not ((namespace == "latent" and cursor is None)
            or (namespace == "report" and type(cursor) is int and cursor >= 0)):
        raise ValueError("Exact latent or report key")
    key = ["R13-one-more-v1", namespace, stable_id, replicate, physical, cursor]
    word = hashlib.sha256(json.dumps(key, separators=(",", ":"), ensure_ascii=True).encode()).digest()[:8]
    return key, Fraction(2 * int.from_bytes(word, "big") + 1, 2**65)


def inverse_cdf(u, cumulative, support=100):
    """Accept only a proved left-closed, right-open bin."""
    bounds = [tuple(map(Fraction, bound)) for bound in cumulative]
    if len(bounds) != support or not 0 <= u < 1 or bounds[-1] != (1, 1):
        raise ValueError("Exact support and unit terminal CDF")
    previous = (Fraction(0), Fraction(0))
    for lo, hi in bounds:
        if not previous[0] <= lo <= hi <= 1 or previous[1] > hi:
            raise ValueError("Invalid cumulative enclosure")
        previous = lo, hi
    previous_hi = Fraction(0)
    for index, (lo, hi) in enumerate(bounds, 1):
        if previous_hi <= u < lo:
            return index
        previous_hi = hi
    return None


def cdf_certificate(u, cumulative, selected):
    if selected is None:
        return {"uniform": str(u), "selected": None, "cdf": [list(map(str, b)) for b in cumulative]}
    previous = cumulative[selected-2] if selected > 1 else (0,0)
    return {"uniform": str(u), "selected": selected, "previous": list(map(str, previous)),
            "upperEdge": list(map(str, cumulative[selected-1]))}


def tie_uniform(stable_id, replicate, step):
    key = ["R13-one-more-v1", "recipient_tie", stable_id, replicate, step]
    word = hashlib.sha256(json.dumps(key, separators=(",", ":"), ensure_ascii=True).encode()).digest()[:8]
    return key, Fraction(2 * int.from_bytes(word, "big") + 1, 2**65)


def choose(stop, asks):
    stop_lo, stop_hi = map(Fraction, stop)
    ask = {slot: tuple(map(Fraction, interval)) for slot, interval in asks.items()}
    if stop_lo > stop_hi or any(lo > hi for lo, hi in ask.values()):
        raise ValueError("Invalid criterion interval")
    if all(hi <= stop_lo for lo, hi in ask.values()):
        return "stop", None
    for slot, (lo, _) in ask.items():
        if lo > stop_hi and all(lo > hi for other, (_, hi) in ask.items() if other != slot):
            return "ask", slot
    tied = [slot for slot, (lo, hi) in ask.items() if lo == hi and lo > stop_hi]
    if len(tied) == len(ask) == 2 and ask[tied[0]] == ask[tied[1]]:
        return "ask_tie", None
    return "unresolved", None


def resume_step(root, replicate, step, final, budget, history, latents, index):
    """Check committed control and draw metadata without replaying the oracle."""
    before = list(map(len, history))
    physical = root["orderedPhysical"]
    cost = root["task"]["cost_millihours"]

    def need(ok):
        if not ok:
            raise ValueError("Inconsistent adaptive decision prefix")

    def bin_matches(field, uniform, selected):
        proof = step[field]
        need(proof["uniform"] == str(uniform) and proof["selected"] == selected)
        if selected is None:
            need(inverse_cdf(uniform, proof["cdf"]) is None)
        else:
            prev = list(map(Fraction, proof["previous"]))
            edge = list(map(Fraction, proof["upperEdge"]))
            need(0 <= prev[0] <= prev[1] <= uniform < edge[0] <= edge[1] <= 1)

    need(step["budgetBefore"] == budget and step["cursorBefore"] == before)
    saved = step["proof"]
    if "orderedHistory" in saved:
        need(saved["orderedHistory"] == [list(row) for row in history])
    asks = {int(slot): interval for slot, interval in saved["asks"].items()}
    need(set(asks) == ({0, 1} if budget >= cost else set()))
    decision, slot = choose(saved["stop"], asks)
    need(step["decision"] == decision)
    if decision == "ask_tie":
        key, uniform = tie_uniform(root["id"], replicate, index)
        need(step["tieKey"] == key and step["tieUniform"] == str(uniform))
        slot = physical.index("A" if uniform < Fraction(1, 2) else "B")
    else:
        need("tieKey" not in step and "tieUniform" not in step)

    if slot is not None and slot not in latents:
        key, uniform = key_uniform("latent", root["id"], replicate, physical[slot])
        need(step["latentKey"] == key and step["latentUniform"] == str(uniform))
        latent = step.get("latentNeed")
        need(latent is None or type(latent) is int and 1 <= latent <= 100)
        bin_matches("latentCDF", uniform, latent)
        if latent is not None:
            latents[slot] = latent
    else:
        need(not any(k in step for k in ("latentKey", "latentUniform", "latentCDF", "latentNeed")))

    if slot is not None and slot in latents:
        key, uniform = key_uniform("report", root["id"], replicate, physical[slot], before[slot])
        need(step["orderedSlot"] == slot and step["physicalRecipient"] == physical[slot]
             and step["reportKey"] == key and step["reportUniform"] == str(uniform))
        report = step.get("report")
        selected = step["reportCDF"]["selected"]
        need(selected is None or type(selected) is int and 1 <= selected <= 100)
        bin_matches("reportCDF", uniform, selected)
        if final is None:
            need(type(report) is int and report == selected)
            history[slot].append(report)
            budget -= cost
        else:
            need(report is None and final == {"status": "unresolved_report_or_normalizer"})
    else:
        need(not any(k in step for k in ("orderedSlot", "physicalRecipient", "reportKey",
                                        "reportUniform", "reportCDF", "report")))
        need(final is not None)
        if slot is not None:
            need(final == {"status": "unresolved_latent_boundary"})
        elif decision == "unresolved":
            need(final == {"status": "unresolved_comparison"})
        else:
            target = saved.get("certifiedTarget") if budget else None
            if budget and target is None:
                need(final == {"status": "unresolved_terminal_index"})
            else:
                need(target is None or type(target) is int and 0 <= target <= root["task"]["allocation_denominator"])
                need(final == {"status": "terminal", "target": target,
                               "conditionalExpectedGPAInterval": saved["stop"]})
    need(step["budgetAfter"] == budget and step["cursorAfter"] == list(map(len, history)))
    return budget


def path(root, replicate, oracle, *, prefix=(), commit=None):
    budget = root["task"]["total_millihours"]
    cost = root["task"]["cost_millihours"]
    physical = root["orderedPhysical"]
    if cost <= 0 or budget < 0 or sorted(physical) != ["A", "B"]:
        raise ValueError("Exact resource and recipient mapping")
    history, steps = [[], []], []
    latents = {}
    for index, record in enumerate(prefix):
        step = record["step"]
        final = record["final"]
        budget = resume_step(root, replicate, step, final, budget, history, latents, index)
        steps.append(step)
        if final is not None:
            if index != len(prefix) - 1:
                raise ValueError("Decision after final adaptive frontier")
            return dict(final, steps=steps)

    def finish(step, **result):
        if commit is not None:
            commit(len(steps)-1, step, result)
        return dict(result, steps=steps)

    while True:
        proof = oracle.compare(root, budget, tuple(tuple(h) for h in history))
        legal = {0, 1} if budget >= cost else set()
        if set(proof["asks"]) != legal:
            raise ValueError("Complete affordable question roster")
        decision, slot = choose(proof["stop"], proof["asks"])
        step = dict(budgetBefore=budget, budgetAfter=budget,
                    cursorBefore=list(map(len, history)), cursorAfter=list(map(len, history)),
                    proof=proof, decision=decision)
        steps.append(step)
        if decision == "ask_tie":
            tie_key, tie_u = tie_uniform(root["id"], replicate, len(steps) - 1)
            slot = physical.index("A" if tie_u < Fraction(1, 2) else "B")
            step.update(tieKey=tie_key, tieUniform=str(tie_u))
        if decision == "stop":
            target = proof.get("certifiedTarget")
            if budget == 0:
                return finish(step, status="terminal", target=None,
                              conditionalExpectedGPAInterval=proof["stop"])
            if target is None:
                return finish(step, status="unresolved_terminal_index")
            if type(target) is not int or not 0 <= target <= root["task"]["allocation_denominator"]:
                raise ValueError("Invalid terminal index")
            return finish(step, status="terminal", target=target,
                          conditionalExpectedGPAInterval=proof["stop"])
        if decision == "unresolved":
            return finish(step, status="unresolved_comparison")
        key, u = key_uniform("report", root["id"], replicate, physical[slot], len(history[slot]))
        if slot not in latents:
            latent_key, latent_u = key_uniform("latent", root["id"], replicate, physical[slot])
            latent_cdf = oracle.latent_cdf(root, slot)
            n = inverse_cdf(latent_u, latent_cdf)
            step["latentCDF"] = cdf_certificate(latent_u, latent_cdf, n)
            step["latentKey"] = latent_key
            step["latentUniform"] = str(latent_u)
            if n is None:
                return finish(step, status="unresolved_latent_boundary")
            step["latentNeed"] = n
            latents[slot] = n
        step.update(reportKey=key, reportUniform=str(u), orderedSlot=slot, physicalRecipient=physical[slot])
        conditional_cdf = oracle.report_cdf(root, slot, latents[slot])
        report = inverse_cdf(u, conditional_cdf)
        step["reportCDF"] = cdf_certificate(u, conditional_cdf, report)
        if report is None or not oracle.update_is_certified(root, slot, history[slot], report):
            return finish(step, status="unresolved_report_or_normalizer")
        history[slot].append(report)
        budget -= cost
        step.update(report=report, budgetAfter=budget, cursorAfter=list(map(len, history)))
        if commit is not None:
            commit(len(steps)-1, step, None)
