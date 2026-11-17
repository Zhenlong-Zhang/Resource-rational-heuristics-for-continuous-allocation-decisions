"""Frozen-source binary64 decisions with a separately supplied external evaluator."""
from fractions import Fraction
import math

import numpy as np


def point(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Finite binary64 comparison operand required")
    exact = str(Fraction.from_float(value))
    return [exact, exact]


class Policy:
    def __init__(self, root, source, external):
        self.root = root
        self.source = source
        self.external = external
        cfg = source.TaskConfig(**root["task"])
        self.task = source.BoundedTask(cfg)
        self.root_logs = tuple(np.array(list(map(float.fromhex, logs)), dtype=np.float64)
                               for logs in root["logsHex"])

    def compare(self, root, budget, history):
        if root is not self.root:
            raise ValueError("Exact owned input")
        config = self.task.config
        price = config.cost_millihours
        if budget != config.total_millihours - price*sum(map(len, history)) or budget < 0:
            raise ValueError("Exact cumulative physical fee")
        logs = [x.copy() for x in self.root_logs]
        for slot, reports in enumerate(history):
            for report in reports:
                logs[slot] = self.task.posterior(logs[slot], report)
        values = self.task.terminal_values(*logs, budget)
        best = float(values[0] if budget == 0 else values.max())
        targets = np.flatnonzero(values == best)
        ask = {}
        if budget >= price:
            table = self.source._utility_table(budget-price, config.allocation_denominator,
                                               config.alpha, config.loss_multiplier)
            p0, p1 = [np.exp(self.source.normalize_log_weights(x)) for x in logs]
            m0, post0 = self.task.predictive(logs[0])
            m1, post1 = self.task.predictive(logs[1])
            for mass in (m0, m1):
                if not np.isfinite(mass).all() or min(mass) < 0 or abs(float(mass.sum())-1) > 1e-12:
                    raise ValueError("Complete finite predictive support")
            v0 = np.exp(post0) @ table.T + (table[::-1] @ p1)[None, :]
            v1 = (table @ p0)[None, :] + np.exp(post1) @ table[::-1].T
            ask = {0: point(m0 @ v0.max(axis=1)), 1: point(m1 @ v1.max(axis=1))}
        return dict(stop=point(best), asks=ask,
                    certifiedTarget=int(targets[0]) if budget and len(targets) == 1 else None,
                    orderedHistory=[list(row) for row in history],
                    terminalIndexTieSet=targets.tolist() if budget else [],
                    law="current_runtime_source_binary64_internal_utility")

    def latent_cdf(self, root, slot):
        return self.external.latent_cdf(root, slot)

    def report_cdf(self, root, slot, latent):
        return self.external.report_cdf(root, slot, latent)

    def update_is_certified(self, root, slot, history, report):
        return self.external.update_is_certified(root, slot, history, report)


def evaluate(root, replicate, source, external_module, path):
    """Label internal decision utility separately from external terminal GPA."""
    external = external_module.CertifiedOracle(root)
    result = path(root, replicate, Policy(root, source, external))
    if result["status"] == "terminal":
        result["internalBinary64Utility"] = result.pop("conditionalExpectedGPAInterval")
        final = result["steps"][-1]
        budget = final["budgetBefore"]
        index = result["target"] if budget else 0
        history = final["proof"]["orderedHistory"]
        table = external_module.utilities(budget, 2)
        scores = []
        for slot, j in enumerate((index, 1000-index)):
            laws = external_module.round_law([external.belief(slot, history[slot])])
            bounds = external_module.dots(laws, table)
            scores.append([int(b[0, j]) for b in bounds])
        result["externalConditionalExpectedGPAInterval"] = external.gpa([
            Fraction(sum(x[side] for x in scores), external_module.P*external_module.U)
            for side in (0, 1)])
    return result
