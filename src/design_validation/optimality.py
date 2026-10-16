"""Finite all-horizon Bellman envelopes with explicitly numerical arithmetic.

The analytical frontier upper bound reveals both needs after the next paid
question. A finite refinement budget limits computation, never legal questioning.
Float64 results are numerical certificates, not validated arithmetic intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import time
from typing import Any, Hashable, Protocol

import numpy as np

from .bounded_task import NEEDS, normalize_log_weights, logsumexp
from .budget_horizon import BudgetTask, native_utility_table, native_perfect_information

ACTIONS = ("stop", "ask_A", "ask_B")


class NumericalEnvelopeError(ValueError):
    """A calculation cannot support even the declared numerical envelope claim."""


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.lower, self.upper)):
            raise NumericalEnvelopeError("Nonfinite interval")
        if self.lower > self.upper:
            raise NumericalEnvelopeError("Inverted interval")


@dataclass(frozen=True)
class Seed:
    """Terminal interval and all-continuation intervals for each legal ask."""
    stop: Interval
    asks: dict[str, Interval]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Branch:
    report: int
    probability: float
    child: Hashable


class EnvelopeModel(Protocol):
    """Small adapter also permitting independent finite rational toy models."""
    root: Hashable
    utility_abs_bound: float

    def seed(self, state: Hashable) -> Seed: ...
    def branches(self, state: Hashable) -> dict[str, list[Branch]]: ...
    def depth(self, state: Hashable) -> int: ...


@dataclass
class Node:
    state: Hashable
    seed: Seed
    actions: dict[str, Interval]
    lower: float
    upper: float
    children: dict[str, list[Branch]] | None = None

    def witness_action(self):
        # Unexpanded lower envelope is realized by stopping.
        legal = self.actions if self.children is not None else {"stop": self.actions["stop"]}
        return max(legal, key=lambda a: (legal[a].lower, -ACTIONS.index(a)))


def guarded(value, error, direction):
    """Outward final operation plus an explicitly declared numerical guard."""
    return math.nextafter(value+direction*error, -math.inf if direction < 0 else math.inf)


def current_rss_bytes():
    """Current RSS on Linux; None when the platform exposes no portable source."""
    try:
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])*1024
    except OSError:
        pass
    return None


class BellmanEnvelopes:
    def __init__(self, model: EnvelopeModel, *, numerical_guard=1e-10,
                 normalization_tolerance=1e-12, certificate_margin=1e-8):
        self.model = model
        self.guard = numerical_guard
        self.normalization_tolerance = normalization_tolerance
        self.margin = certificate_margin
        if min(numerical_guard, normalization_tolerance, certificate_margin) < 0:
            raise ValueError("Nonnegative tolerances required")
        self.nodes: dict[Hashable, Node] = {}
        self.expanded = 0
        self.branch_count = 0
        self.checkpoints = []
        self._materialize(model.root)

    def _materialize(self, state):
        if state not in self.nodes:
            seed = self.model.seed(state)
            actions = {"stop": seed.stop, **seed.asks}
            lower = seed.stop.lower  # witness must stop before expansion
            upper = max(x.upper for x in actions.values())
            self.nodes[state] = Node(state, seed, actions, lower, upper)
        return self.nodes[state]

    def expand(self, state, max_created_states):
        node = self.nodes[state]
        if node.children is not None or not node.seed.asks:
            return False
        children = self.model.branches(state)
        if set(children) != set(node.seed.asks):
            raise NumericalEnvelopeError("Missing or extra ask action")
        for action, branches in children.items():
            if not branches or len({b.report for b in branches}) != len(branches):
                raise NumericalEnvelopeError("Missing or duplicate report branches")
            if any(not math.isfinite(b.probability) or b.probability <= 0 for b in branches):
                raise NumericalEnvelopeError("Nonpositive/nonfinite report probability")
            if abs(math.fsum(b.probability for b in branches)-1) > self.normalization_tolerance:
                raise NumericalEnvelopeError("Predictive probability normalization failed")
            if any(self.model.depth(b.child) != self.model.depth(state)+1 for b in branches):
                raise NumericalEnvelopeError("Ask must consume exactly one count")
        new_states = {b.child for bs in children.values() for b in bs}-self.nodes.keys()
        # Coverage is atomic: never materialize only a prefix of an ask action.
        if len(self.nodes)+len(new_states) > max_created_states:
            return False
        for key in sorted(new_states, key=repr):
            self._materialize(key)
        node.children = children
        self.expanded += 1
        self.branch_count += sum(len(bs) for bs in children.values())
        self.backup()
        return True

    def backup(self):
        for node in sorted(self.nodes.values(), key=lambda n: (-self.model.depth(n.state), repr(n.state))):
            if node.children is None:
                continue
            new_actions = {"stop": node.seed.stop}
            for action, branches in node.children.items():
                mass_error = abs(math.fsum(b.probability for b in branches)-1)
                error = self.guard+mass_error*self.model.utility_abs_bound
                lower = guarded(math.fsum(b.probability*self.nodes[b.child].lower for b in branches), error, -1)
                upper = guarded(math.fsum(b.probability*self.nodes[b.child].upper for b in branches), error, 1)
                old = node.actions[action]
                new_actions[action] = Interval(max(old.lower, lower), min(old.upper, upper))
            lower = max(node.lower, max(v.lower for v in new_actions.values()))
            upper = min(node.upper, max(v.upper for v in new_actions.values()))
            if lower > upper:
                raise NumericalEnvelopeError("Bellman lower exceeds upper")
            node.actions, node.lower, node.upper = new_actions, lower, upper

    def frontier(self, max_depth):
        """Predictive uncertainty priority over all currently viable ask actions."""
        reach = {self.model.root: 1.}
        candidates = []
        for node in sorted(self.nodes.values(), key=lambda n: (self.model.depth(n.state), repr(n.state))):
            weight = reach.get(node.state, 0.)
            if not weight or node.upper-node.lower <= 2*self.guard:
                continue
            if node.children is None:
                if node.seed.asks and self.model.depth(node.state) < max_depth:
                    if max(v.upper for v in node.seed.asks.values()) > node.seed.stop.lower+self.margin:
                        candidates.append((weight*(node.upper-node.lower), node.state))
                continue
            for action, branches in node.children.items():
                if node.actions[action].upper <= node.lower+self.margin:
                    continue
                for branch in branches:
                    reach[branch.child] = reach.get(branch.child, 0.)+weight*branch.probability
        return sorted(candidates, key=lambda row: (-row[0], repr(row[1])))

    def summary(self):
        node = self.nodes[self.model.root]
        bounds = {a: {"lower": v.lower, "upper": v.upper} for a, v in node.actions.items()}
        dominated = [a for a, v in node.actions.items() if v.upper < node.lower-self.margin]
        winners = [a for a, v in node.actions.items()
                   if all(v.lower > other.upper+self.margin for b, other in node.actions.items() if b != a)]
        stop = node.actions["stop"]
        ask_lower = max((v.lower for a, v in node.actions.items() if a != "stop"), default=-math.inf)
        search_required = ask_lower > stop.upper+self.margin
        stop_certified = bool(node.seed.asks) and all(stop.lower > v.upper+self.margin for v in node.seed.asks.values())
        if not node.seed.asks:
            status = "numerical_stop_only_affordable_action"
        elif "stop" in winners or stop_certified:
            status = "numerical_all_horizon_stop_optimal"
        elif winners:
            status = "numerical_unique_"+winners[0]+"_optimal"
        elif search_required:
            status = "numerical_ask_class_optimal_recipient_unresolved"
        else:
            status = "unresolved_root_action"
        symmetric = bool(node.seed.details.get("analytical_initial_recipient_symmetry"))
        if symmetric and search_required:
            status = "numerical_optimal_ask_tie_with_analytical_symmetry"
        gap = node.upper-node.lower
        return {"lower": node.lower, "upper": node.upper, "gap": gap,
                "attained_epsilon_levels": [x for x in (.1, .01, .001) if gap <= x],
                "action_bounds": bounds, "dominated_actions": dominated,
                "unique_action_certificates": winners, "action_status": status,
                "search_required": search_required, "witness_action": node.witness_action(),
                "created_states": len(self.nodes), "expanded_states": self.expanded,
                "explicit_report_edges": self.branch_count,
                "max_materialized_depth": max(self.model.depth(k) for k in self.nodes),
                "unexpanded_frontier_states": sum(n.children is None and bool(n.seed.asks) for n in self.nodes.values()),
                "machine_certified_arithmetic": False, "whole_policy_optimality_claimed": False,
                "lower_policy": "lower_max_action_at_expanded_nodes_stop_at_unexpanded_nodes",
                "epsilon_scope": "constructed_feasible_policy_not_old_repeated_myopic_policy",
                "arithmetic": "float64_analytic_envelopes_with_declared_numerical_guards",
                "numerical_guard": self.guard, "certificate_margin": self.margin,
                "initial_details": node.seed.details}

    def run(self, *, max_created_states, max_depth, max_seconds, target_epsilon=.001,
            rss_limit_bytes=None, checkpoint=None, checkpoint_seconds=30., started_at=None):
        if max_created_states < len(self.nodes) or max_depth < 0 or max_seconds < 0 or target_epsilon < 0:
            raise ValueError("Invalid computational limits")
        start = time.perf_counter() if started_at is None else started_at
        last_checkpoint = start
        reason = "unresolved_frontier"
        while True:
            elapsed = time.perf_counter()-start
            root = self.nodes[self.model.root]
            if root.upper-root.lower <= target_epsilon:
                reason = "target_epsilon"; break
            if elapsed >= max_seconds:
                reason = "elapsed_time_limit"; break
            rss = current_rss_bytes()
            if rss_limit_bytes is not None and rss is not None and rss >= rss_limit_bytes:
                reason = "rss_limit"; break
            front = self.frontier(max_depth)
            if not front:
                reason = "depth_or_numerical_closure"; break
            # A state cap cannot discard children. Try smaller complete expansions.
            progressed = False
            for _, key in front:
                if self.expand(key, max_created_states):
                    progressed = True; break
            if not progressed:
                reason = "created_state_limit"; break
            now = time.perf_counter()
            if checkpoint is not None and now-last_checkpoint >= checkpoint_seconds:
                snap = {**self.summary(), "elapsed_seconds": now-start, "limit_reason": "running"}
                checkpoint(self, snap)
                last_checkpoint = time.perf_counter()
        result = {**self.summary(), "elapsed_seconds": time.perf_counter()-start,
                  "limit_reason": reason, "rss_bytes": current_rss_bytes(),
                  "limits": {"max_created_states": max_created_states, "max_depth": max_depth,
                             "max_seconds": max_seconds, "target_epsilon": target_epsilon,
                             "rss_limit_bytes": rss_limit_bytes}}
        if checkpoint is not None:
            checkpoint(self, result)
        result["elapsed_including_final_checkpoint_seconds"] = time.perf_counter()-start
        return result

    def witness_rows(self):
        """All dependency nodes/edges; each node records its feasible policy action."""
        for key, node in sorted(self.nodes.items(), key=lambda row: (self.model.depth(row[0]), repr(row[0]))):
            yield {"state": key, "depth": self.model.depth(key), "lower": node.lower, "upper": node.upper,
                   "stop": {"lower": node.seed.stop.lower, "upper": node.seed.stop.upper},
                   "actions": {a: {"lower": v.lower, "upper": v.upper} for a, v in node.actions.items()},
                   "witness_action": node.witness_action(), "expanded": node.children is not None,
                   "children": None if node.children is None else {
                       a: [{"report": b.report, "probability": b.probability, "child": b.child} for b in bs]
                       for a, bs in node.children.items()}, "details": node.seed.details}


class BudgetEnvelopeModel:
    """Installed native utility grid and exact mathematical report sufficient stats."""
    root = (0, 0, 0, 0)

    def __init__(self, config, *, numerical_guard=1e-10):
        if config.max_assessments != config.total_millihours//config.cost_millihours:
            raise ValueError("Budget-only affordable horizon required, no artificial cap")
        self.task = BudgetTask(config)
        self.config = config
        self.guard = numerical_guard
        raw = -.5*((NEEDS[:, None]-NEEDS[None, :])/config.report_width)**2
        self.log_normalizer = logsumexp(raw, axis=1)
        self.utility_abs_bound = 2*max(1., config.loss_multiplier)*max(100., config.total_millihours/1000)**config.alpha
        self.probability_underflow_evaluations = 0
        self.maximum_omitted_latent_mass_bound = 0.

    def depth(self, state):
        return state[0]+state[2]

    def budget(self, state):
        return self.config.total_millihours-self.depth(state)*self.config.cost_millihours

    @lru_cache(maxsize=4096)
    def recipient_logs(self, recipient, count, report_sum):
        if count < 0 or (count == 0 and report_sum != 0) or not count <= report_sum <= 100*count:
            raise ValueError("Invalid report sufficient statistic")
        if count == 0:
            return self.task.initial_logs[recipient]
        # Complete the square to reduce avoidable cancellation; report-square
        # constants cancel. The latent-dependent report normalizer remains.
        raw = (self.task.initial_logs[recipient]
               -.5*count*((NEEDS-report_sum/count)/self.config.report_width)**2
               -count*self.log_normalizer)
        result = normalize_log_weights(raw)
        if not np.isfinite(result).all():
            raise NumericalEnvelopeError("Nonfinite canonical posterior")
        result.setflags(write=False)
        return result

    def logs(self, state):
        return (self.recipient_logs(0, state[0], state[1]),
                self.recipient_logs(1, state[2], state[3]))

    def probabilities(self, logs):
        if not np.isfinite(logs).all():
            raise NumericalEnvelopeError("Nonfinite latent log weights")
        p = np.exp(logs)
        omitted = int(np.count_nonzero(p == 0))
        # Bound each underflowed probability represented by these finite logs.
        # This does not certify the preceding float log/normalizer arithmetic.
        mass = math.nextafter(omitted*math.ulp(0.), math.inf) if omitted else 0.
        self.probability_underflow_evaluations += omitted
        self.maximum_omitted_latent_mass_bound = max(self.maximum_omitted_latent_mass_bound, mass)
        error = mass+abs(float(p.sum())-1.)
        if abs(float(p.sum())-1.) > 1e-12:
            raise NumericalEnvelopeError("Latent probability normalization failed")
        return p, error, omitted, mass

    def seed(self, state):
        budget = self.budget(state)
        if budget < 0:
            raise ValueError("Unaffordable state")
        logs = self.logs(state)
        # Match terminal_values' explicit renormalization before expectation.
        pA, errorA, tailsA, massA = self.probabilities(normalize_log_weights(logs[0]))
        pB, errorB, tailsB, massB = self.probabilities(normalize_log_weights(logs[1]))
        c = self.config
        table = native_utility_table(budget, c.allocation_denominator, c.alpha, c.loss_multiplier)
        values = table@pA+table[::-1]@pB
        stop_max = float(values.max())
        error = self.guard+(errorA+errorB)*self.utility_abs_bound
        stop = Interval(guarded(stop_max, error, -1), guarded(stop_max, error, 1))
        installed_share, installed_value = self.task.terminal_solution(*logs, budget)
        installed_index = 0 if installed_share is None else round(installed_share*c.allocation_denominator)
        selected_vector_value = float(values[installed_index])
        details = {"budget_millihours": budget, "stop_grid_max": stop_max,
                   "installed_stop_share": installed_share, "installed_stop_value": selected_vector_value,
                   "installed_runtime_returned_value": installed_value,
                   "runtime_selected_value_parity_delta": installed_value-selected_vector_value,
                   "installed_selector_regret": stop_max-selected_vector_value,
                   "grid_argmax_index": int(np.argmax(values)),
                   "latent_underflow_counts": [tailsA, tailsB],
                   "omitted_latent_mass_upper_bounds": [massA, massB],
                   "latent_normalization_errors": [errorA, errorB],
                   "analytical_initial_recipient_symmetry": state == self.root and
                       c.prior_centres[0] == c.prior_centres[1] and c.prior_spreads[0] == c.prior_spreads[1]}
        asks = {}
        if budget >= c.cost_millihours:
            matrix = native_perfect_information(budget-c.cost_millihours, c.allocation_denominator, c.alpha, c.loss_multiplier)
            pi = float(pA@matrix@pB)
            pi_upper = guarded(pi, error, 1)
            floor = guarded(-self.utility_abs_bound, error, -1)
            asks = {a: Interval(floor, pi_upper) for a in ("ask_A", "ask_B")}
            details["perfect_information_after_next_cost"] = pi
            details["perfect_information_upper_gain_vs_stop"] = pi_upper-stop.lower
        return Seed(stop, asks, details)

    def branches(self, state):
        if self.budget(state) < self.config.cost_millihours:
            return {}
        result = {}
        for who, action in enumerate(("ask_A", "ask_B")):
            probability, posteriors = self.task.predictive(self.logs(state)[who])
            if not np.isfinite(posteriors).all() or np.any(probability <= 0):
                raise NumericalEnvelopeError("A mathematical report branch underflowed or is nonfinite")
            branches = []
            for report, p in enumerate(probability, 1):
                key = list(state)
                key[2*who] += 1
                key[2*who+1] += report
                branches.append(Branch(report, float(p), tuple(key)))
            if len(branches) != 100:
                raise NumericalEnvelopeError("All100 report branches required")
            result[action] = branches
        return result

    def posterior_parity(self, report_orders, tolerance=1e-10):
        rows = []
        for who, reports in report_orders:
            sequential = self.task.initial_logs[who]
            for report in reports:
                sequential = self.task.posterior(sequential, report)
            canonical = self.recipient_logs(who, len(reports), sum(reports))
            log_error = float(np.max(abs(sequential-canonical)))
            probability_error = float(np.max(abs(np.exp(sequential)-np.exp(canonical))))
            # Terminal parity directly exercises the same current integer-first grid.
            other = self.task.initial_logs[1-who]
            before = self.task.terminal_values(sequential, other, self.config.total_millihours) if who == 0 else self.task.terminal_values(other, sequential, self.config.total_millihours)
            after = self.task.terminal_values(canonical, other, self.config.total_millihours) if who == 0 else self.task.terminal_values(other, canonical, self.config.total_millihours)
            terminal_error = float(np.max(abs(before-after)))
            rows.append({"recipient": who, "reports": reports, "max_log_error": log_error,
                         "max_probability_error": probability_error, "max_terminal_error": terminal_error,
                         "pass": max(log_error, probability_error, terminal_error) <= tolerance})
        return {"cases": rows, "all_pass": all(r["pass"] for r in rows),
                "bit_identical_native_history_claimed": False, "tolerance": tolerance}
