"""Pure fixed-roster allocation reductions, not policy or source authentication.

True-need fits exclude exact category ties. The legacy initial-mean slope and
its paired C3 decomposition remain separate estimands. Error bars are sample
SD across nine fixed schedules, not confidence intervals or Monte Carlo error.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import math

import numpy as np

SCHEMA = 'true-need-allocation-diagnostics-v1'
CONDITIONS = ('C1', 'C2', 'C3', 'C4', 'C5')
LABELS = ('rho_0', 'rho_0.00001', 'rho_0.0001', 'rho_0.001',
          'rho_geometric_0.001_0.01', 'rho_0.01')
BASE_PRICE, MIDPOINT_PRICE, HIGH_PRICE = LABELS[3:]
MACROS = ('ES', 'EO_expected_GPA', 'EO_raw_surplus', 'PN', 'AL', 'GN', 'ML',
          'MFN_A', 'MFN_B', 'OPT_expected_GPA', 'random')
MAINS = tuple(range(1, 11))
SCHEDULES = tuple(range(9))
INITIAL_SS_FLOOR = 1e-12
FIELDS = ('condition', 'price', 'variant', 'main', 'share', 'index',
          'initial_budget_milli', 'needs_displayed', 'logical_order',
          'rank_x', 'rank_y', 'initial_ask', 'mean_paid',
          'terminal_action', 'selected_rule')
METRICS = ('gap_slope', 'lower_need_partial', 'initial_mean_slope')
C3_METRICS = ('base_slope', 'midpoint_slope', 'slope_change',
              'search_contribution', 'stop_contribution')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _orientations(row):
    if row['rank_y'] is None:
        return set()
    return {sign for sign, y in ((1, row['share']), (-1, 1 - row['share']))
            if math.isclose(row['rank_y'], y, rel_tol=0, abs_tol=1e-12)}


def _validate(row):
    _need(isinstance(row, Mapping) and all(k in row for k in FIELDS),
          'Every endpoint must supply all diagnostic fields; use explicit None for unknowns')
    r = {k: row[k] for k in FIELDS}
    _need(type(r['condition']) is str and r['condition'] in CONDITIONS
          and type(r['price']) is str and r['price'] in LABELS
          and type(r['variant']) is int and r['variant'] in SCHEDULES
          and type(r['main']) is int and r['main'] in MAINS,
          'Endpoint outside the fixed condition/price/schedule/main roster')
    _need(type(r['index']) is int and 0 <= r['index'] <= 1000
          and _finite(r['share']) and 0 <= r['share'] <= 1
          and math.isclose(r['share'], r['index'] / 1000, rel_tol=0, abs_tol=1e-12),
          'Share must match the saved integer allocation index')
    _need(type(r['initial_budget_milli']) is int
          and _finite(r['initial_budget_milli']) and r['initial_budget_milli'] > 0,
          'Positive finite integer initial budget required')
    order = r['logical_order']
    _need(type(order) in (list, tuple) and len(order) == 2
          and all(type(x) is str for x in order) and set(order) == {'A', 'B'},
          'Logical order must map displayed slots to logical A/B exactly once')
    r['logical_order'] = tuple(order)
    needs = r['needs_displayed']
    _need(needs is None or type(needs) in (list, tuple) and len(needs) == 2
          and all(n is None or type(n) is int and 1 <= n <= 100 for n in needs),
          'Needs must be two integer categories 1-100, or explicitly unavailable')
    r['needs_displayed'] = (None, None) if needs is None else tuple(needs)
    x, y = r['rank_x'], r['rank_y']
    _need((x is None and y is None) or
          (_finite(x) and x >= 0 and _finite(y) and 0 <= y <= 1),
          'Legacy rank predictor/outcome must be finite and paired, or both None')
    _need(y is None or bool(_orientations(r)), 'Legacy rank outcome must orient the saved share')
    _need(type(r['initial_ask']) is int and r['initial_ask'] in (0, 1)
          and type(r['mean_paid']) is int and r['mean_paid'] >= 0
          and r['initial_ask'] == int(r['mean_paid'] > 0),
          'Initial ask and saved nonnegative paid count must agree')
    for field in ('terminal_action', 'selected_rule'):
        value = r[field]
        _need(value is None or type(value) is str and bool(value.strip()) and value == value.strip(),
              field + ' must be a nonempty saved label or explicit None')
    return r


def _mean(values):
    return math.fsum(values) / len(values)


def _slope(points):
    if len(points) < 2:
        return None
    mx, my = (_mean(list(values)) for values in zip(*points))
    ss = math.fsum((x - mx)**2 for x, _ in points)
    return math.fsum((x - mx) * (y - my) for x, y in points) / ss if ss > INITIAL_SS_FLOOR else None


def _ols(design, y, columns):
    if not y:
        return 0, None
    beta, _, rank, _ = np.linalg.lstsq(np.asarray(design, dtype=float),
                                      np.asarray(y, dtype=float), rcond=None)
    _need(bool(np.all(np.isfinite(beta))), 'Nonfinite least-squares result')
    return int(rank), [float(b) for b in beta] if rank == columns else None


def _metric(values):
    available = [v for v in values if v is not None]
    _need(all(_finite(v) for v in available), 'Nonfinite schedule coefficient')
    complete = len(values) == 9 and len(available) == 9
    mean = _mean(available) if complete else None
    sd = math.sqrt(math.fsum((v - mean)**2 for v in available) / 8) if complete else None
    return {'mean': mean, 'sample_sd': sd, 'defined_schedules': len(available),
            'expected_schedules': 9}


def _schedule(group):
    group = sorted(group, key=lambda r: r['main'])
    first = group[0]
    mains = [r['main'] for r in group]
    complete = mains == list(MAINS)
    gap_design, partial_design, outcomes, initial = [], [], [], []
    missing = ties = 0
    for row in group:
        if row['rank_x'] is not None:
            initial.append((row['rank_x'], row['rank_y']))
        a, b = row['needs_displayed']
        if a is None or b is None:
            missing += 1
        elif a == b:
            ties += 1
        else:
            low = 1000 * min(a, b) / row['initial_budget_milli']
            high = 1000 * max(a, b) / row['initial_budget_milli']
            gap_design.append([1., high - low])
            partial_design.append([1., low, high])
            outcomes.append(row['share'] if a > b else 1 - row['share'])
    gap_rank, gap_beta = _ols(gap_design, outcomes, 2)
    partial_rank, partial_beta = _ols(partial_design, outcomes, 3)
    return {'condition': first['condition'], 'price': first['price'], 'schedule': first['variant'],
            'available_rows': len(group), 'missing': missing, 'ties': ties, 'eligible': len(outcomes),
            'present_mains': mains, 'missing_mains': sorted(set(MAINS) - set(mains)),
            'complete': complete, 'gap_rank': gap_rank, 'partial_rank': partial_rank,
            'gap_slope': gap_beta[1] if complete and gap_beta is not None else None,
            'lower_need_partial': partial_beta[1] if complete and partial_beta is not None else None,
            'initial_mean_eligible': len(initial),
            'initial_mean_slope': _slope(initial) if complete else None}


def _conditions(schedules):
    groups = defaultdict(list)
    for row in schedules:
        groups[row['condition'], row['price']].append(row)
    result = []
    for (condition, price), group in sorted(groups.items()):
        complete = [s for s in group if s['complete']]
        result.append({'condition': condition, 'price': price,
                       'available_rows': sum(s['available_rows'] for s in group),
                       'missing': sum(s['missing'] for s in group),
                       'ties': sum(s['ties'] for s in group),
                       'eligible': sum(s['eligible'] for s in group),
                       'present_schedules': [s['schedule'] for s in group],
                       'missing_schedules': sorted(set(SCHEDULES) - {s['schedule'] for s in group}),
                       'complete_schedules': len(complete), 'complete': len(complete) == 9,
                       **{k: _metric([s[k] for s in group]) for k in METRICS}})
    return result


def _paired_c3(lookup):
    schedules, partial = [], []
    for variant in SCHEDULES:
        base = {m: lookup[('C3', BASE_PRICE, variant, m)] for m in MAINS
                if ('C3', BASE_PRICE, variant, m) in lookup}
        midpoint = {m: lookup[('C3', MIDPOINT_PRICE, variant, m)] for m in MAINS
                    if ('C3', MIDPOINT_PRICE, variant, m) in lookup}
        paired = sorted(base.keys() & midpoint.keys())
        unavailable, points = [], []
        for m in paired:
            a, b = base[m], midpoint[m]
            _need(a['rank_x'] == b['rank_x'], 'C3 paired initial predictor or availability changed')
            _need(a['initial_budget_milli'] == b['initial_budget_milli']
                  and a['logical_order'] == b['logical_order'], 'C3 paired initial identity changed')
            if a['rank_x'] is None:
                unavailable.append(m)
                continue
            _need(bool(_orientations(a) & _orientations(b)), 'C3 paired initial-mean orientation changed')
            points.append((a['rank_x'], a['rank_y'], b['rank_y'], b['mean_paid'] > 0))
        missing_base, missing_mid = sorted(set(MAINS) - base.keys()), sorted(set(MAINS) - midpoint.keys())
        detail = {'schedule': variant, 'missing_base_mains': missing_base,
                  'missing_midpoint_mains': missing_mid, 'unavailable_initial_mean_mains': unavailable}
        if missing_base or missing_mid or unavailable:
            partial.append(detail)
        if not base and not midpoint:
            continue
        row = {'schedule': variant, 'paired_targets': len(paired), 'eligible_pairs': len(points),
               'search_targets': sum(midpoint[m]['mean_paid'] > 0 for m in paired),
               'stop_targets': sum(midpoint[m]['mean_paid'] == 0 for m in paired),
               'complete': len(paired) == 10, 'status': 'incomplete_pairs',
               'centered_predictor_ss': None, 'additivity_residual': None,
               'additivity_verified': None, **{k: None for k in C3_METRICS}, **detail}
        if len(paired) == 10:
            row['status'] = 'unavailable_initial_mean'
        if len(points) == 10:
            mx = _mean([p[0] for p in points])
            ss = math.fsum((p[0] - mx)**2 for p in points)
            row['centered_predictor_ss'] = ss
            row['status'] = 'rank_deficient' if ss <= INITIAL_SS_FLOOR else 'available'
            if ss > INITIAL_SS_FLOOR:
                a_beta = _slope([(x, a) for x, a, _, _ in points])
                b_beta = _slope([(x, b) for x, _, b, _ in points])
                stop, search = [math.fsum((x - mx) * (b - a)
                                         for x, a, b, ask in points if ask == subset) / ss
                                for subset in (False, True)]
                delta = b_beta - a_beta
                residual = math.fsum((stop, search, -delta))
                _need(math.isclose(stop + search, delta, rel_tol=1e-12, abs_tol=1e-10),
                      'C3 search and stop contributions do not sum to the paired slope change')
                row.update(base_slope=a_beta, midpoint_slope=b_beta, slope_change=delta,
                           search_contribution=search, stop_contribution=stop,
                           additivity_residual=residual, additivity_verified=True)
        schedules.append(row)
    summary = {k: _metric([s[k] for s in schedules]) for k in C3_METRICS}
    summary.update(paired_targets=sum(s['paired_targets'] for s in schedules),
                   eligible_pairs=sum(s['eligible_pairs'] for s in schedules),
                   search_targets=sum(s['search_targets'] for s in schedules),
                   stop_targets=sum(s['stop_targets'] for s in schedules),
                   complete_schedules=sum(s['complete'] for s in schedules),
                   available_schedules=sum(s['status'] == 'available' for s in schedules),
                   expected_schedules=9)
    return {'condition': 'C3', 'base_price': BASE_PRICE, 'midpoint_price': MIDPOINT_PRICE,
            'predictor': 'legacy_absolute_initial_mean_gap_over_initial_budget',
            'search_group': 'midpoint_saved_paid_count_positive', 'causal': False,
            'schedules': schedules, 'summary': summary, 'partial': partial}


def _c5_counts(lookup):
    group = [lookup[('C5', HIGH_PRICE, v, m)] for v in SCHEDULES for m in MAINS
             if ('C5', HIGH_PRICE, v, m) in lookup]
    actions, rules, indices = Counter(), Counter(), Counter()
    displayed, logical = Counter(), Counter()
    action_displayed, action_logical, mfn_logical = Counter(), Counter(), Counter()
    endpoints = []
    for r in group:
        action, rule, index = r['terminal_action'], r['selected_rule'], r['index']
        if action is not None:
            actions[action] += 1
        if rule is not None:
            rules[rule] += 1
        indices[str(index)] += 1
        slot = 0 if index == 1000 else 1 if index == 0 else None
        d = ('first', 'second')[slot] if slot is not None else 'split'
        logical_recipient = r['logical_order'][slot] if slot is not None else 'split'
        displayed[d] += 1
        logical[logical_recipient] += 1
        action_slot = 0 if action == 'MFN_A' else 1 if action == 'MFN_B' else None
        action_d = ('first', 'second')[action_slot] if action_slot is not None else (
            'not_fixed_recipient' if action in MACROS else 'unknown')
        action_l = r['logical_order'][action_slot] if action_slot is not None else action_d
        action_displayed[action_d] += 1
        action_logical[action_l] += 1
        if action == 'MFN_A' and index == 1000:
            mfn_logical[r['logical_order'][0]] += 1
        endpoints.append({'schedule': r['variant'], 'main': r['main'],
                          'terminal_action': action, 'selected_rule': rule, 'index': index,
                          'allocation_displayed_recipient': d, 'allocation_logical_recipient': logical_recipient,
                          'saved_action_displayed_recipient': action_d,
                          'saved_action_logical_recipient': action_l})
    counts = {name: dict(sorted(counter.items())) for name, counter in (
        ('terminal_action_counts', actions), ('selected_rule_counts', rules),
        ('allocation_index_counts', indices), ('allocation_displayed_recipient_counts', displayed),
        ('allocation_logical_recipient_counts', logical),
        ('saved_action_displayed_recipient_counts', action_displayed),
        ('saved_action_logical_recipient_counts', action_logical),
        ('mfn_a_index_1000_logical_counts', mfn_logical))}
    counts.update(endpoints=len(group), expected_endpoints=90,
                  mfn_a_index_1000=sum(mfn_logical.values()),
                  missing_terminal_actions=sum(r['terminal_action'] is None for r in group),
                  missing_selected_rules=sum(r['selected_rule'] is None for r in group),
                  unknown_terminal_actions=sum(r['terminal_action'] not in MACROS for r in group),
                  unknown_selected_rules=sum(r['selected_rule'] not in MACROS for r in group))
    missing = [[v, m] for v in SCHEDULES for m in MAINS if ('C5', HIGH_PRICE, v, m) not in lookup]
    return {'condition': 'C5', 'price': HIGH_PRICE, 'complete': not missing,
            'counts': counts, 'endpoints': endpoints, 'partial': missing,
            'action_identity_inferred_from_allocation': False}


def summarize(rows, *, full=False):
    """Reduce admitted rows without mutating them or reading their source files.

    Required fields are FIELDS. Explicit None denotes unavailable needs or saved
    labels; legacy rank_x/rank_y must be jointly available. Unknown action names
    are preserved and counted, never relabeled as a known rule. Full mode checks
    the exact endpoint roster, not historical provenance or scientific results.
    Partial mode reports missing coordinates and never substitutes partial means.
    """
    _need(type(full) is bool, 'full must be a boolean')
    _need(not isinstance(rows, (Mapping, str, bytes)), 'rows must be an iterable of endpoint mappings')
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise ValueError('rows must be an iterable of endpoint mappings') from exc
    lookup, groups = {}, defaultdict(list)
    for raw in iterator:
        r = _validate(raw)
        key = (r['condition'], r['price'], r['variant'], r['main'])
        _need(key not in lookup, 'Duplicate endpoint coordinate')
        lookup[key] = r
        groups[key[:3]].append(r)
    expected = {(c, p, v, m) for c in CONDITIONS for p in LABELS for v in SCHEDULES for m in MAINS}
    complete = lookup.keys() == expected
    _need(not full or complete, 'Full mode requires the exact fixed 2700-endpoint roster')
    try:
        schedules = [_schedule(group) for _, group in sorted(groups.items())]
        conditions, c3, c5 = _conditions(schedules), _paired_c3(lookup), _c5_counts(lookup)
    except (OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError('Numerical reduction failed for the supplied finite inputs') from exc
    return {'schema': SCHEMA, 'fullFiniteTarget': complete, 'verifiedTrials': len(lookup),
            'expectedEndpoints': 2700, 'expectedSchedules': 270,
            'schedules': schedules, 'conditions': conditions, 'c3': c3, 'c5': c5,
            'missingCoordinates': {p: [[c, v, m] for c in CONDITIONS for v in SCHEDULES for m in MAINS
                                       if (c, p, v, m) not in lookup] for p in LABELS},
            'uncertainty': 'sample_sd_across_nine_fixed_schedules_ddof1',
            'rowSourceAuthenticated': False, 'policyRerunByAnalysis': False,
            'historicalEventClaimed': False, 'monteCarloPrecisionEstablished': False,
            'humanInferenceAdopted': False}
