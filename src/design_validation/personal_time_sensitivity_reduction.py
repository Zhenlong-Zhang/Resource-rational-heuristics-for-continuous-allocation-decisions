"""Whole-ten then equal-nine finite proxy reductions; no population precision."""
from __future__ import annotations

import math

from .personal_time_inputs import CONDITIONS, ORIGINAL, need
from .personal_time_sensitivity_analysis import nominal as E

SCALARS = ('Q', 'paidCount', 'expectedFinalUtility', 'expectedCombinedGPA', 'expectedNetObjective',
           'totalPersonalSeconds', 'searchSeconds', 'terminalSeconds')
CHOICE = ('D_ES_pp', 'R_surplus_pp', 'R_EGPA_pp', 'L_lower_mean_share', 'share')
LABELS = ('rho_0', 'rho_0.00001', 'rho_0.0001', 'rho_0.001', 'rho_geometric_0.001_0.01', 'rho_0.01')
MACROS = ('ES', 'EO_expected_GPA', 'EO_raw_surplus', 'PN', 'AL', 'GN', 'ML', 'MFN_A', 'MFN_B',
          'OPT_expected_GPA', 'random')


def person(rows):
    """Original producer arithmetic, requiring all ten mains in original order."""
    need(all(type(r['mainIndex']) is int for r in rows)
         and [r['mainIndex'] for r in rows] == list(range(1, 11)), 'Exact whole-ten denominator')
    for row in rows:
        need(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in SCALARS),
             'Finite scalar endpoints, including negative utility')
        need(type(row['choiceAvailable']) is bool and type(row['remainingIntegerBudget']) is int,
             'Typed choice/resource state')
        need(row['realizedCombinedGPA'] is None or type(row['realizedCombinedGPA']) in (int, float)
             and math.isfinite(row['realizedCombinedGPA']), 'Finite or unavailable realized outcome')
        if row['choiceAvailable']:
            need(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in (*CHOICE, 'x0')),
                 'Finite available choice endpoints')
    eligible = all(r['choiceAvailable'] and r['remainingIntegerBudget'] > 0 for r in rows)
    means = {k: float(E.canonical_mean([r[k] for r in rows])) for k in SCALARS}
    realized = [r['realizedCombinedGPA'] for r in rows if r['realizedCombinedGPA'] is not None]
    means['realizedCombinedGPA'] = float(E.canonical_mean(realized)) if len(realized) == 10 else None
    choice = {k: float(E.canonical_mean([r[k] for r in rows])) if eligible else None for k in CHOICE}
    slope = E._slope(rows, 'x0') if eligible else {
        'status': 'unavailable', 'value': None, 'reason': 'incomplete_ten_positive_defined_choices'}
    plan = E._projection_plan(rows, 'x0') if eligible else None
    if plan is not None and plan['status'] == 'available':
        plan = {**plan, 'weights': [float(x) for x in plan['weights']]}
    return {'assignedMains': 10, 'completeMains': 10, 'choiceEligibleWholeTen': eligible,
            'means': means, 'choiceMeans': choice, 'initialSlope': slope,
            'realizedAvailableMains': len(realized), 'realizedAssignedMains': 10,
            'availableRealizedMean': float(E.canonical_mean(realized)) if realized else None,
            'initialProjectionPlan': plan, 'monteCarloSE': None, 'precisionStatus': 'unreplicated_schedule_cell'}


def condition_rows(people, labels):
    result = []
    identities = [(p['rateLabel'], p['condition'], p['variant']) for p in people]
    need(len(identities) == len(set(identities)), 'Duplicate whole-person reduction')
    need(all(label in labels and c in CONDITIONS and type(v) is int and 0 <= v < 9
             for label, c, v in identities), 'Original person reduction identities')
    for label in labels:
        for condition in CONDITIONS:
            group = sorted((p for p in people if p['rateLabel'] == label and p['condition'] == condition),
                           key=lambda p: p['variant'])
            if not group:
                continue
            complete = len(group) == 9 and {p['variant'] for p in group} == set(range(9))
            eligible = [p for p in group if p['choiceEligibleWholeTen']]
            def reduce(values):
                available = [v for v in values if v is not None]
                return {'fullNineMean': float(E.canonical_mean(available)) if complete and len(available) == 9 else None,
                        'availableMean': float(E.canonical_mean(available)) if available else None,
                        'availableSchedules': len(available), 'assignedSchedules': 9}
            result.append({'condition': condition, 'rateLabel': label, 'assignedSchedules': 9,
                           'completeSchedules': len(group), 'choiceEligibleWholeTenSchedules': len(eligible),
                           'completeOriginalNineSchedules': complete,
                           'means': {k: reduce([p['means'][k] for p in group]) for k in group[0]['means']},
                           'choiceMeans': {k: reduce([p['choiceMeans'][k] for p in eligible]) for k in group[0]['choiceMeans']},
                           'initialSlope': reduce([p['initialSlope'].get('value') for p in eligible
                                                   if p['initialSlope'].get('status') == 'available']),
                           'monteCarloSE': None, 'finiteReplicateId': 0})
    return result


def aggregate(rows, labels=LABELS):
    """Reduce metric wrappers; partial persons never contribute a person mean."""
    labels = tuple(labels)
    need(bool(labels) and len(labels) == len(set(labels)) and set(labels) <= set(LABELS), 'Frozen unique rate labels')
    groups, actual = {}, set()
    for row in rows:
        coords = tuple(row['coordinates'])
        need(len(coords) == 3 and type(coords[0]) is str and type(coords[1]) is int
             and type(coords[2]) is int and coords in ORIGINAL, 'Original metric coordinate')
        label = row['rateLabel']
        need(label in labels and type(row['metrics']['mainIndex']) is int
             and row['metrics']['mainIndex'] == coords[2], 'Metric rate/main binding')
        key = (label, *coords)
        need(key not in actual, 'Duplicate metric coordinate')
        actual.add(key)
        groups.setdefault((label, *coords[:2]), []).append(row['metrics'])
    people, partial = [], []
    for label in labels:
        for condition, variant in sorted({(c, v) for c, v, _ in ORIGINAL}):
            group = sorted(groups.get((label, condition, variant), []), key=lambda r: r['mainIndex'])
            if not group:
                continue
            identity = {'condition': condition, 'variant': variant, 'rateLabel': label}
            if [r['mainIndex'] for r in group] != list(range(1, 11)):
                partial.append({**identity, 'completeMains': len(group), 'assignedMains': 10,
                                'missingMains': sorted(set(range(1, 11))-{r['mainIndex'] for r in group})})
            else:
                people.append({**identity, **person(group)})
    missing = {label: [list(c) for c in sorted(ORIGINAL) if (label, *c) not in actual] for label in labels}
    complete = all(not values for values in missing.values())
    return {'schema': 'personal-time-sensitivity-scalars-v1', 'persons': people,
            'conditions': condition_rows(people, labels), 'partialPersons': partial,
            'verifiedTrials': len(rows), 'completePerRate': complete,
            'fullFiniteTarget': complete and set(labels) == set(LABELS), 'missingCoordinates': missing,
            'assignedMainsPerRate': 450, 'assignedPersonsPerRate': 45, 'assignedSchedulesPerCondition': 9,
            'finiteReplicateId': 0, 'monteCarloPrecisionEstablished': False,
            'humanInferenceAdopted': False, 'modelAdopted': False}


def counts(records, labels=LABELS):
    """Acquisition and executed-terminal inventories, never counterfactual counts."""
    initial, final = [], []
    for label in labels:
        for c in CONDITIONS:
            group = [r for r in records if r['rateLabel'] == label and r['coordinates'][0] == c]
            if not group:
                continue
            roster = {(r['coordinates'][1], r['coordinates'][2]) for r in group}
            need(len(roster) == len(group), 'Exactly once count inventory')
            complete = roster == {(v, m) for v in range(9) for m in range(1, 11)}
            base = {'condition': c, 'rateLabel': label, 'assignedEndpoints': 90, 'completeOriginalCell': complete}
            a = {**base, 'roots': len(group), 'ask': 0, 'deterministic_terminal': 0,
                 'random_terminal': 0, 'nearRequestBoundaryRoots': 0, 'unresolved': 90-len(group)}
            f = {**base, 'endpoints': len(group), 'selectedMacroCounts': dict.fromkeys(MACROS, 0),
                 'ESExactChoiceCount': 0, 'choiceAvailableCount': 0, 'expectedOnlyCount': 0,
                 'genuineRealizedCount': 0, 'forcedRandomCount': 0, 'anyPaidCount': 0}
            for row in group:
                d, metric = row['initialDecision'], row['metrics']
                kind = 'ask' if d['action'] in ('ask_A', 'ask_B') else (
                    'random_terminal' if d['action'] == 'random' else 'deterministic_terminal')
                a[kind] += 1
                gaps = [q['objective']-d['stop']['objective'] for q in d['asks']]
                a['nearRequestBoundaryRoots'] += int(bool(gaps) and abs(max(gaps)) <= 1e-8)
                need(metric['action'] in MACROS, 'Original terminal macro')
                f['selectedMacroCounts'][metric['action']] += 1
                f['ESExactChoiceCount'] += int(metric['choiceAvailable'] and metric['shareIndex'] == 500)
                f['choiceAvailableCount'] += int(metric['choiceAvailable'])
                f['expectedOnlyCount'] += int(not metric['realizedAvailable'])
                f['genuineRealizedCount'] += int(metric['realizedAvailable'])
                f['forcedRandomCount'] += int(metric['forced'] and metric['action'] == 'random')
                f['anyPaidCount'] += metric['Q']
            initial.append(a)
            final.append(f)
    return {'initialCounts': initial, 'finalCounts': final}
