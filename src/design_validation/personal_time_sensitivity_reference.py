"""Independent original-law draw, paid-state, endpoint and scalar calculations.

No producer stream/transition/metric functions are imported. Independence is
limited by the explicitly shared finite Bayesian and terminal-target kernels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from .bounded_task import TaskConfig, normalize_log_weights
from .budget_horizon import native_utility_table
from .personal_time_reference import decision_value, seconds
from .terminal_time import expected_curves, terminal_summary

VERSION = 'r10-dynamic-native-path-candidate-v1'
STREAM = 'r10-private-need-report-v1'


def need(ok, label):
    if not ok:
        raise ValueError(label)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value) + b'\n').hexdigest()


def key_value(value):
    if value is None:
        return {'null': True}
    if type(value) is bool:
        return {'bool': value}
    if type(value) is int:
        return {'int': str(value)}
    if type(value) is float:
        need(not math.isnan(value) and value != math.inf, 'Invalid float key')
        return {'floatHex': value.hex()}
    if type(value) is str:
        return {'str': value}
    if type(value) is list:
        return {'list': [key_value(x) for x in value]}
    need(type(value) is dict and all(type(k) is str for k in value), 'Typed model key')
    return {'dict': [[k, key_value(value[k])] for k in sorted(value)]}


def model_binding(cfg):
    value = key_value(cfg)
    return {'encoding': 'typed-json-floathex-v1', 'candidate': value,
            'SHA256': hashlib.sha256(encoded(value)).hexdigest()}


def stable_identity(cfg, seed, population, condition, variant, replicate, main):
    common = {k: v for k, v in cfg.items() if k not in ('task', 'state', 'pathReports')}
    person = digest({'namespace': VERSION, 'seed': seed, 'populationId': population,
                     'modelHash': digest(key_value(common)), 'conditionId': condition,
                     'variant': variant, 'replicateId': replicate})
    return {'seed': seed, 'modelId': cfg['candidateId'], 'personId': person,
            'trialId': digest({'personId': person, 'mainIndex': main}), 'conditionId': condition,
            'variant': variant, 'replicateId': replicate, 'mainIndex': main}


def make_key(ctx, logical, kind, index):
    i, b = ctx['identity'], ctx['bindings']
    return {'streamVersion': STREAM, 'populationId': b['populationId'], 'seed': i['seed'],
            'randomizationGroup': b['randomizationGroup'], 'conditionId': i['conditionId'],
            'scheduleVariant': i['variant'], 'replicateId': i['replicateId'], 'mainIndex': i['mainIndex'],
            'logicalRecipient': logical, 'streamKind': kind, 'reportIndex': index}


def stream_draw(key, logs):
    values = np.asarray(logs, dtype=float)
    mass = np.exp(values)
    need(values.shape == (100,) and np.isfinite(mass).all() and np.all(mass >= 0)
         and abs(float(mass.sum())-1) <= 1e-12, 'Independent unaltered sampling law')
    h = hashlib.sha256(encoded(key)).hexdigest()
    seed = h[:32]
    u = float(np.random.Generator(np.random.PCG64(int(seed, 16))).random())
    total, chosen, previous, selected, last = 0., None, 0., 0., 0.
    for index, p in enumerate(mass):
        last = total
        total = float(np.add(np.float64(total), np.float64(p)))
        if chosen is None and total > u:
            chosen, previous, selected = index+1, last, total
    fallback = chosen is None
    if fallback:
        chosen, previous, selected = 100, last, total
    return {'schema': STREAM, 'key': key, 'keySHA256': h, 'seedHex': seed, 'uniformHex': u.hex(),
            'logHex': [float(x).hex() for x in values], 'probabilityHex': [float(x).hex() for x in mass],
            'rawMassHex': float(mass.sum()).hex(), 'category': chosen,
            'cdfPreviousHex': previous.hex(), 'cdfSelectedHex': selected.hex(), 'roundingFallback': fallback}


def unhex(rows):
    return [np.asarray([float.fromhex(x) for x in row]) for row in rows]


def decision(cfg, task, state):
    return decision_value(cfg, task, unhex(state['logHex']), state['budget'], lambda: None)


def paid_value(cfg, task, state, chosen, draw):
    slot = 0 if chosen['action'] == 'ask_A' else 1
    logs = unhex(state['logHex'])
    logs[slot] = normalize_log_weights(logs[slot] + task.log_likelihood[:, draw['category']-1])
    duration = seconds(cfg, slot, draw['category'])
    counts = list(state['counts'])
    counts[slot] += 1
    need(state['budget'] >= task.config.cost_millihours, 'Independent no tutoring overdraft')
    return {'before': copy.deepcopy(state), 'slot': slot, 'report': draw['category'],
            'incrementalSecondsHex': duration.hex(),
            'after': {'budget': state['budget']-task.config.cost_millihours, 'counts': counts,
                      'queryIndex': state['queryIndex']+1,
                      'logHex': [[float(x).hex() for x in row] for row in logs],
                      'searchSecondsHex': (float.fromhex(state['searchSecondsHex'])+duration).hex()}}


def expected_terminal(cfg, state, stop):
    need(state['queryIndex'] == 0 and stop['action'] != 'random', 'Expected-only initial deterministic stop')
    return {'action': stop['action'], 'forced': bool(stop.get('forced', False)), 'budget': state['budget'],
            'intendedIndex': stop['index'], 'actionExpectedCombinedGPA': stop['expectedCombinedGPA'],
            'expectedNetObjective': stop['expectedCombinedGPA']-cfg['valuation']['rhoGPAperSecond']*stop['seconds'],
            'searchSeconds': 0., 'terminalSeconds': stop['seconds'], 'totalPersonalSeconds': stop['seconds'],
            'realizedCombinedGPA': None, 'realizedNetObjective': None, 'newScientificDraws': 0,
            'targetHistoryFabricated': False, 'scope': 'exact_initial_stop_expected_endpoint_not_realized_history'}


def terminal_value(cfg, ctx, state, stop, needs):
    snapshot = copy.deepcopy(cfg)
    logs, budget = unhex(state['logHex']), state['budget']
    snapshot['state'] = {'budget': budget, 'logs': [[float(x) for x in row] for row in logs]}
    event = None
    if stop['action'] == 'random':
        key = {'version': 'r10-native-terminal-pcg64-sha256-v1', 'identity': ctx['identity'],
               'candidate': key_value(snapshot), 'keyEncoding': 'typed-json-floathex-v1'}
        h = hashlib.sha256(encoded(key)).hexdigest()
        u = float(np.random.Generator(np.random.PCG64(int(h, 16))).random())
        total, j = 0., len(cfg['random']['probabilities'])-1
        for index, p in enumerate(cfg['random']['probabilities']):
            total += p
            if total > u:
                j = index
                break
        event = {'schema': 'r10-native-terminal-pcg64-sha256-v1', 'key': key, 'keySHA256': h,
                 'uniform': u, 'index': j, 'cdfRule': 'ordered_binary64_strict_greater_final_index_rounding_fallback'}
    else:
        j = stop['index']
    c = TaskConfig(**cfg['task'])
    table = native_utility_table(budget, 1000, c.alpha, c.loss_multiplier)
    _, _, ga, gb = expected_curves(c, *logs, budget)
    realized = 2*c.gpa_baseline+c.gpa_scale*float(table[j, needs[0]-1]+table[1000-j, needs[1]-1])
    search, duration, rho = float.fromhex(state['searchSecondsHex']), stop['seconds'], cfg['valuation']['rhoGPAperSecond']
    return {'schema': 'r10-dynamic-terminal-v1', 'contextHash': digest(ctx), 'identity': ctx['identity'],
            'modelBinding': ctx['model'], 'terminalSnapshotBinding': model_binding(snapshot), 'stream': event,
            'action': stop['action'], 'forced': bool(stop.get('forced', False)), 'budget': budget,
            'intendedIndex': j, 'intendedShare': j/1000, 'deliveredShare': None if budget == 0 else j/1000,
            'deliveredAmountNumerators': [budget*j, budget*(1000-j)], 'amountDenominator': 1000000,
            'resourceUnit': 'tutoring_minutes', 'paidCount': state['queryIndex'], 'Q': int(state['queryIndex'] > 0),
            'choiceAvailable': budget > 0, 'searchSeconds': search, 'terminalSeconds': duration,
            'totalPersonalSeconds': search+duration, 'actionExpectedCombinedGPA': stop['expectedCombinedGPA'],
            'sampledIndexExpectedCombinedGPA': float(ga[j]+gb[j]), 'realizedCombinedGPA': realized,
            'expectedNetObjective': stop['expectedCombinedGPA']-rho*(search+duration),
            'realizedNetObjective': realized-rho*(search+duration)}


def metric_row(cfg, root, state, terminal, targets):
    def mean(logs):
        values = [float(x) for x in logs]
        peak = max(values)
        weights = [math.exp(x-peak) for x in values]
        return math.fsum((k+1)*w for k, w in enumerate(weights))/math.fsum(weights)
    initial = [mean(v) for v in root['initialLogs']]
    final = [mean([float.fromhex(v) for v in row]) for row in state['logHex']]
    budget, c = state['budget'], TaskConfig(**cfg['task'])
    if targets is None:
        summary = terminal_summary(c, *unhex(state['logHex']), budget)
        targets = {name: row['index'] for name, row in summary['macros'].items()}
    available = budget > 0
    j = terminal['intendedIndex'] if available else None
    row = {'mainIndex': root['mainIndex'], 'initialIntegerBudget': c.total_millihours,
           'remainingIntegerBudget': budget, 'initialMeans': initial, 'terminalMeans': final,
           'jEGPA': targets['EO_expected_GPA'], 'jSurplus': targets['EO_raw_surplus'],
           'jAL': targets['AL'], 'jML': targets['ML'], 'x0': (initial[0]-initial[1])/(c.total_millihours/1000),
           'xT': (final[0]-final[1])/(budget/1000) if available else None, 'rawInitialGap': initial[0]-initial[1],
           'Q': int(state['queryIndex'] != 0), 'paidCount': state['queryIndex'], 'counts': list(state['counts']),
           'expectedFinalUtility': (terminal['actionExpectedCombinedGPA']-2*c.gpa_baseline)/c.gpa_scale,
           'expectedCombinedGPA': terminal['actionExpectedCombinedGPA'], 'expectedNetObjective': terminal['expectedNetObjective'],
           'realizedCombinedGPA': terminal['realizedCombinedGPA'], 'totalPersonalSeconds': terminal['totalPersonalSeconds'],
           'searchSeconds': terminal['searchSeconds'], 'terminalSeconds': terminal['terminalSeconds'],
           'action': terminal['action'], 'forced': terminal['forced'],
           'realizedAvailable': terminal['realizedCombinedGPA'] is not None, 'shareIndex': j,
           'share': j/1000 if available else None, 'choiceAvailable': available,
           'choiceReason': None if available else 'zero_budget'}
    if available:
        tie = abs(final[0]-final[1]) <= 1e-9*max(1., abs(final[0]), abs(final[1]))
        row.update(D_ES_pp=abs(j-500)/10, D_EGPA_pp=abs(j-targets['EO_expected_GPA'])/10,
                   R_EGPA_pp=(abs(j-500)-abs(j-targets['EO_expected_GPA']))/10,
                   D_surplus_pp=abs(j-targets['EO_raw_surplus'])/10,
                   R_surplus_pp=(abs(j-500)-abs(j-targets['EO_raw_surplus']))/10,
                   L_lower_mean_share=0. if tie else (j-500)/1000*(1 if final[0] < final[1] else -1),
                   lowerMeanTie=tie, allToLowerMatch=j == targets['AL'], meetLowerFirstMatch=j == targets['ML'])
    else:
        row.update({k: None for k in ('D_ES_pp', 'D_EGPA_pp', 'R_EGPA_pp', 'D_surplus_pp', 'R_surplus_pp',
                                      'L_lower_mean_share', 'lowerMeanTie', 'allToLowerMatch', 'meetLowerFirstMatch')})
    if terminal['realizedCombinedGPA'] is None:
        need(terminal['scope'] == 'exact_initial_stop_expected_endpoint_not_realized_history'
             and terminal['newScientificDraws'] == 0 and state['queryIndex'] == 0, 'Explicit expected mapping required')
    return row


def independent_person(rows):
    need([r['mainIndex'] for r in rows] == list(range(1, 11)), 'Independent whole-ten roster')
    scalar_keys = ('Q', 'paidCount', 'expectedFinalUtility', 'expectedCombinedGPA', 'expectedNetObjective',
                   'totalPersonalSeconds', 'searchSeconds', 'terminalSeconds')
    means = {key: math.fsum(r[key] for r in rows)/10 for key in scalar_keys}
    realized = [r['realizedCombinedGPA'] for r in rows if r['realizedCombinedGPA'] is not None]
    means['realizedCombinedGPA'] = math.fsum(realized)/10 if len(realized) == 10 else None
    eligible = all(r['choiceAvailable'] and r['remainingIntegerBudget'] > 0 for r in rows)
    choice = {key: math.fsum(r[key] for r in rows)/10 if eligible else None
              for key in ('D_ES_pp', 'R_surplus_pp', 'R_EGPA_pp', 'L_lower_mean_share', 'share')}
    slope = None
    if eligible:
        x, y = [r['x0'] for r in rows], [r['share'] for r in rows]
        xm, ym = math.fsum(x)/10, math.fsum(y)/10
        sxx = math.fsum((v-xm)**2 for v in x)
        if len(set(x)) >= 2 and sxx > 1e-12:
            slope = math.fsum((a-xm)*(b-ym) for a, b in zip(x, y))/sxx
    return {'means': means, 'choiceMeans': choice, 'initialSlopeValue': slope, 'choiceEligibleWholeTen': eligible,
            'realizedAvailableMains': len(realized),
            'availableRealizedMean': math.fsum(realized)/len(realized) if realized else None}


def independent_conditions(people, labels):
    result = []
    for label in labels:
        for condition in ('C1', 'C2', 'C3', 'C4', 'C5'):
            group = [p for p in people if p['rateLabel'] == label and p['condition'] == condition]
            if not group:
                continue
            complete = len(group) == 9 and {p['variant'] for p in group} == set(range(9))
            eligible = [p for p in group if p['choiceEligibleWholeTen']]
            def reduce(values):
                values = [v for v in values if v is not None]
                return {'fullNineMean': math.fsum(values)/9 if complete and len(values) == 9 else None,
                        'availableMean': math.fsum(values)/len(values) if values else None,
                        'availableSchedules': len(values), 'assignedSchedules': 9}
            result.append({'condition': condition, 'rateLabel': label, 'assignedSchedules': 9,
                           'completeSchedules': len(group), 'choiceEligibleWholeTenSchedules': len(eligible),
                           'completeOriginalNineSchedules': complete,
                           'means': {k: reduce([p['means'][k] for p in group]) for k in group[0]['means']},
                           'choiceMeans': {k: reduce([p['choiceMeans'][k] for p in eligible]) for k in group[0]['choiceMeans']},
                           'initialSlope': reduce([p['initialSlopeValue'] for p in eligible]),
                           'monteCarloSE': None, 'finiteReplicateId': 0})
    return result
