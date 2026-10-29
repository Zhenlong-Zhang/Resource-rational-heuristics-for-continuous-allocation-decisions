"""Original proxy stream, paid-state and endpoint producer kernels.

Scientific namespace strings and newline-sensitive hashes are frozen identities.
Policy state starts with raw JS logs; only latent sampling laws are normalized.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from . import personal_time_policy as A
from .budget_horizon import native_utility_table
from .personal_time_sensitivity_analysis import geometry as G

VERSION = 'r10-dynamic-native-path-candidate-v1'
STREAM = 'r10-private-need-report-v1'
TERMINAL_STREAM = 'r10-native-terminal-pcg64-sha256-v1'
IDENTITY = {'seed', 'modelId', 'personId', 'trialId', 'conditionId', 'variant', 'replicateId', 'mainIndex'}


def payload(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def canonical(value):
    if value is None:
        return {'null': True}
    if type(value) is bool:
        return {'bool': value}
    if type(value) is int:
        return {'int': str(value)}
    if type(value) is float:
        A.need(not math.isnan(value) and value != math.inf, 'No NaN/positive infinity in key')
        return {'floatHex': value.hex()}
    if type(value) is str:
        return {'str': value}
    if type(value) is list:
        return {'list': [canonical(v) for v in value]}
    A.need(type(value) is dict and all(type(k) is str for k in value), 'Typed JSON model key')
    return {'dict': [[k, canonical(value[k])] for k in sorted(value)]}


def model_binding(config):
    value = canonical(config)
    return {'encoding': 'typed-json-floathex-v1', 'candidate': value,
            'SHA256': hashlib.sha256(payload(value)).hexdigest()}


def identity(value):
    A.exact(value, IDENTITY, 'stream identity')
    for key in ('seed', 'variant', 'replicateId', 'mainIndex'):
        A.integer(value[key], key)
    A.need(1 <= value['mainIndex'] <= 10 and value['variant'] <= 8, 'Original main and variant identity')
    A.need(value['conditionId'] in ('C1', 'C2', 'C3', 'C4', 'C5'), 'Original condition identity')
    for key in ('modelId', 'personId', 'trialId'):
        A.need(type(value[key]) is str and bool(value[key]), 'Nonempty ' + key)
    return value


def digest(value):
    return hashlib.sha256(payload(value) + b'\n').hexdigest()


def stable_identity(config, seed, population, condition, variant, replicate, main):
    common = {k: v for k, v in config.items() if k not in ('task', 'state', 'pathReports')}
    model_hash = digest(canonical(common))
    person = digest({'namespace': VERSION, 'seed': seed, 'populationId': population,
                     'modelHash': model_hash, 'conditionId': condition, 'variant': variant,
                     'replicateId': replicate})
    result = {'seed': seed, 'modelId': config['candidateId'], 'personId': person,
              'trialId': digest({'personId': person, 'mainIndex': main}),
              'conditionId': condition, 'variant': variant, 'replicateId': replicate, 'mainIndex': main}
    return identity(result)


def make_key(ctx, logical, kind, index):
    ids, binding = ctx['identity'], ctx['bindings']
    return {'streamVersion': STREAM, 'populationId': binding['populationId'], 'seed': ids['seed'],
            'randomizationGroup': binding['randomizationGroup'], 'conditionId': ids['conditionId'],
            'scheduleVariant': ids['variant'], 'replicateId': ids['replicateId'],
            'mainIndex': ids['mainIndex'], 'logicalRecipient': logical, 'streamKind': kind, 'reportIndex': index}


def stream_draw(key, logs):
    fields = {'streamVersion', 'populationId', 'seed', 'randomizationGroup', 'conditionId',
              'scheduleVariant', 'replicateId', 'mainIndex', 'logicalRecipient', 'streamKind', 'reportIndex'}
    A.exact(key, fields, 'private need/report key')
    A.need(key['streamVersion'] == STREAM and key['conditionId'] in ('C1', 'C2', 'C3', 'C4', 'C5')
           and key['logicalRecipient'] in ('A', 'B') and key['streamKind'] in ('latent', 'report'),
           'Stream kind/original identity')
    for k in ('seed', 'scheduleVariant', 'replicateId', 'mainIndex', 'reportIndex'):
        A.integer(key[k], k)
    A.need(key['scheduleVariant'] <= 8 and 1 <= key['mainIndex'] <= 10
           and (key['streamKind'] != 'latent' or key['reportIndex'] == 0), 'Original stream bounds')
    for k in ('populationId', 'randomizationGroup'):
        A.need(type(key[k]) is str and bool(key[k]), 'Explicit ' + k)
    values = A.np.asarray(logs, dtype=float)
    A.need(values.shape == (100,) and not A.np.isnan(values).any() and not A.np.isposinf(values).any()
           and A.np.isfinite(values).any(), 'Finite-support sampling log law')
    probabilities = A.np.exp(values)
    A.need(A.np.isfinite(probabilities).all() and A.np.all(probabilities >= 0)
           and abs(float(probabilities.sum()) - 1.) <= 1e-12, 'Unaltered finite sampling mass')
    h = hashlib.sha256(payload(key)).hexdigest()
    seed = h[:32]
    u = float(A.np.random.Generator(A.np.random.PCG64(int(seed, 16))).random())
    total, chosen, before, selected, last_before = 0., None, 0., 0., 0.
    for index, p in enumerate(probabilities):
        last_before = total
        total = float(A.np.add(A.np.float64(total), A.np.float64(p)))
        if chosen is None and total > u:
            chosen, before, selected = index + 1, last_before, total
    fallback = chosen is None
    if fallback:
        chosen, before, selected = 100, last_before, total
    return {'schema': STREAM, 'key': copy.deepcopy(key), 'keySHA256': h, 'seedHex': seed,
            'uniformHex': u.hex(), 'logHex': [float(v).hex() for v in values],
            'probabilityHex': [float(v).hex() for v in probabilities],
            'rawMassHex': float(probabilities.sum()).hex(), 'category': chosen,
            'cdfPreviousHex': before.hex(), 'cdfSelectedHex': selected.hex(), 'roundingFallback': fallback}


def draw_terminal(config, ids):
    A.validate(config)
    identity(ids)
    A.need(ids['modelId'] == config['candidateId'], 'Declared model/instance identity')
    A.need(ids['conditionId'] == config['task']['condition_id'], 'Stream/task condition binding')
    key = {'version': TERMINAL_STREAM, 'identity': ids, 'candidate': canonical(config),
           'keyEncoding': 'typed-json-floathex-v1'}
    h = hashlib.sha256(payload(key)).hexdigest()
    u = float(A.np.random.Generator(A.np.random.PCG64(int(h, 16))).random())
    cumulative, chosen = 0., len(config['random']['probabilities']) - 1
    for j, p in enumerate(config['random']['probabilities']):
        cumulative += p
        if cumulative > u:
            chosen = j
            break
    return {'schema': TERMINAL_STREAM, 'key': key, 'keySHA256': h, 'uniform': u, 'index': chosen,
            'cdfRule': 'ordered_binary64_strict_greater_final_index_rounding_fallback'}


def unhex(rows):
    return [A.np.asarray([float.fromhex(x) for x in row]) for row in rows]


def decision(cfg, task, state):
    return A.evaluate(cfg, task, unhex(state['logHex']), state['budget'])


def paid_value(cfg, task, state, chosen, draw):
    slot = 0 if chosen['action'] == 'ask_A' else 1
    after = unhex(state['logHex'])
    after[slot] = task.posterior(after[slot], draw['category'])
    duration = A.request_seconds(cfg, slot, draw['category'])
    counts = list(state['counts'])
    counts[slot] += 1
    A.need(state['budget'] >= task.config.cost_millihours, 'No tutoring overdraft')
    return {'before': copy.deepcopy(state), 'slot': slot, 'report': draw['category'],
            'incrementalSecondsHex': duration.hex(),
            'after': {'budget': state['budget'] - task.config.cost_millihours, 'counts': counts,
                      'queryIndex': state['queryIndex'] + 1,
                      'logHex': [[float(v).hex() for v in row] for row in after],
                      'searchSecondsHex': (float.fromhex(state['searchSecondsHex']) + duration).hex()}}


def expected_terminal(cfg, state, stop):
    A.need(stop['action'] != 'random' and state['queryIndex'] == 0, 'Deterministic initial endpoint only')
    return {'action': stop['action'], 'forced': bool(stop.get('forced', False)), 'budget': state['budget'],
            'intendedIndex': stop['index'], 'actionExpectedCombinedGPA': stop['expectedCombinedGPA'],
            'expectedNetObjective': stop['objective'], 'searchSeconds': 0., 'terminalSeconds': stop['seconds'],
            'totalPersonalSeconds': stop['seconds'], 'realizedCombinedGPA': None, 'realizedNetObjective': None,
            'newScientificDraws': 0, 'targetHistoryFabricated': False,
            'scope': 'exact_initial_stop_expected_endpoint_not_realized_history'}


def terminal_value(cfg, ctx, state, stop, needs):
    snapshot = copy.deepcopy(cfg)
    snapshot['state'] = {'budget': state['budget'],
                         'logs': [[float(v) for v in row] for row in unhex(state['logHex'])]}
    event = draw_terminal(snapshot, ctx['identity']) if stop['action'] == 'random' else None
    j = event['index'] if event is not None else stop['index']
    c, b = A.TaskConfig(**cfg['task']), state['budget']
    table = native_utility_table(b, 1000, c.alpha, c.loss_multiplier)
    _, _, ga, gb = A.expected_curves(c, *unhex(state['logHex']), b)
    realized = 2*c.gpa_baseline + c.gpa_scale*float(table[j, needs[0]-1] + table[1000-j, needs[1]-1])
    search, seconds, rho = float.fromhex(state['searchSecondsHex']), stop['seconds'], cfg['valuation']['rhoGPAperSecond']
    return {'schema': 'r10-dynamic-terminal-v1', 'contextHash': digest(ctx), 'identity': ctx['identity'],
            'modelBinding': ctx['model'], 'terminalSnapshotBinding': model_binding(snapshot), 'stream': event,
            'action': stop['action'], 'forced': bool(stop.get('forced', False)), 'budget': b,
            'intendedIndex': j, 'intendedShare': j/1000, 'deliveredShare': None if b == 0 else j/1000,
            'deliveredAmountNumerators': [b*j, b*(1000-j)], 'amountDenominator': 1000000,
            'resourceUnit': 'tutoring_minutes', 'paidCount': state['queryIndex'], 'Q': int(state['queryIndex'] > 0),
            'choiceAvailable': b > 0, 'searchSeconds': search, 'terminalSeconds': seconds,
            'totalPersonalSeconds': search+seconds, 'actionExpectedCombinedGPA': stop['expectedCombinedGPA'],
            'sampledIndexExpectedCombinedGPA': float(ga[j]+gb[j]), 'realizedCombinedGPA': realized,
            'expectedNetObjective': stop['expectedCombinedGPA']-rho*(search+seconds),
            'realizedNetObjective': realized-rho*(search+seconds)}


def metric_row(cfg, root, state, terminal, targets):
    initial = [G.finite_mean(row) for row in root['initialLogs']]
    final = [G.finite_mean([float.fromhex(x) for x in row]) for row in state['logHex']]
    if targets is None:
        summary = A.terminal_summary(A.TaskConfig(**cfg['task']), *unhex(state['logHex']), state['budget'])
        targets = {name: item['index'] for name, item in summary['macros'].items()}
    positive = state['budget'] > 0
    row = {'mainIndex': root['mainIndex'], 'initialIntegerBudget': cfg['task']['total_millihours'],
           'remainingIntegerBudget': state['budget'], 'initialMeans': initial, 'terminalMeans': final,
           'jEGPA': targets['EO_expected_GPA'], 'jSurplus': targets['EO_raw_surplus'],
           'jAL': targets['AL'], 'jML': targets['ML'],
           'x0': (initial[0]-initial[1])/(cfg['task']['total_millihours']/1000),
           'xT': (final[0]-final[1])/(state['budget']/1000) if positive else None,
           'rawInitialGap': initial[0]-initial[1], 'Q': int(state['queryIndex'] > 0),
           'paidCount': state['queryIndex'], 'counts': state['counts'],
           'expectedFinalUtility': (terminal['actionExpectedCombinedGPA']-2*cfg['task']['gpa_baseline'])/cfg['task']['gpa_scale'],
           'expectedCombinedGPA': terminal['actionExpectedCombinedGPA'],
           'expectedNetObjective': terminal['expectedNetObjective'], 'realizedCombinedGPA': terminal['realizedCombinedGPA'],
           'totalPersonalSeconds': terminal['totalPersonalSeconds'], 'searchSeconds': terminal['searchSeconds'],
           'terminalSeconds': terminal['terminalSeconds'], 'action': terminal['action'], 'forced': terminal['forced'],
           'realizedAvailable': terminal['realizedCombinedGPA'] is not None}
    row.update(G.allocation_metrics(row, terminal['intendedIndex'] if positive else None))
    return row
