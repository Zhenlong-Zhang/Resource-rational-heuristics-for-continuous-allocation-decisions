"""Independent event replay for the portable C5 trajectory representation.

This checks one supplied result against supplied scientific inputs. It does not
authenticate those inputs, certify a numerical profile, or verify a population.
Policy replay uses the existing independent policy kernel; strict equality is a
conservative gate, not a claim that distinct numerical runtimes are equivalent.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import budget_horizon as horizon
from . import c5_public_streams as streams
from . import personal_time_sensitivity_reference as ref
from . import c5_public_inputs as inputs
from . import c5_reference as geometry_reference
from .bounded_task import TaskConfig


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _require_local_type(value, expected, label):
    if type(value) is not expected:
        raise ValueError(label + ' must belong to this import namespace; do not mix src.design_validation and design_validation')


def _same(a, b, label):
    if _bytes(a) != _bytes(b):
        raise ValueError('Independent mismatch: ' + label)


def _draw(key, counts, cumulative):
    digest = hashlib.sha256(_bytes(key)).hexdigest()
    u = float(np.random.Generator(np.random.PCG64(int(digest[:32], 16))).random())
    edges = [float.fromhex(x) for x in cumulative]
    index = next((i for i, edge in enumerate(edges) if edge > u), 99)
    if len(counts) != 100 or sum(counts) != 1 << 53:
        raise ValueError('Invalid independently supplied partition')
    n, d = u.as_integer_ratio()
    grid, rem = divmod(n * (1 << 53), d)
    if rem or not 0 <= grid < 1 << 53:
        raise ValueError('Non-grid private random value')
    acc = 0
    for j, count in enumerate(counts):
        acc += count
        if grid < acc:
            if j != index:
                raise ValueError('CDF and integer partition disagree')
            break
    else:
        raise ValueError('Uncovered private random value')
    return digest, u, index + 1, (edges[index-1] if index else 0.0), edges[index], u >= edges[-1]


def _fraction(value):
    if type(value) is not list or len(value) != 2 or any(type(x) is not str for x in value):
        raise ValueError('Invalid fraction')
    f = Fraction(int(value[0]), int(value[1]))
    if [str(f.numerator), str(f.denominator)] != value:
        raise ValueError('Noncanonical fraction')
    return f


@dataclass(frozen=True, init=False)
class AdmittedSpec:
    """Receiver-owned immutable snapshot; integrity is not provenance."""
    variant: int
    rows: tuple
    latent: object
    table: tuple
    likelihood: tuple
    design_bytes: bytes
    binding_bytes: bytes

    def __init__(self, spec, latent_config, report_bytes):
        if type(spec) is not dict or set(spec) != {'lookupPath', 'lookupSHA256', 'geometryCertificate'}:
            raise ValueError('A complete pinned scientific spec is required')
        _require_local_type(latent_config, streams.ValidatedLatentConfig, 'Validated latent input')
        lookup = inputs.load_fixed_inputs(spec['lookupPath'], spec['lookupSHA256'])
        certificate = spec['geometryCertificate']
        if (type(certificate) is not dict or set(certificate) !=
                {'scheduleVariant', 'lookupSHA256', 'latentConfigSHA256', 'design'}
                or type(certificate['scheduleVariant']) is not int
                or not 0 <= certificate['scheduleVariant'] < 9
                or certificate['lookupSHA256'] != spec['lookupSHA256']
                or certificate['latentConfigSHA256'] != latent_config.input_sha256):
            raise ValueError('Schedule geometry certificate binding mismatch')
        v = certificate['scheduleVariant']
        rows = lookup['roots'][10*v:10*v+10]
        roots = [{'rootKey': fixed['rootKey'], 'budget': fixed['config']['task']['total_millihours'],
                  'laws': [{'counts': law[0]} for law in latent_config.roots[10*v+i]]}
                 for i, fixed in enumerate(rows)]
        computed = geometry_reference.design_reference(roots)
        _same(certificate['design'], computed, 'independent schedule geometry')
        if any(computed[name]['status'] != 'available' for name in ('gap', 'partial_low')):
            raise ValueError('Singular schedule geometry')
        widths = {float(row['reportWidth']).hex() for row in rows}
        # A schedule may contain multiple widths: admit one table per width.
        if type(report_bytes) is not dict or set(report_bytes) != widths:
            raise ValueError('Complete schedule report tables required')
        tables, likelihoods = [], []
        for width in sorted(widths):
            raw = report_bytes[width]
            table = streams.ValidatedReportTable(raw, width)
            if not table.accepted_product_bytes:
                raise ValueError('Report table is not the accepted finite product')
            product = json.loads(raw)
            likelihood = np.asarray([[float.fromhex(x) for x in row['normalizedLogHex']]
                                     for row in product['rows']], dtype=np.float64).tobytes()
            tables.append((width, table))
            likelihoods.append((width, likelihood))
        binding = {'lookupSHA256': spec['lookupSHA256'],
                   'geometrySHA256': hashlib.sha256(_bytes(certificate)).hexdigest(),
                   'geometryScope': 'independently-recomputed-caller-inputs-not-historical-authentication'}
        for key, value in (('variant', v), ('rows', tuple(_bytes(row) for row in rows)),
                           ('latent', latent_config), ('table', tuple(tables)),
                           ('likelihood', tuple(likelihoods)), ('design_bytes', _bytes(computed)),
                           ('binding_bytes', _bytes(binding))):
            object.__setattr__(self, key, value)


def validate_spec(root, cfg, initial_decision, latent_config, design, spec):
    """Check per-root bindings against the receiver's admitted snapshot."""
    _require_local_type(spec, AdmittedSpec, 'Receiver admitted spec')
    _require_local_type(latent_config, streams.ValidatedLatentConfig, 'Validated latent input')
    if latent_config is not spec.latent:
        raise ValueError('Receiver-admitted spec and its latent snapshot required')
    v, m = root['scheduleVariant'], root['mainIndex']
    if type(v) is not int or v != spec.variant or type(m) is not int or not 1 <= m <= 10:
        raise ValueError('Invalid root identity')
    row = json.loads(spec.rows[m-1])
    definition = root['definition']
    if (inputs._digest(root) != row['rootSHA256'] or inputs._digest(cfg) != row['configSHA256']
            or _bytes(cfg) != _bytes(row['config']) or _bytes(initial_decision) != _bytes(row['decision'])
            or root['logicalToDisplayed'] != row['logicalToDisplayed']
            or definition['reportWidth'] != row['reportWidth']
            or cfg['task']['report_width'] != row['reportWidth']
            or definition['budgetMillihours'] != cfg['task']['total_millihours']
            or definition['costMillihours'] != cfg['task']['cost_millihours']):
        raise ValueError('Root, configuration, decision or display binding mismatch')
    if _bytes(design) != spec.design_bytes:
        raise ValueError('Trajectory geometry mismatch')
    return json.loads(spec.binding_bytes)


def receive_root(root, cfg, initial_decision, latent_config, report_bytes, design, replicate, result, *, spec):
    """Independently replay every event, state transition, endpoint and weight.

    Use this receiver's own admitted spec and its report table object; no
    producer admission is trusted. Rep1 cannot be established by this replay.
    """
    return _receive_root(root, cfg, initial_decision, latent_config, report_bytes, design,
                         replicate, result, spec=spec, fresh=None)


def receive_fresh_root(root, cfg, initial_decision, latent_config, report_bytes, design,
                       replicate, result, *, spec, reference_streams):
    """Replay one prospective member against a separately admitted receiver context.

    This establishes neither historical equivalence nor statistical authorization.
    The input identity is root-specific; it is not a population-wide run ID.
    """
    from . import c5_fresh_inputs, c5_fresh_streams_reference
    from .c5_fresh_domain import RunIdentity

    if type(reference_streams) is not c5_fresh_streams_reference.ReferenceStreams:
        raise ValueError('A fresh ReferenceStreams instance is required')
    s = reference_streams
    identity, context = s.identity, s.context
    if type(identity) is not RunIdentity or type(replicate) is not int or not 1 <= replicate <= 1840:
        raise ValueError('Invalid fresh identity or replicate')
    c5_fresh_inputs.check_unchanged(context)
    admitted = context['spec']
    if (identity.input_identity_sha256 != context['identity_sha256'] or
            s.replicate != replicate or s.variant != root['scheduleVariant'] or
            s.main != root['mainIndex'] or context['rs'] is not spec or
            context['r_latent'] is not latent_config or context['r_table'] is not report_bytes or
            _bytes(root) != _bytes(admitted['rawRoot']) or _bytes(cfg) != _bytes(admitted['config']) or
            _bytes(initial_decision) != _bytes(admitted['initialDecision']) or
            _bytes(design) != _bytes(admitted['design'])):
        raise ValueError('Fresh receiver context or member binding mismatch')
    fields = {'schema', 'rootKey', 'replicateId', 'ordinal', 'inputSHA256', 'latent', 'stages',
              'terminalDecision', 'terminalStream', 'terminalIndex', 'finalState', 'endpoint',
              'contribution', 'authority', 'historicalEquivalence', 'scientificSpec', 'runIdentity',
              'identityStatus', 'statisticalAuthorization', 'independentReceiver', 'resultSHA256'}
    if type(result) is not dict or set(result) != fields:
        raise ValueError('Invalid complete fresh trajectory envelope')
    if type(result['latent']) is not dict or type(result['stages']) is not list:
        raise ValueError('Invalid fresh event collections')
    digest = hashlib.sha256(_bytes({k: v for k, v in result.items() if k != 'resultSHA256'})).hexdigest()
    _same([result['resultSHA256'], result['runIdentity'], result['identityStatus'],
           result['statisticalAuthorization'], result['independentReceiver']],
          [digest, {'domain': identity.domain, 'runId': identity.run_id,
                    'inputIdentitySHA256': identity.input_identity_sha256},
           'new-unverified', False, False], 'fresh envelope')
    s._check()
    received = _receive_root(root, cfg, initial_decision, latent_config, report_bytes, design,
                             replicate, result, spec=spec, fresh=s)
    s._check()
    if s.identity != identity or s.context is not context or s.replicate != replicate:
        raise ValueError('Fresh receiver identity or context changed during replay')
    _same(hashlib.sha256(_bytes({k: v for k, v in result.items() if k != 'resultSHA256'})).hexdigest(),
          digest, 'fresh result changed during replay')
    _same(result['resultSHA256'], digest, 'fresh result digest changed during replay')
    received.update({'runIdentity': {'domain': identity.domain, 'runId': identity.run_id,
                                     'inputIdentitySHA256': identity.input_identity_sha256}, 'resultSHA256': digest,
                     'identityStatus': 'new-unverified', 'statisticalAuthorization': False})
    return received


def _receive_root(root, cfg, initial_decision, latent_config, report_bytes, design, replicate, result, *, spec, fresh):
    v, m = root['scheduleVariant'], root['mainIndex']
    if (type(v) is not int or not 0 <= v < 9 or type(m) is not int or not 1 <= m <= 10 or
            type(replicate) is not int or not (1 if fresh is not None else 2) <= replicate <= 1840 or
            type(root['logicalToDisplayed']) is not list or sorted(root['logicalToDisplayed']) != ['A', 'B']):
        raise ValueError('Invalid future root identity or display mapping')
    binding = validate_spec(root, cfg, initial_decision, latent_config, design, spec)
    order = root['logicalToDisplayed']
    c = TaskConfig(**cfg['task'])
    b, cost, limit = root['definition']['budgetMillihours'], c.cost_millihours, c.max_assessments
    if (cfg['task']['condition_id'] != 'C5' or cfg['valuation']['rhoGPAperSecond'] != 0.001 or
            initial_decision['budget'] != b):
        raise ValueError('Expected fixed low-price C5 root and initial decision')
    if (type(b) is not int or type(cost) is not int or type(limit) is not int or cost <= 0 or
            limit < 0 or divmod(b, cost)[0] != limit or not 0 < b % cost < cost):
        raise ValueError('Independent all-path budget failure')
    width = float(c.report_width).hex()
    table = dict(spec.table).get(width)
    if table is None or report_bytes is not table:
        raise ValueError('Receiver-admitted report table required')
    task = object.__new__(horizon.BudgetTask)
    task.config = c
    task.log_likelihood = np.frombuffer(dict(spec.likelihood)[width], dtype=np.float64).reshape(100, 100)
    task.initial_logs = tuple(np.asarray(side, dtype=np.float64) for side in root['initialLogs'])
    if len(task.initial_logs) != 2 or any(len(side) != 100 for side in task.initial_logs):
        raise ValueError('Invalid root priors')
    for side in task.initial_logs:
        side.setflags(write=False)
    task._initial_values = None
    state = {'budget': b, 'logHex': [[float(x).hex() for x in side] for side in task.initial_logs],
             'counts': [0, 0], 'queryIndex': 0, 'searchSecondsHex': 0.0.hex()}
    identity = ref.stable_identity(cfg, 20261019, streams.POPULATION, 'C5', v, replicate, m)
    context = {'identity': identity, 'bindings': {'populationId': streams.POPULATION,
                                                   'randomizationGroup': streams.GROUP}}
    expected_hash = hashlib.sha256(_bytes({'root': root, 'config': cfg, 'initialDecision': initial_decision,
        'design': design, 'latentConfigSHA256': latent_config.input_sha256,
        'reportProductSHA256': table.product_sha256})).hexdigest()
    _same([result['schema'], result['rootKey'], result['replicateId'], result['ordinal'], result['inputSHA256'],
           result['authority'], result['historicalEquivalence'], result['scientificSpec']],
          ['c5-fresh-trajectory-producer-v1' if fresh is not None else 'c5-public-trajectory-v1',
           ['C5', v, m], replicate, 90*(replicate-1)+10*v+m-1, expected_hash,
           'caller-supplied-inputs-not-authenticated', False, binding], 'identity and input binding')
    latent = {}
    for logical in ('A', 'B'):
        if fresh is not None:
            expected = fresh.verify_latent(result['latent'][logical], logical)
            latent[logical] = expected['category']
            continue
        key = ref.make_key(context, logical, 'latent', 0)
        counts, cumulative = latent_config.roots[v*10+m-1][('A', 'B').index(logical)]
        digest, u, category, before, selected, fallback = _draw(key, counts, cumulative)
        expected = {'schema': 'r14-c5-fixed-latent-partition-draw-v1', 'key': key, 'keySHA256': digest,
                    'seedHex': digest[:32], 'uniformHex': u.hex(), 'latentConfigSHA256': latent_config.input_sha256,
                    'category': category, 'cdfPreviousHex': before.hex(), 'cdfSelectedHex': selected.hex(),
                    'roundingFallback': fallback}
        _same(result['latent'][logical], expected, 'latent event')
        latent[logical] = category
    if set(result['latent']) != {'A', 'B'}:
        raise ValueError('Unexpected latent event')
    needs = [latent[x] for x in order]
    decision = copy.deepcopy(initial_decision)
    stages = result['stages']
    cursor = 0
    while decision['action'] in ('ask_A', 'ask_B'):
        if state['queryIndex'] >= limit or cursor >= len(stages):
            raise ValueError('Missing event or finite horizon exceeded')
        slot = 0 if decision['action'] == 'ask_A' else 1
        logical = order[slot]
        stage = stages[cursor]
        if fresh is not None and (type(stage) is not dict or set(stage) != {'decision', 'report', 'paid', 'nextDecision'}):
            raise ValueError('Invalid complete stage')
        if fresh is not None:
            expected = fresh.verify_report(stage['report'], logical, state['counts'][slot], result['latent'][logical])
        else:
            key = ref.make_key(context, logical, 'report', state['counts'][slot])
            counts, cumulative = table.rows[needs[slot]-1]
            digest, u, category, before, selected, fallback = _draw(key, counts, cumulative)
            latent_event = result['latent'][logical]
            expected = {'schema': 'r14-c5-prospective-report-partition-draw-v1', 'key': key,
                    'keySHA256': digest, 'seedHex': digest[:32], 'uniformHex': u.hex(), 'category': category,
                    'cdfPreviousHex': before.hex(), 'cdfSelectedHex': selected.hex(), 'roundingFallback': fallback,
                    'latentKeySHA256': latent_event['keySHA256'], 'latentConfigSHA256': latent_config.input_sha256,
                    'latentCategory': needs[slot], 'reportWidthHex': width, 'tableProductSHA256': table.product_sha256,
                    'acceptedProductByteMatch': True, 'tableIdentity': 'accepted_finite_table_bytes',
                    'historicalEventEstablished': False}
        _same(stage['decision'], decision, 'decision')
        _same(stage['report'], expected, 'report')
        paid = ref.paid_value(cfg, task, state, decision, expected)
        _same(stage['paid'], paid, 'paid transition')
        state = paid['after']
        if state['budget'] != b-cost*state['queryIndex'] or state['budget'] <= 0 or sum(state['counts']) != state['queryIndex']:
            raise ValueError('Independent paid budget failure')
        decision = ref.decision(cfg, task, state)
        _same(stage['nextDecision'], decision, 'next decision')
        cursor += 1
        horizon.native_utility_table.cache_clear()
        horizon.native_perfect_information.cache_clear()
    if cursor != len(stages):
        raise ValueError('Unexpected trailing stages')
    _same(result['terminalDecision'], decision, 'terminal decision')
    _same(result['finalState'], state, 'final state')
    stop = decision['stop']
    if decision['action'] != stop['action']:
        raise ValueError('Invalid terminal decision')
    event = None
    if stop['action'] == 'random':
        snapshot = copy.deepcopy(cfg)
        snapshot['state'] = {'budget': state['budget'], 'logs': [[float.fromhex(x) for x in row] for row in state['logHex']]}
        if fresh is not None:
            event = fresh.verify_terminal(result['terminalStream'], snapshot, identity)
            index = event['index']
        else:
            key = {'version': 'r10-native-terminal-pcg64-sha256-v1', 'randomizationGroup': streams.TERMINAL,
               'identity': identity, 'candidate': ref.key_value(snapshot), 'keyEncoding': 'typed-json-floathex-v1'}
            digest = hashlib.sha256(_bytes(key)).hexdigest()
            u = float(np.random.Generator(np.random.PCG64(int(digest, 16))).random())
            probabilities = snapshot['random']['probabilities']
            total, index = 0., len(probabilities)-1
            for j, p in enumerate(probabilities):
                total += p
                if total > u:
                    index = j
                    break
            event = {'schema': 'r14-c5-grouped-terminal-draw-v1', 'key': key, 'keySHA256': digest,
                 'uniform': u, 'index': index, 'cdfRule': 'ordered_binary64_strict_greater_final_index_rounding_fallback'}
    else:
        index = stop['index']
    if type(index) is not int or not 0 <= index <= 1000 or state['budget'] <= 0:
        raise ValueError('Undefined or invalid terminal allocation')
    _same(result['terminalStream'], event, 'terminal stream')
    _same(result['terminalIndex'], index, 'terminal index')
    tie = needs[0] == needs[1]
    share = None if tie else Fraction(index if needs[0] > needs[1] else 1000-index, 1000)
    encode = lambda x: [str(x.numerator), str(x.denominator)]
    _same(result['endpoint'], {'latentDisplayed': needs, 'status': 'exact_tie' if tie else 'complete',
                               'higherNeedShare': None if tie else encode(share)}, 'endpoint')
    low, high = sorted(needs)
    vectors = {'gap': [Fraction(1), Fraction(1000*(high-low), b)],
               'partial_low': [Fraction(1), Fraction(1000*low, b), Fraction(1000*high, b)]}
    values = {}
    for name, z in vectors.items():
        row = design[name]['inverse'][1]
        if len(row) != len(z):
            raise ValueError('Invalid geometry row')
        weight = sum((_fraction(x)*y for x, y in zip(row, z)), Fraction(0))
        values[name] = encode(Fraction(0) if tie else weight*(share-Fraction(1, 2)))
    _same(result['contribution'], values, 'contribution')
    return {'status': 'independent_replay_pass', 'rootKey': ['C5', v, m], 'replicateId': replicate,
            'inputSHA256': expected_hash, 'authority': 'caller-supplied-inputs-not-authenticated',
            'historicalEquivalence': False}
