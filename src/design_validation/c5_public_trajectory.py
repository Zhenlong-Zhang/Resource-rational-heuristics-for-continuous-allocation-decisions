"""Portable C5 trajectory kernel, not an input or runtime authority.

The caller must authenticate the fixed root, low-price configuration and full
initial decision, latent configuration, report-product bytes and numerical
profile *before* calling.  These objects are scientific inputs, not receipts.
This module neither establishes that authority nor reconstructs historical rep1.
The independent receiver must be run before a result is admitted.
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
from . import personal_time_sensitivity_streams as policy
from . import c5_public_inputs as inputs
from . import c5_moments as geometry
from .bounded_task import TaskConfig


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _require_local_type(value, expected, label):
    if type(value) is not expected:
        raise ValueError(label + ' must belong to this import namespace; do not mix src.design_validation and design_validation')


def _rational(x):
    return [str(x.numerator), str(x.denominator)]


def _fraction(x):
    if type(x) is not list or len(x) != 2 or any(type(v) is not str for v in x):
        raise ValueError('Expected canonical fraction')
    try:
        f = Fraction(int(x[0]), int(x[1]))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('Invalid fraction') from exc
    if _rational(f) != x:
        raise ValueError('Noncanonical fraction')
    return f


@dataclass(frozen=True, init=False)
class AdmittedSpec:
    """Producer-owned snapshot of checked inputs, not historical authority."""
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
        cert = spec['geometryCertificate']
        if (type(cert) is not dict or set(cert) !=
                {'scheduleVariant', 'lookupSHA256', 'latentConfigSHA256', 'design'}
                or type(cert['scheduleVariant']) is not int or not 0 <= cert['scheduleVariant'] < 9
                or cert['lookupSHA256'] != spec['lookupSHA256']
                or cert['latentConfigSHA256'] != latent_config.input_sha256):
            raise ValueError('Schedule geometry certificate binding mismatch')
        v = cert['scheduleVariant']
        rows = lookup['roots'][10*v:10*v+10]
        roots = [{'rootKey': row['rootKey'], 'budget': row['config']['task']['total_millihours'],
                  'laws': [{'counts': list(law[0])} for law in latent_config.roots[10*v+i]]}
                 for i, row in enumerate(rows)]
        computed = geometry.design_payload(roots)
        if _digest(computed) != _digest(cert['design']):
            raise ValueError('Producer schedule geometry mismatch')
        if any(computed[name]['status'] != 'available' for name in ('gap', 'partial_low')):
            raise ValueError('Singular schedule geometry')
        widths = {float(row['reportWidth']).hex() for row in rows}
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
        encode = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        binding = {'lookupSHA256': spec['lookupSHA256'], 'geometrySHA256': _digest(cert),
                   'geometryScope': 'independently-recomputed-caller-inputs-not-historical-authentication'}
        for key, value in (('variant', v), ('rows', tuple(encode(row) for row in rows)),
                           ('latent', latent_config), ('table', tuple(tables)),
                           ('likelihood', tuple(likelihoods)), ('design_bytes', encode(computed)),
                           ('binding_bytes', encode(binding))):
            object.__setattr__(self, key, value)


def _validate_root(root, cfg, initial_decision, latent_config, design, spec):
    _require_local_type(spec, AdmittedSpec, 'Producer admitted spec')
    _require_local_type(latent_config, streams.ValidatedLatentConfig, 'Validated latent input')
    if latent_config is not spec.latent:
        raise ValueError('Producer-admitted spec and its latent snapshot required')
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
            or definition['costMillihours'] != cfg['task']['cost_millihours']
            or json.dumps(design, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() != spec.design_bytes):
        raise ValueError('Root, configuration, decision or geometry binding mismatch')
    return json.loads(spec.binding_bytes)


def produce_root(root, cfg, initial_decision, latent_config, report_bytes, design, replicate, *, spec):
    """Produce one new future root; inputs must already be independently admitted.

    ``root`` has scheduleVariant, mainIndex, logicalToDisplayed, initialLogs,
    and definition.budgetMillihours. ``cfg`` is the original rate_config for
    rho_0.001; ``initial_decision`` is the complete fixed low decision for this
    exact root/config, not a recomputation. ``design`` supplies exact inverse
    matrices for gap and partial_low. ``report_bytes`` is the report table object
    from this producer's admitted spec for cfg.task.report_width, not raw bytes.
    Rep1 is intentionally not generated here.
    Output hashes bind supplied values but do not authenticate their authority.
    No filesystem checkpoint or cross-platform equivalence is implied.
    """
    if type(replicate) is not int or not 2 <= replicate <= 1840:
        raise ValueError('Only future replicates 2..1840; historical rep1 is supplied data')
    return _produce(root, cfg, initial_decision, latent_config, report_bytes, design,
                    replicate, spec=spec, fresh=None)


def produce_fresh_root(root, cfg, initial_decision, latent_config, report_bytes, design,
                       replicate, *, spec, producer_streams):
    """Produce a prospective member, not a receiver or an accepted result.

    The stream instance must belong to the exact admitted context and member.
    Independent receiving and any statistical authorization remain separate.
    """
    from . import c5_fresh_inputs, c5_fresh_streams
    from .c5_fresh_domain import RunIdentity

    if type(producer_streams) is not c5_fresh_streams.ProducerStreams:
        raise ValueError('A fresh ProducerStreams instance is required')
    s = producer_streams
    if type(s.identity) is not RunIdentity or type(replicate) is not int or not 1 <= replicate <= 1840:
        raise ValueError('Invalid fresh identity or replicate')
    identity = s.identity
    c5_fresh_inputs.check_unchanged(s.context)
    context = s.context
    admitted = context['spec']
    if (s.identity.input_identity_sha256 != context['identity_sha256'] or
            s.replicate != replicate or s.variant != root['scheduleVariant'] or
            s.main != root['mainIndex'] or context['ps'] is not spec or
            context['p_latent'] is not latent_config or context['p_table'] is not report_bytes or
            _bytes(root) != _bytes(admitted['rawRoot']) or
            _bytes(cfg) != _bytes(admitted['config']) or
            _bytes(initial_decision) != _bytes(admitted['initialDecision']) or
            _bytes(design) != _bytes(admitted['design'])):
        raise ValueError('Fresh stream context or member binding mismatch')
    s._check()
    result = _produce(root, cfg, initial_decision, latent_config, report_bytes, design,
                      replicate, spec=spec, fresh=s)
    s._check()
    if s.identity != identity or s.context is not context or s.replicate != replicate:
        raise ValueError('Fresh stream identity or context changed during production')
    result['schema'] = 'c5-fresh-trajectory-producer-v1'
    result['runIdentity'] = {'domain': identity.domain, 'runId': identity.run_id,
                             'inputIdentitySHA256': identity.input_identity_sha256}
    result['identityStatus'] = 'new-unverified'
    result['statisticalAuthorization'] = False
    result['independentReceiver'] = False
    result['resultSHA256'] = _digest(result)
    return result


def _produce(root, cfg, initial_decision, latent_config, report_bytes, design, replicate, *, spec, fresh):
    v, m = root['scheduleVariant'], root['mainIndex']
    if fresh is None:
        streams.private_key(v, m, replicate, 'A', 'latent', 0)
    order = root['logicalToDisplayed']
    if type(order) is not list or sorted(order) != ['A', 'B']:
        raise ValueError('Invalid logical/display mapping')
    binding = _validate_root(root, cfg, initial_decision, latent_config, design, spec)
    b = root['definition']['budgetMillihours']
    c = TaskConfig(**cfg['task'])
    cost, qmax = c.cost_millihours, c.max_assessments
    if (cfg['task']['condition_id'] != 'C5' or cfg['valuation']['rhoGPAperSecond'] != 0.001 or
            initial_decision['budget'] != b):
        raise ValueError('Expected fixed low-price C5 root and initial decision')
    if (type(b) is not int or type(cost) is not int or type(qmax) is not int or
            cost <= 0 or qmax < 0 or divmod(b, cost)[0] != qmax or not 0 < b % cost < cost):
        raise ValueError('Positive all-path budget certificate failed')
    width = float(c.report_width).hex()
    table = dict(spec.table).get(width)
    if table is None or report_bytes is not table:
        raise ValueError('Producer-admitted report table required')
    task = object.__new__(horizon.BudgetTask)
    task.config = c
    task.log_likelihood = np.frombuffer(dict(spec.likelihood)[width], dtype=np.float64).reshape(100, 100)
    task.initial_logs = tuple(np.asarray(side, dtype=np.float64) for side in root['initialLogs'])
    if len(task.initial_logs) != 2 or any(len(side) != 100 for side in task.initial_logs):
        raise ValueError('Expected two complete root priors')
    for side in task.initial_logs:
        side.setflags(write=False)
    task._initial_values = None
    state = {'budget': b, 'logHex': [[float(x).hex() for x in side] for side in task.initial_logs],
             'counts': [0, 0], 'queryIndex': 0, 'searchSecondsHex': 0.0.hex()}
    latent_events = {}
    for logical in ('A', 'B'):
        if fresh is None:
            key = streams.private_key(v, m, replicate, logical, 'latent', 0)
            latent_events[logical] = streams.private_draw(key, latent_config)
        else:
            latent_events[logical] = fresh.latent(logical)
    needs = [latent_events[logical]['category'] for logical in order]
    decision = copy.deepcopy(initial_decision)
    stages = []
    while decision['action'] in ('ask_A', 'ask_B'):
        slot = 0 if decision['action'] == 'ask_A' else 1
        if state['queryIndex'] >= qmax:
            raise ValueError('Decision exceeds finite horizon')
        logical = order[slot]
        if fresh is None:
            key = streams.private_key(v, m, replicate, logical, 'report', state['counts'][slot])
            event = streams.report_draw(key, table, width, latent_events[logical], latent_config)
        else:
            event = fresh.report(logical, state['counts'][slot], latent_events[logical])
        paid = policy.paid_value(cfg, task, state, decision, event)
        after = paid['after']
        if (after['budget'] != b - cost * after['queryIndex'] or after['budget'] <= 0 or
                sum(after['counts']) != after['queryIndex']):
            raise ValueError('Invalid paid transition budget or counts')
        following = policy.decision(cfg, task, after)
        stages.append({'decision': decision, 'report': event, 'paid': paid, 'nextDecision': following})
        state, decision = after, following
        horizon.native_utility_table.cache_clear()
        horizon.native_perfect_information.cache_clear()
    stop = decision['stop']
    if decision['action'] != stop['action']:
        raise ValueError('Terminal decision and stop action disagree')
    identity = policy.stable_identity(cfg, 20261019, streams.POPULATION, 'C5', v, replicate, m)
    terminal_event = None
    if stop['action'] == 'random':
        snapshot = copy.deepcopy(cfg)
        snapshot['state'] = {'budget': state['budget'], 'logs': [[float.fromhex(x) for x in row]
                                                                   for row in state['logHex']]}
        terminal_event = (streams.terminal_draw(snapshot, identity) if fresh is None
                          else fresh.terminal(snapshot, identity))
        index = terminal_event['index']
    else:
        index = stop['index']
    if type(index) is not int or not 0 <= index <= 1000:
        raise ValueError('Invalid terminal allocation index')
    if state['budget'] <= 0:
        raise ValueError('Undefined terminal share')
    tie = needs[0] == needs[1]
    share = None if tie else Fraction(index if needs[0] > needs[1] else 1000-index, 1000)
    values = {}
    low, high = sorted(needs)
    vectors = {'gap': [Fraction(1), Fraction(1000*(high-low), b)],
               'partial_low': [Fraction(1), Fraction(1000*low, b), Fraction(1000*high, b)]}
    for name, vector in vectors.items():
        row = design[name]['inverse'][1]
        if len(row) != len(vector):
            raise ValueError('Invalid geometry inverse row')
        weight = sum((_fraction(x)*z for x, z in zip(row, vector)), Fraction(0))
        values[name] = _rational(Fraction(0) if tie else weight*(share-Fraction(1, 2)))
    return {'schema': 'c5-public-trajectory-v1', 'rootKey': ['C5', v, m],
            'replicateId': replicate, 'ordinal': 90*(replicate-1)+10*v+m-1,
            'inputSHA256': _digest({'root': root, 'config': cfg, 'initialDecision': initial_decision,
                                    'design': design, 'latentConfigSHA256': latent_config.input_sha256,
                                    'reportProductSHA256': table.product_sha256}),
            'latent': latent_events, 'stages': stages, 'terminalDecision': decision,
            'terminalStream': terminal_event, 'terminalIndex': index, 'finalState': state,
            'endpoint': {'latentDisplayed': needs, 'status': 'exact_tie' if tie else 'complete',
                         'higherNeedShare': None if tie else _rational(share)},
            'contribution': values,
            'authority': 'caller-supplied-inputs-not-authenticated', 'historicalEquivalence': False,
            'scientificSpec': binding}
