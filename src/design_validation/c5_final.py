"""Portable exact C5 fixed-1840 reduction, not historical authentication.

Inputs are caller-supplied endpoint and design operands.  This module neither
generates a policy nor establishes their origin, display mapping, share, law,
independence or bounds. Fresh and saved identify input routes, not historical
actors. This arithmetic does not issue confidence bands or analytic zeros.
"""
from fractions import Fraction
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from contextlib import ExitStack

ROOTS = 90
REPLICATES = 1840
POPULATION = ROOTS * REPLICATES
TARGETS = ('gap', 'partial_low')
_INTEGER = re.compile(r'(?:0|-?[1-9][0-9]*)\Z')
_ENDPOINT_FIELDS = {'rootKey', 'latentDisplayed', 'latentLogical', 'initialBudget',
                    'remainingBudget', 'queryCount', 'status', 'reason', 'nonTie',
                    'higherNeedShare'}


def _same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def rational(value):
    if (type(value) is not list or len(value) != 2 or
            any(type(x) is not str or not _INTEGER.fullmatch(x) for x in value)):
        raise ValueError('Expected canonical integer-string rational pair')
    numerator, denominator = map(int, value)
    if denominator <= 0:
        raise ValueError('Positive denominator required')
    result = Fraction(numerator, denominator)
    if encode(result) != value:
        raise ValueError('Rational must be reduced')
    return result


def encode(value):
    value = Fraction(value)
    return [str(value.numerator), str(value.denominator)]


def coordinate(rep, root):
    if (type(rep) is not int or not 1 <= rep <= REPLICATES or
            type(root) is not list or len(root) != 3 or root[0] != 'C5' or
            type(root[1]) is not int or not 0 <= root[1] < 9 or
            type(root[2]) is not int or not 1 <= root[2] <= 10):
        raise ValueError('Invalid fixed C5 coordinate')
    return ROOTS * (rep - 1) + 10 * root[1] + root[2] - 1


def contribution(design, endpoint):
    """Calculate I*w*(Y-1/2) from a complete endpoint and inverse rows."""
    if (type(endpoint) is not dict or set(endpoint) != _ENDPOINT_FIELDS or
            type(design) is not dict or set(design) != set(TARGETS)):
        raise ValueError('Endpoint and two-target design required')
    needs = endpoint.get('latentDisplayed')
    logical = endpoint.get('latentLogical')
    budget, remaining = endpoint.get('initialBudget'), endpoint.get('remainingBudget')
    queries = endpoint.get('queryCount')
    if (type(needs) is not list or len(needs) != 2 or
            any(type(x) is not int or not 1 <= x <= 100 for x in needs) or
            type(logical) is not dict or set(logical) != {'A', 'B'} or
            any(type(x) is not int or not 1 <= x <= 100 for x in logical.values()) or
            sorted(logical.values()) != sorted(needs) or
            type(budget) is not int or budget <= 0 or type(remaining) is not int or
            not 0 < remaining <= budget or type(queries) is not int or queries < 0):
        raise ValueError('Incomplete or unavailable endpoint')
    rows = {}
    for target, size in (('gap', 2), ('partial_low', 3)):
        row = design[target]
        if type(row) is not list or len(row) != size:
            raise ValueError('Expected inverse matrix second row')
        rows[target] = [rational(x) for x in row]
    tie = needs[0] == needs[1]
    if tie:
        if (endpoint.get('status') != 'exact_tie' or endpoint.get('nonTie') is not False or
                endpoint.get('reason') != 'equal_true_needs' or endpoint.get('higherNeedShare') is not None):
            raise ValueError('Invalid exact tie')
        return {name: ['0', '1'] for name in TARGETS}
    if endpoint.get('status') != 'complete' or endpoint.get('nonTie') is not True or endpoint.get('reason') is not None:
        raise ValueError('Non-tie outcome unavailable')
    share = rational(endpoint.get('higherNeedShare'))
    if not 0 <= share <= 1:
        raise ValueError('Share must be bounded')
    low, high = sorted(needs)
    vectors = {'gap': [Fraction(1), Fraction(1000 * (high - low), budget)],
               'partial_low': [Fraction(1), Fraction(1000 * low, budget), Fraction(1000 * high, budget)]}
    return {name: encode(sum((a * b for a, b in zip(rows[name], vectors[name])), Fraction()) *
                         (share - Fraction(1, 2))) for name in TARGETS}


def reduce_final(rows, *, input_kind):
    """Reduce all fixed members; saved rows must include their supplied values.

    Each row has replicateId, rootKey, endpoint and design.  A design maps
    each target to inverse[1], encoded as reduced rational pairs.  Saved rows
    additionally carry contributions; those values are checked against operands.
    This checks arithmetic on supplied operands, not endpoint parity, policy
    execution, historical raw provenance, or independent producer validation.
    No inferential band or high-price analytic zero is issued by this function.
    """
    if input_kind not in ('fresh', 'saved'):
        raise ValueError('Explicit fresh or saved input_kind required')
    seen, total, ties = bytearray(POPULATION), {x: Fraction() for x in TARGETS}, 0
    root_geometry = {}
    count = 0
    expected = {'replicateId', 'rootKey', 'endpoint', 'design'}
    if input_kind == 'saved':
        expected = expected | {'contributions'}
    for row in rows:
        if type(row) is not dict or set(row) != expected:
            raise ValueError('Unexpected row schema for input route')
        ordinal = coordinate(row['replicateId'], row['rootKey'])
        if seen[ordinal]:
            raise ValueError('Duplicate scientific member')
        if type(row['endpoint']) is not dict or not _same(row['endpoint'].get('rootKey'), row['rootKey']):
            raise ValueError('Endpoint root mismatch')
        values = contribution(row['design'], row['endpoint'])
        if input_kind == 'saved' and not _same(row['contributions'], values):
            raise ValueError('Saved contributions differ from supplied operands')
        root_id = tuple(row['rootKey'])
        geometry = (tuple(tuple(tuple(x) for x in row['design'][name]) for name in TARGETS),
                    row['endpoint']['initialBudget'])
        if root_id in root_geometry and root_geometry[root_id] != geometry:
            raise ValueError('Root design or initial budget changed across repetitions')
        root_geometry[root_id] = geometry
        seen[ordinal] = 1
        count += 1
        ties += int(row['endpoint']['status'] == 'exact_tie')
        for name in TARGETS:
            total[name] += rational(values[name])
    if count != POPULATION or not all(seen):
        raise ValueError('Full fixed population required')
    results = {}
    for name in TARGETS:
        estimate = total[name] / POPULATION
        results[name] = {'total': encode(total[name]), 'denominator': POPULATION,
                         'lowPriceEstimate': encode(estimate)}
    return {'inputKind': input_kind, 'historicalRawParityEstablished': False,
            'endpointParityEstablished': False,
            'population': POPULATION, 'exactTies': ties, 'results': results,
            'scope': 'arithmetic_on_supplied_operands_only'}


def prefix_bound(root_bounds, variance, maximum, log_upper, *, n=REPLICATES):
    """Evaluate the accepted sufficient inequality on supplied exact geometry.

    This is arithmetic only, not a certificate or inference authorization.
    The caller must independently establish valid M, V, b and an upper bound
    on ln(80), along with the fixed-law and independence assumptions.
    """
    if type(n) is not int or n < 2 or type(root_bounds) is not list or len(root_bounds) != ROOTS:
        raise ValueError('Expected 90 bounds and at least two replicates')
    bounds = [rational(x) for x in root_bounds]
    v, b, ell = map(rational, (variance, maximum, log_upper))
    if any(x < 0 for x in bounds) or v < 0 or b < max(bounds) or ell <= 0:
        raise ValueError('Invalid supplied bound operands')
    delta = sum(bounds, Fraction()) / (ROOTS * n)
    e = Fraction(1, 10) - delta
    lhs = 8100 * n * e * e
    rhs = 2 * (v + 30 * b * e) * ell
    return {'n': n, 'delta': encode(delta), 'remainingRadius': encode(e),
            'lhs': encode(lhs), 'rhs': encode(rhs), 'inequalitySatisfied': e > 0 and lhs >= rhs,
            'scope': 'arithmetic_only_not_certificate_supplied_geometry_and_log_unverified'}


def _typed_candidate(value):
    if type(value) is not dict or len(value) != 1:
        return False
    tag, item = next(iter(value.items()))
    if tag == 'null':
        return item is True
    if tag == 'bool':
        return type(item) is bool
    if tag == 'int':
        return type(item) is str and _INTEGER.fullmatch(item) is not None
    if tag == 'str':
        return type(item) is str
    if tag == 'floatHex':
        if type(item) is not str:
            return False
        try:
            number = float.fromhex(item)
        except (ValueError, OverflowError):
            return False
        return not math.isnan(number) and number != math.inf and number.hex() == item
    if tag == 'list':
        return type(item) is list and all(_typed_candidate(x) for x in item)
    if tag == 'dict':
        if type(item) is not list or any(type(x) is not list or len(x) != 2 or
                                         type(x[0]) is not str or not _typed_candidate(x[1])
                                         for x in item):
            return False
        names = [x[0] for x in item]
        return names == sorted(set(names))
    return False


def _event_seed_bits(key):
    if type(key) is not dict:
        raise ValueError('Event key must be an object')
    private_fields = {'streamVersion', 'populationId', 'seed', 'randomizationGroup',
                      'conditionId', 'scheduleVariant', 'replicateId', 'mainIndex',
                      'logicalRecipient', 'streamKind', 'reportIndex'}
    if set(key) == private_fields:
        if (key['streamVersion'] != 'r10-private-need-report-v1' or
                key['populationId'] != 'r10-working-five-v0-rep0-candidate-not-population-estimate' or
                key['randomizationGroup'] != 'r14-c5-paired-projection-private-stream-v1' or
                key['conditionId'] != 'C5' or type(key['seed']) is not int or key['seed'] != 20261019 or
                type(key['scheduleVariant']) is not int or not 0 <= key['scheduleVariant'] < 9 or
                type(key['replicateId']) is not int or not 1 <= key['replicateId'] <= 1840 or
                type(key['mainIndex']) is not int or not 1 <= key['mainIndex'] <= 10 or
                type(key['logicalRecipient']) is not str or key['logicalRecipient'] not in ('A', 'B') or
                type(key['streamKind']) is not str or key['streamKind'] not in ('latent', 'report') or
                type(key['reportIndex']) is not int or key['reportIndex'] < 0 or
                (key['streamKind'] == 'latent' and key['reportIndex'] != 0)):
            raise ValueError('Invalid private event domain')
        return 128
    if set(key) != {'version', 'randomizationGroup', 'identity', 'candidate', 'keyEncoding'}:
        raise ValueError('Unknown event key domain')
    identity = key['identity']
    if (key['version'] != 'r10-native-terminal-pcg64-sha256-v1' or
            key['randomizationGroup'] != 'r14-c5-paired-projection-terminal-v1' or
            key['keyEncoding'] != 'typed-json-floathex-v1' or
            type(identity) is not dict or set(identity) != {'seed', 'modelId', 'personId', 'trialId',
                                                           'conditionId', 'variant', 'replicateId', 'mainIndex'} or
            type(identity['seed']) is not int or identity['seed'] != 20261019 or
            identity['conditionId'] != 'C5' or
            type(identity['variant']) is not int or not 0 <= identity['variant'] < 9 or
            type(identity['replicateId']) is not int or not 1 <= identity['replicateId'] <= 1840 or
            type(identity['mainIndex']) is not int or not 1 <= identity['mainIndex'] <= 10 or
            any(type(identity[k]) is not str or not identity[k] for k in ('modelId', 'personId', 'trialId')) or
            type(key['candidate']) is not dict or set(key['candidate']) != {'dict'} or
            not _typed_candidate(key['candidate'])):
        raise ValueError('Invalid terminal event domain')
    return 256


def audit_supplied_events(events):
    """Audit supplied key bytes and integer seeds, not completeness or provenance.

    An event is exactly {key, keySHA256, seedBits, seedInteger}.  Keys are
    canonical JSON objects, hashed without a newline.  Private streams use
    the first 128 digest bits; terminal streams use all 256.  This inventory
    alone cannot establish that events were actually used, or that none are
    missing.  In particular it does not authenticate the historical prefix.
    Terminal candidate syntax is checked, not its policy or snapshot binding.

    A future actual-use admission must bind each complete root to its two latent
    events, ordered report frontier, optional terminal and independent P/R proof,
    then bind the complete 165600-root roster and both audits to an authority.
    No such admission or historical-prefix authority is asserted here.
    """
    keys, digests, seeds, records = set(), set(), set(), []
    for event in events:
        if type(event) is not dict or set(event) != {'key', 'keySHA256', 'seedBits', 'seedInteger'}:
            raise ValueError('Invalid supplied event schema')
        key, bits, seed = event['key'], event['seedBits'], event['seedInteger']
        if type(bits) is not int or bits != _event_seed_bits(key):
            raise ValueError('Invalid key or seed width')
        try:
            payload = json.dumps(key, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=True, allow_nan=False).encode('utf-8')
        except (TypeError, ValueError) as exc:
            raise ValueError('Key is not canonical JSON') from exc
        digest = hashlib.sha256(payload).hexdigest()
        if (type(event['keySHA256']) is not str or event['keySHA256'] != digest or
                type(seed) is not str or not re.fullmatch(r'(?:0|[1-9][0-9]*)', seed) or
                int(seed) != int(digest[:bits // 4], 16)):
            raise ValueError('Key digest or integer seed mismatch')
        if payload in keys or digest in digests or int(seed) in seeds:
            raise ValueError('Duplicate key, digest or integer seed, including across widths')
        keys.add(payload)
        digests.add(digest)
        seeds.add(int(seed))
        records.append([digest, str(int(seed)), bits])
    records.sort()
    encoded = json.dumps(records, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    return {'eventCount': len(records), 'inventorySHA256': hashlib.sha256(encoded).hexdigest(),
            'scope': 'supplied_events_only', 'actualUseEstablished': False,
            'eventCompletenessEstablished': False, 'historicalPrefixEstablished': False,
            'independenceEstablished': False}


class _DiskEventAudit:
    """Producer inventory with disk-backed exact uniqueness, not authority."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE events (key_bytes BLOB UNIQUE NOT NULL, digest TEXT UNIQUE NOT NULL, '
                        'seed TEXT UNIQUE NOT NULL, bits INTEGER NOT NULL)')
        self.count = 0

    def add(self, event):
        if type(event) is not dict or set(event) != {'key', 'keySHA256', 'seedBits', 'seedInteger'}:
            raise ValueError('Invalid supplied event schema')
        key, bits, seed = event['key'], event['seedBits'], event['seedInteger']
        if type(bits) is not int or bits != _event_seed_bits(key):
            raise ValueError('Invalid key or seed width')
        try:
            raw = json.dumps(key, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                             allow_nan=False).encode('utf-8')
        except (TypeError, ValueError) as exc:
            raise ValueError('Key is not canonical JSON') from exc
        digest = hashlib.sha256(raw).hexdigest()
        if (type(event['keySHA256']) is not str or event['keySHA256'] != digest or
                type(seed) is not str or not re.fullmatch(r'(?:0|[1-9][0-9]*)', seed) or
                int(seed) != int(digest[:bits // 4], 16)):
            raise ValueError('Key digest or integer seed mismatch')
        try:
            self.db.execute('INSERT INTO events VALUES (?, ?, ?, ?)', (raw, digest, seed, bits))
        except sqlite3.IntegrityError as exc:
            raise ValueError('Duplicate key, digest or integer seed, including across widths') from exc
        self.count += 1

    def finish(self):
        self.db.commit()
        digest = hashlib.sha256()
        digest.update(b'[')
        for i, record in enumerate(self.db.execute('SELECT digest, seed, bits FROM events ORDER BY digest')):
            if i:
                digest.update(b',')
            digest.update(json.dumps(list(record), separators=(',', ':'), ensure_ascii=True).encode('ascii'))
        digest.update(b']')
        return {'eventCount': self.count, 'inventorySHA256': digest.hexdigest(),
                'scope': 'supplied_events_only', 'actualUseEstablished': False,
                'eventCompletenessEstablished': False, 'historicalPrefixEstablished': False,
                'independenceEstablished': False}

    def close(self):
        self.db.close()


def _future_frontier(root, replicate, result):
    """Check a trajectory's claimed frontier; this alone is not event admission."""
    if type(root) is not dict or type(replicate) is not int or not 2 <= replicate <= REPLICATES:
        raise ValueError('Future replicate and root required')
    variant, main, order = root.get('scheduleVariant'), root.get('mainIndex'), root.get('logicalToDisplayed')
    ordinal = coordinate(replicate, ['C5', variant, main])
    if (type(order) is not list or len(order) != 2 or
            any(type(x) is not str for x in order) or sorted(order) != ['A', 'B']):
        raise ValueError('Invalid display mapping')
    fields = {'schema', 'rootKey', 'replicateId', 'ordinal', 'inputSHA256', 'latent', 'stages',
              'terminalDecision', 'terminalStream', 'terminalIndex', 'finalState', 'endpoint',
              'contribution', 'authority', 'historicalEquivalence', 'scientificSpec'}
    if (type(result) is not dict or set(result) != fields or result['schema'] != 'c5-public-trajectory-v1' or
            not _same(result['rootKey'], ['C5', variant, main]) or
            type(result['replicateId']) is not int or result['replicateId'] != replicate or
            type(result['ordinal']) is not int or result['ordinal'] != ordinal or
            result['authority'] != 'caller-supplied-inputs-not-authenticated' or
            result['historicalEquivalence'] is not False):
        raise ValueError('Invalid trajectory envelope')
    events = []

    def add(event, role, logical=None, index=None):
        if type(event) is not dict or type(event.get('key')) is not dict:
            raise ValueError('Missing event key')
        key = event['key']
        width = _event_seed_bits(key)
        if role == 'terminal':
            identity = key.get('identity', {})
            if (width != 256 or identity.get('variant') != variant or identity.get('mainIndex') != main or
                    identity.get('replicateId') != replicate or
                    event.get('schema') != 'r14-c5-grouped-terminal-draw-v1'):
                raise ValueError('Terminal event belongs to another root or domain')
        elif (width != 128 or key['scheduleVariant'] != variant or key['mainIndex'] != main or
              key['replicateId'] != replicate or key['logicalRecipient'] != logical or
              key['streamKind'] != role or key['reportIndex'] != index or
              event.get('schema') != ('r14-c5-fixed-latent-partition-draw-v1' if role == 'latent'
                                      else 'r14-c5-prospective-report-partition-draw-v1')):
            raise ValueError('Private event belongs to another root or frontier')
        raw = json.dumps(key, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        if event.get('keySHA256') != digest or (width == 128 and event.get('seedHex') != digest[:32]):
            raise ValueError('Event key or actual seed mismatch')
        events.append({'key': key, 'keySHA256': digest, 'seedBits': width,
                       'seedInteger': str(int(digest[:width // 4], 16))})

    latent = result['latent']
    if type(latent) is not dict or set(latent) != {'A', 'B'}:
        raise ValueError('Exactly two logical latent events required')
    for logical in ('A', 'B'):
        add(latent[logical], 'latent', logical, 0)
    stages = result['stages']
    if type(stages) is not list:
        raise ValueError('Report stages must be a list')
    counts = {'A': 0, 'B': 0}
    for stage in stages:
        if (type(stage) is not dict or set(stage) != {'decision', 'report', 'paid', 'nextDecision'} or
                type(stage['decision']) is not dict or stage['decision'].get('action') not in ('ask_A', 'ask_B')):
            raise ValueError('Invalid report stage')
        logical = order[0 if stage['decision']['action'] == 'ask_A' else 1]
        add(stage['report'], 'report', logical, counts[logical])
        counts[logical] += 1
    decision = result['terminalDecision']
    if (type(decision) is not dict or type(decision.get('stop')) is not dict or
            type(decision.get('action')) is not str or decision['action'] != decision['stop'].get('action') or
            decision['action'] in ('ask_A', 'ask_B')):
        raise ValueError('Invalid terminal decision')
    if decision['action'] == 'random':
        add(result['terminalStream'], 'terminal')
    elif result['terminalStream'] is not None:
        raise ValueError('Terminal event requires random action')
    state = result['finalState']
    if (type(state) is not dict or type(state.get('queryIndex')) is not int or
            state['queryIndex'] != len(stages) or
            not _same(state.get('counts'), [counts[order[0]], counts[order[1]]])):
        raise ValueError('Final report frontier mismatch')
    return events


def admit_future_root(root, cfg, initial_decision, latent_config, report_bytes, design,
                      replicate, result_bytes, *, spec):
    """Independently replay one canonical future trajectory; no historical authority.

    The receiver is called directly, never accepted as a caller-provided PASS.
    Its fieldwise canonical byte comparisons cover the strict result envelope,
    but it does not emit an independently serialized full-result artifact.
    """
    if type(result_bytes) is not bytes:
        raise ValueError('Canonical producer result bytes required')
    try:
        result = json.loads(result_bytes)
        canonical = json.dumps(result, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=True, allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError('Invalid result JSON') from exc
    if result_bytes != canonical:
        raise ValueError('Result bytes must use canonical JSON')
    events = _future_frontier(root, replicate, result)
    inventory = audit_supplied_events(events)
    from . import c5_public_trajectory_reference
    receipt = c5_public_trajectory_reference.receive_root(
        root, cfg, initial_decision, latent_config, report_bytes, design, replicate, result, spec=spec)
    expected = {'status': 'independent_replay_pass', 'rootKey': result['rootKey'],
                'replicateId': replicate, 'inputSHA256': result['inputSHA256'],
                'authority': 'caller-supplied-inputs-not-authenticated', 'historicalEquivalence': False}
    if not _same(receipt, expected):
        raise ValueError('Independent receiver returned an unexpected receipt')
    needs = result['endpoint']['latentDisplayed']
    tie = result['endpoint']['status'] == 'exact_tie'
    row = {'replicateId': replicate, 'rootKey': result['rootKey'],
           'endpoint': {'rootKey': result['rootKey'], 'latentDisplayed': needs,
                        'latentLogical': {name: result['latent'][name]['category'] for name in ('A', 'B')},
                        'initialBudget': root['definition']['budgetMillihours'],
                        'remainingBudget': result['finalState']['budget'],
                        'queryCount': result['finalState']['queryIndex'],
                        'status': result['endpoint']['status'],
                        'reason': 'equal_true_needs' if tie else None, 'nonTie': not tie,
                        'higherNeedShare': result['endpoint']['higherNeedShare']},
           'design': {name: design[name]['inverse'][1] for name in TARGETS}}
    if not _same(contribution(row['design'], row['endpoint']), result['contribution']):
        raise ValueError('Trajectory and reduction contributions differ')
    from . import c5_final_reference
    c5_final_reference.check_trajectory_row(root, design, result, row)
    # Snapshot mutable input objects, so the returned association is stable.
    row = json.loads(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return {'rootKey': result['rootKey'], 'replicateId': replicate,
            'producerResultSHA256': hashlib.sha256(result_bytes).hexdigest(),
            'inputSHA256': result['inputSHA256'], 'events': events, 'eventInventory': inventory,
            'row': row,
            'independentReplay': 'direct_fieldwise_canonical_byte_comparison',
            'independentFullResultArtifactParityEstablished': False,
            'authority': 'caller-supplied-inputs-not-authenticated',
            'actualUseEstablished': False, 'historicalPrefixEstablished': False,
            'finalCertificateAuthorized': False}


def audit_population(prefix_path, prefix_sha256, *, repository_root, lookup, future, temp_dir=None):
    """Replay all future members and audit the supplied prefix and full roster.

    ``future`` yields keyword dictionaries for admit_future_root. No caller
    receipt, row or precomputed inventory is accepted. This is a provisional
    The complete lookup is checked against every receiver-admitted schedule;
    neither that consistency check nor a caller pin authenticates the lookup.
    Uses disk proportional to the input; temp_dir selects an existing scratch
    parent. All temporary databases are removed on success or failure. This is
    a provisional audit, never an inferential final.
    """
    from . import c5_final_reference
    with tempfile.TemporaryDirectory(prefix='c5-audit-', dir=temp_dir) as directory, ExitStack() as stack:
        producer = _DiskEventAudit(os.path.join(directory, 'events-p.sqlite'))
        stack.callback(producer.close)
        receiver = c5_final_reference.DiskEventChecker(os.path.join(directory, 'events-r.sqlite'))
        stack.callback(receiver.close)
        db = sqlite3.connect(os.path.join(directory, 'rows.sqlite'))
        stack.callback(db.close)
        db.execute('CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL, binding TEXT)')
        return _audit_population(prefix_path, prefix_sha256, repository_root, lookup, future,
                                 db, producer, receiver)


def _admitted_report_sha256(table, spec, report_width):
    """Bind the receiver's exact validated table, not a second byte object."""
    from . import c5_public_streams
    width = float(report_width).hex()
    if (type(table) is not c5_public_streams.ValidatedReportTable or
            table.width_hex != width or dict(spec.table).get(width) is not table or
            type(table.product_sha256) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', table.product_sha256)):
        raise ValueError('Receiver-admitted validated report table required')
    return table.product_sha256


def _audit_population(prefix_path, prefix_sha256, repository_root, lookup, future, db, producer, receiver):
    from . import c5_public_inputs, c5_final_reference, c5_public_streams
    if (type(lookup) is not dict or type(lookup.get('roots')) is not list or
            len(lookup['roots']) != ROOTS):
        raise ValueError('Complete lookup required')
    try:
        lookup = json.loads(json.dumps(lookup, sort_keys=True, separators=(',', ':'), allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError('Lookup must be JSON') from exc
    prefix = c5_public_inputs.load_historical_prefix(prefix_path, prefix_sha256,
                                                     repository_root=repository_root, lookup=lookup)
    lookup_rows = tuple(json.dumps(item, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode() for item in lookup['roots'])
    seen, geometry, root_bindings, count = bytearray(POPULATION), {}, {}, 0
    lookup_hash = None
    checked_schedules = set()
    fields = {'root', 'cfg', 'initial_decision', 'latent_config', 'report_bytes',
              'design', 'replicate', 'result_bytes', 'spec'}
    for args in future:
        if type(args) is not dict or set(args) != fields:
            raise ValueError('A complete future replay input is required')
        admitted = admit_future_root(**args)
        row = admitted['row']
        index = coordinate(row['replicateId'], row['rootKey'])
        variant = row['rootKey'][1]
        spec = args['spec']
        try:
            spec_rows = spec.rows
            spec_binding = json.loads(spec.binding_bytes)
            spec_hash = spec_binding['lookupSHA256']
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            raise ValueError('Invalid receiver lookup snapshot') from exc
        if (type(spec_hash) is not str or not re.fullmatch(r'[0-9a-f]{64}', spec_hash) or
                type(spec_rows) is not tuple or
                spec_rows != lookup_rows[10 * variant:10 * (variant + 1)] or
                (lookup_hash is not None and spec_hash != lookup_hash)):
            raise ValueError('Lookup differs from independently replayed receiver snapshot')
        lookup_hash = spec_hash
        checked_schedules.add(variant)
        if seen[index]:
            raise ValueError('Duplicate future member')
        seen[index] = 1
        key = tuple(row['rootKey'])
        if key in geometry and not _same(geometry[key], row['design']):
            raise ValueError('Future geometry changed')
        geometry[key] = row['design']
        root, cfg, latent = args['root'], args['cfg'], args['latent_config']
        fixed = lookup['roots'][10 * key[1] + key[2] - 1]
        if (type(latent) is not c5_public_streams.ValidatedLatentConfig or
                not _same(fixed['rootKey'], row['rootKey']) or
                c5_public_inputs._digest(root) != fixed['rootSHA256'] or
                c5_public_inputs._digest(cfg) != fixed['configSHA256'] or
                not _same(cfg, fixed['config']) or
                not _same(root['logicalToDisplayed'], fixed['logicalToDisplayed']) or
                root['definition']['reportWidth'] != fixed['reportWidth']):
            raise ValueError('Replayed root differs from fixed lookup or validated inputs')
        binding = {'rootKey': row['rootKey'], 'rootSHA256': fixed['rootSHA256'],
                   'configSHA256': fixed['configSHA256'],
                   'logicalToDisplayed': fixed['logicalToDisplayed'],
                   'reportWidth': fixed['reportWidth'],
                   'budget': row['endpoint']['initialBudget'], 'design': row['design'],
                   'latentConfigSHA256': latent.input_sha256,
                   'latentCounts': [list(law[0]) for law in latent.roots[10 * key[1] + key[2] - 1]],
                   'reportProductSHA256': _admitted_report_sha256(args['report_bytes'], spec,
                                                                  fixed['reportWidth'])}
        if key in root_bindings and not _same(root_bindings[key], binding):
            raise ValueError('Replayed root input binding changed across repetitions')
        root_bindings[key] = binding
        payload = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
        binding = [index, admitted['producerResultSHA256'], admitted['inputSHA256'],
                   hashlib.sha256(payload.encode()).hexdigest(), admitted['eventInventory']['inventorySHA256']]
        db.execute('INSERT INTO rows VALUES (?, ?, ?)',
                   (index, payload, json.dumps(binding, separators=(',', ':'), ensure_ascii=True)))
        count += 1
        for event in admitted['events']:
            producer.add(event)
            receiver.add(event)
    if (count != POPULATION - ROOTS or not all(seen[ROOTS:]) or len(geometry) != ROOTS or
            checked_schedules != set(range(9))):
        raise ValueError('Complete future roster required')
    for item in prefix['payload']['rows']:
        root = item['rootKey']
        index = coordinate(1, root)
        if seen[index] or tuple(root) not in geometry:
            raise ValueError('Invalid prefix roster')
        seen[index] = 1
        row = {'replicateId': 1, 'rootKey': root, 'endpoint': item['endpoint'],
               'design': geometry[tuple(root)]}
        if not _same(contribution(row['design'], row['endpoint']), item['contribution']['values']):
            raise ValueError('Prefix contribution differs from geometry and endpoint')
        db.execute('INSERT INTO rows VALUES (?, ?, NULL)',
                   (index, json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                      allow_nan=False)))
        for event in item['usedSeeds']:
            producer.add(event)
            receiver.add(event)
    if not all(seen):
        raise ValueError('Complete population required')
    db.commit()
    inventory = producer.finish()
    receiver.check(inventory)
    def rows():
        for (payload,) in db.execute('SELECT payload FROM rows ORDER BY ordinal'):
            yield json.loads(payload)
    reduced = reduce_final(rows(), input_kind='fresh')
    checked = c5_final_reference.check_final(rows(), reduced, input_kind='fresh')
    row_digest = hashlib.sha256()
    row_digest.update(b'[')
    separator = b''
    for (payload,) in db.execute('SELECT payload FROM rows ORDER BY ordinal'):
        row_digest.update(separator)
        row_digest.update(payload.encode('ascii'))
        separator = b','
    row_digest.update(b']')
    rows_sha = row_digest.hexdigest()
    if checked['orderedRowsSHA256'] != rows_sha:
        raise ValueError('Independent population row binding mismatch')
    digest = hashlib.sha256()
    digest.update(b'[')
    separator = b''
    for (binding,) in db.execute('SELECT binding FROM rows WHERE binding IS NOT NULL ORDER BY ordinal'):
        digest.update(separator)
        digest.update(binding.encode('ascii'))
        separator = b','
    digest.update(b']')
    binding_sha = digest.hexdigest()
    return {'population': POPULATION, 'prefixContentSHA256': prefix['contentSHA256'],
            'lookupSHA256': lookup_hash, 'lookupAuthenticationEstablished': False,
            'rootBindings': [root_bindings[('C5', v, m)] for v in range(9) for m in range(1, 11)],
            'prefixMappingCheck': 'receiver_lookup_consistency_only',
            'futureRowEventBindingsSHA256': binding_sha, 'eventInventory': inventory,
            'orderedPopulationRowsSHA256': rows_sha,
            'independentPopulationRowBinding': True,
            'independentEventAudit': True, 'arithmetic': reduced,
            'historicalPrefixEstablished': False, 'finalCertificateAuthorized': False,
            'scope': 'provisional_replay_and_supplied_prefix_not_certified_final'}


def certify_supplied_geometry(designs, summary):
    """Check the two-target bounded-prefix planning math on supplied geometry.

    Neither the matrices nor the summary acquire scientific authority here.
    The independent verifier checks both adjacent inequalities and the family.
    """
    from . import c5_precision_planning, c5_precision_reference, c5_final_reference
    certificates = {}
    for name in TARGETS:
        geometry = c5_precision_reference.geometry(designs, summary, name)
        certificate = c5_precision_planning.bounded_prefix_certificate(
            geometry['radii'], geometry['varianceSumUpper'], geometry['maximumRange'])
        certificates[name] = certificate
    result = {'targets': certificates,
              'familyReplicates': max(certificates[name]['totalReplicates'] for name in TARGETS),
              'halfWidth': ['1', '10'], 'alpha': ['1', '20'], 'inferenceLooks': 1,
              'scope': 'supplied_geometry_only_not_input_authority_or_final'}
    c5_final_reference.check_supplied_geometry(designs, summary, result)
    return result


def issue_certified_final(*args, **kwargs):
    """Fail closed pending independently pinned external authority material.

    Required authority contract: an independently published digest binding the
    exported rep1 scientific bytes to its accepted 90-root source and exact
    export schema; and an independently authenticated pin for the actual fixed
    lookup export, bound to the accepted canonical input bytes identified by
    c5_public_inputs.INPUT_SHA256 and all 90 root/config/decision/display
    identities. That accepted source contains the high-price no-question facts,
    recipient laws and mappings; the exported fields must be checked against
    those source bytes, and equal recipient integer laws must be checked rather
    than inferred from a caller-supplied flag. The actual export pin is not yet
    available here. An accepted numerical-profile binding is also required.
    These anchors must be authenticated independently of the caller and bound
    to the population replay, latent laws and design, not supplied as PASS flags
    or a digest alongside arbitrary input. A supplied-geometry certificate and
    key noncollision are not substitutes for these anchors or the conditional
    root-by-replicate independence assumption.
    """
    raise ValueError('Certified final unavailable: authenticated rep1 export/source pin, '
                     'actual fixed-lookup export pin and source/law bindings, and '
                     'numerical-profile binding are not implemented')
