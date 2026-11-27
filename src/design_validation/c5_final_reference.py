"""Independent balanced-sum checker for caller-supplied fixed C5 operands."""
from fractions import Fraction
import hashlib
import json
import math
import re
import sqlite3

_INT = re.compile(r'(?:0|-?[1-9][0-9]*)\Z')
_TOTAL = 165600
_MODELS = ('gap', 'partial_low')
_FIELDS = {'rootKey', 'latentDisplayed', 'latentLogical', 'initialBudget',
           'remainingBudget', 'queryCount', 'status', 'reason', 'nonTie',
           'higherNeedShare'}


def _identical(left, right):
    """Exact schema and types, including bool versus int at every depth."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_identical(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_identical(a, b) for a, b in zip(left, right))
    return left == right


def _read(pair):
    if (type(pair) is not list or len(pair) != 2 or
            any(type(x) is not str or not _INT.fullmatch(x) for x in pair)):
        raise ValueError('Invalid rational')
    a, b = map(int, pair)
    if b <= 0:
        raise ValueError('Invalid denominator')
    q = Fraction(a, b)
    if [str(q.numerator), str(q.denominator)] != pair:
        raise ValueError('Noncanonical rational')
    return q


def _pair(q):
    return [str(q.numerator), str(q.denominator)]


class BalancedSum:
    def __init__(self):
        self.levels = []
        self.count = 0

    def add(self, value):
        if type(value) is not Fraction:
            raise ValueError('Exact fraction required')
        self.count += 1
        i = 0
        while i < len(self.levels) and self.levels[i] is not None:
            value += self.levels[i]
            self.levels[i] = None
            i += 1
        if i == len(self.levels):
            self.levels.append(value)
        else:
            self.levels[i] = value

    def finish(self):
        if self.count != _TOTAL:
            raise ValueError('Incomplete population')
        return sum((x for x in reversed(self.levels) if x is not None), Fraction())


def _independent_values(design, endpoint):
    if (type(design) is not dict or set(design) != set(_MODELS) or
            type(endpoint) is not dict or set(endpoint) != _FIELDS):
        raise ValueError('Missing design or endpoint')
    needs, logical = endpoint.get('latentDisplayed'), endpoint.get('latentLogical')
    budget, remaining, queries = (endpoint.get('initialBudget'), endpoint.get('remainingBudget'),
                                  endpoint.get('queryCount'))
    if (type(needs) is not list or len(needs) != 2 or
            any(type(x) is not int or not 1 <= x <= 100 for x in needs) or
            type(logical) is not dict or set(logical) != {'A', 'B'} or
            any(type(x) is not int or not 1 <= x <= 100 for x in logical.values()) or
            sorted(logical.values()) != sorted(needs) or
            type(budget) is not int or budget <= 0 or type(remaining) is not int or
            remaining <= 0 or remaining > budget or type(queries) is not int or queries < 0):
        raise ValueError('Invalid endpoint support')
    g, p = design['gap'], design['partial_low']
    if type(g) is not list or len(g) != 2 or type(p) is not list or len(p) != 3:
        raise ValueError('Invalid inverse rows')
    g, p = [_read(x) for x in g], [_read(x) for x in p]
    low, high = sorted(needs)
    if low == high:
        if (endpoint.get('status') != 'exact_tie' or endpoint.get('nonTie') is not False or
                endpoint.get('reason') != 'equal_true_needs' or endpoint.get('higherNeedShare') is not None):
            raise ValueError('Invalid tie')
        return {name: Fraction() for name in _MODELS}
    if endpoint.get('status') != 'complete' or endpoint.get('nonTie') is not True or endpoint.get('reason') is not None:
        raise ValueError('Unavailable non-tie')
    share = _read(endpoint.get('higherNeedShare'))
    if not 0 <= share <= 1:
        raise ValueError('Invalid share')
    scale = Fraction(1000, budget)
    centered = share - Fraction(1, 2)
    return {'gap': (g[0] + g[1] * (high - low) * scale) * centered,
            'partial_low': (p[0] + p[1] * low * scale + p[2] * high * scale) * centered}


def check_trajectory_row(root, design, trajectory, row):
    """Check the replayed trajectory's reduction adapter, not its provenance."""
    logical = {name: trajectory['latent'][name]['category'] for name in ('A', 'B')}
    displayed = [logical[name] for name in root['logicalToDisplayed']]
    tie = displayed[0] == displayed[1]
    share = trajectory['endpoint']['higherNeedShare']
    expected = {'replicateId': trajectory['replicateId'], 'rootKey': trajectory['rootKey'],
                'endpoint': {'rootKey': trajectory['rootKey'], 'latentDisplayed': displayed,
                             'latentLogical': logical,
                             'initialBudget': root['definition']['budgetMillihours'],
                             'remainingBudget': trajectory['finalState']['budget'],
                             'queryCount': trajectory['finalState']['queryIndex'],
                             'status': 'exact_tie' if tie else 'complete',
                             'reason': 'equal_true_needs' if tie else None,
                             'nonTie': not tie, 'higherNeedShare': share},
                'design': {name: design[name]['inverse'][1] for name in _MODELS}}
    if (not _identical(trajectory['endpoint']['latentDisplayed'], displayed) or
            trajectory['endpoint']['status'] != expected['endpoint']['status'] or
            not _identical(row, expected)):
        raise ValueError('Independent trajectory row mismatch')
    values = _independent_values(expected['design'], expected['endpoint'])
    if not _identical(trajectory['contribution'], {k: _pair(v) for k, v in values.items()}):
        raise ValueError('Independent trajectory contribution mismatch')
    return {'passed': True, 'scope': 'replayed_trajectory_row_only'}


def check_final(rows, result, *, input_kind):
    """Check arithmetic and hash ordered operands, not provenance or parity.

    The digest covers a compact canonical JSON array in the caller's row order;
    the population audit supplies scientific ordinal order. It is a content
    binding, not evidence that the supplied rows are authentic or independent.
    """
    if input_kind not in ('fresh', 'saved'):
        raise ValueError('Explicit input route required')
    seen = bytearray(_TOTAL)
    sums = {name: BalancedSum() for name in _MODELS}
    count = ties = 0
    row_digest = hashlib.sha256()
    row_digest.update(b'[')
    fixed_roots = {}
    fields = {'replicateId', 'rootKey', 'endpoint', 'design'}
    if input_kind == 'saved':
        fields = fields | {'contributions'}
    for row in rows:
        if type(row) is not dict or set(row) != fields:
            raise ValueError('Invalid row schema')
        if count:
            row_digest.update(b',')
        try:
            row_digest.update(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True, allow_nan=False).encode('ascii'))
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid row serialization') from exc
        rep, root = row['replicateId'], row['rootKey']
        if (type(rep) is not int or not 1 <= rep <= 1840 or type(root) is not list or
                len(root) != 3 or root[0] != 'C5' or type(root[1]) is not int or
                not 0 <= root[1] <= 8 or type(root[2]) is not int or not 1 <= root[2] <= 10):
            raise ValueError('Invalid member coordinate')
        index = (rep - 1) * 90 + root[1] * 10 + root[2] - 1
        if (seen[index] or type(row['endpoint']) is not dict or
                not _identical(row['endpoint'].get('rootKey'), root)):
            raise ValueError('Duplicate member or endpoint mismatch')
        values = _independent_values(row['design'], row['endpoint'])
        if input_kind == 'saved' and not _identical(row['contributions'], {k: _pair(v) for k, v in values.items()}):
            raise ValueError('Saved values do not match operands')
        root_id = (root[1], root[2])
        geometry = (tuple(_read(x) for x in row['design']['gap']),
                    tuple(_read(x) for x in row['design']['partial_low']),
                    row['endpoint']['initialBudget'])
        if root_id in fixed_roots and fixed_roots[root_id] != geometry:
            raise ValueError('Changed fixed root geometry')
        fixed_roots[root_id] = geometry
        seen[index] = 1
        count += 1
        ties += int(row['endpoint']['status'] == 'exact_tie')
        for name in _MODELS:
            sums[name].add(values[name])
    if count != _TOTAL or not all(seen):
        raise ValueError('Incomplete fixed roster')
    results = {}
    for name in _MODELS:
        total = sums[name].finish()
        mean = total / _TOTAL
        results[name] = {'total': _pair(total), 'denominator': _TOTAL,
                         'lowPriceEstimate': _pair(mean)}
    expected = {'inputKind': input_kind, 'historicalRawParityEstablished': False,
                'endpointParityEstablished': False,
                'population': _TOTAL, 'exactTies': ties, 'results': results,
                'scope': 'arithmetic_on_supplied_operands_only'}
    if not _identical(result, expected):
        raise ValueError('Independent final summary mismatch')
    row_digest.update(b']')
    return {'passed': True, 'inputKind': input_kind, 'historicalRawParityEstablished': False,
            'endpointParityEstablished': False, 'orderedRowsSHA256': row_digest.hexdigest(),
            'scope': 'arithmetic_on_supplied_operands_only'}


def check_prefix_bound(root_bounds, variance, maximum, log_upper, result, *, n=1840):
    """Check the original sum-tail inequality without producer bound code."""
    if type(n) is not int or n < 2 or type(root_bounds) is not list or len(root_bounds) != 90:
        raise ValueError('Invalid bound dimensions')
    m = [_read(x) for x in root_bounds]
    v, b, ell = (_read(x) for x in (variance, maximum, log_upper))
    if any(x < 0 for x in m) or v < 0 or b < max(m) or ell <= 0:
        raise ValueError('Invalid bound operands')
    delta = sum(m, Fraction()) / (90 * n)
    e = Fraction(1, 10) - delta
    lhs = (90 * n * e) ** 2
    rhs = 2 * (n * v + b * 90 * n * e / 3) * ell
    expected = {'n': n, 'delta': _pair(delta), 'remainingRadius': _pair(e),
                'lhs': _pair(lhs / n), 'rhs': _pair(rhs / n),
                'inequalitySatisfied': e > 0 and lhs >= rhs,
                'scope': 'arithmetic_only_not_certificate_supplied_geometry_and_log_unverified'}
    if not _identical(result, expected):
        raise ValueError('Independent bound mismatch')
    return {'passed': True, 'inequalitySatisfied': expected['inequalitySatisfied'],
            'scope': 'arithmetic_only_not_certificate_supplied_geometry_and_log_unverified'}


def check_supplied_geometry(designs, summary, result):
    """Independently verify the bounded-prefix family on supplied designs only."""
    from . import c5_precision_reference as precision, c5_prefix_reference as prefix
    fields = {'targets', 'familyReplicates', 'halfWidth', 'alpha', 'inferenceLooks', 'scope'}
    if (type(result) is not dict or set(result) != fields or
            not _identical(result['halfWidth'], ['1', '10']) or
            not _identical(result['alpha'], ['1', '20']) or
            type(result['inferenceLooks']) is not int or result['inferenceLooks'] != 1 or
            result['scope'] != 'supplied_geometry_only_not_input_authority_or_final' or
            type(result['targets']) is not dict or set(result['targets']) != set(_MODELS)):
        raise ValueError('Invalid two-target single-look certificate')
    counts = []
    for name in _MODELS:
        item = result['targets'][name]
        if (type(item) is not dict or set(item) != {'totalReplicates', 'futureCanonicalReplicates',
                'currentPassesUpper', 'previousFailsLower', 'logBounds'} or
                type(item['totalReplicates']) is not int or item['totalReplicates'] < 2 or
                type(item['futureCanonicalReplicates']) is not int or
                item['futureCanonicalReplicates'] != item['totalReplicates'] - 1 or
                item['currentPassesUpper'] is not True or item['previousFailsLower'] is not True or
                type(item['logBounds']) is not list or len(item['logBounds']) != 2):
            raise ValueError('Invalid target certificate envelope')
        lower, upper = map(_read, item['logBounds'])
        independent_lower, independent_upper = precision.prime_log_interval(80, 256)
        if not 0 < lower <= independent_lower < independent_upper <= upper:
            raise ValueError('Producer logarithm does not enclose independent log bounds')
        geometry = precision.geometry(designs, summary, name)
        prefix.verify(geometry['radii'], geometry['varianceSumUpper'], geometry['maximumRange'],
                      item['totalReplicates'])
        counts.append(item['totalReplicates'])
    if (type(result['familyReplicates']) is not int or
            result['familyReplicates'] != max(counts) or result['familyReplicates'] != 1840):
        raise ValueError('Expected the fixed-1840 two-target family')
    return {'passed': True, 'scope': 'supplied_geometry_only_not_input_authority_or_final',
            'familyReplicates': 1840, 'rootReplicateIndependenceIsAssumption': True}


def _valid_typed_value(node):
    if type(node) is not dict or len(node) != 1:
        return False
    if 'null' in node:
        return node['null'] is True
    if 'bool' in node:
        return type(node['bool']) is bool
    if 'int' in node:
        return type(node['int']) is str and _INT.fullmatch(node['int']) is not None
    if 'str' in node:
        return type(node['str']) is str
    if 'floatHex' in node:
        text = node['floatHex']
        if type(text) is not str:
            return False
        try:
            value = float.fromhex(text)
        except (ValueError, OverflowError):
            return False
        return not math.isnan(value) and value != math.inf and value.hex() == text
    if 'list' in node:
        values = node['list']
        return type(values) is list and all(_valid_typed_value(v) for v in values)
    if 'dict' in node:
        entries = node['dict']
        if type(entries) is not list:
            return False
        previous = None
        for entry in entries:
            if (type(entry) is not list or len(entry) != 2 or type(entry[0]) is not str or
                    (previous is not None and entry[0] <= previous) or not _valid_typed_value(entry[1])):
                return False
            previous = entry[0]
        return True
    return False


def _registered_width(key):
    if type(key) is not dict:
        raise ValueError('Invalid event key')
    private = {'streamVersion', 'populationId', 'seed', 'randomizationGroup', 'conditionId',
               'scheduleVariant', 'replicateId', 'mainIndex', 'logicalRecipient', 'streamKind', 'reportIndex'}
    if set(key) == private:
        constants = {'streamVersion': 'r10-private-need-report-v1',
                     'populationId': 'r10-working-five-v0-rep0-candidate-not-population-estimate',
                     'randomizationGroup': 'r14-c5-paired-projection-private-stream-v1', 'conditionId': 'C5'}
        if (any(type(key[k]) is not str or key[k] != v for k, v in constants.items()) or
                type(key['seed']) is not int or key['seed'] != 20261019 or
                any(type(key[k]) is not int or not lo <= key[k] <= hi for k, lo, hi in
                    (('scheduleVariant', 0, 8), ('replicateId', 1, 1840), ('mainIndex', 1, 10))) or
                type(key['logicalRecipient']) is not str or key['logicalRecipient'] not in ('A', 'B') or
                type(key['streamKind']) is not str or key['streamKind'] not in ('latent', 'report') or
                type(key['reportIndex']) is not int or key['reportIndex'] < 0 or
                (key['streamKind'] == 'latent' and key['reportIndex'] != 0)):
            raise ValueError('Invalid private domain')
        return 128
    if set(key) != {'version', 'randomizationGroup', 'identity', 'candidate', 'keyEncoding'}:
        raise ValueError('Unknown event domain')
    constants = {'version': 'r10-native-terminal-pcg64-sha256-v1',
                 'randomizationGroup': 'r14-c5-paired-projection-terminal-v1',
                 'keyEncoding': 'typed-json-floathex-v1'}
    ident = key['identity']
    if (any(type(key[k]) is not str or key[k] != v for k, v in constants.items()) or
            type(ident) is not dict or set(ident) != {'seed', 'modelId', 'personId', 'trialId',
                                                     'conditionId', 'variant', 'replicateId', 'mainIndex'} or
            type(ident['seed']) is not int or ident['seed'] != 20261019 or ident['conditionId'] != 'C5' or
            any(type(ident[k]) is not int or not lo <= ident[k] <= hi for k, lo, hi in
                (('variant', 0, 8), ('replicateId', 1, 1840), ('mainIndex', 1, 10))) or
            any(type(ident[k]) is not str or not ident[k] for k in ('modelId', 'personId', 'trialId')) or
            type(key['candidate']) is not dict or set(key['candidate']) != {'dict'} or
            not _valid_typed_value(key['candidate'])):
        raise ValueError('Invalid terminal domain')
    return 256


def check_supplied_events(events, result):
    """Independently check the supplied inventory, not actual-use completeness."""
    by_key, by_hash, by_seed, inventory = {}, {}, {}, []
    for event in events:
        if type(event) is not dict or set(event) != {'key', 'keySHA256', 'seedBits', 'seedInteger'}:
            raise ValueError('Invalid event')
        key, bits, seed = event['key'], event['seedBits'], event['seedInteger']
        if (type(bits) is not int or bits != _registered_width(key) or
                type(seed) is not str or not re.fullmatch(r'(?:0|[1-9][0-9]*)', seed)):
            raise ValueError('Invalid event key or seed')
        try:
            raw = json.dumps(key, ensure_ascii=True, sort_keys=True, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid key JSON') from exc
        full = hashlib.sha256(raw).hexdigest()
        number = int(seed)
        if type(event['keySHA256']) is not str or event['keySHA256'] != full:
            raise ValueError('Key digest mismatch')
        if number != int(full[:32] if bits == 128 else full, 16):
            raise ValueError('Wrong integer seed')
        if raw in by_key or full in by_hash or number in by_seed:
            raise ValueError('Repeated key, digest or integer seed')
        by_key[raw], by_hash[full], by_seed[number] = True, True, True
        inventory.append([full, str(number), bits])
    inventory.sort()
    raw_inventory = json.dumps(inventory, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    expected = {'eventCount': len(inventory),
                'inventorySHA256': hashlib.sha256(raw_inventory).hexdigest(),
                'scope': 'supplied_events_only', 'actualUseEstablished': False,
                'eventCompletenessEstablished': False, 'historicalPrefixEstablished': False,
                'independenceEstablished': False}
    if not _identical(result, expected):
        raise ValueError('Independent event inventory mismatch')
    return {'passed': True, 'scope': 'supplied_events_only',
            'actualUseEstablished': False, 'eventCompletenessEstablished': False}


class DiskEventChecker:
    """Independent on-disk checker; seed uniqueness spans both widths."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE used (key BLOB PRIMARY KEY, sha TEXT UNIQUE NOT NULL, '
                        'seed BLOB UNIQUE NOT NULL, seed_text TEXT NOT NULL, width INTEGER NOT NULL)')
        self.count = 0

    def add(self, event):
        if type(event) is not dict or set(event) != {'key', 'keySHA256', 'seedBits', 'seedInteger'}:
            raise ValueError('Invalid event')
        key, bits, seed = event['key'], event['seedBits'], event['seedInteger']
        if (type(bits) is not int or bits != _registered_width(key) or
                type(seed) is not str or not re.fullmatch(r'(?:0|[1-9][0-9]*)', seed)):
            raise ValueError('Invalid event key or seed')
        try:
            raw = json.dumps(key, ensure_ascii=True, sort_keys=True, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid key JSON') from exc
        full = hashlib.sha256(raw).hexdigest()
        number = int(seed)
        if (type(event['keySHA256']) is not str or event['keySHA256'] != full or
                number != int(full[:32] if bits == 128 else full, 16)):
            raise ValueError('Invalid digest or integer seed')
        try:
            self.db.execute('INSERT INTO used VALUES (?, ?, ?, ?, ?)',
                            (raw, full, number.to_bytes(32, 'big'), seed, bits))
        except sqlite3.IntegrityError as exc:
            raise ValueError('Repeated key, digest or integer seed') from exc
        self.count += 1

    def check(self, result):
        self.db.commit()
        digest = hashlib.sha256()
        digest.update(b'[')
        separator = b''
        for full, seed, width in self.db.execute('SELECT sha, seed_text, width FROM used ORDER BY sha'):
            digest.update(separator)
            digest.update(json.dumps([full, seed, width], ensure_ascii=True, separators=(',', ':')).encode('ascii'))
            separator = b','
        digest.update(b']')
        expected = {'eventCount': self.count, 'inventorySHA256': digest.hexdigest(),
                    'scope': 'supplied_events_only', 'actualUseEstablished': False,
                    'eventCompletenessEstablished': False, 'historicalPrefixEstablished': False,
                    'independenceEstablished': False}
        if not _identical(result, expected):
            raise ValueError('Independent event inventory mismatch')
        return {'passed': True, 'scope': 'supplied_events_only',
                'actualUseEstablished': False, 'eventCompletenessEstablished': False}

    def close(self):
        self.db.close()
