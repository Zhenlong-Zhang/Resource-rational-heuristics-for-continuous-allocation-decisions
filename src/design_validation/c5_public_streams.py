"""C5 fixed-1840 typed streams; accepted integer laws are explicit inputs.

This module does not reconstruct historical dispatches or authenticate inputs.
NumPy PCG64 and the supplied candidate/identity remain part of the numerical
identity; matching a stream is not a cross-platform policy-equivalence claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from . import personal_time_sensitivity_streams as original
from . import canonical_latent_reference
from . import canonical_report_reference

GRID = 1 << 53
POPULATION = 'r10-working-five-v0-rep0-candidate-not-population-estimate'
GROUP = 'r14-c5-paired-projection-private-stream-v1'
TERMINAL = 'r14-c5-paired-projection-terminal-v1'
ACCEPTED_REPORT_BYTES = {
    '0x1.9333333333334p+1': 'ba563fd341874d99c0a74df66a2dee4c7aea19d7eb52ce64b7569dccfebba85c',
    '0x1.6ccccccccccccp+1': 'f85a6c3b83cca15bc83b8b35c57754a3a43beeb59f31cb0fd4128d2d4453d95c',
    '0x1.8000000000000p+1': 'e3a4632ba2924afae5df9be893c9c2e3b209e8250d31c9c386d6c3e922728bb6',
}


def private_key(variant, main, replicate, logical, kind, index):
    if not (type(variant) is int and 0 <= variant < 9 and
            type(main) is int and 1 <= main <= 10 and
            type(replicate) is int and 1 <= replicate <= 1840 and
            logical in ('A', 'B') and kind in ('latent', 'report') and
            type(index) is int and index >= 0 and (kind != 'latent' or index == 0)):
        raise ValueError('Invalid fixed-1840 private coordinate')
    ctx = {'identity': {'seed': 20261019, 'conditionId': 'C5', 'variant': variant,
                        'replicateId': replicate, 'mainIndex': main},
           'bindings': {'populationId': POPULATION, 'randomizationGroup': GROUP}}
    return original.make_key(ctx, logical, kind, index)


def _hash(key):
    return hashlib.sha256(original.payload(key)).hexdigest()


def accepted_index(law, integer_uniform):
    if type(integer_uniform) is not int or not 0 <= integer_uniform < GRID:
        raise ValueError('Expected a 53-bit integer uniform')
    counts = law['counts']
    if (law['denominator'] != GRID or len(counts) != 100 or
            any(type(x) is not int or x < 0 for x in counts) or sum(counts) != GRID):
        raise ValueError('Invalid accepted integer law')
    bound = 0
    for category, count in enumerate(counts, 1):
        bound += count
        if integer_uniform < bound:
            return category
    raise ValueError('Uncovered integer uniform')


def accepted_uniform(law, uniform):
    if type(uniform) is not float or not 0.0 <= uniform < 1.0:
        raise ValueError('Expected binary64 uniform in [0, 1)')
    numerator, denominator = uniform.as_integer_ratio()
    integer, remainder = divmod(numerator * GRID, denominator)
    if remainder:
        raise ValueError('Uniform is not on the 53-bit grid')
    return accepted_index(law, integer)


def _validate_private_key(key):
    try:
        expected = private_key(key['scheduleVariant'], key['mainIndex'], key['replicateId'],
                               key['logicalRecipient'], key['streamKind'], key['reportIndex'])
        matches = type(key) is dict and original.payload(key) == original.payload(expected)
    except (KeyError, TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError('Private key does not match fixed-1840 domain')


@dataclass(frozen=True, init=False)
class ValidatedLatentConfig:
    """Immutable snapshot of a completely checked fixed scientific input.

    Validation establishes internal consistency, not historical authentication.
    """
    roots: tuple
    input_sha256: str

    def __init__(self, config):
        roots = _validate_latent_config(config)
        object.__setattr__(self, 'roots', roots)
        object.__setattr__(self, 'input_sha256', _hash(config))


@dataclass(frozen=True, init=False)
class ValidatedReportTable:
    """Prospective complete table verified in this numerical environment.

    Accepted status applies only to matching the published product bytes.
    """
    width_hex: str
    product_sha256: str
    accepted_product_bytes: bool
    rows: tuple

    def __init__(self, product_bytes, width_hex):
        if type(product_bytes) is not bytes:
            raise ValueError('Exact report product bytes required')
        try:
            table = json.loads(product_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError('Invalid report product JSON') from exc
        canonical_report_reference.verify_report_table(table, width_hex)
        digest = hashlib.sha256(product_bytes).hexdigest()
        rows = tuple((tuple(row['counts']), tuple(row['cumulativeHex'])) for row in table['rows'])
        object.__setattr__(self, 'width_hex', width_hex)
        object.__setattr__(self, 'product_sha256', digest)
        object.__setattr__(self, 'accepted_product_bytes', digest == ACCEPTED_REPORT_BYTES.get(width_hex))
        object.__setattr__(self, 'rows', rows)


def _validate_latent_config(config):
    fields = {'schema', 'denominator', 'recipientOrder', 'lastCategoryIncludesFallback',
              'identity', 'roots'}
    if (type(config) is not dict or set(config) != fields or
            config['schema'] != 'fixed-c5-latent-53bit-v1' or
            type(config['denominator']) is not int or config['denominator'] != GRID or
            config['recipientOrder'] != ['A', 'B'] or
            config['lastCategoryIncludesFallback'] is not True or
            config['identity'] != 'adopted-fixed-integer-scientific-input-not-fresh-exp-log' or
            type(config['roots']) is not list or len(config['roots']) != 90):
        raise ValueError('Invalid fixed latent configuration')
    roots = []
    for position, root in enumerate(config['roots']):
        variant, main = divmod(position, 10)
        if (type(root) is not dict or set(root) != {'rootKey', 'recipientLaws'} or
                type(root['rootKey']) is not list or
                original.payload(root['rootKey']) != original.payload(['C5', variant, main + 1]) or
                type(root['recipientLaws']) is not list or len(root['recipientLaws']) != 2):
            raise ValueError('Invalid fixed latent root roster')
        laws = []
        for law in root['recipientLaws']:
            if type(law) is not dict or set(law) != {'counts', 'cumulativeHex'}:
                raise ValueError('Invalid fixed latent law schema')
            canonical_latent_reference.verify_historical_cdf(law['counts'], law['cumulativeHex'])
            laws.append((tuple(law['counts']), tuple(law['cumulativeHex'])))
        roots.append(tuple(laws))
    return tuple(roots)


def latent_partition(config, key):
    """Select a validated partition using the complete fixed latent key."""
    _validate_private_key(key)
    if key['streamKind'] != 'latent' or type(config) is not ValidatedLatentConfig:
        raise ValueError('A validated fixed latent configuration and latent key are required')
    counts, cumulative = config.roots[key['scheduleVariant'] * 10 + key['mainIndex'] - 1][
        ('A', 'B').index(key['logicalRecipient'])]
    return {'denominator': GRID, 'counts': counts, 'cumulativeHex': cumulative}


def private_draw(key, law):
    _validate_private_key(key)
    if key['streamKind'] != 'latent':
        raise ValueError('Report draws unavailable: validated width, table and latent binding required')
    law_config_sha = law.input_sha256 if type(law) is ValidatedLatentConfig else None
    law = latent_partition(law, key)
    digest = _hash(key)
    seed = digest[:32]
    uniform = float(np.random.Generator(np.random.PCG64(int(seed, 16))).random())
    chosen = accepted_uniform(law, uniform)
    cumulative = law['cumulativeHex']
    before = float.fromhex(cumulative[chosen - 2]) if chosen > 1 else 0.0
    selected = float.fromhex(cumulative[chosen - 1])
    fallback = uniform >= float.fromhex(cumulative[-1])
    if not ((fallback and chosen == 100) or before <= uniform < selected):
        raise ValueError('Integer law disagrees with strict CDF')
    event = {'schema': 'r14-c5-fixed-latent-partition-draw-v1',
            'key': copy.deepcopy(key), 'keySHA256': digest,
            'seedHex': seed, 'uniformHex': uniform.hex(),
            'latentConfigSHA256': law_config_sha,
            'category': chosen, 'cdfPreviousHex': before.hex(), 'cdfSelectedHex': selected.hex(),
            'roundingFallback': fallback}
    return event


def report_draw(key, table, width_hex, latent_event, latent_config):
    """Draw from a validated prospective table bound to a reproducible latent event.

    The caller supplies the width; consistency does not authenticate historical
    dispatch, prove event ordering, or establish a root's scientific width.
    """
    _validate_private_key(key)
    if (key['streamKind'] != 'report' or type(table) is not ValidatedReportTable or
            type(width_hex) is not str or width_hex != table.width_hex or
            type(latent_event) is not dict or type(latent_config) is not ValidatedLatentConfig):
        raise ValueError('Validated report table, width and latent event required')
    latent_key = private_key(key['scheduleVariant'], key['mainIndex'], key['replicateId'],
                             key['logicalRecipient'], 'latent', 0)
    expected = private_draw(latent_key, latent_config)
    try:
        matches = original.payload(latent_event) == original.payload(expected)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError('Latent event does not match report coordinates and fixed input')
    counts, cumulative = table.rows[expected['category'] - 1]
    law = {'denominator': GRID, 'counts': counts}
    digest = _hash(key)
    seed = digest[:32]
    uniform = float(np.random.Generator(np.random.PCG64(int(seed, 16))).random())
    chosen = accepted_uniform(law, uniform)
    before = float.fromhex(cumulative[chosen - 2]) if chosen > 1 else 0.0
    selected = float.fromhex(cumulative[chosen - 1])
    fallback = uniform >= float.fromhex(cumulative[-1])
    if not ((fallback and chosen == 100) or before <= uniform < selected):
        raise ValueError('Integer law disagrees with strict CDF')
    return {'schema': 'r14-c5-prospective-report-partition-draw-v1',
            'key': copy.deepcopy(key), 'keySHA256': digest, 'seedHex': seed,
            'uniformHex': uniform.hex(), 'category': chosen,
            'cdfPreviousHex': before.hex(), 'cdfSelectedHex': selected.hex(),
            'roundingFallback': fallback, 'latentKeySHA256': expected['keySHA256'],
            'latentConfigSHA256': latent_config.input_sha256,
            'latentCategory': expected['category'], 'reportWidthHex': width_hex,
            'tableProductSHA256': table.product_sha256,
            'acceptedProductByteMatch': table.accepted_product_bytes,
            'tableIdentity': ('accepted_finite_table_bytes' if table.accepted_product_bytes
                              else 'new_verified_prospective_table_bytes'),
            'historicalEventEstablished': False}


def terminal_key(snapshot, identity):
    original.A.validate(snapshot)
    original.identity(identity)
    if (identity['seed'] != 20261019 or identity['conditionId'] != 'C5' or
            not 0 <= identity['variant'] < 9 or not 1 <= identity['replicateId'] <= 1840 or
            identity['modelId'] != snapshot['candidateId'] or
            identity['conditionId'] != snapshot['task']['condition_id']):
        raise ValueError('Terminal configuration and identity do not match')
    return {'version': original.TERMINAL_STREAM, 'randomizationGroup': TERMINAL,
            'identity': identity, 'candidate': original.canonical(snapshot),
            'keyEncoding': 'typed-json-floathex-v1'}


def choose_terminal(probabilities, uniform):
    if (type(probabilities) is not list or not probabilities or
            type(uniform) not in (int, float) or not math.isfinite(uniform) or
            not 0 <= uniform < 1 or
            any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities) or abs(math.fsum(probabilities) - 1.0) > 1e-12):
        raise ValueError('Invalid terminal probabilities or uniform')
    total, chosen = 0.0, len(probabilities) - 1
    for index, probability in enumerate(probabilities):
        total += probability
        if total > uniform:
            chosen = index
            break
    return chosen


def terminal_draw(snapshot, identity):
    key = terminal_key(snapshot, identity)
    digest = _hash(key)
    uniform = float(np.random.Generator(np.random.PCG64(int(digest, 16))).random())
    return {'schema': 'r14-c5-grouped-terminal-draw-v1', 'key': key, 'keySHA256': digest,
            'uniform': uniform, 'index': choose_terminal(snapshot['random']['probabilities'], uniform),
            'cdfRule': 'ordered_binary64_strict_greater_final_index_rounding_fallback'}
