"""Independent event replay for prospective streams, not historical authentication.

NumPy PCG64 and binary64 arithmetic are part of this numerical identity. Typed
domain separation and SHA-256 do not prove absence of collisions. Callers must
independently construct terminal snapshot and member identity, and isolate the
admitted inputs from concurrent writers; file checks are not locks.
"""
import hashlib
import json
import math

import numpy as np

from . import c5_fresh_inputs
from . import c5_public_trajectory_reference
from .c5_fresh_domain import RunIdentity


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _canonical(value):
    if value is None:
        return {'null': True}
    if type(value) is bool:
        return {'bool': value}
    if type(value) is int:
        return {'int': str(value)}
    if type(value) is float:
        if math.isnan(value) or value == math.inf:
            raise ValueError('Invalid canonical float')
        return {'floatHex': value.hex()}
    if type(value) is str:
        return {'str': value}
    if type(value) is list:
        return {'list': [_canonical(v) for v in value]}
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise ValueError('Invalid canonical value')
    return {'dict': [[k, _canonical(value[k])] for k in sorted(value)]}


def _draw(key):
    digest = _digest(key)
    uniform = float(np.random.Generator(np.random.PCG64(int(digest, 16))).random())
    return digest, uniform


def _partition(key, counts, cumulative, bindings):
    if (len(counts) != 100 or len(cumulative) != 100 or
            any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 1 << 53):
        raise ValueError('Invalid 53-bit law')
    digest, uniform = _draw(key)
    numerator, denominator = uniform.as_integer_ratio()
    integer, remainder = divmod(numerator * (1 << 53), denominator)
    if remainder or not 0 <= integer < 1 << 53:
        raise ValueError('Uniform is not on the 53-bit grid')
    bound = 0
    for category, count in enumerate(counts, 1):
        bound += count
        if integer < bound:
            break
    previous = float.fromhex(cumulative[category - 2]) if category > 1 else 0.0
    selected = float.fromhex(cumulative[category - 1])
    fallback = uniform >= float.fromhex(cumulative[-1])
    if not ((fallback and category == 100) or previous <= uniform < selected):
        raise ValueError('Partition and CDF disagree')
    return {'schema': 'c5-fresh-partition-producer-v1', 'key': key,
            'keySHA256': digest, 'uniformHex': uniform.hex(), 'category': category,
            'cdfPreviousHex': previous.hex(), 'cdfSelectedHex': selected.hex(),
            'roundingFallback': fallback, 'bindings': bindings, 'identityStatus': 'new-unverified'}


def _equal(actual, expected):
    if _bytes(actual) != _bytes(expected):
        raise ValueError('Event does not match independently reconstructed event')
    return expected


class ReferenceStreams:
    """Replay against separately admitted receiver inputs, not producer internals."""
    def __init__(self, identity, context, replicate):
        if type(identity) is not RunIdentity or identity.input_identity_sha256 != context['identity_sha256']:
            raise ValueError('Run and admitted input identity mismatch')
        if type(replicate) is not int or not 1 <= replicate <= 1840:
            raise ValueError('Invalid replicate')
        self.identity, self.context, self.replicate = identity, context, replicate
        root = context['spec']['rawRoot']
        self.variant, self.main = root['scheduleVariant'], root['mainIndex']
        if (type(self.variant) is not int or not 0 <= self.variant < 9 or
                type(self.main) is not int or not 1 <= self.main <= 10):
            raise ValueError('Invalid root')
        self.width = float(context['spec']['config']['task']['report_width']).hex()
        self._check()

    def _check(self):
        c5_fresh_inputs.check_unchanged(self.context)
        s = self.context['spec']
        if (self.identity.input_identity_sha256 != self.context['identity_sha256'] or
                s['rawRoot']['scheduleVariant'] != self.variant or s['rawRoot']['mainIndex'] != self.main or
                float(s['config']['task']['report_width']).hex() != self.width or
                dict(self.context['rs'].table).get(self.width) is not self.context['r_table']):
            raise ValueError('Receiver root or table changed')
        c5_public_trajectory_reference.validate_spec(s['rawRoot'], s['config'], s['initialDecision'],
                                                    self.context['r_latent'], s['design'], self.context['rs'])

    def _key(self, kind, recipient=None, index=None, binding=None):
        if kind in ('latent', 'report'):
            if (recipient not in ('A', 'B') or type(index) is not int or not 0 <= index < 2**53 or
                    (kind == 'latent' and index != 0) or binding is not None):
                raise ValueError('Invalid stream coordinate')
        elif kind != 'terminal' or recipient is not None or index is not None or type(binding) is not str:
            raise ValueError('Invalid terminal coordinate')
        return {'schema': 'c5-prospective-typed-key-v1', 'domain': self.identity.domain,
                'runId': self.identity.run_id, 'inputIdentitySHA256': self.identity.input_identity_sha256,
                'kind': kind, 'variant': self.variant, 'main': self.main, 'replicate': self.replicate,
                'logicalRecipient': recipient, 'reportIndex': index, 'terminalBindingSHA256': binding}

    def _latent(self, recipient):
        key = self._key('latent', recipient, 0)
        law = self.context['r_latent']
        counts, cdf = law.roots[self.variant * 10 + self.main - 1][('A', 'B').index(recipient)]
        return _partition(key, counts, cdf, {'latentConfigSHA256': law.input_sha256,
                                            'inputIdentitySHA256': self.context['identity_sha256']})

    def verify_latent(self, event, recipient):
        self._check()
        return _equal(event, self._latent(recipient))

    def verify_report(self, event, recipient, index, latent_event):
        self._check()
        latent = _equal(latent_event, self._latent(recipient))
        table = self.context['r_table']
        if table.width_hex != self.width:
            raise ValueError('Report width mismatch')
        counts, cdf = table.rows[latent['category'] - 1]
        bindings = {'latentEventSHA256': _digest(latent),
                    'latentConfigSHA256': self.context['r_latent'].input_sha256,
                    'reportTableSHA256': table.product_sha256, 'reportWidthHex': self.width,
                    'inputIdentitySHA256': self.context['identity_sha256']}
        return _equal(event, _partition(self._key('report', recipient, index), counts, cdf, bindings))

    def verify_terminal(self, event, snapshot, member_identity):
        """Compare to caller-reconstructed snapshot and member identity, never event binding."""
        self._check()
        fields = {'seed', 'modelId', 'personId', 'trialId', 'conditionId', 'variant', 'replicateId', 'mainIndex'}
        if (type(member_identity) is not dict or set(member_identity) != fields or
                any(type(member_identity[k]) is not int or member_identity[k] < 0
                    for k in ('seed', 'variant', 'replicateId', 'mainIndex')) or
                any(type(member_identity[k]) is not str or not member_identity[k]
                    for k in ('modelId', 'personId', 'trialId')) or
                member_identity['conditionId'] != 'C5' or member_identity['variant'] != self.variant or
                member_identity['mainIndex'] != self.main or member_identity['replicateId'] != self.replicate or
                member_identity['modelId'] != snapshot['candidateId'] or snapshot['task']['condition_id'] != 'C5'):
            raise ValueError('Terminal member does not match root and candidate')
        cfg = self.context['spec']['config']
        if _canonical({k: v for k, v in snapshot.items() if k != 'state'}) != _canonical({k: v for k, v in cfg.items() if k != 'state'}):
            raise ValueError('Terminal candidate differs from admitted config')
        binding = {'schema': 'c5-fresh-terminal-binding-v1', 'run': {
            'domain': self.identity.domain, 'runId': self.identity.run_id, 'variant': self.variant,
            'main': self.main, 'replicate': self.replicate},
            'inputIdentitySHA256': self.context['identity_sha256'],
            'memberIdentity': _canonical(member_identity), 'candidate': _canonical(snapshot)}
        key = self._key('terminal', binding=_digest(binding))
        digest, uniform = _draw(key)
        probabilities = snapshot['random']['probabilities']
        if (type(probabilities) is not list or not probabilities or
                any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or
                abs(math.fsum(probabilities) - 1.0) > 1e-12):
            raise ValueError('Invalid terminal probabilities')
        total, chosen = 0.0, len(probabilities) - 1
        for i, p in enumerate(probabilities):
            total += p
            if total > uniform:
                chosen = i
                break
        return _equal(event, {'schema': 'c5-fresh-terminal-producer-v1', 'key': key,
                              'keySHA256': digest, 'binding': binding, 'uniformHex': uniform.hex(),
                              'index': chosen, 'identityStatus': 'new-unverified'})
