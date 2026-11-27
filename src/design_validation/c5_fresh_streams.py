"""Prospective producer streams. These are not historical replay or a receiver."""
import hashlib
import json

import numpy as np

from . import c5_fresh_inputs
from . import c5_public_streams as arithmetic
from . import c5_public_trajectory as trajectory
from . import personal_time_sensitivity_streams as encoding
from .c5_fresh_domain import RunIdentity, TypedKey


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _uniform(key):
    if type(key) is not TypedKey:
        raise ValueError('A typed prospective key is required')
    digest = _sha(key.as_dict())
    # The entire digest, not a truncated historical seed, defines this stream.
    return digest, float(np.random.Generator(np.random.PCG64(int(digest, 16))).random())


class ProducerStreams:
    """Use a separately admitted immutable input snapshot for one root and run.

    Callers must prevent concurrent changes to input files; the checks here are
    detection rather than a filesystem lock. Output has no inference authority.
    """
    def __init__(self, identity, context, replicate):
        if type(identity) is not RunIdentity:
            raise ValueError('Explicit prospective run identity required')
        c5_fresh_inputs.check_unchanged(context)
        if identity.input_identity_sha256 != context['identity_sha256']:
            raise ValueError('Run and admitted input identity mismatch')
        spec = context['spec']
        root, cfg = spec['rawRoot'], spec['config']
        trajectory._validate_root(root, cfg, spec['initialDecision'], context['p_latent'],
                                  spec['design'], context['ps'])
        self.identity, self.context, self.replicate = identity, context, replicate
        self.variant, self.main = root['scheduleVariant'], root['mainIndex']
        self.width = float(cfg['task']['report_width']).hex()
        self._key('latent', 'A', 0)
        if dict(context['ps'].table).get(self.width) is not context['p_table']:
            raise ValueError('Root-bound admitted report table required')

    def _check(self):
        c5_fresh_inputs.check_unchanged(self.context)
        s = self.context['spec']
        trajectory._validate_root(s['rawRoot'], s['config'], s['initialDecision'],
                                  self.context['p_latent'], s['design'], self.context['ps'])
        if (s['rawRoot']['scheduleVariant'] != self.variant or
                s['rawRoot']['mainIndex'] != self.main or
                float(s['config']['task']['report_width']).hex() != self.width or
                dict(self.context['ps'].table).get(self.width) is not self.context['p_table']):
            raise ValueError('Root or table changed')

    def _key(self, kind, recipient=None, index=None, binding=None):
        return TypedKey(self.identity, kind, self.variant, self.main, self.replicate,
                        recipient, index, binding)

    def _partition(self, key, counts, cumulative, **bindings):
        digest, uniform = _uniform(key)
        category = arithmetic.accepted_uniform({'denominator': arithmetic.GRID, 'counts': counts}, uniform)
        previous = float.fromhex(cumulative[category - 2]) if category > 1 else 0.0
        selected = float.fromhex(cumulative[category - 1])
        fallback = uniform >= float.fromhex(cumulative[-1])
        if not ((fallback and category == 100) or previous <= uniform < selected):
            raise ValueError('Partition and CDF disagree')
        return {'schema': 'c5-fresh-partition-producer-v1', 'key': key.as_dict(),
                'keySHA256': digest, 'uniformHex': uniform.hex(), 'category': category,
                'cdfPreviousHex': previous.hex(), 'cdfSelectedHex': selected.hex(),
                'roundingFallback': fallback, 'bindings': bindings,
                'identityStatus': 'new-unverified'}

    def latent(self, recipient):
        self._check()
        key = self._key('latent', recipient, 0)
        config = self.context['p_latent']
        if type(config) is not arithmetic.ValidatedLatentConfig:
            raise ValueError('Validated immutable latent config required')
        counts, cumulative = config.roots[self.variant * 10 + self.main - 1][('A', 'B').index(recipient)]
        return self._partition(key, counts, cumulative, latentConfigSHA256=config.input_sha256,
                               inputIdentitySHA256=self.context['identity_sha256'])

    def report(self, recipient, index, latent_event):
        self._check()
        key = self._key('report', recipient, index)
        expected = self.latent(recipient)
        if _bytes(latent_event) != _bytes(expected):
            raise ValueError('Latent event does not match this run, root and recipient')
        table = self.context['p_table']
        if type(table) is not arithmetic.ValidatedReportTable or table.width_hex != self.width:
            raise ValueError('Validated root-width report table required')
        counts, cumulative = table.rows[expected['category'] - 1]
        return self._partition(key, counts, cumulative, latentEventSHA256=_sha(expected),
                               latentConfigSHA256=self.context['p_latent'].input_sha256,
                               reportTableSHA256=table.product_sha256, reportWidthHex=self.width,
                               inputIdentitySHA256=self.context['identity_sha256'])

    def terminal(self, snapshot, member_identity):
        self._check()
        encoding.A.validate(snapshot)
        encoding.identity(member_identity)
        cfg = self.context['spec']['config']
        if (member_identity['conditionId'] != 'C5' or
                member_identity['variant'] != self.variant or
                member_identity['mainIndex'] != self.main or
                member_identity['replicateId'] != self.replicate or
                member_identity['modelId'] != snapshot['candidateId'] or
                snapshot['task']['condition_id'] != 'C5'):
            raise ValueError('Terminal member identity does not match candidate and root')
        if _bytes(encoding.canonical({k: v for k, v in snapshot.items() if k != 'state'})) != _bytes(
                encoding.canonical({k: v for k, v in cfg.items() if k != 'state'})):
            raise ValueError('Terminal candidate does not match admitted root configuration')
        binding = {'schema': 'c5-fresh-terminal-binding-v1', 'run': {
            'domain': self.identity.domain, 'runId': self.identity.run_id,
            'variant': self.variant, 'main': self.main, 'replicate': self.replicate},
            'inputIdentitySHA256': self.context['identity_sha256'],
            'memberIdentity': encoding.canonical(member_identity),
            'candidate': encoding.canonical(snapshot)}
        key = self._key('terminal', binding=_sha(binding))
        digest, uniform = _uniform(key)
        return {'schema': 'c5-fresh-terminal-producer-v1', 'key': key.as_dict(),
                'keySHA256': digest, 'binding': binding, 'uniformHex': uniform.hex(),
                'index': arithmetic.choose_terminal(snapshot['random']['probabilities'], uniform),
                'identityStatus': 'new-unverified'}
