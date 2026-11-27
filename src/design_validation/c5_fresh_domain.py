"""Typed identities for prospective C5 runs, without sampling or historical replay.

The key envelope is deliberately distinct from historical stream envelopes.
It establishes typed domain separation, not absence of hash or seed collisions.
"""
from dataclasses import dataclass
import re


_RUN_ID = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}\Z', re.ASCII)
_SCHEMA = 'c5-prospective-typed-key-v1'
_DOMAINS = ('fresh', 'test-only')


@dataclass(frozen=True)
class RunIdentity:
    domain: str
    run_id: str
    input_identity_sha256: str

    def __post_init__(self):
        if type(self.domain) is not str or self.domain not in _DOMAINS:
            raise ValueError('Expected fresh or test-only domain')
        if type(self.run_id) is not str or _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError('Invalid run ID')
        if type(self.input_identity_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', self.input_identity_sha256) is None:
            raise ValueError('Expected pinned input identity digest')


@dataclass(frozen=True)
class TypedKey:
    identity: RunIdentity
    kind: str
    variant: int
    main: int
    replicate: int
    logical_recipient: str | None = None
    report_index: int | None = None
    terminal_binding_sha256: str | None = None

    def __post_init__(self):
        if type(self.identity) is not RunIdentity:
            raise ValueError('Expected prospective run identity')
        if (type(self.kind) is not str or self.kind not in ('latent', 'report', 'terminal') or
                type(self.variant) is not int or not 0 <= self.variant < 9 or
                type(self.main) is not int or not 1 <= self.main <= 10 or
                type(self.replicate) is not int or not 1 <= self.replicate <= 1840):
            raise ValueError('Invalid key kind or coordinate')
        if self.kind == 'terminal':
            valid = (self.logical_recipient is None and self.report_index is None and
                     type(self.terminal_binding_sha256) is str and
                     re.fullmatch(r'[0-9a-f]{64}', self.terminal_binding_sha256) is not None)
        else:
            valid = (type(self.logical_recipient) is str and self.logical_recipient in ('A', 'B') and
                     type(self.report_index) is int and 0 <= self.report_index < 2**53 and
                     self.terminal_binding_sha256 is None and
                     (self.kind != 'latent' or self.report_index == 0))
        if not valid:
            raise ValueError('Invalid stream-specific coordinate')

    def as_dict(self):
        """Return a new, explicit envelope suitable for independent serialization."""
        return {'schema': _SCHEMA, 'domain': self.identity.domain, 'runId': self.identity.run_id,
                'inputIdentitySHA256': self.identity.input_identity_sha256,
                'kind': self.kind, 'variant': self.variant, 'main': self.main,
                'replicate': self.replicate, 'logicalRecipient': self.logical_recipient,
                'reportIndex': self.report_index, 'terminalBindingSHA256': self.terminal_binding_sha256}


def parse_key(value, identity):
    """Validate a key against the caller's expected run, rejecting cross-domain input."""
    fields = {'schema', 'domain', 'runId', 'inputIdentitySHA256', 'kind', 'variant', 'main', 'replicate',
              'logicalRecipient', 'reportIndex', 'terminalBindingSHA256'}
    if (type(identity) is not RunIdentity or type(value) is not dict or set(value) != fields or
            value['schema'] != _SCHEMA or value['domain'] != identity.domain or
            value['runId'] != identity.run_id or value['inputIdentitySHA256'] != identity.input_identity_sha256):
        raise ValueError('Key envelope or expected run mismatch')
    return TypedKey(identity, value['kind'], value['variant'], value['main'], value['replicate'],
                    value['logicalRecipient'], value['reportIndex'], value['terminalBindingSHA256'])
