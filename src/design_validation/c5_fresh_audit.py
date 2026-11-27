"""Replay and describe a prospective population, without inferential authority.

Inputs and checkpoints must be isolated from concurrent writers throughout this
call: the replay lock does not cover the entire audit. Before/after checks
detect changes, but cannot prevent an ABA replacement.
Temporary disk usage grows with checkpoints and event keys. Memory still depends
on the largest checkpoint, replay implementation, replicate list and exact
fraction sizes; no unbounded-resource guarantee is made. The temporary database
and its directory must also be isolated from concurrent modification.
"""
import json
import os
from pathlib import Path
import stat
import sqlite3
import tempfile
from contextlib import closing
from fractions import Fraction

from . import c5_fresh_population as population
from . import c5_fresh_lazy as lazy
from . import c5_public_reproduction as common
from .c5_fresh_domain import RunIdentity, parse_key


def _snapshot(path, *, keep_raw=False):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink in checkpoint path')
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Checkpoint is not a regular file')
            raw = stream.read()
            after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Checkpoint changed during read')
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns, common._sha(raw))
        return (identity, raw) if keep_raw else identity
    except OSError as exc:
        raise ValueError('Missing or inaccessible checkpoint') from exc


def _fraction(pair):
    if type(pair) is not list or len(pair) != 2 or any(type(x) is not str for x in pair):
        raise ValueError('Invalid rational contribution')
    try:
        value = Fraction(int(pair[0]), int(pair[1]))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('Invalid rational contribution') from exc
    if pair != [str(value.numerator), str(value.denominator)]:
        raise ValueError('Noncanonical rational contribution')
    return value


def audit_population(contexts, reference_admissions, checkpoint_dir, repository, *, domain, run_id, replicates):
    """Independently replay all requested members, then return descriptive sums.

    This does not establish independence of people, statistical validity,
    confidence intervals, sampling authorization, or historical equivalence.
    """
    contexts, reps = list(contexts), list(replicates)
    manifest = population.population_manifest(contexts, domain=domain, run_id=run_id)
    roots = {(c['spec']['rawRoot']['scheduleVariant'], c['spec']['rawRoot']['mainIndex']):
             c['identity_sha256'] for c in contexts}
    return _audit(manifest, roots, checkpoint_dir, domain, run_id, reps,
                  lambda: population.run_population(contexts, reference_admissions, checkpoint_dir, repository,
                                                    domain=domain, run_id=run_id, replicates=reps,
                                                    verify_only=True))


def audit_prepared_manifest(manifest_path, manifest_sha256, checkpoint_dir, repository, *,
                            runtime, domain, run_id, replicates):
    """Replay a complete prepared population one root at a time, then describe it.

    The manifest pin is an integrity check, not provenance or inference authority.
    Inputs and checkpoints must remain isolated from concurrent writers.
    """
    path = Path(manifest_path).resolve(strict=True)
    repository = common._executing_root(repository)
    _, entries = lazy._inventory(path, manifest_sha256, repository, runtime, population.ROSTER)
    roots = {key: record['producerIdentitySHA256'] for key, record in entries.items()}
    manifest = {'schema': 'c5-fresh-population-manifest-v1', 'domain': domain, 'runId': run_id,
                'roots': [{'variant': v, 'main': m, 'inputIdentitySHA256': roots[v, m]}
                          for v, m in sorted(roots)], 'fullRoster': set(roots) == set(population.ROSTER),
                'identityStatus': 'new-unverified', 'statisticalAuthorization': False,
                'historicalEquivalence': False}
    reps = list(replicates)
    return _audit(manifest, roots, checkpoint_dir, domain, run_id, reps,
                  lambda: lazy.run_lazy(path, manifest_sha256, checkpoint_dir, repository, runtime=runtime,
                                        domain=domain, run_id=run_id, replicates=reps, verify_only=True))


def _audit(manifest, roots, checkpoint_dir, domain, run_id, reps, replay):
    if not manifest['fullRoster'] or not reps or len(set(reps)) != len(reps):
        raise ValueError('Complete population and distinct replicates required')
    if set(roots) != set(population.ROSTER):
        raise ValueError('Complete population required')
    for rep in reps:
        common.member_id(rep, 0, 1)
    with tempfile.TemporaryDirectory(prefix='c5-fresh-audit-') as tmp:
        with closing(sqlite3.connect(str(Path(tmp) / 'audit.sqlite'))) as db:
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE snapshots (rep INTEGER, variant INTEGER, main INTEGER, '
                       'path TEXT NOT NULL, identity TEXT NOT NULL, PRIMARY KEY (rep, variant, main))')
            db.execute('CREATE TABLE events (key BLOB PRIMARY KEY, digest TEXT NOT NULL UNIQUE)')
            db.execute('CREATE TABLE results (digest TEXT PRIMARY KEY)')
            return _audit_disk(db, manifest, roots, checkpoint_dir, domain, run_id, reps, replay)


def _audit_disk(db, manifest, roots, checkpoint_dir, domain, run_id, reps, replay):
    # The existing verifier also replays rep 1 for any later member.
    checked_reps = set(reps) | ({1} if any(r > 1 for r in reps) else set())
    for rep in checked_reps:
        for v, m in roots:
            path = Path(checkpoint_dir) / (common.member_id(rep, v, m) + '.json')
            db.execute('INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)',
                       (rep, v, m, str(path), json.dumps(_snapshot(path))))
    if 1 in checked_reps:
        path = Path(checkpoint_dir) / 'fresh-prefix-manifest.json'
        db.execute('INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)', (0, 0, 0, str(path), json.dumps(_snapshot(path))))
    result = replay()
    if (result.get('verificationOnly') is not True or result.get('population') != manifest or
            result.get('replicates') != sorted(reps)):
        raise ValueError('Independent replay did not confirm requested population')
    for path, snapshot in db.execute('SELECT path, identity FROM snapshots'):
        if list(_snapshot(path)) != json.loads(snapshot):
            raise ValueError('Checkpoint changed during verification')
    sums = {'gap': Fraction(), 'partial_low': Fraction()}
    ties = event_count = 0
    for rep in sorted(reps):
        for v, m in sorted(roots):
            path, snapshot = db.execute('SELECT path, identity FROM snapshots WHERE rep=? AND variant=? AND main=?',
                                        (rep, v, m)).fetchone()
            identity, raw = _snapshot(path, keep_raw=True)
            if list(identity) != json.loads(snapshot):
                raise ValueError('Checkpoint changed after verification')
            value = common._strict_json(raw)
            if common.canonical_bytes(value) != raw:
                raise ValueError('Noncanonical checkpoint')
            identity = RunIdentity(domain, run_id, roots[v, m])
            member = population.check_checkpoint_envelope(value, identity=identity, variant=v, main=m, replicate=rep)
            if (member.get('rootKey') != ['C5', v, m] or member.get('replicateId') != rep or
                    member.get('ordinal') != 90 * (rep - 1) + 10 * v + m - 1):
                raise ValueError('Member coordinate mismatch')
            digest = value['payload']['resultSHA256']
            try:
                db.execute('INSERT INTO results VALUES (?)', (digest,))
            except sqlite3.IntegrityError as exc:
                raise ValueError('Duplicate member digest') from exc
            latent, stages = member['latent'], member['stages']
            if type(latent) is not dict or set(latent) != {'A', 'B'} or type(stages) is not list:
                raise ValueError('Invalid event collections')
            events = [(latent[k], 'latent', k) for k in ('A', 'B')]
            events += [(stage['report'], 'report', None) for stage in stages]
            if member['terminalStream'] is not None:
                events.append((member['terminalStream'], 'terminal', None))
            for event, kind, recipient in events:
                key = parse_key(event['key'], identity)
                if ((key.variant, key.main, key.replicate, key.kind) != (v, m, rep, kind) or
                        (recipient is not None and key.logical_recipient != recipient)):
                    raise ValueError('Event coordinate mismatch')
                encoded = common.canonical_bytes(event['key'])
                key_digest = common._sha(encoded)
                if event.get('keySHA256') != key_digest:
                    raise ValueError('Duplicate or invalid event key or digest')
                try:
                    db.execute('INSERT INTO events VALUES (?, ?)', (encoded, key_digest))
                except sqlite3.IntegrityError as exc:
                    raise ValueError('Duplicate or invalid event key or digest') from exc
                event_count += 1
            status = member['endpoint']['status']
            if status not in ('exact_tie', 'complete') or set(member['contribution']) != set(sums):
                raise ValueError('Invalid endpoint or contribution')
            values = {key: _fraction(member['contribution'][key]) for key in sums}
            if status == 'exact_tie':
                if any(values.values()):
                    raise ValueError('Tie must have zero contribution')
                ties += 1
            for key in sums:
                sums[key] += values[key]
    for path, snapshot in db.execute('SELECT path, identity FROM snapshots'):
        if list(_snapshot(path)) != json.loads(snapshot):
            raise ValueError('Checkpoint changed during aggregation')
    return {'schema': 'c5-fresh-descriptive-audit-v1', 'domain': domain, 'runId': run_id,
            'replicates': sorted(reps), 'roots': len(roots), 'members': len(reps) * len(roots),
            'coverageDenominator': len(reps) * 90, 'ties': ties, 'eventCount': event_count,
            'contributionSums': {k: [str(v.numerator), str(v.denominator)] for k, v in sums.items()},
            'identityStatus': 'new-unverified', 'historicalEquivalence': False,
            'statisticalAuthorization': False, 'interpretation': 'descriptive-only-no-inference'}
