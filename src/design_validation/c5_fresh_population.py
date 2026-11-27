"""Prospective root orchestration; no historical or inference authority."""
from pathlib import Path
from .c5_fresh_domain import RunIdentity
from . import c5_public_reproduction as common


ROSTER = frozenset((v, m) for v in range(9) for m in range(1, 11))


def population_manifest(contexts, *, domain, run_id, expected_roster=ROSTER):
    """Bind admitted root snapshots to one run and an explicit exact roster.

    A smaller roster is useful for software tests but is never a complete C5
    population. Caller must still check each admitted snapshot before use.
    """
    roster = list(expected_roster)
    if (not roster or any(type(x) is not tuple or len(x) != 2 or x not in ROSTER
                          or type(x[0]) is not int or type(x[1]) is not int for x in roster)
            or len(set(roster)) != len(roster)):
        raise ValueError('Invalid expected root roster')
    entries = {}
    for context in contexts:
        root = context['spec']['rawRoot']
        key = (root['scheduleVariant'], root['mainIndex'])
        if (type(key[0]) is not int or type(key[1]) is not int or
                key not in roster or key in entries):
            raise ValueError('Unexpected or duplicate root')
        identity = RunIdentity(domain, run_id, context['identity_sha256'])
        if (common._sha(common.canonical_bytes(context['identity'])) != identity.input_identity_sha256 or
                common._sha(common.canonical_bytes(context['spec'])) != context['spec_snapshot_sha256']):
            raise ValueError('Changed admission snapshot')
        entries[key] = identity.input_identity_sha256
    if set(entries) != set(roster):
        raise ValueError('Missing expected root')
    return {'schema': 'c5-fresh-population-manifest-v1', 'domain': domain, 'runId': run_id,
            'roots': [{'variant': v, 'main': m, 'inputIdentitySHA256': entries[v, m]}
                      for v, m in sorted(entries)],
            'fullRoster': set(roster) == ROSTER, 'identityStatus': 'new-unverified',
            'statisticalAuthorization': False, 'historicalEquivalence': False}


def check_checkpoint_envelope(value, *, identity, variant, main, replicate):
    """Check member and input binding only; this is not independent replay."""
    if type(identity) is not RunIdentity:
        raise ValueError('Expected run identity')
    name = common.member_id(replicate, variant, main)
    if (type(value) is not dict or set(value) != {'memberId', 'inputSHA256', 'payload'} or
            value['memberId'] != name or value['inputSHA256'] != identity.input_identity_sha256):
        raise ValueError('Checkpoint member or input mismatch')
    payload = value['payload']
    if (type(payload) is not dict or set(payload) != {'result', 'resultSHA256'} or
            common._sha(common.canonical_bytes(payload['result'])) != payload['resultSHA256'] or
            type(payload['result']) is not dict or
            payload['result'].get('runIdentity') != {
                'domain': identity.domain, 'runId': identity.run_id,
                'inputIdentitySHA256': identity.input_identity_sha256}):
        raise ValueError('Checkpoint payload or run mismatch')
    return payload['result']


def run_population(contexts, reference_admissions, output, repository, *, domain, run_id,
                   replicates, expected_roster=ROSTER, verify_only=False):
    """Produce or independently replay checkpoints with separately admitted inputs.

    reference_admissions maps (variant, main) to keyword arguments for
    admit_reference, including spec_path, expected_inputs and repository_root.
    Input and source files must remain isolated from concurrent writers. A
    mutating failure leaves the output lock for manual investigation; never
    remove it automatically. This does not establish statistical validity.
    """
    from . import c5_fresh_inputs as inputs
    from . import c5_fresh_streams as streams
    from . import c5_fresh_streams_reference as reference_streams
    from . import c5_public_trajectory as producer
    from . import c5_public_trajectory_reference as receiver

    contexts = list(contexts)
    manifest = population_manifest(contexts, domain=domain, run_id=run_id,
                                   expected_roster=expected_roster)
    if not manifest['fullRoster'] and domain != 'test-only':
        raise ValueError('A partial roster is restricted to test-only runs')
    reps = list(replicates)
    if not reps or len(set(reps)) != len(reps):
        raise ValueError('Expected distinct replicates')
    for rep in reps:
        common.member_id(rep, 0, 1)
    if type(reference_admissions) is not dict:
        raise ValueError('Explicit reference admissions required')
    roots = {(c['spec']['rawRoot']['scheduleVariant'], c['spec']['rawRoot']['mainIndex']): c
             for c in contexts}
    if set(reference_admissions) != set(roots):
        raise ValueError('Reference admission roster mismatch')
    repository = common._executing_root(repository)
    output = Path(output)
    common._require_ignored(output, repository)
    references = {}
    for key, context in roots.items():
        inputs.check_unchanged(context)
        ref = inputs.admit_reference(**reference_admissions[key])
        if (ref['identity_sha256'] != context['identity_sha256'] or
                ref['spec_snapshot_sha256'] != context['spec_snapshot_sha256'] or
                Path(ref['repository_root']).resolve() != repository):
            raise ValueError('Independent admission does not match root input identity')
        references[key] = ref

    def replay(key, rep, result):
        ref = references[key]
        s = ref['spec']
        identity = RunIdentity(domain, run_id, ref['identity_sha256'])
        return receiver.receive_fresh_root(
            s['rawRoot'], s['config'], s['initialDecision'], ref['r_latent'], ref['r_table'],
            s['design'], rep, result, spec=ref['rs'],
            reference_streams=reference_streams.ReferenceStreams(identity, ref, rep))

    def process(base, key, rep, *, create):
        context = roots[key]
        identity = RunIdentity(domain, run_id, context['identity_sha256'])
        name = common.member_id(rep, *key)
        path = base / (name + '.json')
        common._require_ignored(path, repository)
        if path.is_symlink():
            raise ValueError('Checkpoint symlink refused')
        if path.exists():
            value = common.inspect_checkpoint(path, name, identity.input_identity_sha256)
            result = check_checkpoint_envelope(value, identity=identity, variant=key[0],
                                               main=key[1], replicate=rep)
            replay(key, rep, result)
            return common._sha(common.canonical_bytes(value))
        if not create:
            raise ValueError('Complete verified rep1 required before later replicates')
        inputs.check_unchanged(context)
        s = context['spec']
        result = producer.produce_fresh_root(
            s['rawRoot'], s['config'], s['initialDecision'], context['p_latent'], context['p_table'],
            s['design'], rep, spec=context['ps'],
            producer_streams=streams.ProducerStreams(identity, context, rep))
        replay(key, rep, result)
        value = {'memberId': name, 'inputSHA256': identity.input_identity_sha256,
                 'payload': {'result': result, 'resultSHA256': common._sha(common.canonical_bytes(result))}}
        check_checkpoint_envelope(value, identity=identity, variant=key[0], main=key[1], replicate=rep)
        return common.write_once(path, value)

    if type(verify_only) is not bool:
        raise ValueError('verify_only must be a boolean')
    if verify_only and not output.is_dir():
        raise ValueError('Verification requires an existing output directory')
    with common._output_lock(output, repository, read_only=verify_only) as base:
        # Later members require a newly replayed complete prefix, not just a manifest.
        if any(rep > 1 for rep in reps) and not manifest['fullRoster']:
            raise ValueError('Later replicates require a full verified rep1 roster')
        prefix = None
        if 1 in reps or any(rep > 1 for rep in reps):
            hashes = [{'memberId': common.member_id(1, *key),
                       'checkpointSHA256': process(base, key, 1, create=1 in reps and not verify_only)}
                      for key in sorted(roots)]
            if manifest['fullRoster']:
                prefix = {'schema': 'c5-fresh-prefix-manifest-v1', 'population': manifest,
                          'replicate': 1, 'checkpoints': hashes,
                          'identityStatus': 'new-unverified', 'historicalEquivalence': False,
                          'statisticalAuthorization': False}
                path = base / 'fresh-prefix-manifest.json'
                common._require_ignored(path, repository)
                if path.is_symlink():
                    raise ValueError('Prefix manifest symlink refused')
                if path.exists():
                    if common._file(path) != common.canonical_bytes(prefix):
                        raise ValueError('Prefix manifest mismatch')
                elif verify_only:
                    raise ValueError('Prefix manifest missing')
                else:
                    common.write_once(path, prefix)
        for rep in sorted(r for r in reps if r != 1):
            for key in sorted(roots):
                process(base, key, rep, create=not verify_only)
        return {'population': manifest, 'prefixManifest': prefix, 'replicates': sorted(reps),
                'verificationOnly': verify_only}
