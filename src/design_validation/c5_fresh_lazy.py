"""Root-at-a-time candidate orchestration; not an inference authorization."""
from pathlib import Path

from . import c5_public_reproduction as common
from . import c5_fresh_population as population
from .c5_fresh_domain import RunIdentity


def _inventory(path, digest, repository, runtime, roster):
    raw = common._file(path)
    if common._sha(raw) != digest:
        raise ValueError('Prepared manifest digest mismatch')
    manifest = common._strict_json(raw)
    fields = {'mode', 'historicalAuthenticated', 'statisticalInferenceAuthorized',
              'populationInputs', 'sources', 'runtime', 'roots'}
    if (type(manifest) is not dict or set(manifest) != fields or
            manifest['mode'] != 'fresh-input-unverified' or
            manifest['historicalAuthenticated'] is not False or
            manifest['statisticalInferenceAuthorized'] is not False or
            type(manifest['roots']) is not list):
        raise ValueError('Invalid prepared manifest')
    observed = common.sanitize_runtime(runtime)
    if observed != manifest['runtime'] or common._sources(repository) != manifest['sources']:
        raise ValueError('Source or runtime changed')
    entries, checked_paths = {}, {}
    for record in manifest['roots']:
        fields = {'rootKey', 'specPath', 'inputs', 'producerIdentitySHA256', 'referenceIdentitySHA256'}
        if (type(record) is not dict or set(record) != fields or
                type(record['rootKey']) is not list or len(record['rootKey']) != 3 or
                record['rootKey'][0] != 'C5' or
                type(record['rootKey'][1]) is not int or type(record['rootKey'][2]) is not int or
                type(record['specPath']) is not str or Path(record['specPath']).name != record['specPath'] or
                type(record['inputs']) is not dict or 'spec' not in record['inputs'] or
                record['producerIdentitySHA256'] != record['referenceIdentitySHA256']):
            raise ValueError('Invalid prepared root entry')
        key = tuple(record['rootKey'][1:])
        if key not in roster or key in entries:
            raise ValueError('Unexpected or duplicate root')
        spec_raw = common._file(path.parent / record['specPath'])
        if common._sha(spec_raw) != record['inputs']['spec']:
            raise ValueError('Root spec digest mismatch')
        spec = common._strict_json(spec_raw)
        root = spec['rawRoot']
        if [root['conditionId'], root['scheduleVariant'], root['mainIndex']] != record['rootKey']:
            raise ValueError('Root spec identity mismatch')
        pins = record['inputs']
        reports = spec['reportPaths']
        if type(reports) is not dict or not reports:
            raise ValueError('Invalid report input map')
        paths = {'lookup': (path.parent / spec['lookupPath']).resolve(strict=True),
                 'latent': (path.parent / spec['latentPath']).resolve(strict=True)}
        paths.update({'report:' + width: (path.parent / name).resolve(strict=True)
                      for width, name in reports.items()})
        if set(pins) != {'spec', *paths}:
            raise ValueError('Incomplete root input pins')
        for name, input_path in paths.items():
            if input_path not in checked_paths:
                checked_paths[input_path] = common._sha(common._file(input_path))
            if pins[name] != checked_paths[input_path]:
                raise ValueError('Root input digest mismatch')
        if pins['lookup'] != spec['lookupSHA256']:
            raise ValueError('Lookup identity mismatch')
        entries[key] = record
    if set(entries) != set(roster) or common._sha(common._file(path)) != digest:
        raise ValueError('Incomplete or changed prepared roster')
    return manifest, entries


def run_lazy(manifest_path, manifest_sha256, output, repository, *, runtime, domain, run_id,
             replicates, verify_only=False, expected_roster=None):
    """Process one root at a time, retaining no admissions between roots.

    Inputs and checkout must be isolated from writers. Failed mutations leave
    the existing output lock for manual investigation. The caller's manifest
    pin is integrity, not provenance or statistical authority.
    Verification does not create checkpoints, but the result directory must be
    writable because an exclusive operation marker is temporarily created.
    """
    from . import c5_fresh_inputs as inputs
    from . import c5_fresh_streams as streams
    from . import c5_fresh_streams_reference as reference_streams
    from . import c5_public_trajectory as producer
    from . import c5_public_trajectory_reference as receiver

    roster = population.ROSTER if expected_roster is None else frozenset(expected_roster)
    if (not roster or not roster <= population.ROSTER or
            (expected_roster is not None and domain != 'test-only')):
        raise ValueError('Invalid or unauthorized roster')
    reps = list(replicates)
    if not reps or len(set(reps)) != len(reps) or type(verify_only) is not bool:
        raise ValueError('Invalid replicates or verification mode')
    for rep in reps:
        common.member_id(rep, 0, 1)
    if any(rep > 1 for rep in reps) and roster != population.ROSTER:
        raise ValueError('Later replicates require a full verified rep1 roster')
    repository = common._executing_root(repository)
    path = Path(manifest_path).resolve(strict=True)
    manifest, entries = _inventory(path, manifest_sha256, repository, runtime, roster)
    runtime_sha = common._sha(common.canonical_bytes(common.sanitize_runtime(runtime)))
    # Validate run identity even before any checkpoint is touched.
    identities = {key: RunIdentity(domain, run_id, record['producerIdentitySHA256'])
                  for key, record in entries.items()}
    pop = {'schema': 'c5-fresh-population-manifest-v1', 'domain': domain, 'runId': run_id,
           'roots': [{'variant': v, 'main': m, 'inputIdentitySHA256': identities[v, m].input_identity_sha256}
                     for v, m in sorted(entries)], 'fullRoster': roster == population.ROSTER,
           'identityStatus': 'new-unverified', 'statisticalAuthorization': False,
           'historicalEquivalence': False}
    output = Path(output)
    common._require_ignored(output, repository)
    if verify_only and not output.is_dir():
        raise ValueError('Verification requires an existing output directory')

    def process(base, key, rep, create):
        record = entries[key]
        args = dict(spec_path=path.parent / record['specPath'], expected_inputs=record['inputs'],
                    repository_root=repository, runtime=runtime, expected_sources=manifest['sources'],
                    expected_runtime_sha256=runtime_sha)
        context = inputs.admit(**args)
        ref = inputs.admit_reference(**args)
        identity = identities[key]
        if (context['identity_sha256'] != identity.input_identity_sha256 or
                ref['identity_sha256'] != identity.input_identity_sha256 or
                context['spec_snapshot_sha256'] != ref['spec_snapshot_sha256'] or
                Path(ref['repository_root']).resolve() != repository):
            raise ValueError('Independent admission mismatch')
        inputs.check_unchanged(context)
        inputs.check_unchanged(ref)
        name = common.member_id(rep, *key)
        checkpoint = base / (name + '.json')
        common._require_ignored(checkpoint, repository)
        if checkpoint.is_symlink():
            raise ValueError('Checkpoint symlink refused')
        existing = checkpoint.exists()
        if existing:
            value = common.inspect_checkpoint(checkpoint, name, identity.input_identity_sha256)
            result = population.check_checkpoint_envelope(value, identity=identity, variant=key[0],
                                                          main=key[1], replicate=rep)
        else:
            if not create:
                raise ValueError('Required checkpoint missing')
            s = context['spec']
            result = producer.produce_fresh_root(
                s['rawRoot'], s['config'], s['initialDecision'], context['p_latent'], context['p_table'],
                s['design'], rep, spec=context['ps'],
                producer_streams=streams.ProducerStreams(identity, context, rep))
            value = {'memberId': name, 'inputSHA256': identity.input_identity_sha256,
                     'payload': {'result': result, 'resultSHA256': common._sha(common.canonical_bytes(result))}}
            population.check_checkpoint_envelope(value, identity=identity, variant=key[0], main=key[1], replicate=rep)
        s = ref['spec']
        receiver.receive_fresh_root(s['rawRoot'], s['config'], s['initialDecision'], ref['r_latent'],
                                    ref['r_table'], s['design'], rep, result, spec=ref['rs'],
                                    reference_streams=reference_streams.ReferenceStreams(identity, ref, rep))
        inputs.check_unchanged(context)
        inputs.check_unchanged(ref)
        if not existing:
            return common.write_once(checkpoint, value)
        if not checkpoint.exists():
            raise ValueError('Verified checkpoint disappeared')
        return common._sha(common.canonical_bytes(value))

    with common._output_lock(output, repository, read_only=verify_only) as base:
        prefix = None
        if 1 in reps or any(rep > 1 for rep in reps):
            hashes = [{'memberId': common.member_id(1, *key),
                       'checkpointSHA256': process(base, key, 1, 1 in reps and not verify_only)}
                      for key in sorted(entries)]
            if pop['fullRoster']:
                prefix = {'schema': 'c5-fresh-prefix-manifest-v1', 'population': pop, 'replicate': 1,
                          'checkpoints': hashes, 'identityStatus': 'new-unverified',
                          'historicalEquivalence': False, 'statisticalAuthorization': False}
                target = base / 'fresh-prefix-manifest.json'
                common._require_ignored(target, repository)
                if target.is_symlink():
                    raise ValueError('Prefix manifest symlink refused')
                if target.exists():
                    if common._file(target) != common.canonical_bytes(prefix):
                        raise ValueError('Prefix manifest mismatch')
                elif verify_only:
                    raise ValueError('Prefix manifest missing')
                else:
                    common.write_once(target, prefix)
        for rep in sorted(r for r in reps if r != 1):
            for key in sorted(entries):
                process(base, key, rep, not verify_only)
        if common._sha(common._file(path)) != manifest_sha256:
            raise ValueError('Prepared manifest changed')
        return {'population': pop, 'prefixManifest': prefix, 'replicates': sorted(reps),
                'verificationOnly': verify_only}
