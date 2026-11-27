"""Prepare a pinned C5 population, without outcomes or inference authority.

Read-only modes and hashes are not locks. The caller must isolate inputs and
the executing checkout from writers throughout preparation and later use.
"""
from pathlib import Path
import re

from . import c5_fresh_inputs as fresh
from . import c5_public_reproduction as common


def inspect_population_pins(manifest_path, manifest_sha256, repository_root):
    """List current byte identities only; caller must isolate and authenticate inputs."""
    from . import personal_time_inputs as personal
    from . import personal_time_sensitivity as sensitivity

    root = common._executing_root(repository_root)
    manifest_path = Path(manifest_path).resolve(strict=True)
    raw = common._file(manifest_path)
    if common._sha(raw) != manifest_sha256:
        raise ValueError('Population manifest byte digest mismatch')
    manifest = common._strict_json(raw)
    if (type(manifest) is not dict or set(manifest) !=
            {'recipePath', 'rootsPath', 'lookupPath', 'lookupSHA256', 'latentPath', 'reportPaths'} or
            type(manifest['reportPaths']) is not dict or not manifest['reportPaths']):
        raise ValueError('Expected complete population input manifest')
    base = manifest_path.parent
    paths = {name: (base / manifest[key]).resolve(strict=True) for name, key in
             (('recipe', 'recipePath'), ('roots', 'rootsPath'), ('lookup', 'lookupPath'),
              ('latent', 'latentPath'))}
    paths.update({'report:' + width: (base / path).resolve(strict=True)
                  for width, path in manifest['reportPaths'].items()})
    paths['manifest'] = manifest_path
    scientific = sensitivity.load_context(paths['recipe'])
    reference = (paths['recipe'].parent / scientific['recipe']['reference']).resolve(strict=True)
    paths['reference'] = reference
    for name in personal.INPUT_SHA256:
        paths['reference:' + name] = (reference.parent /
            scientific['reference']['recipe'][name]).resolve(strict=True)
    hashes = {key: common._sha(common._file(path)) for key, path in paths.items()}
    if hashes['manifest'] != manifest_sha256 or hashes['lookup'] != manifest['lookupSHA256']:
        raise ValueError('Manifest or lookup changed during inspection')
    return {'inputs': hashes, 'sources': common._sources(root),
            'entrySHA256': common._sha(common._file(root / 'scripts' / 'fresh_c5_run.py'))}


def prepare_population(manifest_path, manifest_sha256, output, repository_root, *,
                       expected_inputs, expected_sources, expected_entry_sha256,
                       runtime, expected_runtime_sha256):
    """Stage and independently admit all 90 roots; never create a prefix.

    ``expected_inputs`` pins the manifest and its entire scientific closure,
    using the same names as load_population (except there is no prefix).
    ``expected_entry_sha256`` pins the actual source-calling entry as well.
    The output must not exist. A failed attempt is left in place, without a
    completion manifest, and must not be treated as admitted.
    """
    root = common._executing_root(repository_root)
    manifest_path = Path(manifest_path).resolve(strict=True)
    raw = common._file(manifest_path)
    if common._sha(raw) != manifest_sha256:
        raise ValueError('Population manifest byte digest mismatch')
    manifest = common._strict_json(raw)
    fields = {'recipePath', 'rootsPath', 'lookupPath', 'lookupSHA256', 'latentPath', 'reportPaths'}
    if type(manifest) is not dict or set(manifest) != fields:
        raise ValueError('Expected complete population input manifest')
    if (any(type(manifest[k]) is not str or not manifest[k]
            for k in ('recipePath', 'rootsPath', 'lookupPath', 'latentPath')) or
            type(manifest['lookupSHA256']) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', manifest['lookupSHA256']) or
            type(manifest['reportPaths']) is not dict or not manifest['reportPaths'] or
            any(type(k) is not str or not k or type(v) is not str or not v
                for k, v in manifest['reportPaths'].items())):
        raise ValueError('Invalid population paths or lookup digest')
    from . import c5_input_assembly as assembly
    from . import c5_public_inputs as inputs
    from . import personal_time_inputs as personal
    from . import personal_time_sensitivity as sensitivity

    base = manifest_path.parent
    paths = {name: (base / manifest[key]).resolve(strict=True) for name, key in
             (('recipe', 'recipePath'), ('roots', 'rootsPath'), ('lookup', 'lookupPath'),
              ('latent', 'latentPath'))}
    paths.update({'report:' + width: (base / path).resolve(strict=True)
                  for width, path in manifest['reportPaths'].items()})
    paths['manifest'] = manifest_path
    # Check the recipe itself before following its references.
    if (type(expected_inputs) is not dict or
            expected_inputs.get('recipe') != common._sha(common._file(paths['recipe']))):
        raise ValueError('Recipe pin required')
    scientific_context = sensitivity.load_context(paths['recipe'])
    reference = (paths['recipe'].parent / scientific_context['recipe']['reference']).resolve(strict=True)
    paths['reference'] = reference
    for name in personal.INPUT_SHA256:
        paths['reference:' + name] = (reference.parent /
            scientific_context['reference']['recipe'][name]).resolve(strict=True)
    blobs = {key: common._file(path) for key, path in paths.items()}
    hashes = {key: common._sha(value) for key, value in blobs.items()}
    if (hashes != expected_inputs or hashes['manifest'] != manifest_sha256 or
            hashes['lookup'] != manifest['lookupSHA256']):
        raise ValueError('Complete population input byte pins required')
    if common._sources(root) != expected_sources:
        raise ValueError('Source identity mismatch')
    if common._sha(common._file(root / 'scripts' / 'fresh_c5_run.py')) != expected_entry_sha256:
        raise ValueError('Entry source identity mismatch')
    observed_runtime = common.sanitize_runtime(runtime)
    if common._sha(common.canonical_bytes(observed_runtime)) != expected_runtime_sha256:
        raise ValueError('Runtime identity mismatch')
    common._require_readonly_snapshot({'paths': paths, 'identity': {'sources': expected_sources},
                                       'repository_root': root})
    selected = personal.selection([dict(zip(personal.FIELDS, key)) for key in inputs.ROSTER])
    roots = personal.validate_roots(common._strict_json(blobs['roots']), selected,
                                    scientific_context['reference'])
    reports = {width: blobs['report:' + width] for width in manifest['reportPaths']}
    assembled = assembly.assemble(paths['lookup'], manifest['lookupSHA256'], blobs['latent'],
                                  roots, scientific_context, reports)
    rows = assembled['lookup']['roots']
    if len(roots) != 90 or len(rows) != 90 or len(assembled['geometryCertificates']) != 9:
        raise ValueError('Complete 90-root roster required')
    destination = Path(output).absolute()
    if destination.parent.resolve(strict=True) != destination.parent or destination.is_relative_to(root):
        raise ValueError('Output must have a resolved parent outside the source checkout')
    destination.mkdir(mode=0o700)  # exclusive: never reuse or replace output

    def stage(name, data):
        path = destination / name
        with path.open('xb') as handle:
            handle.write(data)
        path.chmod(0o444)
        return path

    stage('lookup.json', blobs['lookup'])
    stage('latent.json', blobs['latent'])
    report_names = {width: 'report-%02d.json' % i for i, width in enumerate(sorted(reports))}
    for width, name in report_names.items():
        stage(name, reports[width])
    records = []
    for position, (raw_root, row) in enumerate(zip(roots, rows)):
        cert = assembled['geometryCertificates'][position // 10]
        schedule = rows[(position // 10) * 10:(position // 10 + 1) * 10]
        widths = {float(item['reportWidth']).hex() for item in schedule}
        spec = {'rawRoot': raw_root, 'config': row['config'], 'initialDecision': row['decision'],
                'design': cert['design'], 'lookupPath': 'lookup.json',
                'lookupSHA256': hashes['lookup'], 'geometryCertificate': cert,
                'latentPath': 'latent.json',
                'reportPaths': {key: report_names[key] for key in sorted(widths)}}
        spec_raw = common.canonical_bytes(spec)
        name = 'root-%02d.json' % position
        spec_path = stage(name, spec_raw)
        pins = {'spec': common._sha(spec_raw), 'lookup': hashes['lookup'],
                'latent': hashes['latent']}
        pins.update({'report:' + key: hashes['report:' + key] for key in widths})
        kwargs = dict(runtime=runtime, expected_sources=expected_sources,
                      expected_runtime_sha256=expected_runtime_sha256)
        producer = fresh.admit(spec_path, pins, root, **kwargs)
        reference = fresh.admit_reference(spec_path, pins, root, **kwargs)
        fresh.check_unchanged(producer)
        fresh.check_unchanged(reference)
        records.append({'rootKey': row['rootKey'], 'specPath': name, 'inputs': pins,
                        'producerIdentitySHA256': producer['identity_sha256'],
                        'referenceIdentitySHA256': reference['identity_sha256']})
    if ({key: common._sha(common._file(path)) for key, path in paths.items()} != hashes or
            common._sources(root) != expected_sources or
            common._sha(common._file(root / 'scripts' / 'fresh_c5_run.py')) != expected_entry_sha256):
        raise ValueError('Source input changed during preparation')
    result = {'mode': 'fresh-input-unverified', 'historicalAuthenticated': False,
              'statisticalInferenceAuthorized': False, 'populationInputs': hashes,
              'sources': expected_sources, 'runtime': observed_runtime, 'roots': records}
    stage('input-manifest.json', common.canonical_bytes(result))
    return result


def load_prepared(manifest_path, manifest_sha256, repository_root, *, runtime):
    """Read a pinned prepared roster and independently admit both readers.

    A caller-provided digest is an integrity check, not a provenance claim.
    The checkout and staged inputs must remain isolated from all writers.
    """
    from . import c5_fresh_population as population

    root = common._executing_root(repository_root)
    path = Path(manifest_path).resolve(strict=True)
    raw = common._file(path)
    if common._sha(raw) != manifest_sha256:
        raise ValueError('Prepared input manifest byte digest mismatch')
    manifest = common._strict_json(raw)
    fields = {'mode', 'historicalAuthenticated', 'statisticalInferenceAuthorized',
              'populationInputs', 'sources', 'runtime', 'roots'}
    if (type(manifest) is not dict or set(manifest) != fields or
            manifest['mode'] != 'fresh-input-unverified' or
            manifest['historicalAuthenticated'] is not False or
            manifest['statisticalInferenceAuthorized'] is not False or
            type(manifest['roots']) is not list or len(manifest['roots']) != 90):
        raise ValueError('Expected complete fresh input manifest')
    observed = common.sanitize_runtime(runtime)
    if observed != manifest['runtime'] or common._sources(root) != manifest['sources']:
        raise ValueError('Prepared input source or runtime identity changed')
    runtime_sha = common._sha(common.canonical_bytes(observed))
    contexts, reference_args, roster = [], {}, set()
    for record in manifest['roots']:
        keys = {'rootKey', 'specPath', 'inputs', 'producerIdentitySHA256',
                'referenceIdentitySHA256'}
        if (type(record) is not dict or set(record) != keys or
                type(record['specPath']) is not str or
                Path(record['specPath']).name != record['specPath']):
            raise ValueError('Invalid prepared root entry')
        spec_path = path.parent / record['specPath']
        args = dict(spec_path=spec_path, expected_inputs=record['inputs'],
                    repository_root=root, runtime=runtime, expected_sources=manifest['sources'],
                    expected_runtime_sha256=runtime_sha)
        context = fresh.admit(**args)
        ref = fresh.admit_reference(**args)
        if (context['identity_sha256'] != record['producerIdentitySHA256'] or
                ref['identity_sha256'] != record['referenceIdentitySHA256'] or
                context['identity_sha256'] != ref['identity_sha256']):
            raise ValueError('Prepared root identity mismatch')
        raw_root = context['spec']['rawRoot']
        key = (raw_root['scheduleVariant'], raw_root['mainIndex'])
        if (record['rootKey'] != [raw_root['conditionId'], *key] or
                raw_root['conditionId'] != 'C5' or key not in population.ROSTER or key in roster):
            raise ValueError('Duplicate or invalid prepared root')
        roster.add(key)
        contexts.append(context)
        reference_args[key] = args
    if roster != population.ROSTER or common._sha(common._file(path)) != manifest_sha256:
        raise ValueError('Incomplete or changed prepared roster')
    return contexts, reference_args
