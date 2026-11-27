"""Admission of caller-pinned inputs, without historical or inference authority.

Digests establish byte integrity, not provenance or numerical equivalence.
This module does not generate a prefix or authorize statistical inference.
"""
from pathlib import Path
import re

from . import c5_public_reproduction as common


def admit(spec_path, expected_inputs, repository_root, *, runtime, expected_sources,
          expected_runtime_sha256):
    """Construct separately validated producer and receiver snapshots.

    expected_inputs pins exactly spec, lookup, latent, and every report:<width>.
    expected_sources is the complete source digest map from the executing checkout;
    expected_runtime_sha256 pins the sanitized runtime observation, not its truth.
    Files must remain isolated from concurrent writers throughout use; call
    ``check_unchanged`` immediately before using an admitted context.
    """
    return _admit(spec_path, expected_inputs, repository_root, runtime=runtime,
                  expected_sources=expected_sources, expected_runtime_sha256=expected_runtime_sha256,
                  reference_only=False)


def admit_reference(spec_path, expected_inputs, repository_root, *, runtime, expected_sources,
                    expected_runtime_sha256):
    """Admit a receiver snapshot without invoking the producer validator."""
    return _admit(spec_path, expected_inputs, repository_root, runtime=runtime,
                  expected_sources=expected_sources, expected_runtime_sha256=expected_runtime_sha256,
                  reference_only=True)


def _admit(spec_path, expected_inputs, repository_root, *, runtime, expected_sources,
           expected_runtime_sha256, reference_only):
    root = common._executing_root(repository_root)
    spec_path = Path(spec_path).resolve(strict=True)
    raw = common._file(spec_path)
    spec = common._strict_json(raw)
    fields = {'rawRoot', 'config', 'initialDecision', 'design', 'lookupPath',
              'lookupSHA256', 'geometryCertificate', 'latentPath', 'reportPaths'}
    if type(spec) is not dict or set(spec) != fields:
        raise ValueError('Expected complete single-root scientific spec')
    if (type(spec['lookupPath']) is not str or not spec['lookupPath'] or
            type(spec['latentPath']) is not str or not spec['latentPath'] or
            type(spec['lookupSHA256']) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', spec['lookupSHA256']) or
            type(spec['reportPaths']) is not dict or not spec['reportPaths'] or
            any(type(k) is not str or not k or type(v) is not str or not v
                for k, v in spec['reportPaths'].items())):
        raise ValueError('Invalid input paths or lookup digest')
    base = spec_path.parent
    paths = {'spec': spec_path, 'lookup': (base / spec['lookupPath']).resolve(strict=True),
             'latent': (base / spec['latentPath']).resolve(strict=True)}
    paths.update({'report:' + k: (base / v).resolve(strict=True)
                  for k, v in spec['reportPaths'].items()})
    if (type(expected_inputs) is not dict or set(expected_inputs) != set(paths) or
            any(type(v) is not str or not re.fullmatch(r'[0-9a-f]{64}', v)
                for v in expected_inputs.values())):
        raise ValueError('Complete input byte digest map required')
    blobs = {key: common._file(path) for key, path in paths.items()}
    files = {key: common._sha(value) for key, value in blobs.items()}
    if files != expected_inputs or files['lookup'] != spec['lookupSHA256'] or blobs['spec'] != raw:
        raise ValueError('Input byte identity mismatch')
    # Bind the source-calling entry separately: common._sources covers src/ only.
    entry = root / 'scripts' / 'fresh_c5_run.py'
    paths['entry'] = entry
    files['entry'] = common._sha(common._file(entry))
    sources = common._sources(root)
    if type(expected_sources) is not dict or sources != expected_sources:
        raise ValueError('Source identity mismatch')
    observed_runtime = common.sanitize_runtime(runtime)
    if common._sha(common.canonical_bytes(observed_runtime)) != expected_runtime_sha256:
        raise ValueError('Runtime identity mismatch')
    common._require_readonly_snapshot({'paths': paths, 'identity': {'sources': sources},
                                       'repository_root': root})

    from . import c5_public_streams as streams
    from . import c5_public_trajectory_reference as receiver

    latent = common._strict_json(blobs['latent'])
    r_latent = streams.ValidatedLatentConfig(common._strict_json(common.canonical_bytes(latent)))
    reports = {width: blobs['report:' + width] for width in spec['reportPaths']}
    admission = {'lookupPath': str(paths['lookup']), 'lookupSHA256': spec['lookupSHA256'],
                 'geometryCertificate': spec['geometryCertificate']}
    rs = receiver.AdmittedSpec(common._strict_json(common.canonical_bytes(admission)), r_latent, reports)
    args = (spec['rawRoot'], spec['config'], spec['initialDecision'])
    receiver.validate_spec(*args, r_latent, spec['design'], rs)
    if not reference_only:
        from . import c5_public_trajectory as producer
        p_latent = streams.ValidatedLatentConfig(common._strict_json(common.canonical_bytes(latent)))
        ps = producer.AdmittedSpec(common._strict_json(common.canonical_bytes(admission)), p_latent, reports)
        producer._validate_root(*args, p_latent, spec['design'], ps)
    width = float(spec['config']['task']['report_width']).hex()
    identity = {'mode': 'fresh-input-unverified', 'inputs': files, 'sources': sources,
                'runtime': observed_runtime}
    context = {'spec': spec, 'rs': rs, 'r_latent': r_latent, 'r_table': dict(rs.table)[width],
               'paths': paths, 'repository_root': root, 'identity': identity,
               'identity_sha256': common._sha(common.canonical_bytes(identity)),
               'spec_snapshot_sha256': common._sha(common.canonical_bytes(spec)),
               'historicalAuthenticated': False, 'statisticalInferenceAuthorized': False}
    if not reference_only:
        context.update(ps=ps, p_latent=p_latent, p_table=dict(ps.table)[width])
    common._require_readonly_snapshot(context)
    check_unchanged(context)
    return context


def check_unchanged(context):
    """Detect file/source replacement; this is not a filesystem lock."""
    if (common._sha(common.canonical_bytes(context['spec'])) != context['spec_snapshot_sha256'] or
            common._sha(common.canonical_bytes(context['identity'])) != context['identity_sha256']):
        raise ValueError('Admission snapshot changed')
    common._require_readonly_snapshot(context)
    common._unchanged(context)
