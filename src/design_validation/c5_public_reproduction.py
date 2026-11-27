"""C5 source-calling orchestration boundaries, not an authority for results.

The supplied prefix is non-regenerable data. A caller-provided digest checks
transport integrity only; it does not authenticate the accepted historical data.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager, ExitStack
from pathlib import Path
import re
import subprocess
import tempfile


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def member_id(replicate, variant, main):
    if (type(replicate) is not int or not 1 <= replicate <= 1840 or
            type(variant) is not int or not 0 <= variant < 9 or
            type(main) is not int or not 1 <= main <= 10):
        raise ValueError('Expected rep 1..1840, schedule 0..8 and main 1..10')
    return f'rep-{replicate:04d}-v{variant}-m{main:02d}'


def write_once(path, value):
    """Atomically publish a JSON checkpoint without replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    fd, tmp = tempfile.mkstemp(prefix='.c5-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)  # Atomic no-clobber, including simultaneous writers.
    finally:
        os.unlink(tmp)
    return hashlib.sha256(payload).hexdigest()


def inspect_checkpoint(path, expected_id, expected_input_sha256):
    """Check identity and transport binding, not scientific admission."""
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if (canonical_bytes(value) != raw or type(value) is not dict or
            set(value) != {'memberId', 'inputSHA256', 'payload'} or
            value['memberId'] != expected_id or value['inputSHA256'] != expected_input_sha256):
        raise ValueError('Checkpoint identity, input binding, or canonical bytes differ')
    return value


def prepare(prefix, prefix_sha256, lookup, lookup_sha256, repository_root):
    """Check supplied bytes and scientific shape without claiming authority."""
    repository_root = _executing_root(repository_root)
    from . import c5_public_inputs as inputs
    fixed = inputs.load_fixed_inputs(lookup, lookup_sha256)
    supplied = inputs.load_historical_prefix(prefix, prefix_sha256,
                                             repository_root=repository_root, lookup=fixed)
    return {'status': 'input_shape_checked_not_authenticated',
            'target': inputs.TARGET, 'prefixSHA256': prefix_sha256,
            'lookupSHA256': lookup_sha256, 'prefixRows': supplied['rootCount'],
            'lookupRoots': len(fixed['roots']), 'historicalEquivalence': False,
            'samplingAuthorized': False, 'finalAuthorized': False}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file(path):
    return Path(path).read_bytes()


def _executing_root(repository_root):
    actual = Path(__file__).resolve(strict=True).parents[2]
    supplied = Path(repository_root).resolve(strict=True)
    if supplied != actual:
        raise ValueError('Repository root must be the executing source checkout')
    return actual


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('Nonfinite JSON value: ' + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def _sources(repository_root):
    root = Path(repository_root)
    base = root / 'src'
    return {str(p.relative_to(root)): _sha(_file(p)) for p in sorted(base.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.cjs', '.R')}


def sanitize_runtime(runtime):
    """Bind an unverified observation; it is not numerical or final authority."""
    if (type(runtime) is not dict or set(runtime) != {'schema', 'identity', 'acceptedProfileCertified',
            'historicalPrefixRegenerationCertified', 'observation', 'observationSHA256', 'limitations'}
            or runtime.get('schema') != 'c5-new-numerical-profile-observation-v2'
            or runtime.get('identity') != 'new-unverified'
            or runtime.get('acceptedProfileCertified') is not False
            or runtime.get('historicalPrefixRegenerationCertified') is not False
            or type(runtime.get('observation')) is not dict):
        raise ValueError('Explicit unverified bootstrap observation required')
    observation = json.loads(canonical_bytes(runtime['observation']))
    if _sha(canonical_bytes(observation)) != runtime.get('observationSHA256'):
        raise ValueError('Bootstrap observation digest mismatch')
    def fields(value, names):
        if type(value) is not dict or set(value) != set(names):
            raise ValueError('Unexpected bootstrap observation fields')

    def token(value):
        return type(value) is str and bool(re.fullmatch(r'[A-Za-z0-9_.+-]+', value))

    def digest(value):
        return value is None or (type(value) is str and bool(re.fullmatch(r'[0-9a-f]{64}', value)))

    fields(observation, ('python', 'machine', 'pythonExecutableSHA256', 'floatInfo',
           'roundingRawBeforeNumpy', 'roundingRawAfterNumpy', 'x87Control', 'mxcsrControl',
           'numpyAlreadyImported', 'environmentBeforeNumpy', 'numpy', 'mappedBlasFiles',
           'selectedBlasCore', 'actualBlasThreads', 'libmResolvedSymbols'))
    if (not token(observation['python']) or not token(observation['machine']) or
            not digest(observation['pythonExecutableSHA256']) or
            type(observation['numpyAlreadyImported']) is not bool or
            any(observation[k] is not None for k in ('x87Control', 'mxcsrControl',
                'selectedBlasCore', 'actualBlasThreads', 'libmResolvedSymbols')) or
            any(observation[k] is not None and type(observation[k]) is not int
                for k in ('roundingRawBeforeNumpy', 'roundingRawAfterNumpy'))):
        raise ValueError('Invalid path-free runtime observation')
    floating = observation['floatInfo']
    fields(floating, ('radix', 'mantissaBits', 'maxExponent'))
    if any(type(value) is not int for value in floating.values()):
        raise ValueError('Invalid floating-point observation')
    environment = observation['environmentBeforeNumpy']
    fields(environment, ('NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS',
           'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'GLIBC_TUNABLES', 'LD_PRELOAD', 'LD_AUDIT',
           'LD_HWCAP_MASK', 'LD_LIBRARY_PATH'))
    for value in environment.values():
        if (type(value) is not dict or set(value) != {'present', 'valueSHA256'} or
                type(value['present']) is not bool or
                (value['present'] and (type(value['valueSHA256']) is not str or
                 len(value['valueSHA256']) != 64 or
                 any(c not in '0123456789abcdef' for c in value['valueSHA256']))) or
                (not value['present'] and value['valueSHA256'] is not None)):
            raise ValueError('Bootstrap controls must be path-free digests')
    numpy = observation['numpy']
    fields(numpy, ('version', 'fileSHA256', 'coreBinarySHA256', 'baseline', 'compiledDispatch', 'features'))
    if (not token(numpy['version']) or not digest(numpy['fileSHA256']) or
            not digest(numpy['coreBinarySHA256']) or
            any(type(numpy[k]) is not list or not all(token(x) for x in numpy[k])
                for k in ('baseline', 'compiledDispatch')) or
            (numpy['features'] is not None and
             (type(numpy['features']) is not dict or
              not all(token(k) and type(v) is bool for k, v in numpy['features'].items())))):
        raise ValueError('Invalid NumPy observation')
    blas = observation['mappedBlasFiles']
    if blas is not None:
        if type(blas) is not list:
            raise ValueError('Invalid BLAS observation')
        for item in blas:
            fields(item, ('sha256',))
            if not digest(item['sha256']):
                raise ValueError('Invalid BLAS digest')
    limitations = runtime['limitations']
    if (type(limitations) is not list or not all(
            type(x) is str and '\\' not in x and
            ('/' not in x or x == 'CPU/CPUID and loaded numerical closure not authenticated')
            for x in limitations)):
        raise ValueError('Invalid runtime limitations')
    return {'schema': 'c5-sanitized-runtime-observation-v1', 'identity': 'new-unverified',
            'acceptedProfileCertified': False, 'observation': observation,
            'observationSHA256': _sha(canonical_bytes(observation))}


def load_context(spec_path, spec_sha256, prefix, prefix_sha256, repository_root, *, runtime):
    """Load a pinned, single-root scientific fixture into separate P/R snapshots.

    Paths in the fixture are relative to that fixture. Caller pins establish
    integrity, not authority. The source and runtime fingerprint is a resume
    guard, not a proof of numerical equivalence or a canonical profile.
    """
    repository_root = _executing_root(repository_root)
    spec_path = Path(spec_path).resolve(strict=True)
    raw = _file(spec_path)
    if _sha(raw) != spec_sha256:
        raise ValueError('Scientific spec byte digest mismatch')
    spec = _strict_json(raw)
    expected = {'rawRoot', 'config', 'initialDecision', 'design', 'lookupPath',
                'lookupSHA256', 'geometryCertificate', 'latentPath', 'reportPaths'}
    if type(spec) is not dict or set(spec) != expected:
        raise ValueError('Expected complete single-root scientific spec')
    if (type(spec['lookupPath']) is not str or not spec['lookupPath'] or
            type(spec['latentPath']) is not str or not spec['latentPath'] or
            type(spec['lookupSHA256']) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', spec['lookupSHA256']) or
            type(spec['reportPaths']) is not dict or not spec['reportPaths'] or
            any(type(k) is not str or type(v) is not str or not v
                for k, v in spec['reportPaths'].items())):
        raise ValueError('Invalid spec input paths or lookup digest')
    base = spec_path.parent
    lookup = (base / spec['lookupPath']).resolve(strict=True)
    latent_path = (base / spec['latentPath']).resolve(strict=True)
    reports = {width: (base / path).resolve(strict=True) for width, path in spec['reportPaths'].items()}
    prepare(prefix, prefix_sha256, lookup, spec['lookupSHA256'], repository_root)
    paths = {'spec': spec_path, 'lookup': lookup, 'latent': latent_path,
             'prefix': Path(prefix).resolve(strict=True)}
    paths.update({'report:' + width: path for width, path in reports.items()})
    files = {key: _sha(_file(path)) for key, path in paths.items()}
    if files['spec'] != spec_sha256 or files['lookup'] != spec['lookupSHA256'] or files['prefix'] != prefix_sha256:
        raise ValueError('Pinned input changed during admission')
    from . import c5_public_streams as streams
    from . import c5_public_trajectory as producer
    from . import c5_public_trajectory_reference as receiver
    latent_json = _strict_json(_file(latent_path))
    report_bytes = {width: _file(path) for width, path in reports.items()}
    ps_latent = streams.ValidatedLatentConfig(json.loads(canonical_bytes(latent_json)))
    rs_latent = streams.ValidatedLatentConfig(json.loads(canonical_bytes(latent_json)))
    admission = {'lookupPath': str(lookup), 'lookupSHA256': spec['lookupSHA256'],
                 'geometryCertificate': spec['geometryCertificate']}
    ps = producer.AdmittedSpec(json.loads(canonical_bytes(admission)), ps_latent, report_bytes)
    rs = receiver.AdmittedSpec(json.loads(canonical_bytes(admission)), rs_latent, report_bytes)
    width = float(spec['config']['task']['report_width']).hex()
    identity = {'mode': 'new-unverified', 'inputs': files, 'sources': _sources(repository_root),
                'runtime': sanitize_runtime(runtime)}
    context = {'spec': spec, 'ps': ps, 'rs': rs, 'p_latent': ps_latent, 'r_latent': rs_latent,
               'p_table': dict(ps.table)[width], 'r_table': dict(rs.table)[width],
               'producer': producer, 'receiver': receiver, 'paths': paths,
               'repository_root': Path(repository_root), 'identity': identity,
               'identity_sha256': _sha(canonical_bytes(identity))}
    _unchanged(context)
    return context


def _unchanged(context):
    _executing_root(context['repository_root'])
    if ({key: _sha(_file(path)) for key, path in context['paths'].items()} !=
            context['identity']['inputs'] or
            _sources(context['repository_root']) != context['identity']['sources']):
        raise ValueError('Input or source changed during run; do not reuse this checkpoint')


def _require_readonly_snapshot(context):
    """Check modes, not mount immutability; callers must isolate the staged tree."""
    root = _executing_root(context['repository_root'])
    paths = list(context['paths'].values())
    paths += [root / name for name in context['identity']['sources']]
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise ValueError('Population source and inputs must be staged as read-only files')


def _require_ignored(path, repository):
    resolved = Path(path).resolve()
    if resolved.is_relative_to(repository):
        check = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-q', '--', str(resolved)],
                               capture_output=True, check=False)
        if check.returncode != 0:
            raise ValueError('Checkpoint or lock inside checkout must be git-ignored')


def _checked_member_paths(output, repository, root, reps, *, block_size=256):
    """Check every real path with bounded NUL-delimited Git requests."""
    for offset in range(0, len(reps), block_size):
        block = []
        inside = set()
        for rep in reps[offset:offset + block_size]:
            name = member_id(rep, root['scheduleVariant'], root['mainIndex'])
            path = Path(output) / (name + '.json')
            if path.is_symlink():
                raise ValueError('Checkpoint symlink refused')
            resolved = path.resolve()
            block.append((rep, name, path, resolved))
            if resolved.is_relative_to(repository):
                inside.add(os.fsencode(str(resolved)))
        if inside:
            request = b''.join(item + b'\0' for item in sorted(inside))
            check = subprocess.run(['git', '-C', str(repository), 'check-ignore', '-z', '--stdin'],
                                   input=request, capture_output=True, check=False)
            if check.returncode not in (0, 1):
                raise ValueError('Could not verify checkpoint ignore rules')
            returned = check.stdout
            if not isinstance(returned, bytes) or (returned and not returned.endswith(b'\0')):
                raise ValueError('Invalid Git ignore response')
            ignored = set(returned[:-1].split(b'\0')) if returned else set()
            if ignored != inside:
                raise ValueError('Checkpoint inside checkout must be git-ignored')
        yield from block


def _check_member_path(path, resolved):
    if path.is_symlink() or path.resolve() != resolved:
        raise ValueError('Checkpoint path changed or became a symlink')


@contextmanager
def _output_lock(output, repository, *, read_only=False):
    """Release ordinary read-only refusals; retain interrupted or mutating failures."""
    output = Path(output).resolve()
    marker = output / '.c5-population-active'
    _require_ignored(marker, repository)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open('x', encoding='ascii') as handle:
            handle.write('Unfinished C5 operation: isolate and review before reuse.\n')
    except FileExistsError as exc:
        raise ValueError('Unfinished or concurrent C5 operation; do not reuse output') from exc
    try:
        yield output
    except Exception:
        if read_only:
            marker.unlink()
        raise
    else:
        marker.unlink()


def _admit_population_prefix(mode, lookup, lookup_sha256, prefix, prefix_sha256, repository_root):
    if mode == 'accepted-prefix':
        from . import c5_public_authority as authority
        if (lookup_sha256 != authority.ADOPTED_INITIAL_SHA256 or
                prefix_sha256 != authority.ACCEPTED_PREFIX_SHA256):
            raise ValueError('Accepted-prefix mode requires both adopted byte pins')
        return authority.load_adopted_inputs(initial_lookup=lookup, historical_prefix=prefix)
    if mode == 'caller-prefix':
        return prepare(prefix, prefix_sha256, lookup, lookup_sha256, repository_root)
    raise ValueError('Choose accepted-prefix or caller-prefix input mode')


def load_population(manifest_path, manifest_sha256, prefix, prefix_sha256, repository_root, *, runtime,
                    input_mode):
    """Assemble all 90 roots from public-generated inputs, without authority claims.

    The manifest names existing input files relative to itself. The prepared
    roots must be the complete C5 output of the public personal-time generator.
    Assembly checks fixed integer counts and report-product identities, not
    historical numerical dispatch or the caller's lookup/prefix authority.
    """
    repository_root = _executing_root(repository_root)
    source_hashes = _sources(repository_root)
    from . import c5_input_assembly as assembly
    from . import c5_public_inputs as inputs
    from . import c5_public_streams as streams
    from . import c5_public_trajectory as producer
    from . import c5_public_trajectory_reference as receiver
    from . import personal_time_inputs as personal
    from . import personal_time_sensitivity as sensitivity

    manifest_path = Path(manifest_path).resolve(strict=True)
    raw = _file(manifest_path)
    if _sha(raw) != manifest_sha256:
        raise ValueError('Population manifest byte digest mismatch')
    manifest = _strict_json(raw)
    fields = {'recipePath', 'rootsPath', 'lookupPath', 'lookupSHA256', 'latentPath', 'reportPaths'}
    if type(manifest) is not dict or set(manifest) != fields:
        raise ValueError('Expected complete population input manifest')
    if (any(type(manifest[k]) is not str or not manifest[k]
            for k in ('recipePath', 'rootsPath', 'lookupPath', 'latentPath')) or
            type(manifest['lookupSHA256']) is not str or
            not re.fullmatch(r'[0-9a-f]{64}', manifest['lookupSHA256']) or
            type(manifest['reportPaths']) is not dict or not manifest['reportPaths'] or
            any(type(k) is not str or type(v) is not str or not v
                for k, v in manifest['reportPaths'].items())):
        raise ValueError('Invalid population paths or lookup digest')
    base = manifest_path.parent
    paths = {name: (base / manifest[key]).resolve(strict=True) for name, key in
             (('recipe', 'recipePath'), ('roots', 'rootsPath'), ('lookup', 'lookupPath'),
              ('latent', 'latentPath'))}
    paths.update({'report:' + width: (base / path).resolve(strict=True)
                  for width, path in manifest['reportPaths'].items()})
    paths.update({'manifest': manifest_path, 'prefix': Path(prefix).resolve(strict=True)})
    scientific_context = sensitivity.load_context(paths['recipe'])
    recipe_dir = paths['recipe'].parent
    reference = recipe_dir / scientific_context['recipe']['reference']
    paths['reference'] = reference.resolve(strict=True)
    for name in personal.INPUT_SHA256:
        paths['reference:' + name] = (reference.parent /
            scientific_context['reference']['recipe'][name]).resolve(strict=True)
    files = {key: _sha(_file(path)) for key, path in paths.items()}
    if (files['manifest'] != manifest_sha256 or files['lookup'] != manifest['lookupSHA256']
            or files['prefix'] != prefix_sha256):
        raise ValueError('Pinned population input changed during admission')
    selected = personal.selection([dict(zip(personal.FIELDS, key)) for key in inputs.ROSTER])
    roots = personal.validate_roots(_strict_json(_file(paths['roots'])), selected,
                                    scientific_context['reference'])
    reports = {width: _file(paths['report:' + width]) for width in manifest['reportPaths']}
    latent_bytes = _file(paths['latent'])
    assembled = assembly.assemble(paths['lookup'], manifest['lookupSHA256'], latent_bytes,
                                  roots, scientific_context, reports)
    _admit_population_prefix(input_mode, paths['lookup'], manifest['lookupSHA256'],
                             paths['prefix'], prefix_sha256, repository_root)
    runtime_identity = sanitize_runtime(runtime)
    p_latent = streams.ValidatedLatentConfig(_strict_json(latent_bytes))
    r_latent = streams.ValidatedLatentConfig(_strict_json(latent_bytes))
    admitted = []
    for v in range(9):
        cert = assembled['geometryCertificates'][v]
        schedule_rows = assembled['lookup']['roots'][10*v:10*v+10]
        widths = {float(item['reportWidth']).hex() for item in schedule_rows}
        schedule_reports = {width: reports[width] for width in widths}
        admission = {'lookupPath': str(paths['lookup']), 'lookupSHA256': manifest['lookupSHA256'],
                     'geometryCertificate': cert}
        admitted.append((producer.AdmittedSpec(admission, p_latent, schedule_reports),
                         receiver.AdmittedSpec(admission, r_latent, schedule_reports)))
    contexts = []
    for position, (root, row) in enumerate(zip(roots, assembled['lookup']['roots'])):
        cert = assembled['geometryCertificates'][position // 10]
        v = position // 10
        ps, rs = admitted[v]
        spec = {'rawRoot': root, 'config': row['config'], 'initialDecision': row['decision'],
                'design': cert['design']}
        identity = {'mode': 'new-unverified', 'inputMode': input_mode, 'inputs': files, 'sources': source_hashes,
                    'runtime': runtime_identity, 'rootKey': row['rootKey']}
        width = float(row['reportWidth']).hex()
        context = {'spec': spec, 'ps': ps, 'rs': rs, 'p_latent': p_latent, 'r_latent': r_latent,
                   'p_table': dict(ps.table)[width], 'r_table': dict(rs.table)[width],
                   'producer': producer, 'receiver': receiver, 'paths': paths,
                   'repository_root': Path(repository_root), 'identity': identity,
                   'identity_sha256': _sha(canonical_bytes(identity))}
        contexts.append(context)
    _unchanged(contexts[0])
    _require_readonly_snapshot(contexts[0])
    return {'contexts': contexts, 'lookup': assembled['lookup'], 'prefix': paths['prefix'],
            'prefix_sha256': prefix_sha256, 'repository_root': Path(repository_root), 'input_mode': input_mode}


def run_population(population, output, replicates, *, verify_only=False):
    """Run a batch against an externally isolated, read-only staged snapshot.

    Boundary hashes detect persistent changes, not transient changes followed by
    restoration. The caller must prevent writes or replacements throughout the
    batch. Interrupted operations and failed writing batches retain a marker;
    ordinary verify-only refusals release their locks.
    """
    reps = tuple(replicates)
    if len(population['contexts']) != 90:
        raise ValueError('Complete 90-root population required')
    context = population['contexts'][0]
    _require_readonly_snapshot(context)
    _unchanged(context)
    repository = _executing_root(context['repository_root'])
    totals = {'created': 0, 'verifiedExisting': 0}
    with _output_lock(output, repository, read_only=verify_only) as locked_output:
        for context in population['contexts']:
            result = _run_members(context, locked_output, reps, verify_only=verify_only, batch=True)
            for key in totals:
                totals[key] += result[key]
        _require_readonly_snapshot(population['contexts'][0])
        _unchanged(population['contexts'][0])
    return {'status': 'new-unverified-independent-replay', 'rootCount': len(population['contexts']),
            'replicateCount': len(reps), **totals, 'historicalEquivalence': False,
            'inputMode': population['input_mode'], 'finalAuthorized': False}


def _shard_directories(output, shard_dirs):
    if (output is None) == (shard_dirs is None):
        raise ValueError('Choose exactly one output or explicit shard directory list')
    supplied = [output] if shard_dirs is None else list(shard_dirs)
    if not supplied:
        raise ValueError('At least one shard directory is required')
    paths, inodes = [], set()
    for item in supplied:
        path = Path(item)
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Each shard must be an existing non-symlink directory')
        path = path.resolve(strict=True)
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes or any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths):
            raise ValueError('Duplicate or overlapping shard directories refused')
        inodes.add(inode)
        paths.append(path)
    return sorted(paths)


@contextmanager
def _shard_locks(paths, repository):
    stack = ExitStack()
    try:
        for path in paths:
            stack.enter_context(_output_lock(path, repository, read_only=True))
    except Exception:
        # A failed acquisition has not started the audit; release only our locks.
        stack.close()
        raise
    except BaseException as exc:
        stack.__exit__(type(exc), exc, exc.__traceback__)
        raise
    with stack:
        yield


def _shard_roster(paths, contexts, *, replicates=range(2, 1841)):
    """Index exact expected names; no shard contents are copied or replaced."""
    reps = set(replicates)
    roots = {}
    for context in contexts:
        root = context['spec']['rawRoot']
        key = (root['scheduleVariant'], root['mainIndex'])
        if key in roots:
            raise ValueError('Duplicate population root')
        roots[key] = context
    roster = {}
    pattern = re.compile(r'rep-(\d{4})-v(\d)-m(\d{2})\.json\Z')
    for directory in paths:
        for path in directory.iterdir():
            if path.name == '.c5-population-active':
                continue
            match = pattern.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                raise ValueError('Foreign or unsafe shard entry: ' + path.name)
            rep, variant, main = map(int, match.groups())
            if rep not in reps or (variant, main) not in roots or path.name != member_id(rep, variant, main) + '.json':
                raise ValueError('Foreign checkpoint identity: ' + path.name)
            if path.name in roster:
                raise ValueError('Duplicate checkpoint across shards: ' + path.name)
            roster[path.name] = path
    if len(roster) != len(reps) * len(roots):
        raise ValueError('Missing checkpoint in shard roster')
    return roster


def _shard_payload(path, name, context):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Unsafe shard checkpoint: ' + name)
    saved = inspect_checkpoint(path, name, context['identity_sha256'])
    payload = saved['payload']
    if (type(payload) is not dict or set(payload) != {'result', 'resultSHA256', 'identity'} or
            payload['identity'] != context['identity'] or
            _sha(canonical_bytes(payload['result'])) != payload['resultSHA256']):
        raise ValueError('Checkpoint result digest mismatch: ' + name)
    return payload['result']


def audit_population(population, output=None, *, shard_dirs=None, temp_dir=None):
    """Replay the complete roster and exact-reduce provisionally, never certify it."""
    from . import c5_final
    contexts = population['contexts']
    if len(contexts) != 90:
        raise ValueError('Complete 90-root population required')
    _require_readonly_snapshot(contexts[0])
    _unchanged(contexts[0])
    paths = _shard_directories(output, shard_dirs)
    roster = None
    def future():
        for rep in range(2, 1841):
            for context in contexts:
                spec = context['spec']
                root = spec['rawRoot']
                name = member_id(rep, root['scheduleVariant'], root['mainIndex'])
                result = _shard_payload(roster[name + '.json'], name, context)
                yield {'root': root, 'cfg': spec['config'], 'initial_decision': spec['initialDecision'],
                       'latent_config': context['r_latent'], 'report_bytes': context['r_table'],
                       'design': spec['design'], 'replicate': rep,
                       'result_bytes': canonical_bytes(result), 'spec': context['rs']}

    with _shard_locks(paths, _executing_root(contexts[0]['repository_root'])):
        roster = _shard_roster(paths, contexts)
        result = c5_final.audit_population(population['prefix'], population['prefix_sha256'],
                                           repository_root=population['repository_root'],
                                           lookup=population['lookup'], future=future(), temp_dir=temp_dir)
        _unchanged(contexts[0])
        _require_readonly_snapshot(contexts[0])
    result['inputMode'] = population['input_mode']
    result['shardCount'] = len(paths)
    return result


def _receive(context, rep, result):
    spec = json.loads(canonical_bytes(context['spec']))
    receipt = context['receiver'].receive_root(
        spec['rawRoot'], spec['config'], spec['initialDecision'], context['r_latent'],
        context['r_table'], spec['design'], rep, result, spec=context['rs'])
    if (type(receipt) is not dict or receipt.get('status') != 'independent_replay_pass' or
            receipt.get('rootKey') != ['C5', spec['rawRoot']['scheduleVariant'], spec['rawRoot']['mainIndex']] or
            receipt.get('replicateId') != rep or receipt.get('historicalEquivalence') is not False or
            receipt.get('authority') != 'caller-supplied-inputs-not-authenticated'):
        raise ValueError('Unexpected independent receiver response')
    return receipt


def run_members(context, output, replicates, *, verify_only=False):
    """Replay reused members; an unverified profile never authorizes final results.

    Dynamic runtime drift is not authenticated; canonical long-run work remains
    fail-closed rather than relying on this observation as exact identity.
    """
    repository = _executing_root(context['repository_root'])
    with _output_lock(output, repository, read_only=verify_only) as locked_output:
        return _run_members(context, locked_output, replicates, verify_only=verify_only, batch=False)


def _run_members(context, output, replicates, *, verify_only, batch):
    spec = context['spec']
    root = spec['rawRoot']
    reps = list(replicates)
    if len(reps) != len(set(reps)):
        raise ValueError('Duplicate replicate requested')
    for rep in reps:
        if type(rep) is not int or not 2 <= rep <= 1840:
            raise ValueError('Future replicate must be 2..1840; rep1 must be supplied')
        member_id(rep, root['scheduleVariant'], root['mainIndex'])
    created = reused = 0
    repository = Path(context['repository_root']).resolve(strict=True)
    for rep, name, path, resolved in _checked_member_paths(output, repository, root, reps):
        _check_member_path(path, resolved)
        if not batch:
            _unchanged(context)
        if path.exists():
            saved = inspect_checkpoint(path, name, context['identity_sha256'])
            payload = saved['payload']
            if (type(payload) is not dict or set(payload) != {'result', 'resultSHA256', 'identity'} or
                    payload['identity'] != context['identity'] or
                    _sha(canonical_bytes(payload['result'])) != payload['resultSHA256']):
                raise ValueError('Checkpoint result digest mismatch')
            _receive(context, rep, payload['result'])
            if not batch:
                _unchanged(context)
            reused += 1
            continue
        if verify_only:
            raise ValueError('Missing checkpoint: ' + name)
        data = json.loads(canonical_bytes(spec))
        result = context['producer'].produce_root(
            data['rawRoot'], data['config'], data['initialDecision'], context['p_latent'],
            context['p_table'], data['design'], rep, spec=context['ps'])
        _receive(context, rep, result)
        if not batch:
            _unchanged(context)
        value = {'memberId': name, 'inputSHA256': context['identity_sha256'],
                 'payload': {'result': result, 'resultSHA256': _sha(canonical_bytes(result)),
                             'identity': context['identity']}}
        _check_member_path(path, resolved)
        try:
            write_once(path, value)
            created += 1
        except FileExistsError:
            # Another writer won; its output is reusable only after full replay.
            saved = inspect_checkpoint(path, name, context['identity_sha256'])
            payload = saved['payload']
            if (type(payload) is not dict or set(payload) != {'result', 'resultSHA256', 'identity'} or
                    payload['identity'] != context['identity'] or
                    _sha(canonical_bytes(payload['result'])) != payload['resultSHA256']):
                raise ValueError('Competing checkpoint result digest mismatch')
            _receive(context, rep, payload['result'])
            if not batch:
                _unchanged(context)
            reused += 1
    return {'status': 'new-unverified-independent-replay', 'created': created, 'verifiedExisting': reused,
            'inputIdentitySHA256': context['identity_sha256'], 'historicalEquivalence': False,
            'finalAuthorized': False}


def unavailable(stage):
    gaps = {
        'sample': 'authenticate the fixed lookup and historical prefix, bind the numerical profile and complete the portable admitted-spec loader before drawing',
        'verify': 'implement the portable admitted-spec loader and independent trajectory replay binding; checkpoint bytes alone are not acceptance',
        'audit': 'complete independently admitted prefix and future-member replay before the full key and integer-seed audit',
        'final': 'authenticate prefix, lookup, law and numerical-profile bindings and connect the independent confidence/high-zero authority',
        'redraw': 'provide a certified final-result schema and its independently validated authority binding',
    }
    raise RuntimeError(f'{stage} unavailable: {gaps[stage]}; no scientific result was produced')
