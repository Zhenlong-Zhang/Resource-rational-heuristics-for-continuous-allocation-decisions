"""Conservative runtime admission for C5 numerical work.

Load this file directly, not through ``src``: the package imports NumPy.
Canonical admission is disabled until an authenticated public export is pinned.
Observation imports NumPy but does not alter environment variables or FP state.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


class RuntimeProfileError(ValueError):
    """The requested numerical identity cannot be admitted."""


_CONTROLS = ('NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE',
             'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'GLIBC_TUNABLES', 'LD_PRELOAD', 'LD_AUDIT', 'LD_HWCAP_MASK',
             'LD_LIBRARY_PATH')

# Set only after the path-free export has been authenticated independently.
_PUBLIC_PROFILE_SHA256 = None
_HELPER_SHA256 = 'a7d56b30c8232256d2c20b1000b3abe0401f8fb6dc92b1b16f674cc8e7a7fb5e'


def _require(condition, message):
    if not condition:
        raise RuntimeProfileError(message)


def export_scientific_profile(raw, source_sha256):
    """Return deterministic path-free bytes; source SHA is provenance, not approval.

    This pure projection does not load native code, import NumPy, or certify its
    input. The caller must independently authenticate the source and export.
    """
    _require(isinstance(raw, bytes) and isinstance(source_sha256, str) and
             len(source_sha256) == 64 and hashlib.sha256(raw).hexdigest() == source_sha256,
             'Source profile bytes/hash mismatch')
    try:
        source = json.loads(raw)
        _require(source['schema'] == 'c5-public-port-profile-evidence-v1', 'Source schema mismatch')
        original = source['actualNumericFiles']
        _require(isinstance(original, dict) and original, 'Missing numeric file roster')
        _require(all(isinstance(k, str) and k.startswith('/') and isinstance(v, str) and
                     len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                     for k, v in original.items()), 'Invalid numeric file roster')
        names = {path: 'file-%04d' % i for i, path in enumerate(sorted(original))}
        pick = lambda obj, keys: {key: obj[key] for key in keys}
        controls = pick(source['controls'], ('mxcsrControl', 'numpyBaseline', 'numpyDisable',
                         'openblasCore', 'openblasThreads', 'rounding', 'x87Control'))
        numpy = pick(source['numpy'], ('version', 'baseline', 'compiledDispatch', 'features'))
        cpu = source['cpuSignature']
        signature = {'cpu': pick(cpu['cpu'], ('vendor_id', 'cpu family', 'model', 'stepping',
                                              'model name', 'microcode', 'flags')),
                     'cpuid': [pick(row, ('leaf', 'subleaf', 'available', 'registers'))
                               for row in cpu['cpuid']]}
        blas = source['openblas']
        queries = {key: pick(blas['queries'][key], ('symbol', 'value')) for key in
                   ('openblas_get_config', 'openblas_get_corename', 'openblas_get_num_threads')}
        symbols = []
        for row in source['libmResolvedSymbols']:
            item = pick(row, ('name', 'version', 'sha256', 'implementationOffset'))
            item['path'] = names[row['path']]
            symbols.append(item)
        exported = {'schema': 'c5-portable-scientific-profile-v1', 'sourceProfileSHA256': source_sha256,
                    'python': source['python'], 'machine': source['machine'],
                    'controls': controls, 'numpy': numpy, 'cpuSignature': signature,
                    'floatingPoint': pick(source['floatingPoint'], ('mxcsrControl', 'x87Control',
                                                                    'rounding', 'nearest')),
                    'files': {names[path]: original[path] for path in sorted(original)},
                    'executable': names[source['executable']['path']],
                    'openblas': {'path': names[blas['path']], 'queries': queries},
                    'libmResolvedSymbols': symbols,
                    'libmResolverCoverage': 'default_symbols_only_not_complete_ELF_relocations'}
        def no_paths(value):
            if isinstance(value, str):
                return '/' not in value and '\\' not in value
            if isinstance(value, dict):
                return all(no_paths(k) and no_paths(v) for k, v in value.items())
            if isinstance(value, list):
                return all(no_paths(v) for v in value)
            return value is None or type(value) in (bool, int)
        _require(no_paths(exported), 'Unexpected path or non-scientific value in selected fields')
        return (json.dumps(exported, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeProfileError):
            raise
        raise RuntimeProfileError('Malformed source profile') from exc


def _maps():
    try:
        lines = Path('/proc/self/maps').read_text().splitlines()
        result = set()
        for line in lines:
            fields = line.split(None, 5)
            if len(fields) < 6 or not fields[5].startswith('/'):
                continue
            path = fields[5]
            if ('x' not in fields[1] and '.so' not in Path(path).name and
                    'ld-linux' not in Path(path).name):
                continue
            _require(not path.endswith(' (deleted)'), 'Deleted mapped binary cannot be authenticated')
            stat = os.stat(path)
            major, minor = (int(x, 16) for x in fields[3].split(':'))
            _require(stat.st_ino == int(fields[4]) and os.major(stat.st_dev) == major and
                     os.minor(stat.st_dev) == minor, 'Mapped binary no longer matches its file')
            result.add(str(Path(path).resolve(strict=True)))
        return result
    except (OSError, ValueError, UnicodeError) as exc:
        raise RuntimeProfileError('Cannot authenticate Linux loaded DSO map') from exc


def _cpu_signature(lib):
    try:
        affinity = os.sched_getaffinity(0)
        _require(len(affinity) == 1, 'Conditional admission requires single-CPU affinity')
        cpu = next(iter(affinity))
        paragraphs = Path('/proc/cpuinfo').read_text().strip().split('\n\n')
        rows = [dict(line.split(':', 1) for line in p.splitlines() if ':' in line)
                for p in paragraphs]
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
        matching = [row for row in rows if row.get('processor') == str(cpu)]
        _require(len(matching) == 1, 'Actual CPU metadata unavailable')
        keys = ('vendor_id', 'cpu family', 'model', 'stepping', 'model name', 'microcode', 'flags')
        info = {k: matching[0].get(k) for k in keys}
        lib.r14_cpuid.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        lib.r14_cpuid.restype = ctypes.c_int
        cpuid = []
        for leaf, sub in ((0, 0), (1, 0), (7, 0), (7, 1), (0x80000000, 0), (0x80000001, 0)):
            regs = (ctypes.c_uint * 4)()
            available = bool(lib.r14_cpuid(leaf, sub, regs))
            values = list(regs)
            if leaf == 1:
                values[1] &= 0x00ffffff
            cpuid.append({'leaf': leaf, 'subleaf': sub, 'available': available, 'registers': values})
        return {'cpu': info, 'cpuid': cpuid}
    except (OSError, ValueError, AttributeError) as exc:
        raise RuntimeProfileError('Cannot inspect CPU signature') from exc


def _native_profile(lib, expected, mapping, loaded):
    for name in ('r14_x87', 'r14_mxcsr', 'r14_round', 'r14_nearest'):
        function = getattr(lib, name)
        function.argtypes = []
        function.restype = ctypes.c_uint if name in ('r14_x87', 'r14_mxcsr') else ctypes.c_int
    fp = {'x87Control': lib.r14_x87(), 'mxcsrControl': lib.r14_mxcsr() & 0xffc0,
          'rounding': lib.r14_round(), 'nearest': lib.r14_nearest()}
    _require(fp == expected['floatingPoint'] and fp['rounding'] == fp['nearest'], 'FP control mismatch')
    _require(_cpu_signature(lib) == expected['cpuSignature'], 'Actual CPU/CPUID mismatch')

    blas_path = mapping[expected['openblas']['path']]
    blas_loaded = [p for p in loaded if 'openblas' in Path(p).name.lower()]
    _require(blas_loaded == [blas_path], 'Actual loaded OpenBLAS identity mismatch')
    blas = ctypes.CDLL(blas_path, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    for stem, saved in expected['openblas']['queries'].items():
        fn = getattr(blas, saved['symbol'])
        fn.argtypes = []
        fn.restype = ctypes.c_int if stem == 'openblas_get_num_threads' else ctypes.c_char_p
        value = fn()
        if isinstance(value, bytes):
            value = value.decode('ascii')
        _require(value == saved['value'], 'Actual OpenBLAS query mismatch: ' + stem)

    lib.r14_symbol.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                               ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64)]
    lib.r14_symbol.restype = ctypes.c_int
    for row in expected['libmResolvedSymbols']:
        _require(row['version'] is None, 'Unsupported versioned symbol')
        library = mapping[row['path']]
        _require(library in loaded, 'Expected libm is not loaded')
        out, offset = ctypes.create_string_buffer(4096), ctypes.c_uint64()
        status = lib.r14_symbol(os.fsencode(library), row['name'].encode('ascii'), b'',
                                out, len(out), ctypes.byref(offset))
        _require(status == 0 and str(Path(os.fsdecode(out.value)).resolve()) == library and
                 offset.value == row['implementationOffset'] and _sha256(library) == row['sha256'],
                 'Resolved default libm symbol mismatch: ' + row['name'])
    return fp


def _numpy_sources(mapping, files):
    reverse = {value: key for key, value in mapping.items()}
    for name, module in list(sys.modules.items()):
        if name != 'numpy' and not name.startswith('numpy.'):
            continue
        path = getattr(module, '__file__', None)
        if not path:
            continue
        cached = getattr(module, '__cached__', None)
        _require(not cached or not Path(cached).exists(),
                 'NumPy bytecode cache prevents source-byte authentication: ' + name)
        if path.endswith(('.pyc', '.pyo')):
            try:
                path = importlib.util.source_from_cache(path)
            except ValueError as exc:
                raise RuntimeProfileError('Unresolvable NumPy bytecode') from exc
        path = str(Path(path).resolve(strict=True))
        _require(path in reverse and _sha256(path) == files[reverse[path]],
                 'Imported NumPy source/binary not authenticated: ' + name)


def _canonical(spec):
    _require(_PUBLIC_PROFILE_SHA256 is not None,
             'Public canonical admission unavailable: the authenticated path-free profile export and its pin are not yet supplied')
    _require(isinstance(spec, dict), 'Public profile bytes, SHA and logical file mapping required')
    raw, pinned, paths = spec.get('profile_bytes'), spec.get('profile_sha256'), spec.get('file_map')
    _require(isinstance(raw, bytes) and pinned == _PUBLIC_PROFILE_SHA256 and
             hashlib.sha256(raw).hexdigest() == pinned, 'Public profile bytes/hash mismatch')
    try:
        expected = json.loads(raw)
        _require(expected['schema'] == 'c5-portable-scientific-profile-v1', 'Profile schema mismatch')
        files = expected['files']
        _require(isinstance(paths, dict) and set(paths) == set(files), 'Complete exact logical mapping required')
        mapping = {old: str(Path(new).resolve(strict=True)) for old, new in paths.items()}
        _require(len(set(mapping.values())) == len(mapping), 'Ambiguous file mapping')
        for old, current in mapping.items():
            _require(_sha256(current) == files[old], 'Numerical file byte mismatch: ' + old)
        helpers = [mapping[p] for p, digest in files.items() if digest == _HELPER_SHA256]
        _require(len(helpers) == 1, 'Original authenticated FP helper required')
        _require(platform.system() == 'Linux' and platform.machine() == expected['machine'] and
                 platform.python_version() == expected['python'], 'Python/platform mismatch')
        _require(str(Path(sys.executable).resolve()) == mapping[expected['executable']],
                 'Actual Python executable differs from mapped executable')
        _require(not any(x == 'numpy' or x.startswith('numpy.') for x in sys.modules),
                 'NumPy already imported before admission')
        before = _maps()
        _require(not any('numpy' in p.lower() or 'blas' in Path(p).name.lower() for p in before),
                 'NumPy or BLAS already mapped before admission')
        controls = expected['controls']
        wanted = {'NPY_DISABLE_CPU_FEATURES': ','.join(controls['numpyDisable']),
                  'OPENBLAS_CORETYPE': controls['openblasCore'], 'OPENBLAS_NUM_THREADS': '1',
                  'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        _require(all(os.environ.get(k) == v for k, v in wanted.items()), 'Pre-import controls mismatch')
        _require(all(k not in os.environ for k in ('GLIBC_TUNABLES', 'LD_PRELOAD', 'LD_AUDIT', 'LD_HWCAP_MASK')),
                 'Unsupported loader override')
        affinity = os.sched_getaffinity(0)
        _require(len(affinity) == 1, 'Conditional admission requires single-CPU affinity')
        lib = ctypes.CDLL(helpers[0], mode=os.RTLD_LOCAL | os.RTLD_NOW)
        for name in ('r14_x87', 'r14_mxcsr', 'r14_round', 'r14_nearest'):
            fn = getattr(lib, name)
            fn.argtypes = []
            fn.restype = ctypes.c_uint if name in ('r14_x87', 'r14_mxcsr') else ctypes.c_int
        _require(lib.r14_x87() == controls['x87Control'] and
                 (lib.r14_mxcsr() & 0xffc0) == controls['mxcsrControl'] and
                 lib.r14_round() == lib.r14_nearest(), 'Pre-import FP control mismatch')
        import numpy as np
        core = np.core._multiarray_umath
        actual_numpy = {'version': np.__version__, 'baseline': list(core.__cpu_baseline__),
                        'compiledDispatch': list(core.__cpu_dispatch__),
                        'features': {k: bool(v) for k, v in core.__cpu_features__.items()}}
        _require(actual_numpy == expected['numpy'], 'NumPy actual dispatch mismatch')
        _require(str(Path(np.__file__).resolve()) in mapping.values() and
                 str(Path(core.__file__).resolve()) in mapping.values(), 'Unmapped NumPy implementation')
        _numpy_sources(mapping, files)
        loaded = _maps()
        _require(loaded <= set(mapping.values()), 'Loaded DSO outside authenticated numeric file map')
        for path in loaded:
            old = next(k for k, v in mapping.items() if v == path)
            _require(_sha256(path) == files[old], 'Loaded DSO byte mismatch')
        _native_profile(lib, expected, mapping, loaded)
        _require(_maps() == loaded and os.sched_getaffinity(0) == affinity,
                 'DSO or affinity drift during admission')
        return {'schema': 'c5-conditional-numerical-admission-v1',
                'identity': 'prospective-recorded-profile-scope', 'profileSHA256': pinned,
                'prospectiveProfileMatched': True, 'futureSamplingAuthorized': False,
                'historicalPrefixRegenerationCertified': False,
                'completeRelocationCoverageProved': False,
                'crossCpuEquivalenceProved': False,
                'scope': 'current process, current thread FP state, default libm symbols only',
                'requires': 'immutable files and revalidation of process/thread before numerical work; '
                            'prevent subsequent state or loaded-code drift'}
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        if isinstance(exc, RuntimeProfileError):
            raise
        raise RuntimeProfileError('Canonical evidence unavailable or malformed') from exc


def _sha256(path):
    try:
        with open(path, 'rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
            return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _rounding():
    # Observe only. C's numeric encoding is platform-specific and is not
    # interpreted as FE_TONEAREST without the platform's native constants.
    try:
        function = ctypes.CDLL(None).fegetround
        function.argtypes = []
        function.restype = ctypes.c_int
        value = function()
        return value if value >= 0 else None
    except (AttributeError, OSError):
        return None


def _loaded_blas():
    """Describe Linux mapped BLAS files, not the selected kernel or ABI."""
    try:
        lines = Path('/proc/self/maps').read_text().splitlines()
    except (OSError, UnicodeError):
        return None
    paths = set()
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = fields[5]
            if 'blas' in Path(path).name.lower():
                paths.add(path)
    return sorted(({'sha256': _sha256(p)} for p in paths),
                  key=lambda item: item['sha256'] or '')


def _sanitized_controls(environ):
    """Bind each captured value without exposing it; hashes are not secret."""
    result = {}
    for name in _CONTROLS:
        value = environ.get(name)
        result[name] = {'present': value is not None,
                        'valueSHA256': (hashlib.sha256(os.fsencode(value)).hexdigest()
                                        if value is not None else None)}
    return result


def validate_runtime(spec=None, *, mode='historical'):
    """Refuse unavailable canonical scope or opt into an unverified profile.

    Canonical mode fails closed until the public, authenticated path-free
    profile and its content pin are available. Historical mode always refuses
    seed reconstruction of the old prefix. New mode returns an observation
    label, never a certificate.
    """
    if mode == 'historical':
        raise RuntimeProfileError('Historical numerical identity unavailable: the portable gate '
                                  'cannot verify the accepted native profile or historical dispatch')
    if mode == 'canonical':
        return _canonical(spec)
    if mode != 'new':
        raise RuntimeProfileError('Select historical, canonical, or explicitly request new numerical profile')

    before = 'numpy' in sys.modules
    environment = _sanitized_controls(os.environ)
    rounding_before = _rounding()
    import numpy as np

    core = getattr(getattr(np, 'core', None), '_multiarray_umath', None)
    features = getattr(core, '__cpu_features__', None)
    observation = {
        'python': platform.python_version(),
        'machine': platform.machine(),
        'pythonExecutableSHA256': _sha256(sys.executable),
        'floatInfo': {'radix': sys.float_info.radix, 'mantissaBits': sys.float_info.mant_dig,
                      'maxExponent': sys.float_info.max_exp},
        'roundingRawBeforeNumpy': rounding_before,
        'roundingRawAfterNumpy': _rounding(),
        'x87Control': None, 'mxcsrControl': None,
        'numpyAlreadyImported': before,
        'environmentBeforeNumpy': environment,
        'numpy': {'version': np.__version__, 'fileSHA256': _sha256(np.__file__),
                  'coreBinarySHA256': _sha256(getattr(core, '__file__', '')),
                  'baseline': list(getattr(core, '__cpu_baseline__', ())),
                  'compiledDispatch': list(getattr(core, '__cpu_dispatch__', ())),
                  'features': ({str(k): bool(v) for k, v in features.items()}
                               if isinstance(features, dict) else None)},
        'mappedBlasFiles': _loaded_blas(),
        'selectedBlasCore': None, 'actualBlasThreads': None,
        'libmResolvedSymbols': None,
    }
    encoded = json.dumps(observation, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True).encode('ascii')
    return {'schema': 'c5-new-numerical-profile-observation-v2',
            'identity': 'new-unverified', 'acceptedProfileCertified': False,
            'historicalPrefixRegenerationCertified': False,
            'observationSHA256': hashlib.sha256(encoded).hexdigest(),
            'observation': observation,
            'limitations': ['pre-import control effectiveness not established',
                            'x87 and MXCSR not inspected',
                            'CPU/CPUID and loaded numerical closure not authenticated',
                            'BLAS dispatch and threads not queried',
                            'libm symbol and relocation identity not established',
                            'no cross-CPU or IFUNC equivalence established']}
