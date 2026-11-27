import sys
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from src.design_validation import c5_runtime_profile as profile


class RuntimeProfileTest(unittest.TestCase):
    def test_historical_refuses_even_with_supplied_spec(self):
        with self.assertRaisesRegex(profile.RuntimeProfileError, 'Historical numerical identity'):
            profile.validate_runtime({'acceptedProfileCertified': True})

    def test_unknown_mode_refuses(self):
        with self.assertRaises(profile.RuntimeProfileError):
            profile.validate_runtime(mode='accepted')

    def test_new_is_explicitly_unverified_and_does_not_change_controls(self):
        with patch.object(profile, '_rounding', side_effect=[0, 1]), \
             patch.object(profile, '_loaded_blas', return_value=None), \
             patch.dict(profile.os.environ, {'OPENBLAS_NUM_THREADS': '7'}):
            already = 'numpy' in sys.modules
            result = profile.validate_runtime(mode='new')
            self.assertEqual(profile.os.environ['OPENBLAS_NUM_THREADS'], '7')
        self.assertEqual(result['identity'], 'new-unverified')
        self.assertFalse(result['acceptedProfileCertified'])
        self.assertFalse(result['historicalPrefixRegenerationCertified'])
        self.assertEqual(result['observation']['numpyAlreadyImported'], already)
        self.assertEqual(result['observation']['roundingRawBeforeNumpy'], 0)
        self.assertEqual(result['observation']['roundingRawAfterNumpy'], 1)
        self.assertIsNone(result['observation']['mxcsrControl'])
        self.assertIsNone(result['observation']['selectedBlasCore'])
        self.assertIsNone(result['observation']['mappedBlasFiles'])
        self.assertEqual(len(result['observationSHA256']), 64)

    def test_missing_file_is_not_a_verified_hash(self):
        self.assertIsNone(profile._sha256('/nonexistent/c5-runtime-file'))

    def test_public_observation_redacts_controls_and_hashes_sanitized_content(self):
        environment = {'LD_LIBRARY_PATH': '/private/first:/private/second',
                       'LD_PRELOAD': '/private/libsecret.so',
                       'LD_AUDIT': '', 'OPENBLAS_CORETYPE': '/unexpected/path'}
        with patch.object(profile, '_rounding', return_value=0), \
             patch.object(profile, '_loaded_blas', return_value=[{'sha256': 'a' * 64}]), \
             patch.dict(profile.os.environ, environment, clear=True):
            result = profile.validate_runtime(mode='new')
        observation = result['observation']
        encoded = json.dumps(observation, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True).encode('ascii')
        self.assertEqual(result['observationSHA256'], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(result['schema'], 'c5-new-numerical-profile-observation-v2')
        for secret in environment.values():
            if secret:
                self.assertNotIn(secret, json.dumps(result))
        controls = observation['environmentBeforeNumpy']
        self.assertTrue(controls['LD_AUDIT']['present'])
        self.assertEqual(controls['LD_AUDIT']['valueSHA256'], hashlib.sha256(b'').hexdigest())
        self.assertFalse(controls['LD_HWCAP_MASK']['present'])
        self.assertIsNone(controls['LD_HWCAP_MASK']['valueSHA256'])
        self.assertEqual(controls['LD_PRELOAD']['valueSHA256'],
                         hashlib.sha256(environment['LD_PRELOAD'].encode()).hexdigest())

    def test_canonical_requires_original_bytes_and_explicit_mapping(self):
        for spec in (None, {}, {'profile_bytes': b'{}', 'profile_sha256': '0' * 64,
                                'file_map': {}}):
            with self.assertRaises(profile.RuntimeProfileError):
                profile.validate_runtime(spec, mode='canonical')

    def test_canonical_refuses_private_profile_even_if_hash_matches(self):
        raw = json.dumps({'schema': 'c5-public-port-profile-evidence-v1',
                          'actualNumericFiles': {'original': 'a' * 64}}).encode()
        digest = hashlib.sha256(raw).hexdigest()
        with patch.object(profile.ctypes, 'CDLL') as load:
            with self.assertRaisesRegex(profile.RuntimeProfileError, 'path-free profile export'):
                profile.validate_runtime({'profile_bytes': raw, 'profile_sha256': digest,
                                          'file_map': {}}, mode='canonical')
            load.assert_not_called()

    def test_export_strips_paths_and_actor_and_binds_logical_ids(self):
        path = '/private/example/numeric.so'
        query = {'symbol': 'query', 'value': 'Nehalem'}
        source = {
            'schema': 'c5-public-port-profile-evidence-v1',
            'actualOriginalExecution': {'host': 'secret-host', 'argv': [path]},
            'actualNumericFiles': {path: 'a' * 64},
            'executable': {'path': path}, 'python': '3.10.20', 'machine': 'x86_64',
            'controls': {'mxcsrControl': 8064, 'numpyBaseline': [], 'numpyDisable': [],
                         'openblasCore': 'NEHALEM', 'openblasThreads': 1,
                         'rounding': 'nearest', 'x87Control': 639},
            'numpy': {'version': '1.23.5', 'baseline': [], 'compiledDispatch': [], 'features': {}},
            'cpuSignature': {'cpu': {'vendor_id': 'x', 'cpu family': '6', 'model': '1',
                                     'stepping': '1', 'model name': 'x', 'microcode': 'x',
                                     'flags': 'x'}, 'cpuid': []},
            'floatingPoint': {'mxcsrControl': 8064, 'x87Control': 639, 'rounding': 0, 'nearest': 0},
            'openblas': {'path': path, 'queries': {'openblas_get_config': query,
                         'openblas_get_corename': query, 'openblas_get_num_threads': query}},
            'libmResolvedSymbols': [{'name': 'exp', 'path': path, 'version': None,
                                     'sha256': 'a' * 64, 'implementationOffset': 1}]}
        raw = json.dumps(source).encode()
        digest = hashlib.sha256(raw).hexdigest()
        public = profile.export_scientific_profile(raw, digest)
        self.assertNotIn(path.encode(), public)
        self.assertNotIn(b'secret-host', public)
        data = json.loads(public)
        self.assertEqual(data['files'], {'file-0000': 'a' * 64})
        self.assertEqual(data['executable'], data['openblas']['path'])
        self.assertEqual(data['libmResolvedSymbols'][0]['path'], data['executable'])
        self.assertEqual(public, profile.export_scientific_profile(raw, digest))
        with self.assertRaises(profile.RuntimeProfileError):
            profile.export_scientific_profile(raw, '0' * 64)

    def test_pinned_public_profile_still_requires_complete_logical_mapping(self):
        raw = json.dumps({'schema': 'c5-portable-scientific-profile-v1',
                          'files': {'file-0000': 'a' * 64}}).encode()
        digest = hashlib.sha256(raw).hexdigest()
        with patch.object(profile, '_PUBLIC_PROFILE_SHA256', digest), \
             patch.object(profile.ctypes, 'CDLL') as load:
            with self.assertRaisesRegex(profile.RuntimeProfileError, 'logical mapping'):
                profile.validate_runtime({'profile_bytes': raw, 'profile_sha256': digest,
                                          'file_map': {}}, mode='canonical')
            load.assert_not_called()

    def test_numpy_source_cache_refused(self):
        class Module:
            __file__ = '/not-used/numpy.py'
            __cached__ = '/not-used/numpy.pyc'

        with patch.dict(sys.modules, {'numpy.untrusted_test': Module()}), \
             patch.object(profile.Path, 'exists', return_value=True):
            with self.assertRaisesRegex(profile.RuntimeProfileError, 'bytecode cache'):
                profile._numpy_sources({}, {})

    def test_direct_bootstrap_does_not_import_src_or_numpy(self):
        path = Path(__file__).resolve().parents[1] / 'src/design_validation/c5_runtime_profile.py'
        program = ("import runpy,sys; runpy.run_path(sys.argv[1]); "
                   "assert 'src' not in sys.modules and 'numpy' not in sys.modules")
        subprocess.run([sys.executable, '-S', '-c', program, str(path)], check=True)
