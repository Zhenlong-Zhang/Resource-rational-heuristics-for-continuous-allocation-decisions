"""Non-numerical orchestration and no-clobber contract tests."""
import json
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from src.design_validation import c5_public_inputs
from src.design_validation import c5_public_reproduction as runner


class ReproductionTests(unittest.TestCase):
    def test_sanitized_runtime_binds_controls_without_disclosing_paths(self):
        controls = ('NPY_DISABLE_CPU_FEATURES', 'OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'GLIBC_TUNABLES', 'LD_PRELOAD',
                    'LD_AUDIT', 'LD_HWCAP_MASK', 'LD_LIBRARY_PATH')
        environment = {name: {'present': False, 'valueSHA256': None} for name in controls}
        environment['LD_LIBRARY_PATH'] = {'present': True, 'valueSHA256': hashlib.sha256(b'/private/one').hexdigest()}
        observation = {'python': '3.10.20', 'machine': 'x86_64', 'pythonExecutableSHA256': None,
                       'floatInfo': {'radix': 2, 'mantissaBits': 53, 'maxExponent': 1024},
                       'roundingRawBeforeNumpy': 0, 'roundingRawAfterNumpy': 0,
                       'x87Control': None, 'mxcsrControl': None, 'numpyAlreadyImported': False,
                       'environmentBeforeNumpy': environment,
                       'numpy': {'version': '1.23.5', 'fileSHA256': None, 'coreBinarySHA256': None,
                                 'baseline': ['SSE'], 'compiledDispatch': [], 'features': {'SSE': True}},
                       'mappedBlasFiles': [{'sha256': None}], 'selectedBlasCore': None,
                       'actualBlasThreads': None, 'libmResolvedSymbols': None}
        runtime = {'schema': 'c5-new-numerical-profile-observation-v2', 'identity': 'new-unverified',
                   'acceptedProfileCertified': False, 'historicalPrefixRegenerationCertified': False,
                   'observation': observation,
                   'observationSHA256': hashlib.sha256(runner.canonical_bytes(observation)).hexdigest(),
                   'limitations': []}
        sanitized = runner.sanitize_runtime(runtime)
        self.assertNotIn('/private/one', json.dumps(sanitized))
        self.assertEqual(sanitized['observationSHA256'], runtime['observationSHA256'])
        runtime['limitations'] = ['CPU/CPUID and loaded numerical closure not authenticated']
        runner.sanitize_runtime(runtime)
        runtime['limitations'] = ['/private/path']
        with self.assertRaises(ValueError):
            runner.sanitize_runtime(runtime)
        runtime['limitations'] = []
        observation['environmentBeforeNumpy']['LD_LIBRARY_PATH']['valueSHA256'] = hashlib.sha256(b'/private/two').hexdigest()
        with self.assertRaises(ValueError):
            runner.sanitize_runtime(runtime)
        runtime['observationSHA256'] = hashlib.sha256(runner.canonical_bytes(observation)).hexdigest()
        self.assertNotEqual(sanitized['observationSHA256'], runner.sanitize_runtime(runtime)['observationSHA256'])
        observation['environmentBeforeNumpy']['LD_LIBRARY_PATH'] = {'present': False, 'valueSHA256': None}
        for container, key, value in ((observation, 'unexpected', '/private/path'),
                                      (observation['numpy'], 'unexpected', 'safe'),
                                      (observation['numpy']['features'], '/private/path', True),
                                      (observation['mappedBlasFiles'][0], 'path', '/private/path')):
            container[key] = value
            runtime['observationSHA256'] = hashlib.sha256(runner.canonical_bytes(observation)).hexdigest()
            with self.assertRaises(ValueError):
                runner.sanitize_runtime(runtime)
            del container[key]
        observation['machine'] = '/private/path'
        runtime['observationSHA256'] = hashlib.sha256(runner.canonical_bytes(observation)).hexdigest()
        with self.assertRaises(ValueError):
            runner.sanitize_runtime(runtime)
        observation['machine'] = 'x86_64'
        observation['environmentBeforeNumpy']['LD_LIBRARY_PATH'] = '/private/raw'
        runtime['observationSHA256'] = hashlib.sha256(runner.canonical_bytes(observation)).hexdigest()
        with self.assertRaises(ValueError):
            runner.sanitize_runtime(runtime)

    def test_member_identity_and_invalid_coordinates(self):
        self.assertEqual(runner.member_id(2, 0, 1), 'rep-0002-v0-m01')
        for args in ((True, 0, 1), (1841, 0, 1), (2, 9, 1), (2, 0, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.member_id(*args)

    def test_prepare_uses_prefix_loader_envelope(self):
        lookup = {'roots': [None] * 90}
        prefix = {'payload': {'rows': [None] * 90}, 'rootCount': 90}
        with patch.object(c5_public_inputs, 'load_fixed_inputs', return_value=lookup), \
                patch.object(c5_public_inputs, 'load_historical_prefix', return_value=prefix) as load:
            result = runner.prepare('prefix', 'p', 'lookup', 'l', Path(__file__).resolve().parents[1])
        self.assertEqual(result['prefixRows'], 90)
        self.assertFalse(result['finalAuthorized'])
        self.assertIs(load.call_args.kwargs['lookup'], lookup)

    def test_repository_root_is_executing_checkout(self):
        self.assertEqual(runner._executing_root(Path(__file__).resolve().parents[1]),
                         Path(runner.__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory() as other, self.assertRaisesRegex(ValueError, 'executing source'):
            runner._executing_root(other)

    def test_accepted_prefix_rejects_altered_hash_and_bytes(self):
        from src.design_validation import c5_public_authority as authority
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'prefix.json'
            prefix.write_bytes(b'{"altered":true}')
            altered_sha = hashlib.sha256(prefix.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'adopted byte pins'):
                runner._admit_population_prefix('accepted-prefix', 'lookup', authority.ADOPTED_INITIAL_SHA256,
                                                prefix, altered_sha, root)
            with patch.object(c5_public_inputs, 'load_fixed_inputs', return_value={'roots': [None] * 90}):
                with self.assertRaisesRegex(ValueError, 'input byte SHA256 mismatch'):
                    runner._admit_population_prefix('accepted-prefix', 'lookup', authority.ADOPTED_INITIAL_SHA256,
                                                    prefix, authority.ACCEPTED_PREFIX_SHA256, root)

    def test_population_batch_boundary_and_failed_marker(self):
        root = Path(__file__).resolve().parents[1]
        context = {'repository_root': root}
        population = {'contexts': [context] * 90, 'input_mode': 'caller-prefix'}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_require_readonly_snapshot'), \
                patch.object(runner, '_unchanged') as unchanged, \
                patch.object(runner, '_run_members', return_value={'created': 1, 'verifiedExisting': 0}) as member:
            result = runner.run_population(population, directory, [2])
            self.assertEqual(result['created'], 90)
            self.assertEqual(unchanged.call_count, 2)
            self.assertEqual(member.call_count, 90)
            self.assertTrue(all(call.kwargs['batch'] for call in member.call_args_list))
            self.assertFalse((Path(directory) / '.c5-population-active').exists())
            member.side_effect = ValueError('failed member')
            with self.assertRaisesRegex(ValueError, 'failed member'):
                runner.run_population(population, directory, [2])
            self.assertTrue((Path(directory) / '.c5-population-active').exists())
            with self.assertRaisesRegex(ValueError, 'Unfinished or concurrent'):
                runner.run_population(population, directory, [2])

    def test_checkpoint_no_clobber_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'member.json'
            value = {'memberId': runner.member_id(2, 0, 1), 'inputSHA256': 'a' * 64, 'payload': {}}
            runner.write_once(path, value)
            self.assertEqual(runner.inspect_checkpoint(path, value['memberId'], 'a' * 64), value)
            with self.assertRaises(FileExistsError):
                runner.write_once(path, {'different': True})
            with self.assertRaises(ValueError):
                runner.inspect_checkpoint(path, value['memberId'], 'b' * 64)

    def test_missing_only_and_independent_replay(self):
        calls = []
        def produce(root, cfg, decision, latent, table, design, rep, *, spec):
            calls.append(('produce', rep))
            return {'replicateId': rep}
        def receive(root, cfg, decision, latent, table, design, rep, result, *, spec):
            calls.append(('receive', rep))
            if result != {'replicateId': rep}:
                raise ValueError('bad result')
            return {'status': 'independent_replay_pass', 'rootKey': ['C5', 0, 1],
                    'replicateId': rep, 'historicalEquivalence': False,
                    'authority': 'caller-supplied-inputs-not-authenticated'}
        context = {'spec': {'rawRoot': {'scheduleVariant': 0, 'mainIndex': 1},
                            'config': {}, 'initialDecision': {}, 'design': {}},
                   'producer': SimpleNamespace(produce_root=produce),
                   'receiver': SimpleNamespace(receive_root=receive),
                   'p_latent': object(), 'r_latent': object(), 'p_table': object(),
                   'r_table': object(), 'ps': object(), 'rs': object(),
                   'identity': {'mode': 'new-unverified'}, 'identity_sha256': 'x',
                   'repository_root': Path(__file__).resolve().parents[1]}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, '_unchanged'):
            first = runner.run_members(context, directory, [2])
            second = runner.run_members(context, directory, [2, 3])
            self.assertEqual((first['created'], second['created'], second['verifiedExisting']), (1, 1, 1))
            self.assertEqual(calls.count(('produce', 2)), 1)
            self.assertEqual(calls.count(('receive', 2)), 2)
            runner.run_members(context, directory, [2, 3], verify_only=True)
            with self.assertRaises(ValueError):
                runner.run_members(context, directory, [4], verify_only=True)
            self.assertFalse((Path(directory) / '.c5-population-active').exists())
            path = Path(directory) / (runner.member_id(2, 0, 1) + '.json')
            saved = json.loads(path.read_bytes())
            saved['payload']['result']['replicateId'] = 3
            path.write_bytes(runner.canonical_bytes(saved))
            with self.assertRaises(ValueError):
                runner.run_members(context, directory, [2])
            self.assertTrue((Path(directory) / '.c5-population-active').exists())
        for invalid in ([2, 2], [1]):
            with tempfile.TemporaryDirectory() as directory, patch.object(runner, '_unchanged'):
                with self.assertRaises(ValueError):
                    runner.run_members(context, directory, invalid)

    def test_concurrent_winner_is_replayed_and_bad_winner_refused(self):
        def receive(root, cfg, decision, latent, table, design, rep, result, *, spec):
            if result != {'replicateId': rep}:
                raise ValueError('bad winner')
            return {'status': 'independent_replay_pass', 'rootKey': ['C5', 0, 1],
                    'replicateId': rep, 'historicalEquivalence': False,
                    'authority': 'caller-supplied-inputs-not-authenticated'}
        context = {'spec': {'rawRoot': {'scheduleVariant': 0, 'mainIndex': 1},
                            'config': {}, 'initialDecision': {}, 'design': {}},
                   'producer': SimpleNamespace(produce_root=lambda *a, **k: {'replicateId': 2}),
                   'receiver': SimpleNamespace(receive_root=receive),
                   'p_latent': object(), 'r_latent': object(), 'p_table': object(),
                   'r_table': object(), 'ps': object(), 'rs': object(),
                   'identity': {'mode': 'new-unverified'}, 'identity_sha256': 'x',
                   'repository_root': Path(__file__).resolve().parents[1]}
        original = runner.write_once
        for bad in (False, True):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                def race(path, value):
                    if bad:
                        value['payload']['result']['replicateId'] = 3
                        value['payload']['resultSHA256'] = runner._sha(runner.canonical_bytes(value['payload']['result']))
                    original(path, value)
                    raise FileExistsError('concurrent writer won')
                with patch.object(runner, '_unchanged'), patch.object(runner, 'write_once', side_effect=race):
                    if bad:
                        with self.assertRaisesRegex(ValueError, 'bad winner'):
                            runner.run_members(context, directory, [2])
                    else:
                        result = runner.run_members(context, directory, [2])
                        self.assertEqual((result['created'], result['verifiedExisting']), (0, 1))

    def test_checkout_output_requires_ignored_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = {'spec': {'rawRoot': {'scheduleVariant': 0, 'mainIndex': 1}},
                       'repository_root': root}
            with patch.object(runner, '_executing_root', return_value=root), \
                    patch.object(runner.subprocess, 'run', side_effect=[SimpleNamespace(returncode=0),
                                                                       SimpleNamespace(returncode=1, stdout=b'')]) as check:
                with self.assertRaisesRegex(ValueError, 'git-ignored'):
                    runner.run_members(context, root / 'output', [2])
                self.assertIn(b'rep-0002-v0-m01.json\0', check.call_args.kwargs['input'])
            with patch.object(runner, '_executing_root', return_value=root), \
                    patch.object(runner.subprocess, 'run', side_effect=lambda *a, **k:
                                 SimpleNamespace(returncode=0, stdout=k.get('input', b''))), \
                    patch.object(runner, '_unchanged'):
                with self.assertRaisesRegex(ValueError, 'Missing checkpoint'):
                    runner.run_members(context, root / 'other-output', [2], verify_only=True)
            with tempfile.TemporaryDirectory() as external, \
                    patch.object(runner, '_executing_root', return_value=root), \
                    patch.object(runner.subprocess, 'run') as check, patch.object(runner, '_unchanged'):
                with self.assertRaisesRegex(ValueError, 'Missing checkpoint'):
                    runner.run_members(context, external, [2], verify_only=True)
                check.assert_not_called()

    def test_population_refuses_negated_checkpoint_ignore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
            (root / '.gitignore').write_text('results/*\n!results/rep-0002-v0-m01.json\n')
            context = {'spec': {'rawRoot': {'scheduleVariant': 0, 'mainIndex': 1}},
                       'repository_root': root}
            population = {'contexts': [context] * 90, 'input_mode': 'caller-prefix'}
            with patch.object(runner, '_executing_root', return_value=root), \
                    patch.object(runner, '_require_readonly_snapshot'), patch.object(runner, '_unchanged'):
                with self.assertRaisesRegex(ValueError, 'git-ignored'):
                    runner.run_population(population, root / 'results', [2])
            self.assertFalse((root / 'results/rep-0002-v0-m01.json').exists())
            self.assertTrue((root / 'results/.c5-population-active').exists())

    def test_bounded_ignore_checks_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / 'results'
            output.mkdir()
            source = {'scheduleVariant': 0, 'mainIndex': 1}
            def ignored(*args, **kwargs):
                return SimpleNamespace(returncode=0, stdout=kwargs['input'])
            with patch.object(runner.subprocess, 'run', side_effect=ignored) as check:
                paths = list(runner._checked_member_paths(output, root, source, list(range(2, 602))))
            self.assertEqual(len(paths), 600)
            self.assertEqual(check.call_count, 3)
            self.assertTrue(all(len(call.kwargs['input'].split(b'\0')) - 1 <= 256
                                for call in check.call_args_list))
            first = output / 'rep-0002-v0-m01.json'
            first.symlink_to(output / 'elsewhere')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                list(runner._checked_member_paths(output, root, source, [2]))
            first.unlink()
            entry = list(runner._checked_member_paths(output, Path('/nonexistent-checkout'), source, [2]))[0]
            first.symlink_to(output / 'elsewhere')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                runner._check_member_path(entry[2], entry[3])

    def test_single_and_population_share_atomic_exclusion(self):
        root = Path(__file__).resolve().parents[1]
        context = {'repository_root': root}
        population = {'contexts': [context] * 90, 'input_mode': 'caller-prefix'}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_require_readonly_snapshot'), patch.object(runner, '_unchanged'):
            def single_started(*args, **kwargs):
                with self.assertRaisesRegex(ValueError, 'Unfinished or concurrent'):
                    runner.run_population(population, directory, [2])
                return {'created': 1, 'verifiedExisting': 0}
            with patch.object(runner, '_run_members', side_effect=single_started):
                runner.run_members(context, directory, [2])

            def population_started(*args, **kwargs):
                with self.assertRaisesRegex(ValueError, 'Unfinished or concurrent'):
                    runner.run_members(context, directory, [2])
                return {'created': 1, 'verifiedExisting': 0}
            with patch.object(runner, '_run_members', side_effect=population_started):
                result = runner.run_population(population, directory, [2])
            self.assertEqual(result['created'], 90)

    def test_shard_roster_rejects_missing_duplicate_foreign_and_bad_identity(self):
        contexts = [{'spec': {'rawRoot': {'scheduleVariant': 0, 'mainIndex': main}},
                     'identity': {'mode': 'new-unverified'}, 'identity_sha256': 'x'}
                    for main in (1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'left', Path(directory) / 'right'
            left.mkdir()
            right.mkdir()
            paths = runner._shard_directories(None, [right, left])
            def checkpoint(target, rep, context):
                root = context['spec']['rawRoot']
                name = runner.member_id(rep, root['scheduleVariant'], root['mainIndex'])
                result = {'replicateId': rep}
                runner.write_once(target / (name + '.json'),
                                  {'memberId': name, 'inputSHA256': 'x',
                                   'payload': {'identity': context['identity'], 'result': result,
                                               'resultSHA256': runner._sha(runner.canonical_bytes(result))}})
                return name
            first = checkpoint(left, 2, contexts[0])
            checkpoint(left, 2, contexts[1])
            checkpoint(right, 3, contexts[0])
            with self.assertRaisesRegex(ValueError, 'Missing checkpoint'):
                runner._shard_roster(paths, contexts, replicates=[2, 3])
            checkpoint(right, 3, contexts[1])
            roster = runner._shard_roster(paths, contexts, replicates=[2, 3])
            self.assertEqual(len(roster), 4)
            self.assertEqual(runner._shard_payload(roster[first + '.json'], first, contexts[0]),
                             {'replicateId': 2})
            checkpoint(right, 2, contexts[0])
            with self.assertRaisesRegex(ValueError, 'Duplicate checkpoint'):
                runner._shard_roster(paths, contexts, replicates=[2, 3])
            (right / (first + '.json')).unlink()
            foreign = right / 'rep-0004-v0-m01.json'
            foreign.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Foreign checkpoint'):
                runner._shard_roster(paths, contexts, replicates=[2, 3])
            foreign.unlink()
            context = dict(contexts[0], identity_sha256='foreign')
            with self.assertRaisesRegex(ValueError, 'Checkpoint identity'):
                runner._shard_payload(roster[first + '.json'], first, context)
            with self.assertRaisesRegex(ValueError, 'Duplicate or overlapping'):
                runner._shard_directories(None, [left, left])
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                runner._shard_directories(left, [right])

    def test_shard_partial_lock_failure_releases_only_owned_locks(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / 'a', Path(directory) / 'b'
            left.mkdir()
            right.mkdir()
            existing = right / '.c5-population-active'
            existing.write_text('other owner')
            with self.assertRaisesRegex(ValueError, 'Unfinished or concurrent'):
                with runner._shard_locks([left, right], root):
                    self.fail('must not enter with another owner')
            self.assertFalse((left / '.c5-population-active').exists())
            self.assertEqual(existing.read_text(), 'other owner')

    def test_read_only_refusals_release_locks_but_interruptions_do_not(self):
        root = Path(__file__).resolve().parents[1]
        context = {'repository_root': root}
        population = {'contexts': [context] * 90, 'input_mode': 'caller-prefix'}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, '_require_readonly_snapshot'), patch.object(runner, '_unchanged'):
            left, right = Path(directory) / 'left', Path(directory) / 'right'
            left.mkdir()
            right.mkdir()
            with patch.object(runner, '_shard_roster', side_effect=ValueError('missing member')):
                with self.assertRaisesRegex(ValueError, 'missing member'):
                    runner.audit_population(population, shard_dirs=[left, right])
            self.assertFalse((left / '.c5-population-active').exists())
            self.assertFalse((right / '.c5-population-active').exists())
            with patch.object(runner, '_run_members', side_effect=ValueError('bad checkpoint')):
                with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                    runner.run_members(context, left, [2], verify_only=True)
                self.assertFalse((left / '.c5-population-active').exists())
                with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                    runner.run_population(population, left, [2], verify_only=True)
                self.assertFalse((left / '.c5-population-active').exists())
            with patch.object(runner, '_run_members', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_members(context, left, [2], verify_only=True)
            self.assertTrue((left / '.c5-population-active').exists())
            with self.assertRaisesRegex(ValueError, 'Unfinished or concurrent'):
                runner.run_members(context, left, [2], verify_only=True)

    def test_conditional_final_cli_requires_inputs_and_routes_shards(self):
        from scripts import reproduce_c5_projection as cli
        from src.design_validation import c5_conditional_final
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(['final'])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            high, shard_a, shard_b = (base / name for name in ('high', 'shard-a', 'shard-b'))
            for path in (high, shard_a, shard_b):
                path.mkdir()
            summary, certificate = base / 'summary.json', base / 'certificate.json'
            summary.write_text('{"summary":1}')
            certificate.write_text('{"certificate":2}')
            args = ['final', '--mode', 'new-unverified', '--input-mode', 'accepted-prefix',
                    '--manifest', 'manifest', '--manifest-sha256', 'm', '--prefix', 'prefix',
                    '--prefix-sha256', 'p', '--shard-dir', str(shard_a), '--shard-dir', str(shard_b),
                    '--high-output', str(high), '--high-config', 'config', '--adopted-lookup', 'lookup',
                    '--latent-config', 'latent', '--geometry-summary', str(summary),
                    '--geometry-certificate', str(certificate), '--policy', 'policy']
            observed = io.StringIO()
            with patch.object(cli.runpy, 'run_path', return_value={'validate_runtime': lambda **k: 'runtime'}), \
                    patch.object(runner, 'load_population', return_value={'population': 'fixture'}) as load, \
                    patch.object(c5_conditional_final, 'conditional_final', return_value={
                        'historicalExactResultEstablished': False, 'runtimeEquivalenceEstablished': False}) as final, \
                    redirect_stdout(observed):
                self.assertEqual(cli.main(args), 0)
            result = json.loads(observed.getvalue())
            self.assertEqual(result['status'], 'conditional-new-unverified')
            self.assertFalse(result['result']['historicalExactResultEstablished'])
            self.assertEqual(load.call_args.kwargs['input_mode'], 'accepted-prefix')
            self.assertEqual(final.call_args.kwargs['shard_dirs'], [str(shard_a), str(shard_b)])
            self.assertEqual(final.call_args.kwargs['geometry_summary'], {'summary': 1})
            self.assertEqual(final.call_args.kwargs['geometry_certificate'], {'certificate': 2})
            self.assertEqual(final.call_args.kwargs['policy'], 'policy')
            self.assertIsNone(final.call_args.kwargs['output'])

            overlap = args.copy()
            overlap[overlap.index('--high-output') + 1] = str(shard_a)
            with patch.object(cli.runpy, 'run_path', return_value={'validate_runtime': lambda **k: 'runtime'}), \
                    patch.object(runner, 'load_population', return_value={}), \
                    patch.object(c5_conditional_final, 'conditional_final') as final, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(overlap), 1)
                final.assert_not_called()

    def test_redraw_cli_unverified_input_and_no_overwrite(self):
        from scripts import reproduce_c5_projection as cli
        from src.design_validation import c5_projection_plot
        import matplotlib.pyplot as plt
        class Figure:
            def savefig(self, handle, *, format):
                self.format = format
                handle.write(b'figure fixture')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / 'input.json', base / 'output'
            result = {'schema': 'fixture-caller-data'}
            source.write_text(json.dumps({'status': 'conditional-new-unverified', 'result': result}))
            args = ['redraw', '--conditional-json', str(source), '--output-dir', str(output)]
            response = io.StringIO()
            fig = Figure()
            with patch.object(c5_projection_plot, 'plot_conditional_projection', return_value=(fig, None)) as plot, \
                    patch.object(plt, 'close') as close, redirect_stdout(response):
                self.assertEqual(cli.main(args), 0)
            plot.assert_called_once_with(result)
            close.assert_called_once_with(fig)
            self.assertEqual(fig.format, 'pdf')
            self.assertEqual(json.loads(response.getvalue())['status'], 'conditional-redraw-unverified')
            self.assertFalse(json.loads(response.getvalue())['inputAuthenticated'])
            target = output / 'c5_conditional_projection.pdf'
            self.assertEqual(target.read_bytes(), b'figure fixture')
            with patch.object(c5_projection_plot, 'plot_conditional_projection', return_value=(Figure(), None)), \
                    patch.object(plt, 'close'), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 1)
            self.assertEqual(target.read_bytes(), b'figure fixture')
            source.write_text(json.dumps({'status': 'accepted', 'result': result}))
            with patch.object(c5_projection_plot, 'plot_conditional_projection') as plot, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 1)
                plot.assert_not_called()

    def test_redraw_cli_refuses_unignored_output(self):
        from scripts import reproduce_c5_projection as cli
        from src.design_validation import c5_projection_plot
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            source.write_text('{}')
            with patch.object(runner, '_require_ignored', side_effect=ValueError('not ignored')), \
                    patch.object(c5_projection_plot, 'plot_conditional_projection') as plot, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['redraw', '--conditional-json', str(source),
                                           '--output-dir', str(Path(directory) / 'out')]), 1)
                plot.assert_not_called()

    def test_historical_redraw_cli_uses_fixed_pin_and_no_clobber(self):
        from scripts import reproduce_c5_projection as cli
        from src.design_validation import c5_projection_plot
        import matplotlib.pyplot as plt
        class Figure:
            def savefig(self, handle, *, format):
                self.format = format
                handle.write(b'historical fixture')
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'summary.json', Path(directory) / 'figures'
            source.write_text('{}')  # Route mock only; not a validated historical artifact.
            pin = c5_projection_plot.ACCEPTED_PLOT_SUMMARY_SHA256
            args = ['historical-redraw', '--summary-json', str(source), '--summary-sha256', pin,
                    '--output-dir', str(output)]
            response = io.StringIO()
            figure = Figure()
            with patch.object(c5_projection_plot, 'plot_verified_saved_summary', return_value=(figure, None)) as plot, \
                    patch.object(plt, 'close') as close, redirect_stdout(response):
                self.assertEqual(cli.main(args), 0)
            plot.assert_called_once_with(str(source))
            close.assert_called_once_with(figure)
            value = json.loads(response.getvalue())
            self.assertEqual(value['status'], 'pinned-saved-result-redraw')
            self.assertEqual(value['savedSummarySHA256'], pin)
            self.assertFalse(value['freshSimulation'])
            self.assertFalse(value['runtimeEquivalenceEstablished'])
            target = output / 'c5_historical_saved_projection.pdf'
            self.assertEqual(target.read_bytes(), b'historical fixture')
            with patch.object(c5_projection_plot, 'plot_verified_saved_summary', return_value=(Figure(), None)), \
                    patch.object(plt, 'close'), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 1)
            self.assertEqual(target.read_bytes(), b'historical fixture')
            wrong = args.copy()
            wrong[wrong.index('--summary-sha256') + 1] = '0' * 64
            with patch.object(c5_projection_plot, 'plot_verified_saved_summary') as plot, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(wrong), 1)
                plot.assert_not_called()
            # The real plotting entry must reject bytes that do not match its pin.
            with redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 1)
            self.assertEqual(target.read_bytes(), b'historical fixture')

    def test_historical_redraw_cli_refuses_unignored_output(self):
        from scripts import reproduce_c5_projection as cli
        from src.design_validation import c5_projection_plot
        with tempfile.TemporaryDirectory() as directory:
            args = ['historical-redraw', '--summary-json', str(Path(directory) / 'summary.json'),
                    '--summary-sha256', c5_projection_plot.ACCEPTED_PLOT_SUMMARY_SHA256,
                    '--output-dir', str(Path(directory) / 'output')]
            with patch.object(runner, '_require_ignored', side_effect=ValueError('not ignored')), \
                    patch.object(c5_projection_plot, 'plot_verified_saved_summary') as plot, \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 1)
                plot.assert_not_called()

    def test_unimplemented_stages_fail_closed(self):
        for stage in ('audit', 'final', 'redraw'):
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, 'unavailable'):
                runner.unavailable(stage)

    def test_required_real_fixture_gate(self):
        """Scheduled gate: C5_REPRO_REQUIRE_FIXTURE=1 forbids a silent skip.

        Supply C5_REPRO_SPEC, C5_REPRO_SPEC_SHA256, C5_REPRO_PREFIX and
        C5_REPRO_PREFIX_SHA256. These are caller inputs, not authority evidence.
        """
        names = ('C5_REPRO_SPEC', 'C5_REPRO_SPEC_SHA256', 'C5_REPRO_PREFIX', 'C5_REPRO_PREFIX_SHA256')
        if not all(os.environ.get(name) for name in names):
            if os.environ.get('C5_REPRO_REQUIRE_FIXTURE') == '1':
                self.fail('Required real fixture variables missing: ' + ', '.join(name for name in names if not os.environ.get(name)))
            self.skipTest('requires pinned external reproduction spec and prefix')
        root = Path(__file__).resolve().parents[1]
        spec, spec_hash, prefix, prefix_hash = (os.environ[name] for name in names)
        with tempfile.TemporaryDirectory() as output:
            common = ['--mode', 'new-unverified', '--spec', spec, '--spec-sha256', spec_hash,
                      '--prefix', prefix, '--prefix-sha256', prefix_hash,
                      '--output', output, '--replicates', '2:2']
            results = []
            for stage in ('sample', 'verify'):
                result = subprocess.run([sys.executable, str(root / 'scripts/reproduce_c5_projection.py'), stage, *common],
                                        cwd=root, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertFalse(payload['result']['finalAuthorized'])
                self.assertFalse(payload['result']['historicalEquivalence'])
                results.append(payload)
            self.assertEqual(results[0]['result']['created'], 1)
            self.assertEqual(results[1]['result']['verifiedExisting'], 1)
            self.assertEqual(results[0]['result']['inputIdentitySHA256'], results[1]['result']['inputIdentitySHA256'])

    def test_population_real_fixture_sample_and_replay(self):
        """Opt-in real 90-root seam; must run from an isolated read-only staged checkout.

        Set C5_POP_REQUIRE_FIXTURE=1 to make missing inputs fail instead of skip.
        Manifest paths and inputs must be staged read-only; no historical parity
        or final certificate follows from this single-replicate software check.
        """
        names = ('C5_POP_MANIFEST', 'C5_POP_MANIFEST_SHA256', 'C5_POP_PREFIX', 'C5_POP_PREFIX_SHA256')
        if not all(os.environ.get(name) for name in names):
            if os.environ.get('C5_POP_REQUIRE_FIXTURE') == '1':
                self.fail('Required population fixture variables missing: ' +
                          ', '.join(name for name in names if not os.environ.get(name)))
            self.skipTest('requires read-only 90-root population fixture and accepted prefix')
        root = Path(__file__).resolve().parents[1]
        manifest, manifest_hash, prefix, prefix_hash = (os.environ[name] for name in names)
        with tempfile.TemporaryDirectory() as output:
            common = ['--mode', 'new-unverified', '--input-mode', 'accepted-prefix',
                      '--manifest', manifest, '--manifest-sha256', manifest_hash,
                      '--prefix', prefix, '--prefix-sha256', prefix_hash,
                      '--output', output, '--replicates', '2:2']
            results = []
            for stage in ('population-sample', 'population-verify'):
                result = subprocess.run([sys.executable, str(root / 'scripts/reproduce_c5_projection.py'),
                                         stage, *common], cwd=root, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload['rootCount'], 90)
                self.assertEqual(payload['inputMode'], 'accepted-prefix')
                self.assertFalse(payload['historicalEquivalence'])
                self.assertFalse(payload['finalAuthorized'])
                results.append(payload)
            self.assertEqual(results[0]['created'], 90)
            self.assertEqual(results[1]['verifiedExisting'], 90)


if __name__ == '__main__':
    unittest.main()
