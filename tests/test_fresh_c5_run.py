"""Stub-only CLI routing checks; these do not validate scientific computation."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.design_validation import c5_fresh_lazy as lazy
from src.design_validation import c5_fresh_prepare as prepare
from src.design_validation import c5_public_reproduction as common
from src.design_validation import c5_fresh_audit as audit


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'fresh_c5_run.py'
spec = importlib.util.spec_from_file_location('fresh_c5_run_test', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class FreshCliStubTests(unittest.TestCase):
    def test_inspect_pins_is_explicitly_unverified(self):
        bootstrap = {'validate_runtime': lambda **kwargs: {'stub': True}}
        with mock.patch.object(cli.runpy, 'run_path', return_value=bootstrap), \
             mock.patch.object(prepare, 'inspect_population_pins', return_value={'inputs': {}, 'sources': {}, 'entrySHA256': 'x'}) as inspect, \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main(['inspect-pins', '--manifest', 'manifest', '--manifest-sha256', 'digest'])
        inspect.assert_called_once()

    def test_missing_arguments_rejected_before_runtime(self):
        with mock.patch.object(cli.runpy, 'run_path') as bootstrap, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(['sample', '--manifest', 'x'])
        bootstrap.assert_not_called()

    def test_sample_and_verify_routing_and_range(self):
        bootstrap = {'validate_runtime': lambda **kwargs: {'stub': True}}
        args = ['--manifest', 'manifest', '--manifest-sha256', 'digest',
                '--output', 'out', '--run-id', 'test-only', '--replicates', '2:3']
        with mock.patch.object(cli.runpy, 'run_path', return_value=bootstrap), \
             mock.patch.object(lazy, 'run_lazy', return_value={'stub': True}) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            for command in ('sample', 'verify'):
                cli.main([command, *args])
                self.assertEqual(list(run.call_args.kwargs['replicates']), [2, 3])
                self.assertEqual(run.call_args.kwargs['domain'], 'fresh')
                self.assertEqual(run.call_args.kwargs['run_id'], 'test-only')
                self.assertEqual(run.call_args.kwargs['verify_only'], command == 'verify')
            for value in ('1', '0:1', '3:2', '1:1841'):
                with self.assertRaises(ValueError):
                    cli.main(['sample', *args[:-1], value])
            self.assertEqual(run.call_count, 2)

    def test_audit_routes_to_descriptive_auditor(self):
        bootstrap = {'validate_runtime': lambda **kwargs: {'stub': True}}
        with mock.patch.object(cli.runpy, 'run_path', return_value=bootstrap), \
             mock.patch.object(audit, 'audit_prepared_manifest', return_value={'statisticalAuthorization': False}) as check, \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main(['audit', '--manifest', 'm', '--manifest-sha256', 'digest',
                      '--output', 'out', '--run-id', 'fresh-a', '--replicates', '1:2'])
        self.assertEqual(list(check.call_args.kwargs['replicates']), [1, 2])
        self.assertEqual(check.call_args.kwargs['domain'], 'fresh')

    def test_prepare_passes_pins_and_reports_staged_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pins = base / 'pins.json'
            pins.write_text(json.dumps({'inputs': {'a': 'b'}, 'sources': {}, 'entrySHA256': 'x'}))
            out = base / 'out'
            out.mkdir()
            (out / 'input-manifest.json').write_text('{}')
            bootstrap = {'validate_runtime': lambda **kwargs: {}}
            with mock.patch.object(cli.runpy, 'run_path', return_value=bootstrap), \
                 mock.patch.object(common, 'sanitize_runtime', return_value={}), \
                 mock.patch.object(prepare, 'prepare_population', return_value={'roots': [None] * 90}) as call, \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                cli.main(['prepare', '--manifest', 'm', '--manifest-sha256', 'digest',
                          '--pins', str(pins), '--pins-sha256', common._sha(pins.read_bytes()), '--output', str(out)])
            self.assertEqual(call.call_args.kwargs['expected_inputs'], {'a': 'b'})
            self.assertEqual(json.loads(stdout.getvalue())['rootCount'], 90)
            with mock.patch.object(cli.runpy, 'run_path', return_value=bootstrap), \
                 mock.patch.object(prepare, 'prepare_population') as call, \
                 self.assertRaisesRegex(ValueError, 'Reviewed pins'):
                cli.main(['prepare', '--manifest', 'm', '--manifest-sha256', 'digest',
                          '--pins', str(pins), '--pins-sha256', '0' * 64, '--output', str(out)])
            call.assert_not_called()
            pins.write_text('{}')
            with mock.patch.object(cli.runpy, 'run_path', return_value=bootstrap), \
                 mock.patch.object(prepare, 'prepare_population') as call, \
                 self.assertRaisesRegex(ValueError, 'Pins'):
                cli.main(['prepare', '--manifest', 'm', '--manifest-sha256', 'digest',
                          '--pins', str(pins), '--pins-sha256', common._sha(pins.read_bytes()), '--output', str(out)])
            call.assert_not_called()
