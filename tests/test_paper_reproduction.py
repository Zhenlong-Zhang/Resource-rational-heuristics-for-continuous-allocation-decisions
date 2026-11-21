"""Purpose: resume and admission gates for the paper's four-figure entry.

Fixtures check software behavior, not historical result agreement. A separate
small native CLI run and saved-result comparison verify the publication entry.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import paper_reproduction as P
from src.design_validation import personal_time_inputs as I
from src.design_validation import personal_time_sensitivity as S
import test_personal_time_sensitivity as sensitivity_fixture
from test_paper_figures_plot import full_fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/paper_figures.json'


class PaperReproductionTests(unittest.TestCase):
    def test_help_and_explicit_scope_without_numerical_imports(self):
        cli = ROOT / 'scripts/reproduce_paper_figures.py'
        for args, code in ((['--help'], 0), (['all', '--output', 'unused'], 2),
                           (['redraw', '--output', 'unused', '--full'], 2),
                           (['all', '--output', 'unused', '--full', '--workers', '0'], 2)):
            result = subprocess.run([sys.executable, '-S', str(cli), *args],
                                    capture_output=True, text=True, cwd='/tmp')
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_independent_resume_never_calls_trial_and_preserves_bytes(self):
        context, root, cfg = sensitivity_fixture.fixture()
        record, _ = sensitivity_fixture.TestImmutableTrials().trial_record()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            path = S.record_path(output, 'rho_0.01', root)
            I.write(path, record)
            with patch.object(S, 'trial', return_value=record['payload']) as compute:
                P.check_trial(context, root, cfg, {}, 'rho_0.01', output)
                compute.assert_called_once()
            before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in output.rglob('*.json')}
            with patch.object(S, 'trial', side_effect=AssertionError('Must not recompute')):
                P.check_trial(context, root, cfg, {}, 'rho_0.01', output)
            self.assertEqual(before, {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                                     for p in output.rglob('*.json')})

    def test_missing_independent_checkpoint_is_recomputed_not_assumed(self):
        context, root, cfg = sensitivity_fixture.fixture()
        record, _ = sensitivity_fixture.TestImmutableTrials().trial_record()
        with tempfile.TemporaryDirectory() as temp:
            I.write(S.record_path(temp, 'rho_0.01', root), record)
            with patch.object(S, 'trial', return_value=record['payload']) as compute:
                P.check_trial(context, root, cfg, {}, 'rho_0.01', temp)
            self.assertIs(compute.call_args.args[-1], S.R)

    def test_altered_independent_binding_or_payload_is_refused(self):
        context, root, cfg = sensitivity_fixture.fixture()
        record, _ = sensitivity_fixture.TestImmutableTrials().trial_record()
        for field in ('binding', 'payload', 'producerSHA256', 'absoluteTolerance'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                I.write(S.record_path(temp, 'rho_0.01', root), record)
                with patch.object(S, 'trial', return_value=record['payload']):
                    P.check_trial(context, root, cfg, {}, 'rho_0.01', temp)
                path = S.record_path(temp, 'rho_0.01', root, 'independent')
                changed = I.read(path)
                if field == 'payload':
                    changed[field]['events'].clear()
                elif field == 'binding':
                    changed[field]['rootSHA256'] = 'foreign'
                else:
                    changed[field] = 'changed'
                path.write_text(json.dumps(changed))
                with patch.object(S, 'trial', side_effect=AssertionError('Do not repair by replay')):
                    with self.assertRaises(ValueError):
                        P.check_trial(context, root, cfg, {}, 'rho_0.01', temp)

    def test_null_or_malformed_checkpoint_never_replays_policy(self):
        context, root, cfg = sensitivity_fixture.fixture()
        record, _ = sensitivity_fixture.TestImmutableTrials().trial_record()
        for bad in (None, [], {}, 1):
            with self.subTest(value=bad), tempfile.TemporaryDirectory() as temp:
                I.write(S.record_path(temp, 'rho_0.01', root), record)
                I.write(S.record_path(temp, 'rho_0.01', root, 'independent'), bad)
                with patch.object(S, 'trial', side_effect=AssertionError('Corrupt is not missing')):
                    with self.assertRaisesRegex(ValueError, 'Malformed independent checkpoint'):
                        P.check_trial(context, root, cfg, {}, 'rho_0.01', temp)

    def test_partial_analysis_never_claims_paper_figures(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(P.T, 'analyze', return_value={'verifiedTrials': 6, 'fullFiniteTarget': False}):
            result = P.reduce_checked(Path(temp) / 'input', Path(temp) / 'out', CONFIG, full=False)
        self.assertFalse(result['paperComplete'])
        self.assertEqual(result['figures'], [])

    def test_all_keeps_stage_counts_not_reduction_only_zero_claim(self):
        selected = I.select_roots(only='C1:0:1')
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(P.S, 'binding', return_value={'fixture': True}), \
             patch.object(P.S, 'prepare'), \
             patch.object(P, 'produce_checked', return_value={'new_producer': 6, 'new_independent': 6}), \
             patch.object(P, 'reduce_checked', return_value={'paperComplete': False, 'newPolicySimulations': 0}):
            result = P.run('all', temp, CONFIG, selected=selected)
        self.assertEqual(result['new_producer'], 6)
        self.assertEqual(result['new_independent'], 6)
        self.assertNotIn('newPolicySimulations', result)

    def test_redraw_rejects_missing_and_tampered_seal_before_rendering(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'bundle', Path(temp) / 'plots'
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                P.redraw(source, output, CONFIG)
            I.write(source / 'figure_bundle.json', {'schema': P.BUNDLE_SCHEMA,
                'payload': {}, 'evidence': {}, 'identity': {}, 'contentSHA256': 'altered'})
            with self.assertRaisesRegex(ValueError, 'content changed'):
                P.redraw(source, output, CONFIG)
            self.assertFalse(output.exists())

    def test_old_policy_source_binding_is_unchanged(self):
        # Adding a presentation entry must not invalidate already checked inputs.
        self.assertNotIn('paper_reproduction.py', S.SOURCE_FILES)
        self.assertNotIn('paper_figures_plot.py', S.SOURCE_FILES)
        self.assertIn('personal_time_sensitivity.py', S.SOURCE_FILES)

    def test_complete_fixture_seal_and_real_redraw_cli(self):
        """A synthetic full bundle tests the interface, not historical provenance."""
        allocation, true_need, design, recipe = full_fixture()
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'input', Path(temp) / 'paper'
            selection = I.select_roots(full=True)
            producer_inputs = {str(S.record_path(Path('.'), label, row)): 'a'*64
                               for row in selection for label, _ in S.RATES}
            policy_inputs = {**producer_inputs, **{p.replace('producer/', 'independent/', 1): 'b'*64
                                                  for p in producer_inputs}}
            policy_check = {'schema': 'personal-time-sensitivity-independent-check-v1',
                            'binding': {'sources': S.implementation_identity(),
                                       'selection': I.select_roots(full=True)},
                            'producerInputs': producer_inputs,
                            'fullFiniteTarget': True, 'verifiedTrials': 2700,
                            'expectedOnlyEndpoints': 1860, 'genuineRealizedEndpoints': 840,
                            'absoluteTolerance': 1e-10, 'historicalParityEstablished': False,
                            'monteCarloPrecisionEstablished': False,
                            'completeIndependentReconstruction': True}
            I.write(source / 'independent_check.json', policy_check)

            def fixture_analysis(input_path, analysis, config, *, full):
                self.assertTrue(full)
                I.write(analysis / 'summary.json', true_need)
                I.write(analysis / 'fixed_outcomes/summary.json', allocation)
                # These rows are explicitly empty fixtures, not historical inputs.
                I.write(analysis / 'trial_metrics.json', [])
                I.write(analysis / 'fixed_outcomes/trial_metrics.json', [])
                checked = {'fullFiniteTarget': True, 'verifiedTrials': 2700,
                           'policyCheckSHA256': S.file_sha(source / 'independent_check.json'),
                           'policyInputs': policy_inputs, 'newPolicySimulations': 0,
                           'historicalProvenanceReconstructed': False}
                fixed = {**checked, 'recipe': P.T.D.RECIPE, 'sources': {
                    p.name: S.file_sha(p) for p in (Path(P.T.D.__file__),
                    ROOT / 'configs/personal_time_diagnostics.json')},
                    'absoluteTolerance': 1e-10, 'independentArithmetic': 'scalar_power_and_centered_least_squares',
                    'sharedKernels': ['native_law_normalization', 'PCG64', 'record_admission', 'group_traversal']}
                I.write(analysis / 'fixed_outcomes/independent_check.json', fixed)
                true = {**checked, 'recipe': P.T.RECIPE, 'sources': {
                    p.name: S.file_sha(p) for p in (Path(P.T.__file__), Path(P.T.T.__file__),
                    ROOT / 'configs/true_need_diagnostics.json')},
                    'fixedOutcomeCheckSHA256': S.file_sha(analysis / 'fixed_outcomes/independent_check.json'),
                    'interpretation': 'descriptive_fixed_simulated_targets_not_human_inference'}
                I.write(analysis / 'source_check.json', true)
                return {'verifiedTrials': 2700, 'fullFiniteTarget': True}

            with patch.object(P.T, 'analyze', side_effect=fixture_analysis):
                made = P.reduce_checked(source, output, CONFIG, full=True)
            self.assertTrue(made['paperComplete'])
            self.assertEqual(len(made['figures']), 4)
            redraw = Path(temp) / 'redraw'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_paper_figures.py'),
                'redraw', '--input', str(output), '--output', str(redraw)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(redraw.glob('figure_*.png'))), 4)
            self.assertEqual(json.loads(result.stdout)['newPolicySimulations'], 0)
            before = {p.name: p.read_bytes() for p in redraw.glob('*.png')}
            P.redraw(output, redraw, CONFIG)
            self.assertEqual(before, {p.name: p.read_bytes() for p in redraw.glob('*.png')})

            for mutate in ('summary', 'check'):
                target = output / 'analysis' / ('summary.json' if mutate == 'summary' else 'source_check.json')
                original = target.read_bytes()
                bad = json.loads(original)
                bad['verifiedTrials'] = 12
                target.write_text(json.dumps(bad))
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    P.redraw(output, Path(temp) / 'refused', CONFIG)
                target.write_bytes(original)

            for change in ('recipe', 'sources', 'fixedOutcomeCheckSHA256', 'policyInputs', 'schema_field'):
                path = output / 'analysis/source_check.json'
                original = path.read_bytes()
                seal_path = output / 'figure_bundle.json'
                old_seal = seal_path.read_bytes()
                check = json.loads(original)
                if change == 'schema_field':
                    del check['interpretation']
                elif change == 'policyInputs':
                    check[change] = {}
                else:
                    check[change] = 'invalid'
                path.write_text(json.dumps(check))
                sealed = json.loads(old_seal)
                sealed['evidence']['reductionChecks']['source_check.json'] = S.file_sha(path)
                sealed['contentSHA256'] = I.digest({k: v for k, v in sealed.items() if k != 'contentSHA256'})
                seal_path.write_text(json.dumps(sealed))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    P.redraw(output, Path(temp) / 'refused', CONFIG)
                path.write_bytes(original)
                seal_path.write_bytes(old_seal)


if __name__ == '__main__':
    unittest.main()
