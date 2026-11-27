"""Small interface/refusal checks; scientific parity requires admitted inputs."""
import copy
import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from src.design_validation import c5_public_trajectory as producer
from src.design_validation import c5_public_trajectory_reference as receiver
from src.design_validation import c5_public_streams as streams
from src.design_validation import c5_final


class TrajectoryBoundaryTests(unittest.TestCase):
    def test_namespace_imports_reject_cross_namespace_objects(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(root), str(root / 'src'), env.get('PYTHONPATH', '')))
        code = '''
from src.design_validation import c5_public_trajectory as a
from design_validation import c5_public_trajectory as b
from src.design_validation import c5_public_trajectory_reference as ar
from design_validation import c5_public_trajectory_reference as br
from src.design_validation import c5_public_streams as sa
from design_validation import c5_public_streams as sb
for local, foreign in ((a, b), (ar, br), (b, a), (br, ar)):
    try:
        check = local._validate_root if hasattr(local, '_validate_root') else local.validate_spec
        check({}, {}, {}, None, {}, object.__new__(foreign.AdmittedSpec))
    except ValueError as e:
        assert 'do not mix' in str(e)
    else:
        raise AssertionError('cross-namespace spec accepted')
for local, foreign in ((sa, sb), (sb, sa)):
    assert local.ValidatedLatentConfig is not foreign.ValidatedLatentConfig
    try:
        module = a if local is sa else b
        module.AdmittedSpec({'lookupPath': '', 'lookupSHA256': '', 'geometryCertificate': {}},
                            object.__new__(foreign.ValidatedLatentConfig), {})
    except ValueError as e:
        assert 'do not mix' in str(e)
    else:
        raise AssertionError('cross-namespace latent accepted')
'''
        completed = subprocess.run([sys.executable, '-c', code], env=env, cwd=root,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_required_fixture_gate(self):
        if os.environ.get('C5_TRAJECTORY_REQUIRE_FIXTURE') == '1':
            path = os.environ.get('C5_TRAJECTORY_FIXTURE')
            self.assertTrue(path, 'C5_TRAJECTORY_FIXTURE is required by the scheduled gate')
            self.assertTrue(Path(path).is_file(), 'C5_TRAJECTORY_FIXTURE must name a real fixture file')

    def test_canonical_fraction(self):
        self.assertEqual(producer._fraction(['-3', '7']), Fraction(-3, 7))
        self.assertEqual(receiver._fraction(['-3', '7']), Fraction(-3, 7))
        for value in (['2', '4'], ['-0', '1'], ['1', '-2'], ['1', '0'], [1, '2']):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    producer._fraction(value)
                with self.assertRaises((ValueError, ZeroDivisionError)):
                    receiver._fraction(value)

    def test_historical_prefix_and_out_of_roster_refused_before_draw(self):
        root = {'scheduleVariant': 0, 'mainIndex': 1}
        for rep in (True, 0, 1, 1841):
            with self.subTest(rep=rep):
                with self.assertRaises(ValueError):
                    producer.produce_root(root, {}, {}, None, b'', {}, rep, spec=None)
                with self.assertRaises(ValueError):
                    receiver.receive_root(root, {}, {}, None, b'', {}, rep, {}, spec=None)

    def test_receiver_rejects_bad_display_mapping(self):
        root = {'scheduleVariant': 0, 'mainIndex': 1, 'logicalToDisplayed': ['A', 'A']}
        with self.assertRaises(ValueError):
            receiver.receive_root(root, {}, {}, None, b'', {}, 2, {}, spec=None)

    def test_producer_rejects_bad_display_mapping(self):
        root = {'scheduleVariant': 0, 'mainIndex': 1, 'logicalToDisplayed': ['B', 'B']}
        with self.assertRaises(ValueError):
            producer.produce_root(root, {}, {}, None, b'', {}, 2, spec=None)

    def test_independent_draw_rejects_missing_mass(self):
        key = {'test': 'not-a-scientific-key'}
        counts = [0] * 100
        cumulative = [0.0.hex()] * 100
        with self.assertRaises(ValueError):
            receiver._draw(key, counts, cumulative)

    def test_independent_equality_includes_all_fields(self):
        result = {'event': {'seed': '01', 'index': 1}}
        altered = copy.deepcopy(result)
        altered['event']['seed'] = '02'
        with self.assertRaises(ValueError):
            receiver._same(result, altered, 'event')


@unittest.skipUnless(os.environ.get('C5_TRAJECTORY_FIXTURE'), 'requires externally supplied real scientific fixture')
class RealSourceTrajectoryTests(unittest.TestCase):
    """Fixture paths are relative to C5_TRAJECTORY_FIXTURE, not private defaults.

    JSON fields: rawRoot, config, initialDecision, design, lookupPath,
    lookupSHA256, geometryCertificate, latentPath, reportPaths, replicateId.
    reportPaths maps each schedule width hex to its complete product path.
    The lookup is the export accepted by c5_public_inputs.load_fixed_inputs;
    the certificate contains the independent ten-root design_reference output.
    Supplying the fixture or a digest does not authenticate historical origin.
    """
    def setUp(self):
        path = Path(os.environ['C5_TRAJECTORY_FIXTURE']).resolve()
        self.fixture = json.loads(path.read_text())
        f = self.fixture
        self.root, self.cfg, self.decision = f['rawRoot'], f['config'], f['initialDecision']
        self.design, self.replicate = f['design'], f['replicateId']
        self.latent = streams.ValidatedLatentConfig(json.loads((path.parent / f['latentPath']).read_text()))
        self.reports = {width: (path.parent / name).read_bytes() for width, name in f['reportPaths'].items()}
        self.source_spec = {'lookupPath': str(path.parent / f['lookupPath']), 'lookupSHA256': f['lookupSHA256'],
                            'geometryCertificate': f['geometryCertificate']}
        self.pspec = producer.AdmittedSpec(self.source_spec, self.latent, self.reports)
        self.rspec = receiver.AdmittedSpec(self.source_spec, self.latent, self.reports)
        self.width = float(self.cfg['task']['report_width']).hex()
        self.ptable = dict(self.pspec.table)[self.width]
        self.rtable = dict(self.rspec.table)[self.width]

    def produce(self, root=None, cfg=None, decision=None, design=None, spec=None):
        return producer.produce_root(self.root if root is None else root, self.cfg if cfg is None else cfg,
                                     self.decision if decision is None else decision, self.latent, self.ptable,
                                     self.design if design is None else design, self.replicate,
                                     spec=self.pspec if spec is None else spec)

    def receive(self, result):
        return receiver.receive_root(self.root, self.cfg, self.decision, self.latent, self.rtable,
                                     self.design, self.replicate, result, spec=self.rspec)

    def test_complete_produce_receive_and_event_refusal(self):
        result = self.produce()
        self.assertEqual(self.receive(result)['status'], 'independent_replay_pass')
        raw = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        admission = c5_final.admit_future_root(self.root, self.cfg, self.decision, self.latent,
                                               self.rtable, self.design, self.replicate, raw, spec=self.rspec)
        self.assertFalse(admission['actualUseEstablished'])
        self.assertFalse(admission['historicalPrefixEstablished'])
        self.assertFalse(admission['independentFullResultArtifactParityEstablished'])
        self.assertFalse(admission['finalCertificateAuthorized'])
        altered = copy.deepcopy(result)
        altered['latent']['A']['category'] = 101
        with self.assertRaises(ValueError):
            self.receive(altered)
        altered_raw = json.dumps(altered, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        with self.assertRaises(ValueError):
            c5_final.admit_future_root(self.root, self.cfg, self.decision, self.latent,
                                       self.rtable, self.design, self.replicate, altered_raw, spec=self.rspec)
        self.assertFalse(result['historicalEquivalence'])

    def test_cross_root_and_scientific_binding_refusals(self):
        lookup = receiver.inputs.load_fixed_inputs(self.source_spec['lookupPath'], self.source_spec['lookupSHA256'])
        other = next(row for row in lookup['roots'] if row['rootKey'] !=
                     ['C5', self.root['scheduleVariant'], self.root['mainIndex']]
                     and row['decision'] != self.decision)
        with self.assertRaises(ValueError):
            self.produce(decision=other['decision'])
        root = copy.deepcopy(self.root)
        root['logicalToDisplayed'].reverse()
        with self.assertRaises(ValueError):
            self.produce(root=root)
        cfg = copy.deepcopy(self.cfg)
        cfg['task']['report_width'] = other['reportWidth'] if other['reportWidth'] != cfg['task']['report_width'] else -1
        with self.assertRaises(ValueError):
            self.produce(cfg=cfg)
        root = copy.deepcopy(self.root)
        root['mainIndex'] = 1 if root['mainIndex'] != 1 else 2
        with self.assertRaises(ValueError):
            self.produce(root=root)

    def test_nested_decision_type_spoof_rejected_by_both_sides(self):
        for field, replacement in (('budget', float(self.decision['budget'])),):
            changed = copy.deepcopy(self.decision)
            changed[field] = replacement
            self.assertEqual(changed, self.decision)
            with self.assertRaises(ValueError):
                producer._validate_root(self.root, self.cfg, changed, self.latent, self.design, self.pspec)
            with self.assertRaises(ValueError):
                receiver.validate_spec(self.root, self.cfg, changed, self.latent, self.design, self.rspec)
        changed = copy.deepcopy(self.decision)
        changed['stop']['forced'] = int(changed['stop']['forced'])
        self.assertEqual(changed, self.decision)
        with self.assertRaises(ValueError):
            producer._validate_root(self.root, self.cfg, changed, self.latent, self.design, self.pspec)
        with self.assertRaises(ValueError):
            receiver.validate_spec(self.root, self.cfg, changed, self.latent, self.design, self.rspec)

    def test_geometry_and_certificate_refusals(self):
        design = copy.deepcopy(self.design)
        design['gap']['inverse'][1][1] = ['0', '1']
        with self.assertRaises(ValueError):
            self.produce(design=design)
        spec = copy.deepcopy(self.source_spec)
        spec['geometryCertificate']['scheduleVariant'] = (self.root['scheduleVariant'] + 1) % 9
        with self.assertRaises(ValueError):
            producer.AdmittedSpec(spec, self.latent, self.reports)
        spec = copy.deepcopy(self.source_spec)
        del spec['geometryCertificate']
        with self.assertRaises(ValueError):
            receiver.AdmittedSpec(spec, self.latent, self.reports)

    def test_separate_admissions_reject_cross_source_and_mutation(self):
        with self.assertRaises(FrozenInstanceError):
            self.pspec.variant = 8
        with self.assertRaises(ValueError):
            self.produce(spec=self.rspec)
        with self.assertRaises(ValueError):
            receiver.validate_spec(self.root, self.cfg, self.decision, self.latent, self.design, self.pspec)
        with self.assertRaises(ValueError):
            producer.produce_root(self.root, self.cfg, self.decision, self.latent, self.rtable,
                                  self.design, self.replicate, spec=self.pspec)
        other_latent = streams.ValidatedLatentConfig(json.loads(
            (Path(os.environ['C5_TRAJECTORY_FIXTURE']).resolve().parent / self.fixture['latentPath']).read_text()))
        with self.assertRaises(ValueError):
            producer._validate_root(self.root, self.cfg, self.decision, other_latent, self.design, self.pspec)
        modified = copy.deepcopy(self.source_spec)
        modified['lookupSHA256'] = '0' * 64
        with self.assertRaises(ValueError):
            producer.AdmittedSpec(modified, self.latent, self.reports)
        modified = dict(self.reports)
        modified[self.width] += b' '
        with self.assertRaises(ValueError):
            receiver.AdmittedSpec(self.source_spec, self.latent, modified)
        # Changes to the caller's certificate after admission cannot mutate snapshots.
        before = self.pspec.design_bytes
        self.source_spec['geometryCertificate']['design']['gap']['status'] = 'tampered'
        self.assertEqual(self.pspec.design_bytes, before)
        producer._validate_root(self.root, self.cfg, self.decision, self.latent, self.design, self.pspec)

    def test_repeated_root_binding_does_not_revalidate_geometry_or_tables(self):
        with patch.object(producer.geometry, 'design_payload', side_effect=AssertionError('recomputed')), \
             patch.object(receiver.geometry_reference, 'design_reference', side_effect=AssertionError('recomputed')), \
             patch.object(streams, 'ValidatedReportTable', side_effect=AssertionError('revalidated')), \
             patch.object(receiver.inputs, 'load_fixed_inputs', side_effect=AssertionError('reloaded')):
            for _ in range(2):
                producer._validate_root(self.root, self.cfg, self.decision, self.latent, self.design, self.pspec)
                receiver.validate_spec(self.root, self.cfg, self.decision, self.latent, self.design, self.rspec)


if __name__ == '__main__':
    unittest.main()
