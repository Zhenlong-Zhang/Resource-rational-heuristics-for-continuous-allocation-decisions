"""Assembly wiring and rejection tests; fixtures are not scientific evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import c5_input_assembly as assembly
from src.design_validation import c5_public_inputs as inputs


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.raw = b'{}'
        self.roots, rows = [], []
        for key in inputs.ROSTER:
            root = {'conditionId': key[0], 'scheduleVariant': key[1], 'mainIndex': key[2],
                    'logicalToDisplayed': ['A', 'B'], 'initialLogs': [[0], [0]],
                    'definition': {'reportWidth': 3.0, 'budgetMillihours': 11, 'costMillihours': 5}}
            cfg = {'task': {'report_width': 3.0, 'total_millihours': 11,
                            'cost_millihours': 5, 'max_assessments': 2}}
            self.roots.append(root)
            rows.append({'rootSHA256': inputs._digest(root), 'configSHA256': inputs._digest(cfg),
                         'config': cfg, 'logicalToDisplayed': ['A', 'B'], 'reportWidth': 3.0})
        self.lookup = {'roots': rows}
        self.cfg = rows[0]['config']
        self.latent = type('Latent', (), {'roots': tuple((([7], []), ([7], [])) for _ in range(90)),
                                          'input_sha256': 'latent-object-digest'})()
        self.table = type('Table', (), {'accepted_product_bytes': True, 'product_sha256': 'table-digest'})()
        self.design = {'gap': {'status': 'available'}, 'partial_low': {'status': 'available'}}
        self.patches = [
            patch.object(assembly, 'LATENT_BYTES_SHA256', hashlib.sha256(self.raw).hexdigest()),
            patch.object(assembly, 'LOOKUP_BYTES_SHA256', 'lookup-sha'),
            patch.object(inputs, 'load_fixed_inputs', return_value=self.lookup),
            patch.object(assembly.c5_public_streams, 'ValidatedLatentConfig', return_value=self.latent),
            patch.object(assembly.c5_public_streams, 'ValidatedReportTable', return_value=self.table),
            patch.object(assembly.personal_time_sensitivity, 'rate_config', return_value=self.cfg),
            patch.object(assembly.c5_moments, 'design_payload', return_value=self.design),
            patch.object(assembly.c5_reference, 'design_reference', return_value=self.design),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def run_assembly(self):
        return assembly.assemble('lookup', 'lookup-sha', self.raw, self.roots, {},
                                 {float(3).hex(): b'fixture'})

    def test_complete_wiring_is_not_authority(self):
        result = self.run_assembly()
        self.assertEqual(len(result['geometryCertificates']), 9)
        self.assertEqual(result['geometryCertificates'][8]['lookupSHA256'], 'lookup-sha')
        self.assertFalse(result['samplingAuthorized'])
        self.assertFalse(result['historicalRuntimeEquivalence'])
        self.assertEqual(result['latentBytesSHA256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(result['latentCanonicalObjectSHA256'], 'latent-object-digest')
        self.assertEqual(result['lawIdentity'], 'adopted-fixed-integer-partitions-not-recomputed')

    def test_producer_envelope_and_existing_list_interface(self):
        with patch.object(assembly.personal_time_inputs, 'validate_roots', return_value=self.roots) as validate:
            result = assembly.assemble('lookup', 'lookup-sha', self.raw, {'roots': self.roots},
                                       {'reference': {}}, {float(3).hex(): b'fixture'})
        self.assertEqual(len(result['geometryCertificates']), 9)
        validate.assert_called_once()
        self.run_assembly()

    def test_unaccepted_lookup_pin_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lookup byte identity'):
            assembly.assemble('lookup', 'wrong-sha', self.raw, self.roots, {}, {})

    def test_root_mapping_and_roster_rejected(self):
        self.roots[0]['logicalToDisplayed'].reverse()
        with self.assertRaises(ValueError):
            self.run_assembly()
        self.roots[0]['logicalToDisplayed'].reverse()
        self.roots.reverse()
        with self.assertRaises(ValueError):
            self.run_assembly()

    def test_geometry_mismatch_rejected(self):
        with patch.object(assembly.c5_reference, 'design_reference', return_value={}):
            with self.assertRaises(ValueError):
                self.run_assembly()

    def test_fixed_laws_do_not_recompute_floating_point_priors(self):
        with patch.object(assembly.c5_moments, 'finite_law', side_effect=AssertionError('unexpected recomputation')), \
             patch.object(assembly.c5_reference, 'finite_law', side_effect=AssertionError('unexpected recomputation')):
            self.assertEqual(self.run_assembly()['lawIdentity'],
                             'adopted-fixed-integer-partitions-not-recomputed')

    def test_unverified_product_and_bad_budget_rejected(self):
        self.table.accepted_product_bytes = False
        with self.assertRaises(ValueError):
            self.run_assembly()
        self.table.accepted_product_bytes = True
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg['task']['max_assessments'] = 3
        self.lookup['roots'][0]['config'] = self.cfg
        self.lookup['roots'][0]['configSHA256'] = inputs._digest(self.cfg)
        with patch.object(assembly.personal_time_sensitivity, 'rate_config', return_value=self.cfg):
            with self.assertRaises(ValueError):
                self.run_assembly()

    def test_wrong_latent_bytes_rejected(self):
        with self.assertRaises(ValueError):
            assembly.assemble('lookup', 'lookup-sha', b'changed', self.roots, {}, {})


@unittest.skipUnless(os.environ.get('C5_RUN_SOURCE_ASSEMBLY_INTEGRATION') == '1',
                     'Opt in on the compatible scientific runtime; generates all C5 source inputs')
class SourceProducerIntegration(unittest.TestCase):
    """Actual native root and report generators, not a historical outcome replay.

    This deliberately exercises all 90 roots and nine exact designs. It is
    opt-in because it is computational work, and requires the compatible
    numerical environment and Node, not just the normal unit-test fixtures.
    """

    def test_real_source_envelope_assembly(self):
        from src.design_validation import canonical_report, personal_time_inputs, personal_time_sensitivity

        root = Path(__file__).resolve().parents[1]
        context = personal_time_sensitivity.load_context(root / 'configs/personal_time_sensitivity.json')
        selection = personal_time_inputs.selection(
            [dict(zip(personal_time_inputs.FIELDS, key)) for key in inputs.ROSTER])
        with tempfile.TemporaryDirectory() as directory:
            envelope = personal_time_inputs.prepare(context['reference'], selection, directory)
            widths = {float(item['definition']['reportWidth']).hex() for item in envelope['roots']}
            reports = {}
            for width in widths:
                table = canonical_report.generate_report_table(width)
                reports[width] = (json.dumps(table, sort_keys=True, separators=(',', ':')) + '\n').encode()
            result = assembly.assemble(root / 'configs/c5_fixed_initial_policy.json',
                                       assembly.LOOKUP_BYTES_SHA256,
                                       (root / 'configs/canonical_c5_latent.json').read_bytes(),
                                       envelope, context, reports)
        self.assertEqual(len(result['geometryCertificates']), 9)
        self.assertFalse(result['samplingAuthorized'])


if __name__ == '__main__':
    unittest.main()
