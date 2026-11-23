"""Portable menu routing and refusal tests using labelled synthetic operands.

No actor, law or utility is generated. Mocking native input loading isolates
the saved-product adapter; this is not upstream numerical validation.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import operation_menu_reproduction as R
from src.design_validation import personal_time_inputs as I
from src.design_validation.operation_costs.checkpoint import Workspace
from test_operation_menu import fixture

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def workspace(path, bundle):
    binding = {'testOnly': 'synthetic_saved_products_not_original_science'}
    native = [r['root'] for r in bundle['roots']]
    for unit in R.R.units(native, ('noask', 'ask_once'), ('A', 'B')):
        name, family, root, side = unit
        value = (bundle['rawCheap'][root['id'][0]] if family == 'noask' else
                 next(d for d in bundle['rawAskDirections']
                      if d['id'] == root['id'] and d['physicalRecipient'] == side))
        context = R.R.scientific_context(unit)
        Workspace(path, binding).put('result_' + name, context, value)
        product = dict(context=context, result=value)
        proof = dict(status='PASS', id=root['id'])
        proof.update(dict(actors=5, priceRows=40) if family == 'noask' else
                     dict(reportRows=100, comparisons=16, physicalRecipient=side))
        I.write(path / 'products' / (name + '.json'), product)
        I.write(path / 'checks' / (name + '.json'), dict(unit=name, status='PASS',
            bindingSHA256=I.digest(binding), productSHA256=I.digest(product), independent=proof))
    return binding, native


class MenuReproductionTests(unittest.TestCase):
    def test_checked_product_adapter_and_immutable_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source', Path(tmp) / 'result'
            binding, native = workspace(source, fixture())
            with patch.object(R.I, 'load', return_value=(binding, native)):
                result = R.run(source, output, CONFIGS)
                self.assertEqual(result['verifiedPairs'], 42)
                self.assertTrue(result['newSavedReduction'])
                self.assertFalse(R.run(source, output, CONFIGS, checking=True)['newSavedReduction'])
                self.assertTrue((output / 'menu_best.csv').is_file())
                self.assertTrue((output / 'pair_prices.csv').is_file())
                self.assertTrue((output / 'sufficient_conditions.json').is_file())
                path = source / 'checks/ask_C5_B.json'
                changed = I.read(path)
                changed['independent']['physicalRecipient'] = 'A'
                path.unlink()
                I.write(path, changed)
                with self.assertRaises(ValueError):
                    R.run(source, output, CONFIGS, checking=True)

    def test_explicit_compact_scope_and_changed_input_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source', Path(tmp) / 'result'
            bundle = fixture(('C5',))
            R.export_compact(bundle, source / 'operands.json')
            result = R.run(source, output, CONFIGS, conditions=('C5',), input_kind='saved-operands')
            self.assertFalse(result['historicalExecutionRevalidated'])
            self.assertFalse(result['fullSelectedScope'])
            self.assertEqual(result['verifiedPairPriceCells'], 168)
            self.assertFalse(R.run(source, output, CONFIGS, conditions=('C5',),
                                  input_kind='saved-operands', checking=True)['newSavedReduction'])
            changed_sources = dict(R.source_identity(), changed_dependency='not_the_original_code')
            with patch.object(R, 'source_identity', return_value=changed_sources):
                with self.assertRaises(ValueError):
                    R.run(source, output, CONFIGS, conditions=('C5',), input_kind='saved-operands',
                          checking=True)
            changed = copy.deepcopy(bundle)
            changed['sourceEvidence']['changed'] = True
            changed['sourceEvidenceSHA256'] = I.digest(changed['sourceEvidence'])
            (source / 'operands.json').unlink()
            R.export_compact(changed, source / 'operands.json')
            with self.assertRaises(ValueError):
                R.run(source, output, CONFIGS, conditions=('C5',), input_kind='saved-operands')

    def test_missing_upstream_is_not_silently_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            binding, native = workspace(source, fixture())
            (source / 'products/ask_C3_B.json').unlink()
            with patch.object(R.I, 'load', return_value=(binding, native)):
                with self.assertRaises(ValueError):
                    R.run(source, Path(tmp) / 'result', CONFIGS)


if __name__ == '__main__':
    unittest.main()
