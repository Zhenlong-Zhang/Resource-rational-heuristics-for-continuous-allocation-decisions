"""Purpose: retain literal operation arithmetic, independent ES and scoped resume.

Synthetic numerical fixtures test the source interface, not historical five-root
parity. All executable verification is scheduled separately; no data is bundled.
"""
import copy
from dataclasses import asdict
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from threadpoolctl import threadpool_limits

from src.design_validation.bounded_task import TaskConfig
from src.design_validation.operation_costs import costs as C, literal as P, literal_reference as R
from src.design_validation.operation_costs.checkpoint import Workspace


def root_fixture():
    task = asdict(TaskConfig(condition_id='C1', template_id='synthetic-source-interface',
                            scale=1., gap_offset=0., prior_centres=(52., 53.),
                            prior_spreads=(3., 3.), report_width=30.,
                            total_millihours=114000, cost_millihours=1140,
                            repeat_count_per_30=10))
    logs = [[float(-.5*((need-mean)/3.)**2).hex() for need in range(1, 101)]
            for mean in (52., 53.)]
    return dict(id=['C1', 0, 1], task=task, logsHex=logs, orderedPhysical=['A', 'B'])


class CountTests(unittest.TestCase):
    def test_original_fee_and_complete_ledger_identities(self):
        self.assertEqual(C.RESPONSE, F(73, 20))
        self.assertEqual(C.REQUEST, F(93, 20))
        for budget, fee, count in ((178500, 525, 0), (177975, 525, 1), (0, 525, 3),
                                   (524, 525, 2), (525, 525, 2)):
            produced = C.computation(budget, fee, count)
            counts, seconds = R.counts_reference(budget, fee, count)
            self.assertEqual(produced['counts'], counts)
            self.assertEqual(F(produced['seconds']), seconds)
        self.assertEqual(F(C.computation(178500, 525, 0)['seconds'])+C.RESPONSE,
                         F(586824422, 5))
        total = sum((F(C.computation(178500-525*h, 525, h)['seconds']) for h in range(3)), F())
        self.assertEqual(total+2*C.REQUEST+C.RESPONSE, F(7041981619, 20))

    def test_exact_price_and_current_cost_sunk(self):
        root = root_fixture()
        root['costModel'] = {'rhoHex': P.PRICES[1]}
        stop, asks, ledger = C.price(root, 114000, [[], []], F(2), {0: F(3), 1: F(4)})
        rho = F.from_float(float.fromhex(P.PRICES[1]))
        base, scale = 2*F(3.5), F(.03)
        self.assertEqual(stop, base+scale*2-rho*C.RESPONSE)
        self.assertEqual(asks[0], base+scale*3-rho*(C.REQUEST+F(ledger['future']['seconds'])+C.RESPONSE))
        self.assertGreater(F(ledger['current']['seconds']), 0)
        _, noasks, blocked = C.price(root, 0, [[], []], F(-2), {})
        self.assertEqual(noasks, {})
        self.assertIsNone(blocked['future'])

    def test_invalid_event_domains(self):
        for args in ((True, 1, 0), (-1, 1, 0), (1, 0, 0), (1, 1, True)):
            with self.assertRaises(ValueError):
                C.computation(*args)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = root_fixture()
        with threadpool_limits(limits=1):
            cls.result = P.produce(cls.root, P.PRICES[2])

    def test_fresh_source_to_independent_reference(self):
        with threadpool_limits(limits=1):
            checked = R.verify(self.root, P.PRICES[2], self.result)
        self.assertEqual(checked['status'], 'PASS')
        self.assertEqual(checked['paidReports'], 0)
        self.assertEqual(self.result['immediateES']['ESFeeSeconds'], '73/20')
        self.assertEqual(self.result['comparison']['comparisonScope'], 'same_initial_information')
        self.assertNotIn('costModel', self.root)
        self.assertNotIn('acceptedNumericProofs', self.root)

    def test_scalar_signed_enclosure(self):
        expected = R.root_value(self.root)
        actual = list(map(F, P.es_value(self.root)['expectedGPAInterval']))
        self.assertLessEqual(actual[0], expected[0])
        self.assertLessEqual(expected[0], expected[1])
        self.assertLessEqual(expected[1], actual[1])

    def test_source_comparison_does_not_read_historical_proof_cache(self):
        root = copy.deepcopy(self.root)
        root.update(costModel={'rhoHex': P.PRICES[2]}, acceptedNumericProofs={'bogus': 'rejected'},
                    acceptedES={'expectedGPAInterval': ['-999', '-999']})
        raw = copy.deepcopy(self.result['steps'][0]['proof']['unpricedSourceProof'])
        oracle = object()
        with patch.object(P.adaptive.Policy, 'compare', return_value=raw) as compare:
            policy = P.Policy(root, oracle)
            result = policy.compare(root, root['task']['total_millihours'], ((), ()))
        compare.assert_called_once()
        self.assertEqual(result['unpricedSourceProof'], raw)

    def test_reference_rejects_altered_cost_score_and_conditioning(self):
        bad = []
        changed = copy.deepcopy(self.result)
        changed['steps'][0]['proof']['costLedger']['current']['counts']['store'] += 1
        bad.append(changed)
        changed = copy.deepcopy(self.result)
        changed['immediateES']['expectedGPAInterval'] = ['0', '0']
        bad.append(changed)
        changed = copy.deepcopy(self.result)
        changed['actualPersonalSeconds'] = '0'
        bad.append(changed)
        changed = copy.deepcopy(self.result)
        changed['comparison']['comparisonScope'] = 'conditional_path_only'
        bad.append(changed)
        with threadpool_limits(limits=1):
            for changed in bad:
                with self.subTest(changed=changed['actualPersonalSeconds']), self.assertRaises(ValueError):
                    R.verify(self.root, P.PRICES[2], changed)

    def test_conditional_endpoint_cannot_rank_against_initial_es(self):
        changed = copy.deepcopy(self.result)
        last = changed['steps'][-1]
        last['cursorAfter'] = [1, 1]
        last['budgetAfter'] -= 2*self.root['task']['cost_millihours']
        last['proof']['orderedHistory'] = [[50], [50]]
        row = P.comparison(self.root, P.PRICES[0], changed, changed['immediateES'])
        self.assertEqual(row['comparison'], 'conditioning_not_comparable')
        self.assertNotIn('costedMinusESCriterionInterval', row)

    def test_exact_scope_refusals(self):
        for root, rho in ((self.root, '0.0'), (dict(self.root, id=['C1', True, 1]), P.PRICES[0]),
                          (dict(self.root, id=['C1', 0, 2]), P.PRICES[0])):
            with self.assertRaises(ValueError):
                P.produce(root, rho)

    def test_bound_complete_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            workspace = Workspace(directory, {'synthetic_test_binding': 1})
            first = P.produce(self.root, P.PRICES[2], workspace)
            with patch.object(P.Policy, 'compare', side_effect=AssertionError('replayed completed decision')):
                second = P.produce(self.root, P.PRICES[2], Workspace(directory, {'synthetic_test_binding': 1}))
            self.assertEqual(json.loads(json.dumps(first)), second)
            # A committed final path with missing postprocessing is a valid prefix.
            (Path(directory)/'checkpoints/literal_C1_2_result.json').unlink()
            with patch.object(P.Policy, 'compare', side_effect=AssertionError('replayed committed prefix')):
                resumed = P.produce(self.root, P.PRICES[2], Workspace(directory, {'synthetic_test_binding': 1}))
            self.assertEqual(json.loads(json.dumps(first)), resumed)
            R.verify(self.root, P.PRICES[2], resumed)
            with self.assertRaises(ValueError):
                P.produce(self.root, P.PRICES[2], Workspace(directory, {'synthetic_test_binding': 2}))
            changed = copy.deepcopy(self.root)
            changed['logsHex'][0][0] = float(-1).hex()
            with self.assertRaises(ValueError):
                P.produce(changed, P.PRICES[2], Workspace(directory, {'synthetic_test_binding': 1}))
            path = Path(directory)/'checkpoints/literal_C1_2_result.json'
            path.write_text('{')
            with self.assertRaises(ValueError):
                P.produce(self.root, P.PRICES[2], Workspace(directory, {'synthetic_test_binding': 1}))


if __name__ == '__main__':
    unittest.main()
