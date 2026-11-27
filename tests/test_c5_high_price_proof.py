"""Rejection contracts; fixtures are not scientific evidence."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.design_validation import c5_high_price_proof as proof
from src.design_validation import c5_moments as moments
from src.design_validation import c5_public_authority as authority
from src.design_validation import c5_public_inputs as public_inputs
from src.design_validation import c5_reproduction as reproduction
from src.design_validation import coefficient_io
from src.design_validation import personal_time_inputs as inputs
from src.design_validation import personal_time_sensitivity as sensitivity


def member():
    root = {'conditionId': 'C5', 'scheduleVariant': 0, 'mainIndex': 1, 'seed': 1,
            'definition': {'budgetMillihours': 1000}, 'logicalToDisplayed': ['A', 'B'],
            'initialLogs': [[], []], 'common': {}}
    cfg = {'valuation': {'rhoGPAperSecond': 0.01}}
    endpoint = {'decision': {'action': 'MFN_A', 'stop': {'action': 'MFN_A', 'index': 1000}},
                'terminal': {'action': 'MFN_A', 'intendedIndex': 1000, 'budget': 1000,
                             'newScientificDraws': 0, 'realizedCombinedGPA': None,
                             'realizedNetObjective': None, 'searchSeconds': 0,
                             'targetHistoryFabricated': False,
                             'scope': 'exact_initial_stop_expected_endpoint_not_realized_history'}}
    mass = {'counts': [moments.GRID] + [0] * 99, 'denominator': moments.GRID,
            'lastCategoryIncludesFallback': True}
    law = {'rootKey': list(inputs.coordinates(root)), 'rootSHA256': inputs.scientific_digest(root),
           'budget': 1000, 'logicalToDisplayed': ['A', 'B'], 'laws': [mass, copy.deepcopy(mass)]}
    return root, cfg, endpoint, copy.deepcopy(endpoint), law


class HighPriceProofTests(unittest.TestCase):
    def test_member_requires_actual_consistent_endpoint_and_laws(self):
        args = member()
        proof._validate_member(*args)
        mutations = (
            (1, lambda x: x['valuation'].__setitem__('rhoGPAperSecond', 0.001)),
            (2, lambda x: x['decision'].__setitem__('action', 'ask_A')),
            (3, lambda x: x['terminal'].__setitem__('newScientificDraws', 1)),
            (2, lambda x: x['terminal'].__setitem__('searchSeconds', 1)),
            (4, lambda x: x.__setitem__('budget', 999)),
            (4, lambda x: x.__setitem__('logicalToDisplayed', ['B', 'A'])),
            (4, lambda x: x['laws'][1]['counts'].__setitem__(0, moments.GRID - 1)),
            (4, lambda x: x['laws'][0].__setitem__('lastCategoryIncludesFallback', False)),
        )
        for index, mutate in mutations:
            changed = copy.deepcopy(args)
            mutate(changed[index])
            with self.subTest(index=index, mutation=mutate), self.assertRaises(ValueError):
                proof._validate_member(*changed)

    def test_boolean_pass_is_not_an_endpoint(self):
        root, cfg, _, independent, law = member()
        with self.assertRaises(ValueError):
            proof._validate_member(root, cfg, True, independent, law)

    def test_missing_duplicate_and_singular_rosters_are_refused(self):
        _, _, _, _, law = member()
        records = []
        for v in range(9):
            for m in range(1, 11):
                row = copy.deepcopy(law)
                row['rootKey'] = ['C5', v, m]
                records.append(row)
        for invalid in (records[:-1], records[:-1] + [records[0]], records):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                proof._full_rank(invalid)

    def test_full_rank_fixture_covers_both_models_and_nine_schedules(self):
        _, _, _, _, law = member()
        counts = [moments.GRID // 4, moments.GRID // 4, moments.GRID // 2] + [0] * 97
        law['laws'][0]['counts'] = counts
        law['laws'][1]['counts'] = counts[:]
        records = []
        for v in range(9):
            for m in range(1, 11):
                row = copy.deepcopy(law)
                row['rootKey'] = ['C5', v, m]
                records.append(row)
        ranks = proof._full_rank(records)
        self.assertEqual([row['schedule'] for row in ranks], list(range(9)))
        self.assertTrue(all(set(row['models']) == {'gap', 'partial_low'} for row in ranks))
        for row in ranks:
            self.assertEqual(row['models']['gap']['coefficients'], [['1', '2'], ['0', '1']])
            self.assertEqual(row['models']['partial_low']['coefficients'],
                             [['1', '2'], ['0', '1'], ['0', '1']])

    def test_rehashed_forgery_cannot_replace_actual_evaluator(self):
        root, cfg, actual, independent, _ = member()
        forged = copy.deepcopy(actual)
        forged['terminal']['searchSeconds'] = 0.5
        binding = {'root': 'fixture'}
        body = {'schema': 'coefficient-checkpoint-v1', 'binding': binding, 'payload': forged}
        envelope = {**body, 'contentSHA256': inputs.digest(body)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forged.json'
            inputs.write(path, envelope)
            payload, _ = coefficient_io.stage(path, binding, None, readonly=True)
            with patch.object(proof.C, '_high_source', side_effect=[actual, independent]):
                with self.assertRaisesRegex(ValueError, 'source replay mismatch'):
                    proof._source_replay(root, cfg, payload, independent)

    def test_adopted_target_rejects_changed_root_even_if_rehashed(self):
        root, _, _, _, _ = member()
        low_cfg = {'task': {'total_millihours': 1000, 'cost_millihours': 10}}
        root['definition'].update(reportWidth=3, costMillihours=10)
        row = {'rootKey': list(inputs.coordinates(root)), 'rootSHA256': inputs.digest(root),
               'configSHA256': inputs.digest(low_cfg), 'config': low_cfg,
               'logicalToDisplayed': ['A', 'B'], 'reportWidth': 3}
        with patch.object(sensitivity, 'rate_config', return_value=low_cfg):
            proof._target_member(root, {}, row)
            changed = copy.deepcopy(root)
            changed['definition']['budgetMillihours'] = 999
            with self.assertRaisesRegex(ValueError, 'adopted fixed lookup'):
                proof._target_member(changed, {}, row)

    def test_public_entry_requires_source_setup_not_pass_flags(self):
        with patch.object(proof.C, '_setup', side_effect=ValueError('missing bound source')):
            with self.assertRaisesRegex(ValueError, 'missing bound source'):
                with patch.object(proof.public_inputs, 'load_fixed_inputs',
                                  return_value={'roots': [{'rootKey': list(k)} for k in sorted(proof.C.ROSTER)]}):
                    proof.prove('missing', 'missing', adopted_lookup='missing')


@unittest.skipUnless(os.environ.get('C5_HIGH_PRICE_SOURCE_FIXTURE') == '1',
                     'scheduled source-bound C5 workspace required')
class RealSourceFixtureTests(unittest.TestCase):
    def test_source_replay_and_rehashed_checkpoint_forgery(self):
        output = os.environ['C5_HIGH_PRICE_OUTPUT']
        config = os.environ['C5_HIGH_PRICE_CONFIG']
        lookup_path = os.environ['C5_HIGH_PRICE_LOOKUP']
        policy_path = os.environ.get('C5_HIGH_PRICE_POLICY') or None
        lookup = public_inputs.load_fixed_inputs(lookup_path, authority.ADOPTED_INITIAL_SHA256)
        output, context, roots, _, run, policy, _ = reproduction._setup(
            output, config, reproduction.selection(full=True), policy_path, checking=True)
        root = roots[0]
        row = next(row for row in lookup['roots'] if row['rootKey'] == list(inputs.coordinates(root)))
        proof._target_member(root, context, row)
        high, _ = reproduction._high(output, root, context, run, policy, readonly=True)
        independent, _ = reproduction._high(output, root, context, run, policy,
                                             independent=True, readonly=True)
        cfg = sensitivity.rate_config(context, root, 'rho_0.01')
        proof._source_replay(root, cfg, high, independent)
        law = reproduction.law_record(root, independent, independent=True)
        proof._validate_member(root, cfg, high, independent, law)
        binding = reproduction._root_binding(run, root)
        forged = copy.deepcopy(high)
        forged['terminal']['searchSeconds'] = 1.0
        body = {'schema': 'coefficient-checkpoint-v1', 'binding': binding, 'payload': forged}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forged.json'
            inputs.write(path, {**body, 'contentSHA256': inputs.digest(body)})
            payload, _ = coefficient_io.stage(path, binding, None, readonly=True)
            with self.assertRaisesRegex(ValueError, 'source replay mismatch'):
                proof._source_replay(root, cfg, payload, independent)

    @unittest.skipUnless(os.environ.get('C5_HIGH_PRICE_FULL_PROOF') == '1',
                         'full 90-root source replay explicitly requested')
    def test_full_source_bound_entry(self):
        result = proof.prove(os.environ['C5_HIGH_PRICE_OUTPUT'], os.environ['C5_HIGH_PRICE_CONFIG'],
                             adopted_lookup=os.environ['C5_HIGH_PRICE_LOOKUP'],
                             policy=os.environ.get('C5_HIGH_PRICE_POLICY') or None)
        self.assertEqual(result['verifiedRoots'], 90)
        self.assertEqual(result['highPricePopulationSlopes'],
                         {'gap': ['0', '1'], 'partial_low': ['0', '1']})


if __name__ == '__main__':
    unittest.main()
