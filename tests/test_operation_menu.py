"""Saved-only interval/menu science and public payload bindings.

Synthetic saved scalars/report metadata exercise the accepted algebra without
actors, probability laws, utility vectors or integrations. They are not the
historical scientific results or proof that arbitrary input was accepted.
"""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from src.design_validation import operation_menu as P
from src.design_validation import operation_menu_reference as R
from src.design_validation.personal_time_inputs import digest


def point(value):
    return [str(value), str(value)]


def fixture_grant(root, side=None, report=None):
    """Synthetic complete displayed H, independently assembled by the fixture."""
    means = list(root['display']['root'])
    budget = root['task']['total_millihours']
    latest, history = [None, None], []
    if side is not None:
        slot = root['orderedPhysical'].index(side)
        means[slot] = root['display']['singleReport'][slot][report-1]
        budget -= root['task']['cost_millihours']
        latest[slot] = report
        history = [dict(slot=slot, report=report)]
    return dict(stateId='synthetic-opaque-grant', budgetInteger=budget, meanStrings=means,
                resourceString=f'{budget//1000}.{budget%1000:03d}', latest=latest, history=history,
                ranges=[dict(lower=1, upper=100), dict(lower=1, upper=100)],
                channel=dict(source='task_width_metadata_not_actor_operand',
                             reportWidthHex=float(root['task']['report_width']).hex()))


def refresh(bundle):
    """Test-only portable adapter; no numeric source generation or re-evaluation."""
    bundle['sourceEvidence'] = {
        'cheap': {'producers': {name: {'sha256': digest(raw)} for name, raw in bundle['rawCheap'].items()}},
        'ask': {'producer': {'sha256': digest(bundle['rawAskDirections'])}},
        'fixture': 'synthetic_saved_values_not_historical_evidence',
    }
    bundle['sourceEvidenceSHA256'] = digest(bundle['sourceEvidence'])
    for source in bundle['roots']:
        root = source['root']
        source['rootSHA256'] = digest(root)
        source['lawContext'] = dict(schema='operation-menu-law-context-v1', rootSHA256=digest(root),
                                    logsHex=root['logsHex'], widthHex=float(root['task']['report_width']).hex(),
                                    precisionBits=160,
                                    law='normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
                                    support=[1, 100], latentSpecificRowNormalization=True)
        raw = bundle['rawCheap'][root['id'][0]]
        menu = []
        for actor in raw['actors']:
            ledger = actor['ledger']
            menu.append(dict(id='noask:'+actor['family'], kind='noask', family=actor['family'],
                             index=actor['index'], budget=root['task']['total_millihours'],
                             GPAInterval=actor['GPAInterval'],
                             knownSecondsInterval=point(ledger['knownSeconds']),
                             symbolCountIntervals={k: point(v) for k, v in ledger['symbols'].items()},
                             memberSecondsInterval=point(ledger['oneSecondMemberSeconds']),
                             sourceRaw=digest(raw), knownCoefficients=ledger['knownCoefficients']))
        source['askRows'] = {}
        for side in ('A', 'B'):
            row = next(x for x in bundle['rawAskDirections']
                       if x['id'] == root['id'] and x['physicalRecipient'] == side)
            source['askRows'][side] = [dict(report=z['report'], queriedVectorRef=z['queriedVectorRef'],
                                           otherVectorRef=z['otherVectorRef'],
                                           ledger={k: z['ledger'][k] for k in
                                                   ('knownSeconds', 'symbols', 'oneSecondMemberSeconds')},
                                           rawSHA256=digest(z)) for z in row['reports']]
            menu.append(dict(id='ask:'+side, kind='ask_once', physicalRecipient=side,
                             orderedSlot=row['orderedSlot'], budget=row['nextBudget'],
                             GPAInterval=row['expectedGPAInterval'],
                             knownSecondsInterval=row['expectedKnownSecondsInterval'],
                             symbolCountIntervals=row['expectedSymbolCountsIntervals'],
                             memberSecondsInterval=row['expectedMemberSecondsInterval'],
                             sourceRaw=digest(bundle['rawAskDirections']),
                             knownCoefficients=dict(R.KNOWN, request_execution='53/20')))
        source['menu'] = menu
    return bundle


def fixture(conditions=('C3', 'C5')):
    rhos = [str(F.from_float(float.fromhex(x))) for x in R.HEX]
    bundle = dict(roots=[], rawCheap={}, rawAskDirections=[], rhoValues=rhos)
    for condition in conditions:
        symmetric = condition == 'C5'
        root = dict(id=[condition, 0, 1], orderedPhysical=['A', 'B'],
                    logsHex=[['0x0.0p+0']*100, ['0x0.0p+0' if symmetric else '0x1.0p+0']*100],
                    task=dict(total_millihours=100000, cost_millihours=500,
                              alpha=.5, loss_multiplier=2, allocation_denominator=1000, report_width=2.0),
                    display={'root': ['50.0', '50.0' if symmetric else '60.0'],
                             'singleReport': [[str(y)+'.0' for y in range(1, 101)] for _ in range(2)]})
        grant = fixture_grant(root)
        indices = [500]*5 if symmetric else [500, 400, 600, 0, 0]
        scores = {str(j): {'interval': ['10', '11'] if symmetric else
                          {500: ['4', '5'], 400: ['6', '7'], 600: ['3', '4'], 0: ['2', '3']}[j]}
                  for j in indices}
        actors = []
        for family, index, seconds in zip(R.FAMILIES, indices, (1, 2, 3, 4, 4)):
            actors.append(dict(family=family, index=index, GPAInterval=scores[str(index)]['interval'],
                               ledger=dict(knownSeconds=str(seconds), symbols={},
                                           oneSecondMemberSeconds=str(seconds), knownCoefficients=dict(R.KNOWN)),
                               prices=[dict(rho=rho) for rho in rhos]))
        bundle['rawCheap'][condition] = dict(id=root['id'], H=grant, HSourceSHA256=digest(grant),
                                             actors=actors, scores=scores, newReports=0,
                                             scope='five_precommitted_noask_rules_not_free_best_of_five',
                                             unknownMember='each_used_unknown_one_second_not_measured')
        bundle['roots'].append(dict(id=root['id'], root=root))
        for side, slot in (('A', 0), ('B', 1)):
            rows = []
            for report in range(1, 101):
                report_grant = fixture_grant(root, side, report)
                rows.append(dict(id=root['id'], physicalRecipient=side, orderedSlot=slot, report=report,
                                 H=report_grant, HSHA256=digest(report_grant),
                                 queriedVectorRef='99500:500' if symmetric or side == 'A' else '99500:400',
                                 otherVectorRef='99500:500' if symmetric or side == 'A' else '99500:600',
                                 ledger=dict(knownSeconds='5', symbols={'store': 1},
                                             oneSecondMemberSeconds='6',
                                             knownCoefficients=dict(R.KNOWN, request_execution='53/20'))))
            bundle['rawAskDirections'].append(dict(
                id=root['id'], physicalRecipient=side, orderedSlot=slot, nextBudget=99500,
                expectedGPAInterval=['7', '8'] if symmetric or side == 'B' else ['9', '10'],
                expectedKnownSecondsInterval=['5', '6'], expectedSymbolCountsIntervals={'store': ['1', '2']},
                expectedMemberSecondsInterval=['6', '8'], reports=rows,
                prices=[dict(rho=rho) for rho in rhos], actualReportsDrawn=0, newPolicyTrajectories=0,
                scope='full_exante_precommitted_direction_not_reached_posterior',
                unknownMember='each_used_unknown_one_second_not_measured'))
    return refresh(bundle)


class OperationMenuTests(unittest.TestCase):
    def test_full_counts_independent_receiving_and_twelve_mutations(self):
        bundle = fixture()
        original = copy.deepcopy(bundle)
        result = P.produce(bundle)
        check = R.verify(bundle, result)
        self.assertEqual(check['status'], 'PASS')
        self.assertEqual((check['verifiedPairs'], check['verifiedPairPriceCells'], check['verifiedMenuPriceCells']),
                         (42, 336, 16))
        self.assertEqual(result['schema'], 'operation-menu-v1')
        self.assertTrue(result['fullSelectedScope'])
        for key in ('newActors', 'newLaws', 'newUtilities', 'newIntegrations', 'newTrajectories'):
            self.assertEqual(result[key], 0)
        self.assertEqual(R.semantic_checks(bundle, result)['status'], 'PASS')
        self.assertEqual(bundle, original)

    def test_explicit_subset_never_claims_full_scope(self):
        for conditions in (('C3',), ('C5',)):
            bundle = fixture(conditions)
            with self.assertRaises(ValueError):
                P.produce(bundle)
            result = P.produce(bundle, conditions=conditions)
            check = R.verify(bundle, result, conditions=conditions)
            self.assertFalse(check['fullSelectedScope'])
            self.assertEqual(result['counts'], dict(roots=1, policies=7, pairs=21,
                                                    pairPriceCells=168, menuPriceCells=8))
            with self.assertRaises(ValueError):
                R.verify(bundle, result)
            bad = copy.deepcopy(result)
            bad['fullSelectedScope'] = True
            with self.assertRaises(ValueError):
                R.verify(bundle, bad, conditions=conditions)

    def test_c5_role_exchange_is_structural_not_overlap(self):
        bundle = fixture()
        root = bundle['roots'][1]
        result = P.produce(bundle)['roots'][1]
        pair = next(x for x in result['pairs'] if (x['first'], x['second']) == ('ask:A', 'ask:B'))
        self.assertEqual(pair['gpaDifferenceInterval'], ['0', '0'])
        self.assertEqual(pair['memberTimeDifferenceInterval'], ['0', '0'])
        self.assertTrue(all(x['status'] == 'provable_tie' for x in pair['prices']))
        self.assertEqual(pair['symbolicConditions']['provableTieScope'], 'all_nonnegative_rho_and_unknown_seconds')
        self.assertIn('exact_equal_input_priors_shared_channel_payoff_and100role_operand_exchange',
                      pair['identityProof']['evidence'])
        # Recommit a synthetic changed operand, including its compact projection.
        direction = next(x for x in bundle['rawAskDirections']
                         if x['id'][0] == 'C5' and x['physicalRecipient'] == 'B')
        direction['reports'][0]['queriedVectorRef'] = '99500:501'
        refresh(bundle)
        a, b = root['menu'][-2:]
        changed = P.pair(root, a, b, bundle['rhoValues'])
        self.assertEqual(changed, R.checked_pair(root, a, b, bundle['rhoValues']))
        self.assertFalse(changed['identityProof']['sameUnderlyingG'])
        self.assertTrue(all(x['status'] == 'overlap' for x in changed['prices']))

    def test_overlap_is_not_equivalence_and_signed_open_regions(self):
        root = fixture(('C3',))['roots'][0]
        a, b = copy.deepcopy(root['menu'][:2])
        a['GPAInterval'] = b['GPAInterval'] = ['0', '2']
        a['memberSecondsInterval'] = b['memberSecondsInterval'] = ['1', '1']
        pair = P.pair(root, a, b, ['0', '1'])
        self.assertEqual(pair, R.checked_pair(root, a, b, ['0', '1']))
        self.assertTrue(all(x['status'] == 'overlap' for x in pair['prices']))
        menu = P.menu_sets([a, b], [pair], ['0', '1'])
        self.assertEqual(menu, R.best_sets([a, b], [pair], ['0', '1']))
        self.assertTrue(all(x['status'] == 'unresolved' and len(x['equivalenceGroups']) == 2 for x in menu))
        for intercept, penalty, expected in (
                (2, 3, dict(kind='below', threshold='2/3', lower='0', lowerClosed=True, upperClosed=False)),
                (-2, -3, dict(kind='above', threshold='2/3', lowerClosed=False)),
                (1, -3, dict(kind='all_nonnegative')), (0, 0, dict(kind='empty')),
                (1, 0, dict(kind='all_nonnegative')), (0, 1, dict(kind='empty'))):
            self.assertEqual(P.region(F(intercept), F(penalty)), expected)
            self.assertEqual(R.positive_prices(F(intercept), F(penalty)), expected)

    def test_exact_crossing_extra_time_and_symbolic_orientation(self):
        root = fixture(('C3',))['roots'][0]
        a, b = copy.deepcopy(root['menu'][:2])
        a.update(GPAInterval=['3', '3'], knownSecondsInterval=['1', '1'],
                 symbolCountIntervals={'store': ['1', '1']}, memberSecondsInterval=['2', '2'])
        b.update(GPAInterval=['1', '1'], knownSecondsInterval=['1', '1'],
                 symbolCountIntervals={}, memberSecondsInterval=['1', '1'])
        result = P.pair(root, a, b, ['0', '1', '2', '3'])
        self.assertEqual(result, R.checked_pair(root, a, b, ['0', '1', '2', '3']))
        self.assertEqual([x['status'] for x in result['prices']],
                         ['first_strictly_better', 'first_strictly_better', 'provable_tie', 'second_strictly_better'])
        self.assertEqual(result['prices'][0]['extraTimeCondition']['status'], 'all_nonnegative_personal_extra_time')
        self.assertEqual(result['prices'][1]['extraTimeCondition']['secondsThreshold'], '1')
        self.assertEqual(result['prices'][1]['extraTimeCondition']['policy'], a['id'])
        self.assertEqual(result['prices'][3]['extraTimeCondition']['secondsThreshold'], '1/3')
        self.assertEqual(result['prices'][3]['extraTimeCondition']['policy'], b['id'])
        self.assertEqual(result['symbolicConditions']['firstStrict']['unknownCountUpperDifferences'], {'store': '1'})
        self.assertEqual(result['symbolicConditions']['secondStrict']['unknownCountUpperDifferences'], {'store': '-1'})
        self.assertEqual(result['oneSecondMemberRhoConditions']['provableTie'], dict(kind='exact_price', rho='2'))

    def test_public_H_and_root_digests_must_not_be_conflated(self):
        bundle = fixture()
        root = bundle['roots'][0]
        raw = bundle['rawCheap']['C3']
        self.assertNotEqual(raw['HSourceSHA256'], root['rootSHA256'])
        self.assertEqual(raw['HSourceSHA256'], digest(raw['H']))
        raw['HSourceSHA256'] = root['rootSHA256']
        refresh(bundle)
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))

    def test_recommitted_report_channel_and_semantic_grants_are_refused(self):
        for edit in (
                lambda h: h['channel'].update(reportWidthHex='0x1.0p+2'),
                lambda h: h['channel'].update(source='actor_operand'),
                lambda h: h.update(budgetInteger=99999),
                lambda h: h.update(resourceString='99.999'),
                lambda h: h['meanStrings'].__setitem__(1, 'forged'),
                lambda h: h.update(history=[dict(slot=0, report=1)]),
                lambda h: h.update(latest=[1, None])):
            bundle = fixture()
            row = next(d for d in bundle['rawAskDirections']
                       if d['id'][0] == 'C5' and d['physicalRecipient'] == 'B')['reports'][0]
            edit(row['H'])
            row['HSHA256'] = digest(row['H'])
            refresh(bundle)
            with self.assertRaises(ValueError):
                R.verify(bundle, P.produce(bundle))
        bundle = fixture()
        bundle['rawCheap']['C5']['H']['channel']['reportWidthHex'] = '0x1.0p+2'
        bundle['rawCheap']['C5']['HSourceSHA256'] = digest(bundle['rawCheap']['C5']['H'])
        refresh(bundle)
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))
        bundle = fixture()
        bundle['roots'][1]['root']['display']['singleReport'][1][0] = 'forged-root-display'
        refresh(bundle)
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))

    def test_canonical_law_context_and_actual_H_hash_not_state_label_spelling(self):
        bundle = fixture()
        row = bundle['rawAskDirections'][0]['reports'][0]
        row['H']['stateId'] = 'alternative-opaque-provenance-label'
        row['HSHA256'] = digest(row['H'])
        refresh(bundle)
        self.assertEqual(R.verify(bundle, P.produce(bundle))['status'], 'PASS')
        row['HSHA256'] = '0'*64
        refresh(bundle)
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))
        bundle = fixture()
        bundle['roots'][1]['lawContext']['widthHex'] = '0x1.0p+2'
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))

    def test_current_report_counter_and_compact_binding_required(self):
        for edit in (
                lambda b: b['rawAskDirections'][0].pop('actualReportsDrawn'),
                lambda b: b['rawAskDirections'][0].update(actualReportsDrawn=1),
                lambda b: b['rawAskDirections'][0].update(scope='conditional_reached_posterior'),
                lambda b: b['rawAskDirections'][0]['reports'].pop(),
                lambda b: b['rawAskDirections'][0]['reports'][0].update(orderedSlot=1)):
            bundle = fixture()
            edit(bundle)
            refresh(bundle)
            with self.assertRaises((ValueError, KeyError)):
                R.verify(bundle, P.produce(bundle))
        bundle = fixture()
        bundle['roots'][0]['askRows']['A'][0]['rawSHA256'] = '0'*64
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))

    def test_source_commitment_and_frozen_prices_cannot_be_silently_reused(self):
        bundle = fixture()
        bundle['sourceEvidence']['ask']['producer']['sha256'] = '0'*64
        bundle['sourceEvidenceSHA256'] = digest(bundle['sourceEvidence'])
        with self.assertRaises(ValueError):
            R.verify(bundle, P.produce(bundle))
        for value in ('-1', '1e-8'):
            bundle = fixture()
            bundle['rhoValues'][0] = value
            with self.assertRaises(ValueError):
                P.produce(bundle)

    def test_public_adapter_projects_saved_payloads_without_running_policies(self):
        from src.design_validation import operation_menu_reproduction as adapter

        for conditions in (('C3', 'C5'), ('C3',), ('C5',)):
            saved = fixture(conditions)
            native = [row['root'] for row in saved['roots']]
            admitted = []

            def checked_product(source, binding, unit):
                admitted.append(unit[0])
                if unit[1] == 'noask':
                    raw = saved['rawCheap'][unit[2]['id'][0]]
                else:
                    raw = next(row for row in saved['rawAskDirections']
                               if row['id'] == unit[2]['id'] and row['physicalRecipient'] == unit[3])
                # Preserve envelope pins separately from normalized payload pins.
                return raw, dict(productSHA256=digest({'result': raw}),
                                 checkSHA256=digest({'checkedSyntheticUnit': unit[0]}))

            # Only the admission boundary is mocked; public normalization is real.
            with patch.object(adapter.I, 'load', return_value=({'fixture': True}, native)), \
                    patch.object(adapter, '_product', side_effect=checked_product):
                bundle = adapter.load_inputs('unused-saved-input', 'unused-config', conditions)
            self.assertEqual(admitted, [name for c in conditions
                                        for name in ('noask_'+c, 'ask_'+c+'_A', 'ask_'+c+'_B')])
            self.assertEqual(bundle['roots'], saved['roots'])
            self.assertEqual(bundle['rawCheap'], saved['rawCheap'])
            self.assertEqual(bundle['rawAskDirections'], saved['rawAskDirections'])
            for c in conditions:
                evidence = bundle['sourceEvidence']
                self.assertEqual(evidence['cheap']['producers'][c]['sha256'], digest(saved['rawCheap'][c]))
                self.assertNotEqual(evidence['products']['noask_'+c]['productSHA256'],
                                    evidence['cheap']['producers'][c]['sha256'])
            result = P.produce(bundle, conditions=conditions)
            self.assertEqual(R.verify(bundle, result, conditions=conditions)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
