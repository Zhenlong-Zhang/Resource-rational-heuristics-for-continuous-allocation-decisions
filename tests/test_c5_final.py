"""Software fixtures only: no historical identity or policy execution."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from src.design_validation import c5_final as P
from src.design_validation import c5_final_reference as R


DESIGN = {'gap': [['1', '1'], ['1', '2']],
          'partial_low': [['0', '1'], ['1', '1'], ['-1', '2']]}


def row(rep=1, variant=0, main=1, *, tie=False, saved=False):
    root = ['C5', variant, main]
    needs = [3, 3] if tie else [2, 4]
    endpoint = {'rootKey': root, 'latentDisplayed': needs,
                'latentLogical': {'A': needs[0], 'B': needs[1]},
                'initialBudget': 1000, 'remainingBudget': 1000, 'queryCount': 0,
                'status': 'exact_tie' if tie else 'complete', 'nonTie': not tie,
                'reason': 'equal_true_needs' if tie else None,
                'higherNeedShare': None if tie else ['3', '4']}
    result = {'replicateId': rep, 'rootKey': root, 'endpoint': endpoint, 'design': DESIGN}
    if saved:
        result['contributions'] = {'gap': ['1', '2'], 'partial_low': ['0', '1']}
        if tie:
            result['contributions']['gap'] = ['0', '1']
    return result


def population(*, saved=False):
    for rep in range(1, 1841):
        for variant in range(9):
            for main in range(1, 11):
                yield row(rep, variant, main, tie=(rep == 1 and variant == 0 and main == 1), saved=saved)


class C5FinalTests(unittest.TestCase):
    def test_report_binding_uses_receiver_admitted_table_object(self):
        from src.design_validation import c5_public_streams as streams
        raw = b'{"rows":[]}'
        width = (3.0).hex()
        # Synthetic table: patch only the numerical table verifier, while using
        # the actual validated-table constructor and receiver-style identity.
        with mock.patch.object(streams.canonical_report_reference, 'verify_report_table'):
            table = streams.ValidatedReportTable(raw, width)
            same_bytes_other_object = streams.ValidatedReportTable(raw, width)
        spec = SimpleNamespace(table=((width, table),))
        self.assertEqual(P._admitted_report_sha256(table, spec, 3.0), hashlib.sha256(raw).hexdigest())
        for wrong in (raw, same_bytes_other_object):
            with self.subTest(value=type(wrong)), self.assertRaises(ValueError):
                P._admitted_report_sha256(wrong, spec, 3.0)
        with self.assertRaises(ValueError):
            P._admitted_report_sha256(table, spec, 2.0)

    def test_disk_audits_match_and_reject_duplicates(self):
        key = {'streamVersion': 'r10-private-need-report-v1',
               'populationId': 'r10-working-five-v0-rep0-candidate-not-population-estimate',
               'seed': 20261019, 'randomizationGroup': 'r14-c5-paired-projection-private-stream-v1',
               'conditionId': 'C5', 'scheduleVariant': 0, 'replicateId': 2, 'mainIndex': 1,
               'logicalRecipient': 'A', 'streamKind': 'latent', 'reportIndex': 0}
        raw = json.dumps(key, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        digest = hashlib.sha256(raw).hexdigest()
        event = {'key': key, 'keySHA256': digest, 'seedBits': 128,
                 'seedInteger': str(int(digest[:32], 16))}
        with tempfile.TemporaryDirectory() as directory:
            producer = P._DiskEventAudit(os.path.join(directory, 'p.sqlite'))
            receiver = R.DiskEventChecker(os.path.join(directory, 'r.sqlite'))
            try:
                producer.add(event)
                receiver.add(event)
                result = producer.finish()
                self.assertEqual(result, P.audit_supplied_events([event]))
                self.assertTrue(receiver.check(result)['passed'])
                with self.assertRaises(ValueError):
                    producer.add(event)
                with self.assertRaises(ValueError):
                    receiver.add(event)
                bad = dict(result, eventCount=2)
                with self.assertRaises(ValueError):
                    receiver.check(bad)
            finally:
                producer.close()
                receiver.close()

    def test_disk_audits_reject_cross_width_integer_collision(self):
        from types import SimpleNamespace
        private = {'streamVersion': 'r10-private-need-report-v1',
                   'populationId': 'r10-working-five-v0-rep0-candidate-not-population-estimate',
                   'seed': 20261019, 'randomizationGroup': 'r14-c5-paired-projection-private-stream-v1',
                   'conditionId': 'C5', 'scheduleVariant': 0, 'replicateId': 2, 'mainIndex': 1,
                   'logicalRecipient': 'A', 'streamKind': 'latent', 'reportIndex': 0}
        terminal = {'version': 'r10-native-terminal-pcg64-sha256-v1',
                    'randomizationGroup': 'r14-c5-paired-projection-terminal-v1',
                    'identity': {'seed': 20261019, 'modelId': 'm', 'personId': 'p', 'trialId': 't',
                                 'conditionId': 'C5', 'variant': 0, 'replicateId': 2, 'mainIndex': 1},
                    'candidate': {'dict': []}, 'keyEncoding': 'typed-json-floathex-v1'}
        short_hash = '0' * 31 + '1' + 'a' * 32
        full_hash = '0' * 63 + '1'
        events = [{'key': private, 'keySHA256': short_hash, 'seedBits': 128, 'seedInteger': '1'},
                  {'key': terminal, 'keySHA256': full_hash, 'seedBits': 256, 'seedInteger': '1'}]
        def fake_hash(raw):
            return SimpleNamespace(hexdigest=lambda: short_hash if b'streamVersion' in raw else full_hash)
        with tempfile.TemporaryDirectory() as directory:
            producer = P._DiskEventAudit(os.path.join(directory, 'p.sqlite'))
            receiver = R.DiskEventChecker(os.path.join(directory, 'r.sqlite'))
            try:
                with mock.patch.object(hashlib, 'sha256', side_effect=fake_hash):
                    producer.add(events[0]); receiver.add(events[0])
                    with self.assertRaises(ValueError):
                        producer.add(events[1])
                    with self.assertRaises(ValueError):
                        receiver.add(events[1])
            finally:
                producer.close()
                receiver.close()

    def test_replayed_trajectory_row_and_receiver_refusal(self):
        from src.design_validation import c5_public_trajectory_reference as receiver
        root = {'scheduleVariant': 0, 'mainIndex': 1, 'logicalToDisplayed': ['B', 'A'],
                'definition': {'budgetMillihours': 1000}}
        design = {name: {'inverse': [[], copy.deepcopy(values)]} for name, values in DESIGN.items()}
        trajectory = {'schema': 'c5-public-trajectory-v1', 'rootKey': ['C5', 0, 1],
                      'replicateId': 2, 'inputSHA256': 'fixture',
                      'latent': {'A': {'category': 4}, 'B': {'category': 2}},
                      'endpoint': {'latentDisplayed': [2, 4], 'status': 'complete',
                                   'higherNeedShare': ['3', '4']},
                      'finalState': {'budget': 900, 'queryIndex': 1},
                      'contribution': {'gap': ['1', '2'], 'partial_low': ['0', '1']}}
        raw = json.dumps(trajectory, sort_keys=True, separators=(',', ':')).encode()
        receipt = {'status': 'independent_replay_pass', 'rootKey': ['C5', 0, 1],
                   'replicateId': 2, 'inputSHA256': 'fixture',
                   'authority': 'caller-supplied-inputs-not-authenticated',
                   'historicalEquivalence': False}
        with mock.patch.object(P, '_future_frontier', return_value=[]), \
             mock.patch.object(receiver, 'receive_root', return_value=receipt) as replay:
            result = P.admit_future_root(root, {}, {}, None, None, design, 2, raw, spec=None)
            replay.assert_called_once()
            self.assertEqual(result['row']['endpoint']['latentLogical'], {'A': 4, 'B': 2})
            self.assertEqual(result['row']['endpoint']['remainingBudget'], 900)
            self.assertEqual(P.contribution(result['row']['design'], result['row']['endpoint']),
                             trajectory['contribution'])
            self.assertTrue(R.check_trajectory_row(root, design, trajectory, result['row'])['passed'])
            altered = copy.deepcopy(result['row'])
            altered['endpoint']['latentLogical'] = {'A': 2, 'B': 4}
            with self.assertRaises(ValueError):
                R.check_trajectory_row(root, design, trajectory, altered)
            design['gap']['inverse'][1][0][0] = '9'
            self.assertEqual(result['row']['design']['gap'][0], ['1', '1'])
        with mock.patch.object(P, '_future_frontier', return_value=[]), \
             mock.patch.object(receiver, 'receive_root', return_value={'passed': True}):
            with self.assertRaises(ValueError):
                P.admit_future_root(root, {}, {}, None, None, design, 2, raw, spec=None)

    def test_population_bridge_refuses_unverified_or_incomplete_inputs(self):
        from src.design_validation import c5_public_inputs
        prefix = {'payload': {'rows': []}, 'contentSHA256': 'fixture'}
        lookup = {'roots': [{'rootKey': list(key), 'logicalToDisplayed': ['A', 'B']}
                            for key in c5_public_inputs.ROSTER]}
        with mock.patch.object(c5_public_inputs, 'load_historical_prefix', return_value=prefix) as load:
            with self.assertRaises(ValueError):
                P.audit_population('fixture', 'fixture', repository_root='.', lookup=lookup, future=[])
            self.assertEqual(load.call_args.kwargs['lookup'], lookup)
            with self.assertRaises(ValueError):
                P.audit_population('fixture', 'fixture', repository_root='.',
                                   lookup=lookup, future=[{'row': row(rep=2), 'passed': True}])
        with self.assertRaises(TypeError):
            P.audit_population('fixture', 'fixture', repository_root='.', future=[])
        with self.assertRaises(ValueError):
            P.issue_certified_final({'prefixAuthority': True, 'statisticalCertificate': True})

    def test_population_prefix_mapping_and_forged_lookup_refused(self):
        from types import SimpleNamespace
        from src.design_validation import c5_public_inputs as inputs
        lookup = {'roots': [{'rootKey': list(key), 'logicalToDisplayed': ['A', 'B']}
                            for key in inputs.ROSTER]}
        endpoint = row()['endpoint']
        endpoint['latentLogical'] = {'A': 4, 'B': 2}
        data = {'schema': inputs.PREFIX_SCHEMA, 'target': inputs.TARGET, 'replicateId': 1,
                'rows': [{'rootKey': list(key), 'endpoint': dict(endpoint, rootKey=list(key)),
                          'contribution': {'status': 'complete', 'reason': None,
                                           'values': {'gap': ['0', '1'], 'partial_low': ['0', '1']}},
                          'usedSeeds': [{}]} for key in inputs.ROSTER]}
        with mock.patch.object(inputs, '_read', return_value=data):
            with self.assertRaisesRegex(ValueError, 'mapping'):
                P.audit_population('/dev/null', 'fixture', repository_root='.', lookup=lookup, future=[])
        actual = copy.deepcopy(lookup)
        actual['roots'][0]['logicalToDisplayed'] = ['B', 'A']
        spec = SimpleNamespace(rows=tuple(json.dumps(item).encode() for item in actual['roots'][:10]),
                               binding_bytes=json.dumps({'lookupSHA256': 'a' * 64}).encode())
        args = dict(root={}, cfg={}, initial_decision={}, latent_config=None, report_bytes=None,
                    design={}, replicate=2, result_bytes=b'', spec=spec)
        with mock.patch.object(inputs, 'load_historical_prefix',
                               return_value={'payload': {'rows': []}, 'contentSHA256': 'fixture'}), \
             mock.patch.object(P, 'admit_future_root', return_value={'row': row(rep=2)}):
            with self.assertRaisesRegex(ValueError, 'receiver snapshot'):
                P.audit_population('fixture', 'fixture', repository_root='.', lookup=lookup, future=[args])

    def test_future_frontier_refusals(self):
        root = {'scheduleVariant': 0, 'mainIndex': 1, 'logicalToDisplayed': ['B', 'A']}
        def private(logical, kind, index, rep=2):
            key = {'streamVersion': 'r10-private-need-report-v1',
                   'populationId': 'r10-working-five-v0-rep0-candidate-not-population-estimate',
                   'seed': 20261019, 'randomizationGroup': 'r14-c5-paired-projection-private-stream-v1',
                   'conditionId': 'C5', 'scheduleVariant': 0, 'replicateId': rep, 'mainIndex': 1,
                   'logicalRecipient': logical, 'streamKind': kind, 'reportIndex': index}
            digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':'),
                                               ensure_ascii=True).encode()).hexdigest()
            return {'schema': ('r14-c5-fixed-latent-partition-draw-v1' if kind == 'latent' else
                               'r14-c5-prospective-report-partition-draw-v1'),
                    'key': key, 'keySHA256': digest, 'seedHex': digest[:32]}
        decision = {'action': 'ask_A'}
        stage = {'decision': decision, 'report': private('B', 'report', 0),
                 'paid': {}, 'nextDecision': {}}
        result = {'schema': 'c5-public-trajectory-v1', 'rootKey': ['C5', 0, 1],
                  'replicateId': 2, 'ordinal': 90, 'inputSHA256': 'fixture',
                  'latent': {'A': private('A', 'latent', 0), 'B': private('B', 'latent', 0)},
                  'stages': [stage], 'terminalDecision': {'action': 'fixed', 'stop': {'action': 'fixed'}},
                  'terminalStream': None, 'terminalIndex': 0, 'finalState': {'counts': [1, 0], 'queryIndex': 1},
                  'endpoint': {}, 'contribution': {}, 'authority': 'caller-supplied-inputs-not-authenticated',
                  'historicalEquivalence': False, 'scientificSpec': {}}
        self.assertEqual(len(P._future_frontier(root, 2, result)), 3)
        terminal_key = {'version': 'r10-native-terminal-pcg64-sha256-v1',
                        'randomizationGroup': 'r14-c5-paired-projection-terminal-v1',
                        'identity': {'seed': 20261019, 'modelId': 'model', 'personId': 'person',
                                     'trialId': 'trial', 'conditionId': 'C5', 'variant': 0,
                                     'replicateId': 2, 'mainIndex': 1},
                        'candidate': {'dict': []}, 'keyEncoding': 'typed-json-floathex-v1'}
        digest = hashlib.sha256(json.dumps(terminal_key, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        random_result = copy.deepcopy(result)
        random_result['terminalDecision'] = {'action': 'random', 'stop': {'action': 'random'}}
        random_result['terminalStream'] = {'schema': 'r14-c5-grouped-terminal-draw-v1',
                                           'key': terminal_key, 'keySHA256': digest}
        self.assertEqual(len(P._future_frontier(root, 2, random_result)), 4)
        cases = []
        altered = copy.deepcopy(result); del altered['latent']['A']; cases.append(altered)
        altered = copy.deepcopy(result); altered['latent']['A'] = private('A', 'latent', 0, rep=3); cases.append(altered)
        altered = copy.deepcopy(result); altered['stages'][0]['report'] = private('B', 'report', 1); cases.append(altered)
        altered = copy.deepcopy(result); altered['stages'][0]['report'] = private('A', 'report', 0); cases.append(altered)
        altered = copy.deepcopy(result); altered['terminalStream'] = {}; cases.append(altered)
        altered = copy.deepcopy(result); altered['terminalDecision'] = {'action': 'random', 'stop': {'action': 'random'}}; cases.append(altered)
        altered = copy.deepcopy(result); altered['finalState']['counts'] = [0, 1]; cases.append(altered)
        altered = copy.deepcopy(result); altered['stages'][0]['unexpected'] = 1; cases.append(altered)
        altered = copy.deepcopy(random_result); altered['terminalStream']['key']['identity']['replicateId'] = 3; cases.append(altered)
        for altered in cases:
            with self.assertRaises(ValueError):
                P._future_frontier(root, 2, altered)
        with self.assertRaises(ValueError):
            P._future_frontier(root, 1, result)

    def test_supplied_event_inventory_and_authority_refusal(self):
        def event(key, bits):
            payload = json.dumps(key, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=True, allow_nan=False).encode('utf-8')
            digest = hashlib.sha256(payload).hexdigest()
            return {'key': key, 'keySHA256': digest, 'seedBits': bits,
                    'seedInteger': str(int(digest[:bits // 4], 16))}

        private = {'streamVersion': 'r10-private-need-report-v1',
                   'populationId': 'r10-working-five-v0-rep0-candidate-not-population-estimate',
                   'seed': 20261019, 'randomizationGroup': 'r14-c5-paired-projection-private-stream-v1',
                   'conditionId': 'C5', 'scheduleVariant': 0, 'replicateId': 1, 'mainIndex': 1,
                   'logicalRecipient': 'A', 'streamKind': 'latent', 'reportIndex': 0}
        terminal = {'version': 'r10-native-terminal-pcg64-sha256-v1',
                    'randomizationGroup': 'r14-c5-paired-projection-terminal-v1',
                    'identity': {'seed': 20261019, 'modelId': 'model', 'personId': 'person',
                                 'trialId': 'trial', 'conditionId': 'C5', 'variant': 0,
                                 'replicateId': 1, 'mainIndex': 1},
                    'candidate': {'dict': [['value', {'floatHex': '0x1.0000000000000p+0'}]]},
                    'keyEncoding': 'typed-json-floathex-v1'}
        events = [event(private, 128), event(terminal, 256)]
        result = P.audit_supplied_events(events)
        self.assertEqual(result['eventCount'], 2)
        self.assertFalse(result['eventCompletenessEstablished'])
        self.assertFalse(result['historicalPrefixEstablished'])
        self.assertTrue(R.check_supplied_events(reversed(events), result)['passed'])
        report = event(dict(private, streamKind='report', reportIndex=3, logicalRecipient='B'), 128)
        self.assertEqual(P.audit_supplied_events([report])['eventCount'], 1)
        empty = P.audit_supplied_events([])
        self.assertFalse(empty['actualUseEstablished'])
        self.assertFalse(empty['eventCompletenessEstablished'])
        self.assertTrue(R.check_supplied_events([], empty)['passed'])
        for bad in ([events[0], events[0]],
                    [dict(events[0], seedInteger='01')],
                    [dict(events[0], seedBits=True)],
                    [dict(events[0], keySHA256='0' * 64)],
                    [event(private, 256)], [event(terminal, 128)],
                    [event(dict(private, streamKind='latent', reportIndex=1), 128)],
                    [event(dict(private, replicateId=True), 128)],
                    [event(dict(private, randomizationGroup='other'), 128)],
                    [event(dict(terminal, identity=dict(terminal['identity'], variant=True)), 256)],
                    [event(dict(terminal, candidate={'dict': [['x', {'int': '01'}]]}), 256)],
                    [event(dict(terminal, candidate={'dict': [['x', {'str': 'a'}],
                                                              ['x', {'str': 'b'}]]}), 256)],
                    [event({'kind': 'latent'}, 128)]):
            with self.assertRaises(ValueError):
                P.audit_supplied_events(bad)
            with self.assertRaises(ValueError):
                R.check_supplied_events(bad, result)
        with self.assertRaises(ValueError):
            R.check_supplied_events(events, dict(result, actualUseEstablished=True))
        with self.assertRaises(ValueError):
            P.issue_certified_final(result, authority={'passed': True})

    def test_exact_contribution_and_tie(self):
        self.assertEqual(P.contribution(DESIGN, row()['endpoint']),
                         {'gap': ['1', '2'], 'partial_low': ['0', '1']})
        self.assertEqual(P.contribution(DESIGN, row(tie=True)['endpoint']),
                         {'gap': ['0', '1'], 'partial_low': ['0', '1']})
        bad = row()['endpoint']
        bad['remainingBudget'] = 0
        with self.assertRaises(ValueError):
            P.contribution(DESIGN, bad)
        bad = row(tie=True)['endpoint']
        bad['status'] = 'complete'
        with self.assertRaises(ValueError):
            P.contribution(DESIGN, bad)

    def test_complete_population_and_independent_reduction(self):
        result = P.reduce_final(population(), input_kind='fresh')
        self.assertEqual(result['population'], 165600)
        self.assertEqual(result['exactTies'], 1)
        self.assertEqual(result['results']['gap']['total'], ['165599', '2'])
        self.assertEqual(result['results']['gap']['lowPriceEstimate'], ['165599', '331200'])
        self.assertNotIn('highPrice', result['results']['gap'])
        self.assertNotIn('band', result['results']['gap'])
        self.assertFalse(result['endpointParityEstablished'])
        checked = R.check_final(population(), result, input_kind='fresh')
        self.assertTrue(checked['passed'])
        digest = hashlib.sha256()
        digest.update(b'[')
        for index, item in enumerate(population()):
            if index:
                digest.update(b',')
            digest.update(json.dumps(item, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii'))
        digest.update(b']')
        self.assertEqual(checked['orderedRowsSHA256'], digest.hexdigest())
        altered = copy.deepcopy(result)
        altered['results']['gap']['denominator'] -= 1
        with self.assertRaises(ValueError):
            R.check_final(population(), altered, input_kind='fresh')
        altered = copy.deepcopy(result)
        altered['exactTies'] = True
        with self.assertRaises(ValueError):
            R.check_final(population(), altered, input_kind='fresh')
        altered = copy.deepcopy(result)
        altered['results']['gap']['band'] = [['0', '1'], ['1', '1']]
        with self.assertRaises(ValueError):
            R.check_final(population(), altered, input_kind='fresh')

    def test_saved_operands_are_not_historical_authentication(self):
        result = P.reduce_final(population(saved=True), input_kind='saved')
        self.assertFalse(result['historicalRawParityEstablished'])
        self.assertTrue(R.check_final(population(saved=True), result, input_kind='saved')['passed'])
        bad = row(saved=True)
        bad['contributions']['gap'] = ['0', '1']
        with self.assertRaises(ValueError):
            P.reduce_final([bad], input_kind='saved')

    def test_missing_duplicate_and_route_refusal(self):
        with self.assertRaises(ValueError):
            P.reduce_final([row()], input_kind='fresh')
        with self.assertRaises(ValueError):
            P.reduce_final([row(), row()], input_kind='fresh')
        with self.assertRaises(ValueError):
            P.reduce_final([row(saved=True)], input_kind='fresh')
        with self.assertRaises(ValueError):
            P.reduce_final([], input_kind='historical')
        with self.assertRaises(ValueError):
            R.check_final([row(), row()], {}, input_kind='fresh')
        changed = row(rep=2)
        changed['design'] = copy.deepcopy(DESIGN)
        changed['design']['gap'][0] = ['2', '1']
        with self.assertRaises(ValueError):
            P.reduce_final([row(), changed], input_kind='fresh')
        with self.assertRaises(ValueError):
            R.check_final([row(), changed], {}, input_kind='fresh')
        wrong_type = row()
        wrong_type['endpoint']['rootKey'] = ['C5', False, 1]
        with self.assertRaises(ValueError):
            P.reduce_final([wrong_type], input_kind='fresh')
        with self.assertRaises(ValueError):
            R.check_final([wrong_type], {}, input_kind='fresh')
        extra = row()
        extra['endpoint']['unexpected'] = 'unverified'
        with self.assertRaises(ValueError):
            P.reduce_final([extra], input_kind='fresh')
        self.assertFalse(R._identical({'a': [1]}, {'a': [True]}))

    def test_prefix_inequality_and_tampering(self):
        bounds = [['1', '1']] * 90
        result = P.prefix_bound(bounds, ['1', '1'], ['1', '1'], ['5', '1'])
        self.assertEqual(result['delta'], ['1', '1840'])
        self.assertTrue(R.check_prefix_bound(bounds, ['1', '1'], ['1', '1'], ['5', '1'], result)['passed'])
        bad = dict(result, inequalitySatisfied=not result['inequalitySatisfied'])
        with self.assertRaises(ValueError):
            R.check_prefix_bound(bounds, ['1', '1'], ['1', '1'], ['5', '1'], bad)
        bad = dict(result, n=float(result['n']))
        with self.assertRaises(ValueError):
            R.check_prefix_bound(bounds, ['1', '1'], ['1', '1'], ['5', '1'], bad)
        with self.assertRaises(ValueError):
            P.prefix_bound(bounds, ['1', '1'], ['1', '2'], ['5', '1'])

    def test_geometry_family_requires_independent_adjacent_checks(self):
        from fractions import Fraction
        from src.design_validation import c5_precision_planning, c5_precision_reference, c5_prefix_reference
        geometry = {'radii': [Fraction(1)] * 90, 'varianceSumUpper': Fraction(1),
                    'maximumRange': Fraction(1)}
        def certificate(n):
            return {'totalReplicates': n, 'futureCanonicalReplicates': n - 1,
                    'currentPassesUpper': True, 'previousFailsLower': True,
                    'logBounds': [['4', '1'], ['5', '1']]}
        with mock.patch.object(c5_precision_reference, 'geometry', return_value=geometry), \
             mock.patch.object(c5_precision_reference, 'prime_log_interval',
                               return_value=(Fraction(43, 10), Fraction(44, 10))), \
             mock.patch.object(c5_precision_planning, 'bounded_prefix_certificate',
                               side_effect=[certificate(1685), certificate(1840)]), \
             mock.patch.object(c5_prefix_reference, 'verify', return_value={'passed': True}) as verify:
            result = P.certify_supplied_geometry([], {})
            self.assertEqual(result['familyReplicates'], 1840)
            self.assertEqual(verify.call_count, 2)
            altered = copy.deepcopy(result)
            altered['targets']['gap']['previousFailsLower'] = False
            with self.assertRaises(ValueError):
                R.check_supplied_geometry([], {}, altered)
            altered = copy.deepcopy(result)
            altered['familyReplicates'] = 1685
            with self.assertRaises(ValueError):
                R.check_supplied_geometry([], {}, altered)
            altered = copy.deepcopy(result)
            altered['inferenceLooks'] = True
            with self.assertRaises(ValueError):
                R.check_supplied_geometry([], {}, altered)
            with mock.patch.object(c5_prefix_reference, 'verify', side_effect=ValueError('bad boundary')):
                with self.assertRaises(ValueError):
                    R.check_supplied_geometry([], {}, result)


if __name__ == '__main__':
    unittest.main()
