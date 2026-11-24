"""Purpose: actual saved action/count extraction, admission and sealed-table gates.

All payloads and full bundles below are explicitly synthetic software fixtures.
They do not reproduce or authenticate historical endpoints, policies or results.
No policy evaluation, numerical integration, fitting or native generator is used.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import paper_action_tables as A

I, S = A.I, A.S
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/personal_time_sensitivity.json'
CFG = {'candidateId': 'synthetic_action_fixture', 'task': {}}


def endpoint_fixture(c='C3', v=0, m=1, *, order=('B', 'A'), slots=(1, 0, 1), action='EO_raw_surplus'):
    """Construct explicit saved events without invoking a model or a policy."""
    root = dict(conditionId=c, scheduleVariant=v, mainIndex=m,
                logicalToDisplayed=list(order), definition={'budgetMillihours': 120000})
    context = dict(rootSHA256=I.digest(root), runSHA256=I.digest({}),
                   identity=S.P.stable_identity(CFG, S.RECIPE['seed'], S.RECIPE['populationId'], c, v, 0, m),
                   bindings={k: S.RECIPE[k] for k in ('populationId', 'randomizationGroup')})
    state = dict(budget=120000, counts=[0, 0], queryIndex=0, logHex=[['0x0.0p+0']]*2)
    events = []

    def save(name, value):
        events.append(dict(name=name, value=deepcopy(value)))

    def decision(k):
        stop = dict(action=action, index=500,
                    legacySelection=dict(selected_macro=action, selected_index=500))
        return dict(action=('ask_A', 'ask_B')[slots[k]] if k < len(slots) else action, stop=stop)

    initial = decision(0)
    save('decision_000000', dict(before=state, value=initial))
    if slots:
        save('latent_A', {'category': 42})
        save('latent_B', {'category': 54})
    for k, slot in enumerate(slots):
        key = S.P.make_key(context, order[slot], 'report', state['counts'][slot])
        save('report_%06d' % k, dict(key=key, category=50))
        before = deepcopy(state)
        state['counts'][slot] += 1
        state['queryIndex'] += 1
        state['budget'] -= 1000
        save('paid_%06d' % k, dict(before=before, slot=slot, report=50, after=state))
        save('decision_%06d' % (k+1), dict(before=state, value=decision(k+1)))
    terminal = dict(action=action, intendedIndex=500, budget=state['budget'],
                    realizedCombinedGPA=7. if slots else None, newScientificDraws=0)
    metrics = dict(counts=list(state['counts']), paidCount=len(slots), Q=int(bool(slots)),
                   initialIntegerBudget=120000, remainingIntegerBudget=state['budget'],
                   shareIndex=500, action=action)
    save('terminal' if slots else 'expected_terminal', terminal)
    save('metrics', metrics)
    return root, dict(context=context, events=events, initialDecision=initial, finalState=state,
                      terminal=terminal, metrics=metrics,
                      endpointKind='realized_trajectory' if slots else 'expected_only')


def synthetic_policy(selected=None, *, full=False, realized=5):
    selected = I.select_roots(full=True) if full else I.selection(selected or [
        dict(conditionId='C3', scheduleVariant=0, mainIndex=1)])
    n = 6*len(selected)
    return dict(schema='personal-time-sensitivity-independent-check-v1',
                binding=dict(selection=selected, sources=S.implementation_identity()),
                producerInputs={str(S.record_path(Path('.'), label, root)): 'a'*64
                                for root in selected for label, _ in S.RATES},
                verifiedTrials=n, fullFiniteTarget=full, expectedOnlyEndpoints=n-realized,
                genuineRealizedEndpoints=realized, absoluteTolerance=1e-10,
                completeIndependentReconstruction=True, historicalParityEstablished=False,
                monteCarloPrecisionEstablished=False)


def full_action_fixture(output, policy, config):
    """Write a full SYNTHETIC action bundle for orchestration tests, not history.

    config is personal_time_sensitivity.json. Assign one question to the first
    840 non-highest-price rows, giving the requested 1860/840 type counts only.
    This is deliberately not a claim of historical per-cell behavior.
    """
    output = Path(output)
    rows, root_inputs = [], {}
    remaining = 840
    for c, v, m in sorted(I.ORIGINAL):
        for label, _ in S.RATES:
            ask = label != 'rho_0.01' and remaining > 0
            remaining -= int(ask)
            root, payload = endpoint_fixture(c, v, m, order=('A', 'B') if m % 2 else ('B', 'A'),
                                             slots=(0,) if ask else (), action='ES')
            rows.append(A.extract(root, label, payload))
            root_inputs[A._root_key(root)] = I.digest(root)
    result = A.summarize(rows, full=True)
    result['rowSourceAuthenticated'] = True
    A._write_tables(output, rows, result)
    pins = {**policy['producerInputs'], **{
        p.replace('producer/', 'independent/', 1): 'b'*64 for p in policy['producerInputs']}}
    I.write(output / 'source_check.json', dict(
        schema=A.SCHEMA, sources=A._source_identity(config),
        policyRunSHA256=hashlib.sha256(I.encoded(policy['binding']) + b'\n').hexdigest(),
        policyCheckSHA256=hashlib.sha256(I.encoded(policy) + b'\n').hexdigest(),
        policyContentSHA256=I.digest(policy), policyBindingSHA256=I.digest(policy['binding']),
        policyInputs=pins, rootInputs=root_inputs,
        outputFiles={name: S.file_sha(output / name) for name in A.TABLE_FILES},
        verifiedTrials=2700, fullFiniteTarget=True, newPolicySimulations=0,
        historicalProvenanceReconstructed=False, payloadAbsoluteTolerance=1e-10,
        extractedFieldsComparison='exact_typed_json'))
    return dict(output=str(output), verifiedTrials=2700, fullFiniteTarget=True, newPolicySimulations=0,
                tables=[str(output / name) for name in (*A.TABLE_FILES, 'source_check.json')])


def checked_fixture(source):
    """Write six synthetic producer/independent pairs, with real envelope hashes."""
    source = Path(source)
    policy = synthetic_policy()
    run = policy['binding']
    root = None
    for label, _ in S.RATES:
        root, payload = endpoint_fixture(slots=() if label == 'rho_0.01' else (1, 0, 1))
        payload['context']['runSHA256'] = I.digest(run)
        binding = S.record_binding(root, CFG, run, label)
        path = S.record_path(source, label, root)
        I.write(path, S.envelope(binding, payload))
        sha = S.file_sha(path)
        policy['producerInputs'][str(path.relative_to(source))] = sha
        I.write(S.record_path(source, label, root, 'independent'), dict(
            schema='personal-time-sensitivity-independent-trial-v1', binding=binding,
            producerSHA256=sha, payload=payload, absoluteTolerance=1e-10,
            sharedKernels=['bounded_task', 'budget_horizon', 'terminal_time',
                           'authenticated_input_adapter_and_event_traversal']))
    I.write(source / 'run.json', run)
    I.write(source / 'independent_check.json', policy)
    return root, run, policy


class ActionTableTests(unittest.TestCase):
    def test_saved_counts_use_display_orientation_not_action_letters_or_total_split(self):
        root, payload = endpoint_fixture()
        original = deepcopy((root, payload))
        row = A.extract(root, 'rho_0.001', payload)
        self.assertEqual([row[k] for k in A.COUNTS], [3, 1, 2, 2, 1])
        self.assertEqual(row['logical_order'], ['B', 'A'])
        self.assertEqual((row['terminal_action'], row['selected_rule'], row['index']),
                         ('EO_raw_surplus', 'EO_raw_surplus', 500))
        self.assertEqual(original, (root, payload))
        # Same allocation index is not an ES action, and no benchmark is consulted.
        self.assertNotEqual(row['terminal_action'], 'ES')

    def test_expected_only_zero_counts_are_observed_zeros_not_missing(self):
        root, payload = endpoint_fixture(slots=(), action='MFN_A')
        row = A.extract(root, 'rho_0.01', payload)
        self.assertEqual([row[k] for k in A.COUNTS], [0]*5)
        summary = A.summarize([row])
        observed = next(c for c in summary['conditions'] if c['condition'] == 'C3' and c['price'] == 'rho_0.01')
        self.assertEqual(observed['status'], 'PARTIAL')
        self.assertEqual(observed['questionMeans']['total_questions'], 0)
        missing = summary['conditions'][0]
        self.assertEqual((missing['status'], missing['meanDenominator']), ('UNKNOWN', 0))
        self.assertIsNone(missing['questionMeans']['total_questions'])
        self.assertEqual(len(summary['conditions']), 30)
        self.assertFalse(summary['rowSourceAuthenticated'])

    def test_extract_refuses_missing_counts_wrong_types_and_metric_disagreement(self):
        root, payload = endpoint_fixture()
        for section, field, value in (
            ('finalState', 'counts', [True, 2]), ('finalState', 'counts', [-1, 4]),
            ('finalState', 'queryIndex', 3.), ('metrics', 'counts', [2, 1]),
            ('metrics', 'paidCount', 2), ('metrics', 'Q', True),
            ('metrics', 'action', 'ES'), ('terminal', 'intendedIndex', 501),
        ):
            bad = deepcopy(payload)
            bad[section][field] = value
            if section in ('metrics', 'terminal'):
                bad['events'][-1 if section == 'metrics' else -2]['value'] = deepcopy(bad[section])
            with self.subTest(section=section, field=field, value=value), self.assertRaises(ValueError):
                A.extract(root, 'rho_0.001', bad)
        del payload['metrics']['counts']
        with self.assertRaises(ValueError):
            A.extract(root, 'rho_0.001', payload)

    def test_extract_refuses_paid_event_slot_chain_report_and_final_selection_changes(self):
        root, payload = endpoint_fixture()
        for case in ('missing', 'duplicate', 'slot', 'before', 'after', 'logical', 'report_index', 'selection'):
            bad = deepcopy(payload)
            by = {e['name']: e['value'] for e in bad['events']}
            if case == 'missing':
                bad['events'].pop(4)
            elif case == 'duplicate':
                bad['events'].insert(4, deepcopy(bad['events'][4]))
            elif case == 'slot':
                by['paid_000000']['slot'] = 0
            elif case == 'before':
                by['paid_000001']['before']['counts'] = [1, 0]
            elif case == 'after':
                by['paid_000000']['after']['counts'] = [1, 0]
            elif case == 'logical':
                by['report_000000']['key']['logicalRecipient'] = 'B'
            elif case == 'report_index':
                by['report_000002']['key']['reportIndex'] = 0
            else:
                by['decision_000003']['value']['stop']['legacySelection']['selected_macro'] = 'ES'
            with self.subTest(case=case), self.assertRaises(ValueError):
                A.extract(root, 'rho_0.001', bad)

    def test_extract_refuses_wrong_root_orientation_seed_and_frozen_high_price_scope(self):
        root, payload = endpoint_fixture()
        for order in (None, ['A', 'A'], ['A', 1], ['A', 'B']):
            bad = deepcopy(root)
            bad['logicalToDisplayed'] = order
            with self.subTest(order=order), self.assertRaises(ValueError):
                A.extract(bad, 'rho_0.001', payload)
        for key, value in (('seed', 0), ('replicateId', 1), ('mainIndex', True)):
            bad = deepcopy(payload)
            bad['context']['identity'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                A.extract(root, 'rho_0.001', bad)
        with self.assertRaises(ValueError):
            A.extract(root, 'rho_0.01', payload)
        with self.assertRaises(ValueError):
            A.extract(root, 'unknown', payload)

    def test_summarize_complete_cell_partial_roster_and_exact_counts(self):
        rows = []
        for v in range(9):
            for m in range(1, 11):
                root, payload = endpoint_fixture(v=v, m=m)
                rows.append(A.extract(root, 'rho_0.001', payload))
        result = A.summarize(rows)
        cell = next(c for c in result['conditions'] if c['condition'] == 'C3' and c['price'] == 'rho_0.001')
        self.assertEqual(cell['status'], 'COMPLETE')
        self.assertEqual(cell['meanDenominator'], 90)
        self.assertEqual(cell['questionTotals']['total_questions'], 270)
        self.assertEqual(cell['questionMeans']['logical_A_questions'], 2)
        self.assertEqual(cell['actionCounts']['EO_raw_surplus'], 90)
        self.assertEqual(sum(cell['actionCounts'].values()), 90)
        self.assertFalse(result['fullFiniteTarget'])
        with self.assertRaisesRegex(ValueError, '2700-endpoint'):
            A.summarize(rows, full=True)
        self.assertEqual(A.summarize(list(reversed(rows))), result)

    def test_summary_refuses_duplicate_foreign_or_inconsistent_extracted_rows(self):
        root, payload = endpoint_fixture()
        row = A.extract(root, 'rho_0.001', payload)
        with self.assertRaises(ValueError):
            A.summarize([row, row])
        for field, value in (('variant', True), ('main', 11), ('total_questions', 2),
                             ('logical_A_questions', 1), ('terminal_action', 'nearest_MU'),
                             ('endpoint_kind', 'expected_only'), ('price', 'other')):
            bad = deepcopy(row)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                A.summarize([bad])
        other = {**row, 'price': 'rho_0.0001', 'rootSHA256': 'f'*64}
        with self.assertRaisesRegex(ValueError, 'changed across prices'):
            A.summarize([row, other])

    def test_full_fixture_saved_verifier_and_no_source_tree_or_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = synthetic_policy(full=True, realized=840)
            output = Path(temp) / 'actions'
            full_action_fixture(output, policy, CONFIG)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(S, 'trial', side_effect=AssertionError('No policy replay')), \
                 patch.object(S, 'load_roots', side_effect=AssertionError('No original source tree')):
                check = A.verify_saved(output, CONFIG, policy=policy)
            self.assertEqual(check['verifiedTrials'], 2700)
            self.assertEqual(check['newPolicySimulations'], 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            summary = I.read(output / 'summary.json')
            self.assertEqual(sum(c['realizedEndpoints'] for c in summary['conditions']), 840)
            self.assertTrue(all(c['completeOriginalCell'] for c in summary['conditions']))

    def test_saved_verifier_refuses_tampered_pins_summary_csv_roster_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = synthetic_policy(full=True, realized=840)
            output = Path(temp) / 'actions'
            full_action_fixture(output, policy, CONFIG)
            originals = {p.name: p.read_bytes() for p in output.iterdir()}
            cases = ('sources', 'policyInputs', 'rootInputs', 'policyBindingSHA256',
                     'policyContentSHA256', 'newPolicySimulations', 'summary', 'csv', 'rows')
            for case in cases:
                for name, blob in originals.items():
                    (output / name).write_bytes(blob)
                check = I.read(output / 'source_check.json')
                if case in ('sources', 'policyInputs', 'rootInputs'):
                    check[case].pop(next(iter(check[case])))
                elif case in ('policyBindingSHA256', 'policyContentSHA256'):
                    check[case] = 'f'*64
                elif case == 'newPolicySimulations':
                    check[case] = 1
                else:
                    name = {'summary': 'summary.json', 'csv': 'action_conditions.csv', 'rows': 'trial_actions.json'}[case]
                    path = output / name
                    if case == 'csv':
                        path.write_bytes(path.read_bytes() + b'changed\n')
                    else:
                        data = I.read(path)
                        if case == 'summary':
                            data['conditions'][0]['actionCounts']['ES'] -= 1
                        else:
                            data.pop()
                        path.write_bytes(I.encoded(data) + b'\n')
                    # Rehashing the local table still cannot evade re-reduction.
                    check['outputFiles'][name] = S.file_sha(path)
                (output / 'source_check.json').write_bytes(I.encoded(check) + b'\n')
                with self.subTest(case=case), self.assertRaises(ValueError):
                    A.verify_saved(output, CONFIG, policy=policy)

    def test_analyze_checked_synthetic_pairs_is_read_only_resumable_and_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'policy', Path(temp) / 'actions'
            root, run, policy = checked_fixture(source)
            before = {str(p): p.read_bytes() for p in source.rglob('*.json')}
            with self.admission_mocks(source, root, run):
                made = A.analyze(source, output, CONFIG)
                self.assertEqual(made['verifiedTrials'], 6)
                self.assertFalse(made['fullFiniteTarget'])
                A.analyze(source, output, CONFIG)
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob('*.json')})
            A.verify_saved(output, CONFIG, policy=policy, full=False,
                           policy_sha=S.file_sha(source / 'independent_check.json'))
            with self.assertRaisesRegex(ValueError, 'raw policy-check'):
                A.verify_saved(output, CONFIG, policy=policy, full=False, policy_sha='f'*64)

    def admission_mocks(self, source, root, run):
        stack = ExitStack()
        stack.enter_context(patch.object(S, 'load_context', return_value={}))
        stack.enter_context(patch.object(S, 'load_roots', return_value=([root], run, source)))
        stack.enter_context(patch.object(S, 'rate_config', return_value=CFG))
        stack.enter_context(patch.object(S, 'trial', side_effect=AssertionError('Never reconstruct')))
        return stack

    def test_analyze_refuses_missing_reader_tampered_checkpoint_and_foreign_inventory(self):
        for case in ('missing_reader', 'reader_pin', 'reader_schema', 'reader_null',
                     'reader_float_count', 'producer', 'extra'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                source, output = Path(temp) / 'policy', Path(temp) / 'actions'
                root, run, _ = checked_fixture(source)
                reader = S.record_path(source, 'rho_0.001', root, 'independent')
                if case == 'missing_reader':
                    reader.unlink()
                elif case in ('reader_pin', 'reader_schema'):
                    other = I.read(reader)
                    other['producerSHA256' if case == 'reader_pin' else 'schema'] = 'changed'
                    reader.write_bytes(I.encoded(other))
                elif case == 'reader_null':
                    reader.write_bytes(b'null\n')
                elif case == 'reader_float_count':
                    other = I.read(reader)
                    other['payload']['metrics']['counts'][0] = 1.00000000001
                    other['payload']['events'][-1]['value'] = deepcopy(other['payload']['metrics'])
                    reader.write_bytes(I.encoded(other))
                elif case == 'producer':
                    path = S.record_path(source, 'rho_0.001', root)
                    path.write_bytes(path.read_bytes() + b' ')
                else:
                    I.write(source / 'producer/extra.json', {})
                with self.admission_mocks(source, root, run), self.assertRaises(ValueError):
                    A.analyze(source, output, CONFIG)
                self.assertFalse(output.exists())

    def test_analyze_refuses_overlapping_trees_before_context_and_full_on_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'policy'
            root, run, _ = checked_fixture(source)
            with patch.object(S, 'load_context', side_effect=AssertionError('No read before path gate')):
                with self.assertRaises(ValueError):
                    A.analyze(source, source / 'actions', CONFIG)
            with self.admission_mocks(source, root, run), self.assertRaises(ValueError):
                A.analyze(source, Path(temp) / 'actions', CONFIG, full=True)


if __name__ == '__main__':
    unittest.main()
