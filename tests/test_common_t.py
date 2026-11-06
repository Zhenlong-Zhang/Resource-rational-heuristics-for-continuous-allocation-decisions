"""Deterministic common-t branch and scalar checks, not new scientific inputs.

No original within estimator, QR, random generator or full-population fixture is
used. Genuine saved-reference calculate parity belongs to the portable-entry
checks on existing inputs, separately from these software tests.
"""
from collections import Counter
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tutoring_statistics import common_t as producer
from tutoring_statistics import independent_common_t as reader


def design_rows():
    pattern = ((1, 1), (1, -1), (-1, 1), (-1, -1)) + ((0, 0),)*6
    return [dict(participant=arm, condition=arm, need_A=a*sa, need_B=b*sb)
            for arm, sa, sb in (('C1', 1, 1), ('C2', 2, 1), ('C3', .5, 2))
            for a, b in pattern]


def independent_certificate(rows):
    predictors = [tuple(row[k] for k in ('participant', 'condition', 'need_A', 'need_B')) for row in rows]
    counts = dict.fromkeys(producer.ARMS, 1)
    return reader.certificate_from_blocks(reader.exact_design(predictors, counts), counts)


def scalar_row(ordinal, state='available', old=False, new=False):
    """Small hand-made scalar disposition, not a generated or fitted dataset."""
    index = 1 if ordinal < 10000 else 2
    flags = {}
    for side, reject in (('old', old), ('new', new)):
        flags[side] = dict.fromkeys(producer.COMPARISONS, None if state == 'technical_unresolved' else False)
        if state == 'available':
            flags[side][producer.COMPARISONS[0]] = reject
    truth = dict.fromkeys(producer.COMPARISONS, index == 1)
    values = dict(status=state, reason=None if state == 'available' else 'fixture_unavailable',
                  truth=truth, actualK=dict.fromkeys(producer.ARMS, 1), signs=dict.fromkeys(producer.COMPARISONS, 1),
                  branches=dict.fromkeys(producer.COMPARISONS,
                      'certified_nonnegative_common_t' if state == 'available' else state))
    for side in ('old', 'new'):
        values[side+'Holm'] = flags[side]
        event = None if state == 'technical_unresolved' else any(flags[side].values())
        values[side+'Events'] = dict(R=event, F=event if index == 1 else None, D=event if index == 2 else None)
        values[side+'P'] = {k: (.01 if flag else 1.) if state == 'available' else None
                           for k, flag in flags[side].items()}
        values[side+'Adjusted'] = {k: (.04 if flag else 1.) if state == 'available' else None
                                  for k, flag in flags[side].items()}
    return dict(ordinal=ordinal, caseIndex=index, repID=ordinal-(0 if index == 1 else 10000)+1,
                caseID='h10wcv1_initial_amount_'+('s1' if index == 1 else 's5_l20')+'_recruit193_mcar08',
                analysis=values, endpointUnknown=dict.fromkeys(producer.REFS, False))


class CommonTTests(unittest.TestCase):
    def test_exact_design_signs_and_independent_certificate(self):
        rows = design_rows()
        cert = producer.design_certificate(rows, dict.fromkeys(producer.ARMS, 1))
        self.assertEqual(list(cert['signs'].values()), [1, 0, -1, 1])
        self.assertTrue(cert['fullRank'])
        cert.pop('formula')
        reader.compare(cert, independent_certificate(rows), 'exact certificate')

    def test_one_ulp_difference_is_not_zero(self):
        rows = design_rows()
        for row in rows:
            if row['condition'] == 'C2':
                row['need_B'] *= math.nextafter(1., 2.)
        cert = producer.design_certificate(rows, dict.fromkeys(producer.ARMS, 1))
        self.assertEqual(cert['signs']['C1-C2:B'], 1)
        self.assertEqual(independent_certificate(rows)['signs'], cert['signs'])

    def test_predictor_only_boundary_and_response_blindness(self):
        rows = design_rows()
        for extra in ({'y': 0.}, {'truth': .25}):
            contaminated = [dict(row, **extra) for row in rows]
            with self.assertRaises(ValueError):
                producer.design_certificate(contaminated, dict.fromkeys(producer.ARMS, 1))
        actual = dict(model='H10', reference='C2', rows=[dict(row, y=0.) for row in rows])
        original = reader.exact_design(reader.predictor_rows(actual), dict.fromkeys(producer.ARMS, 1))
        for i, row in enumerate(actual['rows']):
            row['y'] = float(i)
        self.assertEqual(original, reader.exact_design(reader.predictor_rows(actual), dict.fromkeys(producer.ARMS, 1)))

    def test_invalid_predictors_and_membership(self):
        for value in (True, float('nan'), float('inf')):
            rows = design_rows(); rows[0]['need_A'] = value
            for call in (lambda: producer.design_certificate(rows, dict.fromkeys(producer.ARMS, 1)),
                         lambda: independent_certificate(rows)):
                with self.assertRaises(ValueError):
                    call()
        for rows in (design_rows()[:-1], [dict(row, participant='same') for row in design_rows()]):
            with self.assertRaises(ValueError):
                producer.design_certificate(rows, dict.fromkeys(producer.ARMS, 1))
            with self.assertRaises(ValueError):
                independent_certificate(rows)

    def test_singular_certificate_is_not_available_proof(self):
        rows = [dict(row, need_A=0., need_B=0.) for row in design_rows()]
        cert = producer.design_certificate(rows, dict.fromkeys(producer.ARMS, 1))
        self.assertFalse(cert['fullRank'])
        self.assertEqual(cert['signs'], dict.fromkeys(producer.COMPARISONS))
        cert.pop('formula')
        reader.compare(cert, independent_certificate(rows), 'singular certificate')

    def test_candidate_branches(self):
        for function in (producer.candidate_probability, reader.magnitude_from_t):
            for sign in (-1, None):
                self.assertEqual(function(-2., 2., 21, sign, .4), ('original_iut2_fallback', .4))
            for d, s in ((2., 2.), (-2., -2.), (0., 2.), (-2., 0.), (0., 0.)):
                self.assertEqual(function(d, s, 21, -1, .4), ('same_or_zero_sign', 1.))
            for sign in (0, 1):
                branch, probability = function(-2., 2., 21, sign, .4)
                self.assertEqual(branch, 'certified_nonnegative_common_t')
                self.assertGreater(probability, 0.)
                self.assertLess(probability, .5)
        for d, s in ((-2., 2.), (2., -3.), (-20., 30.)):
            reader.close_number(producer.candidate_probability(d, s, 21, 1, .4)[1],
                                reader.magnitude_from_t(d, s, 21, 1, .4)[1], 'independent tail')
        self.assertAlmostEqual(producer.candidate_probability(-1., 1., 1, 0, .5)[1], .25)

    def test_invalid_df_and_candidate_inputs(self):
        for function in (producer.candidate_probability, reader.magnitude_from_t):
            for df in (None, True, 0, -1, 1.5):
                for sign in (-1, 1):
                    with self.assertRaises(ValueError):
                        function(-2., 2., df, sign, .4)
            for args in ((True, 2., 21, 1, .4), (-2., 2., 21, True, .4),
                         (-2., 2., 21, 1, None), (-2., 2., 21, 1, 1.1),
                         (-2., float('nan'), 21, 1, .4)):
                with self.assertRaises(ValueError):
                    function(*args)

    def test_strict_four_family_holm(self):
        for function in (producer.holm, reader.holm_flags):
            states = dict.fromkeys(producer.COMPARISONS, 'available')
            ps = dict.fromkeys(producer.COMPARISONS, 1.)
            ps[producer.COMPARISONS[0]] = .0125
            self.assertFalse(any(function(ps, states)[1].values()))
            ps[producer.COMPARISONS[0]] = math.nextafter(.0125, 0.)
            self.assertTrue(function(ps, states)[1][producer.COMPARISONS[0]])
            ps = dict.fromkeys(producer.COMPARISONS, .01)
            self.assertTrue(all(function(ps, states)[1].values()))
            states[producer.COMPARISONS[3]] = 'statistical_unavailable'
            ps[producer.COMPARISONS[3]] = None
            ps[producer.COMPARISONS[0]] = .0125
            ps[producer.COMPARISONS[1]] = ps[producer.COMPARISONS[2]] = 1.
            adjusted, flags = function(ps, states)
            self.assertFalse(any(flags.values()))
            self.assertIsNone(adjusted[producer.COMPARISONS[3]])

    def test_unknown_and_inapplicable(self):
        for holm, events in ((producer.holm, producer.events), (reader.holm_flags, reader.family_events)):
            states = dict.fromkeys(producer.COMPARISONS, 'statistical_unavailable')
            adjusted, flags = holm(dict.fromkeys(producer.COMPARISONS), states)
            self.assertTrue(all(v is None for v in adjusted.values()))
            self.assertTrue(all(v is False for v in flags.values()))
            self.assertEqual(events(flags, dict.fromkeys(producer.COMPARISONS, True)), dict(R=False, F=False, D=None))
            self.assertEqual(events(flags, dict.fromkeys(producer.COMPARISONS, False)), dict(R=False, F=None, D=False))
            states[producer.COMPARISONS[0]] = 'technical_unresolved'
            _, flags = holm(dict.fromkeys(producer.COMPARISONS), states)
            self.assertEqual(events(flags, dict.fromkeys(producer.COMPARISONS, False)), dict(R=None, F=None, D=None))

    def test_paired_counts_q_zero_one_and_losses(self):
        for pairs in ([], [(False, True)], [(False, True), (False, True)],
                      [(False, False), (False, True), (True, False), (True, True)]):
            cells = Counter(f'{int(a)}{int(b)}' for a, b in pairs)
            reader.compare(producer.paired_table(pairs), reader.paired(cells), 'paired counts')
        self.assertIsNone(producer.paired_table([])['delta'])
        self.assertIsNone(producer.paired_table([(False, True)])['pairedMCSE'])
        self.assertEqual(producer.paired_table([(False, True), (False, True)])['pairedMCSE'], 0.)
        self.assertEqual(reader.pair_cell(None, True), 'unknown')
        with self.assertRaises(ValueError):
            producer.paired_table([(None, False)])

    def test_cp_boundaries(self):
        for count, size in ((0, 0), (0, 1), (1, 1), (0, 10), (10, 10), (3, 10)):
            reader.compare(producer.proportion(count, size), reader.proportion(count, size), 'CP boundaries')
        self.assertIsNone(producer.proportion(0, 0)['CP95'])
        self.assertEqual(producer.proportion(0, 10)['CP95'][0], 0.)
        self.assertEqual(producer.proportion(10, 10)['CP95'][1], 1.)
        for function in (producer.proportion, reader.proportion):
            for pair in ((1, 0), (-1, 2), (True, 2), (0, -1)):
                with self.assertRaises(ValueError):
                    function(*pair)

    def test_partial_scalar_report_unknowns_and_endpoint_separation(self):
        rows = [scalar_row(0), scalar_row(1, 'technical_unresolved'),
                scalar_row(2, 'statistical_unavailable'), scalar_row(10000, new=True)]
        rows[-1]['endpointUnknown'][producer.REFS[0]] = True
        original = copy.deepcopy(rows)
        summary = producer.report(rows)
        reader.compare(summary, reader.report(iter(rows)), 'partial scalar report')
        self.assertEqual(rows, original)
        self.assertFalse(summary['fullMeasurement'])
        self.assertEqual(summary['cases']['1']['events']['F']['Q'], 2)
        self.assertEqual(summary['cases']['1']['events']['F']['unknown'], 1)
        self.assertEqual(summary['cases']['1']['missing'], 9997)
        self.assertIsNone(summary['cases']['1']['events']['D'])
        self.assertIsNone(summary['cases']['2']['events']['F'])
        self.assertEqual(summary['cases']['2']['events']['D']['table']['01'], 1)
        self.assertIsNone(summary['cases']['2']['events']['D']['fullNew'])
        reader.check_no_original_losses(summary)
        reader.compare(producer.report([]), reader.report([]), 'empty scalar report')

    def test_scalar_identity_and_disposition_refusals(self):
        cases = [[scalar_row(0), scalar_row(0)]]
        for field, value in (('ordinal', -1), ('caseIndex', 2), ('repID', 2), ('caseID', 'other')):
            row = scalar_row(0); row[field] = value; cases.append([row])
        row = scalar_row(0, 'technical_unresolved'); row['analysis']['newHolm'][producer.COMPARISONS[0]] = False
        cases.append([row])
        row = scalar_row(10000); row['analysis']['newEvents']['F'] = False; cases.append([row])
        for rows in cases:
            for function in (producer.report, reader.report):
                with self.assertRaises(ValueError):
                    function(rows)
        with self.assertRaises(ValueError):
            reader.check_no_original_losses(reader.report([scalar_row(10000, old=True)]))

    def test_reader_does_not_delegate_mathematics(self):
        rows = design_rows()
        with patch.object(producer, 'design_certificate', side_effect=AssertionError('producer used')), \
             patch.object(producer, 'candidate_probability', side_effect=AssertionError('producer used')), \
             patch.object(producer, 'holm', side_effect=AssertionError('producer used')), \
             patch.object(producer, 'report', side_effect=AssertionError('producer used')):
            self.assertTrue(independent_certificate(rows)['fullRank'])
            self.assertEqual(reader.magnitude_from_t(-1., 1., 1, 0, .5)[0], 'certified_nonnegative_common_t')
            reader.holm_flags(dict.fromkeys(producer.COMPARISONS, .5), dict.fromkeys(producer.COMPARISONS, 'available'))
            self.assertFalse(reader.report([scalar_row(0)])['fullMeasurement'])

    def test_typed_comparison(self):
        for actual, expected in ((False, None), (0, False), (True, 1), ({'x': 0}, {'x': None})):
            with self.assertRaises(ValueError):
                reader.compare(actual, expected, 'typed mismatch')


if __name__ == '__main__':
    unittest.main()
