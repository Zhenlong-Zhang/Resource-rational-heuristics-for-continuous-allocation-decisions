"""Frozen-law, independent numerical, scalar-denominator and portable-interface tests.

Deterministic fixtures are software checks, not additional scientific inputs.
"""
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.tutoring_statistics import within_design as d
from src.tutoring_statistics import within_projection as core
from src.tutoring_statistics import independent_within_projection as qr
from src.tutoring_statistics import independent_within_check as check
from src.tutoring_statistics import magnitude_composition as composition
from src.tutoring_statistics import within_reproduction as runner
from src.tutoring_statistics import within_reduction as coverage
from src.tutoring_statistics import independent_within_reduction as other_coverage
from src.tutoring_statistics import midpoint_generation as midpoint
from src.tutoring_statistics import midpoint_inputs as native_midpoint
from src.tutoring_statistics import midpoint_reduction as power
from src.tutoring_statistics import independent_midpoint_reduction as other_power
from src.tutoring_statistics import proxy_h10_mapping as mapping


def fixture():
    a = [-2., -1., 0., 1., 2.] * 2
    b = [-1.] * 5 + [1.] * 5
    e = [1., -2., 2., -2., 1.] * 2
    slopes = [.25, -.25, .55, -.55, .75, -.75]
    rows = []
    for ai, arm in enumerate(core.ARMS):
        for person in range(2):
            for x, z, error in zip(a, b, e):
                rows.append(dict(participant=arm+str(person), condition=arm,
                    need_A=50+x, need_B=60+z,
                    y=20+100*person+slopes[2*ai]*(50+x)+slopes[2*ai+1]*(60+z)+error))
    return dict(model='H10', reference='C2', rows=rows), slopes


def empty_event(technical=False):
    actual = dict(model='H10', reference='C2', rows=[])
    counts = dict.fromkeys(d.ARMS, 0)
    value = core.estimate_within(actual, composition.compose_family)
    if technical:
        value = core.unavailable('technical_unresolved', 'fixture', value['design'], composition.compose_family)
    return midpoint.compact_event(0, value, midpoint.evaluation(value), counts)


class FrozenDesign(unittest.TestCase):
    def test_notebook_settings_match_frozen_population(self):
        settings = json.loads((Path(__file__).resolve().parents[1]/'configs/within_statistics.json').read_text())
        for study in ('coverage', 'midpoint'):
            self.assertEqual(settings[study]['full_count'], d.plan(study)['expectedTotal'])
            self.assertEqual(runner.selection(study, settings[study]['selected_original_ordinals']),
                             [0, 10000] if study == 'coverage' else [0, 1])
        self.assertEqual(settings['reference_python'], '3.10.9')
        self.assertFalse(settings['large_execution_default'])

    def test_case_major_boundaries_and_all_seeds(self):
        from src.tutoring_statistics.magnitude_design import manifest, case_for
        plan = manifest('fresh_within_validation')
        for ordinal, case, rep in ((0, 1, 1), (9999, 1, 10000), (10000, 2, 1), (12128, 2, 2129)):
            value = case_for(plan, ordinal)
            self.assertEqual((value['caseIndex'], value['repID']), (case, rep))
            self.assertEqual(value['responseSeed'], 2022610280+100000*case+rep)
        seeds = [{d.coverage_case(i)[role+'Seed'] for i in range(12129)} for role in d.COVERAGE_BASES]
        self.assertTrue(all(len(s) == 12129 for s in seeds))
        self.assertEqual(len(set.union(*seeds)), 3*12129)
        self.assertEqual(d.midpoint_case(4611)['seeds']['response'], 2052611040+4612)
        self.assertEqual(d.MIDPOINT_SLOPES['C3']['B'], .586772857425495)
        self.assertEqual(check.precision_readback(4612)['B'], 4612)
        with self.assertRaises(ValueError):
            check.precision_readback(4611)

    def test_selectors_reject_changed_population(self):
        self.assertEqual(runner.selection('coverage', [0, 10000]), [0, 10000])
        for ids in ([], [0, 0], [10000, 0], [-1], [12129], [True]):
            with self.assertRaises(ValueError):
                runner.selection('coverage', ids)
        with self.assertRaises(ValueError):
            runner.selection('midpoint', [4612])
        with self.assertRaises(ValueError):
            runner.selection('midpoint', [0], True)


class NumericalKernel(unittest.TestCase):
    def test_accepted_deterministic_projection_and_independent_qr(self):
        actual, slopes = fixture()
        produced = core.estimate_within(actual, composition.compose_family)
        independent = qr.calculate(actual['rows'])
        check.near(produced['beta'], slopes, 'fixture slopes')
        check.near(produced['beta'], independent['beta'], 'QR slopes')
        check.near(produced['covariance'], independent['covariance'], 'QR covariance')
        self.assertAlmostEqual(produced['RSS'], 168.)
        self.assertEqual(produced['df'], 48)
        for key, ref in produced['references'].items():
            for field in ('estimate', 'SE', 'df', 'statistic', 'lessP', 'greaterP', 'CI'):
                check.near(ref[field], independent['references'][key][field], key+field)

    def test_truth_and_intercept_do_not_change_estimation(self):
        actual, _ = fixture()
        value = core.estimate_within(actual, composition.compose_family)
        before = copy.deepcopy(value)
        core.evaluate_truth(value, d.MIDPOINT_SLOPES)
        self.assertEqual(value, before)
        for row in actual['rows']:
            row['y'] += 1700. if row['participant'].endswith('1') else -311.
        actual['rows'].reverse()
        after = core.estimate_within(actual, composition.compose_family)
        check.near(value['beta'], after['beta'], 'subject offset')
        check.near(value['covariance'], after['covariance'], 'subject offset covariance')

    def test_typed_unavailability_and_empty_qr_adapter(self):
        counts = dict.fromkeys(d.ARMS, 0)
        self.assertEqual(check.fitted_reference(qr, [], counts)['status'], 'statistical_unavailable')
        for technical in (False, True):
            event = empty_event(technical)
            result = runner.reduce_events('midpoint', [event])
            self.assertEqual(result['report']['counts']['M'], 4612)
            self.assertEqual(result['report']['counts']['missing'], 4611)
            self.assertEqual(result['report']['events']['D']['unknown'], 4612 if technical else 4611)
            self.assertIsNone(result['report']['primary']['pHat'])
            self.assertEqual(result['report']['primary']['decision'], 'unresolved')
        actual, _ = fixture()
        for row in actual['rows']:
            row['y'] = 0.
        self.assertEqual(core.estimate_within(actual, composition.compose_family)['status'], 'technical_unresolved')


class ScalarReduction(unittest.TestCase):
    def test_original_primary_paired_table_and_incomplete_denominator(self):
        key = 'C1-C2:B:D'
        a, b = coverage.Accumulator(), other_coverage.Paired()
        table = {'00': 449, '01': 52, '10': 51, '11': 9448}
        a.cases[1]['present'] = 10000
        a.cases[1]['pairs'][key]['table'].update(table)
        b.table = table
        value = a.report()['cases']['1']['pairedWithinMinusCR2'][key]
        other_coverage.compare(value, b.report(10000, 10000, True))
        self.assertEqual(value['exactMcNemarP'], 1.)
        self.assertEqual(value['primaryDecision'], 'unresolved')
        b.table['11'] -= 1
        self.assertIsNone(b.report(10000, 10000, True)['exactMcNemarP'])

    def test_original_power_rule_not_cp_selection(self):
        for successes, label in ((82, 'supports_at_least_80_percent'), (78, 'unresolved'), (77, 'below_80_percent')):
            self.assertEqual(power.primary(successes, 100-successes, M=100)['decision'], label)
            self.assertEqual(other_power.classify_primary(successes, 0, 100), label)
        self.assertFalse(power.primary(82, 17, M=100)['fullFlowEligible'])
        self.assertEqual(power.primary(3080, 1532)['decision'], 'below_80_percent')
        self.assertAlmostEqual(power.primary(3080, 1532)['pHat'], 3080/4612)
        for k, n in ((0, 0), (0, 100), (100, 100), (47, 100)):
            check.near(power.proportion(k, n), other_power.proportion(k, n), 'CP endpoint')
            check.near(coverage.cp(k, n), other_coverage.exact_cp(k, n), 'coverage CP')

    def test_full_union_and_contradictory_event_refusals(self):
        event = empty_event()
        for rows, full in (([event, event], False), ([event], True)):
            with self.assertRaises(ValueError):
                runner.reduce_events('midpoint', rows, full=full)
        for key in ('D', 'F', 'R'):
            wrong = copy.deepcopy(event)
            wrong[key] = True
            with self.assertRaises(ValueError):
                runner.reduce_events('midpoint', [wrong])
        with self.assertRaises(ValueError):
            runner.reduce_events('coverage', [], full=True)

    def test_strict_holm_boundary(self):
        rows = [dict(comparisonID=k, status='available', reason=None,
            p=dict(D_less=.00625, D_greater=1., S_less=1., S_greater=.00625)) for k in d.COMPARISONS]
        family = composition.compose_family(rows)
        self.assertFalse(family['anyMagnitudeRejects'])
        self.assertEqual(family['tests'][d.COMPARISONS[0]]['holmP'], .05)


class PortableAdmission(unittest.TestCase):
    def test_native_midpoint_roster_is_initial_not_policy(self):
        expected = [(a, v, m) for a in d.ARMS for v in range(9) for m in range(1, 11)]
        selected = native_midpoint.selected()
        self.assertEqual([(r['conditionId'], r['scheduleVariant'], r['mainIndex'])
                          for r in selected], expected)
        self.assertEqual(len(selected), 270)
        for roots in ([], selected[:-1], selected[::-1], selected+[selected[0]]):
            with self.assertRaisesRegex(ValueError, 'Exact ordered 270'):
                native_midpoint.library(roots, {})

    def test_native_and_checked_mapping_sources_cannot_be_mixed(self):
        for study, mapping in (('coverage', None), ('midpoint', Path('mapping'))):
            with self.assertRaisesRegex(ValueError, 'distinct exclusive sources'):
                runner.prepare(SimpleNamespace(study=study, native_midpoint=True, mapping=mapping))

    def test_unknown_midpoint_source_mode_refuses_before_estimation(self):
        plan = d.plan('midpoint')
        value = dict(schema=runner.SCHEMA, plan=plan, sources=[], upstream=None,
                     mapping=None, library=None, nativeLibrary=None, initialLibraryMode='unverified')
        with patch.object(runner.io, 'pin', return_value={'path': 'unused/manifest.json'}), \
             patch.object(runner.io, 'read', return_value=value), \
             patch.object(runner.io, 'inventory', return_value={'files': []}):
            with self.assertRaisesRegex(ValueError, 'Unknown midpoint library source'):
                runner.context(Path('unused'))

    def test_fresh_routes_keep_exact_python_pin(self):
        with patch.object(runner.io.sys, 'version_info', (3, 10, 20)):
            for study in ('coverage', 'midpoint'):
                with self.assertRaisesRegex(ValueError, '3.10.9'):
                    runner.numerical_runtime(study)

    def test_original_guard_name_adapter_preserves_all_flags(self):
        from src.tutoring_statistics import saved_fit_magnitude as source
        raw = dict(status='fitted', diagnostics=dict(converged=True, singular=False,
            usableCandidateInference=True, gradient=[0.], hessianEigenvalues=[1.]),
            coefficientNames=list(source.NAMES), coefficients=dict.fromkeys(source.NAMES, 0.),
            vcov=[[float(i == j) for j in range(9)] for i in range(9)])
        guard = runner.original_guard(raw)
        self.assertEqual(set(guard), {'converged', 'nonsingular', 'hessianOK', 'gradientOK', 'finiteBeta', 'finiteV'})
        self.assertTrue(all(guard.values()))
        raw['vcov'][0][0] = float('nan')
        raw['diagnostics']['usableCandidateInference'] = False
        self.assertIs(runner.original_guard(raw)['finiteV'], False)
        raw['diagnostics']['usableCandidateInference'] = True
        with self.assertRaises(ValueError):
            runner.original_guard(raw)

    def test_coverage_event_is_admitted_by_original_guard_and_truth_schema(self):
        from src.tutoring_statistics import saved_fit_magnitude as source
        raw = dict(status='fitted', diagnostics=dict(converged=True, singular=True,
            usableCandidateInference=False, gradient=[0.], hessianEigenvalues=[1.]),
            coefficientNames=list(source.NAMES), coefficients=dict.fromkeys(source.NAMES, 0.),
            vcov=[[float(i == j) for j in range(9)] for i in range(9)])
        actual, _ = fixture()
        within = core.estimate_within(actual, composition.compose_family)
        unavailable = source.unavailable_method('statistical_unavailable',
                                               'accepted_original_guard_false', composition)
        files = {'commit': {'result': {'path': 'result'}, 'input': {'path': 'binding'}},
                 'result': {'record': {'methods': {'baseline': unavailable, 'candidate': unavailable}}},
                 'binding': {'baseline': {'path': 'baseline'}}, 'baseline': {'raw': raw}}
        for ordinal in (0, 10000):
            case = d.coverage_case(ordinal)
            inputs = dict(case=case, comparatorCommit={'path': 'commit'}, populationCounts=dict.fromkeys(d.ARMS, 2))
            evaluation = core.evaluate_truth(within, case['parameters']['slopes'])
            with patch.object(runner.io, 'read', side_effect=lambda path: files[path]):
                event = runner.coverage_event(inputs, within, evaluation)
            other_coverage.audit_event(event)
            self.assertEqual(set(event['originalGuard']),
                             {'converged', 'nonsingular', 'hessianOK', 'gradientOK', 'finiteBeta', 'finiteV'})
            self.assertFalse(event['originalGuard']['nonsingular'])
            self.assertEqual(event['methods']['within']['status'], 'available')
            self.assertEqual(event['methods']['baseline']['status'], 'statistical_unavailable')
            self.assertFalse(event['methods']['within']['conditionedOnOriginalGuard'])
            self.assertEqual(event['magnitudeTrueNull'], dict.fromkeys(d.COMPARISONS, ordinal == 0))
            wrong = copy.deepcopy(event)
            wrong['originalGuard']['finiteModelV'] = wrong['originalGuard'].pop('finiteV')
            with self.assertRaises(ValueError):
                other_coverage.audit_event(wrong)

    def test_real_record_resume_and_changed_input_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            ctx = dict(output=output, plan=d.plan('midpoint'), manifestPin={'path': 'fixture', 'sha256': 'fixture'})
            directory = runner.paths(ctx, 0)
            actual = runner.io.write(directory / 'actual_request.json', dict(model='H10', reference='C2', rows=[]))
            generation = runner.io.write(directory / 'generation.json', {})
            inputs = dict(case=d.midpoint_case(0), request=actual, generation=generation,
                          comparatorCommit=None, populationCounts=dict.fromkeys(d.ARMS, 0))
            input_pin = runner.io.write(directory / 'input.json', inputs)
            runner.io.write(directory / 'record.json', dict(schema=runner.SCHEMA, manifest=ctx['manifestPin'],
                            ordinal=0, input=input_pin))
            with patch.object(midpoint, 'generate', side_effect=AssertionError('resume drew inputs')):
                self.assertTrue(runner.produce_one(ctx, 0)['reused'])
            # A source mutation is rejected at the actual record-admission boundary.
            (directory / 'generation.json').write_text('{"changed": true}\n')
            with self.assertRaises(ValueError):
                runner.produce_one(ctx, 0)

    def test_no_automatic_partial_input_redraw(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = dict(output=Path(tmp), plan=d.plan('midpoint'), manifestPin={})
            directory = runner.paths(ctx, 0)
            runner.io.write(directory / 'actual_request.json', {})
            with patch.object(runner, 'numerical_runtime'), patch.object(midpoint, 'generate', side_effect=AssertionError('redraw')):
                with self.assertRaises(ValueError):
                    runner.produce_one(ctx, 0)

    def test_cli_help_and_reduction_requires_execute(self):
        root = Path(__file__).resolve().parents[1]
        for script in ('reproduce_within_statistics.py', 'reproduce_proxy_h10_mapping.py'):
            completed = subprocess.run([sys.executable, str(root/'scripts'/script), '--help'], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        with self.assertRaises(ValueError):
            runner.main(['produce', '--output', 'unused', '--ids', '0'])


class MappingSchema(unittest.TestCase):
    def test_description_version_literals_are_not_canonicalized(self):
        expected = mapping.io.read(mapping.ROOT/'data/numerical_environment.json')['statistical_reference']
        self.assertEqual(expected['R_packages']['lme4'], '1.1-35.5')
        self.assertEqual(expected['R_packages']['lmerTest'], '3.1-3')
        self.assertIn('read.dcf(file.path(location,"DESCRIPTION"),fields="Version")', mapping.R_PROBE)
        self.assertNotIn('packageVersion(', mapping.R_PROBE)
        self.assertIn("'--vanilla'", (mapping.ROOT/'proxy_h10_mapping.py').read_text())

    def test_runtime_changed_or_missing_is_refused_without_fit(self):
        with patch.object(mapping.shutil, 'which', return_value=None), patch.object(mapping.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                mapping.runtime('missing-Rscript')
            run.assert_not_called()
        with self.assertRaises(ValueError):
            mapping.runtime_matches({'R': 'old', 'files': []}, {'R': 'changed', 'files': []})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'runtime-file'
            path.write_text('original')
            value = dict(files=[mapping.io.pin(path)])
            path.write_text('changed')
            with self.assertRaises(ValueError):
                mapping.runtime_matches(value, value)

    def test_completed_mapping_rechecks_runtime_without_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, csv_path = root/'source', root/'mapping', root/'rows.csv'
            source.mkdir()
            output.mkdir()
            csv_path.write_text('source fixture')
            saved = dict(executable='old', files=[])
            mapping.io.write(output/'runtime.json', saved)
            mapping.io.write(output/'mapping.json', dict(artifacts=[], source=dict(
                rows=mapping.io.pin(csv_path), inputs=[], statisticalSources=[],
                inputCoordinateFrame=mapping.LOGICAL_FRAME,
                analysisCoordinateFrame=mapping.ANALYSIS_FRAME)))
            with patch.object(mapping, 'load_rows', return_value=[]), \
                 patch.object(mapping, 'authenticate', return_value=[]), \
                 patch.object(mapping.io, 'inventory', return_value={'files': []}), \
                 patch.object(mapping, 'runtime', return_value=dict(executable='changed', files=[])), \
                 patch.object(mapping, 'fit_rows') as fit:
                with self.assertRaisesRegex(ValueError, 'Changed R executable'):
                    mapping.main(['produce', '--rows', str(csv_path), '--source', str(source), '--output', str(output)])
                fit.assert_not_called()

    def rows(self):
        return [dict(rate=r, condition=a, variant=v, main=m, person=a+'_v'+str(v),
            price=price, trial_id=a+':'+str(v)+':'+str(m), need_a=50., need_b=60.,
            amount_a=50., budget_milli=100000)
            for r, price in mapping.RATES.items() for a in ('C1', 'C2', 'C3', 'C4', 'C5')
            for v in range(9) for m in range(1, 11)]

    def test_displayed_frame_preserves_native_integer_allocation(self):
        rows = [dict(self.rows()[i], amount_a=123000*271/1000000,
                     budget_milli=123000) for i in range(2)]
        orientations = {('C1', 0, 1): ['A', 'B'], ('C1', 0, 2): ['B', 'A']}
        panel = dict(coordinateFrame=mapping.LOGICAL_FRAME, rows=rows)
        other = mapping.displayed_panel(panel, orientations)
        self.assertEqual(other['coordinateFrame'], mapping.ANALYSIS_FRAME)
        self.assertEqual(other['rows'][0], rows[0])
        self.assertEqual((other['rows'][1]['need_a'], other['rows'][1]['need_b']), (60., 50.))
        self.assertEqual(other['rows'][1]['amount_a'], 123000*729/1000000)
        self.assertEqual(rows[1]['amount_a'], 123000*271/1000000)
        with self.assertRaises(ValueError):
            mapping.displayed_panel(other, orientations)
        for invalid in ({}, {('C1', 0, 1): ['A', 'A']}):
            with self.assertRaises(ValueError):
                mapping.displayed_panel(panel, invalid)
        off_grid = copy.deepcopy(panel)
        off_grid['rows'][0]['amount_a'] += .001
        with self.assertRaises(ValueError):
            mapping.displayed_panel(off_grid, orientations)

    def test_native_scenario_rejects_wrong_coordinate_frame(self):
        for frame in (mapping.LOGICAL_FRAME, 'unknown'):
            value = dict(schema=native_midpoint.SCHEMA, coordinateFrame=frame)
            with patch.object(native_midpoint.io, 'numerical_runtime'), \
                 patch.object(native_midpoint.io, 'read', return_value=value), \
                 patch.object(native_midpoint, 'source', return_value=(SimpleNamespace(), {})):
                with self.assertRaisesRegex(ValueError, 'Changed frozen native scenario'):
                    native_midpoint.admit(Path('unused'))

    def save(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=mapping.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_roster_grouping_price_and_endpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows.csv'
            rows = self.rows()
            self.save(path, rows)
            loaded = mapping.load_rows(path)
            library = mapping.library(loaded)
            self.assertEqual([r['main'] for r in library['C1'][0]], list(range(1, 11)))
            self.assertEqual(len(library['C3']), 9)
            for mutate in ('missing', 'duplicate', 'price', 'trial', 'person'):
                wrong = copy.deepcopy(rows)
                if mutate == 'missing':
                    wrong.pop()
                elif mutate == 'duplicate':
                    wrong[1] = wrong[0]
                elif mutate == 'price':
                    wrong[0]['price'] = .5
                elif mutate == 'trial':
                    wrong[1]['trial_id'] = wrong[0]['trial_id']
                else:
                    wrong[10]['person'] = wrong[0]['person']
                self.save(path, wrong)
                with self.assertRaises(ValueError, msg=mutate):
                    mapping.load_rows(path)

    def test_string_main_fit_order_distinct_from_numeric_library(self):
        rows = self.rows()
        table = list(csv.DictReader(mapping.ordered_csv(rows).splitlines()))
        self.assertEqual([r['main'] for r in table[:10]], ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertEqual(len(table), 2700)


if __name__ == '__main__':
    unittest.main()
