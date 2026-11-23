"""Conditional saved-geometry algebra, independent certificates and portable I/O.

Synthetic operands test mathematics and admission boundaries, not the accepted
ninety-law computation or any real policy's achievable precision. No generator,
policy, finite-law enumeration, inverse or draw is needed by these tests.
"""
import copy
import importlib.util
import json
import os
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.design_validation import c5_precision_planning as P
from src.design_validation import c5_precision_reference as V
from src.design_validation import c5_precision_reproduction as R
from src.design_validation import coefficient_io as IO
from src.design_validation import personal_time_inputs as I
from src.design_validation import personal_time_sensitivity as S

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('originalEBNecessaryOnly', 'fixedLookHoeffdingSufficient', 'fixedLookBernsteinSufficient')


def synthetic_operands():
    """Typed hand-specified geometry, explicitly not finite-C5-law evidence."""
    designs = []
    for variant in range(9):
        design = {}
        for model, size in (('gap', 2), ('partial_low', 3)):
            gram = [[['1', '100'] if i == j == 1 else ['1', '1'] if i == j else ['0', '1']
                     for j in range(size)] for i in range(size)]
            inverse = [[['100', '1'] if i == j == 1 else ['1', '1'] if i == j else ['0', '1']
                        for j in range(size)] for i in range(size)]
            design[model] = dict(G=gram, inverse=inverse, determinant=['1', '100'], status='available',
                rootRanges=[dict(rootKey=['C5', variant, m], weightMinimum=['-10', '1'],
                                 weightMaximum=['10', '1'], singlePrice=[['-10', '1'], ['10', '1']],
                                 pairedDifference=[['-10', '1'], ['10', '1']]) for m in range(1, 11)])
        designs.append(design)
    summary = dict(activeJ=2, precisionBranch='analytic_high_zero_low_two', reportQuantityCount=6,
                   highPriceSlopes=[['0', '1'], ['0', '1']], pairedDifference='negative_low_price',
                   policyDrawsReleased=False, precisionEstablished=False, floorFeasible=False,
                   ranges=[dict(model=model, singlePriceRange=[['-10', '1'], ['10', '1']],
                                nominalPairedDifferenceRange=[['-10', '1'], ['10', '1']],
                                zeroVarianceFloors=[dict(replicates=n, status='unattainable')
                                                    for n in (16, 32, 64, 128, 256)]) for model in P.MODELS])
    return designs, summary


def write_public_fixture(path, designs, summary):
    """Mock public admission fields only; never label this as an actual execution."""
    run = dict(schema='c5-finite-preparation-v1',
               selection=[dict(conditionId='C5', scheduleVariant=v, mainIndex=m)
                          for v in range(9) for m in range(1, 11)],
               sources={'syntheticSource': True}, contextSHA256=I.digest({'syntheticContext': True}),
               law=R.RECIPE['law'], looks=R.RECIPE['originalLooks'], alpha=['1', '20'], radius=['1', '10'],
               estimand='equal_nine_conditional_nontie_population_linear_projections',
               highSource='fresh_public_initial_decisions')
    I.write(path/'run.json', run)
    for v, design in enumerate(designs):
        binding = dict(runSHA256=I.digest(run), lawsSHA256='a'*64, schedule=v)
        IO.stage(path/'designs'/('C5_v%d.json' % v), binding, lambda: design)
        IO.stage(path/'independent_designs'/('C5_v%d.json' % v),
                 {**binding, 'producerSHA256': I.digest(design)}, lambda: design)
    IO.stage(path/'summary.json', dict(runSHA256=I.digest(run), lawsSHA256='b'*64,
             designsSHA256=I.digest([dict(schedule=v, payload=d) for v, d in enumerate(designs)])), lambda: summary)
    I.write(path/'independent_check.json', dict(passed=True, verifiedRoots=90, verifiedDesigns=18,
            fullFiniteTarget=True, symmetryCertified=True, summary=summary,
            sourceSemantics='public_source_epoch_not_historical_receipts', highSource=run['highSource'],
            policyDrawsReleased=False, precisionEstablished=False, wholeReplicateRNGReady=False))


class PrecisionPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.designs, cls.summary = synthetic_operands()
        cls.plan = P.calculations(cls.designs, cls.summary)
        cls.proof = V.verify_plan(cls.plan, cls.designs, cls.summary)
        cls.log_intervals = {argument: V.prime_log_interval(argument, 256) for argument in (80, 800)}

    def test_six_original_inequalities_and_scope(self):
        self.assertTrue(self.proof['passed'])
        self.assertEqual(len(self.proof['boundaryCertificates']), 6)
        for certificate in self.proof['boundaryCertificates']:
            self.assertTrue(certificate['currentPasses'])
            self.assertTrue(certificate['previousFails'])
            self.assertTrue(certificate['publishedEnclosureCertified'])
            self.assertEqual(certificate['previousReplicates'], certificate['replicates']-1)
        for kind in KINDS:
            expected = max(row[kind]['replicates'] for row in self.plan['rows'])
            actual = (self.plan['originalEBNecessaryFullFamilyReplicates'] if kind == KINDS[0]
                      else self.plan['candidateFullFamilyReplicates'][kind])
            self.assertEqual(actual, expected)
        self.assertTrue(self.proof['futurePlanNotAdopted'])
        self.assertFalse(self.plan['policyReleaseAuthorized'])
        self.assertFalse(self.proof['policyDrawsReleased'])
        self.assertNotIn('costs', self.plan)
        self.assertEqual(self.plan['inferenceLooks'], 1)

    def test_reference_does_not_call_producer_numerics(self):
        with patch.object(P, 'calculations', side_effect=AssertionError('Producer called')), \
                patch.object(P, 'log_bounds', side_effect=AssertionError('Producer log called')), \
                patch.object(P, 'integer_boundary', side_effect=AssertionError('Producer ceiling called')):
            self.assertEqual(V.verify_plan(self.plan, self.designs, self.summary), self.proof)

    def test_adjacent_integer_and_old_new_allocation_mutations(self):
        known = V.geometry(self.designs, self.summary, 'gap')
        # Cache the independently derived enclosure so refusals do not expand
        # thousands of unnecessary series terms for a deliberately false count.
        with patch.object(V, 'prime_log_interval', side_effect=lambda n, terms: self.log_intervals[n]):
            for kind in KINDS:
                for delta in (-1, 1):
                    item = copy.deepcopy(self.plan['rows'][0][kind])
                    item['replicates'] += delta
                    item['previousReplicates'] += delta
                    with self.assertRaises((ValueError, ArithmeticError)):
                        V.check_integer_certificate(item, kind, known)
                item = copy.deepcopy(self.plan['rows'][0][kind])
                item['logArgument'] = 80 if kind == KINDS[0] else 800
                with self.assertRaises(ValueError):
                    V.check_integer_certificate(item, kind, known)

    def test_wrong_geometry_roster_noncanonical_and_floor_refused(self):
        for update in (
                lambda d, s: d[0]['gap']['rootRanges'][0].update(rootKey=['C5', 0, 2]),
                lambda d, s: d[0]['gap']['inverse'][1].__setitem__(1, ['200', '2']),
                lambda d, s: d[0]['gap']['rootRanges'][0].update(weightMaximum=['-11', '1']),
                lambda d, s: s['ranges'][0]['zeroVarianceFloors'][0].update(status='not_ruled_out')):
            designs, summary = copy.deepcopy((self.designs, self.summary))
            update(designs, summary)
            with self.assertRaises(ValueError):
                P.calculations(designs, summary)
            with self.assertRaises(ValueError):
                V.verify_plan(self.plan, designs, summary)
        with self.assertRaises(ValueError):
            P.integer_boundary(F(0), 80)
        with self.assertRaises(ValueError):
            R.operand_document(self.designs[:-1], self.summary)

    def test_assumption_selection_and_nonrelease_mutations(self):
        for field, value in (('inferenceLooks', 5), ('activeJ', 1), ('fixedLowPriceEndpoints', 1),
                             ('requiredIndependence', 'replicates_only'), ('policyReleaseAuthorized', True),
                             ('withinPrior256ReplicateLimit', not self.plan['withinPrior256ReplicateLimit']),
                             ('estimator', 'old_T_unchanged_samplewise')):
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            with self.assertRaises(ValueError):
                V.verify_plan(plan, self.designs, self.summary)

    def test_all_tie_root_zero_is_not_missing_or_undefined_outcome(self):
        designs, summary = copy.deepcopy((self.designs, self.summary))
        for model in P.MODELS:
            designs[0][model]['rootRanges'][0].update(weightMinimum=None, weightMaximum=None,
                singlePrice=[['0', '1'], ['0', '1']], pairedDifference=[['0', '1'], ['0', '1']])
            row = next(x for x in summary['ranges'] if x['model'] == model)
            row['singlePriceRange'] = [['-89', '9'], ['89', '9']]
        plan = P.calculations(designs, summary)
        self.assertEqual(plan['rows'][0]['centeredRootRadii'][0], ['0', '1'])
        self.assertTrue(V.verify_plan(plan, designs, summary)['passed'])
        self.assertIn('cannot_be_zeroed_or_dropped', R.RECIPE['assumptions']['failures'])

    def test_optional_operand_entry_resume_tables_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'operands.json', Path(tmp)/'plan'
            I.write(source, R.operand_document(self.designs, self.summary))
            first = R.run(source, output, mathematical_operands=True)
            self.assertTrue(first['passed'])
            self.assertEqual(first['sourceScope'], R.OPERAND_SCOPE)
            self.assertEqual(first['newPlanningResults'], 1)
            self.assertTrue(all(value == 0 for value in first['counters'].values()))
            self.assertEqual(I.read(output/'plan.json')['payload'], self.plan)
            self.assertEqual(len((output/'precision_counts.csv').read_text().splitlines()), 10)
            before = (output/'plan.json').read_bytes(), (output/'plan.json').stat().st_mtime_ns
            with patch.object(P, 'calculations', side_effect=AssertionError('Closed plan repeated')):
                second = R.run(source, output, mathematical_operands=True, checking=True)
            self.assertEqual(second['newPlanningResults'], 0)
            self.assertEqual(before, ((output/'plan.json').read_bytes(), (output/'plan.json').stat().st_mtime_ns))
            bad = I.read(output/'plan.json')
            bad['payload']['fixedReplicates'] += 1
            (output/'plan.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                R.run(source, output, mathematical_operands=True, checking=True)

    def test_public_preparation_compact_adapter_and_changed_independent_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'prep'
            write_public_fixture(source, self.designs, self.summary)
            with patch.object(IO, 'source_identity', return_value={'syntheticSource': True}), \
                    patch.object(S, 'load_context', return_value={'syntheticContext': True}):
                designs, summary, evidence = R.load_preparation(source, ROOT/'configs')
                self.assertEqual((designs, summary), (self.designs, self.summary))
                self.assertEqual(evidence['scope'], R.PUBLIC_SCOPE)
                self.assertEqual(len(evidence['inputs']), 21)
                path = source/'independent_designs/C5_v0.json'
                value = I.read(path)
                value['payload']['gap']['inverse'][1][1] = ['101', '1']
                value['contentSHA256'] = I.digest({k: value[k] for k in ('schema', 'binding', 'payload')})
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    R.load_preparation(source, ROOT/'configs')

    def test_saved_mode_binds_actual_helpers_and_refuses_dependency_change(self):
        identity = R.source_identity()
        required = {'src/design_validation/'+name for name in (
            'c5_precision_planning.py', 'c5_precision_reference.py', 'c5_precision_reproduction.py',
            'coefficient_io.py', 'personal_time_inputs.py', 'personal_time_policy.py',
            'cost_aware_inputs.py', 'personal_time_sensitivity.py')}
        self.assertEqual(set(identity), required | {'scripts/reproduce_precision_planning.py'})
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'operands.json', Path(tmp)/'plan'
            I.write(source, R.operand_document(self.designs, self.summary))
            R.run(source, output, mathematical_operands=True)
            for name in ('coefficient_io.py', 'personal_time_inputs.py', 'personal_time_policy.py',
                         'cost_aware_inputs.py', 'personal_time_sensitivity.py'):
                changed = dict(identity)
                changed['src/design_validation/'+name] = '0'*64
                with patch.object(R, 'source_identity', return_value=changed), \
                        patch.object(P, 'calculations', side_effect=AssertionError('Foreign resume recomputed')):
                    with self.assertRaises(ValueError):
                        R.run(source, output, mathematical_operands=True, checking=True)

    def test_cli_no_implicit_source_and_actual_saved_operand_entry(self):
        cli = ROOT/'scripts/reproduce_precision_planning.py'
        for args, code in ((['--help'], 0), (['all', '--output', 'unused'], 2),
                           (['all', '--input', 'unused', '--output', 'unused2', '--input-kind', 'receipt'], 2)):
            result = subprocess.run([sys.executable, '-S', str(cli), *args],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'operands.json', Path(tmp)/'cli'
            I.write(source, R.operand_document(self.designs, self.summary))
            args = ['--input-kind', 'saved-designs', '--input', str(source), '--output', str(output)]
            result = subprocess.run([sys.executable, str(cli), 'all', *args],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            first = json.loads(result.stdout)
            spec = importlib.util.spec_from_file_location('precision_cli_test', cli)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                second = module.main(['check', *args])
            finally:
                os.chdir(previous)
            self.assertEqual(first['newPlanningResults'], 1)
            self.assertEqual(second['newPlanningResults'], 0)
            for value in (first, second):
                self.assertTrue(value['passed'])
                self.assertFalse(value['precisionEstablished'])
                self.assertFalse(value['policyDrawsReleased'])


if __name__ == '__main__':
    unittest.main()
