"""Portable saved-design admission, conditional planning and scientific tables.

Public preparation checkpoints and explicitly supplied mathematical operands are
different routes. Neither invokes C5 preparation, policy evaluation or sampling.
"""
import csv
import hashlib
import io
from pathlib import Path

from . import c5_precision_planning as P
from . import c5_precision_reference as V
from . import coefficient_io as IO
from . import personal_time_inputs as I
from . import personal_time_sensitivity as sensitivity
from . import cost_aware_inputs

RECIPE = {
    'schema': 'c5-precision-recipe-v1',
    'models': ['gap', 'partial_low'], 'schedules': list(range(9)), 'rootsPerSchedule': 10,
    'law': 'strict_CDF_53_bit_uniform_grid_with_category100_fallback',
    'lawIsNotExactRealGaussian': True,
    'target': 'equal_nine_unnormalised_nontie_population_projection_share_slopes',
    'designVectors': {'gap': ['1', '(H-L)*1000/B'], 'partial_low': ['1', 'L*1000/B', 'H*1000/B']},
    'predictorUnits': 'L,H_integer_need_categories;B_original_budget_millihours',
    'gramDefinition': 'G_s=(1/10)*sum_m_E[I*z*z_transpose]_without_nontie_normalization',
    'slopeIndex': 1, 'control': 'I*w*(Y-1/2)',
    'alpha': ['1', '20'], 'epsilon': ['1', '10'], 'activeTargets': 2,
    'originalLooks': [16, 32, 64, 128, 256], 'originalLogArgument': 800,
    'candidateInferenceLooks': 1, 'candidateLogArgument': 80,
    'selectionTieRule': 'Bernstein',
    'assumptions': {
        'independence': 'root_times_replicate_is_assumed_not_proved_by_PRNG_keys',
        'withinRoot': 'adaptive_dependence_within_one_root_experiment_allowed',
        'betweenTargets': 'correlation_allowed_simultaneous_guarantee_uses_union_bound',
        'outcome': 'complete_well_defined_higher_true_need_share_in_[0,1]_on_full_nontie_support',
        'ties': 'I=0_contribution_zero',
        'failures': 'undefined_nontie_share_missing_root_or_technical_failure_cannot_be_zeroed_or_dropped',
        'centering': 'intercept_population_identity_not_samplewise_cancellation',
        'variance': 'design_upper_envelope_not_actual_policy_variance',
        'numerics': 'future_nonexact_contributions_require_error_budget_and_prospective_replanning',
    },
    'interpretation': 'conditional_planning_not_achieved_precision_confidence_or_policy_adoption',
}
ZERO = dict(newScientificDraws=0, newPolicyEvaluations=0, newNativeRoots=0,
            newCoefficientFits=0, oldDecisionRecalculations=0,
            newFiniteLawEnumerations=0, newGramInversions=0)
PUBLIC_SCOPE = 'compatible_public_checked_designs_not_new_finite_law_verification'
OPERAND_SCOPE = 'supplied_mathematical_operands_not_execution_or_law_certification'


def source_identity():
    """Bind the actual entry, math and saved-input/I/O helpers, not a broad epoch."""
    root = Path(__file__).resolve().parents[2]
    files = (Path(__file__), Path(P.__file__), Path(V.__file__), Path(IO.__file__),
             Path(I.__file__), Path(I.A.__file__), Path(cost_aware_inputs.__file__),
             Path(sensitivity.__file__), root/'scripts/reproduce_precision_planning.py')
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _read(path, evidence):
    path = Path(path)
    I.need(path.is_file() and not path.is_symlink(), 'Ordinary nonsymlink input required')
    raw = path.read_bytes()
    value = I.parse(raw.decode())
    evidence[path.name if path.parent.name not in ('designs', 'independent_designs')
             else path.parent.name+'/'+path.name] = hashlib.sha256(raw).hexdigest()
    return value


def _checkpoint(path, evidence, binding=None):
    value = _read(path, evidence)
    I.need(type(value) is dict and set(value) == {'schema', 'binding', 'payload', 'contentSHA256'},
           'Exact public preparation checkpoint')
    body = {key: value[key] for key in ('schema', 'binding', 'payload')}
    I.need(value['schema'] == 'coefficient-checkpoint-v1'
           and value['contentSHA256'] == I.digest(body)
           and (binding is None or value['binding'] == binding), 'Changed or foreign saved checkpoint')
    return value['payload'], value['binding']


def validate_operands(designs, summary):
    """Validate typed saved geometry, without recomputing a law, G or its inverse."""
    I.need(type(designs) is list and len(designs) == 9, 'Exactly nine ordered design payloads')
    for variant, design in enumerate(designs):
        I.need(type(design) is dict and set(design) == set(P.MODELS), 'Exact design model roster')
        for model, size in (('gap', 2), ('partial_low', 3)):
            row = design[model]
            I.need(set(row) == {'G', 'determinant', 'status', 'inverse', 'rootRanges'}
                   and row['status'] == 'available', 'Complete saved full-rank matrix operands')
            I.need(P.fraction(row['determinant']) > 0, 'Positive saved determinant')
            for name in ('G', 'inverse'):
                matrix = row[name]
                I.need(type(matrix) is list and len(matrix) == size
                       and all(type(r) is list and len(r) == size for r in matrix), 'Exact matrix dimensions')
                for line in matrix:
                    for value in line:
                        P.fraction(value)
                I.need(all(matrix[i][j] == matrix[j][i] for i in range(size) for j in range(size)),
                       'Symmetric saved matrix')
            I.need(P.fraction(row['inverse'][1][1]) > 0, 'Positive saved slope inverse diagonal')
            I.need([r['rootKey'] for r in row['rootRanges']] ==
                   [['C5', variant, m] for m in range(1, 11)], 'Exact ordered ninety-root target')
            for root in row['rootRanges']:
                I.need(set(root) == {'rootKey', 'weightMinimum', 'weightMaximum',
                                     'singlePrice', 'pairedDifference'}, 'Exact signed root-range fields')
                for key in ('singlePrice', 'pairedDifference'):
                    I.need(type(root[key]) is list and len(root[key]) == 2, 'Saved interval shape')
                    lo, hi = map(P.fraction, root[key])
                    I.need(lo <= hi, 'Ordered saved interval')
                for key in ('weightMinimum', 'weightMaximum'):
                    if root[key] is not None:
                        P.fraction(root[key])
    I.need(summary['activeJ'] == 2 and summary['precisionBranch'] == 'analytic_high_zero_low_two'
           and summary['reportQuantityCount'] == 6 and summary['pairedDifference'] == 'negative_low_price'
           and summary['highPriceSlopes'] == [['0', '1'], ['0', '1']]
           and summary['floorFeasible'] is False and summary['policyDrawsReleased'] is False
           and summary['precisionEstablished'] is False, 'Analytic high-zero and failed original-floor branch')
    I.need([r['model'] for r in summary['ranges']] == list(P.MODELS), 'Exactly two ordered summary targets')


def operand_document(designs, summary):
    """Build an explicit conditional operand document, never an execution seal."""
    validate_operands(designs, summary)
    return dict(schema='c5-precision-design-operands-v1', scope=OPERAND_SCOPE,
                recipe=RECIPE, designs=designs, summary=summary)


def load_operands(path):
    evidence = {}
    value = _read(path, evidence)
    I.need(type(value) is dict and set(value) == {'schema', 'scope', 'recipe', 'designs', 'summary'}
           and value['schema'] == 'c5-precision-design-operands-v1'
           and value['scope'] == OPERAND_SCOPE and value['recipe'] == RECIPE,
           'Explicit mathematical operands, not a historical or public preparation seal')
    validate_operands(value['designs'], value['summary'])
    return value['designs'], value['summary'], dict(scope=OPERAND_SCOPE, inputs=evidence)


def load_preparation(source, configs):
    """Read the compact checked design chain; do not repeat its ninety-law work."""
    source, configs = Path(source), Path(configs)
    evidence = {}
    run = _read(source/'run.json', evidence)
    roster = [dict(conditionId='C5', scheduleVariant=v, mainIndex=m)
              for v in range(9) for m in range(1, 11)]
    I.need(run['schema'] == 'c5-finite-preparation-v1' and run['selection'] == roster
           and run['sources'] == IO.source_identity()
           and run['contextSHA256'] == I.digest(sensitivity.load_context(configs/'personal_time_sensitivity.json'))
           and run['law'] == RECIPE['law'] and run['looks'] == RECIPE['originalLooks']
           and run['alpha'] == RECIPE['alpha'] and run['radius'] == RECIPE['epsilon']
           and run['estimand'] == 'equal_nine_conditional_nontie_population_linear_projections'
           and run['highSource'] in ('checked_public_reuse', 'fresh_public_initial_decisions'),
           'Compatible complete public preparation source/recipe/selection binding')
    check = _read(source/'independent_check.json', evidence)
    I.need(check['passed'] is True and check['verifiedRoots'] == 90 and check['verifiedDesigns'] == 18
           and check['fullFiniteTarget'] is True and check['symmetryCertified'] is True
           and check['sourceSemantics'] == 'public_source_epoch_not_historical_receipts'
           and check['highSource'] == run['highSource'] and check['policyDrawsReleased'] is False
           and check['precisionEstablished'] is False and check['wholeReplicateRNGReady'] is False,
           'Complete independent public preparation scope; no precision or execution release')
    expected = {'C5_v%d.json' % v for v in range(9)}
    for folder in ('designs', 'independent_designs'):
        I.need(not (source/folder).is_symlink()
               and {p.name for p in (source/folder).iterdir()} == expected, 'Exact nine-stage roster')
    designs = []
    for variant in range(9):
        name = 'C5_v%d.json' % variant
        design, bound = _checkpoint(source/'designs'/name, evidence)
        I.need(set(bound) == {'runSHA256', 'lawsSHA256', 'schedule'}
               and bound['runSHA256'] == I.digest(run) and bound['schedule'] == variant
               and type(bound['lawsSHA256']) is str and len(bound['lawsSHA256']) == 64,
               'Design belongs to this public run and schedule')
        independent, _ = _checkpoint(source/'independent_designs'/name, evidence,
                                     {**bound, 'producerSHA256': I.digest(design)})
        I.need(independent == design, 'Exact independent saved G/inverse/range agreement')
        designs.append(design)
    summary, bound = _checkpoint(source/'summary.json', evidence)
    I.need(set(bound) == {'runSHA256', 'designsSHA256', 'lawsSHA256'}
           and bound['runSHA256'] == I.digest(run)
           and bound['designsSHA256'] == I.digest([dict(schedule=v, payload=d) for v, d in enumerate(designs)])
           and type(bound['lawsSHA256']) is str and len(bound['lawsSHA256']) == 64
           and summary == check['summary'], 'Checked summary bound to exactly the consumed design payloads')
    validate_operands(designs, summary)
    return designs, summary, dict(scope=PUBLIC_SCOPE, inputs=evidence,
                                 admission='compact_public_stage_chain_not_repeated_law_or_runtime_audit')


def _tables(output, plan):
    rows = []
    for row in plan['rows']:
        for kind in ('originalEBNecessaryOnly', 'fixedLookHoeffdingSufficient', 'fixedLookBernsteinSufficient'):
            certificate = row[kind]
            rows.append(dict(target=row['model'], bound=kind, replicates=certificate['replicates'],
                             previous_replicates=certificate['previousReplicates'],
                             log_argument=certificate['logArgument'],
                             interpretation='necessary_floor_only' if kind == 'originalEBNecessaryOnly'
                             else 'conditional_sufficient_fixed_single_look'))
    family = {'originalEBNecessaryOnly': plan['originalEBNecessaryFullFamilyReplicates'],
              **plan['candidateFullFamilyReplicates']}
    for kind, number in sorted(family.items()):
        rows.append(dict(target='simultaneous_two_target_family', bound=kind, replicates=number,
                         previous_replicates=number-1, log_argument=800 if kind == 'originalEBNecessaryOnly' else 80,
                         interpretation='necessary_floor_only' if kind == 'originalEBNecessaryOnly'
                         else 'conditional_sufficient_fixed_single_look'))
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    I.write_bytes(Path(output)/'precision_counts.csv', text.getvalue().encode())


def run(source, output, configs=None, *, mathematical_operands=False, checking=False):
    """Missing-only planning resume; every call independently checks all six bounds."""
    source, output = IO.separate(source, output)
    if mathematical_operands:
        designs, summary, evidence = load_operands(source)
    else:
        I.need(configs is not None, 'Public configuration directory required')
        designs, summary, evidence = load_preparation(source, configs)
    binding = dict(schema='c5-precision-planning-v1', recipe=RECIPE,
                   inputsSHA256=I.digest(dict(designs=designs, summary=summary)), sourceEvidence=evidence,
                   sources=source_identity())
    allowed = {'plan.json', 'independent_check.json', 'precision_counts.csv'}
    I.need(not output.exists() or {p.name for p in output.iterdir()} <= allowed,
           'Foreign planning output members')
    plan, new = IO.stage(output/'plan.json', binding, lambda: P.calculations(designs, summary), readonly=checking)
    checked = V.verify_plan(plan, designs, summary)
    proof = dict(schema='c5-precision-independent-v1', binding=binding, planSHA256=I.digest(plan),
                 counters=ZERO, precisionEstablished=False, **checked)
    I.write(output/'independent_check.json', proof)
    _tables(output, plan)
    return dict(output=str(output), sourceScope=evidence['scope'], newPlanningResults=int(new),
                newIndependentChecks=1, counters=ZERO, precisionEstablished=False,
                **{k: checked[k] for k in ('passed', 'candidateFullFamilyReplicates', 'recommendedMethod',
                                          'fixedReplicates', 'futurePlanNotAdopted', 'policyDrawsReleased')})
