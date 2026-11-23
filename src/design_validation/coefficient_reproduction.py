"""Read-only coefficient diagnosis from sealed public reductions, or fresh source calls."""
from pathlib import Path

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import paper_reproduction as Paper
from . import true_need_reproduction as T
from . import coefficient_diagnostics as D
from . import coefficient_reference as V
from . import coefficient_io as IO
from . import coefficient_original as O
from .personal_time_reproduction import csv_bytes


def admit_bundle(source, configs):
    """Validate the existing public figure seal without rendering or refitting.

    A seal verifies compatible saved bytes and their declared source chain. It
    does not independently authenticate historical execution or recreate policies.
    """
    source, configs = Path(source), Path(configs)
    data = I.read(source / 'figure_bundle.json')
    I.need(type(data) is dict and set(data) == {'schema', 'payload', 'identity', 'evidence', 'contentSHA256'},
           'Exact complete public reduction bundle required')
    I.need(data['schema'] == Paper.BUNDLE_SCHEMA and data['contentSHA256'] == I.digest(
        {k: v for k, v in data.items() if k != 'contentSHA256'}), 'Changed bundle seal')
    I.need(data['identity'] == Paper._plot_identity(configs / 'paper_figures.json'),
           'Compatible public source/recipe required')
    e, payload = data['evidence'], data['payload']
    I.need(e['verifiedTrials'] == 2700 and e['completeIndependentReconstruction'] is True,
           'Complete checked 2700 endpoint chain required')
    I.need(e['policyBinding']['sources'] == S.implementation_identity(), 'Policy source mismatch')
    context = S.load_context(configs / 'personal_time_sensitivity.json')
    I.need(e['policyBinding']['contextSHA256'] == I.digest(context), 'Policy configuration mismatch')
    selected = I.selection(e['policyBinding']['selection'])
    I.need({I.coordinates(r) for r in selected} == I.ORIGINAL, 'Exact original root roster')
    I.need(S.file_sha(source / 'policy_check.json') == e['policyCheckSHA256'], 'Changed policy check')
    policy = I.read(source / 'policy_check.json')
    I.need(policy['schema'] == 'personal-time-sensitivity-independent-check-v1'
           and policy['binding'] == e['policyBinding'] and policy['verifiedTrials'] == 2700
           and policy['fullFiniteTarget'] is True and policy['completeIndependentReconstruction'] is True
           and policy['absoluteTolerance'] == 1e-10
           and (policy['expectedOnlyEndpoints'], policy['genuineRealizedEndpoints']) == (1860, 840),
           'Policy check scope mismatch')
    required = {'summary.json', 'fixed_outcomes/summary.json', 'trial_metrics.json',
                'fixed_outcomes/trial_metrics.json'}
    I.need(set(e['reductionData']) == required and set(e['reductionChecks']) ==
           {'source_check.json', 'fixed_outcomes/independent_check.json'}, 'Exact reduction files')
    analysis = source / 'analysis'
    for name, digest in {**e['reductionData'], **e['reductionChecks']}.items():
        I.need(S.file_sha(analysis / name) == digest, 'Changed saved reduction: ' + name)
    I.need(set(payload) == {'allocation', 'true_need', 'design', 'recipe'}
           and payload['allocation'] == I.read(analysis / 'fixed_outcomes/summary.json')
           and payload['true_need'] == I.read(analysis / 'summary.json')
           and payload['recipe'] == I.read(configs / 'paper_figures.json')
           and payload['design'] == I.read(configs / 'personal_time_design.json'), 'Bundle/data mismatch')
    fixed = I.read(analysis / 'fixed_outcomes/independent_check.json')
    true = I.read(analysis / 'source_check.json')
    for checked in (fixed, true):
        I.need(checked['verifiedTrials'] == 2700 and checked['fullFiniteTarget'] is True
               and checked['newPolicySimulations'] == 0
               and checked['policyCheckSHA256'] == e['policyCheckSHA256'], 'Reduction check scope')
    I.need(fixed['recipe'] == T.D.RECIPE and true['recipe'] == T.RECIPE
           and fixed['absoluteTolerance'] == 1e-10, 'Frozen reduction recipes')
    I.need(fixed['sources'] == {p.name: S.file_sha(p) for p in
           (Path(T.D.__file__), configs / 'personal_time_diagnostics.json')}, 'Allocation source mismatch')
    I.need(true['sources'] == {p.name: S.file_sha(p) for p in
           (Path(T.__file__), Path(T.T.__file__), configs / 'true_need_diagnostics.json')},
           'True-need source mismatch')
    I.need(true['fixedOutcomeCheckSHA256'] == S.file_sha(analysis / 'fixed_outcomes/independent_check.json')
           and true['policyInputs'] == fixed['policyInputs'], 'Reduction check linkage')
    producers = {str(S.record_path(Path('.'), price, r)) for r in selected for price, _ in S.RATES}
    I.need(set(policy['producerInputs']) == producers and set(fixed['policyInputs']) == producers |
           {p.replace('producer/', 'independent/', 1) for p in producers}, 'Complete endpoint input inventory')
    I.need(all(fixed['policyInputs'][p] == digest for p, digest in policy['producerInputs'].items()),
           'Producer/check input binding')
    rows = [{name: row[name] for name in D.FIELDS} for row in I.read(analysis / 'trial_metrics.json')]
    D.admit(rows, payload['true_need'])
    return rows, payload['true_need'], S.file_sha(source / 'figure_bundle.json')


def diagnose(source, output, configs, *, checking=False):
    source, output = IO.separate(source, output)
    rows, summary, seal = admit_bundle(source, configs)
    bound = dict(schema='coefficient-saved-diagnosis-v1', sources=IO.source_identity(),
                 bundleSHA256=seal, rowsSHA256=I.digest(rows), summarySHA256=I.digest(summary),
                 validationScope='compatible_sealed_reduction_not_fresh_policy_reconstruction')
    return _diagnose(rows, summary, output, bound, checking=checking)


def diagnose_original(source, output, configs, *, checking=False):
    """Diagnose the exact original compact data, not a fabricated paper seal."""
    source, output = IO.separate(source, output)
    rows, summary, original_binding, imported = O.load(source, configs)
    bound = dict(schema='coefficient-original-diagnosis-v1', sources=IO.source_identity(),
                 originalImportSHA256=imported, originalBinding=original_binding,
                 rowsSHA256=I.digest(rows), summarySHA256=I.digest(summary),
                 validationScope='pinned_original_compact_input_diagnosis_not_public_policy_epoch')
    result = _diagnose(rows, summary, output, bound, checking=checking)
    return dict(result, originalInputsByteMatched=True, **O.BOUNDARY)


def _diagnose(rows, summary, output, bound, *, checking=False):
    """Common producer/independent receiver after distinct input admissions."""
    value, new = IO.stage(output / 'diagnostics.json', bound, lambda: D.diagnose(rows, summary),
                          readonly=checking)
    checked = V.verify(rows, summary, value)
    I.write(output / 'independent_check.json', dict(binding=bound, **checked))
    table = []
    for row in value['schedules']:
        geometry = row['geometry']
        table.append(dict(schedule=row['scheduleID'], eligible=geometry['n'],
                          normalized_gram_determinant=geometry['normalizedDeterminant']['decimal'],
                          gap_slope=row['saved']['gap_slope'],
                          lower_need_partial=row['saved']['lower_need_partial'],
                          exact_rank_agrees=row['exactRankAgreesWithSaved']))
    I.write_bytes(output / 'coefficient_designs.csv', csv_bytes(table, tuple(table[0])))
    return dict(output=str(output), newDesignSchedules=270 if new else 0, **checked,
                validationScope=bound['validationScope'], historicalParityEstablished=False)


def fresh(output, configs, *, node='node', workers=1):
    """Explicit expensive original population generation; resume matching public work."""
    from . import c5_reproduction as C5
    output, configs = S.safe_output(output), Path(configs)
    source = output / 'original_endpoints'
    original = Paper.run('all', source, configs / 'paper_figures.json',
                         selected=I.select_roots(full=True), node=node, workers=workers)
    diagnostic = diagnose(source / 'paper', output / 'coefficient_diagnostics', configs)
    chosen = C5.selection(full=True)
    prepared = C5.produce(output / 'c5', configs / 'personal_time_sensitivity.json', chosen,
                          policy=source / 'policy', node=node)
    received = C5.check(output / 'c5', configs / 'personal_time_sensitivity.json', chosen,
                       policy=source / 'policy')
    return dict(original=original, diagnostic=diagnostic, c5=prepared, c5Check=received,
                supplementalPolicyDraws=0, newPrecisionMethod=False)
