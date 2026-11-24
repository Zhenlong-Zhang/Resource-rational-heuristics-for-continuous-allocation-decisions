"""Reproduce the four finite-allocation paper figures from checked model outputs.

The costly policy stage is resumable per root and price. Redraw only admits a
complete, sealed reduction bundle; it never evaluates a policy. Integrity of a
bundle does not establish the historical provenance of an external dataset.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import true_need_reproduction as T
from . import paper_action_tables as A

BUNDLE_SCHEMA = 'finite-allocation-paper-figures-v1'
ACTION_FILES = ('trial_actions.json', 'summary.json', 'action_endpoints.csv',
                'action_conditions.csv', 'source_check.json')


def check_trial(context, root, cfg, run, label, output):
    """Reconstruct missing independent records; validate existing exact checkpoints.

    Keep the existing numerical modules byte-identical so previously checked
    public outputs retain their source binding. This is transport orchestration,
    not a second policy implementation.
    """
    expected = S.record_binding(root, cfg, run, label)
    path = S.record_path(output, label, root)
    record = I.read(path)
    S.validate_record(record, expected)
    reader_path = S.record_path(output, label, root, 'independent')
    exists = reader_path.exists()
    cached = I.read(reader_path) if exists else None
    if exists:
        I.need(type(cached) is dict and set(cached) == {
            'schema', 'binding', 'producerSHA256', 'payload', 'absoluteTolerance', 'sharedKernels'},
            'Malformed independent checkpoint')
        I.need(cached.get('producerSHA256') == S.file_sha(path)
               and cached.get('binding') == expected, 'Stale independent checkpoint')
        independent = cached['payload']
        S.validate_record(S.envelope(expected, independent), expected)
    else:
        independent = S.trial(context, root, cfg, run, label, S.R)
    S.compare(record['payload'], independent, 'independent_complete_trial')
    reader_record = {'schema': 'personal-time-sensitivity-independent-trial-v1', 'binding': expected,
                     'producerSHA256': S.file_sha(path), 'payload': independent,
                     'absoluteTolerance': 1e-10, 'sharedKernels': ['bounded_task', 'budget_horizon',
                     'terminal_time', 'authenticated_input_adapter_and_event_traversal']}
    if exists:
        I.need(cached == reader_record, 'Changed independent checkpoint schema')
    I.write(reader_path, reader_record)
    return record


def _root_work(context, root, run, output, mode='all'):
    """Each process owns one root, so no two workers write a trial record."""
    counts = {'new_producer': 0, 'new_independent': 0}
    for label, _ in S.RATES:
        cfg = S.rate_config(context, root, label)
        expected = S.record_binding(root, cfg, run, label)
        path = S.record_path(output, label, root)
        if path.exists():
            S.validate_record(I.read(path), expected)
        else:
            I.need(mode != 'check', 'Check mode cannot generate missing producer records')
            value = S.trial(context, root, cfg, run, label, S.P)
            record = S.envelope(expected, value)
            S.validate_record(record, expected)
            I.write(path, record)
            counts['new_producer'] += 1
        if mode != 'produce':
            reader = S.record_path(output, label, root, 'independent')
            counts['new_independent'] += int(not reader.exists())
            check_trial(context, root, cfg, run, label, output)
        S.H.native_utility_table.cache_clear()
        S.H.native_perfect_information.cache_clear()
    return counts


def produce_checked(context, selected, output, *, workers=1, mode='all'):
    I.need(type(workers) is int and workers > 0, 'Positive worker count')
    I.need(mode in ('all', 'produce', 'check'), 'Known policy stage')
    roots, run, output = S.load_roots(context, selected, output)
    counts = {'new_producer': 0, 'new_independent': 0}

    def record(result, completed):
        for key in counts:
            counts[key] += result[key]
        print('Checked roots: %d/%d' % (completed, len(roots)), flush=True)

    if workers == 1:
        for i, root in enumerate(roots, 1):
            record(_root_work(context, root, run, output, mode), i)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(roots)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = [pool.submit(_root_work, context, root, run, output, mode) for root in roots]
            for i, future in enumerate(as_completed(pending), 1):
                record(future.result(), i)
    I.write(output / 'producer_complete.json', {
        'schema': 'personal-time-sensitivity-producer-v1', 'binding': run,
        'selectedTrials': len(roots) * len(S.RATES), 'independentlyChecked': False})
    expected_paths = {S.record_path(output, label, root) for root in roots for label, _ in S.RATES}
    I.need(set((output / 'producer').rglob('*.json')) == expected_paths,
           'Missing or foreign producer trial inventory')
    if mode == 'produce':
        return {**counts, 'producedTrials': len(expected_paths), 'independentlyChecked': False}
    rows, pins = [], {}
    for root in roots:
        for label, _ in S.RATES:
            cfg = S.rate_config(context, root, label)
            record = check_trial(context, root, cfg, run, label, output)
            path = S.record_path(output, label, root)
            pins[str(path.relative_to(output))] = S.file_sha(path)
            rows.append({'coordinates': list(I.coordinates(root)), 'rateLabel': label,
                         'metrics': record['payload']['metrics'],
                         'initialDecision': record['payload']['initialDecision']})
    endpoint_counts = S.D.counts(rows)
    expected_only = sum(row['expectedOnlyCount'] for row in endpoint_counts['finalCounts'])
    realized = sum(row['genuineRealizedCount'] for row in endpoint_counts['finalCounts'])
    full = {I.coordinates(root) for root in roots} == I.ORIGINAL
    if full:
        I.need((expected_only, realized) == (1860, 840), 'Original full endpoint-kind counts')
    checked = {'schema': 'personal-time-sensitivity-independent-check-v1', 'binding': run,
               'producerInputs': pins, 'verifiedTrials': len(rows), 'fullFiniteTarget': full,
               'expectedOnlyEndpoints': expected_only, 'genuineRealizedEndpoints': realized,
               'absoluteTolerance': 1e-10, 'completeIndependentReconstruction': True,
               'historicalParityEstablished': False, 'monteCarloPrecisionEstablished': False}
    I.write(output / 'independent_check.json', checked)
    return {**counts, 'verifiedTrials': checked['verifiedTrials'],
            'fullFiniteTarget': checked['fullFiniteTarget']}


def _plot_identity(recipe_path):
    from . import paper_figures_plot as plot
    return {'recipe': S.file_sha(recipe_path), 'plot': S.file_sha(plot.__file__),
            'orchestration': S.file_sha(__file__),
            'action_tables': S.file_sha(A.__file__),
            'allocation': S.file_sha(T.D.__file__),
            'true_need': S.file_sha(T.T.__file__),
            'true_need_adapter': S.file_sha(T.__file__)}


def reduce_checked(source, output, config_path, *, full):
    """Authenticate checked policies, then generate a self-contained figure bundle."""
    from . import paper_figures_plot as plot
    source, output = Path(source).resolve(), S.safe_output(output)
    config_path = Path(config_path).resolve()
    true_config = config_path.parent / 'true_need_diagnostics.json'
    result = T.analyze(source, output / 'analysis', true_config, full=full)
    actions = A.analyze(source, output / 'actions',
                        config_path.parent / 'personal_time_sensitivity.json', full=full)
    I.need(actions['verifiedTrials'] == result['verifiedTrials'], 'Action/reduction endpoint scope mismatch')
    if not full:
        return {**result, 'figures': [], 'paperComplete': False, 'actionTables': actions['tables']}
    analysis = output / 'analysis'
    allocation = I.read(analysis / 'fixed_outcomes/summary.json')
    true_need = I.read(analysis / 'summary.json')
    design = I.read(config_path.parent / 'personal_time_design.json')
    recipe = I.read(config_path)
    # Validate completeness and scientific definitions before publishing a seal.
    plot.figure_data(allocation, true_need, design, recipe)
    payload = {'allocation': allocation, 'true_need': true_need, 'design': design,
               'recipe': recipe}
    paths = ('fixed_outcomes/independent_check.json', 'source_check.json')
    evidence = {name: S.file_sha(analysis / name) for name in paths}
    policy = I.read(source / 'independent_check.json')
    I.write_bytes(output / 'policy_check.json', (source / 'independent_check.json').read_bytes())
    artifacts = ('summary.json', 'fixed_outcomes/summary.json', 'trial_metrics.json',
                 'fixed_outcomes/trial_metrics.json')
    data = {'schema': BUNDLE_SCHEMA, 'payload': payload,
            'identity': _plot_identity(config_path),
            'evidence': {'reductionChecks': evidence,
                         'actionFiles': {name: S.file_sha(output / 'actions' / name) for name in ACTION_FILES},
                         'reductionData': {name: S.file_sha(analysis / name) for name in artifacts},
                         'policyCheckSHA256': S.file_sha(source / 'independent_check.json'),
                         'policyBinding': policy['binding'],
                         'verifiedTrials': result['verifiedTrials'],
                         'completeIndependentReconstruction': policy['completeIndependentReconstruction'],
                         'historicalProvenanceReconstructed': False,
                         'outcomeScope': 'same_target_fixed_model_outcomes_not_all_historical_trajectories'}}
    I.write(output / 'figure_bundle.json', {**data, 'contentSHA256': I.digest(data)})
    return {**result, **redraw(output, output / 'figures', config_path)}


def redraw(bundle_dir, output, config_path):
    """Validate immutable reduced inputs; no native generator or policy is invoked."""
    from . import paper_figures_plot as plot
    source, output = Path(bundle_dir).resolve(), S.safe_output(output)
    I.need(not source.is_relative_to(output.resolve()), 'Output cannot replace bundle directory')
    data = I.read(source / 'figure_bundle.json')
    I.need(set(data) == {'schema', 'payload', 'identity', 'evidence', 'contentSHA256'},
           'Exact figure bundle schema')
    I.need(data['schema'] == BUNDLE_SCHEMA and data['contentSHA256'] == I.digest(
        {k: v for k, v in data.items() if k != 'contentSHA256'}), 'Figure bundle content changed')
    I.need(data['identity'] == _plot_identity(config_path), 'Figure recipe/source version changed')
    e = data['evidence']
    I.need(e['verifiedTrials'] == 2700 and e['completeIndependentReconstruction'] is True,
           'Complete independently checked numerical chain required')
    I.need(e['policyBinding']['sources'] == S.implementation_identity(),
           'Numerical source version changed')
    selected = I.selection(e['policyBinding']['selection'])
    I.need({I.coordinates(row) for row in selected} == I.ORIGINAL,
           'Figure bundle requires all original roots')
    policy_path = source / 'policy_check.json'
    I.need(S.file_sha(policy_path) == e['policyCheckSHA256'], 'Saved policy check changed')
    policy = I.read(policy_path)
    I.need(type(policy) is dict and set(policy) == {'schema', 'binding', 'producerInputs',
        'verifiedTrials', 'fullFiniteTarget', 'expectedOnlyEndpoints', 'genuineRealizedEndpoints',
        'absoluteTolerance', 'completeIndependentReconstruction', 'historicalParityEstablished',
        'monteCarloPrecisionEstablished'}, 'Exact policy-check schema required')
    I.need(policy['schema'] == 'personal-time-sensitivity-independent-check-v1'
           and policy['binding'] == e['policyBinding'] and policy['verifiedTrials'] == 2700
           and policy['fullFiniteTarget'] is True and policy['completeIndependentReconstruction'] is True
           and policy['absoluteTolerance'] == 1e-10
           and (policy['expectedOnlyEndpoints'], policy['genuineRealizedEndpoints']) == (1860, 840),
           'Saved policy check scope or binding mismatch')
    for name, sha in e['reductionChecks'].items():
        I.need(name in ('fixed_outcomes/independent_check.json', 'source_check.json'),
               'Unexpected reduction check path')
        I.need(S.file_sha(source / 'analysis' / name) == sha, 'Reduction check changed')
    I.need(set(e['reductionChecks']) == {'fixed_outcomes/independent_check.json', 'source_check.json'},
           'Both reduction checks required')
    payload = data['payload']
    I.need(set(payload) == {'allocation', 'true_need', 'design', 'recipe'}, 'Exact plot inputs')
    I.need(payload['recipe'] == I.read(config_path), 'Plot recipe mismatch')
    I.need(payload['design'] == I.read(Path(config_path).parent / 'personal_time_design.json'),
           'Scientific task design changed')
    analysis = source / 'analysis'
    required_data = {'summary.json', 'fixed_outcomes/summary.json', 'trial_metrics.json',
                     'fixed_outcomes/trial_metrics.json'}
    I.need(set(e['reductionData']) == required_data, 'Complete saved reduction data required')
    for name in required_data:
        I.need(S.file_sha(analysis / name) == e['reductionData'][name], 'Saved reduction data changed')
    I.need(payload['allocation'] == I.read(analysis / 'fixed_outcomes/summary.json')
           and payload['true_need'] == I.read(analysis / 'summary.json'),
           'Bundle differs from its saved reductions')
    for name in e['reductionChecks']:
        checked = I.read(analysis / name)
        I.need(checked['fullFiniteTarget'] is True and checked['verifiedTrials'] == 2700
               and checked['policyCheckSHA256'] == e['policyCheckSHA256'],
               'Reduction scope or policy identity mismatch')
    fixed = I.read(analysis / 'fixed_outcomes/independent_check.json')
    true = I.read(analysis / 'source_check.json')
    I.need(type(fixed) is dict and set(fixed) == {'recipe', 'sources', 'policyCheckSHA256',
        'policyInputs', 'verifiedTrials', 'fullFiniteTarget', 'absoluteTolerance',
        'independentArithmetic', 'sharedKernels', 'historicalProvenanceReconstructed',
        'newPolicySimulations'}, 'Exact fixed-outcome check schema required')
    I.need(type(true) is dict and set(true) == {'recipe', 'sources', 'fixedOutcomeCheckSHA256',
        'policyCheckSHA256', 'policyInputs', 'verifiedTrials', 'fullFiniteTarget',
        'newPolicySimulations', 'historicalProvenanceReconstructed', 'interpretation'},
        'Exact true-need check schema required')
    I.need(fixed['recipe'] == T.D.RECIPE and true['recipe'] == T.RECIPE,
           'Reduction recipe mismatch')
    I.need(fixed['sources'] == {p.name: S.file_sha(p) for p in (
        Path(T.D.__file__), Path(config_path).parent / 'personal_time_diagnostics.json')},
        'Fixed-outcome reduction source mismatch')
    I.need(true['sources'] == {p.name: S.file_sha(p) for p in (
        Path(T.__file__), Path(T.T.__file__), Path(config_path).parent / 'true_need_diagnostics.json')},
        'True-need reduction source mismatch')
    I.need(true['fixedOutcomeCheckSHA256'] == S.file_sha(analysis / 'fixed_outcomes/independent_check.json')
           and true['policyInputs'] == fixed['policyInputs'], 'Reduction check chain mismatch')
    expected_producers = {str(S.record_path(Path('.'), label, row)) for row in selected for label, _ in S.RATES}
    I.need(set(policy['producerInputs']) == expected_producers,
           'Policy check must cover every original endpoint')
    expected_inputs = expected_producers | {p.replace('producer/', 'independent/', 1) for p in expected_producers}
    I.need(set(fixed['policyInputs']) == expected_inputs
           and all(fixed['policyInputs'][p] == sha for p, sha in policy['producerInputs'].items()),
           'Reduction policy-input inventory mismatch')
    I.need(fixed['absoluteTolerance'] == 1e-10 and fixed['newPolicySimulations'] == 0
           and true['newPolicySimulations'] == 0, 'Reduction cannot claim new policy evidence')
    I.need(set(e['actionFiles']) == set(ACTION_FILES), 'Complete action/question files required')
    for name, sha in e['actionFiles'].items():
        I.need(S.file_sha(source / 'actions' / name) == sha, 'Action/question file changed')
    A.verify_saved(source / 'actions', Path(config_path).parent / 'personal_time_sensitivity.json',
                   policy=policy, full=True, policy_sha=e['policyCheckSHA256'])
    values = plot.figure_data(payload['allocation'], payload['true_need'], payload['design'], payload['recipe'])
    I.write(output / 'figure_data.json', values)
    rendered = plot.render(payload['allocation'], payload['true_need'], payload['design'],
                           output, payload['recipe'])
    return {'paperComplete': True, 'figures': [str(p) for p in rendered.values()],
            'vectorFigures': [str(p.with_suffix('.pdf')) for p in rendered.values()],
            'actionTables': [str(source / 'actions' / name) for name in ACTION_FILES],
            'newPolicySimulations': 0}


def run(mode, output, config_path, *, selected=None, source=None, node='node', workers=1):
    """Execute fresh/resumed production, checked-input reduction, or fast redraw."""
    config_path, output = Path(config_path).resolve(), S.safe_output(output)
    if mode == 'redraw':
        I.need(source is not None and selected is None, 'Redraw needs a bundle, not a root selection')
        return redraw(source, output, config_path)
    if mode == 'redraw-original':
        from . import paper_figure_original
        I.need(source is not None and selected is None, 'Historical redraw needs compact inputs only')
        return paper_figure_original.redraw(source, output, config_path)
    I.need(selected is not None, 'Explicit finite root selection required')
    selected = I.selection(selected)
    full = {I.coordinates(row) for row in selected} == I.ORIGINAL
    if mode in ('all', 'produce'):
        I.need(source is None, 'Fresh/resume mode owns its policy input directory')
        source = output / 'policy'
        context = S.load_context(config_path.parent / 'personal_time_sensitivity.json')
        I.write(output / 'paper_run.json', {'schema': BUNDLE_SCHEMA,
                'selection': selected, 'sources': _plot_identity(config_path),
                'policyBinding': S.binding(context, selected)})
        S.prepare(context, selected, source, node=node)
        counts = produce_checked(context, selected, source, workers=workers, mode=mode)
        if mode == 'produce':
            return {**counts, 'policyDirectory': str(source), 'paperComplete': False}
    elif mode == 'check':
        I.need(source is not None, 'Check mode requires producer input')
        context = S.load_context(config_path.parent / 'personal_time_sensitivity.json')
        result = produce_checked(context, selected, source, workers=workers, mode='check')
        I.write(output / 'check_stage.json', {
            'schema': 'paper-policy-check-stage-v1',
            'policyCheckSHA256': S.file_sha(Path(source) / 'independent_check.json'),
            'verifiedTrials': result['verifiedTrials'], 'fullFiniteTarget': result['fullFiniteTarget']})
        return {**result, 'paperComplete': False}
    else:
        I.need(mode == 'reduce' and source is not None, 'Known mode and checked input required')
        I.need(I.read(Path(source) / 'run.json')['selection'] == selected, 'Checked selection mismatch')
        counts = {'new_producer': 0, 'new_independent': 0}
    result = reduce_checked(source, output / 'paper', config_path, full=full)
    if mode == 'all':
        result.pop('newPolicySimulations', None)
    return {**result, **counts}
