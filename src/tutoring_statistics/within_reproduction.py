"""Portable preparation, production, independent checking and scalar reduction.

Coverage reuses the public native/once-REML/eight-contrast chain. Midpoint uses
frozen slopes with either native initial covariates or a checked source mapping;
it never fits a random-intercept model to synthetic midpoint responses.
"""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from . import reproduction as io
from . import within_design as design
from .within_design import require

SCHEMA = 'within-statistics-public-v1'


def numerical_runtime(study):
    require(study in ('coverage', 'midpoint'), 'Unknown numerical study')
    io.numerical_runtime()
    return dict(python=sys.version, executable=sys.executable,
                historicalBinaryIdentityClaimed=False)


def selection(study, ids=None, all_ids=False):
    roster = design.plan(study)['expectedOrdinals']
    require((ids is not None) != bool(all_ids), 'Choose --ids or --all')
    selected = roster if all_ids else ids
    require(type(selected) is list and selected and all(type(i) is int for i in selected)
            and selected == sorted(set(selected)) and set(selected) <= set(roster),
            'IDs must be nonempty, unique, sorted members of the fixed roster')
    return selected


def admit_mapping(path):
    from .proxy_h10_mapping import ANALYSIS_FRAME
    pin = io.pin(Path(path) / 'mapping.json')
    value = io.read(pin['path'])
    require(value['checked'] is True and value['fullSixPrices'] is True
            and value['coordinateFrame'] == ANALYSIS_FRAME
            and value['selectedMidpointSlopes'] == design.MIDPOINT_SLOPES,
            'Complete checked source-derived slope mapping required')
    for descriptor in value['artifacts']:
        io.validate(descriptor)
    for descriptor in value['source']['statisticalSources']:
        io.validate(descriptor)
    io.validate(value['source']['rows'])
    for descriptor in value['source']['inputs']:
        io.validate(descriptor)
    proof = io.read(Path(path) / 'independent.json')
    require(proof['allSixIndependentFits'] is True and proof['midpointFrozenDecimalParity'] is True
            and proof['substitutedScenarioCoefficients'] is False, 'Unverified mapping fit')
    from .proxy_h10_mapping import load_rows, library
    expected = library(load_rows(Path(path) / 'six_prices.csv'))
    require(expected == io.read(value['library']['path']), 'Changed source-derived midpoint library')
    return pin, value['library']


def prepare(args):
    require(args.study in ('coverage', 'midpoint'), 'Preparation requires --study')
    native = bool(getattr(args, 'native_midpoint', False))
    require(not native or (args.study == 'midpoint' and args.mapping is None),
            'Native midpoint and checked mapping are distinct exclusive sources')
    output = io.validate_output(args.output, args.mapping)
    marker = output / 'manifest.json'
    if marker.exists():
        value = context(output)
        require(value['plan'] == design.plan(args.study), 'Changed resumed study')
        require(value['initialLibraryMode'] == ('native' if native else 'mapping' if args.study == 'midpoint' else None),
                'Changed resumed initial library source')
        return dict(manifest=io.pin(marker), reused=True, newInputs=0, newFits=0)
    environment = numerical_runtime(args.study)
    output.mkdir(parents=True, exist_ok=True)
    sources = io.inventory()['files']
    if args.study == 'coverage':
        require(args.mapping is None, 'Coverage uses its original native library, not midpoint mapping')
        from . import magnitude_reproduction as upstream
        upstream_pin = upstream.initialize(output / 'comparators', 'fresh_within_validation',
            SimpleNamespace(power_directory=None, rscript=args.rscript, node=args.node))
        extra = dict(upstream=upstream_pin, mapping=None, library=None, precision=None,
                     nativeLibrary=None, initialLibraryMode=None)
    else:
        require(native or args.mapping is not None, 'Midpoint requires --native-midpoint or complete --mapping')
        if native:
            from .midpoint_inputs import prepare as prepare_native
            native_pin, library_pin = prepare_native(output / 'initial_scenario', args.node)
            mapping = None
        else:
            mapping, library_pin = admit_mapping(args.mapping)
            native_pin = None
        from .independent_within_check import precision_readback
        b = math.ceil(math.log(2/.05)/(2*.02**2))
        require(b == 4612, 'Changed fixed Hoeffding population')
        extra = dict(upstream=None, mapping=mapping, library=library_pin,
                     precision=precision_readback(b), nativeLibrary=native_pin,
                     initialLibraryMode='native' if native else 'mapping')
    return io.write(marker, dict(schema=SCHEMA, plan=design.plan(args.study), sources=sources, environment=environment,
        identity='new_public_reproduction_not_historical_archive_bytes', **extra))


def context(output):
    pin = io.pin(Path(output) / 'manifest.json')
    value = io.read(pin['path'])
    require(value['schema'] == SCHEMA and value['plan'] == design.plan(value['plan']['study'])
            and value['sources'] == io.inventory()['files'], 'Changed scientific plan or source')
    for key in ('upstream', 'mapping', 'library', 'nativeLibrary'):
        if value[key] is not None:
            io.validate(value[key])
    mode = value['initialLibraryMode']
    if value['plan']['study'] == 'coverage':
        require(mode is None and value['nativeLibrary'] is None and value['mapping'] is None,
                'Foreign coverage library source')
    elif mode == 'native':
        from .midpoint_inputs import admit
        require(value['mapping'] is None and value['nativeLibrary'] is not None,
                'Invalid native library source')
        proof, library_pin = admit(value['nativeLibrary']['path'])
        require(proof == value['nativeLibrary'] and library_pin == value['library'],
                'Changed admitted native library')
    else:
        require(mode == 'mapping' and value['nativeLibrary'] is None and value['mapping'] is not None,
                'Unknown midpoint library source')
        proof, library_pin = admit_mapping(Path(value['mapping']['path']).parent)
        require(proof == value['mapping'] and library_pin == value['library'], 'Changed admitted mapping')
    return dict(value, manifestPin=pin, output=Path(output).resolve())


def paths(ctx, ordinal):
    directory = ctx['output'] / 'units' / f'{ordinal:05d}'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def coverage_input(ctx, ordinal):
    from . import magnitude_reproduction as upstream
    root = Path(ctx['upstream']['path']).parent / 'units' / f'{ordinal:05d}'
    had_input = (root / 'spec.json').exists()
    had_fit = (root / 'fit_receipt.json').exists()
    upstream.member((ctx['upstream'], ordinal))
    commitment_pin = io.pin(root / 'commit.json')
    commitment = upstream.admit_commit(ctx['upstream'], commitment_pin, expected_ordinal=ordinal)
    binding = io.read(commitment['input']['path'])
    spec = io.read(binding['spec']['path'])
    for key in ('request', 'generation'):
        io.validate(spec[key])
    return dict(case=design.coverage_case(ordinal), request=spec['request'], generation=spec['generation'],
        comparatorCommit=commitment_pin, populationCounts=spec['populationCounts'],
        upstreamNewInputs=int(not had_input), upstreamNewFits=int(not had_fit))


def midpoint_input(ctx, ordinal, directory):
    from .midpoint_generation import generate
    case = design.midpoint_case(ordinal)
    actual, generation = generate(case, io.read(ctx['library']['path']))
    request = io.write(directory / 'actual_request.json', actual)
    generated = io.write(directory / 'generation.json', generation)
    return dict(case=case, request=request, generation=generated, comparatorCommit=None,
                populationCounts=generation['actualK'])


def normalize_references(refs, state, reason, slopes, *, evaluation=None):
    output = {}
    for key, ref in refs.items():
        pair, need, mode = key.split(':')
        left, right = pair.split('-')
        truth = slopes[left][need] + (-1. if mode == 'D' else 1.)*slopes[right][need]
        status = ref.get('status', 'available' if state == 'complete' else state)
        available = status == 'available'
        uncertain = available and any(math.isclose(truth, x, abs_tol=2e-9, rel_tol=2e-8) for x in ref['CI'])
        covered = ref['CI'][0] < truth < ref['CI'][1] if available and not uncertain else None
        if evaluation is not None:
            uncertain, covered = evaluation[key]['endpointUnresolved'], evaluation[key]['covered']
        output[key] = dict(status=status, reason=None if available else reason, L=ref['L'], truth=truth,
            **{k: ref[k] if available else None for k in
               ('estimate', 'SE', 'df', 'CI', 'statistic', 'lessP', 'greaterP', 'twoSidedP')},
            covered=covered, coverageNumericallyUnresolved=uncertain)
    return output


def original_guard(raw):
    """Rename the covariance flag only; retain every original guard condition."""
    from .saved_fit_magnitude import guard_components
    guard = guard_components(raw)
    expected = {'converged', 'nonsingular', 'hessianOK', 'gradientOK', 'finiteBeta', 'finiteModelV'}
    require(set(guard) == expected and all(type(v) is bool for v in guard.values()),
            'Changed original guard schema')
    return {('finiteV' if k == 'finiteModelV' else k): v for k, v in guard.items()}


def coverage_event(input_value, projection, evaluation, *, independent=False):
    from . import magnitude_composition as composition
    from .saved_fit_magnitude import compose_references
    commit = io.read(input_value['comparatorCommit']['path'])
    result = io.read(commit['result']['path'])
    binding = io.read(commit['input']['path'])
    baseline = io.read(binding['baseline']['path'])
    numerical = io.read(result['artifacts']['independent']['output']['path']) if independent else None
    case = input_value['case']
    slopes = case['parameters']['slopes']
    methods = {}
    for name in ('baseline', 'candidate'):
        source = copy.deepcopy(result['record']['methods'][name])
        if independent and source['status'] == 'complete':
            for key, ref in source['references'].items():
                for field in ('estimate', 'SE', 'df', 'CI', 'statistic', 'lessP', 'greaterP', 'twoSidedP'):
                    ref[field] = numerical['methods'][name]['references'][key][field]
            source['composition'] = compose_references(source['references'], composition)
        reason = next((r['reason'] for r in source['references'].values() if r.get('reason')), None)
        methods[name] = dict(status=source['status'], reason=reason, composition=source['composition'],
            references=normalize_references(source['references'], source['status'], reason, slopes))
    methods['within'] = dict(status=projection['status'], reason=projection['reason'],
        composition=projection['composition'], conditionedOnOriginalGuard=False,
        references=normalize_references(projection['references'], projection['status'],
            projection['reason'], slopes, evaluation=evaluation['references']))
    ordinal = case['repID']-1 if case['caseIndex'] == 1 else case['repID']+9999
    return dict(ordinal=ordinal, caseIndex=case['caseIndex'], caseID=case['caseID'], repID=case['repID'],
        actualK=input_value['populationCounts'], originalGuard=original_guard(baseline['raw']),
        methods=methods, magnitudeTrueNull=evaluation['magnitudeTrueNull'])


def admit_record(ctx, directory, ordinal):
    value = io.read(directory / 'record.json')
    require(value['schema'] == SCHEMA and value['manifest'] == ctx['manifestPin']
            and value['ordinal'] == ordinal, 'Changed stored result identity')
    io.validate(value['input'])
    inputs = io.read(value['input']['path'])
    expected = design.coverage_case(ordinal) if ctx['plan']['study'] == 'coverage' else design.midpoint_case(ordinal)
    require(inputs['case'] == expected, 'Changed input law or seed')
    for key in ('request', 'generation', 'comparatorCommit'):
        if inputs[key] is not None:
            io.validate(inputs[key])
    return value, inputs


def produce_one(ctx, ordinal):
    from . import within_projection as kernel, magnitude_composition as composition
    directory = paths(ctx, ordinal)
    with (directory / 'unit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'record.json').exists():
            admit_record(ctx, directory, ordinal)
            return dict(ordinal=ordinal, reused=True, newInputs=0, newFits=0, newWithin=0)
        numerical_runtime(ctx['plan']['study'])
        input_path = directory / 'input.json'
        if input_path.exists():
            inputs = io.read(input_path)
            expected = design.coverage_case(ordinal) if ctx['plan']['study'] == 'coverage' else design.midpoint_case(ordinal)
            require(inputs['case'] == expected, 'Changed saved input checkpoint')
            for key in ('request', 'generation', 'comparatorCommit'):
                if inputs[key] is not None:
                    io.validate(inputs[key])
            new_inputs = 0
            new_fits = 0
        else:
            require(not (directory / 'actual_request.json').exists() and not (directory / 'generation.json').exists(),
                    'Uncommitted partial input remains; refuse automatic redraw')
            inputs = coverage_input(ctx, ordinal) if ctx['plan']['study'] == 'coverage' else midpoint_input(ctx, ordinal, directory)
            io.write(input_path, inputs)
            new_inputs = inputs.get('upstreamNewInputs', 1)
            new_fits = inputs.get('upstreamNewFits', 0)
        require(not (directory / 'projection_started.json').exists(),
                'Uncommitted numerical attempt remains; inspect before retrying')
        io.write(directory / 'projection_started.json', dict(input=io.pin(input_path), ordinal=ordinal))
        actual = io.read(inputs['request']['path'])
        projection = kernel.estimate_within(actual, composition.compose_family)
        slopes = inputs['case']['parameters']['slopes'] if ctx['plan']['study'] == 'coverage' else design.MIDPOINT_SLOPES
        evaluation = kernel.evaluate_truth(projection, slopes)
        if ctx['plan']['study'] == 'coverage':
            event = coverage_event(inputs, projection, evaluation)
        else:
            from .midpoint_generation import evaluation as midpoint_truth, compact_event
            evaluation = midpoint_truth(projection)
            event = compact_event(ordinal, projection, evaluation, inputs['populationCounts'])
        io.write(directory / 'record.json', dict(schema=SCHEMA, manifest=ctx['manifestPin'], ordinal=ordinal,
            input=io.pin(input_path), within=projection, evaluation=evaluation, event=event,
            humanMethodAdopted=False))
        return dict(ordinal=ordinal, reused=False, newInputs=new_inputs, newFits=new_fits, newWithin=1)


def independent_projection(projection, checked):
    value = copy.deepcopy(projection)
    if value['status'] == 'available':
        for key, ref in value['references'].items():
            other = checked['references'][key]
            for field in ('estimate', 'SE', 'df', 'CI', 'statistic', 'lessP', 'greaterP'):
                ref[field] = other[field]
            ref['twoSidedP'] = min(1., 2*min(ref['lessP'], ref['greaterP']))
    value['composition'] = checked['family']
    return value


def check_one(ctx, ordinal):
    from . import independent_within_check as reader, independent_within_projection as qr
    from . import magnitude_composition as composition
    directory = paths(ctx, ordinal)
    with (directory / 'unit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record, inputs = admit_record(ctx, directory, ordinal)
        record_pin = io.pin(directory / 'record.json')
        if (directory / 'checked.json').exists():
            proof = io.read(directory / 'checked.json')
            require(proof['manifest'] == ctx['manifestPin'] and proof['record'] == record_pin
                    and proof['ordinal'] == ordinal, 'Changed resumed independent proof')
            return dict(ordinal=ordinal, reused=True, newIndependentQR=0, inputLawReplays=0)
        numerical_runtime(ctx['plan']['study'])
        actual, generation = io.read(inputs['request']['path']), io.read(inputs['generation']['path'])
        if ctx['plan']['study'] == 'coverage':
            from . import magnitude_reproduction as upstream
            upstream.admit_commit(ctx['upstream'], inputs['comparatorCommit'], expected_ordinal=ordinal)
            slopes = inputs['case']['parameters']['slopes']
            replay_count = 0  # The admitted upstream proof already replayed the original law.
        else:
            slopes = design.MIDPOINT_SLOPES
            reader.replay_input(io.read(ctx['library']['path']), slopes, design.MIDPOINT_NAMESPACE,
                                ordinal+1, actual, generation)
            replay_count = 1
        counts = inputs['populationCounts']
        reference = reader.fitted_reference(qr, actual['rows'], counts)
        checked = reader.check_projection(record['within'], record['evaluation'], reference,
                                           slopes, counts, composition.compose_family)
        independent = independent_projection(record['within'], checked)
        if ctx['plan']['study'] == 'coverage':
            other_event = coverage_event(inputs, independent, record['evaluation'], independent=True)
            from .independent_within_reduction import audit_event
            audit_event(other_event)
        else:
            from .midpoint_generation import compact_event
            other_event = compact_event(ordinal, independent, record['evaluation'], counts)
        reader.near(record['event'], other_event, 'independently received scalar event')
        io.write(directory / 'checked.json', dict(schema=SCHEMA, manifest=ctx['manifestPin'],
            ordinal=ordinal, record=record_pin, event=other_event, independent=checked,
            independentQR=True, sharedComposition=True, inputLawReplays=replay_count))
        return dict(ordinal=ordinal, reused=False, newIndependentQR=1, inputLawReplays=replay_count)


def reduce_events(study, events, independent_events=None, *, full=False):
    """Pure scalar interface; callers separately establish source authenticity."""
    plan = design.plan(study)
    rows = list(events)
    other_rows = rows if independent_events is None else list(independent_events)
    ids, other_ids = [e['ordinal'] for e in rows], [e['ordinal'] for e in other_rows]
    require(ids == sorted(set(ids)) and other_ids == ids and set(ids) <= set(plan['expectedOrdinals']),
            'Unordered, duplicate, foreign or mismatched scalar membership')
    complete = ids == plan['expectedOrdinals']
    require(not full or complete, 'Full reduction requires the exact frozen union')
    if study == 'coverage':
        from . import within_reduction as producer, independent_within_reduction as reader
        a, b = producer.Accumulator(), reader.Accumulator()
        for event in rows:
            a.add(event)
        for event in other_rows:
            b.add(event)
        reported, independent = a.report(), b.report()
        reader.compare(reported, independent)
    else:
        from . import midpoint_reduction as producer, independent_midpoint_reduction as reader
        a = producer.Accumulator()
        for event in rows:
            a.add(event, event['ordinal'])
        truth = reader.linear_truths(design.MIDPOINT_SLOPES)
        independent = reader.reduce_rows(reader.classified_event(e, truth) for e in other_rows)
        reported = a.report()
        reader.compare(reported, independent)
    return dict(study=study, fullRoster=complete, expectedTotal=plan['expectedTotal'],
        present=len(rows), ordinals=ids, report=reported, independentReport=independent,
        independentReduction=True, newInputs=0, newFits=0, newQR=0,
        scientificScope=plan['scientificScope'])


def reduce_public(ctx, full):
    produced, checked, pins = [], [], []
    for ordinal in ctx['plan']['expectedOrdinals']:
        directory = ctx['output'] / 'units' / f'{ordinal:05d}'
        if not (directory / 'record.json').exists():
            continue
        record_pin, proof_pin = io.pin(directory / 'record.json'), io.pin(directory / 'checked.json')
        record, proof = io.read(record_pin['path']), io.read(proof_pin['path'])
        require(record['manifest'] == proof['manifest'] == ctx['manifestPin']
                and proof['record'] == record_pin and record['ordinal'] == proof['ordinal'] == ordinal
                and proof['independentQR'] is True, 'Unclosed or mismatched scalar proof')
        produced.append(record['event'])
        checked.append(proof['event'])
        pins.extend((record_pin, proof_pin))
    result = reduce_events(ctx['plan']['study'], produced, checked, full=full)
    result.update(manifest=ctx['manifestPin'], scalarSources=pins,
                  evidence='new_public_production_independently_checked_scalar_reduction')
    name = 'aggregate_full.json' if result['fullRoster'] else 'aggregate_partial_'+str(result['present'])+'.json'
    return io.write(ctx['output'] / name, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'prepare', 'produce', 'check', 'reduce'))
    parser.add_argument('--study', choices=('coverage', 'midpoint'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--native-midpoint', action='store_true',
                        help='Regenerate unpaid covariates for the fixed midpoint scenario, without policies or refitting slopes')
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--node', default='node')
    parser.add_argument('--ids', help='Comma-separated original ordinals; never a new sample size')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--full', action='store_true', help='Require exact full membership during reduction')
    parser.add_argument('--events', type=Path, help='Scalar JSONL arithmetic check only; not source authentication')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'plan':
        return design.plan(args.study)
    if args.stage == 'prepare':
        return prepare(args)
    require(args.execute, 'Numerical production/check/reduction requires --execute')
    if args.events is not None:
        require(args.stage == 'reduce' and args.study is not None and args.ids is None and not args.all,
                '--events is only a supplied-scalar reduction')
        output = io.validate_output(args.output, args.events)
        with args.events.open() as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        result = reduce_events(args.study, events, full=args.full)
        result.update(input=io.pin(args.events),
                      evidence='supplied_scalar_arithmetic_only_not_source_authentication')
        return io.write(output / ('scalar_full.json' if result['fullRoster'] else 'scalar_partial.json'), result)
    ctx = context(args.output)
    require(args.study is None or args.study == ctx['plan']['study'], 'Changed selected study')
    if args.stage == 'reduce':
        require(args.ids is None and not args.all, 'Reduction admits the committed roster, not a new selection')
        return reduce_public(ctx, args.full)
    ids = selection(ctx['plan']['study'], [int(x) for x in args.ids.split(',')] if args.ids is not None else None, args.all)
    action = produce_one if args.stage == 'produce' else check_one
    return dict(stage=args.stage, study=ctx['plan']['study'], members=[action(ctx, i) for i in ids])
