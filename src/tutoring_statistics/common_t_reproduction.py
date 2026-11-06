"""Saved-coverage common-t comparison; no generation, fitting or old QR replay."""
import argparse
import fcntl
import json
from pathlib import Path

from . import reproduction as io, within_design as design
from .within_design import require

SCHEMA = 'common-t-public-reproduction-v1'
METHOD = 'h10_within_common_t_sign_congruence_IUT2fallback_Holm4_candidate_v1'
NAMESPACE = 'r11-h10-within-common-t-saved-development-v1-20261104'


def plan():
    return dict(methodID=METHOD, namespace=NAMESPACE, inputPlan=design.plan('coverage'),
        expectedTotal=12129, expectedOrdinals=list(range(12129)),
        originalMethod=design.WITHIN_METHOD, alpha=.05,
        scientificScope='retrospective_fixed_input_comparison_not_human_adoption',
        linearIntervalsUnchanged=True, newInputs=0, newFits=0, newWithin=0, newQR=0)


def read_pin(descriptor):
    require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'},
            'Invalid artifact descriptor')
    io.validate(descriptor)
    return io.read(descriptor['path'])


def source_bindings(sources, source_root=None):
    """Authenticate an explicitly preserved upstream checkout, not its execution."""
    require(type(sources) is list and sources, 'Missing upstream source inventory')
    output, seen, parents = [], set(), set()
    for descriptor in sources:
        require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'},
                'Invalid upstream source descriptor')
        path = Path(descriptor['path'])
        parts = path.parts
        require(path.is_absolute() and '..' not in parts
                and parts.count('tutoring_statistics') == 1, 'Ambiguous upstream source path')
        index = parts.index('tutoring_statistics')
        require(index >= 1 and parts[index-1] == 'src', 'Foreign upstream source root')
        relative = Path(*parts[index-1:])
        parents.add(str(Path(*parts[:index-1])))
        require(str(relative) not in seen and path.suffix in ('.py', '.R', '.cjs', '.json'),
                'Duplicate or foreign upstream source')
        seen.add(str(relative))
        resolved = Path(source_root).resolve() / relative if source_root is not None else path
        require(not resolved.is_symlink() and (source_root is None or
                resolved.resolve().is_relative_to(Path(source_root).resolve())),
                'Escaping upstream source archive')
        observed = io.pin(resolved)
        require(observed['sha256'] == descriptor['sha256'], 'Changed archived upstream source')
        output.append(dict(original=descriptor, observed=observed))
    require(len(parents) == 1 and 'src/tutoring_statistics/within_projection.py' in seen
            and 'src/tutoring_statistics/within_reproduction.py' in seen,
            'Incomplete or mixed upstream source roots')
    return output


def prepare(coverage, output, source_root=None):
    output = io.validate_output(output, coverage)
    manifest = io.pin(Path(coverage) / 'manifest.json')
    upstream = read_pin(manifest)
    require(upstream['schema'] == 'within-statistics-public-v1'
            and upstream['plan'] == design.plan('coverage')
            and upstream['initialLibraryMode'] is None, 'Original coverage population required')
    bindings = source_bindings(upstream['sources'], source_root)
    read_pin(upstream['upstream'])
    value = dict(schema=SCHEMA, plan=plan(), coverage=manifest,
        coverageSourceRoot=str(Path(source_root).resolve()) if source_root is not None else None,
        upstreamSources=bindings, sources=io.inventory()['files'],
        identity='new_saved_input_comparison_not_historical_execution')
    return io.write(output / 'manifest.json', value)


def context(output):
    marker = io.pin(Path(output) / 'manifest.json')
    value = read_pin(marker)
    require(value['schema'] == SCHEMA and value['plan'] == plan()
            and value['sources'] == io.inventory()['files'], 'Changed comparison source or plan')
    upstream = read_pin(value['coverage'])
    require(upstream['schema'] == 'within-statistics-public-v1'
            and upstream['plan'] == design.plan('coverage'), 'Changed upstream coverage identity')
    require(source_bindings(upstream['sources'], value['coverageSourceRoot']) == value['upstreamSources'],
            'Changed original source archive binding')
    read_pin(upstream['upstream'])
    return dict(value, manifestPin=marker, upstream=upstream, output=Path(output).resolve())


def selected(ids=None, all_ids=False):
    require((ids is not None) != bool(all_ids), 'Choose --ids or --all')
    values = list(range(12129)) if all_ids else ids
    require(type(values) is list and values and values == sorted(set(values))
            and all(type(i) is int and 0 <= i < 12129 for i in values),
            'IDs must be sorted unique original coverage ordinals')
    return values


def coverage_member(ctx, ordinal):
    """Read the checked request/fit linkage; never call the upstream producer."""
    from . import independent_common_t as reader
    from .saved_fit_magnitude import validate_population
    require(type(ordinal) is int and 0 <= ordinal < 12129, 'Foreign ordinal')
    directory = Path(ctx['coverage']['path']).parent / 'units' / f'{ordinal:05d}'
    record_pin, proof_pin = io.pin(directory / 'record.json'), io.pin(directory / 'checked.json')
    record, proof = read_pin(record_pin), read_pin(proof_pin)
    require(record['schema'] == proof['schema'] == 'within-statistics-public-v1'
            and record['manifest'] == proof['manifest'] == ctx['coverage']
            and type(record['ordinal']) is type(proof['ordinal']) is int
            and record['ordinal'] == proof['ordinal'] == ordinal
            and proof['record'] == record_pin and proof['independentQR'] is True,
            'Unclosed or mismatched original within record/proof')
    inputs = read_pin(record['input'])
    case = design.coverage_case(ordinal)
    require(inputs['case'] == case, 'Wrong original case or seed')
    actual, generation = read_pin(inputs['request']), read_pin(inputs['generation'])
    commit = read_pin(inputs['comparatorCommit'])
    binding = read_pin(commit['input'])
    spec = read_pin(binding['spec'])
    require(commit['manifest'] == ctx['upstream']['upstream'] and commit['ordinal'] == ordinal
            and commit['sourceClosure'] is True and binding['ordinal'] == ordinal
            and binding['mode'] == 'fresh_within_validation'
            and spec['manifest'] == commit['manifest'] and spec['ordinal'] == ordinal
            and spec['case'] == generation['case'] == case
            and spec['request'] == inputs['request'] and spec['generation'] == inputs['generation'],
            'Comparator commitment belongs to another saved input')
    generation_proof = read_pin(commit['generationProof'])
    require(generation_proof == dict(spec=binding['spec'], manifest=commit['manifest'],
        request=inputs['request'], generation=inputs['generation'], independentReplay=True),
        'Changed original input-law proof')
    population = dict(spec)
    for field in ('retentionMask', 'recruitedCounts'):
        require(field not in population or population[field] == generation[field],
                'Conflicting whole-person retention metadata')
        population.setdefault(field, generation[field])
    counts = validate_population(actual, population, generation)
    require(counts == inputs['populationCounts'] == record['within']['design']['actualK']
            and record['within']['methodID'] == design.WITHIN_METHOD
            and record['within']['assumptions']['responseDependentGuardConditioning'] is False
            and record['evaluation']['usedByEstimator'] is False
            and record['event']['ordinal'] == ordinal and record['event']['caseIndex'] == case['caseIndex']
            and record['event']['caseID'] == case['caseID'] and record['event']['repID'] == case['repID'],
            'Changed original projection population, method or truth role')
    reader.compare(record['event'], proof['event'], 'saved within event/proof')
    data = dict(actual=actual, counts=counts, case=case, ordinal=ordinal,
        within=dict(result=record['within'], truthEvaluation=record['evaluation'],
                    conditionedOnOriginalGuard=False), accepted=record['event'])
    ancestry = dict(record=record_pin, proof=proof_pin, input=record['input'],
                    request=inputs['request'], generation=inputs['generation'],
                    comparatorCommit=inputs['comparatorCommit'])
    return data, ancestry


def event(ordinal, data, analysis):
    return dict(ordinal=ordinal, caseIndex=data['case']['caseIndex'],
        caseID=data['case']['caseID'], repID=data['case']['repID'], analysis=analysis,
        endpointUnknown={key: value['endpointUnresolved']
                         for key, value in data['within']['truthEvaluation']['references'].items()})


def produce_one(ctx, ordinal):
    from . import common_t as kernel
    directory = ctx['output'] / 'units' / f'{ordinal:05d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'unit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data, ancestry = coverage_member(ctx, ordinal)
        path = directory / 'record.json'
        if path.exists():
            pin, record = admit_result(ctx, ordinal, ancestry)
            if (directory / 'checked.json').exists():
                admit_proof(ctx, ordinal, pin, record)
            return dict(ordinal=ordinal, reused=True, newCandidateFamilies=0)
        require(not (directory / 'started.json').exists(), 'Incomplete comparison attempt; inspect before retrying')
        io.numerical_runtime()
        io.write(directory / 'started.json', dict(manifest=ctx['manifestPin'], ancestry=ancestry))
        calculated = kernel.calculate(data)
        certificate, analysis = calculated['certificate'], calculated['analysis']
        require(coverage_member(ctx, ordinal)[1] == ancestry, 'Upstream input changed during calculation')
        io.write(path, dict(schema=SCHEMA, manifest=ctx['manifestPin'], ordinal=ordinal,
            ancestry=ancestry, designCertificate=certificate, analysis=analysis,
            event=event(ordinal, data, analysis), linearIntervalsUnchanged=True,
            newInputs=0, newFits=0, newWithin=0, newQR=0))
        return dict(ordinal=ordinal, reused=False, newCandidateFamilies=1)


def admit_result(ctx, ordinal, ancestry):
    path = ctx['output'] / 'units' / f'{ordinal:05d}' / 'record.json'
    pin = io.pin(path)
    value = read_pin(pin)
    require(value['schema'] == SCHEMA and value['manifest'] == ctx['manifestPin']
            and value['ordinal'] == ordinal and value['ancestry'] == ancestry
            and value['linearIntervalsUnchanged'] is True
            and all(value[key] == 0 for key in ('newInputs', 'newFits', 'newWithin', 'newQR')),
            'Changed saved comparison identity')
    original = read_pin(ancestry['record'])
    canonical = design.coverage_case(ordinal)
    require(original['ordinal'] == ordinal and original['manifest'] == ctx['coverage'],
            'Changed comparison ancestry')
    expected = event(ordinal, dict(case=canonical,
        within=dict(truthEvaluation=original['evaluation'])), value['analysis'])
    require(value['event'] == expected, 'Stored event differs from its own ordinal, analysis or endpoints')
    return pin, value


def admit_proof(ctx, ordinal, record_pin, record):
    from . import independent_common_t as reader
    pin = io.pin(ctx['output'] / 'units' / f'{ordinal:05d}' / 'checked.json')
    proof = read_pin(pin)
    require(proof['schema'] == SCHEMA and proof['record'] == record_pin
            and proof['manifest'] == ctx['manifestPin'] and type(proof['ordinal']) is int
            and proof['ordinal'] == ordinal and proof['passed'] is True
            and all(proof[key] == 0 for key in ('newInputs', 'newFits', 'newWithin', 'newQR'))
            and proof['designCertificate'] == {k: v for k, v in record['designCertificate'].items() if k != 'formula'},
            'Changed independent comparison identity or exact certificate')
    reader.compare(record['event'], proof['event'], 'stored independent scalar event')
    reader.compare(record['analysis'], proof['event']['analysis'], 'stored independent analysis')
    return pin, proof


def check_one(ctx, ordinal):
    from . import independent_common_t as reader
    directory = ctx['output'] / 'units' / f'{ordinal:05d}'
    require(directory.is_dir(), 'Produce the selected comparison before checking it')
    with (directory / 'unit.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data, ancestry = coverage_member(ctx, ordinal)
        pin, record = admit_result(ctx, ordinal, ancestry)
        path = directory / 'checked.json'
        if path.exists():
            admit_proof(ctx, ordinal, pin, record)
            return dict(ordinal=ordinal, reused=True, newIndependentFamilies=0)
        require(not (directory / 'check_started.json').exists(), 'Incomplete independent attempt; inspect before retrying')
        io.numerical_runtime()
        io.write(directory / 'check_started.json', dict(record=pin, manifest=ctx['manifestPin']))
        calculated = reader.calculate(data)
        certificate, analysis = calculated['certificate'], calculated['analysis']
        require(certificate == {k: v for k, v in record['designCertificate'].items() if k != 'formula'},
                'Independent exact rational certificate differs')
        reader.compare(record['analysis'], analysis, 'independent comparison')
        scalar = event(ordinal, data, analysis)
        reader.compare(record['event'], scalar, 'independent scalar event')
        require(coverage_member(ctx, ordinal)[1] == ancestry, 'Upstream input changed during independent check')
        io.write(path, dict(schema=SCHEMA, manifest=ctx['manifestPin'], ordinal=ordinal,
            record=pin, event=scalar, designCertificate=certificate, passed=True,
            newInputs=0, newFits=0, newWithin=0, newQR=0))
        return dict(ordinal=ordinal, reused=False, newIndependentFamilies=1)


def reduce_events(rows, independent_rows=None, *, full=False):
    from . import common_t as producer, independent_common_t as reader
    rows = list(rows)
    independent_rows = rows if independent_rows is None else list(independent_rows)
    ids, other = [r['ordinal'] for r in rows], [r['ordinal'] for r in independent_rows]
    require(ids == sorted(set(ids)) and ids == other
            and all(type(i) is int and 0 <= i < 12129 for i in ids),
            'Duplicate, foreign, unordered or unmatched comparison roster')
    require(all(r['caseIndex'] == (1 if r['ordinal'] < 10000 else 2)
                for r in rows + independent_rows), 'Wrong saved cohort')
    complete = ids == list(range(12129))
    require(not full or complete, 'Full reduction requires exactly12129 original members')
    reported, checked = producer.report(rows), reader.report(independent_rows)
    reader.compare(reported, checked, 'independent paired reduction')
    reader.check_no_original_losses(checked)
    return dict(schema=SCHEMA, fullRoster=complete, present=len(rows), expectedTotal=12129,
        ordinals=ids, report=reported, independentReport=checked, independentReduction=True,
        newInputs=0, newFits=0, newWithin=0, newQR=0, newCandidateFamilies=0)


def reduce_public(ctx, full=False):
    produced, checked, pins = [], [], []
    for directory in sorted((ctx['output'] / 'units').glob('*')):
        require(directory.is_dir() and len(directory.name) == 5 and directory.name.isdigit(),
                'Foreign comparison directory')
        ordinal = int(directory.name)
        # Numerical checking already bound the raw request. Reduction consumes
        # immutable scalar records, not every original X/y array again.
        saved = io.read(directory / 'record.json')
        ancestry = saved['ancestry']
        original = read_pin(ancestry['record'])
        previous_proof = read_pin(ancestry['proof'])
        require(original['ordinal'] == previous_proof['ordinal'] == ordinal
                and original['manifest'] == previous_proof['manifest'] == ctx['coverage']
                and previous_proof['record'] == ancestry['record']
                and previous_proof['independentQR'] is True, 'Changed checked upstream scalar identity')
        record_pin, record = admit_result(ctx, ordinal, ancestry)
        proof_pin, proof = admit_proof(ctx, ordinal, record_pin, record)
        produced.append(record['event'])
        checked.append(proof['event'])
        pins.extend((record_pin, proof_pin))
    result = reduce_events(produced, checked, full=full)
    result.update(manifest=ctx['manifestPin'], scalarSources=pins,
        evidence='new_saved_input_comparison_with_independent_numerical_checks')
    name = 'aggregate_full.json' if result['fullRoster'] else f'aggregate_partial_{len(produced)}.json'
    context(ctx['output'])
    return io.write(ctx['output'] / name, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'prepare', 'produce', 'check', 'reduce'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coverage', type=Path)
    parser.add_argument('--coverage-source-root', type=Path)
    parser.add_argument('--ids')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--events', type=Path, help='Supplied scalar arithmetic only; not archive authentication')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'plan':
        return plan()
    if args.stage == 'prepare':
        require(args.coverage is not None, 'Preparation requires checked --coverage')
        return prepare(args.coverage, args.output, args.coverage_source_root)
    require(args.execute, 'Production/check/reduction requires --execute')
    require(args.coverage is None and args.coverage_source_root is None, 'Upstream inputs are fixed at preparation')
    if args.events is not None:
        require(args.stage == 'reduce' and args.ids is None and not args.all, 'Events are reduction only')
        output = io.validate_output(args.output, args.events)
        with args.events.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        result = reduce_events(rows, full=args.full)
        result.update(input=io.pin(args.events), evidence='supplied_scalar_arithmetic_only_not_source_authentication')
        return io.write(output / 'scalar_comparison.json', result)
    ctx = context(args.output)
    if args.stage == 'reduce':
        require(args.ids is None and not args.all, 'Reduction uses the committed roster')
        return reduce_public(ctx, args.full)
    require(not args.full, 'Use --all to select the full production/check roster')
    ids = selected([int(i) for i in args.ids.split(',')] if args.ids is not None else None, args.all)
    function = produce_one if args.stage == 'produce' else check_one
    results = [function(ctx, i) for i in ids]
    context(ctx['output'])
    return dict(results=results, newInputs=0, newFits=0, newWithin=0, newQR=0)
