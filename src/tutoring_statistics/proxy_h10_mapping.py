"""Six-price joint amount-slope REML bridge from authenticated public endpoints.

All six descriptive fits are retained. The separately selected midpoint law
uses its original stored decimals only after source-derived coefficient parity.
"""
import argparse
import csv
import io as text_io
import math
from pathlib import Path
import json
import shutil
import subprocess

from . import reproduction as io
from .within_design import ARMS, MIDPOINT_RATE, MIDPOINT_SLOPES, require

ROOT = Path(__file__).resolve().parent
FIELDS = ('rate', 'condition', 'variant', 'main', 'person', 'price', 'trial_id',
          'need_a', 'need_b', 'amount_a', 'budget_milli')
RATES = {'rho_0': 0., 'rho_0.00001': .00001, 'rho_0.0001': .0001,
         'rho_0.001': .001, MIDPOINT_RATE: .0031622776601683794, 'rho_0.01': .01}
LOGICAL_FRAME = 'logical_recipients'
ANALYSIS_FRAME = 'displayed_recipients'

# This is a declared public reproduction environment, not an assertion that
# the original descriptive mapping was executed in the synthetic-study runtime.
R_PROBE = r'''
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(jsonlite))
expected <- read_json(args[[1]],simplifyVector=FALSE)$statistical_reference
stopifnot(identical(as.character(getRversion()),expected$R))
packages <- list()
for (name in names(expected$R_packages)) {
  stopifnot(requireNamespace(name,quietly=TRUE))
  location <- normalizePath(find.package(name),mustWork=TRUE)
  version <- unname(read.dcf(file.path(location,"DESCRIPTION"),fields="Version")[1,1])
  stopifnot(identical(version,expected$R_packages[[name]]))
  files <- c(file.path(location,"DESCRIPTION"),list.files(file.path(location,"R"),full.names=TRUE))
  packages[[name]] <- list(version=version,files=as.list(files[file.exists(files)]))
}
dlls <- vapply(getLoadedDLLs(),function(x)x[["path"]],character(1))
# R's built-in base entry is a virtual DLL, not a pathname.
virtual <- names(dlls) == "base" & dlls == "base"
dlls <- normalizePath(c(dlls[!virtual & nzchar(dlls)],
  file.path(R.home("bin"),"exec","R"),file.path(R.home("lib"),"libR.so")),mustWork=TRUE)
cat(toJSON(list(R=as.character(getRversion()),Rhome=normalizePath(R.home()),
  packages=packages,dlls=as.list(dlls),external=as.list(extSoftVersion())),
  auto_unbox=TRUE,digits=NA,null="null"))
'''


def runtime(rscript):
    executable = shutil.which(str(rscript))
    require(executable is not None, 'Declared Rscript executable is unavailable')
    executable = str(Path(executable).resolve())
    config = ROOT / 'data/numerical_environment.json'
    expected = io.read(config)['statistical_reference']
    command = [executable, '--vanilla', '-e', R_PROBE, str(config)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    require(result.returncode == 0, 'Reference mapping R environment unavailable: '+result.stderr)
    value = json.loads(result.stdout)
    require(value['R'] == expected['R'] and set(value['packages']) == set(expected['R_packages']),
            'Changed declared mapping reference environment')
    files = {executable, str(config), *value['dlls']}
    for name, package in value['packages'].items():
        require(package['version'] == expected['R_packages'][name] and package['files'],
                'Changed loaded mapping package '+name)
        files.update(package['files'])
    return dict(executable=executable, actual=value, files=[io.pin(p) for p in sorted(files)],
        reference='public_statistical_reference_for_descriptive_reproduction',
        originalMappingRuntimeIdentityEstablished=False, vanilla=True)


def runtime_matches(saved, actual):
    require(saved == actual, 'Changed R executable, package/library bytes or runtime; do not refit')
    for pin in saved['files']:
        io.validate(pin)


def load_rows(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == FIELDS, 'Changed shared amount CSV schema')
        rows = list(reader)
    require(len(rows) == 2700, 'Complete six-price source roster required, not a prefix')
    seen, identities, people = set(), set(), {}
    for row in rows:
        rate, arm = row['rate'], row['condition']
        require(rate in RATES and arm in ('C1', 'C2', 'C3', 'C4', 'C5'), 'Foreign price or arm')
        for key in ('variant', 'main', 'budget_milli'):
            value = int(row[key])
            require(str(value) == row[key], 'Noncanonical integer '+key)
            row[key] = value
        v, m = row['variant'], row['main']
        require(0 <= v < 9 and 1 <= m <= 10 and row['budget_milli'] > 0, 'Wrong complete schedule')
        key = rate, arm, v, m
        identity = rate, row['trial_id']
        require(key not in seen and identity not in identities and row['trial_id'], 'Duplicate coordinate/endpoint ID')
        seen.add(key)
        identities.add(identity)
        group = (rate, row['person'])
        require(people.setdefault(group, (arm, v)) == (arm, v), 'Person crosses condition or variant')
        for key in ('price', 'need_a', 'need_b', 'amount_a'):
            row[key] = float(row[key])
            require(math.isfinite(row[key]), 'Nonfinite '+key)
        require(row['price'] == RATES[rate] and 0 <= row['amount_a'] <= row['budget_milli']/1000,
                'Changed price or invalid logical amount')
    require(len(people) == 270, 'Exactly nine whole-ten people per arm/price')
    require(seen == {(r, a, v, m) for r in RATES for a in ('C1', 'C2', 'C3', 'C4', 'C5')
                     for v in range(9) for m in range(1, 11)}, 'Missing scientific coordinate')
    return rows


def authenticate(rows, source):
    """Admit producer/independent endpoint identities without running policies."""
    return _authenticate_selected(rows, source, require_full=True)


def _authenticate_selected(rows, source, *, require_full):
    """Shared row-level checks; only the public full-roster entry admits a mapping."""
    from design_validation import personal_time_sensitivity as s
    from design_validation.personal_time_reproduction import compare
    source = Path(source).resolve()
    checkout = ROOT.parents[1]
    context = s.load_context(checkout / 'configs/personal_time_sensitivity.json')
    selection = io.read(source / 'run.json')['selection']
    roots, run, _ = s.load_roots(context, selection, source)
    check = io.read(source / 'independent_check.json')
    require(check['binding'] == run and check['completeIndependentReconstruction'] is True
            and check['verifiedTrials'] == len(rows) == len(roots)*len(RATES),
            'Genuine selected source check required')
    if require_full:
        require(check['fullFiniteTarget'] is True and check['verifiedTrials'] == 2700,
                'Complete genuine public source check required')
    table = {(r['rate'], r['condition'], r['variant'], r['main']): r for r in rows}
    require(len(table) == len(rows), 'Duplicate source row')
    pins, expected_paths = [], set()
    for root in roots:
        c, v, m = s.I.coordinates(root)
        for rate in RATES:
            path = s.record_path(source, rate, root)
            relative = str(path.relative_to(source))
            expected_paths.add(relative)
            pin = io.pin(path)
            require(check['producerInputs'][relative] == pin['sha256'], 'Changed checked source record')
            cfg = s.rate_config(context, root, rate)
            binding = s.record_binding(root, cfg, run, rate)
            record = io.read(path)
            s.validate_record(record, binding)
            other_path = s.record_path(source, rate, root, 'independent')
            other = io.read(other_path)
            require(other['binding'] == binding and other['producerSHA256'] == pin['sha256']
                    and other['absoluteTolerance'] == 1e-10, 'Changed independent endpoint linkage')
            compare(record['payload'], other['payload'], 'independent source endpoint')
            value = record['payload']
            ids = value['context']['identity']
            row = table[rate, c, v, m]
            require((ids['conditionId'], ids['variant'], ids['mainIndex'], ids['replicateId'])
                    == (c, v, m, 0) and row['trial_id'] == ids['trialId'], 'Changed source-qualified trial identity')
            means = dict(zip(root['logicalToDisplayed'], value['metrics']['initialMeans']))
            budget, j = value['terminal']['budget'], value['terminal']['intendedIndex']
            logical_j = j if root['logicalToDisplayed'][0] == 'A' else 1000-j
            require(row['need_a'] == means['A'] and row['need_b'] == means['B']
                    and row['budget_milli'] == budget and row['amount_a'] == budget*logical_j/1000000,
                    'CSV differs from original logical needs, budget or allocation')
            pins.extend((pin, io.pin(other_path)))
    require(set(check['producerInputs']) == expected_paths and len(expected_paths) == len(table),
            'Source endpoint roster differs')
    return [io.pin(source / 'run.json'), io.pin(source / 'independent_check.json'), *pins]


def ordered_csv(rows):
    """Preserve the original string-main ordering used for the descriptive fit."""
    normalized = [dict(r, person=r['condition']+'_v'+str(r['variant'])) for r in rows]
    normalized.sort(key=lambda r: (r['rate'], r['condition'], r['person'], str(r['main'])))
    stream = text_io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(normalized)
    return stream.getvalue()


def displayed_panel(panel, orientations):
    """Convert authenticated logical exports to the historical displayed frame."""
    require(panel['coordinateFrame'] == LOGICAL_FRAME, 'Conversion requires logical recipient frame')
    result = []
    for row in panel['rows']:
        key = row['condition'], row['variant'], row['main']
        require(key in orientations and orientations[key] in (['A', 'B'], ['B', 'A']),
                'Missing or invalid source orientation')
        budget, amount = row['budget_milli'], row['amount_a']
        require(type(budget) is int and budget > 0 and math.isfinite(amount), 'Invalid native allocation')
        j = round(amount*1000000/budget)
        require(0 <= j <= 1000 and amount == budget*j/1000000,
                'Amount must round-trip exactly through the native integer index')
        value = dict(row)
        if orientations[key] == ['B', 'A']:
            value['need_a'], value['need_b'] = row['need_b'], row['need_a']
            j = 1000-j
        value['amount_a'] = budget*j/1000000
        result.append(value)
    return dict(coordinateFrame=ANALYSIS_FRAME, rows=result)


def source_panel(rows, source):
    from design_validation import personal_time_sensitivity as s
    context = s.load_context(ROOT.parents[1]/'configs/personal_time_sensitivity.json')
    selection = io.read(Path(source)/'run.json')['selection']
    roots, _, _ = s.load_roots(context, selection, source)
    return displayed_panel(dict(coordinateFrame=LOGICAL_FRAME, rows=rows),
                           {s.I.coordinates(r): r['logicalToDisplayed'] for r in roots})


def library(rows):
    result = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for variant in range(9):
            group = sorted((r for r in rows if r['rate'] == MIDPOINT_RATE
                            and r['condition'] == arm and r['variant'] == variant), key=lambda r: r['main'])
            require([r['main'] for r in group] == list(range(1, 11)), 'Wrong midpoint whole-ten order')
            result[arm].append([dict(main=r['main'], need_A=r['need_a'], need_B=r['need_b'],
                trialID=r['trial_id']) for r in group])
    return result


def fit_rows(rows, output, rscript):
    """Descriptive numerical bridge; input authentication is a separate obligation."""
    output = io.validate_output(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = runtime(rscript)
    io.write(output / 'runtime.json', environment)
    path = output / 'six_prices.csv'
    content = ordered_csv(rows)
    if path.exists():
        require(path.read_text() == content, 'Changed prepared descriptive rows')
    else:
        with path.open('x', newline='') as stream:
            stream.write(content)
    fitted = output / 'six_fits.csv'
    require(not fitted.exists() and not (output / 'fit.stderr').exists(), 'Preserve prior fit attempt; do not refit')
    command = [environment['executable'], '--vanilla', str(ROOT / 'proxy_h10_fit.R'), str(path)]
    with fitted.open('x') as stdout, (output / 'fit.stderr').open('x') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    require(result.returncode == 0, 'Descriptive REML failed; preserve its output')
    runtime_matches(environment, runtime(rscript))
    return command


def check_fits(output, rscript):
    output = Path(output).resolve()
    environment = io.read(output / 'runtime.json')
    runtime_matches(environment, runtime(rscript))
    command = [environment['executable'], '--vanilla', str(ROOT / 'independent_proxy_h10_fit.R'),
               str(output / 'six_prices.csv'), str(output / 'six_fits.csv')]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    require(result.returncode == 0, 'Independent profile REML failed: '+result.stderr)
    runtime_matches(environment, runtime(rscript))
    with (output / 'six_fits.csv').open(newline='') as stream:
        fits = list(csv.DictReader(stream))
    require(len(fits) == 6 and {r['rate'] for r in fits} == set(RATES), 'All six fits required')
    midpoint = next(r for r in fits if r['rate'] == MIDPOINT_RATE)
    for arm in ARMS:
        for need in ('A', 'B'):
            value = float(midpoint[need+'_'+arm])
            require(math.isfinite(value) and abs(value-MIDPOINT_SLOPES[arm][need]) < 1e-7,
                    'Changed midpoint source-derived slope; frozen scenario not admitted')
    return dict(command=command, stdout=result.stdout, stderr=result.stderr,
                runtime=io.pin(output / 'runtime.json'),
                allSixIndependentFits=True, midpointFrozenDecimalParity=True,
                coefficientTolerance=1e-7, substitutedScenarioCoefficients=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('produce', 'check'))
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Complete checked public six-price policy output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rscript', default='Rscript')
    args = parser.parse_args(argv)
    output = io.validate_output(args.output, args.source)
    require(not args.rows.resolve().is_relative_to(output), 'Source CSV must be outside mapping output')
    rows = load_rows(args.rows)
    source_pins = authenticate(rows, args.source)
    source = dict(rows=io.pin(args.rows), inputs=source_pins, statisticalSources=io.inventory()['files'],
                  inputCoordinateFrame=LOGICAL_FRAME, analysisCoordinateFrame=ANALYSIS_FRAME)
    marker = output / 'mapping.json'
    if marker.exists():
        value = io.read(marker)
        require(value['source'] == source, 'Changed completed mapping source')
        for pin in value['artifacts']:
            io.validate(pin)
        runtime_matches(io.read(output / 'runtime.json'), runtime(args.rscript))
        return dict(reused=True, newFits=0, mapping=io.pin(marker))
    prepared = output / 'prepared.json'
    if args.stage == 'produce':
        require(not prepared.exists(), 'Prepared mapping exists; use check, not another fit')
        command = fit_rows(source_panel(rows, args.source)['rows'], output, args.rscript)
        io.write(prepared, dict(source=source, command=command,
            fits=io.pin(output / 'six_fits.csv'), rows=io.pin(output / 'six_prices.csv'),
            runtime=io.pin(output / 'runtime.json')))
        return dict(newFits=6, prepared=io.pin(prepared))
    value = io.read(prepared)
    require(value['source'] == source, 'Changed mapping preparation source')
    io.validate(value['fits'])
    io.validate(value['rows'])
    io.validate(value['runtime'])
    proof = check_fits(output, args.rscript)
    library_pin = io.write(output / 'midpoint_library.json', library(source_panel(rows, args.source)['rows']))
    proof_pin = io.write(output / 'independent.json', proof)
    for pin in source_pins:
        io.validate(pin)
    return io.write(marker, dict(source=source, prepared=io.pin(prepared),
        artifacts=[value['fits'], value['rows'], value['runtime'], library_pin, proof_pin],
        library=library_pin, checked=True, fullSixPrices=True,
        coordinateFrame=ANALYSIS_FRAME,
        scientificScope='finite_descriptive_proxy_not_human_power',
        selectedMidpointSlopes=MIDPOINT_SLOPES, coefficientTolerance=1e-7))
