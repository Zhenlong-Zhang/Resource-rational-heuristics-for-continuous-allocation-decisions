"""Portable production, independent receiving and fixed-design card reduction."""
from contextlib import contextmanager
import csv
import io
import os
from pathlib import Path

from threadpoolctl import threadpool_limits

from .. import bounded_task
from . import adaptive, adaptive_path, adaptive_reference, external_bounds, external_reference
from . import fixed, fixed_reference, inputs as I, summary as S


def selected_roots(roots, *, full=False, only=None):
    selected = I.I.select_roots(full=full, only=only)
    by_id = {tuple(r['id']): r for r in roots}
    keys = [I.I.coordinates(row) for row in selected]
    I.need(set(keys) <= by_id.keys(), 'Requested roots must be prepared first')
    return [by_id[key] for key in keys]


def name(root):
    S.coordinate(root['id'])
    return '-'.join(map(str, root['id'])) + '-r0.json'


@contextmanager
def lock(output, phase, root):
    directory = Path(output) / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (phase + '-' + name(root))
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(descriptor)
        yield
    finally:
        target.unlink()


def produce_root(root):
    """One named original replicate, never a replacement draw or repeated rollout."""
    return {'id': root['id'], 'replicate': 0, 'fixed': fixed.evaluate(root),
            'adaptive': adaptive.evaluate(root, 0, bounded_task, external_bounds, adaptive_path.path)}


def check_root(root, record):
    I.need(record['id'] == root['id'] and type(record['replicate']) is int and record['replicate'] == 0,
           'Producer root and replicate identity')
    return {'fixed': fixed_reference.verify(root, record['fixed']),
            'adaptive': adaptive_reference.verify(root, 0, record['adaptive'],
                                                  model=I.MODEL, reference=external_reference)}


def process(output, config, *, phase, full=False, only=None, node='node'):
    I.need(phase in ('produce', 'check'), 'Explicit production or independent check')
    output = Path(output)
    with threadpool_limits(limits=1):
        binding, roots = I.load(output, config, node=node if phase == 'check' else None)
        I.need(binding['runtime'] == I.runtime(), 'Recorded numerical build changed')
        selected = selected_roots(roots, full=full, only=only)
        run_sha = I.digest(binding)
        for root in selected:
            p_path = output / 'producer' / name(root)
            r_path = output / 'independent' / name(root)
            with lock(output, phase, root):
                common = {'bindingSHA256': run_sha, 'rootSHA256': I.digest(root)}
                if phase == 'produce':
                    if p_path.exists():
                        old = I.read(p_path)
                        I.need(all(old[k] == value for k, value in common.items())
                               and old['result']['id'] == root['id'] and old['result']['replicate'] == 0,
                               'Conflicting producer output')
                        continue
                    I.write(p_path, dict(common, result=produce_root(root)))
                else:
                    producer = I.read(p_path)
                    I.need(all(producer[k] == value for k, value in common.items()), 'Changed producer binding')
                    pin = I.file_sha(p_path)
                    if r_path.exists():
                        old = I.read(r_path)
                        I.need(all(old[k] == value for k, value in common.items())
                               and old['producerSHA256'] == pin and old['status'] == 'PASS',
                               'Conflicting independent output')
                        continue
                    checked = check_root(root, producer['result'])
                    I.need(I.file_sha(p_path) == pin, 'Producer changed during verification')
                    I.write(r_path, dict(common, producerSHA256=pin, status='PASS', check=checked))
    return {'phase': phase, 'selectedRoots': len(selected), 'output': str(output),
            'scientificScope': I.RECIPE['scope']}


def aggregate(output, config, *, full=False):
    output = Path(output)
    binding, roots = I.load(output, config)
    run_sha = I.digest(binding)
    records, pins = [], {}
    expected_files = {name(root) for root in roots}
    for folder in ('producer', 'independent'):
        I.need({p.name for p in (output / folder).glob('*.json')} <= expected_files,
               'Unexpected record outside the prepared root roster')
    for root in roots:
        p_path, r_path = output / 'producer' / name(root), output / 'independent' / name(root)
        if not p_path.exists() or not r_path.exists():
            continue
        p, r = I.read(p_path), I.read(r_path)
        I.need(p['bindingSHA256'] == r['bindingSHA256'] == run_sha
               and p['rootSHA256'] == r['rootSHA256'] == I.digest(root)
               and r['producerSHA256'] == I.file_sha(p_path) and r['status'] == 'PASS',
               'Source-bound independent receiving required')
        value = p['result']
        I.need(value['id'] == root['id'] and value['fixed']['id'] == root['id'], 'Checked result identity')
        row = S.adaptive_row(root, value['adaptive'], r['check']['adaptive'])
        records.append({'id': root['id'], 'replicate': value['replicate'],
                        'fixed': value['fixed']['rows'], 'adaptive': row})
        pins[str(p_path.relative_to(output))] = I.file_sha(p_path)
        pins[str(r_path.relative_to(output))] = I.file_sha(r_path)
    result = S.summarize(records, full=full)
    result['rowSourceAuthenticated'] = True
    result['bindingSHA256'] = run_sha
    # Partial checks have separate immutable names and cannot overwrite the full panel.
    suffix = 'full' if result['fullFiniteTarget'] else 'partial_' + I.digest(pins)[:16]
    destination = output / 'summary' / suffix
    I.write(destination / 'summary.json', result)
    I.write(destination / 'source_check.json', {'binding': binding, 'files': pins})
    stream = io.StringIO(newline='')
    fields = ('condition', 'card', 'observedRoots', 'missingRoots', 'pointGPA', 'lowerGPA',
              'upperGPA', 'meanQuestions', 'meanTutoringMinutes', 'intervalType')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for cell in result['cells']:
        is_adaptive = cell['card'] == S.ADAPTIVE
        bounds = cell['means']['GPAInterval'] if is_adaptive else cell['meanPayoffSupport']
        writer.writerow({**{k: cell[k] for k in fields[:4]},
                         'pointGPA': None if is_adaptive else cell['expectedCombinedGPA'],
                         'lowerGPA': bounds[0] if bounds else None, 'upperGPA': bounds[1] if bounds else None,
                         'meanQuestions': cell['means']['questions'] if is_adaptive else cell['meanQuestions'],
                         'meanTutoringMinutes': cell['means']['tutoringMinutes'] if is_adaptive else cell['meanTutoringMinutes'],
                         'intervalType': cell['intervalType']})
    I.I.write_bytes(destination / 'condition_cards.csv', stream.getvalue().encode())
    return {'summary': str(destination / 'summary.json'), 'observedRoots': len(records),
            'fullFiniteTarget': result['fullFiniteTarget']}
