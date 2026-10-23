"""Exact personal-time inputs, explicit selections and immutable portable files."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from . import personal_time_policy as A
from .cost_aware_inputs import read, parse, source_root

CONDITIONS = ('C1', 'C2', 'C3', 'C4', 'C5')
FIELDS = ('conditionId', 'scheduleVariant', 'mainIndex')
SCIENTIFIC_FIELDS = (*FIELDS, 'seed', 'definition', 'logicalToDisplayed', 'initialLogs', 'common')
ORIGINAL = {(c, v, m) for c in CONDITIONS for v in range(9) for m in range(1, 11)}
# Original scientific input identities, not generated result receipts.
INPUT_SHA256 = {
    'base': '9bec46f78e28fb2b289187fc15e90d1f7156c1460f1d49b0bf30e8cfd62ec85e',
    'instance': '45021145a06dd485543317d8415926098b472cdd9c402e8759b618c8646b25c9',
    'design': 'a2fed8bfc9fee3edcf7a6eb68ccbd76979396ee3a65a61faba701ad0a8988d54',
    'root_identity': 'd0a20cd482e7429bdea88515587cab33835cfb3f1e7ffdfd5739bee1b001d30d',
}
need = A.need


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def numeric_bits(value):
    if type(value) in (int, float):
        need(math.isfinite(value), 'Finite binary64 input')
        return {'number_binary64': struct.pack('>d', float(value)).hex()}
    if type(value) is list:
        return [numeric_bits(x) for x in value]
    if type(value) is dict:
        return {k: numeric_bits(v) for k, v in value.items()}
    return value


def scientific_digest(root):
    return hashlib.sha256(json.dumps(numeric_bits({k: root[k] for k in SCIENTIFIC_FIELDS}),
                                    sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write_bytes(path, data):
    """Publish without replacing anything; identical concurrent/resumed writes are OK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.is_symlink(), 'Output symlink refused: ' + str(path))
    if path.exists():
        need(path.is_file() and path.read_bytes() == data, 'Conflicting resumed output: ' + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            need(not path.is_symlink() and path.is_file() and path.read_bytes() == data,
                 'Conflicting resumed output: ' + str(path))
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write(path, value):
    write_bytes(path, encoded(value) + b'\n')


def coordinates(root):
    return tuple(root[k] for k in FIELDS)


def selection(rows):
    need(type(rows) is list and bool(rows), 'Explicit nonempty selection required')
    keys = []
    for row in rows:
        need(type(row) is dict and set(row) == set(FIELDS), 'Exact original root coordinates required')
        need(type(row['conditionId']) is str and type(row['scheduleVariant']) is int
             and type(row['mainIndex']) is int, 'Strict coordinate types')
        key = coordinates(row)
        need(key in ORIGINAL, 'Root outside original finite support')
        keys.append(key)
    need(len(keys) == len(set(keys)), 'Duplicate requested coordinates are rejected')
    return [dict(zip(FIELDS, key)) for key in sorted(keys)]


def select_roots(*, full=False, only=None, selected=None):
    need(sum((bool(full), only is not None, selected is not None)) == 1,
         'Choose exactly one of full, only, or a selection file; no implicit computation')
    if full:
        return selection([dict(zip(FIELDS, key)) for key in sorted(ORIGINAL)])
    if selected is not None:
        return selection(selected)
    rows = []
    for item in only.split(','):
        parts = item.split(':')
        need(len(parts) == 3, 'Use C3:0:1,C5:0:1 coordinates')
        rows.append(dict(zip(FIELDS, (parts[0], int(parts[1]), int(parts[2])))))
    return selection(rows)


def load_context(path):
    path = Path(path)
    recipe = read(path)
    need(recipe['schema'] == 'personal-time-finite-reference-v1', 'Personal-time recipe schema')
    inputs = {}
    for key, expected in INPUT_SHA256.items():
        filename = path.parent / recipe[key]
        need(hashlib.sha256(filename.read_bytes()).hexdigest() == expected,
             'Changed original scientific input: ' + key)
        inputs[key] = read(filename)
    return {'recipe': recipe, **inputs}


def reference_runtime():
    need(sys.version_info[:2] == (3, 10), 'Reference Python is 3.10')
    need(A.np.__version__ == '1.23.5', 'Reference NumPy is 1.23.5')


def implementation_identity():
    here = Path(__file__).resolve().parent
    names = ('personal_time_policy.py', 'personal_time_reference.py', 'personal_time_inputs.py',
             'personal_time_reproduction.py', 'personal_time_source.cjs', 'personal_time_examples.py',
             'cost_aware_inputs.py', 'cost_aware_source.cjs', 'bounded_task.py', 'budget_horizon.py',
             'terminal_time.py', '../tutoring_analysis/native_engine.cjs')
    return digest({name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names})


def materialize(base, instance):
    A.exact(instance, {'candidateId','law','actionCosts','valuation','insufficientBudgetMode'}, 'native instance')
    result = copy.deepcopy(base)
    result['candidateId'] = instance['candidateId']
    result['task']['allocation_denominator'] = 1000
    result['actionCosts'] = copy.deepcopy(instance['actionCosts'])
    result['valuation'] = copy.deepcopy(instance['valuation'])
    result['insufficientBudgetMode'] = instance['insufficientBudgetMode']
    law = instance['law']
    A.need(isinstance(law,dict), 'declared native law')
    if law.get('kind') == 'uniform_native_grid':
        A.exact(law, {'kind','denominator','counts','overheadSeconds','voluntaryEnabled','streamNamespace'}, 'uniform native law')
        probabilities = [1./1001]*1001
    else:
        A.exact(law, {'kind','denominator','probabilities','counts','overheadSeconds','voluntaryEnabled','streamNamespace'}, 'categorical native law')
        A.need(law['kind'] == 'categorical_native_grid', 'explicit native law kind')
        probabilities = copy.deepcopy(law['probabilities'])
    A.need(type(law['denominator']) is int and law['denominator'] == 1000, 'actual native D1000')
    result['random'] = {k:copy.deepcopy(law[k]) for k in ('counts','overheadSeconds','voluntaryEnabled','streamNamespace')}
    result['random']['probabilities'] = probabilities
    A.validate(result)
    return result


def make_config(root, context):
    # Reuse the public exact TaskConfig constructor, including one side reversal.
    task = source_root(root, root['session'])['config']
    cfg = materialize(context['base'], context['instance'])
    cfg['task'] = task
    cfg['pathReports'] = []
    cfg['state'] = {'budget': task['total_millihours'], 'logs': copy.deepcopy(root['initialLogs'])}
    A.validate(cfg)
    return cfg


def validate_roots(data, selected, context):
    selected = selection(selected)
    need(data['schema'] == 'personal-time-original-roots-v1' and data['selection'] == selected,
         'Exact prepared source selection')
    need(data['nodeVersion'] == '22.14.0' and data['realizedPolicyHistoriesGenerated'] is False,
         'Reference source runtime and expected-root-only scope')
    expected = {(r['condition'], r['variant'], r['main']): r
                for r in context['root_identity']['coordinates']}
    need(set(expected) == ORIGINAL and len(context['root_identity']['coordinates']) == 450,
         'Complete original root identity roster')
    roots = data['roots']
    need(len(roots) == len(selected) and [coordinates(r) for r in roots] == [coordinates(r) for r in selected],
         'Exact ordered unique source roots')
    for root in roots:
        ref = expected[coordinates(root)]
        need(scientific_digest(root) == ref['scientificSHA256'], 'Original root binary64 identity')
        need(root['stateSnapshotSHA256'] == ref['snapshotSHA256'], 'Original root snapshot identity')
        need(root['prefixKind'] == 'fixed' and root['prefixLength'] == 0 and root['paid'] == [],
             'Only original unpaid roots')
    return roots


def binding(context, selected):
    return {'schema': 'personal-time-run-binding-v1', 'selection': selection(selected),
            'contextSHA256': digest(context), 'implementationSHA256': implementation_identity()}


def prepare(context, selected, output, node='node'):
    selected = selection(selected)
    output = Path(output)
    write(output / 'run.json', binding(context, selected))
    target = output / 'inputs' / 'roots.json'
    if target.exists():
        data = read(target)
        validate_roots(data, selected, context)
        return data
    request = {'selection': selected, 'identity': context['root_identity'], 'design': context['design'],
               'analysis': context['recipe']['analysis'], 'construction': context['recipe']['construction']}
    helper = Path(__file__).with_name('personal_time_source.cjs')
    # Native snapshots serialize design fields in their original insertion order.
    # Canonical sorting is for identity hashes, not this transport payload.
    payload = json.dumps(request, separators=(',', ':'), allow_nan=False)
    try:
        result = subprocess.run([node, str(helper)], input=payload,
                                text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError('Native source generation failed: ' + error.stderr.strip()) from error
    data = parse(result.stdout)
    validate_roots(data, selected, context)
    write(target, data)
    return data


def load_roots(context, selected, output):
    need(read(Path(output) / 'run.json') == binding(context, selected), 'Run/input/implementation binding mismatch')
    return validate_roots(read(Path(output) / 'inputs' / 'roots.json'), selected, context)
