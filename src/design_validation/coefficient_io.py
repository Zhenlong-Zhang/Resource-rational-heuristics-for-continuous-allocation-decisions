"""Immutable portable checkpoints for the coefficient reproduction family."""
from pathlib import Path

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S


def source_identity():
    here = Path(__file__).resolve().parent
    names = ('coefficient_io.py', 'coefficient_diagnostics.py', 'coefficient_reference.py',
             'coefficient_reproduction.py', 'coefficient_original.py', 'c5_moments.py', 'c5_reference.py',
             'c5_reproduction.py', 'c5_streams.py')
    return {**S.implementation_identity(),
            **{name: S.file_sha(here / name) for name in names}}


def stage(path, binding, compute, *, readonly=False):
    """Reuse only an exact, self-consistent stage; never repair changed bytes."""
    path = Path(path)
    if path.exists():
        I.need(not path.is_symlink(), 'Checkpoint symlink refused')
        value = I.read(path)
        I.need(type(value) is dict and set(value) == {'schema', 'binding', 'payload', 'contentSHA256'},
               'Exact checkpoint envelope required')
        body = {key: value[key] for key in ('schema', 'binding', 'payload')}
        I.need(value['schema'] == 'coefficient-checkpoint-v1' and value['binding'] == binding
               and value['contentSHA256'] == I.digest(body), 'Foreign or changed checkpoint')
        return value['payload'], False
    I.need(not readonly, 'Missing checkpoint: ' + str(path))
    payload = compute()
    body = dict(schema='coefficient-checkpoint-v1', binding=binding, payload=payload)
    I.write(path, {**body, 'contentSHA256': I.digest(body)})
    return payload, True


def separate(source, output):
    source, output = Path(source).resolve(), S.safe_output(output).resolve()
    I.need(not source.is_relative_to(output) and not output.is_relative_to(source),
           'Input and output trees must be separate')
    return source, output


def inventory(output, expected):
    """Reject foreign files instead of silently ignoring a stale completed unit."""
    output = Path(output)
    for folder in ('laws', 'independent_laws', 'designs', 'independent_designs', 'high', 'independent_high'):
        base = output / folder
        actual = {str(p.relative_to(output)) for p in base.rglob('*') if p.is_file()} if base.exists() else set()
        I.need(actual <= expected, 'Foreign stage members in ' + folder)
