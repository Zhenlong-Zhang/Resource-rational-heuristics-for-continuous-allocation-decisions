"""Unpaid native covariates for the fixed synthetic midpoint scenario.

This constructs initial beliefs, not policy trajectories or new fitted slopes.
The separate six-price mapping checks the origin of the frozen scenario.
"""
import importlib
import math
from pathlib import Path

from . import reproduction as io
from .within_design import ARMS, MIDPOINT_RATE, MIDPOINT_SLOPES, require

SCHEMA = 'frozen_scenario_native_library'
FRAME = 'displayed_recipients'


def module(name):
    prefix = __package__.rsplit('.', 1)[0]+'.' if '.' in __package__ else ''
    return importlib.import_module(prefix+'design_validation.'+name)


def source():
    s = module('personal_time_sensitivity')
    config = Path(__file__).resolve().parents[2] / 'configs/personal_time_sensitivity.json'
    return s, s.load_context(config)


def selected():
    return [dict(conditionId=a, scheduleVariant=v, mainIndex=m)
            for a in ARMS for v in range(9) for m in range(1, 11)]


def library(roots, context):
    s = module('personal_time_sensitivity')
    ref = module('personal_time_sensitivity_reference')
    require([s.I.coordinates(r) for r in roots] ==
            [(a, v, m) for a in ARMS for v in range(9) for m in range(1, 11)],
            'Exact ordered 270 initial covariates required')
    result = {a: [[] for _ in range(9)] for a in ARMS}
    for root in roots:
        a, v, m = s.I.coordinates(root)
        cfg = s.rate_config(context, root, MIDPOINT_RATE)
        args = cfg, s.RECIPE['seed'], s.RECIPE['populationId'], a, v, 0, m
        identity = s.P.stable_identity(*args)
        require(identity == ref.stable_identity(*args), 'Independent native identity mismatch')
        means = [s.P.G.finite_mean(logs) for logs in root['initialLogs']]
        for logs, mean in zip(root['initialLogs'], means):
            weights = [math.exp(x-max(logs)) for x in logs]
            other = sum((i+1)*w for i, w in enumerate(weights))/sum(weights)
            require(math.isfinite(mean) and abs(mean-other) <= 1e-10,
                    'Independent initial mean mismatch')
        require(sorted(root['logicalToDisplayed']) == ['A', 'B'], 'Invalid recipient orientation')
        # The frozen synthetic scenario uses the historical table's display order.
        result[a][v].append(dict(main=m, need_A=means[0], need_B=means[1],
                                 trialID=identity['trialId']))
    return result


def prepare(output, node):
    io.numerical_runtime()
    output = Path(output)
    marker = output / 'native_library.json'
    if marker.exists():
        return admit(marker)
    s, context = source()
    s.I.prepare(context['reference'], selected(), output / 'native', node=node)
    roots = s.I.load_roots(context['reference'], selected(), output / 'native')
    values = library(roots, context)
    payload = io.write(output / 'midpoint_library.json', values)
    io.write(marker, dict(schema=SCHEMA, coordinateFrame=FRAME, library=payload,
        roots=io.pin(output / 'native/inputs/roots.json'), run=io.pin(output / 'native/run.json'),
        sourceIdentity=s.implementation_identity(), contextIdentity=s.I.digest(context),
        runtime=s.runtime_identity(), slopes=MIDPOINT_SLOPES,
        initialRows=270, policyTrajectoriesGenerated=False, newlyFittedSlopes=False))
    return admit(marker)


def admit(marker):
    io.numerical_runtime()
    marker = Path(marker)
    value = io.read(marker)
    s, context = source()
    require(value['schema'] == SCHEMA and value['coordinateFrame'] == FRAME
            and value['slopes'] == MIDPOINT_SLOPES
            and value['sourceIdentity'] == s.implementation_identity()
            and value['contextIdentity'] == s.I.digest(context)
            and value['runtime'] == s.runtime_identity()
            and value['initialRows'] == 270
            and value['policyTrajectoriesGenerated'] is False
            and value['newlyFittedSlopes'] is False, 'Changed frozen native scenario')
    for name in ('roots', 'run', 'library'):
        io.validate(value[name])
    require(Path(value['roots']['path']).resolve() == (marker.parent/'native/inputs/roots.json').resolve()
            and Path(value['run']['path']).resolve() == (marker.parent/'native/run.json').resolve(),
            'Foreign native source')
    roots = s.I.load_roots(context['reference'], selected(), marker.parent/'native')
    require(library(roots, context) == io.read(value['library']['path']), 'Changed ordered native needs')
    return io.pin(marker), value['library']
