"""Original root identities, displayed guesses and portable numerical bindings."""
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess

import numpy as np

from .. import personal_time_inputs as I
from ..cost_aware_inputs import source_root

need = I.need
read = I.read
write = I.write
digest = I.digest
ORIGINAL = I.ORIGINAL
MODEL = {
    'internal': 'current_runtime_frozen_source_binary64_utility_strict_float_comparison',
    'external': 'normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
    'history': 'owned_ordered_cursor_and_exact_integer_budget',
    'historyReuse': False,
    'recipientTies': 'physical_symmetric_keyed_midpoint',
    'rootWeights': 'one_ninetieth_within_each_of_five_conditions',
    'terminalTies': 'retain_unresolved_native_set',
}
RECIPE = {
    'schema': 'fixed-design-strategy-cards-v1', 'source_recipe': 'personal_time_reference.json',
    'conditions': ['C1', 'C2', 'C3', 'C4', 'C5'], 'variants': list(range(9)),
    'mains': list(range(1, 11)), 'replicate': 0, 'question_limit': 'budget_only',
    'display_version': 'r9-finite95-channel80-decimal-v1',
    'rng': 'sha256-first64-big-endian-midpoint-v1', 'stream_domain': 'R13-one-more-v1',
    'runtime': {'python': '3.11.5', 'numpy': '1.24.3', 'node': '22.14.0', 'numerical_threads': 1},
    'gpa_comparison_tolerance': 2e-11, 'model': MODEL,
    'scope': 'fixed_450_roots_one_adaptive_path_each_descriptive_not_human_targets',
}
ORDER_PROOF = {
    'orders': [[0, 1], [1, 0]], 'weights': ['1/2', '1/2'],
    'law': 'independent_recipient_latents_and_recipient_local_reports',
    'normalization': 'each_latent_row_divided_by_sum_over_reports_1_to_100',
    'posterior': 'one_recipient_local_update_each',
    'terminal': 'ordered_terminal_visible_means_and_remaining_budget_only',
    'questionCharges': 'same_cost_twice_or_skip_both',
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    here = Path(__file__).resolve().parent
    files = sorted([*here.glob('*.py'), *here.glob('*.cjs')])
    return {'cards': {p.name: file_sha(p) for p in files},
            'original_inputs': I.implementation_identity()}


def runtime():
    need(platform.python_version() == RECIPE['runtime']['python'] and np.__version__ == '1.24.3',
         'Use the separate Python 3.11.5 / NumPy 1.24.3 card environment')
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    need(all(p['num_threads'] == 1 for p in pools), 'Use one numerical thread per card process')
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'libraries': [{k: p.get(k) for k in ('internal_api', 'version', 'architecture', 'num_threads')}
                          | {'sha256': file_sha(p['filepath'])} for p in pools],
            'scope': 'recorded_build_not_arbitrary_hardware_bitwise_guarantee'}


def context(config):
    config = Path(config)
    recipe = read(config)
    need(recipe == RECIPE, 'Frozen six-card numerical recipe required')
    return I.load_context(config.parent / recipe['source_recipe'])


def root_adapter(root):
    """The constructor already orders logs by displayed side; reverse only task priors."""
    task = source_root(root, root['session'])['config']
    need(root['definition']['displayVersion'] == RECIPE['display_version'], 'Original display version')
    ordered = root['logicalToDisplayed']
    need(ordered == (['B', 'A'] if root['definition']['sideReversed'] else ['A', 'B']),
         'Original physical/displayed mapping')
    logs = [[float(x).hex() for x in row] for row in root['initialLogs']]
    return {'id': list(I.coordinates(root)), 'task': task, 'logsHex': logs,
            'logBits': [[struct.pack('>d', float.fromhex(x)).hex() for x in row] for row in logs],
            'orderedPhysical': list(ordered), 'orderProof': ORDER_PROOF,
            'scientificSHA256': I.scientific_digest(root), 'stateSnapshotSHA256': root['stateSnapshotSHA256']}


def render(roots, node):
    command = [node, str(Path(__file__).with_name('display.cjs'))]
    process = subprocess.run(command, input=json.dumps(roots, allow_nan=False), text=True,
                             capture_output=True, check=True)
    displays = json.loads(process.stdout)
    need(len(displays) == len(roots), 'Complete rendered root roster')
    for root, display in zip(roots, displays):
        need(display['id'] == root['id'] and display['status'] == 'authenticated'
             and display['priorLogBits'] == root['logBits'] and display['orderEquivalent'] is True,
             'Display must retain its original ordered prior')
        root['display'] = display
    return roots


def prepare(output, config, selected, *, node='node'):
    output = Path(output)
    ctx = context(config)
    data = I.prepare(ctx, selected, output / 'original_inputs', node)
    roots = render([root_adapter(root) for root in data['roots']], node)
    binding = {'recipe': RECIPE, 'selection': I.selection(selected), 'sources': sources(),
               'runtime': runtime(), 'rootSHA256': digest(roots)}
    write(output / 'roots.json', roots)
    write(output / 'run.json', binding)
    return binding


def load(output, config, *, node=None):
    output = Path(output)
    binding = read(output / 'run.json')
    ctx = context(config)
    need(binding['recipe'] == RECIPE and binding['sources'] == sources(), 'Changed card source or recipe')
    original = I.load_roots(ctx, binding['selection'], output / 'original_inputs')
    roots = read(output / 'roots.json')
    need(digest(roots) == binding['rootSHA256'] and len(roots) == len(original), 'Changed prepared card roots')
    for root, source in zip(roots, original):
        expected = root_adapter(source)
        need({k: root[k] for k in expected} == expected and root['display']['id'] == root['id']
             and root['display']['priorLogBits'] == root['logBits'], 'Original input/display binding')
    if node is not None:
        need(render([root_adapter(source) for source in original], node) == roots,
             'Independent display regeneration differs from prepared inputs')
    return binding, roots
