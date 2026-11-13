"""Source-bound true-need reductions of checked six-price policy outputs."""
import json
from pathlib import Path

from . import personal_time_diagnostics as D
from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import true_need_diagnostics as T
from .personal_time_reproduction import csv_bytes

RECIPE = {
    'schema': 'true-need-allocation-diagnostics-v1',
    'allocationRecipe': 'personal_time_diagnostics.json',
    'tieRule': 'exact_integer_need_category_equality',
    'response': 'higher_true_need_recipient_final_share',
    'predictorDenominator': 'initial_tutoring_budget',
    'gapModel': 'intercept_plus_high_minus_low_true_need',
    'partialModel': 'intercept_plus_low_and_high_true_need',
    'aggregation': 'nine_equal_weight_fixed_schedules_sample_sd_ddof1',
    'scope': 'fixed_finite_roster_not_population_or_human_inference',
}


def terminal_fields(payload):
    """Read the actual selected routine, not a nearby benchmark allocation."""
    terminal = payload['terminal']
    decisions = [event['value']['value'] for event in payload['events']
                 if event['name'].startswith('decision_')]
    I.need(bool(decisions), 'A saved final decision is required')
    stop = decisions[-1]['stop']
    I.need(stop['action'] == terminal['action']
           and stop['index'] == terminal['intendedIndex'], 'Terminal decision linkage')
    selected = stop['legacySelection'].get('selected_macro')
    I.need(isinstance(selected, str) and bool(selected), 'Explicit selected legacy rule required')
    I.need(selected == terminal['action'], 'Selected macro and terminal action must agree')
    I.need(stop['legacySelection'].get('selected_index') == terminal['intendedIndex'],
           'Selected macro index and terminal index must agree')
    return {'terminal_action': terminal['action'], 'selected_rule': selected}


def table_rows(rows):
    """Flatten coefficient summaries into numeric CSV columns."""
    result = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flat.update({key + '_' + name: scalar for name, scalar in value.items()})
            elif isinstance(value, (list, tuple)):
                flat[key] = json.dumps(value)
            else:
                flat[key] = value
        result.append(flat)
    return result


def analyze(input_path, output_path, config_path, *, full=False):
    """Reuse authenticated endpoints and fit descriptive slopes, never a new policy."""
    source, output = Path(input_path).resolve(), S.safe_output(output_path)
    I.need(not source.is_relative_to(output.resolve())
           and not output.resolve().is_relative_to(source), 'Separate input and output trees required')
    config_path = Path(config_path).resolve()
    I.need(I.read(config_path) == RECIPE, 'Frozen true-need recipe required')
    if full:
        selection = I.read(source / 'run.json')['selection']
        I.need({(r['conditionId'], r['scheduleVariant'], r['mainIndex']) for r in selection}
               == I.ORIGINAL and len(selection) == len(I.ORIGINAL),
               'Full analysis requires the exact original 450-root selection')

    # The existing reader checks source/runtime identities, producer/reader pairs,
    # target-specific latent keys and fixed-outcome arithmetic before reduction.
    fixed = output / 'fixed_outcomes'
    D.analyze(source, fixed, config_path.parent / RECIPE['allocationRecipe'])
    rows = I.read(fixed / 'trial_metrics.json')
    check = I.read(fixed / 'independent_check.json')
    saved = I.read(source / 'independent_check.json')
    I.need(S.file_sha(source / 'independent_check.json') == check['policyCheckSHA256'],
           'Policy check changed after endpoint admission')
    context = S.load_context(config_path.parent / D.RECIPE['policyRecipe'])
    roots, _, _ = S.load_roots(context, saved['binding']['selection'], source)
    roots = {I.coordinates(root): root for root in roots}
    for row in rows:
        root = roots[row['condition'], row['variant'], row['main']]
        path = S.record_path(source, row['price'], root)
        relative = str(path.relative_to(source))
        I.need(S.file_sha(path) == check['policyInputs'][relative],
               'Policy record changed after endpoint admission')
        payload = I.read(path)['payload']
        fields = terminal_fields(payload)
        reader_path = S.record_path(source, row['price'], root, 'independent')
        reader_relative = str(reader_path.relative_to(source))
        I.need(S.file_sha(reader_path) == check['policyInputs'][reader_relative],
               'Independent record changed after endpoint admission')
        I.need(fields == terminal_fields(I.read(reader_path)['payload']),
               'Independent selected-routine agreement required')
        row.update(fields)

    result = T.summarize(rows, full=full)
    result['rowSourceAuthenticated'] = True
    I.write(output / 'trial_metrics.json', rows)
    I.write(output / 'summary.json', result)
    for name, values in (('true_need_schedules.csv', result['schedules']),
                         ('true_need_conditions.csv', result['conditions'])):
        if values:
            table = table_rows(values)
            I.write_bytes(output / name, csv_bytes(table, tuple(table[0])))
    I.write(output / 'c3_paired_decomposition.json', result['c3'])
    I.write(output / 'c5_selected_rules.json', result['c5'])
    I.write(output / 'source_check.json', {
        'recipe': RECIPE,
        'sources': {path.name: S.file_sha(path) for path in
                    (Path(__file__), Path(T.__file__), config_path)},
        'fixedOutcomeCheckSHA256': S.file_sha(fixed / 'independent_check.json'),
        'policyCheckSHA256': check['policyCheckSHA256'],
        'policyInputs': check['policyInputs'],
        'verifiedTrials': len(rows),
        'fullFiniteTarget': result['fullFiniteTarget'],
        'newPolicySimulations': 0,
        'historicalProvenanceReconstructed': False,
        'interpretation': 'descriptive_fixed_simulated_targets_not_human_inference',
    })
    return {'output': str(output), 'verifiedTrials': len(rows),
            'fullFiniteTarget': result['fullFiniteTarget'], 'newPolicySimulations': 0}
