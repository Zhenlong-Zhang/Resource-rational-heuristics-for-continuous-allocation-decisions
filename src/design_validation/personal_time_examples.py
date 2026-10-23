"""Four fixed equal-outcome target indices, not welfare-loss estimates."""
from pathlib import Path

from .bounded_task import load_design
from .budget_horizon import BudgetTask, budget_templates
from .terminal_time import terminal_summary
from .personal_time_inputs import digest, implementation_identity, need, read, reference_runtime, write, write_bytes

EXPECTED = {('C2_s1_g0', 'EO_expected_GPA'): 367, ('C2_s1_g0', 'EO_raw_surplus'): 367,
            ('C3_s1_g0', 'EO_expected_GPA'): 451, ('C3_s1_g0', 'EO_raw_surplus'): 446}


def target_rows(design):
    configs = {c.template_id: c for c in budget_templates(design)}
    rows = []
    for template_id in ('C2_s1_g0', 'C3_s1_g0'):
        config = configs[template_id]
        task = BudgetTask(config)
        value = terminal_summary(config, *task.initial_logs, config.total_millihours)
        for macro in ('EO_expected_GPA', 'EO_raw_surplus'):
            index = value['macros'][macro]['index']
            rows.append({'template_id': template_id, 'rule': macro, 'index': index,
                         'share': index / config.allocation_denominator,
                         'reference_index': EXPECTED[template_id, macro],
                         'exact_index_agreement': index == EXPECTED[template_id, macro]})
    return rows


def reproduce(design_path, output):
    from .personal_time_reproduction import csv_bytes
    reference_runtime()
    output = Path(output)
    write(output / 'examples_binding.json', {'schema': 'personal-time-equal-targets-v1',
          'designSHA256': digest(read(design_path)), 'implementationSHA256': implementation_identity()})
    rows = target_rows(load_design(design_path))
    write(output / 'equal_targets.json', rows)
    write_bytes(output / 'equal_targets.csv', csv_bytes(rows, (
        'template_id', 'rule', 'index', 'share', 'reference_index', 'exact_index_agreement')))
    need(all(r['exact_index_agreement'] for r in rows), 'Target-index discrepancy; computed rows retained')
    return {'targets': 4, 'exactIndexAgreement': True, 'welfareLossInferred': False,
            'policyEvidence': False, 'humanDataInference': False}
