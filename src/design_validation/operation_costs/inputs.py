"""Original five-root input generation and recipient-local public grants."""
from fractions import Fraction as F
from pathlib import Path

from ..strategy_cards import inputs as cards
from ..personal_time_inputs import digest, need, read, write

PRICES = ('0x0.0p+0', '0x1.5798ee2308c3ap-27', '0x1.0c6f7a0b5ed8dp-20',
          '0x1.4f8b588e368f1p-17', '0x1.a36e2eb1c432dp-14',
          '0x1.0624dd2f1a9fcp-10', '0x1.9e7c6e43390b7p-9', '0x1.47ae147ae147bp-7')
RECIPE = {
    'schema': 'operation-cost-reproduction-v1',
    'source_recipe': 'strategy_cards.json',
    'roots': [[f'C{i}', 0, 1] for i in range(1, 6)],
    'replicate': 0,
    'rhoHex': list(PRICES),
    'literalRhoHex': list(PRICES[:3]),
    'families': ['ES', 'EO_raw_surplus', 'PN', 'AL', 'GN'],
    'askOnceRoots': ['C3', 'C5'],
    'directions': ['A', 'B'],
    'externalLaw': 'normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
    'utility': 'signed_sqrt_loss2_integer_millihours_native1000',
    'objective': 'combined_GPA_minus_rho_times_personal_seconds',
    'unknownCostMember': 'each_used_unknown_one_second_not_measured',
    'scope': 'fixed_roots_precommitted_procedures_not_free_selector',
}


def sources():
    here = Path(__file__).parent
    return {p.name: cards.file_sha(p) for p in sorted(here.glob('*.py'))}


def price_values():
    return [str(F.from_float(float.fromhex(x))) for x in PRICES]


def config(path):
    path = Path(path)
    need(read(path) == RECIPE, 'Exact accepted operation-cost scientific recipe')
    return path.parent / RECIPE['source_recipe']


def prepare(output, path, conditions, node='node'):
    output = Path(output)
    need(bool(conditions) and len(set(conditions)) == len(conditions)
         and set(conditions) <= {f'C{i}' for i in range(1, 6)}, 'Original five-root selection')
    selected = [dict(conditionId=c, scheduleVariant=0, mainIndex=1) for c in sorted(conditions)]
    binding = cards.prepare(output / 'inputs', config(path), selected, node=node)
    _, roots = cards.load(output / 'inputs', config(path))
    write(output / 'grants.json', source_grants(roots))
    record = dict(recipe=RECIPE, sources=sources(), inputBindingSHA256=digest(binding),
                  selection=selected, runtime=cards.runtime())
    write(output / 'run.json', record)
    return record


def load(output, path, node=None):
    output = Path(output)
    record = read(output / 'run.json')
    binding, roots = cards.load(output / 'inputs', config(path), node=node)
    need(record == dict(recipe=RECIPE, sources=sources(), inputBindingSHA256=digest(binding),
                        selection=binding['selection'], runtime=cards.runtime()),
         'Changed scientific source, recipe, root roster or runtime')
    need(all(root['id'] in RECIPE['roots'] for root in roots), 'Original root coordinates')
    need(read(output / 'grants.json') == source_grants(roots), 'Changed public display grants')
    return record, roots


def public_h(root, physical=None, report=None):
    """Only displayed fields reach the actor; latent evaluator operands stay outside."""
    t = root['task']
    means = root['display']['root'].copy()
    B = t['total_millihours']
    history, latest = [], [None, None]
    suffix = 'initial'
    if physical is not None:
        need(physical in ('A', 'B') and type(report) is int and 1 <= report <= 100,
             'Exactly one attributed hypothetical report')
        slot = root['orderedPhysical'].index(physical)
        means[slot] = root['display']['singleReport'][slot][report-1]
        latest[slot] = report
        history = [dict(slot=slot, report=report)]
        B -= t['cost_millihours']
        suffix = f'ask_{physical}|report_{report}'
    else:
        need(report is None, 'No report without recipient')
    need(type(B) is int and B > 0, 'Affordable positive remaining resource')
    return dict(stateId='cost|'+ '-'.join(map(str, root['id']))+'|'+suffix,
                budgetInteger=B, meanStrings=means,
                resourceString=f'{B//1000}.{B%1000:03d}',
                ranges=[dict(lower=1, upper=100), dict(lower=1, upper=100)],
                channel=dict(source='task_width_metadata_not_actor_operand',
                             reportWidthHex=float(t['report_width']).hex()),
                latest=latest, history=history)


def record(root, physical, report):
    H = public_h(root, physical, report)
    return dict(id=root['id'], physicalRecipient=physical,
                orderedSlot=root['orderedPhysical'].index(physical),
                report=report, H=H, HSHA256=digest(H))


def source_grants(roots):
    """Admit displayed operands only; no actor, score, actual report or update."""
    return dict(initial=[dict(id=r['id'],H=public_h(r)) for r in roots],
                afterOne=[record(r,d,y) for r in roots if r['id'][0] in ('C3','C5')
                          for d in ('A','B') for y in range(1,101)],
                actualReports=0,actorExecutions=0,valueEvaluations=0,
                scope='hypothetical_public_input_only_not_a_paid_event')
