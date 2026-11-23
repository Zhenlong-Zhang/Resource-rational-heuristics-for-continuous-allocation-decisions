"""Fresh source64 costed VOC paths and matched initial-information equal split.

The caller authenticates native roots and the numerical runtime through inputs.
An optional checkpoint.Workspace carries that source/config/runtime binding.
No historical numeric proof, receipt or accepted-score cache is an input.
"""
import copy
from fractions import Fraction as F
from math import isfinite, isqrt

from .. import bounded_task
from ..personal_time_inputs import digest
from ..strategy_cards import adaptive, adaptive_path, external_bounds as external
from . import costs

PRICES = ('0x0.0p+0', '0x1.5798ee2308c3ap-27', '0x1.0c6f7a0b5ed8dp-20')
MODEL = {
    'internal': 'literal_operation_costed_source_binary64',
    'external': 'normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
    'costKnowledge': 'supplied_duration_schedule_lookup_charged',
    'constants': 'supplied_baseline_scale_rho_three_reads_per_comparison',
}
LAW = 'literal_costed_repeated_comparison'
ES_COUNTS = dict(retrieve_rule=1, read_scalar=1, compare=1, common_response=1)
ES_PRICES = dict(retrieve_rule='1/4', read_scalar='1/2', compare='1/4', common_response='53/20')
ES_GRANT = 'known_native_half_index_500_for_D1000_positive_root_budget'


def _root(root, rho_hex):
    sid = root['id']
    task = root['task']
    if (rho_hex not in PRICES or type(sid) is not list or len(sid) != 3
            or sid[0] not in ('C1', 'C2', 'C3', 'C4', 'C5')
            or type(sid[1]) is not int or sid[1] != 0
            or type(sid[2]) is not int or sid[2] != 1
            or sorted(root['orderedPhysical']) != ['A', 'B']
            or type(task['total_millihours']) is not int or task['total_millihours'] <= 0
            or type(task['cost_millihours']) is not int or task['cost_millihours'] <= 0
            or task['allocation_denominator'] != 1000 or task['alpha'] != .5
            or task['loss_multiplier'] != 2 or not isfinite(task['report_width'])
            or task['report_width'] <= 0 or not isfinite(task['gpa_baseline'])
            or not isfinite(task['gpa_scale']) or task['gpa_scale'] <= 0
            or len(root['logsHex']) != 2
            or any(len(row) != 100 or any(not isfinite(float.fromhex(x)) for x in row)
                   for row in root['logsHex'])):
        raise ValueError('Original five-root literal scope, three exact prices and signed payoff')
    owned = copy.deepcopy(root)
    owned['costModel'] = {'rhoHex': rho_hex}
    return owned


class Policy:
    def __init__(self, root, oracle):
        self.root = root
        self.base = adaptive.Policy(root, bounded_task, oracle)
        self.external = oracle

    def compare(self, root, budget, history):
        raw = self.base.compare(root, budget, history)
        stop = F(raw['stop'][0])
        asks = {int(slot): F(value[0]) for slot, value in raw['asks'].items()}
        stop, asks, ledger = costs.price(root, budget, history, stop, asks)
        return dict(raw, stop=[str(stop)]*2,
                    asks={slot: [str(value)]*2 for slot, value in asks.items()},
                    unpricedSourceProof=raw, costLedger=ledger, law=LAW)

    def latent_cdf(self, *args):
        return self.external.latent_cdf(*args)

    def report_cdf(self, *args):
        return self.external.report_cdf(*args)

    def update_is_certified(self, *args):
        return self.external.update_is_certified(*args)


def es_value(root):
    """Original isqrt80 signed half-index scalar, freshly evaluated at the root."""
    task = root['task']
    denominator = 1 << 80
    values = []
    for need in range(1, 101):
        x = task['total_millihours']*500-need*1_000_000
        low = isqrt(abs(x)*denominator*denominator//1_000_000)
        high = low+(low*low*1_000_000 != abs(x)*denominator*denominator)
        values.append((-F(2*high, denominator), -F(2*low, denominator)) if x < 0
                      else (F(low, denominator), F(high, denominator)))
    total = [F(), F()]
    for logs in root['logsHex']:
        for (a, b), (x, y) in zip(external.prior(tuple(logs)), values):
            products = (a*x, a*y, b*x, b*y)
            total[0] += min(products)
            total[1] += max(products)
    gpa = [str(2*F(task['gpa_baseline'])+F(task['gpa_scale'])*v) for v in total]
    return dict(id=root['id'], amountIndex=500, budget=task['total_millihours'],
                expectedGPAInterval=gpa, ESKnowledgeGrant=ES_GRANT,
                ESOperationCounts=ES_COUNTS.copy(), ESOperationPrices=ES_PRICES.copy(),
                ESFeeSeconds=str(costs.RESPONSE),
                origin='fresh_initial_root_signed_pair_not_historical_alias')


def comparison(root, rho_hex, result, es):
    rho = F.from_float(float.fromhex(rho_hex))
    gpa = list(map(F, es['expectedGPAInterval']))
    value = [v-rho*costs.RESPONSE for v in gpa]
    row = dict(rhoHex=rho_hex, immediateESCriterionInterval=list(map(str, value)),
               actualESSeconds=str(costs.RESPONSE), costedDisposition=result['status'])
    terminal = result['status'] == 'terminal'
    last = result['steps'][-1]
    same = (terminal and last['cursorAfter'] == [0, 0]
            and last['budgetAfter'] == root['task']['total_millihours']
            and last['proof']['orderedHistory'] == [[], []])
    row['comparisonScope'] = 'same_initial_information' if same else 'conditional_path_only'
    if terminal:
        row.update(costedConditionalGPAInterval=result['externalConditionalExpectedGPAInterval'],
                   costedActualPersonalSeconds=result['actualPersonalSeconds'])
    if terminal and not same:
        row['comparison'] = 'conditioning_not_comparable'
    if same:
        low, high = map(F, result['externalCriterionInterval'])
        delta = [low-value[1], high-value[0]]
        raw = list(map(F, result['externalConditionalExpectedGPAInterval']))
        row.update(costedMinusESCriterionInterval=list(map(str, delta)),
                   costedMinusESGPAInterval=[str(raw[0]-gpa[1]), str(raw[1]-gpa[0])],
                   costedMinusESPersonalSeconds=str(F(result['actualPersonalSeconds'])-costs.RESPONSE),
                   comparison=('costed_strictly_better' if delta[0] > 0 else
                               'ES_strictly_better' if delta[1] < 0 else 'unresolved_overlap'))
    return row


def produce(root, rho_hex, workspace=None):
    """One original replicate; workspace is an optional bound Workspace object."""
    owned = _root(root, rho_hex)
    key = 'literal_'+root['id'][0]+'_'+str(PRICES.index(rho_hex))
    context = dict(rootSHA256=digest(root), rhoHex=rho_hex, replicate=0, model=MODEL)
    if workspace is not None:
        saved = workspace.get(key+'_result', context)
        if saved is not None:
            return saved
    oracle = external.CertifiedOracle(owned)
    prefix = []
    if workspace is not None:
        while not prefix or prefix[-1]['final'] is None:
            stage = workspace.get(key+f'_step_{len(prefix):04}',
                                  dict(context, prefixSHA256=digest(prefix)))
            if stage is None:
                break
            prefix.append(stage)

    def commit(index, step, final):
        stage = dict(step=step, final=final)
        if workspace is not None:
            workspace.put(key+f'_step_{index:04}',
                          dict(context, prefixSHA256=digest(prefix)), stage)
        prefix.append(stage)

    result = adaptive_path.path(owned, 0, Policy(owned, oracle), prefix=tuple(prefix), commit=commit)
    seconds = sum((F(step['proof']['costLedger']['current']['seconds'])
                   for step in result['steps']), F())
    seconds += costs.REQUEST*sum('report' in step for step in result['steps'])
    if result['status'] == 'terminal':
        seconds += costs.RESPONSE
        last = result['steps'][-1]
        index = result['target'] if last['budgetAfter'] else 0
        table = external.utilities(last['budgetAfter'], 2)
        pair = []
        for slot, target in enumerate((index, 1000-index)):
            law = external.round_law([oracle.belief(slot, last['proof']['orderedHistory'][slot])])
            bounds = external.dots(law, table)
            pair.append([int(value[0, target]) for value in bounds])
        gpa = oracle.gpa([F(sum(v[side] for v in pair), external.P*external.U) for side in (0, 1)])
        result['externalConditionalExpectedGPAInterval'] = gpa
        rho = F.from_float(float.fromhex(rho_hex))
        result['externalCriterionInterval'] = [str(F(value)-rho*seconds) for value in gpa]
    result.update(id=root['id'], replicate=0, rhoHex=rho_hex, model=MODEL.copy(),
                  actualPersonalSeconds=str(seconds))
    es_context = dict(rootSHA256=digest(root), law=MODEL['external'], amountIndex=500,
                      budget=root['task']['total_millihours'], precisionBits=80)
    es = (es_value(owned) if workspace is None else
          workspace.obtain('literal_es_'+root['id'][0], es_context, lambda: es_value(owned)))
    result['immediateES'] = es
    row = comparison(root, rho_hex, result, es)
    result['immediateESCriterionSupport'] = row['immediateESCriterionInterval']
    result['comparison'] = row
    if workspace is not None:
        workspace.put(key+'_result', context, result)
    return result
