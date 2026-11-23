"""Independent literal-cost path, source64 operand and initial ES receiving.

Reconstructs comparison operands, counts, CDF bins and external enclosures;
does not import the literal producer, cost calculator or its saved operands.
"""
from fractions import Fraction as F
import math

from ..strategy_cards import adaptive_reference, external_reference as reference

PRICES = ('0x0.0p+0', '0x1.5798ee2308c3ap-27', '0x1.0c6f7a0b5ed8dp-20')
MODEL = {
    'internal': 'literal_operation_costed_source_binary64',
    'external': 'normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
    'costKnowledge': 'supplied_duration_schedule_lookup_charged',
    'constants': 'supplied_baseline_scale_rho_three_reads_per_comparison',
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def counts_reference(budget, fee, h):
    n, k, d = 100, 1001, 100
    c = dict(add_subtract=4*n*n+4*n*h+6*n+2*k*n+2*k*(n-1)+k,
             multiply_divide=3*n*n+2*k+3*k*n,
             compare=n*(n-1)+(n-1)*h+2*(n-1)+3*k*n+k-1,
             exp=n*n+n*h+4*n, log=n+h+2, sqrt=2*k*n)
    c['compare'] += k
    c['multiply_divide'] += 6 if budget >= fee else 2
    c['add_subtract'] += 10 if budget >= fee else 2
    if budget >= fee:
        c['add_subtract'] += 2*k*n+6*n+10*n*d+2*d*k*(n-1)+2*k*(n-1)+2*d*k+2*(d-1)
        c['multiply_divide'] += 2*k+k*n+2*d*k*n+2*k*n+2*d
        c['compare'] += 3+3*k*n+2*(n-1)+2*d*(n-1)+2*d*(k-1)
        c['exp'] += 2*n+4*n*d+2*d
        c['log'] += 2+2*d
        c['sqrt'] += 2*k*n
    c['read_scalar'] = (2*(c['add_subtract']+c['multiply_divide']+c['compare'])
                        + c['exp']+c['log']+c['sqrt']+200+3+(2 if budget >= fee else 1))
    c['store'] = sum(c.values())-c['read_scalar']
    c['retrieve_rule'] = 1+(2 if budget >= fee else 1)
    prices = {op: F(1) for op in c}
    prices.update(read_scalar=F(1, 2), retrieve_rule=F(1, 4),
                  add_subtract=F(1, 2), compare=F(1, 4))
    return c, sum((prices[key]*value for key, value in c.items()), F())


def internal(root, budget, history, rho_hex):
    value, asks, targets = adaptive_reference.internal(root, budget, history)
    task = root['task']
    count = sum(map(len, history))
    rho = F.from_float(float.fromhex(rho_hex))
    child = (counts_reference(budget-task['cost_millihours'],
                              task['cost_millihours'], count+1)[1] if asks else F())
    base, scale = 2*F(task['gpa_baseline']), F(task['gpa_scale'])
    return (base+scale*value-rho*F(73, 20),
            {slot: base+scale*v-rho*(F(93, 20)+child+F(73, 20)) for slot, v in asks.items()},
            targets, value, asks)


def _operands(saved, stop, asks, targets, history, law):
    parsed = {int(k): tuple(map(F, v)) for k, v in saved['asks'].items()}
    need(len(parsed) == len(saved['asks']) and parsed == {k: (v, v) for k, v in asks.items()}
         and tuple(map(F, saved['stop'])) == (stop, stop)
         and saved['terminalIndexTieSet'] == targets
         and saved['certifiedTarget'] == (targets[0] if len(targets) == 1 else None)
         and saved['orderedHistory'] == history and saved['law'] == law,
         'Independent source64 operands, target and history')


def _ledger(step, task, rho_hex):
    budget, count = step['budgetBefore'], sum(step['cursorBefore'])
    ledger = step['proof']['costLedger']
    expected, seconds = counts_reference(budget, task['cost_millihours'], count)
    need(ledger['current']['counts'] == expected
         and F(ledger['current']['seconds']) == seconds, 'Independent actual comparison cost')
    need(F(ledger['requestSeconds']) == F(93, 20)
         and F(ledger['responseSeconds']) == F(73, 20)
         and F(ledger['rho']) == F.from_float(float.fromhex(rho_hex)),
         'Single request/read, response and exact price')
    checked = [(ledger['current'], expected)]
    if budget >= task['cost_millihours']:
        counts, duration = counts_reference(budget-task['cost_millihours'],
                                            task['cost_millihours'], count+1)
        need(ledger['future']['counts'] == counts and F(ledger['future']['seconds']) == duration,
             'Future comparison cannot be omitted')
        checked.append((ledger['future'], counts))
    else:
        need(ledger['future'] is None, 'No unaffordable child')
    for part, counts in checked:
        need(all(type(n) is int and n >= 0 for n in part['counts'].values()), 'Typed event counts')
        for op in ('add_subtract', 'multiply_divide', 'compare', 'exp', 'log', 'sqrt'):
            need(sum(block['counts'].get(op, 0) for block in part['blocks']) == counts[op],
                 'Typed blocks conserve operation inventory')
    return seconds


def root_value(root):
    task = root['task']
    accum = [F(), F()]
    for encoded in root['logsHex']:
        law = reference.root_law(tuple(encoded))
        for n, (p0, p1) in enumerate(law, 1):
            signed = task['total_millihours']*500-1_000_000*n
            scale = 1 << 96
            low = math.isqrt(abs(signed)*scale*scale//1_000_000)
            high = low+(low*low*1_000_000 != abs(signed)*scale*scale)
            a, b = ((-F(2*high, scale), -F(2*low, scale)) if signed < 0
                    else (F(low, scale), F(high, scale)))
            accum[0] += p1*a if a < 0 else p0*a
            accum[1] += p0*b if b < 0 else p1*b
    return [2*F(task['gpa_baseline'])+F(task['gpa_scale'])*v for v in accum]


def _es(root, rho_hex, result, checked):
    es = result['immediateES']
    need(es['id'] == root['id'] and es['amountIndex'] == 500
         and es['budget'] == root['task']['total_millihours']
         and es['ESKnowledgeGrant'] == 'known_native_half_index_500_for_D1000_positive_root_budget'
         and es['ESOperationCounts'] == dict(retrieve_rule=1, read_scalar=1, compare=1, common_response=1)
         and es['ESOperationPrices'] == dict(retrieve_rule='1/4', read_scalar='1/2',
                                            compare='1/4', common_response='53/20')
         and F(es['ESFeeSeconds']) == F(73, 20)
         and es['origin'] == 'fresh_initial_root_signed_pair_not_historical_alias',
         'Complete original root, ES grant and once-only fee')
    own = root_value(root)
    reference.contained(es['expectedGPAInterval'], own, 'Independent isqrt96 initial ES enclosure')
    gpa = list(map(F, es['expectedGPAInterval']))
    rho = F.from_float(float.fromhex(rho_hex))
    value = [x-rho*F(73, 20) for x in gpa]
    need(list(map(F, result['immediateESCriterionSupport'])) == value, 'Root ES priced support')
    row = result['comparison']
    expected = dict(rhoHex=rho_hex, immediateESCriterionInterval=list(map(str, value)),
                    actualESSeconds='73/20', costedDisposition=result['status'])
    terminal = checked['disposition'] == 'terminal'
    same = (terminal and checked['terminal']['cursors'] == [0, 0]
            and checked['terminal']['budget'] == root['task']['total_millihours'])
    expected['comparisonScope'] = 'same_initial_information' if same else 'conditional_path_only'
    if terminal:
        expected.update(costedConditionalGPAInterval=result['externalConditionalExpectedGPAInterval'],
                        costedActualPersonalSeconds=result['actualPersonalSeconds'])
    if terminal and not same:
        expected['comparison'] = 'conditioning_not_comparable'
    if same:
        a, b = map(F, result['externalCriterionInterval'])
        delta = [a-value[1], b-value[0]]
        raw = list(map(F, result['externalConditionalExpectedGPAInterval']))
        expected.update(costedMinusESCriterionInterval=list(map(str, delta)),
                        costedMinusESGPAInterval=[str(raw[0]-gpa[1]), str(raw[1]-gpa[0])],
                        costedMinusESPersonalSeconds=str(F(result['actualPersonalSeconds'])-F(73, 20)),
                        comparison=('costed_strictly_better' if delta[0] > 0 else
                                    'ES_strictly_better' if delta[1] < 0 else 'unresolved_overlap'))
    need(row == expected, 'Independent same-information comparison; no conditional ranking')
    return list(map(str, own))


def verify(root, rho_hex, result):
    """Independently receive a single root/price record, never a producer rerun."""
    sid, task = root['id'], root['task']
    need(type(sid) is list and len(sid) == 3 and sid[0] in ('C1', 'C2', 'C3', 'C4', 'C5')
         and type(sid[1]) is int and sid[1] == 0 and type(sid[2]) is int and sid[2] == 1
         and rho_hex in PRICES and result['id'] == sid and result['rhoHex'] == rho_hex
         and type(result['replicate']) is int and result['replicate'] == 0
         and result['model'] == MODEL, 'Original finite scope and numerical model')
    physical = root['orderedPhysical']
    need(sorted(physical) == ['A', 'B'] and type(task['total_millihours']) is int
         and task['total_millihours'] > 0, 'Original resource and physical order')
    ext = reference.Reference(root)
    budget = task['total_millihours']
    history, latents, seconds = [[], []], {}, F()
    steps = result['steps']
    need(bool(steps), 'Nonempty literal path')
    for index, step in enumerate(steps):
        status = None
        before, start = list(map(len, history)), budget
        proof = step['proof']
        need(step['budgetBefore'] == start and step['cursorBefore'] == before, 'Owned reached state')
        stop, ask, targets, raw_stop, raw_asks = internal(root, budget, history, rho_hex)
        _operands(proof, stop, ask, targets, history, 'literal_costed_repeated_comparison')
        _operands(proof['unpricedSourceProof'], raw_stop, raw_asks, targets, history,
                  'current_runtime_source_binary64_internal_utility')
        seconds += _ledger(step, task, rho_hex)
        slot = None
        if not ask or max(ask.values()) <= stop:
            need(step['decision'] == 'stop', 'Strict stop')
            if budget and len(targets) != 1:
                status = 'unresolved_terminal_index'
            else:
                status = 'terminal'
                j = targets[0] if budget else 0
                need(result['target'] == (j if budget else None)
                     and result['conditionalExpectedGPAInterval'] == proof['stop'], 'Terminal target/criterion')
                payoff = reference.terms(budget, 2)
                a, b = [reference.curve([v], payoff) for v in ext.laws]
                lo, hi = (F(int(a[k][0, j])+int(b[k][0, 1000-j]), reference.P*reference.U)
                          for k in (0, 1))
                reference.contained(result['externalConditionalExpectedGPAInterval'],
                                    ext.gpa((lo, hi)), 'External conditional terminal GPA')
        elif ask[0] == ask[1]:
            key = ['R13-one-more-v1', 'recipient_tie', root['id'], 0, index]
            u = reference.key_value(key)
            need(step['decision'] == 'ask_tie' and step['tieKey'] == key
                 and F(step['tieUniform']) == u, 'Independent physical symmetric tie')
            slot = physical.index('A' if u < F(1, 2) else 'B')
        else:
            slot = 0 if ask[0] > ask[1] else 1
            need(step['decision'] == 'ask', 'Strict best question')
        if step['decision'] != 'ask_tie':
            need('tieKey' not in step and 'tieUniform' not in step, 'No unused tie draw')
        if slot is None:
            need(not any(k in step for k in ('report', 'reportKey', 'latentKey', 'physicalRecipient')),
                 'Stopping carries no paid events')
        if slot is not None and slot not in latents:
            key = ['R13-one-more-v1', 'latent', root['id'], 0, physical[slot], None]
            latent = reference.read_bin(step, 'latent', key, ext.laws[slot])
            if latent is None:
                status = 'unresolved_latent_boundary'
            else:
                need(step['latentNeed'] == latent, 'One retained physical latent')
                latents[slot] = latent
        elif slot is not None:
            need('latentKey' not in step, 'No repeated latent draw')
        if slot is not None and status is None:
            need(step['orderedSlot'] == slot and step['physicalRecipient'] == physical[slot],
                 'Physical report recipient')
            key = ['R13-one-more-v1', 'report', root['id'], 0, physical[slot], before[slot]]
            report = reference.read_bin(step, 'report', key, ext.channel[latents[slot]-1])
            if report is None:
                status = 'unresolved_report_or_normalizer'
            else:
                law, _ = reference.update(ext.laws[slot], ext.channel, report)
                need(step.get('report') == report, 'Certified paid report')
                ext.laws[slot] = law
                history[slot].append(report)
                budget -= ext.cost
                seconds += F(93, 20)
        need(step['budgetAfter'] == budget and step['cursorAfter'] == list(map(len, history)),
             'Exact physical debit and selected cursor')
        if status is not None:
            need(index == len(steps)-1 and result['status'] == status, 'Closing disposition')
            checked = dict(status='PASS', disposition=status, paidReports=sum(map(len, history)))
            if status == 'terminal':
                seconds += F(73, 20)
                rho = F.from_float(float.fromhex(rho_hex))
                need(list(map(F, result['externalCriterionInterval'])) ==
                     [F(x)-rho*seconds for x in result['externalConditionalExpectedGPAInterval']],
                     'Actual criterion includes every sunk comparison')
                checked['terminal'] = dict(budget=budget, cursors=list(map(len, history)),
                                           target=result['target'],
                                           independentlyEnclosedGPA=list(map(str, ext.gpa((lo, hi)))))
            else:
                need(not any(k in result for k in ('externalCriterionInterval',
                                                  'externalConditionalExpectedGPAInterval')),
                     'Unresolved frontier is not a terminal value')
            need(F(result['actualPersonalSeconds']) == seconds, 'No sunk comparison refund')
            checked.update(actualPersonalSeconds=str(seconds), verifiedRoot=sid, verifiedScenarios=1)
            checked['ESExpectedGPAInterval'] = _es(root, rho_hex, result, checked)
            checked['ESFeeSeconds'] = '73/20'
            return checked
        need(budget < start, 'Strict budget descent after a paid report')
    raise ValueError('Missing closing frontier')
