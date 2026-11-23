"""Source-bound saved-only comparisons of seven precommitted procedures."""
from pathlib import Path

from . import personal_time_inputs as P
from . import coefficient_io as IO
from . import personal_time_sensitivity as sensitivity
from . import cost_aware_inputs
from . import operation_menu as M
from . import operation_menu_reference as V
from .operation_costs import inputs as I, reproduction as R
from .operation_costs.checkpoint import Workspace

RECIPE = {
    'schema': 'operation-menu-reproduction-v1',
    'roots': [['C3', 0, 1], ['C5', 0, 1]],
    'noaskFamilies': ['ES', 'EO_raw_surplus', 'PN', 'AL', 'GN'],
    'directions': ['A', 'B'],
    'rhoHex': list(I.PRICES),
    'objective': 'combined_GPA_minus_rho_times_personal_seconds',
    'unknownMember': 'each_used_unknown_one_second_not_measured',
    'symbolicDomain': 'nonnegative_global_unknown_operation_seconds',
    'scope': 'saved_evaluator_not_free_policy_selector_or_global_optimum',
}
COMPACT_SCHEMA = 'operation-menu-saved-operands-v1'
COMPACT_SCOPE = 'saved_scientific_operands_not_public_policy_execution_proof'


def source_identity():
    """Bind the menu entry and the actual saved-input/I/O/table dependencies."""
    root = Path(__file__).resolve().parents[2]
    files = (Path(__file__), Path(M.__file__), Path(V.__file__), Path(P.__file__),
             Path(P.A.__file__), Path(IO.__file__), Path(sensitivity.__file__),
             Path(cost_aware_inputs.__file__), Path(I.__file__), Path(R.__file__),
             Path(I.cards.__file__), Path(I.__file__).with_name('checkpoint.py'),
             root / 'scripts/reproduce_operation_menu.py')
    return {str(path.relative_to(root)): I.cards.file_sha(path) for path in files}


def point(value):
    return [str(value), str(value)]


def _product(source, binding, unit):
    name = unit[0]
    paths = [source / folder / (name + '.json') for folder in ('products', 'checks')]
    P.need(all(p.is_file() and not p.is_symlink() for p in paths), 'Checked ordinary product files required')
    product, checked = map(P.read, paths)
    context = R.scientific_context(unit)
    P.need(set(product) == {'context', 'result'} and product['context'] == context,
           'Exact original scientific product context')
    P.need(checked['unit'] == name and checked['status'] == 'PASS'
           and checked['bindingSHA256'] == P.digest(binding)
           and checked['productSHA256'] == P.digest(product), 'Independent product binding')
    P.need(Workspace(source, binding).get('result_' + name, context) == product['result'],
           'Complete checked checkpoint identity')
    proof = checked['independent']
    P.need(proof['status'] == 'PASS' and proof['id'] == unit[2]['id'], 'Original receiving scope')
    if unit[1] == 'noask':
        P.need(proof['actors'] == 5 and proof['priceRows'] == 40, 'Complete no-question source')
    else:
        P.need(proof['reportRows'] == 100 and proof['comparisons'] == 16
               and proof['physicalRecipient'] == unit[3], 'Complete ex-ante direction source')
    return product['result'], dict(productSHA256=P.digest(product), checkSHA256=P.digest(checked))


def load_inputs(source, cost_config, conditions):
    """Admit compatible public checked products, never historical server receipts."""
    source = Path(source)
    P.need(conditions and len(set(conditions)) == len(conditions)
           and set(conditions) <= {'C3', 'C5'}, 'Distinct evaluated menu roots')
    binding, native = I.load(source, cost_config)
    roots = [r for r in native if r['id'][0] in conditions]
    P.need([r['id'] for r in roots] == [[c, 0, 1] for c in sorted(conditions)], 'Complete selected original roots')
    raw, directions, evidence = {}, [], {}
    for unit in R.units(roots, ('noask', 'ask_once'), ('A', 'B')):
        result, pins = _product(source, binding, unit)
        evidence[unit[0]] = pins
        if unit[1] == 'noask':
            raw[unit[2]['id'][0]] = result
        else:
            directions.append(result)
    source_evidence = dict(
        kind='compatible_checked_public_numerical_products', runSHA256=P.digest(binding),
        products=evidence,
        cheap=dict(producers={c: dict(sha256=P.digest(raw[c])) for c in raw}),
        ask=dict(producer=dict(sha256=P.digest(directions))),
    )
    normalized = []
    for root in roots:
        menu, ask_rows = [], {}
        for actor in raw[root['id'][0]]['actors']:
            ledger = actor['ledger']
            P.need([p['rho'] for p in actor['prices']] == I.price_values(), 'Exact noask price identities')
            menu.append(dict(id='noask:' + actor['family'], kind='noask', family=actor['family'],
                index=actor['index'], budget=root['task']['total_millihours'], GPAInterval=actor['GPAInterval'],
                knownSecondsInterval=point(ledger['knownSeconds']),
                symbolCountIntervals={k: point(v) for k, v in ledger['symbols'].items()},
                memberSecondsInterval=point(ledger['oneSecondMemberSeconds']),
                sourceRaw=source_evidence['cheap']['producers'][root['id'][0]]['sha256'],
                knownCoefficients=ledger['knownCoefficients']))
        for side in ('A', 'B'):
            direction = next(d for d in directions if d['id'] == root['id'] and d['physicalRecipient'] == side)
            P.need([p['rho'] for p in direction['prices']] == I.price_values(), 'Exact ask price identities')
            ask_rows[side] = [dict(report=r['report'], queriedVectorRef=r['queriedVectorRef'],
                otherVectorRef=r['otherVectorRef'],
                ledger={k: r['ledger'][k] for k in ('knownSeconds', 'symbols', 'oneSecondMemberSeconds')},
                rawSHA256=P.digest(r))
                for r in direction['reports']]
            menu.append(dict(id='ask:' + side, kind='ask_once', physicalRecipient=side,
                orderedSlot=direction['orderedSlot'], budget=direction['nextBudget'],
                GPAInterval=direction['expectedGPAInterval'],
                knownSecondsInterval=direction['expectedKnownSecondsInterval'],
                symbolCountIntervals=direction['expectedSymbolCountsIntervals'],
                memberSecondsInterval=direction['expectedMemberSecondsInterval'],
                sourceRaw=source_evidence['ask']['producer']['sha256'],
                knownCoefficients=direction['reports'][0]['ledger']['knownCoefficients']))
        law_context = dict(schema='operation-menu-law-context-v1',
            rootSHA256=P.digest(root), logsHex=root['logsHex'],
            widthHex=float(root['task']['report_width']).hex(), precisionBits=160,
            law='normalized_real_exp_of_admitted_binary64_logs_and_width_dyadic160',
            support=[1, 100], latentSpecificRowNormalization=True)
        normalized.append(dict(id=root['id'], rootSHA256=P.digest(root), root=root,
                               lawContext=law_context, menu=menu, askRows=ask_rows))
    return dict(roots=normalized, rhoValues=I.price_values(), sourceEvidence=source_evidence,
                sourceEvidenceSHA256=P.digest(source_evidence), rawCheap=raw, rawAskDirections=directions)


def export_compact(bundle, path):
    """Save exact mathematical operands, without asserting a policy execution."""
    P.write(path, dict(schema=COMPACT_SCHEMA, scope=COMPACT_SCOPE, bundle=bundle,
                       bundleSHA256=P.digest(bundle)))


def load_compact(source, conditions):
    """Read explicit scientific operands; hashes do not authenticate their history."""
    path = Path(source) / 'operands.json'
    P.need(path.is_file() and not path.is_symlink(), 'Ordinary saved-operand file')
    value = P.read(path)
    P.need(set(value) == {'schema', 'scope', 'bundle', 'bundleSHA256'}
           and value['schema'] == COMPACT_SCHEMA and value['scope'] == COMPACT_SCOPE
           and value['bundleSHA256'] == P.digest(value['bundle']), 'Exact saved-operand envelope')
    bundle = value['bundle']
    P.need([r['id'] for r in bundle['roots']] == [[c, 0, 1] for c in conditions],
           'Saved operands match explicit selected roots')
    return bundle


def run(source, output, configs, *, conditions=('C3', 'C5'), checking=False,
        input_kind='checked-products', exporting=False):
    source, output = IO.separate(source, output)
    configs = Path(configs)
    P.need(P.read(configs / 'operation_menu.json') == RECIPE, 'Exact saved-menu scientific recipe')
    conditions = tuple(sorted(conditions))
    P.need(input_kind in ('checked-products', 'saved-operands'), 'Explicit scientific input kind')
    P.need(not exporting or input_kind == 'checked-products', 'Export requires checked public products')
    bundle = (load_inputs(source, configs / 'operation_costs.json', conditions)
              if input_kind == 'checked-products' else load_compact(source, conditions))
    if exporting:
        export_compact(bundle, output / 'operands.json')
        return dict(output=str(output), scope=COMPACT_SCOPE, newPolicyCalls=0)
    binding = dict(schema=RECIPE['schema'], recipe=RECIPE, inputSHA256=P.digest(bundle),
                   inputKind=input_kind, historicalExecutionRevalidated=False,
                   sources=source_identity())
    value, new = IO.stage(output / 'menu.json', binding,
                          lambda: M.produce(bundle, conditions=conditions), readonly=checking)
    checked = V.verify(bundle, value, conditions=conditions)
    P.write(output / 'independent_check.json', dict(binding=binding, **checked))
    tables(output, value, input_kind)
    return dict(output=str(output), newSavedReduction=new, inputKind=input_kind,
                historicalExecutionRevalidated=False, **checked)


def tables(output, value, input_kind):
    menu, pairs, conditions = [], [], []
    for root in value['roots']:
        condition = root['id'][0]
        for row in root['menuBestSets']:
            menu.append(dict(condition=condition, rho=row['rho'], input_kind=input_kind,
                             historical_execution_revalidated=False, status=row['status'],
                             best_policies='|'.join(row['bestPolicies']), scope=row['scope']))
        for pair in root['pairs']:
            for row in pair['prices']:
                extra = row['extraTimeCondition']
                pairs.append(dict(condition=condition, input_kind=input_kind,
                    historical_execution_revalidated=False, first=pair['first'], second=pair['second'],
                    rho=row['rho'], status=row['status'], lower=row['differenceInterval'][0],
                    upper=row['differenceInterval'][1], extra_status=extra['status'],
                    extra_seconds_policy=extra.get('policy'), extra_seconds_threshold=extra.get('secondsThreshold')))
            conditions.append(dict(condition=condition, first=pair['first'], second=pair['second'],
                identityProof=pair['identityProof'], priceConditions=pair['oneSecondMemberRhoConditions'],
                symbolicConditions=pair['symbolicConditions']))
    P.write_bytes(Path(output) / 'menu_best.csv', R.csv_bytes(menu))
    P.write_bytes(Path(output) / 'pair_prices.csv', R.csv_bytes(pairs))
    P.write(Path(output) / 'sufficient_conditions.json', dict(inputKind=input_kind,
        historicalExecutionRevalidated=False, conditions=conditions))
