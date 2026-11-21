"""Four paper plots from complete fixed-roster summaries, without model work.

``figure_data`` is pure and returns JSON-compatible plotting values. ``render``
uses those exact values and returns figure-name -> Path. Inputs are parsed
mappings, including the frozen recipe. Source admission belongs to the caller;
neither complete flags nor agreement of summaries establishes provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path


CONDITIONS = ('C1', 'C2', 'C3', 'C4', 'C5')
PRICES = ('rho_0', 'rho_0.00001', 'rho_0.0001', 'rho_0.001',
          'rho_geometric_0.001_0.01', 'rho_0.01')
FIGURES = ('figure_1_framework', 'figure_2_allocation_distances',
           'figure_3_utility_and_search', 'figure_4_true_need_coefficients')
FIELDS = ('expected_u', 'realized_u', 'initial_ask', 'mean_paid',
          'ES_pp', 'EO_pp', 'OPT_pp', 'AL_pp')
METRICS = ('gap_slope', 'lower_need_partial')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, lower=None, upper=None):
    _need(type(value) in (int, float), label + ': finite number required')
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    _need(valid and (lower is None or value >= lower)
          and (upper is None or value <= upper), label + ': invalid or nonfinite value')
    return value


def _integer(value, label, lower, upper):
    _need(type(value) is int and lower <= value <= upper, label + ': invalid count')
    return value


def _mean(values):
    return math.fsum(values) / len(values)


def _match(actual, expected, label, tolerance):
    _number(actual, label)
    _need(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
          label + ': schedule reduction does not match summary')


def _index(rows, fields, expected, label):
    _need(type(rows) is list, label + ': rows must be a list')
    result = {}
    for row in rows:
        _need(isinstance(row, Mapping), label + ': row must be a mapping')
        key = tuple(row[field] for field in fields)
        _need(all(type(x) is str for x in key[:2])
              and (len(key) == 2 or type(key[2]) is int), label + ': invalid coordinate')
        _need(key in expected and key not in result, label + ': duplicate or unexpected coordinate')
        result[key] = row
    _need(result.keys() == expected, label + ': incomplete fixed roster')
    return result


def _recipe(recipe):
    _need(isinstance(recipe, Mapping) and recipe['schema'] == 'paper-figures-v1', 'Figure recipe schema')
    _need(recipe['allocationSchema'] == 'finite-allocation-welfare-v1'
          and recipe['trueNeedSchema'] == 'true-need-allocation-diagnostics-v1', 'Summary schemas')
    _need(recipe['scope'] == dict(conditions=5, prices=6, schedulesPerCell=9,
                                   trialsPerSchedule=10, endpoints=2700), 'Fixed figure scope')
    _need([p['key'] for p in recipe['prices']] == list(PRICES), 'Frozen six-price order')
    _need([p['value'] for p in recipe['prices']] ==
          [0, 1e-5, 1e-4, .001, .0031622776601683794, .01], 'Frozen six-price values')
    _need([p['label'] for p in recipe['prices']] ==
          ['0', '10\u207b\u2075', '10\u207b\u2074', '0.001', '0.00316', '0.01'], 'Frozen price labels')
    _need([c['id'] for c in recipe['conditions']] == list(CONDITIONS), 'Frozen condition order')
    _need([(b['label'], b['field']) for b in recipe['benchmarks']] ==
          list(zip(('ES', 'EO', 'MU', 'LN'), FIELDS[4:])), 'Frozen benchmark mapping')
    _need(recipe['frameworkPrice'] == 'rho_0.001' and recipe['decimalPlaces'] == 2
          and recipe['absoluteTolerance'] == 1e-10, 'Frozen figure estimands and tolerance')
    _need(recipe['definitions']['uncertainty'] == 'sample_sd_across_nine_fixed_schedules_ddof1',
          'Sample SD, not a confidence interval')
    _need(set(recipe['figures']) == set(FIGURES), 'Exactly four named figures')
    for name, size, dpi in zip(FIGURES, ([11, 6.4], [10, 4.3], [9, 3.4], [10, 4]), (180, 240, 240, 240)):
        _need(recipe['figures'][name] == dict(sizeInches=size, dpi=dpi), 'Frozen size/dpi: ' + name)
    _need(len(recipe['style']['colors']) == 5, 'One color per condition')
    for c in recipe['conditions']:
        _need(all(type(c[k]) is str and c[k].strip() for k in ('label', 'shortLabel')),
              'Descriptive condition labels required')


def _design(design, recipe):
    _need(isinstance(design, Mapping) and type(design['conditions']) is list, 'Design conditions')
    rows = design['conditions']
    _need(len(rows) == 5 and {c['id'] for c in rows} == set(CONDITIONS), 'Exact design conditions')
    by = {c['id']: c for c in rows}
    result = []
    for label in recipe['conditions']:
        c = by[label['id']]
        item = deepcopy(label)
        for key in ('prior_kernel_centres', 'prior_kernel_spreads'):
            values = c[key]
            _need(type(values) is list and len(values) == 2, 'Two prior parameters: ' + key)
            item[key] = [_number(v, key, 0) for v in values]
            _need(all(v > 0 for v in values), 'Positive prior parameters')
        item['total_time'] = _number(c['total_time'], 'Initial tutoring budget', 0)
        _need(item['total_time'] > 0, 'Positive initial tutoring budget')
        result.append(item)
    return result


def _scope(summary, schema):
    _need(isinstance(summary, Mapping) and summary['schema'] == schema, 'Summary schema mismatch')
    _need(summary['fullFiniteTarget'] is True and type(summary['verifiedTrials']) is int
          and summary['verifiedTrials'] == 2700, 'Full 2700-endpoint scope required')


def _allocation(summary, cells, schedules, tolerance):
    _need(summary['partialPersons'] == [], 'No partial allocation schedules')
    people = _index(summary['persons'], ('condition', 'price', 'variant'), schedules, 'Allocation schedules')
    conditions = _index(summary['conditions'], ('condition', 'price'), cells, 'Allocation cells')
    result = {}
    for key, cell in conditions.items():
        _integer(cell['endpoints'], str(key) + ' endpoints', 90, 90)
        group = [people[(*key, v)] for v in range(9)]
        row = {}
        for field in FIELDS:
            low = 0 if field not in ('expected_u', 'realized_u') else None
            high = 1 if field == 'initial_ask' else 50 if field == 'ES_pp' else 100 if field.endswith('_pp') else None
            values = [_number(p[field], str((*key, v, field)), low, high) for v, p in enumerate(group)]
            row[field] = _mean(values)
            _match(cell[field], row[field], str((*key, field)), tolerance)
        result[key] = row
    return result


def _true_need(summary, cells, schedules, tolerance):
    _integer(summary['expectedEndpoints'], 'Expected endpoints', 2700, 2700)
    _integer(summary['expectedSchedules'], 'Expected schedules', 270, 270)
    _need(summary['missingCoordinates'] == {p: [] for p in PRICES}, 'Missing endpoint coordinates')
    _need(summary['uncertainty'] == 'sample_sd_across_nine_fixed_schedules_ddof1', 'Wrong uncertainty definition')
    people = _index(summary['schedules'], ('condition', 'price', 'schedule'), schedules, 'True-need schedules')
    conditions = _index(summary['conditions'], ('condition', 'price'), cells, 'True-need cells')
    for key, row in people.items():
        label = str(key)
        _need(row['complete'] is True and row['present_mains'] == list(range(1, 11))
              and row['missing_mains'] == [], label + ': complete ten-trial schedule required')
        _integer(row['available_rows'], label + ' available rows', 10, 10)
        _integer(row['missing'], label + ' missing true needs', 0, 0)
        ties = _integer(row['ties'], label + ' exact ties', 0, 10)
        eligible = _integer(row['eligible'], label + ' eligible rows', 3, 10)
        _need(ties + eligible == 10, label + ': ties and eligible rows must exhaust schedule')
        for field in METRICS:
            _need(row[field] is not None, label + ': undefined ' + field + '; no plotted fits may be omitted')
            _number(row[field], label + ' ' + field)
        _integer(row['gap_rank'], label + ' gap rank', 2, 2)
        _integer(row['partial_rank'], label + ' partial rank', 3, 3)
    result = {}
    for key, cell in conditions.items():
        group = [people[(*key, v)] for v in range(9)]
        _need(cell['complete'] is True and sorted(cell['present_schedules']) == list(range(9))
              and cell['missing_schedules'] == [], str(key) + ': complete nine-schedule cell required')
        _integer(cell['complete_schedules'], str(key) + ' complete schedules', 9, 9)
        counts = {}
        for field in ('available_rows', 'missing', 'ties', 'eligible'):
            counts[field] = sum(p[field] for p in group)
            _integer(cell[field], str((*key, field)), counts[field], counts[field])
        row = dict(counts)
        for field in METRICS:
            values = [p[field] for p in group]
            mean = _mean(values)
            sd = math.sqrt(math.fsum((x - mean)**2 for x in values) / 8)
            supplied = cell[field]
            for count in ('defined_schedules', 'expected_schedules'):
                _integer(supplied[count], str((*key, field, count)), 9, 9)
            _match(supplied['mean'], mean, str((*key, field, 'mean')), tolerance)
            _match(supplied['sample_sd'], sd, str((*key, field, 'sample_sd')), tolerance)
            row[field] = dict(mean=mean, sample_sd=sd, schedules=values, defined_schedules=9)
        result[key] = row
    return result


def _distance(row, recipe):
    values = [row[b['field']] for b in recipe['benchmarks']]
    displayed = [format(x, '.2f') for x in values]
    rounded = [float(x) for x in displayed]
    minimum = min(rounded)
    winners = [b['label'] for b, x in zip(recipe['benchmarks'], rounded) if x == minimum]
    # Escaped Unicode retains the established star and superscript figure style.
    lines = [('\u2605 ' if x == minimum else '    ') + b['label'] + '  ' + text
             for b, x, text in zip(recipe['benchmarks'], rounded, displayed)]
    return dict(values=values, displayed=displayed, starred=[x == minimum for x in rounded],
                nearest=winners, minimum_displayed=format(minimum, '.2f'), lines=lines)


def _behavior(row, distance):
    # Relative EO resemblance is not a claim that EO is the nearest benchmark.
    if row['ES_pp'] == 0:
        allocation = 'exact equal division'
    elif row['AL_pp'] == 0:
        allocation = 'favor lower terminal mean need'
    elif row['EO_pp'] < row['ES_pp']:
        allocation = 'resembles equal mean surplus'
    else:
        allocation = 'nearest: ' + ' / '.join(distance['nearest'])
    ask = row['initial_ask']
    search = 'Stop' if ask == 0 else 'Ask' if ask == 1 else f'Ask {100 * ask:.1f}%'
    return search + ' \u00b7 ' + allocation


def figure_data(allocation_summary, true_need_summary, design, recipe):
    """Return complete, deterministic plotting values; reject inconsistent inputs.

    Missing or undefined plotted fits are errors, including when a supplied mean
    was computed from only the available schedules. Legacy initial-mean fits are
    not used here and may be undefined. No files or model modules are accessed.
    """
    try:
        _recipe(recipe)
        descriptions = _design(design, recipe)
        _scope(allocation_summary, recipe['allocationSchema'])
        _scope(true_need_summary, recipe['trueNeedSchema'])
        cells = {(c, p) for c in CONDITIONS for p in PRICES}
        schedules = {(*cell, v) for cell in cells for v in range(9)}
        allocation = _allocation(allocation_summary, cells, schedules, recipe['absoluteTolerance'])
        coefficients = _true_need(true_need_summary, cells, schedules, recipe['absoluteTolerance'])
        framework, distances, utility, fits = [], [], [], []
        for description in descriptions:
            c = description['id']
            base = allocation[c, recipe['frameworkPrice']]
            distance = _distance(base, recipe)
            ask = 100 * base['initial_ask']
            framework.append(dict(condition=c, initial_search_percent=ask, distances=distance,
                                  behavior=_behavior(base, distance)))
            distances.append([dict(condition=c, price=p, **_distance(allocation[c, p], recipe)) for p in PRICES])
            utility.append(dict(condition=c, realized_u=[allocation[c, p]['realized_u'] for p in PRICES],
                                initial_search_percent=[100 * allocation[c, p]['initial_ask'] for p in PRICES]))
            fits.append(dict(condition=c, cells=[dict(price=p, **coefficients[c, p]) for p in PRICES]))
        return dict(schema=recipe['schema'], prices=deepcopy(recipe['prices']), conditions=descriptions,
                    definitions=deepcopy(recipe['definitions']),
                    scope=dict(endpoints=2700, cells=30, schedules=270, schedules_per_cell=9,
                               trials_per_schedule=10, ties=sum(r['ties'] for r in coefficients.values()),
                               eligible=sum(r['eligible'] for r in coefficients.values()),
                               source_authenticated=False, historical_trajectories_claimed=False),
                    figures=dict(zip(FIGURES, (dict(price=recipe['frameworkPrice'], rows=framework),
                                              dict(rows=distances), dict(series=utility), dict(series=fits)))))
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed or numerically invalid figure inputs: ' + str(exc)) from exc


def _framework(fig, data, recipe):
    from matplotlib.patches import Rectangle

    ax = fig.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=.055, right=.97, bottom=.18, top=.84)
    fig.suptitle('From welfare to allocation under costly deliberation', x=.055, y=.97,
                 ha='left', fontsize=17, fontweight='bold')
    for x, title, body in (
            (.03, 'Objective', 'Combined recipient welfare\nminus the value of personal time'),
            (.37, 'Process', 'Ask or stop, then choose a\nterminal allocation routine'),
            (.71, 'Environment', 'Needs, uncertainty and scarcity\nchange the payoff to each choice')):
        ax.text(x, .91, title, fontsize=13, fontweight='bold', color=recipe['style']['accent'], transform=ax.transAxes)
        ax.text(x, .76, body, fontsize=10.8, transform=ax.transAxes)
    for y, description, row in zip((.59, .47, .35, .23, .11), data['conditions'], data['figures'][FIGURES[0]]['rows']):
        ax.add_patch(Rectangle((0, y-.034), 1, .095, transform=ax.transAxes, color=recipe['style']['band'], zorder=0))
        ax.text(.025, y, description['id'], weight='bold', transform=ax.transAxes)
        ax.text(.12, y, description['label'], fontsize=10.5, transform=ax.transAxes)
        ax.text(.56, y, row['behavior'], fontsize=10, color=recipe['style']['accent'], transform=ax.transAxes)
    fig.text(.055, .025,
             'At 0.001 combined GPA/s; finite model predictions, not human observations.\n'
             'Equal-mean-surplus resemblance means closer than equal split, not necessarily the nearest benchmark.\n'
             'Resemblance to a rule does not identify the computation used to generate it.',
             fontsize=8.5, color='#43505d', linespacing=1.5)


def _matrix(fig, data, recipe):
    ax = fig.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=.01, right=.985, bottom=.06, top=.88)
    rows = data['figures'][FIGURES[1]]['rows']
    # Give the long descriptive row labels a real column, not an off-axis label.
    table = ax.table(cellText=[[c['label']] + ['\n'.join(v['lines']) for v in row]
                               for c, row in zip(data['conditions'], rows)],
                     colLabels=[''] + [p['label'] for p in data['prices']],
                     colWidths=[.29] + [.71/6]*6, cellLoc='left', bbox=[0, 0, 1, .98])
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor(recipe['style']['tableEdge'])
        cell.set_facecolor(recipe['style']['tableBand'] if i % 2 else 'white')
        cell.set_height(.08 if i == 0 else .18)
        cell.PAD = .045
        if i == 0:
            cell.get_text().set_ha('center')
        if j == 0:
            cell.get_text().set_wrap(True)
    ax.set_title('Allocation distance to four benchmarks (percentage points)', pad=10)


def _curves(fig, data, recipe, coefficients=False):
    axes = fig.subplots(1, 2)
    labels = [p['label'] for p in data['prices']]
    x = list(range(6))
    series = data['figures'][FIGURES[3] if coefficients else FIGURES[2]]['series']
    for description, values, color in zip(data['conditions'], series, recipe['style']['colors']):
        for ax, field in zip(axes, METRICS if coefficients else ('realized_u', 'initial_search_percent')):
            if coefficients:
                ax.errorbar(x, [r[field]['mean'] for r in values['cells']],
                            yerr=[r[field]['sample_sd'] for r in values['cells']],
                            marker='o', capsize=3, color=color, label=description['shortLabel'])
            else:
                ax.plot(x, values[field], marker='o', color=color, label=description['label'])
    for i, ax in enumerate(axes):
        ax.set_xticks(x, labels)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel(recipe['style']['priceAxis'])
        if coefficients:
            ax.axhline(0, color='gray', lw=.6)
            ax.set_title(('True-need gap slope', 'Partial slope on lower true need')[i])
            ax.set_ylabel('Higher-need recipient share coefficient')
        else:
            ax.set_ylabel(('Mean realized utility', 'Initial search (%)')[i])
            ax.grid(axis='y', alpha=.2)
    if coefficients:
        fig.text(.5, .025, 'Mean \u00b1 SD across nine fixed schedules (not confidence intervals)', ha='center', fontsize=9)
    else:
        axes[1].set_ylim(-4, 104)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, 1), ncols=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .075 if coefficients else 0, 1, .86))


def render(allocation_summary, true_need_summary, design, output, recipe):
    """Render four descriptive PNGs; return names mapped to filesystem Paths.

    Validation precedes directory creation and importing Matplotlib. Agg figures
    avoid changing a caller's interactive backend or leaving pyplot figures open.
    The separate figure_data return value is suitable for JSON parity inspection.
    """
    data = figure_data(allocation_summary, true_need_summary, design, recipe)
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    with mpl.rc_context({'font.family': recipe['style']['fontFamily'],
                         'font.size': recipe['style']['fontSize']}):
        for name in FIGURES:
            settings = recipe['figures'][name]
            fig = Figure(figsize=settings['sizeInches'], dpi=settings['dpi'], facecolor='white')
            FigureCanvasAgg(fig)
            try:
                if name == FIGURES[0]:
                    _framework(fig, data, recipe)
                elif name == FIGURES[1]:
                    _matrix(fig, data, recipe)
                else:
                    _curves(fig, data, recipe, coefficients=name == FIGURES[3])
                path = output / (name + '.png')
                fig.savefig(path, dpi=settings['dpi'], facecolor='white')
                paths[name] = path
            finally:
                fig.clear()
    return paths
