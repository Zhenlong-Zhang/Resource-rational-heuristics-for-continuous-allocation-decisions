"""Purpose: check four-figure plotting parity using complete synthetic summaries.

No policies, model kernels, historical outcomes or private artifacts are used.
Synthetic schedule values test rounding ties, scientific metric selection,
complete-scope failures, descriptive SDs and four PNG/vector-PDF output pairs.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import struct
import tempfile
import unittest

from src.design_validation import paper_figures_plot as P


ROOT = Path(__file__).resolve().parents[1]


def full_fixture():
    """Produce the shapes returned by both full=True summary APIs, without fits."""
    recipe = json.loads((ROOT / 'configs/paper_figures.json').read_text())
    design = json.loads((ROOT / 'configs/personal_time_design.json').read_text())
    allocation = dict(schema=recipe['allocationSchema'], fullFiniteTarget=True,
                      verifiedTrials=2700, persons=[], conditions=[], partialPersons=[],
                      historicalEventClaimed=False, policyRerunByAnalysis=False,
                      monteCarloPrecisionEstablished=False)
    true_need = dict(schema=recipe['trueNeedSchema'], fullFiniteTarget=True,
                     verifiedTrials=2700, expectedEndpoints=2700, expectedSchedules=270,
                     schedules=[], conditions=[], missingCoordinates={p: [] for p in P.PRICES},
                     uncertainty=recipe['definitions']['uncertainty'], c3={}, c5={},
                     rowSourceAuthenticated=False, policyRerunByAnalysis=False,
                     historicalEventClaimed=False, monteCarloPrecisionEstablished=False,
                     humanInferenceAdopted=False)
    for ci, c in enumerate(P.CONDITIONS):
        for pi, p in enumerate(P.PRICES):
            for v in range(9):
                allocation['persons'].append(dict(
                    condition=c, price=p, variant=v, beta_high_need=None, rank_eligible=0,
                    expected_u=80 + ci + pi + v/10, realized_u=-8 + ci - pi/2 + v/10,
                    initial_ask=(v % 3)/10, mean_paid=v % 3,
                    ES_pp=2 + ci + v*.02, EO_pp=1 + pi*.2 + v*.03,
                    OPT_pp=.5 + pi*.01 + v*.01, AL_pp=10 + ci + v*.05))
                true_need['schedules'].append(dict(
                    condition=c, price=p, schedule=v, available_rows=10, missing=0,
                    ties=v % 2, eligible=10-v % 2, present_mains=list(range(1, 11)),
                    missing_mains=[], complete=True, gap_rank=2, partial_rank=3,
                    gap_slope=(ci-2)*.1 + pi*.02 + v*.03,
                    lower_need_partial=(2-ci)*.2 - pi*.03 - v*.07,
                    initial_mean_eligible=0, initial_mean_slope=None))
            group = [s for s in true_need['schedules'] if (s['condition'], s['price']) == (c, p)]
            true_need['conditions'].append(dict(
                condition=c, price=p, available_rows=90, missing=0,
                ties=sum(s['ties'] for s in group), eligible=sum(s['eligible'] for s in group),
                present_schedules=list(range(9)), missing_schedules=[], complete_schedules=9,
                complete=True, **{field: dict(mean=statistics.mean(s[field] for s in group),
                                             sample_sd=statistics.stdev(s[field] for s in group),
                                             defined_schedules=9, expected_schedules=9)
                                  for field in P.METRICS},
                initial_mean_slope=dict(mean=None, sample_sd=None, defined_schedules=0, expected_schedules=9)))
    refresh_allocation(allocation)
    return allocation, true_need, design, recipe


def refresh_allocation(allocation):
    allocation['conditions'] = []
    for c in P.CONDITIONS:
        for p in P.PRICES:
            group = [s for s in allocation['persons'] if (s['condition'], s['price']) == (c, p)]
            allocation['conditions'].append(dict(
                condition=c, price=p, endpoints=90, rank_eligible=0,
                defined_schedules=0, beta_high_need=None,
                **{field: statistics.mean(s[field] for s in group) for field in P.FIELDS}))


def set_cell(allocation, condition, price, **values):
    for row in allocation['persons']:
        if (row['condition'], row['price']) == (condition, price):
            row.update(values)
    refresh_allocation(allocation)


class TestPaperFiguresPlot(unittest.TestCase):
    def setUp(self):
        self.allocation, self.true_need, self.design, self.recipe = full_fixture()

    def data(self):
        return P.figure_data(self.allocation, self.true_need, self.design, self.recipe)

    def test_complete_pure_deterministic_json_and_exact_metric_sources(self):
        original = deepcopy((self.allocation, self.true_need, self.design, self.recipe))
        data = self.data()
        self.assertEqual(json.loads(json.dumps(data, allow_nan=False)), data)
        self.assertEqual(original, (self.allocation, self.true_need, self.design, self.recipe))
        self.assertEqual((data['scope']['endpoints'], data['scope']['cells'], data['scope']['schedules']), (2700, 30, 270))
        self.assertEqual(data['scope']['ties'] + data['scope']['eligible'], 2700)
        self.assertFalse(data['scope']['source_authenticated'])
        self.assertFalse(data['scope']['historical_trajectories_claimed'])
        series = data['figures'][P.FIGURES[2]]['series'][0]
        self.assertAlmostEqual(series['realized_u'][0], -7.6)
        self.assertAlmostEqual(series['initial_search_percent'][0], 10)
        self.assertNotEqual(series['realized_u'][0], self.allocation['conditions'][0]['expected_u'])
        self.assertEqual([r['value'] for r in data['prices']], [0, 1e-5, 1e-4, .001, .0031622776601683794, .01])
        for summary in (self.allocation, self.true_need):
            summary['conditions'].reverse()
        self.allocation['persons'].reverse()
        self.true_need['schedules'].reverse()
        self.design['conditions'].reverse()
        self.assertEqual(self.data(), data)

    def test_displayed_ties_not_unique_unrounded_minimum(self):
        set_cell(self.allocation, 'C1', P.PRICES[0], ES_pp=1.231, EO_pp=1.234, OPT_pp=2, AL_pp=3)
        row = self.data()['figures'][P.FIGURES[1]]['rows'][0][0]
        self.assertEqual(row['displayed'], ['1.23', '1.23', '2.00', '3.00'])
        self.assertEqual(row['starred'], [True, True, False, False])
        self.assertEqual(row['nearest'], ['ES', 'EO'])
        self.assertEqual([line.startswith('\u2605') for line in row['lines']], row['starred'])
        set_cell(self.allocation, 'C1', P.PRICES[0], ES_pp=0, EO_pp=0, OPT_pp=0, AL_pp=0)
        row = self.data()['figures'][P.FIGURES[1]]['rows'][0][0]
        self.assertEqual(row['starred'], [True]*4)

    def test_framework_rules_are_condition_independent_and_use_actual_slice(self):
        cases = [
            (dict(ES_pp=0, EO_pp=0, OPT_pp=0, AL_pp=0, initial_ask=0), 'Stop', 'exact equal division'),
            (dict(ES_pp=20, EO_pp=1, OPT_pp=.5, AL_pp=0, initial_ask=1), 'Ask', 'favor lower terminal mean need'),
            (dict(ES_pp=20, EO_pp=1, OPT_pp=.5, AL_pp=5, initial_ask=.3), 'Ask 30.0%', 'resembles equal mean surplus'),
            (dict(ES_pp=1, EO_pp=2, OPT_pp=.5, AL_pp=5, initial_ask=0), 'Stop', 'nearest: MU'),
        ]
        for values, search, allocation in cases:
            for c in P.CONDITIONS:
                set_cell(self.allocation, c, self.recipe['frameworkPrice'], **values)
            for condition in self.design['conditions']:
                condition['acquisition_prediction'] = 'contradictory prediction must be ignored'
                condition['allocation_prediction'] = 'not empirical data'
            rows = self.data()['figures'][P.FIGURES[0]]['rows']
            self.assertEqual([r['behavior'] for r in rows], [search + ' \u00b7 ' + allocation]*5)
        set_cell(self.allocation, 'C1', P.PRICES[0], ES_pp=0, initial_ask=1)
        self.assertEqual(self.data()['figures'][P.FIGURES[0]]['rows'][0]['behavior'], 'Stop \u00b7 nearest: MU')

    def test_framework_exact_means_are_not_rounded_to_zero_for_rules(self):
        set_cell(self.allocation, 'C1', self.recipe['frameworkPrice'], ES_pp=.001,
                 EO_pp=.0001, OPT_pp=.00001, AL_pp=.00001, initial_ask=0)
        row = self.data()['figures'][P.FIGURES[0]]['rows'][0]
        self.assertEqual(row['behavior'], 'Stop \u00b7 resembles equal mean surplus')

    def test_framework_compares_both_observed_prices_and_total_questions(self):
        for ci, c in enumerate(P.CONDITIONS):
            set_cell(self.allocation, c, 'rho_0.001', ES_pp=20, EO_pp=1,
                     OPT_pp=.5, AL_pp=5, initial_ask=.3, mean_paid=ci + 1.25)
            set_cell(self.allocation, c, 'rho_0.01', ES_pp=0, EO_pp=2,
                     OPT_pp=.5, AL_pp=5, initial_ask=0, mean_paid=0)
            set_cell(self.allocation, c, P.PRICES[0], initial_ask=1, mean_paid=99)
        data = self.data()['figures'][P.FIGURES[0]]
        self.assertEqual(data['price'], 'rho_0.001')
        self.assertEqual([(r['condition'], r['price']) for r in data['comparisons']],
                         [(c, p) for c in P.CONDITIONS for p in ('rho_0.001', 'rho_0.01')])
        for ci, c in enumerate(P.CONDITIONS):
            low, high = data['comparisons'][2*ci:2*ci+2]
            self.assertEqual(low, dict(condition=c, price='rho_0.001', mean_paid=ci + 1.25,
                                      behavior='Ask 30.0% \u00b7 resembles equal mean surplus'))
            self.assertEqual(high, dict(condition=c, price='rho_0.01', mean_paid=0,
                                       behavior='Stop \u00b7 exact equal division'))
            self.assertEqual(low['behavior'], data['rows'][ci]['behavior'])
        # The second displayed slice must pass the same schedule/summary checks.
        for cell in self.allocation['conditions']:
            if (cell['condition'], cell['price']) == ('C5', 'rho_0.01'):
                cell['mean_paid'] += 1
        with self.assertRaisesRegex(ValueError, 'does not match summary'):
            self.data()

    def test_framework_environment_table_and_arrows_point_to_strategy(self):
        from matplotlib.figure import Figure
        from matplotlib.text import Annotation

        data = self.data()
        fig = Figure()
        try:
            P._framework(fig, data, self.recipe)
            ax = fig.axes[0]
            text = [item.get_text() for item in ax.texts]
            self.assertIn('Recipient welfare minus\npersonal-time opportunity cost', text)
            self.assertIn('Resource-Rational Strategy', text)
            self.assertIn('Needs, uncertainty, scarcity\nand opportunity cost', text)
            arrows = [item for item in ax.texts if isinstance(item, Annotation)]
            self.assertEqual([(item.get_position(), item.xy) for item in arrows],
                             [((.18, .84), (.42, .84)), ((.87, .84), (.62, .84))])
            self.assertTrue(all(item.arrowprops['arrowstyle'] == '->' for item in arrows))
            table = ax.tables[0]
            self.assertEqual([table[0, j].get_text().get_text() for j in range(4)],
                             ['Environment: needs', 'Environment: resources',
                              'Opportunity cost', 'Resource-Rational Strategy'])
            self.assertEqual(len(table.get_celld()), 44)
            for i, comparison in enumerate(data['figures'][P.FIGURES[0]]['comparisons'], 1):
                description = data['conditions'][(i-1)//2]
                self.assertEqual(table[i, 0].get_text().get_text(), description['label'])
                self.assertEqual(table[i, 1].get_text().get_text(),
                                 f"Reference budget: {description['total_time']:g}")
                self.assertEqual(table[i, 2].get_text().get_text(), '0.001' if i % 2 else '0.01')
                self.assertEqual(table[i, 3].get_text().get_text(),
                                 comparison['behavior'] + f"\nMean questions: {comparison['mean_paid']:.2f}")
            self.assertEqual(list(fig.texts), [])
        finally:
            fig.clear()

    def test_search_overlap_markers_preserve_y_values_and_opportunity_cost_units(self):
        from matplotlib.figure import Figure

        for row in self.allocation['persons']:
            row['initial_ask'] = 0
        refresh_allocation(self.allocation)
        data = self.data()
        fig = Figure()
        try:
            P._curves(fig, data, self.recipe)
            utility, search = fig.axes
            for ci, (u, s) in enumerate(zip(utility.lines, search.lines)):
                self.assertEqual(list(u.get_xdata()), list(range(6)))
                self.assertEqual(list(s.get_xdata()), [x + .035*(ci-2) for x in range(6)])
                self.assertEqual(list(u.get_ydata()),
                                 data['figures'][P.FIGURES[2]]['series'][ci]['realized_u'])
                self.assertEqual(list(s.get_ydata()), [0]*6)
                self.assertEqual(s.get_marker(), ('o', 's', '^', 'D', 'v')[ci])
            self.assertEqual(len(search.lines), 5)
            for ax in fig.axes:
                self.assertEqual(ax.get_xlabel(), self.recipe['style']['priceAxis'])
                self.assertEqual(list(ax.get_xticks()), list(range(6)))
            self.assertEqual(utility.get_ylabel(), 'Mean realized utility')
            self.assertEqual(search.get_ylabel(), 'Initial search (%)')
        finally:
            fig.clear()
        matrix = Figure()
        try:
            P._matrix(matrix, data, self.recipe)
            self.assertEqual(matrix.axes[0].get_title(),
                             'Opportunity Cost of Time (combined GPA / personal second)')
        finally:
            matrix.clear()

    def test_coefficient_axes_and_bars_use_partial_lower_need_and_fixed_schedule_sd(self):
        from matplotlib.figure import Figure

        data = self.data()
        fig = Figure()
        try:
            P._curves(fig, data, self.recipe, coefficients=True)
            self.assertEqual([ax.get_ylabel() for ax in fig.axes],
                             ['Coefficient on true-need gap', 'Partial coefficient on lower true need'])
            for ax, field in zip(fig.axes, P.METRICS):
                self.assertEqual(ax.get_xlabel(), self.recipe['style']['priceAxis'])
                self.assertEqual(len(ax.containers), 5)
                for container, series in zip(ax.containers, data['figures'][P.FIGURES[3]]['series']):
                    line, _, bars = container.lines
                    self.assertEqual(list(line.get_xdata()), list(range(6)))
                    self.assertEqual(list(line.get_ydata()), [cell[field]['mean'] for cell in series['cells']])
                    for x, (segment, cell) in enumerate(zip(bars[0].get_segments(), series['cells'])):
                        metric = cell[field]
                        self.assertEqual(list(segment[:, 0]), [x, x])
                        self.assertAlmostEqual(segment[0, 1], metric['mean'] - metric['sample_sd'])
                        self.assertAlmostEqual(segment[1, 1], metric['mean'] + metric['sample_sd'])
        finally:
            fig.clear()

    def test_recomputed_nine_schedule_sample_sd_not_sem_or_population_sd(self):
        data = self.data()
        for series in data['figures'][P.FIGURES[3]]['series']:
            for cell in series['cells']:
                for field in P.METRICS:
                    metric = cell[field]
                    self.assertEqual(len(metric['schedules']), 9)
                    self.assertAlmostEqual(metric['mean'], statistics.mean(metric['schedules']))
                    self.assertAlmostEqual(metric['sample_sd'], statistics.stdev(metric['schedules']))
                    self.assertNotAlmostEqual(metric['sample_sd'], statistics.pstdev(metric['schedules']))
                    self.assertNotAlmostEqual(metric['sample_sd'], statistics.stdev(metric['schedules'])/3)
        first = data['figures'][P.FIGURES[3]]['series'][0]['cells'][0]['gap_slope']
        self.assertAlmostEqual(first['sample_sd'], .03 * math.sqrt(7.5))

    def test_rejects_partial_and_duplicate_rosters_even_with_full_flags(self):
        for summary_name, table in (('allocation', 'persons'), ('allocation', 'conditions'),
                                    ('true_need', 'schedules'), ('true_need', 'conditions')):
            for duplicate in (False, True):
                with self.subTest(summary=summary_name, table=table, duplicate=duplicate):
                    original = deepcopy(getattr(self, summary_name))
                    rows = getattr(self, summary_name)[table]
                    if duplicate:
                        rows[-1] = deepcopy(rows[0])
                    else:
                        rows.pop()
                    with self.assertRaises(ValueError):
                        self.data()
                    setattr(self, summary_name, original)

    def test_rejects_wrong_scope_counts_prices_and_partial_markers(self):
        for name in ('allocation', 'true_need'):
            for key, bad in (('fullFiniteTarget', False), ('fullFiniteTarget', 1), ('verifiedTrials', 2699), ('schema', 'other')):
                with self.subTest(summary=name, field=key):
                    original = deepcopy(getattr(self, name))
                    getattr(self, name)[key] = bad
                    with self.assertRaises(ValueError):
                        self.data()
                    setattr(self, name, original)
        self.allocation['persons'][0]['price'] = 'rho_unknown'
        with self.assertRaises(ValueError):
            self.data()
        self.allocation, self.true_need, _, _ = full_fixture()
        self.allocation['partialPersons'] = [dict(condition='C1')]
        with self.assertRaises(ValueError):
            self.data()

    def test_rejects_stale_means_and_sd_instead_of_trusting_summary(self):
        self.allocation['conditions'][0]['realized_u'] += .001
        with self.assertRaisesRegex(ValueError, 'does not match summary'):
            self.data()
        refresh_allocation(self.allocation)
        for field in P.METRICS:
            for key in ('mean', 'sample_sd', 'defined_schedules', 'expected_schedules'):
                with self.subTest(field=field, key=key):
                    original = deepcopy(self.true_need)
                    self.true_need['conditions'][0][field][key] += 1
                    with self.assertRaises(ValueError):
                        self.data()
                    self.true_need = original

    def test_undefined_fits_missing_pairs_and_rank_deficiency_are_not_dropped(self):
        for key, value in (('gap_slope', None), ('lower_need_partial', None), ('gap_rank', 1),
                           ('partial_rank', 2), ('missing', 1), ('eligible', 2), ('complete', False),
                           ('present_mains', list(range(1, 10))), ('missing_mains', [10])):
            with self.subTest(field=key):
                original = deepcopy(self.true_need)
                self.true_need['schedules'][0][key] = value
                with self.assertRaises(ValueError):
                    self.data()
                self.true_need = original
        self.true_need['missingCoordinates'][P.PRICES[0]] = [['C1', 0, 1]]
        with self.assertRaises(ValueError):
            self.data()

    def test_invalid_scalars_and_coordinate_types_fail_closed(self):
        for value in (None, True, '1.0', float('nan'), float('inf'), -float('inf'), 10**400):
            for name, table, field in (('allocation', 'persons', 'realized_u'),
                                       ('true_need', 'schedules', 'gap_slope')):
                with self.subTest(summary=name, value=str(value)):
                    original = deepcopy(getattr(self, name))
                    getattr(self, name)[table][0][field] = value
                    with self.assertRaises(ValueError):
                        self.data()
                    setattr(self, name, original)
        for field, value in (('initial_ask', 1.01), ('ES_pp', 50.01), ('EO_pp', -1), ('OPT_pp', 101), ('AL_pp', 101)):
            original = deepcopy(self.allocation)
            self.allocation['persons'][0][field] = value
            with self.assertRaises(ValueError):
                self.data()
            self.allocation = original
        self.true_need['schedules'][0]['schedule'] = False
        with self.assertRaises(ValueError):
            self.data()

    def test_all_ties_or_inconsistent_count_metadata_fail(self):
        for table, field, value in (('schedules', 'ties', 10), ('conditions', 'ties', 100),
                                    ('conditions', 'available_rows', 89), ('conditions', 'complete_schedules', 8),
                                    ('conditions', 'missing_schedules', [8])):
            original = deepcopy(self.true_need)
            self.true_need[table][0][field] = value
            with self.assertRaises(ValueError):
                self.data()
            self.true_need = original

    def test_invalid_recipe_and_design_fail(self):
        self.recipe['prices'][4]['value'] = math.nextafter(.0031622776601683794, math.inf)
        with self.assertRaisesRegex(ValueError, 'six-price values'):
            self.data()
        self.recipe['prices'][4]['value'] = .0031622776601683794
        self.recipe['figures'][P.FIGURES[0]]['dpi'] = 72
        with self.assertRaises(ValueError):
            self.data()
        _, _, _, self.recipe = full_fixture()
        self.design['conditions'][0]['total_time'] = 0
        with self.assertRaises(ValueError):
            self.data()
        with self.assertRaises(ValueError):
            P.figure_data({}, {}, {}, {})

    def test_render_rejects_incomplete_scope_before_creating_output(self):
        self.true_need['schedules'].pop()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'figures'
            with self.assertRaises(ValueError):
                P.render(self.allocation, self.true_need, self.design, output, self.recipe)
            self.assertFalse(output.exists())

    def test_render_writes_four_pngs_and_vector_pdf_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = P.render(self.allocation, self.true_need, self.design, directory, self.recipe)
            self.assertEqual(set(paths), set(P.FIGURES))
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             sorted(n + ext for n in P.FIGURES for ext in ('.png', '.pdf')))
            for name, path in paths.items():
                self.assertIsInstance(path, Path)
                self.assertEqual(path.name, name + '.png')
                blob = path.read_bytes()
                self.assertEqual(blob[:8], b'\x89PNG\r\n\x1a\n')
                self.assertEqual(blob[12:16], b'IHDR')
                settings = self.recipe['figures'][name]
                self.assertEqual(struct.unpack('>II', blob[16:24]),
                                 tuple(int(x * settings['dpi']) for x in settings['sizeInches']))
                self.assertGreater(len(blob), 1000)
                pdf = path.with_suffix('.pdf').read_bytes()
                self.assertTrue(pdf.startswith(b'%PDF-'))
                self.assertTrue(pdf.rstrip().endswith(b'%%EOF'))
                self.assertIn(b'/Type /Page', pdf)
                self.assertNotIn(b'/Subtype /Image', pdf)
                self.assertGreater(len(pdf), 1000)


if __name__ == '__main__':
    unittest.main()
