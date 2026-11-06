"""Scientific plots of complete, finite proxy allocation/welfare reductions."""
from pathlib import Path

import numpy as np

from . import personal_time_inputs as I
from .personal_time_sensitivity import RATES


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    summary = I.read(output / 'summary.json')
    I.need(summary['fullFiniteTarget'] and summary['verifiedTrials'] == 2700,
           'Figures require the complete finite roster, not smoke output')
    rows = {(r['condition'], r['price']): r for r in summary['conditions']}
    I.need(len(rows) == 30, 'All condition/price summaries')
    labels = ['0', '1e-5', '1e-4', '.001', '.00316', '.01']
    colors = ['#255c89', '#087f77', '#b75f19', '#607a40', '#a94350']
    matrix = lambda key: np.asarray([[rows[c, p][key] for p, _ in RATES]
                                    for c in I.CONDITIONS], dtype=float)

    def heat(ax, key, title, cmap, bounds):
        data = matrix(key)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad('#e9edf0')
        artist = ax.imshow(data, aspect='auto', cmap=cm, vmin=bounds[0], vmax=bounds[1])
        ax.set_xticks(range(6), labels)
        ax.set_yticks(range(5), I.CONDITIONS)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Price (combined GPA / personal second)')
        for i in range(5):
            for j in range(6):
                value = data[i, j]
                ax.text(j, i, 'U' if np.isnan(value) else f'{value:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if np.isfinite(value) and abs(value) > .7*max(map(abs, bounds)) else '#18232b')
        return artist

    files = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, key, title in zip(axes[0], ('expected_u', 'realized_u'),
                                 ('Posterior expected recipient utility', 'Target-key realized recipient utility')):
            for i, c in enumerate(I.CONDITIONS):
                ax.plot(range(6), matrix(key)[i], 'o', color=colors[i], label=c)
            ax.set_xticks(range(6), labels)
            ax.set_xlabel('Price (combined GPA / personal second)')
            ax.set_title(title, loc='left')
            ax.set_ylabel('Sum of recipient utilities')
        axes[0, 0].legend(ncols=5, frameon=False, fontsize=8)
        heat(axes[1, 0], 'initial_ask', 'Initial asking fraction', 'Blues', (0, 1))
        heat(axes[1, 1], 'mean_paid', 'Mean paid questions', 'Blues', (0, max(1., float(matrix('mean_paid').max()))))
        fig.suptitle('Finite proxy outcomes: 90 endpoints per cell, nine equal schedule weights')
        path = output / 'welfare_and_search.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path.name)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        beta = matrix('beta_high_need')
        limit = max(1., float(np.nanmax(np.abs(beta))))
        heat(axes[1], 'beta_high_need', 'Condition by price', 'RdBu_r', (-limit, limit))
        axes[0].axhline(0, color='#687782', linewidth=1)
        for i, c in enumerate(I.CONDITIONS):
            value = rows[c, 'rho_0.001']['beta_high_need']
            if value is None:
                axes[0].text(i, 0, 'U', ha='center')
            else:
                axes[0].bar(i, value, color=colors[i])
        axes[0].set_xticks(range(5), I.CONDITIONS)
        axes[0].set_title('At price .001', loc='left')
        axes[0].set_ylabel('Higher-initial-need share slope')
        fig.suptitle('Rank-oriented initial need response; U = undefined, not zero')
        path = output / 'allocation_slopes.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path.name)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, key, title in zip(axes.ravel(), ('ES_pp', 'EO_pp', 'OPT_pp', 'AL_pp'),
                ('Equal split', 'Equal mean surplus', 'Representative expected-utility maximizer', 'All to lower terminal mean need')):
            heat(ax, key, title, 'Blues', (0, 100))
        fig.suptitle('Matched terminal allocation distances (percentage points), not welfare regret')
        path = output / 'benchmark_distances.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(path.name)
    return files
