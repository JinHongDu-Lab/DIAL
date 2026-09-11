"""Study 3 -- judge panels across budget regimes (manuscript appendix G.4.2).

One study, three figures, one 3 x 3 grid (datasets x judge panels) in all of them:

  fig_small_panel.pdf            human budget n_H sweep, as-collected LLM data
  fig_small_panel_llmbudget.pdf  LLM budget n_L sweep at the fixed default n_H
  fig_intermediate_budget.pdf    human budget sweep at a fixed intermediate n_L

The first two read the 50-seed endpoint-margin run (`results/endpoint_margin_appendix`)
together with the main robustness rows; the third reads an `intermediate_budget` run
directory. Both sources carry the same methods and the same 50 record splits, so the three
figures are directly comparable -- which is why they live in one module rather than three.

    python -m experiments.real_data.panel_budget --figures all
    python -m experiments.real_data.panel_budget --figures intermediate --intermediate-root /tmp/dial-intermediate-budget-50

Data generation stays where it was: `endpoint_margin.py` and `intermediate_budget.py`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dial-panel-budget-matplotlib')

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from experiments.style import DATASET_LABEL, GRID, RCPARAMS

from .robustness import ROOT
from .robustness_plot import load

PANELS = ['small6', 'large6', 'all']
PANEL_TITLE = ['small6 (36.5B)', 'large6 (203B)', 'all, 21 judges (367.5B+, 3 paid APIs)']
NH = {'arena_33k': 300, 'mt_bench': 80, 'pandalm': 60}          # default human budget per dataset
INTERMEDIATE_NL = {'arena_33k': 2000, 'mt_bench': 160, 'pandalm': 100}
XTICKS = {'arena_33k': [50, 200, 800, 3200], 'mt_bench': [20, 80, 320], 'pandalm': [20, 80, 320]}

# (key, label, colour, linestyle, marker); human_only is first and is dropped where it is absent
METHODS = [('human_only', 'Human', '#52514e', '-', None),
           ('consensus_cal', 'Cons-Cal', '#1baf7a', '--', 'v'),
           ('dial_mu', r'DIAL-$\mu$ (raw)', '#4279ad', ':', 's'),
           ('dial_margin_1', r'DIAL-$\mu$', '#c0392b', '-', 'o')]
SEEDS = 50


# --------------------------------------------------------------------------- data
def panel_budget_data(margin_root):
    """Rows behind the human- and LLM-budget figures, as mean excess with its Monte Carlo s.e.

    The adaptive `dial_margin_1` curve comes from the endpoint-margin run, the other methods
    from the main robustness rows; both are checked to cover the same cells and all 50 seeds.
    """
    adaptive = pd.read_csv(Path(margin_root) / 'rows.csv')
    adaptive = adaptive[adaptive.method == 'dial_margin_1'].copy()
    frames = []
    for ds in NH:
        raw = load(ds, ROOT)
        raw = raw[raw.sweep.isin(['budget', 'llm_budget']) & raw.panel.isin(PANELS)
                  & raw.method.isin(['human_only', 'consensus_cal', 'dial_mu']) & (raw.seed < SEEDS)]
        raw = raw[(raw.sweep == 'budget') | ((raw.sweep == 'llm_budget') & (raw.n_H_level == NH[ds]))].copy()
        # Human-only predictions use the same calibration draw for every judge panel.
        human = raw[(raw.panel == 'all') & (raw.method == 'human_only')]
        raw = raw[raw.method != 'human_only']
        raw = pd.concat([raw] + [human.assign(panel=p) for p in PANELS], ignore_index=True)
        raw['dataset'] = ds

        ad = adaptive[(adaptive.dataset == ds) & adaptive.panel.isin(PANELS) & (adaptive.seed < SEEDS)]
        ad = ad[(ad.sweep == 'budget') | ((ad.sweep == 'llm_budget') & (ad.n_H_level == NH[ds]))]
        cells = ['sweep', 'panel', 'level', 'n_H_level', 'seed']
        raw_keys = set(map(tuple, raw[raw.method == 'dial_mu'][cells].to_numpy()))
        ad_keys = set(map(tuple, ad[cells].to_numpy()))
        assert raw_keys == ad_keys, (ds, 'raw/adaptive cell mismatch', len(raw_keys), len(ad_keys))
        frames.extend([raw, ad])

    data = pd.concat(frames, ignore_index=True)
    keys = ['dataset', 'sweep', 'panel', 'level', 'n_H_level', 'method']
    assert not data.duplicated(keys + ['seed']).any()
    assert data.groupby(keys).seed.apply(lambda x: set(x) == set(range(SEEDS))).all()
    return data.groupby(keys).agg(excess=('excess', 'mean'), se=('excess', 'sem'),
                                  n_H=('n_H', 'mean'), n_L=('n_L', 'mean'), n=('seed', 'size')).reset_index()


def intermediate_data(root):
    """Rows behind the intermediate-LLM-budget figure, with the run's own completeness checks."""
    root = Path(root)
    x = pd.read_csv(root / 'rows.csv')
    seeds = json.loads((root / 'design.json').read_text())['seeds']
    keys = ['dataset', 'panel', 'n_H_level', 'method']
    assert not x.duplicated(keys + ['seed']).any()
    assert x.groupby(keys).seed.apply(lambda z: set(z) == set(range(seeds))).all()
    assert len(x) == 57 * 3 * seeds

    base = x[x.method == 'consensus_cal'].set_index(['dataset', 'panel', 'n_H_level', 'seed'])
    z = x.join(base.excess.rename('cons_excess'), on=['dataset', 'panel', 'n_H_level', 'seed'], validate='many_to_one')
    z['delta_vs_cons'] = z.excess - z.cons_excess
    summary = z.reset_index().groupby(keys).agg(
        n_H=('n_H', 'mean'), excess=('excess', 'mean'), se=('excess', 'sem'),
        delta_vs_cons=('delta_vs_cons', 'mean'), paired_mcse=('delta_vs_cons', 'sem')).reset_index()
    return summary, seeds


# --------------------------------------------------------------------------- figure
def panel_grid(summary, x_column, ylabel, xlabel, methods, flat=(), titles=PANEL_TITLE, suptitle=None, xticks=True):
    """The shared 3 x 3 figure: one row per dataset, one column per judge panel.

    `flat` names methods drawn as a horizontal reference line (their value does not vary along
    the x axis of that sweep); the y axis is shared within a row and excludes Human from its
    upper limit, which is off the chart at the smallest budgets.
    """
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6), sharey='row')
    for i, ds in enumerate(NH):
        row = summary[summary.dataset == ds]
        scaled = row[row.method != 'human_only']
        upper = max(.01, float((scaled.excess + 1.96 * scaled.se).max()) * 1.08)
        for j, panel in enumerate(PANELS):
            ax = axes[i, j]
            for method, label, color, ls, marker in methods:
                g = row[(row.panel == panel) & (row.method == method)].sort_values(x_column)
                if g.empty:
                    continue
                if method in flat:
                    ax.axhline(g.excess.mean(), color=color, lw=1, label=label)
                    continue
                lw = 1.45 if method == 'dial_margin_1' else 1.05
                ax.plot(g[x_column], g.excess, color=color, ls=ls, marker=marker, ms=2.5, lw=lw, label=label)
                if method != 'human_only':
                    ax.fill_between(g[x_column], g.excess - 1.96 * g.se, g.excess + 1.96 * g.se,
                                    color=color, alpha=.085, lw=0)
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(NullFormatter())
            if xticks:
                ax.set_xticks(XTICKS[ds])
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylim(-.002, upper)
            ax.grid(color=GRID, lw=.5)
            ax.tick_params(length=2.5, labelsize=6.5)
            if i == 0:
                ax.set_title(titles[j], fontsize=7.7)
            if j == 0:
                ax.set_ylabel(ylabel(ds))
            if i == 2:
                ax.set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, bbox_to_anchor=(.5, -.005))
    if suptitle:
        fig.suptitle(suptitle, fontsize=8.5, y=.995)
    fig.tight_layout(rect=(0, .03, 1, .975 if suptitle else 1))
    return fig


def _save(fig, out, name):
    """Save inside the caller's rc context: the tight bbox and embedded fonts depend on it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out / f'{name}.pdf'


def make_figures(which=('small', 'llm', 'intermediate'),
                 margin_root=ROOT / 'results' / 'endpoint_margin_appendix',
                 intermediate_root=ROOT / 'results' / 'intermediate_budget',
                 out=ROOT / 'figures'):
    """Draw and save the requested figures; return {figure name: plotted summary frame}."""
    drawn = {}
    with plt.rc_context(RCPARAMS):
        if {'small', 'llm'} & set(which):
            summary = panel_budget_data(margin_root)
            Path(out).mkdir(parents=True, exist_ok=True)
            summary.to_csv(Path(out) / 'panel_budget_plot_data.csv', index=False)
            if 'small' in which:
                rows = summary[summary.sweep == 'budget']
                fig = panel_grid(rows, 'n_H', lambda ds: f'{DATASET_LABEL[ds]}\nexcess held-out log loss',
                            r'human calibration labels $n_{\mathrm{H}}$', METHODS)
                _save(fig, out, 'fig_small_panel')
                drawn['fig_small_panel'] = rows
            if 'llm' in which:
                rows = summary[summary.sweep == 'llm_budget']
                fig = panel_grid(rows, 'level', lambda ds: f'{DATASET_LABEL[ds]}\nexcess held-out log loss',
                            r'LLM comparisons $n_{\mathrm{L}}$', METHODS, flat=('human_only',),
                            suptitle=r'Fixed human budget $n_{\mathrm{H}}=300\,/\,80\,/\,60$', xticks=False)
                _save(fig, out, 'fig_small_panel_llmbudget')
                drawn['fig_small_panel_llmbudget'] = rows
        if 'intermediate' in which:
            rows, seeds = intermediate_data(intermediate_root)
            fig = panel_grid(rows, 'n_H',
                        lambda ds: f'{DATASET_LABEL[ds]}, $n_L={INTERMEDIATE_NL[ds]}$\nexcess held-out log loss',
                        r'human calibration labels $n_{\mathrm{H}}$', METHODS[1:],
                        titles=PANELS, suptitle=f'Intermediate LLM budget: {seeds} matched seeds')
            _save(fig, out, 'fig_intermediate_budget')
            drawn['fig_intermediate_budget'] = rows
    return drawn


def main():
    matplotlib.use('Agg')                      # CLI is headless; the notebook keeps its own backend
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures', default='all', help="comma-separated: small, llm, intermediate, or all")
    p.add_argument('--margin-root', type=Path, default=ROOT / 'results' / 'endpoint_margin_appendix')
    p.add_argument('--intermediate-root', type=Path, default=ROOT / 'results' / 'intermediate_budget')
    p.add_argument('--out', type=Path, default=ROOT / 'figures')
    a = p.parse_args()
    which = ('small', 'llm', 'intermediate') if a.figures == 'all' else tuple(s.strip() for s in a.figures.split(','))

    drawn = make_figures(which, a.margin_root, a.intermediate_root, a.out)
    for name, summary in drawn.items():
        print(f'{a.out / name}.pdf: {len(summary)} plotted method/setting rows')


if __name__ == '__main__':
    main()
