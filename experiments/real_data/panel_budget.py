"""Study 3 -- judge panels across budget regimes (manuscript appendix G.4.2).

One study, three figures, one 3 x 3 grid (datasets x judge panels) in all of them:

  fig_small_panel.pdf            human budget n_H sweep, as-collected LLM data
  fig_small_panel_llmbudget.pdf  LLM budget n_L sweep at a fixed n_H (300 / 80 / 60)
  fig_intermediate_budget.pdf    human budget sweep at a fixed n_L (2000 / 160 / 100)

All three share one layout -- judge panels as columns, datasets as rows, and each row labelled
with whichever budget is held fixed -- plus a `_tau` companion in Kendall's tau.

The first two read the main robustness rows; the third reads an `intermediate_budget` run
directory. Both sources carry the same three estimators of `experiments/style.py` -- Human,
Cons-Cal, DIAL-mu with its GACV weight -- over the same 50 record splits, so the figures are
directly comparable, which is why they live in one module rather than three.

    python -m experiments.real_data.panel_budget --figures all
    python -m experiments.real_data.panel_budget --figures intermediate --intermediate-root /tmp/dial-intermediate-budget-50

Data generation stays where it was: `run_real_data.py --study robustness` and
`intermediate_budget.py`.
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

from experiments.style import DATASET_LABEL, GRID, LABEL, RCPARAMS, STYLE

from .robustness import ROOT
from .robustness_plot import load

PANELS = ['small6', 'large6', 'all']
PANEL_TITLE = ['small6 (36.5B)', 'large6 (203B)', 'all, 21 judges (367.5B+, 3 paid APIs)']
NH = {'arena_33k': 300, 'mt_bench': 80, 'pandalm': 60}          # human budget held fixed in the LLM sweep
INTERMEDIATE_NL = {'arena_33k': 2000, 'mt_bench': 160, 'pandalm': 100}
XTICKS = {'arena_33k': [50, 200, 800, 3200], 'mt_bench': [20, 80, 320], 'pandalm': [20, 80, 320]}

# Names, colours and dashes come from the shared panel of experiments/style.py, so this study
# draws the same three estimators the simulation and robustness studies do. human_only is first
# and is dropped where it is absent.
METHOD_KEYS = ['human_only', 'consensus_cal', 'dial_mu']
METHODS = [(k, LABEL[k], STYLE[k]['color'], STYLE[k]['ls'], STYLE[k]['marker']) for k in METHOD_KEYS]
SEEDS = 50

# Both metrics are recorded per cell by every runner; `excess` is minimized, `tau` maximized,
# so they need different y limits but nothing else.
METRIC = {
    'excess': dict(value='excess', err='se', label='excess held-out log loss'),
    'tau': dict(value='tau', err='tau_se', label=r"Kendall's $\tau$, held-out ranking"),
}


# --------------------------------------------------------------------------- data
def panel_budget_data():
    """Rows behind the human- and LLM-budget figures, as means with their Monte Carlo s.e.

    All three methods come from the main robustness rows, which are checked to cover every
    cell with all 50 seeds.
    """
    frames = []
    for ds in NH:
        raw = load(ds, ROOT)
        raw = raw[raw.sweep.isin(['budget', 'llm_budget']) & raw.panel.isin(PANELS)
                  & raw.method.isin(METHOD_KEYS) & (raw.seed < SEEDS)]
        raw = raw[(raw.sweep == 'budget') | ((raw.sweep == 'llm_budget') & (raw.n_H_level == NH[ds]))].copy()
        # Human-only predictions use the same calibration draw for every judge panel.
        human = raw[(raw.panel == 'all') & (raw.method == 'human_only')]
        raw = raw[raw.method != 'human_only']
        raw = pd.concat([raw] + [human.assign(panel=p) for p in PANELS], ignore_index=True)
        raw['dataset'] = ds
        frames.append(raw)

    data = pd.concat(frames, ignore_index=True)
    keys = ['dataset', 'sweep', 'panel', 'level', 'n_H_level', 'method']
    assert not data.duplicated(keys + ['seed']).any()
    assert data.groupby(keys).seed.apply(lambda x: set(x) == set(range(SEEDS))).all()
    return data.groupby(keys).agg(excess=('excess', 'mean'), se=('excess', 'sem'),
                                  tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem'),
                                  n_H=('n_H', 'mean'), n_L=('n_L', 'mean'), n=('seed', 'size')).reset_index()


def human_only_curve():
    """Human-only excess and tau against the human budget, from the robustness `budget` sweep.

    The human-only fit never touches the LLM sample, and both the record split and the
    calibration draw key off [seed, dataset] and n_H alone, so this curve is the same in every
    sweep and on every judge panel -- verified exactly against the llm_budget rows. That is why
    the intermediate run, which fitted only the two calibrated estimators, can borrow it.
    """
    frames = []
    for ds in NH:
        raw = load(ds, ROOT)
        raw = raw[(raw.sweep == 'budget') & (raw.method == 'human_only') & (~raw.failed) & (raw.seed < SEEDS)
                  & (raw.panel == 'all')]
        g = raw.groupby('level').agg(n_H=('n_H', 'mean'), excess=('excess', 'mean'), se=('excess', 'sem'),
                                     tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem')).reset_index()
        frames.append(g.assign(dataset=ds, method='human_only').rename(columns={'level': 'n_H_level'}))
    return pd.concat(frames, ignore_index=True)


def intermediate_data(root):
    """Rows behind the intermediate-LLM-budget figure, with the run's own completeness checks."""
    root = Path(root)
    x = pd.read_csv(root / 'rows.csv')
    x = x[x.method.isin(METHOD_KEYS)].copy()          # the run also stores endpoint-margin variants
    seeds = json.loads((root / 'design.json').read_text())['seeds']
    keys = ['dataset', 'panel', 'n_H_level', 'method']
    assert not x.duplicated(keys + ['seed']).any()
    assert x.groupby(keys).seed.apply(lambda z: set(z) == set(range(seeds))).all()
    assert len(x) == 57 * 2 * seeds

    base = x[x.method == 'consensus_cal'].set_index(['dataset', 'panel', 'n_H_level', 'seed'])
    z = x.join(base.excess.rename('cons_excess'), on=['dataset', 'panel', 'n_H_level', 'seed'], validate='many_to_one')
    z['delta_vs_cons'] = z.excess - z.cons_excess
    summary = z.reset_index().groupby(keys).agg(
        n_H=('n_H', 'mean'), excess=('excess', 'mean'), se=('excess', 'sem'),
        tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem'),
        delta_vs_cons=('delta_vs_cons', 'mean'), paired_mcse=('delta_vs_cons', 'sem')).reset_index()

    # the run fitted only the calibrated estimators, so add the panel-independent human reference
    human = human_only_curve()
    human = human[human.n_H_level.isin(summary.n_H_level.unique())]
    summary = pd.concat([summary] + [human.assign(panel=p) for p in PANELS], ignore_index=True)
    return summary, seeds


# --------------------------------------------------------------------------- figure
def panel_grid(summary, x_column, ylabel, xlabel, methods, flat=(), titles=PANEL_TITLE, suptitle=None,
               xticks=True, metric='excess'):
    """The shared 3 x 3 figure: one row per dataset, one column per judge panel.

    `metric` selects the plotted column through `METRIC`: 'excess' (held-out log loss, lower is
    better, y starts at zero) or 'tau' (Kendall's tau against the held-out human ranking, higher
    is better, y limits follow the data and stop at 1). `flat` names methods drawn as a
    horizontal reference line, their value not varying along that sweep's x axis. The y axis is
    shared within a row; for log loss it excludes Human from its upper limit, which is off the
    chart at the smallest budgets.
    """
    value, err = METRIC[metric]['value'], METRIC[metric]['err']
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6), sharey='row')
    for i, ds in enumerate(NH):
        row = summary[summary.dataset == ds]
        if metric == 'excess':
            scaled = row[row.method != 'human_only']
            limits = (-.002, max(.01, float((scaled[value] + 1.96 * scaled[err]).max()) * 1.08))
        else:
            lo = float((row[value] - 1.96 * row[err]).min())
            hi = float((row[value] + 1.96 * row[err]).max())
            pad = .04 * max(hi - lo, .05)
            limits = (lo - pad, min(1.0, hi + pad))
        for j, panel in enumerate(PANELS):
            ax = axes[i, j]
            for method, label, color, ls, marker in methods:
                g = row[(row.panel == panel) & (row.method == method)].sort_values(x_column)
                if g.empty:
                    continue
                if method in flat:
                    # dash-dotted so the reference reads as a series rather than a grid line
                    ax.axhline(g[value].mean(), color=color, lw=1.2, ls='-.', label=label)
                    continue
                lw = 1.45 if method == 'dial_mu' else 1.05
                ax.plot(g[x_column], g[value], color=color, ls=ls, marker=marker, ms=2.5, lw=lw, label=label)
                if method != 'human_only':
                    ax.fill_between(g[x_column], g[value] - 1.96 * g[err], g[value] + 1.96 * g[err],
                                    color=color, alpha=.085, lw=0)
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(NullFormatter())
            if xticks:
                ax.set_xticks(XTICKS[ds])
                ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylim(*limits)
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
        fig.suptitle(suptitle, fontsize=8.5, y=.975 if metric == 'tau' else .995)
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
                 intermediate_root=ROOT / 'results' / 'intermediate_budget',
                 out=ROOT / 'figures',
                 metrics=('excess', 'tau')):
    """Draw and save the requested figures; return {figure name: plotted summary frame}.

    Each figure is drawn once per metric; the Kendall-tau versions take a `_tau` suffix.
    """
    drawn = {}

    def draw(rows, name, metric, **kwargs):
        name += '' if metric == 'excess' else '_tau'
        _save(panel_grid(rows, metric=metric, **kwargs), out, name)
        drawn[name] = rows

    with plt.rc_context(RCPARAMS):
        if {'small', 'llm'} & set(which):
            summary = panel_budget_data()
            Path(out).mkdir(parents=True, exist_ok=True)
            summary.to_csv(Path(out) / 'panel_budget_plot_data.csv', index=False)
            for metric in metrics:
                label = METRIC[metric]['label']
                if 'small' in which:
                    # n_H is this sweep's x axis, so the row label names the LLM side instead
                    draw(summary[summary.sweep == 'budget'], 'fig_small_panel', metric,
                         x_column='n_H', ylabel=lambda ds, _l=label: f'{DATASET_LABEL[ds]}, all LLM data\n{_l}',
                         xlabel=r'human calibration labels $n_{\mathrm{H}}$', methods=METHODS)
                if 'llm' in which:
                    # the fixed budget goes in the row label, as in the intermediate figure
                    draw(summary[summary.sweep == 'llm_budget'], 'fig_small_panel_llmbudget', metric,
                         x_column='level',
                         ylabel=lambda ds, _l=label: f'{DATASET_LABEL[ds]}, $n_H={NH[ds]}$\n{_l}',
                         xlabel=r'LLM comparisons $n_{\mathrm{L}}$', methods=METHODS, flat=('human_only',),
                         xticks=False)
        if 'intermediate' in which:
            rows, seeds = intermediate_data(intermediate_root)
            for metric in metrics:
                label = METRIC[metric]['label']
                draw(rows, 'fig_intermediate_budget', metric,
                     x_column='n_H',
                     ylabel=lambda ds, _l=label: f'{DATASET_LABEL[ds]}, $n_L={INTERMEDIATE_NL[ds]}$\n{_l}',
                     xlabel=r'human calibration labels $n_{\mathrm{H}}$', methods=METHODS)
    return drawn


def main():
    matplotlib.use('Agg')                      # CLI is headless; the notebook keeps its own backend
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures', default='all', help="comma-separated: small, llm, intermediate, or all")
    p.add_argument('--intermediate-root', type=Path, default=ROOT / 'results' / 'intermediate_budget')
    p.add_argument('--out', type=Path, default=ROOT / 'figures')
    a = p.parse_args()
    which = ('small', 'llm', 'intermediate') if a.figures == 'all' else tuple(s.strip() for s in a.figures.split(','))

    drawn = make_figures(which, a.intermediate_root, a.out)
    for name, summary in drawn.items():
        print(f'{a.out / name}.pdf: {len(summary)} plotted method/setting rows')


if __name__ == '__main__':
    main()
