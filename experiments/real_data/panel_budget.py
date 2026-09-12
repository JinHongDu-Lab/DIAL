"""Study 3 -- judge panels across budget regimes (manuscript appendix G.4.2).

One study, three figures, one 3 x 3 grid (datasets x judge panels) in all of them:

  fig_small_panel.pdf            human budget n_H sweep, as-collected LLM data
  fig_small_panel_llmbudget.pdf  LLM budget n_L sweep at a fixed n_H (300 / 80 / 60)
  fig_intermediate_budget.pdf    human budget sweep at a fixed n_L (2000 / 160 / 100)

All three share one layout -- judge panels as columns, datasets as rows, and each row labelled
with whichever budget is held fixed. Only these excess-log-loss versions appear in the
manuscript; `--metrics excess,tau` also writes a `_tau` companion in Kendall's tau.

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

from experiments.style import DATASET_LABEL, GRID, LABEL, RCPARAMS, STYLE, mark_better

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
                                  N=('N', 'first'), n_H=('n_H', 'mean'), n_L=('n_L', 'mean'),
                                  n=('seed', 'size')).reset_index()


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
                                     tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem'),
                                     N=('N', 'first')).reset_index()
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
        tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem'), N=('N', 'first'),
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
    fig.savefig(out / f'{name}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out / f'{name}.pdf'


FIGURES = {
    # name -> everything panel_grid needs except the metric, so the CLI and the notebook cannot
    # drift apart: both ask figure_kwargs for the spec rather than spelling the arguments out.
    'fig_small_panel': dict(
        source='budget',
        # n_H is this sweep's x axis, so the row label names the LLM side instead
        row=lambda ds, N: f'{DATASET_LABEL[ds]} ($N={N}$), all LLM data',
        x_column='n_H', xlabel=r'human calibration labels $n_{\mathrm{H}}$'),
    'fig_small_panel_llmbudget': dict(
        source='llm_budget',
        row=lambda ds, N: f'{DATASET_LABEL[ds]} ($N={N}$), $n_H={NH[ds]}$',
        x_column='level', xlabel=r'LLM comparisons $n_{\mathrm{L}}$',
        flat=('human_only',), xticks=False),
    'fig_intermediate_budget': dict(
        source='intermediate',
        row=lambda ds, N: f'{DATASET_LABEL[ds]} ($N={N}$), $n_L={INTERMEDIATE_NL[ds]}$',
        x_column='n_H', xlabel=r'human calibration labels $n_{\mathrm{H}}$'),
}


def figure_kwargs(name, metric, rows):
    """panel_grid keyword arguments for one figure and metric (the single source of truth).

    `rows` is the frame being drawn; the item count N of each dataset is read from it rather
    than hard-coded, so the row labels cannot fall out of step with the fitted panels.
    """
    spec = {k: v for k, v in FIGURES[name].items() if k not in ('source', 'row')}
    label, row = METRIC[metric]['label'], FIGURES[name]['row']
    n_items = rows.groupby('dataset')['N'].agg(lambda x: int(x.dropna().iloc[0])).to_dict()
    return dict(metric=metric, methods=METHODS,
                ylabel=lambda ds: f'{row(ds, n_items[ds])}\n{label}', **spec)


def figure_name(name, metric):
    """File name for one figure and metric; Kendall's tau takes a `_tau` suffix."""
    return name + ('' if metric == 'excess' else '_tau')


def figure_rows(name, summary=None, intermediate=None):
    """The frame `name` is drawn from, selected out of the two loaders' output."""
    source = FIGURES[name]['source']
    return intermediate if source == 'intermediate' else summary[summary.sweep == source]


def make_figures(which=('small', 'llm', 'intermediate'),
                 intermediate_root=ROOT / 'results' / 'intermediate_budget',
                 out=ROOT / 'figures',
                 metrics=('excess',)):
    """Draw and save the requested figures; return {figure name: plotted summary frame}.

    One file per figure and metric. Only the `excess` versions appear in the manuscript, so they
    are the default; `metrics=('excess', 'tau')` also writes the `_tau` companions.
    """
    names = {'small': 'fig_small_panel', 'llm': 'fig_small_panel_llmbudget',
             'intermediate': 'fig_intermediate_budget'}
    wanted = [names[w] for w in which]
    summary = intermediate = None
    drawn = {}
    with plt.rc_context(RCPARAMS):
        if any(FIGURES[n]['source'] != 'intermediate' for n in wanted):
            summary = panel_budget_data()
            Path(out).mkdir(parents=True, exist_ok=True)
            summary.to_csv(Path(out) / 'panel_budget_plot_data.csv', index=False)
        if any(FIGURES[n]['source'] == 'intermediate' for n in wanted):
            intermediate, _ = intermediate_data(intermediate_root)
        for name in wanted:
            rows = figure_rows(name, summary, intermediate)
            for metric in metrics:
                _save(panel_grid(rows, **figure_kwargs(name, metric, rows)), out, figure_name(name, metric))
                drawn[figure_name(name, metric)] = rows
    return drawn



# --------------------------------------------------------------------------- main-text calibration figure
CALIBRATION_DATASET = 'arena_33k'      # the only benchmark whose item set makes tau informative
CALIBRATION_PANEL = 'all'              # the paper's 21-judge panel
HUMAN_EXISTS_LABEL = r'human-only MLE exists ($n_{{\mathrm{{H}}}}\geq{:,}$)'


def human_exists_budgets(ds=CALIBRATION_DATASET):
    """Human budgets at which the unrestricted human MLE exists in every record split.

    Below them the reported human fit is the bounded maximizer, not the MLE, so the main-text
    ranking figure drops those points instead of plotting a mixture of the two. The budget is
    identified by the realized mean n_H, which is what both frames carry (the `budget` sweep
    keys its levels through `level`, the intermediate run through `n_H_level`).
    """
    raw = load(ds, ROOT)
    raw = raw[(raw.sweep == 'budget') & (raw.panel == CALIBRATION_PANEL) & (raw.method == 'human_only')
              & (raw.seed < SEEDS)]
    g = raw.groupby('level').agg(exists=('human_only_exists', 'mean'), n_H=('n_H', 'mean'))
    return {round(float(v)) for v in g.loc[g.exists >= 1, 'n_H']}


def calibration_figure(summary, intermediate, ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL):
    """Main-text figure: human-label efficiency and the anchor--adapt crossover in Kendall's tau.

    (a) the collected LLM judgments of the full judge panel, (b) the same panel at the
    intermediate LLM budget, both against the human calibration budget on one shared y axis.
    """
    keep = human_exists_budgets(ds)
    n_star = min(keep)   # smallest budget at which the unrestricted human MLE exists in every split
    rows = [summary[(summary.sweep == 'budget')], intermediate]
    titles = [f'(a) all LLM judgments', f'(b) $n_{{\\mathrm{{L}}}}={INTERMEDIATE_NL[ds]:,}$']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), sharey=True)
    for ax, frame, title in zip(axes, rows, titles):
        g0 = frame[(frame.dataset == ds) & (frame.panel == panel)]
        for method, label, color, ls, marker in METHODS:
            g = g0[g0.method == method].sort_values('n_H')
            if method == 'human_only':
                g = g[g.n_H.round().isin(keep)]
            if g.empty:
                continue
            lw = 1.45 if method == 'dial_mu' else 1.05
            ax.plot(g.n_H, g.tau, color=color, ls=ls, marker=marker, ms=3, lw=lw, label=label)
            ax.fill_between(g.n_H, g.tau - 1.96 * g.tau_se, g.tau + 1.96 * g.tau_se,
                            color=color, alpha=.085, lw=0)
        # The human MLE exists above n_star regardless of the LLM budget, so the same line is
        # drawn in both panels; left of it the human-only fit is the bounded maximizer and is not
        # plotted.
        ax.axvline(n_star, color=STYLE['human_only']['color'], ls='--', lw=.8, zorder=1,
                   label='_nolegend_' if ax is not axes[0] else HUMAN_EXISTS_LABEL.format(n_star))
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(XTICKS[ds])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r'human calibration labels $n_{\mathrm{H}}$')
        ax.set_title(title, fontsize=8.5)
        ax.grid(color=GRID, lw=.5)
        ax.tick_params(length=2.5, labelsize=6.5)
    axes[0].set_ylabel(r"Kendall's $\tau$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, bbox_to_anchor=(.5, -.12))
    fig.tight_layout()
    mark_better(axes[0], 'up')   # after the layout: the arrow is placed from the rendered label
    return fig

def main():
    matplotlib.use('Agg')                      # CLI is headless; the notebook keeps its own backend
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures', default='all', help="comma-separated: small, llm, intermediate, or all")
    p.add_argument('--intermediate-root', type=Path, default=ROOT / 'results' / 'intermediate_budget')
    p.add_argument('--out', type=Path, default=ROOT / 'figures')
    p.add_argument('--metrics', default='excess', help="comma-separated: excess, tau, or both")
    a = p.parse_args()
    which = ('small', 'llm', 'intermediate') if a.figures == 'all' else tuple(s.strip() for s in a.figures.split(','))

    drawn = make_figures(which, a.intermediate_root, a.out,
                         metrics=tuple(m.strip() for m in a.metrics.split(',')))
    for name, summary in drawn.items():
        print(f'{a.out / name}.pdf: {len(summary)} plotted method/setting rows')


if __name__ == '__main__':
    main()
