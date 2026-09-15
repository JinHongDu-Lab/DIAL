"""Study 3 -- judge panels across budget regimes (manuscript appendix G.4.2): the data behind the figures.

One study, three figures, one 3 x 3 grid (datasets x judge panels) in all of them:

  fig_panel_human_budget.pdf      human budget n_H sweep, as-collected LLM data
  fig_panel_llm_budget.pdf        LLM budget n_L sweep at a fixed n_H (300 / 80 / 60)
  fig_panel_intermediate_llm.pdf  human budget sweep at a fixed n_L (2000 / 160 / 100)

plus the main-text calibration figure, which draws the Arena `all` cells of the first and third
on one Kendall-tau axis.

This module is data only: the loaders below return tidy per-cell means with their Monte Carlo
s.e., and every figure is drawn in `notebooks/3_real_data_calibration.ipynb`, which is the only
place figure code lives. It imports nothing from the fitting library, so the notebook runs from
the stored result rows alone.

The first two figures read the main robustness rows; the third reads an `intermediate_budget`
run directory. Both sources carry the same three estimators of `experiments/style.py` -- Human,
Cons-Cal, DIAL-mu with its GACV weight -- over the same 50 record splits, so the figures are
directly comparable, which is why they load through one module rather than three.

Data generation stays where it was: `run_real_data.py --study robustness` and
`intermediate_budget.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .robustness_plot import load

ROOT = Path(__file__).resolve().parents[2]

PANELS = ['small6', 'large6', 'all']
NH = {'arena_33k': 300, 'mt_bench': 80, 'pandalm': 60}          # human budget held fixed in the LLM sweep
INTERMEDIATE_NL = {'arena_33k': 2000, 'mt_bench': 160, 'pandalm': 100}

# The three estimators every figure of this study draws, in the order of `experiments/style.py`:
# human_only is first and is dropped where it is absent.
METHOD_KEYS = ['human_only', 'consensus_cal', 'dial_mu']
SEEDS = 50


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


def intermediate_data(root=None):
    """Rows behind the intermediate-LLM-budget figure, with the run's own completeness checks."""
    root = Path(ROOT / 'results' / 'intermediate_budget' if root is None else root)
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


# --------------------------------------------------------------------------- main-text calibration figure
CALIBRATION_DATASET = 'arena_33k'      # the only benchmark whose item set makes tau informative
CALIBRATION_PANEL = 'all'              # the paper's 21-judge panel
ATC = 'atc_btl'                        # stage-matched external baseline, main-text figure only


def human_exists_budgets(ds=CALIBRATION_DATASET):
    """Human budgets at which the unrestricted human MLE exists in every record split.

    Below them the reported human fit is the bounded maximizer, not the MLE, so the main-text
    ranking figure drops those points instead of plotting a mixture of the two. The budget is
    identified by the realized mean n_H, which is what both frames carry (the `budget` sweep
    keys its levels through `level`, the intermediate run through `n_H_level`). AtC aggregates the
    same comparisons with the same fit, so it is gated on the same budgets.
    """
    raw = load(ds, ROOT)
    raw = raw[(raw.sweep == 'budget') & (raw.panel == CALIBRATION_PANEL) & (raw.method == 'human_only')
              & (raw.seed < SEEDS)]
    g = raw.groupby('level').agg(exists=('human_only_exists', 'mean'), n_H=('n_H', 'mean'))
    return {round(float(v)) for v in g.loc[g.exists >= 1, 'n_H']}


def atc_rows(source, level_column, sweep=None, ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL):
    """AtC-BTL means and Monte Carlo s.e. for one sweep, in the columns the calibration figure plots.

    The AtC cells were run as their own pass over the same splits, so they are loaded here rather
    than through `panel_budget_data` / `intermediate_data`, whose completeness assertions cover
    only the three estimators those figures draw. `source` is either the raw robustness rows or an
    intermediate-run directory, and `level_column` names that frame's human-budget level (`level`
    in the robustness `budget` sweep, whose `n_H_level` is unset, `n_H_level` in the LLM-budget
    runs). Every budget must carry all 50 seeds.
    """
    raw = pd.read_csv(Path(source) / 'rows.csv') if isinstance(source, (str, Path)) else source
    raw = raw[(raw.dataset == ds) & (raw.panel == panel) & (raw.method == ATC) & (raw.seed < SEEDS)]
    if sweep is not None:
        raw = raw[raw.sweep == sweep]
    if 'failed' in raw:
        raw = raw[~raw.failed]
    assert not raw.empty and raw.groupby(level_column).seed.apply(lambda x: set(x) == set(range(SEEDS))).all()
    g = raw.groupby(level_column).agg(n_H=('n_H', 'mean'), excess=('excess', 'mean'), se=('excess', 'sem'),
                                      tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem'),
                                      N=('N', 'first')).reset_index()
    return g.assign(dataset=ds, panel=panel, method=ATC)


def calibration_rows(summary, intermediate, ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL,
                     intermediate_root=None):
    """The two frames the main-text calibration figure draws, each with its AtC pass appended.

    (a) the collected LLM judgments of the full judge panel, (b) the same panel at the
    intermediate LLM budget.
    """
    intermediate_root = ROOT / 'results' / 'intermediate_budget' if intermediate_root is None else intermediate_root
    atc = [atc_rows(load(ds, ROOT), 'level', sweep='budget', ds=ds, panel=panel),
           atc_rows(intermediate_root, 'n_H_level', ds=ds, panel=panel)]
    return [pd.concat([summary[summary.sweep == 'budget'], atc[0]], ignore_index=True),
            pd.concat([intermediate, atc[1]], ignore_index=True)]
