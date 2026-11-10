"""Judge panels across budget regimes: the data behind the calibration figures.

Appendix grids (datasets x judge panels), read from the robustness rows:

  fig_panel_human_budget.pdf      human budget n_H sweep, as-collected LLM data
  fig_panel_llm_budget.pdf        LLM budget n_L sweep at a fixed n_H (300 / 80 / 60)

The main-text calibration figure draws the Arena `all` cells of the human-budget sweep and of the
`intermediate_budget` run (fixed n_L = 2000) on one Kendall-tau axis.

This module only loads and aggregates per-cell means with their Monte Carlo s.e.; the figures are
drawn in `notebooks/3_real_data_calibration.ipynb`.
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
    # replications with no admissible GACV candidate carry no usable score (robustness_plot.load)
    data = data[~data.failed]
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
    if 'all_dropped' in x:                            # no admissible GACV candidate: no usable score
        dropped = x['all_dropped'].fillna(False).astype(bool)
        x = x[~dropped]
    seeds = json.loads((root / 'design.json').read_text())['seeds']
    keys = ['dataset', 'panel', 'n_H_level', 'method']
    assert not x.duplicated(keys + ['seed']).any()
    assert x.groupby(keys).seed.apply(lambda z: set(z) <= set(range(seeds))).all()
    assert len(x) <= 57 * 2 * seeds

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

# Alternative estimators drawn with the three DIAL-study methods and AtC (experiments/style.py
# `ALTERNATIVES`): PPI++ with the panel-majority proxy, the ridge-to-consensus shrinkage fit, and the
# Firth-regularized human and AtC fits, which exist at every budget.
ALTERNATIVE = ['ppi_bt_panel', 'ridge_mu', 'human_firth', 'atc_firth']
ALTERNATIVE_ROOT = ROOT / 'results' / 'alternatives'


def _alternative_agg(raw, level_column):
    return raw.groupby([level_column, 'method']).agg(n_H=('n_H', 'mean'), excess=('excess', 'mean'), se=('excess', 'sem'),
                                                      tau=('ref_kendall', 'mean'), tau_se=('ref_kendall', 'sem'),
                                                      N=('N', 'first'), n_fits=('seed', 'size')).reset_index()


def alternative_rows(ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL, methods=ALTERNATIVE):
    """Means and Monte Carlo s.e. of the alternative estimators on the robustness `budget` sweep.

    Read from results/alternatives/budget, which holds only these methods on the stored splits.
    `n_fits` counts the finite fits per budget: PPI++ has no finite fit in some splits at the
    smallest budgets, and the figures draw it only where all 50 are finite.
    """
    raw = load(ds, ROOT, results='alternatives/budget')
    raw = raw[(raw.sweep == 'budget') & (raw.panel == panel) & raw.method.isin(methods) & (raw.seed < SEEDS) & (~raw.failed)]
    return _alternative_agg(raw, 'level').assign(dataset=ds, panel=panel)


def llm_only_tau(ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL):
    """Kendall's tau of the LLM-only consensus ranking (zero human budget, `budget_full` level 0)."""
    raw = load(ds, ROOT, results='alternatives/budget')
    raw = raw[(raw.panel == panel) & (raw.method == 'llm_only') & (raw.level == 0) & (raw.seed < SEEDS) & (~raw.failed)]
    assert len(raw) == SEEDS
    return float(raw.ref_kendall.mean())


def llm_only_raw_tau(ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL, n_L=None):
    """Kendall's tau of the uncalibrated LLM-only consensus (`llm_only_raw`: DIAL's LLM-only fit, both
    display orders), from the robustness rows: with all LLM judgments (`budget` sweep) when
    `n_L` is None, else at that LLM budget (`llm_budget` sweep). It uses no human labels, so every
    human-budget cell of a split carries the same value; one cell per split is kept. The LLM
    subsample of an `llm_budget` cell depends only on the split seed, so it is the subsample of the
    intermediate-budget run at the same n_L."""
    raw = load(ds, ROOT)
    raw = raw[(raw.panel == panel) & (raw.method == 'llm_only_raw') & (raw.seed < SEEDS) & (~raw.failed)]
    raw = raw[raw.sweep == 'budget'] if n_L is None else raw[(raw.sweep == 'llm_budget') & (raw.level == n_L)]
    raw = raw.drop_duplicates('seed')
    assert len(raw) == SEEDS, len(raw)
    return float(raw.ref_kendall.mean())


def alternative_intermediate_rows(ds=CALIBRATION_DATASET, panel=CALIBRATION_PANEL, methods=ALTERNATIVE):
    """The same estimators on the intermediate-LLM-budget run (its own `--tag alternatives` pass)."""
    path = ALTERNATIVE_ROOT / 'intermediate_budget' / 'jobs_alternatives.jsonl'
    x = pd.DataFrame([r for line in path.open() if line.strip() for r in json.loads(line)['rows']])
    x = x[(x.dataset == ds) & (x.panel == panel) & x.method.isin(methods) & (x.seed < SEEDS) & x.error.isna()]
    return _alternative_agg(x, 'n_H_level').assign(dataset=ds, panel=panel)


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
    """The two frames the main-text calibration figure draws, each with its AtC pass and the
    alternative estimators appended.

    (a) the collected LLM judgments of the full judge panel, (b) the same panel at the
    intermediate LLM budget.
    """
    intermediate_root = ROOT / 'results' / 'intermediate_budget' if intermediate_root is None else intermediate_root
    atc = [atc_rows(load(ds, ROOT), 'level', sweep='budget', ds=ds, panel=panel),
           atc_rows(intermediate_root, 'n_H_level', ds=ds, panel=panel)]
    alt = [alternative_rows(ds=ds, panel=panel), alternative_intermediate_rows(ds=ds, panel=panel)]
    return [pd.concat([summary[summary.sweep == 'budget'], atc[0], alt[0]], ignore_index=True),
            pd.concat([intermediate, atc[1], alt[1]], ignore_index=True)]
