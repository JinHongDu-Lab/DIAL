"""Aggregation of the real-data results; the figures are drawn in the notebooks.

`load` / `aggregate` / `aggregate_spectest`: per (sweep, panel, kind, level, n_H_level, method)
mean and Monte Carlo s.e. of the held-out metrics, the paired difference to human-only on the same
split, endpoint shares of the selected weight, selected rank, and failure counts
(notebooks/2_real_data_position.ipynb).

`panel_budget_data`, `intermediate_data`, `calibration_rows`, ...: the rows behind the
human-calibration figures (notebooks/3_real_data_calibration.ipynb): the human-budget and
LLM-budget grids over datasets and judge panels, and the main-text calibration figure, which draws
the Arena `all` cells of the human-budget sweep and of the intermediate-LLM-budget run on one
Kendall-tau axis.

`dataset_stats`: the dataset-statistics table after preprocessing.

    python -m experiments.real_data.aggregate    # results/real_robustness/<dataset>/summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_LABEL = {"arena_33k": "Chatbot Arena", "mt_bench": "MT-Bench", "pandalm": "PandaLM"}

KEY = ["sweep", "panel", "kind", "level", "n_H_level"]
METRICS = ["excess", "logloss", "acc", "ref_kendall", "b_mean_abs", "b_inj_mean_abs", "gamma_inj_mean_abs", "lam_rel", "r_sel", "first_share", "n_L", "n_H"]
MAIN_METHODS = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_nodeb"]


def load(dataset, base=".", results="real_robustness"):
    """Rows of one dataset under results/<results>/; `results` names the run directory, e.g.
    `alternatives/budget` for the alternative estimators."""
    path = Path(base) / "results" / results / dataset / "rows.jsonl"
    with open(path) as handle:
        df = pd.DataFrame([json.loads(line) for line in handle])
    for old, new in (("n_0", "n_H"), ("n_0_level", "n_H_level")):  # pre-rename rows.jsonl; coalesce if both schemas are present
        if old in df.columns:
            df[new] = df[new].combine_first(df[old]) if new in df.columns else df[old]
            df = df.drop(columns=old)
    for c in METRICS + ["error", "converged", "lam", "human_only_exists", "n_H_level", "n_at_bound", "b_at_bound", "K_dropped"]:
        if c not in df:
            df[c] = np.nan
    df["n_H_level"] = df["n_H_level"].fillna(-1).astype(int)
    if "all_dropped" not in df:
        df["all_dropped"] = False
    df["all_dropped"] = df["all_dropped"].fillna(False).astype(bool)
    # a replication in which every GACV candidate failed a guard has no admissible fit, so it is
    # reported as failed and excluded from the aggregated means (it is counted in `n_failed`)
    df["failed"] = df["error"].notna() | df["all_dropped"]
    df["lam_rel"] = df["lam_rel"].replace([np.inf, -np.inf], np.nan)
    for sw in ("noise", "noise_scarce"):
        noise0 = df[(df.sweep == sw) & (df.level == 0)]
        if not noise0.empty:
            kinds = sorted(set(df.loc[df.sweep == sw, "kind"]) - set(noise0["kind"]))
            df = pd.concat([df] + [noise0.assign(kind=k) for k in kinds], ignore_index=True)
    return df


def _se(x):
    x = x.dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def aggregate(df):
    df = df[~df.method.isin(["spectest", "planner"])]   # "planner": a retired method in older rows.jsonl files
    ok = df[~df.failed]
    g = ok.groupby(KEY + ["method"])
    agg = g.agg(n=("seed", "size"), K=("K", "first"), n_test=("n_test", "first"), floor=("floor", "mean"),
                **{f"{m}_mean": (m, "mean") for m in METRICS}, **{f"{m}_se": (m, _se) for m in METRICS},
                lam_rel_med=("lam_rel", "median"),
                lam_inf_frac=("lam", lambda x: float(np.mean(np.isinf(x.astype(float))))),
                lam_zero_frac=("lam", lambda x: float(np.mean(x.astype(float) == 0))),
                nonconv_frac=("converged", lambda x: float(np.mean(~x.fillna(True).astype(bool)))),
                sep_frac=("n_at_bound", lambda x: float(np.mean(x.fillna(0).astype(float) > 0))),      # fit touched the finite-fit box (a separated judge)
                bsep_frac=("b_at_bound", lambda x: float(np.mean(x.fillna(0).astype(float) > 0))),    # a position effect at the bound
                k_dropped_mean=("K_dropped", lambda x: float(np.nanmean(x.astype(float))) if x.notna().any() else 0.0),
                ho_exists_frac=("human_only_exists", lambda x: float(np.mean(x.fillna(True).astype(bool))))).reset_index()
    ho = ok[ok.method == "human_only"][KEY + ["seed", "excess"]].rename(columns={"excess": "ex_ho"})
    paired = ok.merge(ho, on=KEY + ["seed"], how="inner")
    paired["d_excess"] = paired["excess"] - paired["ex_ho"]
    pd_agg = paired.groupby(KEY + ["method"]).agg(d_excess_mean=("d_excess", "mean"), d_excess_se=("d_excess", _se)).reset_index()
    agg = agg.merge(pd_agg, on=KEY + ["method"], how="left")
    failed = df[df.failed].groupby(KEY + ["method"]).size().rename("n_failed").reset_index()
    return agg.merge(failed, on=KEY + ["method"], how="left").fillna({"n_failed": 0})


def aggregate_spectest(df):
    s = df[df.method == "spectest"]
    if s.empty:
        return s
    return s.groupby(["kind", "level"]).agg(n=("seed", "size"), reject_rate=("reject05", "mean"), stat_mean=("stat", "mean"), df=("df", "first"),
                                            ho_exists=("human_only_exists", "mean"), excess_restricted=("excess_restricted", "mean"), excess_full=("excess_full", "mean"),
                                            tau_mu=("tau_mu_vs_test_human", "mean"), n_H=("n_H", "first")).reset_index()


# --------------------------------------------------------------------------- judge panels across budget regimes
PANELS = ['small6', 'large6', 'all']
NH = {'arena_33k': 300, 'mt_bench': 80, 'pandalm': 60}          # human budget held fixed in the LLM sweep
INTERMEDIATE_NL = {'arena_33k': 2000, 'mt_bench': 160, 'pandalm': 100}

# The three estimators every figure of this study draws, in the order of `experiments/style.py`:
# human_only is first and is dropped where it is absent.
METHOD_KEYS = ['human_only', 'consensus_cal', 'dial_mu']
SEEDS = 50


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
    # replications with no admissible GACV candidate carry no usable score (`load`)
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
    x = x[x.method.isin(METHOD_KEYS)].copy()          # the directory may hold other methods' passes
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


# --------------------------------------------------------------------------- dataset tables
def dataset_stats(dataset, cfg=None):
    """One column of the dataset-statistics table; judges are kept as in the real-data study."""
    from .prepare import load_canonical
    from .robustness import build_panel, load_config
    s = (cfg or load_config())["study"]
    frame = load_canonical(dataset)
    panel = build_panel(frame, exclude=s["exclude_judges"], tie_rate_exclude=s["tie_rate_exclude"],
                        min_coverage=s["min_coverage"], llm_ties=s["llm_ties"], human_ties=s["human_ties"])
    N, K, llm = panel["N"], panel["K"], panel["llm"]
    records = frame.drop_duplicates("record_id")
    labels = records["human_label"].value_counts()
    n_pairs = N * (N - 1) // 2
    cells = llm.groupby(["k", "i", "j"]).size()
    kept = frame[frame["judge"].isin(panel["judges"])]
    tie_rate = kept.groupby("judge")["judge_tie"].mean()
    return {
        "records": panel["n_records"],
        "N": N,
        "K": K,
        "observed pairs": f"{records.groupby(['item_i', 'item_j']).ngroups} / {n_pairs}",
        "human a / b / tie / both bad": " / ".join(str(int(labels.get(x, 0))) for x in ("model_a", "model_b", "tie", "both_bad")),
        "decisive human labels": panel["n_human_decisive"],
        "LLM judgments, kept judges": panel["n_llm_rows"],
        "of which ties": f"{panel['n_llm_ties']} ({panel['n_llm_ties'] / panel['n_llm_rows']:.1%})",
        "tie-excluded n_L": len(llm),
        "display order +1 / -1": f"{(llm.a == 1).sum()} / {(llm.a == -1).sum()}",
        "cells observed / possible": f"{len(cells)} / {K * n_pairs}",
        "n_kij per cell (min / mean / max)": f"{cells.min()} / {cells.mean():.1f} / {cells.max()}",
        "judge tie rate (min-max)": f"{tie_rate.min():.3f}-{tie_rate.max():.3f}",
        "candidate models": ", ".join(panel["items"]),
    }


def print_dataset_tables():
    """Print the dataset statistics and the candidate models of the three benchmarks."""
    table = pd.DataFrame({label: dataset_stats(d) for d, label in DATASET_LABEL.items()})
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(table.drop("candidate models").to_string())
        for label, items in table.loc["candidate models"].items():
            print(f"\n{label}: {items}")


# --------------------------------------------------------------------------- summary tables
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    a = p.parse_args(argv)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 800)
    datasets = ("arena_33k", "mt_bench", "pandalm") if a.dataset == "all" else (a.dataset,)
    for d in datasets:
        path = ROOT / "results" / "real_robustness" / d / "rows.jsonl"
        if not path.exists():
            continue
        df = load(d, ROOT)
        agg = aggregate(df)
        agg.to_csv(path.parent / "summary.csv", index=False)
        print(f"=== {d}")
        for (sw, pn, kd), g in agg.groupby(["sweep", "panel", "kind"]):
            piv = g.pivot_table(index=["level", "n_H_level"], columns="method", values="excess_mean").reindex(columns=MAIN_METHODS + ["dial_mle_mu", "oracle_test"])
            print(f"--- {sw} | panel={pn} | kind={kd}   excess held-out log loss (mean over seeds)")
            print(piv.round(4).to_string())
        st = aggregate_spectest(df)
        if not st.empty:
            print("--- spectest"); print(st.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
