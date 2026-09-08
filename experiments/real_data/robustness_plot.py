"""Aggregate results/real_robustness/<dataset>/rows.jsonl for the Section 22 figures.

Per (sweep, panel, kind, level, n_0_level, method): mean and Monte Carlo s.e. of the
held-out metrics, the paired difference to human-only on the same split, endpoint
shares of the selected weight, selected rank, and failure counts. Figures are drawn in
notebooks/real_data_robustness.ipynb.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["sweep", "panel", "kind", "level", "n_0_level"]
METRICS = ["excess", "logloss", "acc", "ref_kendall", "b_mean_abs", "b_inj_mean_abs", "gamma_inj_mean_abs", "lam_rel", "r_sel", "first_share", "n_L", "n_0"]
MAIN_METHODS = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial", "dial_nodeb"]


def load(dataset, base="."):
    path = Path(base) / "results" / "real_robustness" / dataset / "rows.jsonl"
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    for c in METRICS + ["error", "converged", "lam", "human_only_exists", "n_0_level", "n_at_bound", "b_at_bound", "K_dropped"]:
        if c not in df:
            df[c] = np.nan
    df["n_0_level"] = df["n_0_level"].fillna(-1).astype(int)
    df["failed"] = df["error"].notna()
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
    df = df[~df.method.isin(["spectest", "planner"])]
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
                                            tau_mu=("tau_mu_vs_test_human", "mean"), n_0=("n_0", "first")).reset_index()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all")
    a = p.parse_args()
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 800)
    datasets = ("arena_33k", "mt_bench", "pandalm") if a.dataset == "all" else (a.dataset,)
    for d in datasets:
        path = Path("results/real_robustness") / d / "rows.jsonl"
        if not path.exists():
            continue
        df = load(d)
        agg = aggregate(df)
        agg.to_csv(path.parent / "summary.csv", index=False)
        print(f"=== {d}")
        for (sw, pn, kd), g in agg.groupby(["sweep", "panel", "kind"]):
            piv = g.pivot_table(index=["level", "n_0_level"], columns="method", values="excess_mean").reindex(columns=MAIN_METHODS + ["dial_mle_r0", "oracle_test"])
            print(f"--- {sw} | panel={pn} | kind={kd}   excess held-out log loss (mean over seeds)")
            print(piv.round(4).to_string())
        st = aggregate_spectest(df)
        if not st.empty:
            print("--- spectest"); print(st.round(3).to_string(index=False))


def planner_frame(df):
    """Planner rows with the first-order predictions attached (per seed and pilot size)."""
    pl = df[df.method == "planner"].copy()
    if pl.empty:
        return pl
    pl["opt_floor"] = (pl["N"] - 1) / (2.0 * pl["n_test"])          # first-order optimism of the in-sample test-pool floor
    pl["n_star"] = np.where(pl["delta_hat"] > 0, (pl["N"] - pl["r"] - 2) / (2.0 * pl["delta_hat"]), np.inf)
    pl["n_star_all"] = np.where(pl["delta_all"] > 0, (pl["N"] - pl["r"] - 2) / (2.0 * pl["delta_all"]), np.inf)
    return pl


def predicted_curves(pl_row, grid):
    """Predicted excess (relative to the in-sample test-pool floor) at human budgets `grid`."""
    grid = np.asarray(grid, dtype=float)
    anchor = pl_row["delta_hat"] + (pl_row["r"] + 1) / (2.0 * grid) + pl_row["opt_floor"]
    human = (pl_row["N"] - 1) / (2.0 * grid) + pl_row["opt_floor"]
    return anchor, human


def noncentrality_interval(T, df, level=0.90):
    """Two-sided confidence interval for the noncentrality of a chi-square(df) from one observation T.

    The lower end is the largest noncentrality whose upper tail beyond T has mass at most
    (1 - level)/2 (zero when T is below the corresponding central quantile); the upper end
    is the smallest noncentrality whose lower tail below T has mass at most (1 - level)/2.
    Dividing both ends by 2 n_0 gives the interval for Delta_W under the first-order
    approximation E[T] = df + 2 n_0 Delta_W of app:calibration-test.
    """
    from scipy.optimize import brentq
    from scipy.stats import ncx2

    a = (1.0 - level) / 2.0
    f_lo = lambda l: ncx2.sf(T, df, l) - a
    if f_lo(0.0) >= 0:
        lo = 0.0
    else:
        b = 10.0
        while f_lo(b) < 0:
            b *= 2
        lo = brentq(f_lo, 0.0, b)
    f_hi = lambda l: ncx2.cdf(T, df, l) - a
    if f_hi(0.0) <= 0:
        hi = 0.0
    else:
        b = 10.0
        while f_hi(b) > 0:
            b *= 2
        hi = brentq(f_hi, 0.0, b)
    return lo, hi


def planner_with_intervals(pl, level=0.90):
    """Attach per-pilot interval endpoints for Delta_W (columns delta_lo, delta_hi) to a planner frame."""
    pl = pl.copy()
    ends = np.array([noncentrality_interval(t, d, level) for t, d in zip(pl["stat"], pl["df"])])
    pl["delta_lo"] = ends[:, 0] / (2.0 * pl["n_0"])
    pl["delta_hi"] = ends[:, 1] / (2.0 * pl["n_0"])
    return pl


# --------------------------------------------------------------------------- tables
