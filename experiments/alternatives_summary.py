"""Tables and figures of the alternative-estimator runs (results/alternatives/, plus the stored rows of the
methods they are compared with). Run from the repository root:

    python -m experiments.alternatives_summary

Writes results/alternatives/summary/*.csv (per-cell means with Monte Carlo s.e. and fit counts) and
figures/fig_real_alternatives.pdf, fig_real_matched.pdf, fig_simu_matched.pdf, fig_simu_orth.pdf with the shared method panel of experiments/style.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import style
from experiments.style import DATASET_LABEL, LABEL, legend_handles, mark_better, method_lines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "alternatives" / "summary"
FIG = ROOT / "figures"

BUDGET_METHODS = ["human_only", "human_box", "human_ridge", "human_firth", "ppi_bt_best", "ppi_bt_panel", "ridge_mu", "atc_btl", "atc_firth", "consensus_cal", "dial_mu"]
MATCHED_METHODS = ["human_only", "human_firth", "ppi_bt_best", "ppi_bt_panel", "ridge_mu", "atc_btl", "consensus_cal", "dial_mu"]
SYN_METHODS = ["human_only", "human_firth", "ppi_bt_best", "ppi_bt_panel", "ridge_mu", "consensus_cal", "dial_mu", "dial_w"]
ORTH_METHODS = ["human_only", "ridge_mu", "consensus_cal", "dial_mu", "dial_w", "oracle_mu"]


# --------------------------------------------------------------------------- loading
def _rows(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.open() if l.strip()])


def _job_rows(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([r for l in path.open() if l.strip() for r in json.loads(l)["rows"]])


def budget_rows(dataset):
    old = _rows(ROOT / "results" / "real_robustness" / dataset / "rows.jsonl")
    old = old[(old.sweep == "budget") & (old.panel == "all")]
    new = _rows(ROOT / "results" / "alternatives" / "budget" / dataset / "rows.jsonl")
    return pd.concat([old, new[new.panel == "all"]], ignore_index=True)


def matched_rows(dataset):
    return _rows(ROOT / "results" / "alternatives" / "matched" / dataset / "rows.jsonl")


def intermediate_rows():
    old = _job_rows(ROOT / "results" / "intermediate_budget" / "jobs.jsonl")
    new = _job_rows(ROOT / "results" / "alternatives" / "intermediate_budget" / "jobs_alternatives.jsonl")
    return pd.concat([old, new], ignore_index=True)


def synthetic_rows(config):
    return pd.concat([_rows(ROOT / "results" / "simulation" / config / "rows.jsonl"),
                      _rows(ROOT / "results" / "alternatives" / "simulation" / config / "rows.jsonl")], ignore_index=True)


# --------------------------------------------------------------------------- aggregation
def aggregate(df, x, metrics, methods, extra=()):
    """Per (x, method) mean, Monte Carlo s.e., and count of finite fits for each metric."""
    d = df[df["method"].isin(methods)].copy()
    if "error" in d:
        d = d[d["error"].isna()]
    out = []
    for (xv, m), g in d.groupby([x, "method"], sort=True):
        row = {x: xv, "method": m, "n_fits": int(len(g))}
        for met in metrics:
            v = g[met].astype(float)
            v = v[np.isfinite(v)]
            row[met] = float(v.mean()) if len(v) else np.nan
            row[met + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        for e in extra:
            if e in g:
                row[e] = float(pd.to_numeric(g[e], errors="coerce").mean())
        out.append(row)
    return pd.DataFrame(out)


def ppi_status(df, x):
    d = df[df["method"].str.startswith("ppi")].copy()
    d["status"] = d.get("ppi_status", pd.Series(index=d.index, dtype=object)).fillna("failed")
    return d.groupby([x, "method"])["status"].value_counts().unstack(fill_value=0).reset_index()


# --------------------------------------------------------------------------- figures
def _panel(ax, agg, x, y, methods, log_x=True, ylabel=None, title=None, better=None):
    ax.set_xlabel(r"$n_{\mathrm{H}}$" if x in ("level", "n_H", "n_H_level") else x)
    ax.set_ylabel(y if ylabel is None else ylabel)
    method_lines(ax, agg, x, y, methods, se=y + "_se", caps=True, emphasize=("dial_mu",))
    if log_x:
        ax.set_xscale("log")
    if title:
        ax.set_title(title)
    if better:
        mark_better(ax, better)


def fig_budget(agg_by_dataset):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), constrained_layout=True)
    for c, (ds, agg) in enumerate(agg_by_dataset.items()):
        a = agg[agg["level"] > 0]
        _panel(axes[0, c], a, "level", "ref_kendall", BUDGET_METHODS, ylabel=r"Kendall's $\tau$" if c == 0 else "", title=DATASET_LABEL[ds], better="up" if c == 0 else None)
        _panel(axes[1, c], a, "level", "excess", BUDGET_METHODS, ylabel="excess human log loss" if c == 0 else "", better="down" if c == 0 else None)
        axes[1, c].set_yscale("log")
    h, l = legend_handles(BUDGET_METHODS)
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.savefig(FIG / "fig_real_alternatives.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_matched(agg_by_dataset):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), constrained_layout=True)
    for c, (ds, agg) in enumerate(agg_by_dataset.items()):
        _panel(axes[0, c], agg, "level", "ref_kendall", MATCHED_METHODS, ylabel=r"Kendall's $\tau$" if c == 0 else "", title=DATASET_LABEL[ds] + " (matched records)", better="up" if c == 0 else None)
        _panel(axes[1, c], agg, "level", "excess", MATCHED_METHODS, ylabel="excess human log loss" if c == 0 else "", better="down" if c == 0 else None)
        axes[1, c].set_yscale("log")
    h, l = legend_handles(MATCHED_METHODS)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.savefig(FIG / "fig_real_matched.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_matched(agg, N=10, d=1):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    _panel(axes[0], agg, "n_H", "excess", SYN_METHODS, ylabel="excess human risk", title="matched synthetic design", better="down")
    axes[0].set_yscale("log")
    nh = np.array(sorted(agg["n_H"].unique()), dtype=float)
    axes[0].plot(nh, (N - 1) / (2 * nh), ls=":", color=style.INK, lw=0.9)
    axes[0].plot(nh, d / (2 * nh), ls=":", color=style.STYLE["consensus_cal"]["color"], lw=0.9)
    _panel(axes[1], agg, "n_H", "kendall", SYN_METHODS, ylabel=r"Kendall's $\tau$", better="up")
    h, l = legend_handles(SYN_METHODS, extra=[(r"$(N-1)/(2n_{\mathrm{H}})$, $d/(2n_{\mathrm{H}})$", dict(color=style.INK, ls=":"))])
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(FIG / "fig_simu_matched.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_orth(agg_by_nh):
    fig, axes = plt.subplots(2, len(agg_by_nh), figsize=(3.5 * len(agg_by_nh), 5.0), constrained_layout=True, squeeze=False)
    for c, (nh, agg) in enumerate(agg_by_nh.items()):
        a0, a1 = axes[0, c], axes[1, c]
        _panel(a0, agg, "rho_orth", "excess", ORTH_METHODS, log_x=False, ylabel="excess human risk" if c == 0 else "", title=rf"$n_{{\mathrm{{H}}}}={nh}$", better="down" if c == 0 else None)
        a0.set_yscale("log"); a0.set_xlabel(r"orthogonal share $\rho$")
        _panel(a1, agg, "rho_orth", "kendall", ORTH_METHODS, log_x=False, ylabel=r"Kendall's $\tau$" if c == 0 else "", better="up" if c == 0 else None)
        a1.set_xlabel(r"orthogonal share $\rho$")
        # Theorem 3 crossover: the rho at which n_H Delta_mu reaches (N - 2)/2, by interpolation
        t = agg[agg["method"] == "consensus_cal"].sort_values("rho_orth")
        if "nH_delta_mu" in t and t["nH_delta_mu"].notna().any():
            x, y = t["rho_orth"].to_numpy(), t["nH_delta_mu"].to_numpy()
            cross = float(t["crossover_mu"].iloc[0])
            if y.max() > cross:
                rho_c = float(np.interp(cross, y, x))
                for ax in (a0, a1):
                    ax.axvline(rho_c, ls=":", color=style.INK, lw=0.8)
    h, l = legend_handles(ORTH_METHODS, extra=[("Theorem 3 crossover", dict(color=style.INK, ls=":"))])
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.savefig(FIG / "fig_simu_orth.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(style.RCPARAMS)
    datasets = ["arena_33k", "mt_bench", "pandalm"]

    budget = {}
    for ds in datasets:
        df = budget_rows(ds)
        if df.empty:
            continue
        agg = aggregate(df, "level", ["ref_kendall", "excess"], BUDGET_METHODS + ["llm_only", "atc_ridge"], extra=("box_active",))
        agg.insert(0, "dataset", ds)
        budget[ds] = agg
        ppi_status(df, "level").assign(dataset=ds).to_csv(OUT / f"budget_ppi_status_{ds}.csv", index=False)
    if budget:
        pd.concat(budget.values()).to_csv(OUT / "budget.csv", index=False)
        fig_budget(budget)

    matched = {}
    for ds in datasets:
        df = matched_rows(ds)
        if df.empty:
            continue
        sub = df[(df.panel == "all") & (df.n_H_level == -1)]
        agg = aggregate(sub, "level", ["ref_kendall", "excess"], MATCHED_METHODS)
        agg.insert(0, "dataset", ds)
        matched[ds] = agg
        full = aggregate(df, "level", ["ref_kendall", "excess"], MATCHED_METHODS + ["panel", "n_H_level"]) if False else None
        for (panel, nu), g in df.groupby(["panel", "n_H_level"]):
            aggregate(g, "level", ["ref_kendall", "excess"], MATCHED_METHODS).assign(dataset=ds, panel=panel, n_U_level=nu).to_csv(OUT / f"matched_{ds}_{panel}_nU{nu}.csv", index=False)
        ppi_status(sub, "level").assign(dataset=ds).to_csv(OUT / f"matched_ppi_status_{ds}.csv", index=False)
    if matched:
        pd.concat(matched.values()).to_csv(OUT / "matched_all_panel.csv", index=False)
        fig_matched(matched)

    inter = intermediate_rows()
    if not inter.empty:
        aggregate(inter[inter.panel == "all"], "n_H_level", ["ref_kendall", "excess"], BUDGET_METHODS).to_csv(OUT / "intermediate_all_panel.csv", index=False)

    syn = synthetic_rows("main10_matched")
    if not syn.empty:
        agg = aggregate(syn[syn.row == "nH"], "n_H", ["excess", "kendall"], SYN_METHODS + ["oracle_mu"], extra=("lam",))
        agg.to_csv(OUT / "synthetic_matched.csv", index=False)
        fig_synthetic_matched(agg)
    syn0 = synthetic_rows("main10")
    if not syn0.empty:
        aggregate(syn0[syn0.row == "nH"], "n_H", ["excess", "kendall"], SYN_METHODS).to_csv(OUT / "synthetic_main10_addon.csv", index=False)

    orth = synthetic_rows("main10_orth")
    if not orth.empty:
        by = {}
        for nh, g in orth.groupby("n_H"):
            agg = aggregate(g, "rho_orth", ["excess", "kendall"], ORTH_METHODS + ["human_firth"], extra=("delta_mu", "nH_delta_mu", "crossover_mu", "lam"))
            agg.insert(0, "n_H", nh)
            by[int(nh)] = agg
        pd.concat(by.values()).to_csv(OUT / "orth.csv", index=False)
        fig_orth(by)
    bootstrap_tables()
    print("wrote", sorted(p.name for p in OUT.iterdir()), "and", [f"fig_{n}.pdf" for n in ("real_alternatives","real_matched","simu_matched","simu_orth")])


# --------------------------------------------------------------------------- group 4: record-level bootstrap
BOOT_REAL_COLS = ["n_boot", "lam_boot_inf_frac", "lam_boot_zero_frac", "lam_boot_same_frac", "se_ratio_med", "se_ratio_q1", "se_ratio_q3", "se_ratio_fixed_med",
                  "se_iid_over_cell_med", "width_sand", "width_boot", "adj_sep_sand", "adj_sep_boot", "agree_sand", "agree_boot", "n_sep_sand", "n_sep_boot", "human_only_exists"]
BOOT_SYN_COLS = ["attenuation", "n_boot", "cov_sand_cell", "cov_sand_iid", "cov_sand_cell_s0", "cov_boot_pct", "cov_boot_wald", "cov_boot_pct_s0", "cov_boot_fixed_pct",
                 "cov_boot_fixed_wald", "width_sand", "width_boot", "se_ratio_med", "se_ratio_fixed_med", "lam_boot_inf_frac", "lam_boot_zero_frac", "lam_boot_same_frac", "sand_dev"]


def bootstrap_tables():
    """Per-cell means (and s.e. of the coverage columns) of the record-level bootstrap runs."""
    real = _rows(ROOT / "results" / "alternatives" / "bootstrap" / "rows.jsonl")
    if not real.empty:
        if "error" in real:
            real = real[real["error"].isna()]
        real = real.assign(lam_inf_frac=np.isinf(real["lam"].astype(float)).astype(float), lam_zero_frac=(real["lam"].astype(float) == 0).astype(float))
        cols = [c for c in BOOT_REAL_COLS + ["lam_inf_frac", "lam_zero_frac"] if c in real]
        g = real.groupby(["dataset", "level", "method"])
        out = g[cols].mean(numeric_only=True)
        out["n_splits"] = g.size()
        out.reset_index().to_csv(OUT / "bootstrap_real.csv", index=False)
    syn = _rows(ROOT / "results" / "alternatives" / "simulation" / "main10_record" / "bootstrap.jsonl")
    if not syn.empty:
        if "error" in syn:
            syn = syn[syn["error"].isna()]
        syn = syn.assign(lam_inf_frac=np.isinf(syn["lam"].astype(float)).astype(float))
        cols = [c for c in BOOT_SYN_COLS + ["lam_inf_frac"] if c in syn]
        g = syn.groupby(["n_H", "record_sd", "method"])
        out = g[cols].mean(numeric_only=True)
        for c in ("cov_sand_cell", "cov_boot_pct", "cov_boot_fixed_pct"):
            if c in syn:
                out[c + "_se"] = g[c].sem()
        out["n_seeds"] = g.size()
        out.reset_index().to_csv(OUT / "bootstrap_synthetic.csv", index=False)


if __name__ == "__main__":
    main()
