"""Tables of the comparisons with alternative estimators, the matched and orthogonal-misalignment
synthetic designs (with their figures), the resampling checks of the intervals, and the rank
sensitivity. Run from the repository root:

    python run_real_data.py summary

Writes results/alternatives/summary/*.csv (per-cell means with Monte Carlo s.e. and fit counts),
results/real_robustness_rank/{summary.csv, bic.json}, and figures/fig_simu_matched.pdf and
fig_simu_orth.pdf with the shared method panel of experiments/style.py. Each part is skipped when
its rows are absent.
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
from experiments.style import legend_handles, mark_better, method_lines

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "alternatives" / "summary"
FIG = ROOT / "figures"
DATASETS = ["arena_33k", "mt_bench", "pandalm"]

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
        # risk crossover: the rho at which n_H Delta_mu reaches (N - 2)/2, by interpolation
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
def main(argv=None):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(style.RCPARAMS)
    datasets = DATASETS

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

    matched = {}
    for ds in datasets:
        df = matched_rows(ds)
        if df.empty:
            continue
        sub = df[(df.panel == "all") & (df.n_H_level == -1)]
        agg = aggregate(sub, "level", ["ref_kendall", "excess"], MATCHED_METHODS)
        agg.insert(0, "dataset", ds)
        matched[ds] = agg
        for (panel, nu), g in df.groupby(["panel", "n_H_level"]):
            aggregate(g, "level", ["ref_kendall", "excess"], MATCHED_METHODS).assign(dataset=ds, panel=panel, n_U_level=nu).to_csv(OUT / f"matched_{ds}_{panel}_nU{nu}.csv", index=False)
        ppi_status(sub, "level").assign(dataset=ds).to_csv(OUT / f"matched_ppi_status_{ds}.csv", index=False)
    if matched:
        pd.concat(matched.values()).to_csv(OUT / "matched_all_panel.csv", index=False)

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
    rank_tables()
    print("wrote", sorted(p.name for p in OUT.iterdir()))


# --------------------------------------------------------------------------- record-level bootstrap
BOOT_REAL_COLS = ["n_boot", "lam_boot_inf_frac", "lam_boot_zero_frac", "lam_boot_same_frac", "se_ratio_med", "se_ratio_q1", "se_ratio_q3", "se_ratio_fixed_med",
                  "se_iid_over_cell_med", "width_sand", "width_boot", "adj_sep_sand", "adj_sep_boot", "agree_sand", "agree_boot", "n_sep_sand", "n_sep_boot", "human_only_exists",
                  "se_rec_over_cell_med", "se_ratio_rec_med", "se_ratio_fixed_rec_med", "width_rec", "adj_sep_rec", "agree_rec", "n_sep_rec",
                  "se_two_over_cell_med", "se_two_over_rec_med", "se_ratio_two_med", "se_ratio_fixed_two_med", "width_two", "adj_sep_two", "agree_two", "n_sep_two"]
BOOT_SYN_COLS = ["attenuation", "n_boot", "cov_sand_cell", "cov_sand_iid", "cov_sand_cell_s0", "cov_boot_pct", "cov_boot_wald", "cov_boot_pct_s0", "cov_boot_fixed_pct",
                 "cov_boot_fixed_wald", "width_sand", "width_boot", "se_ratio_med", "se_ratio_fixed_med", "lam_boot_inf_frac", "lam_boot_zero_frac", "lam_boot_same_frac", "sand_dev",
                 "cov_sand_rec", "width_sand_rec", "se_rec_over_cell_med", "se_ratio_rec_med", "se_ratio_fixed_rec_med",
                 "lam_pre_rel", "cov_sand_cell_pre", "cov_sand_rec_pre", "se_rec_over_cell_pre_med",
                 "cov_sand_two", "width_sand_two", "se_two_over_cell_med", "se_two_over_rec_med", "se_ratio_two_med", "se_ratio_fixed_two_med", "cov_sand_two_pre"]


def _adjacent_se_ratio(row, key="contrast_se_rec"):
    """Median record- (or two-way-) to-cell sandwich s.e. ratio over the N - 1 ranking-adjacent contrasts of one split."""
    rec, cell, s_hat = row.get(key), row.get("contrast_se_sand"), row.get("s_hat")
    if not isinstance(rec, list) or not isinstance(cell, list) or not isinstance(s_hat, list):
        return np.nan
    s_hat = np.asarray(s_hat, float)
    tri = {pair: q for q, pair in enumerate(zip(*np.triu_indices(s_hat.size, 1)))}
    order = np.argsort(-s_hat)
    adj = [tri[tuple(sorted((int(a), int(b))))] for a, b in zip(order[:-1], order[1:])]
    return float(np.median(np.asarray(rec, float)[adj] / np.asarray(cell, float)[adj]))


def bootstrap_tables():
    """Per-cell means (and s.e. of the coverage columns) of the record-level resampling runs; real-data s.e. ratios are medians over splits.

    `rows.jsonl`/`bootstrap.jsonl` hold the with-replacement record bootstrap and `rows_half.jsonl`/`bootstrap_half.jsonl` the
    half-sampling without replacement, which avoids duplicate records in the GACV selection; their tables get a `_half` suffix."""
    for tag in ("", "_half"):
        real = _rows(ROOT / "results" / "alternatives" / "bootstrap" / f"rows{tag}.jsonl")
        if not real.empty:
            if "error" in real:
                real = real[real["error"].isna()]
            real = real.assign(lam_inf_frac=np.isinf(real["lam"].astype(float)).astype(float), lam_zero_frac=(real["lam"].astype(float) == 0).astype(float))
            if "contrast_se_rec" in real:
                real["se_rec_over_cell_adj_med"] = [_adjacent_se_ratio(r) for _, r in real.iterrows()]
            if "contrast_se_two" in real:
                real["se_two_over_cell_adj_med"] = [_adjacent_se_ratio(r, "contrast_se_two") for _, r in real.iterrows()]
            cols = [c for c in BOOT_REAL_COLS + ["lam_inf_frac", "lam_zero_frac", "se_rec_over_cell_adj_med", "se_two_over_cell_adj_med"] if c in real]
            g = real.groupby(["dataset", "level", "method"])
            out = g[cols].mean(numeric_only=True)
            # s.e. ratios: median over splits, since a single near-separated split can dominate the mean
            ratio_cols = [c for c in cols if c.startswith(("se_ratio", "se_rec", "se_two"))]
            out[ratio_cols] = g[ratio_cols].median(numeric_only=True)
            out["lam_boot_same_frac_med"] = g["lam_boot_same_frac"].median() if "lam_boot_same_frac" in real else np.nan
            out["n_splits"] = g.size()
            out.reset_index().to_csv(OUT / f"bootstrap_real{tag}.csv", index=False)
        syn = _rows(ROOT / "results" / "alternatives" / "simulation" / "main10_record" / f"bootstrap{tag}.jsonl")
        if not syn.empty:
            if "error" in syn:
                syn = syn[syn["error"].isna()]
            syn = syn.assign(lam_inf_frac=np.isinf(syn["lam"].astype(float)).astype(float))
            cols = [c for c in BOOT_SYN_COLS + ["lam_inf_frac"] if c in syn]
            g = syn.groupby(["n_H", "record_sd", "method"])
            out = g[cols].mean(numeric_only=True)
            for c in ("cov_sand_cell", "cov_boot_pct", "cov_boot_fixed_pct", "cov_sand_rec", "cov_sand_cell_pre", "cov_sand_rec_pre", "cov_sand_two", "cov_sand_two_pre"):
                if c in syn:
                    out[c + "_se"] = g[c].sem()
            out["n_seeds"] = g.size()
            out.reset_index().to_csv(OUT / f"bootstrap_synthetic{tag}.csv", index=False)


# --------------------------------------------------------------------------- rank sensitivity
RANK_METHODS = ("consensus_cal", "dial_mu", "dial_w")


def rank_tables():
    """The `budget` sweep refitted with LLM-side rank 0 and 2 (`--llm-rank`, results/real_robustness_rank/r{0,2})
    next to the main study's rank 1, for the rank-dependent estimators, and the BIC choice of the rank on the full LLM panels."""
    from experiments.real_data import robustness as rb
    dst = ROOT / "results" / "real_robustness_rank"
    frames = []
    for r, root in ((0, dst / "r0"), (1, ROOT / "results" / "real_robustness"), (2, dst / "r2")):
        for d in DATASETS:
            df = _rows(root / d / "rows.jsonl")
            if not df.empty:
                frames.append(df[(df["sweep"] == "budget") & (df["panel"] == "all") & df["method"].isin(RANK_METHODS)].assign(rank=r))
    if not (dst / "r0").exists() and not (dst / "r2").exists():
        return
    out = pd.concat(frames, ignore_index=True).groupby(["dataset", "level", "method", "rank"]).agg(
        excess=("excess", "mean"), excess_se=("excess", "sem"), kendall=("ref_kendall", "mean"),
        kendall_se=("ref_kendall", "sem"), n=("excess", "size")).reset_index()
    out.to_csv(dst / "summary.csv", index=False)
    cfg, bic = rb.load_config(), {}
    for d in DATASETS:
        best, table = rb.select_rank(d, cfg, rank_cap=3)
        bic[d] = dict(best=int(best), bic={int(k): float(v) for k, v in table.items()})
    json.dump(bic, open(dst / "bic.json", "w"), indent=1)
    print("BIC rank:", {d: v["best"] for d, v in bic.items()})


if __name__ == "__main__":
    main()
