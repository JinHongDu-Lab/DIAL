"""Aggregate results/r1/<config>/rows.jsonl into per-cell summaries (mean, MC s.e., selection rates).

Aggregation is over all stored seeds (append more with r1_main.py). Cells whose staged or
joint fit is flagged non-convergent are excluded and counted. Figures are drawn in
notebooks/1_simulation.ipynb, which imports `load` and `aggregate` from here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = [
    ("excess", "excess human risk", True),
    ("sign_acc", "pairwise sign accuracy", False),
    ("spearman", "Spearman$(\\hat s, s_0)$", False),
    ("cov_s", "95% CI coverage", False),
    ("b_rmse", "RMSE$(\\hat b, b)$", True),
    ("S_mse", "MSE(Ŝ, S)", True),
    ("cov_s_iid", "coverage, binomial sandwich", False),
]
MAIN_COLS = [0, 1, 2, 3, 4]  # columns of COLS shown in the main figure


def load(cfg, base="."):
    """Load results/r1/<cfg>/rows.jsonl (relative to `base`) and flag irregular replications."""
    rows = [json.loads(l) for l in open(Path(base) / "results" / "r1" / cfg / "rows.jsonl")]
    df = pd.DataFrame(rows)
    for c, _, _ in COLS:
        if c not in df:
            df[c] = np.nan
    if "error" in df:
        df = df[df["error"].isna()]
    # regularity: exclude a (row, cell, seed) if staged or dial_mle did not converge
    key = ["row", "n_L", "n_0", "seed"]
    bad = df[(df.method.isin(["staged", "dial_mle"])) & (~df["converged"].fillna(True).astype(bool))][key].drop_duplicates()
    df = df.merge(bad.assign(irregular=True), on=key, how="left")
    df["irregular"] = df["irregular"].fillna(False).astype(bool)
    return df


def aggregate(df):
    reg = df[~df.irregular]
    g = reg.groupby(["row", "n_L", "n_0", "method"])
    agg = g.agg(n=("seed", "size"), sigma_L=("sigma_L", "first"), swap_fraction=("swap_fraction", "first"), **{f"{c}_mean": (c, "mean") for c, _, _ in COLS}, **{f"{c}_se": (c, lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else np.nan) for c, _, _ in COLS},
                lam_med=("lam", lambda x: np.nanmedian(x.replace(np.inf, np.nan))) if "lam" in reg else ("seed", "size"),
                lam_inf_frac=("lam", lambda x: np.mean(np.isinf(x))) if "lam" in reg else ("seed", "size"),
                lam_zero_frac=("lam", lambda x: np.mean(x == 0)) if "lam" in reg else ("seed", "size")).reset_index()
    irr = df[df.irregular][["row", "n_L", "n_0", "seed"]].drop_duplicates().groupby(["row", "n_L", "n_0"]).size().rename("n_irregular").reset_index()
    return agg.merge(irr, on=["row", "n_L", "n_0"], how="left").fillna({"n_irregular": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="main10")
    a = p.parse_args()
    df = load(a.config)
    agg = aggregate(df)
    N, r = int(df.N.iloc[0]), int(df.r.iloc[0])
    outdir = Path(f"results/r1/{a.config}")
    agg.to_csv(outdir / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = agg[["row", "n_L", "n_0", "method", "n", "excess_mean", "sign_acc_mean", "spearman_mean", "cov_s_mean", "b_rmse_mean", "lam_med", "lam_inf_frac", "lam_zero_frac", "n_irregular"]]
    print(show.round(4).to_string(index=False))
    print("summary:", outdir / "summary.csv")
