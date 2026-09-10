"""Stage 0 oracle sanity checks (plan Section 3): known rank, well-specified DGP.

For each (config, n_L, n_H, seed) fit human-only, oracle-W calibration, the
staged endpoint, and the joint MLE (lambda = n_L / n_H), and record bias,
error, population excess risk, and 95% coverage of pairwise contrasts of s_0,
entries of b, and entries of S. Also checks invariance of the joint fit to a
rotation of the initial factor basis and logs optimizer regularity.

Pass criteria are evaluated in `summarize`.
"""
from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dial_judge.baselines import fit_human_only_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.dial_model import calibrated_score_uq, fit_human_calibration
from dial_judge.hja import uncertainty_quantification
from dial_judge.inference import (
    contrast_intervals,
    human_only_uq,
    joint_sandwich,
    pairwise_contrast_matrix,
    population_excess_risk,
)
from dial_judge.simulate import (
    compute_human_score,
    compute_score_matrix,
    generate_plan_calibration,
    generate_plan_parameters,
    generate_plan_position_effects,
    generate_random_human_comparisons,
    generate_random_llm_comparisons,
)

STUDY_NAME = "stage0_oracle"

CONFIGS = {
    "inspect": dict(N=10, K=4, r=1),   # quick inspection: N^2 = 100 comparisons is a natural unit
    "hja_size": dict(N=8, K=4, r=1),
    "main": dict(N=30, K=6, r=2),
}
DEFAULT_GRID = dict(n_L=[3000, 30000], n_H=[60, 240, 960, 3840])


def _cover(truth, ci):
    return float(np.mean((truth >= ci["lower"]) & (truth <= ci["upper"])))


def _width(ci):
    return float(np.mean(ci["upper"] - ci["lower"]))


def one_seed(args):
    cfg_name, n_L, n_H, seed, check_basis = args
    cfg = CONFIGS[cfg_name]
    N, K, r = cfg["N"], cfg["K"], cfg["r"]
    out = dict(config=cfg_name, N=N, K=K, r=r, n_L=n_L, n_H=n_H, seed=seed, failures={})
    t0 = time.perf_counter()

    mu, gamma, U, V = generate_plan_parameters(N, K, r, random_seed=seed)
    b = generate_plan_position_effects(K, random_seed=seed + 1)
    c_mu, c_v = generate_plan_calibration(V, random_seed=seed + 2)
    S = compute_score_matrix(mu, gamma, U, V)
    s0 = compute_human_score(mu, V, c_mu, c_v)
    llm = generate_random_llm_comparisons(S, b, n_L, random_seed=seed + 3)
    hum = generate_random_human_comparisons(s0, n_H, random_seed=seed + 4)
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    pairs = pool_pairs(hum)
    D = pairwise_contrast_matrix(N)
    tc = D @ s0
    methods = {}

    def s_metrics(s_hat):
        return dict(
            excess=population_excess_risk(s_hat, s0),
            mse=float(np.mean((s_hat - s0) ** 2)),
            bias=(s_hat - s0).tolist(),
            max_abs=float(np.max(np.abs(s_hat))),
        )

    # human-only
    try:
        h = fit_human_only_btl(N, pairs)
        uq = human_only_uq(N, pairs, h["s_H"])
        methods["human_only"] = {**s_metrics(h["s_H"]), "cov_s": _cover(tc, uq["contrasts"]), "width_s": _width(uq["contrasts"])}
    except Exception as e:  # noqa: BLE001
        out["failures"]["human_only"] = repr(e)

    # oracle W
    try:
        al, a = fit_human_calibration(mu, V, pairs)
        s_or = al * mu + (V @ a if r > 0 else 0.0)
        uq = calibrated_score_uq(al, a, mu, V, pairs)
        ci = contrast_intervals(s_or, uq["covariance"])
        methods["oracle_W"] = {**s_metrics(s_or), "cov_s": _cover(tc, ci), "width_s": _width(ci)}
    except Exception as e:  # noqa: BLE001
        out["failures"]["oracle_W"] = repr(e)

    # staged endpoint
    st = None
    try:
        st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
        uq = calibrated_score_uq(st["alpha_H"], st["a"], st["mu"], st["V"], pairs)
        ci = contrast_intervals(st["s_H"], uq["covariance"])
        targets = [{"type": "position_effect", "k": k} for k in range(K)] + [
            {"type": "score_entry", "k": k, "i": i} for k in range(K) for i in range(N)
        ]
        fis = uncertainty_quantification(st["gamma"], st["mu"], st["U"], st["V"], n_ijk, targets, b=st["b"], n_order=n_order)
        lo = np.array([iv["lower"] for iv in fis["intervals"]])
        hi = np.array([iv["upper"] for iv in fis["intervals"]])
        truth = np.concatenate([b, S.ravel()])
        cov_all = (truth >= lo) & (truth <= hi)
        methods["staged"] = {
            **s_metrics(st["s_H"]),
            "cov_s_wfixed": _cover(tc, ci),
            "width_s_wfixed": _width(ci),
            "b_bias": (st["b"] - b).tolist(),
            "b_rmse": float(np.sqrt(np.mean((st["b"] - b) ** 2))),
            "cov_b": float(cov_all[:K].mean()),
            "S_mse": float(np.mean((st["S"] - S) ** 2)),
            "S_bias_mean": float(np.mean(st["S"] - S)),
            "cov_S": float(cov_all[K:].mean()),
            "n_iter": st["fit_info"]["n_iter"],
            "converged": bool(st["fit_info"]["converged"]),
            "grad_norm": st["fit_info"].get("polish_grad_norm"),
        }
    except Exception as e:  # noqa: BLE001
        out["failures"]["staged"] = repr(e) + traceback.format_exc()[-300:]

    # joint MLE, warm-started from staged
    try:
        init = (st["gamma"], st["mu"], st["U"], st["V"], st["b"]) if st is not None else None
        jt = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=init, tol=1e-6, max_steps=30)
        sw = joint_sandwich(jt, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
        sw0 = joint_sandwich(jt, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, llm_variation=False)
        S_hat = np.outer(jt["gamma"], jt["mu"]) + (jt["U"] @ jt["V"].T if r > 0 else 0.0)
        m = {
            **s_metrics(jt["s_H"]),
            "cov_s": _cover(tc, sw["contrasts"]),
            "width_s": _width(sw["contrasts"]),
            "cov_s_tau0": _cover(tc, sw0["contrasts"]),
            "b_bias": (jt["b"] - b).tolist(),
            "b_rmse": float(np.sqrt(np.mean((jt["b"] - b) ** 2))),
            "cov_b": _cover(b, sw["b"]),
            "S_mse": float(np.mean((S_hat - S) ** 2)),
            "S_bias_mean": float(np.mean(S_hat - S)),
            "cov_S": _cover(S, sw["S"]),
            "n_iter": jt["fit_info"]["n_iter"],
            "converged": bool(jt["fit_info"]["converged"]),
            "grad_norm": jt["fit_info"].get("polish_grad_norm"),
        }
        if check_basis and r > 0:
            rng = np.random.default_rng(seed + 99)
            Q, _ = np.linalg.qr(rng.normal(size=(r, r)))
            init_rot = (st["gamma"], st["mu"], st["U"] @ Q, st["V"] @ Q, st["b"])
            jt2 = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=init_rot, tol=1e-6, max_steps=30)
            m["basis_max_diff_s"] = float(np.max(np.abs(jt2["s_H"] - jt["s_H"])))
            m["basis_max_diff_b"] = float(np.max(np.abs(jt2["b"] - jt["b"])))
        methods["joint_mle"] = m
    except Exception as e:  # noqa: BLE001
        out["failures"]["joint_mle"] = repr(e) + traceback.format_exc()[-300:]

    out["methods"] = methods
    out["seconds"] = time.perf_counter() - t0
    return out


def run(configs=("hja_size", "main"), grid=None, seeds=100, basis_seeds=10, workers=2, out_path=None):
    grid = grid or DEFAULT_GRID
    jobs = [
        (c, n_L, n_H, s, s < basis_seeds)
        for c in configs
        for n_L in grid["n_L"]
        for n_H in grid["n_H"]
        for s in range(seeds)
    ]
    records = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, rec in enumerate(ex.map(one_seed, jobs, chunksize=4)):
            records.append(rec)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s", flush=True)
    payload = {"study": STUDY_NAME, "configs": {c: CONFIGS[c] for c in configs}, "grid": grid, "seeds": seeds, "records": records}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload) + "\n")
    return payload


def summarize(payload, regular_only=False):
    """Aggregate to one row per (config, n_L, n_H, method) and evaluate pass criteria."""
    import pandas as pd

    rows = []
    for rec in payload["records"]:
        for m, v in rec["methods"].items():
            row = {k: rec[k] for k in ("config", "N", "K", "r", "n_L", "n_H", "seed")}
            row["method"] = m
            for key, val in v.items():
                if key in ("bias", "b_bias"):
                    row[key + "_norm"] = float(np.linalg.norm(val))
                    row[key + "_vec"] = val
                else:
                    row[key] = val
            rows.append(row)
    df = pd.DataFrame(rows)
    # regularity flag from the polish gradient norm (regular fits end at <= 1e-5; separated samples at >= 1e-4)
    if "grad_norm" in df:
        df["converged"] = np.where(df["grad_norm"].notna(), df["grad_norm"] <= 1e-5, df.get("converged", True))
        bad = df.loc[(df.method.isin(["staged", "joint_mle"])) & (~df["converged"].astype(bool)), ["config", "n_L", "seed"]].drop_duplicates()
        df = df.merge(bad.assign(irregular=True), on=["config", "n_L", "seed"], how="left")
        df["irregular"] = df["irregular"].fillna(False).astype(bool)
    else:
        df["irregular"] = False
    if regular_only:
        df = df[~df["irregular"]]
    fail = pd.DataFrame([{**{k: rec[k] for k in ("config", "n_L", "n_H", "seed")}, "method": m, "err": e[:120]} for rec in payload["records"] for m, e in rec["failures"].items()])

    grp = df.groupby(["config", "N", "r", "n_L", "n_H", "method"])
    agg = grp.agg(
        n=("seed", "size"),
        excess=("excess", "mean"),
        excess_se=("excess", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        mse=("mse", "mean"),
        cov_s=("cov_s", "mean"),
        cov_s_tau0=("cov_s_tau0", "mean"),
        cov_s_wfixed=("cov_s_wfixed", "mean"),
        width_s=("width_s", "mean"),
        b_rmse=("b_rmse", "mean"),
        cov_b=("cov_b", "mean"),
        S_mse=("S_mse", "mean"),
        S_bias=("S_bias_mean", "mean"),
        cov_S=("cov_S", "mean"),
        n_iter=("n_iter", "mean"),
        converged=("converged", "mean"),
        grad_norm=("grad_norm", "max"),
        basis_s=("basis_max_diff_s", "max"),
        basis_b=("basis_max_diff_b", "max"),
    ).reset_index()
    agg["nH_excess"] = agg["n_H"] * agg["excess"]
    agg["nH_excess_se"] = agg["n_H"] * agg["excess_se"]
    agg["theory_nH_excess"] = np.where(agg["method"] == "human_only", (agg["N"] - 1) / 2, (agg["r"] + 1) / 2)

    # bias z-scores: mean bias vector / MC se, max over coordinates
    def bias_z(sub, col):
        vecs = np.array([v for v in sub[col + "_vec"] if isinstance(v, list)])
        if vecs.size == 0:
            return np.nan
        mean = vecs.mean(axis=0)
        se = vecs.std(axis=0, ddof=1) / np.sqrt(vecs.shape[0])
        return float(np.max(np.abs(mean) / np.where(se > 0, se, np.inf)))

    bz = grp.apply(lambda sub: pd.Series({"bias_z_s": bias_z(sub, "bias"), "bias_z_b": bias_z(sub, "b_bias") if "b_bias_vec" in sub else np.nan})).reset_index()
    agg = agg.merge(bz, on=["config", "N", "r", "n_L", "n_H", "method"])
    return agg, fail, df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--configs", nargs="+", default=["hja_size", "main"])
    p.add_argument("--n_L", nargs="+", type=int, default=DEFAULT_GRID["n_L"])
    p.add_argument("--n_H", nargs="+", type=int, default=DEFAULT_GRID["n_H"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="results/stage0_oracle/latest.json")
    a = p.parse_args()
    payload = run(configs=a.configs, grid=dict(n_L=a.n_L, n_H=a.n_H), seeds=a.seeds, workers=a.workers, out_path=a.out)
    agg, fail, df_all = summarize(payload)
    agg_reg, _, _ = summarize(payload, regular_only=True)
    import pandas as pd

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(agg.round(4).to_string())
    print("\nfailures:", len(fail))
    if len(fail):
        print(fail.groupby(["config", "n_L", "n_H", "method"]).size())
    agg.to_csv(Path(a.out).with_suffix(".summary.csv"), index=False)
    agg_reg.to_csv(Path(a.out).with_suffix(".summary_regular.csv"), index=False)
    irr = df_all[df_all["irregular"]][["config", "n_L", "seed"]].drop_duplicates()
    print("\nirregular (config, n_L, seed):", irr.values.tolist())
