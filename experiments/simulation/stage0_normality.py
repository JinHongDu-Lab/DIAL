"""Stage 0 addendum: is sqrt(n_0)(s_hat - s_0) approximately normal with a consistently estimated variance?

For each seed store the pairwise-contrast estimates and their estimated standard
errors (joint sandwich; W-fixed for staged; Fisher for human-only), then compare
(i) mean estimated variance to the Monte Carlo variance across seeds, per contrast,
(ii) the distribution of standardized errors z = (d^T s_hat - d^T s_0) / se to N(0,1).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from dial_judge.baselines import fit_human_only_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.dial_model import calibrated_score_uq
from dial_judge.inference import contrast_intervals, human_only_uq, joint_sandwich, pairwise_contrast_matrix
from dial_judge.simulate import *  # noqa: F401,F403
from dial_judge.simulate import (
    compute_human_score,
    compute_score_matrix,
    generate_plan_calibration,
    generate_plan_parameters,
    generate_plan_position_effects,
    generate_random_human_comparisons,
    generate_random_llm_comparisons,
)

CFG = dict(inspect=(10, 4, 1), hja_size=(8, 4, 1), main=(30, 6, 2))
N, K, r = CFG["inspect"]


def one(args):
    n_L, n_0, seed = args
    mu, gamma, U, V = generate_plan_parameters(N, K, r, random_seed=seed)
    b = generate_plan_position_effects(K, random_seed=seed + 1)
    c_mu, c_v = generate_plan_calibration(V, random_seed=seed + 2)
    S = compute_score_matrix(mu, gamma, U, V)
    s0 = compute_human_score(mu, V, c_mu, c_v)
    llm = generate_random_llm_comparisons(S, b, n_L, random_seed=seed + 3)
    hum = generate_random_human_comparisons(s0, n_0, random_seed=seed + 4)
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    pairs = pool_pairs(hum)
    D = pairwise_contrast_matrix(N)
    out = {"seed": seed, "truth": (D @ s0).tolist(), "b": b.tolist()}
    h = fit_human_only_btl(N, pairs)
    ci = human_only_uq(N, pairs, h["s_H"])["contrasts"]
    out["human_only"] = {"est": ci["estimate"].tolist(), "se": ci["se"].tolist()}
    st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
    uq = calibrated_score_uq(st["alpha_H"], st["a"], st["mu"], st["V"], pairs)
    ci = contrast_intervals(st["s_H"], uq["covariance"])
    out["staged_wfixed"] = {"est": ci["estimate"].tolist(), "se": ci["se"].tolist()}
    jt = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=(st["gamma"], st["mu"], st["U"], st["V"], st["b"]), tol=1e-6, max_steps=30)
    sw = joint_sandwich(jt, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
    out["joint_mle"] = {"est": sw["contrasts"]["estimate"].tolist(), "se": sw["contrasts"]["se"].tolist(), "b_est": jt["b"].tolist(), "b_se": sw["b"]["se"].tolist()}
    out["converged"] = bool(jt["fit_info"]["converged"] and st["fit_info"]["converged"])
    return out


def analyze(recs, label):
    recs = [x for x in recs if x["converged"]]
    truth = np.array([x["truth"] for x in recs])
    print(f"\n== {label}: {len(recs)} regular seeds")
    for m in ["human_only", "staged_wfixed", "joint_mle"]:
        est = np.array([x[m]["est"] for x in recs])
        se = np.array([x[m]["se"] for x in recs])
        err = est - truth
        emp_var = err.var(axis=0, ddof=1)          # per contrast, across seeds
        est_var = (se ** 2).mean(axis=0)
        ratio = est_var / emp_var
        z = (err / se).ravel()
        ks = stats.kstest(z, "norm")
        print(f"{m:14s} var ratio est/emp: median {np.median(ratio):.3f}  IQR [{np.quantile(ratio,.25):.3f}, {np.quantile(ratio,.75):.3f}]"
              f" | z: mean {z.mean():+.3f} sd {z.std():.3f} skew {stats.skew(z):+.3f} kurt {stats.kurtosis(z):+.3f} KS D={ks.statistic:.3f}"
              f" | P(|z|>1.96)={np.mean(np.abs(z)>1.96):.3f}")
    b = np.array([x["b"] for x in recs]); be = np.array([x["joint_mle"]["b_est"] for x in recs]); bs = np.array([x["joint_mle"]["b_se"] for x in recs])
    zb = ((be - b) / bs).ravel()
    print(f"{'b (joint)':14s} var ratio est/emp: {np.median((bs**2).mean(0)/(be-b).var(0,ddof=1)):.3f} | z: mean {zb.mean():+.3f} sd {zb.std():.3f} kurt {stats.kurtosis(zb):+.3f} KS D={stats.kstest(zb,'norm').statistic:.3f}")


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cells = [(1000, 100), (1000, 400), (400, 400)]
    all_out = {}
    with ProcessPoolExecutor(2) as ex:
        for n_L, n_0 in cells:
            recs = list(ex.map(one, [(n_L, n_0, s) for s in range(seeds)], chunksize=4))
            all_out[f"{n_L}_{n_0}"] = recs
            analyze(recs, f"n_L={n_L}, n_0={n_0}")
    json.dump(all_out, open("results/stage0_oracle/normality.json", "w"))
