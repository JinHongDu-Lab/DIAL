"""Stage 0 addendum: GACV(lambda) versus the actual population risk R_0(s_hat_lambda).

Prop. gacv-additional: max_lambda |GACV(lambda) - R_0(s_hat_lambda)| = O_p(n_0^{-1/2}), and the
selected lambda_hat has risk within o_p(1) of the grid minimum. Here R_0 is the population
human negative log-likelihood (absolute, not excess), computed exactly from the DGP.
No leave-one-out refits are needed, so a fine lambda grid and several n_0 are cheap.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import expit

from dial_judge.baselines import fit_human_only_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.dial_model import calibration_design
from dial_judge.gacv import gacv_for_fit, observations_from_pairs
from dial_judge.hja import make_centering_basis
from dial_judge.inference import population_human_risk
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


def _endpoint_gacv(X, s_hat, i_obs, j_obs, z_obs):
    """GACV for a fixed-design logistic fit with design rows X (n_0, d): chart = its coefficients."""
    n_0 = float(z_obs.size)
    d = s_hat[i_obs] - s_hat[j_obs]
    p = expit(d)
    H = (X * (p * (1 - p))[:, None]).T @ X / n_0
    G = X * (p - z_obs)[:, None]
    J = (G - G.mean(0)).T @ (G - G.mean(0)) / n_0
    ell = float(np.mean(np.logaddexp(0.0, d) - z_obs * d))
    return ell + float(np.trace(np.linalg.pinv(H) @ J)) / (n_0 - 1)


def one(args):
    cfg, n_L, n_0, seed, n_grid = args
    N, K, r = CFG[cfg]
    t0 = time.perf_counter()
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
    i_obs, j_obs, z_obs = observations_from_pairs(pairs)
    R0 = population_human_risk(s0, s0)
    grid = [float(m * n_L / n_0) for m in np.geomspace(0.02, 50.0, n_grid)]

    st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
    init = (st["gamma"], st["mu"], st["U"], st["V"], st["b"])
    rows = []
    h = fit_human_only_btl(N, pairs)
    B = make_centering_basis(N)
    rows.append(dict(lam=0.0, gacv=_endpoint_gacv(B[i_obs] - B[j_obs], h["s_H"], i_obs, j_obs, z_obs), risk=population_human_risk(h["s_H"], s0), max_abs=float(np.abs(h["s_H"]).max())))
    for lam in grid:
        fit = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, lam=lam, init_params=init, tol=1e-6, max_steps=30)
        g = gacv_for_fit(fit, N, K, r, n_ijk, y_ijk, n_order, y_order, pairs, i_obs, j_obs, z_obs)
        rows.append(dict(lam=lam, gacv=g["gacv"], risk=population_human_risk(fit["s_H"], s0), converged=bool(fit["fit_info"]["converged"]), max_abs=float(np.abs(fit["s_H"]).max())))
    W = calibration_design(st["mu"], st["V"])
    rows.append(dict(lam=np.inf, gacv=_endpoint_gacv(W[i_obs] - W[j_obs], st["s_H"], i_obs, j_obs, z_obs), risk=population_human_risk(st["s_H"], s0), max_abs=float(np.abs(st["s_H"]).max())))

    lams = np.array([x["lam"] for x in rows]); gacv = np.array([x["gacv"] for x in rows]); risk = np.array([x["risk"] for x in rows])
    k_g, k_r = int(np.argmin(gacv)), int(np.argmin(risk))
    return dict(
        cfg=cfg, seed=seed, n_0=n_0, n_L=n_L, R0=R0, rows=rows, seconds=time.perf_counter() - t0,
        max_gap=float(np.max(np.abs(gacv - risk))), max_gap_regular=float(np.max(np.abs(gacv - risk)[1:])),
        lam_gacv=float(lams[k_g]), lam_star=float(lams[k_r]), idx_gacv=k_g, idx_star=k_r,
        regret=float(risk[k_g] - risk[k_r]), excess_min=float(risk[k_r] - R0), excess_gacv=float(risk[k_g] - R0),
        excess_human=float(risk[0] - R0), excess_staged=float(risk[-1] - R0),
    )


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_L = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    cells = [int(c) for c in sys.argv[4].split(",")] if len(sys.argv) > 4 else [100, 200, 400, 800, 1600]
    n_grid = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    tag = sys.argv[6] if len(sys.argv) > 6 else f"gacv_risk_{cfg}"
    out = {}
    with ProcessPoolExecutor(2) as ex:
        for n_0 in cells:
            recs = list(ex.map(one, [(cfg, n_L, n_0, s, n_grid) for s in range(seeds)]))
            out[str(n_0)] = recs
            json.dump(out, open(f"results/stage0_oracle/{tag}.json", "w"))
            gap = np.array([x["max_gap"] for x in recs]); gapr = np.array([x["max_gap_regular"] for x in recs])
            print(f"n_0={n_0:5d}: max|GACV-R0(s_hat)| all {gap.mean():.4f}  excl. lam=0 {gapr.mean():.4f} (sd {gapr.std():.4f})"
                  f" | argmin agree {np.mean([x['idx_gacv']==x['idx_star'] for x in recs]):.2f}, |idx diff|<=1 {np.mean([abs(x['idx_gacv']-x['idx_star'])<=1 for x in recs]):.2f}"
                  f" | excess: human {np.mean([x['excess_human'] for x in recs]):.4f} staged {np.mean([x['excess_staged'] for x in recs]):.4f} gacv-sel {np.mean([x['excess_gacv'] for x in recs]):.4f} min {np.mean([x['excess_min'] for x in recs]):.4f}"
                  f" | regret/min {np.mean([x['regret'] for x in recs])/np.mean([x['excess_min'] for x in recs]):.2f} | {np.mean([x['seconds'] for x in recs]):.1f}s/seed", flush=True)
    ns = np.array(sorted(int(k) for k in out)); g = np.array([np.mean([x["max_gap_regular"] for x in out[str(n)]]) for n in ns])
    print(f"log-log slope of max|GACV-R0| (excl. lam=0) vs n_0: {np.polyfit(np.log(ns), np.log(g), 1)[0]:.2f}  (Prop: -1/2)")
