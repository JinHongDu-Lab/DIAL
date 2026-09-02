"""Stage 0 addendum: GACV versus exact leave-one-human-comparison-out CV (Thm GACV, Prop gacv-additional).

For each seed and n_0, over a lambda grid plus the endpoints {0, inf}:
  - fit DIAL, compute GACV(lambda);
  - compute CV_loo(lambda) exactly by refitting after deleting each human comparison
    (warm-started from the full fit, so each refit is an L-BFGS polish);
  - compute the population human risk R_0(s_hat_lambda).
Report max_lambda |GACV - CV_loo| (should scale like n_0^{-2}), agreement of the
selected lambda, and the excess-risk regret of lambda_hat relative to the grid minimizer.
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
from dial_judge.dial_model import calibration_design, fit_human_calibration
from dial_judge.gacv import gacv_for_fit, observations_from_pairs
from dial_judge.hja import make_centering_basis
from dial_judge.inference import population_excess_risk
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


def _h(s, i, j, z):
    d = s[i] - s[j]
    return float(np.logaddexp(0.0, d) - z * d)


def _delete_obs(pairs, i, j, z):
    out = []
    for (pi, pj, n, y) in pairs:
        if pi == i and pj == j:
            n2, y2 = n - 1.0, y - z
            if n2 > 0:
                out.append((pi, pj, n2, y2))
        else:
            out.append((pi, pj, n, y))
    return out


def _gacv_endpoint_human_only(N, pairs, s_hat, i_obs, j_obs, z_obs):
    """GACV at lambda = 0: chart = centered scores, H = Hessian of ell_H^full / n_0."""
    B = make_centering_basis(N)
    n_0 = float(z_obs.size)
    X = B[i_obs] - B[j_obs]  # (n_0, N-1)
    p = expit(s_hat[i_obs] - s_hat[j_obs])
    H = (X * (p * (1 - p))[:, None]).T @ X / n_0
    G = X * (p - z_obs)[:, None]
    J = (G - G.mean(0)).T @ (G - G.mean(0)) / n_0
    ell = np.mean([_h(s_hat, i, j, z) for i, j, z in zip(i_obs, j_obs, z_obs)])
    return ell + np.trace(np.linalg.pinv(H) @ J) / (n_0 - 1)


def _gacv_endpoint_staged(W, c_hat, i_obs, j_obs, z_obs):
    """GACV at lambda = inf: chart = c with W fixed."""
    n_0 = float(z_obs.size)
    s_hat = W @ c_hat
    X = W[i_obs] - W[j_obs]
    p = expit(s_hat[i_obs] - s_hat[j_obs])
    H = (X * (p * (1 - p))[:, None]).T @ X / n_0
    G = X * (p - z_obs)[:, None]
    J = (G - G.mean(0)).T @ (G - G.mean(0)) / n_0
    ell = np.mean([_h(s_hat, i, j, z) for i, j, z in zip(i_obs, j_obs, z_obs)])
    return ell + np.trace(np.linalg.pinv(H) @ J) / (n_0 - 1)


def one(args):
    n_L, n_0, seed, n_grid = args
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
    mle = n_L / n_0
    grid = [float(m * mle) for m in np.geomspace(0.05, 20.0, n_grid)]

    st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
    init = (st["gamma"], st["mu"], st["U"], st["V"], st["b"])
    rows = []

    # lambda = 0 endpoint
    h = fit_human_only_btl(N, pairs)
    loo = np.mean([_h(fit_human_only_btl(N, _delete_obs(pairs, i, j, z))["s_H"], i, j, z) for i, j, z in zip(i_obs, j_obs, z_obs)])
    rows.append(dict(lam=0.0, gacv=_gacv_endpoint_human_only(N, pairs, h["s_H"], i_obs, j_obs, z_obs), loo=float(loo), risk=population_excess_risk(h["s_H"], s0)))

    # finite lambdas
    for lam in grid:
        fit = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, lam=lam, init_params=init, tol=1e-6, max_steps=30)
        g = gacv_for_fit(fit, N, K, r, n_ijk, y_ijk, n_order, y_order, pairs, i_obs, j_obs, z_obs)
        warm = (fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"])
        loo_vals = []
        for i, j, z in zip(i_obs, j_obs, z_obs):
            f_t = fit_dial(N, K, r, n_ijk, y_ijk, _delete_obs(pairs, i, j, z), n_order=n_order, y_order=y_order, lam=lam, init_params=warm, max_steps=0)
            loo_vals.append(_h(f_t["s_H"], i, j, z))
        rows.append(dict(lam=lam, gacv=g["gacv"], loo=float(np.mean(loo_vals)), risk=population_excess_risk(fit["s_H"], s0), converged=bool(fit["fit_info"]["converged"])))

    # lambda = inf endpoint (W fixed at the LLM-only fit)
    W = calibration_design(st["mu"], st["V"])
    c_hat = np.concatenate([[st["alpha_H"]], st["a"]])
    loo_vals = []
    for i, j, z in zip(i_obs, j_obs, z_obs):
        al, a = fit_human_calibration(st["mu"], st["V"], _delete_obs(pairs, i, j, z), initial=c_hat)
        loo_vals.append(_h(W @ np.concatenate([[al], a]), i, j, z))
    rows.append(dict(lam=np.inf, gacv=_gacv_endpoint_staged(W, c_hat, i_obs, j_obs, z_obs), loo=float(np.mean(loo_vals)), risk=population_excess_risk(st["s_H"], s0)))

    lams = np.array([x["lam"] for x in rows]); gacv = np.array([x["gacv"] for x in rows]); loo = np.array([x["loo"] for x in rows]); risk = np.array([x["risk"] for x in rows])
    return dict(
        seed=seed, n_0=n_0, n_L=n_L, rows=rows, seconds=time.perf_counter() - t0,
        max_gap=float(np.max(np.abs(gacv - loo))),
        lam_gacv=float(lams[np.argmin(gacv)]), lam_loo=float(lams[np.argmin(loo)]), lam_star=float(lams[np.argmin(risk)]),
        regret_gacv=float(risk[np.argmin(gacv)] - risk.min()), regret_loo=float(risk[np.argmin(loo)] - risk.min()),
        risk_min=float(risk.min()), risk_gacv=float(risk[np.argmin(gacv)]), risk_human=float(risk[0]), risk_staged=float(risk[-1]),
    )


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_grid = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cells = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [50, 100, 200]
    tag = sys.argv[4] if len(sys.argv) > 4 else "gacv_inspect"
    n_L = 1000
    out = {}
    with ProcessPoolExecutor(2) as ex:
        for n_0 in cells:
            recs = list(ex.map(one, [(n_L, n_0, s, n_grid) for s in range(seeds)]))
            out[str(n_0)] = recs
            json.dump(out, open(f"results/stage0_oracle/{tag}.json", "w"))
            gap = np.array([x["max_gap"] for x in recs])
            print(f"n_0={n_0:4d}: max|GACV-LOO| mean {gap.mean():.2e} (sd {gap.std():.1e}) | argmin agree GACV=LOO {np.mean([x['lam_gacv']==x['lam_loo'] for x in recs]):.2f}, GACV=risk* {np.mean([x['lam_gacv']==x['lam_star'] for x in recs]):.2f}"
                  f" | regret/min-risk: GACV {np.mean([x['regret_gacv'] for x in recs])/np.mean([x['risk_min'] for x in recs]):.3f}, LOO {np.mean([x['regret_loo'] for x in recs])/np.mean([x['risk_min'] for x in recs]):.3f}"
                  f" | mean risk: human {np.mean([x['risk_human'] for x in recs]):.4f} staged {np.mean([x['risk_staged'] for x in recs]):.4f} gacv {np.mean([x['risk_gacv'] for x in recs]):.4f} min {np.mean([x['risk_min'] for x in recs]):.4f}"
                  f" | {np.mean([x['seconds'] for x in recs]):.0f}s/seed", flush=True)
    gaps = {int(k): np.mean([x["max_gap"] for x in v]) for k, v in out.items()}
    xs, ys = np.log(np.array(list(gaps.keys()), float)), np.log(np.array(list(gaps.values())))
    print(f"log-log slope of mean max|GACV-LOO| vs n_0: {np.polyfit(xs, ys, 1)[0]:.2f}  (Thm GACV: -2)")
