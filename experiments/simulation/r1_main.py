"""R1: well-specified simulation with the unified method panel, appendable per-seed storage.

Rows of the main figure:
  row "n0": x = human budget n_0 at fixed abundant n_L      (limited human supervision; theory lines)
  row "nL": x = LLM budget n_L at fixed moderate n_0        (adaptive weighting under LLM overdispersion)

Data-generating process (plan Section 2 of code/plan/2026-09-02-main-experiments-plan.md, revised
2026-09-07): LLM side S = gamma mu^T + U V^T at rank r with judge-specific position effects; human
target s_0 = W c_0 with c_V ~ N(0, c_v_sd^2). The main configurations use c_v_sd = 0 (the human
preference is aligned with the consensus, as on the three benchmarks); the "*_mis" configuration keeps
c_v_sd = 0.5 so that the W-calibration has a direction to recover.

Presented methods (experiments/style.py): human_only, pooled_cal, consensus_cal (staged endpoint of
DIAL at rank r, s = alpha mu), dial_mu ("DIAL": joint weighted likelihood at rank r aligned to mu,
GACV weight), dial_nodeb (DIAL without the order term). Diagnostics: staged_w, dial_w (W-calibration),
dial_mle_mu, dial_mle_w (fixed weight n_L/n_0), oracle_mu, oracle_w (population-risk minimizers on the
respective GACV paths).

Each (config, row, cell, seed) produces one JSON line per method in results/r1/<config>/rows.jsonl.
Existing keys are skipped, so calling again with a new seed range appends. Aggregation lives in r1_plot.py.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dial_judge.baselines import fit_consensus_only_calibrated, fit_human_only_btl, fit_pooled_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.dial_model import fit_human_calibration
from dial_judge.evaluate import sign_accuracy, spearman
from dial_judge.gacv import select_lambda
from dial_judge.inference import human_only_uq, joint_sandwich, pairwise_contrast_matrix, population_excess_risk
from dial_judge.simulate import (
    compute_human_score,
    compute_score_matrix,
    generate_plan_calibration,
    generate_plan_parameters,
    generate_plan_position_effects,
    generate_random_human_comparisons,
    generate_random_llm_comparisons,
)

CONFIGS = {
    # main-text figure: human target aligned with the consensus
    "main10": dict(N=10, K=4, r=1, n_L_fixed=20000, n_0_fixed=800, swap_fraction=0.25, c_v_sd=0.0,
                   sigma_L=dict(n0=0.0, nL=1.0),  # row 2 adds pair-level LLM overdispersion (LLM likelihood misspecified, human exact)
                   n_0_grid=[100, 200, 400, 800, 1600], n_L_grid=[400, 800, 1600, 3200, 6400, 12800]),
    # appendix replicate
    "app20": dict(N=20, K=6, r=2, n_L_fixed=60000, n_0_fixed=2400, swap_fraction=0.25, c_v_sd=0.0, sigma_L=dict(n0=0.0, nL=1.0),
                  n_0_grid=[300, 600, 1200, 2400, 4800], n_L_grid=[2400, 4800, 9600, 19200, 38400, 76800]),
    # appendix: human target with a component in col(V) (the pre-2026-09-07 DGP), where the W-calibration is needed
    "main10_mis": dict(N=10, K=4, r=1, n_L_fixed=20000, n_0_fixed=800, swap_fraction=0.25, c_v_sd=0.5,
                       sigma_L=dict(n0=0.0, nL=1.0), n_0_grid=[100, 200, 400, 800, 1600], n_L_grid=[400, 800, 1600, 3200, 6400, 12800]),
}
METHODS = ["human_only", "pooled_cal", "consensus_cal", "dial_mu", "dial_nodeb", "staged_w", "dial_w", "dial_mle_mu", "dial_mle_w", "oracle_mu", "oracle_w"]


def cells_for(cfg, row):
    c = CONFIGS[cfg]
    if row == "n0":
        return [(c["n_L_fixed"], n0) for n0 in c["n_0_grid"]]
    return [(nL, c["n_0_fixed"]) for nL in c["n_L_grid"]]


def _cover(truth, ci):
    return float(np.mean((truth >= ci["lower"]) & (truth <= ci["upper"])))


def run_cell(args):
    cfg_name, row, n_L, n_0, seed = args
    c = CONFIGS[cfg_name]
    N, K, r = c["N"], c["K"], c["r"]
    rho = c.get("swap_fraction", 0.5)
    sigma_L = c.get("sigma_L", {}).get(row, 0.0)
    base = dict(config=cfg_name, row=row, N=N, K=K, r=r, n_L=n_L, n_0=n_0, seed=seed, swap_fraction=rho, sigma_L=sigma_L, c_v_sd=c.get("c_v_sd", 0.0))
    t0 = time.perf_counter()
    out = []

    mu, gamma, U, V = generate_plan_parameters(N, K, r, random_seed=seed)
    b = generate_plan_position_effects(K, random_seed=seed + 1)
    c_mu, c_v = generate_plan_calibration(V, c_v_sd=c.get("c_v_sd", 0.0), random_seed=seed + 2)
    S = compute_score_matrix(mu, gamma, U, V)
    s0 = compute_human_score(mu, V, c_mu, c_v)
    llm = generate_random_llm_comparisons(S, b, n_L, random_seed=seed + 3, swap_fraction=rho, overdispersion=sigma_L)
    hum = generate_random_human_comparisons(s0, n_0, random_seed=seed + 4)
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    pairs = pool_pairs(hum)
    D = pairwise_contrast_matrix(N)
    tc = D @ s0

    def rec(method, s_hat, **extra):
        s_hat = np.asarray(s_hat, dtype=float)
        acc = sign_accuracy(s0, s_hat)
        m = dict(base, method=method, excess=population_excess_risk(s_hat, s0), mse=float(np.mean((s_hat - s0) ** 2)), sign_acc=acc,
                 kendall=2.0 * acc - 1.0, spearman=spearman(s0, s_hat), max_abs=float(np.max(np.abs(s_hat))))
        m.update(extra)
        out.append(m)

    def fail(method, e):
        out.append(dict(base, method=method, error=repr(e) + traceback.format_exc()[-200:]))

    def b_metrics(b_hat):
        b_hat = np.asarray(b_hat, dtype=float)
        return dict(b_rmse=float(np.sqrt(np.mean((b_hat - b) ** 2))), b_sign_acc=float(np.mean(np.sign(b_hat) == np.sign(b))))

    def S_of(fit):
        return np.outer(fit["gamma"], fit["mu"]) + (fit["U"] @ fit["V"].T if r > 0 else 0.0)

    def sandwich_fields(fit):
        sw = joint_sandwich(fit, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="pair")
        sw_iid = joint_sandwich(fit, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
        return dict(cov_s=_cover(tc, sw["contrasts"]), width_s=float(np.mean(sw["contrasts"]["upper"] - sw["contrasts"]["lower"])), cov_s_iid=_cover(tc, sw_iid["contrasts"]), cov_b=_cover(b, sw["b"]))

    # human-only
    try:
        h = fit_human_only_btl(N, pairs)
        uq = human_only_uq(N, pairs, h["s_H"])
        rec("human_only", h["s_H"], cov_s=_cover(tc, uq["contrasts"]), width_s=float(np.mean(uq["contrasts"]["upper"] - uq["contrasts"]["lower"])), converged=True, lam=0.0)
    except Exception as e:  # noqa: BLE001
        fail("human_only", e)

    # pooled LLM BTL (no judges, no order) + scale
    try:
        pooled = fit_pooled_btl(N, n_ijk, y_ijk)["score"]
        al, _ = fit_human_calibration(pooled, np.zeros((N, 0)), pairs)
        rec("pooled_cal", al * pooled, alpha=float(al))
    except Exception as e:  # noqa: BLE001
        fail("pooled_cal", e)

    # Consensus-cal (staged endpoint of DIAL) and DIAL (align mu)
    st_mu = None
    try:
        st_mu = fit_consensus_only_calibrated(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
        rec("consensus_cal", st_mu["s_H"], lam=float("inf"), converged=bool(st_mu["fit_info"]["converged"]), S_mse=float(np.mean((S_of(st_mu) - S) ** 2)), **b_metrics(st_mu["b"]))
    except Exception as e:  # noqa: BLE001
        fail("consensus_cal", e)
    if st_mu is not None:
        try:
            sel = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, n_order=n_order, y_order=y_order, staged_fit=st_mu, return_all=True, align="mu")
            lam_hat = float(sel["lam"])
            extra = dict(lam=lam_hat, n_dropped=len(sel["dropped"]), converged=bool(sel.get("fit_info", {}).get("converged", True)))
            if np.isfinite(lam_hat) and lam_hat > 0:
                extra.update(sandwich_fields(sel))
            if lam_hat > 0:
                extra.update(S_mse=float(np.mean((S_of(sel) - S) ** 2)), **b_metrics(sel["b"]))
            rec("dial_mu", sel["s_H"], **extra)
            risks = [(lam, population_excess_risk(f["s_H"], s0), f) for lam, f in sel["candidates"]]
            lam_o, _, f_o = min(risks, key=lambda x: x[1])
            rec("oracle_mu", f_o["s_H"], lam=float(lam_o), **(b_metrics(f_o["b"]) if lam_o > 0 else {}))
        except Exception as e:  # noqa: BLE001
            fail("dial_mu", e)
        try:
            jt = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=(st_mu["gamma"], st_mu["mu"], st_mu["U"], st_mu["V"], st_mu["b"]), tol=1e-6, max_steps=30, align="mu")
            rec("dial_mle_mu", jt["s_H"], lam=float(jt["lam"]), converged=bool(jt["fit_info"]["converged"]), S_mse=float(np.mean((S_of(jt) - S) ** 2)), **b_metrics(jt["b"]), **sandwich_fields(jt))
        except Exception as e:  # noqa: BLE001
            fail("dial_mle_mu", e)

    # DIAL-noDeb: DIAL without the order term
    try:
        st0 = fit_consensus_only_calibrated(N, K, r, n_ijk, y_ijk, pairs)
        sel0 = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, staged_fit=st0, align="mu")
        rec("dial_nodeb", sel0["s_H"], lam=float(sel0["lam"]), n_dropped=len(sel0["dropped"]), converged=bool(sel0.get("fit_info", {}).get("converged", True)),
            S_mse=float(np.mean((S_of(sel0) - S) ** 2)) if sel0.get("mu") is not None else None)
    except Exception as e:  # noqa: BLE001
        fail("dial_nodeb", e)

    # W-calibration diagnostics
    try:
        st_w = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
        rec("staged_w", st_w["s_H"], lam=float("inf"), converged=bool(st_w["fit_info"]["converged"]), S_mse=float(np.mean((S_of(st_w) - S) ** 2)), **b_metrics(st_w["b"]))
        selw = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, n_order=n_order, y_order=y_order, staged_fit=st_w, return_all=True, align="W")
        lamw = float(selw["lam"])
        extra = dict(lam=lamw, n_dropped=len(selw["dropped"]), converged=bool(selw.get("fit_info", {}).get("converged", True)))
        if np.isfinite(lamw) and lamw > 0:
            extra.update(sandwich_fields(selw))
        if lamw > 0:
            extra.update(S_mse=float(np.mean((S_of(selw) - S) ** 2)), **b_metrics(selw["b"]))
        rec("dial_w", selw["s_H"], **extra)
        risks = [(lam, population_excess_risk(f["s_H"], s0), f) for lam, f in selw["candidates"]]
        lam_o, _, f_o = min(risks, key=lambda x: x[1])
        rec("oracle_w", f_o["s_H"], lam=float(lam_o))
        jtw = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=(st_w["gamma"], st_w["mu"], st_w["U"], st_w["V"], st_w["b"]), tol=1e-6, max_steps=30, align="W")
        rec("dial_mle_w", jtw["s_H"], lam=float(jtw["lam"]), converged=bool(jtw["fit_info"]["converged"]), **b_metrics(jtw["b"]), **sandwich_fields(jtw))
    except Exception as e:  # noqa: BLE001
        fail("dial_w", e)

    secs = time.perf_counter() - t0
    for m in out:
        m["seconds_cell"] = secs
    return out


def existing_keys(path):
    keys = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                keys.add((d["row"], d["n_L"], d["n_0"], d["seed"]))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="main10", choices=sorted(CONFIGS))
    p.add_argument("--row", default="both", choices=["n0", "nL", "both"])
    p.add_argument("--seeds", default="0:10", help="start:stop seed range (stop exclusive)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    out = Path(a.out or f"results/r1/{a.config}/rows.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(out)
    rows = ["n0", "nL"] if a.row == "both" else [a.row]
    jobs = [(a.config, row, nL, n0, s) for row in rows for (nL, n0) in cells_for(a.config, row) for s in range(s0, s1) if (row, nL, n0, s) not in done]
    print(f"{len(jobs)} cells to run ({len(done)} already stored) -> {out}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(out, "a") as f:
        for k, recs in enumerate(ex.map(run_cell, jobs, chunksize=1)):
            for m in recs:
                f.write(json.dumps(m) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(jobs)} cells, {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
