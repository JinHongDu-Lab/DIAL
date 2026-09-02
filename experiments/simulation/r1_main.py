"""R1: well-specified simulation (plan Section 4), appendable per-seed storage.

Rows of the main figure:
  row "n0": x = human budget n_0 at fixed abundant n_L      (Claim 1a, theory lines)
  row "nL": x = LLM budget n_L at fixed moderate n_0        (Claim 1b, GACV tracking)

Each (config, row, cell, seed) produces one JSON line per method in
results/r1/<config>/rows.jsonl. Existing keys are skipped, so calling again with a
new seed range appends. Aggregation and plotting live in r1_plot.py.

Methods: human_only (lambda=0), staged (lambda=inf), dial_mle (lambda=n_L/n_0),
dial (GACV lambda-hat with endpoints and regularity guard), oracle_lambda (true-risk
minimizer over the same candidates), dial_nodeb (same pipeline ignoring display
order), consensus_cal (mu-only calibration).
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

from dial_judge.baselines import fit_human_only_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.dial_model import calibration_design, fit_human_calibration
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
    # main-text figure
    "main10": dict(N=10, K=4, r=1, n_L_fixed=20000, n_0_fixed=800, swap_fraction=0.25,
                   sigma_L=dict(n0=0.0, nL=1.0),  # row 2 adds pair-level LLM overdispersion (LLM likelihood misspecified, human exact)
                   n_0_grid=[100, 200, 400, 800, 1600], n_L_grid=[400, 800, 1600, 3200, 6400, 12800]),
    # appendix replicate
    "app20": dict(N=20, K=6, r=2, n_L_fixed=60000, n_0_fixed=2400, swap_fraction=0.25, sigma_L=dict(n0=0.0, nL=1.0),
                  n_0_grid=[300, 600, 1200, 2400, 4800], n_L_grid=[2400, 4800, 9600, 19200, 38400, 76800]),
}
METHODS = ["human_only", "staged", "dial_mle", "dial", "oracle_lambda", "dial_nodeb", "consensus_cal"]


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
    base = dict(config=cfg_name, row=row, N=N, K=K, r=r, n_L=n_L, n_0=n_0, seed=seed, swap_fraction=rho, sigma_L=sigma_L)
    t0 = time.perf_counter()
    out = []

    mu, gamma, U, V = generate_plan_parameters(N, K, r, random_seed=seed)
    b = generate_plan_position_effects(K, random_seed=seed + 1)
    c_mu, c_v = generate_plan_calibration(V, random_seed=seed + 2)
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
        m = dict(base, method=method,
                 excess=population_excess_risk(s_hat, s0),
                 mse=float(np.mean((s_hat - s0) ** 2)),
                 sign_acc=sign_accuracy(s0, s_hat),
                 spearman=spearman(s0, s_hat),
                 max_abs=float(np.max(np.abs(s_hat))))
        m.update(extra)
        out.append(m)

    def b_metrics(b_hat):
        return dict(b_rmse=float(np.sqrt(np.mean((b_hat - b) ** 2))), b_sign_acc=float(np.mean(np.sign(b_hat) == np.sign(b))))

    # human-only
    try:
        h = fit_human_only_btl(N, pairs)
        uq = human_only_uq(N, pairs, h["s_H"])
        rec("human_only", h["s_H"], cov_s=_cover(tc, uq["contrasts"]), width_s=float(np.mean(uq["contrasts"]["upper"] - uq["contrasts"]["lower"])), converged=True)
    except Exception as e:  # noqa: BLE001
        out.append(dict(base, method="human_only", error=repr(e)))

    # staged
    st = None
    try:
        st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order)
        rec("staged", st["s_H"], converged=bool(st["fit_info"]["converged"]), S_mse=float(np.mean((st["S"] - S) ** 2)), **b_metrics(st["b"]))
    except Exception as e:  # noqa: BLE001
        out.append(dict(base, method="staged", error=repr(e) + traceback.format_exc()[-200:]))

    # consensus-only calibration (mu direction of the same LLM fit)
    if st is not None:
        try:
            V0 = np.zeros((N, 0))
            al, _ = fit_human_calibration(st["mu"], V0, pairs)
            rec("consensus_cal", al * st["mu"], converged=bool(st["fit_info"]["converged"]))
        except Exception as e:  # noqa: BLE001
            out.append(dict(base, method="consensus_cal", error=repr(e)))

    # joint MLE at lambda = n_L / n_0
    if st is not None:
        try:
            init = (st["gamma"], st["mu"], st["U"], st["V"], st["b"])
            jt = fit_dial(N, K, r, n_ijk, y_ijk, pairs, n_order=n_order, y_order=y_order, init_params=init, tol=1e-6, max_steps=30)
            sw = joint_sandwich(jt, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="pair")
            sw_iid = joint_sandwich(jt, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
            S_hat = np.outer(jt["gamma"], jt["mu"]) + (jt["U"] @ jt["V"].T if r > 0 else 0.0)
            rec("dial_mle", jt["s_H"], cov_s=_cover(tc, sw["contrasts"]), width_s=float(np.mean(sw["contrasts"]["upper"] - sw["contrasts"]["lower"])), cov_s_iid=_cover(tc, sw_iid["contrasts"]),
                cov_b=_cover(b, sw["b"]), converged=bool(jt["fit_info"]["converged"]), S_mse=float(np.mean((S_hat - S) ** 2)), lam=float(jt["lam"]), **b_metrics(jt["b"]))
        except Exception as e:  # noqa: BLE001
            out.append(dict(base, method="dial_mle", error=repr(e) + traceback.format_exc()[-200:]))

    # DIAL with GACV, plus oracle lambda over the same candidates
    if st is not None:
        try:
            sel = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, n_order=n_order, y_order=y_order, staged_fit=st, return_all=True)
            lam_hat = float(sel["lam"])
            extra = dict(lam=lam_hat, n_dropped=len(sel["dropped"]), converged=bool(sel.get("fit_info", {}).get("converged", True)))
            if np.isfinite(lam_hat) and lam_hat > 0:
                sw = joint_sandwich(sel, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="pair")
                sw_iid = joint_sandwich(sel, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
                extra.update(cov_s=_cover(tc, sw["contrasts"]), width_s=float(np.mean(sw["contrasts"]["upper"] - sw["contrasts"]["lower"])), cov_b=_cover(b, sw["b"]), cov_s_iid=_cover(tc, sw_iid["contrasts"]))
            if lam_hat > 0:
                extra.update(**b_metrics(np.asarray(sel["b"])))
                S_sel = np.outer(sel["gamma"], sel["mu"]) + (sel["U"] @ sel["V"].T if r > 0 else 0.0)
                extra["S_mse"] = float(np.mean((S_sel - S) ** 2))
            rec("dial", sel["s_H"], **extra)
            risks = [(lam, population_excess_risk(f["s_H"], s0), f) for lam, f in sel["candidates"]]
            lam_o, _, f_o = min(risks, key=lambda x: x[1])
            oextra = dict(lam=float(lam_o))
            if lam_o > 0:
                oextra.update(**b_metrics(np.asarray(f_o["b"])))
            rec("oracle_lambda", f_o["s_H"], **oextra)
        except Exception as e:  # noqa: BLE001
            out.append(dict(base, method="dial", error=repr(e) + traceback.format_exc()[-300:]))

    # ablation: same pipeline ignoring display order
    try:
        st0 = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, pairs)
        sel0 = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, staged_fit=st0)
        S_hat0 = np.outer(sel0["gamma"], sel0["mu"]) + (sel0["U"] @ sel0["V"].T if r > 0 else 0.0) if sel0["mu"] is not None else None
        rec("dial_nodeb", sel0["s_H"], lam=float(sel0["lam"]), n_dropped=len(sel0["dropped"]),
            S_mse=float(np.mean((S_hat0 - S) ** 2)) if S_hat0 is not None else None, converged=bool(sel0.get("fit_info", {}).get("converged", True)))
    except Exception as e:  # noqa: BLE001
        out.append(dict(base, method="dial_nodeb", error=repr(e) + traceback.format_exc()[-200:]))

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
