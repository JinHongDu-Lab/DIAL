"""Two-way cluster-robust check of the real-data sandwich intervals (Appendix E.2, G).

The record kernel joins all verdicts and the human label of a record; the cell kernel joins one
judge's verdicts on one model pair across records. Two-way clustering (Cameron, Gelbach, and
Miller, 2011) joins both, with meat J_rec + J_cell - J_(rec and cell), where the intersection
groups one judge's verdicts on one record. This script compares it with the one-way kernels for

  * the calibrated score of DIAL-Ada at the GACV-selected weight, on the `budget` cells of the
    record-bootstrap study (Table G4), and
  * the LLM-only order effects b_k of the full-panel fit (Figure G1(a)).

    python -m experiments.real_data.twoway_check --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dial_judge.baselines import fit_staged_structured_calibration
from dial_judge.inference import joint_sandwich, llm_only_sandwich, pairwise_contrast_matrix
from experiments.real_data import robustness as rb
from experiments.real_data.record_bootstrap import _fit
from experiments.simulation.record_bootstrap import candidate_at


def _labels(llm):
    rec = llm["record"].astype(str).to_numpy()
    judge = llm["k"].astype(str).to_numpy()
    cell = (llm["k"].astype(str) + "_" + llm["i"].astype(str) + "_" + llm["j"].astype(str)).to_numpy()
    return rec, cell, np.char.add(np.char.add(rec, "|"), judge)


def _llm_obs(llm, rec):
    return dict(k=llm["k"].to_numpy(int), i=llm["i"].to_numpy(int), j=llm["j"].to_numpy(int), y=llm["y"].to_numpy(float),
                a=llm["a"].to_numpy(float), rec=rec)


def score_job(job):
    """Contrast s.e. of the calibrated score under the independent, cell, record, and two-way meats."""
    dataset, level, seed, cfg = job
    cell = rb.build_cell((dataset, "budget", "all", "none", level, None, seed, cfg))
    N, K, A, pairs, llm, cal, grid = (cell[k] for k in ("N", "K", "A", "pairs", "llm", "cal", "lam_grid"))
    sel = _fit(N, K, 1, A, pairs, grid)
    lam = float(sel["lam"])
    out = dict(dataset=dataset, level=level, seed=seed, lam=lam)
    if lam == 0:
        return dict(out, skipped="human-only endpoint")
    if np.isinf(lam):
        lam = max(l for l, _ in sel["candidates"] if np.isfinite(l) and l > 0)
    fit = dict(candidate_at(sel, lam), lam=lam)
    rec, _, inter = _labels(llm)
    hum = dict(i=cal["i"].to_numpy(int), j=cal["j"].to_numpy(int), y=cal["y"].to_numpy(float))
    hrec = cal["record"].astype(str).to_numpy()
    D = pairwise_contrast_matrix(N)

    def se(cov):
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, cov, D), 0.0))

    cov_iid = joint_sandwich(fit, N, K, 1, pairs, *A)["cov_s"]
    cov_cell = joint_sandwich(fit, N, K, 1, pairs, *A, cluster="pair")["cov_s"]
    cov_rec = joint_sandwich(fit, N, K, 1, pairs, *A, cluster="record", llm_obs=_llm_obs(llm, rec), human_obs=dict(hum, rec=hrec))["cov_s"]
    # intersection: one judge's verdicts on one record; human labels are singletons in the cell kernel
    cov_int = joint_sandwich(fit, N, K, 1, pairs, *A, cluster="record", llm_obs=_llm_obs(llm, inter), human_obs=dict(hum, rec=np.char.add("H", hrec)))["cov_s"]
    s_iid, s_cell, s_rec, s_two = se(cov_iid), se(cov_cell), se(cov_rec), se(cov_rec + cov_cell - cov_int)
    return dict(out, cell_over_iid=float(np.median(s_cell / s_iid)), rec_over_cell=float(np.median(s_rec / s_cell)),
                twoway_over_rec=float(np.median(s_two / s_rec)), twoway_over_rec_max=float(np.max(np.abs(s_two / s_rec - 1))))


def order_effect_check(dataset, cfg, r=1):
    """b_k s.e. of the full-panel LLM-only fit under the cell, record, and two-way meats."""
    panel = rb.get_panel(dataset, cfg)
    N, K, llm = panel["N"], panel["K"], panel["llm"]
    n_ijk, y_ijk, n_order, y_order = rb.llm_arrays(llm, N, K)
    st = fit_staged_structured_calibration(N, K, r, n_ijk, y_ijk, rb.human_pairs(panel["human"]), n_order=n_order, y_order=y_order)
    rec, _, inter = _labels(llm)
    kw = dict(n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order)
    cell = llm_only_sandwich(st, N, K, r, cluster="pair", **kw)
    by_rec = llm_only_sandwich(st, N, K, r, cluster="record", llm_obs=_llm_obs(llm, rec), **kw)
    by_int = llm_only_sandwich(st, N, K, r, cluster="record", llm_obs=_llm_obs(llm, inter), **kw)
    cov_two = by_rec["cov_b"] + cell["cov_b"] - by_int["cov_b"]
    se_cell, se_rec, se_two = cell["b"]["se"], by_rec["b"]["se"], np.sqrt(np.maximum(np.diag(cov_two), 0.0))
    return dict(dataset=dataset, rec_over_cell=(se_rec / se_cell).tolist(), twoway_over_cell=(se_two / se_cell).tolist(),
                negative_variance=int(np.sum(np.diag(cov_two) < 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seeds", default="0:10")
    p.add_argument("--out", default="results/alternatives/bootstrap/twoway_check.json")
    a = p.parse_args()
    cfg = rb.load_config()
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    jobs = [(d, lv, s, cfg) for d in ("arena_33k", "mt_bench") for lv in (200, 800) for s in range(s0, s1)]
    with ProcessPoolExecutor(a.workers) as ex:
        scores = list(ex.map(score_job, jobs, chunksize=1))
    orders = [order_effect_check(d, cfg) for d in cfg["study"]["datasets"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(calibrated_score=scores, order_effects=orders), open(a.out, "w"), indent=1)
    print(f"{len(scores)} score cells and {len(orders)} order-effect fits -> {a.out}")


if __name__ == "__main__":
    main()
