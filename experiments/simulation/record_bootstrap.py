"""Sandwich against record-level bootstrap intervals on the matched synthetic design (group 4).

For each cell of the `record` row of `main10_record` (human budget x record-effect s.d.), the
DIAL-Ada fit is computed once with its GACV-selected weight, and B record-level bootstrap
replicates (labeled and proxy-only records resampled within strata) repeat the whole fit
including the weight selection. Reported per cell and method:

  cov_sand_cell / cov_sand_iid   coverage of the cell-clustered and the independent sandwich
                                  Wald intervals at the selected weight (Theorem 4, Appendix E)
  cov_sand_rec                    the same with the record-clustered meat (Prop. record-cluster-inference)
  cov_sand_two                    the same with the two-way meat (records and judge-pair cells)
  cov_*_pre                       cell-, record-, and two-way-sandwich coverage at the prespecified weight
                                  lambda = n_L / n_H (equal per-comparison weight), the setting of the proposition
  cov_boot_pct / cov_boot_wald    coverage of the bootstrap percentile and bootstrap-s.e. intervals
                                  that repeat the weight selection
  cov_boot_fixed_pct              the same with the weight held at the original selection
  se_ratio_med                    median over contrasts of bootstrap s.e. / cell-sandwich s.e.
  lam_boot_*                      the selected weight across replicates

Human-only rows compare the analytic BTL intervals with the bootstrap of the labeled records.

    python -m experiments.simulation.record_bootstrap --config main10_record --seeds 0:30 --B 100 --workers 6
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dial_judge.baselines import calibrate_llm_fit, fit_human_only_btl
from dial_judge.benchmarks import fit_hja
from dial_judge.bootstrap import coverage, draw_within_strata, summarize_contrasts
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.gacv import select_lambda
from dial_judge.inference import human_only_uq, joint_sandwich, pairwise_contrast_matrix
from experiments.simulation.synthetic import simulate_cell


def _arrays(llm, hum, N, K):
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    return n_ijk, y_ijk, n_order, y_order, pool_pairs(hum)


def fit_dial_mu(N, K, r, llm, hum, lam_grid=None):
    """DIAL-Ada (consensus space) with GACV selection; returns the selection and the arrays."""
    n_ijk, y_ijk, n_order, y_order, pairs = _arrays(llm, hum, N, K)
    st = calibrate_llm_fit(fit_hja(N, K, r, n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order), pairs, consensus_only=True)
    sel = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, n_order=n_order, y_order=y_order, staged_fit=st,
                        lambda_grid=lam_grid, return_all=True, align="mu")
    return sel, (n_ijk, y_ijk, n_order, y_order, pairs)


def candidate_at(sel, lam):
    for l, f in sel["candidates"]:
        if l == lam or (np.isinf(l) and np.isinf(lam)):
            return f
    return None


def matched_observations(llm, hum, K):
    """Comparison-level arrays with record ids for the record-clustered meat: record t holds the K
    judge rows llm[t K : (t + 1) K] and, for t < len(hum), the human label hum[t]."""
    arr = np.asarray(llm, dtype=float)
    harr = np.asarray(hum, dtype=float)
    llm_obs = dict(k=arr[:, 0].astype(int), i=arr[:, 1].astype(int), j=arr[:, 2].astype(int), y=arr[:, 3], a=arr[:, 4],
                   rec=np.repeat(np.arange(arr.shape[0] // K), K))
    human_obs = dict(i=harr[:, 1].astype(int), j=harr[:, 2].astype(int), y=harr[:, 3], rec=np.arange(harr.shape[0]))
    return llm_obs, human_obs


def sandwich_at_weight(sel, lam, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="pair", llm_obs=None, human_obs=None):
    """Sandwich contrasts of the candidate fit at weight `lam` of a GACV selection, with the given meat.

    lam = 0 gives the human-only BTL intervals (human comparisons are independent records), and
    lam = infinity is evaluated at the largest finite weight of the grid as in `sandwich_at_selection`."""
    if lam == 0:
        f0 = candidate_at(sel, 0.0)
        return human_only_uq(N, pairs, (f0 if f0 is not None else sel)["s_H"])["contrasts"]
    if np.isinf(lam):
        lam = max(l for l, _ in sel["candidates"] if np.isfinite(l) and l > 0)
    fit = dict(candidate_at(sel, lam), lam=lam)
    sw = joint_sandwich(fit, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster=cluster, llm_obs=llm_obs, human_obs=human_obs)
    return sw["contrasts"]


def mixture_target(s0, record_sd, n_nodes=40):
    """Population BTL score under the record-effect mixture: the minimizer of the expected human log
    loss when every pair's win probability is E_delta sigma(s0_i - s0_j + delta), delta ~ N(0, sd^2).
    With sd = 0 it is s0; otherwise it is an attenuated s0, which is what every estimator here
    targets, so coverage is read against it (`cov_*`) and, for reference, against s0 (`cov_*_s0`)."""
    from scipy.optimize import minimize
    from scipy.special import expit, log_expit
    s0 = np.asarray(s0, dtype=float)
    if record_sd <= 0:
        return s0.copy()
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    tri_i, tri_j = np.triu_indices(s0.size, k=1)
    d = s0[tri_i] - s0[tri_j]
    p = (expit(d[:, None] + record_sd * x[None, :]) * w[None, :]).sum(1)

    def loss(s):
        e = s[tri_i] - s[tri_j]
        return -float(np.sum(p * log_expit(e) + (1 - p) * log_expit(-e))) + 1e3 * s.sum() ** 2

    res = minimize(loss, x0=s0, method="BFGS")
    return res.x - res.x.mean()


def sandwich_at_selection(sel, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order):
    """Cell-clustered and independent sandwich intervals at the selected weight.

    The sandwich of Theorem 4 is defined for finite positive weights. At the lambda = 0 endpoint
    the human-only BTL intervals apply. At the lambda = infinity endpoint the fit is evaluated at
    the largest finite weight of the GACV grid, whose calibrated score coincides with the staged
    endpoint to numerical precision (the maximal deviation is returned as `sand_dev`; it is below
    0.003 in every cell of this study) and whose sandwich is the staged limit of Corollary 1.
    """
    lam = float(sel["lam"])
    if lam == 0:
        uq = human_only_uq(N, pairs, sel["s_H"])
        return uq["contrasts"], uq["contrasts"], dict(sand_lam=0.0, sand_dev=0.0)
    fit, dev = sel, 0.0
    if np.isinf(lam):
        lam_max = max(l for l, _ in sel["candidates"] if np.isfinite(l) and l > 0)
        fit = candidate_at(sel, lam_max)
        dev = float(np.max(np.abs(fit["s_H"] - sel["s_H"])))
        lam = lam_max
    sw = joint_sandwich(fit, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="pair")
    sw0 = joint_sandwich(fit, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
    return sw["contrasts"], sw0["contrasts"], dict(sand_lam=lam, sand_dev=dev)


def _records_of(matched, K):
    """Row index blocks of the llm list per record (records are stored judge-major within record)."""
    n_rec = len(matched["llm"]) // K
    return [np.arange(t * K, (t + 1) * K) for t in range(n_rec)]


def run_cell(args):
    cfg_name, row, n_L, n_H, record_sd, seed, B, *rest = args
    scheme = rest[0] if rest else "boot"
    t0 = time.perf_counter()
    cell = simulate_cell(cfg_name, row, n_L, n_H, record_sd, seed)
    N, K, r, base, s0, matched = (cell[k] for k in ("N", "K", "r", "base", "s0", "matched"))
    llm, hum = matched["llm"], matched["human"]
    n_lab = len(hum)
    blocks = _records_of(matched, K)
    D = pairwise_contrast_matrix(N)
    s_mix = mixture_target(s0, record_sd)
    truth, truth0 = D @ s_mix, D @ s0
    base = dict(base, attenuation=float(s_mix @ s0 / (s0 @ s0)))
    rng = np.random.default_rng([seed, 11, int(round(100 * record_sd)), n_H])
    out = []

    def fail(method, e):
        out.append(dict(base, method=method, B=B, error=repr(e) + traceback.format_exc()[-200:]))

    # ---- DIAL-Ada
    try:
        sel, (n_ijk, y_ijk, n_order, y_order, pairs) = fit_dial_mu(N, K, r, llm, hum)
        lam_hat = float(sel["lam"])
        lam_grid = sorted(float(l) for l, _ in sel["candidates"] if np.isfinite(l) and l > 0)
        row_out = dict(base, method="dial_mu", B=B, lam=lam_hat, n_dropped=len(sel["dropped"]))
        try:
            c_cell, c_iid, info = sandwich_at_selection(sel, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order)
            row_out.update(info, cov_sand_cell=coverage(truth, c_cell["lower"], c_cell["upper"]), cov_sand_cell_s0=coverage(truth0, c_cell["lower"], c_cell["upper"]),
                           cov_sand_iid=coverage(truth, c_iid["lower"], c_iid["upper"]), width_sand=float(np.mean(c_cell["upper"] - c_cell["lower"])))
            se_sand = np.asarray(c_cell["se"], float)
        except Exception as e:  # noqa: BLE001
            row_out.update(sandwich_error=repr(e))
            se_sand = None
        llm_obs, human_obs = matched_observations(llm, hum, K)
        se_rec = None
        try:
            c_rec = sandwich_at_weight(sel, lam_hat, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="record", llm_obs=llm_obs, human_obs=human_obs)
            se_rec = np.asarray(c_rec["se"], float)
            row_out.update(cov_sand_rec=coverage(truth, c_rec["lower"], c_rec["upper"]), width_sand_rec=float(np.mean(c_rec["upper"] - c_rec["lower"])))
            if se_sand is not None:
                row_out.update(se_rec_over_cell_med=float(np.nanmedian(se_rec / se_sand)))
        except Exception as e:  # noqa: BLE001
            row_out.update(record_sandwich_error=repr(e))
        se_two = None
        try:
            c_two = sandwich_at_weight(sel, lam_hat, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="twoway", llm_obs=llm_obs, human_obs=human_obs)
            se_two = np.asarray(c_two["se"], float)
            row_out.update(cov_sand_two=coverage(truth, c_two["lower"], c_two["upper"]), width_sand_two=float(np.mean(c_two["upper"] - c_two["lower"])))
            if se_sand is not None:
                row_out.update(se_two_over_cell_med=float(np.nanmedian(se_two / se_sand)))
            if se_rec is not None:
                row_out.update(se_two_over_rec_med=float(np.nanmedian(se_two / se_rec)))
        except Exception as e:  # noqa: BLE001
            row_out.update(twoway_sandwich_error=repr(e))
        try:
            lam_pre = min((l for l, _ in sel["candidates"] if np.isfinite(l) and l > 0), key=lambda l: abs(np.log(l * n_lab / len(llm))))
            c_pre = sandwich_at_weight(sel, lam_pre, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="pair")
            c_pre_rec = sandwich_at_weight(sel, lam_pre, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="record", llm_obs=llm_obs, human_obs=human_obs)
            row_out.update(lam_pre=float(lam_pre), lam_pre_rel=float(lam_pre * n_lab / len(llm)),
                           cov_sand_cell_pre=coverage(truth, c_pre["lower"], c_pre["upper"]), cov_sand_rec_pre=coverage(truth, c_pre_rec["lower"], c_pre_rec["upper"]),
                           se_rec_over_cell_pre_med=float(np.nanmedian(np.asarray(c_pre_rec["se"], float) / np.asarray(c_pre["se"], float))))
            c_pre_two = sandwich_at_weight(sel, lam_pre, N, K, r, pairs, n_ijk, y_ijk, n_order, y_order, cluster="twoway", llm_obs=llm_obs, human_obs=human_obs)
            row_out.update(cov_sand_two_pre=coverage(truth, c_pre_two["lower"], c_pre_two["upper"]))
        except Exception as e:  # noqa: BLE001
            row_out.update(prespecified_error=repr(e))
        S_post, S_fixed, lams = np.full((B, N), np.nan), np.full((B, N), np.nan), np.full(B, np.nan)
        for bidx in range(B):
            draw = draw_within_strata([np.arange(n_lab), np.arange(n_lab, len(blocks))], rng, scheme)
            llm_b = [llm[i] for t in draw for i in blocks[t]]
            hum_b = [hum[t] for t in draw if t < n_lab]
            try:
                sel_b, _ = fit_dial_mu(N, K, r, llm_b, hum_b, lam_grid=lam_grid)
                S_post[bidx] = sel_b["s_H"]
                lams[bidx] = float(sel_b["lam"])
                fx = candidate_at(sel_b, lam_hat)
                if fx is not None:
                    S_fixed[bidx] = fx["s_H"]
            except Exception:  # noqa: BLE001
                continue
        bs = summarize_contrasts(sel["s_H"], S_post)
        bf = summarize_contrasts(sel["s_H"], S_fixed)
        row_out.update(cov_boot_pct=coverage(truth, bs["lower_pct"], bs["upper_pct"]), cov_boot_wald=coverage(truth, bs["lower_wald"], bs["upper_wald"]),
                       cov_boot_pct_s0=coverage(truth0, bs["lower_pct"], bs["upper_pct"]),
                       width_boot=float(np.nanmean(bs["upper_pct"] - bs["lower_pct"])), n_boot=bs["n_boot"],
                       cov_boot_fixed_pct=coverage(truth, bf["lower_pct"], bf["upper_pct"]), cov_boot_fixed_wald=coverage(truth, bf["lower_wald"], bf["upper_wald"]),
                       n_boot_fixed=bf["n_boot"],
                       lam_boot_inf_frac=float(np.mean(np.isinf(lams[np.isfinite(lams) | np.isinf(lams)]))) if np.any(~np.isnan(lams)) else np.nan,
                       lam_boot_zero_frac=float(np.mean(lams[~np.isnan(lams)] == 0)) if np.any(~np.isnan(lams)) else np.nan,
                       lam_boot_same_frac=float(np.mean(lams[~np.isnan(lams)] == lam_hat)) if np.any(~np.isnan(lams)) else np.nan)
        if se_sand is not None:
            row_out.update(se_ratio_med=float(np.nanmedian(bs["se"] / se_sand)), se_ratio_fixed_med=float(np.nanmedian(bf["se"] / se_sand)))
        if se_rec is not None:
            row_out.update(se_ratio_rec_med=float(np.nanmedian(bs["se"] / se_rec)), se_ratio_fixed_rec_med=float(np.nanmedian(bf["se"] / se_rec)))
        if se_two is not None:
            row_out.update(se_ratio_two_med=float(np.nanmedian(bs["se"] / se_two)), se_ratio_fixed_two_med=float(np.nanmedian(bf["se"] / se_two)))
        out.append(row_out)
    except Exception as e:  # noqa: BLE001
        fail("dial_mu", e)

    # ---- Human-only: analytic BTL intervals against the bootstrap of the labeled records
    try:
        pairs = pool_pairs(hum)
        h = fit_human_only_btl(N, pairs)
        uq = human_only_uq(N, pairs, h["s_H"])
        S_h = np.full((B, N), np.nan)
        for bidx in range(B):
            draw = rng.choice(n_lab, size=n_lab, replace=True) if scheme == "boot" else rng.choice(n_lab, size=n_lab // 2, replace=False)
            try:
                S_h[bidx] = fit_human_only_btl(N, pool_pairs([hum[t] for t in draw]))["s_H"]
            except Exception:  # noqa: BLE001
                continue
        bh = summarize_contrasts(h["s_H"], S_h)
        out.append(dict(base, method="human_only", B=B, lam=0.0,
                        cov_sand_cell=coverage(truth, uq["contrasts"]["lower"], uq["contrasts"]["upper"]),
                        cov_sand_cell_s0=coverage(truth0, uq["contrasts"]["lower"], uq["contrasts"]["upper"]),
                        width_sand=float(np.mean(uq["contrasts"]["upper"] - uq["contrasts"]["lower"])),
                        cov_boot_pct=coverage(truth, bh["lower_pct"], bh["upper_pct"]), cov_boot_wald=coverage(truth, bh["lower_wald"], bh["upper_wald"]),
                        width_boot=float(np.nanmean(bh["upper_pct"] - bh["lower_pct"])), n_boot=bh["n_boot"],
                        se_ratio_med=float(np.nanmedian(bh["se"] / np.asarray(uq["contrasts"]["se"], float)))))
    except Exception as e:  # noqa: BLE001
        fail("human_only", e)

    secs = time.perf_counter() - t0
    for m in out:
        m["seconds_cell"] = secs
    return out


def existing_keys(path):
    keys = set()
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            keys.add((d["n_L"], d["n_H"], float(d.get("record_sd", 0.0)), d["seed"]))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="main10_record")
    p.add_argument("--seeds", default="0:30")
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--scheme", choices=("boot", "half"), default="boot", help="record bootstrap or half-sampling without replacement")
    a = p.parse_args()
    from experiments.simulation.synthetic import cells_for
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    out = Path(a.out or f"results/alternatives/simulation/{a.config}/bootstrap.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(out)
    jobs = [(a.config, "record", nL, nH, sd, s, a.B, a.scheme) for (nL, nH, sd) in cells_for(a.config, "record") for s in range(s0, s1) if (nL, nH, sd, s) not in done]
    print(f"{len(jobs)} cells to run ({len(done)} stored) -> {out}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(out, "a") as f:
        for k, recs in enumerate(ex.map(run_cell, jobs, chunksize=1)):
            for m in recs:
                f.write(json.dumps(m) + "\n")
            f.flush()
            if (k + 1) % 5 == 0:
                print(f"{k + 1}/{len(jobs)} cells, {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
