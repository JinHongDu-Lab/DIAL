"""Record-level bootstrap of the DIAL-Ada intervals on real data.

One job is a stored `budget` cell (dataset, panel, human budget, split seed) built exactly as the
robustness study builds it. The original fit gives the GACV-selected weight and the cell-clustered
sandwich intervals of all pairwise contrasts. B replicates resample the train records within two
strata, the calibration records (with their human label and all their judge rows) and the
remaining records (judge rows only), and repeat the whole fit, weight selection included, on the
original GACV grid. The job row stores, for DIAL-Ada and (where its MLE exists) Human-only:

  se ratio of bootstrap to sandwich over the contrasts (median and quartiles)
  the record-clustered sandwich (cluster-robust inference): its s.e. relative to the cell
  sandwich, the bootstrap-to-record-sandwich ratios, and its pair separation (`*_rec`), and the
  same for the two-way sandwich over records and judge-pair cells (`*_two`)
  the selected weight across replicates
  share of ranking-adjacent pairs separated at 95% by each interval, and the sign agreement of the
  separated contrasts with the held-out human ranking (the practical-value demonstration)

Two further checks run on the same cells:

  --check twoway     the two-way cluster-robust sandwich (Cameron, Gelbach, and Miller, 2011) against
                     the one-way kernels. The record kernel joins all verdicts and the human label
                     of a record; the cell kernel joins one judge's verdicts on one model pair
                     across records; the two-way meat is J_rec + J_cell - J_(rec and cell), where
                     the intersection groups one judge's verdicts on one record. Compared for the
                     calibrated score of DIAL at the selected weight and for the LLM-only order
                     effects b_k of the full-panel fit.
  --check selection  the selected weight and the spread of the calibrated contrasts under three
                     resampling schemes of the records, within the labeled and LLM-only strata:
                     `boot` (with replacement; about 37% of draws duplicate a record), `sub`
                     (without replacement, about 63% of each stratum: as many distinct records as a
                     bootstrap sample, no duplicates), and `dup` (the original records plus a
                     duplicate of 37% of them). GACV approximates leave-one-out prediction error,
                     so duplicated records can make human-heavy fits look better; if duplication
                     drives the selection shift, `boot` and `dup` move toward small weights and
                     `sub` does not.

    python run_real_data.py bootstrap --datasets arena_33k,mt_bench --levels 200,800 --seeds 0:10 --B 200 --workers 4
    python run_real_data.py bootstrap --check twoway --workers 8
    python run_real_data.py bootstrap --check selection --workers 16
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dial_judge.baselines import calibrate_llm_fit, fit_human_only_btl, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_hja
from dial_judge.bootstrap import adjacent_separation, draw_within_strata, separated_sign_agreement, summarize_contrasts
from dial_judge.gacv import select_lambda
from dial_judge.inference import human_only_uq, joint_sandwich, llm_only_sandwich, pairwise_contrast_matrix
from experiments.real_data import robustness as rb
from experiments.simulation.record_bootstrap import candidate_at, sandwich_at_selection, sandwich_at_weight


def _fit(N, K, r, A, pairs, lam_grid):
    st = calibrate_llm_fit(fit_hja(N, K, r, n_ijk=A[0], y_ijk=A[1], n_order=A[2], y_order=A[3]), pairs, consensus_only=True)
    return select_lambda(N, K, r, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2], y_order=A[3], staged_fit=st, lambda_grid=lam_grid, return_all=True, align="mu")


def _quartiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(med=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75))) if x.size else dict(med=np.nan, q1=np.nan, q3=np.nan)


def run_job(job):
    dataset, panel_name, level, seed, cfg, B, *rest = job
    scheme = rest[0] if rest else "boot"
    t0 = time.perf_counter()
    cell = rb.build_cell((dataset, "budget", panel_name, "none", level, None, seed, cfg))
    N, K, A, pairs, llm, cal, lam_grid, hum_test = (cell[k] for k in ("N", "K", "A", "pairs", "llm", "cal", "lam_grid", "hum_test"))
    r = int(min(cell["scfg"].get("llm_rank", 1), K - 1, N - 2))
    ref = fit_human_only_btl(N, rb.human_pairs(hum_test))["s_H"]
    base = dict(dataset=dataset, panel=panel_name, level=float(level), n_H=int(len(cal)), seed=int(seed), N=N, K=K, r=r, n_L=int(len(llm)), B=B,
                human_only_exists=bool(rb.btl_mle_exists(N, pairs)))
    out = []
    try:
        sel = _fit(N, K, r, A, pairs, lam_grid)
        lam_hat = float(sel["lam"])
        row = dict(base, method="dial_mu", lam=lam_hat, lam_rel=lam_hat * len(cal) / len(llm) if np.isfinite(lam_hat) else float("inf"))
        sw = None
        try:
            c, c0, info = sandwich_at_selection(sel, N, K, r, pairs, A[0], A[1], A[2], A[3])
            sw = dict(contrasts=c)
            row.update(info, width_sand=float(np.mean(c["upper"] - c["lower"])), se_iid_over_cell_med=float(np.median(np.asarray(c0["se"]) / np.asarray(c["se"]))))
            frac, n_sep = adjacent_separation(sel["s_H"], c["lower"], c["upper"])
            agree, n_all = separated_sign_agreement(sel["s_H"], c["lower"], c["upper"], ref)
            row.update(adj_sep_sand=frac, n_adj_sep_sand=n_sep, agree_sand=agree, n_sep_sand=n_all)
        except Exception as e:  # noqa: BLE001
            row.update(sandwich_error=repr(e))
        # ---- record-clustered sandwich at the selected weight: all judge rows and the human label of a record form one cluster
        se_rec = None
        try:
            llm_obs = dict(k=llm["k"].to_numpy(int), i=llm["i"].to_numpy(int), j=llm["j"].to_numpy(int), y=llm["y"].to_numpy(float),
                           a=llm["a"].to_numpy(float), rec=llm["record"].to_numpy())
            human_obs = dict(i=cal["i"].to_numpy(int), j=cal["j"].to_numpy(int), y=cal["y"].to_numpy(float), rec=cal["record"].to_numpy())
            c_rec = sandwich_at_weight(sel, lam_hat, N, K, r, pairs, A[0], A[1], A[2], A[3], cluster="record", llm_obs=llm_obs, human_obs=human_obs)
            se_rec = np.asarray(c_rec["se"], float)
            frac, n_sep = adjacent_separation(sel["s_H"], c_rec["lower"], c_rec["upper"])
            agree, n_all = separated_sign_agreement(sel["s_H"], c_rec["lower"], c_rec["upper"], ref)
            row.update(width_rec=float(np.mean(c_rec["upper"] - c_rec["lower"])), adj_sep_rec=frac, n_adj_sep_rec=n_sep, agree_rec=agree, n_sep_rec=n_all,
                       contrast_se_rec=np.round(se_rec, 4).tolist())
            if sw is not None:
                row.update(se_rec_over_cell_med=float(np.median(se_rec / np.asarray(sw["contrasts"]["se"], float))))
        except Exception as e:  # noqa: BLE001
            row.update(record_sandwich_error=repr(e))
        # ---- two-way sandwich (records and judge-pair cells), the interval reported on real data
        se_two = None
        try:
            c_two = sandwich_at_weight(sel, lam_hat, N, K, r, pairs, A[0], A[1], A[2], A[3], cluster="twoway", llm_obs=llm_obs, human_obs=human_obs)
            se_two = np.asarray(c_two["se"], float)
            frac, n_sep = adjacent_separation(sel["s_H"], c_two["lower"], c_two["upper"])
            agree, n_all = separated_sign_agreement(sel["s_H"], c_two["lower"], c_two["upper"], ref)
            row.update(width_two=float(np.mean(c_two["upper"] - c_two["lower"])), adj_sep_two=frac, n_adj_sep_two=n_sep, agree_two=agree, n_sep_two=n_all,
                       contrast_se_two=np.round(se_two, 4).tolist())
            if sw is not None:
                row.update(se_two_over_cell_med=float(np.median(se_two / np.asarray(sw["contrasts"]["se"], float))))
            if se_rec is not None:
                row.update(se_two_over_rec_med=float(np.median(se_two / se_rec)))
        except Exception as e:  # noqa: BLE001
            row.update(twoway_sandwich_error=repr(e))
        # ---- bootstrap: records resampled within the calibration and the proxy-only strata
        groups = llm.groupby("record").indices
        records = np.array(list(groups.keys()))
        cal_recs = set(cal["record"].tolist())
        is_cal = np.array([rr in cal_recs for rr in records])
        cal_rows = {rr: np.flatnonzero(cal["record"].to_numpy() == rr) for rr in cal_recs}
        strata = [np.flatnonzero(is_cal), np.flatnonzero(~is_cal)]
        S_post, S_fixed, lams = np.full((B, N), np.nan), np.full((B, N), np.nan), np.full(B, np.nan)
        rng = np.random.default_rng([int(seed), rb.DATASET_CODE[dataset], 7, int(level)])
        for bidx in range(B):
            draw = draw_within_strata(strata, rng, scheme)
            idx = np.concatenate([groups[records[t]] for t in draw])
            llm_b = llm.iloc[idx]
            cal_b = cal.iloc[np.concatenate([cal_rows[records[t]] for t in draw if is_cal[t]])]
            try:
                llm_b, K_b, _, _ = rb.drop_empty_judges(llm_b, K, K)
                if K_b != K:
                    continue
                sel_b = _fit(N, K, r, rb.llm_arrays(llm_b, N, K), rb.human_pairs(cal_b), lam_grid)
                S_post[bidx] = sel_b["s_H"]
                lams[bidx] = float(sel_b["lam"])
                fx = candidate_at(sel_b, lam_hat)
                if fx is not None:
                    S_fixed[bidx] = fx["s_H"]
            except Exception:  # noqa: BLE001
                continue
        bs, bf = summarize_contrasts(sel["s_H"], S_post), summarize_contrasts(sel["s_H"], S_fixed)
        ok = ~np.isnan(lams)
        row.update(n_boot=bs["n_boot"], n_boot_fixed=bf["n_boot"], width_boot=float(np.nanmean(bs["upper_pct"] - bs["lower_pct"])),
                   lam_boot_inf_frac=float(np.mean(np.isinf(lams[ok]))) if ok.any() else np.nan,
                   lam_boot_zero_frac=float(np.mean(lams[ok] == 0)) if ok.any() else np.nan,
                   lam_boot_same_frac=float(np.mean(lams[ok] == lam_hat)) if ok.any() else np.nan,
                   lam_boot_rel_med=float(np.nanmedian(np.where(np.isinf(lams[ok]), np.nan, lams[ok]) * len(cal) / len(llm))) if ok.any() else np.nan)
        frac, n_sep = adjacent_separation(sel["s_H"], bs["lower_pct"], bs["upper_pct"])
        agree, n_all = separated_sign_agreement(sel["s_H"], bs["lower_pct"], bs["upper_pct"], ref)
        row.update(adj_sep_boot=frac, n_adj_sep_boot=n_sep, agree_boot=agree, n_sep_boot=n_all)
        if sw is not None:
            se_sand = np.asarray(sw["contrasts"]["se"], float)
            row.update({f"se_ratio_{k}": v for k, v in _quartiles(bs["se"] / se_sand).items()})
            row.update({f"se_ratio_fixed_{k}": v for k, v in _quartiles(bf["se"] / se_sand).items()})
        if se_rec is not None:
            row.update({f"se_ratio_rec_{k}": v for k, v in _quartiles(bs["se"] / se_rec).items()})
            row.update({f"se_ratio_fixed_rec_{k}": v for k, v in _quartiles(bf["se"] / se_rec).items()})
        if se_two is not None:
            row.update({f"se_ratio_two_{k}": v for k, v in _quartiles(bs["se"] / se_two).items()})
            row.update({f"se_ratio_fixed_two_{k}": v for k, v in _quartiles(bf["se"] / se_two).items()})
        row.update(contrast_est=np.round(bs["estimate"], 4).tolist(), contrast_se_boot=np.round(bs["se"], 4).tolist(),
                   contrast_se_sand=(np.round(sw["contrasts"]["se"], 4).tolist() if sw is not None else None), s_hat=np.round(sel["s_H"], 4).tolist())
        out.append(row)
    except Exception as e:  # noqa: BLE001
        out.append(dict(base, method="dial_mu", error=repr(e) + traceback.format_exc()[-200:]))
    # ---- Human-only, where its MLE exists: analytic intervals against the bootstrap of the calibration records
    if base["human_only_exists"]:
        try:
            h = fit_human_only_btl(N, pairs)
            uq = human_only_uq(N, pairs, h["s_H"])
            rng = np.random.default_rng([int(seed), rb.DATASET_CODE[dataset], 8, int(level)])
            S_h = np.full((B, N), np.nan)
            for bidx in range(B):
                cal_b = cal.iloc[rng.integers(len(cal), size=len(cal))] if scheme == "boot" else cal.iloc[rng.choice(len(cal), size=len(cal) // 2, replace=False)]
                try:
                    S_h[bidx] = fit_human_only_btl(N, rb.human_pairs(cal_b))["s_H"]
                except Exception:  # noqa: BLE001
                    continue
            bh = summarize_contrasts(h["s_H"], S_h)
            c = uq["contrasts"]
            frac, n_sep = adjacent_separation(h["s_H"], c["lower"], c["upper"])
            agree, n_all = separated_sign_agreement(h["s_H"], c["lower"], c["upper"], ref)
            fracb, n_sepb = adjacent_separation(h["s_H"], bh["lower_pct"], bh["upper_pct"])
            agreeb, n_allb = separated_sign_agreement(h["s_H"], bh["lower_pct"], bh["upper_pct"], ref)
            row = dict(base, method="human_only", lam=0.0, width_sand=float(np.mean(c["upper"] - c["lower"])), width_boot=float(np.nanmean(bh["upper_pct"] - bh["lower_pct"])),
                       n_boot=bh["n_boot"], adj_sep_sand=frac, n_adj_sep_sand=n_sep, agree_sand=agree, n_sep_sand=n_all,
                       adj_sep_boot=fracb, n_adj_sep_boot=n_sepb, agree_boot=agreeb, n_sep_boot=n_allb)
            row.update({f"se_ratio_{k}": v for k, v in _quartiles(np.asarray(bh["se"]) / np.asarray(c["se"], float)).items()})
            out.append(row)
        except Exception as e:  # noqa: BLE001
            out.append(dict(base, method="human_only", error=repr(e) + traceback.format_exc()[-200:]))
    secs = time.perf_counter() - t0
    for m in out:
        m["seconds_cell"] = secs
    return out


# --------------------------------------------------------------------------- two-way cluster-robust check
def _labels(llm):
    rec = llm["record"].astype(str).to_numpy()
    judge = llm["k"].astype(str).to_numpy()
    cell = (llm["k"].astype(str) + "_" + llm["i"].astype(str) + "_" + llm["j"].astype(str)).to_numpy()
    return rec, cell, np.char.add(np.char.add(rec, "|"), judge)


def _llm_obs(llm, rec):
    return dict(k=llm["k"].to_numpy(int), i=llm["i"].to_numpy(int), j=llm["j"].to_numpy(int), y=llm["y"].to_numpy(float),
                a=llm["a"].to_numpy(float), rec=rec)


def twoway_score_job(job):
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


def twoway_check(seeds, workers, out):
    cfg = rb.load_config()
    jobs = [(d, lv, s, cfg) for d in ("arena_33k", "mt_bench") for lv in (200, 800) for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        scores = list(ex.map(twoway_score_job, jobs, chunksize=1))
    orders = [order_effect_check(d, cfg) for d in cfg["study"]["datasets"]]
    json.dump(dict(calibrated_score=scores, order_effects=orders), open(out, "w"), indent=1)
    print(f"{len(scores)} score cells and {len(orders)} order-effect fits -> {out}")


# --------------------------------------------------------------------------- resampling scheme and the weight selection
SCHEMES = ("boot", "sub", "dup")


def _draw(scheme, strata, rng):
    out = []
    for idx in strata:
        n = idx.size
        if scheme == "boot":
            out.append(rng.choice(idx, size=n, replace=True))
        elif scheme == "sub":
            out.append(rng.choice(idx, size=int(round((1 - np.exp(-1)) * n)), replace=False))
        else:
            out.append(np.concatenate([idx, rng.choice(idx, size=int(round(np.exp(-1) * n)), replace=False)]))
    return np.concatenate(out)


def _lam_code(lam):
    if lam == 0:
        return "0"
    if np.isinf(lam):
        return "inf"
    return f"{float(lam):.4g}"


def selection_job(job):
    dataset, level, seed, scheme, B, cfg = job
    cell = rb.build_cell((dataset, "budget", "all", "none", level, None, seed, cfg))
    N, K, A, pairs, llm, cal, grid = (cell[k] for k in ("N", "K", "A", "pairs", "llm", "cal", "lam_grid"))
    sel = _fit(N, K, 1, A, pairs, grid)
    lam_hat = float(sel["lam"])
    D = pairwise_contrast_matrix(N)
    groups = llm.groupby("record").indices
    records = np.array(list(groups.keys()))
    cal_recs = set(cal["record"].tolist())
    is_cal = np.array([r in cal_recs for r in records])
    cal_rows = {r: np.flatnonzero(cal["record"].to_numpy() == r) for r in cal_recs}
    strata = [np.flatnonzero(is_cal), np.flatnonzero(~is_cal)]
    rng = np.random.default_rng([int(seed), rb.DATASET_CODE[dataset], 17, int(level), SCHEMES.index(scheme)])
    lams, S_sel, S_fix = [], [], []
    for _ in range(B):
        draw = _draw(scheme, strata, rng)
        llm_b = llm.iloc[np.concatenate([groups[records[t]] for t in draw])]
        cal_b = cal.iloc[np.concatenate([cal_rows[records[t]] for t in draw if is_cal[t]])]
        try:
            llm_b, K_b, _, _ = rb.drop_empty_judges(llm_b, K, K)
            if K_b != K:
                continue
            sb = _fit(N, K, 1, rb.llm_arrays(llm_b, N, K), rb.human_pairs(cal_b), grid)
        except Exception:  # noqa: BLE001
            continue
        fx = candidate_at(sb, lam_hat)
        lams.append(float(sb["lam"]))
        S_sel.append(D @ sb["s_H"])
        S_fix.append(D @ fx["s_H"] if fx is not None else np.full(D.shape[0], np.nan))
    S_sel, S_fix = np.array(S_sel), np.array(S_fix)
    sd_sel, sd_fix = np.nanstd(S_sel, axis=0, ddof=1), np.nanstd(S_fix, axis=0, ddof=1)
    iqr = np.nanpercentile(S_sel, 75, axis=0) - np.nanpercentile(S_sel, 25, axis=0)
    codes = [_lam_code(l) for l in lams]
    finite = sorted(float(g) for g in grid)
    rank = {**{"0": 0}, **{_lam_code(g): i + 1 for i, g in enumerate(finite)}, "inf": len(finite) + 1}
    r_hat = rank[_lam_code(lam_hat)]
    rb_ = np.array([rank[c] for c in codes])
    return dict(dataset=dataset, level=level, seed=seed, scheme=scheme, n_rep=len(lams), lam_hat=_lam_code(lam_hat),
                same=float(np.mean(rb_ == r_hat)), smaller=float(np.mean(rb_ < r_hat)), larger=float(np.mean(rb_ > r_hat)),
                zero=float(np.mean(rb_ == 0)), inf=float(np.mean(rb_ == len(finite) + 1)),
                sel_over_fix=float(np.nanmedian(sd_sel / sd_fix)), sd_over_iqr=float(np.nanmedian(sd_sel / (iqr / 1.349))),
                lam_counts={c: codes.count(c) for c in sorted(set(codes), key=lambda c: rank[c])})


def selection_check(seeds, B, workers, out):
    cfg = rb.load_config()
    jobs = [(d, lv, s, sc, B, cfg) for d in ("arena_33k", "mt_bench") for lv in (200, 800) for s in seeds for sc in SCHEMES]
    with ProcessPoolExecutor(workers) as ex, open(out, "w") as f:
        for r in ex.map(selection_job, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("done", flush=True)


# --------------------------------------------------------------------------- command line
def existing_keys(path):
    keys = set()
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            keys.add((d["dataset"], d["panel"], float(d["level"]), d["seed"]))
    return keys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="arena_33k")
    p.add_argument("--panels", default="all")
    p.add_argument("--levels", default="200,800", help="human budgets, comma-separated")
    p.add_argument("--seeds", default=None, help="start:stop split seeds (default 0:5; 0:10 / 0:3 for the twoway / selection check)")
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--scheme", choices=("boot", "half"), default="boot", help="record bootstrap or half-sampling without replacement")
    p.add_argument("--check", choices=("twoway", "selection"), default=None, help="run one of the two checks instead of the resampling study")
    a = p.parse_args(argv)
    s0, s1 = (int(x) for x in (a.seeds or {None: "0:5", "twoway": "0:10", "selection": "0:3"}[a.check]).split(":"))
    name = {None: "rows_half.jsonl" if a.scheme == "half" else "rows.jsonl", "twoway": "twoway_check.json", "selection": "selection_check.jsonl"}[a.check]
    out = Path(a.out or f"results/alternatives/bootstrap/{name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.check == "twoway":
        return twoway_check(range(s0, s1), a.workers, out)
    if a.check == "selection":
        return selection_check(range(s0, s1), a.B, a.workers, out)
    cfg = rb.load_config()
    done = existing_keys(out)
    jobs = [(d, pn, int(lv), s, cfg, a.B, a.scheme) for d in a.datasets.split(",") for pn in a.panels.split(",") for lv in a.levels.split(",") for s in range(s0, s1)
            if (d, pn, float(lv), s) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} stored) -> {out}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(out, "a") as f:
        for k, recs in enumerate(ex.map(run_job, jobs, chunksize=1)):
            for m in recs:
                f.write(json.dumps(m) + "\n")
            f.flush()
            print(f"{k + 1}/{len(jobs)} jobs, {time.perf_counter() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
