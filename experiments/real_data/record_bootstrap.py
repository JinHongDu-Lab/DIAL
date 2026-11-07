"""Record-level bootstrap of the DIAL-Ada intervals on real data (group 4: R1-Q4, R2-W2(ii)).

One job is a stored `budget` cell (dataset, panel, human budget, split seed) built exactly as the
robustness study builds it. The original fit gives the GACV-selected weight and the cell-clustered
sandwich intervals of all pairwise contrasts. B replicates resample the train records within two
strata, the calibration records (with their human label and all their judge rows) and the
remaining records (judge rows only), and repeat the whole fit, weight selection included, on the
original GACV grid. The job row stores, for DIAL-Ada and (where its MLE exists) Human-only:

  se ratio of bootstrap to sandwich over the contrasts (median and quartiles)
  the selected weight across replicates
  share of ranking-adjacent pairs separated at 95% by each interval, and the sign agreement of the
  separated contrasts with the held-out human ranking (the practical-value demonstration)

    python -m experiments.real_data.record_bootstrap --datasets arena_33k --levels 200,800 --seeds 0:5 --B 100 --workers 8
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
from dial_judge.bootstrap import adjacent_separation, resample_within_strata, separated_sign_agreement, summarize_contrasts
from dial_judge.gacv import select_lambda
from dial_judge.inference import human_only_uq
from experiments.real_data import robustness as rb
from experiments.simulation.record_bootstrap import candidate_at, sandwich_at_selection


def _fit(N, K, r, A, pairs, lam_grid):
    st = calibrate_llm_fit(fit_hja(N, K, r, n_ijk=A[0], y_ijk=A[1], n_order=A[2], y_order=A[3]), pairs, consensus_only=True)
    return select_lambda(N, K, r, pairs, n_ijk_llm=A[0], y_ijk_llm=A[1], n_order=A[2], y_order=A[3], staged_fit=st, lambda_grid=lam_grid, return_all=True, align="mu")


def _quartiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(med=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75))) if x.size else dict(med=np.nan, q1=np.nan, q3=np.nan)


def run_job(job):
    dataset, panel_name, level, seed, cfg, B = job
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
        # ---- bootstrap: records resampled within the calibration and the proxy-only strata
        rec_llm = llm["record"].to_numpy()
        groups = llm.groupby("record").indices
        records = np.array(list(groups.keys()))
        cal_recs = set(cal["record"].tolist())
        is_cal = np.array([rr in cal_recs for rr in records])
        cal_rows = {rr: np.flatnonzero(cal["record"].to_numpy() == rr) for rr in cal_recs}
        strata = [np.flatnonzero(is_cal), np.flatnonzero(~is_cal)]
        S_post, S_fixed, lams = np.full((B, N), np.nan), np.full((B, N), np.nan), np.full(B, np.nan)
        rng = np.random.default_rng([int(seed), rb.DATASET_CODE[dataset], 7, int(level)])
        for bidx in range(B):
            draw = resample_within_strata(strata, rng)
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
                cal_b = cal.iloc[rng.integers(len(cal), size=len(cal))]
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
    p.add_argument("--seeds", default="0:5")
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results/alternatives/bootstrap/rows.jsonl")
    a = p.parse_args(argv)
    cfg = rb.load_config()
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(out)
    jobs = [(d, pn, int(lv), s, cfg, a.B) for d in a.datasets.split(",") for pn in a.panels.split(",") for lv in a.levels.split(",") for s in range(s0, s1)
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
