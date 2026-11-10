"""Why does repeating the GACV selection inflate the record bootstrap? (diagnostic)

For a few `budget` cells of the record-bootstrap study, compare the GACV-selected weight and the
spread of the calibrated contrasts under three resampling schemes of the records (within the
labeled and proxy-only strata):

  boot   with replacement (the record bootstrap; about 37% of draws duplicate a record)
  sub    without replacement, about 63% of each stratum (the same number of distinct records as a
         bootstrap sample, with no duplicates)
  dup    the original records plus a duplicate of 37% of them (duplication without loss of data)

GACV approximates leave-one-out prediction error, so duplicated records can make human-heavy fits
look better. If duplication drives the selection shift, `boot` and `dup` move toward small weights
and `sub` does not. For each scheme the script also stores the s.d. of the contrasts after
selection and at the original selected weight, so selection-induced inflation is measured within
the same scheme.

    python -m experiments.real_data.selection_bootstrap_check --workers 16
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from dial_judge.inference import pairwise_contrast_matrix
from experiments.real_data import robustness as rb
from experiments.real_data.record_bootstrap import _fit
from experiments.simulation.record_bootstrap import candidate_at

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


def _lam_code(lam, grid):
    if lam == 0:
        return "0"
    if np.isinf(lam):
        return "inf"
    return f"{float(lam) * 1.0:.4g}"


def run(job):
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
    codes = [_lam_code(l, grid) for l in lams]
    finite = sorted(float(g) for g in grid)
    rank = {**{"0": 0}, **{_lam_code(g, grid): i + 1 for i, g in enumerate(finite)}, "inf": len(finite) + 1}
    r_hat = rank[_lam_code(lam_hat, grid)]
    rb_ = np.array([rank[c] for c in codes])
    return dict(dataset=dataset, level=level, seed=seed, scheme=scheme, n_rep=len(lams), lam_hat=_lam_code(lam_hat, grid),
                same=float(np.mean(rb_ == r_hat)), smaller=float(np.mean(rb_ < r_hat)), larger=float(np.mean(rb_ > r_hat)),
                zero=float(np.mean(rb_ == 0)), inf=float(np.mean(rb_ == len(finite) + 1)),
                sel_over_fix=float(np.nanmedian(sd_sel / sd_fix)), sd_over_iqr=float(np.nanmedian(sd_sel / (iqr / 1.349))),
                lam_counts={c: codes.count(c) for c in sorted(set(codes), key=lambda c: rank[c])})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--B", type=int, default=100)
    p.add_argument("--seeds", default="0:3")
    p.add_argument("--out", default="results/alternatives/bootstrap/selection_check.jsonl")
    a = p.parse_args()
    cfg = rb.load_config()
    s0, s1 = (int(x) for x in a.seeds.split(":"))
    jobs = [(d, lv, s, sc, a.B, cfg) for d in ("arena_33k", "mt_bench") for lv in (200, 800) for s in range(s0, s1) for sc in SCHEMES]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as f:
        for r in ex.map(run, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
