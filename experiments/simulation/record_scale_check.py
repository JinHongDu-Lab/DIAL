"""Large-budget check of the record-clustered sandwich (Prop. record-cluster-inference).

Coverage of the cell- and record-clustered sandwich intervals of DIAL-Ada in the `record` row of
`main10_record` with record-effect s.d. 1, at larger human and proxy-only budgets than the main
grid, to see whether the finite-sample shortfall of the record sandwich shrinks as the theory
predicts. Only the sandwiches are of interest, so the bootstrap is run with B = 2.

    python -m experiments.simulation.record_scale_check --workers 16
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.simulation.record_bootstrap import run_cell

CELLS = [(2000, 800), (2000, 3200), (20000, 3200)]   # (proxy-only records n_U, human budget n_H)
KEYS = ("cov_sand_cell", "cov_sand_rec", "cov_sand_two", "cov_sand_cell_pre", "cov_sand_rec_pre", "cov_sand_two_pre", "lam", "lam_pre_rel")


def job(a):
    n_U, n_H, sd, seed = a
    d = [r for r in run_cell(("main10_record", "record", n_U, n_H, sd, seed, 2)) if r["method"] == "dial_mu"][0]
    return dict(n_U=n_U, n_H=n_H, record_sd=sd, seed=seed, **{k: d.get(k) for k in KEYS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="results/alternatives/simulation/main10_record/record_scale.json")
    a = p.parse_args()
    jobs = [(nU, nH, 1.0, s) for nU, nH in CELLS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, jobs, chunksize=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} cells -> {a.out}")


if __name__ == "__main__":
    main()
