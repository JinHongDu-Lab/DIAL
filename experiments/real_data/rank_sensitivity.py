"""Rank sensitivity of the real-data budget study (Appendix G).

The main study fits the LLM side with rank r = 1 (`configs/real_robustness.toml`). This script
summarizes the same `budget` sweep refitted with r = 0 and r = 2
(`--llm-rank 0` and `--llm-rank 2`, results under `results/real_robustness_rank/r{0,2}`)
next to r = 1, for the rank-dependent estimators, and the BIC choice of r on the full LLM panels.

    python -m experiments.real_data.robustness --sweep budget --seeds 0:50 --workers 8 \
        --llm-rank 0 --methods consensus_cal,dial_mu,dial_w \
        --out results/real_robustness_rank/r0
    python -m experiments.real_data.rank_sensitivity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.real_data import robustness as rb

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("arena_33k", "mt_bench", "pandalm")
METHODS = ("consensus_cal", "dial_mu", "dial_w")


def _rows(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])


def load():
    frames = []
    for r, root in ((0, ROOT / "results" / "real_robustness_rank" / "r0"), (1, ROOT / "results" / "real_robustness"),
                    (2, ROOT / "results" / "real_robustness_rank" / "r2")):
        for d in DATASETS:
            df = _rows(root / d / "rows.jsonl")
            if df.empty:
                continue
            df = df[(df["sweep"] == "budget") & (df["panel"] == "all") & df["method"].isin(METHODS)]
            frames.append(df.assign(rank=r))
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    g = df.groupby(["dataset", "level", "method", "rank"])
    out = g.agg(excess=("excess", "mean"), excess_se=("excess", "sem"), kendall=("ref_kendall", "mean"),
                kendall_se=("ref_kendall", "sem"), n=("excess", "size")).reset_index()
    cfg = rb.load_config()
    bic = {}
    for d in DATASETS:
        best, table = rb.select_rank(d, cfg, rank_cap=3)
        bic[d] = dict(best=int(best), bic={int(k): float(v) for k, v in table.items()})
    dst = ROOT / "results" / "real_robustness_rank"
    dst.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / "summary.csv", index=False)
    json.dump(bic, open(dst / "bic.json", "w"), indent=1)
    print(out[out["method"] == "dial_mu"].pivot_table(index=["dataset", "level"], columns="rank", values=["excess", "kendall"]).round(4).to_string())
    print(json.dumps({d: v["best"] for d, v in bic.items()}))


if __name__ == "__main__":
    main()
