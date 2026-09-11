"""Exploratory human-budget sweep at the prechosen middle G7 LLM budgets."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import robustness as rb
from ._runner import completed_keys, run_keyed_jobs, write_design
from .endpoint_margin import summarize

LLM_LEVELS = {"arena_33k": [2000], "mt_bench": [160], "pandalm": [100]}
PANELS = ["small6", "large6", "all"]
PURPOSE = ("Fixed intermediate human-budget sweep; LLM budgets fixed to middle G7 levels before "
           "inspecting outcomes; c=1 fixed; no test-based selection.")


def build_config():
    cfg = rb.load_config()
    cfg["_only_methods"] = ["consensus_cal", "dial_mu"]
    cfg["_gacv_endpoint_margins"] = [1.0]
    sweep = cfg["sweeps"]["llm_budget"]
    sweep["panels"] = PANELS
    sweep["levels"] = LLM_LEVELS
    sweep["n_H_grid"] = cfg["sweeps"]["budget"]["levels"]
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("/tmp/dial-intermediate-budget"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = build_config()
    write_design(a.out / "design.json", dict(seeds=a.seeds, config=cfg, purpose=PURPOSE))

    def key_of(job):
        return (job[0],) + rb.job_key(job)

    done = completed_keys(a.out / "jobs.jsonl")
    jobs = [j for j in rb.jobs_for("llm_budget", cfg, range(a.seeds)) if key_of(j) not in done]
    run_keyed_jobs(jobs, rb.run_cell, a.out / "jobs.jsonl", key_of, workers=a.workers)
    summarize(a.out)


if __name__ == "__main__":
    main()
