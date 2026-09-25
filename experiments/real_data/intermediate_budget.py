"""Human-budget sweep at a fixed intermediate LLM budget (main-text calibration figure)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import robustness as rb
from ._runner import completed_keys, run_keyed_jobs, write_design
from .endpoint_margin import summarize

LLM_LEVELS = {"arena_33k": [2000], "mt_bench": [160], "pandalm": [100]}
PANELS = ["small6", "large6", "all"]
PURPOSE = ("Human-budget sweep at the middle LLM budget of the llm_budget sweep, fixed before "
           "inspecting outcomes; endpoint margin c = 1; no test-based selection.")


def build_config(methods=None, datasets=None, panels=None):
    """The run's design; `methods`, `datasets` and `panels` narrow it for a later add-on pass."""
    cfg = rb.load_config()
    cfg["_only_methods"] = list(methods) if methods else ["consensus_cal", "dial_mu"]
    if not methods:
        cfg["_gacv_endpoint_margins"] = [1.0]
    sweep = cfg["sweeps"]["llm_budget"]
    sweep["panels"] = list(panels) if panels else PANELS
    sweep["levels"] = {d: v for d, v in LLM_LEVELS.items() if datasets is None or d in datasets}
    sweep["datasets"] = [d for d in sweep["datasets"] if datasets is None or d in datasets]
    sweep["n_H_grid"] = cfg["sweeps"]["budget"]["levels"]
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=rb.ROOT / "results" / "intermediate_budget")
    p.add_argument("--methods", default=None, help="comma-separated method keys for an add-on pass over the same cells")
    p.add_argument("--datasets", default=None, help="comma-separated datasets (default: all three)")
    p.add_argument("--panels", default=None, help="comma-separated judge panels (default: all three)")
    p.add_argument("--tag", default=None, help="suffix of the pass's own jobs/design files, e.g. `atc`")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    split = lambda v: [x.strip() for x in v.split(",")] if v else None

    cfg = build_config(split(a.methods), split(a.datasets), split(a.panels))
    suffix = f"_{a.tag}" if a.tag else ""
    write_design(a.out / f"design{suffix}.json", dict(seeds=a.seeds, config=cfg, purpose=PURPOSE))

    def key_of(job):
        return (job[0],) + rb.job_key(job)

    done = completed_keys(a.out / f"jobs{suffix}.jsonl")
    jobs = [j for j in rb.jobs_for("llm_budget", cfg, range(a.seeds)) if key_of(j) not in done]
    run_keyed_jobs(jobs, rb.run_cell, a.out / f"jobs{suffix}.jsonl", key_of, workers=a.workers)
    summarize(a.out)


if __name__ == "__main__":
    main()
