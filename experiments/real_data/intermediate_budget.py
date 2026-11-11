"""Human-budget sweep at a fixed intermediate LLM budget (main-text calibration figure).

The LLM sample of every judge panel is subsampled to the middle level of the `llm_budget` sweep
(n_L = 2000 / 160 / 100 on Chatbot Arena / MT-Bench / PandaLM), and the human budget runs over the
levels of the `budget` sweep. Cells are those of `robustness.run_cell`; one keyed JSONL line is
written per cell, so an interrupted run resumes, and `--methods` with a `--tag` adds methods to the
same cells in a file of their own.

    python run_real_data.py intermediate --seeds 50 --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import robustness as rb

LLM_LEVELS = {"arena_33k": [2000], "mt_bench": [160], "pandalm": [100]}
PANELS = ["small6", "large6", "all"]
GROUP = ["dataset", "sweep", "panel", "level", "n_H_level"]


def build_config(methods=None, datasets=None, panels=None):
    """The run's design; `methods`, `datasets` and `panels` narrow it to a subset of the design."""
    cfg = rb.load_config()
    cfg["_only_methods"] = list(methods) if methods else ["consensus_cal", "dial_mu"]
    sweep = cfg["sweeps"]["llm_budget"]
    sweep["panels"] = list(panels) if panels else PANELS
    sweep["levels"] = {d: v for d, v in LLM_LEVELS.items() if datasets is None or d in datasets}
    sweep["datasets"] = [d for d in sweep["datasets"] if datasets is None or d in datasets]
    sweep["n_H_grid"] = cfg["sweeps"]["budget"]["levels"]
    return cfg


def completed_keys(path):
    """Keys already written to a keyed JSONL file (empty when it does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return set()
    return {tuple(json.loads(line)["key"]) for line in path.read_text().splitlines() if line.strip()}


def run_keyed_jobs(jobs, fn, dest, key_of, workers=8, every=25):
    """Run `fn(job)` over `jobs`, appending one `{"key", "rows"}` line per completed cell."""
    jobs = list(jobs)
    print(f"{len(jobs)} cells", flush=True)
    start = time.monotonic()
    with Path(dest).open("a") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, job): job for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            handle.write(json.dumps(dict(key=list(key_of(futures[future])), rows=future.result())) + "\n")
            handle.flush()
            if count % every == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)} cells; {time.monotonic() - start:.0f}s", flush=True)


def summarize(root):
    """Aggregate a run directory into rows.csv, summary.csv, and diagnostics.json.

    Every `jobs*.jsonl` of the directory is read, so a pass that adds a method to the same cells
    (written to its own file, e.g. `jobs_atc.jsonl`) is aggregated with the original run.
    """
    root = Path(root)
    rows = [r for path in sorted(root.glob("jobs*.jsonl"))
            for line in path.read_text().splitlines() for r in json.loads(line)["rows"]]
    x = pd.DataFrame(rows)
    x.to_csv(root / "rows.csv", index=False)

    # paired differences against the two reference methods within each cell, by seed
    out = []
    for key, g in x.groupby(GROUP):
        dial = g[g.method == "dial_mu"].set_index("seed")
        cons = g[g.method == "consensus_cal"].set_index("seed")
        for method, z in g.groupby("method"):
            z = z.set_index("seed")
            delta = z.excess - dial.excess
            out.append(dict(zip(GROUP, key), method=method, seeds=len(z), n_H=z.n_H.mean(), n_L=z.n_L.mean(),
                            excess=z.excess.mean(), excess_se=z.excess.sem(), tau=z.ref_kendall.mean(),
                            tau_se=z.ref_kendall.sem(), delta_vs_gacv=delta.mean(), paired_mcse=delta.sem(),
                            delta_vs_cons=(z.excess - cons.excess).mean(), infinite_share=np.isinf(z.lam).mean()))
    pd.DataFrame(out).to_csv(root / "summary.csv", index=False)

    diag = dict(rows=len(x),
                exception_rows=int(x.error.notna().sum()) if "error" in x else 0,
                nonconverged_rows=int((x.converged == False).sum()),          # noqa: E712 - NaN must not count
                irregular_selected_rows=int((x.selected_regular == False).sum()))  # noqa: E712
    (root / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print(diag, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=rb.ROOT / "results" / "intermediate_budget")
    p.add_argument("--methods", default=None, help="comma-separated method keys to run on the same cells")
    p.add_argument("--datasets", default=None, help="comma-separated datasets (default: all three)")
    p.add_argument("--panels", default=None, help="comma-separated judge panels (default: all three)")
    p.add_argument("--tag", default=None, help="suffix of the pass's own jobs/design files, e.g. `atc`")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    split = lambda v: [x.strip() for x in v.split(",")] if v else None

    cfg = build_config(split(a.methods), split(a.datasets), split(a.panels))
    suffix = f"_{a.tag}" if a.tag else ""
    design = a.out / f"design{suffix}.json"
    seeds = max(a.seeds, json.loads(design.read_text())["seeds"]) if design.exists() else a.seeds   # runs append seeds
    design.write_text(json.dumps(dict(seeds=seeds, config=cfg), indent=2) + "\n")

    def key_of(job):
        return (job[0],) + rb.job_key(job)

    done = completed_keys(a.out / f"jobs{suffix}.jsonl")
    jobs = [j for j in rb.jobs_for("llm_budget", cfg, range(a.seeds)) if key_of(j) not in done]
    run_keyed_jobs(jobs, rb.run_cell, a.out / f"jobs{suffix}.jsonl", key_of, workers=a.workers)
    summarize(a.out)


if __name__ == "__main__":
    main()
