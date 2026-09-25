"""Resumable process-pool runner: one `{"key": ..., "rows": [...]}` JSONL line per cell."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def completed_keys(path):
    """Keys already written to a keyed JSONL file (empty when it does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return set()
    return {tuple(json.loads(line)["key"]) for line in path.read_text().splitlines() if line.strip()}


def _progress(count, total, start, every, label):
    if count % every == 0 or count == total:
        print(f"{count}/{total} {label}; {time.monotonic() - start:.0f}s", flush=True)


def run_keyed_jobs(jobs, fn, dest, key_of, workers=8, every=25, label="cells"):
    """Run `fn(job)` over `jobs`, appending one `{"key", "rows"}` line per completed cell.

    `key_of(job)` must return the same JSON-serializable key used by `completed_keys`, so an
    interrupted run resumes by filtering `jobs` on that set beforehand.
    """
    jobs = list(jobs)
    dest = Path(dest)
    print(f"{len(jobs)} {label}", flush=True)
    start = time.monotonic()
    with dest.open("a") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, job): job for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            handle.write(json.dumps(dict(key=list(key_of(futures[future])), rows=rows)) + "\n")
            handle.flush()
            _progress(count, len(jobs), start, every, label)


def write_design(path, design, strict=True):
    """Record a study's design.

    With `strict`, refuse to mix a changed design into a directory that already holds results
    (the free-text `purpose` is not compared).
    """
    path = Path(path)
    strip = lambda d: {k: v for k, v in d.items() if k != "purpose"}
    if strict and path.exists() and strip(json.loads(path.read_text())) != strip(design):
        raise SystemExit(f"{path} records a different design; use a fresh output directory.")
    path.write_text(json.dumps(design, indent=2) + "\n")
