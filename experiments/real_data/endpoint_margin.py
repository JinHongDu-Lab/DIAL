"""Direct c/nH preference for DIAL's infinity endpoint; no extra fits or CV.

Run: python -m experiments.real_data.endpoint_margin --seeds 20 --workers 6
The optional robustness-runner hook evaluates all c values on the same fit path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = ["dataset", "sweep", "panel", "level", "n_H_level"]


def choose_endpoint_margin(path, n_h, c, fallback=math.inf):
    """Select from regular candidates using training GACV only.

    Return (lambda, reason, endpoint_minus_best_finite_GACV).
    Includes lambda=0 among finite candidates. Ties favor infinity, then larger
    finite weights. An irregular infinity fit cannot be reinstated by the margin.
    """
    if n_h <= 0 or c < 0 or not math.isfinite(c):
        raise ValueError('n_h must be positive and c finite and nonnegative')
    eligible = [p for p in path if p['regular'] and math.isfinite(p['gacv'])]
    finite = [p for p in eligible if math.isfinite(p['lam'])]
    endpoint = next((p for p in eligible if math.isinf(p['lam'])), None)
    if not eligible:
        return fallback, 'all_irregular_fallback', None
    if not finite:
        return math.inf, 'only_endpoint', None
    best = min(finite, key=lambda p: (p['gacv'], -p['lam']))
    if endpoint is None:
        return best['lam'], 'endpoint_irregular', None
    gap = endpoint['gacv'] - best['gacv']
    if gap <= c / n_h:
        return math.inf, 'within_margin', gap
    return best['lam'], 'finite_gain', gap


def summarize(root):
    """Aggregate one run's cells into rows.csv, summary.csv, and diagnostics.json."""
    root = Path(root)
    # every `jobs*.jsonl` of the directory, so a later pass that adds a method to the same cells
    # (written to its own file, e.g. `jobs_atc.jsonl`) is aggregated with the original run
    rows = [r for path in sorted(root.glob('jobs*.jsonl'))
            for line in path.read_text().splitlines() for r in json.loads(line)['rows']]
    x = pd.DataFrame(rows)
    x.to_csv(root / 'rows.csv', index=False)

    # paired differences against the two reference methods within each cell, by seed
    out = []
    for key, g in x.groupby(GROUP):
        baseline = g[g.method == 'dial_mu'].set_index('seed')
        cons = g[g.method == 'consensus_cal'].set_index('seed')
        for method, z in g.groupby('method'):
            z = z.set_index('seed')
            delta = z.excess - baseline.excess
            out.append(dict(zip(GROUP, key), method=method, seeds=len(z), n_H=z.n_H.mean(), n_L=z.n_L.mean(),
                            excess=z.excess.mean(), excess_se=z.excess.sem(), tau=z.ref_kendall.mean(),
                            tau_se=z.ref_kendall.sem(), delta_vs_gacv=delta.mean(), paired_mcse=delta.sem(),
                            delta_vs_cons=(z.excess - cons.excess).mean(), infinite_share=np.isinf(z.lam).mean()))
    pd.DataFrame(out).to_csv(root / 'summary.csv', index=False)

    diag = dict(rows=len(x),
                exception_rows=int(x.error.notna().sum()) if 'error' in x else 0,
                nonconverged_rows=int((x.converged == False).sum()),          # noqa: E712 - NaN must not count
                irregular_selected_rows=int((x.selected_regular == False).sum()),  # noqa: E712
                fallback_margin_rows=int((x.get('margin_reason') == 'all_irregular_fallback').sum()))
    (root / 'diagnostics.json').write_text(json.dumps(diag, indent=2) + '\n')
    print(diag, flush=True)


def main():
    from . import robustness as rb
    from ._runner import completed_keys, run_keyed_jobs, write_design

    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--panels', nargs='+', default=['all', 'small6'])
    a = p.parse_args()

    root = a.out or rb.ROOT / 'results' / 'endpoint_margin'
    root.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(root)
        return

    cfg = rb.load_config()
    cfg['_only_methods'] = ['consensus_cal', 'dial_mu']
    cfg['_gacv_endpoint_margins'] = [.5, 1., 2.]
    for sweep in ('budget', 'llm_budget'):
        cfg['sweeps'][sweep]['panels'] = a.panels
    cfg['sweeps']['llm_budget']['n_H_grid'] = {d: [cfg[d]['n_H']] for d in cfg['study']['datasets']}
    write_design(root / 'design.json', dict(
        seeds=a.seeds, config=cfg,
        rule='infinity if GACV(infinity)-min_finite_GACV <= c/nH; same regularity guards; c not selected using test results'),
        strict=False)   # --panels may legitimately differ between runs into one directory

    def key_of(job):
        return (job[0],) + rb.job_key(job)

    done = completed_keys(root / 'jobs.jsonl')
    jobs = [j for sweep in ('budget', 'llm_budget') for j in rb.jobs_for(sweep, cfg, range(a.seeds))
            if key_of(j) not in done]
    run_keyed_jobs(jobs, rb.run_cell, root / 'jobs.jsonl', key_of, workers=a.workers)
    summarize(root)


if __name__ == '__main__':
    main()
