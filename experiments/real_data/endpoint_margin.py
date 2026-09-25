"""Endpoint margin for GACV selection and the summary of keyed real-data runs.

The margin rule keeps DIAL's infinity endpoint unless the best finite weight improves GACV by more
than c / n_H; it reuses the GACV path, so it needs no extra fits.
"""
from __future__ import annotations

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
