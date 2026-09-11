"""Low-human-budget LLM thinning pilot; independent of manuscript outputs.

Run: python -m experiments.real_data.llm_thinning --seeds 50 --workers 8
Uses recorded LLM rows, record-disjoint human test pools, and the existing
rank-1 DIAL/GACV implementation. No human test outcomes enter selection.
Subsets are uniform draws from the same available training pool (not nested).
"""
from __future__ import annotations
import argparse
import json
from . import robustness as rb
from ._runner import run_row_jobs
import numpy as np
import pandas as pd

FRACTIONS = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
BUDGETS = {'arena_33k': [50, 100], 'mt_bench': [20, 40], 'pandalm': [20, 40]}
OUT = rb.ROOT / 'results' / 'llm_thinning'


def select_fraction(dial):
    """Choose using training GACV only, with regularity guards and a full-data fallback."""
    eligible = dial[dial.selected_regular.fillna(False).astype(bool) & np.isfinite(dial.gacv)]
    if len(eligible):
        return eligible.sort_values(['gacv', 'fraction'], ascending=[True, False]).iloc[0]
    return dial[dial.fraction == 1].iloc[0]


def run(job):
    dataset, panel_name, nh, seed = job
    cfg = rb.load_config()
    cfg['_only_methods'] = ['consensus_cal', 'dial_mu']
    panel = rb.restrict_panel(rb.get_panel(dataset, cfg), None if panel_name == 'all' else cfg['panels'][panel_name])
    _, train = rb.split_records(panel['records'], cfg[dataset]['f_test'], np.random.default_rng([seed, rb.DATASET_CODE[dataset], 1]))
    available = int(panel['llm']['record'].isin(train).sum())
    rows = []
    for fraction in FRACTIONS:
        nl = max(1, int(round(available * fraction)))
        fitrows = rb.run_cell((dataset, 'llm_budget', panel_name, 'none', nl, nh, seed, cfg))
        for row in fitrows:
            row.update(fraction=fraction, n_L_available=available)
        rows.extend(fitrows)
    return rows


def summarize():
    rows = [json.loads(line) for line in (OUT / 'rows.jsonl').read_text().splitlines()]
    df = pd.DataFrame(rows)
    failures = df[df['error'].notna()] if 'error' in df else df.iloc[:0]
    good = df[df['logloss'].notna()].copy()
    selections = []
    keys = ['dataset', 'panel', 'n_H', 'seed']
    for key, group in good.groupby(keys):
        dial = group[group.method == 'dial_mu']
        selected = select_fraction(dial)
        row = selected.to_dict()
        row['method'] = 'dial_select_nL_gacv'
        selections.append(row)
        oracle = dial.loc[dial.logloss.idxmin()].to_dict()
        oracle['method'] = 'oracle_nL_test_DIAGNOSTIC'
        selections.append(oracle)
    selected = pd.DataFrame(selections)
    selected.to_csv(OUT / 'selected.csv', index=False)
    combined = pd.concat([good, selected], ignore_index=True)
    combined.groupby(['dataset','panel','n_H','method','fraction']).agg(n=('excess','size'), excess=('excess','mean'), tau=('ref_kendall','mean')).to_csv(OUT / 'curves.csv')
    comparisons = []
    for key, group in good.groupby(['dataset','panel','n_H']):
        full = group[(group.method == 'dial_mu') & (group.fraction == 1)].set_index('seed')
        cons = group[(group.method == 'consensus_cal') & (group.fraction == 1)].set_index('seed')
        choices = [(f'fixed_{f:g}', group[(group.method == 'dial_mu') & (group.fraction == f)].set_index('seed')) for f in FRACTIONS]
        sub = selected[(selected.dataset == key[0]) & (selected.panel == key[1]) & (selected.n_H == key[2])]
        choices += [(m, g.set_index('seed')) for m,g in sub.groupby('method')]
        choices += [('consensus_full',cons)]
        for label, candidate in choices:
            delta = candidate.excess - full.excess
            comparisons.append(dict(dataset=key[0],panel=key[1],n_H=key[2],method=label,n=len(delta.dropna()),excess=candidate.excess.mean(),tau=candidate.ref_kendall.mean(),delta_vs_full=delta.mean(),paired_se=delta.std(ddof=1)/np.sqrt(delta.count()),delta_vs_cons=(candidate.excess-cons.excess).mean(),fraction_mean=candidate.fraction.mean()))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / 'comparisons.csv',index=False)
    diagnostics = dict(rows=len(df), failures=len(failures), nonconverged=int((good.converged == False).sum()), irregular_dial=int(((good.method == 'dial_mu') & (good.selected_regular == False)).sum()))
    (OUT / 'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(diagnostics,flush=True)
    print(comparison[comparison.method.isin(['fixed_1','consensus_full','dial_select_nL_gacv','oracle_nL_test_DIAGNOSTIC'])].round(4).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=50)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--summarize',action='store_true')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        summarize(); return
    done=set()
    if (OUT/'rows.jsonl').exists():
        existing=pd.read_json(OUT/'rows.jsonl',lines=True)
        for key,g in existing.groupby(['dataset','panel','n_H','seed']):
            if len(g)==2*len(FRACTIONS): done.add(key)
    jobs=[(d,p,h,s) for d,hs in BUDGETS.items() for p in ['all','small6','large6'] for h in hs for s in range(args.seeds) if (d,p,h,s) not in done]
    (OUT/'design.json').write_text(json.dumps(dict(fractions=FRACTIONS,budgets=BUDGETS,seeds=args.seeds,panels=['all','small6','large6'],selection='GACV across nL and lambda; full-data tie break',subsampling='uniform nonnested LLM rows; shared human train/test and calibration samples'),indent=2)+'\n')
    print(f'{len(FRACTIONS)} fits/job',flush=True)
    run_row_jobs(jobs, run, OUT/'rows.jsonl', workers=args.workers)
    summarize()

if __name__ == '__main__': main()
