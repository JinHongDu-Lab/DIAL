"""Exploratory exact human-comparison LOOCV for DIAL-mu; no reserved tuning set.

All LLM rows and numerical lambda stay fixed in each deletion. Identical human
(pair, outcome) deletions are evaluated once and weighted by multiplicity.
The independent outer human test pool is used only for experiment evaluation.
"""
from __future__ import annotations
from . import robustness as rb  # sets BLAS thread limits before numpy import
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from dial_judge.baselines import fit_consensus_only_calibrated, fit_human_only_btl
from dial_judge.dial_model import fit_human_calibration, joint
from dial_judge.gacv import select_lambda, btl_mle_exists
from dial_judge.evaluate import heldout_log_loss

OUT = rb.ROOT / 'results' / 'loocv_pilot'
LOW_H = {'arena_33k': 50, 'mt_bench': 20, 'pandalm': 20}
LOW_L = {'arena_33k': 500, 'mt_bench': 40, 'pandalm': 25}


def delete_one(pairs, i, j, y):
    reduced=[]
    for a,b,n,w in pairs:
        if (a,b)==(i,j):
            n,w=n-1,w-y
        if n>0: reduced.append((a,b,n,w))
    return reduced


def deletion_cells(pairs):
    for i,j,n,y in pairs:
        if n-y>0: yield i,j,0,int(n-y)
        if y>0: yield i,j,1,int(y)


def choose(candidates, nh):
    valid=[c for c in candidates if c['loo_valid']]
    if not valid:
        return {'loo':np.inf,'loo_conservative':np.inf}, dict(fallback=True)
    best=min(valid,key=lambda c:(c['loo'], -c['lam']))
    endpoint=next((c for c in valid if np.isinf(c['lam'])),None)
    conservative=best['lam']
    info=dict(fallback=False, gain_vs_inf=None, paired_dispersion_se=None)
    if endpoint is not None:
        d=np.asarray(endpoint['deletion_losses'])-np.asarray(best['deletion_losses'])
        count=np.asarray(best['multiplicities'])
        gain=float(np.sum(count*d)/nh)
        # Descriptive one-SE-style margin, NOT a valid independent-fold SE or test.
        se=float(np.sqrt(np.sum(count*(d-gain)**2)/(nh-1)/nh))
        conservative=best['lam'] if gain>se else np.inf
        info.update(gain_vs_inf=gain,paired_dispersion_se=se)
    return {'loo':best['lam'],'loo_conservative':conservative},info


def run(job):
    dataset,panel_name,regime,seed=job
    start=time.monotonic()
    cfg=rb.load_config()
    panel=rb.restrict_panel(rb.get_panel(dataset,cfg),None if panel_name=='all' else cfg['panels'][panel_name])
    code=rb.DATASET_CODE[dataset]
    test,train=rb.split_records(panel['records'],cfg[dataset]['f_test'],np.random.default_rng([seed,code,1]))
    llm=panel['llm'][panel['llm']['record'].isin(train)].reset_index(drop=True)
    nh=LOW_H[dataset] if regime=='human_scarce' else cfg[dataset]['n_H']
    llm=rb.subsample_rows(llm,-1 if regime=='human_scarce' else LOW_L[dataset],np.random.default_rng([seed,code,3]))
    llm,K,_,_=rb.drop_empty_judges(llm,panel['K'],panel['K'])
    cal=rb.draw_budget(panel['human'][panel['human']['record'].isin(train)].reset_index(drop=True),nh,np.random.default_rng([seed,code,2]))
    test_h=panel['human'][panel['human']['record'].isin(test)&(panel['human']['y']!=.5)]
    pairs=rb.human_pairs(cal)
    assert all(float(y).is_integer() and float(n).is_integer() for i,j,n,y in pairs)
    nh=len(cal); N=panel['N']; rank=min(1,K-1,N-2)
    A=rb.llm_arrays(llm,N,K)
    grid=[m*len(llm)/nh for m in cfg['study']['lambda_multipliers']]
    st=fit_consensus_only_calibrated(N,K,rank,A[0],A[1],pairs,n_order=A[2],y_order=A[3])
    selected=select_lambda(N,K,rank,pairs,n_ijk_llm=A[0],y_ijk_llm=A[1],n_order=A[2],y_order=A[3],lambda_grid=grid,staged_fit=st,return_all=True,align='mu')
    fitmap={float(l):fit for l,fit in selected['candidates']}
    cells=list(deletion_cells(pairs))
    base=dict(dataset=dataset,panel=panel_name,regime=regime,seed=seed,n_H=nh,n_L=len(llm),N=N,K=K,unique_deletions=len(cells))
    paths=[]
    for entry in selected['gacv_path']:
        lam=float(entry['lam']); full=fitmap[lam]
        row=dict(base,lam=lam,gacv=float(entry['gacv']),full_regular=bool(entry['regular']),deletion_losses=[],multiplicities=[c[3] for c in cells],deletion_issues=[])
        if not entry['regular']:
            row.update(loo_valid=False,loo=None)
            paths.append(row);continue
        for i,j,y,count in cells:
            reduced=delete_one(pairs,i,j,y)
            try:
                if np.isinf(lam):
                    alpha,_=fit_human_calibration(st['mu'],np.zeros((N,0)),reduced)
                    fit={'s_H':alpha*st['mu'],'fit_info':{'converged':True}}
                elif lam==0:
                    if not btl_mle_exists(N,reduced): raise ValueError('human MLE absent after deletion')
                    fit=fit_human_only_btl(N,reduced)
                else:
                    fit=joint(N,K,rank,reduced,n_ijk_llm=A[0],y_ijk_llm=A[1],n_order=A[2],y_order=A[3],lam=lam,init_params=tuple(full[k] for k in ['gamma','mu','U','V','b']),align='mu')
                score=np.asarray(fit['s_H']); diff=float(score[i]-score[j])
                loss=float(np.logaddexp(0,diff)-y*diff)
                issues=[]
                if not fit['fit_info'].get('converged',True): issues.append('nonconverged')
                if not np.isfinite(score).all() or np.max(np.abs(score))>10: issues.append('score_guard')
                if issues: row['deletion_issues'].append(dict(i=i,j=j,y=y,issues=issues))
                row['deletion_losses'].append(loss)
            except Exception as e:
                row['deletion_losses'].append(None)
                row['deletion_issues'].append(dict(i=i,j=j,y=y,issues=[repr(e)]))
        row['loo_valid']=not row['deletion_issues']
        row['loo']=float(np.average(row['deletion_losses'],weights=row['multiplicities'])) if all(v is not None for v in row['deletion_losses']) else None
        paths.append(row)
    choices,info=choose(paths,nh)
    choices.update(gacv=float(selected['lam']),consensus=np.inf)
    # Outer test labels are first used here, after all tuning decisions are frozen.
    test_records=rb.human_records(test_h)
    ref=fit_human_only_btl(N,rb.human_pairs(test_h))['s_H']
    floor=heldout_log_loss(ref,test_records)
    rows=[]
    for method,lam in choices.items():
        score=fitmap[lam]['s_H']
        rows.append(dict(base,method=method,lam=lam,excess=heldout_log_loss(score,test_records)-floor,logloss=heldout_log_loss(score,test_records),tau=float(kendalltau(ref,score).statistic),**info))
    return dict(**base,rows=rows,paths=paths,seconds=time.monotonic()-start)


def hybrid_rows(job):
    """Illustrative small-nH / close-GACV trigger; no outer outcomes used."""
    regular=[p for p in job['paths'] if p['full_regular']]
    finite=[p['gacv'] for p in regular if np.isfinite(p['lam'])]
    endpoint=next((p['gacv'] for p in regular if np.isinf(p['lam'])),None)
    gap=abs(min(finite)-endpoint) if finite and endpoint is not None else np.inf
    trigger=job['n_H']<=50 or gap<=1/job['n_H']
    by_method={r['method']:r for r in job['rows']}
    return [dict(by_method[source if trigger else 'gacv'],method=name,loo_trigger=trigger,gacv_gap_to_inf=gap)
            for name,source in [('hybrid_loo','loo'),('hybrid_conservative','loo_conservative')]]


def summarize():
    jobs=[json.loads(s) for s in (OUT/'jobs.jsonl').read_text().splitlines()]
    rows=pd.DataFrame([r for j in jobs for r in j['rows']+hybrid_rows(j)])
    rows.to_csv(OUT/'rows.csv',index=False)
    agg=[]
    for key,g in rows.groupby(['dataset','panel','regime']):
        baseline=g[g.method=='gacv'].set_index('seed')
        for method,z in g.groupby('method'):
            z=z.set_index('seed');delta=z.excess-baseline.excess
            agg.append(dict(dataset=key[0],panel=key[1],regime=key[2],method=method,seeds=len(z),excess=z.excess.mean(),tau=z.tau.mean(),delta_vs_gacv=delta.mean(),paired_mcse=delta.sem(),infinite_share=np.isinf(z.lam).mean()))
    table=pd.DataFrame(agg);table.to_csv(OUT/'summary.csv',index=False)
    paths=[p for j in jobs for p in j['paths']]
    diag=dict(jobs=len(jobs),evaluated_candidates=sum(p['full_regular'] for p in paths),full_irregular_candidates=sum(not p['full_regular'] for p in paths),candidates_with_deletion_issues=sum(bool(p['deletion_issues']) for p in paths),deletion_issues=sum(len(p['deletion_issues']) for p in paths),loo_all_invalid_fallbacks=sum(j['rows'][0]['fallback'] for j in jobs),summed_job_seconds=sum(j['seconds'] for j in jobs))
    (OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    print(table.round(4).to_string(index=False));print(diag)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--workers',type=int,default=6);p.add_argument('--summarize',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.summarize: summarize();return
    done=set()
    if (OUT/'jobs.jsonl').exists():
        done={(j['dataset'],j['panel'],j['regime'],j['seed']) for j in map(json.loads,(OUT/'jobs.jsonl').read_text().splitlines())}
    jobs=[(d,'small6',r,s) for d in LOW_H for r in ['human_scarce','llm_scarce'] for s in range(args.seeds) if (d,'small6',r,s) not in done]
    (OUT/'design.json').write_text(json.dumps(dict(seeds=args.seeds,panels=['small6'],low_h=LOW_H,low_l=LOW_L,config=rb.load_config(),conservative_rule='one paired dispersion SE; exploratory heuristic, not confidence bound',fixed_numerical_lambda=True,conditional_on_full_llm=True),indent=2)+'\n')
    start=time.monotonic();print(f'{len(jobs)} jobs',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool,(OUT/'jobs.jsonl').open('a') as handle:
        futures={pool.submit(run,j):j for j in jobs}
        for count,fut in enumerate(as_completed(futures),1):
            result=fut.result();handle.write(json.dumps(result)+'\n');handle.flush()
            print(f'{count}/{len(jobs)} {futures[fut]} in {result["seconds"]:.1f}s; elapsed {time.monotonic()-start:.0f}s',flush=True)
    summarize()

if __name__=='__main__': main()
