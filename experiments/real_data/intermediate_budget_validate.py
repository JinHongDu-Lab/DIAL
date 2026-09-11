"""Validate paired intermediate-budget outputs and their overlap with G7/pilot."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from .robustness import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    design=json.loads((root/'design.json').read_text());seeds=design['seeds']
    jobs=[json.loads(s) for s in (root/'jobs.jsonl').read_text().splitlines()]
    assert len(jobs)==57*seeds==len({tuple(j['key']) for j in jobs})
    for j in jobs:
        d={r['method']:r for r in j['rows']};assert len(j['rows'])==3;assert set(d)=={'consensus_cal','dial_mu','dial_margin_1'}
        ad=d['dial_margin_1'];ref=d['consensus_cal'] if np.isinf(ad['lam']) else d['dial_mu']
        assert ad['lam']==ref['lam'];assert abs(ad['excess']-ref['excess'])<1e-12;assert abs(ad['ref_kendall']-ref['ref_kendall'])<1e-12
        if ad['margin_reason']=='within_margin':assert np.isinf(ad['lam']) and ad['endpoint_gacv_gap']<=1/ad['n_H']
        if ad['margin_reason']=='finite_gain':assert np.isfinite(ad['lam']) and ad['endpoint_gacv_gap']>1/ad['n_H']
    x=pd.read_csv(root/'rows.csv');assert len(x)==171*seeds
    keys=['dataset','panel','level','n_H_level','method','seed'];assert not x.duplicated(keys).any()
    assert x.groupby(keys[:-1]).seed.apply(lambda z:set(z)==set(range(seeds))).all()
    assert x.groupby('dataset').n_L.unique().apply(list).to_dict()=={'arena_33k':[2000],'mt_bench':[160],'pandalm':[100]}
    checks=[]
    for name,expected in [('endpoint_margin_appendix',9*seeds),('intermediate_budget_pilot',171*min(seeds,5))]:
        old=pd.read_csv(ROOT/'results'/name/'rows.csv');old=old[old.sweep=='llm_budget']
        y=x.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(y)==expected
        delta=float((y.excess_new-y.excess_old).abs().max());assert delta<1e-12
        checks.append(dict(source=name,matched_rows=len(y),maximum_loss_difference=delta))
    diag=json.loads((root/'diagnostics.json').read_text());assert diag['exception_rows']==0
    bymethod=x.groupby('method').apply(lambda g:pd.Series(dict(nonconverged=int((g.converged==False).sum()),irregular=int((g.selected_regular==False).sum()),fallback=int((g.margin_reason=='all_irregular_fallback').sum()))),include_groups=False)
    bymethod.to_csv(root/'fit_diagnostics.csv')
    report=dict(cells=len(jobs),rows=len(x),seeds=seeds,paired_checks=checks,decision_formula_verified=True,candidate_predictions_reused=True,diagnostics=diag)
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    paths=['intermediate_budget.py','intermediate_budget_plot.py','intermediate_budget_validate.py','endpoint_margin.py','robustness.py','human_budget_graph.py']
    (root/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(report,indent=2));print(bymethod.to_string())

if __name__=='__main__':main()
