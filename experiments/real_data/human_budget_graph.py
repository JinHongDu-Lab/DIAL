"""Reconstruct training-only graph diagnostics for the paired G6/G8 human draws."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from . import robustness as rb


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=50);p.add_argument('--out',type=Path,default=Path('/tmp/dial-human-budget-graph'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=rb.load_config();rows=[]
    for ds in cfg['study']['datasets']:
        panel=rb.get_panel(ds,cfg);N=panel['N'];code=rb.DATASET_CODE[ds]
        for seed in range(a.seeds):
            _,train=rb.split_records(panel['records'],cfg[ds]['f_test'],np.random.default_rng([seed,code,1]))
            human=panel['human'][panel['human'].record.isin(train)].reset_index(drop=True)
            for level in cfg['sweeps']['budget']['levels'][ds]:
                cal=rb.draw_budget(human,level,np.random.default_rng([seed,code,2]));pairs=rb.human_pairs(cal);L=np.zeros((N,N));A=np.zeros((N,N))
                for i,j,n,_ in pairs:L[i,i]+=n;L[j,j]+=n;L[i,j]-=n;L[j,i]-=n;A[i,j]=A[j,i]=1
                gap=float(np.linalg.eigvalsh(L)[1]);rows.append(dict(dataset=ds,seed=seed,level=level,N=N,n_H=len(cal),connected=gap>1e-8,laplacian_gap=max(0,gap),minimum_comparisons=int(np.diag(L).min()),minimum_opponents=int(A.sum(axis=1).min()),human_only_exists=rb.btl_mle_exists(N,pairs)))
    x=pd.DataFrame(rows);assert len(x)==19*a.seeds;x.to_csv(a.out/'rows.csv',index=False)
    summary=x.groupby(['dataset','level']).agg(n_H=('n_H','mean'),connected_share=('connected','mean'),human_mle_exists_share=('human_only_exists','mean'),median_minimum_comparisons=('minimum_comparisons','median'),median_minimum_opponents=('minimum_opponents','median'),median_laplacian_gap=('laplacian_gap','median')).reset_index()
    summary.to_csv(a.out/'summary.csv',index=False);print(summary.round(3).to_string(index=False))

if __name__=='__main__':main()
