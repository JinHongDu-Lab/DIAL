import numpy as np
import pytest
from experiments.real_data.loocv_pilot import delete_one, deletion_cells, choose
from dial_judge.dial_model import fit_human_calibration


def test_grouped_deletions_equal_individual_loocv():
    pairs=[(0,1,5.,3.),(0,2,4.,2.),(1,2,6.,4.)]
    mu=np.array([-.5,.1,.4]); losses=[]; weights=[]
    for i,j,y,count in deletion_cells(pairs):
        reduced=delete_one(pairs,i,j,y)
        assert sum(p[2] for p in reduced)==14
        assert sum(p[3] for p in reduced)==9-y
        alpha,_=fit_human_calibration(mu,np.zeros((3,0)),reduced)
        d=alpha*(mu[i]-mu[j]);loss=float(np.logaddexp(0,d)-y*d)
        losses.append(loss);weights.append(count)
    individual=[]
    records=[(i,j,y) for i,j,y,count in deletion_cells(pairs) for _ in range(count)]
    for t,(i,j,y) in enumerate(records):
        remaining=records[:t]+records[t+1:]
        reduced=[(a,b,sum(u==a and v==b for u,v,w in remaining),sum(w for u,v,w in remaining if u==a and v==b)) for a,b,n,w in pairs]
        alpha,_=fit_human_calibration(mu,np.zeros((3,0)),reduced)
        d=alpha*(mu[i]-mu[j]);individual.append(float(np.logaddexp(0,d)-y*d))
    assert np.average(losses,weights=weights)==pytest.approx(np.mean(individual),abs=1e-12)


def test_deleting_only_observation_drops_empty_pair():
    assert delete_one([(0,1,1.,1.),(1,2,2.,1.)],0,1,1)==[(1,2,2.,1.)]


def candidate(lam,losses,valid=True):
    return dict(lam=lam,loo=float(np.mean(losses)),loo_valid=valid,deletion_losses=losses,multiplicities=[1]*len(losses))


def test_conservative_selection_and_fallback():
    full=candidate(np.inf,[.5,.5,.5,.5])
    noisy=candidate(1.,[.1,.8,.5,.5])
    choices,_=choose([full,noisy],4)
    assert choices['loo']==1.
    assert np.isinf(choices['loo_conservative'])
    clear=candidate(1.,[.4,.4,.4,.4])
    assert choose([full,clear],4)[0]['loo_conservative']==1.
    assert np.isinf(choose([full,candidate(1.,[0.]*4,False)],4)[0]['loo'])
    assert choose([candidate(np.inf,[.5]*4,False)],4)[1]['fallback']


def test_hybrid_trigger_uses_training_scores_only():
    from experiments.real_data.loocv_pilot import hybrid_rows
    job=dict(n_H=80,paths=[dict(full_regular=True,lam=1.,gacv=.4),dict(full_regular=True,lam=np.inf,gacv=.6)],rows=[dict(method=m,lam=l,excess=e) for m,l,e in [('gacv',1.,10.),('loo',2.,0.),('loo_conservative',np.inf,-10.)]])
    assert not hybrid_rows(job)[0]['loo_trigger']
    assert hybrid_rows(job)[0]['lam']==1.
    job['paths'][1]['gacv']=.405
    assert hybrid_rows(job)[0]['loo_trigger']
    assert hybrid_rows(job)[0]['lam']==2.
    job['paths'][1]['gacv']=.6
    job['n_H']=20
    assert hybrid_rows(job)[0]['loo_trigger']
