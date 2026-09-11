import numpy as np
import pandas as pd
from experiments.real_data.llm_thinning import select_fraction


def candidates():
    return pd.DataFrame(dict(fraction=[.1,.5,1.],gacv=[.5,.6,.7],selected_regular=[True,True,True],logloss=[.9,.8,.1]))


def test_selection_does_not_use_test_loss():
    frame=candidates()
    assert select_fraction(frame).fraction==.1
    frame['logloss']=[0.,100.,-100.]
    assert select_fraction(frame).fraction==.1


def test_guards_ties_and_fallback():
    frame=candidates()
    frame.loc[0,'selected_regular']=False
    frame.loc[1,'gacv']=np.nan
    assert select_fraction(frame).fraction==1.
    frame['gacv']=.5
    frame['selected_regular']=True
    assert select_fraction(frame).fraction==1.
    frame['selected_regular']=False
    assert select_fraction(frame).fraction==1.
