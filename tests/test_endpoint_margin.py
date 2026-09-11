import math
import pytest
from experiments.real_data.endpoint_margin import choose_endpoint_margin


def path():
    return [dict(lam=math.inf,gacv=.50,regular=True),dict(lam=10.,gacv=.48,regular=True),dict(lam=1.,gacv=.47,regular=True),dict(lam=0.,gacv=.49,regular=True)]


def test_margin_and_ordinary_minimizer():
    p=path()
    assert choose_endpoint_margin(p,20,0)[0]==1.
    assert choose_endpoint_margin(p,20,.5)[0]==1.
    assert math.isinf(choose_endpoint_margin(p,20,1)[0])
    assert choose_endpoint_margin(p,100,1)[0]==1.
    assert math.isinf(choose_endpoint_margin(p,20,2)[0])


def test_guards_and_zero_endpoint():
    p=path();p[0]['regular']=False
    assert choose_endpoint_margin(p,20,100)[0]==1.
    p=path();p[1]['regular']=p[2]['regular']=False
    assert choose_endpoint_margin(p,100,.5)[0]==0.
    for point in p:point['regular']=False
    assert choose_endpoint_margin(p,20,1,fallback=math.inf)[1]=='all_irregular_fallback'


def test_ties_and_input_validation():
    p=path();p[0]['gacv']=.47
    assert math.isinf(choose_endpoint_margin(p,20,0)[0])
    with pytest.raises(ValueError):choose_endpoint_margin(p,0,1)
    with pytest.raises(ValueError):choose_endpoint_margin(p,20,-1)


def test_selection_does_not_read_test_performance():
    p=path()
    for i,point in enumerate(p):point['test_loss']=100*i
    before=choose_endpoint_margin(p,20,1)
    for point in p:point['test_loss']=-point['test_loss']
    assert choose_endpoint_margin(p,20,1)==before
