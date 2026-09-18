"""Alignment option: human score aligned to the consensus only (align="mu") versus within W (align="W")."""
from __future__ import annotations

import numpy as np
import pytest

from dial_judge.baselines import fit_consensus_only_calibrated, fit_staged_structured_calibration
from dial_judge.benchmarks import fit_dial
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated, pool_pairs
from dial_judge.gacv import select_lambda
from dial_judge.inference import joint_sandwich
from dial_judge.simulate import (
    compute_human_score, compute_score_matrix, generate_plan_calibration, generate_plan_parameters,
    generate_plan_position_effects, generate_random_human_comparisons, generate_random_llm_comparisons,
)


@pytest.fixture(scope="module")
def data():
    N, K, r = 8, 4, 1
    mu, gamma, U, V = generate_plan_parameters(N, K, r, random_seed=3)
    b = generate_plan_position_effects(K, random_seed=4)
    c_mu, c_v = generate_plan_calibration(V, c_v_sd=0.0, random_seed=5)      # rank-0 aligned human target
    S = compute_score_matrix(mu, gamma, U, V)
    s0 = compute_human_score(mu, V, c_mu, c_v)
    llm = generate_random_llm_comparisons(S, b, 6000, random_seed=6, swap_fraction=0.25)
    hum = generate_random_human_comparisons(s0, 300, random_seed=7)
    n_ijk, y_ijk = comparisons_to_aggregated(llm, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm, N, K)
    return dict(N=N, K=K, r=r, n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order, pairs=pool_pairs(hum), s0=s0)


def test_joint_align_mu_shapes_and_score(data):
    d = data
    fit = fit_dial(d["N"], d["K"], d["r"], d["n_ijk"], d["y_ijk"], d["pairs"], n_order=d["n_order"], y_order=d["y_order"], align="mu", max_steps=10)
    assert fit["a"].size == 0 and fit["V"].shape == (d["N"], d["r"]) and fit["align"] == "mu"
    assert np.allclose(fit["s_H"], fit["alpha_H"] * fit["mu"])
    assert fit["W"].shape == (d["N"], 1)
    fit_w = fit_dial(d["N"], d["K"], d["r"], d["n_ijk"], d["y_ijk"], d["pairs"], n_order=d["n_order"], y_order=d["y_order"], align="W", max_steps=10)
    assert fit_w["a"].size == d["r"] and fit_w["W"].shape == (d["N"], d["r"] + 1)
    # the mu-aligned fit is at least as good on the LLM side is not required; but both must be finite and close in excess
    assert np.isfinite(fit["s_H"]).all() and np.isfinite(fit_w["s_H"]).all()


def test_rank0_aligns_coincide(data):
    d = data
    f_mu = fit_dial(d["N"], d["K"], 0, d["n_ijk"], d["y_ijk"], d["pairs"], n_order=d["n_order"], y_order=d["y_order"], align="mu", max_steps=10)
    f_w = fit_dial(d["N"], d["K"], 0, d["n_ijk"], d["y_ijk"], d["pairs"], n_order=d["n_order"], y_order=d["y_order"], align="W", max_steps=10)
    assert np.allclose(f_mu["s_H"], f_w["s_H"], atol=1e-6)


def test_select_lambda_align_mu_and_sandwich(data):
    d = data
    st = fit_consensus_only_calibrated(d["N"], d["K"], d["r"], d["n_ijk"], d["y_ijk"], d["pairs"], n_order=d["n_order"], y_order=d["y_order"])
    assert np.atleast_1d(st["a"]).size == 0
    sel = select_lambda(d["N"], d["K"], d["r"], d["pairs"], n_ijk_llm=d["n_ijk"], y_ijk_llm=d["y_ijk"], n_order=d["n_order"], y_order=d["y_order"],
                        staged_fit=st, align="mu", lambda_grid=[20.0, 200.0], return_all=True)
    assert all(np.isfinite(p["gacv"]) for p in sel["gacv_path"])
    lams = {lam for lam, _ in sel["candidates"]}
    assert lams == {np.inf, 200.0, 20.0, 0.0}
    for lam, f in sel["candidates"]:
        if 0 < lam < np.inf:
            assert np.atleast_1d(f["a"]).size == 0 and f["V"].shape[1] == d["r"]
            sw = joint_sandwich(f, d["N"], d["K"], d["r"], d["pairs"], d["n_ijk"], d["y_ijk"], d["n_order"], d["y_order"], cluster="pair")
            dim = (d["K"] - 1) * (1 + d["r"]) + d["K"] + (d["N"] - 1) * (1 + d["r"]) + 1      # no a-coordinates
            assert sw["hessian"].shape == (dim, dim) and np.isfinite(sw["contrasts"]["lower"]).all()
    # a mismatched staged fit is rejected
    st_w = fit_staged_structured_calibration(d["N"], d["K"], d["r"], d["n_ijk"], d["y_ijk"], d["pairs"], n_order=d["n_order"], y_order=d["y_order"])
    with pytest.raises(ValueError):
        select_lambda(d["N"], d["K"], d["r"], d["pairs"], n_ijk_llm=d["n_ijk"], y_ijk_llm=d["y_ijk"], n_order=d["n_order"], y_order=d["y_order"], staged_fit=st_w, align="mu", lambda_grid=[20.0])
    # default alignment stays "W" and still works with the W staged fit
    sel_w = select_lambda(d["N"], d["K"], d["r"], d["pairs"], n_ijk_llm=d["n_ijk"], y_ijk_llm=d["y_ijk"], n_order=d["n_order"], y_order=d["y_order"], staged_fit=st_w, lambda_grid=[20.0])
    assert np.isfinite(sel_w["gacv"])


def test_select_lambda_all_irregular_falls_back_to_anchor():
    """When every candidate fails the guards, the selector returns the lam = inf endpoint."""
    from dial_judge.gacv import select_lambda
    rng = np.random.default_rng(3)
    N, K, r = 5, 3, 1
    n_ijk = np.full((K, N, N), 40.0); y_ijk = np.zeros((K, N, N))
    s_true = np.linspace(-1.5, 1.5, N)
    for k in range(K):
        for i in range(N):
            for j in range(N):
                if i != j:
                    y_ijk[k, i, j] = rng.binomial(40, 1 / (1 + np.exp(-(s_true[i] - s_true[j]))))
                    n_ijk[k, i, j] = 40.0
    # (i, j, n, y): item 0 is undefeated, so the win graph is not strongly connected and the
    # human-only MLE does not exist, while the other items have mixed results (finite scale fit)
    pairs = [(0, j, 3, 3) for j in range(1, N)] + [(1, 2, 3, 2), (2, 3, 3, 2), (3, 4, 3, 2), (1, 4, 3, 1)]
    sel = select_lambda(N, K, r, pairs, n_ijk_llm=n_ijk, y_ijk_llm=y_ijk, lambda_grid=[1e6, 1e7], max_abs_score=1e-9, align="mu")
    assert not sel["human_only_exists"]
    assert len(sel["dropped"]) == len(sel["gacv_path"])
    assert sel["lam"] == np.inf and np.all(np.isfinite(sel["s_H"]))


def test_select_lambda_excludes_box_active_but_warm_starts(data, monkeypatch):
    """A converged box-active finite-lambda fit warm-starts the next weight but never enters the argmin."""
    import dial_judge.gacv as gacv_mod

    d = data
    grid = [5.0, 20.0, 80.0]
    real_joint, inits = gacv_mod.joint, []

    def flagged_joint(*args, **kwargs):
        inits.append(kwargs.get("init_params"))
        fit = real_joint(*args, **kwargs)
        if kwargs["lam"] == 80.0:          # the first (largest) weight on the downward path
            fit["fit_info"] = {**fit["fit_info"], "boundary_any": True}
        return fit

    monkeypatch.setattr(gacv_mod, "joint", flagged_joint)
    sel = select_lambda(d["N"], d["K"], d["r"], d["pairs"], n_ijk_llm=d["n_ijk"], y_ijk_llm=d["y_ijk"], n_order=d["n_order"], y_order=d["y_order"],
                        lambda_grid=grid, return_all=True, align="mu")
    point = next(p for p in sel["gacv_path"] if p["lam"] == 80.0)
    assert not point["regular"] and "box_active" in point["reasons"]
    assert 80.0 in sel["dropped"] and sel["lam"] != 80.0
    fit80 = dict(sel["candidates"])[80.0]
    assert inits[1] is not None and np.allclose(inits[1][1], fit80["mu"])
