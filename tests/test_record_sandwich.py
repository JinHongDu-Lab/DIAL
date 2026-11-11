"""Tests for the record-clustered sandwich meat (Appendix E.2, Prop. record-cluster-inference)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from dial_judge.gacv import hessian_from_grad, pack_reduced, q_lambda_grad, unpack_reduced
from dial_judge.hja import make_centering_basis
from dial_judge.inference import _llm_cell_design, joint_sandwich, llm_score_covariance, record_score_meat


@pytest.fixture(scope="module")
def cell():
    """A small matched design with a record effect, fitted once at a finite weight."""
    from experiments.simulation.record_bootstrap import fit_dial_mu
    from experiments.simulation.synthetic import simulate_cell

    c = simulate_cell("main10_record", "record", 300, 60, 1.0, 0)
    N, K, r, matched = c["N"], c["K"], c["r"], c["matched"]
    llm, hum = matched["llm"], matched["human"]
    sel, (n_ijk, y_ijk, n_order, y_order, pairs) = fit_dial_mu(N, K, r, llm, hum)
    lam, fit = next((l, f) for l, f in sel["candidates"] if np.isfinite(l) and l > 0)
    arr = np.asarray(llm, dtype=float)
    n_lab = len(hum)
    llm_obs = dict(k=arr[:, 0].astype(int), i=arr[:, 1].astype(int), j=arr[:, 2].astype(int), y=arr[:, 3], a=arr[:, 4],
                   rec=np.repeat(np.arange(arr.shape[0] // K), K))
    harr = np.asarray(hum, dtype=float)
    human_obs = dict(i=harr[:, 1].astype(int), j=harr[:, 2].astype(int), y=harr[:, 3], rec=np.arange(n_lab))
    zeta = pack_reduced(fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"], fit["alpha_H"], fit["a"], True)
    return dict(N=N, K=K, r=r, lam=float(lam), fit=dict(fit, lam=lam), zeta=zeta, pairs=pairs, A=(n_ijk, y_ijk, n_order, y_order),
                llm_obs=llm_obs, human_obs=human_obs, n_H=float(n_lab), n_L=float(arr.shape[0]))


def _meat(c, llm_obs=None, human_obs=None):
    N, K, r = c["N"], c["K"], c["r"]
    n_ijk, y_ijk, n_order, y_order = c["A"]
    return record_score_meat(c["zeta"], N, K, r, make_centering_basis(K), make_centering_basis(N), True, n_order, y_order, n_ijk, y_ijk,
                             c["lam"], c["n_H"], c["n_L"], llm_obs or c["llm_obs"], human_obs or c["human_obs"])


def _comparison_grads(c):
    """Brute-force per-comparison loss gradients by central differences of the linear predictors."""
    N, K, r, zeta = c["N"], c["K"], c["r"], c["zeta"]
    jb, ib = make_centering_basis(K), make_centering_basis(N)
    n_ijk, y_ijk, n_order, y_order = c["A"]
    o, h = c["llm_obs"], c["human_obs"]
    tri_i, tri_j = np.triu_indices(N, k=1)
    pidx = np.full((N, N), -1)
    pidx[tri_i, tri_j] = np.arange(tri_i.size)

    def etas(z):
        _, _, _, eta, keys = _llm_cell_design(z, N, K, r, jb, ib, True, n_order, y_order, n_ijk, y_ijk, return_keys=True)
        look = dict(zip(keys.tolist(), eta))
        key = (o["k"] * 2 + (o["a"] > 0)) * tri_i.size + pidx[o["i"], o["j"]]
        eL = np.array([look[q] for q in key])
        g, mu, U, V, b, alpha, a = unpack_reduced(z, N, K, r, jb, ib, True)
        a = np.atleast_1d(a)
        s = alpha * mu + (V[:, :a.size] @ a if a.size else 0.0)
        return eL, s[h["i"]] - s[h["j"]]

    eL0, eH0 = etas(zeta)
    eps = 1e-6
    dL, dH = np.zeros((eL0.size, zeta.size)), np.zeros((eH0.size, zeta.size))
    for d in range(zeta.size):
        e = np.zeros_like(zeta); e[d] = eps
        (lp, hp), (lm, hm) = etas(zeta + e), etas(zeta - e)
        dL[:, d], dH[:, d] = (lp - lm) / (2 * eps), (hp - hm) / (2 * eps)
    # d/dzeta of -[y log sigma(eta) + (1 - y) log(1 - sigma(eta))] = (sigma(eta) - y) d eta
    return (expit(eL0) - o["y"])[:, None] * dL, (expit(eH0) - h["y"])[:, None] * dH


def test_comparison_scores_sum_to_the_criterion_gradient(cell):
    c = cell
    N, K, r = c["N"], c["K"], c["r"]
    n_ijk, y_ijk, n_order, y_order = c["A"]
    from dial_judge.dial_model import pairs_to_arrays
    gL, gH = _comparison_grads(c)
    total = gH.sum(0) / c["n_H"] + c["lam"] * gL.sum(0) / c["n_L"]
    ref = q_lambda_grad(c["zeta"], N, K, r, make_centering_basis(K), make_centering_basis(N), True, n_ijk, y_ijk, n_order, y_order,
                        pairs_to_arrays(c["pairs"]), c["lam"], c["n_H"], c["n_L"])
    ref = ref[1] if isinstance(ref, tuple) else ref
    assert np.allclose(total, ref, atol=1e-6)


def test_record_meat_matches_brute_force(cell):
    c = cell
    gL, gH = _comparison_grads(c)
    psiL = c["lam"] * (gL - gL.mean(0)) / c["n_L"]
    psiH = (gH - gH.mean(0)) / c["n_H"]
    n_rec = int(c["llm_obs"]["rec"].max()) + 1
    U = np.zeros((n_rec, gL.shape[1]))
    np.add.at(U, c["llm_obs"]["rec"], psiL)
    np.add.at(U, c["human_obs"]["rec"], psiH)
    brute = c["n_H"] * U.T @ U
    J = _meat(c)
    assert np.allclose(J, brute, rtol=1e-5, atol=1e-8 * np.abs(brute).max())


def _reference_meat(c, cluster):
    N, K, r = c["N"], c["K"], c["r"]
    n_ijk, y_ijk, n_order, y_order = c["A"]
    sw = joint_sandwich(c["fit"], N, K, r, c["pairs"], n_ijk, y_ijk, n_order, y_order, cluster=cluster)
    return sw["score_cov"]


def test_singleton_records_give_the_independent_meat(cell):
    c = cell
    n_L = c["llm_obs"]["k"].size
    llm_obs = dict(c["llm_obs"], rec=np.array([f"L{u}" for u in range(n_L)]))
    human_obs = dict(c["human_obs"], rec=np.array([f"H{t}" for t in range(c["human_obs"]["i"].size)]))
    assert np.allclose(_meat(c, llm_obs, human_obs), _reference_meat(c, None), rtol=1e-8, atol=1e-12)


def test_cell_records_give_the_cell_clustered_meat(cell):
    c = cell
    o = c["llm_obs"]
    llm_obs = dict(o, rec=np.array([f"L{k}_{i}_{j}" for k, i, j in zip(o["k"], o["i"], o["j"])]))
    human_obs = dict(c["human_obs"], rec=np.array([f"H{t}" for t in range(c["human_obs"]["i"].size)]))
    assert np.allclose(_meat(c, llm_obs, human_obs), _reference_meat(c, "pair"), rtol=1e-8, atol=1e-12)


def test_record_sandwich_runs_and_is_psd(cell):
    c = cell
    N, K, r = c["N"], c["K"], c["r"]
    n_ijk, y_ijk, n_order, y_order = c["A"]
    sw = joint_sandwich(c["fit"], N, K, r, c["pairs"], n_ijk, y_ijk, n_order, y_order, cluster="record",
                        llm_obs=c["llm_obs"], human_obs=c["human_obs"])
    assert np.linalg.eigvalsh(sw["score_cov"]).min() > -1e-10
    assert np.all(np.isfinite(sw["contrasts"]["se"])) and np.all(sw["contrasts"]["se"] > 0)


def test_mismatched_observations_are_rejected(cell):
    c = cell
    bad = {k: v[:-1] for k, v in c["llm_obs"].items()}
    with pytest.raises(ValueError):
        _meat(c, llm_obs=bad)


def _llm_only(c, cluster, llm_obs=None):
    from dial_judge.inference import llm_only_sandwich
    n_ijk, y_ijk, n_order, y_order = c["A"]
    return llm_only_sandwich(c["fit"], c["N"], c["K"], c["r"], n_ijk=n_ijk, y_ijk=y_ijk, n_order=n_order, y_order=y_order,
                             cluster=cluster, llm_obs=llm_obs)


def test_llm_only_record_meat_reduces_to_existing_kernels(cell):
    c = cell
    o = c["llm_obs"]
    single = dict(o, rec=np.arange(o["k"].size))
    cells = dict(o, rec=np.array([f"{k}_{i}_{j}" for k, i, j in zip(o["k"], o["i"], o["j"])]))
    assert np.allclose(_llm_only(c, "record", single)["score_cov"], _llm_only(c, None)["score_cov"], rtol=1e-8, atol=1e-12)
    assert np.allclose(_llm_only(c, "record", cells)["score_cov"], _llm_only(c, "pair")["score_cov"], rtol=1e-8, atol=1e-12)
    rec = _llm_only(c, "record", o)
    assert np.all(np.isfinite(rec["b"]["se"])) and np.all(rec["b"]["se"] > 0)


def test_twoway_meat_reduces_to_one_way_kernels(cell):
    """Singleton records give the cell meat; records that coincide with cells give it too."""
    c = cell
    N, K, r = c["N"], c["K"], c["r"]
    n_ijk, y_ijk, n_order, y_order = c["A"]
    o, h = c["llm_obs"], c["human_obs"]

    def sw(cluster, llm_obs=None, human_obs=None):
        return joint_sandwich(c["fit"], N, K, r, c["pairs"], n_ijk, y_ijk, n_order, y_order, cluster=cluster,
                              llm_obs=llm_obs, human_obs=human_obs)["score_cov"]

    ref = sw("pair")
    single = dict(o, rec=np.array([f"L{u}" for u in range(o["k"].size)]))
    hs = dict(h, rec=np.array([f"H{t}" for t in range(h["i"].size)]))
    assert np.allclose(sw("twoway", single, hs), ref, rtol=1e-7, atol=1e-11)
    assert np.allclose(sw("twoway"), ref)
    assert np.allclose(_llm_only(c, "twoway", single)["score_cov"], _llm_only(c, "pair")["score_cov"], rtol=1e-7, atol=1e-11)
    assert np.all(_llm_only(c, "twoway", o)["b"]["se"] > 0)
