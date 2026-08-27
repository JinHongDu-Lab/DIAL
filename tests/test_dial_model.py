import numpy as np
import pytest

from dial_judge.baselines import (
    fit_consensus_only_calibrated,
    fit_human_only_btl,
    fit_staged_structured_calibration,
    unavailable_methods,
)
from dial_judge.benchmarks import fit_dial
from dial_judge.dial_model import human_nll_and_grad, pairs_to_arrays


def test_human_calibration_gradient_matches_finite_difference():
    mu = np.array([-0.7, -0.1, 0.3, 0.5])
    V = np.array([[-0.3], [0.4], [0.2], [-0.3]])
    pairs = [(0, 1, 8, 6), (0, 2, 7, 5), (1, 3, 9, 2), (2, 3, 6, 3)]
    arrays = pairs_to_arrays(pairs)
    x = np.array([1.2, -0.4])

    def objective(value):
        return human_nll_and_grad(value[0], value[1:], mu, V, arrays)[0]

    _, grad_alpha, grad_a, _, _ = human_nll_and_grad(x[0], x[1:], mu, V, arrays)
    analytic = np.concatenate([[grad_alpha], grad_a])
    eps = 1e-6
    numeric = np.empty_like(x)
    for idx in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric[idx] = (objective(plus) - objective(minus)) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_available_endpoints_return_human_scores(small_study):
    human = fit_human_only_btl(small_study["N"], small_study["human_pairs"])
    assert human["s_H"].shape == (small_study["N"],)

    kwargs = dict(max_steps=30, tol=1e-4, inner_maxiter=200)
    consensus = fit_consensus_only_calibrated(
        small_study["N"],
        small_study["K"],
        small_study["r"],
        small_study["n_ijk"],
        small_study["y_ijk"],
        small_study["human_pairs"],
        n_order=small_study["n_order"],
        y_order=small_study["y_order"],
        **kwargs,
    )
    staged = fit_staged_structured_calibration(
        small_study["N"],
        small_study["K"],
        small_study["r"],
        small_study["n_ijk"],
        small_study["y_ijk"],
        small_study["human_pairs"],
        n_order=small_study["n_order"],
        y_order=small_study["y_order"],
        **kwargs,
    )
    assert consensus["a"].size == 0
    assert staged["a"].size == small_study["r"]
    assert np.all(np.isfinite(consensus["s_H"]))
    assert np.all(np.isfinite(staged["s_H"]))


def test_unavailable_method_registry_is_explicit():
    assert "atc" in unavailable_methods()
    assert "paired_order_logit_average" in unavailable_methods()


@pytest.mark.slow
def test_joint_dial_smoke(small_study):
    fit = fit_dial(
        small_study["N"],
        small_study["K"],
        small_study["r"],
        small_study["n_ijk"],
        small_study["y_ijk"],
        small_study["human_pairs"],
        n_order=small_study["n_order"],
        y_order=small_study["y_order"],
        lam=1.0,
        max_steps=120,
        tol=1e-4,
        inner_maxiter=200,
    )
    assert fit["fit_info"]["converged"]
    assert fit["s_H"].shape == (small_study["N"],)
    assert np.all(np.isfinite(fit["s_H"]))
