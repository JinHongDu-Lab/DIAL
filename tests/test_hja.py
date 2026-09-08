import numpy as np

from dial_judge.benchmarks import fit_hja
from dial_judge.hja import (
    negative_log_likelihood,
    negative_log_likelihood_and_grad,
    reanchor,
)


def _central_difference(objective, value, index, eps=1e-6):
    plus = value.copy()
    minus = value.copy()
    plus.flat[index] += eps
    minus.flat[index] -= eps
    return (objective(plus) - objective(minus)) / (2 * eps)


def test_order_likelihood_gradients_match_finite_differences():
    N, K, r = 3, 2, 1
    mu = np.array([-0.5, 0.1, 0.4])
    gamma = np.array([0.8, 1.2])
    U = np.array([[-0.2], [0.2]])
    V = np.array([[-0.3], [0.5], [-0.2]])
    b = np.array([0.15, -0.1])
    n_order = np.zeros((K, N, N, 2))
    y_order = np.zeros_like(n_order)
    n_order[:, 0, 1, :] = [[5, 6], [4, 7]]
    n_order[:, 0, 2, :] = [[6, 5], [7, 4]]
    n_order[:, 1, 2, :] = [[4, 6], [5, 5]]
    y_order[:, 0, 1, :] = [[2, 4], [1, 5]]
    y_order[:, 0, 2, :] = [[4, 2], [5, 1]]
    y_order[:, 1, 2, :] = [[1, 4], [2, 3]]
    n_ijk = n_order.sum(axis=-1)
    y_ijk = y_order.sum(axis=-1)

    _, g_mu, g_gamma, g_U, g_V, g_b = negative_log_likelihood_and_grad(
        mu, gamma, U, V, n_ijk, y_ijk, b=b, n_order=n_order, y_order=y_order
    )

    def loss(mu_arg=mu, gamma_arg=gamma, U_arg=U, V_arg=V, b_arg=b):
        return negative_log_likelihood(
            mu_arg,
            gamma_arg,
            U_arg,
            V_arg,
            n_ijk,
            y_ijk,
            b=b_arg,
            n_order=n_order,
            y_order=y_order,
        )

    blocks = [
        (mu, g_mu, lambda x: loss(mu_arg=x)),
        (gamma, g_gamma, lambda x: loss(gamma_arg=x)),
        (U, g_U, lambda x: loss(U_arg=x)),
        (V, g_V, lambda x: loss(V_arg=x)),
        (b, g_b, lambda x: loss(b_arg=x)),
    ]
    for value, analytic, objective in blocks:
        numeric = np.array(
            [_central_difference(objective, value, idx) for idx in range(value.size)]
        ).reshape(value.shape)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


def test_reanchor_preserves_scores_and_constraints():
    mu = np.array([-1.0, -0.2, 0.4, 0.8])
    gamma = np.array([0.7, 1.0, 1.3])
    U = np.array([[-0.4], [0.1], [0.3]])
    V = np.array([[-0.5], [0.3], [0.4], [-0.2]]) + np.outer(mu, [0.35])
    score_before = np.outer(gamma, mu) + U @ V.T
    gamma_new, mu_new, U_new, V_new = reanchor(gamma, mu, U, V)
    score_after = np.outer(gamma_new, mu_new) + U_new @ V_new.T
    np.testing.assert_allclose(score_after, score_before, atol=1e-10)
    np.testing.assert_allclose(gamma_new.sum(), 3.0, atol=1e-10)
    np.testing.assert_allclose(U_new.sum(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(V_new.sum(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(mu_new @ V_new, 0.0, atol=1e-10)


def test_hja_order_fit_smoke(small_study):
    fit = fit_hja(
        small_study["N"],
        small_study["K"],
        small_study["r"],
        small_study["n_ijk"],
        small_study["y_ijk"],
        n_order=small_study["n_order"],
        y_order=small_study["y_order"],
        max_steps=30,
        tol=1e-4,
        inner_maxiter=200,
    )
    assert fit["fit_info"]["converged"]
    assert fit["S"].shape == (small_study["K"], small_study["N"])
    assert np.all(np.isfinite(fit["b"]))


def test_separated_position_effect_hits_bound_and_converges():
    """A judge that always picks the first-displayed response has no finite MLE for b_k; the
    bounded fit converges with that judge at the bound and finite scores for everything else."""
    from dial_judge.hja import POSITION_EFFECT_BOUND
    rng = np.random.default_rng(0)
    N, K, r = 6, 4, 1
    s = np.linspace(-1, 1, N)
    n_order = np.zeros((K, N, N, 2)); y_order = np.zeros((K, N, N, 2))
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                for a_idx, a in enumerate((-1.0, 1.0)):
                    n = 6
                    if k == K - 1:
                        y = n if a > 0 else 0          # position-only judge: first-displayed always wins
                    else:
                        y = rng.binomial(n, 1 / (1 + np.exp(-(s[i] - s[j] + 0.3 * a))))
                    n_order[k, i, j, a_idx] = n; y_order[k, i, j, a_idx] = y
    fit = fit_hja(N, K, r, n_order=n_order, y_order=y_order, max_steps=60, tol=1e-5, inner_maxiter=200)
    assert fit["fit_info"]["converged"]
    assert fit["fit_info"]["b_at_bound"] == 1
    assert abs(fit["b"][K - 1]) >= POSITION_EFFECT_BOUND - 1e-6 and np.all(np.abs(fit["b"][:-1]) < 2.0)
    assert np.all(np.isfinite(fit["mu"])) and np.max(np.abs(fit["mu"])) < 5.0
