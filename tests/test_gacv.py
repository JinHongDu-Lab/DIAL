import numpy as np

from dial_judge.dial_model import pairs_to_arrays, total_human_n, total_llm_n
from dial_judge.gacv import (
    default_lambda_grid,
    pack_reduced,
    q_lambda_grad,
    unpack_reduced,
)
from dial_judge.hja import make_centering_basis


def test_default_lambda_grid_contains_joint_mle_weight():
    grid = default_lambda_grid(n_L=600, n_H=120, n_points=6)
    assert 5.0 in grid
    assert grid == sorted(grid)
    assert all(value > 0 for value in grid)


def test_reduced_coordinate_round_trip(small_study):
    zeta = pack_reduced(
        small_study["gamma"],
        small_study["mu"],
        small_study["U"],
        small_study["V"],
        small_study["b"],
        1.1,
        np.array([0.3]),
        use_order=True,
    )
    unpacked = unpack_reduced(
        zeta,
        small_study["N"],
        small_study["K"],
        small_study["r"],
        make_centering_basis(small_study["K"]),
        make_centering_basis(small_study["N"]),
        use_order=True,
    )
    gamma, mu, U, V, b, alpha, a = unpacked
    np.testing.assert_allclose(gamma, small_study["gamma"], atol=1e-12)
    np.testing.assert_allclose(mu, small_study["mu"], atol=1e-12)
    np.testing.assert_allclose(U, small_study["U"], atol=1e-12)
    np.testing.assert_allclose(V, small_study["V"], atol=1e-12)
    np.testing.assert_allclose(b, small_study["b"], atol=1e-12)
    np.testing.assert_allclose(alpha, 1.1)
    np.testing.assert_allclose(a, [0.3])


def test_gacv_objective_gradient_matches_finite_difference(small_study):
    N, K, r = small_study["N"], small_study["K"], small_study["r"]
    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    zeta = pack_reduced(
        small_study["gamma"],
        small_study["mu"],
        small_study["U"],
        small_study["V"],
        small_study["b"],
        1.1,
        np.array([0.3]),
        use_order=True,
    )
    pair_arrays = pairs_to_arrays(small_study["human_pairs"])
    n_H = total_human_n(small_study["human_pairs"])
    n_L = total_llm_n(small_study["n_ijk"], n_order=small_study["n_order"])

    def objective(value):
        return q_lambda_grad(
            value,
            N,
            K,
            r,
            judge_basis,
            item_basis,
            True,
            small_study["n_ijk"],
            small_study["y_ijk"],
            small_study["n_order"],
            small_study["y_order"],
            pair_arrays,
            1.2,
            n_H,
            n_L,
        )

    _, analytic = objective(zeta)
    eps = 1e-6
    numeric = np.empty_like(zeta)
    for idx in range(zeta.size):
        plus, minus = zeta.copy(), zeta.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric[idx] = (objective(plus)[0] - objective(minus)[0]) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


def test_analytic_hessian_matches_finite_difference(small_study):
    from dial_judge.gacv import hessian_from_grad, q_lambda_hessian

    N, K = small_study["N"], small_study["K"]
    pair_arrays = pairs_to_arrays(small_study["human_pairs"])
    rng = np.random.default_rng(0)
    for r, use_order, m in [(1, True, 1), (1, True, 0), (1, False, 1), (0, True, 0)]:
        jb, ib = make_centering_basis(K), make_centering_basis(N)
        zeta = pack_reduced(1 + 0.3 * rng.normal(size=K), rng.normal(size=N), rng.normal(size=(K, r)), rng.normal(size=(N, r)),
                            0.4 * rng.normal(size=K), 1.1, rng.normal(size=m), use_order)
        args = (N, K, r, jb, ib, use_order, small_study["n_ijk"], small_study["y_ijk"],
                small_study["n_order"] if use_order else None, small_study["y_order"] if use_order else None,
                pair_arrays, 3.0, total_human_n(small_study["human_pairs"]), total_llm_n(small_study["n_ijk"]))
        exact = q_lambda_hessian(zeta, *args)
        numeric = hessian_from_grad(lambda z: q_lambda_grad(z, *args), zeta)
        np.testing.assert_allclose(exact, numeric, rtol=1e-6, atol=1e-7 * np.abs(numeric).max())
