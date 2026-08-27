import numpy as np

from dial_judge.simulate import (
    compute_score_matrix,
    generate_balanced_comparisons,
    generate_true_parameters,
)


def test_generated_parameters_satisfy_identification_constraints():
    N, K, r = 6, 4, 2
    mu, gamma, U, V = generate_true_parameters(N, K, r, random_seed=17)
    np.testing.assert_allclose(mu.sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(gamma.sum(), K, atol=1e-12)
    np.testing.assert_allclose(U.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(V.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(mu @ V, np.zeros(r), atol=1e-12)


def test_balanced_generator_is_reproducible_and_exact_size():
    params = generate_true_parameters(5, 3, 1, random_seed=2)
    score = compute_score_matrix(*params)
    first = generate_balanced_comparisons(score, 137, random_seed=9)
    second = generate_balanced_comparisons(score, 137, random_seed=9)
    assert first == second
    assert len(first) == 137


def test_order_generator_uses_both_display_orders(small_study):
    n_order = small_study["n_order"]
    assert np.all(n_order.sum(axis=(0, 1, 2)) > 0)
    assert n_order.sum() == 1200
