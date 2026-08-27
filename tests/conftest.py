import numpy as np
import pytest

from dial_judge.data import (
    comparisons_to_aggregated,
    comparisons_to_order_aggregated,
    pool_pairs,
)
from dial_judge.simulate import (
    compute_human_score,
    compute_score_matrix,
    generate_balanced_comparisons_with_order,
    generate_human_calibration_params,
    generate_human_comparisons,
    generate_position_effects,
    generate_true_parameters,
)


@pytest.fixture(scope="session")
def small_study():
    N, K, r = 4, 3, 1
    mu, gamma, U, V = generate_true_parameters(N, K, r, random_seed=7)
    score = compute_score_matrix(mu, gamma, U, V)
    b = generate_position_effects(K, scale=0.2, random_seed=8)
    llm_records = generate_balanced_comparisons_with_order(
        score, b, total_comparisons=1200, random_seed=9
    )
    n_ijk, y_ijk = comparisons_to_aggregated(llm_records, N, K)
    n_order, y_order = comparisons_to_order_aggregated(llm_records, N, K)

    alpha, a = generate_human_calibration_params(V, random_seed=10)
    human_score = compute_human_score(mu, V, alpha, a)
    human_records = generate_human_comparisons(
        human_score, total_comparisons=600, random_seed=11
    )
    human_pairs = pool_pairs(human_records)
    return {
        "N": N,
        "K": K,
        "r": r,
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "S": score,
        "b": b,
        "n_ijk": n_ijk,
        "y_ijk": y_ijk,
        "n_order": n_order,
        "y_order": y_order,
        "human_score": human_score,
        "human_pairs": human_pairs,
    }
