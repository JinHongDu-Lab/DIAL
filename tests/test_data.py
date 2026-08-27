import numpy as np
import pytest

from dial_judge.data import (
    comparisons_to_aggregated,
    comparisons_to_order_aggregated,
    display_order_a,
    pool_pairs,
    split_records,
)


def test_aggregation_preserves_counts_and_outcomes():
    records = [
        (0, 0, 1, 1, 1),
        (0, 0, 1, 0, -1),
        (1, 1, 2, 0.5, 1),
    ]
    n_ijk, y_ijk = comparisons_to_aggregated(records, N=3, K=2)
    n_order, y_order = comparisons_to_order_aggregated(records, N=3, K=2)

    assert n_ijk.sum() == 3
    assert y_ijk.sum() == 1.5
    assert n_order[0, 0, 1, 0] == 1
    assert n_order[0, 0, 1, 1] == 1
    assert y_order[0, 0, 1, 0] == 0
    assert y_order[0, 0, 1, 1] == 1


def test_order_sign_and_pooling_contract():
    assert display_order_a(0, 2) == (0, 2, 1)
    assert display_order_a(2, 0) == (0, 2, -1)
    pooled = pool_pairs([(0, 0, 2, 1, 1), (2, 0, 2, 0, -1)])
    assert pooled == [(0, 2, 2.0, 1.0)]


def test_split_is_deterministic_and_disjoint():
    records = [(0, 0, 1, value) for value in range(10)]
    train_a, test_a = split_records(records, test_ratio=0.3, random_seed=4)
    train_b, test_b = split_records(records, test_ratio=0.3, random_seed=4)
    assert train_a == train_b
    assert test_a == test_b
    assert len(train_a) == 7
    assert len(test_a) == 3
    assert set(map(tuple, train_a)).isdisjoint(set(map(tuple, test_a)))


def test_aggregation_rejects_noncanonical_pairs():
    with pytest.raises(ValueError, match="i < j"):
        comparisons_to_aggregated([(0, 2, 1, 1)], N=3, K=1)
