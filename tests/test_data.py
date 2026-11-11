import pytest

from dial_judge.data import (
    comparisons_to_aggregated,
    comparisons_to_order_aggregated,
    pool_pairs,
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


def test_pooling_contract():
    pooled = pool_pairs([(0, 0, 2, 1, 1), (2, 0, 2, 0, -1)])
    assert pooled == [(0, 2, 2.0, 1.0)]


def test_aggregation_rejects_noncanonical_pairs():
    with pytest.raises(ValueError, match="i < j"):
        comparisons_to_aggregated([(0, 2, 1, 1)], N=3, K=1)
