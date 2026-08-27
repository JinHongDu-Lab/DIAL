import numpy as np

from dial_judge.evaluate import (
    interval_coverage,
    ndcg,
    score_recovery_mse,
    sign_accuracy,
    spearman,
)


def test_exact_ranking_metrics():
    truth = np.array([2.0, 1.0, -1.0])
    assert spearman(truth, truth) == 1.0
    assert ndcg(truth, truth) == 1.0
    assert sign_accuracy(truth, truth) == 1.0
    assert sign_accuracy(truth, -truth) == 0.0


def test_score_mse_alignment_handles_shift_and_scale():
    truth = np.array([-1.0, 0.0, 1.0])
    estimate = 3.0 * truth + 7.0
    np.testing.assert_allclose(score_recovery_mse(truth, estimate, align=True), 0.0)


def test_interval_coverage():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    lower = np.array([-1.0, 0.5, 2.1, 4.0])
    upper = np.array([0.5, 1.5, 2.5, 5.0])
    assert interval_coverage(truth, lower, upper) == 0.5
