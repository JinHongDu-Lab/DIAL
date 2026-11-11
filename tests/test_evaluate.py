import numpy as np

from dial_judge.evaluate import heldout_log_loss, score_accuracy, sign_accuracy, spearman


def test_exact_ranking_metrics():
    truth = np.array([2.0, 1.0, -1.0])
    assert spearman(truth, truth) == 1.0
    assert sign_accuracy(truth, truth) == 1.0
    assert sign_accuracy(truth, -truth) == 0.0


def test_heldout_metrics_on_records():
    score = np.array([1.0, 0.0, -1.0])
    records = [(0, 0, 1, 1.0), (0, 1, 2, 0.0), (0, 0, 2, 0.5)]      # the tie is skipped
    assert score_accuracy(score, records) == 0.5
    np.testing.assert_allclose(heldout_log_loss(score, records), 0.5 * (np.log1p(np.exp(-1.0)) + np.log1p(np.exp(1.0))))
