"""Evaluation metrics: rank agreement with a target score, and held-out pairwise accuracy and
log loss on human comparison records."""
import numpy as np
from scipy.stats import spearmanr

from .data import records_to_array



def spearman(a, b):
    return float(spearmanr(a, b).correlation)


def pairwise_accuracy(mu_true, mu_est):
    """Share of item pairs whose estimated ordering matches the true one.

    A pair counts as correct when the two sign differences agree exactly, so a tie in
    `mu_true` is scored correct only against an exact tie in `mu_est` (the simulation DGPs
    draw continuous scores, so exact ties do not arise there).
    """
    mu_true = np.asarray(mu_true, dtype=float)
    mu_est = np.asarray(mu_est, dtype=float)
    iu = np.triu_indices(mu_true.size, k=1)
    if iu[0].size == 0:
        return 1.0
    d_true = np.sign(mu_true[iu[0]] - mu_true[iu[1]])
    d_est = np.sign(mu_est[iu[0]] - mu_est[iu[1]])
    return float(np.mean(d_true == d_est))


def sign_accuracy(true_scores, est_scores):
    """Pairwise sign accuracy of an estimated ranking vs a true (human) ranking."""
    return pairwise_accuracy(true_scores, est_scores)


def score_accuracy(score, records, skip_ties=True):
    # Fraction of (k, i, j, y) comparisons whose observed winner matches the
    # sign of score[i] - score[j]. Used for held-out pairwise-prediction
    # accuracy in the real-data studies, and as sign accuracy vs observed
    # human comparisons when no latent ground truth is available.
    if len(records) == 0:
        return None
    arr = records_to_array(records)[:, 1:4]
    if skip_ties:
        arr = arr[arr[:, 2] != 0.5]
    if arr.shape[0] == 0:
        return None
    score = np.asarray(score, dtype=float)
    i, j = arr[:, 0].astype(int), arr[:, 1].astype(int)
    pred = (score[i] > score[j]).astype(int)
    return float(np.mean(pred == arr[:, 2].astype(int)))


def heldout_log_loss(score, records, skip_ties=True):
    """Mean Bernoulli negative log-likelihood of sigma(score_i - score_j) over
    (k, i, j, y) comparisons. The real-data analogue of the excess human risk:
    it is minimized in population by the true human score, and differences
    between methods on the same held-out comparisons are directly comparable.
    Ties (y = 0.5) are skipped by default, matching `score_accuracy`."""
    if len(records) == 0:
        return None
    arr = records_to_array(records)[:, 1:4]
    if skip_ties:
        arr = arr[arr[:, 2] != 0.5]
    if arr.shape[0] == 0:
        return None
    i_idx = arr[:, 0].astype(int)
    j_idx = arr[:, 1].astype(int)
    y = arr[:, 2]
    score = np.asarray(score, dtype=float)
    diff = score[i_idx] - score[j_idx]
    loss = y * np.logaddexp(0.0, -diff) + (1.0 - y) * np.logaddexp(0.0, diff)
    return float(np.mean(loss))
