"""
Evaluation metrics for HJA and DIAL, matching the DIAL paper's synthetic
experiment list:

  - Score recovery (MSE)
  - Human preference ranking (Spearman)
  - Human preference ranking (NDCG)
  - Uncertainty calibration (Coverage)
  - Sign accuracy (human)

Also keeps the held-out pairwise prediction helper used by the real-data
studies (`score_accuracy`).
"""
import numpy as np
from scipy.stats import spearmanr

from .data import records_to_array
from .hja import uncertainty_quantification


EPS = 1e-12


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


def human_sign_accuracy(score, true_scores=None, records=None, skip_ties=True):
    """Sign accuracy against a human target.

    Pass `true_scores` in simulation (latent human ranking) or `records` of
    (k, i, j, y) human comparisons in real-data evaluation.
    """
    if true_scores is not None:
        return sign_accuracy(true_scores, score)
    if records is not None:
        return score_accuracy(score, records, skip_ties=skip_ties)
    raise ValueError("human_sign_accuracy requires true_scores or records")


def align_mu(mu_true, mu_est):
    # Least-squares scale alignment of a centered estimate onto the centered
    # truth, for reporting an error that is invariant to the model's overall
    # score scale.
    mu_true = np.asarray(mu_true, dtype=float)
    mu_est = np.asarray(mu_est, dtype=float)
    mu_true_c = mu_true - np.mean(mu_true)
    mu_est_c = mu_est - np.mean(mu_est)
    denom = float(mu_est_c @ mu_est_c)
    if denom <= EPS:
        raise ValueError("estimated mu has near-zero norm and cannot be aligned")
    scale = float((mu_true_c @ mu_est_c) / denom)
    return mu_true_c, scale * mu_est_c, scale


def score_recovery_mse(true_scores, est_scores, align=True):
    """Score-recovery MSE. BTL scores are shift-invariant; with `align=True`
    they are also scale-aligned before the mean squared error is taken."""
    true_scores = np.asarray(true_scores, dtype=float)
    est_scores = np.asarray(est_scores, dtype=float)
    if align:
        true_c, est_c, _ = align_mu(true_scores, est_scores)
        return float(np.mean((true_c - est_c) ** 2))
    true_c = true_scores - np.mean(true_scores)
    est_c = est_scores - np.mean(est_scores)
    return float(np.mean((true_c - est_c) ** 2))


def ndcg(true_scores, est_scores, k=None):
    """NDCG@k of the estimated ranking, with graded relevance from the true
    ranking (item at true rank t, 0-best, gets relevance N-1-t). Scale-invariant
    and the ranking metric named in the DIAL paper."""
    true_scores = np.asarray(true_scores, dtype=float)
    est_scores = np.asarray(est_scores, dtype=float)
    n = true_scores.size
    if n == 0:
        return 1.0
    if k is None:
        k = n
    k = int(min(max(k, 1), n))

    true_rank = np.argsort(np.argsort(-true_scores))
    relevance = (n - 1 - true_rank).astype(float)

    pred_order = np.argsort(-est_scores)[:k]
    gains = (np.power(2.0, relevance[pred_order]) - 1.0) / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains))

    ideal_order = np.argsort(-relevance)[:k]
    ideal_gains = (np.power(2.0, relevance[ideal_order]) - 1.0) / np.log2(np.arange(2, k + 2))
    idcg = float(np.sum(ideal_gains))
    return dcg / idcg if idcg > 0 else 1.0


def interval_coverage(true_values, lower, upper):
    """Uncertainty-calibration coverage: fraction of true values that fall
    inside the reported [lower, upper] interval."""
    true_values = np.asarray(true_values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if true_values.shape != lower.shape or true_values.shape != upper.shape:
        raise ValueError("true_values, lower, and upper must have the same shape")
    return float(np.mean((true_values >= lower) & (true_values <= upper)))


def hja_consensus_intervals(gamma, mu, U, V, n_ijk, alpha=0.05, b=None, n_order=None):
    """Wald intervals for HJA consensus scores mu[i] (delta-method UQ)."""
    N = np.asarray(mu).size
    targets = [{"type": "consensus_score", "i": i, "label": f"mu[{i}]"} for i in range(N)]
    uq = uncertainty_quantification(gamma, mu, U, V, n_ijk, targets, alpha=alpha, b=b, n_order=n_order)
    lower = np.array([iv["lower"] for iv in uq["intervals"]], dtype=float)
    upper = np.array([iv["upper"] for iv in uq["intervals"]], dtype=float)
    se = np.array([iv["se"] for iv in uq["intervals"]], dtype=float)
    return {"lower": lower, "upper": upper, "se": se, "uq": uq}


def evaluate_scores(true_scores, est_scores, ci_lower=None, ci_upper=None, ndcg_k=None, align_mse=True):
    """All DIAL-paper metrics for one estimated score vector vs a human target."""
    metrics = {
        "mse": score_recovery_mse(true_scores, est_scores, align=align_mse),
        "spearman": spearman(true_scores, est_scores),
        "ndcg": ndcg(true_scores, est_scores, k=ndcg_k),
        "sign_accuracy": sign_accuracy(true_scores, est_scores),
    }
    if ci_lower is not None and ci_upper is not None:
        metrics["coverage"] = interval_coverage(true_scores, ci_lower, ci_upper)
    else:
        metrics["coverage"] = None
    return metrics


def mu_recovery_metrics(mu_true, mu_est, ci_lower=None, ci_upper=None, ndcg_k=None):
    # Backward-compatible name used by the simulation runner; now returns the
    # full DIAL-paper metric set plus the older aligned-L2 / scale fields.
    mu_true_aligned, mu_est_aligned, scale = align_mu(mu_true, mu_est)
    metrics = evaluate_scores(mu_true, mu_est, ci_lower=ci_lower, ci_upper=ci_upper, ndcg_k=ndcg_k)
    metrics["pairwise_accuracy"] = metrics["sign_accuracy"]
    metrics["aligned_l2_error"] = float(np.linalg.norm(mu_true_aligned - mu_est_aligned))
    metrics["scale"] = scale
    return metrics
