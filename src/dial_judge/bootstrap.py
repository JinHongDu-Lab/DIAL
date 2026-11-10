"""Record-level bootstrap summaries for the calibrated score.

The unit of resampling is the record: all judge verdicts on a record and its human label, if
any, move together, so dependence among the comparisons of one record is preserved. Records are
resampled within strata (human-labeled and proxy-only) so that the human budget is fixed and the
GACV grid of the original fit applies to every replicate. The functions here only resample
indices and summarize bootstrap draws; each study supplies its own refit.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .inference import pairwise_contrast_matrix


def resample_within_strata(strata, rng):
    """Bootstrap draw of record indices, one draw per stratum (a list of index arrays), concatenated."""
    out = []
    for idx in strata:
        idx = np.asarray(idx)
        if idx.size:
            out.append(rng.choice(idx, size=idx.size, replace=True))
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def subsample_within_strata(strata, rng, frac=0.5):
    """Half-sampling draw (without replacement) of record indices within each stratum.

    With frac = 1/2 the spread of a smooth estimator over half-samples around the full-sample
    estimate matches its sampling variance, var(theta_m - theta_n) ~ sigma^2 (1/m - 1/n) = sigma^2/n,
    so no rescaling is needed; unlike the bootstrap it creates no duplicate records, which would
    bias the leave-one-out-type GACV criterion toward human-heavy weights."""
    out = []
    for idx in strata:
        idx = np.asarray(idx)
        if idx.size:
            out.append(rng.choice(idx, size=int(round(frac * idx.size)), replace=False))
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def draw_within_strata(strata, rng, scheme="boot"):
    """`boot`: with-replacement record bootstrap; `half`: half-sampling without replacement."""
    return resample_within_strata(strata, rng) if scheme == "boot" else subsample_within_strata(strata, rng)


def summarize_contrasts(s_hat, s_boot, alpha=0.05):
    """Bootstrap standard errors and intervals for all pairwise contrasts of a score vector.

    `s_boot` is (B, N) with NaN rows for failed replicates. Returns the percentile interval and the
    bootstrap-s.e. Wald interval of every contrast, both centered as usual for the bootstrap, plus
    the number of replicates used.
    """
    s_hat = np.asarray(s_hat, dtype=float)
    S = np.asarray(s_boot, dtype=float)
    S = S[np.all(np.isfinite(S), axis=1)]
    D = pairwise_contrast_matrix(s_hat.size)
    est = D @ s_hat
    C = S @ D.T                                                   # (B, n_pairs)
    se = C.std(axis=0, ddof=1) if len(C) > 1 else np.full(est.size, np.nan)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    lo_p, hi_p = (np.percentile(C, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0) if len(C) > 1
                  else (np.full(est.size, np.nan), np.full(est.size, np.nan)))
    return dict(estimate=est, se=se, lower_wald=est - z * se, upper_wald=est + z * se, lower_pct=lo_p, upper_pct=hi_p, n_boot=int(len(C)))


def coverage(truth, lower, upper):
    truth = np.asarray(truth, dtype=float)
    ok = np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean((truth[ok] >= lower[ok]) & (truth[ok] <= upper[ok]))) if ok.any() else float("nan")


def adjacent_separation(s_hat, lower, upper):
    """Among the N - 1 pairs adjacent in the ranking of `s_hat`, the share whose contrast interval excludes zero."""
    s_hat = np.asarray(s_hat, dtype=float)
    N = s_hat.size
    order = np.argsort(-s_hat)
    tri_i, tri_j = np.triu_indices(N, k=1)
    index = {(int(i), int(j)): p for p, (i, j) in enumerate(zip(tri_i, tri_j))}
    sep = []
    for a, b in zip(order[:-1], order[1:]):
        i, j = (int(a), int(b)) if a < b else (int(b), int(a))
        p = index[(i, j)]
        sep.append(bool(np.isfinite(lower[p]) and np.isfinite(upper[p]) and (lower[p] > 0 or upper[p] < 0)))
    return float(np.mean(sep)), int(np.sum(sep))


def separated_sign_agreement(s_hat, lower, upper, reference):
    """Among contrasts whose interval excludes zero, the share whose sign agrees with `reference`."""
    D = pairwise_contrast_matrix(np.asarray(s_hat).size)
    est, ref = D @ np.asarray(s_hat, float), D @ np.asarray(reference, float)
    sep = np.isfinite(lower) & np.isfinite(upper) & ((lower > 0) | (upper < 0))
    if not sep.any():
        return float("nan"), 0
    return float(np.mean(np.sign(est[sep]) == np.sign(ref[sep]))), int(sep.sum())
