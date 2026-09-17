"""Baseline and endpoint estimators used by DIAL experiments.

This module contains the endpoints that can be expressed using the current
core implementation. More involved paper comparisons are registered below as
explicitly unavailable until their method-specific implementations are added.
"""

from __future__ import annotations

import numpy as np

from .benchmarks import fit_hja
from .dial_model import calibration_design, fit_human_calibration
from .hja import aggregate_judge_pairs, fit_centered_btl_from_pairs


METHOD_STATUS = {
    "human_only_btl": True,
    "llm_consensus": True,
    "consensus_only_calibrated": True,
    "staged_structured_calibration": True,
    "dial_gacv": True,
    "ordinary_structured_btl": True,
    "pooled_btl": True,
    "swap_probability_average": False,
    "paired_order_logit_average": False,
    "unstructured_btl_svd_calibration": False,
    "atc": True,
}


def fit_human_only_btl(N, human_pairs, maxiter=1000):
    """Fit an unrestricted centered BTL score using human comparisons only."""
    score = fit_centered_btl_from_pairs(N, human_pairs, maxiter=maxiter)
    return {
        "method": "human_only_btl",
        "s_H": score,
        "fit_info": {"converged": True},
    }


def _pava(values):
    """Pool-adjacent-violators: the nondecreasing least-squares fit of `values` (unit weights)."""
    level, weight = [], []
    for v in np.asarray(values, dtype=float):
        level.append(float(v))
        weight.append(1)
        while len(level) > 1 and level[-2] > level[-1]:
            w = weight[-2] + weight[-1]
            level[-2] = (weight[-2] * level[-2] + weight[-1] * level[-1]) / w
            weight[-2] = w
            level.pop()
            weight.pop()
    return np.repeat(np.asarray(level, dtype=float), weight)


def fit_atc_btl(N, mu, human_pairs, maxiter=1000):
    """AtC: aggregate the human comparisons into a ranking, then isotonic-calibrate `mu` to it.

    Stage 1 is the same unrestricted centered BTL fit as ``fit_human_only_btl``, whose induced
    ordering is the consensus ranking; stage 2 is the Euclidean projection of the predictive score
    `mu` onto the monotone cone of that ordering, computed by PAVA. The result is centered, so it
    lives on the same scale as every other estimator of the study. ``mle_exists`` is left to the
    caller: when the stage-1 MLE is not finite the reported fit is the bounded maximizer.
    """
    s_human = fit_centered_btl_from_pairs(N, human_pairs, maxiter=maxiter)
    order = np.argsort(s_human, kind="stable")          # ascending human rank
    mu = np.asarray(mu, dtype=float)
    fitted = np.empty(N, dtype=float)
    fitted[order] = _pava(mu[order])
    return {
        "method": "atc",
        "s_H": fitted - fitted.mean(),
        "s_human_stage1": s_human,
        "fit_info": {"converged": True},
    }


def fit_pooled_btl(N, n_ijk, y_ijk, maxiter=1000):
    """Fit ordinary BTL after pooling all LLM judges and ignoring order."""
    pairs = aggregate_judge_pairs(n_ijk, y_ijk)
    score = fit_centered_btl_from_pairs(N, pairs, maxiter=maxiter)
    return {
        "method": "pooled_btl",
        "score": score,
        "fit_info": {"converged": True},
    }


def _calibrated_result(method, hja_fit, human_pairs, consensus_only=False, maxiter=1000):
    mu = np.asarray(hja_fit["mu"], dtype=float)
    if consensus_only:
        V = np.zeros((mu.size, 0), dtype=float)
    else:
        V = np.asarray(hja_fit["V"], dtype=float)
    alpha, a, cal_info = fit_human_calibration(mu, V, human_pairs, maxiter=maxiter, return_info=True)
    design = calibration_design(mu, V)
    coefficients = np.concatenate([[alpha], a])
    return {
        **hja_fit,
        "method": method,
        "cal_info": cal_info,
        "alpha_H": alpha,
        "a": a,
        "W": design,
        "s_H": design @ coefficients,
    }


def fit_llm_consensus(N, K, r, n_ijk, y_ijk, **fit_kwargs):
    """Fit the LLM panel and expose its centered consensus as the target score."""
    fit = fit_hja(N, K, r, n_ijk=n_ijk, y_ijk=y_ijk, **fit_kwargs)
    return {**fit, "method": "llm_consensus", "s_H": np.asarray(fit["mu"], dtype=float)}


def fit_consensus_only_calibrated(
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    human_pairs,
    n_order=None,
    y_order=None,
    calibration_maxiter=1000,
    **fit_kwargs,
):
    """Fit the LLM panel, then calibrate only its consensus direction."""
    fit = fit_hja(
        N,
        K,
        r,
        n_ijk=n_ijk,
        y_ijk=y_ijk,
        n_order=n_order,
        y_order=y_order,
        **fit_kwargs,
    )
    return _calibrated_result(
        "consensus_only_calibrated",
        fit,
        human_pairs,
        consensus_only=True,
        maxiter=calibration_maxiter,
    )


def fit_staged_structured_calibration(
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    human_pairs,
    n_order=None,
    y_order=None,
    calibration_maxiter=1000,
    **fit_kwargs,
):
    """Fit the staged ``lambda = infinity`` DIAL endpoint."""
    fit = fit_hja(
        N,
        K,
        r,
        n_ijk=n_ijk,
        y_ijk=y_ijk,
        n_order=n_order,
        y_order=y_order,
        **fit_kwargs,
    )
    return _calibrated_result(
        "staged_structured_calibration",
        fit,
        human_pairs,
        consensus_only=False,
        maxiter=calibration_maxiter,
    )


def unavailable_methods():
    """Return paper comparison methods that still need implementations."""
    return tuple(name for name, implemented in METHOD_STATUS.items() if not implemented)
