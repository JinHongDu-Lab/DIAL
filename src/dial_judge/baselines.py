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
    "atc": False,
}


def fit_human_only_btl(N, human_pairs, maxiter=1000):
    """Fit an unrestricted centered BTL score using human comparisons only."""
    score = fit_centered_btl_from_pairs(N, human_pairs, maxiter=maxiter)
    return {
        "method": "human_only_btl",
        "s_H": score,
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
    alpha, a = fit_human_calibration(mu, V, human_pairs, maxiter=maxiter)
    design = calibration_design(mu, V)
    coefficients = np.concatenate([[alpha], a])
    return {
        **hja_fit,
        "method": method,
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
