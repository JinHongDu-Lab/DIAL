"""
Unified fit_* interface.

Methods:
  - fit_hja: LLM-panel HJA, optionally with position-effect BTL
  - fit_dial / fit_dial_joint: DIAL weighted joint estimator (paper Eq. 14)
  - fit_dial_gacv: same estimator with GACV-selected lambda (paper Eq. 18)
"""
import numpy as np

from .dial_model import joint as _joint
from .gacv import select_lambda as _select_lambda
from .hja import collapse_order_counts, estimate_parameters, uncertainty_quantification


def compute_score_matrix(mu, gamma, U, V):
    mu = np.asarray(mu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    return np.outer(gamma, mu) + U @ V.T


def fit_hja(
    N,
    K,
    r,
    n_ijk=None,
    y_ijk=None,
    n_order=None,
    y_order=None,
    max_steps=120,
    tol=1e-5,
    tau=10.0,
    inner_maxiter=500,
    with_uq=False,
    uq_alpha=0.05,
):
    """Fit HJA. Pass n_order/y_order to estimate judge-specific position
    effects b_k in the order-effect BTL; otherwise the original no-order HJA."""
    if n_order is not None and n_ijk is None:
        n_ijk, y_ijk = collapse_order_counts(n_order, y_order)
    if n_ijk is None or y_ijk is None:
        raise ValueError("fit_hja requires n_ijk/y_ijk or n_order/y_order")
    mu, gamma, U, V, fit_info = estimate_parameters(
        N, K, r, n_ijk, y_ijk,
        max_steps=max_steps, tol=tol, tau=tau, inner_maxiter=inner_maxiter,
        n_order=n_order, y_order=y_order,
    )
    b = np.asarray(fit_info.get("b", np.zeros(K, dtype=float)), dtype=float)
    result = {
        "mu": mu,
        "gamma": gamma,
        "U": U,
        "V": V,
        "b": b,
        "S": compute_score_matrix(mu, gamma, U, V),
        "fit_info": fit_info,
    }
    if with_uq:
        targets = [{"type": "consensus_score", "i": i, "label": f"mu[{i}]"} for i in range(mu.size)]
        result["uq"] = uncertainty_quantification(
            gamma, mu, U, V, n_ijk, targets, alpha=uq_alpha, b=b if n_order is not None else None, n_order=n_order
        )
    return result


def fit_dial(
    N,
    K,
    r,
    n_ijk_llm,
    y_ijk_llm,
    human_pairs,
    n_order=None,
    y_order=None,
    lam=None,
    max_steps=150,
    tol=1e-6,
    tau=10.0,
    inner_maxiter=500,
    with_uq=False,
    uq_alpha=0.05,
):
    """DIAL joint estimator: ell_H + lambda ell_L, with optional position effects.

    `lam=None` uses n_L / n_0 (ordinary joint likelihood). For GACV selection
    of lambda use fit_dial_gacv.
    """
    return _joint(
        N, K, r, human_pairs,
        n_ijk_llm=n_ijk_llm, y_ijk_llm=y_ijk_llm,
        n_order=n_order, y_order=y_order,
        lam=lam, max_steps=max_steps, tol=tol, tau=tau, inner_maxiter=inner_maxiter,
        with_uq=with_uq, uq_alpha=uq_alpha,
    )


def fit_dial_gacv(
    N,
    K,
    r,
    n_ijk_llm,
    y_ijk_llm,
    human_pairs,
    n_order=None,
    y_order=None,
    human_records=None,
    lambda_grid=None,
    max_steps=150,
    tol=1e-6,
    tau=10.0,
    inner_maxiter=500,
    with_uq=False,
    uq_alpha=0.05,
):
    """DIAL joint estimator with GACV-selected lambda."""
    return _select_lambda(
        N, K, r, human_pairs,
        n_ijk_llm=n_ijk_llm, y_ijk_llm=y_ijk_llm,
        n_order=n_order, y_order=y_order,
        human_records=human_records, lambda_grid=lambda_grid,
        max_steps=max_steps, tol=tol, tau=tau, inner_maxiter=inner_maxiter,
        with_uq=with_uq, uq_alpha=uq_alpha,
    )


fit_dial_joint = fit_dial
fit_ht_hja_joint = fit_dial
