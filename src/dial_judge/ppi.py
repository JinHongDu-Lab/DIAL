"""Bradley--Terry PPI++ (Boyeau et al., 2025; Angelopoulos et al., 2023), run as defined.

Prediction-powered inference for the centered Bradley--Terry coefficient: a small labeled sample
of records carrying the human outcome ``y`` and a proxy outcome ``f`` (the verdict of one LLM
judge, or a panel vote, on the same record), and a large proxy-only sample with ``f`` alone.
With ``l(s; i, j, z)`` the Bernoulli log loss of outcome ``z`` under ``s_i - s_j``,

    L_lambda(s) = (1/n) sum_lab l(s; y_t) + lambda [ (1/m) sum_unl l(s; f_u) - (1/n) sum_lab l(s; f_t) ],

minimized over the zero-sum vector ``s``; lambda = 0 is the human-only fit, lambda = 1 classical
PPI, and PPI++ chooses lambda in [0, 1] by the plug-in variance formula. Nothing here departs from
the published estimator: no penalty, no box, no imputation of missing proxies. A record without a
proxy is not a labeled record for this estimator, and the caller reports that gap.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .hja import centered_btl_loss_and_grad_from_pairs, make_centering_basis, project_zero_sum

FINITE_BOUND = 25.0


def _aggregate(N, i, j, z):
    """(i, j, n, sum z) pairs from per-record arrays (fractional z allowed)."""
    if i.size == 0:
        return []
    key = i.astype(np.int64) * N + j.astype(np.int64)
    keys, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, minlength=keys.size).astype(float)
    s = np.bincount(inv, weights=z, minlength=keys.size)
    return [(int(k // N), int(k % N), float(nn), float(ss)) for k, nn, ss in zip(keys, n, s)]


def _unpack(labeled, unlabeled):
    lab = np.asarray(labeled, dtype=float).reshape(-1, 4)
    unl = np.asarray(unlabeled, dtype=float).reshape(-1, 3)
    return (lab[:, 0].astype(int), lab[:, 1].astype(int), lab[:, 2], lab[:, 3],
            unl[:, 0].astype(int), unl[:, 1].astype(int), unl[:, 2])


def ppi_objective(N, s, lab_y, lab_f, unl_f, lam, n, m):
    """L_lambda(s) and its gradient from the three aggregated pair lists."""
    loss_y, g_y = centered_btl_loss_and_grad_from_pairs(N, lab_y, s)
    if lam == 0.0 or not unl_f:
        return loss_y / n, g_y / n
    loss_f, g_f = centered_btl_loss_and_grad_from_pairs(N, lab_f, s)
    loss_u, g_u = centered_btl_loss_and_grad_from_pairs(N, unl_f, s)
    loss = loss_y / n + lam * (loss_u / m - loss_f / n)
    grad = g_y / n + lam * (g_u / m - g_f / n)
    return loss, grad


def _minimize(N, lab_y, lab_f, unl_f, lam, n, m, x0, maxiter=1000):
    def objective(s_free):
        s = project_zero_sum(s_free)
        loss, grad = ppi_objective(N, s, lab_y, lab_f, unl_f, lam, n, m)
        return loss, project_zero_sum(grad)

    res = minimize(objective, x0=project_zero_sum(x0), method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "gtol": 1e-8})
    s = project_zero_sum(res.x)
    finite = bool(np.all(np.isfinite(s))) and float(np.max(np.abs(s))) < FINITE_BOUND
    return s, bool(res.success), finite


def ppi_lambda(N, s, li, lj, ly, lf, ui, uj):
    """PPI++ plug-in weight at ``s``: the trace-optimal lambda in [0, 1].

    With per-record score gradients g_y = (p - y) x and g_f = (p - f) x, x = e_i - e_j, and the
    pooled Hessian H of the log loss, lambda* = tr(H^-1 Cov(g_y, g_f) H^-1) /
    ((1 + n/m) tr(H^-1 Var(g_f) H^-1)) (Angelopoulos et al., 2023, Section 6), clipped to [0, 1].
    """
    n, m = li.size, ui.size
    if n == 0 or m == 0:
        return 0.0, {"lam_raw": 0.0, "reason": "no_labeled_or_unlabeled"}
    B = make_centering_basis(N)
    X_l = np.zeros((n, N)); X_l[np.arange(n), li] = 1.0; X_l[np.arange(n), lj] = -1.0
    X_u = np.zeros((m, N)); X_u[np.arange(m), ui] = 1.0; X_u[np.arange(m), uj] = -1.0
    p_l = expit(s[li] - s[lj]); p_u = expit(s[ui] - s[uj])
    w = np.concatenate([p_l * (1 - p_l), p_u * (1 - p_u)])
    X = np.vstack([X_l, X_u]) @ B
    H = (X * w[:, None]).T @ X / (n + m)
    H_inv = np.linalg.pinv(H)
    G_y = (X_l @ B) * (p_l - ly)[:, None]
    G_f = (X_l @ B) * (p_l - lf)[:, None]
    G_y = G_y - G_y.mean(0); G_f = G_f - G_f.mean(0)
    cov_yf = G_y.T @ G_f / n
    var_f = G_f.T @ G_f / n
    num = float(np.trace(H_inv @ cov_yf @ H_inv))
    den = float((1.0 + n / m) * np.trace(H_inv @ var_f @ H_inv))
    lam_raw = num / den if den > 0 else 0.0
    return float(np.clip(lam_raw, 0.0, 1.0)), {"lam_raw": float(lam_raw)}


def fit_bt_ppi(N, labeled, unlabeled, lam=None, initial=None, maxiter=1000):
    """Bradley--Terry PPI++ point estimate.

    ``labeled``: rows (i, j, y, f) with the human outcome ``y`` and the proxy ``f`` on the same
    record; ``unlabeled``: rows (i, j, f). ``lam=None`` selects the PPI++ weight at the lambda = 1
    fit (falling back to the human-only fit when that one is not finite); a float fixes it.

    Returns ``s_H`` (zero-sum), ``lam``, and ``fit_info`` with ``converged`` and ``finite``; a
    non-finite result is the estimator's failure on that sample, reported, not repaired.
    """
    li, lj, ly, lf, ui, uj, uf = _unpack(labeled, unlabeled)
    n, m = int(li.size), int(ui.size)
    if n == 0:
        raise ValueError("PPI++ needs at least one labeled record with a proxy")
    lab_y = _aggregate(N, li, lj, ly)
    lab_f = _aggregate(N, li, lj, lf)
    unl_f = _aggregate(N, ui, uj, uf)
    x0 = np.zeros(N) if initial is None else np.asarray(initial, dtype=float)
    info = {"n_labeled": n, "n_unlabeled": m}
    if lam is None:
        if m == 0:
            lam, info["lam_reason"] = 0.0, "no_unlabeled"
        else:
            s1, ok1, fin1 = _minimize(N, lab_y, lab_f, unl_f, 1.0, n, m, x0, maxiter)
            if ok1 and fin1:
                s_pre, info["lam_at"] = s1, "ppi"
            else:
                s0, ok0, fin0 = _minimize(N, lab_y, lab_f, unl_f, 0.0, n, m, x0, maxiter)
                s_pre, info["lam_at"] = (s0, "human_only") if (ok0 and fin0) else (np.zeros(N), "zero")
            lam, lam_info = ppi_lambda(N, s_pre, li, lj, ly, lf, ui, uj)
            info.update(lam_info)
            x0 = s_pre
    lam = float(lam)
    s, ok, finite = _minimize(N, lab_y, lab_f, unl_f, lam, n, m, x0, maxiter)
    info.update(converged=ok, finite=finite, max_abs=float(np.max(np.abs(s))))
    return {"method": "bt_ppi", "s_H": s, "lam": lam, "fit_info": info}
