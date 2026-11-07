"""Penalized human-only Bradley--Terry fits used as rebuttal baselines.

* ``fit_ridge_btl``: centered BTL plus ``kappa * ||P s||^2``; with a consensus direction ``mu`` the
  projector ``P`` removes ``span(mu)``, so the penalty shrinks the human score toward the line
  ``alpha mu`` with ``alpha`` free (kappa -> 0: human-only BTL; kappa -> inf: the staged
  consensus calibration). With ``mu=None`` it is plain ridge toward zero. ``kappa`` is chosen on a
  grid by the approximate leave-one-comparison-out log loss (ALO), the same leave-one-out
  expansion that underlies GACV.
* ``fit_boxed_btl``: the unpenalized centered BTL fit on the same coordinate box that DIAL uses
  (|z_j| <= bound in the centering-reduced chart), reported with its box activity.

These exist to answer whether DIAL's small-budget gains come from bounding or shrinkage rather
than from the LLM preference representation; they are not part of DIAL.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .gacv import observations_from_pairs
from .hja import centered_btl_loss_and_grad_from_pairs, make_centering_basis, project_zero_sum

KAPPA_MULTIPLIERS = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
CHART_BOUND = 10.0


def _projector(N, mu):
    if mu is None:
        return np.eye(N)
    mu = project_zero_sum(np.asarray(mu, dtype=float))
    nrm = float(mu @ mu)
    if nrm <= 0:
        return np.eye(N)
    return np.eye(N) - np.outer(mu, mu) / nrm


def ridge_objective(N, pairs, s, kappa, P, arrays=None):
    loss, grad = centered_btl_loss_and_grad_from_pairs(N, pairs, s, arrays=arrays)
    Ps = P @ s
    return loss + kappa * float(Ps @ Ps), grad + 2.0 * kappa * Ps


def fit_ridge_btl_fixed(N, pairs, kappa, mu=None, initial=None, maxiter=1000):
    """Penalized fit at one ``kappa`` (zero-sum ``s``)."""
    P = _projector(N, mu)
    arrays = _pairs(pairs)
    x0 = np.zeros(N) if initial is None else project_zero_sum(np.asarray(initial, dtype=float))

    def objective(s_free):
        s = project_zero_sum(s_free)
        loss, grad = ridge_objective(N, pairs, s, kappa, P, arrays)
        return loss, project_zero_sum(grad)

    res = minimize(objective, x0=x0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "gtol": 1e-8})
    return project_zero_sum(res.x), bool(res.success)


def _pairs(pairs):
    arr = np.asarray(pairs, dtype=float).reshape(-1, 4)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3]


def alo_criterion(N, pairs, s, kappa, P, rcond=1e-10):
    """Approximate leave-one-comparison-out log loss of the penalized fit at ``s``:
    sum_t l_t(s) + tr(H^-1 sum_t g_t g_t^T) on the centered chart, with H the penalized Hessian."""
    i_obs, j_obs, z = observations_from_pairs(pairs)
    B = make_centering_basis(N)
    X = np.zeros((i_obs.size, N)); X[np.arange(i_obs.size), i_obs] = 1.0; X[np.arange(i_obs.size), j_obs] = -1.0
    d = s[i_obs] - s[j_obs]
    p = expit(d)
    ell = float(np.sum(np.logaddexp(0.0, d) - z * d))
    Xb = X @ B
    H = (Xb * (p * (1 - p))[:, None]).T @ Xb + 2.0 * kappa * (B.T @ P @ B)
    G = Xb * (p - z)[:, None]
    eig = np.linalg.eigvalsh(H)
    regular = bool(eig.min() > rcond * max(eig.max(), 1e-300)) and float(np.max(np.abs(s))) < 25.0
    tr = float(np.trace(np.linalg.pinv(H, rcond=rcond) @ (G.T @ G)))
    return {"alo": ell + tr, "ell_H": ell, "trace": tr, "regular": regular}


def fit_ridge_btl(N, pairs, mu=None, kappa_grid=None, multipliers=KAPPA_MULTIPLIERS, maxiter=1000):
    """Ridge-penalized centered BTL with ``kappa`` chosen by ALO over ``kappa_grid`` (default:
    ``n_H`` times ``multipliers``). Returns ``s_H``, ``kappa``, and the selection path."""
    n_H = float(sum(p[2] for p in pairs))
    grid = list(kappa_grid) if kappa_grid is not None else [n_H * m for m in multipliers]
    P = _projector(N, mu)
    path, best, s_prev = [], None, None
    for kappa in sorted(grid, reverse=True):          # warm-start from the most regularized fit
        s, ok = fit_ridge_btl_fixed(N, pairs, kappa, mu=mu, initial=s_prev, maxiter=maxiter)
        crit = alo_criterion(N, pairs, s, kappa, P)
        entry = dict(kappa=float(kappa), converged=ok, **crit)
        path.append(entry)
        if ok and crit["regular"] and (best is None or crit["alo"] < best[0]):
            best = (crit["alo"], kappa, s)
        s_prev = s
    if best is None:                                  # every candidate irregular: keep the most regularized one
        kappa, s = path[0]["kappa"], fit_ridge_btl_fixed(N, pairs, path[0]["kappa"], mu=mu, maxiter=maxiter)[0]
        selected_regular = False
    else:
        _, kappa, s = best
        selected_regular = True
    return {"method": "ridge_btl", "s_H": s, "kappa": float(kappa), "kappa_rel": float(kappa / n_H) if n_H else np.nan,
            "path": path, "fit_info": {"converged": True, "selected_regular": selected_regular, "max_abs": float(np.max(np.abs(s)))}}


def fit_boxed_btl(N, pairs, bound=CHART_BOUND, initial=None, maxiter=1000):
    """Unpenalized centered BTL on the coordinate box |z_j| <= bound of the centering-reduced
    chart ``s = B z``; ``fit_info['box_active']`` marks a solution on the boundary, which is the
    separated case where the unconstrained MLE does not exist."""
    B = make_centering_basis(N)
    arrays = _pairs(pairs)
    z0 = np.zeros(N - 1) if initial is None else B.T @ np.asarray(initial, dtype=float)

    def objective(z):
        s = B @ z
        loss, grad = centered_btl_loss_and_grad_from_pairs(N, pairs, s, arrays=arrays)
        return loss, B.T @ grad

    res = minimize(objective, x0=np.clip(z0, -bound, bound), method="L-BFGS-B", jac=True, bounds=[(-bound, bound)] * (N - 1), options={"maxiter": maxiter, "gtol": 1e-8})
    z = res.x
    active = bool(np.any(np.abs(np.abs(z) - bound) < 1e-6))
    s = project_zero_sum(B @ z)
    return {"method": "boxed_btl", "s_H": s, "fit_info": {"converged": bool(res.success), "box_active": active, "bound": float(bound), "max_abs": float(np.max(np.abs(s)))}}
