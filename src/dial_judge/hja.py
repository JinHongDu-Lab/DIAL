"""
Core HJA model: S = gamma * mu^T + U V^T for a panel of judges.

When order-specific counts n_order/y_order are supplied, the likelihood is the
position-effect BTL (DIAL paper Eq. 1):

    logit(p_kij^(a)) = S_ki - S_kj + a * b_k

Self-contained port of src/models.py so the DIAL package does not depend on
the original src/ package at import time. The algorithm (anchored alternating
MLE, ReAnchor, delta-method UQ, BIC rank selection) is unchanged from
src/models.py, with position effects added as judge-specific parameters b.
"""
import time
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


EPS = 1e-10


def validate_rank(N, K, r):
    r_max = min(K - 1, N - 2)
    if r < 0:
        raise ValueError(f"rank r must be nonnegative, got {r}")
    if r > r_max:
        raise ValueError(f"rank r={r} exceeds identifiable maximum {r_max} for N={N}, K={K}")
    return r_max


def sigmoid(x):
    return expit(x)


def project_zero_sum(vec):
    arr = np.asarray(vec, dtype=float)
    return arr - np.mean(arr)


def canonicalize_columns(U, V):
    U_out = np.array(U, dtype=float, copy=True)
    V_out = np.array(V, dtype=float, copy=True)
    for col in range(U_out.shape[1]):
        nz = np.flatnonzero(np.abs(U_out[:, col]) > EPS)
        if nz.size == 0:
            raise ValueError(f"column {col} is numerically zero and cannot be canonicalized")
        if U_out[nz[0], col] < 0:
            U_out[:, col] *= -1.0
            V_out[:, col] *= -1.0
    return U_out, V_out


ORDER_A = np.array([-1.0, 1.0])  # a_idx 0 -> A=-1 (i second), 1 -> A=+1 (i first)


def collapse_order_counts(n_order, y_order):
    n_order = np.asarray(n_order, dtype=float)
    y_order = np.asarray(y_order, dtype=float)
    return n_order.sum(axis=-1), y_order.sum(axis=-1)


def negative_log_likelihood(mu, gamma, U, V, n_ijk, y_ijk, b=None, n_order=None, y_order=None):
    loss, *_ = negative_log_likelihood_and_grad(
        mu, gamma, U, V, n_ijk, y_ijk, b=b, n_order=n_order, y_order=y_order
    )
    return loss


def negative_log_likelihood_and_grad(mu, gamma, U, V, n_ijk, y_ijk, b=None, n_order=None, y_order=None):
    # logit(p_kij^(a)) = S_ki - S_kj + a * b_k  (paper Eq. 1). When n_order is
    # None this reduces to the original HJA likelihood (b = 0).
    score = np.outer(gamma, mu) + U @ V.T
    K, N = score.shape
    tri_i, tri_j = np.triu_indices(N, k=1)
    delta = score[:, tri_i] - score[:, tri_j]
    zero_b = np.zeros(K, dtype=float)

    if n_order is None:
        n_obs = n_ijk[:, tri_i, tri_j]
        mask = n_obs > 0
        if not np.any(mask):
            return 0.0, np.zeros_like(mu), np.zeros_like(gamma), np.zeros_like(U), np.zeros_like(V), zero_b
        y_obs = y_ijk[:, tri_i, tri_j]
        eta = delta
        residual = (n_obs * expit(eta) - y_obs) * mask
        loss = y_obs * np.logaddexp(0.0, -eta) + (n_obs - y_obs) * np.logaddexp(0.0, eta)
        loss_value = float(np.sum(loss[mask]))
        grad_score = np.zeros_like(score)
        for judge_index in range(K):
            np.add.at(grad_score[judge_index], tri_i, residual[judge_index])
            np.add.at(grad_score[judge_index], tri_j, -residual[judge_index])
        grad_gamma = grad_score @ mu
        grad_mu = grad_score.T @ gamma
        grad_U = grad_score @ V if U.size else np.zeros_like(U)
        grad_V = grad_score.T @ U if V.size else np.zeros_like(V)
        return loss_value, grad_mu, grad_gamma, grad_U, grad_V, zero_b

    b = zero_b if b is None else np.asarray(b, dtype=float)
    eta = delta[:, :, None] + b[:, None, None] * ORDER_A[None, None, :]
    n_obs = np.asarray(n_order, dtype=float)[:, tri_i, tri_j, :]
    y_obs = np.asarray(y_order, dtype=float)[:, tri_i, tri_j, :]
    mask = n_obs > 0
    if not np.any(mask):
        return 0.0, np.zeros_like(mu), np.zeros_like(gamma), np.zeros_like(U), np.zeros_like(V), zero_b
    residual = (n_obs * expit(eta) - y_obs) * mask
    loss = y_obs * np.logaddexp(0.0, -eta) + (n_obs - y_obs) * np.logaddexp(0.0, eta)
    loss_value = float(np.sum(loss[mask]))
    residual_score = residual.sum(axis=2)
    grad_score = np.zeros_like(score)
    for judge_index in range(K):
        np.add.at(grad_score[judge_index], tri_i, residual_score[judge_index])
        np.add.at(grad_score[judge_index], tri_j, -residual_score[judge_index])
    grad_gamma = grad_score @ mu
    grad_mu = grad_score.T @ gamma
    grad_U = grad_score @ V if U.size else np.zeros_like(U)
    grad_V = grad_score.T @ U if V.size else np.zeros_like(V)
    grad_b = np.sum(residual * ORDER_A[None, None, :], axis=(1, 2))
    return loss_value, grad_mu, grad_gamma, grad_U, grad_V, grad_b


# Finite-fit bound on the judge position effects. A centered log-odds of 10 is a
# first-position probability of 0.99995, beyond what any judge's sample supports; a judge
# whose order term is separated (every retained verdict explained by display position) has
# no finite MLE for b_k, and the bound makes the maximizer exist so that the remaining
# parameters, in particular the consensus, are estimated at the limiting profile likelihood.
# Judges at the bound are reported as `b_at_bound` in fit_info.
POSITION_EFFECT_BOUND = 10.0
# The same box is applied to every coordinate of the centering-reduced chart (gamma, U, mu, V and,
# in the joint fit, alpha and a): when a judge's verdicts never contradict its fitted scores its
# loading (gamma_k, U_k) has no finite maximizer either, and without a box the divergence stalls
# the optimizer and, along a warm-started lambda path, disqualifies every candidate. The box is
# inert on regular fits (all coordinates of the anchored chart are O(1) to O(5) there) and makes
# the maximizer exist otherwise; coordinates on the boundary are reported as `n_at_bound`.
CHART_BOUND = 10.0
_BOUND_TOL = 1e-6


def chart_bounds(n, bound=CHART_BOUND):
    """L-BFGS-B box [-bound, bound] for all n chart coordinates."""
    return [(-bound, bound)] * n


def position_bounds(n_before, K, n_after, bound=POSITION_EFFECT_BOUND):
    """Kept for callers that bound only the K position effects."""
    return [(None, None)] * n_before + [(-bound, bound)] * K + [(None, None)] * n_after


def projected_gradient(x, grad, bounds):
    """Zero the gradient components that point outward at an active bound."""
    g = np.array(grad, dtype=float, copy=True)
    if bounds is None:
        return g
    for idx, (lo, hi) in enumerate(bounds):
        if lo is not None and x[idx] <= lo + _BOUND_TOL and g[idx] > 0:
            g[idx] = 0.0
        if hi is not None and x[idx] >= hi - _BOUND_TOL and g[idx] < 0:
            g[idx] = 0.0
    return g


def count_at_bound(b, bound=POSITION_EFFECT_BOUND):
    b = np.asarray(b, dtype=float)
    return int(np.sum(np.abs(b) >= bound - _BOUND_TOL)) if b.size else 0


def count_chart_at_bound(x, bounds):
    """Number of coordinates of x on the boundary of `bounds`."""
    if bounds is None:
        return 0
    x = np.asarray(x, dtype=float)
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds]); hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    return int(np.sum((x <= lo + _BOUND_TOL) | (x >= hi - _BOUND_TOL)))


def boundary_activity(x, bounds, blocks):
    """Box activity of the optimizer's final chart point, per parameter block.

    `blocks` maps a block name to the index array of its coordinates in `x`. Uses the same
    active-bound tolerance as `count_chart_at_bound` and `projected_gradient`. Returns
    `boundary_<name>` flags, `boundary_any`, and the largest absolute chart coordinate.
    """
    x = np.asarray(x, dtype=float)
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds]); hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    active = (x <= lo + _BOUND_TOL) | (x >= hi - _BOUND_TOL)
    out = {f"boundary_{name}": bool(active[np.asarray(idx, dtype=int)].any()) for name, idx in blocks.items()}
    out["boundary_any"] = bool(active.any())
    out["max_abs_coord"] = float(np.max(np.abs(x))) if x.size else 0.0
    return out


def accept_block_step(result, f0, label):
    """A proximal block update need not be solved exactly: accept L-BFGS-B's iterate whenever it
    did not increase the objective (its line search guarantees descent), and report whether the
    inner iteration limit was hit. Raise only when the solver returned a worse point."""
    if result.success:
        return False
    if float(result.fun) <= f0 + 1e-12 * (1.0 + abs(f0)):
        return True
    raise RuntimeError(f"{label} failed: {result.message} (nit={getattr(result, 'nit', 'NA')})")


def aggregate_judge_pairs(n_ijk, y_ijk, judge_index=None):
    """Observed (i, j, n, y) pairs, summed over judges or taken from `judge_index` alone."""
    K, N, _ = n_ijk.shape
    if judge_index is None:
        n_ij, y_ij = np.sum(n_ijk, axis=0), np.sum(y_ijk, axis=0)
    else:
        n_ij, y_ij = n_ijk[judge_index], y_ijk[judge_index]
    tri_i, tri_j = np.triu_indices(N, k=1)
    n, y = n_ij[tri_i, tri_j], y_ij[tri_i, tri_j]
    keep = n > 0
    if not np.any(keep):
        raise ValueError("no observed pairs available for BTL fit")
    return [(int(i), int(j), float(nn), float(yy))
            for i, j, nn, yy in zip(tri_i[keep], tri_j[keep], n[keep], y[keep])]


def _pairs_arrays(pairs):
    arr = np.asarray(pairs, dtype=float).reshape(-1, 4)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3]


def centered_btl_loss_and_grad_from_pairs(N, pairs, s, arrays=None):
    i_idx, j_idx, n_arr, y_arr = _pairs_arrays(pairs) if arrays is None else arrays
    s = np.asarray(s, dtype=float)
    diff = s[i_idx] - s[j_idx]
    loss = float(np.sum(y_arr * np.logaddexp(0.0, -diff) + (n_arr - y_arr) * np.logaddexp(0.0, diff)))
    residual = n_arr * expit(diff) - y_arr
    grad = np.zeros(N, dtype=float)
    np.add.at(grad, i_idx, residual)
    np.add.at(grad, j_idx, -residual)
    return loss, grad


def fit_centered_btl_from_pairs(N, pairs, initial=None, maxiter=500):
    if initial is None:
        x0 = np.zeros(N, dtype=float)
    else:
        x0 = project_zero_sum(initial)
    arrays = _pairs_arrays(pairs)

    def objective(s_free):
        s = project_zero_sum(s_free)
        loss, grad = centered_btl_loss_and_grad_from_pairs(N, pairs, s, arrays=arrays)
        return loss, project_zero_sum(grad)

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"centered BTL fit failed: {result.message}")
    return project_zero_sum(result.x)


def firth_penalized_loss_and_grad(N, pairs, s, arrays=None):
    """Firth-penalized centered BTL objective NLL(s) - (1/2) log det J(s) and its gradient, with
    J(s) the centered Fisher information (comparison Laplacian of app:clustered) restricted to
    the zero-sum subspace via `make_centering_basis`. See `fit_centered_btl_firth`.

    The log-determinant gradient uses the standard hat-matrix identity
    d/ds_k log det J(s) = sum_c (1 - 2 p_c) h_c x_{c,k}, h_c = w_c (X J(s)^+ X^T)_{cc}, the
    weighted-least-squares leverage of comparison c (sum_c h_c = N - 1 by construction).
    """
    i_idx, j_idx, n_arr, y_arr = _pairs_arrays(pairs) if arrays is None else arrays
    s = np.asarray(s, dtype=float)
    nll, grad_nll = centered_btl_loss_and_grad_from_pairs(N, pairs, s, arrays=(i_idx, j_idx, n_arr, y_arr))
    X = np.zeros((i_idx.size, N), dtype=float)
    X[np.arange(i_idx.size), i_idx] = 1.0
    X[np.arange(i_idx.size), j_idx] = -1.0
    diff = s[i_idx] - s[j_idx]
    p = expit(diff)
    w = n_arr * p * (1.0 - p)
    info = X.T @ (X * w[:, None])
    B = make_centering_basis(N)
    sign, logdet = np.linalg.slogdet(B.T @ info @ B)
    logdet = logdet if sign > 0 else -1e10          # degenerate point (all weights collapsing); steers the optimizer away
    info_pinv = np.linalg.pinv(info)
    h = w * np.einsum("cj,cj->c", X @ info_pinv, X)
    grad_logdet = X.T @ ((1.0 - 2.0 * p) * h)
    loss = nll - 0.5 * logdet
    grad = grad_nll - 0.5 * grad_logdet
    return loss, grad


def fit_centered_btl_firth(N, pairs, initial=None, maxiter=500):
    """Firth (1993) bias-reduced centered BTL fit: minimizes `firth_penalized_loss_and_grad`
    over the zero-sum subspace. Firth's penalty makes the objective finite everywhere and gives
    a unique interior minimizer even under Ford (1957) separation, where
    `fit_centered_btl_from_pairs` diverges; used as the existence fallback in
    `calibration_restriction_test`. This is a standard
    bias-reduction heuristic (bias O(1/n) versus the plain MLE's O(1/sqrt(n)), Firth 1993); using
    it specifically to replace a non-existent MLE is outside what that asymptotic argument covers
    and is validated empirically rather than proved (see the manuscript remark).
    """
    arrays = _pairs_arrays(pairs)

    def objective(s_free):
        s = project_zero_sum(s_free)
        loss, grad = firth_penalized_loss_and_grad(N, pairs, s, arrays=arrays)
        return loss, project_zero_sum(grad)

    x0 = np.zeros(N, dtype=float) if initial is None else project_zero_sum(initial)
    bounds = [(-30.0, 30.0)] * N          # keeps unbounded L-BFGS-B trial steps from overflowing the weight/log-det computation; never binds at a genuine optimum
    result = minimize(objective, x0=x0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": maxiter, "gtol": 1e-6})
    if not result.success:
        raise RuntimeError(f"Firth centered BTL fit failed: {result.message}")
    return project_zero_sum(result.x)


def reanchor(gamma, mu, U, V, delta_mu=1e-8, delta_sigma=1e-8, sum_tol=1e-8, check_sum=True):
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    mu_plus = np.array(mu, dtype=float, copy=True)
    mu_norm_sq = float(mu_plus @ mu_plus)
    if mu_norm_sq < delta_mu:
        raise ValueError("ReAnchor failed: mu norm is too small")

    if V.shape[1] == 0:
        if check_sum and abs(np.sum(gamma) - gamma.size) > sum_tol:
            raise ValueError("ReAnchor failed: gamma sum constraint violated")
        return gamma, mu_plus, np.zeros_like(U), np.zeros_like(V)

    a = (V.T @ mu_plus) / mu_norm_sq
    V_bar = V - np.outer(mu_plus, a)
    gamma_cand = gamma + U @ a

    if check_sum and abs(np.sum(gamma_cand) - gamma_cand.size) > sum_tol:
        raise ValueError("ReAnchor failed: gamma candidate sum constraint violated")

    H_bar = U @ V_bar.T
    P, singular_values, Qt = np.linalg.svd(H_bar, full_matrices=False)
    rank = V.shape[1]
    if singular_values.shape[0] < rank or singular_values[rank - 1] < delta_sigma:
        raise ValueError("ReAnchor failed: heterogeneity term is rank deficient")

    sigma = singular_values[:rank]
    U_plus = P[:, :rank] @ np.diag(sigma / np.sqrt(mu_plus.size))
    V_plus = Qt[:rank, :].T * np.sqrt(mu_plus.size)
    U_plus, V_plus = canonicalize_columns(U_plus, V_plus)
    return gamma_cand, mu_plus, U_plus, V_plus


def initialize_position_effects(mu, gamma, U, V, n_order, y_order):
    # Per-judge 1D logistic for b_k with S held at the pooled-order init.
    score = np.outer(gamma, mu) + U @ V.T
    K, N = score.shape
    tri_i, tri_j = np.triu_indices(N, k=1)
    delta = score[:, tri_i] - score[:, tri_j]
    b0 = np.zeros(K, dtype=float)
    n_order = np.asarray(n_order, dtype=float)
    y_order = np.asarray(y_order, dtype=float)
    for k in range(K):
        n_k = n_order[k, tri_i, tri_j, :]
        y_k = y_order[k, tri_i, tri_j, :]
        mask = n_k > 0
        if not np.any(mask):
            continue
        delta_rep = np.stack([delta[k], delta[k]], axis=1)
        a_rep = np.broadcast_to(ORDER_A, delta_rep.shape)

        def objective(bk):
            eta = delta_rep[mask] + a_rep[mask] * bk[0]
            n_obs = n_k[mask]
            y_obs = y_k[mask]
            loss = float(np.sum(y_obs * np.logaddexp(0.0, -eta) + (n_obs - y_obs) * np.logaddexp(0.0, eta)))
            grad = np.array([float(np.sum(a_rep[mask] * (n_obs * expit(eta) - y_obs)))])
            return loss, grad

        result = minimize(objective, x0=np.zeros(1), method="L-BFGS-B", jac=True, bounds=[(-POSITION_EFFECT_BOUND, POSITION_EFFECT_BOUND)], options={"maxiter": 200, "gtol": 1e-6})
        b0[k] = float(np.clip(result.x[0], -POSITION_EFFECT_BOUND, POSITION_EFFECT_BOUND))
    return b0


def initialize_parameters(N, K, r, n_ijk, y_ijk, n_order=None, y_order=None):
    validate_rank(N, K, r)
    if n_order is not None and n_ijk is None:
        n_ijk, y_ijk = collapse_order_counts(n_order, y_order)

    mu0 = fit_centered_btl_from_pairs(N, aggregate_judge_pairs(n_ijk, y_ijk))
    mu0 = project_zero_sum(mu0)
    gamma0 = np.ones(K, dtype=float)

    judge_scores = np.zeros((K, N), dtype=float)
    for k in range(K):
        judge_scores[k, :] = fit_centered_btl_from_pairs(N, aggregate_judge_pairs(n_ijk, y_ijk, judge_index=k), initial=mu0)

    if r == 0:
        U0 = np.zeros((K, 0), dtype=float)
        V0 = np.zeros((N, 0), dtype=float)
        gamma0, mu0, U0, V0 = reanchor(gamma0, mu0, U0, V0, check_sum=False)
        gamma0 = gamma0 / np.sum(gamma0) * K
    else:
        residual = judge_scores - np.outer(gamma0, mu0)
        residual_centered = residual - np.mean(residual, axis=0, keepdims=True)
        P, singular_values, Qt = np.linalg.svd(residual_centered, full_matrices=False)
        if singular_values.shape[0] < r or singular_values[r - 1] <= EPS:
            raise ValueError("initialization failed: residual SVD is rank deficient")
        sigma_root = np.sqrt(singular_values[:r])
        U0 = P[:, :r] @ np.diag(sigma_root)
        V0 = Qt[:r, :].T @ np.diag(sigma_root)
        gamma0, mu0, U0, V0 = reanchor(gamma0, mu0, U0, V0, check_sum=False)
        gamma0 = gamma0 / np.sum(gamma0) * K

    if n_order is None:
        b0 = np.zeros(K, dtype=float)
    else:
        b0 = initialize_position_effects(mu0, gamma0, U0, V0, n_order, y_order)
    return gamma0, mu0, U0, V0, b0


def _pack_params(gamma, mu, U, V, b=None):
    parts = [gamma, mu, U.ravel(), V.ravel()]
    if b is not None:
        parts.append(np.asarray(b, dtype=float))
    return np.concatenate(parts)


def _param_slices(K, N, r, with_b=False):
    gamma_end = K
    mu_end = gamma_end + N
    U_end = mu_end + K * r
    V_end = U_end + N * r
    slices = {
        "gamma": slice(0, gamma_end),
        "mu": slice(gamma_end, mu_end),
        "U": slice(mu_end, U_end),
        "V": slice(U_end, V_end),
        "size": V_end,
    }
    if with_b:
        slices["b"] = slice(V_end, V_end + K)
        slices["size"] = V_end + K
    return slices


@lru_cache(maxsize=None)
def make_centering_basis(dim):
    if dim <= 1:
        return np.zeros((dim, 0), dtype=float)
    raw = np.eye(dim, dim - 1, dtype=float)
    raw[-1, :] = -1.0
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return basis


def reduce_judge_block(gamma, U):
    basis = make_centering_basis(gamma.size)
    gamma_reduced = basis.T @ (np.asarray(gamma, dtype=float) - 1.0)
    U_reduced = basis.T @ np.asarray(U, dtype=float)
    return gamma_reduced, U_reduced


def reduce_item_block(mu, V):
    basis = make_centering_basis(mu.size)
    mu_reduced = basis.T @ np.asarray(mu, dtype=float)
    V_reduced = basis.T @ np.asarray(V, dtype=float)
    return mu_reduced, V_reduced


def polish_llm_fit(gamma, mu, U, V, b, n_ijk, y_ijk, n_order=None, y_order=None, maxiter=2000, gtol=1e-8):
    """
    Finish an LLM-only fit by L-BFGS over the centering-reduced factor chart
    (gamma_red, U_red, [b], mu_red, V_red), then ReAnchor. The chart has exact
    invariance directions (factor rotations), along which the objective is flat;
    L-BFGS handles these without difficulty. Returns the anchored parameters and
    a dict with the final gradient norm, iteration count, and convergence flag.
    """
    use_order = n_order is not None
    K, N, r = gamma.size, mu.size, U.shape[1]
    jb, ib = make_centering_basis(K), make_centering_basis(N)
    n_g, n_U = K - 1, (K - 1) * r

    def unpack(z):
        idx = 0
        g = np.ones(K) + jb @ z[idx: idx + n_g]
        idx += n_g
        Uu = jb @ z[idx: idx + n_U].reshape(K - 1, r)
        idx += n_U
        if use_order:
            bb = z[idx: idx + K]
            idx += K
        else:
            bb = None
        m = ib @ z[idx: idx + N - 1]
        idx += N - 1
        Vv = ib @ z[idx:].reshape(N - 1, r)
        return g, m, Uu, Vv, bb

    def fg(z):
        g, m, Uu, Vv, bb = unpack(z)
        loss, gmu, ggamma, gU, gV, gb = negative_log_likelihood_and_grad(m, g, Uu, Vv, n_ijk, y_ijk, b=bb, n_order=n_order, y_order=y_order)
        parts = [jb.T @ ggamma, (jb.T @ gU).ravel()]
        if use_order:
            parts.append(gb)
        parts.extend([ib.T @ gmu, (ib.T @ gV).ravel()])
        return loss, np.concatenate(parts)

    g_red, U_red = reduce_judge_block(gamma, U)
    m_red, V_red = reduce_item_block(mu, V)
    parts = [g_red, U_red.ravel()]
    if use_order:
        parts.append(np.asarray(b, dtype=float))
    parts.extend([m_red, V_red.ravel()])
    z0 = np.concatenate(parts)
    # scale-free stopping: gradient relative to total comparisons
    n_total = float(np.sum(n_order) if use_order else np.sum(n_ijk))
    bounds = chart_bounds(z0.size)
    res = minimize(fg, z0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": gtol * n_total, "ftol": 1e-15})
    g, m, Uu, Vv, bb = unpack(res.x)
    if r > 0:
        g, m, Uu, Vv = reanchor(g, m, Uu, Vv)
    grad_norm = float(np.linalg.norm(projected_gradient(res.x, res.jac, bounds))) / n_total
    # Divergence diagnostics: a judge whose verdicts never contradict its fitted scores has no finite
    # loading (gamma_k, U_k), and the polish then stops at the iteration cap with a small but
    # nonzero gradient; `max_abs_S` and `max_gamma` let the caller report such fits.
    S = np.outer(g, m) + Uu @ Vv.T
    info = {"polish_nit": int(res.nit), "polish_grad_norm": grad_norm, "polish_converged": bool(grad_norm <= 1e-5), "nll": float(res.fun),
            "b_at_bound": count_at_bound(bb) if use_order else 0, "n_at_bound": count_chart_at_bound(res.x, bounds),
            "max_abs_S": float(np.max(np.abs(S))), "max_gamma": float(np.max(np.abs(g)))}
    n_b = K if use_order else 0
    idx = np.arange(z0.size)
    info.update(boundary_activity(res.x, bounds, {"b": idx[n_g + n_U: n_g + n_U + n_b],
                                                   "llm_other": np.concatenate([idx[: n_g + n_U], idx[n_g + n_U + n_b:]])}))
    return g, m, Uu, Vv, (bb if use_order else np.zeros(K)), info


def alternating_mle(
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    max_steps=120,
    tol=1e-5,
    tau=10.0,
    inner_maxiter=100,
    reanchor_steps=True,
    n_order=None,
    y_order=None,
    polish=True,
    raise_on_nonconvergence=False,
):
    use_order = n_order is not None
    gamma, mu, U, V, b = initialize_parameters(N, K, r, n_ijk, y_ijk, n_order=n_order, y_order=y_order)
    history = []
    converged = False
    inner_limit_hits = 0

    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    n_gamma = K - 1
    n_U = (K - 1) * r

    for step in range(max_steps):
        step_index = step + 1
        step_start = time.perf_counter()
        current = _pack_params(gamma, mu, U, V, b if use_order else None)
        current_gamma_reduced, current_U_reduced = reduce_judge_block(gamma, U)
        judge_parts = [current_gamma_reduced, current_U_reduced.ravel()]
        if use_order:
            judge_parts.append(b)
        current_judge_block = np.concatenate(judge_parts)
        current_mu_reduced, current_V_reduced = reduce_item_block(mu, V)
        current_item_block = np.concatenate([current_mu_reduced, current_V_reduced.ravel()])

        def objective_judge(block):
            gamma_new = np.ones(K, dtype=float) + judge_basis @ block[:n_gamma]
            U_new = judge_basis @ block[n_gamma: n_gamma + n_U].reshape(K - 1, r)
            b_new = block[n_gamma + n_U:] if use_order else None
            loss, _, grad_gamma, grad_U, _, grad_b = negative_log_likelihood_and_grad(
                mu, gamma_new, U_new, V, n_ijk, y_ijk, b=b_new, n_order=n_order, y_order=y_order
            )
            objective_value = loss + 0.5 * tau * np.sum((block - current_judge_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[:n_gamma] = judge_basis.T @ grad_gamma + tau * (block[:n_gamma] - current_gamma_reduced)
            grad_reduced[n_gamma: n_gamma + n_U] = (
                (judge_basis.T @ grad_U).ravel() + tau * (block[n_gamma: n_gamma + n_U] - current_U_reduced.ravel())
            )
            if use_order:
                grad_reduced[n_gamma + n_U:] = grad_b + tau * (block[n_gamma + n_U:] - b)
            return objective_value, grad_reduced

        result_j = minimize(
            objective_judge,
            x0=current_judge_block,
            method="L-BFGS-B",
            jac=True,
            bounds=chart_bounds(current_judge_block.size),
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        inner_limit_hits += accept_block_step(result_j, objective_judge(current_judge_block)[0], "judge-side update")
        gamma_tilde = np.ones(K, dtype=float) + judge_basis @ result_j.x[:n_gamma]
        U_tilde = judge_basis @ result_j.x[n_gamma: n_gamma + n_U].reshape(K - 1, r)
        b_tilde = result_j.x[n_gamma + n_U:].copy() if use_order else np.zeros(K, dtype=float)

        def objective_item(block):
            mu_new = item_basis @ block[: N - 1]
            V_new = item_basis @ block[N - 1:].reshape(N - 1, r)
            loss, grad_mu, _, _, grad_V, _ = negative_log_likelihood_and_grad(
                mu_new, gamma_tilde, U_tilde, V_new, n_ijk, y_ijk, b=b_tilde, n_order=n_order, y_order=y_order
            )
            objective_value = loss + 0.5 * tau * np.sum((block - current_item_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[: N - 1] = item_basis.T @ grad_mu + tau * (block[: N - 1] - current_mu_reduced)
            grad_reduced[N - 1:] = (item_basis.T @ grad_V).ravel() + tau * (block[N - 1:] - current_V_reduced.ravel())
            return objective_value, grad_reduced

        result_i = minimize(
            objective_item,
            x0=current_item_block,
            method="L-BFGS-B",
            jac=True,
            bounds=chart_bounds(current_item_block.size),
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        inner_limit_hits += accept_block_step(result_i, objective_item(current_item_block)[0], "item-side update")
        mu_tilde = item_basis @ result_i.x[: N - 1]
        V_tilde = item_basis @ result_i.x[N - 1:].reshape(N - 1, r)

        if reanchor_steps:
            gamma_new, mu_new, U_new, V_new = reanchor(gamma_tilde, mu_tilde, U_tilde, V_tilde)
        else:
            gamma_new, mu_new, U_new, V_new = gamma_tilde, mu_tilde, U_tilde, V_tilde
        b_new = b_tilde

        nll = negative_log_likelihood(mu_new, gamma_new, U_new, V_new, n_ijk, y_ijk, b=b_new, n_order=n_order, y_order=y_order)
        prev_nll = negative_log_likelihood(mu, gamma, U, V, n_ijk, y_ijk, b=b, n_order=n_order, y_order=y_order)
        diff = np.linalg.norm(_pack_params(gamma_new, mu_new, U_new, V_new, b_new if use_order else None) - current)
        rel_nll = abs(nll - prev_nll) / (1.0 + abs(prev_nll))
        step_elapsed = time.perf_counter() - step_start
        history.append({"iteration": step_index, "nll": float(nll), "diff": float(diff), "rel_nll": float(rel_nll), "tau": float(tau), "seconds": float(step_elapsed)})

        gamma, mu, U, V, b = gamma_new, mu_new, U_new, V_new, b_new
        if rel_nll < tol:
            converged = True
            break

    fit_info = {
        "n_iter": len(history),
        "alternating_converged": converged,
        "history": history,
        "nll": float(history[-1]["nll"]) if history else None,
        "reanchor_steps": bool(reanchor_steps),
        "inner_limit_hits": int(inner_limit_hits),
        "b_at_bound": count_at_bound(b) if use_order else 0,
    }
    if polish:
        gamma, mu, U, V, b, pinfo = polish_llm_fit(gamma, mu, U, V, b, n_ijk, y_ijk, n_order=n_order, y_order=y_order)
        fit_info.update(pinfo)
        converged = pinfo["polish_converged"]
    fit_info["converged"] = bool(converged)
    fit_info["b"] = b
    if not converged and raise_on_nonconvergence:
        raise RuntimeError("alternating MLE failed to converge within max_steps")
    return mu, gamma, U, V, b, fit_info


def fit_rank0_model(N, K, n_ijk, y_ijk, maxiter=2000, n_order=None, y_order=None):
    item_basis = make_centering_basis(N)
    judge_basis = make_centering_basis(K)
    gamma0, mu0, U0, V0, b0 = initialize_parameters(N, K, 0, n_ijk, y_ijk, n_order=n_order, y_order=y_order)
    use_order = n_order is not None
    x0_parts = [item_basis.T @ mu0, np.zeros(K - 1, dtype=float)]
    if use_order:
        x0_parts.append(b0)
    x0 = np.concatenate(x0_parts)
    n_mu = N - 1
    n_gamma = K - 1

    def objective(params):
        mu = item_basis @ params[:n_mu]
        gamma = np.ones(K, dtype=float) + judge_basis @ params[n_mu: n_mu + n_gamma]
        b = params[n_mu + n_gamma:] if use_order else None
        loss, grad_mu, grad_gamma, _, _, grad_b = negative_log_likelihood_and_grad(
            mu,
            gamma,
            np.zeros((K, 0), dtype=float),
            np.zeros((N, 0), dtype=float),
            n_ijk,
            y_ijk,
            b=b,
            n_order=n_order,
            y_order=y_order,
        )
        grad_parts = [item_basis.T @ grad_mu, judge_basis.T @ grad_gamma]
        if use_order:
            grad_parts.append(grad_b)
        return loss, np.concatenate(grad_parts)

    bounds = chart_bounds(x0.size)
    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "gtol": 1e-5, "maxls": 50},
    )
    n_total = float(np.sum(n_order) if use_order else np.sum(n_ijk))
    grad_norm = float(np.linalg.norm(projected_gradient(result.x, result.jac, bounds))) / max(n_total, 1.0)
    converged = bool(result.success) or grad_norm <= 1e-5
    if not converged and float(result.fun) > objective(x0)[0]:
        raise RuntimeError(f"rank-0 fit failed: {result.message}")

    mu = item_basis @ result.x[:n_mu]
    gamma = np.ones(K, dtype=float) + judge_basis @ result.x[n_mu: n_mu + n_gamma]
    b = result.x[n_mu + n_gamma:].copy() if use_order else np.zeros(K, dtype=float)
    U = np.zeros((K, 0), dtype=float)
    V = np.zeros((N, 0), dtype=float)
    gamma, mu, U, V = reanchor(gamma, mu, U, V)
    fit_info = {"n_iter": int(result.nit), "converged": converged, "polish_grad_norm": grad_norm, "nll": float(result.fun), "b": b,
                "b_at_bound": count_at_bound(b) if use_order else 0, "n_at_bound": count_chart_at_bound(result.x, bounds), "inner_limit_hits": int(not result.success)}
    idx = np.arange(x0.size)
    fit_info.update(boundary_activity(result.x, bounds, {"b": idx[n_mu + n_gamma:], "llm_other": idx[: n_mu + n_gamma]}))
    return mu, gamma, U, V, b, fit_info


def estimate_parameters(
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    max_steps=120,
    tol=1e-5,
    tau=10.0,
    inner_maxiter=500,
    reanchor_steps=True,
    n_order=None,
    y_order=None,
    polish=True,
):
    validate_rank(N, K, r)
    if n_order is not None and n_ijk is None:
        n_ijk, y_ijk = collapse_order_counts(n_order, y_order)
    if r == 0:
        mu, gamma, U, V, b, fit_info = fit_rank0_model(N, K, n_ijk, y_ijk, n_order=n_order, y_order=y_order)
    else:
        mu, gamma, U, V, b, fit_info = alternating_mle(
            N,
            K,
            r,
            n_ijk,
            y_ijk,
            max_steps=max_steps,
            tol=tol,
            tau=tau,
            inner_maxiter=inner_maxiter,
            reanchor_steps=reanchor_steps,
            n_order=n_order,
            y_order=y_order,
            polish=polish,
        )
    fit_info = dict(fit_info)
    fit_info["b"] = b
    return mu, gamma, U, V, fit_info


def score_contrast_gradient(gamma, mu, U, V, k, i, j, b=None, a_order=None):
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if not (0 <= k < K and 0 <= i < N and 0 <= j < N and i != j):
        raise ValueError(f"invalid score contrast indices {(k, i, j)} for K={K}, N={N}")
    with_b = b is not None
    slices = _param_slices(K, N, r, with_b=with_b)
    grad = np.zeros(slices["size"], dtype=float)
    grad[slices["gamma"].start + k] = mu[i] - mu[j]
    grad[slices["mu"].start + i] = gamma[k]
    grad[slices["mu"].start + j] = -gamma[k]
    if r > 0:
        U_offset = slices["U"].start + k * r
        grad[U_offset: U_offset + r] = V[i, :] - V[j, :]
        V_i_offset = slices["V"].start + i * r
        V_j_offset = slices["V"].start + j * r
        grad[V_i_offset: V_i_offset + r] = U[k, :]
        grad[V_j_offset: V_j_offset + r] = -U[k, :]
    if with_b and a_order is not None:
        grad[slices["b"].start + k] = float(a_order)
    return grad


def score_entry_gradient(gamma, mu, U, V, k, i):
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if not (0 <= k < K and 0 <= i < N):
        raise ValueError(f"invalid score entry indices {(k, i)} for K={K}, N={N}")
    slices = _param_slices(K, N, r)
    grad = np.zeros(slices["size"], dtype=float)
    grad[slices["gamma"].start + k] = mu[i]
    grad[slices["mu"].start + i] = gamma[k]
    if r > 0:
        U_offset = slices["U"].start + k * r
        V_i_offset = slices["V"].start + i * r
        grad[U_offset: U_offset + r] = V[i, :]
        grad[V_i_offset: V_i_offset + r] = U[k, :]
    return grad


def consensus_contrast_gradient(gamma, mu, U, V, i, j):
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    if not (0 <= i < N and 0 <= j < N and i != j):
        raise ValueError(f"invalid consensus contrast indices {(i, j)} for N={N}")
    slices = _param_slices(K, N, r)
    grad = np.zeros(slices["size"], dtype=float)
    grad[slices["mu"].start + i] = 1.0
    grad[slices["mu"].start + j] = -1.0
    return grad


def compute_information_matrix(gamma, mu, U, V, n_ijk, b=None, n_order=None):
    """Plug-in Fisher information (per comparison) in the full factor chart, vectorized over cells."""
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    K = gamma.size
    N = mu.size
    r = U.shape[1]
    with_b = n_order is not None
    tri_i, tri_j = np.triu_indices(N, k=1)
    if with_b:
        b = np.zeros(K, dtype=float) if b is None else np.asarray(b, dtype=float)
        n_order = np.asarray(n_order, dtype=float)
        if n_order.shape != (K, N, N, 2):
            raise ValueError(f"n_order must have shape {(K, N, N, 2)}, got {n_order.shape}")
        counts = n_order[:, tri_i, tri_j, :]  # (K, P, 2)
        total_n = float(np.sum(counts))
    else:
        n_ijk = np.asarray(n_ijk, dtype=float)
        if n_ijk.shape != (K, N, N):
            raise ValueError(f"n_ijk must have shape {(K, N, N)}, got {n_ijk.shape}")
        counts = n_ijk[:, tri_i, tri_j][:, :, None]  # (K, P, 1)
        total_n = float(np.sum(counts))
    if total_n <= 0:
        raise ValueError("information matrix requires at least one observed comparison")

    slices = _param_slices(K, N, r, with_b=with_b)
    dim = slices["size"]
    score = np.outer(gamma, mu) + U @ V.T
    delta = score[:, tri_i] - score[:, tri_j]  # (K, P)
    dmu = mu[tri_i] - mu[tri_j]
    dV = V[tri_i] - V[tri_j]  # (P, r)
    P = tri_i.size
    orders = ORDER_A if with_b else np.array([0.0])
    info = np.zeros((dim, dim), dtype=float)
    for k in range(K):
        # gradient of the linear predictor wrt full chart for judge k, all pairs: (P, dim)
        X = np.zeros((P, dim), dtype=float)
        X[:, slices["gamma"].start + k] = dmu
        X[np.arange(P), slices["mu"].start + tri_i] = gamma[k]
        X[np.arange(P), slices["mu"].start + tri_j] = -gamma[k]
        if r > 0:
            X[:, slices["U"].start + k * r: slices["U"].start + (k + 1) * r] = dV
            rows_i = slices["V"].start + tri_i[:, None] * r + np.arange(r)[None, :]
            rows_j = slices["V"].start + tri_j[:, None] * r + np.arange(r)[None, :]
            X[np.arange(P)[:, None], rows_i] = U[k][None, :]
            X[np.arange(P)[:, None], rows_j] = -U[k][None, :]
        for a_idx, a_val in enumerate(orders):
            cell_n = counts[k, :, a_idx]
            if not np.any(cell_n > 0):
                continue
            Xa = X.copy()
            if with_b:
                Xa[:, slices["b"].start + k] = a_val
            p = expit(delta[k] + (a_val * b[k] if with_b else 0.0))
            w = (cell_n / total_n) * p * (1.0 - p)
            info += (Xa * w[:, None]).T @ Xa
    return info


def _evaluate_uq_target(target, gamma, mu, U, V, b=None):
    with_b = b is not None
    target_type = target.get("type", "score_diff")
    if target_type in {"score_diff", "judge_score_diff"}:
        k = int(target["k"])
        i = int(target["i"])
        j = int(target["j"])
        value = gamma[k] * (mu[i] - mu[j])
        if U.shape[1] > 0:
            value += U[k, :] @ (V[i, :] - V[j, :])
        a_order = target.get("a_order")
        if with_b and a_order is not None:
            value += float(a_order) * b[k]
        grad = score_contrast_gradient(gamma, mu, U, V, k, i, j, b=b, a_order=a_order)
        label = target.get("label", f"S[{k},{i}]-S[{k},{j}]")
    elif target_type in {"score_entry", "judge_score_entry"}:
        k = int(target["k"])
        i = int(target["i"])
        value = gamma[k] * mu[i]
        if U.shape[1] > 0:
            value += U[k, :] @ V[i, :]
        grad = score_entry_gradient(gamma, mu, U, V, k, i)
        if with_b:
            padded = np.zeros(_param_slices(gamma.size, mu.size, U.shape[1], with_b=True)["size"])
            padded[: grad.size] = grad
            grad = padded
        label = target.get("label", f"S[{k},{i}]")
    elif target_type == "consensus_diff":
        i = int(target["i"])
        j = int(target["j"])
        value = mu[i] - mu[j]
        grad = consensus_contrast_gradient(gamma, mu, U, V, i, j)
        if with_b:
            padded = np.zeros(_param_slices(gamma.size, mu.size, U.shape[1], with_b=True)["size"])
            padded[: grad.size] = grad
            grad = padded
        label = target.get("label", f"mu[{i}]-mu[{j}]")
    elif target_type == "consensus_score":
        i = int(target["i"])
        N = mu.size
        if not (0 <= i < N):
            raise ValueError(f"invalid consensus score index {i} for N={N}")
        value = mu[i]
        slices = _param_slices(gamma.size, N, U.shape[1], with_b=with_b)
        grad = np.zeros(slices["size"], dtype=float)
        grad[slices["mu"].start + i] = 1.0
        label = target.get("label", f"mu[{i}]")
    elif target_type == "gamma":
        k = int(target["k"])
        K = gamma.size
        if not (0 <= k < K):
            raise ValueError(f"invalid gamma index {k} for K={K}")
        value = gamma[k]
        slices = _param_slices(K, mu.size, U.shape[1], with_b=with_b)
        grad = np.zeros(slices["size"], dtype=float)
        grad[slices["gamma"].start + k] = 1.0
        label = target.get("label", f"gamma[{k}]")
    elif target_type == "position_effect":
        k = int(target["k"])
        if b is None:
            raise ValueError("position_effect UQ requires fitted b")
        K = gamma.size
        if not (0 <= k < K):
            raise ValueError(f"invalid position-effect index {k} for K={K}")
        value = float(b[k])
        slices = _param_slices(K, mu.size, U.shape[1], with_b=True)
        grad = np.zeros(slices["size"], dtype=float)
        grad[slices["b"].start + k] = 1.0
        label = target.get("label", f"b[{k}]")
    else:
        raise ValueError(f"unknown UQ target type: {target_type}")
    return label, float(value), grad


def uncertainty_quantification(gamma, mu, U, V, n_ijk, targets, alpha=0.05, rcond=1e-8, b=None, n_order=None):
    info = compute_information_matrix(gamma, mu, U, V, n_ijk, b=b, n_order=n_order)
    N = mu.size
    if n_order is not None:
        total_n = float(np.sum(n_order[:, np.triu_indices(N, k=1)[0], np.triu_indices(N, k=1)[1], :]))
        b = np.zeros(gamma.size, dtype=float) if b is None else np.asarray(b, dtype=float)
    else:
        total_n = float(np.sum(n_ijk[:, np.triu_indices(N, k=1)[0], np.triu_indices(N, k=1)[1]]))
        b = None
    covariance = np.linalg.pinv(info, rcond=rcond) / total_n
    z_value = float(norm.ppf(1.0 - alpha / 2.0))

    intervals = []
    for target in targets:
        label, estimate, grad = _evaluate_uq_target(target, gamma, mu, U, V, b=b)
        variance = float(grad @ covariance @ grad)
        se = float(np.sqrt(max(variance, 0.0)))
        intervals.append(
            {
                "label": label,
                "type": target.get("type", "score_diff"),
                "estimate": estimate,
                "se": se,
                "alpha": float(alpha),
                "lower": estimate - z_value * se,
                "upper": estimate + z_value * se,
            }
        )
    return {
        "intervals": intervals,
        "information_matrix": info,
        "covariance": covariance,
        "total_n": total_n,
        "z_value": z_value,
    }


def select_rank_by_bic(N, K, n_ijk, y_ijk, candidate_ranks=None, max_steps=800, tol=1e-5, n_order=None, y_order=None):
    """BIC over candidate ranks on the LLM likelihood.

    With `n_order`/`y_order` the order-effect BTL is fitted at every rank and
    the K position effects are added to the parameter count; that offset is
    constant in r, so it changes the reported BIC values but not the argmin.
    """
    r_max = min(K - 1, N - 2)
    if candidate_ranks is None:
        candidate_ranks = list(range(r_max + 1))
    results = {}
    if n_order is not None and n_ijk is None:
        n_ijk, y_ijk = collapse_order_counts(n_order, y_order)
    n_total = float(np.sum(n_ijk[:, np.triu_indices(N, 1)[0], np.triu_indices(N, 1)[1]]))
    if n_total <= 0:
        raise ValueError("BIC selection requires positive total comparisons")

    for r in candidate_ranks:
        mu, gamma, U, V, fit_info = estimate_parameters(
            N, K, r, n_ijk, y_ijk, max_steps=max_steps, tol=tol, tau=10.0, n_order=n_order, y_order=y_order
        )
        d_r = r * (K + N - r - 3) + (K if n_order is not None else 0)
        bic = 2.0 * fit_info["nll"] + d_r * np.log(n_total)
        results[r] = {
            "bic": float(bic),
            "mu": mu,
            "gamma": gamma,
            "U": U,
            "V": V,
            "fit_info": fit_info,
        }
    best_rank = min(results, key=lambda r: results[r]["bic"])
    return best_rank, results
