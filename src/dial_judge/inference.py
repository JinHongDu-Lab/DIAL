"""
Inference and population-risk utilities for DIAL.

- population_excess_risk: R_0(s) - R_0(s_0) under a pair-sampling distribution
  (the quantity in Thm. population-risk-tradeoff; equals the rho-weighted KL).
- human_only_uq: Fisher-information Wald intervals for the unrestricted human
  BTL estimator (the lambda = 0 endpoint).
- joint_sandwich: the fixed-lambda sandwich covariance of Thm.
  weighted-asymptotic-normality for the joint DIAL estimator,
      Cov(zeta_hat) ~= H_lambda^{-1} J_{lambda,tau} H_lambda^{-1} / n_H,
      H_lambda = Hessian of q_lambda = ell_H / n_H + lambda ell_L / n_L,
      J = V_0 + lambda^2 (n_H / n_L) V_L,
  with V_0, V_L the centered per-observation score covariances of the human and
  LLM likelihood contributions, computed in the centering-reduced factor chart
  used by gacv.py. Covariances of s_cal, S, and b follow by the delta method.
  Chart redundancies (factor rotations) are exact invariances of both
  likelihoods, so pseudo-inverses are used throughout; the delta-method
  variances of invariant targets are unaffected.
"""
import numpy as np
from scipy.special import expit
from scipy.stats import norm

from .dial_model import human_nll_and_grad, pairs_to_arrays, total_llm_n
from .hja import negative_log_likelihood_and_grad
from .gacv import (
    hessian_from_grad,
    human_observation_grads,
    pack_reduced,
    q_lambda_grad,
    unpack_reduced,
)
from .hja import ORDER_A, make_centering_basis


# ---------------------------------------------------------------------------
# Population risk
# ---------------------------------------------------------------------------

def _uniform_pair_weights(N):
    tri_i, tri_j = np.triu_indices(N, k=1)
    return tri_i, tri_j, np.ones(tri_i.size) / tri_i.size


def population_human_risk(s, s0, pair_weights=None):
    """R_0(s) = sum_{i<j} rho_ij [log(1 + exp(s_i - s_j)) - p_0ij (s_i - s_j)]."""
    s = np.asarray(s, dtype=float)
    s0 = np.asarray(s0, dtype=float)
    tri_i, tri_j, w = _uniform_pair_weights(s.size)
    if pair_weights is not None:
        w = np.asarray(pair_weights, dtype=float)
    d = s[tri_i] - s[tri_j]
    p0 = expit(s0[tri_i] - s0[tri_j])
    return float(np.sum(w * (np.logaddexp(0.0, d) - p0 * d)))


def population_excess_risk(s, s0, pair_weights=None):
    """R_0(s) - R_0(s_0) >= 0, the rho-weighted Bernoulli KL (Prop. population-alignment)."""
    return population_human_risk(s, s0, pair_weights) - population_human_risk(s0, s0, pair_weights)


def pairwise_contrast_matrix(N):
    """D in {-1,0,1}^{N(N-1)/2 x N} with rows e_i - e_j, i < j."""
    tri_i, tri_j = np.triu_indices(N, k=1)
    D = np.zeros((tri_i.size, N))
    D[np.arange(tri_i.size), tri_i] = 1.0
    D[np.arange(tri_i.size), tri_j] = -1.0
    return D


def contrast_intervals(s, cov_s, alpha=0.05):
    """Wald intervals for all pairwise contrasts s_i - s_j from a score covariance."""
    N = np.asarray(s).size
    D = pairwise_contrast_matrix(N)
    est = D @ np.asarray(s, dtype=float)
    var = np.einsum("pi,ij,pj->p", D, cov_s, D)
    se = np.sqrt(np.maximum(var, 0.0))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return {"estimate": est, "se": se, "lower": est - z * se, "upper": est + z * se}


# ---------------------------------------------------------------------------
# Human-only endpoint
# ---------------------------------------------------------------------------

def comparison_laplacian(N, human_pairs, s, normalize=True):
    """Empirical comparison Laplacian at plug-in score `s`.

    sum_{(i,j)} w_ij p_ij(1-p_ij) (e_i-e_j)(e_i-e_j)^T, w_ij the pair count.
    With normalize=True (default) divides by n_H = sum(w_ij), giving the per-observation
    average that matches the O(1) scale of H_lambda = Hessian of q_lambda = ell_H/n_H + ...;
    with normalize=False returns the raw (n_H x) Fisher information used by human_only_uq.
    """
    i_idx, j_idx, n_arr, _ = pairs_to_arrays(human_pairs)
    s = np.asarray(s, dtype=float)
    p = expit(s[i_idx] - s[j_idx])
    w = n_arr * p * (1.0 - p)
    X = np.zeros((i_idx.size, N))
    X[np.arange(i_idx.size), i_idx] = 1.0
    X[np.arange(i_idx.size), j_idx] = -1.0
    info = X.T @ (X * w[:, None])
    return info / n_arr.sum() if normalize else info


def human_only_uq(N, human_pairs, s_hat, alpha=0.05, rcond=1e-10):
    """Fisher-information covariance of the centered human-only BTL estimator."""
    s_hat = np.asarray(s_hat, dtype=float)
    info = comparison_laplacian(N, human_pairs, s_hat, normalize=False)
    B = make_centering_basis(N)
    cov_s = B @ np.linalg.pinv(B.T @ info @ B, rcond=rcond) @ B.T
    return {"s_H": s_hat, "covariance": cov_s, "contrasts": contrast_intervals(s_hat, cov_s, alpha)}


# ---------------------------------------------------------------------------
# Joint DIAL sandwich
# ---------------------------------------------------------------------------

def _llm_cell_design(zeta, N, K, r, judge_basis, item_basis, use_order, n_order, y_order, n_ijk, y_ijk, return_cluster=False):
    """Per-cell linear-predictor gradients x_c in chart coordinates, with counts (n_c, y_c) and eta_c.

    With return_cluster=True also returns an integer cluster id per cell (judge x pair)."""
    gamma, mu, U, V, b, _alpha, _a = unpack_reduced(zeta, N, K, r, judge_basis, item_basis, use_order)
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_g = K - 1
    n_U = (K - 1) * r
    off_b = n_g + n_U
    off_mu = off_b + (K if use_order else 0)
    off_V = off_mu + (N - 1)
    dim = zeta.size

    rows, n_c, y_c, eta_c, cl = [], [], [], [], []
    n_pairs_total = tri_i.size
    orders = ORDER_A if use_order else np.array([0.0])
    for k in range(K):
        for a_idx, a_val in enumerate(orders):
            if use_order:
                nn = np.asarray(n_order, dtype=float)[k, tri_i, tri_j, a_idx]
                yy = np.asarray(y_order, dtype=float)[k, tri_i, tri_j, a_idx]
            else:
                nn = np.asarray(n_ijk, dtype=float)[k, tri_i, tri_j]
                yy = np.asarray(y_ijk, dtype=float)[k, tri_i, tri_j]
            mask = nn > 0
            if not np.any(mask):
                continue
            ii, jj = tri_i[mask], tri_j[mask]
            dB = item_basis[ii] - item_basis[jj]  # (m, N-1)
            dmu = mu[ii] - mu[jj]
            dV = V[ii] - V[jj]  # (m, r)
            X = np.zeros((ii.size, dim))
            X[:, :n_g] = np.outer(dmu, judge_basis[k])
            if r > 0:
                X[:, n_g:off_b] = np.einsum("l,mq->mlq", judge_basis[k], dV).reshape(ii.size, -1)
            if use_order:
                X[:, off_b + k] = a_val
            X[:, off_mu:off_V] = gamma[k] * dB
            if r > 0:
                X[:, off_V:off_V + (N - 1) * r] = np.einsum("ml,q->mlq", dB, U[k]).reshape(ii.size, -1)
            eta = gamma[k] * dmu + (dV @ U[k] if r > 0 else 0.0) + a_val * b[k]
            rows.append(X)
            n_c.append(nn[mask])
            y_c.append(yy[mask])
            eta_c.append(eta)
            cl.append(k * n_pairs_total + np.flatnonzero(mask))
    out = (np.vstack(rows), np.concatenate(n_c), np.concatenate(y_c), np.concatenate(eta_c))
    return out + (np.concatenate(cl),) if return_cluster else out


def llm_score_covariance(zeta, N, K, r, judge_basis, item_basis, use_order, n_order, y_order, n_ijk, y_ijk, cluster=None):
    """
    Per-LLM-observation score covariance V_L.

    cluster=None: observations independent given the cell (binomial), centered closed form.
    cluster="pair": cluster-robust version treating all queries of judge k on pair (i, j)
    (both display orders) as one cluster, V_L = n_L^{-1} sum_c (sum_{u in c} (g_u - g_bar))(...)^T
    with g_bar the grand-mean per-observation score. This is the appropriate variance when repeated queries of the same pair share an
    unmodeled pair-level effect.
    """
    X, n_c, y_c, eta, cl = _llm_cell_design(zeta, N, K, r, judge_basis, item_basis, use_order, n_order, y_order, n_ijk, y_ijk, return_cluster=True)
    p = expit(eta)
    n_L = float(n_c.sum())
    if cluster is None:
        second = y_c * (p - 1.0) ** 2 + (n_c - y_c) * p ** 2
        raw = (X * second[:, None]).T @ X / n_L
        g_bar = X.T @ (n_c * p - y_c) / n_L
        return raw - np.outer(g_bar, g_bar), n_L
    # clustered form: sum over clusters of the summed centered per-observation
    # scores, sum_{u in cluster} (g_u - g_bar) = g_c - n_c g_bar, with g_bar the grand mean score.
    resid = n_c * p - y_c  # cell-level summed residual
    g_bar = X.T @ resid / n_L
    uniq, inv = np.unique(cl, return_inverse=True)
    G = np.zeros((uniq.size, X.shape[1]))
    np.add.at(G, inv, X * resid[:, None] - np.outer(n_c, g_bar))
    return G.T @ G / n_L, n_L


def _target_jacobian(zeta, N, K, r, judge_basis, item_basis, use_order, eps=1e-6):
    """Finite-difference Jacobians of zeta -> (s_cal, vec(S), b); the map is polynomial so FD is near exact."""
    def targets(z):
        gamma, mu, U, V, b, alpha, a = unpack_reduced(z, N, K, r, judge_basis, item_basis, use_order)
        a = np.atleast_1d(a)
        s = alpha * mu + (V[:, :a.size] @ a if a.size else 0.0)
        S = np.outer(gamma, mu) + (U @ V.T if r > 0 else 0.0)
        return np.concatenate([s, S.ravel(), b])

    base = targets(zeta)
    steps = eps * np.eye(zeta.size)
    plus = np.column_stack([targets(zeta + e) for e in steps])
    minus = np.column_stack([targets(zeta - e) for e in steps])
    return (plus - minus) / (2.0 * eps)


def joint_sandwich(
    fit,
    N,
    K,
    r,
    human_pairs,
    n_ijk=None,
    y_ijk=None,
    n_order=None,
    y_order=None,
    alpha=0.05,
    rcond=1e-9,
    hess_eps=1e-5,
    llm_variation=True,
    cluster=None,
):
    """
    Sandwich covariance for a joint DIAL fit at fixed lambda (Thm. weighted-asymptotic-normality).

    Returns covariances of s_cal (N x N), S (KN x KN), b (K x K) and Wald
    intervals for pairwise contrasts of s_cal and for b. With
    llm_variation=False the LLM sampling term is dropped (tau = 0 formula).
    """
    use_order = n_order is not None
    if use_order and n_ijk is None:
        from .hja import collapse_order_counts

        n_ijk, y_ijk = collapse_order_counts(n_order, y_order)
    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    pair_arrays = pairs_to_arrays(human_pairs)
    n_H = float(pair_arrays[2].sum())
    n_L = total_llm_n(n_ijk, n_order=n_order)
    lam = float(fit["lam"])
    zeta = pack_reduced(fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"], fit["alpha_H"], fit["a"], use_order)

    def grad_fn(z):
        return q_lambda_grad(z, N, K, r, judge_basis, item_basis, use_order, n_ijk, y_ijk, n_order, y_order, pair_arrays, lam, n_H, n_L)

    H = hessian_from_grad(grad_fn, zeta, eps=hess_eps)

    from .gacv import observations_from_pairs

    i_obs, j_obs, z_obs = observations_from_pairs(human_pairs)
    g_t = human_observation_grads(zeta, N, K, r, judge_basis, item_basis, use_order, i_obs, j_obs, z_obs)
    g_bar = g_t.mean(axis=0)
    V0 = (g_t - g_bar).T @ (g_t - g_bar) / n_H
    if llm_variation:
        VL, _ = llm_score_covariance(zeta, N, K, r, judge_basis, item_basis, use_order, n_order, y_order, n_ijk, y_ijk, cluster=cluster)
        J = V0 + lam ** 2 * (n_H / n_L) * VL
    else:
        J = V0
    H_inv = np.linalg.pinv(H, rcond=rcond)
    cov_zeta = H_inv @ J @ H_inv / n_H

    Jac = _target_jacobian(zeta, N, K, r, judge_basis, item_basis, use_order)
    cov_targets = Jac @ cov_zeta @ Jac.T
    cov_s = cov_targets[:N, :N]
    cov_S = cov_targets[N:N + K * N, N:N + K * N]
    cov_b = cov_targets[N + K * N:, N + K * N:]

    z = float(norm.ppf(1.0 - alpha / 2.0))
    b = np.asarray(fit["b"], dtype=float)
    se_b = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    S_hat = np.outer(fit["gamma"], fit["mu"]) + (fit["U"] @ fit["V"].T if r > 0 else 0.0)
    se_S = np.sqrt(np.maximum(np.diag(cov_S), 0.0)).reshape(K, N)
    return {
        "cov_zeta": cov_zeta,
        "hessian": H,
        "dot_s0": Jac[:N, :],
        "score_cov": J,
        "cov_s": cov_s,
        "cov_S": cov_S,
        "cov_b": cov_b,
        "contrasts": contrast_intervals(fit["s_H"], cov_s, alpha),
        "b": {"estimate": b, "se": se_b, "lower": b - z * se_b, "upper": b + z * se_b},
        "S": {"estimate": S_hat, "se": se_S, "lower": S_hat - z * se_S, "upper": S_hat + z * se_S},
        "n_H": n_H,
        "n_L": n_L,
        "lam": lam,
    }


def llm_only_sandwich(fit, N, K, r, n_ijk=None, y_ijk=None, n_order=None, y_order=None,
                      alpha=0.05, rcond=1e-9, hess_eps=1e-5, cluster="pair"):
    """Cluster-robust Wald intervals for b from the LLM-only structured fit (the lambda = infinity fit).

    Same cell-clustered meat as `joint_sandwich`, specialized to the LLM block:
    the criterion is ell_L / n_L alone, so

        Cov(zeta_hat) ~= H_L^{-1} V_L^{cell} H_L^{-1} / n_L,

    with H_L the Hessian of the per-observation LLM negative log-likelihood. The human-side
    coordinates (alpha, a) of the reduced chart do not enter that criterion and are dropped
    before inverting. Factor rotations are exact invariances of the LLM likelihood, so H_L is
    singular in those directions and a pseudo-inverse is used, as in the joint case; b is
    invariant under them.
    """
    use_order = n_order is not None
    if not use_order:
        raise ValueError("an order-effect fit is required for intervals on b")
    if n_ijk is None:
        from .hja import collapse_order_counts

        n_ijk, y_ijk = collapse_order_counts(n_order, y_order)
    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    zeta = pack_reduced(fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"], 0.0, np.zeros(max(r, 1)), use_order)
    n_L = total_llm_n(n_ijk, n_order=n_order)

    n_tail = 1 + max(r, 1)          # the (alpha, a) coordinates of the chart, unused by the LLM criterion

    def grad_fn(z):
        gamma, mu, U, V, b, _alpha, _a = unpack_reduced(z, N, K, r, judge_basis, item_basis, use_order)
        l_l, gmu, ggamma, gU, gV, gb = negative_log_likelihood_and_grad(
            mu, gamma, U, V, n_ijk, y_ijk, b=b, n_order=n_order, y_order=y_order)
        parts = [judge_basis.T @ ggamma, (judge_basis.T @ gU).ravel(), gb,
                 item_basis.T @ gmu, (item_basis.T @ gV).ravel()]
        return l_l / n_L, np.concatenate([p / n_L for p in parts] + [np.zeros(n_tail)])

    H = hessian_from_grad(grad_fn, zeta, eps=hess_eps)
    J, _ = llm_score_covariance(zeta, N, K, r, judge_basis, item_basis, use_order,
                                n_order, y_order, n_ijk, y_ijk, cluster=cluster)
    # drop the human-side coordinates (alpha and a), which the LLM criterion does not involve
    keep = np.arange(zeta.size - n_tail)
    H_inv = np.linalg.pinv(H[np.ix_(keep, keep)], rcond=rcond)
    cov = H_inv @ J[np.ix_(keep, keep)] @ H_inv / n_L

    off_b = (K - 1) + (K - 1) * r
    idx = np.arange(off_b, off_b + K)
    cov_b = cov[np.ix_(idx, idx)]
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    b = np.asarray(fit["b"], dtype=float)
    se_b = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    return {"cov_zeta": cov, "hessian": H, "score_cov": J, "cov_b": cov_b, "n_L": n_L,
            "b": {"estimate": b, "se": se_b, "lower": b - z_crit * se_b, "upper": b + z_crit * se_b}}


# ---------------------------------------------------------------------------
# Interval validity under local misalignment
# ---------------------------------------------------------------------------

def bias_projection_operator(dot_s0, H, L, rcond=1e-9):
    """A_lambda := dot_s0 H_lambda^{-1} dot_s0^T L.

    L-self-adjoint (w.r.t. <x,y>_L = x^T L y), eigenvalues in [0,1] nonincreasing in
    lambda, A_lambda h = h for h in col(W), and A_lambda -> P^L_W as lambda -> infty
    (validated numerically in tests, not asserted here).
    """
    H_inv = np.linalg.pinv(np.asarray(H, dtype=float), rcond=rcond)
    return dot_s0 @ H_inv @ dot_s0.T @ np.asarray(L, dtype=float)


def bias_sensitivity(D, A_lambda, L, rcond=1e-9):
    """kappa(d, lambda) := {d^T (I-A_lambda) L^+ (I-A_lambda)^T d}^{1/2} for each row d of D.

    D defaults to the pairwise-contrast matrix elsewhere in this module; any set of
    contrasts with D @ ones(N) = 0 is valid.
    """
    D = np.asarray(D, dtype=float)
    N = A_lambda.shape[0]
    M = np.eye(N) - A_lambda
    L_pinv = np.linalg.pinv(np.asarray(L, dtype=float), rcond=rcond)
    Dt = D @ M
    quad = np.einsum("pi,ij,pj->p", Dt, L_pinv, Dt)
    return np.sqrt(np.maximum(quad, 0.0))


def noncentrality_interval(T, df, level=0.90):
    """Two-sided confidence interval for the noncentrality of a chi-square(df) from one observation T.

    The lower end is the largest noncentrality whose upper tail beyond T has mass at most
    (1 - level)/2 (zero when T is below the corresponding central quantile); the upper end
    is the smallest noncentrality whose lower tail below T has mass at most (1 - level)/2.
    Dividing both ends by 2 n_H gives the interval for Delta_W under the first-order
    approximation E[T] = df + 2 n_H Delta_W of the calibration test (also the delta_W of the
    local-misalignment widening, since delta_W = n_human Delta_W and
    the n_human factors cancel against the widening formula's own 1/sqrt(n_human): see
    misalignment_widened_contrasts).
    """
    from scipy.optimize import brentq
    from scipy.stats import ncx2

    a = (1.0 - level) / 2.0
    f_lo = lambda l: ncx2.sf(T, df, l) - a
    if f_lo(0.0) >= 0:
        lo = 0.0
    else:
        b = 10.0
        while f_lo(b) < 0:
            b *= 2
        lo = brentq(f_lo, 0.0, b)
    f_hi = lambda l: ncx2.cdf(T, df, l) - a
    if f_hi(0.0) <= 0:
        hi = 0.0
    else:
        b = 10.0
        while f_hi(b) > 0:
            b *= 2
        hi = brentq(f_hi, 0.0, b)
    return lo, hi


def misalignment_widened_contrasts(sw, N, human_pairs, s_full, delta_hi, alpha=0.05, rcond=1e-9, D=None):
    """Widen the pairwise-contrast intervals of `sw = joint_sandwich(...)` for asymptotic
    coverage under local misalignment.

    `s_full` is a consistent human-only estimate of s_human (e.g. `calibration_restriction_test`'s
    "s_full"), used only to plug into the empirical comparison Laplacian L.
    `delta_hi` is the upper end of a confidence interval for Delta_W at the pilot used to fit
    `sw` (e.g. the upper end of `noncentrality_interval` divided by 2 n_human), standing in for the confidence upper bound on delta_W = n_human * Delta_W
    of the calibration test; the n_human factor cancels against the widening formula's own
    1/sqrt(n_human), so the widening below is kappa(d,lambda) * sqrt(2 * delta_hi) exactly,
    with no separate n_human term.

    Returns the original contrasts plus "kappa", "widening", "lower_widened", "upper_widened",
    and the bias operator "A_lambda" for diagnostics.
    """
    if D is None:
        D = pairwise_contrast_matrix(N)
    L = comparison_laplacian(N, human_pairs, s_full, normalize=True)
    A_lambda = bias_projection_operator(sw["dot_s0"], sw["hessian"], L, rcond=rcond)
    kappa = bias_sensitivity(D, A_lambda, L, rcond=rcond)
    widening = kappa * np.sqrt(2.0 * max(delta_hi, 0.0))
    base = sw["contrasts"]
    return {
        "estimate": base["estimate"],
        "se": base["se"],
        "lower": base["lower"],
        "upper": base["upper"],
        "kappa": kappa,
        "widening": widening,
        "lower_widened": base["lower"] - widening,
        "upper_widened": base["upper"] + widening,
        "A_lambda": A_lambda,
        "L": L,
    }


def calibration_restriction_test(N, human_pairs, W, maxiter=1000, firth_fallback=True):
    """Likelihood-ratio test of the calibration restriction s_0 = W c against the unrestricted centered BTL.

    Statistic 2 { L_H(c-hat; W) - L_H^full(s-hat_0) } with L_H the total human negative
    log-likelihood, compared with chi-square on (N - 1) - d degrees of freedom, d = W.shape[1]
    (W treated as known, i.e. the n_H / n_L -> 0 regime).
    Returns the statistic, degrees of freedom, p-value, both fitted scores, and whether the
    unrestricted MLE exists (`btl_mle_exists`).

    When it does not (Ford 1957 separation) and `firth_fallback` is set, `s_full` is instead the
    Firth (1993) bias-reduced fit (`hja.fit_centered_btl_firth`),
    which is always finite, and `s_full_method` records which fit was used ("mle" or "firth")
    instead of silently returning the inflated statistic of a separated maximizer as before.
    """
    from scipy.stats import chi2

    from .dial_model import fit_human_calibration
    from .gacv import btl_mle_exists
    from .hja import centered_btl_loss_and_grad_from_pairs, fit_centered_btl_firth, fit_centered_btl_from_pairs

    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    d = W.shape[1]
    alpha, a = fit_human_calibration(W[:, 0], W[:, 1:], human_pairs, maxiter=maxiter)
    s_restricted = W @ np.concatenate([[alpha], a])
    exists = btl_mle_exists(N, human_pairs)
    if exists or not firth_fallback:
        s_full = fit_centered_btl_from_pairs(N, human_pairs, maxiter=maxiter)
        s_full_method = "mle"
    else:
        s_full = fit_centered_btl_firth(N, human_pairs, maxiter=maxiter)
        s_full_method = "firth"
    nll_r = centered_btl_loss_and_grad_from_pairs(N, human_pairs, s_restricted)[0]
    nll_u = centered_btl_loss_and_grad_from_pairs(N, human_pairs, s_full)[0]
    stat = max(0.0, 2.0 * (nll_r - nll_u))
    df = (N - 1) - d
    return {"stat": float(stat), "df": int(df), "pvalue": float(chi2.sf(stat, df)) if df > 0 else float("nan"),
            "s_restricted": s_restricted, "s_full": s_full, "human_only_exists": bool(exists), "s_full_method": s_full_method,
            "coefficients": np.concatenate([[alpha], a])}
