"""
Inference and population-risk utilities for DIAL.

- population_excess_risk: R_0(s) - R_0(s_0) under a pair-sampling distribution
  (the quantity in Thm. population-risk-tradeoff; equals the rho-weighted KL).
- human_only_uq: Fisher-information Wald intervals for the unrestricted human
  BTL estimator (the lambda = 0 endpoint).
- joint_sandwich: the fixed-lambda sandwich covariance of Thm.
  weighted-asymptotic-normality for the joint DIAL estimator,
      Cov(zeta_hat) ~= H_lambda^{-1} J_{lambda,tau} H_lambda^{-1} / n_0,
      H_lambda = Hessian of q_lambda = ell_H / n_0 + lambda ell_L / n_L,
      J = V_0 + lambda^2 (n_0 / n_L) V_L,
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

def human_only_uq(N, human_pairs, s_hat, alpha=0.05, rcond=1e-10):
    """Fisher-information covariance of the centered human-only BTL estimator."""
    i_idx, j_idx, n_arr, _ = pairs_to_arrays(human_pairs)
    s_hat = np.asarray(s_hat, dtype=float)
    p = expit(s_hat[i_idx] - s_hat[j_idx])
    w = n_arr * p * (1.0 - p)
    X = np.zeros((i_idx.size, N))
    X[np.arange(i_idx.size), i_idx] = 1.0
    X[np.arange(i_idx.size), j_idx] = -1.0
    info = X.T @ (X * w[:, None])
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
    with g_bar the grand-mean per-observation score (manuscript, paragraph "Comparisons that share
    a prompt"). This is the appropriate variance when repeated queries of the same pair share an
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
    # manuscript form (app:clustered): sum over clusters of the summed centered per-observation
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
        s = alpha * mu + (V @ a if r > 0 else 0.0)
        S = np.outer(gamma, mu) + (U @ V.T if r > 0 else 0.0)
        return np.concatenate([s, S.ravel(), b])

    base = targets(zeta)
    J = np.zeros((base.size, zeta.size))
    for d in range(zeta.size):
        e = np.zeros(zeta.size)
        e[d] = eps
        J[:, d] = (targets(zeta + e) - targets(zeta - e)) / (2.0 * eps)
    return J


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
    n_0 = float(pair_arrays[2].sum())
    n_L = total_llm_n(n_ijk, n_order=n_order)
    lam = float(fit["lam"])
    zeta = pack_reduced(fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"], fit["alpha_H"], fit["a"], use_order)

    def grad_fn(z):
        return q_lambda_grad(z, N, K, r, judge_basis, item_basis, use_order, n_ijk, y_ijk, n_order, y_order, pair_arrays, lam, n_0, n_L)

    H = hessian_from_grad(grad_fn, zeta, eps=hess_eps)

    from .gacv import observations_from_pairs

    i_obs, j_obs, z_obs = observations_from_pairs(human_pairs)
    g_t = human_observation_grads(zeta, N, K, r, judge_basis, item_basis, use_order, i_obs, j_obs, z_obs)
    g_bar = g_t.mean(axis=0)
    V0 = (g_t - g_bar).T @ (g_t - g_bar) / n_0
    if llm_variation:
        VL, _ = llm_score_covariance(zeta, N, K, r, judge_basis, item_basis, use_order, n_order, y_order, n_ijk, y_ijk, cluster=cluster)
        J = V0 + lam ** 2 * (n_0 / n_L) * VL
    else:
        J = V0
    H_inv = np.linalg.pinv(H, rcond=rcond)
    cov_zeta = H_inv @ J @ H_inv / n_0

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
        "score_cov": J,
        "cov_s": cov_s,
        "cov_S": cov_S,
        "cov_b": cov_b,
        "contrasts": contrast_intervals(fit["s_H"], cov_s, alpha),
        "b": {"estimate": b, "se": se_b, "lower": b - z * se_b, "upper": b + z * se_b},
        "S": {"estimate": S_hat, "se": se_S, "lower": S_hat - z * se_S, "upper": S_hat + z * se_S},
        "n_0": n_0,
        "n_L": n_L,
        "lam": lam,
    }
