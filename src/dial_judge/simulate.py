"""
Synthetic data generation for the DIAL simulation study.

Extends the LLM-panel DGP (mu, gamma, U, V under the HJA decomposition) with
the human score model s^H = alpha_H * mu + V @ a (+ delta, kept 0 in the primary
model), so a simulation run has ground truth for both the LLM panel and the
human target and can check whether DIAL's calibrated s_H recovers the human
preference better than an LLM-only HJA fit does.
"""
import math

import numpy as np

from .data import comparisons_to_aggregated  # noqa: F401  (re-exported for convenience)

EPS = 1e-12


def validate_rank(N, K, r):
    r_max = min(K - 1, N - 2)
    if r < 0:
        raise ValueError(f"rank r must be nonnegative, got {r}")
    if r > r_max:
        raise ValueError(f"rank r={r} exceeds identifiable maximum {r_max} for N={N}, K={K}")
    return r_max


def _canonicalize_columns(matrix):
    out = np.array(matrix, dtype=float, copy=True)
    for col in range(out.shape[1]):
        nz = np.flatnonzero(np.abs(out[:, col]) > EPS)
        if nz.size == 0:
            raise ValueError(f"column {col} is numerically zero and cannot be canonicalized")
        if out[nz[0], col] < 0:
            out[:, col] *= -1.0
    return out


def generate_true_parameters(N, K, r, random_seed=42, heterogeneity_scale=1.0):
    """
    Returns mu_true (N,), gamma_true (K,), U_true (K, r), V_true (N, r)
    satisfying the HJA identification constraints: sum(mu)=0, sum(gamma)=K,
    1_K^T U = 0, 1_N^T V = 0, mu^T V = 0.
    """
    validate_rank(N, K, r)
    if heterogeneity_scale < 0:
        raise ValueError(f"heterogeneity_scale must be nonnegative, got {heterogeneity_scale}")

    rng = np.random.default_rng(random_seed)

    mu_true = rng.normal(size=N)
    mu_true = mu_true - np.mean(mu_true)

    gamma_true = K * rng.dirichlet(np.ones(K))

    if r == 0:
        U_true = np.zeros((K, 0), dtype=float)
        V_true = np.zeros((N, 0), dtype=float)
        return mu_true, gamma_true, U_true, V_true

    raw_v = rng.normal(size=(N, r))
    raw_v = raw_v - np.mean(raw_v, axis=0, keepdims=True)
    raw_v = raw_v - np.outer(mu_true, (mu_true @ raw_v) / max(mu_true @ mu_true, EPS))
    q_v, _ = np.linalg.qr(raw_v)
    strengths = np.linspace(r, 1, r, dtype=float)
    V_true = q_v[:, :r] @ np.diag(np.sqrt(N) * strengths)
    V_true = V_true - np.mean(V_true, axis=0, keepdims=True)
    V_true = V_true - np.outer(mu_true, (mu_true @ V_true) / max(mu_true @ mu_true, EPS))

    raw_u = rng.normal(size=(K, r))
    raw_u = raw_u - np.mean(raw_u, axis=0, keepdims=True)
    q_u, _ = np.linalg.qr(raw_u)
    U_true = q_u[:, :r] @ np.diag(np.sqrt(K) * strengths)

    U_true = _canonicalize_columns(U_true) * math.sqrt(heterogeneity_scale)
    V_true = _canonicalize_columns(V_true) * math.sqrt(heterogeneity_scale)
    mu_true = mu_true - np.mean(mu_true)

    return mu_true, gamma_true, U_true, V_true


def generate_human_calibration_params(V_true, alpha_true=1.0, a_scale=1.0, random_seed=42):
    """
    Draw ground-truth DIAL human calibration coefficients (alpha_H, a) for
    the human score model s^H = alpha_H * mu + V @ a. `a_scale` controls how much
    of the LLM panel's disagreement direction the human target loads onto
    (0 => humans follow the LLM consensus exactly with no V-alignment).
    """
    r = V_true.shape[1]
    rng = np.random.default_rng(random_seed)
    a_true = a_scale * rng.normal(size=r) if r > 0 else np.zeros(0)
    return float(alpha_true), a_true


def compute_score_matrix(mu, gamma, U, V):
    mu = np.asarray(mu, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    return np.outer(gamma, mu) + U @ V.T


def compute_human_score(mu, V, alpha_H, a, delta=None):
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    s = alpha_H * mu + (V @ a if V.shape[1] else np.zeros_like(mu))
    if delta is not None:
        s = s + np.asarray(delta, dtype=float)
    return s


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_position_effects(K, scale=0.5, random_seed=42):
    """Draw judge-specific position effects b_k. b_k > 0 prefers the first-displayed item."""
    rng = np.random.default_rng(random_seed)
    return rng.normal(0.0, scale, size=K)


def generate_balanced_comparisons(S, total_comparisons, random_seed=42, b=None):
    # Near-balanced design. If b is None, no order effect (records are (k,i,j,y)).
    # If b is given, both display orders are used (records are (k,i,j,y,a)).
    S = np.asarray(S, dtype=float)
    K, N = S.shape
    n_pairs = N * (N - 1) // 2
    use_order = b is not None
    n_cells = K * n_pairs * (2 if use_order else 1)
    if total_comparisons <= 0:
        raise ValueError(f"total_comparisons must be positive, got {total_comparisons}")

    base_per_cell = total_comparisons // n_cells
    remainder = total_comparisons % n_cells
    rng = np.random.default_rng(random_seed)
    extra_cell_indices = rng.choice(n_cells, size=remainder, replace=False) if remainder > 0 else np.empty(0, dtype=int)
    b = np.zeros(K, dtype=float) if b is None else np.asarray(b, dtype=float)

    # cells enumerate (k, i < j, a) in that order; counts differ only by the +1 of the remainder
    order_values = np.array([-1.0, 1.0]) if use_order else np.array([0.0])
    tri_i, tri_j = np.triu_indices(N, k=1)
    kk, pp, aa = (idx.ravel() for idx in np.meshgrid(np.arange(K), np.arange(n_pairs), np.arange(order_values.size), indexing="ij"))
    i_cell, j_cell, a_cell = tri_i[pp], tri_j[pp], order_values[aa]
    counts = np.full(n_cells, base_per_cell, dtype=int)
    counts[extra_cell_indices] += 1

    eta = S[kk, i_cell] - S[kk, j_cell] + a_cell * b[kk]
    rep = np.repeat(np.arange(n_cells), counts)
    draws = rng.binomial(1, _logistic(eta[rep]))
    k_obs, i_obs, j_obs, a_obs = kk[rep], i_cell[rep], j_cell[rep], a_cell[rep].astype(int)
    if use_order:
        return [(int(k_), int(i_), int(j_), int(y_), int(a_)) for k_, i_, j_, y_, a_ in zip(k_obs, i_obs, j_obs, draws, a_obs)]
    return [(int(k_), int(i_), int(j_), int(y_)) for k_, i_, j_, y_ in zip(k_obs, i_obs, j_obs, draws)]


def generate_balanced_comparisons_with_order(S, b, total_comparisons, random_seed=42):
    return generate_balanced_comparisons(S, total_comparisons, random_seed=random_seed, b=b)


def generate_study_parameters(N, K, r, random_seed=42, heterogeneity_scale=1.0, gamma_sd=0.3):
    """
    DGP of the synthetic study.

    mu Gaussian, centered, ||mu||_2^2 = N; V Gaussian, orthonormalized against
    1 and mu with N^{-1} V^T V = I_r; gamma_k = 1 + gamma_sd z_k rescaled to sum
    K; U Gaussian with zero column means scaled by sqrt(heterogeneity_scale).
    Differs from generate_true_parameters in the gamma law (Gaussian around 1
    rather than K * Dirichlet) and in the exact unit-norm item constraints.
    """
    validate_rank(N, K, r)
    rng = np.random.default_rng(random_seed)

    mu = rng.normal(size=N)
    mu = mu - mu.mean()
    mu = mu * math.sqrt(N) / np.linalg.norm(mu)

    gamma = 1.0 + gamma_sd * rng.normal(size=K)
    gamma = gamma / gamma.sum() * K

    if r == 0:
        return mu, gamma, np.zeros((K, 0)), np.zeros((N, 0))

    raw_v = rng.normal(size=(N, r))
    basis = np.column_stack([np.ones(N) / math.sqrt(N), mu / math.sqrt(N)])
    raw_v = raw_v - basis @ (basis.T @ raw_v)
    q_v, _ = np.linalg.qr(raw_v)
    V = q_v[:, :r] * math.sqrt(N)

    raw_u = rng.normal(size=(K, r))
    raw_u = raw_u - raw_u.mean(axis=0, keepdims=True)
    U = raw_u * math.sqrt(heterogeneity_scale)
    U, V = _canonicalize_columns(U), _canonicalize_columns(V)
    return mu, gamma, U, V


def generate_study_position_effects(K, tau_b=1.0, random_seed=42, low=0.3, high=1.2, p_first=0.8, null_judge=True):
    """b_k = tau_b * beta_k, beta_k ~ U(low, high) with sign +1 w.p. p_first; optionally one near-null judge."""
    rng = np.random.default_rng(random_seed)
    beta = rng.uniform(low, high, size=K)
    signs = np.where(rng.uniform(size=K) < p_first, 1.0, -1.0)
    b = beta * signs
    if null_judge and K > 0:
        b[-1] = 0.05
    return tau_b * b


def generate_study_calibration(V, c_mu=1.0, c_v_sd=0.5, random_seed=42):
    """c_0 = (c_mu, c_V) with c_V ~ N(0, c_v_sd^2 I_r) for s_0 = W c_0 (exact calibration)."""
    rng = np.random.default_rng(random_seed)
    r = V.shape[1]
    return float(c_mu), c_v_sd * rng.normal(size=r) if r > 0 else np.zeros(0)


def generate_random_llm_comparisons(S, b, total_comparisons, random_seed=42, swap_fraction=0.5, pair_weights=None, overdispersion=0.0, position_heterogeneity=0.0):
    """
    Draw `total_comparisons` LLM records (k, i, j, y, a) with judge uniform, pair
    drawn from `pair_weights` (uniform if None), and a = +1 (canonical i first)
    with probability 1 - swap_fraction, a = -1 with probability swap_fraction.
    swap_fraction = 0 gives a design in which z_k is constant and the position
    effect is not identified (Cond. LLM design fails).

    `position_heterogeneity` = sigma_pos > 0 replaces the constant b_k by a pair-varying
    b_kij = b_k + delta_kij, with delta centered within judge over pairs and of pairwise
    standard deviation sigma_pos, so judge k's average pair effect stays exactly b_k while the
    constant-b working model is misspecified.
    """
    S = np.asarray(S, dtype=float)
    b = np.asarray(b, dtype=float)
    K, N = S.shape
    rng = np.random.default_rng(random_seed)
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = tri_i.size
    if pair_weights is None:
        pair_weights = np.ones(n_pairs) / n_pairs
    k = rng.integers(0, K, size=total_comparisons)
    pair = rng.choice(n_pairs, size=total_comparisons, p=pair_weights)
    i, j = tri_i[pair], tri_j[pair]
    a = np.where(rng.uniform(size=total_comparisons) < swap_fraction, -1, 1)
    b_eff = b[k]
    if position_heterogeneity > 0:
        d = rng.normal(size=(K, n_pairs))
        d = d - d.mean(axis=1, keepdims=True)
        d = position_heterogeneity * d / d.std(axis=1, ddof=0, keepdims=True)
        b_eff = b_eff + d[k, pair]
    eta = S[k, i] - S[k, j] + a * b_eff
    if overdispersion > 0:
        # pair-level idiosyncrasy of judge k on pair (i, j), fixed across repeated queries:
        # eta += eps_kij, eps ~ N(0, overdispersion^2). The LLM likelihood is then misspecified
        # (extra-binomial variation) while the human model stays exact.
        eps = rng.normal(0.0, overdispersion, size=(K, n_pairs))
        eta = eta + eps[k, pair]
    y = rng.binomial(1, _logistic(eta))
    return [(int(kk), int(ii), int(jj), int(yy), int(aa)) for kk, ii, jj, yy, aa in zip(k, i, j, y, a)]


def generate_random_human_comparisons(s_H, total_comparisons, random_seed=42, pair_weights=None):
    """Draw pooled human records (0, i, j, y) with pairs uniform (or `pair_weights`)."""
    s_H = np.asarray(s_H, dtype=float)
    N = s_H.size
    rng = np.random.default_rng(random_seed)
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = tri_i.size
    if pair_weights is None:
        pair_weights = np.ones(n_pairs) / n_pairs
    pair = rng.choice(n_pairs, size=total_comparisons, p=pair_weights)
    i, j = tri_i[pair], tri_j[pair]
    y = rng.binomial(1, _logistic(s_H[i] - s_H[j]))
    return [(0, int(ii), int(jj), int(yy)) for ii, jj, yy in zip(i, j, y)]


def generate_human_comparisons(s_H, total_comparisons, random_seed=42, pair_subset=None):
    """
    Draw pooled (single-judge, K=1) human comparisons from the BTL model
    implied by a human score vector s_H. If `pair_subset` is
    given, only those (i, j) item pairs are sampled from (near-balanced);
    otherwise all N(N-1)/2 pairs are used.
    """
    s_H = np.asarray(s_H, dtype=float)
    N = s_H.size
    if total_comparisons <= 0:
        raise ValueError(f"total_comparisons must be positive, got {total_comparisons}")

    pairs = pair_subset if pair_subset is not None else np.column_stack(np.triu_indices(N, k=1))
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("pair_subset must be non-empty")

    base_per_cell = total_comparisons // n_pairs
    remainder = total_comparisons % n_pairs
    rng = np.random.default_rng(random_seed)
    extra_indices = rng.choice(n_pairs, size=remainder, replace=False) if remainder > 0 else np.empty(0, dtype=int)

    pair_arr = np.asarray(pairs, dtype=int).reshape(-1, 2)
    counts = np.full(n_pairs, base_per_cell, dtype=int)
    counts[extra_indices] += 1
    rep = np.repeat(np.arange(n_pairs), counts)
    i, j = pair_arr[rep, 0], pair_arr[rep, 1]
    draws = rng.binomial(1, _logistic(s_H[i] - s_H[j]))
    return [(0, int(i_), int(j_), int(y_)) for i_, j_, y_ in zip(i, j, draws)]
