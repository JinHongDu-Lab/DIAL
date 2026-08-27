"""
Synthetic data generation for the DIAL simulation study.

Extends the LLM-panel DGP (mu, gamma, U, V under the HJA decomposition,
ported from src/generate_simulation_data.py) with the human score model
(Eq. 2.3-2.5): s^H = alpha_H * mu + V @ a (+ delta, kept 0 in the primary
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
    comparisons = []
    extra_cell_indices = set(rng.choice(n_cells, size=remainder, replace=False).tolist()) if remainder > 0 else set()
    b = np.zeros(K, dtype=float) if b is None else np.asarray(b, dtype=float)

    cell_idx = 0
    order_values = (-1.0, 1.0) if use_order else (0.0,)
    for k in range(K):
        for i in range(N):
            for j in range(i + 1, N):
                for a in order_values:
                    eta = S[k, i] - S[k, j] + a * b[k]
                    prob = _logistic(eta)
                    cell_count = base_per_cell + int(cell_idx in extra_cell_indices)
                    if cell_count > 0:
                        draws = rng.binomial(1, prob, size=cell_count)
                        if use_order:
                            comparisons.extend((k, i, j, int(y), int(a)) for y in draws)
                        else:
                            comparisons.extend((k, i, j, int(y)) for y in draws)
                    cell_idx += 1

    return comparisons


def generate_balanced_comparisons_with_order(S, b, total_comparisons, random_seed=42):
    return generate_balanced_comparisons(S, total_comparisons, random_seed=random_seed, b=b)


def generate_human_comparisons(s_H, total_comparisons, random_seed=42, pair_subset=None):
    """
    Draw pooled (single-judge, K=1) human comparisons from the BTL model
    implied by a human score vector s_H (Eq. 2.4). If `pair_subset` is
    given, only those (i, j) item pairs are sampled from (near-balanced);
    otherwise all N(N-1)/2 pairs are used.
    """
    s_H = np.asarray(s_H, dtype=float)
    N = s_H.size
    if total_comparisons <= 0:
        raise ValueError(f"total_comparisons must be positive, got {total_comparisons}")

    pairs = pair_subset if pair_subset is not None else [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("pair_subset must be non-empty")

    base_per_cell = total_comparisons // n_pairs
    remainder = total_comparisons % n_pairs
    rng = np.random.default_rng(random_seed)
    extra_indices = set(rng.choice(n_pairs, size=remainder, replace=False).tolist()) if remainder > 0 else set()

    comparisons = []
    for idx, (i, j) in enumerate(pairs):
        prob = _logistic(s_H[i] - s_H[j])
        cell_count = base_per_cell + int(idx in extra_indices)
        if cell_count > 0:
            draws = rng.binomial(1, prob, size=cell_count)
            comparisons.extend((0, i, j, int(y)) for y in draws)
    return comparisons
