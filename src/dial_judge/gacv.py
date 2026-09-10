"""
GACV selection of the DIAL LLM weight lambda (paper Section 5.3, Eq. 18).

For a finite candidate set Lambda subset (0, inf),

    GACV(lambda) = ell_H^full(s_hat_lambda) + 1/(n_H - 1) tr(H_hat^{-1} J_hat)

approximates leave-one-human-comparison-out predictive loss while keeping all
LLM comparisons in every candidate fit (Theorem 5). Coordinates are the
centering-reduced factor chart (gamma, U, b, mu, V, c) used by the joint
alternating algorithm; the trace is invariant to a smooth change of chart.
"""
import numpy as np
from scipy.special import expit

from .dial_model import (
    default_lambda,
    human_nll_and_grad,
    joint,
    pairs_to_arrays,
    total_human_n,
    total_llm_n,
)
from .hja import (
    collapse_order_counts,
    make_centering_basis,
    negative_log_likelihood_and_grad,
    reduce_item_block,
    reduce_judge_block,
)


def default_lambda_grid(n_L, n_H, n_points=7):
    mle = default_lambda(n_L, n_H)
    multipliers = np.geomspace(0.1, 10.0, n_points)
    grid = sorted({float(m * mle) for m in multipliers} | {float(mle)})
    return grid


def observations_from_records(records):
    i_idx, j_idx, z = [], [], []
    for rec in records:
        if len(rec) >= 4:
            _k, i, j, y = rec[0], rec[1], rec[2], rec[3]
        else:
            i, j, y = rec[0], rec[1], rec[2]
        i_idx.append(int(i))
        j_idx.append(int(j))
        z.append(float(y))
    if not i_idx:
        raise ValueError("no human observations supplied")
    return np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int), np.asarray(z, dtype=float)


def observations_from_pairs(pairs):
    i_idx, j_idx, z = [], [], []
    for i, j, n, y in pairs:
        n_win = float(y)
        n_loss = float(n) - float(y)
        n_win_i = int(round(n_win))
        n_loss_i = int(round(n_loss))
        if abs(n_win - n_win_i) < 1e-8 and abs(n_loss - n_loss_i) < 1e-8:
            i_idx.extend([int(i)] * n_win_i)
            j_idx.extend([int(j)] * n_win_i)
            z.extend([1.0] * n_win_i)
            i_idx.extend([int(i)] * n_loss_i)
            j_idx.extend([int(j)] * n_loss_i)
            z.extend([0.0] * n_loss_i)
        else:
            i_idx.append(int(i))
            j_idx.append(int(j))
            z.append(float(y) / float(n) if n else 0.5)
    if not i_idx:
        raise ValueError("no human observations supplied")
    return np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int), np.asarray(z, dtype=float)


def pack_reduced(gamma, mu, U, V, b, alpha, a, use_order):
    gamma_red, U_red = reduce_judge_block(gamma, U)
    mu_red, V_red = reduce_item_block(mu, V)
    parts = [gamma_red, U_red.ravel()]
    if use_order:
        parts.append(np.asarray(b, dtype=float))
    parts.extend([mu_red, V_red.ravel(), np.array([float(alpha)], dtype=float), np.atleast_1d(a)])
    return np.concatenate(parts)


def unpack_reduced(zeta, N, K, r, judge_basis, item_basis, use_order):
    n_g = K - 1
    n_U = (K - 1) * r
    idx = 0
    gamma_red = zeta[idx: idx + n_g]
    idx += n_g
    U_red = zeta[idx: idx + n_U].reshape(K - 1, r)
    idx += n_U
    if use_order:
        b = zeta[idx: idx + K]
        idx += K
    else:
        b = np.zeros(K, dtype=float)
    mu_red = zeta[idx: idx + (N - 1)]
    idx += N - 1
    V_red = zeta[idx: idx + (N - 1) * r].reshape(N - 1, r)
    idx += (N - 1) * r
    alpha = float(zeta[idx])
    a = zeta[idx + 1 :]
    gamma = np.ones(K, dtype=float) + judge_basis @ gamma_red
    U = judge_basis @ U_red
    mu = item_basis @ mu_red
    V = item_basis @ V_red
    return gamma, mu, U, V, b, alpha, a


def q_lambda_grad(
    zeta,
    N,
    K,
    r,
    judge_basis,
    item_basis,
    use_order,
    n_ijk,
    y_ijk,
    n_order,
    y_order,
    pair_arrays,
    lam,
    n_H,
    n_L,
):
    gamma, mu, U, V, b, alpha, a = unpack_reduced(zeta, N, K, r, judge_basis, item_basis, use_order)
    l_l, gmu_l, ggamma_l, gU_l, gV_l, gb_l = negative_log_likelihood_and_grad(
        mu, gamma, U, V, n_ijk, y_ijk, b=b, n_order=n_order, y_order=y_order
    )
    a = np.atleast_1d(a)
    l_h, galpha, ga, gmu_h, gV_h_cal = human_nll_and_grad(alpha, a, mu, V[:, :a.size], pair_arrays)
    gV_h = np.zeros_like(V)
    gV_h[:, :a.size] = gV_h_cal
    q = (l_h / n_H) + lam * (l_l / n_L)
    scale_l = lam / n_L
    g_gamma_red = judge_basis.T @ (scale_l * ggamma_l)
    g_U_red = (judge_basis.T @ (scale_l * gU_l)).ravel()
    g_mu_red = item_basis.T @ (gmu_h / n_H + scale_l * gmu_l)
    g_V_red = (item_basis.T @ (gV_h / n_H + scale_l * gV_l)).ravel()
    parts = [g_gamma_red, g_U_red]
    if use_order:
        parts.append(scale_l * gb_l)
    parts.extend([g_mu_red, g_V_red, np.array([galpha / n_H]), ga / n_H])
    return q, np.concatenate(parts)


def hessian_from_grad(grad_fn, x, eps=1e-5):
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    for i in range(n):
        e = np.zeros(n, dtype=float)
        e[i] = eps
        _, g_plus = grad_fn(x + e)
        _, g_minus = grad_fn(x - e)
        hess[i] = (g_plus - g_minus) / (2.0 * eps)
    return 0.5 * (hess + hess.T)


def human_observation_grads(zeta, N, K, r, judge_basis, item_basis, use_order, i_obs, j_obs, z_obs):
    """Per-human-observation gradients of h_t in the reduced chart, vectorized: (n_obs, dim)."""
    gamma, mu, U, V, b, alpha, a = unpack_reduced(zeta, N, K, r, judge_basis, item_basis, use_order)
    a = np.atleast_1d(a)
    s = alpha * mu + (V[:, :a.size] @ a if a.size else np.zeros_like(mu))
    i_obs = np.asarray(i_obs, dtype=int)
    j_obs = np.asarray(j_obs, dtype=int)
    z_obs = np.asarray(z_obs, dtype=float)
    n_obs = z_obs.size
    dim = zeta.size

    n_g = K - 1
    n_U = (K - 1) * r
    offset_b = n_g + n_U
    offset_mu = offset_b + (K if use_order else 0)
    offset_V = offset_mu + (N - 1)
    offset_alpha = offset_V + (N - 1) * r
    offset_a = offset_alpha + 1

    resid = expit(s[i_obs] - s[j_obs]) - z_obs  # (n_obs,)
    dB = item_basis[i_obs] - item_basis[j_obs]  # (n_obs, N-1): item_basis^T grad_s with grad_s = resid (e_i - e_j)
    grads = np.zeros((n_obs, dim), dtype=float)
    grads[:, offset_mu:offset_V] = alpha * resid[:, None] * dB
    if a.size > 0:
        # only the first a.size columns of V enter the calibration; the remaining V columns get zero human gradient
        gV = np.zeros((n_obs, N - 1, r))
        gV[:, :, :a.size] = resid[:, None, None] * dB[:, :, None] * a[None, None, :]
        grads[:, offset_V:offset_alpha] = gV.reshape(n_obs, -1)
    grads[:, offset_alpha] = resid * (mu[i_obs] - mu[j_obs])
    if a.size > 0:
        grads[:, offset_a:] = resid[:, None] * (V[i_obs, :a.size] - V[j_obs, :a.size])
    return grads


def gacv_for_fit(
    fit,
    N,
    K,
    r,
    n_ijk,
    y_ijk,
    n_order,
    y_order,
    human_pairs,
    i_obs,
    j_obs,
    z_obs,
    rcond=1e-8,
    hess_eps=1e-5,
):
    use_order = n_order is not None
    n_H = float(z_obs.size)
    n_L = total_llm_n(n_ijk, n_order=n_order)
    lam = float(fit["lam"])
    pair_arrays = pairs_to_arrays(human_pairs)
    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    zeta = pack_reduced(fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"], fit["alpha_H"], fit["a"], use_order)

    def grad_fn(z):
        return q_lambda_grad(
            z, N, K, r, judge_basis, item_basis, use_order,
            n_ijk, y_ijk, n_order, y_order, pair_arrays, lam, n_H, n_L,
        )

    q_val, _ = grad_fn(zeta)
    hess = hessian_from_grad(grad_fn, zeta, eps=hess_eps)
    g_t = human_observation_grads(zeta, N, K, r, judge_basis, item_basis, use_order, i_obs, j_obs, z_obs)
    g_bar = g_t.mean(axis=0)
    emp_j = ((g_t - g_bar).T @ (g_t - g_bar)) / n_H
    hess_inv = np.linalg.pinv(hess, rcond=rcond)
    a_fit = np.atleast_1d(np.asarray(fit["a"], dtype=float))
    ell_h = human_nll_and_grad(fit["alpha_H"], a_fit, fit["mu"], np.asarray(fit["V"], dtype=float)[:, :a_fit.size], pair_arrays)[0] / n_H
    if n_H <= 1:
        raise ValueError("GACV requires at least two human comparisons")
    gacv = float(ell_h + tr_product(hess_inv, emp_j) / (n_H - 1.0))
    # Regularity (Assumption a2-gacv): the chart has exactly r(r+1) exact invariance directions
    # (factor rotations and mu-mixing), so the Hessian may have that many zero eigenvalues and no more.
    eig = np.linalg.eigvalsh(hess)
    n_null = int(np.sum(eig <= rcond * max(eig.max(), 1e-300)))
    regular = bool(fit["fit_info"].get("converged", True)) and n_null <= r * (r + 1) and float(np.max(np.abs(fit["s_H"]))) < 25.0
    return {
        "lam": lam,
        "gacv": gacv,
        "ell_H": float(ell_h),
        "q_lambda": float(q_val),
        "trace_term": float(tr_product(hess_inv, emp_j)),
        "hess_null_dim": n_null,
        "regular": regular,
    }


def endpoint_gacv(X, s_hat, i_obs, j_obs, z_obs, rcond=1e-8):
    """GACV for a fixed-design logistic fit with per-observation design rows X (n_H, d).

    Used for the endpoints: lambda = 0 (X = centered item-difference rows, d = N - 1) and
    lambda = inf (X = W_i - W_j with W fixed at the LLM-only fit, d = r + 1).
    Returns the criterion and a regularity flag (finite fit, nonsingular Hessian).
    """
    n_H = float(z_obs.size)
    d = s_hat[i_obs] - s_hat[j_obs]
    p = expit(d)
    H = (X * (p * (1 - p))[:, None]).T @ X / n_H
    G = X * (p - z_obs)[:, None]
    J = (G - G.mean(0)).T @ (G - G.mean(0)) / n_H
    ell = float(np.mean(np.logaddexp(0.0, d) - z_obs * d))
    eig = np.linalg.eigvalsh(H)
    regular = bool(eig.min() > rcond * max(eig.max(), 1e-300)) and float(np.max(np.abs(s_hat))) < 25.0
    return {"gacv": ell + float(np.trace(np.linalg.pinv(H, rcond=rcond) @ J)) / (n_H - 1), "ell_H": ell, "regular": regular}


def tr_product(hess_inv, emp_j):
    return float(np.trace(hess_inv @ emp_j))


def btl_mle_exists(N, pairs):
    """Existence of the (centered) BTL maximum-likelihood estimate.

    Ford (1957) / Zermelo: the MLE exists and is finite iff the directed win
    graph, with an edge i -> j whenever i beat j at least once (a tie counts in
    both directions), is strongly connected over all N items. When it fails the
    human-only endpoint is a separated fit whose training loss can be driven to
    zero, so it is not an admissible GACV candidate.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows, cols = [], []
    for i, j, n, y in pairs:
        if y > 0:
            rows.append(int(i))
            cols.append(int(j))
        if n - y > 0:
            rows.append(int(j))
            cols.append(int(i))
    if not rows:
        return False
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    n_components, _ = connected_components(graph, directed=True, connection="strong")
    return bool(n_components == 1)


def select_lambda(
    N,
    K,
    r,
    human_pairs,
    n_ijk_llm=None,
    y_ijk_llm=None,
    n_order=None,
    y_order=None,
    human_records=None,
    lambda_grid=None,
    max_steps=30,
    tol=1e-6,
    tau=1.0,
    inner_maxiter=100,
    with_uq=False,
    uq_alpha=0.05,
    warm_start=True,
    init_params=None,
    include_endpoints=True,
    guard=True,
    staged_fit=None,
    return_all=False,
    endpoint_existence_check=True,
    max_abs_score=10.0,
    align="W",
):
    """Fit DIAL at each lambda in the grid (plus the endpoints 0 and inf) and return the GACV minimizer.

    `align` ("W" or "mu") is passed to `joint`; with "mu" the human score is aligned to the
    consensus only while V is still estimated. A supplied `staged_fit` must use the same
    alignment (its `a` has length r for "W" and 0 for "mu").

    The grid is traversed from the largest lambda downward; the first fit is
    initialized from the LLM-only staged fit (computed here unless `staged_fit`
    is passed) and each subsequent fit from the previous one (plan Algorithm 1).
    With `guard`, candidates whose fit is non-convergent or whose criterion
    Hessian is singular beyond the chart's exact invariances are excluded from
    the argmin (Assumption a2-gacv) and listed in `dropped`.
    Every candidate, endpoints included, is marked irregular when its calibrated
    score exceeds `max_abs_score` in absolute value (a finite-fit bound: a
    centered log-odds of 10 is a pairwise probability above 0.99999, which no
    human comparison sample of this size supports; this catches nearly separated
    fits whose MLE exists but is unstable).
    With `endpoint_existence_check` (default), the lam = 0 endpoint is also
    marked irregular when the human-only BTL MLE does not exist
    (`btl_mle_exists`), i.e. the human comparison graph is not strongly
    connected and the human-only fit is separated. The flag is returned as
    `human_only_exists`.
    When every candidate is irregular the selector returns the largest lambda (the
    lam = inf endpoint when included) rather than the raw GACV minimizer.
    The returned dict has `lam` in [0, inf]; for lam = 0 it carries the human-only
    score and for lam = inf the staged calibrated score.
    """
    from .baselines import fit_human_only_btl, fit_staged_structured_calibration
    from .dial_model import calibration_design

    use_order = n_order is not None
    if use_order and n_ijk_llm is None:
        n_ijk_llm, y_ijk_llm = collapse_order_counts(n_order, y_order)
    n_L = total_llm_n(n_ijk_llm, n_order=n_order)
    n_H_pairs = total_human_n(human_pairs)
    if lambda_grid is None:
        lambda_grid = default_lambda_grid(n_L, n_H_pairs)
    lambda_grid = sorted(float(l) for l in lambda_grid)
    if warm_start:
        lambda_grid = lambda_grid[::-1]

    if human_records is not None:
        i_obs, j_obs, z_obs = observations_from_records(human_records)
    else:
        i_obs, j_obs, z_obs = observations_from_pairs(human_pairs)

    if staged_fit is None:
        if align == "mu":
            from .baselines import fit_consensus_only_calibrated

            staged_fit = fit_consensus_only_calibrated(N, K, r, n_ijk_llm, y_ijk_llm, human_pairs, n_order=n_order, y_order=y_order)
        else:
            staged_fit = fit_staged_structured_calibration(N, K, r, n_ijk_llm, y_ijk_llm, human_pairs, n_order=n_order, y_order=y_order)
    a_staged = np.atleast_1d(np.asarray(staged_fit["a"], dtype=float))
    if r > 0 and (align == "mu") != (a_staged.size == 0):
        raise ValueError("staged_fit alignment does not match `align`")
    if init_params is None:
        init_params = (staged_fit["gamma"], staged_fit["mu"], staged_fit["U"], staged_fit["V"], staged_fit["b"])

    candidates = []  # (lam, gacv, regular, fit)
    if include_endpoints:
        W = calibration_design(staged_fit["mu"], np.asarray(staged_fit["V"], dtype=float)[:, :a_staged.size])
        e = endpoint_gacv(W[i_obs] - W[j_obs], staged_fit["s_H"], i_obs, j_obs, z_obs)
        e["regular"] = e["regular"] and bool(staged_fit["fit_info"].get("converged", True))
        candidates.append((np.inf, e["gacv"], e["regular"], {**staged_fit, "lam": np.inf}))

    current_init = init_params
    for lam in lambda_grid:
        fit = joint(
            N, K, r, human_pairs,
            n_ijk_llm=n_ijk_llm, y_ijk_llm=y_ijk_llm,
            n_order=n_order, y_order=y_order,
            lam=float(lam),
            max_steps=max_steps, tol=tol, tau=tau, inner_maxiter=inner_maxiter,
            with_uq=False, init_params=current_init, align=align,
        )
        if warm_start:
            current_init = (fit["gamma"], fit["mu"], fit["U"], fit["V"], fit["b"])
        scores = gacv_for_fit(fit, N, K, r, n_ijk_llm, y_ijk_llm, n_order, y_order, human_pairs, i_obs, j_obs, z_obs)
        candidates.append((float(lam), scores["gacv"], scores["regular"], fit))

    if include_endpoints:
        h = fit_human_only_btl(N, human_pairs)
        B = make_centering_basis(N)
        e = endpoint_gacv(B[i_obs] - B[j_obs], h["s_H"], i_obs, j_obs, z_obs)
        human_only_exists = btl_mle_exists(N, human_pairs)
        if endpoint_existence_check:
            e["regular"] = e["regular"] and human_only_exists
        candidates.append((0.0, e["gacv"], e["regular"], {**h, "lam": 0.0, "b": np.zeros(K), "mu": None, "V": None}))

    if max_abs_score is not None:
        candidates = [(c[0], c[1], c[2] and bool(np.max(np.abs(np.asarray(c[3]["s_H"], dtype=float))) <= max_abs_score), c[3]) for c in candidates]
    path = [{"lam": c[0], "gacv": c[1], "regular": c[2]} for c in candidates]
    admissible = [c for c in candidates if (c[2] or not guard)]
    if not admissible:
        # Every candidate is irregular: the human sample is (nearly) separated at every
        # weight, so the guards give no basis for comparing GACV values. Fall back to the
        # most LLM-anchored candidate (lam = inf when the endpoints are included), whose
        # calibrated score is finite whenever the LLM-only fit is.
        admissible = [max(candidates, key=lambda c: c[0])]
    best = min(admissible, key=lambda c: c[1])
    selected = dict(best[3])
    selected["lam"] = best[0]
    selected["gacv"] = best[1]
    selected["gacv_path"] = path
    selected["dropped"] = [c[0] for c in candidates if guard and not c[2]]
    selected["human_only_exists"] = bool(human_only_exists) if include_endpoints else None
    if return_all:
        selected["candidates"] = [(c[0], c[3]) for c in candidates]
    if with_uq and np.isfinite(best[0]) and best[0] > 0:
        from .dial_model import calibrated_score_uq

        a_sel = np.atleast_1d(np.asarray(selected["a"], dtype=float))
        selected["uq"] = calibrated_score_uq(selected["alpha_H"], a_sel, selected["mu"], np.asarray(selected["V"], dtype=float)[:, :a_sel.size], human_pairs, alpha_level=uq_alpha)
    return selected
