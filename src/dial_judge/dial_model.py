"""
DIAL joint estimator (paper Section 5.1, Eq. 14).

Position-debiased LLM scores follow the HJA factorization S = gamma*mu^T + U V^T
inside the order-effect BTL

    logit(p_kij^(a)) = S_ki - S_kj + a * b_k

Human calibration uses s_cal = W c with W = [mu, V] and c = (alpha_H, a):

    logit(p_ij^H) = alpha_H (mu_i - mu_j) + a^T (V_i - V_j)

The weighted joint criterion is the paper's normalized form

    ell_lambda(c, theta) = ell_H(c; W_theta) + lambda * ell_L(theta)

with ell_H = L_H / n_H and ell_L = L_L / n_L. Ordinary joint likelihood is
lambda = n_L / n_H. Adaptive choice of lambda is in gacv.py.

The two-stage / lambda=inf endpoint is not part of this module.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from .hja import (
    collapse_order_counts,
    initialize_parameters,
    make_centering_basis,
    negative_log_likelihood,
    negative_log_likelihood_and_grad,
    reanchor,
    reduce_item_block,
    reduce_judge_block,
    accept_block_step,
    boundary_activity,
    chart_bounds,
    count_at_bound,
    count_chart_at_bound,
    position_bounds,
    projected_gradient,
)


def pairs_to_arrays(pairs):
    if not pairs:
        raise ValueError("no human pairs supplied")
    i_idx = np.array([p[0] for p in pairs], dtype=int)
    j_idx = np.array([p[1] for p in pairs], dtype=int)
    n_arr = np.array([p[2] for p in pairs], dtype=float)
    y_arr = np.array([p[3] for p in pairs], dtype=float)
    return i_idx, j_idx, n_arr, y_arr


def total_human_n(pairs):
    return float(np.sum(np.array([p[2] for p in pairs], dtype=float)))


def total_llm_n(n_ijk, n_order=None):
    if n_order is not None:
        return float(np.sum(n_order))
    return float(np.sum(n_ijk))


def calibration_design(mu, V):
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.size == 0 or V.shape[1] == 0:
        return mu.reshape(-1, 1)
    return np.column_stack([mu, V])


def calibration_columns(V, a):
    """The columns of V that enter the calibration: the first a.size of them.

    Alignment convention: the calibration rank is the length of the coefficient
    vector `a` (r for align="W", 0 for align="mu"), while V keeps all r
    disagreement directions on the LLM side.
    """
    V = np.asarray(V, dtype=float)
    r_cal = int(np.atleast_1d(a).size)
    return V[:, :r_cal] if V.ndim == 2 else np.zeros((V.shape[0], 0))


def cal_score(alpha, a, mu, V):
    """s_cal = alpha mu + V[:, :a.size] a."""
    mu = np.asarray(mu, dtype=float)
    a = np.atleast_1d(np.asarray(a, dtype=float))
    return float(alpha) * mu + (calibration_columns(V, a) @ a if a.size else np.zeros_like(mu))


def human_nll_and_grad(alpha, a, mu, V, pair_arrays):
    i_idx, j_idx, n_arr, y_arr = pair_arrays
    a = np.atleast_1d(a)
    s = alpha * mu + (V @ a if V.shape[1] else np.zeros_like(mu))

    diff = s[i_idx] - s[j_idx]
    loss = float(np.sum(y_arr * np.logaddexp(0.0, -diff) + (n_arr - y_arr) * np.logaddexp(0.0, diff)))

    residual = n_arr * expit(diff) - y_arr
    grad_s = np.zeros_like(mu)
    np.add.at(grad_s, i_idx, residual)
    np.add.at(grad_s, j_idx, -residual)

    grad_alpha = float(grad_s @ mu)
    grad_a = V.T @ grad_s if V.shape[1] else np.zeros(0)
    grad_mu = alpha * grad_s
    grad_V = np.outer(grad_s, a) if V.shape[1] else np.zeros_like(V)
    return loss, grad_alpha, grad_a, grad_mu, grad_V


def fit_human_calibration(mu, V, pairs, initial=None, maxiter=1000, return_info=False):
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    r = V.shape[1]
    pair_arrays = pairs_to_arrays(pairs)

    if initial is None:
        x0 = np.zeros(1 + r, dtype=float)
        x0[0] = 1.0
    else:
        x0 = np.asarray(initial, dtype=float)

    def objective(x):
        alpha = x[0]
        a = x[1:]
        loss, grad_alpha, grad_a, _, _ = human_nll_and_grad(alpha, a, mu, V, pair_arrays)
        return loss, np.concatenate([[grad_alpha], grad_a])

    result = minimize(objective, x0=x0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "gtol": 1e-8})
    bounds = None
    if not result.success:
        # Line-search breakdown happens when the calibration likelihood is separated (alpha or a
        # runs off) or flat to machine precision. Retry on a box that keeps the calibrated score
        # within the finite-fit range |s| <= 25, and accept a point whose projected gradient is
        # small relative to the sample size; the GACV guards flag such fits downstream.
        n_h = float(np.sum(pair_arrays[2]))
        scale = np.concatenate([[np.max(np.abs(mu))], np.max(np.abs(V), axis=0) if r else []])
        bounds = [(-25.0 / max(sc, 1e-8), 25.0 / max(sc, 1e-8)) for sc in scale]
        result = minimize(objective, x0=np.clip(x0, [lo for lo, _ in bounds], [hi for _, hi in bounds]), method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": maxiter, "gtol": 1e-8})
        if not result.success:
            pg = np.linalg.norm(projected_gradient(result.x, result.jac, bounds)) / max(n_h, 1.0)
            if pg > 1e-6:
                raise RuntimeError(f"human calibration fit failed: {result.message} (projected gradient {pg:.2e})")
    if return_info:
        # the unconstrained fit has no box; the retry box bounds the calibrated score, not the chart
        info = {"cal_box_used": bool(bounds is not None), "max_abs_coord": float(np.max(np.abs(result.x)))}
        info["boundary_cal"] = bool(boundary_activity(result.x, bounds, {})["boundary_any"]) if bounds is not None else False
        return float(result.x[0]), result.x[1:].copy(), info
    return float(result.x[0]), result.x[1:].copy()


def calibrated_score_uq(alpha, a, mu, V, pairs, alpha_level=0.05, rcond=1e-8):
    """Wald intervals for s_cal = W c with (mu, V) treated as fixed (abundant-LLM)."""
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)
    a = np.atleast_1d(np.asarray(a, dtype=float))
    pair_arrays = pairs_to_arrays(pairs)
    i_idx, j_idx, n_arr, _y_arr = pair_arrays

    W = calibration_design(mu, V)
    c = np.concatenate([[float(alpha)], a]) if V.shape[1] else np.array([float(alpha)])
    s = W @ c

    diff = s[i_idx] - s[j_idx]
    weight = n_arr * expit(diff) * (1.0 - expit(diff))
    X = W[i_idx] - W[j_idx]
    hessian_c = X.T @ (X * weight[:, None])
    cov_c = np.linalg.pinv(hessian_c, rcond=rcond)
    cov_s = W @ cov_c @ W.T

    z_value = float(norm.ppf(1.0 - alpha_level / 2.0))
    se = np.sqrt(np.maximum(np.diag(cov_s), 0.0))
    return {
        "s_H": s,
        "se": se,
        "lower": s - z_value * se,
        "upper": s + z_value * se,
        "covariance": cov_s,
        "cov_c": cov_c,
        "hessian_c": hessian_c,
        "z_value": z_value,
        "alpha": float(alpha_level),
    }


def polish_joint_fit(gamma, mu, U, V, b, alpha, a, n_ijk, y_ijk, n_order, y_order, pair_arrays, lam, n_H, n_L, maxiter=2000, gtol=1e-8):
    """
    L-BFGS on the weighted criterion q_lambda over the centering-reduced factor
    chart (gamma_red, U_red, [b], mu_red, V_red, alpha, a), followed by ReAnchor
    and re-expression of (alpha, a) in the anchored basis. The chart's exact
    invariance directions leave q_lambda flat and are harmless for L-BFGS.
    """
    from .gacv import pack_reduced, q_lambda_grad, unpack_reduced  # local import: gacv imports this module

    use_order = n_order is not None
    K, N, r = gamma.size, mu.size, U.shape[1]
    jb, ib = make_centering_basis(K), make_centering_basis(N)
    z0 = pack_reduced(gamma, mu, U, V, b, alpha, a, use_order)

    def fg(z):
        return q_lambda_grad(z, N, K, r, jb, ib, use_order, n_ijk, y_ijk, n_order, y_order, pair_arrays, lam, n_H, n_L)

    bounds = chart_bounds(z0.size)
    res = minimize(fg, z0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15})
    gamma, mu, U, V, b, alpha, a = unpack_reduced(res.x, N, K, r, jb, ib, use_order)
    a = np.atleast_1d(a)
    if r > 0:
        target = cal_score(alpha, a, mu, V)
        gamma, mu, U, V = reanchor(gamma, mu, U, V)
        coef, *_ = np.linalg.lstsq(calibration_design(mu, calibration_columns(V, a)), target, rcond=None)
        alpha, a = float(coef[0]), coef[1:]
    grad_norm = float(np.linalg.norm(projected_gradient(res.x, res.jac, bounds)))
    S = np.outer(gamma, mu) + U @ V.T
    info = {"polish_nit": int(res.nit), "polish_grad_norm": grad_norm, "polish_converged": bool(grad_norm <= 1e-5), "total_loss": float(res.fun),
            "b_at_bound": count_at_bound(b) if use_order else 0, "n_at_bound": count_chart_at_bound(res.x, bounds),
            "max_abs_S": float(np.max(np.abs(S))), "max_gamma": float(np.max(np.abs(gamma)))}
    # chart layout of pack_reduced: gamma_red, U_red, [b], mu_red, V_red, alpha, a
    n_judge, n_b = (K - 1) * (r + 1), (K if use_order else 0)
    n_cal = 1 + np.atleast_1d(a).size
    idx = np.arange(res.x.size)
    info.update(boundary_activity(res.x, bounds, {"b": idx[n_judge: n_judge + n_b], "llm_other": np.concatenate([idx[:n_judge], idx[n_judge + n_b: idx.size - n_cal]]),
                                                  "cal": idx[idx.size - n_cal:]}))
    return gamma, mu, U, V, (b if use_order else np.zeros(K)), alpha, a, info


def default_lambda(n_L, n_H):
    if n_H <= 0:
        raise ValueError("n_H must be positive")
    return float(n_L / n_H)


def joint(
    N,
    K,
    r,
    human_pairs,
    n_ijk_llm=None,
    y_ijk_llm=None,
    n_order=None,
    y_order=None,
    lam=None,
    max_steps=30,
    tol=1e-6,
    tau=1.0,
    inner_maxiter=100,
    with_uq=False,
    uq_alpha=0.05,
    init_params=None,
    polish=True,
    polish_gtol=1e-8,
    align="W",
):
    """Minimize ell_H + lambda * ell_L by anchored alternating MLE, then L-BFGS polish.

    `align="W"` calibrates within W = [mu, V] (coefficients (alpha, a), a of length r);
    `align="mu"` aligns the human score to the consensus only, s_cal = alpha mu, while V
    is still estimated on the LLM side (a has length 0). The two coincide when r = 0.

    Judge block: (gamma, U, b) from the LLM order-effect likelihood.
    Item + human block: (mu, V, alpha_H, a) from the weighted joint criterion.
    `init_params`, if given, is (gamma, mu, U, V, b) overriding the default
    initialization (used, e.g., to check invariance to the factor basis).
    """
    if n_order is not None:
        if n_ijk_llm is None:
            n_ijk_llm, y_ijk_llm = collapse_order_counts(n_order, y_order)
        use_order = True
    else:
        if n_ijk_llm is None or y_ijk_llm is None:
            raise ValueError("joint requires n_order/y_order or n_ijk_llm/y_ijk_llm")
        use_order = False

    n_L = total_llm_n(n_ijk_llm, n_order=n_order)
    n_H = total_human_n(human_pairs)
    if n_L <= 0 or n_H <= 0:
        raise ValueError("joint requires positive LLM and human comparison counts")
    if lam is None:
        lam = default_lambda(n_L, n_H)
    lam = float(lam)
    if lam <= 0:
        raise ValueError(f"lambda must be positive, got {lam}")

    if init_params is None:
        gamma, mu, U, V, b = initialize_parameters(N, K, r, n_ijk_llm, y_ijk_llm, n_order=n_order, y_order=y_order)
    else:
        gamma, mu, U, V, b = (np.array(p, dtype=float, copy=True) for p in init_params)
    if align not in ("W", "mu"):
        raise ValueError(f"align must be 'W' or 'mu', got {align!r}")
    r_cal = r if align == "W" else 0
    alpha, a = fit_human_calibration(mu, V[:, :r_cal], human_pairs)
    a = np.atleast_1d(np.asarray(a, dtype=float))

    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    pair_arrays = pairs_to_arrays(human_pairs)
    n_gamma = K - 1
    n_U = (K - 1) * r

    def weighted_loss(gamma, mu, U, V, b, alpha, a):
        l_l = negative_log_likelihood(mu, gamma, U, V, n_ijk_llm, y_ijk_llm, b=b, n_order=n_order, y_order=y_order)
        l_h, _, _, _, _ = human_nll_and_grad(alpha, a, mu, V[:, :r_cal], pair_arrays)
        return (l_h / n_H) + lam * (l_l / n_L), l_l, l_h

    history = []
    inner_limit_hits = 0
    for step in range(max_steps):
        step_index = step + 1
        current_gamma_reduced, current_U_reduced = reduce_judge_block(gamma, U)
        judge_parts = [current_gamma_reduced, current_U_reduced.ravel()]
        if use_order:
            judge_parts.append(b)
        current_judge_block = np.concatenate(judge_parts)
        current_mu_reduced, current_V_reduced = reduce_item_block(mu, V)
        current_item_block = np.concatenate([current_mu_reduced, current_V_reduced.ravel(), [alpha], a])

        prev_total, _, _ = weighted_loss(gamma, mu, U, V, b, alpha, a)

        def objective_judge(block):
            gamma_new = np.ones(K, dtype=float) + judge_basis @ block[:n_gamma]
            U_new = judge_basis @ block[n_gamma: n_gamma + n_U].reshape(K - 1, r)
            b_new = block[n_gamma + n_U:] if use_order else None
            loss, _, grad_gamma, grad_U, _, grad_b = negative_log_likelihood_and_grad(
                mu, gamma_new, U_new, V, n_ijk_llm, y_ijk_llm, b=b_new, n_order=n_order, y_order=y_order
            )
            # Judge parameters appear only in ell_L; lambda / n_L is a common scale.
            scale = lam / n_L
            objective_value = scale * loss + 0.5 * tau * np.sum((block - current_judge_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[:n_gamma] = scale * (judge_basis.T @ grad_gamma) + tau * (block[:n_gamma] - current_gamma_reduced)
            grad_reduced[n_gamma: n_gamma + n_U] = (
                scale * (judge_basis.T @ grad_U).ravel() + tau * (block[n_gamma: n_gamma + n_U] - current_U_reduced.ravel())
            )
            if use_order:
                grad_reduced[n_gamma + n_U:] = scale * grad_b + tau * (block[n_gamma + n_U:] - b)
            return objective_value, grad_reduced

        result_j = minimize(
            objective_judge,
            x0=current_judge_block,
            method="L-BFGS-B",
            jac=True,
            bounds=chart_bounds(current_judge_block.size),
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        inner_limit_hits += accept_block_step(result_j, objective_judge(current_judge_block)[0], "joint judge-block update")
        gamma_tilde = np.ones(K, dtype=float) + judge_basis @ result_j.x[:n_gamma]
        U_tilde = judge_basis @ result_j.x[n_gamma: n_gamma + n_U].reshape(K - 1, r)
        b_tilde = result_j.x[n_gamma + n_U:].copy() if use_order else np.zeros(K, dtype=float)

        def objective_item(block):
            mu_new = item_basis @ block[: N - 1]
            V_new = item_basis @ block[N - 1: N - 1 + (N - 1) * r].reshape(N - 1, r)
            alpha_new = block[N - 1 + (N - 1) * r]
            a_new = block[N - 1 + (N - 1) * r + 1:]

            l_l, grad_mu_l, _, _, grad_V_l, _ = negative_log_likelihood_and_grad(
                mu_new, gamma_tilde, U_tilde, V_new, n_ijk_llm, y_ijk_llm, b=b_tilde, n_order=n_order, y_order=y_order
            )
            l_h, grad_alpha_h, grad_a_h, grad_mu_h, grad_V_h_cal = human_nll_and_grad(
                alpha_new, a_new, mu_new, V_new[:, :r_cal], pair_arrays
            )
            grad_V_h = np.zeros_like(V_new)
            grad_V_h[:, :r_cal] = grad_V_h_cal
            grad_mu = grad_mu_h / n_H + lam * grad_mu_l / n_L
            grad_V = grad_V_h / n_H + lam * grad_V_l / n_L
            objective_value = (l_h / n_H) + lam * (l_l / n_L) + 0.5 * tau * np.sum((block - current_item_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[: N - 1] = item_basis.T @ grad_mu + tau * (block[: N - 1] - current_mu_reduced)
            grad_reduced[N - 1: N - 1 + (N - 1) * r] = (
                (item_basis.T @ grad_V).ravel()
                + tau * (block[N - 1: N - 1 + (N - 1) * r] - current_V_reduced.ravel())
            )
            grad_reduced[N - 1 + (N - 1) * r] = grad_alpha_h / n_H + tau * (block[N - 1 + (N - 1) * r] - alpha)
            grad_reduced[N - 1 + (N - 1) * r + 1:] = grad_a_h / n_H + tau * (block[N - 1 + (N - 1) * r + 1:] - a)
            return objective_value, grad_reduced

        result_i = minimize(
            objective_item,
            x0=current_item_block,
            method="L-BFGS-B",
            jac=True,
            bounds=chart_bounds(current_item_block.size),
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        inner_limit_hits += accept_block_step(result_i, objective_item(current_item_block)[0], "joint item-block update")
        mu_tilde = item_basis @ result_i.x[: N - 1]
        V_tilde = item_basis @ result_i.x[N - 1: N - 1 + (N - 1) * r].reshape(N - 1, r)
        alpha_tilde = float(result_i.x[N - 1 + (N - 1) * r])
        a_tilde = result_i.x[N - 1 + (N - 1) * r + 1:].copy()

        gamma_new, mu_new, U_new, V_new = reanchor(gamma_tilde, mu_tilde, U_tilde, V_tilde)
        if r > 0:
            target = cal_score(alpha_tilde, a_tilde, mu_tilde, V_tilde)
            design = calibration_design(mu_new, calibration_columns(V_new, a_tilde))
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            alpha_new, a_new = float(coef[0]), coef[1:]
        else:
            alpha_new, a_new = alpha_tilde, a_tilde
        b_new = b_tilde

        new_total, l_l_new, l_h_new = weighted_loss(gamma_new, mu_new, U_new, V_new, b_new, alpha_new, a_new)
        rel_change = abs(new_total - prev_total) / (1.0 + abs(prev_total))
        history.append(
            {
                "iteration": step_index,
                "total_loss": float(new_total),
                "ell_L": float(l_l_new / n_L),
                "ell_H": float(l_h_new / n_H),
                "rel_change": float(rel_change),
            }
        )
        gamma, mu, U, V, b, alpha, a = gamma_new, mu_new, U_new, V_new, b_new, alpha_new, a_new

        if rel_change < tol and step_index > 1:
            break

    alternating_converged = bool(history and history[-1]["rel_change"] < tol and history[-1]["iteration"] > 1)
    polish_info = {}
    if polish:
        gamma, mu, U, V, b, alpha, a, polish_info = polish_joint_fit(
            gamma, mu, U, V, b, alpha, a, n_ijk_llm, y_ijk_llm, n_order, y_order, pair_arrays, lam, n_H, n_L, gtol=polish_gtol
        )

    a = np.atleast_1d(np.asarray(a, dtype=float))
    s_hat = cal_score(alpha, a, mu, V)
    result = {
        "gamma": gamma,
        "mu": mu,
        "U": U,
        "V": V,
        "b": b,
        "alpha_H": alpha,
        "a": a,
        "align": align,
        "s_H": s_hat,
        "W": calibration_design(mu, calibration_columns(V, a)),
        "lam": lam,
        "n_L": n_L,
        "n_H": n_H,
        "fit_info": {
            "n_iter": history[-1]["iteration"] if history else 0,
            "alternating_converged": alternating_converged,
            "converged": bool(polish_info["polish_converged"]) if polish else alternating_converged,
            "history": history,
            "total_loss": float(polish_info["total_loss"]) if polish else (float(history[-1]["total_loss"]) if history else None),
            "inner_limit_hits": int(inner_limit_hits),
            "b_at_bound": count_at_bound(b) if use_order else 0,
            **polish_info,
        },
    }
    if with_uq:
        result["uq"] = calibrated_score_uq(alpha, a, mu, V, human_pairs, alpha_level=uq_alpha)
    return result
