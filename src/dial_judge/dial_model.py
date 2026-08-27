"""
DIAL joint estimator (paper Section 5.1, Eq. 14).

Position-debiased LLM scores follow the HJA factorization S = gamma*mu^T + U V^T
inside the order-effect BTL

    logit(p_kij^(a)) = S_ki - S_kj + a * b_k

Human calibration uses s_cal = W c with W = [mu, V] and c = (alpha_H, a):

    logit(p_ij^H) = alpha_H (mu_i - mu_j) + a^T (V_i - V_j)

The weighted joint criterion is the paper's normalized form

    ell_lambda(c, theta) = ell_H(c; W_theta) + lambda * ell_L(theta)

with ell_H = L_H / n_0 and ell_L = L_L / n_L. Ordinary joint likelihood is
lambda = n_L / n_0. Adaptive choice of lambda is in gacv.py.

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


def fit_human_calibration(mu, V, pairs, initial=None, maxiter=1000):
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
    if not result.success:
        raise RuntimeError(f"human calibration fit failed: {result.message}")
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


def default_lambda(n_L, n_0):
    if n_0 <= 0:
        raise ValueError("n_0 must be positive")
    return float(n_L / n_0)


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
    max_steps=150,
    tol=1e-6,
    tau=10.0,
    inner_maxiter=500,
    with_uq=False,
    uq_alpha=0.05,
):
    """Minimize ell_H + lambda * ell_L by anchored alternating MLE.

    Judge block: (gamma, U, b) from the LLM order-effect likelihood.
    Item + human block: (mu, V, alpha_H, a) from the weighted joint criterion.
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
    n_0 = total_human_n(human_pairs)
    if n_L <= 0 or n_0 <= 0:
        raise ValueError("joint requires positive LLM and human comparison counts")
    if lam is None:
        lam = default_lambda(n_L, n_0)
    lam = float(lam)
    if lam <= 0:
        raise ValueError(f"lambda must be positive, got {lam}")

    gamma, mu, U, V, b = initialize_parameters(N, K, r, n_ijk_llm, y_ijk_llm, n_order=n_order, y_order=y_order)
    alpha, a = fit_human_calibration(mu, V, human_pairs)

    judge_basis = make_centering_basis(K)
    item_basis = make_centering_basis(N)
    pair_arrays = pairs_to_arrays(human_pairs)
    n_gamma = K - 1
    n_U = (K - 1) * r

    def weighted_loss(gamma, mu, U, V, b, alpha, a):
        l_l = negative_log_likelihood(mu, gamma, U, V, n_ijk_llm, y_ijk_llm, b=b, n_order=n_order, y_order=y_order)
        l_h, _, _, _, _ = human_nll_and_grad(alpha, a, mu, V, pair_arrays)
        return (l_h / n_0) + lam * (l_l / n_L), l_l, l_h

    history = []
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
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        if not result_j.success:
            raise RuntimeError(f"joint judge-block update failed: {result_j.message}")
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
            l_h, grad_alpha_h, grad_a_h, grad_mu_h, grad_V_h = human_nll_and_grad(
                alpha_new, a_new, mu_new, V_new, pair_arrays
            )
            grad_mu = grad_mu_h / n_0 + lam * grad_mu_l / n_L
            grad_V = grad_V_h / n_0 + lam * grad_V_l / n_L
            objective_value = (l_h / n_0) + lam * (l_l / n_L) + 0.5 * tau * np.sum((block - current_item_block) ** 2)
            grad_reduced = np.empty_like(block)
            grad_reduced[: N - 1] = item_basis.T @ grad_mu + tau * (block[: N - 1] - current_mu_reduced)
            grad_reduced[N - 1: N - 1 + (N - 1) * r] = (
                (item_basis.T @ grad_V).ravel()
                + tau * (block[N - 1: N - 1 + (N - 1) * r] - current_V_reduced.ravel())
            )
            grad_reduced[N - 1 + (N - 1) * r] = grad_alpha_h / n_0 + tau * (block[N - 1 + (N - 1) * r] - alpha)
            grad_reduced[N - 1 + (N - 1) * r + 1:] = grad_a_h / n_0 + tau * (block[N - 1 + (N - 1) * r + 1:] - a)
            return objective_value, grad_reduced

        result_i = minimize(
            objective_item,
            x0=current_item_block,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": inner_maxiter, "gtol": 1e-5, "maxls": 50},
        )
        if not result_i.success:
            raise RuntimeError(f"joint item-block update failed: {result_i.message}")
        mu_tilde = item_basis @ result_i.x[: N - 1]
        V_tilde = item_basis @ result_i.x[N - 1: N - 1 + (N - 1) * r].reshape(N - 1, r)
        alpha_tilde = float(result_i.x[N - 1 + (N - 1) * r])
        a_tilde = result_i.x[N - 1 + (N - 1) * r + 1:].copy()

        gamma_new, mu_new, U_new, V_new = reanchor(gamma_tilde, mu_tilde, U_tilde, V_tilde)
        if r > 0:
            target = alpha_tilde * mu_tilde + V_tilde @ a_tilde
            design = np.column_stack([mu_new, V_new])
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
                "ell_H": float(l_h_new / n_0),
                "rel_change": float(rel_change),
            }
        )
        gamma, mu, U, V, b, alpha, a = gamma_new, mu_new, U_new, V_new, b_new, alpha_new, a_new

        if rel_change < tol and step_index > 1:
            break

    s_hat = alpha * mu + (V @ a if V.shape[1] else np.zeros_like(mu))
    result = {
        "gamma": gamma,
        "mu": mu,
        "U": U,
        "V": V,
        "b": b,
        "alpha_H": alpha,
        "a": a,
        "s_H": s_hat,
        "W": calibration_design(mu, V),
        "lam": lam,
        "n_L": n_L,
        "n_0": n_0,
        "fit_info": {
            "n_iter": history[-1]["iteration"] if history else 0,
            "converged": bool(history and history[-1]["rel_change"] < tol and history[-1]["iteration"] > 1),
            "history": history,
            "total_loss": float(history[-1]["total_loss"]) if history else None,
        },
    }
    if with_uq:
        result["uq"] = calibrated_score_uq(alpha, a, mu, V, human_pairs, alpha_level=uq_alpha)
    return result
