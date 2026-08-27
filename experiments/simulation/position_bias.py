"""Position-bias simulation study.

The initial implementation runs the two estimable structured comparisons. The
order-averaging estimators remain explicit student tasks in METHOD_STATUS.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from dial_judge.baselines import METHOD_STATUS
from dial_judge.benchmarks import fit_hja
from dial_judge.data import comparisons_to_aggregated, comparisons_to_order_aggregated
from dial_judge.simulate import (
    compute_score_matrix,
    generate_balanced_comparisons_with_order,
    generate_position_effects,
    generate_true_parameters,
)


STUDY_NAME = "position_bias"


def describe():
    return {
        "study": STUDY_NAME,
        "purpose": "Show why display order must be modeled explicitly.",
        "implemented_methods": ["dial_order_effect_btl", "ordinary_structured_btl"],
        "pending_methods": [
            "swap_probability_average",
            "paired_order_logit_average",
        ],
        "primary_metrics": ["position_effect_rmse", "score_mse", "debiased_probability_mse"],
    }


def _probability_mse(score_true, score_est):
    K, N = score_true.shape
    i, j = np.triu_indices(N, k=1)
    truth = expit(score_true[:, i] - score_true[:, j])
    estimate = expit(score_est[:, i] - score_est[:, j])
    return float(np.mean((truth - estimate) ** 2))


def run(smoke=False, seed=42):
    N, K, r = 4, 3, 1
    repetitions = 1 if smoke else 20
    budgets = [1200] if smoke else [600, 1200, 2400]
    bias_scales = [0.0, 0.35]
    records = []

    for bias_scale in bias_scales:
        for budget in budgets:
            for repetition in range(repetitions):
                run_seed = int(seed + 10000 * repetition + 100 * budget + round(100 * bias_scale))
                mu, gamma, U, V = generate_true_parameters(N, K, r, random_seed=run_seed)
                score_true = compute_score_matrix(mu, gamma, U, V)
                b_true = generate_position_effects(K, scale=bias_scale, random_seed=run_seed + 1)
                comparisons = generate_balanced_comparisons_with_order(
                    score_true, b_true, budget, random_seed=run_seed + 2
                )
                n_ijk, y_ijk = comparisons_to_aggregated(comparisons, N, K)
                n_order, y_order = comparisons_to_order_aggregated(comparisons, N, K)

                common = dict(max_steps=120, tol=1e-4, inner_maxiter=200)
                fits = {
                    "dial_order_effect_btl": fit_hja(
                        N,
                        K,
                        r,
                        n_ijk,
                        y_ijk,
                        n_order=n_order,
                        y_order=y_order,
                        **common,
                    ),
                    "ordinary_structured_btl": fit_hja(N, K, r, n_ijk, y_ijk, **common),
                }
                for method, fit in fits.items():
                    entry = {
                        "study": STUDY_NAME,
                        "method": method,
                        "seed": run_seed,
                        "repetition": repetition,
                        "n_llm": budget,
                        "bias_scale": bias_scale,
                        "score_mse": float(np.mean((fit["S"] - score_true) ** 2)),
                        "debiased_probability_mse": _probability_mse(score_true, fit["S"]),
                        "converged": bool(fit["fit_info"]["converged"]),
                    }
                    entry["position_effect_rmse"] = (
                        float(np.sqrt(np.mean((fit["b"] - b_true) ** 2)))
                        if method == "dial_order_effect_btl"
                        else None
                    )
                    records.append(entry)

    return {
        "description": describe(),
        "method_status": METHOD_STATUS,
        "records": records,
    }
