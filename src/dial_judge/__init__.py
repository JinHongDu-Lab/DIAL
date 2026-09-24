"""
DIAL: Debiasing and human-preference Informed Alignment for LLM judges.

  - HJA (hja.py): S = gamma*mu^T + U V^T, optionally with position-effect BTL
    logit(p_kij^(a)) = S_ki - S_kj + a * b_k
  - DIAL (dial_model.py): weighted joint estimator
    ell_H(c; W) + lambda * ell_L(theta)
  - GACV (gacv.py): selects lambda from human comparisons

Metrics in evaluate.py: MSE, Spearman, NDCG, Coverage, Sign accuracy (human).
"""
from .benchmarks import fit_hja, fit_dial, fit_dial_joint, fit_dial_gacv, fit_ht_hja_joint
from .baselines import (
    fit_atc_btl,
    fit_consensus_only_calibrated,
    fit_human_only_btl,
    fit_llm_consensus,
    fit_pooled_btl,
    fit_staged_structured_calibration,
)
from .evaluate import (
    evaluate_scores,
    score_recovery_mse,
    spearman,
    ndcg,
    interval_coverage,
    sign_accuracy,
    human_sign_accuracy,
)

__all__ = [
    "fit_hja",
    "fit_dial",
    "fit_dial_joint",
    "fit_dial_gacv",
    "fit_ht_hja_joint",
    "fit_atc_btl",
    "fit_consensus_only_calibrated",
    "fit_human_only_btl",
    "fit_llm_consensus",
    "fit_pooled_btl",
    "fit_staged_structured_calibration",
    "evaluate_scores",
    "score_recovery_mse",
    "spearman",
    "ndcg",
    "interval_coverage",
    "sign_accuracy",
    "human_sign_accuracy",
]
