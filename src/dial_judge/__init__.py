"""DIAL: position-debiased LLM judges with adaptive human-preference calibration.

  hja.py         structured LLM-panel model S = gamma mu^T + U V^T with the order-effect BTL
                 logit(p_kij^(a)) = S_ki - S_kj + a b_k
  dial_model.py  weighted joint estimator ell_H(c; W) + lambda ell_L(theta)
  gacv.py        selection of lambda from the human comparisons
  inference.py   sandwich and cluster-robust intervals, calibration-space test
  baselines.py, penalized.py, ppi.py   comparison estimators
"""
from .baselines import (
    fit_atc_btl,
    fit_consensus_only_calibrated,
    fit_human_only_btl,
    fit_pooled_btl,
    fit_staged_structured_calibration,
)
from .benchmarks import fit_dial, fit_hja
from .gacv import select_lambda

__all__ = [
    "fit_hja",
    "fit_dial",
    "select_lambda",
    "fit_atc_btl",
    "fit_consensus_only_calibrated",
    "fit_human_only_btl",
    "fit_pooled_btl",
    "fit_staged_structured_calibration",
]
