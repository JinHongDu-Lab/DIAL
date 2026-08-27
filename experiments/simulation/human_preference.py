"""Human-preference simulation specification and implementation status."""

from __future__ import annotations

from dial_judge.baselines import METHOD_STATUS


STUDY_NAME = "human_preference"


def describe():
    return {
        "study": STUDY_NAME,
        "purpose": "Compare human-preference estimators under the well-specified DIAL DGP.",
        "rows": ["no_position_bias", "mild_position_bias"],
        "x_axis": "number_of_llm_comparisons_at_fixed_human_budget",
        "methods": [
            "human_only_btl",
            "llm_consensus",
            "consensus_only_calibrated",
            "staged_structured_calibration",
            "dial_gacv",
            "unstructured_btl_svd_calibration",
            "atc",
        ],
        "method_status": METHOD_STATUS,
        "primary_metrics": ["mse", "heldout_log_loss", "spearman", "ndcg", "sign_accuracy"],
        "implementation": "scaffold",
    }


def run(smoke=False, seed=42):
    return {
        "description": describe(),
        "seed": int(seed),
        "smoke": bool(smoke),
        "status": "scaffold",
        "records": [],
        "next_task": "Implement the shared fit/evaluate repetition using the registered baseline wrappers.",
    }
