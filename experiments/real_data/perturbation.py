"""Semi-synthetic perturbation study specification."""

from .prepare import describe as describe_adapter


def describe(dataset):
    return {
        "study": "perturbation",
        "dataset": dataset,
        "perturbations": [
            "human_subsample",
            "llm_subsample",
            "label_flip_noise",
            "judge_removal",
            "display_order_imbalance",
        ],
        "evaluation": "held_out_human_comparisons",
        "adapter": describe_adapter(),
        "status": "superseded by experiments/real_data/robustness.py (--study robustness)",
    }
