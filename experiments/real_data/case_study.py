"""Full real-data case-study specification."""

from .prepare import describe as describe_adapter


def describe(dataset):
    return {
        "study": "case_study",
        "dataset": dataset,
        "outputs": [
            "judge_position_effects",
            "debiased_ranking_changes",
            "endpoint_comparison",
            "dial_human_ranking",
            "clustered_uncertainty",
        ],
        "adapter": describe_adapter(),
        "status": "scaffold",
    }
