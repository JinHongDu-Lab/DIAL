"""Analysis-ready data adapter contract.

Implementation is intentionally deferred until the large-data distribution
policy is settled. Collection files must not be silently copied or rewritten.
"""

CANONICAL_FIELDS = (
    "dataset",
    "record_id",
    "judge",
    "item_i",
    "item_j",
    "outcome",
    "display_order",
    "human_outcome",
)


def describe():
    return {
        "status": "scaffold",
        "canonical_fields": CANONICAL_FIELDS,
        "split_unit": "record_id",
        "supported_datasets": ("arena_33k", "mt_bench", "pandalm"),
    }
