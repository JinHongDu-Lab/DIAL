"""Comparison records and their aggregated counts.

An LLM record is (k, i, j, y) or (k, i, j, y, a): judge k, canonical pair i < j, outcome
y in {0, 0.5, 1} (1 when item i wins), and display order a in {-1, 1} (+1 when item i is
shown first). A pooled human pair is (i, j, n, y_sum).
"""
import numpy as np


def records_to_array(records, min_fields=4):
    """(n_rec, n_fields) float view of a record list, validating i < j.

    `np.asarray` on a list of equal-length tuples converts in C, so this is the cheap way
    into array form; records must therefore all carry the same number of fields.
    """
    try:
        arr = np.asarray(records, dtype=float)
    except ValueError:                                  # ragged: 4- and 5-field records mixed
        arr = np.array([rec[:min_fields] for rec in records], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < min_fields:
        raise ValueError(f"records must be a rectangular list of >= {min_fields} fields, got shape {arr.shape}")
    bad = np.flatnonzero(arr[:, 1] >= arr[:, 2])
    if bad.size:
        i, j = arr[bad[0], 1], arr[bad[0], 2]
        raise ValueError(f"comparison indices must satisfy i < j, got {(int(i), int(j))}")
    return arr


def comparisons_to_aggregated(comparisons, N, K):
    # comparisons: (k, i, j, y) or (k, i, j, y, a) with i < j, y in {0, 0.5, 1}.
    # Order, if present, is collapsed. Returns n_ijk, y_ijk as (K, N, N).
    n_ijk = np.zeros((K, N, N), dtype=float)
    y_ijk = np.zeros((K, N, N), dtype=float)
    if not comparisons:
        return n_ijk, y_ijk
    arr = records_to_array(comparisons, 4)
    idx = (arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int))
    np.add.at(n_ijk, idx, 1.0)
    np.add.at(y_ijk, idx, arr[:, 3])
    return n_ijk, y_ijk


def comparisons_to_order_aggregated(comparisons, N, K):
    # comparisons: (k, i, j, y, a) with a in {-1, 1}, i < j.
    # Returns n_order, y_order as (K, N, N, 2) with last axis a_idx:
    #   0 -> A=-1 (canonical i displayed second), 1 -> A=+1 (i displayed first).
    n_order = np.zeros((K, N, N, 2), dtype=float)
    y_order = np.zeros((K, N, N, 2), dtype=float)
    if not comparisons:
        return n_order, y_order
    try:
        arr = records_to_array(comparisons, 5)
    except ValueError as exc:
        raise ValueError("order aggregation requires records (k, i, j, y, a)") from exc
    idx = (arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2].astype(int), (arr[:, 4] >= 0).astype(int))
    np.add.at(n_order, idx, 1.0)
    np.add.at(y_order, idx, arr[:, 3])
    return n_order, y_order


def pool_pairs(records):
    # Collapse a list of (k, i, j, y[, a]) comparisons into (i, j, n, y_sum) pairs,
    # discarding the judge index and display order.
    if not records:
        return []
    arr = records_to_array(records, 4)
    i, j = arr[:, 1].astype(np.int64), arr[:, 2].astype(np.int64)
    stride = int(j.max()) + 1
    keys, first, inverse = np.unique(i * stride + j, return_index=True, return_inverse=True)
    n = np.bincount(inverse, minlength=keys.size)
    y = np.bincount(inverse, weights=arr[:, 3], minlength=keys.size)
    order = np.argsort(first)          # keep the pairs in order of first appearance
    return [(int(keys[o] // stride), int(keys[o] % stride), float(n[o]), float(y[o])) for o in order]
