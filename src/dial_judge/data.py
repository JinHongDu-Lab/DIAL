"""
Data loading for DIAL: reads "chatbot_arena-JARanking-merged.csv", which
merges a 10k-row LLM-judge chatbot-arena dataset (10 distinct LLM judges)
with a ~33k-row human chatbot-arena preference dataset (thousands of
distinct human annotators), joined on question_id + identical
(model_a, model_b) pair. Only the intersection (rows with both an LLM-judge
label and a matching human label) is present in the merged file.

Self-contained: does not import from src/ or the original new-experiment/
flat scripts.
"""
import os

import numpy as np
import pandas as pd

DEFAULT_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "chatbot_arena-JARanking-merged.csv")
)

TIE_LABELS = {"tie", "tie (bothbad)"}


def comparisons_to_aggregated(comparisons, N, K):
    # comparisons: (k, i, j, y) or (k, i, j, y, a) with i < j, y in {0, 0.5, 1}.
    # Order, if present, is collapsed. Returns n_ijk, y_ijk as (K, N, N).
    n_ijk = np.zeros((K, N, N), dtype=float)
    y_ijk = np.zeros((K, N, N), dtype=float)
    for rec in comparisons:
        k, i, j, y = rec[0], rec[1], rec[2], rec[3]
        if i >= j:
            raise ValueError(f"comparison indices must satisfy i < j, got {(i, j)}")
        n_ijk[k, i, j] += 1.0
        y_ijk[k, i, j] += float(y)
    return n_ijk, y_ijk


def comparisons_to_order_aggregated(comparisons, N, K):
    # comparisons: (k, i, j, y, a) with a in {-1, 1}, i < j.
    # Returns n_order, y_order as (K, N, N, 2) with last axis a_idx:
    #   0 -> A=-1 (canonical i displayed second), 1 -> A=+1 (i displayed first).
    n_order = np.zeros((K, N, N, 2), dtype=float)
    y_order = np.zeros((K, N, N, 2), dtype=float)
    for rec in comparisons:
        if len(rec) < 5:
            raise ValueError("order aggregation requires records (k, i, j, y, a)")
        k, i, j, y, a = rec[0], rec[1], rec[2], rec[3], rec[4]
        if i >= j:
            raise ValueError(f"comparison indices must satisfy i < j, got {(i, j)}")
        a_idx = 0 if a < 0 else 1
        n_order[k, i, j, a_idx] += 1.0
        y_order[k, i, j, a_idx] += float(y)
    return n_order, y_order


def pool_pairs(records):
    # Collapse a list of (k, i, j, y[, a]) comparisons into (i, j, n, y_sum) pairs,
    # discarding the judge index and display order.
    pooled = {}
    for rec in records:
        i, j, y = rec[1], rec[2], rec[3]
        key = (i, j)
        n_prev, y_prev = pooled.get(key, (0.0, 0.0))
        pooled[key] = (n_prev + 1.0, y_prev + y)
    return [(i, j, n, y) for (i, j), (n, y) in pooled.items()]


def pool_as_single_judge(records):
    # Collapse all individual human annotators into one pooled "judge"
    # (K=1). Human annotators average ~1.7 comparisons each, so
    # per-annotator sensitivity/disagreement parameters are essentially
    # unidentified -- the human score model itself treats humans
    # as a single pooled signal, so this matches the intended model rather
    # than being merely a numerical convenience.
    return [(0, rec[1], rec[2], rec[3]) for rec in records]


def display_order_a(i_first, i_second):
    # Canonical pair is i < j. A = +1 if canonical i is displayed first.
    i, j = (i_first, i_second) if i_first < i_second else (i_second, i_first)
    a = 1 if i_first == i else -1
    return i, j, a


def _row_to_comparison(model_a, model_b, preferred, judge_key, item_to_idx, judge_to_idx, with_order=False):
    if pd.isna(preferred) or pd.isna(model_a) or pd.isna(model_b):
        return None
    i_a = item_to_idx[model_a]
    i_b = item_to_idx[model_b]
    if with_order:
        i, j, a = display_order_a(i_a, i_b)
    else:
        i, j = (i_a, i_b) if i_a < i_b else (i_b, i_a)
        a = None
    if preferred in TIE_LABELS:
        y = 0.5
    elif preferred == model_a:
        y = 1.0 if i_a == i else 0.0
    elif preferred == model_b:
        y = 1.0 if i_b == i else 0.0
    else:
        return None
    k = judge_to_idx[judge_key]
    if with_order:
        return (k, i, j, y, a)
    return (k, i, j, y)


def load_merged_dataframe(csv_path=DEFAULT_CSV_PATH):
    return pd.read_csv(csv_path)


def build_item_index(df):
    items = sorted(set(df["judge_model_a_original"].dropna()) | set(df["judge_model_b_original"].dropna()))
    return items, {name: idx for idx, name in enumerate(items)}


def build_llm_dataset(df, item_to_idx):
    llm_df = df.drop_duplicates("judge_row_id_10k")
    judge_names = sorted(llm_df["llm_judge_model"].dropna().unique().tolist())
    judge_to_idx = {name: idx for idx, name in enumerate(judge_names)}

    records = []
    for row in llm_df.itertuples(index=False):
        comparison = _row_to_comparison(
            row.judge_model_a_original,
            row.judge_model_b_original,
            row.llm_judge_actual_preferred_model,
            row.llm_judge_model,
            item_to_idx,
            judge_to_idx,
            with_order=True,
        )
        if comparison is not None:
            records.append(comparison)

    return {
        "judge_names": judge_names,
        "judge_to_idx": judge_to_idx,
        "records": records,
        "n_dropped_unknown": len(llm_df) - len(records),
        "n_total_rows": len(llm_df),
    }


def build_human_dataset(df, item_to_idx):
    human_df = df.drop_duplicates("human_row_id_33k")
    judge_names = sorted(human_df["human_judge_id"].dropna().unique().tolist())
    judge_to_idx = {name: idx for idx, name in enumerate(judge_names)}

    records = []
    for row in human_df.itertuples(index=False):
        comparison = _row_to_comparison(
            row.human_model_a_original,
            row.human_model_b_original,
            row.human_actual_preferred_model,
            row.human_judge_id,
            item_to_idx,
            judge_to_idx,
            with_order=False,
        )
        if comparison is not None:
            records.append(comparison)

    return {
        "judge_names": judge_names,
        "judge_to_idx": judge_to_idx,
        "records": records,
        "n_dropped_unknown": len(human_df) - len(records),
        "n_total_rows": len(human_df),
    }


def records_to_aggregated(records, N, K):
    return comparisons_to_aggregated(records, N, K)


def records_to_order_aggregated(records, N, K):
    return comparisons_to_order_aggregated(records, N, K)


def sample_human_records(records, n_sample=500, random_seed=42):
    if n_sample > len(records):
        raise ValueError(f"requested sample {n_sample} exceeds available human records {len(records)}")
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(records), size=n_sample, replace=False)
    return [records[int(i)] for i in idx]


def sample_human_records_with_holdout(records, n_sample=500, random_seed=42):
    # Same draw as sample_human_records, but also returns the complementary
    # held-out records -- used to fit a human-only reference consensus that
    # is disjoint from the records used for DIAL calibration, so a
    # comparison against it isn't contaminated by reusing the same rows.
    if n_sample > len(records):
        raise ValueError(f"requested sample {n_sample} exceeds available human records {len(records)}")
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(records), size=n_sample, replace=False)
    idx_set = set(int(i) for i in idx)
    sample = [records[int(i)] for i in idx]
    holdout = [records[i] for i in range(len(records)) if i not in idx_set]
    return sample, holdout


def split_records(records, test_ratio=0.2, random_seed=42):
    rng = np.random.default_rng(random_seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(records) * test_ratio)))
    test_idx = set(idx[:n_test].tolist())
    train = [records[i] for i in range(len(records)) if i not in test_idx]
    test = [records[i] for i in range(len(records)) if i in test_idx]
    return train, test


def load_experiment_data(csv_path=DEFAULT_CSV_PATH):
    df = load_merged_dataframe(csv_path)
    items, item_to_idx = build_item_index(df)
    llm = build_llm_dataset(df, item_to_idx)
    human = build_human_dataset(df, item_to_idx)
    return {
        "items": items,
        "item_to_idx": item_to_idx,
        "N": len(items),
        "llm": llm,
        "human": human,
    }


if __name__ == "__main__":
    data = load_experiment_data()
    print("N items:", data["N"])
    print("LLM judges (K):", len(data["llm"]["judge_names"]), data["llm"]["judge_names"])
    print("LLM usable records:", len(data["llm"]["records"]), "dropped:", data["llm"]["n_dropped_unknown"])
    print("Human judges (K):", len(data["human"]["judge_names"]))
    print("Human usable records:", len(data["human"]["records"]), "dropped:", data["human"]["n_dropped_unknown"])
