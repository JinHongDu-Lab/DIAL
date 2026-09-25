"""Dataset statistics and candidate models of the three benchmarks after preprocessing."""
from __future__ import annotations

import pandas as pd

from .prepare import load_canonical
from .robustness import build_panel, load_config

DATASETS = {"arena_33k": "Chatbot Arena", "mt_bench": "MT-Bench", "pandalm": "PandaLM"}


def dataset_stats(dataset, cfg=None):
    """One column of the dataset-statistics table; judges are kept as in the real-data study."""
    s = (cfg or load_config())["study"]
    frame = load_canonical(dataset)
    panel = build_panel(frame, exclude=s["exclude_judges"], tie_rate_exclude=s["tie_rate_exclude"],
                        min_coverage=s["min_coverage"], llm_ties=s["llm_ties"], human_ties=s["human_ties"])
    N, K, llm = panel["N"], panel["K"], panel["llm"]
    records = frame.drop_duplicates("record_id")
    labels = records["human_label"].value_counts()
    n_pairs = N * (N - 1) // 2
    cells = llm.groupby(["k", "i", "j"]).size()
    kept = frame[frame["judge"].isin(panel["judges"])]
    tie_rate = kept.groupby("judge")["judge_tie"].mean()
    return {
        "records": panel["n_records"],
        "N": N,
        "K": K,
        "observed pairs": f"{records.groupby(['item_i', 'item_j']).ngroups} / {n_pairs}",
        "human a / b / tie / both bad": " / ".join(str(int(labels.get(x, 0))) for x in ("model_a", "model_b", "tie", "both_bad")),
        "decisive human labels": panel["n_human_decisive"],
        "LLM judgments, kept judges": panel["n_llm_rows"],
        "of which ties": f"{panel['n_llm_ties']} ({panel['n_llm_ties'] / panel['n_llm_rows']:.1%})",
        "tie-excluded n_L": len(llm),
        "display order +1 / -1": f"{(llm.a == 1).sum()} / {(llm.a == -1).sum()}",
        "cells observed / possible": f"{len(cells)} / {K * n_pairs}",
        "n_kij per cell (min / mean / max)": f"{cells.min()} / {cells.mean():.1f} / {cells.max()}",
        "judge tie rate (min-max)": f"{tie_rate.min():.3f}-{tie_rate.max():.3f}",
        "candidate models": ", ".join(panel["items"]),
    }


def main():
    cfg = load_config()
    table = pd.DataFrame({label: dataset_stats(d, cfg) for d, label in DATASETS.items()})
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(table.drop("candidate models").to_string())
        for label, items in table.loc["candidate models"].items():
            print(f"\n{label}: {items}")


if __name__ == "__main__":
    main()
