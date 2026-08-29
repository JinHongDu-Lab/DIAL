"""Descriptive motivation study behind Figure 1(b).

Summarizes, per judge and study, how strongly judgments depend on display order
and how far they sit from human preference.  Every reported quantity is a plain
sample proportion computed from the collected comparisons; no Bradley--Terry
model is fitted here.  The model-based counterparts belong to the case study.

Two facts make the proportions interpretable:

* Both display orders were collected for every record, so the verdict of a judge
  that ignores display order cannot change when the two responses are swapped,
  whatever the items are.  ``swap_inconsistency_rate`` measures that directly
  and is the position-bias panel: it captures order-dependence irrespective of
  direction, whereas the directional ``first_position_rate`` reads ``0.5`` both
  for a judge with no order effect and for one whose effects offset.  Judges
  favouring the *second* position are common in these data (21 of 60
  judge-study cells), so the directional rate alone would mislabel them.
* Human agreement is reported on records whose canonical verdict is identical
  in both orders.  On that subset the verdict is invariant to display order, so
  residual disagreement with the human label is not attributable to position
  bias.

``agreement_consistent`` rather than ``agreement_pooled`` is the alignment
panel, and the reason is mechanical: when a judge flips with display order,
exactly one of its two verdicts matches the human label, contributing ``0.5``.
Order-averaged agreement is therefore a mixture dominated by inconsistency --
empirically it correlates about ``0.97`` with ``swap_inconsistency_rate``,
against about ``0.25`` for ``agreement_consistent`` -- so it would restate
position bias rather than measure a second phenomenon.  The cost is that the
conditioning subset is judge-dependent and favourable to the judge (its own
order-stable comparisons), which makes the reported agreement an upper bound;
``n_agreement_consistent`` is saved so the denominator can be stated.

See ``code/plan/2026-08-29-figure1b-motivation-analysis.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prepare import describe as describe_adapter
from .prepare import load_canonical


#: Ratio metrics, each defined by a (numerator, denominator) count pair.
RATIO_METRICS = (
    "first_position_rate",
    "swap_inconsistency_rate",
    "agreement_consistent",
    "agreement_original",
    "agreement_swapped",
    "agreement_pooled",
    "judge_tie_rate",
    "human_nondecisive_rate",
)

DEFAULT_N_BOOT = 200
SMOKE_N_BOOT = 20

#: Judges with a tie rate above this are reported but excluded from the figure.
#: Dropping ties from a judge that ties on most comparisons would leave a
#: subset self-selected on the judge's own decisiveness.
TIE_RATE_EXCLUDE = 0.5

#: Judges covering less than this fraction of the study's records (in both
#: display orders) are reported but excluded from the figure as uncollected.
MIN_COVERAGE = 0.5


def _per_record_counts(frame, record_index):
    """Per-record numerator/denominator arrays for one judge.

    Each array has one entry per record in ``record_index`` (zero where the
    judge has no data), so a single bootstrap draw of record positions can be
    shared across judges.
    """
    n = len(record_index)
    counts = {name: (np.zeros(n), np.zeros(n)) for name in RATIO_METRICS}

    positions = frame["record_id"].map(record_index)
    frame = frame.assign(_pos=positions).dropna(subset=["_pos"])
    frame["_pos"] = frame["_pos"].astype(int)

    # Judge-tie and human-decisiveness rates use every collected judgment, so
    # they describe what the other metrics had to drop.
    for pos, is_tie in zip(frame["_pos"], frame["judge_tie"]):
        counts["judge_tie_rate"][1][pos] += 1
        counts["judge_tie_rate"][0][pos] += bool(is_tie)

    # One row per record and display order; keep records observed in both.
    for pos, group in frame.groupby("_pos", sort=False):
        orig = group[~group["swapped"]]
        swap = group[group["swapped"]]
        if orig.empty or swap.empty:
            continue
        orig = orig.iloc[0]
        swap = swap.iloc[0]

        human = orig["human_outcome"]
        human_decisive = bool(orig["human_decisive"])
        counts["human_nondecisive_rate"][1][pos] += 1
        counts["human_nondecisive_rate"][0][pos] += not human_decisive

        orig_tie = bool(orig["judge_tie"])
        swap_tie = bool(swap["judge_tie"])

        # (a) Directional order preference, pooled over the two orders.  The
        # design is order-balanced, so 0.5 is the no-bias reference.
        for row, tie in ((orig, orig_tie), (swap, swap_tie)):
            if tie or row["chose_first"] is None:
                continue
            counts["first_position_rate"][1][pos] += 1
            counts["first_position_rate"][0][pos] += bool(row["chose_first"])

        both_decisive = not orig_tie and not swap_tie
        consistent = both_decisive and orig["outcome"] == swap["outcome"]

        # (a) Order sensitivity regardless of direction.
        if both_decisive:
            counts["swap_inconsistency_rate"][1][pos] += 1
            counts["swap_inconsistency_rate"][0][pos] += not consistent

        if not human_decisive:
            continue

        # (b) Agreement with human preference, on human-decisive records.
        if consistent:
            counts["agreement_consistent"][1][pos] += 1
            counts["agreement_consistent"][0][pos] += orig["outcome"] == human

        for name, row, tie in (
            ("agreement_original", orig, orig_tie),
            ("agreement_swapped", swap, swap_tie),
        ):
            if tie:
                continue
            counts[name][1][pos] += 1
            counts[name][0][pos] += row["outcome"] == human
            counts["agreement_pooled"][1][pos] += 1
            counts["agreement_pooled"][0][pos] += row["outcome"] == human

    return counts


def _ratio(num, den):
    total = den.sum()
    if total == 0:
        return float("nan")
    return float(num.sum() / total)


def run(
    dataset,
    judges=None,
    n_boot=DEFAULT_N_BOOT,
    seed=42,
    smoke=False,
    data_root=None,
    use_cache=True,
):
    """Compute the per-judge summary and its clustered bootstrap distribution."""
    if smoke:
        n_boot = min(n_boot, SMOKE_N_BOOT)

    frame = load_canonical(dataset, judges=judges, data_root=data_root, use_cache=use_cache)
    record_ids = pd.Index(sorted(frame["record_id"].unique()))
    record_index = pd.Series(np.arange(len(record_ids)), index=record_ids)
    judge_names = sorted(frame["judge"].unique())

    per_judge = {}
    for judge in judge_names:
        per_judge[judge] = _per_record_counts(
            frame[frame["judge"] == judge].reset_index(drop=True), record_index
        )

    # Judge-level metadata, constant within a judge; carried through so the
    # figure can label reasoning-toggle pairs without re-reading the raw files.
    meta = (
        frame.groupby("judge")[["judge_model", "reasoning", "reasoning_effort"]]
        .first()
        .to_dict("index")
    )

    rows = []
    for judge in judge_names:
        counts = per_judge[judge]
        row = {"dataset": dataset, "judge": judge}
        row.update(meta.get(judge, {}))
        for name in RATIO_METRICS:
            num, den = counts[name]
            row[name] = _ratio(num, den)
            row[f"n_{name}"] = int(den.sum())
        row["n_records"] = int((counts["human_nondecisive_rate"][1] > 0).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["coverage"] = summary["n_records"] / max(len(record_ids), 1)
    summary["low_coverage"] = summary["coverage"] < MIN_COVERAGE
    summary["high_tie_rate"] = summary["judge_tie_rate"] > TIE_RATE_EXCLUDE
    summary["excluded_from_figure"] = summary["low_coverage"] | summary["high_tie_rate"]

    # Clustered bootstrap: records are resampled, and the two display orders of
    # a record move together because they are the same comparison.  The same
    # draw is reused for every judge so between-judge contrasts within a panel
    # are not inflated by independent resampling noise.
    boot_rows = []
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        n_records = len(record_ids)
        stacked = {
            name: (
                np.vstack([per_judge[j][name][0] for j in judge_names]),
                np.vstack([per_judge[j][name][1] for j in judge_names]),
            )
            for name in RATIO_METRICS
        }
        for rep in range(n_boot):
            idx = rng.integers(0, n_records, size=n_records)
            for name, (num, den) in stacked.items():
                num_b = num[:, idx].sum(axis=1)
                den_b = den[:, idx].sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    value = np.where(den_b > 0, num_b / np.maximum(den_b, 1), np.nan)
                for judge, val in zip(judge_names, value):
                    boot_rows.append(
                        {
                            "dataset": dataset,
                            "judge": judge,
                            "replicate": rep,
                            "metric": name,
                            "value": float(val),
                        }
                    )
    bootstrap = pd.DataFrame(boot_rows, columns=["dataset", "judge", "replicate", "metric", "value"])

    if not bootstrap.empty:
        quantiles = (
            bootstrap.groupby(["judge", "metric"])["value"]
            .quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: "lo", 0.975: "hi"})
            .reset_index()
        )
        wide = quantiles.pivot(index="judge", columns="metric", values=["lo", "hi"])
        wide.columns = [f"{metric}_{bound}" for bound, metric in wide.columns]
        summary = summary.merge(wide.reset_index(), on="judge", how="left")

    return summary, bootstrap, record_ids


def describe(dataset):
    return {
        "study": "motivation",
        "dataset": dataset,
        "outputs": ["per_judge", "bootstrap", "summary"],
        "metrics": list(RATIO_METRICS),
        "position_panel": "swap_inconsistency_rate",
        "position_direction": "first_position_rate",
        "alignment_panel": "agreement_consistent",
        "reference_value": 0.5,
        "tie_rate_exclude": TIE_RATE_EXCLUDE,
        "min_coverage": MIN_COVERAGE,
        "cluster_unit": "record_id",
        "model_free": True,
        "adapter": describe_adapter(),
        "status": "implemented",
    }
