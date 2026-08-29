"""Tests for the real-data adapter and the descriptive motivation study.

These run on a small synthetic collection tree, never on the shipped data, so
the expected proportions can be written down by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.real_data import motivation, prepare


def _row(record_id, judge, swapped, choice, verdict, human, model_a="m1", model_b="m2"):
    """One collected judgment, mirroring the collection kit's schema.

    ``choice`` is the displayed slot ("a" is displayed first); ``verdict`` is
    the already de-swapped canonical side.
    """
    return {
        "record_id": record_id,
        "judge": judge,
        "model_a": model_a,
        "model_b": model_b,
        "condition_id": "swapped_anonymous" if swapped else "original_anonymous",
        "response_order_default": not swapped,
        "choice": choice,
        "canonical_verdict": verdict,
        "human_winner": human,
    }


@pytest.fixture()
def collection_tree(tmp_path):
    """Four records, three judges with hand-checkable behaviour.

    ``always-first`` always picks the first-displayed response, so it flips on
    every record.  ``stable`` always returns ``model_a`` regardless of order.
    ``tie-heavy`` ties on three of four records.
    """
    rows = []
    humans = {"r1": "model_a", "r2": "model_a", "r3": "model_a", "r4": "model_b"}
    for record, human in humans.items():
        # Picks whichever response is shown first: original -> model_a,
        # swapped -> model_b.
        rows.append(_row(record, "always-first", False, "a", "model_a", human))
        rows.append(_row(record, "always-first", True, "a", "model_b", human))
        # Order-invariant: always model_a, displayed second when swapped.
        rows.append(_row(record, "stable", False, "a", "model_a", human))
        rows.append(_row(record, "stable", True, "b", "model_a", human))
        # Ties except on r1.
        tie = record != "r1"
        rows.append(_row(record, "tie-heavy", False, "c" if tie else "a",
                         "tie" if tie else "model_a", human))
        rows.append(_row(record, "tie-heavy", True, "c" if tie else "b",
                         "tie" if tie else "model_a", human))

    root = tmp_path / "data"
    for judge in ("always-first", "stable", "tie-heavy"):
        directory = root / "mt_bench" / "results" / judge
        directory.mkdir(parents=True)
        lines = [json.dumps(r) for r in rows if r["judge"] == judge]
        (directory / "responses.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture()
def frame(collection_tree):
    return prepare.load_canonical("mt_bench", data_root=collection_tree, use_cache=False)


def test_adapter_deswaps_verdicts(frame):
    """``chose_first`` comes from the displayed slot, ``outcome`` from the verdict."""
    swapped = frame[(frame["judge"] == "stable") & frame["swapped"]]
    assert (swapped["outcome"] == "m1").all()
    # model_a is displayed second under the swap, so it is not the first choice.
    assert (~swapped["chose_first"].astype(bool)).all()

    original = frame[(frame["judge"] == "stable") & ~frame["swapped"]]
    assert (original["outcome"] == "m1").all()
    assert original["chose_first"].astype(bool).all()


def test_adapter_canonical_pair_and_display_order(collection_tree):
    """Items are name-sorted and ``display_order`` tracks the canonical first item."""
    directory = collection_tree / "mt_bench" / "results" / "reversed-pair"
    directory.mkdir(parents=True)
    # model_a sorts after model_b, so the canonical pair reverses the sides.
    row = _row("r9", "reversed-pair", False, "a", "model_a", "model_b", model_a="zz", model_b="aa")
    (directory / "responses.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    frame = prepare.load_canonical("mt_bench", data_root=collection_tree, use_cache=False)
    entry = frame[frame["judge"] == "reversed-pair"].iloc[0]
    assert entry["item_i"] == "aa" and entry["item_j"] == "zz"
    # "zz" was displayed first, so the canonical first item "aa" was not.
    assert entry["display_order"] == -1
    assert entry["outcome"] == "zz"
    assert entry["human_outcome"] == "aa"


def test_adapter_skips_judges_without_responses(collection_tree):
    """A results directory holding only metadata is omitted, not reported empty."""
    (collection_tree / "mt_bench" / "results" / "uncollected").mkdir(parents=True)
    (collection_tree / "mt_bench" / "results" / "uncollected" / "metadata.json").write_text(
        "{}", encoding="utf-8"
    )
    judges = prepare.judge_directories("mt_bench", data_root=collection_tree)
    assert "uncollected" not in judges
    assert set(judges) == {"always-first", "stable", "tie-heavy"}


def test_position_bias_metrics(collection_tree):
    summary, _, _ = motivation.run(
        "mt_bench", n_boot=0, data_root=collection_tree, use_cache=False
    )
    summary = summary.set_index("judge")

    # Always picks the first-displayed response, and therefore always flips.
    assert summary.loc["always-first", "first_position_rate"] == pytest.approx(1.0)
    assert summary.loc["always-first", "swap_inconsistency_rate"] == pytest.approx(1.0)

    # Order-invariant: picks first in one order and second in the other.
    assert summary.loc["stable", "first_position_rate"] == pytest.approx(0.5)
    assert summary.loc["stable", "swap_inconsistency_rate"] == pytest.approx(0.0)


def test_agreement_uses_order_consistent_records(collection_tree):
    summary, _, _ = motivation.run(
        "mt_bench", n_boot=0, data_root=collection_tree, use_cache=False
    )
    summary = summary.set_index("judge")

    # "stable" agrees with the human on r1, r2, r3 but not r4.
    assert summary.loc["stable", "n_agreement_consistent"] == 4
    assert summary.loc["stable", "agreement_consistent"] == pytest.approx(0.75)

    # "always-first" is never order-consistent, so the subset is empty.
    assert summary.loc["always-first", "n_agreement_consistent"] == 0
    assert np.isnan(summary.loc["always-first", "agreement_consistent"])


def test_tie_accounting_and_figure_exclusion(collection_tree):
    summary, _, _ = motivation.run(
        "mt_bench", n_boot=0, data_root=collection_tree, use_cache=False
    )
    summary = summary.set_index("judge")

    # Ties on r2, r3, r4 in both orders: 6 of 8 judgments.
    assert summary.loc["tie-heavy", "judge_tie_rate"] == pytest.approx(0.75)
    assert summary.loc["tie-heavy", "high_tie_rate"]
    assert summary.loc["tie-heavy", "excluded_from_figure"]
    # Ties are dropped from the proportions, leaving only r1.
    assert summary.loc["tie-heavy", "n_first_position_rate"] == 2
    assert not summary.loc["stable", "excluded_from_figure"]


def test_human_nondecisive_records_are_excluded(collection_tree):
    """A tie or ``both_bad`` human label leaves the agreement denominators."""
    path = collection_tree / "mt_bench" / "results" / "stable" / "responses.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["record_id"] == "r4":
            row["human_winner"] = "both_bad"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary, _, _ = motivation.run(
        "mt_bench", n_boot=0, data_root=collection_tree, use_cache=False
    )
    summary = summary.set_index("judge")
    assert summary.loc["stable", "human_nondecisive_rate"] == pytest.approx(0.25)
    # r4 was the only disagreement, so agreement over the remaining three is 1.
    assert summary.loc["stable", "n_agreement_consistent"] == 3
    assert summary.loc["stable", "agreement_consistent"] == pytest.approx(1.0)


def test_bootstrap_is_seed_reproducible(collection_tree):
    kwargs = dict(n_boot=8, data_root=collection_tree, use_cache=False)
    _, first, _ = motivation.run("mt_bench", seed=3, **kwargs)
    _, again, _ = motivation.run("mt_bench", seed=3, **kwargs)
    _, other, _ = motivation.run("mt_bench", seed=4, **kwargs)

    assert first["value"].equals(again["value"])
    assert not first["value"].equals(other["value"])
    assert first["replicate"].nunique() == 8


def test_bootstrap_is_clustered_on_records(collection_tree):
    """Both display orders of a record move together under resampling."""
    _, boot, _ = motivation.run(
        "mt_bench", n_boot=50, seed=1, data_root=collection_tree, use_cache=False
    )
    values = boot[(boot["judge"] == "always-first") & (boot["metric"] == "first_position_rate")]
    # Every record contributes the same two judgments, so no resample can move
    # this judge's rate away from 1.
    assert values["value"].nunique() == 1
    assert values["value"].iloc[0] == pytest.approx(1.0)
