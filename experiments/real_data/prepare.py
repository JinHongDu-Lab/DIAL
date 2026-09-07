"""Analysis-ready data adapter for the collected judge responses.

Reads ``data/<dataset>/results/<judge>/responses.jsonl`` and returns one tidy
row per (judge, record, display order).  Collection files are only read, never
rewritten; the normalized table is cached separately under ``results/cache/``.

The collection kit stores two distinct notions of the judge's answer:

``choice``
    the *displayed* slot the judge picked -- ``"a"`` is the first-displayed
    response, ``"b"`` the second-displayed one, ``"c"`` a tie.
``canonical_verdict``
    the same verdict already mapped back to the original sides
    (``"model_a"`` / ``"model_b"`` / ``"tie"``), i.e. de-swapped.

Position-bias quantities read ``choice``; agreement with ``human_winner`` reads
``canonical_verdict``.  Swapping the two inverts the swapped half of the data.

The judge identity is the results *directory* name, which is the panel alias
the study was configured with.  Rows inside one directory may carry a variant
alias in their own ``judge`` field (for example a ``-64tok`` suffix on gap-fill
re-queries of the same model with a larger output cap); that value is kept in
``judge_alias`` so the decoding change stays visible without splitting one
judge into two.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


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

SUPPORTED_DATASETS = ("arena_33k", "mt_bench", "pandalm")

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
CACHE_ROOT = Path(__file__).resolve().parents[2] / "results" / "cache"

#: Verdict labels that resolve to a decisive side.
DECISIVE = ("model_a", "model_b")


def judge_directories(dataset, data_root=None):
    """Return ``{judge: responses.jsonl path}`` for judges with collected rows.

    A judge whose results directory holds only ``metadata.json`` (no responses)
    is omitted rather than reported as empty; ``ollama-deepseek-r1-32b-low`` is
    in this state for ``arena_33k``.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    results = root / dataset / "results"
    if not results.is_dir():
        raise FileNotFoundError(f"no results directory for dataset {dataset!r}: {results}")
    found = {}
    for entry in sorted(results.iterdir()):
        path = entry / "responses.jsonl"
        if entry.is_dir() and path.is_file():
            found[entry.name] = path
    if not found:
        raise FileNotFoundError(f"no responses.jsonl files under {results}")
    return found


def _canonical_pair(model_a, model_b):
    """Order the two items by name so a pair has one canonical representation."""
    if model_a <= model_b:
        return model_a, model_b, False
    return model_b, model_a, True


def _side_to_item(side, model_a, model_b):
    """Map a ``model_a``/``model_b`` side label to the item it names."""
    if side == "model_a":
        return model_a
    if side == "model_b":
        return model_b
    return None


def _reasoning(row):
    """The judge's reasoning configuration, as ``(state, effort)``.

    Taken from the ``inference`` block the collection kit recorded on every
    response, which is authoritative; the alias suffixes (``-direct``, ``-low``,
    ``-thinking``, ``-minimal``) are only a naming convention.

    There are two states. ``direct`` is reasoning disabled; ``thinking`` is
    reasoning enabled or required, however the provider spells it. The effort
    level within ``thinking`` (``low``, ``minimal``, or unset) is a
    vendor-specific knob on no common cross-provider scale, so it is recorded
    per judge rather than treated as an ordered level.
    """
    reasoning = (row.get("inference") or {}).get("reasoning") or {}
    mode = reasoning.get("mode")
    if mode is None:
        return None, None
    return ("direct" if mode == "disabled" else "thinking"), reasoning.get("effort")


def _row_to_record(row, dataset):
    reasoning, reasoning_effort = _reasoning(row)
    model_a = row["model_a"]
    model_b = row["model_b"]
    item_i, item_j, swapped_pair = _canonical_pair(model_a, model_b)

    # ``response_order_default`` is True when model_a was displayed first.
    a_first = bool(row.get("response_order_default", True))
    # display_order is +1 when the canonical first item was displayed first.
    i_first = a_first != swapped_pair
    display_order = 1 if i_first else -1

    choice = row.get("choice")
    verdict = row.get("canonical_verdict")
    human = row.get("human_winner")

    return {
        "dataset": dataset,
        "record_id": row["record_id"],
        "judge": row["judge"],
        "item_i": item_i,
        "item_j": item_j,
        "outcome": _side_to_item(verdict, model_a, model_b),
        "display_order": display_order,
        "human_outcome": _side_to_item(human, model_a, model_b),
        "condition": row.get("condition_id"),
        "swapped": not a_first,
        # True when the judge picked the first-displayed response.
        "chose_first": True if choice == "a" else (False if choice == "b" else None),
        "judge_tie": verdict not in DECISIVE,
        "human_decisive": human in DECISIVE,
        "human_label": human,
        "judge_model": row.get("judge_model"),
        "category": row.get("main_category"),
        "language": row.get("language"),
        "is_code": row.get("is_code"),
        "judge_alias": row.get("judge_alias"),
        "max_output_tokens": (row.get("inference") or {}).get("max_output_tokens"),
        "reasoning": reasoning,
        "reasoning_effort": reasoning_effort,
    }


def load_canonical(dataset, judges=None, data_root=None, use_cache=True, refresh=False):
    """Load the normalized comparison table for one dataset.

    Parameters
    ----------
    dataset : str
        One of :data:`SUPPORTED_DATASETS`.
    judges : sequence of str, optional
        Restrict to these judge aliases.  Filtering is applied after caching so
        the cache stays complete.
    use_cache, refresh : bool
        Read from / rewrite the parquet cache under ``results/cache/``.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")

    cache_path = CACHE_ROOT / f"{dataset}.parquet"
    frame = None
    if use_cache and not refresh and cache_path.is_file():
        frame = pd.read_parquet(cache_path)

    if frame is None:
        rows = []
        for judge, path in judge_directories(dataset, data_root=data_root).items():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    # The directory is the panel alias; a row-level variant
                    # alias is kept separately (see module docstring).
                    payload["judge_alias"] = payload.get("judge", judge)
                    payload["judge"] = judge
                    rows.append(_row_to_record(payload, dataset))
        frame = pd.DataFrame(rows)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)

    if judges is not None:
        frame = frame[frame["judge"].isin(list(judges))].reset_index(drop=True)
    return frame


def describe():
    return {
        "status": "implemented",
        "canonical_fields": CANONICAL_FIELDS,
        "split_unit": "record_id",
        "supported_datasets": SUPPORTED_DATASETS,
    }
