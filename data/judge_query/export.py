"""Pack a study's collected responses into one Parquet file.

``<study>/results/<judge>/responses.jsonl`` -> ``<study>/responses.parquet``, one row per
response with every collected field except the local run label ``source_run``. ``panel_judge`` is
the results directory (the panel alias); nested fields (``inference``, ``usage``,
``provider_options``) are stored as JSON strings.

Usage
-----
    python -m judge_query.export arena_33k mt_bench pandalm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KIT_ROOT = Path(__file__).resolve().parents[1]
NESTED = ("inference", "usage", "provider_options")
DROPPED = ("source_run",)


def load_responses(study_dir):
    """Every collected response of one study as a DataFrame."""
    rows = []
    for path in sorted(Path(study_dir, "results").glob("*/responses.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["panel_judge"] = path.parent.name
                    rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no responses.jsonl files under {study_dir}/results")
    frame = pd.DataFrame(rows).drop(columns=list(DROPPED), errors="ignore")
    for column in NESTED:
        if column in frame:
            frame[column] = frame[column].map(lambda v: None if v is None else json.dumps(v, sort_keys=True))
    return frame


def export(study, root=KIT_ROOT):
    study_dir = Path(root) / study
    frame = load_responses(study_dir)
    out = study_dir / "responses.parquet"
    frame.to_parquet(out, compression="zstd", index=False)
    return out, len(frame)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("studies", nargs="+")
    args = parser.parse_args(argv)
    for study in args.studies:
        out, n = export(study)
        print(f"{out.relative_to(KIT_ROOT)}: {n} rows")


if __name__ == "__main__":
    main()
