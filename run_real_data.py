"""Command-line entry point for DIAL real-data studies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.real_data import case_study, perturbation


DATASETS = ("arena_33k", "mt_bench", "pandalm")
STUDIES = {
    "perturbation": perturbation,
    "case_study": case_study,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(STUDIES))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-studies", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_studies:
        payload = {
            name: {dataset: module.describe(dataset) for dataset in DATASETS}
            for name, module in STUDIES.items()
        }
        print(json.dumps(payload, indent=2))
        return
    if args.study is None or args.dataset is None:
        raise SystemExit("--study and --dataset are required unless --list-studies is used")

    payload = STUDIES[args.study].describe(args.dataset)
    payload.update({"seed": args.seed, "smoke": args.smoke})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results") / args.study / f"{args.dataset}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
