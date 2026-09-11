"""Command-line entry point for DIAL synthetic studies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.simulation import human_preference


STUDIES = {
    human_preference.STUDY_NAME: human_preference,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(STUDIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-studies", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_output(study):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / study / f"{stamp}.json"


def main():
    args = parse_args()
    if args.list_studies:
        print(json.dumps({name: module.describe() for name, module in STUDIES.items()}, indent=2))
        return
    if args.study is None:
        raise SystemExit("--study is required unless --list-studies is used")

    payload = STUDIES[args.study].run(smoke=args.smoke, seed=args.seed)
    output = args.output or default_output(args.study)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output.relative_to(Path.cwd()) if output.is_absolute() and output.is_relative_to(Path.cwd()) else output)


if __name__ == "__main__":
    main()
