"""Command-line entry point for DIAL real-data studies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.real_data import case_study, motivation, perturbation, robustness


DATASETS = ("arena_33k", "mt_bench", "pandalm")
STUDIES = {
    "perturbation": perturbation,
    "case_study": case_study,
    "motivation": motivation,
    "robustness": robustness,
}


def parse_args():
    import sys

    # The robustness study has its own sweep/seed interface; hand it every
    # argument except the study selector.
    if "--study" in sys.argv and sys.argv[sys.argv.index("--study") + 1] == "robustness":
        rest = [a for n, a in enumerate(sys.argv[1:]) if a != "--study" and sys.argv[1:][n - 1] != "--study"]
        robustness.main(rest)
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(STUDIES))
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ("all",),
        help="dataset to analyze, or 'all' for every dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list-studies", action="store_true")
    parser.add_argument("--output", type=Path, help="output directory (runnable studies) or file")
    parser.add_argument("--judges", help="comma-separated judge aliases; default is every judge")
    parser.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="re-parse the raw responses instead of reading results/cache/",
    )
    return parser.parse_args()


def run_study(module, dataset, args, stamp):
    """Execute a study that implements ``run`` and write its artifacts."""
    from experiments.real_data import prepare

    if args.refresh_cache:
        prepare.load_canonical(dataset, use_cache=True, refresh=True)

    n_boot = 0 if args.no_bootstrap else args.n_boot
    kwargs = {"seed": args.seed, "smoke": args.smoke}
    if args.judges:
        kwargs["judges"] = [name.strip() for name in args.judges.split(",") if name.strip()]
    if n_boot is not None:
        kwargs["n_boot"] = n_boot

    summary, bootstrap, record_ids = module.run(dataset, **kwargs)

    outdir = args.output or Path("results") / args.study / f"{dataset}-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "per_judge.csv", index=False)
    bootstrap.to_csv(outdir / "bootstrap.csv", index=False)

    config = module.describe(dataset)
    config.update(
        {
            "seed": args.seed,
            "smoke": args.smoke,
            "n_boot": int(bootstrap["replicate"].nunique()) if not bootstrap.empty else 0,
            "judges": kwargs.get("judges"),
            "generated_at": stamp,
        }
    )
    (outdir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outdir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "n_judges": int(len(summary)),
                "n_records": int(len(record_ids)),
                "judges": summary["judge"].tolist(),
                "excluded_from_figure": summary.loc[
                    summary["excluded_from_figure"], "judge"
                ].tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(outdir)


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

    module = STUDIES[args.study]
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if hasattr(module, "run"):
        if args.output is not None and len(datasets) > 1:
            raise SystemExit("--output cannot be combined with --dataset all")
        for dataset in datasets:
            run_study(module, dataset, args, stamp)
        return

    # Scaffolded studies still only emit their specification.
    for dataset in datasets:
        payload = module.describe(dataset)
        payload.update({"seed": args.seed, "smoke": args.smoke})
        output = args.output or Path("results") / args.study / f"{dataset}-{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
