"""Real-data studies; see experiments/README.md.

    python run_real_data.py prepare                           # rebuild results/cache/ from data/
    python run_real_data.py tables                            # dataset tables
    python run_real_data.py motivation                        # descriptive position-bias and alignment summaries
    python run_real_data.py robustness --sweep all --seeds 0:50 --workers 12
    python run_real_data.py intermediate --seeds 50 --workers 8
    python run_real_data.py bootstrap --datasets arena_33k,mt_bench --seeds 0:10 --B 200
    python run_real_data.py summary                           # tables of the comparisons and checks
"""

import sys

STUDIES = ("prepare", "tables", "motivation", "robustness", "intermediate", "bootstrap", "summary")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STUDIES:
        raise SystemExit(__doc__)
    study, argv = sys.argv[1], sys.argv[2:]
    if study == "prepare":
        from experiments.real_data.prepare import SUPPORTED_DATASETS, load_canonical
        for dataset in SUPPORTED_DATASETS:
            print(dataset, len(load_canonical(dataset, refresh=True)))
    elif study == "tables":
        from experiments.real_data.aggregate import print_dataset_tables
        print_dataset_tables()
    elif study == "motivation":
        from experiments.real_data.motivation import main as run
        run(argv)
    elif study == "robustness":
        from experiments.real_data.robustness import main as run
        run(argv)
    elif study == "intermediate":
        from experiments.real_data.intermediate_budget import main as run
        run(argv)
    elif study == "bootstrap":
        from experiments.real_data.record_bootstrap import main as run
        run(argv)
    else:
        from experiments.summary import main as run
        run(argv)


if __name__ == "__main__":
    main()
