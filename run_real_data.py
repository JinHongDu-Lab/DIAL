"""Real-data studies; see experiments/README.md.

    python run_real_data.py tables                            # dataset tables
    python run_real_data.py robustness --sweep all --seeds 0:50 --workers 12
    python run_real_data.py intermediate --seeds 50 --workers 8
    python run_real_data.py prepare                           # rebuild results/cache/ from data/
"""

import sys

STUDIES = ("tables", "robustness", "intermediate", "prepare")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STUDIES:
        raise SystemExit(__doc__)
    study, argv = sys.argv[1], sys.argv[2:]
    if study == "tables":
        from experiments.real_data.data_tables import main as run
        run()
    elif study == "robustness":
        from experiments.real_data.robustness import main as run
        run(argv)
    elif study == "intermediate":
        from experiments.real_data.intermediate_budget import main as run
        run(argv)
    else:
        from experiments.real_data.prepare import SUPPORTED_DATASETS, load_canonical
        for dataset in SUPPORTED_DATASETS:
            print(dataset, len(load_canonical(dataset, refresh=True)))


if __name__ == "__main__":
    main()
