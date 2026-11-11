"""Synthetic study; see experiments/README.md.

    python run_simulation.py --config main10 --row both --seeds 0:50 --workers 6
    python run_simulation.py bootstrap --config main10_record --seeds 0:50 --B 200 --workers 6
"""

import sys

if __name__ == "__main__":
    if sys.argv[1:2] == ["bootstrap"]:
        from experiments.simulation.record_bootstrap import main
        main(sys.argv[2:])
    else:
        from experiments.simulation.synthetic import main
        main(sys.argv[1:])
