# DIAL

DIAL is research software for learning position-debiased preferences from a
panel of LLM judges and calibrating the resulting structured preference space
to limited human comparisons.

The code currently implements:

- structured HJA estimation with optional judge-specific position effects;
- finite-weight joint DIAL estimation;
- GACV selection of the LLM weight;
- human-only, LLM-consensus, consensus-only calibration, and staged structured
  calibration endpoints;
- synthetic data generators and evaluation metrics.

The experiment runners also list the remaining paper baselines as explicit
student implementation tasks. They never substitute dummy numerical results.

## Installation

Python 3.10 or later is required.

```bash
python -m pip install -e '.[dev]'
```

The data-collection tools have separate optional dependencies:

```bash
python -m pip install -e '.[data]'
```

## Quick checks

```bash
pytest -m 'not slow'
pytest
python run_simulation.py --list-studies
python run_simulation.py --study position_bias --smoke
python run_real_data.py --list-studies
```

Generated run metadata and summaries are written under `results/`, which is
ignored by Git.

## Repository layout

```text
src/dial_judge/       core estimators, baselines, data helpers, and metrics
tests/                deterministic unit tests and marked integration tests
experiments/          simulation and real-data study implementations
configs/              paper-study defaults
data/                 real datasets and the judge-query collection kit
run_simulation.py     synthetic-study entry point
run_real_data.py      real-data entry point
```

See [experiments/README.md](experiments/README.md) for the planned comparisons,
metrics, and student tasks. See [data/README.md](data/README.md) for collection
provenance and the current analysis-integration status.

## Implemented method status

| Method | Status |
|---|---|
| Human-only BTL | Implemented |
| LLM consensus | Implemented |
| Consensus-only calibration | Implemented |
| Staged structured calibration, `lambda = infinity` | Implemented |
| Finite-weight DIAL | Implemented |
| GACV-selected DIAL | Implemented |
| Probability-scale order averaging | Student task |
| Paired-order logit averaging | Student task |
| Unstructured BTL + SVD + calibration | Student task |
| AtC | Optional student task; license and data-contract review required |

## Data policy

The local `data/` tree contains large Parquet and JSONL artifacts. Do not commit
the entire tree directly. Before publication, choose either Git LFS or a
versioned external archive with checksums and a download script.

## License

See [LICENSE](LICENSE).
