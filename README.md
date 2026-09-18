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

This project uses the **`llm`** conda environment (Python 3.11). Activate it
before running anything here:

```bash
conda activate llm
python -m pip install -e '.[dev,viz]'
```

`viz` adds matplotlib, seaborn, pyarrow, and ipykernel for the analysis
notebooks. To build an equivalent environment from scratch instead, use the
checked-in spec (same floors as `pyproject.toml`, plus nbconvert and pymupdf for
re-running and inspecting the figure notebooks):

```bash
conda env create -f environment.yml && conda activate dial
python -m pip install -e .
```

The data-collection tools have separate optional dependencies:

```bash
python -m pip install -e '.[data]'
```

Python 3.10 is the minimum the package declares, but use 3.11+: the collection
kit in `data/` relies on the standard-library `tomllib`.

Notebooks under `notebooks/` are pinned to the `Python (llm)` kernel. If it is
not registered yet:

```bash
conda activate llm
python -m ipykernel install --user --name llm --display-name "Python (llm)"
```

Do not run from the conda `base` environment. A `python3` kernelspec resolving
to `base` shadows the one inside `llm`, so a notebook left on the default kernel
will silently execute against `base` instead.

## Quick checks

```bash
conda activate llm
pytest -m 'not slow'
pytest
python run_simulation.py --list-studies
python run_simulation.py --study human_preference --smoke
python run_real_data.py --list-studies
```

Generated run metadata and summaries are written under `results/`, which is
ignored by Git.

The full paper experiments take about two hours on 16 workers, most of it in the
real-data GACV paths; see the runtime section of
[experiments/README.md](experiments/README.md#runtime) for per-study times and
what dominates them.

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
| Human | Implemented |
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
