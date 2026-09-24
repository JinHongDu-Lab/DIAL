# DIAL

Code and data for DIAL: position-debiased preferences from a panel of LLM judges, calibrated to
limited human comparisons.

## Installation

```bash
conda env create -f environment.yml
conda activate dial
python -m pip install -e .
pytest -m 'not slow'
```

## Data

`data/<dataset>/` holds the prepared benchmark (`data/*.parquet`) and every collected judge
response (`responses.parquet`) for Chatbot Arena, MT-Bench, and PandaLM. See
[data/README.md](data/README.md) for provenance, the response schema, and the collection kit.

## Reproducing the paper

| Paper output | Command | Drawn by |
|---|---|---|
| Synthetic-study figures | `python run_simulation.py --config <main10, app20, main10_mis, main10_pos> ...` | `notebooks/1_simulation.ipynb` |
| Position-debiasing figures | `python run_real_data.py robustness --sweep all --seeds 0:50` and `--clean-fit` | `notebooks/2_real_data_position.ipynb` |
| Human-calibration figures | the robustness run above and `python run_real_data.py intermediate --seeds 50` | `notebooks/3_real_data_calibration.ipynb` |
| Dataset tables | `python run_real_data.py tables` | printed |

Exact commands, the method panel, and runtimes are in [experiments/README.md](experiments/README.md).
Results are written under `results/`; the notebooks save figures to `figures/`.

## Layout

```text
src/dial_judge/   estimators (structured LLM model, DIAL joint fit, GACV, baselines, inference)
experiments/      simulation and real-data studies, shared figure style
notebooks/        figure code
configs/          real-data study design
data/             benchmarks, judge responses, and the judge-query kit
tests/            unit and integration tests
```

## License

MIT; see [LICENSE](LICENSE).
