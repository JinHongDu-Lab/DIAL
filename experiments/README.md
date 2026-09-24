# Experiments

Study modules write per-replication rows to `results/`; the notebooks aggregate them and draw every
figure. Run commands from the repository root.

## Method panel

`experiments/style.py` fixes the presented estimators, their order, names, and colours for all
figures.

| Key | Name | Definition |
|---|---|---|
| `human_only` | Human | centred BTL on the human sample (lambda = 0) |
| `pooled_cal` | Pooled | one BTL over all LLM judgments (no judge identity, no order term) plus a human-fitted scale |
| `consensus_cal` | Cons-Cal | order-effect structured LLM model at rank r, consensus mu, one human-fitted scale (lambda = infinity) |
| `dial_mu` | DIAL-$\mu$ | joint weighted likelihood at rank r, human score aligned to mu, GACV weight |
| `dial_w` | DIAL-$W$ | the same fit calibrating within W = [mu, V] |
| `dial_nodeb` | DIAL-noPos | DIAL-$\mu$ without the order term |
| `atc_btl` | AtC-BTL | human BTL ordering, then an isotonic projection of the position-debiased LLM consensus onto it |

Appendix diagnostics: `staged_w`, `dial_mle_mu` / `dial_mle_w` (fixed weight n_L / n_H),
`oracle_mu` / `oracle_w` / `oracle_test` (risk- or test-loss-minimizing weight on DIAL's path), and `dial_rsel`
((r, lambda) by GACV).

## Synthetic study

`simulation/synthetic.py`; figures in `notebooks/1_simulation.ipynb`.
The LLM side is S = gamma mu^T + U V^T at rank r with judge-specific position effects and
canonical-first display with probability 0.75. The human target is s_0 = alpha_0 mu (`main10`,
`app20`) or s_0 = W c_0 with c_V ~ N(0, 0.5^2) (`main10_mis`). Rows vary n_H at fixed n_L, n_L at
fixed n_H with pair-level LLM overdispersion, and (`main10_pos`) the spread sigma_pos of
pair-varying position effects b_kij = b_k + delta_kij.

```bash
python run_simulation.py --config main10     --row both --seeds 0:50 --workers 6
python run_simulation.py --config app20      --row both --seeds 0:50 --workers 6
python run_simulation.py --config main10_mis --row both --seeds 0:50 --workers 6
python run_simulation.py --config main10_pos --row pos  --seeds 0:50 --workers 8
```

## Real-data study

`real_data/robustness.py` with the design in `configs/real_robustness.toml`. Each cell splits the
records into a human training pool and a held-out test pool; original and swapped responses of a
record stay in the same split. Sweeps: `order` (share of swapped copies), `noise` and
`noise_scarce` (injected anti-consensus or position-only judges), `budget` (human budget),
`llm_budget` (LLM budget), and `spectest` (test of s_0 = alpha mu on Arena subsets).

```bash
python run_real_data.py robustness --sweep all --seeds 0:50 --workers 12
python run_real_data.py robustness --clean-fit          # full-data judge-level position effects
python run_real_data.py robustness --sweep budget --datasets arena_33k --panels all \
    --methods atc_btl --seeds 0:50 --workers 8
python run_real_data.py intermediate --seeds 50 --workers 8                  # fixed n_L = 2000 / 160 / 100
python run_real_data.py intermediate --seeds 50 --workers 8 --methods atc_btl \
    --datasets arena_33k --panels all --tag atc
python run_real_data.py tables                                               # dataset tables
```

Rows go to `results/real_robustness/<dataset>/rows.jsonl` and `results/intermediate_budget/`.
`robustness_plot.py` and `panel_budget.py` aggregate them;
`notebooks/2_real_data_position.ipynb` draws the position-debiasing figures and
`notebooks/3_real_data_calibration.ipynb` the human-calibration figures. Runs resume: cells already
in the output are skipped, and `--methods` adds methods to existing cells.

## Runtime

Wall clock on an 18-core laptop with `--workers 16` and 50 seeds: about 1 to 2 minutes per
simulation config, about 2 hours for the full real-data `--sweep all` (dominated by the GACV paths
of the `llm_budget` sweep on Arena), and about 15 minutes for `intermediate`.
