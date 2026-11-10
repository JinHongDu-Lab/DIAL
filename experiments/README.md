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

## Alternative estimators and further designs

`real_data/alternative_estimators.py` and the `--addon` pass of `simulation/synthetic.py`; rows go
to `results/alternatives/`, and `python -m experiments.alternatives_summary` writes the tables
(`results/alternatives/summary/`) and the matched-design and orthogonal-misalignment figures.

| Key | Name | Definition |
|---|---|---|
| `ppi_bt_best` | PPI++ (best judge) | Bradley--Terry PPI++; proxy = the judge agreeing most with the cell's calibration labels |
| `ppi_bt_panel` | PPI++ (panel) | the same with the panel majority verdict as proxy |
| `ridge_mu` | Joint-l2 | human BTL + kappa ||P s||^2 shrinking toward span(mu_hat), kappa by approximate leave-one-out |
| `human_box` | Human (box) | unpenalized human BTL on DIAL's coordinate box |
| `human_ridge` | Human (ridge) | ridge toward zero, kappa by approximate leave-one-out |
| `human_firth` | Human (Firth) | Firth-penalized human BTL |
| `atc_ridge`, `atc_firth` | AtC (ridge / Firth) | AtC with the stage-1 ordering from the penalized fit |
| `llm_only` | LLM-only consensus | mu_hat alone (zero human budget; ranking metrics only) |

PPI++ is run as defined (no penalty, box, or imputation) and needs a proxy verdict on every
human-labeled record; its rows record `ppi_status` (ok / partial / failed) and the matched counts.

```bash
ALT=ppi_bt_best,ppi_bt_panel,ridge_mu,human_box,human_ridge,human_firth,atc_ridge,atc_firth

# human-budget sweep (n_H = 0 and 3200 on Arena included) and the intermediate LLM budget
python run_real_data.py robustness --sweep budget --panels all --methods $ALT --seeds 0:50 --workers 8 --out results/alternatives/budget
python run_real_data.py robustness --sweep budget_full --panels all --seeds 0:50 --workers 8 --out results/alternatives/budget
python run_real_data.py intermediate --seeds 50 --workers 8 --tag alternatives --methods $ALT --out results/alternatives/intermediate_budget

# matched-pair design: n_H labeled records with every judge's verdicts, plus proxy-only records
python run_real_data.py robustness --sweep matched --seeds 0:50 --workers 8 --out results/alternatives/matched \
    --methods human_only,consensus_cal,dial_mu,atc_btl,ppi_bt_best,ppi_bt_panel,ridge_mu,human_firth
python run_simulation.py --config main10_matched --row nH --seeds 0:50 --workers 6
python run_simulation.py --config main10_matched --row nH --seeds 0:50 --workers 6 --addon
python run_simulation.py --config main10 --row nH --seeds 0:50 --workers 6 --addon

# orthogonal misalignment: s_H = mu + rho ||mu|| u / ||u||, u orthogonal to {1, mu, V}
python run_simulation.py --config main10_orth --row orth --seeds 0:50 --workers 6
python run_simulation.py --config main10_orth --row orth --seeds 0:50 --workers 6 --addon

# rank sensitivity (likewise --llm-rank 2 into .../r2), then the summary and the BIC choice of r
python run_real_data.py robustness --sweep budget --seeds 0:50 --workers 8 --llm-rank 0 \
    --methods consensus_cal,dial_mu,dial_w --out results/real_robustness_rank/r0
python -m experiments.real_data.rank_sensitivity
```

## Resampling checks of the intervals

`simulation/record_bootstrap.py` (config `main10_record`: matched design with a record effect
shared by the judges and the human label) and `real_data/record_bootstrap.py` refit DIAL, GACV
selection included, on resampled records and compare with the sandwich intervals. `--scheme half`
draws half of the records of each stratum without replacement instead of a bootstrap sample.

```bash
python -m experiments.simulation.record_bootstrap --config main10_record --seeds 0:50 --B 200 --workers 5
python -m experiments.real_data.record_bootstrap --datasets arena_33k,mt_bench --levels 200,800 --seeds 0:10 --B 200 --workers 4
python -m experiments.real_data.twoway_check --workers 8                 # record, judge-pair, and two-way sandwiches
python -m experiments.real_data.selection_bootstrap_check --workers 16   # resampling scheme and the GACV selection
python -m experiments.simulation.record_scale_check --workers 16         # sandwich coverage at larger budgets
```

## Motivation figure

`real_data/motivation.py`; Figure 1(b) in `notebooks/0_motivation.ipynb`. Model-free: every
quantity is a sample proportion over records collected in both display orders, with a 200-replicate
bootstrap clustered on `record_id` (seed 42, one resampled record set shared by all judges of a
dataset).

| Metric | Definition |
|---|---|
| `swap_inconsistency_rate` (panel a) | share of records, decisive in both orders, whose canonical verdict changes when the responses are swapped |
| `agreement_consistent` (panel b, plotted as 1 minus) | agreement with the human label on records where the verdict is the same in both orders and the human label is decisive |
| `first_position_rate` | share of non-tie judgments picking the first-displayed response (directional; reported, not plotted) |

Judges with a tie rate above 0.5 or covering under half of a dataset's records are saved but left
out of the figure (`ollama-stablelm2-12b-direct` in all three datasets).

```bash
python run_real_data.py motivation          # results/motivation/<dataset>/per_judge.csv, bootstrap.csv
```

## Runtime

Wall clock on an 18-core laptop with `--workers 16` and 50 seeds: about 1 to 2 minutes per
simulation config, about 2 hours for the full real-data `--sweep all` (dominated by the GACV paths
of the `llm_budget` sweep on Arena), about 15 minutes for `intermediate`, and about 2 minutes for `motivation`.
