# Alternative estimators and matched designs (added after the ICLR 2027 reviews)

Rules: stored results under `results/real_robustness`, `results/intermediate_budget`, and
`results/simulation` are never rewritten; every new row goes to `results/alternatives/`. PPI++ is
run as defined (no penalty, box, or imputation); where the stored unmatched designs give it no
matched verdicts the row records the failure. The matched-pair sweep exists for the comparison on
PPI++'s own terms. Plan and status: `../../plan/2026-11-07-iclr-rebuttal-plan.md`.

## Methods (`experiments/real_data/alternative_estimators.py`, `experiments/style.py` `ALTERNATIVES`)

| Key | Name | Definition |
|---|---|---|
| `ppi_bt_best` | PPI++ (best judge) | Bradley--Terry PPI++; proxy = the judge agreeing most with the cell's calibration labels |
| `ppi_bt_panel` | PPI++ (panel) | the same with the panel majority verdict as proxy |
| `ridge_mu` | Joint (l2) | human BTL + kappa ||P s||^2 shrinking toward span(mu_hat), kappa by approximate leave-one-out |
| `human_box` | Human (box) | unpenalized human BTL on DIAL's coordinate box, box activity recorded |
| `human_ridge` | Human (ridge) | ridge toward zero, kappa by approximate leave-one-out |
| `human_firth` | Human (Firth) | Firth-penalized human BTL |
| `atc_ridge`, `atc_firth` | AtC (ridge / Firth) | AtC with the stage-1 ordering from the penalized fit |
| `llm_only` | LLM-only consensus | mu_hat alone (zero human budget; ranking metrics only) |

Row fields added by PPI++: `ppi_status` (ok / partial / failed), `n_labeled_matched`,
`n_proxy_only`, `lam`, `ppi_lam_raw`, `proxy_judge`, `proxy_agreement`.

## Experiment 1a: add-ons on the stored designs (unmatched verdicts)

```bash
python run_real_data.py robustness --sweep budget --panels all \
    --methods ppi_bt_best,ppi_bt_panel,ridge_mu,human_box,human_ridge,human_firth,atc_ridge,atc_firth \
    --seeds 0:50 --workers 8 --out results/alternatives/budget
python run_real_data.py intermediate --seeds 50 --workers 8 --tag alternatives --out results/alternatives/intermediate_budget \
    --methods ppi_bt_best,ppi_bt_panel,ridge_mu,human_box,human_ridge,human_firth,atc_ridge,atc_firth
python run_simulation.py --config main10 --row nH --seeds 0:50 --workers 6 --addon      # PPI++ rows record unmatched_design
```

## Experiment 1b: matched-pair design

```bash
python run_real_data.py robustness --sweep matched --seeds 0:50 --workers 8 --out results/alternatives/matched \
    --methods human_only,consensus_cal,dial_mu,atc_btl,ppi_bt_best,ppi_bt_panel,ridge_mu,human_firth
python run_simulation.py --config main10_matched --row nH --seeds 0:50 --workers 6           # the stored method panel
python run_simulation.py --config main10_matched --row nH --seeds 0:50 --workers 6 --addon   # PPI++, ridge_mu, human_firth
```

Design (`configs/real_robustness.toml` `[sweeps.matched]`): n_H labeled train records with their
human label and every kept judge's verdicts in both orders, plus n_U proxy-only train records
(-1 = all remaining); the job's `level` is n_H and its `n_H_level` slot carries n_U. Synthetic
`main10_matched`: every record answered once by each of the K judges, n_U = 5000 proxy-only
records, human target s_H = mu.

## Group 2: human-budget extension

```bash
python run_real_data.py robustness --sweep budget_full --panels all --seeds 0:50 --workers 8 --out results/alternatives/budget
```

`budget_full` adds n_H = 0 (only `llm_only` is defined) and n_H = 3200 on Arena with the full
stored method panel plus the add-ons via `--methods`.

## Group 3: orthogonal misalignment (R3-Q1)

```bash
python run_simulation.py --config main10_orth --row orth --seeds 0:50 --workers 6
python run_simulation.py --config main10_orth --row orth --seeds 0:50 --workers 6 --addon
```

`main10_orth` sets s_H = mu + rho ||mu|| u / ||u|| with u orthogonal to {1, mu, V}; the `orth`
row varies rho in {0, 0.1, 0.2, 0.3, 0.5, 0.75, 1} at n_H in {200, 800} (the row's third cell slot
carries rho, stored as `rho_orth`). Each row records `delta_mu`, `delta_W` (Theorem 3's
population approximation errors), `nH_delta_mu`, `nH_delta_W`, and the crossover values
`crossover_mu` = (N - 2)/2 and `crossover_W` = (N - 2 - r)/2.
