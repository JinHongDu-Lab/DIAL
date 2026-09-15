# Intermediate LLM budget pilot

This bridges G6 and G7 by fixing the middle G7 LLM budgets (Arena 2000, MT-Bench 160, PandaLM 100) and sweeping all human budgets from G6.
These LLM budgets were fixed before inspecting pilot outcomes.
All three judge panels and seeds 0 through 4 are included: 57 settings and 285 paired cells.
The adaptive DIAL-mu curve uses the existing c=1 endpoint margin, with all fitting guards unchanged.
Neither the LLM budget nor the margin is selected using test loss.
This is an exploratory five-seed figure; uncertainty bands are mean plus/minus 1.96 Monte Carlo standard errors, not a claim of reliable coverage with five seeds.
The existing record-disjoint evaluation and training samples are reused.
Human and LLM training draws are paired across methods and judge panels.
A relative improvement over Cons-Cal after reducing LLM records does not imply that discarding available LLM data improves absolute accuracy.
No manuscript figure is replaced by this pilot.

```bash
python -m experiments.real_data.intermediate_budget --seeds 5 --workers 8 --out /tmp/dial-intermediate-budget
# figures: run notebooks/3_real_data_calibration.ipynb
```

Copy completed outputs into `results/intermediate_budget_pilot/` after validation.
Use a fresh directory for a changed design.

## Fifty-seed appendix extension

The approved extension retains the exact pilot design and adds seeds 5 through 49: 57 settings, 2850 cells, and 8550 method rows.
Results are stored separately in `results/intermediate_budget/`.

```bash
python -m experiments.real_data.intermediate_budget --seeds 50 --workers 8 --out /tmp/dial-intermediate-budget-50
python -m experiments.real_data.intermediate_budget_validate --root /tmp/dial-intermediate-budget-50
# figures: run notebooks/3_real_data_calibration.ipynb
python -m experiments.real_data.human_budget_graph --seeds 50 --out /tmp/dial-human-budget-graph
```

The validator checks complete matched seeds, candidate-prediction reuse, the margin formula, and exact reproduction of 855 pilot rows and 450 overlapping G7 rows.
`fit_diagnostics.csv` retains nonconvergence and all-irregular fallback counts; no cells are silently removed from the figure.
`converged_pair_sensitivity.csv` is a descriptive restriction to pairs where both methods report convergence, not an unbiased replacement for the full analysis.
`empirical_crossover.csv` summarizes the first tested human budget with a mean adaptive advantage greater than two paired Monte Carlo standard errors there and at every higher tested budget.
That post hoc summary is not a calibrated test or a validated rule for new datasets.
The final PDF is copied to the manuscript as `figures/fig_panel_intermediate_llm.pdf`; the original five-seed pilot is preserved separately.
Human-graph diagnostics use only the corresponding human training draws and are identical across judge panels; undirected connectivity does not imply existence of an unrestricted human BTL MLE.
