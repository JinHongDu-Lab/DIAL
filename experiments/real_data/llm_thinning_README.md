# Fitting DIAL with fewer available LLM comparisons

This exploratory experiment tests whether discarding recorded LLM comparisons can improve DIAL-mu when human calibration labels are scarce.
It does not modify the manuscript or the existing robustness results.

Run from `code/DIAL`, using the experiment environment:

```bash
python -m experiments.real_data.llm_thinning --seeds 50 --workers 8
python -m experiments.real_data.llm_thinning_plot
```

The design crosses Arena/MT-Bench/PandaLM with all/small6/large6 judge panels.
Human budgets are 50 and 100 for Arena, and 20 and 40 for MT-Bench and PandaLM.
For each of 50 seeds, the existing robustness code supplies the record-disjoint training/test split and human calibration draw.
All LLM fractions share those same human samples.
LLM training rows are sampled uniformly without replacement at fractions 0.025, 0.05, 0.1, 0.25, 0.5, and 1 of the available training panel.
Samples at different fractions are not nested; they use the existing deterministic robustness sampling routine.
One LLM subsample is drawn per fraction and outer seed, so Monte Carlo uncertainty includes subsampling variation.
Structural rank, fitting guards, and weight multipliers are unchanged from `configs/real_robustness.toml`.
The numerical weight grid scales with the retained nL/nH, exactly as in the existing LLM-budget sweep; this experiment is therefore not a pure fixed-weight information-removal comparison.

Comparisons include full-data DIAL-mu, full-data Cons-Cal, fixed-fraction DIAL-mu, and DIAL-mu with both the fraction and weight selected by GACV.
For joint selection, minimize each fraction's selected GACV over fractions with regular selected fits, prefer full data on exact ties, and fall back to full-data DIAL when none is regular.
This extended use of GACV across sample sizes is exploratory, not a newly established guarantee.
Human test labels never enter this selection.
The test-loss oracle across fractions is explicitly a diagnostic lower envelope and is not a deployable estimator.
The oracle chooses among GACV-selected fits, rather than every candidate weight.

Primary evaluation is mean human test excess log loss, subtracting the human BTL fit on the test pool as in the existing robustness study.
Kendall's tau also uses the existing test-pool BTL reference.
For comparisons on a shared seed, the test-loss floor cancels exactly.
Paired Monte Carlo standard errors describe variation across randomized splits of these fixed benchmark datasets, not uncertainty over new benchmark datasets.

Outputs live in `results/llm_thinning/`: raw fit rows, design, selected rows, curves, paired comparisons, diagnostics, and rendered PNGs.
PDF figures live in `figures/fig_llm_thinning.pdf` and `figures/fig_llm_thinning_tau.pdf`.
The experiment is resumable by complete dataset/panel/budget/seed groups; use a fresh output directory if changing the design or estimator.
