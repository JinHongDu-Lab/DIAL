# Direct endpoint margin for GACV

Run from `code/DIAL` in the experiment environment:

```bash
python -m experiments.real_data.endpoint_margin --seeds 20 --workers 6
python -m experiments.real_data.endpoint_margin --summarize
```

For each regular candidate path, select the best finite numerical weight by GACV, including zero.
Choose infinity if GACV(infinity) minus that finite candidate's GACV is at most c/nH.
Otherwise retain the best finite candidate.
The primary proposed value is c=1; c=0.5 and 2 are sensitivity checks and are not chosen using test results.
This is a heuristic preference for the two-step DIAL endpoint, not a hypothesis test.
The same existing regularity guards apply to all candidates.
An irregular infinity fit cannot be reinstated by the margin.
When all candidates are irregular, preserve the existing selector's fallback and label it.
Ties favor infinity, then the larger finite weight.

The rule uses only the existing GACV path and adds no model fitting, holdout set, LOOCV, bootstrap, or LLM subsampling for tuning.
Every variant predicts using a fit to the complete training samples available in its experimental cell.
The robustness experiment's independent outer test pool remains solely for evaluation.
The default GACV selector is unchanged; the optional `_gacv_endpoint_margins` runner hook generates additional experiment rows.

The study covers Arena, MT-Bench, and PandaLM, with all and small6 judge panels.
Use all configured human-budget levels (7/6/6 levels) with abundant LLM training data.
Use all five configured LLM-budget levels per dataset with the default human budgets (300/80/60).
Use seeds 0 through 19, paired across estimators: 68 settings and 1,360 cells.
Each cell saves Cons-Cal, original GACV DIAL-mu, and the three margin choices.

Results live in `results/endpoint_margin/`.
`jobs.jsonl` contains atomic complete-cell records with chosen weights, criterion gaps, margins, and fit flags.
`rows.csv` contains individual fit results; `summary.csv` contains mean excess human test log loss, Kendall tau, paired Monte Carlo differences against GACV, and infinity-selection shares.
Monte Carlo errors quantify variation across splits of these fixed datasets, not uncertainty across new datasets.
The run is resumable by complete cell keys; use a fresh output directory if changing the design.
No manuscript files are changed.

## Appendix G.4.2 exports

The manuscript extension uses seeds 0 through 49 on all, small6, and large6 panels: 102 settings and 5,100 cells.
Keep these results separate from the original 20-seed exploration.
Compute outside the Dropbox-synced tree and copy completed, validated artifacts into `results/endpoint_margin_appendix/`.

```bash
python -m experiments.real_data.endpoint_margin --seeds 50 --workers 8 --panels all small6 large6 --out /tmp/dial-endpoint-margin-run
# figures: run notebooks/3_real_data_calibration.ipynb
```

The figure exporter verifies 50 matched seeds per method and setting and plots the fixed c=1 rule as DIAL-mu, with original GACV as DIAL-mu (raw).
It exports the human-budget and LLM-budget PDFs plus the aggregated plotting data.
The final cells of `notebooks/small_panel.ipynb` call this same exporter.
