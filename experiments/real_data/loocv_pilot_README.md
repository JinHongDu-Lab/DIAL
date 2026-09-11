# Small exact-LOOCV pilot

Run from `code/DIAL` in the experiment environment:

```bash
python -m experiments.real_data.loocv_pilot --seeds 5 --workers 6
```

The pilot uses the small6 panel on Arena, MT-Bench, and PandaLM.
The human-scarce settings use nH=50/20/20 and all available LLM training judgments.
The LLM-scarce settings use nL=500/40/25 and nH=300/80/60.
Five outer record splits (seeds 0 through 4) are shared across methods.
The outer test pool is used only for experiment evaluation, never weight selection.
Tuning reserves no validation set: each deletion uses nH-1 human comparisons and all LLM training rows.
Numerical lambda is held fixed across deletions, matching the manuscript's GACV target.

The candidate grid is the existing seven multiples of nL/nH plus zero and infinity.
Only full-data candidates admitted by the existing GACV regularity guards enter LOOCV.
At infinity the LLM representation is cached and the human calibration scale is refitted after each deletion.
Positive finite weights use full-data fits as warm starts and refit the joint model to convergence under the existing optimizer settings.
At zero the human-only BTL model is refitted, provided its MLE exists after deletion.
Identical pair/outcome observations induce identical deletions in this aggregated model, so each deletion is fitted once and its predictive loss weighted by multiplicity.
No Hessian approximation replaces the deletion refits; numerical optimization and local-solution limitations still apply.

A candidate is unusable for LOOCV if any deletion has an optimizer error, reports nonconvergence, violates the existing absolute calibrated-score bound of 10, or has no human-only MLE at lambda=0.
No individual deletion is silently removed from its criterion.
If all candidates are unusable, use the full-data infinity fit and record a fallback.
Deletion regularity checks cover numerical convergence and score bounds, not a new Hessian nonsingularity calculation for every deletion.

Compare full-data Cons-Cal, GACV selection, exact refitted LOOCV minimization, and conservative LOOCV selection.
For the conservative rule, let d_t be the infinity deletion loss minus the best LOOCV candidate's deletion loss.
Use the finite candidate only when mean(d_t) exceeds sd(d_t)/sqrt(nH); otherwise choose infinity.
This paired dispersion margin is an exploratory one-SE-style heuristic, not a valid independent-fold standard error, confidence bound, or hypothesis test.
LOO training samples overlap and weight selection adds adaptivity.
When infinity itself is unusable, use the minimum-LOOCV admissible candidate.
Final predictions always use the stored full-data fit at the selected weight.

Raw results, all candidate criteria, grouped deletion losses, multiplicities, fit issues, and selected weights are stored in `results/loocv_pilot/jobs.jsonl`.
`summary.csv` reports mean test loss, Kendall tau, infinity-selection shares, and paired Monte Carlo differences against GACV.
With five splits, these are diagnostic results rather than a final estimator comparison.
No manuscript files are modified.

The summary additionally evaluates two illustrative hybrids from the saved candidate paths.
Trigger LOOCV when nH<=50 or the absolute GACV gap between the best regular finite candidate (including zero) and the regular infinity endpoint is <=1/nH.
Use ordinary GACV otherwise; inside the trigger use either minimum LOOCV or conservative LOOCV.
These thresholds are exploratory choices, not tuned or validated cutoffs, and the hybrid was added after the first-seed smoke run.
All triggers depend only on training GACV and sample size; test losses do not enter.
Exact LOOCV was computed in every pilot cell so the hybrids can be compared without rerunning fits.
