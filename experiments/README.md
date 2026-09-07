# Experiment guide

The repository has two synthetic studies and two real-data modes. Root-level
runners provide a stable command-line interface; study modules contain the
scientific design and produce JSON-serializable records.

## Synthetic study 1: position bias

Generate `S = gamma mu^T + U V^T`, query both display orders, and vary the
position-effect magnitude and LLM sample size.

Methods:

1. position-aware structured BTL (DIAL stage 1);
2. the same structured BTL after discarding order;
3. probability-scale swap averaging followed by BTL;
4. paired-order logit averaging with documented smoothing.

Primary metrics are position-effect RMSE, score-matrix MSE, and debiased
pairwise-probability MSE. Ranking metrics alone are insufficient because
probability averaging can preserve the population ranking while shrinking
preferences toward one half.

## Synthetic study 2: human preference

Generate `s_H = alpha_H mu + V a` under the well-specified model. Use two rows:
no LLM position bias and mild judge-specific position bias. At fixed human
budget, vary the LLM comparison budget.

Methods:

1. human-only BTL;
2. position-debiased LLM consensus;
3. consensus-only calibration;
4. staged structured calibration, `lambda = infinity`;
5. DIAL with GACV-selected `lambda_hat`;
6. unstructured order-effect BTL + SVD + calibration;
7. optional AtC.

Report score MSE, held-out human log loss, Spearman, NDCG, sign accuracy, and
the selected weight. Monte Carlo standard errors describe repetition noise and
must not be presented as estimator confidence intervals.

## Real-data modes

- `motivation`: descriptive per-judge summary behind Figure 1(b). Implemented.
- `robustness`: the real-data study of
  `code/plan/2026-09-06-real-data-robustness-plan.md` (Section 22), implemented
  in `real_data/robustness.py`. Five sweeps on one record-level protocol:
  `order` (swapped-copy share with a canonical-first default), `noise`
  (anti-consensus / position-only judges injected into a small panel under a
  one-sided display), `budget` (human labels, balanced LLM data), `llm_budget`
  (LLM rows subsampled at fixed human budgets), `spectest` (likelihood-ratio
  test of `s_0 = alpha mu` on Arena human subsets), and `planner` (budget
  prediction from a pilot via the test statistic). Methods: human-only,
  pooled-LLM + scale, Consensus-cal (rank-0 staged), DIAL-mu (rank 0, joint
  weighted likelihood, GACV weight), DIAL (rank and weight by GACV), DIAL-noDeb,
  plus DIAL-mu at the MLE weight and the test-pool oracle weight as references.
  `real_data/ja_reanalysis.py` rescored HJA's JA-Ranking judge files against our
  human labels (`--ja`). Rows append to `results/real_robustness/<dataset>/rows.jsonl`,
  `robustness_plot.py` aggregates, and `notebooks/real_data_robustness.ipynb`
  draws `figures/fig_real_main.pdf` (position bias; robustness and verification),
  `fig_real_efficiency.pdf`, and `fig_real_planner.pdf`.

  ```bash
  python run_real_data.py --study robustness --sweep all --seeds 0:50 --workers 12
  python run_real_data.py --study robustness --ja
  python run_real_data.py --study robustness --clean-fit
  ```

  Results live under Dropbox; an append that overlaps a sync can leave the
  data in a `rows (... conflicted copy ...).jsonl` next to an empty
  `rows.jsonl`, so check for conflicted copies after a run.
- `case_study`: full Arena 33K, MT-Bench, or PandaLM analysis.

### Motivation study

```bash
python run_real_data.py --study motivation --dataset all
```

No model is fitted here; every quantity is a sample proportion.

Position bias is `swap_inconsistency_rate`, the share of records whose verdict
changes when the two responses are swapped, among records the judge decides in
both orders. A judge that ignores display order cannot flip, so `0` is the
reference. The directional `first_position_rate` is also saved but is not the
panel: it reads `0.5` both for a judge with no order effect and for one whose
effects offset, and 21 of 60 judge-study cells here favour the *second*
position, which a first-position framing would mislabel.

Human misalignment is `agreement_consistent`, agreement with the human label on
records where the judge returns the same verdict in both orders, so the residual
gap is not attributable to display order.

`agreement_pooled` is saved but is *not* the alignment metric: when a judge flips
with order, exactly one of its two verdicts matches the human, so order-averaged
agreement is a mixture dominated by inconsistency (empirically correlated about
`0.97` with `swap_inconsistency_rate`, against about `0.25` for
`agreement_consistent`). Using it would restate position bias rather than measure
a second phenomenon.

Each row carries a `reasoning` state read from the recorded inference config, not
parsed from the alias suffix: `direct` (reasoning disabled) or `thinking`
(enabled or required). The effort level within `thinking` — `low` for the Ollama
models, `minimal` for Gemini, unset for GLM — is a per-provider setting on no
common scale, so it is kept in `reasoning_effort` for the record rather than
treated as an ordered level. Note that `direct` also caps `max_output_tokens` at
16 against 4096, so the two states differ in decoding configuration, not
reasoning alone.

Uncertainty is a bootstrap clustered on `record_id`, sharing one resample across
judges within a study. Judges are reported in full but flagged
`excluded_from_figure` when they tie on more than half their judgments
(`ollama-stablelm2-12b-direct`) or cover less than half the study's records.
Since the 2026-09-06 collection update every one of the 22 judges covers all
three studies, so the coverage rule no longer excludes anyone; the adapter keys
each judge on its results directory and keeps a row-level alias variant (for
example the `-64tok` gap-fill re-queries of Claude Haiku) in `judge_alias`.

`notebooks/figure1b.ipynb` reads the saved CSVs and renders the figure.

Original and swapped responses belonging to one record must stay in the same
split. Full-data estimates are empirical references, not literal ground truth.

## Student implementation contracts

Unimplemented methods are listed in `dial_judge.baselines.METHOD_STATUS`.
Students should add one `fit_*` wrapper returning a score vector and `fit_info`,
then add deterministic unit tests before enabling the method in the registry.
Unavailable methods are skipped and reported; they must never emit dummy scores.

The real-data adapter should normalize the existing Parquet and JSONL artifacts
to records containing dataset, record ID, judge, canonical item pair, binary or
tie outcome, display-order sign, and human outcome.

## Output contract

Each completed run uses `results/<study>/<run-id>/` and records the resolved
configuration, seed, package version, method status, failures, per-repetition
results, and summary. Generated output is ignored by Git unless deliberately
selected for the paper.
