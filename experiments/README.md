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

- `perturbation`: controlled human/LLM subsampling, label flips, judge removal,
  and order imbalance, evaluated on held-out human comparisons.
- `case_study`: full Arena 33K, MT-Bench, or PandaLM analysis.

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
