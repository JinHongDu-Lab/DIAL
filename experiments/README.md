# Experiment guide

The repository has two synthetic studies and two real-data modes. Root-level
runners provide a stable command-line interface; study modules contain the
scientific design and produce JSON-serializable records.

## Method panel (both studies)

`experiments/style.py` fixes the five presented estimators, their order, names, and
colours, for every figure of the simulation and the real-data study
(`code/plan/2026-09-07-unified-method-panel-plan.md`):

| Key | Presented name | Definition |
|---|---|---|
| `human_only` | Human-only | centred BTL on the human sample (lambda = 0) |
| `pooled_cal` | Pooled-LLM | one BTL over all LLM judgments (no judge identity, no order term) plus a human-fitted scale |
| `consensus_cal` | Consensus-cal | order-effect structured model at rank r on LLM data, consensus mu, one human-fitted scale (lambda = infinity endpoint of DIAL) |
| `dial_mu` | DIAL | joint weighted likelihood at rank r, human score aligned to mu only (`align="mu"`), GACV weight |
| `dial_nodeb` | DIAL-noDeb | DIAL without the order term |

Diagnostics (appendix only): `staged_w`, `dial_w`, `dial_mle_w` (calibration within
W = [mu, V]), `dial_mle_mu` (fixed weight n_L / n_0), `oracle_mu` / `oracle_test`
(population-risk or test-loss minimizer on DIAL's path), `dial_rsel` ((r, lambda) by GACV).
The alignment is an option of `dial_model.joint`, `gacv.select_lambda`, and
`benchmarks.fit_dial` (`align="W"` remains the library default).

## Simulation R1 (`simulation/r1_main.py`, `r1_plot.py`, `notebooks/1_simulation.ipynb`)

LLM side S = gamma mu^T + U V^T at rank r with judge-specific position effects and an
unbalanced display (canonical item first with probability 0.75); human target
s_0 = alpha_0 mu (configs `main10`, `app20`, aligned with the consensus as on the three
benchmarks) or s_0 = W c_0 with c_V ~ N(0, 0.5^2) (`main10_mis`, appendix, where DIAL-W
is needed). Rows: n_0 at fixed n_L (theory lines (N-1)/(2 n_0) and 1/(2 n_0)); n_L at
fixed n_0 with pair-level LLM overdispersion. Metrics: excess human risk, Kendall tau
(= 2 x pairwise sign accuracy - 1), 95% contrast coverage (cell-clustered sandwich), RMSE
of b-hat; Spearman and MSE(S) in the appendix tables.

```bash
python experiments/simulation/r1_main.py --config main10 --row both --seeds 0:50 --workers 6
python experiments/simulation/r1_main.py --config app20  --row both --seeds 0:50 --workers 6
python experiments/simulation/r1_main.py --config main10_mis --row both --seeds 0:50 --workers 6
```

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
  prediction from a pilot via the test statistic). Methods: the shared panel above, LLM rank
  `llm_rank = 1`, plus the diagnostics; `--methods a,b` recomputes only the listed
  methods for cells that lack them.
  `real_data/inspect_cell.py` rebuilds one cell's data exactly as `run_cell` does and prints
  the staged fit's convergence/separation diagnostics and DIAL's GACV path
  (`python -m experiments.real_data.inspect_cell mt_bench noise_scarce biased5 anti 0 --seed 1`).
  `real_data/ja_reanalysis.py` rescored HJA's JA-Ranking judge files against our
  human labels (`--ja`). Rows append to `results/real_robustness/<dataset>/rows.jsonl`,
  `robustness_plot.py` aggregates, and `notebooks/real_data_robustness.ipynb`
  draws `figures/fig_real_main.pdf` (2 x 4: grouped bars over three levels per
  dataset for position debiasing and for adaptive weighting/robustness, plus the
  judge-level position effects and the specification test), `fig_real_curves.pdf`
  (the Chatbot Arena sweeps behind the bars, in log loss and Kendall tau),
  `fig_real_efficiency.pdf`, `fig_real_planner.pdf`, and `fig_real_diagnostics.pdf`.

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
