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
| `human_only` | Human | centred BTL on the human sample (lambda = 0) |
| `pooled_cal` | Pooled | one BTL over all LLM judgments (no judge identity, no order term) plus a human-fitted scale |
| `consensus_cal` | Cons-Cal | order-effect structured model at rank r on LLM data, consensus mu, one human-fitted scale (lambda = infinity endpoint of DIAL) |
| `dial_mu` | DIAL-$\mu$ | joint weighted likelihood at rank r, human score aligned to mu only (`align="mu"`), GACV weight |
| `dial_w` | DIAL-$W$ | same fit calibrating within W = [mu, V] (`align="W"`), GACV weight; drawn in the simulation figures, appendix diagnostic on real data |
| `dial_nodeb` | DIAL-noPos | DIAL-$\mu$ without the order term |
| `atc_btl` | AtC-BTL | stage-matched external baseline: the same human comparisons aggregated by BTL, then an isotonic (PAVA) projection of the LLM-only position-debiased consensus onto that ordering; main-text calibration figure only |

Diagnostics (appendix only): `staged_w`, `dial_mle_w` (calibration within W = [mu, V]),
`dial_mle_mu` (fixed weight n_L / n_H), `oracle_mu` / `oracle_test` (population-risk or
test-loss minimizer on DIAL-$\mu$'s path), `dial_rsel` ((r, lambda) by GACV).
`dial_w` stays in `style.APPENDIX` but the simulation notebook passes it to `draw(..., extra=["dial_w"])`
for all three r1 figures; the real-data figures keep the five presented methods.
The alignment is an option of `dial_model.joint`, `gacv.select_lambda`, and
`benchmarks.fit_dial` (`align="W"` remains the library default).

## Simulation R1 (`simulation/r1_main.py`, `r1_plot.py`, `notebooks/1_simulation.ipynb`)

LLM side S = gamma mu^T + U V^T at rank r with judge-specific position effects and an
unbalanced display (canonical item first with probability 0.75); human target
s_0 = alpha_0 mu (configs `main10`, `app20`, aligned with the consensus as on the three
benchmarks) or s_0 = W c_0 with c_V ~ N(0, 0.5^2) (`main10_mis`, appendix, where DIAL-$W$
is needed). Rows: n_H at fixed n_L (theory lines (N-1)/(2 n_H), 1/(2 n_H), and
(r+1)/(2 n_H) for the W-calibration); n_L at
fixed n_H with pair-level LLM overdispersion; and `pos` (config `main10_pos`, appendix),
which sweeps sigma_pos, the within-judge pairwise spread of b_kij = b_k + delta_kij, at the
main configuration's fixed budgets while every method keeps a constant b_k working model.
A cell is (row, n_L, n_H, sigma_pos); rows written before sigma_pos existed default to 0.0,
so the three budget configs group exactly as before. Metrics: excess human risk, Kendall tau
(= 2 x pairwise sign accuracy - 1), 95% contrast coverage (cell-clustered sandwich), RMSE
of b-hat; Spearman and MSE(S) in the appendix tables. Pooled and DIAL-noPos have no order
parameter, so they report the RMSE of their implied b-hat = 0 (`b_is_zero` flag; b_sign_acc
omitted, being undefined for a zero estimate) and stay visible in that panel.

```bash
python experiments/simulation/r1_main.py --config main10 --row both --seeds 0:50 --workers 6
python experiments/simulation/r1_main.py --config app20  --row both --seeds 0:50 --workers 6
python experiments/simulation/r1_main.py --config main10_mis --row both --seeds 0:50 --workers 6
python experiments/simulation/r1_main.py --config main10_pos --row pos  --seeds 0:50 --workers 8
```

The first three figures come from the notebook's `draw`; the sigma_pos figure
(`fig_simu_pos_heterogeneity.pdf`, Figure F3) has its own three-panel layout and comes from the
notebook's `pos_figure`. `r1_plot.py` only loads and aggregates: all figure code lives in the
notebook.

## Real-data modes

- `motivation`: descriptive per-judge summary behind Figure 1(b). Implemented.
- `robustness`: the real-data study of
  `code/plan/2026-09-06-real-data-robustness-plan.md` (Section 22), implemented
  in `real_data/robustness.py`. Five sweeps on one record-level protocol:
  `order` (swapped-copy share with a canonical-first default), `noise`
  (anti-consensus / position-only judges injected into a small panel under a
  one-sided display), `budget` (human labels, balanced LLM data), `llm_budget`
  (LLM rows subsampled at fixed human budgets), `spectest` (likelihood-ratio
  test of `s_0 = alpha mu` on Arena human subsets). Methods: the shared panel above, LLM rank
  `llm_rank = 1`, plus the diagnostics; `--methods a,b` recomputes only the listed
  methods for cells that lack them.
  `real_data/inspect_cell.py` rebuilds one cell's data exactly as `run_cell` does and prints
  the staged fit's convergence/separation diagnostics and DIAL's GACV path
  (`python -m experiments.real_data.inspect_cell mt_bench noise_scarce biased5 anti 0 --seed 1`).
  `real_data/ja_reanalysis.py` rescored HJA's JA-Ranking judge files against our
  human labels (`--ja`). Rows append to `results/real_robustness/<dataset>/rows.jsonl`,
  `robustness_plot.py` aggregates, and `notebooks/2_real_data_position.ipynb`
  draws `figures/fig_real_main.pdf` (2 x 4: grouped bars over three levels per
  dataset for position debiasing and for adaptive weighting/robustness, plus the
  judge-level position effects and the specification test), `fig_real_curves.pdf`
  (the Chatbot Arena sweeps behind the bars, in log loss and Kendall tau),
  `fig_real_efficiency.pdf`, and `fig_real_diagnostics.pdf`.

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

`notebooks/0_motivation.ipynb` reads the saved CSVs and renders the figure.

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

## Study 3: judge panels across budget regimes (`real_data/panel_budget.py`, `notebooks/3_real_data_calibration.ipynb`)

Appendix G.4.2. `panel_budget.py` loads and aggregates; the notebook draws all three figures on
the same 3 x 3 grid (datasets x judge panels `small6` / `large6` / `all`), from two runs that
share the same 50 record splits:

| Figure | Sweep | Fixed | Source |
|---|---|---|---|
| `fig_panel_human_budget.pdf` | human budget n_H | as-collected LLM data | robustness rows |
| `fig_panel_llm_budget.pdf` | LLM budget n_L | n_H = 300 / 80 / 60 (named in each row label) | robustness rows |
| `fig_panel_intermediate_llm.pdf` | human budget n_H | n_L = 2000 / 160 / 100 | `results/intermediate_budget` |

Each cell records both metrics of the notebook's `METRIC`: excess held-out log loss and
Kendall's tau against the held-out human ranking. Only the log-loss versions appear in the
manuscript, so they are what `show(name)` draws by default; `show(name, metrics=METRIC)` adds the
`_tau` companions. They rank the methods the same way but weight the regimes
differently: log loss charges by the probability gap and by the number of test comparisons on a
pair, so it is dominated by the low-budget regime where the adaptive fit is unstable, while tau
charges every misordered pair equally.

Methods are the shared panel of `experiments/style.py` -- Human, Cons-Cal, and DIAL-mu at its
GACV-selected weight -- so this study draws the same estimators, colours and dashes as the
other two. `panel_budget_data` and `intermediate_data` assert that every cell carries all 50
seeds before anything is plotted. The main-text calibration figure adds `atc_btl`, loaded by
`panel_budget.atc_rows` from its own pass over the same splits and drawn under the human-only
display rule (both share the pooled-human BTL stage, so both are shown only where that fit is
finite in all 50 splits).

```bash
python run_real_data.py --study robustness --sweep all --seeds 0:50 --workers 12
python -m experiments.real_data.intermediate_budget --seeds 50 --workers 8 --out /tmp/dial-intermediate-budget-50
python -m experiments.real_data.intermediate_budget_validate --root /tmp/dial-intermediate-budget-50
# figures: run notebooks/3_real_data_calibration.ipynb

# AtC baseline of Figure 4: one add-on pass per source, appended to the same result directories
python -m experiments.real_data.robustness --sweep budget --datasets arena_33k --panels all \
    --methods atc_btl --seeds 0:50 --workers 8
python -m experiments.real_data.intermediate_budget --seeds 50 --workers 8 \
    --out results/intermediate_budget --methods atc_btl --datasets arena_33k --panels all --tag atc
```

The notebook draws the three appendix figures and the main-text calibration figure
(`fig_real_calibration.pdf`, Figure 4), and adds the descriptive crossover table. It is the only
place figure code lives, so there is no CLI that can drift away from it.

## Side-study runner

`experiments/real_data/_runner.py` holds the resumable process-pool loop the side studies
share (`endpoint_margin`, `intermediate_budget`, `llm_thinning`): `completed_keys` reads the
keys already in a JSONL file, `run_keyed_jobs` / `run_row_jobs` run the remaining cells and
append their results, and `write_design` records the study's design. A study module should
therefore contain only its design and its summaries.

## Output contract

Each completed run uses `results/<study>/<run-id>/` and records the resolved
configuration, seed, package version, method status, failures, per-repetition
results, and summary. Generated output is ignored by Git unless deliberately
selected for the paper.
