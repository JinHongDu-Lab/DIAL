# Judge query kit

> **Analysis integration status.** The files in this directory are collection
> inputs and raw per-judge outputs. The analysis package does not yet normalize
> them automatically. The planned adapter will join each study's Parquet file
> to `results/<judge>/responses.jsonl`, canonicalize item pairs and display
> order, and split at `record_id` so original/swapped responses cannot leak
> across train and test sets. Until a Git LFS or external-archive policy is
> selected, do not commit the complete local data tree directly.

A minimal, self-contained copy of the "Causal Judge" data-collection pipeline
for three studies -- `arena_33k`, `mt_bench`, `pandalm` -- distilled from the
full `causal-judge` research repository for this paper's reproducibility
appendix. It queries the same 22-judge panel against the same prompts and the
same provider configs as the canonical studies, but drops the web UI,
multi-account routing, quota tracking, and archiving tooling that repo also
has. Each study's `results/` already ships with the previously-collected raw
responses for this panel; running the scripts only fills gaps or adds judges,
it never overwrites an existing row.

## What's here

```
judge_query/       the collection engine (Python package)
configs/           shared apis.json / judges.json catalogs
arena_33k/         study.toml, prompts, raw dataset, results
mt_bench/          study.toml, prompts, raw dataset, results
pandalm/           study.toml, prompts, raw dataset, results
```

## 1. Set up Python

```bash
conda create -n judge-query python=3.11 -y
conda activate judge-query
python -m pip install -r requirements.txt
```

Python 3.11+ is recommended (uses the standard-library `tomllib`); on 3.10
`tomli` from `requirements.txt` covers it.

## 2. Install Ollama and pull the local judge models

19 of the 22 judges run locally via [Ollama](https://ollama.com); the
remaining 3 (OpenAI, Anthropic, Google) call each provider's paid batch API.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Launch the Ollama application, or run `ollama serve` if nothing is listening
yet, then pull every model tag the panel needs:

```bash
ollama pull gpt-oss:20b               # ollama-gpt-oss-20b-low
ollama pull gemma3n:e4b               # ollama-gemma-3n-e4b-direct
ollama pull gemma3:27b                # ollama-gemma-3-27b-direct
ollama pull llama3.3:70b              # ollama-llama-3.3-70b-direct
ollama pull qwen3:30b-a3b-q4_K_M      # ollama-qwen3-30b-a3b-direct / -low
ollama pull mistral-small3.2:24b      # ollama-mistral-small-3.2-24b-direct
ollama pull deepseek-r1:1.5b          # ollama-deepseek-r1-1.5b-low
ollama pull deepseek-r1:7b            # ollama-deepseek-r1-7b-low
ollama pull deepseek-r1:8b            # ollama-deepseek-r1-8b-low
ollama pull deepseek-r1:14b           # ollama-deepseek-r1-14b-low
ollama pull deepseek-r1:32b           # ollama-deepseek-r1-32b-low
ollama pull deepseek-r1:70b           # ollama-deepseek-r1-70b-low
ollama pull glm4:9b                   # ollama-glm4-9b-direct
ollama pull glm-4.7-flash:q4_K_M      # ollama-glm-4.7-flash-direct / -thinking
ollama pull phi4:14b                  # ollama-phi4-14b-direct
ollama pull stablelm2:12b             # ollama-stablelm2-12b-direct
ollama pull starling-lm:7b-alpha      # ollama-starling-lm-7b-direct
```

That's 16 distinct tags (~230 GB total; `deepseek-r1:70b` and `llama3.3:70b`
are ~42 GB each). Verify before running anything:

```bash
ollama --version
curl -fsS http://127.0.0.1:11434/api/tags
```

If a VPN or system proxy is active, bypass it for loopback traffic or Ollama
calls may fail with HTTP 502:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

An Ollama tag/quantization is part of a judge's scientific identity --
`ollama list` / `ollama show <tag>` report the exact digest if you need to
confirm it matches what produced the shipped `results/`.

## 3. Set up API keys (only needed for the 3 paid judges)

```bash
cp .env.example .env
```

Edit `.env` and fill in whichever keys you have:

- `OPENAI_API_KEY` -- for `openai-gpt-4.1-nano-batch-direct`
- `ANTROPIC_API_KEY` -- for `anthropic-claude-haiku-4.5-batch-direct` (this
  exact spelling, matching `configs/apis.json`)
- `GEMINI_API_KEY` -- for `google-gemini-3.1-flash-lite-batch-minimal`

`.env` is git-ignored and read only from your local disk; nothing in this kit
ever contains a literal key (`configs/apis.json` holds only the environment
variable *names* above, never values).

## 4. Run

Each study directory has a `run.sh` wrapper:

```bash
# Validate config/prompts/dataset resolution only, no network calls:
bash arena_33k/run.sh --dry-run

# Live run, all 22 judges in the study's panel:
bash arena_33k/run.sh

# A specific subset, and/or a small slice for testing:
bash arena_33k/run.sh --judges ollama-gemma-3-27b-direct,ollama-deepseek-r1-1.5b-low --max-records 5
```

Equivalent direct invocation from this directory:

```bash
python -m judge_query.run --study arena_33k/study.toml [--judges alias,...] [--max-records N] [--dry-run]
```

Results land in `<study>/results/<judge>/responses.jsonl` (one JSON object
per record x condition) and `<study>/results/<judge>/errors.jsonl` for any
failed jobs. Re-running is safe: each judge's existing `responses.jsonl` is
scanned first, and already-collected record/condition pairs are skipped.

Local Ollama judges run as a plain sequential loop (no concurrency, matching
the original studies). The 3 paid judges submit an asynchronous batch job to
the provider, poll until it completes, then parse the results -- this can
take anywhere from a few minutes to a few hours depending on the provider's
queue, and intermediate batch state is kept under
`<study>/batch-jobs/<judge>/` so a killed process can pick the job back up on
its next successful poll instead of resubmitting.

## What this kit intentionally omits

This is a data-collection-only distillation of the full `causal-judge`
pipeline. It has no SQLite success cache, no multi-account/multi-machine
routing, no cross-run quota ledger, no web UI, no archiving, and no bootstrap
analysis/report generation -- each judge here has exactly one configured
account, and `results/` already ships with the corresponding pre-computed
raw responses. If you need any of that (or want to regenerate a study from a
different judge panel), see the full `causal-judge` repository.
