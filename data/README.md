# Data

```text
<dataset>/data/*.parquet     prepared benchmark: one row per human-labelled comparison
<dataset>/responses.parquet  every collected judge response (both display orders)
<dataset>/study.toml         judge panel, prompts, and display-order conditions
<dataset>/prompts/           judge system and user prompts
configs/                     judge (judges.json) and provider (apis.json) catalogs
judge_query/                 collection kit
```

## Benchmarks

| Dataset | Source | Records | Items |
|---|---|---|---|
| `arena_33k` | Chatbot Arena conversations (33K), restricted to the records covered by the Judge-Aware Ranking framework's judge-vs-human comparisons | 7,738 | 20 |
| `mt_bench` | MT-Bench human judgments, turn 1, majority vote over annotators per (question, model pair) | 1,199 | 6 |
| `pandalm` | PandaLM human-annotated test set, majority vote over 3 annotators | 945 | 5 |

Each prepared row has `id`, `model_a`, `model_b`, `winner` (the human label), `prompt`,
`response_a`, and `response_b`.

## Judge responses

The panel has 22 judges: 19 open-weight models served locally by Ollama and three paid batch APIs
(`openai-gpt-4.1-nano-batch-direct`, `anthropic-claude-haiku-4.5-batch-direct`,
`google-gemini-3.1-flash-lite-batch-minimal`); `configs/judges.json` records each model tag and
inference setting. Every record is judged anonymously in its original and swapped display order.

Main columns of `responses.parquet`:

| Column | Meaning |
|---|---|
| `record_id` | prepared-benchmark record |
| `panel_judge` | judge alias in the study panel |
| `condition_id` | `original_anonymous` or `swapped_anonymous` |
| `response_order_default` | true when `model_a` was displayed first |
| `choice` | displayed slot chosen: `a` (first), `b` (second), `c` (tie) |
| `canonical_verdict` | the same verdict mapped back to `model_a` / `model_b` / `tie` |
| `human_winner` | human label of the record |
| `raw_response` | the judge's raw output |
| `inference` | decoding and reasoning settings (JSON) |

`experiments/real_data/prepare.py` turns this file into the analysis table.

## Collecting responses

```bash
cd data
python -m pip install -r requirements.txt
cp .env.example .env                              # keys for the three paid judges only
bash mt_bench/run.sh --dry-run                    # validate, no API calls
bash mt_bench/run.sh --judges ollama-gemma-3-27b-direct --max-records 5
python -m judge_query.export arena_33k mt_bench pandalm
```

Ollama judges need a local server (`ollama serve`) and the model tags in `configs/judges.json`
(`ollama pull <tag>`); the tag and quantization are part of the judge's identity. Paid judges
submit provider batch jobs and resume from `<study>/batch-jobs/`. Raw output goes to
`<study>/results/<judge>/responses.jsonl` (not tracked); a rerun skips the (record, condition)
pairs already collected. `judge_query.export` packs these files into `<study>/responses.parquet`.
