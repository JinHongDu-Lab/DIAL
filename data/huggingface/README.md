---
pretty_name: DIAL LLM-judge panel
license: cc-by-nc-4.0
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - text-classification
tags:
  - llm-as-a-judge
  - pairwise-comparison
  - human-preference
  - position-bias
  - evaluation
arxiv: 2609.31215
configs:
  - config_name: arena_33k_responses
    default: true
    data_files: arena_33k/responses.parquet
  - config_name: arena_33k_benchmark
    data_files: arena_33k/data/arena_33k_ja_overlap.parquet
  - config_name: mt_bench_responses
    data_files: mt_bench/responses.parquet
  - config_name: mt_bench_benchmark
    data_files: mt_bench/data/mt_bench_prepared.parquet
  - config_name: pandalm_responses
    data_files: pandalm/responses.parquet
  - config_name: pandalm_benchmark
    data_files: pandalm/data/pandalm_prepared.parquet
---

# DIAL: an LLM-judge panel in both display orders

Data released with
[DIAL: Position-Debiased LLM Judges with Adaptive Human Preference Calibration](https://arxiv.org/abs/2609.31215)
(Cai, Fan, Chen, and Du, 2026).
Code: [github.com/JinHongDu-Lab/DIAL](https://github.com/JinHongDu-Lab/DIAL).

Every human-labelled pairwise comparison in three preference benchmarks was judged anonymously by a
panel of 22 LLM judges, once in the original display order and once with the two responses swapped.
The result is 429,879 judge responses over 9,882 comparisons, with the human label attached to every
row, for studying position bias, judge heterogeneity, and human alignment of LLM-as-a-judge.

| Config | Rows | Contents |
|---|---:|---|
| `arena_33k_benchmark` | 7,738 | Chatbot Arena comparisons (20 models) with human labels |
| `arena_33k_responses` | 337,105 | judge responses on those comparisons |
| `mt_bench_benchmark` | 1,199 | MT-Bench turn-1 comparisons (6 models) with human labels |
| `mt_bench_responses` | 51,561 | judge responses on those comparisons |
| `pandalm_benchmark` | 945 | PandaLM test comparisons (5 models) with human labels |
| `pandalm_responses` | 41,213 | judge responses on those comparisons |

## Loading

```python
from datasets import load_dataset

responses = load_dataset("CSML/DIAL", "mt_bench_responses", split="train")
benchmark = load_dataset("CSML/DIAL", "mt_bench_benchmark", split="train")
```

Join `responses["record_id"]` to `benchmark["id"]` to recover the prompt and the two responses a judge saw.
Every file is plain Parquet, so `pandas.read_parquet` on the raw files works as well.

## Benchmarks

| Config prefix | Source | Human label |
|---|---|---|
| `arena_33k` | a 7,738-record subset of [Chatbot Arena conversations (33K)](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | the Arena vote |
| `mt_bench` | [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments), turn 1 | majority vote over annotators per (question, model pair) |
| `pandalm` | [PandaLM](https://github.com/WeOpenML/PandaLM) human-annotated test set | majority vote over 3 annotators |

Each benchmark row has `id`, `model_a`, `model_b`, `winner` (`model_a`, `model_b`, `tie`, or, for
Arena, `both_bad`), `prompt`, `response_a`, and `response_b`, plus source-specific columns such as
`language` (Arena; 89% English), `question_id`, and `num_human_annotators`; the Arena file also carries
auxiliary single-judge fields (`ja_judge_*`) from the subset construction that the paper does not use.

## Judge responses

The panel has 22 judges: 19 open-weight models served locally by Ollama and three paid batch APIs
(`openai-gpt-4.1-nano-batch-direct`, `anthropic-claude-haiku-4.5-batch-direct`,
`google-gemini-3.1-flash-lite-batch-minimal`).
`configs/judges.json` records each model tag, quantization, and decoding setting; `<dataset>/study.toml`
records the study design and `<dataset>/prompts/` the system and user prompts.
Every record is judged in both display orders (`original_anonymous` and `swapped_anonymous`), always
with model names hidden, and the judge answers with one line `<choice> <confidence>`.

| Column | Meaning |
|---|---|
| `record_id` | benchmark row (`id` in the benchmark config) |
| `panel_judge` | judge alias in the study panel; the identity to group by |
| `judge` | alias of the exact query (a few rows carry a `-64tok` re-query suffix) |
| `judge_model`, `judge_provider`, `inference` | model tag, backend, and decoding settings (JSON) |
| `condition_id` | `original_anonymous` or `swapped_anonymous` |
| `response_order_default` | true when `model_a` was displayed first |
| `choice` | displayed slot chosen: `a` (first), `b` (second), `c` (tie) |
| `canonical_verdict` | the same verdict mapped back to `model_a` / `model_b` / `tie` |
| `confidence` | the judge's self-reported confidence, 1 to 5 |
| `raw_response` | the judge's raw output |
| `human_winner` | human label of the record |
| `prompt_hash`, `*_fingerprint` | hashes of the exact prompt and configuration used |
| `usage`, `latency_seconds`, `completed_at`, `provider_*` | collection metadata |

Position-bias quantities read `choice`; agreement with `human_winner` reads `canonical_verdict`.
The paper excludes `ollama-stablelm2-12b-direct` (tie rate above 50% on every benchmark), leaving
21 judges, and drops LLM ties and human ties or `both_bad` labels, which gives the 410K judgments
reported there; the release keeps everything.

## Collection

Responses were collected between August and September 2026 with the kit in
[`data/judge_query`](https://github.com/JinHongDu-Lab/DIAL/tree/main/data) of the code repository.
Open-weight judges ran through a local Ollama server at the tags listed in `configs/judges.json`;
the tag and quantization are part of the judge's identity.
Paid judges were queried through the providers' batch APIs at temperature 0 where the API allows it.

## Licensing

The judge responses, prompts, and study configuration are released under CC BY-NC 4.0.
The benchmark files redistribute third-party material under their own terms: Chatbot Arena user
prompts are CC BY 4.0 and model outputs are CC BY-NC 4.0 (see the source dataset's terms), the MT-Bench
human judgments are CC BY 4.0, and PandaLM is Apache 2.0.
The non-commercial term on the Arena model outputs is what sets the license of this dataset as a whole.

## Citation

```bibtex
@article{cai2026dial,
  title   = {{DIAL}: Position-Debiased {LLM} Judges with Adaptive Human Preference Calibration},
  author  = {Cai, Zesheng and Fan, Yingqi and Chen, Sichang and Du, Jin-Hong},
  journal = {arXiv preprint arXiv:2609.31215},
  year    = {2026}
}
```

Please also cite the source benchmarks:

```bibtex
@inproceedings{zheng2023mtbench,
  title     = {Judging {LLM}-as-a-Judge with {MT}-Bench and Chatbot Arena},
  author    = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {46595--46623},
  year      = {2023}
}
@inproceedings{chiang2024chatbot,
  title     = {Chatbot Arena: An Open Platform for Evaluating {LLM}s by Human Preference},
  author    = {Chiang, Wei-Lin and Zheng, Lianmin and Sheng, Ying and Angelopoulos, Anastasios N. and Li, Tianle and Li, Dacheng and Zhu, Banghua and Zhang, Hao and Jordan, Michael I. and Gonzalez, Joseph E. and Stoica, Ion},
  booktitle = {International Conference on Machine Learning},
  pages     = {8359--8388},
  year      = {2024}
}
@inproceedings{wang2024pandalm,
  title     = {{PandaLM}: An Automatic Evaluation Benchmark for {LLM} Instruction Tuning Optimization},
  author    = {Wang, Yidong and Yu, Zhuohao and Yao, Wenjin and Zeng, Zhengran and Yang, Linyi and Wang, Cunxiang and Chen, Hao and Jiang, Chaoya and Xie, Rui and Wang, Jindong and Xie, Xing and Ye, Wei and Zhang, Shikun and Zhang, Yue},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024}
}
```
