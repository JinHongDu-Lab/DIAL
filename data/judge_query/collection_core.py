"""Verdict parsing, prompt building, and job bookkeeping.

Copied verbatim (unchanged) from causal_judge/collect.py in the main
`causal-judge` repository, keeping only the pure, self-contained pieces that
do not depend on the SQLite success cache, account routing, or usage
ledger. This guarantees byte-identical verdict scoring and prompt
construction between this distilled kit and the canonical studies.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import pyarrow.parquet as pq

from .config import JudgeConfig, canonical_json, fingerprint
from .prompts import PromptSpec

PLAIN_VERDICT_PATTERN = re.compile(
    r"^\s*([abcd])\s+([+-]?\d+)\s*$",
    re.IGNORECASE,
)
VERDICT_PATTERNS = (
    PLAIN_VERDICT_PATTERN,
    re.compile(
        r"^\s*<choice>\s*([abcd])\s*</choice>\s*"
        r"<confidence>\s*([+-]?\d+)\s*</confidence>\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*<choice>\s*([abcd])\s*"
        r"<confidence>\s*([+-]?\d+)\s*</choice>\s*$",
        re.IGNORECASE,
    ),
)
# Some judges (observed on hku-claude-opus-4.6-low with reasoning effort
# "low") ignore the "one line only" instruction and emit a bare choice
# letter followed by rationale, sometimes with a "Confidence: N" label
# buried in that rationale. Recover the verdict from that shape without
# loosening the stricter patterns above (which still reject a choice
# letter embedded mid-sentence, e.g. "The answer is a 5").
BARE_CHOICE_PATTERN = re.compile(r"^\s*([abcd])[.:)]?\s*$", re.IGNORECASE)
CONFIDENCE_LABEL_PATTERN = re.compile(
    r"(?im)^\s*confidence\s*[:=]\s*([+-]?\d+)\b"
)
# Reasoning models sometimes emit the requested verdict after a rationale or
# omit the closing XML-like tags. Treat only line-oriented, explicit
# choice/confidence pairs as recoverable; prose mentions remain invalid.
TAGGED_VERDICT_PATTERN = re.compile(
    r"(?im)^[ \t]*<choice>[ \t]*([abcd])[ \t]*(?:</choice>)?[ \t]*"
    r"(?:\r?\n[ \t]*|[ \t]*)"
    r"<confidence>[ \t]*([+-]?\d+)[ \t]*(?:</confidence>)?[ \t]*"
    r"(?:</choice>)?[ \t]*$"
)
LABELED_VERDICT_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:\*\*)?choice"
    r"[ \t]*[:=](?:\*\*)?[ \t]*(?:\*\*)?([abcd])(?:\*\*)?[ \t]*"
    r"(?:\r?\n[ \t]*|[ \t]+)"
    r"(?:[-*][ \t]*)?(?:\*\*)?confidence"
    r"[ \t]*[:=](?:\*\*)?[ \t]*(?:\*\*)?([+-]?\d+)"
    r"(?:\*\*)?[ \t]*$"
)
ANSWER_VERDICT_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:final[ \t]+)?answer[ \t]*[:=][ \t]*"
    r"([abcd])[ \t]+([+-]?\d+)[ \t]*$"
)
FINAL_ANSWER_MARKER_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*)?final\s+"
    r"(?:answer|verdict|judg(?:e)?ment)(?:\*\*)?\s*:?\s*$"
)
FINAL_COMPACT_VERDICT_PATTERN = re.compile(
    r"(?im)^[ \t]*([abcd])[ \t]*([1-5])[ \t]*$"
)
CHOICE_TOKEN_PATTERNS = (
    re.compile(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?<cho(?:ice|ise)>\s*([abcd])\b"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:\*\*)?choice"
        r"[ \t]*[:=](?:\*\*)?[ \t]*[\"'`*]*([abcd])\b"
    ),
)
CONFIDENCE_TOKEN_PATTERNS = (
    re.compile(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?<confidence>\s*([+-]?\d+)\b"
    ),
    re.compile(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?(?:\*\*)?\(?confidence\)?"
        r"[ \t]*(?::|=)?(?:\*\*)?[ \t]*(?:\*\*)?([+-]?\d+)\b"
    ),
)
# Some judges (observed on ollama-phi4-14b-direct) wrap an otherwise-correct
# verdict in a Markdown code span, e.g. "`a 5`" instead of "a 5". The
# backreference requires matching backtick counts on both ends, so this only
# strips a span wrapping the *entire* string and cannot loosen matching for a
# verdict embedded mid-sentence (e.g. "The answer is `a 5`").
CODE_SPAN_PATTERN = re.compile(r"^\s*(`{1,3})(.*?)\1\s*$", re.DOTALL)


def _strip_code_span(text: str) -> str:
    """Strip a Markdown code span/fence wrapping the entire text."""

    match = CODE_SPAN_PATTERN.fullmatch(text)
    return match.group(2).strip() if match else text


REQUIRED_COLUMNS = {
    "id",
    "evaluation_session_id",
    "evaluation_order",
    "model_a",
    "model_b",
    "winner",
    "prompt",
    "response_a",
    "response_b",
}
OPTIONAL_COLUMNS = {"language", "is_code", "main_category", "prompts_match"}
Record: TypeAlias = dict[str, Any]
Result: TypeAlias = dict[str, Any]
ResumeKey: TypeAlias = tuple[str, str, str, str, bool, bool, int]


class InvalidVerdict(ValueError):
    """Raised when a judge does not return the required verdict format."""


@dataclass(frozen=True)
class Job:
    """One record, repeat, treatment condition, and logical judge."""

    record: Record
    query_index: int
    condition_id: str
    response_order_default: bool
    is_anonymous: bool
    displayed_a_original_side: str
    displayed_b_original_side: str
    messages: list[dict[str, str]]
    prompt_id: str
    prompt_spec_fingerprint: str
    prompt_hash: str
    resume_key: ResumeKey

    @property
    def stable_key(self) -> str:
        """Return a stable string identifying this job."""

        return canonical_json(self.resume_key)


def effective_judge_fingerprint(judge: JudgeConfig, prompt_fingerprint: str) -> str:
    """Return the collection fingerprint for a judge and prompt."""

    return fingerprint(
        {
            "judge_fingerprint": judge.experiment_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
        }
    )


def resolve_parquet_path(path: Path) -> Path:
    """Resolve a file or single-Parquet directory."""

    if path.is_file() and path.suffix == ".parquet":
        return path
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if len(files) == 1:
            return files[0]
        if not files:
            raise ValueError(f"no parquet file found under {path}")
        raise ValueError(f"expected one parquet file under {path}, found {len(files)}")
    raise ValueError(f"dataset path does not exist or is not parquet: {path}")


def load_records(
    path: Path,
    start_index: int = 0,
    max_records: int | None = None,
) -> list[Record]:
    """Load the selected dataset slice from Parquet."""

    parquet_path = resolve_parquet_path(path)
    schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    missing = REQUIRED_COLUMNS - schema_names
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    columns = sorted(REQUIRED_COLUMNS | (OPTIONAL_COLUMNS & schema_names))
    table = pq.read_table(parquet_path, columns=columns)
    if start_index >= table.num_rows:
        return []
    length = table.num_rows - start_index if max_records is None else max_records
    records = table.slice(start_index, length).to_pylist()
    for offset, record in enumerate(records):
        record["_dataset_index"] = start_index + offset
    return records


def validate_record(record: Record) -> None:
    """Validate fields used to build a judge prompt."""

    for field in REQUIRED_COLUMNS:
        if record.get(field) is None:
            raise ValueError(f"missing value for {field}")
    for field in ("id", "model_a", "model_b", "prompt", "response_a", "response_b"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if record["winner"] not in {"model_a", "model_b", "tie", "both_bad"}:
        raise ValueError(f"invalid human winner: {record['winner']!r}")
    if record.get("prompts_match") is False:
        raise ValueError("prompt_a and prompt_b do not match")


def build_judge_messages(
    record: Record,
    response_order_default: bool,
    is_anonymous: bool,
    prompt_spec: PromptSpec,
) -> tuple[list[dict[str, str]], str, str]:
    """Build a judge prompt while preserving the displayed-side mapping.

    Returns
    -------
    messages, displayed_a_side, displayed_b_side
        Provider-neutral messages and the original dataset side shown as A/B.
    """

    validate_record(record)
    if response_order_default:
        side_a, side_b = "model_a", "model_b"
        response_a, response_b = record["response_a"], record["response_b"]
        model_a, model_b = record["model_a"], record["model_b"]
    else:
        side_a, side_b = "model_b", "model_a"
        response_a, response_b = record["response_b"], record["response_a"]
        model_a, model_b = record["model_b"], record["model_a"]

    label_a = "Assistant A" if is_anonymous else f"Assistant A (model: {model_a})"
    label_b = "Assistant B" if is_anonymous else f"Assistant B (model: {model_b})"
    user_prompt = prompt_spec.render_user(
        {
            "question": record["prompt"],
            "assistant_a_label": label_a,
            "assistant_a_response": response_a,
            "assistant_b_label": label_b,
            "assistant_b_response": response_b,
        }
    )
    return (
        [
            {"role": "system", "content": prompt_spec.system},
            {"role": "user", "content": user_prompt},
        ],
        side_a,
        side_b,
    )


def parse_verdict(text: str) -> tuple[str, int]:
    """Parse a verdict and clamp an integer confidence to the 1-5 scale."""

    text = _strip_code_span(text)
    for pattern in VERDICT_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            try:
                confidence = int(match.group(2))
            except ValueError:
                continue
            return match.group(1).lower(), min(5, max(1, confidence))
    final_markers = list(FINAL_ANSWER_MARKER_PATTERN.finditer(text))
    if final_markers:
        final_section = text[final_markers[-1].end() :]
        final_candidates = {
            (
                match.group(1).lower(),
                min(5, max(1, int(match.group(2)))),
            )
            for match in FINAL_COMPACT_VERDICT_PATTERN.finditer(final_section)
        }
        if len(final_candidates) == 1:
            return next(iter(final_candidates))
    candidates: set[tuple[str, int]] = set()
    for line in text.splitlines():
        match = PLAIN_VERDICT_PATTERN.fullmatch(_strip_code_span(line))
        if match:
            candidates.add(
                (
                    match.group(1).lower(),
                    min(5, max(1, int(match.group(2)))),
                )
            )
    for pattern in (
        TAGGED_VERDICT_PATTERN,
        LABELED_VERDICT_PATTERN,
        ANSWER_VERDICT_PATTERN,
    ):
        for match in pattern.finditer(text):
            candidates.add(
                (
                    match.group(1).lower(),
                    min(5, max(1, int(match.group(2)))),
                )
            )
    explicit_choices = {
        match.group(1).lower()
        for pattern in CHOICE_TOKEN_PATTERNS
        for match in pattern.finditer(text)
    }
    explicit_confidences = {
        min(5, max(1, int(match.group(1))))
        for pattern in CONFIDENCE_TOKEN_PATTERNS
        for match in pattern.finditer(text)
    }
    if len(explicit_choices) == 1 and len(explicit_confidences) == 1:
        candidates.add(
            (next(iter(explicit_choices)), next(iter(explicit_confidences)))
        )
    first_nonempty_line = next(
        (line for line in text.splitlines() if line.strip()),
        "",
    )
    first_nonempty_line = _strip_code_span(first_nonempty_line)
    bare_choice = BARE_CHOICE_PATTERN.fullmatch(first_nonempty_line)
    if bare_choice:
        confidence_label = CONFIDENCE_LABEL_PATTERN.search(text)
        if confidence_label:
            candidates.add(
                (
                    bare_choice.group(1).lower(),
                    min(5, max(1, int(confidence_label.group(1)))),
                )
            )
    if len(candidates) == 1:
        return next(iter(candidates))
    raise InvalidVerdict(f"invalid verdict: {text[:200]!r}")


def canonical_verdict(
    choice: str,
    displayed_a_side: str,
    displayed_b_side: str,
) -> str:
    """Map a displayed choice back to the original dataset side."""

    if choice == "a":
        return displayed_a_side
    if choice == "b":
        return displayed_b_side
    return {"c": "tie", "d": "both_bad"}[choice]


def make_resume_key(
    record_id: str,
    judge: str,
    model: str,
    experiment_fingerprint: str,
    response_order_default: bool,
    is_anonymous: bool,
    query_index: int,
) -> ResumeKey:
    """Build the successful-job checkpoint key."""

    return (
        record_id,
        judge,
        model,
        experiment_fingerprint,
        response_order_default,
        is_anonymous,
        query_index,
    )


def resume_key_from_result(result: Result) -> ResumeKey | None:
    """Read a resume key from a saved result."""

    fields = (
        "record_id",
        "judge",
        "judge_model",
        "config_fingerprint",
        "response_order_default",
        "is_anonymous",
        "query_index",
    )
    if any(field not in result for field in fields):
        return None
    values = tuple(result[field] for field in fields)
    return values  # type: ignore[return-value]


def load_resume_keys(path: Path) -> set[ResumeKey]:
    """Load valid successful checkpoint keys from JSONL."""

    keys: set[ResumeKey] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                key = resume_key_from_result(result)
                if key is not None:
                    keys.add(key)
    return keys


def ensure_appendable(path: Path) -> None:
    """Create a JSONL parent and repair a missing final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.seek(0, 2)
            handle.write(b"\n")


def build_job(
    record: Record,
    query_index: int,
    response_order_default: bool,
    is_anonymous: bool,
    config: JudgeConfig,
    prompt_spec: PromptSpec,
    condition_id: str | None = None,
) -> Job:
    """Build one normalized collection job."""

    messages, side_a, side_b = build_judge_messages(
        record, response_order_default, is_anonymous, prompt_spec
    )
    prompt_hash = hashlib.sha256(canonical_json(messages).encode()).hexdigest()
    return Job(
        record=record,
        query_index=query_index,
        condition_id=condition_id or (
            ("original" if response_order_default else "swapped")
            + "_"
            + ("anonymous" if is_anonymous else "named")
        ),
        response_order_default=response_order_default,
        is_anonymous=is_anonymous,
        displayed_a_original_side=side_a,
        displayed_b_original_side=side_b,
        messages=messages,
        prompt_id=prompt_spec.prompt_id,
        prompt_spec_fingerprint=prompt_spec.fingerprint,
        prompt_hash=prompt_hash,
        resume_key=make_resume_key(
            record["id"],
            config.alias,
            config.model,
            config.experiment_fingerprint,
            response_order_default,
            is_anonymous,
            query_index,
        ),
    )


def base_result(
    job: Job,
    config: JudgeConfig,
    account_id: str | None = None,
) -> Result:
    """Build shared success/error metadata."""

    record = job.record
    return {
        "record_id": record["id"],
        "dataset_index": record.get("_dataset_index"),
        "evaluation_session_id": record["evaluation_session_id"],
        "evaluation_order": record["evaluation_order"],
        "human_winner": record["winner"],
        "model_a": record["model_a"],
        "model_b": record["model_b"],
        "language": record.get("language"),
        "is_code": record.get("is_code"),
        "main_category": record.get("main_category"),
        "judge": config.alias,
        "judge_model": config.model,
        "judge_provider": config.api.adapter,
        "api_id": config.api.api_id,
        "api_base": config.api.base_url,
        "account_id": account_id,
        "inference": {
            "temperature": config.inference.temperature,
            "max_output_tokens": config.inference.max_output_tokens,
            "reasoning": {
                "mode": config.inference.reasoning.mode,
                "effort": config.inference.reasoning.effort,
            },
        },
        "provider_options": config.provider_options,
        "config_fingerprint": config.experiment_fingerprint,
        "experiment_fingerprint": config.experiment_fingerprint,
        "routing_fingerprint": config.routing_fingerprint,
        "generation_seed": config.generation_seed,
        "prompt_version": job.prompt_id,
        "prompt_spec_fingerprint": job.prompt_spec_fingerprint,
        "prompt_hash": job.prompt_hash,
        "condition_id": job.condition_id,
        "response_order_default": job.response_order_default,
        "is_anonymous": job.is_anonymous,
        "displayed_a_original_side": job.displayed_a_original_side,
        "displayed_b_original_side": job.displayed_b_original_side,
        "query_index": job.query_index,
    }
