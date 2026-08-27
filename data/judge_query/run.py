"""Flat data-collection script for one study.

Queries every requested judge in a study.toml sequentially against the
shared apis.json/judges.json catalogs: a plain async loop against the
ollama-local adapter for local judges, and the provider batch clients for
the three paid batch judges (OpenAI, Anthropic, Gemini). There is no SQLite
success cache, account router, or usage ledger -- each judge in this
panel has exactly one configured account, and resuming is a plain scan of
the judge's existing responses.jsonl.

Usage
-----
    python -m judge_query.run --study arena_33k/study.toml [--dry-run]
    python -m judge_query.run --study arena_33k/study.toml --judges ollama-gemma-3-27b-direct --max-records 2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import pyarrow.parquet as pq

from .anthropic_batch import (
    AnthropicBatchClient,
    SUCCESS_STATES as ANTHROPIC_SUCCESS_STATES,
    iter_batch_results as iter_anthropic_batch_results,
    operation_state as anthropic_operation_state,
    split_anthropic_requests,
    wait_for_batch as wait_for_anthropic_batch,
)
from .collection_core import (
    InvalidVerdict,
    Job,
    base_result,
    build_job,
    canonical_verdict,
    effective_judge_fingerprint,
    ensure_appendable,
    load_records,
    load_resume_keys,
    parse_verdict,
)
from .config import JudgeConfig, file_sha256, load_config, load_environment_file
from .gemini_batch import (
    GeminiBatchClient,
    SUCCESS_STATES as GEMINI_SUCCESS_STATES,
    iter_batch_results as iter_gemini_batch_results,
    operation_state as gemini_operation_state,
    result_file_name as gemini_result_file_name,
    split_jsonl as split_gemini_jsonl,
    wait_for_batch as wait_for_gemini_batch,
)
from .openai_batch import (
    OpenAIBatchClient,
    SUCCESS_STATES as OPENAI_SUCCESS_STATES,
    error_file_name as openai_error_file_name,
    iter_batch_results as iter_openai_batch_results,
    operation_state as openai_operation_state,
    result_file_name as openai_result_file_name,
    split_openai_jsonl,
    wait_for_batch as wait_for_openai_batch,
)
from .prompts import load_prompt_spec
from .providers import (
    ErrorCategory,
    ProviderError,
    anthropic_messages_payload,
    anthropic_provider_result,
    gemini_generate_payload,
    gemini_provider_result,
    get_adapter,
    openai_responses_payload,
    openai_responses_provider_result,
)
from .status import utc_now

# Both conditions this kit's three studies share: always-anonymous labels,
# original vs. swapped response order. Hardcoded rather than parsed from
# study.toml's [[factors]]/[[conditions]] tables, which are identical across
# arena_33k, mt_bench, and pandalm.
CONDITIONS = (
    {"condition_id": "original_anonymous", "response_order_default": True, "is_anonymous": True},
    {"condition_id": "swapped_anonymous", "response_order_default": False, "is_anonymous": True},
)

BATCH_ADAPTERS = {"openai_batch", "anthropic_batch", "google_gemini_batch"}


class Study:
    """The subset of study.toml fields this script needs."""

    def __init__(self, path: Path) -> None:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        study = data["study"]
        base = path.parent

        def resolve(value: str) -> Path:
            return (base / value).resolve()

        self.study_id: str = study["id"]
        self.dataset = resolve(study["dataset"])
        self.dataset_sha256: str | None = study.get("dataset_sha256")
        self.dataset_rows: int | None = study.get("dataset_rows")
        self.api_config = resolve(study["api_config"])
        self.judge_config = resolve(study["judge_config"])
        self.judges: list[str] = list(study["judges"])
        self.system_prompt = resolve(study["system_prompt"])
        self.user_prompt_template = resolve(study["user_prompt_template"])
        self.results_dir = resolve(study.get("results_dir", "results"))
        collection = data.get("collection", {})
        self.timeout: float = float(collection.get("timeout", 120))
        self.max_retries: int = int(collection.get("max_retries", 3))


def validate_dataset(study: Study) -> None:
    if not study.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {study.dataset}")
    if study.dataset_rows is not None:
        rows = pq.ParquetFile(study.dataset).metadata.num_rows
        if rows != study.dataset_rows:
            raise SystemExit(
                f"dataset row count is {rows}, expected {study.dataset_rows}"
            )
    if study.dataset_sha256 is not None:
        digest = file_sha256(study.dataset)
        if digest != study.dataset_sha256:
            raise SystemExit("dataset SHA-256 does not match study.dataset_sha256")


def write_or_check_metadata(
    path: Path,
    *,
    study: Study,
    judge: JudgeConfig,
    prompt_fingerprint: str,
) -> None:
    metadata = {
        "version": 1,
        "study_id": study.study_id,
        "dataset_sha256": study.dataset_sha256,
        "judge": judge.alias,
        "judge_model": judge.model,
        "config_fingerprint": effective_judge_fingerprint(judge, prompt_fingerprint),
        "prompt_fingerprint": prompt_fingerprint,
    }
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != metadata:
            raise SystemExit(
                f"existing results for judge {judge.alias!r} use different "
                "scientific settings than this study/prompt/config; "
                "resolve before continuing"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_jobs(records: list[dict[str, Any]], judge: JudgeConfig, prompt_spec: Any) -> list[Job]:
    jobs: list[Job] = []
    for record in records:
        for condition in CONDITIONS:
            jobs.append(
                build_job(
                    record,
                    0,
                    condition["response_order_default"],
                    condition["is_anonymous"],
                    judge,
                    prompt_spec,
                    condition_id=condition["condition_id"],
                )
            )
    return jobs


def append_result(path: Path, result: dict[str, Any]) -> None:
    ensure_appendable(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")


def _failure(
    job: Job,
    judge: JudgeConfig,
    account_id: str | None,
    *,
    stage: str,
    error: BaseException,
    category: str,
) -> dict[str, Any]:
    failure = base_result(job, judge, account_id)
    failure.update(
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "error_category": category,
            "error": str(error)[:4_000],
            "attempts": 1,
            "failed_at": utc_now(),
        }
    )
    return failure


async def run_ollama_judge(
    judge: JudgeConfig,
    jobs: list[Job],
    output_path: Path,
    error_path: Path,
    *,
    timeout: float,
    max_retries: int,
    dry_run: bool,
) -> None:
    completed = load_resume_keys(output_path)
    pending = [job for job in jobs if job.resume_key not in completed]
    print(f"{judge.alias}: {len(pending)} of {len(jobs)} jobs pending")
    if dry_run or not pending:
        return
    adapter = get_adapter(judge.api.adapter)
    account = judge.accounts[0]
    api_key = os.environ.get(account.api_key_env) if account.api_key_env else None
    for index, job in enumerate(pending, start=1):
        last_error: BaseException | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = await adapter.complete(judge, account, api_key, job.messages, timeout)
                choice, confidence = parse_verdict(response.text)
            except (ProviderError, InvalidVerdict) as error:
                last_error = error
                if attempt >= max_retries + 1:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 30))
                continue
            result = base_result(job, judge, account.account_id)
            result.update(
                {
                    "choice": choice,
                    "canonical_verdict": canonical_verdict(
                        choice, job.displayed_a_original_side, job.displayed_b_original_side
                    ),
                    "confidence": confidence,
                    "raw_response": response.text,
                    "usage": response.usage,
                    "cost": None,
                    "provider_response_id": response.response_id,
                    "attempts": attempt,
                    "completed_at": utc_now(),
                }
            )
            append_result(output_path, result)
            last_error = None
            break
        if last_error is not None:
            category = (
                last_error.category.value
                if isinstance(last_error, ProviderError)
                else "invalid_response"
            )
            append_result(
                error_path,
                _failure(job, judge, account.account_id, stage="collect", error=last_error, category=category),
            )
        if index % 200 == 0:
            print(f"{judge.alias}: {index}/{len(pending)}")


async def run_openai_batch_judge(
    judge: JudgeConfig,
    jobs: list[Job],
    output_path: Path,
    error_path: Path,
    batch_dir: Path,
    *,
    timeout: float,
    dry_run: bool,
) -> None:
    completed = load_resume_keys(output_path)
    pending = [job for job in jobs if job.resume_key not in completed]
    print(f"{judge.alias}: {len(pending)} of {len(jobs)} jobs pending (OpenAI batch)")
    if dry_run or not pending:
        return
    account = judge.accounts[0]
    api_key = os.environ.get(account.api_key_env or "")
    if not api_key:
        raise SystemExit(f"environment variable {account.api_key_env!r} is required for {judge.alias}")
    keyed = {hashlib.sha256(job.stable_key.encode()).hexdigest()[:32]: job for job in pending}
    rows = [
        (key, openai_responses_payload(judge, job.messages, explicit_disabled_reasoning=True))
        for key, job in keyed.items()
    ]
    split_kwargs: dict[str, Any] = {"encoding_name": "o200k_base"}
    if judge.limits.batch_max_requests is not None:
        split_kwargs["max_requests"] = judge.limits.batch_max_requests
    chunks = split_openai_jsonl(batch_dir, rows, **split_kwargs)
    client = OpenAIBatchClient(
        api_key,
        base_url=judge.api.base_url or "https://api.openai.com",
        timeout=judge.request_timeout or timeout,
    )
    for chunk in chunks:
        chunk_jobs = {key: keyed[key] for key in chunk.keys}
        display_name = f"judge-query-{judge.alias}-{chunk.index}"[:128]
        print(f"{judge.alias}: submitting batch chunk {chunk.index + 1}/{len(chunks)} ({len(chunk.keys)} requests)")
        file_name = await client.upload_file(chunk.path, display_name)
        operation = await client.create_batch(judge.model, file_name, display_name)
        final = await wait_for_openai_batch(client, operation["id"])
        state = openai_operation_state(final)
        if state not in OPENAI_SUCCESS_STATES:
            for job in chunk_jobs.values():
                error = ValueError(f"OpenAI batch ended in state {state!r}")
                append_result(
                    error_path,
                    _failure(job, judge, account.account_id, stage="batch", error=error, category="batch_failed"),
                )
            continue
        content = b""
        output_name = openai_result_file_name(final)
        error_name = openai_error_file_name(final)
        if output_name:
            content += await client.download_results(output_name)
        if error_name:
            content += await client.download_results(error_name)
        seen: set[str] = set()
        async for row in iter_openai_batch_results(content):
            key = row.get("custom_id")
            if not isinstance(key, str) or key not in chunk_jobs or key in seen:
                continue
            seen.add(key)
            job = chunk_jobs[key]
            try:
                if isinstance(row.get("error"), dict):
                    raise ValueError(json.dumps(row["error"], ensure_ascii=False))
                response_value = row.get("response")
                if not isinstance(response_value, dict):
                    raise ValueError("batch result omitted response")
                status_code = response_value.get("status_code")
                body = response_value.get("body")
                if not isinstance(body, dict) or not isinstance(status_code, int) or not (200 <= status_code < 300):
                    raise ValueError(f"OpenAI batch response returned HTTP {status_code}")
                response = openai_responses_provider_result(body, status_code=status_code)
                choice, confidence = parse_verdict(response.text)
            except (ProviderError, InvalidVerdict, ValueError) as error:
                append_result(
                    error_path,
                    _failure(job, judge, account.account_id, stage="parse", error=error, category="invalid_response"),
                )
                continue
            result = base_result(job, judge, account.account_id)
            result.update(
                {
                    "choice": choice,
                    "canonical_verdict": canonical_verdict(
                        choice, job.displayed_a_original_side, job.displayed_b_original_side
                    ),
                    "confidence": confidence,
                    "raw_response": response.text,
                    "usage": response.usage,
                    "cost": None,
                    "provider_response_id": response.response_id,
                    "provider_batch_id": operation["id"],
                    "attempts": 1,
                    "completed_at": utc_now(),
                }
            )
            append_result(output_path, result)


async def run_anthropic_batch_judge(
    judge: JudgeConfig,
    jobs: list[Job],
    output_path: Path,
    error_path: Path,
    batch_dir: Path,
    *,
    timeout: float,
    dry_run: bool,
) -> None:
    completed = load_resume_keys(output_path)
    pending = [job for job in jobs if job.resume_key not in completed]
    print(f"{judge.alias}: {len(pending)} of {len(jobs)} jobs pending (Anthropic batch)")
    if dry_run or not pending:
        return
    account = judge.accounts[0]
    api_key = os.environ.get(account.api_key_env or "")
    if not api_key:
        raise SystemExit(f"environment variable {account.api_key_env!r} is required for {judge.alias}")
    keyed = {hashlib.sha256(job.stable_key.encode()).hexdigest()[:32]: job for job in pending}
    rows = [(key, anthropic_messages_payload(judge, job.messages)) for key, job in keyed.items()]
    split_kwargs: dict[str, Any] = {}
    if judge.limits.batch_max_requests is not None:
        split_kwargs["max_requests"] = judge.limits.batch_max_requests
    chunks = split_anthropic_requests(batch_dir, rows, **split_kwargs)
    client = AnthropicBatchClient(
        api_key,
        base_url=judge.api.base_url or "https://api.anthropic.com",
        timeout=judge.request_timeout or timeout,
    )
    for chunk in chunks:
        chunk_jobs = {key: keyed[key] for key in chunk.keys}
        print(f"{judge.alias}: submitting batch chunk {chunk.index + 1}/{len(chunks)} ({len(chunk.keys)} requests)")
        operation = await client.create_batch(chunk.path)
        final = await wait_for_anthropic_batch(client, operation["id"])
        state = anthropic_operation_state(final)
        if state not in ANTHROPIC_SUCCESS_STATES:
            for job in chunk_jobs.values():
                error = ValueError(f"Anthropic batch ended in state {state!r}")
                append_result(
                    error_path,
                    _failure(job, judge, account.account_id, stage="batch", error=error, category="batch_failed"),
                )
            continue
        content = await client.download_results(operation["id"])
        seen: set[str] = set()
        async for row in iter_anthropic_batch_results(content):
            key = row.get("custom_id")
            if not isinstance(key, str) or key not in chunk_jobs or key in seen:
                continue
            seen.add(key)
            job = chunk_jobs[key]
            try:
                batch_result = row.get("result")
                if not isinstance(batch_result, dict):
                    raise ValueError("Anthropic batch result omitted result")
                if batch_result.get("type") != "succeeded":
                    detail = batch_result.get("error") or {"type": batch_result.get("type") or "unknown"}
                    raise ValueError(json.dumps(detail, ensure_ascii=False)[:1_000])
                message = batch_result.get("message")
                if not isinstance(message, dict):
                    raise ValueError("Anthropic succeeded result omitted message")
                response = anthropic_provider_result(message)
                choice, confidence = parse_verdict(response.text)
            except (ProviderError, InvalidVerdict, ValueError) as error:
                append_result(
                    error_path,
                    _failure(job, judge, account.account_id, stage="parse", error=error, category="invalid_response"),
                )
                continue
            result = base_result(job, judge, account.account_id)
            result.update(
                {
                    "choice": choice,
                    "canonical_verdict": canonical_verdict(
                        choice, job.displayed_a_original_side, job.displayed_b_original_side
                    ),
                    "confidence": confidence,
                    "raw_response": response.text,
                    "usage": response.usage,
                    "cost": None,
                    "provider_response_id": response.response_id,
                    "provider_batch_id": operation["id"],
                    "attempts": 1,
                    "completed_at": utc_now(),
                }
            )
            append_result(output_path, result)


async def run_gemini_batch_judge(
    judge: JudgeConfig,
    jobs: list[Job],
    output_path: Path,
    error_path: Path,
    batch_dir: Path,
    *,
    timeout: float,
    dry_run: bool,
) -> None:
    completed = load_resume_keys(output_path)
    pending = [job for job in jobs if job.resume_key not in completed]
    print(f"{judge.alias}: {len(pending)} of {len(jobs)} jobs pending (Gemini batch)")
    if dry_run or not pending:
        return
    account = judge.accounts[0]
    api_key = os.environ.get(account.api_key_env or "")
    if not api_key:
        raise SystemExit(f"environment variable {account.api_key_env!r} is required for {judge.alias}")
    keyed = {hashlib.sha256(job.stable_key.encode()).hexdigest()[:32]: job for job in pending}
    rows = [(key, gemini_generate_payload(judge, job.messages)) for key, job in keyed.items()]
    split_kwargs: dict[str, Any] = {}
    if judge.limits.batch_max_requests is not None:
        split_kwargs["max_requests"] = judge.limits.batch_max_requests
    chunks = split_gemini_jsonl(batch_dir, rows, **split_kwargs)
    client = GeminiBatchClient(
        api_key,
        base_url=judge.api.base_url or "https://generativelanguage.googleapis.com",
        timeout=judge.request_timeout or timeout,
    )
    for chunk in chunks:
        chunk_jobs = {key: keyed[key] for key in chunk.keys}
        display_name = f"judge-query-{judge.alias}-{chunk.index}"[:128]
        print(f"{judge.alias}: submitting batch chunk {chunk.index + 1}/{len(chunks)} ({len(chunk.keys)} requests)")
        file_name = await client.upload_file(chunk.path, display_name)
        operation = await client.create_batch(judge.model, file_name, display_name)
        final = await wait_for_gemini_batch(client, operation.get("name", ""))
        state = gemini_operation_state(final)
        if state not in GEMINI_SUCCESS_STATES:
            for job in chunk_jobs.values():
                error = ValueError(f"Gemini batch ended in state {state!r}")
                append_result(
                    error_path,
                    _failure(job, judge, account.account_id, stage="batch", error=error, category="batch_failed"),
                )
            continue
        output_name = gemini_result_file_name(final)
        if not output_name:
            raise SystemExit(f"successful Gemini batch omitted its result file: {final}")
        content = await client.download_results(output_name)
        seen: set[str] = set()
        async for row in iter_gemini_batch_results(content):
            key = row.get("key")
            if not isinstance(key, str) or key not in chunk_jobs or key in seen:
                continue
            seen.add(key)
            job = chunk_jobs[key]
            try:
                if isinstance(row.get("error"), dict):
                    raise ValueError(json.dumps(row["error"], ensure_ascii=False))
                response_value = row.get("response")
                if not isinstance(response_value, dict):
                    raise ValueError("batch result omitted response")
                response = gemini_provider_result(response_value)
                choice, confidence = parse_verdict(response.text)
            except (ProviderError, InvalidVerdict, ValueError) as error:
                append_result(
                    error_path,
                    _failure(job, judge, account.account_id, stage="parse", error=error, category="invalid_response"),
                )
                continue
            result = base_result(job, judge, account.account_id)
            result.update(
                {
                    "choice": choice,
                    "canonical_verdict": canonical_verdict(
                        choice, job.displayed_a_original_side, job.displayed_b_original_side
                    ),
                    "confidence": confidence,
                    "raw_response": response.text,
                    "usage": response.usage,
                    "cost": None,
                    "provider_response_id": response.response_id,
                    "provider_batch_id": operation.get("name"),
                    "attempts": 1,
                    "completed_at": utc_now(),
                }
            )
            append_result(output_path, result)


async def run_judge(
    judge: JudgeConfig,
    records: list[dict[str, Any]],
    prompt_spec: Any,
    study: Study,
    *,
    dry_run: bool,
) -> None:
    output_dir = study.results_dir / judge.alias
    output_path = output_dir / "responses.jsonl"
    error_path = output_dir / "errors.jsonl"
    if not dry_run:
        write_or_check_metadata(
            output_dir / "metadata.json",
            study=study,
            judge=judge,
            prompt_fingerprint=prompt_spec.fingerprint,
        )
    # Mirrors collect.py's _collect_conditions: every row/resume-key/metadata
    # field keyed on "experiment_fingerprint" must combine the judge with the
    # prompt, not the judge alone, so a prompt change invalidates resume for
    # every judge that used it. Applied only now (after metadata, which pins
    # the *bare* judge fingerprint together with the prompt fingerprint
    # separately) so it isn't double-hashed.
    judge = replace(
        judge,
        experiment_fingerprint=effective_judge_fingerprint(judge, prompt_spec.fingerprint),
    )
    jobs = build_jobs(records, judge, prompt_spec)
    if judge.api.adapter == "openai_batch":
        await run_openai_batch_judge(
            judge, jobs, output_path, error_path, study.results_dir.parent / "batch-jobs" / judge.alias,
            timeout=study.timeout, dry_run=dry_run,
        )
    elif judge.api.adapter == "anthropic_batch":
        await run_anthropic_batch_judge(
            judge, jobs, output_path, error_path, study.results_dir.parent / "batch-jobs" / judge.alias,
            timeout=study.timeout, dry_run=dry_run,
        )
    elif judge.api.adapter == "google_gemini_batch":
        await run_gemini_batch_judge(
            judge, jobs, output_path, error_path, study.results_dir.parent / "batch-jobs" / judge.alias,
            timeout=study.timeout, dry_run=dry_run,
        )
    else:
        await run_ollama_judge(
            judge, jobs, output_path, error_path,
            timeout=study.timeout, max_retries=study.max_retries, dry_run=dry_run,
        )


async def async_main(args: argparse.Namespace) -> None:
    study_path = Path(args.study).resolve()
    load_environment_file()
    study = Study(study_path)
    validate_dataset(study)
    app_config = load_config(study.api_config, study.judge_config)

    requested = args.judges.split(",") if args.judges else study.judges
    unknown = sorted(set(requested) - set(app_config.judges))
    if unknown:
        raise SystemExit(f"unknown judge alias(es): {unknown}")

    prompt_spec = load_prompt_spec(study.system_prompt, study.user_prompt_template, study.study_id)
    records = load_records(study.dataset, max_records=args.max_records)
    print(f"study {study.study_id}: {len(records)} records x {len(CONDITIONS)} conditions x {len(requested)} judges")

    for alias in requested:
        judge = app_config.judges[alias]
        await run_judge(judge, records, prompt_spec, study, dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="Path to a study.toml file")
    parser.add_argument("--judges", default=None, help="Comma-separated judge aliases (default: all in study.toml)")
    parser.add_argument("--max-records", type=int, default=None, help="Limit the number of dataset records")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/prompts/dataset only, no network calls")
    args = parser.parse_args(argv)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
