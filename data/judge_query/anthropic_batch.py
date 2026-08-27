from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx

from .gemini_batch import update_batch_state
from .providers import ErrorCategory, ProviderError, _safe_http_error
from .status import read_status, utc_now

SUCCESS_STATES = frozenset({"ended"})
TERMINAL_STATES = frozenset({"ended"})
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 250 * 1024 * 1024
DEFAULT_MAX_REQUESTS = 50_000
DEFAULT_POLL_SECONDS = 30.0
ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class AnthropicBatchChunk:
    """One local Anthropic Message Batch request chunk."""

    index: int
    path: Path
    keys: tuple[str, ...]
    size_bytes: int
    estimated_input_tokens: int = 0
    token_estimator: str = "not_applicable"


def split_anthropic_requests(
    directory: Path,
    rows: list[tuple[str, dict[str, Any]]],
    *,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_bytes: int = MAX_BATCH_BYTES,
) -> list[AnthropicBatchChunk]:
    """Write deterministic Anthropic request-array chunks under API limits."""

    if max_requests <= 0:
        raise ValueError("Anthropic max_requests must be positive")
    if max_requests > MAX_BATCH_REQUESTS:
        raise ValueError(
            f"Anthropic batch_max_requests cannot exceed {MAX_BATCH_REQUESTS:,}"
        )
    if max_bytes <= 0:
        raise ValueError("Anthropic max_bytes must be positive")
    prefix = b'{"requests":['
    suffix = b"]}"
    prepared: list[tuple[str, bytes]] = []
    for key, params in rows:
        if not key or len(key) > 64:
            raise ValueError(f"Anthropic custom_id {key!r} must be 1-64 characters")
        line = json.dumps(
            {"custom_id": key, "params": params},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        if len(prefix) + len(line) + len(suffix) > max_bytes:
            raise ValueError(
                f"Anthropic batch request {key!r} exceeds the payload byte limit"
            )
        prepared.append((key, line))

    directory.mkdir(parents=True, exist_ok=True)
    chunks: list[AnthropicBatchChunk] = []
    keys: list[str] = []
    lines: list[bytes] = []
    size = len(prefix) + len(suffix)

    def flush() -> None:
        nonlocal keys, lines, size
        if not lines:
            return
        content = prefix + b",".join(lines) + suffix
        index = len(chunks)
        path = directory / f"input-{index:05d}.json"
        path.write_bytes(content)
        chunks.append(
            AnthropicBatchChunk(
                index=index,
                path=path,
                keys=tuple(keys),
                size_bytes=len(content),
            )
        )
        keys, lines = [], []
        size = len(prefix) + len(suffix)

    for key, line in prepared:
        separator_size = 1 if lines else 0
        if lines and (
            len(lines) >= max_requests
            or size + separator_size + len(line) > max_bytes
        ):
            flush()
            separator_size = 0
        keys.append(key)
        lines.append(line)
        size += separator_size + len(line)
    flush()
    return chunks


def operation_state(value: dict[str, Any]) -> str:
    """Return an Anthropic Message Batch lifecycle state."""

    state = value.get("processing_status")
    return str(state or "in_progress")


def request_count(value: dict[str, Any]) -> int | None:
    """Return the total request count reported for one Message Batch."""

    counts = value.get("request_counts")
    if not isinstance(counts, dict):
        return None
    values = [
        counts.get(name)
        for name in ("processing", "succeeded", "errored", "canceled", "expired")
    ]
    if any(not isinstance(item, int) for item in values):
        return None
    return sum(values)


class AnthropicBatchClient:
    """Small native REST client for Anthropic's Message Batches API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Cache-Control": "no-cache",
        }

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as error:
            raise ProviderError(
                str(error), category=ErrorCategory.TIMEOUT
            ) from error
        except httpx.RequestError as error:
            raise ProviderError(
                str(error), category=ErrorCategory.NETWORK
            ) from error
        if response.status_code >= 400:
            raise _safe_http_error(response)
        return response

    async def create_batch(self, path: Path) -> dict[str, Any]:
        """Create one Message Batch from a persisted request document."""

        response = await self._request(
            "POST",
            f"{self.base_url}/v1/messages/batches",
            headers={**self.headers, "Content-Type": "application/json"},
            content=path.read_bytes(),
        )
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise ValueError("Anthropic batch create response omitted id")
        return result

    async def get_batch(self, batch_id: str) -> dict[str, Any]:
        response = await self._request(
            "GET",
            f"{self.base_url}/v1/messages/batches/{batch_id}",
            headers=self.headers,
        )
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Anthropic batch status response was not an object")
        return value

    async def list_batches(self) -> list[dict[str, Any]]:
        """Return recent batches, following bounded cursor pagination."""

        batches: list[dict[str, Any]] = []
        before_id: str | None = None
        for _ in range(20):
            params: dict[str, Any] = {"limit": 100}
            if before_id:
                params["before_id"] = before_id
            response = await self._request(
                "GET",
                f"{self.base_url}/v1/messages/batches",
                headers=self.headers,
                params=params,
            )
            value = response.json()
            if not isinstance(value, dict) or not isinstance(value.get("data"), list):
                raise ValueError("Anthropic batch list response omitted data")
            page = [item for item in value["data"] if isinstance(item, dict)]
            batches.extend(page)
            if not value.get("has_more") or not page:
                return batches
            last_id = value.get("last_id")
            if not isinstance(last_id, str) or not last_id:
                return batches
            before_id = last_id
        raise ValueError("Anthropic batch list exceeded 20 pages during recovery")

    async def recover_batch(
        self,
        submission_started_at: str,
        expected_requests: int,
    ) -> list[dict[str, Any]]:
        """Find candidate batches created after a persisted submission start."""

        try:
            threshold = datetime.fromisoformat(
                submission_started_at.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError("invalid Anthropic submission_started_at") from error
        if threshold.tzinfo is None:
            threshold = threshold.replace(tzinfo=timezone.utc)
        threshold -= timedelta(seconds=5)
        matches: list[dict[str, Any]] = []
        for batch in await self.list_batches():
            created_at = batch.get("created_at")
            if not isinstance(created_at, str):
                continue
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if created < threshold or request_count(batch) != expected_requests:
                continue
            matches.append(batch)
        return matches

    async def cancel_batch(self, batch_id: str) -> None:
        await self._request(
            "POST",
            f"{self.base_url}/v1/messages/batches/{batch_id}/cancel",
            headers={**self.headers, "Content-Type": "application/json"},
        )

    async def download_results(self, batch_id: str) -> bytes:
        response = await self._request(
            "GET",
            f"{self.base_url}/v1/messages/batches/{batch_id}/results",
            headers=self.headers,
        )
        return response.content


async def wait_for_batch(
    client: AnthropicBatchClient,
    batch_id: str,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll until an Anthropic Message Batch reaches a terminal state."""

    while True:
        value = await client.get_batch(batch_id)
        if on_update is not None:
            on_update(value)
        if operation_state(value) in TERMINAL_STATES:
            return value
        await asyncio.sleep(poll_seconds)


def cancel_persisted_anthropic_batches(
    run_dir: Path,
    api_key: str,
    *,
    base_url: str = "https://api.anthropic.com",
) -> list[str]:
    """Cancel active Anthropic batches recorded under one run."""

    cancelled: list[str] = []
    for state_path in sorted((run_dir / "batch-jobs").glob("*/*.json")):
        state = read_status(state_path)
        if (
            not state
            or state.get("reconciled")
            or state.get("provider") != "anthropic"
        ):
            continue
        batch_id = state.get("batch_name")
        if not isinstance(batch_id, str) or not batch_id:
            continue
        if str(state.get("state", "")) in TERMINAL_STATES:
            continue
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/messages/batches/{batch_id}/cancel",
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
            },
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise _safe_http_error(response)
        update_batch_state(
            state_path,
            state="canceling",
            remote_cancel_requested_at=utc_now(),
        )
        cancelled.append(batch_id)
    return cancelled


def iter_batch_results(content: bytes) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded Anthropic Message Batch result rows."""

    async def iterator() -> AsyncIterator[dict[str, Any]]:
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Anthropic batch result line {line_number} is not an object"
                )
            yield value

    return iterator()
