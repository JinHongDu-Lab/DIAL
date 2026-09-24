from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx

from .batch_budget import OPENAI_TOKEN_ESTIMATOR, estimate_openai_input_tokens
from .providers import ErrorCategory, ProviderError, _safe_http_error
from .status import read_status, utc_now
from .gemini_batch import update_batch_state

SUCCESS_STATES = frozenset({"completed"})
TERMINAL_STATES = frozenset({"completed", "failed", "expired", "cancelled"})
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_BYTES = 190 * 1024 * 1024
DEFAULT_POLL_SECONDS = 30.0


@dataclass(frozen=True)
class OpenAIBatchChunk:
    """One local OpenAI Batch JSONL input chunk."""

    index: int
    path: Path
    keys: tuple[str, ...]
    size_bytes: int
    estimated_input_tokens: int
    token_estimator: str


def split_openai_jsonl(
    directory: Path,
    rows: list[tuple[str, dict[str, Any]]],
    *,
    max_requests: int = MAX_BATCH_REQUESTS,
    max_bytes: int = MAX_BATCH_BYTES,
    max_input_tokens: int | None = None,
    encoding_name: str = "o200k_base",
) -> list[OpenAIBatchChunk]:
    """Write deterministic OpenAI JSONL chunks under provider limits."""

    chunks: list[OpenAIBatchChunk] = []
    prepared: list[tuple[str, bytes, int]] = []
    for key, body in rows:
        line = (
            json.dumps(
                {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        tokens = estimate_openai_input_tokens(
            body,
            encoding_name=encoding_name,
        )
        if len(line) > max_bytes:
            raise ValueError(f"batch request {key!r} exceeds the JSONL byte limit")
        if max_input_tokens is not None and tokens > max_input_tokens:
            raise ValueError(
                f"batch request {key!r} needs an estimated {tokens:,} input "
                f"tokens, above the configured {max_input_tokens:,}-token limit"
            )
        prepared.append((key, line, tokens))

    directory.mkdir(parents=True, exist_ok=True)
    lines: list[bytes] = []
    keys: list[str] = []
    size = 0
    estimated_tokens = 0

    def flush() -> None:
        nonlocal lines, keys, size, estimated_tokens
        if not lines:
            return
        index = len(chunks)
        path = directory / f"input-{index:05d}.jsonl"
        path.write_bytes(b"".join(lines))
        chunks.append(
            OpenAIBatchChunk(
                index,
                path,
                tuple(keys),
                size,
                estimated_tokens,
                OPENAI_TOKEN_ESTIMATOR,
            )
        )
        lines, keys, size, estimated_tokens = [], [], 0, 0

    for key, line, tokens in prepared:
        token_limit_reached = (
            max_input_tokens is not None
            and estimated_tokens + tokens > max_input_tokens
        )
        if lines and (
            len(lines) >= max_requests
            or size + len(line) > max_bytes
            or token_limit_reached
        ):
            flush()
        lines.append(line)
        keys.append(key)
        size += len(line)
        estimated_tokens += tokens
    flush()
    return chunks


def operation_state(value: dict[str, Any]) -> str:
    """Return an OpenAI Batch lifecycle state."""

    state = value.get("status")
    return str(state or "validating")


def result_file_name(value: dict[str, Any]) -> str | None:
    """Return the successful output file ID, if present."""

    file_id = value.get("output_file_id")
    return file_id if isinstance(file_id, str) and file_id else None


def error_file_name(value: dict[str, Any]) -> str | None:
    """Return the per-request error file ID, if present."""

    file_id = value.get("error_file_id")
    return file_id if isinstance(file_id, str) and file_id else None


class OpenAIBatchClient:
    """Small native REST client for OpenAI's Batch API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com",
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
            "Authorization": f"Bearer {self.api_key}",
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

    async def upload_file(self, path: Path, _display_name: str) -> str:
        """Upload a JSONL input file with purpose ``batch``."""

        with path.open("rb") as handle:
            response = await self._request(
                "POST",
                f"{self.base_url}/v1/files",
                headers=self.headers,
                data={"purpose": "batch"},
                files={"file": (path.name, handle, "application/jsonl")},
            )
        value = response.json()
        file_id = value.get("id") if isinstance(value, dict) else None
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("OpenAI file upload response omitted id")
        return file_id

    async def create_batch(
        self, _model: str, file_name: str, display_name: str
    ) -> dict[str, Any]:
        """Create a 24-hour Responses batch."""

        response = await self._request(
            "POST",
            f"{self.base_url}/v1/batches",
            headers={**self.headers, "Content-Type": "application/json"},
            json={
                "input_file_id": file_name,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {"dial_judge_id": display_name},
            },
        )
        value = response.json()
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError("OpenAI batch create response omitted id")
        return value

    async def find_batch(self, display_name: str) -> dict[str, Any] | None:
        """Find a prior batch after a create-response persistence gap."""

        after: str | None = None
        for _ in range(20):
            params: dict[str, Any] = {"limit": 100}
            if after:
                params["after"] = after
            response = await self._request(
                "GET",
                f"{self.base_url}/v1/batches",
                headers=self.headers,
                params=params,
            )
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("OpenAI batch list response was not an object")
            items = value.get("data", [])
            if not isinstance(items, list):
                raise ValueError("OpenAI batch list response omitted data")
            for batch in items:
                if not isinstance(batch, dict):
                    continue
                metadata = batch.get("metadata", {})
                if (
                    isinstance(metadata, dict)
                    and metadata.get("dial_judge_id") == display_name
                ):
                    return batch
            if not value.get("has_more") or not items:
                return None
            last_id = value.get("last_id")
            if not isinstance(last_id, str) or not last_id:
                return None
            after = last_id
        raise ValueError("OpenAI batch list exceeded 20 pages during recovery")

    async def get_batch(self, name: str) -> dict[str, Any]:
        response = await self._request(
            "GET", f"{self.base_url}/v1/batches/{name}", headers=self.headers
        )
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("OpenAI batch status response was not an object")
        return value

    async def cancel_batch(self, name: str) -> None:
        await self._request(
            "POST",
            f"{self.base_url}/v1/batches/{name}/cancel",
            headers={**self.headers, "Content-Type": "application/json"},
        )

    async def download_results(self, file_name: str) -> bytes:
        response = await self._request(
            "GET",
            f"{self.base_url}/v1/files/{file_name}/content",
            headers=self.headers,
        )
        return response.content


async def wait_for_batch(
    client: OpenAIBatchClient,
    name: str,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll until an OpenAI batch reaches a terminal state."""

    while True:
        value = await client.get_batch(name)
        if on_update is not None:
            on_update(value)
        if operation_state(value) in TERMINAL_STATES:
            return value
        await asyncio.sleep(poll_seconds)


def cancel_persisted_openai_batches(
    run_dir: Path,
    api_key: str,
    *,
    base_url: str = "https://api.openai.com",
) -> list[str]:
    """Cancel active OpenAI batches recorded under one run."""

    cancelled: list[str] = []
    for state_path in sorted((run_dir / "batch-jobs").glob("*/*.json")):
        state = read_status(state_path)
        if (
            not state
            or state.get("reconciled")
            or state.get("provider") != "openai"
        ):
            continue
        name = state.get("batch_name")
        if not isinstance(name, str) or not name:
            continue
        current = str(state.get("state", ""))
        if current in TERMINAL_STATES:
            continue
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/batches/{name}/cancel",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
            },
            timeout=30.0,
        )
        if response.status_code == 409:
            status_response = httpx.get(
                f"{base_url.rstrip('/')}/v1/batches/{name}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Cache-Control": "no-cache",
                },
                timeout=30.0,
            )
            if status_response.status_code >= 400:
                raise _safe_http_error(status_response)
            remote = status_response.json()
            if not isinstance(remote, dict):
                raise ValueError(
                    "OpenAI batch status response was not an object"
                )
            terminal = operation_state(remote)
            if terminal in TERMINAL_STATES:
                update_batch_state(
                    state_path,
                    state=terminal,
                    result_file=remote.get("output_file_id"),
                    error_file=remote.get("error_file_id"),
                )
                continue
        if response.status_code >= 400:
            raise _safe_http_error(response)
        update_batch_state(
            state_path,
            state="cancelling",
            remote_cancel_requested_at=utc_now(),
        )
        cancelled.append(name)
    return cancelled


def iter_batch_results(content: bytes) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded OpenAI Batch output or error rows."""

    async def iterator() -> AsyncIterator[dict[str, Any]]:
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"OpenAI batch result line {line_number} is not an object"
                )
            yield value

    return iterator()
