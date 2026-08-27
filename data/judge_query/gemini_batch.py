from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx

from .batch_budget import GEMINI_TOKEN_ESTIMATOR, estimate_gemini_input_tokens
from .providers import ProviderError, _safe_http_error
from .status import atomic_write_json, read_status, utc_now

SUCCESS_STATES = frozenset(
    {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}
)
TERMINAL_STATES = frozenset(
    {
        *SUCCESS_STATES,
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "BATCH_STATE_FAILED",
        "BATCH_STATE_CANCELLED",
        "BATCH_STATE_EXPIRED",
    }
)
MAX_BATCH_REQUESTS = 20_000
MAX_BATCH_BYTES = 256 * 1024 * 1024
DEFAULT_POLL_SECONDS = 30.0


@dataclass(frozen=True)
class BatchChunk:
    """One local JSONL input chunk."""

    index: int
    path: Path
    keys: tuple[str, ...]
    size_bytes: int
    estimated_input_tokens: int
    token_estimator: str


def split_jsonl(
    directory: Path,
    rows: list[tuple[str, dict[str, Any]]],
    *,
    max_requests: int = MAX_BATCH_REQUESTS,
    max_bytes: int = MAX_BATCH_BYTES,
    max_input_tokens: int | None = None,
) -> list[BatchChunk]:
    """Write deterministic Gemini JSONL chunks under provider limits."""

    chunks: list[BatchChunk] = []
    prepared: list[tuple[str, bytes, int]] = []
    for key, request in rows:
        line = (
            json.dumps(
                {"key": key, "request": request},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        tokens = estimate_gemini_input_tokens(request)
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
            BatchChunk(
                index,
                path,
                tuple(keys),
                size,
                estimated_tokens,
                GEMINI_TOKEN_ESTIMATOR,
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
    """Read a batch state from current and legacy operation shapes."""

    state = value.get("state")
    if not isinstance(state, str):
        metadata = value.get("metadata", {})
        state = metadata.get("state") if isinstance(metadata, dict) else None
    return str(state or "JOB_STATE_PENDING")


def result_file_name(value: dict[str, Any]) -> str | None:
    """Read the output file name from current and legacy operation shapes."""

    candidates = (
        value.get("dest"),
        value.get("response"),
        value.get("metadata"),
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = (
            candidate.get("fileName")
            or candidate.get("file_name")
            or candidate.get("responsesFile")
        )
        if isinstance(name, str) and name:
            return name
    return None


class GeminiBatchClient:
    """Small native REST client for the Gemini Batch API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    @property
    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, transport=self.transport
            ) as client:
                response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as error:
            from .providers import ErrorCategory

            raise ProviderError(str(error), category=ErrorCategory.TIMEOUT) from error
        except httpx.RequestError as error:
            from .providers import ErrorCategory

            raise ProviderError(str(error), category=ErrorCategory.NETWORK) from error
        if response.status_code >= 400:
            raise _safe_http_error(response)
        return response

    async def upload_file(self, path: Path, display_name: str) -> str:
        """Upload one JSONL file using Google's resumable upload protocol."""

        size = path.stat().st_size
        start_headers = {
            **self.headers,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        }
        started = await self._request(
            "POST",
            f"{self.base_url}/upload/v1beta/files",
            headers=start_headers,
            json={"file": {"display_name": display_name}},
        )
        upload_url = started.headers.get("x-goog-upload-url")
        if not upload_url:
            raise ValueError("Gemini upload response omitted x-goog-upload-url")
        uploaded = await self._request(
            "POST",
            upload_url,
            headers={
                **self.headers,
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=path.read_bytes(),
        )
        value = uploaded.json()
        name = value.get("file", {}).get("name") if isinstance(value, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("Gemini upload response omitted file.name")
        return name

    async def create_batch(
        self, model: str, file_name: str, display_name: str
    ) -> dict[str, Any]:
        response = await self._request(
            "POST",
            f"{self.base_url}/v1beta/models/{model}:batchGenerateContent",
            headers={**self.headers, "Content-Type": "application/json"},
            json={
                "batch": {
                    "display_name": display_name,
                    "input_config": {"file_name": file_name},
                }
            },
        )
        value = response.json()
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("Gemini batch create response omitted operation name")
        return value

    async def find_batch(self, display_name: str) -> dict[str, Any] | None:
        """Find a prior operation after a create-response persistence gap."""

        page_token: str | None = None
        for _ in range(20):
            params: dict[str, Any] = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            response = await self._request(
                "GET",
                f"{self.base_url}/v1beta/batches",
                headers=self.headers,
                params=params,
            )
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("Gemini batch list response was not an object")
            items = value.get("batches", value.get("operations", []))
            for batch in items:
                if not isinstance(batch, dict):
                    continue
                metadata = batch.get("metadata", {})
                candidate = batch.get("displayName") or batch.get("display_name")
                if not candidate and isinstance(metadata, dict):
                    candidate = metadata.get("displayName") or metadata.get(
                        "display_name"
                    )
                if candidate == display_name:
                    return batch
            page_token = value.get("nextPageToken")
            if not isinstance(page_token, str) or not page_token:
                return None
        raise ValueError("Gemini batch list exceeded 20 pages during recovery")

    async def get_batch(self, name: str) -> dict[str, Any]:
        response = await self._request(
            "GET", f"{self.base_url}/v1beta/{name.lstrip('/')}", headers=self.headers
        )
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Gemini batch status response was not an object")
        return value

    async def cancel_batch(self, name: str) -> None:
        await self._request(
            "POST",
            f"{self.base_url}/v1beta/{name.lstrip('/')}:cancel",
            headers={**self.headers, "Content-Type": "application/json"},
            json={},
        )

    async def delete_file(self, name: str) -> None:
        """Delete an uploaded input file that is not referenced by a batch."""

        await self._request(
            "DELETE",
            f"{self.base_url}/v1beta/{name.lstrip('/')}",
            headers=self.headers,
        )

    async def download_results(self, file_name: str) -> bytes:
        response = await self._request(
            "GET",
            f"{self.base_url}/download/v1beta/{file_name.lstrip('/')}:download",
            headers=self.headers,
            params={"alt": "media"},
        )
        return response.content


async def wait_for_batch(
    client: GeminiBatchClient,
    name: str,
    *,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll until the batch reaches a terminal state."""

    while True:
        value = await client.get_batch(name)
        if on_update is not None:
            on_update(value)
        if operation_state(value) in TERMINAL_STATES:
            return value
        await asyncio.sleep(poll_seconds)


def update_batch_state(path: Path, **values: Any) -> dict[str, Any]:
    """Atomically merge fields into one resumable provider-job state."""

    state = read_status(path) or {"version": 1, "created_at": utc_now()}
    state.update(values)
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)
    return state


def cancel_persisted_batches(
    run_dir: Path,
    api_key: str,
    *,
    base_url: str = "https://generativelanguage.googleapis.com",
) -> list[str]:
    """Cancel active batches recorded under a run, then update their states."""

    cancelled: list[str] = []
    for state_path in sorted((run_dir / "batch-jobs").glob("*/*.json")):
        state = read_status(state_path)
        if (
            not state
            or state.get("reconciled")
            or state.get("provider") not in {None, "google"}
        ):
            continue
        name = state.get("batch_name")
        if not isinstance(name, str) or not name:
            continue
        current = str(state.get("state", ""))
        if current in TERMINAL_STATES:
            continue
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1beta/{name.lstrip('/')}:cancel",
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={},
            timeout=30.0,
        )
        if response.status_code >= 400:
            raise _safe_http_error(response)
        update_batch_state(
            state_path,
            state="CANCEL_REQUESTED",
            remote_cancel_requested_at=utc_now(),
        )
        cancelled.append(name)
    return cancelled


def iter_batch_results(content: bytes) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded result rows from a downloaded JSONL file."""

    async def iterator() -> AsyncIterator[dict[str, Any]]:
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Gemini batch result line {line_number} is not an object"
                )
            yield value

    return iterator()
