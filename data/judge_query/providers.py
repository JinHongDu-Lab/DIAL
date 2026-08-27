from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import httpx

try:
    import litellm
except ModuleNotFoundError:  # Optional for direct HKU adapters.
    litellm = None  # type: ignore[assignment]

from .config import AccountConfig, JudgeConfig


class ErrorCategory(str, Enum):
    """Normalized provider failure categories."""

    AUTHENTICATION = "authentication"
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    SERVER = "server"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class ProviderError(RuntimeError):
    """Normalized error raised by provider adapters."""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory,
        status_code: int | None = None,
        retry_after: float | None = None,
        headers: Mapping[str, str] | None = None,
        response_detail: str | None = None,
        response_json: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.retry_after = retry_after
        self.headers = headers
        self.response_detail = response_detail
        self.response_json = response_json

    @property
    def retryable(self) -> bool:
        return self.category in {
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.SERVER,
            ErrorCategory.INVALID_RESPONSE,
        }

    @property
    def disables_account(self) -> bool:
        return self.category == ErrorCategory.AUTHENTICATION


@dataclass(frozen=True)
class ProviderResult:
    """Normalized provider completion."""

    text: str
    usage: dict[str, Any] | None = None
    cost: float | None = None
    response_id: str | None = None
    status_code: int | None = None
    headers: Mapping[str, str] | None = None


class ProviderAdapter(Protocol):
    """Interface implemented by every completion backend."""

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        """Return one normalized completion."""


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _google_retry_after(response: httpx.Response) -> float | None:
    """Read a Google ``RetryInfo.retryDelay`` protobuf duration."""

    try:
        value = response.json()
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        error = value.get("error", {})
        if isinstance(error, dict):
            for detail in error.get("details", []):
                if not isinstance(detail, dict):
                    continue
                if not str(detail.get("@type", "")).endswith(
                    "google.rpc.RetryInfo"
                ):
                    continue
                delay = detail.get("retryDelay")
                if isinstance(delay, str):
                    match = re.fullmatch(r"(\d+(?:\.\d+)?)s", delay.strip())
                    if match:
                        return max(0.0, float(match.group(1)))
                if isinstance(delay, dict):
                    try:
                        seconds = float(delay.get("seconds", 0))
                        nanos = float(delay.get("nanos", 0))
                    except (TypeError, ValueError):
                        continue
                    return max(0.0, seconds + nanos / 1_000_000_000)
    match = re.search(
        r"please\s+retry\s+in\s+(\d+(?:\.\d+)?)s",
        response.text,
        re.IGNORECASE,
    )
    return max(0.0, float(match.group(1))) if match else None


def quota_retry_after(detail: str) -> float | None:
    """Parse an HKU-style quota replenishment countdown from an error.

    HKU normally reports the countdown as plain ``HH:MM:SS``, resetting
    within a day. On a few occasions (gpt-5.5, claude-opus-4.6,
    qwen3-next-80b-thinking-low) it has instead reported a multi-day
    countdown using .NET's TimeSpan default format ``D.HH:MM:SS`` (a leading
    day count joined by a dot, not the word "day(s)") lasting several weeks.
    Live re-checks showed those same models back to a normal sub-day
    countdown within a day or two, so this looks like a transient gateway
    anomaly rather than a documented cap (HKU's own rate-limit tables list
    every chat model, including these, as a plain daily quota). Both formats
    are matched here regardless of cause; ``routing.py``/``usage.py`` cap how
    long a reported multi-day countdown is trusted before re-verifying live
    (see ``QUOTA_REVALIDATE_AFTER``).
    """

    match = re.search(
        r"quota\s+will\s+be\s+replenished\s+in\s+"
        r"(?:(\d+)\s+days?,?\s+|(\d+)\.)?(\d+):(\d+):(\d+)",
        detail,
        re.IGNORECASE,
    )
    if match is None:
        return None
    day_word, day_dot, hours, minutes, seconds = match.groups()
    days = int(day_word or day_dot or 0)
    hours, minutes, seconds = (int(value) for value in (hours, minutes, seconds))
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _period_quota_exhausted(detail: str) -> bool:
    """Return whether a provider message identifies a non-transient quota."""

    normalized = " ".join(detail.lower().split())
    return any(
        marker in normalized
        for marker in (
            "out of call volume quota",
            "quota will be replenished",
            "daily quota",
            "daily request limit",
            "requests per day",
            "per-day quota",
            "batch enqueued tokens",
            "batchenqueuedtokens",
            "enqueued tokens per model",
        )
    )


def _http_category(status_code: int, detail: str = "") -> ErrorCategory:
    if _period_quota_exhausted(detail):
        return ErrorCategory.QUOTA_EXHAUSTED
    if status_code in {401, 403}:
        return ErrorCategory.AUTHENTICATION
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code in {408, 409, 425}:
        return ErrorCategory.TIMEOUT
    if status_code >= 500:
        return ErrorCategory.SERVER
    if status_code >= 400:
        return ErrorCategory.INVALID_REQUEST
    return ErrorCategory.UNKNOWN


def _safe_http_error(response: httpx.Response) -> ProviderError:
    full_detail = " ".join(response.text.splitlines())[:16_000]
    try:
        response_json = response.json()
    except json.JSONDecodeError:
        response_json = None
    if not isinstance(response_json, dict):
        response_json = None
    category = _http_category(response.status_code, full_detail)
    retry_after = _retry_after(response.headers)
    if retry_after is None:
        retry_after = _google_retry_after(response)
    if category == ErrorCategory.QUOTA_EXHAUSTED and retry_after is None:
        retry_after = quota_retry_after(full_detail)
    if category == ErrorCategory.RATE_LIMIT and retry_after is None:
        retry_after = 60.0
    return ProviderError(
        f"HTTP {response.status_code}: {full_detail[:2_000]}",
        category=category,
        status_code=response.status_code,
        retry_after=retry_after,
        headers=response.headers,
        response_detail=full_detail,
        response_json=response_json,
    )


def _text_from_litellm(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                texts.append(item.text)
        if texts:
            return "".join(texts)
    raise ProviderError(
        "provider returned empty or unsupported content",
        category=ErrorCategory.INVALID_RESPONSE,
    )


class LiteLLMAdapter:
    """Call providers supported by the LiteLLM Python SDK."""

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        if litellm is None:
            raise ProviderError(
                "the litellm adapter requires installation with "
                "`python -m pip install -e '.[litellm]'`",
                category=ErrorCategory.INVALID_REQUEST,
            )
        kwargs = _inference_payload(judge, "litellm")
        kwargs.update(
            model=judge.model,
            messages=messages,
            timeout=timeout,
            num_retries=0,
            stream=False,
        )
        if judge.generation_seed is not None:
            kwargs["seed"] = judge.generation_seed
        if api_key:
            kwargs["api_key"] = api_key
        api_base = account.api_base or judge.api.base_url
        if api_base:
            kwargs["api_base"] = api_base
        try:
            response = await litellm.acompletion(**kwargs)
            usage = getattr(response, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            elif usage is not None and not isinstance(usage, dict):
                try:
                    usage = dict(usage)
                except (TypeError, ValueError):
                    usage = None
            hidden = getattr(response, "_hidden_params", {}) or {}
            cost = hidden.get("response_cost")
            return ProviderResult(
                text=_text_from_litellm(response),
                usage=usage,
                cost=float(cost) if cost is not None else None,
                response_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as error:
            status = getattr(error, "status_code", None)
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            error_name = type(error).__name__
            detail = str(error)
            if _period_quota_exhausted(detail):
                category = ErrorCategory.QUOTA_EXHAUSTED
            elif litellm is not None and isinstance(error, litellm.AuthenticationError):
                category = ErrorCategory.AUTHENTICATION
            elif litellm is not None and isinstance(error, litellm.BadRequestError):
                category = ErrorCategory.INVALID_REQUEST
            elif error_name == "RateLimitError":
                category = ErrorCategory.RATE_LIMIT
            elif status is not None:
                category = _http_category(int(status), detail)
            elif isinstance(error, TimeoutError) or error_name in {
                "Timeout",
                "TimeoutError",
                "APITimeoutError",
            }:
                category = ErrorCategory.TIMEOUT
            else:
                category = ErrorCategory.NETWORK
            retry_after = _retry_after(headers) if headers else None
            if category == ErrorCategory.QUOTA_EXHAUSTED and retry_after is None:
                retry_after = quota_retry_after(detail)
            if category == ErrorCategory.RATE_LIMIT and retry_after is None:
                retry_after = 60.0
            raise ProviderError(
                detail,
                category=category,
                status_code=int(status) if status is not None else None,
                retry_after=retry_after,
                headers=headers,
            ) from error


class _HKUBaseAdapter:
    """Shared HTTP behavior for HKU API Gateway adapters."""

    provider_label = "HKU"

    def _url(self, judge: JudgeConfig, account: AccountConfig) -> str:
        base_url = (account.api_base or judge.api.base_url or "").rstrip("/")
        template = judge.api.endpoint_template or ""
        return f"{base_url}/{template.lstrip('/')}".format(model=judge.model)

    def _headers(self, judge: JudgeConfig, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Cache-Control": "no-cache"}
        if judge.api.auth_header and api_key:
            headers[judge.api.auth_header] = api_key
        return headers

    async def _post(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        payload: dict[str, Any],
        timeout: float,
    ) -> tuple[dict[str, Any], httpx.Response]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self._url(judge, account),
                    headers=self._headers(judge, api_key),
                    json=payload,
                )
        except httpx.TimeoutException as error:
            raise ProviderError(str(error), category=ErrorCategory.TIMEOUT) from error
        except httpx.RequestError as error:
            raise ProviderError(str(error), category=ErrorCategory.NETWORK) from error
        if response.status_code >= 400:
            raise _safe_http_error(response)
        try:
            value = response.json()
        except json.JSONDecodeError as error:
            raise ProviderError(
                f"{self.provider_label} returned invalid JSON",
                category=ErrorCategory.INVALID_RESPONSE,
                status_code=response.status_code,
            ) from error
        if not isinstance(value, dict):
            raise ProviderError(
                f"{self.provider_label} returned a non-object JSON response",
                category=ErrorCategory.INVALID_RESPONSE,
                status_code=response.status_code,
            )
        return value, response


class HKUOpenAIAdapter(_HKUBaseAdapter):
    """Call an HKU OpenAI-compatible chat endpoint."""

    provider_label = "HKU OpenAI"

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        payload = _inference_payload(judge, "openai")
        payload.update(model=judge.model, messages=messages, stream=False)
        if judge.generation_seed is not None:
            payload["seed"] = judge.generation_seed
        value, response = await self._post(judge, account, api_key, payload, timeout)
        try:
            choice = value["choices"][0]
            message = choice["message"]
            text = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                f"{self.provider_label} response is missing "
                "choices[0].message.content",
                category=ErrorCategory.INVALID_RESPONSE,
                status_code=response.status_code,
            ) from error
        if not isinstance(text, str) or not text.strip():
            finish_reason = choice.get("finish_reason")
            reasoning_content = message.get("reasoning_content")
            reasoning_chars = (
                len(reasoning_content)
                if isinstance(reasoning_content, str)
                else 0
            )
            usage = value.get("usage")
            completion_tokens = (
                usage.get("completion_tokens")
                if isinstance(usage, dict)
                else None
            )
            diagnostics = (
                f"finish_reason={finish_reason!r}, "
                f"reasoning_content_chars={reasoning_chars}, "
                f"completion_tokens={completion_tokens!r}"
            )
            if finish_reason == "length":
                raise ProviderError(
                    f"{self.provider_label} exhausted the output token limit "
                    f"before producing final text ({diagnostics})",
                    category=ErrorCategory.INVALID_REQUEST,
                    status_code=response.status_code,
                )
            raise ProviderError(
                f"{self.provider_label} response contains no final text "
                f"({diagnostics})",
                category=ErrorCategory.INVALID_RESPONSE,
                status_code=response.status_code,
            )
        return ProviderResult(
            text=text,
            usage=value.get("usage"),
            response_id=value.get("id"),
            status_code=response.status_code,
            headers=response.headers,
        )


class OpenAIChatAdapter(HKUOpenAIAdapter):
    """Call a configurable OpenAI-compatible chat-completions endpoint."""

    provider_label = "OpenAI-compatible provider"


class HKUResponsesAdapter(_HKUBaseAdapter):
    """Call an HKU/Azure Responses-only deployment."""

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        payload = openai_responses_payload(judge, messages)
        value, response = await self._post(judge, account, api_key, payload, timeout)
        return openai_responses_provider_result(
            value,
            status_code=response.status_code,
            headers=response.headers,
        )


def openai_responses_provider_result(
    value: dict[str, Any],
    *,
    status_code: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> ProviderResult:
    """Normalize one OpenAI-compatible Responses object."""

    text = value.get("output_text")
    if not isinstance(text, str) or not text.strip():
        texts: list[str] = []
        for output in value.get("output", []):
            if not isinstance(output, dict):
                continue
            for content in output.get("content", []):
                if (
                    isinstance(content, dict)
                    and content.get("type") in {"output_text", "text"}
                    and isinstance(content.get("text"), str)
                ):
                    texts.append(content["text"])
        text = "".join(texts)
    if not isinstance(text, str) or not text.strip():
        raise ProviderError(
            "OpenAI Responses object contains no output text",
            category=ErrorCategory.INVALID_RESPONSE,
            status_code=status_code,
        )
    usage = value.get("usage")
    return ProviderResult(
        text=text,
        usage=usage if isinstance(usage, dict) else None,
        response_id=value.get("id") if isinstance(value.get("id"), str) else None,
        status_code=status_code,
        headers=headers,
    )


def _split_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = [item["content"] for item in messages if item["role"] == "system"]
    conversation = [item for item in messages if item["role"] != "system"]
    return "\n\n".join(system_parts) or None, conversation


def openai_responses_payload(
    judge: JudgeConfig,
    messages: list[dict[str, str]],
    *,
    explicit_disabled_reasoning: bool = False,
) -> dict[str, Any]:
    """Build one native Responses request from normalized messages."""

    system, conversation = _split_messages(messages)
    payload = _inference_payload(judge, "responses")
    payload.update(model=judge.model, input=conversation, store=False)
    if system:
        payload["instructions"] = system
    if (
        explicit_disabled_reasoning
        and judge.inference.reasoning.mode == "disabled"
        and "enabled" in judge.capabilities.reasoning_modes
    ):
        payload["reasoning"] = {"effort": "none"}
    return payload


def anthropic_messages_payload(
    judge: JudgeConfig,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Build one native Anthropic Messages request."""

    system, conversation = _split_messages(messages)
    payload = dict(judge.api.request_defaults)
    payload.update(judge.provider_options)
    payload.update(
        {
            "model": judge.model,
            "max_tokens": judge.inference.max_output_tokens,
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in conversation
            ],
        }
    )
    if system:
        payload["system"] = system
    reasoning = judge.inference.reasoning
    if reasoning.mode == "disabled":
        if set(judge.capabilities.reasoning_modes) != {"disabled"}:
            payload["thinking"] = {"type": "disabled"}
        if judge.inference.temperature is not None:
            payload["temperature"] = judge.inference.temperature
    elif "haiku" in judge.model.lower():
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": judge.inference.max_output_tokens,
        }
    else:
        payload["thinking"] = {"type": "adaptive"}
        if reasoning.effort is not None:
            payload["effort"] = reasoning.effort
    return payload


def anthropic_provider_result(
    value: dict[str, Any],
    *,
    status_code: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> ProviderResult:
    """Normalize a native Anthropic Message while excluding thinking blocks."""

    content = value.get("content")
    if not isinstance(content, list):
        raise ProviderError(
            "Anthropic response is missing content",
            category=ErrorCategory.INVALID_RESPONSE,
            status_code=status_code,
        )
    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ).strip()
    if not text:
        raise ProviderError(
            "Anthropic response contains no answer text",
            category=ErrorCategory.INVALID_RESPONSE,
            status_code=status_code,
        )
    usage = value.get("usage")
    return ProviderResult(
        text=text,
        usage=usage if isinstance(usage, dict) else None,
        response_id=value.get("id") if isinstance(value.get("id"), str) else None,
        status_code=status_code,
        headers=headers,
    )


def _inference_payload(judge: JudgeConfig, target: str) -> dict[str, Any]:
    """Translate normalized inference settings to one provider wire schema."""

    payload = dict(judge.api.request_defaults)
    payload.update(judge.provider_options)
    inference = judge.inference
    reasoning = inference.reasoning
    if inference.temperature is not None:
        payload["temperature"] = inference.temperature
    token_field = {
        "openai": "max_completion_tokens",
        "responses": "max_output_tokens",
        "gemini": "maxOutputTokens",
        "claude": "maxTokens",
        "litellm": "max_tokens",
    }[target]
    api_name = judge.api.api_id.lower()
    if target == "openai" and (
        "deepseek" in api_name
        or "vertex-openai" in api_name
        or "ollama" in api_name
    ):
        token_field = "max_tokens"
    payload[token_field] = inference.max_output_tokens
    if target == "responses" and reasoning.mode != "disabled":
        payload["reasoning"] = {"effort": reasoning.effort or "low"}
    elif target == "gemini":
        model = judge.model.lower()
        if model.startswith(("gemini-3", "gemma-4")):
            payload["thinkingConfig"] = {
                "thinkingLevel": (
                    "minimal"
                    if reasoning.mode == "disabled"
                    else reasoning.effort or "low"
                )
            }
        else:
            budget = -1
            if reasoning.mode == "disabled":
                budget = 0
            elif "2.5-pro" in model and reasoning.effort == "low":
                budget = 128
            payload["thinkingConfig"] = {"thinkingBudget": budget}
    elif target == "claude" and reasoning.mode != "disabled":
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": inference.max_output_tokens,
        }
    elif target in {"openai", "litellm"}:
        hybrid = "deepseek" in api_name or "vertex-openai" in api_name
        if hybrid:
            payload["thinking"] = {
                "type": "disabled" if reasoning.mode == "disabled" else "enabled"
            }
        if target == "openai" and "ollama" in api_name:
            payload["reasoning_effort"] = (
                "none" if reasoning.mode == "disabled"
                else reasoning.effort or "low"
            )
        elif reasoning.mode != "disabled" and reasoning.effort is not None:
            payload["reasoning_effort"] = reasoning.effort
    return payload


def gemini_generate_payload(
    judge: JudgeConfig,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a native Gemini ``GenerateContentRequest``."""

    system, conversation = _split_messages(messages)
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
            for item in conversation
        ]
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    generation = _inference_payload(judge, "gemini")
    if judge.generation_seed is not None:
        generation["seed"] = judge.generation_seed
    if generation:
        payload["generationConfig"] = generation
    return payload


def gemini_provider_result(
    value: dict[str, Any],
    *,
    status_code: int | None = None,
    headers: Mapping[str, str] | None = None,
    provider_label: str = "Gemini",
) -> ProviderResult:
    """Normalize native Gemini output while excluding thought parts."""

    try:
        parts = value["candidates"][0]["content"]["parts"]
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and not part.get("thought", False)
        ).strip()
    except (KeyError, IndexError, TypeError) as error:
        raise ProviderError(
            f"{provider_label} response is missing candidate text",
            category=ErrorCategory.INVALID_RESPONSE,
            status_code=status_code,
        ) from error
    if not text:
        raise ProviderError(
            f"{provider_label} response contains no answer text",
            category=ErrorCategory.INVALID_RESPONSE,
            status_code=status_code,
        )
    return ProviderResult(
        text=text,
        usage=value.get("usageMetadata"),
        response_id=value.get("responseId"),
        status_code=status_code,
        headers=headers,
    )


class HKUGeminiAdapter(_HKUBaseAdapter):
    """Call HKU Vertex AI Gemini generateContent."""

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        payload = gemini_generate_payload(judge, messages)
        value, response = await self._post(judge, account, api_key, payload, timeout)
        return gemini_provider_result(
            value,
            status_code=response.status_code,
            headers=response.headers,
            provider_label="HKU Gemini",
        )


class GoogleGeminiAdapter(_HKUBaseAdapter):
    """Call the Google Gemini Developer API directly."""

    provider_label = "Google Gemini"

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        payload = gemini_generate_payload(judge, messages)
        value, response = await self._post(
            judge, account, api_key, payload, timeout
        )
        return gemini_provider_result(
            value,
            status_code=response.status_code,
            headers=response.headers,
            provider_label=self.provider_label,
        )


class HKUClaudeAdapter(_HKUBaseAdapter):
    """Call HKU AWS Bedrock Claude Converse."""

    async def complete(
        self,
        judge: JudgeConfig,
        account: AccountConfig,
        api_key: str | None,
        messages: list[dict[str, str]],
        timeout: float,
    ) -> ProviderResult:
        system, conversation = _split_messages(messages)
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": item["role"],
                    "content": [{"text": item["content"]}],
                }
                for item in conversation
            ]
        }
        if system:
            payload["system"] = [{"text": system}]
        inference = _inference_payload(judge, "claude")
        if inference:
            payload["inferenceConfig"] = inference
        value, response = await self._post(judge, account, api_key, payload, timeout)
        try:
            parts = value["output"]["message"]["content"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, TypeError) as error:
            raise ProviderError(
                "HKU Claude response is missing output.message.content",
                category=ErrorCategory.INVALID_RESPONSE,
                status_code=response.status_code,
            ) from error
        if not text:
            raise ProviderError(
                "HKU Claude response contains no text",
                category=ErrorCategory.INVALID_RESPONSE,
                status_code=response.status_code,
            )
        return ProviderResult(
            text=text,
            usage=value.get("usage"),
            response_id=value.get("id"),
            status_code=response.status_code,
            headers=response.headers,
        )


ADAPTERS: dict[str, ProviderAdapter] = {
    "litellm": LiteLLMAdapter(),
    "openai_chat": OpenAIChatAdapter(),
    "hku_openai": HKUOpenAIAdapter(),
    "hku_responses": HKUResponsesAdapter(),
    "hku_gemini": HKUGeminiAdapter(),
    "hku_claude": HKUClaudeAdapter(),
    "google_gemini": GoogleGeminiAdapter(),
}


def get_adapter(name: str) -> ProviderAdapter:
    """Return a configured provider adapter."""

    try:
        return ADAPTERS[name]
    except KeyError as error:
        raise ValueError(f"unknown provider adapter: {name}") from error
