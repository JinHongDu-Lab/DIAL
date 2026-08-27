from __future__ import annotations

import json
import math
from typing import Any

import tiktoken

OPENAI_TOKEN_ESTIMATOR = "tiktoken_json"
GEMINI_TOKEN_ESTIMATOR = "gemini_local_conservative"


def compact_json(value: Any) -> str:
    """Serialize a provider request in the same compact form used by JSONL."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate_openai_input_tokens(
    request: dict[str, Any],
    *,
    encoding_name: str,
) -> int:
    """Conservatively estimate OpenAI input tokens from the request body."""

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError as error:
        raise ValueError(
            f"unknown OpenAI batch token encoding {encoding_name!r}"
        ) from error
    return len(
        encoding.encode(compact_json(request), disallowed_special=())
    )


def estimate_gemini_input_tokens(request: dict[str, Any]) -> int:
    """Estimate Gemini tokens locally without spending API quota.

    Google documents roughly four ASCII characters per token. Non-ASCII text
    is charged at its full UTF-8 byte length to keep multilingual estimates
    conservative. A small fixed allowance covers request framing.
    """

    serialized = compact_json(request)
    ascii_characters = sum(ord(character) < 128 for character in serialized)
    non_ascii_bytes = sum(
        len(character.encode("utf-8"))
        for character in serialized
        if ord(character) >= 128
    )
    return math.ceil(ascii_characters / 4) + non_ascii_bytes + 32
