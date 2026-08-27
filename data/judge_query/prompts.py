from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .config import canonical_json

REQUIRED_TEMPLATE_FIELDS = {
    "question",
    "assistant_a_label",
    "assistant_a_response",
    "assistant_b_label",
    "assistant_b_response",
}
_TEMPLATE_FIELD = re.compile(r"(?<!\$)\$(?:{([A-Za-z_]\w*)}|([A-Za-z_]\w*))")


@dataclass(frozen=True)
class PromptSpec:
    """Validated provider-neutral prompt content for one study."""

    prompt_id: str
    system: str
    user_template: str
    fingerprint: str

    def render_user(self, values: dict[str, str]) -> str:
        """Render the user prompt after validating its complete value set."""

        return Template(self.user_template).substitute(values)


def _template_fields(template: str) -> set[str]:
    return {first or second for first, second in _TEMPLATE_FIELD.findall(template)}


def create_prompt_spec(
    prompt_id: str,
    system: str,
    user_template: str,
) -> PromptSpec:
    """Validate prompt text and return its content-addressed specification."""

    if not prompt_id.strip():
        raise ValueError("prompt_id must not be empty")
    if not system.strip():
        raise ValueError("system prompt must not be empty")
    fields = _template_fields(user_template)
    missing = REQUIRED_TEMPLATE_FIELDS - fields
    unknown = fields - REQUIRED_TEMPLATE_FIELDS
    if missing:
        raise ValueError(f"user prompt template is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"user prompt template has unknown fields: {sorted(unknown)}")
    try:
        Template(user_template).substitute(
            {field: field for field in REQUIRED_TEMPLATE_FIELDS}
        )
    except ValueError as error:
        raise ValueError(f"user prompt template is invalid: {error}") from error
    fingerprint = hashlib.sha256(
        canonical_json(
            {
                "prompt_id": prompt_id,
                "system": system,
                "user_template": user_template,
            }
        ).encode()
    ).hexdigest()
    return PromptSpec(prompt_id, system, user_template, fingerprint)


def load_prompt_spec(
    system_path: Path,
    user_template_path: Path,
    prompt_id: str,
) -> PromptSpec:
    """Load and validate a study's system and user prompt files."""

    return create_prompt_spec(
        prompt_id,
        system_path.read_text(encoding="utf-8").rstrip(),
        user_template_path.read_text(encoding="utf-8").rstrip(),
    )
