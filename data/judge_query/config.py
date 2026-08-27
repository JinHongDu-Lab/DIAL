from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

Period = Literal["day", "week"]
ReasoningMode = Literal["disabled", "enabled", "required"]
SUPPORTED_ADAPTERS = {
    "litellm",
    "openai_chat",
    "hku_openai",
    "hku_responses",
    "hku_gemini",
    "hku_claude",
    "google_gemini",
    "google_gemini_batch",
    "openai_batch",
    "anthropic_batch",
}


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    """Return a short stable SHA-256 fingerprint for a JSON value."""

    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    """Return the full SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_environment_file(path: Path = Path(".env")) -> None:
    """Load missing environment variables from a simple local file."""

    if not path.is_file():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = raw_line.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(f"invalid environment line {line_number} in {path}")
        os_value = value.strip().strip('"').strip("'")
        if "\x00" in os_value:
            raise ValueError(f"invalid environment value on line {line_number} in {path}")
        os.environ.setdefault(name, os_value)


@dataclass(frozen=True)
class AccountConfig:
    """Non-secret reference to one provider account."""

    account_id: str
    api_key_env: str | None
    api_base: str | None = None


@dataclass(frozen=True)
class ApiConfig:
    """Transport and authentication settings for one API."""

    api_id: str
    adapter: str
    model_family: str
    quota_scope: str
    base_url: str | None = None
    endpoint_template: str | None = None
    auth_header: str | None = None
    request_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchTokenBudgetProfile:
    """Provider queue limits for one configured account."""

    account_id: str
    adapter: str
    tier: str
    safety_ratio: float
    estimator: str
    encoding: str | None
    verified_at: str
    source_url: str
    models: dict[str, int]


@dataclass(frozen=True)
class ApiCatalog:
    """Validated repository-level API infrastructure."""

    source: Path
    source_sha256: str
    accounts: dict[str, AccountConfig]
    apis: dict[str, ApiConfig]
    batch_token_budgets: dict[str, BatchTokenBudgetProfile]


@dataclass(frozen=True)
class ReasoningConfig:
    """Normalized reasoning request."""

    mode: ReasoningMode
    effort: str | None = None


@dataclass(frozen=True)
class InferenceConfig:
    """Provider-independent generation settings."""

    temperature: float | None
    max_output_tokens: int
    reasoning: ReasoningConfig


@dataclass(frozen=True)
class CapabilityConfig:
    """Declared generation capabilities for a model deployment."""

    reasoning_modes: tuple[ReasoningMode, ...]
    reasoning_efforts: tuple[str, ...]
    temperature_allowed_during_reasoning: bool
    seed_supported: bool


@dataclass(frozen=True)
class LimitConfig:
    """Per-account limits for a logical judge."""

    requests_per_minute: int | None = None
    requests_per_period: int | None = None
    period: Period | None = None
    timezone: str = "UTC"
    max_concurrency: int = 1
    batch_max_requests: int | None = None
    batch_max_input_tokens: int | None = None
    batch_max_concurrency: int = 1


@dataclass(frozen=True)
class ResolvedBatchTokenBudget:
    """Effective token ceiling and estimator for one batch judge."""

    provider_limit: int
    max_input_tokens: int
    estimator: str
    encoding: str | None
    tier: str | None
    safety_ratio: float | None
    verified_at: str | None
    source_url: str | None


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved logical judge configuration."""

    alias: str
    api: ApiConfig
    model: str
    accounts: tuple[AccountConfig, ...]
    limits: LimitConfig
    batch_token_budget: ResolvedBatchTokenBudget | None
    request_timeout: float | None
    inference: InferenceConfig
    capabilities: CapabilityConfig
    provider_options: dict[str, Any]
    generation_seed: int | None
    routing_seed: int
    experiment_fingerprint: str
    routing_fingerprint: str

@dataclass(frozen=True)
class AppConfig:
    """Resolved API infrastructure and one judge catalog."""

    api_source: Path
    api_source_sha256: str
    judge_source: Path
    judge_source_sha256: str
    accounts: dict[str, AccountConfig]
    apis: dict[str, ApiConfig]
    batch_token_budgets: dict[str, BatchTokenBudgetProfile]
    judges: dict[str, JudgeConfig]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    with path.open(encoding="utf-8") as handle:
        return _mapping(json.load(handle, object_pairs_hook=unique_object), label)


def _unknown(item: dict[str, Any], allowed: set[str], name: str) -> None:
    extra = set(item) - allowed
    if extra:
        raise ValueError(f"{name} has unknown fields: {sorted(extra)}")


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _contains_secret(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in {"api_key", "token", "secret", "password"}
                or normalized.endswith(("_token", "_secret", "_password"))
            ):
                return True
            if _contains_secret(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return False


def load_api_config(path: Path) -> ApiCatalog:
    """Load independently versioned API infrastructure."""

    source = path.resolve()
    root = _read_json(source, "API configuration")
    if root.get("version") != 1:
        raise ValueError("API configuration version must be 1")
    _unknown(
        root,
        {"version", "accounts", "apis", "batch_token_budgets"},
        "API configuration",
    )
    if _contains_secret(root):
        raise ValueError("API configuration may reference api_key_env but not secrets")

    accounts: dict[str, AccountConfig] = {}
    for account_id, value in _mapping(root.get("accounts"), "accounts").items():
        item = _mapping(value, f"accounts.{account_id}")
        _unknown(item, {"api_key_env", "api_base"}, f"accounts.{account_id}")
        key_env = item.get("api_key_env")
        if key_env is not None and (
            not isinstance(key_env, str) or not key_env.strip()
        ):
            raise ValueError(f"accounts.{account_id}.api_key_env must be non-empty")
        accounts[account_id] = AccountConfig(
            account_id, key_env, item.get("api_base")
        )

    apis: dict[str, ApiConfig] = {}
    for api_id, value in _mapping(root.get("apis"), "apis").items():
        item = _mapping(value, f"apis.{api_id}")
        _unknown(
            item,
            {
                "adapter", "base_url", "endpoint_template", "auth_header",
                "model_family", "quota_scope", "request_defaults",
            },
            f"apis.{api_id}",
        )
        adapter = item.get("adapter")
        if adapter not in SUPPORTED_ADAPTERS:
            raise ValueError(
                f"apis.{api_id}.adapter must be one of {sorted(SUPPORTED_ADAPTERS)}"
            )
        defaults = item.get("request_defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError(f"apis.{api_id}.request_defaults must be an object")
        conflicts = sorted(set(defaults) & _NORMALIZED_CONFLICTS)
        if conflicts:
            raise ValueError(
                f"apis.{api_id}.request_defaults conflicts with normalized fields: "
                f"{conflicts}"
            )
        if adapter != "litellm" and not item.get("endpoint_template"):
            raise ValueError(f"apis.{api_id}.endpoint_template is required")
        model_family = item.get("model_family", api_id)
        if not isinstance(model_family, str) or not model_family.strip():
            raise ValueError(f"apis.{api_id}.model_family must be non-empty")
        quota_scope = item.get("quota_scope", "model")
        if quota_scope not in {"model", "api"}:
            raise ValueError(
                f"apis.{api_id}.quota_scope must be 'model' or 'api'"
            )
        apis[api_id] = ApiConfig(
            api_id=api_id,
            adapter=adapter,
            model_family=model_family.strip(),
            quota_scope=quota_scope,
            base_url=item.get("base_url"),
            endpoint_template=item.get("endpoint_template"),
            auth_header=item.get("auth_header"),
            request_defaults=dict(defaults),
        )
    budgets: dict[str, BatchTokenBudgetProfile] = {}
    for account_id, value in _mapping(
        root.get("batch_token_budgets", {}), "batch_token_budgets"
    ).items():
        if account_id not in accounts:
            raise ValueError(
                f"batch_token_budgets references unknown account {account_id!r}"
            )
        item = _mapping(value, f"batch_token_budgets.{account_id}")
        _unknown(
            item,
            {
                "adapter", "tier", "safety_ratio", "estimator", "encoding",
                "verified_at", "source_url", "models",
            },
            f"batch_token_budgets.{account_id}",
        )
        adapter = item.get("adapter")
        if adapter not in {"google_gemini_batch", "openai_batch"}:
            raise ValueError(
                f"batch_token_budgets.{account_id}.adapter must be a batch adapter"
            )
        tier = item.get("tier")
        estimator = item.get("estimator")
        verified_at = item.get("verified_at")
        source_url = item.get("source_url")
        for field_name, field_value in {
            "tier": tier,
            "estimator": estimator,
            "verified_at": verified_at,
            "source_url": source_url,
        }.items():
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"batch_token_budgets.{account_id}.{field_name} "
                    "must be non-empty"
                )
        if estimator not in {"tiktoken_json", "gemini_local_conservative"}:
            raise ValueError(
                f"batch_token_budgets.{account_id}.estimator is unsupported"
            )
        encoding = item.get("encoding")
        if encoding is not None and (
            not isinstance(encoding, str) or not encoding.strip()
        ):
            raise ValueError(
                f"batch_token_budgets.{account_id}.encoding must be non-empty"
            )
        if estimator == "tiktoken_json" and encoding is None:
            raise ValueError(
                f"batch_token_budgets.{account_id}.encoding is required"
            )
        if estimator != "tiktoken_json" and encoding is not None:
            raise ValueError(
                f"batch_token_budgets.{account_id}.encoding requires tiktoken_json"
            )
        safety_ratio = item.get("safety_ratio")
        if (
            not isinstance(safety_ratio, (int, float))
            or isinstance(safety_ratio, bool)
            or not 0 < safety_ratio <= 1
        ):
            raise ValueError(
                f"batch_token_budgets.{account_id}.safety_ratio "
                "must be in (0, 1]"
            )
        raw_models = _mapping(
            item.get("models"), f"batch_token_budgets.{account_id}.models"
        )
        if any(not model.strip() for model in raw_models):
            raise ValueError(
                f"batch_token_budgets.{account_id}.models has an empty model ID"
            )
        models = {
            model: _optional_positive_int(
                limit, f"batch_token_budgets.{account_id}.models.{model}"
            )
            for model, limit in raw_models.items()
        }
        if not models or any(limit is None for limit in models.values()):
            raise ValueError(
                f"batch_token_budgets.{account_id}.models must not be empty"
            )
        budgets[account_id] = BatchTokenBudgetProfile(
            account_id=account_id,
            adapter=adapter,
            tier=tier.strip(),
            safety_ratio=float(safety_ratio),
            estimator=estimator,
            encoding=encoding,
            verified_at=verified_at.strip(),
            source_url=source_url.strip(),
            models={model: int(limit) for model, limit in models.items()},
        )
    return ApiCatalog(source, file_sha256(source), accounts, apis, budgets)


_NORMALIZED_CONFLICTS = {
    "temperature", "max_tokens", "maxTokens", "max_output_tokens",
    "maxOutputTokens", "max_completion_tokens", "reasoning", "reasoning_effort",
    "thinking", "seed",
}


def _parse_inference(alias: str, item: dict[str, Any]) -> tuple[InferenceConfig, CapabilityConfig]:
    raw = _mapping(item.get("inference"), f"judges.{alias}.inference")
    _unknown(raw, {"temperature", "max_output_tokens", "reasoning"},
             f"judges.{alias}.inference")
    temperature = raw.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
    ):
        raise ValueError(f"judges.{alias}.inference.temperature must be numeric or null")
    max_tokens = _optional_positive_int(
        raw.get("max_output_tokens"), f"judges.{alias}.inference.max_output_tokens"
    )
    if max_tokens is None:
        raise ValueError(f"judges.{alias}.inference.max_output_tokens is required")
    reasoning = _mapping(raw.get("reasoning"), f"judges.{alias}.inference.reasoning")
    _unknown(reasoning, {"mode", "effort"}, f"judges.{alias}.inference.reasoning")
    mode = reasoning.get("mode")
    if mode not in {"disabled", "enabled", "required"}:
        raise ValueError(f"judges.{alias}.inference.reasoning.mode is invalid")
    effort = reasoning.get("effort")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise ValueError(f"judges.{alias}.inference.reasoning.effort must be non-empty")
    if mode == "disabled" and effort is not None:
        raise ValueError(f"judges.{alias} cannot set reasoning effort when disabled")

    caps = _mapping(item.get("capabilities"), f"judges.{alias}.capabilities")
    _unknown(
        caps,
        {
            "reasoning_modes", "reasoning_efforts",
            "temperature_allowed_during_reasoning", "seed_supported",
        },
        f"judges.{alias}.capabilities",
    )
    modes = caps.get("reasoning_modes")
    if (
        not isinstance(modes, list) or not modes or len(modes) != len(set(modes))
        or any(value not in {"disabled", "enabled", "required"} for value in modes)
    ):
        raise ValueError(f"judges.{alias}.capabilities.reasoning_modes is invalid")
    efforts = caps.get("reasoning_efforts", [])
    if (
        not isinstance(efforts, list) or len(efforts) != len(set(efforts))
        or any(not isinstance(value, str) or not value for value in efforts)
    ):
        raise ValueError(f"judges.{alias}.capabilities.reasoning_efforts is invalid")
    temp_reasoning = caps.get("temperature_allowed_during_reasoning")
    seed_supported = caps.get("seed_supported")
    if not isinstance(temp_reasoning, bool) or not isinstance(seed_supported, bool):
        raise ValueError(f"judges.{alias}.capabilities boolean fields are required")
    if set(modes) == {"disabled"} and efforts:
        raise ValueError(
            f"judges.{alias}.capabilities cannot list reasoning efforts for a "
            "non-reasoning model"
        )
    if mode not in modes:
        raise ValueError(f"judges.{alias} requests unsupported reasoning mode {mode!r}")
    if effort is not None and effort not in efforts:
        raise ValueError(f"judges.{alias} requests unsupported reasoning effort {effort!r}")
    if mode != "disabled" and temperature is not None and not temp_reasoning:
        raise ValueError(f"judges.{alias} cannot use temperature during reasoning")
    return (
        InferenceConfig(
            None if temperature is None else float(temperature),
            max_tokens,
            ReasoningConfig(mode, effort),
        ),
        CapabilityConfig(tuple(modes), tuple(efforts), temp_reasoning, seed_supported),
    )


def load_judge_config(path: Path, api_catalog: ApiCatalog) -> AppConfig:
    """Load a judge catalog and validate its infrastructure references."""

    source = path.resolve()
    root = _read_json(source, "judge configuration")
    if root.get("version") != 1:
        raise ValueError("judge configuration version must be 1")
    _unknown(root, {"version", "judges"}, "judge configuration")
    if _contains_secret(root):
        raise ValueError("judge configuration must not contain secrets")
    raw_judges = _mapping(root.get("judges"), "judges")
    if not raw_judges:
        raise ValueError("judge configuration must define at least one judge")
    judges: dict[str, JudgeConfig] = {}
    identities: dict[str, str] = {}
    for alias, value in raw_judges.items():
        if not isinstance(alias, str) or not alias:
            raise ValueError("judge aliases must be non-empty strings")
        item = _mapping(value, f"judges.{alias}")
        _unknown(
            item,
            {
                "api", "model", "accounts", "limits", "inference", "capabilities",
                "provider_options", "generation_seed", "routing_seed",
                "request_timeout",
            },
            f"judges.{alias}",
        )
        api_id = item.get("api")
        if api_id not in api_catalog.apis:
            raise ValueError(f"judges.{alias}.api references unknown API {api_id!r}")
        model = item.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"judges.{alias}.model must be a non-empty string")
        account_ids = item.get("accounts")
        if not isinstance(account_ids, list) or not account_ids:
            raise ValueError(f"judges.{alias}.accounts must be a non-empty list")
        if len(account_ids) != len(set(account_ids)):
            raise ValueError(f"judges.{alias}.accounts contains duplicates")
        missing = [name for name in account_ids if name not in api_catalog.accounts]
        if missing:
            raise ValueError(f"judges.{alias} references unknown accounts: {missing}")
        inference, capabilities = _parse_inference(alias, item)
        options = item.get("provider_options", {})
        if not isinstance(options, dict):
            raise ValueError(f"judges.{alias}.provider_options must be an object")
        conflicts = sorted(set(options) & _NORMALIZED_CONFLICTS)
        if conflicts:
            raise ValueError(
                f"judges.{alias}.provider_options conflicts with normalized fields: "
                f"{conflicts}"
            )
        generation_seed = item.get("generation_seed")
        if generation_seed is not None and (
            not isinstance(generation_seed, int) or isinstance(generation_seed, bool)
        ):
            raise ValueError(f"judges.{alias}.generation_seed must be an integer or null")
        if generation_seed is not None and not capabilities.seed_supported:
            raise ValueError(f"judges.{alias} sets a seed but the model does not support it")
        routing_seed = item.get("routing_seed", 0)
        if not isinstance(routing_seed, int) or isinstance(routing_seed, bool):
            raise ValueError(f"judges.{alias}.routing_seed must be an integer")
        request_timeout = item.get("request_timeout")
        if request_timeout is not None and (
            not isinstance(request_timeout, (int, float))
            or isinstance(request_timeout, bool)
            or request_timeout <= 0
        ):
            raise ValueError(
                f"judges.{alias}.request_timeout must be a positive number"
            )

        raw_limits = _mapping(item.get("limits", {}), f"judges.{alias}.limits")
        _unknown(
            raw_limits,
            {
                "requests_per_minute", "requests_per_period", "period", "timezone",
                "max_concurrency", "batch_max_requests",
                "batch_max_input_tokens", "batch_max_concurrency",
            },
            f"judges.{alias}.limits",
        )
        period = raw_limits.get("period")
        period_requests = _optional_positive_int(
            raw_limits.get("requests_per_period"),
            f"judges.{alias}.limits.requests_per_period",
        )
        if period_requests is not None and period not in {"day", "week"}:
            raise ValueError(f"judges.{alias}.limits.period must be day or week")
        if period is not None and period_requests is None:
            raise ValueError(
                f"judges.{alias}.limits.requests_per_period is required with period"
            )
        limits = LimitConfig(
            requests_per_minute=_optional_positive_int(
                raw_limits.get("requests_per_minute"),
                f"judges.{alias}.limits.requests_per_minute",
            ),
            requests_per_period=period_requests,
            period=period,
            timezone=raw_limits.get("timezone", "UTC"),
            max_concurrency=_optional_positive_int(
                raw_limits.get("max_concurrency", 1),
                f"judges.{alias}.limits.max_concurrency",
            ) or 1,
            batch_max_requests=_optional_positive_int(
                raw_limits.get("batch_max_requests"),
                f"judges.{alias}.limits.batch_max_requests",
            ),
            batch_max_input_tokens=_optional_positive_int(
                raw_limits.get("batch_max_input_tokens"),
                f"judges.{alias}.limits.batch_max_input_tokens",
            ),
            batch_max_concurrency=_optional_positive_int(
                raw_limits.get("batch_max_concurrency", 1),
                f"judges.{alias}.limits.batch_max_concurrency",
            ) or 1,
        )
        batch_adapter = api_catalog.apis[api_id].adapter
        if (
            limits.batch_max_requests is not None
            and batch_adapter
            not in {"google_gemini_batch", "openai_batch", "anthropic_batch"}
        ):
            raise ValueError(
                f"judges.{alias} batch limit configuration requires a batch adapter"
            )
        if (
            limits.batch_max_input_tokens is not None
            and batch_adapter not in {"google_gemini_batch", "openai_batch"}
        ):
            raise ValueError(
                f"judges.{alias}.limits.batch_max_input_tokens requires a "
                "token-budget batch adapter"
            )
        if (
            limits.batch_max_concurrency != 1
            and batch_adapter != "openai_batch"
        ):
            raise ValueError(
                f"judges.{alias}.limits.batch_max_concurrency currently "
                "requires the OpenAI batch adapter"
            )
        try:
            ZoneInfo(limits.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(
                f"judges.{alias}.limits.timezone is unknown: {limits.timezone!r}"
            ) from error

        api = api_catalog.apis[api_id]
        batch_budget: ResolvedBatchTokenBudget | None = None
        if api.adapter in {"google_gemini_batch", "openai_batch"}:
            account_id = str(account_ids[0])
            profile = api_catalog.batch_token_budgets.get(account_id)
            provider_limit = (
                profile.models.get(model)
                if profile is not None and profile.adapter == api.adapter
                else None
            )
            override = limits.batch_max_input_tokens
            if provider_limit is None and override is None:
                raise ValueError(
                    f"judges.{alias} has no batch token budget for "
                    f"account {account_id!r} and model {model!r}"
                )
            if (
                override is not None
                and provider_limit is not None
                and override > provider_limit
            ):
                raise ValueError(
                    f"judges.{alias}.limits.batch_max_input_tokens exceeds "
                    f"the configured provider limit {provider_limit}"
                )
            estimator = (
                profile.estimator
                if profile is not None and profile.adapter == api.adapter
                else (
                    "tiktoken_json"
                    if api.adapter == "openai_batch"
                    else "gemini_local_conservative"
                )
            )
            encoding = (
                profile.encoding
                if profile is not None and profile.adapter == api.adapter
                else ("o200k_base" if api.adapter == "openai_batch" else None)
            )
            effective_limit = override or int(
                int(provider_limit) * float(profile.safety_ratio)
            )
            if limits.batch_max_concurrency > effective_limit:
                raise ValueError(
                    f"judges.{alias}.limits.batch_max_concurrency exceeds "
                    f"the safe input-token budget {effective_limit}"
                )
            batch_budget = ResolvedBatchTokenBudget(
                provider_limit=int(provider_limit or override),
                max_input_tokens=effective_limit,
                estimator=estimator,
                encoding=encoding,
                tier=profile.tier if profile is not None else None,
                safety_ratio=(
                    profile.safety_ratio
                    if override is None and profile is not None
                    else None
                ),
                verified_at=profile.verified_at if profile is not None else None,
                source_url=profile.source_url if profile is not None else None,
            )
        scientific = {
            "alias": alias, "api": api_id, "adapter": api.adapter,
            "model": model, "inference": item.get("inference"),
            "capabilities": item.get("capabilities"),
            "provider_options": options, "generation_seed": generation_seed,
        }
        identity = fingerprint({
            "api": api_id, "model": model, "inference": item.get("inference"),
            "provider_options": options, "generation_seed": generation_seed,
        })
        if identity in identities:
            raise ValueError(
                f"judges.{alias} duplicates model/profile identity of "
                f"{identities[identity]!r}"
            )
        identities[identity] = alias
        routing = {
            "accounts": account_ids, "limits": raw_limits,
            "request_timeout": request_timeout, "routing_seed": routing_seed,
        }
        judges[alias] = JudgeConfig(
            alias=alias,
            api=api,
            model=model,
            accounts=tuple(api_catalog.accounts[name] for name in account_ids),
            limits=limits,
            batch_token_budget=batch_budget,
            request_timeout=(
                float(request_timeout) if request_timeout is not None else None
            ),
            inference=inference,
            capabilities=capabilities,
            provider_options=dict(options),
            generation_seed=generation_seed,
            routing_seed=routing_seed,
            experiment_fingerprint=fingerprint(scientific),
            routing_fingerprint=fingerprint(routing),
        )
    return AppConfig(
        api_catalog.source, api_catalog.source_sha256, source, file_sha256(source),
        api_catalog.accounts, api_catalog.apis, api_catalog.batch_token_budgets,
        judges,
    )


def load_config(api_path: Path, judge_path: Path | None = None) -> AppConfig:
    """Load split API and judge configuration.

    The one-file version-2 schema was deliberately removed.
    """

    if judge_path is None:
        raise ValueError(
            "combined configuration is unsupported; pass both API and judge "
            "configuration paths"
        )
    return load_judge_config(judge_path, load_api_config(api_path))
