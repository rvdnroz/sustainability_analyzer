from __future__ import annotations

import email.utils
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from .llm_client import (
    BaseLLMClient,
    LLMAuthenticationError,
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMDependencyError,
    LLMMessage,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUsage,
)

try:
    import httpx
except ImportError:  # pragma: no cover - yalnızca bağımlılık eksikliğinde
    httpx = None  # type: ignore[assignment]


JsonModePolicy = Literal["auto", "required", "disabled"]


_PROVIDER_DEFAULTS: dict[LLMProvider, dict[str, Any]] = {
    LLMProvider.GROQ: {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_environment_variable": "GROQ_API_KEY",
        "requires_api_key": True,
    },
    LLMProvider.OPENROUTER: {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_environment_variable": "OPENROUTER_API_KEY",
        "requires_api_key": True,
    },
    LLMProvider.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "api_key_environment_variable": None,
        "requires_api_key": False,
    },
    LLMProvider.OPENAI_COMPATIBLE: {
        "base_url": None,
        "api_key_environment_variable": "OPENAI_API_KEY",
        "requires_api_key": False,
    },
}


@dataclass(slots=True, frozen=True)
class OpenAICompatibleClientConfig:
    """OpenAI Chat Completions uyumlu sağlayıcıların ortak ayarları."""

    provider: LLMProvider | str
    model: str

    base_url: str | None = None
    api_key_environment_variable: str | None = None
    requires_api_key: bool | None = None

    temperature: float = 0.0
    maximum_output_tokens: int = 2200
    top_p: float | None = None

    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 15.0

    maximum_retries: int = 3
    initial_retry_delay_seconds: float = 1.0
    maximum_retry_delay_seconds: float = 20.0
    retry_jitter_ratio: float = 0.20

    json_mode_policy: JsonModePolicy = "auto"
    retry_without_json_mode_when_unsupported: bool = True

    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = "Sustainability Analyzer"

    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    verify_ssl: bool = True
    trust_environment: bool = True

    def __post_init__(self) -> None:
        provider = LLMProvider.normalize(self.provider)
        object.__setattr__(self, "provider", provider)

        defaults = _PROVIDER_DEFAULTS.get(provider, {})

        base_url = _clean_text(self.base_url) or defaults.get("base_url")
        if not base_url:
            raise ValueError(
                "OpenAI-compatible sağlayıcı için base_url tanımlanmalıdır."
            )
        object.__setattr__(self, "base_url", _normalize_base_url(base_url))

        environment_variable = (
            _clean_text(self.api_key_environment_variable)
            or defaults.get("api_key_environment_variable")
        )
        object.__setattr__(
            self,
            "api_key_environment_variable",
            environment_variable or None,
        )

        requires_api_key = self.requires_api_key
        if requires_api_key is None:
            requires_api_key = bool(defaults.get("requires_api_key", False))
        object.__setattr__(self, "requires_api_key", requires_api_key)

        if not _clean_text(self.model):
            raise ValueError("Model adı boş olamaz.")

        BaseLLMClient.validate_generation_parameters(
            temperature=self.temperature,
            max_tokens=self.maximum_output_tokens,
            top_p=self.top_p,
        )

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds pozitif olmalıdır.")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds pozitif olmalıdır.")
        if self.maximum_retries < 0:
            raise ValueError("maximum_retries negatif olamaz.")
        if self.initial_retry_delay_seconds <= 0:
            raise ValueError(
                "initial_retry_delay_seconds pozitif olmalıdır."
            )
        if self.maximum_retry_delay_seconds <= 0:
            raise ValueError(
                "maximum_retry_delay_seconds pozitif olmalıdır."
            )
        if not 0.0 <= self.retry_jitter_ratio <= 1.0:
            raise ValueError("retry_jitter_ratio 0 ile 1 arasında olmalıdır.")
        if self.json_mode_policy not in {"auto", "required", "disabled"}:
            raise ValueError(
                "json_mode_policy auto, required veya disabled olmalıdır."
            )

        normalized_headers = {
            str(key).strip(): str(value).strip()
            for key, value in self.extra_headers.items()
            if str(key).strip() and str(value).strip()
        }
        object.__setattr__(self, "extra_headers", normalized_headers)
        object.__setattr__(self, "extra_body", dict(self.extra_body))

    @property
    def chat_completions_url(self) -> str:
        assert self.base_url is not None
        return f"{self.base_url}/chat/completions"


class OpenAICompatibleClient(BaseLLMClient):
    """
    Groq, OpenRouter, Ollama ve özel OpenAI-compatible sunucular için istemci.

    İstemci senkron `POST /chat/completions` çağrısı yapar. Sağlayıcı SDK'sı
    gerektirmez; yalnızca projede zaten kullanılan `httpx` paketine dayanır.
    """

    _RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
    _AUTHENTICATION_STATUS_CODES = {401, 403}

    _JSON_MODE_ERROR_TERMS = (
        "response_format",
        "json mode",
        "json_object",
        "structured output",
        "structured outputs",
        "not support json",
        "does not support json",
        "unsupported json",
        # Groq JSON Object Mode, model geçerli JSON üretemediğinde
        # HTTP 400 ile bu ifadeleri döndürebilir. auto politikasında
        # response_format kaldırılarak prompt-temelli JSON denemesi yapılır.
        "failed to validate json",
        "failed_generation",
        "please adjust your prompt",
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: OpenAICompatibleClientConfig,
        http_client: Any | None = None,
    ) -> None:
        if httpx is None:
            raise LLMDependencyError(
                "OpenAI-compatible istemci için 'httpx' paketi gereklidir. "
                "Kurulum: pip install httpx",
                provider=LLMProvider.normalize(config.provider).value,
                model=config.model,
            )

        self.config = config
        self.provider_name = LLMProvider.normalize(config.provider).value

        environment_api_key = ""
        if config.api_key_environment_variable:
            environment_api_key = _clean_text(
                os.getenv(config.api_key_environment_variable)
            )

        self._api_key = _clean_text(api_key) or environment_api_key
        if config.requires_api_key and not self._api_key:
            variable_name = config.api_key_environment_variable or "API_KEY"
            raise LLMAuthenticationError(
                f"{self.provider_name} API anahtarı bulunamadı. "
                f"'{variable_name}' ortam değişkenini tanımlayın.",
                provider=self.provider_name,
                model=config.model,
            )

        self._owns_http_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(
                timeout=config.timeout_seconds,
                connect=config.connect_timeout_seconds,
            ),
            verify=config.verify_ssl,
            trust_env=config.trust_environment,
            follow_redirects=True,
        )

    @classmethod
    def for_groq(
        cls,
        model: str,
        *,
        api_key: str | None = None,
        **config_overrides: Any,
    ) -> OpenAICompatibleClient:
        return cls(
            api_key,
            config=OpenAICompatibleClientConfig(
                provider=LLMProvider.GROQ,
                model=model,
                **config_overrides,
            ),
        )

    @classmethod
    def for_openrouter(
        cls,
        model: str,
        *,
        api_key: str | None = None,
        **config_overrides: Any,
    ) -> OpenAICompatibleClient:
        return cls(
            api_key,
            config=OpenAICompatibleClientConfig(
                provider=LLMProvider.OPENROUTER,
                model=model,
                **config_overrides,
            ),
        )

    @classmethod
    def for_ollama(
        cls,
        model: str,
        *,
        base_url: str | None = None,
        **config_overrides: Any,
    ) -> OpenAICompatibleClient:
        return cls(
            None,
            config=OpenAICompatibleClientConfig(
                provider=LLMProvider.OLLAMA,
                model=model,
                base_url=base_url,
                **config_overrides,
            ),
        )

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        model: str | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        normalized_messages = self.validate_messages(messages)

        resolved_temperature = (
            self.config.temperature
            if temperature is None
            else temperature
        )
        resolved_max_tokens = (
            self.config.maximum_output_tokens
            if max_tokens is None
            else max_tokens
        )
        resolved_top_p = self.config.top_p if top_p is None else top_p
        resolved_model = _clean_text(model) or self.config.model

        self.validate_generation_parameters(
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            top_p=resolved_top_p,
            top_k=top_k,
        )

        body: dict[str, Any] = {
            "model": resolved_model,
            "messages": list(normalized_messages),
            "stream": False,
            "temperature": resolved_temperature,
        }

        if self.provider_name in {
            LLMProvider.GROQ.value,
            LLMProvider.OPENROUTER.value,
        }:
            body["max_completion_tokens"] = resolved_max_tokens
        else:
            body["max_tokens"] = resolved_max_tokens

        if resolved_top_p is not None:
            body["top_p"] = resolved_top_p

        # Qwen 3.6'nın düşünme çıktısı JSON üretimini gereksiz yere
        # zorlaştırabilir. Groq üzerinde yapılandırılmış veri çıkarımında
        # non-thinking modunu kullan.
        if (
            self.provider_name == LLMProvider.GROQ.value
            and resolved_model.casefold() == "qwen/qwen3.6-27b"
        ):
            body.setdefault("reasoning_effort", "none")

        # top_k OpenAI Chat Completions sözleşmesinin ortak bir alanı değildir.
        # Sağlayıcılar arasında taşınabilirliği korumak için gönderilmez.
        ignored_parameters: dict[str, Any] = {}
        if top_k is not None:
            ignored_parameters["top_k"] = top_k
        if thinking_level is not None:
            ignored_parameters["thinking_level"] = thinking_level

        json_mode_requested = _is_json_response_format(response_format)
        json_mode_applied = (
            json_mode_requested
            and self.config.json_mode_policy != "disabled"
        )
        if json_mode_applied:
            body["response_format"] = {"type": "json_object"}

        body.update(dict(self.config.extra_body))

        headers = self._build_headers()
        started_at = time.perf_counter()

        payload, attempts, fallback_used = self._post_with_retries(
            body=body,
            headers=headers,
            resolved_model=resolved_model,
            json_mode_applied=json_mode_applied,
        )

        latency_seconds = time.perf_counter() - started_at
        response = self._parse_response(
            payload,
            requested_model=resolved_model,
            attempts=attempts,
            latency_seconds=latency_seconds,
            json_mode_requested=json_mode_requested,
            json_mode_applied=(json_mode_applied and not fallback_used),
            json_mode_fallback_used=fallback_used,
            ignored_parameters=ignored_parameters,
        )

        if json_mode_requested:
            self._validate_json_content(response)

        return response

    def close(self) -> None:
        if self._owns_http_client and self._client is not None:
            self._client.close()

    def _post_with_retries(
        self,
        *,
        body: dict[str, Any],
        headers: Mapping[str, str],
        resolved_model: str,
        json_mode_applied: bool,
    ) -> tuple[Mapping[str, Any], int, bool]:
        attempt = 0
        json_mode_fallback_used = False
        active_body = dict(body)

        while True:
            attempt += 1
            try:
                response = self._client.post(
                    self.config.chat_completions_url,
                    headers=dict(headers),
                    json=active_body,
                )
            except Exception as exc:
                mapped_error = self._map_transport_error(
                    exc,
                    model=resolved_model,
                )
                if mapped_error.retryable and attempt <= self.config.maximum_retries:
                    self._sleep_before_retry(attempt, retry_after=None)
                    continue
                raise mapped_error from exc

            if 200 <= response.status_code < 300:
                try:
                    payload = response.json()
                except Exception as exc:
                    raise LLMResponseError(
                        "Model sağlayıcısı geçerli JSON HTTP cevabı döndürmedi.",
                        provider=self.provider_name,
                        model=resolved_model,
                        status_code=response.status_code,
                        metadata={
                            "response_text": _truncate(response.text, 1200),
                        },
                    ) from exc

                if not isinstance(payload, Mapping):
                    raise LLMResponseError(
                        "Model sağlayıcısının cevabı JSON nesnesi olmalıdır.",
                        provider=self.provider_name,
                        model=resolved_model,
                        status_code=response.status_code,
                    )

                if payload.get("error"):
                    error_message = self._extract_error_message(payload)
                    raise LLMResponseError(
                        error_message,
                        provider=self.provider_name,
                        model=resolved_model,
                        status_code=response.status_code,
                        metadata={"error": payload.get("error")},
                    )

                return payload, attempt, json_mode_fallback_used

            error_message = self._extract_error_message_from_response(response)

            if (
                response.status_code == 400
                and json_mode_applied
                and not json_mode_fallback_used
                and self.config.json_mode_policy == "auto"
                and self.config.retry_without_json_mode_when_unsupported
                and self._looks_like_unsupported_json_mode(error_message)
            ):
                active_body = dict(active_body)
                active_body.pop("response_format", None)
                json_mode_fallback_used = True
                continue

            mapped_error = self._map_http_error(
                status_code=response.status_code,
                message=error_message,
                model=resolved_model,
                response_headers=response.headers,
            )

            if mapped_error.retryable and attempt <= self.config.maximum_retries:
                retry_after = self._parse_retry_after(response.headers)
                self._sleep_before_retry(attempt, retry_after=retry_after)
                continue

            raise mapped_error

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "sustainability-analyzer/1.0",
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self.provider_name == LLMProvider.OPENROUTER.value:
            if _clean_text(self.config.openrouter_http_referer):
                headers["HTTP-Referer"] = _clean_text(
                    self.config.openrouter_http_referer
                )
            if _clean_text(self.config.openrouter_app_title):
                headers["X-OpenRouter-Title"] = _clean_text(
                    self.config.openrouter_app_title
                )

        headers.update(dict(self.config.extra_headers))
        return headers

    def _parse_response(
        self,
        payload: Mapping[str, Any],
        *,
        requested_model: str,
        attempts: int,
        latency_seconds: float,
        json_mode_requested: bool,
        json_mode_applied: bool,
        json_mode_fallback_used: bool,
        ignored_parameters: Mapping[str, Any],
    ) -> LLMResponse:
        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise LLMResponseError(
                "Model cevabında choices listesi bulunamadı.",
                provider=self.provider_name,
                model=requested_model,
                metadata={"response": dict(payload)},
            )
        if not choices:
            raise LLMResponseError(
                "Model sağlayıcısı boş choices listesi döndürdü.",
                provider=self.provider_name,
                model=requested_model,
            )

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMResponseError(
                "İlk model seçimi geçerli bir JSON nesnesi değildir.",
                provider=self.provider_name,
                model=requested_model,
            )

        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMResponseError(
                "Model cevabında message nesnesi bulunamadı.",
                provider=self.provider_name,
                model=requested_model,
            )

        content = self._coerce_message_content(message.get("content"))
        if not content:
            refusal = _clean_text(message.get("refusal"))
            reasoning = _clean_text(message.get("reasoning"))
            detail = refusal or reasoning
            suffix = f" Ayrıntı: {_truncate(detail, 500)}" if detail else ""
            raise LLMResponseError(
                f"Model kullanılabilir bir cevap metni döndürmedi.{suffix}",
                provider=self.provider_name,
                model=requested_model,
                metadata={
                    "finish_reason": first_choice.get("finish_reason"),
                    "message": dict(message),
                },
            )

        response_model = _clean_text(payload.get("model")) or requested_model
        finish_reason = _clean_text(first_choice.get("finish_reason")) or None
        usage = self._parse_usage(payload.get("usage"))

        metadata: dict[str, Any] = {
            "attempts": attempts,
            "latency_seconds": latency_seconds,
            "json_mode_requested": json_mode_requested,
            "json_mode_applied": json_mode_applied,
            "json_mode_fallback_used": json_mode_fallback_used,
        }
        if ignored_parameters:
            metadata["ignored_parameters"] = dict(ignored_parameters)

        if message.get("reasoning") is not None:
            metadata["reasoning_present"] = True
        if payload.get("system_fingerprint") is not None:
            metadata["system_fingerprint"] = payload.get("system_fingerprint")

        return LLMResponse(
            content=content,
            model=response_model,
            provider=self.provider_name,
            finish_reason=finish_reason,
            usage=usage,
            response_id=_clean_text(payload.get("id")) or None,
            raw_response=payload,
            metadata=metadata,
        )

    def _parse_usage(self, usage_payload: Any) -> LLMUsage | None:
        if not isinstance(usage_payload, Mapping):
            return None

        prompt_details = usage_payload.get("prompt_tokens_details")
        completion_details = usage_payload.get("completion_tokens_details")

        cached_tokens = None
        if isinstance(prompt_details, Mapping):
            cached_tokens = _as_non_negative_int(
                prompt_details.get("cached_tokens")
            )

        reasoning_tokens = None
        if isinstance(completion_details, Mapping):
            reasoning_tokens = _as_non_negative_int(
                completion_details.get("reasoning_tokens")
            )

        prompt_tokens = _as_non_negative_int(usage_payload.get("prompt_tokens"))
        output_tokens = _as_non_negative_int(
            usage_payload.get("completion_tokens")
        )
        total_tokens = _as_non_negative_int(usage_payload.get("total_tokens"))

        if total_tokens is None and (
            prompt_tokens is not None or output_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (output_tokens or 0)

        known_keys = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        }
        metadata = {
            str(key): value
            for key, value in usage_payload.items()
            if key not in known_keys
        }

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            metadata=metadata,
        )

    def _map_transport_error(
        self,
        exc: Exception,
        *,
        model: str,
    ) -> LLMClientError:
        assert httpx is not None

        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                f"{self.provider_name} isteği zaman aşımına uğradı: {exc}",
                provider=self.provider_name,
                model=model,
                retryable=True,
            )

        if isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        ):
            return LLMConnectionError(
                f"{self.provider_name} sunucusuna bağlanılamadı: {exc}",
                provider=self.provider_name,
                model=model,
                retryable=True,
            )

        if isinstance(exc, httpx.HTTPError):
            return LLMConnectionError(
                f"{self.provider_name} HTTP istemci hatası: {exc}",
                provider=self.provider_name,
                model=model,
                retryable=False,
            )

        return LLMClientError(
            f"Beklenmeyen model istemcisi hatası: {exc}",
            provider=self.provider_name,
            model=model,
            retryable=False,
        )

    def _map_http_error(
        self,
        *,
        status_code: int,
        message: str,
        model: str,
        response_headers: Mapping[str, str],
    ) -> LLMClientError:
        metadata = {
            "response_headers": {
                key: value
                for key, value in response_headers.items()
                if key.casefold() in {
                    "retry-after",
                    "x-ratelimit-limit-requests",
                    "x-ratelimit-remaining-requests",
                    "x-ratelimit-reset-requests",
                    "x-ratelimit-limit-tokens",
                    "x-ratelimit-remaining-tokens",
                    "x-ratelimit-reset-tokens",
                }
            }
        }

        if status_code in self._AUTHENTICATION_STATUS_CODES:
            return LLMAuthenticationError(
                message,
                provider=self.provider_name,
                model=model,
                status_code=status_code,
                retryable=False,
                metadata=metadata,
            )

        if status_code == 429:
            return LLMRateLimitError(
                message,
                provider=self.provider_name,
                model=model,
                status_code=status_code,
                retryable=True,
                metadata=metadata,
            )

        if status_code in {408, 504}:
            return LLMTimeoutError(
                message,
                provider=self.provider_name,
                model=model,
                status_code=status_code,
                retryable=True,
                metadata=metadata,
            )

        if status_code in self._RETRYABLE_STATUS_CODES:
            return LLMConnectionError(
                message,
                provider=self.provider_name,
                model=model,
                status_code=status_code,
                retryable=True,
                metadata=metadata,
            )

        return LLMResponseError(
            message,
            provider=self.provider_name,
            model=model,
            status_code=status_code,
            retryable=False,
            metadata=metadata,
        )

    def _sleep_before_retry(
        self,
        attempt: int,
        *,
        retry_after: float | None,
    ) -> None:
        if retry_after is not None:
            delay = min(retry_after, self.config.maximum_retry_delay_seconds)
        else:
            delay = min(
                self.config.initial_retry_delay_seconds * (2 ** (attempt - 1)),
                self.config.maximum_retry_delay_seconds,
            )

        jitter = delay * self.config.retry_jitter_ratio
        if jitter:
            delay += random.uniform(-jitter, jitter)
        time.sleep(max(0.0, delay))

    @staticmethod
    def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
        raw_value = _clean_text(headers.get("retry-after"))
        if not raw_value:
            return None

        try:
            return max(0.0, float(raw_value))
        except ValueError:
            pass

        try:
            target = email.utils.parsedate_to_datetime(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None

        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return max(0.0, (target - now).total_seconds())

    @classmethod
    def _looks_like_unsupported_json_mode(cls, message: str) -> bool:
        normalized = message.casefold()
        return any(term in normalized for term in cls._JSON_MODE_ERROR_TERMS)

    @staticmethod
    def _extract_error_message_from_response(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None

        if isinstance(payload, Mapping):
            message = OpenAICompatibleClient._extract_error_message(payload)
            if message:
                return message

        text = _truncate(_clean_text(getattr(response, "text", "")), 1500)
        return text or f"Model sağlayıcısı HTTP {response.status_code} döndürdü."

    @staticmethod
    def _extract_error_message(payload: Mapping[str, Any]) -> str:
        error = payload.get("error")

        if isinstance(error, str):
            return error.strip()

        if isinstance(error, Mapping):
            for key in ("message", "detail", "error", "type", "code"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in ("message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return "Model sağlayıcısı bir hata döndürdü."

    @staticmethod
    def _coerce_message_content(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            fragments: list[str] = []
            for item in value:
                if isinstance(item, str):
                    cleaned = item.strip()
                elif isinstance(item, Mapping):
                    cleaned = _clean_text(
                        item.get("text")
                        or item.get("content")
                        or item.get("output_text")
                    )
                else:
                    cleaned = ""

                if cleaned:
                    fragments.append(cleaned)

            return "\n".join(fragments).strip()

        return ""

    def _validate_json_content(self, response: LLMResponse) -> None:
        candidate = _strip_json_code_fence(response.content)
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                "Model JSON istendiği hâlde ayrıştırılabilir JSON döndürmedi: "
                f"{exc.msg} (satır {exc.lineno}, sütun {exc.colno}).",
                provider=self.provider_name,
                model=response.model,
                metadata={
                    "response_content": _truncate(response.content, 2000),
                    "json_mode_fallback_used": response.metadata.get(
                        "json_mode_fallback_used"
                    ),
                },
            ) from exc


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("base_url http:// veya https:// ile başlamalıdır.")
    return normalized


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(value: str, maximum_characters: int) -> str:
    if len(value) <= maximum_characters:
        return value
    return value[: maximum_characters - 1].rstrip() + "…"


def _is_json_response_format(value: str | None) -> bool:
    normalized = _clean_text(value).casefold().replace("-", "_")
    return normalized in {
        "json",
        "json_object",
        "application/json",
        "structured_json",
    }


def _strip_json_code_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().casefold() in {"```", "```json"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _as_non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "JsonModePolicy",
    "OpenAICompatibleClient",
    "OpenAICompatibleClientConfig",
]