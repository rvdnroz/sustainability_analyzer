from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .llm_client import (
    BaseLLMClient,
    LLMAuthenticationError,
    LLMClientError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMDependencyError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUsage,
)


@dataclass(slots=True, frozen=True)
class GeminiClientConfig:
    """Google Gemini istemcisinin çalışma ayarları."""

    model: str = "gemini-3.5-flash"
    api_key_environment_variable: str = "GEMINI_API_KEY"

    temperature: float = 0.0
    maximum_output_tokens: int = 700
    top_p: float | None = None
    top_k: int | None = None

    thinking_level: str | None = None

    timeout_seconds: float = 120.0
    maximum_retries: int = 3
    initial_retry_delay_seconds: float = 1.0
    maximum_retry_delay_seconds: float = 20.0

    response_mime_type: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Gemini model adı boş olamaz.")
        if not self.api_key_environment_variable.strip():
            raise ValueError("API anahtarı ortam değişkeni adı boş olamaz.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature 0 ile 2 arasında olmalıdır.")
        if self.maximum_output_tokens <= 0:
            raise ValueError("maximum_output_tokens pozitif olmalıdır.")
        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p 0 ile 1 arasında olmalıdır.")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds pozitif olmalıdır.")
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


# Geriye dönük uyumluluk: Eski modüller bu adları import etmeye devam edebilir.
GeminiUsage = LLMUsage
GeminiResponse = LLMResponse
GeminiClientError = LLMClientError
GeminiDependencyError = LLMDependencyError
GeminiAuthenticationError = LLMAuthenticationError
GeminiRateLimitError = LLMRateLimitError
GeminiTimeoutError = LLMTimeoutError
GeminiResponseError = LLMResponseError
GeminiConnectionError = LLMConnectionError
GeminiConfigurationError = LLMConfigurationError


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API için provider bağımsız metin üretim istemcisi.

    Dışarıya yalnızca ``LLMResponse`` ve ortak ``LLMClientError`` alt
    sınıflarını verir. Böylece AnswerGenerator ve CategoryAnswerGenerator,
    Gemini'ye özel sınıflara bağlanmadan çalışabilir.
    """

    provider_name = "gemini"

    _RETRYABLE_ERROR_TERMS = (
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "too many requests",
        "500",
        "502",
        "503",
        "504",
        "internal",
        "unavailable",
        "deadline exceeded",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
    )

    _AUTHENTICATION_ERROR_TERMS = (
        "401",
        "403",
        "unauthenticated",
        "permission denied",
        "api key not valid",
        "invalid api key",
        "authentication",
    )

    _CONNECTION_ERROR_TERMS = (
        "connection error",
        "connection refused",
        "connection reset",
        "name resolution",
        "dns",
        "network is unreachable",
        "failed to establish a new connection",
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: GeminiClientConfig | None = None,
    ) -> None:
        self.config = config or GeminiClientConfig()

        self._api_key = (
            self._clean_text(api_key)
            or self._clean_text(
                os.getenv(self.config.api_key_environment_variable)
            )
        )

        if not self._api_key:
            raise LLMAuthenticationError(
                "Gemini API anahtarı bulunamadı. "
                f"'{self.config.api_key_environment_variable}' "
                "ortam değişkenini tanımlayın.",
                provider=self.provider_name,
                model=self.config.model,
                retryable=False,
            )

        self._client: Any | None = None
        self._types: Any | None = None

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
        self._ensure_client()

        resolved_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        resolved_max_tokens = (
            self.config.maximum_output_tokens
            if max_tokens is None
            else max_tokens
        )
        resolved_top_p = self.config.top_p if top_p is None else top_p
        resolved_top_k = self.config.top_k if top_k is None else top_k
        resolved_thinking_level = (
            self.config.thinking_level
            if thinking_level is None
            else thinking_level
        )
        resolved_model = self._clean_text(model) or self.config.model

        self.validate_generation_parameters(
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            top_p=resolved_top_p,
            top_k=resolved_top_k,
        )

        system_instruction, contents = self._convert_messages(
            normalized_messages
        )
        if not contents:
            raise LLMConfigurationError(
                "Gemini isteğinde kullanıcı veya model mesajı bulunamadı.",
                provider=self.provider_name,
                model=resolved_model,
            )

        generation_config = self._build_generation_config(
            system_instruction=system_instruction,
            temperature=resolved_temperature,
            maximum_output_tokens=resolved_max_tokens,
            response_format=response_format,
            top_p=resolved_top_p,
            top_k=resolved_top_k,
            thinking_level=resolved_thinking_level,
        )

        raw_response = self._generate_with_retry(
            model=resolved_model,
            contents=contents,
            generation_config=generation_config,
        )

        return self._normalize_response(raw_response, model=resolved_model)

    def close(self) -> None:
        if self._client is None:
            return

        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass

        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMDependencyError(
                "Google GenAI SDK bulunamadı. "
                "'python -m pip install google-genai' komutuyla kurun.",
                provider=self.provider_name,
                model=self.config.model,
                retryable=False,
            ) from exc

        try:
            http_options = types.HttpOptions(
                timeout=int(self.config.timeout_seconds * 1000)
            )
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=http_options,
            )
        except TypeError:
            # Bazı SDK sürümleri HttpOptions parametresini kabul etmeyebilir.
            try:
                self._client = genai.Client(api_key=self._api_key)
            except Exception as exc:
                self._raise_client_error(exc, model=self.config.model)
        except Exception as exc:
            self._raise_client_error(exc, model=self.config.model)

        self._types = types

    def _convert_messages(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[str | None, list[Any]]:
        system_parts: list[str] = []
        contents: list[Any] = []

        for message in messages:
            role = self._clean_text(message.get("role")).casefold()
            content = self._clean_text(message.get("content"))

            if role == "system":
                system_parts.append(content)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                self._types.Content(
                    role=gemini_role,
                    parts=[self._types.Part(text=content)],
                )
            )

        system_instruction = (
            "\n\n".join(system_parts) if system_parts else None
        )
        return system_instruction, contents

    def _build_generation_config(
        self,
        *,
        system_instruction: str | None,
        temperature: float,
        maximum_output_tokens: int,
        response_format: str | None,
        top_p: float | None,
        top_k: int | None,
        thinking_level: str | None,
    ) -> Any:
        config_values: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": maximum_output_tokens,
        }

        if system_instruction:
            config_values["system_instruction"] = system_instruction
        if top_p is not None:
            config_values["top_p"] = top_p
        if top_k is not None:
            config_values["top_k"] = top_k

        response_mime_type = self._resolve_response_mime_type(response_format)
        if response_mime_type:
            config_values["response_mime_type"] = response_mime_type

        if thinking_level:
            thinking_config = self._build_thinking_config(thinking_level)
            if thinking_config is not None:
                config_values["thinking_config"] = thinking_config

        try:
            return self._types.GenerateContentConfig(**config_values)
        except TypeError:
            # Eski SDK sürümlerinde thinking_config bulunmayabilir.
            config_values.pop("thinking_config", None)
            return self._types.GenerateContentConfig(**config_values)

    def _build_thinking_config(self, thinking_level: str) -> Any | None:
        normalized_level = self._clean_text(thinking_level).upper()
        if not normalized_level:
            return None

        try:
            return self._types.ThinkingConfig(
                thinking_level=normalized_level
            )
        except (AttributeError, TypeError):
            return None

    def _generate_with_retry(
        self,
        *,
        model: str,
        contents: Sequence[Any],
        generation_config: Any,
    ) -> Any:
        last_error: Exception | None = None

        for attempt in range(self.config.maximum_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generation_config,
                )
            except Exception as exc:
                last_error = exc

                if self._is_authentication_error(exc):
                    raise LLMAuthenticationError(
                        "Gemini API kimlik doğrulaması başarısız oldu.",
                        provider=self.provider_name,
                        model=model,
                        status_code=self._extract_status_code(exc),
                        retryable=False,
                    ) from exc

                if not self._is_retryable_error(exc):
                    self._raise_client_error(exc, model=model)

                if attempt >= self.config.maximum_retries:
                    break

                time.sleep(self._calculate_retry_delay(attempt))

        if last_error is None:
            raise LLMClientError(
                "Gemini isteği bilinmeyen nedenle başarısız oldu.",
                provider=self.provider_name,
                model=model,
            )

        self._raise_client_error(
            last_error,
            model=model,
            exhausted_retries=True,
        )
        raise AssertionError("Ulaşılamaz kod")

    def _normalize_response(self, response: Any, *, model: str) -> LLMResponse:
        content = self._extract_response_text(response)

        if not content:
            block_reason = self._extract_block_reason(response)
            if block_reason:
                raise LLMResponseError(
                    "Gemini cevabı güvenlik filtresi tarafından "
                    f"engellendi: {block_reason}",
                    provider=self.provider_name,
                    model=model,
                    retryable=False,
                    metadata={"block_reason": block_reason},
                )

            raise LLMResponseError(
                "Gemini geçerli bir metin cevabı üretmedi.",
                provider=self.provider_name,
                model=model,
                retryable=False,
            )

        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        response_id = self._optional_string(
            getattr(response, "response_id", None)
        )

        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
            raw_response=response,
            metadata={
                "candidate_count": self._candidate_count(response),
            },
        )

    @classmethod
    def _extract_response_text(cls, response: Any) -> str:
        direct_text = getattr(response, "text", None)
        cleaned_direct_text = cls._clean_text(direct_text)
        if cleaned_direct_text:
            return cleaned_direct_text

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        text_parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue

            for part in parts:
                cleaned_text = cls._clean_text(getattr(part, "text", None))
                if cleaned_text:
                    text_parts.append(cleaned_text)

        return "\n".join(text_parts)

    @classmethod
    def _extract_usage(cls, response: Any) -> LLMUsage | None:
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is None:
            return None

        prompt_tokens = cls._optional_int(
            getattr(usage_metadata, "prompt_token_count", None)
        )
        output_tokens = cls._optional_int(
            getattr(usage_metadata, "candidates_token_count", None)
        )
        total_tokens = cls._optional_int(
            getattr(usage_metadata, "total_token_count", None)
        )
        cached_tokens = cls._optional_int(
            getattr(usage_metadata, "cached_content_token_count", None)
        )
        reasoning_tokens = cls._optional_int(
            getattr(usage_metadata, "thoughts_token_count", None)
        )

        if total_tokens is None and (
            prompt_tokens is not None or output_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (output_tokens or 0)

        return LLMUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            metadata={"source": "gemini_usage_metadata"},
        )

    @classmethod
    def _extract_finish_reason(cls, response: Any) -> str | None:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None

        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is None:
            return None

        return cls._clean_text(
            getattr(finish_reason, "value", finish_reason)
        ) or None

    @classmethod
    def _extract_block_reason(cls, response: Any) -> str | None:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is None:
            return None

        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason is None:
            return None

        return cls._clean_text(
            getattr(block_reason, "value", block_reason)
        ) or None

    @staticmethod
    def _candidate_count(response: Any) -> int:
        candidates = getattr(response, "candidates", None)
        if candidates is None:
            return 0
        try:
            return len(candidates)
        except TypeError:
            return 0

    def _resolve_response_mime_type(
        self,
        response_format: str | None,
    ) -> str | None:
        normalized_format = self._clean_text(response_format).casefold()
        if normalized_format in {
            "json",
            "application/json",
            "json_object",
        }:
            return "application/json"
        if normalized_format in {"text", "plain_text", "text/plain"}:
            return "text/plain"
        return self.config.response_mime_type

    def _calculate_retry_delay(self, attempt: int) -> float:
        exponential_delay = self.config.initial_retry_delay_seconds * (
            2**attempt
        )
        jitter = random.uniform(
            0.0,
            self.config.initial_retry_delay_seconds,
        )
        return min(
            exponential_delay + jitter,
            self.config.maximum_retry_delay_seconds,
        )

    @classmethod
    def _is_retryable_error(cls, error: Exception) -> bool:
        status_code = cls._extract_status_code(error)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True

        normalized_error = str(error).casefold()
        return any(
            term in normalized_error for term in cls._RETRYABLE_ERROR_TERMS
        )

    @classmethod
    def _is_authentication_error(cls, error: Exception) -> bool:
        status_code = cls._extract_status_code(error)
        if status_code in {401, 403}:
            return True

        normalized_error = str(error).casefold()
        return any(
            term in normalized_error
            for term in cls._AUTHENTICATION_ERROR_TERMS
        )

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        normalized_error = str(error).casefold()
        return any(
            term in normalized_error
            for term in ("timeout", "timed out", "deadline exceeded")
        )

    @classmethod
    def _is_connection_error(cls, error: Exception) -> bool:
        normalized_error = str(error).casefold()
        return any(
            term in normalized_error for term in cls._CONNECTION_ERROR_TERMS
        )

    @classmethod
    def _is_rate_limit_error(cls, error: Exception) -> bool:
        if cls._extract_status_code(error) == 429:
            return True

        normalized_error = str(error).casefold()
        return any(
            term in normalized_error
            for term in (
                "429",
                "resource_exhausted",
                "quota",
                "rate limit",
                "too many requests",
            )
        )

    @classmethod
    def _raise_client_error(
        cls,
        error: Exception,
        *,
        model: str,
        exhausted_retries: bool = False,
    ) -> None:
        status_code = cls._extract_status_code(error)
        shared = {
            "provider": "gemini",
            "model": model,
            "status_code": status_code,
        }

        if cls._is_authentication_error(error):
            raise LLMAuthenticationError(
                "Gemini API kimlik doğrulaması başarısız oldu.",
                retryable=False,
                **shared,
            ) from error

        if cls._is_rate_limit_error(error):
            message = "Gemini API kota veya hız sınırı aşıldı."
            if exhausted_retries:
                message += " Yeniden deneme sınırına ulaşıldı."
            raise LLMRateLimitError(
                message,
                retryable=True,
                **shared,
            ) from error

        if cls._is_timeout_error(error):
            message = "Gemini API isteği zaman aşımına uğradı."
            if exhausted_retries:
                message += " Yeniden deneme sınırına ulaşıldı."
            raise LLMTimeoutError(
                message,
                retryable=True,
                **shared,
            ) from error

        if cls._is_connection_error(error):
            raise LLMConnectionError(
                f"Gemini API bağlantısı başarısız oldu: {error}",
                retryable=True,
                **shared,
            ) from error

        retryable = cls._is_retryable_error(error)
        message = f"Gemini API isteği başarısız oldu: {error}"
        if exhausted_retries:
            message = (
                "Gemini isteği yeniden denemelere rağmen başarısız oldu: "
                f"{error}"
            )

        raise LLMClientError(
            message,
            retryable=retryable,
            **shared,
        ) from error

    @classmethod
    def _extract_status_code(cls, error: Exception) -> int | None:
        candidates = [
            getattr(error, "status_code", None),
            getattr(error, "code", None),
        ]

        response = getattr(error, "response", None)
        if response is not None:
            candidates.append(getattr(response, "status_code", None))

        for value in candidates:
            if value is None:
                continue
            raw_value = getattr(value, "value", value)
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                text = str(raw_value)
                for code in (401, 403, 408, 409, 425, 429, 500, 502, 503, 504):
                    if str(code) in text:
                        return code

        normalized_error = str(error)
        for code in (401, 403, 408, 409, 425, 429, 500, 502, 503, 504):
            if str(code) in normalized_error:
                return code
        return None

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "GeminiAuthenticationError",
    "GeminiClient",
    "GeminiClientConfig",
    "GeminiClientError",
    "GeminiConfigurationError",
    "GeminiConnectionError",
    "GeminiDependencyError",
    "GeminiRateLimitError",
    "GeminiResponse",
    "GeminiResponseError",
    "GeminiTimeoutError",
    "GeminiUsage",
]