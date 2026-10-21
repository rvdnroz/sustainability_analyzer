from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


LLMMessage = Mapping[str, str]


class LLMProvider(StrEnum):
    """Uygulamanın yerleşik olarak tanıdığı metin modeli sağlayıcıları."""

    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"

    @classmethod
    def normalize(cls, value: str | LLMProvider) -> LLMProvider:
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().casefold().replace("-", "_")
        aliases = {
            "google": cls.GEMINI,
            "google_gemini": cls.GEMINI,
            "open_router": cls.OPENROUTER,
            "local": cls.OLLAMA,
            "openai": cls.OPENAI_COMPATIBLE,
            "openai_compatible_api": cls.OPENAI_COMPATIBLE,
        }

        if normalized in aliases:
            return aliases[normalized]

        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Desteklenmeyen LLM sağlayıcısı: {value!r}. "
                f"Desteklenen değerler: {supported}."
            ) from exc


@dataclass(slots=True, frozen=True)
class LLMUsage:
    """Bir model çağrısına ait provider bağımsız token kullanımı."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "prompt_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} negatif olamaz.")

    @property
    def input_tokens(self) -> int | None:
        return self.prompt_tokens

    @property
    def completion_tokens(self) -> int | None:
        return self.output_tokens

    @classmethod
    def empty(cls) -> LLMUsage:
        return cls()


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Bütün metin modeli istemcilerinin döndürdüğü ortak cevap modeli."""

    content: str
    model: str
    provider: str | None = None

    finish_reason: str | None = None
    usage: LLMUsage | None = None

    response_id: str | None = None
    raw_response: Any = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("LLMResponse.content str olmalıdır.")
        if not self.model.strip():
            raise ValueError("LLMResponse.model boş olamaz.")

    @property
    def text(self) -> str:
        return self.content

    @property
    def input_tokens(self) -> int | None:
        return self.usage.prompt_tokens if self.usage is not None else None

    @property
    def output_tokens(self) -> int | None:
        return self.usage.output_tokens if self.usage is not None else None

    @property
    def total_tokens(self) -> int | None:
        return self.usage.total_tokens if self.usage is not None else None


class LLMClientError(RuntimeError):
    """Bütün model istemcisi hatalarının temel sınıfı."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable
        self.metadata = dict(metadata or {})


class LLMDependencyError(LLMClientError):
    """İstemcinin ihtiyaç duyduğu Python paketi kurulu olmadığında oluşur."""


class LLMConfigurationError(LLMClientError):
    """Provider, URL veya model yapılandırması geçersiz olduğunda oluşur."""


class LLMAuthenticationError(LLMClientError):
    """API anahtarı eksik veya geçersiz olduğunda oluşur."""


class LLMRateLimitError(LLMClientError):
    """Kota ya da hız sınırı aşıldığında oluşur."""


class LLMTimeoutError(LLMClientError):
    """Model isteği zaman aşımına uğradığında oluşur."""


class LLMConnectionError(LLMClientError):
    """Provider sunucusuna bağlantı kurulamadığında oluşur."""


class LLMResponseError(LLMClientError):
    """Provider geçerli veya kullanılabilir bir cevap döndürmediğinde oluşur."""


@runtime_checkable
class JsonGenerationClient(Protocol):
    """JSON çıktısı üretebilen istemciler için en küçük ortak sözleşme."""

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        ...


@runtime_checkable
class TextGenerationClient(JsonGenerationClient, Protocol):
    """Cevap üretim katmanında kullanılacak tam metin istemcisi sözleşmesi."""

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
        ...

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        ...

    def close(self) -> None:
        ...


class BaseLLMClient:
    """
    İstemciler için küçük bir yardımcı taban sınıfı.

    Provider uygulamaları yalnızca `generate()` metodunu yazabilir. Düz metin ve
    JSON yardımcı metotları burada ortak biçimde sağlanır. Sınıf bilinçli olarak
    ağ veya SDK davranışı içermez.
    """

    provider_name: str = "unknown"

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
        raise NotImplementedError

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        cleaned_prompt = self._clean_text(prompt)
        if not cleaned_prompt:
            raise LLMConfigurationError(
                "Model prompt metni boş olamaz.",
                provider=self.provider_name,
                model=model,
            )

        messages: list[dict[str, str]] = []
        cleaned_system_instruction = self._clean_text(system_instruction)
        if cleaned_system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": cleaned_system_instruction,
                }
            )

        messages.append({"role": "user", "content": cleaned_prompt})

        return self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        return self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
            model=model,
        )

    def close(self) -> None:
        """SDK veya HTTP istemcisi kullanmayan alt sınıflar için no-op."""

    def __enter__(self) -> BaseLLMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    @staticmethod
    def validate_messages(
        messages: Sequence[LLMMessage],
    ) -> tuple[dict[str, str], ...]:
        if not messages:
            raise LLMConfigurationError(
                "Model isteği en az bir mesaj içermelidir."
            )

        normalized: list[dict[str, str]] = []
        valid_roles = {"system", "user", "assistant", "model"}

        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise LLMConfigurationError(
                    f"Mesaj {index} bir mapping olmalıdır."
                )

            role = BaseLLMClient._clean_text(message.get("role")).casefold()
            content = BaseLLMClient._clean_text(message.get("content"))

            if not role:
                raise LLMConfigurationError(
                    f"Mesaj {index} için role alanı boş olamaz."
                )
            if role not in valid_roles:
                raise LLMConfigurationError(
                    f"Mesaj {index} için desteklenmeyen rol: {role!r}."
                )
            if not content:
                raise LLMConfigurationError(
                    f"Mesaj {index} için content alanı boş olamaz."
                )

            if role == "model":
                role = "assistant"

            normalized.append({"role": role, "content": content})

        return tuple(normalized)

    @staticmethod
    def validate_generation_parameters(
        *,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature 0 ile 2 arasında olmalıdır.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens pozitif olmalıdır.")
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p 0 ile 1 arasında olmalıdır.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


__all__ = [
    "BaseLLMClient",
    "JsonGenerationClient",
    "LLMAuthenticationError",
    "LLMClientError",
    "LLMConfigurationError",
    "LLMConnectionError",
    "LLMDependencyError",
    "LLMMessage",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUsage",
    "TextGenerationClient",
]