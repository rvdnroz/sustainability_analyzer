from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .llm_client import (
    BaseLLMClient,
    LLMConfigurationError,
    LLMMessage,
    LLMResponse,
)


class MultimodalContentType(StrEnum):
    """Provider bağımsız multimodal içerik türleri."""

    TEXT = "text"
    IMAGE = "image"


class ImageDetail(StrEnum):
    """OpenAI uyumlu vision API'lerinde kullanılan görüntü ayrıntı düzeyi."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"

    @classmethod
    def normalize(
        cls,
        value: str | ImageDetail | None,
    ) -> ImageDetail:
        if value is None:
            return cls.AUTO

        if isinstance(value, cls):
            return value

        normalized = str(value).strip().casefold()

        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(
                item.value
                for item in cls
            )
            raise ValueError(
                f"Desteklenmeyen image detail değeri: {value!r}. "
                f"Desteklenen değerler: {supported}."
            ) from exc


class MultimodalClientError(RuntimeError):
    """Multimodal istemci katmanının temel hata sınıfı."""


class MultimodalImageError(MultimodalClientError):
    """Görsel dosyası okunamadığında veya doğrulanamadığında oluşur."""


@dataclass(slots=True, frozen=True)
class MultimodalImage:
    """
    Bir multimodal model çağrısına eklenecek yerel görseli temsil eder.

    Görsel dosyası oluşturulurken base64'e çevrilmez. Bu işlem provider
    istemcisi tarafından yalnızca çağrı yapılacağı sırada gerçekleştirilir.
    Böylece aday görseller retrieval aşamasında gereksiz bellek tüketmez.
    """

    path: Path

    visual_id: str | None = None
    page_number: int | None = None

    caption: str | None = None
    nearby_text: str | None = None

    region_type: str | None = None
    topic: str | None = None
    category: str | None = None

    question_ids: tuple[int, ...] = ()

    mime_type: str | None = None
    detail: ImageDetail = ImageDetail.AUTO

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        resolved_path = Path(
            self.path
        ).expanduser().resolve()

        object.__setattr__(
            self,
            "path",
            resolved_path,
        )

        if self.page_number is not None:
            if self.page_number < 1:
                raise ValueError(
                    "page_number pozitif olmalıdır."
                )

        invalid_question_ids = [
            value
            for value in self.question_ids
            if int(value) < 1
        ]

        if invalid_question_ids:
            raise ValueError(
                "question_ids pozitif olmalıdır: "
                f"{invalid_question_ids}"
            )

        object.__setattr__(
            self,
            "question_ids",
            tuple(
                sorted(
                    {
                        int(value)
                        for value in self.question_ids
                    }
                )
            ),
        )

        object.__setattr__(
            self,
            "detail",
            ImageDetail.normalize(
                self.detail
            ),
        )

        resolved_mime = (
            self.mime_type.strip().casefold()
            if self.mime_type
            else self._guess_mime_type(
                resolved_path
            )
        )

        if not resolved_mime.startswith(
            "image/"
        ):
            raise ValueError(
                f"Geçersiz görsel MIME türü: {resolved_mime!r}"
            )

        object.__setattr__(
            self,
            "mime_type",
            resolved_mime,
        )

    @classmethod
    def from_visual_match(
        cls,
        match: Any,
        *,
        detail: str | ImageDetail = ImageDetail.AUTO,
    ) -> MultimodalImage:
        """
        `VisualRetrievalMatch` benzeri bir nesneden görsel girdisi üretir.

        Döngüsel import oluşmaması için belirli bir retriever sınıfına sıkı
        bağımlılık kurulmaz.
        """

        asset = getattr(
            match,
            "asset",
            None,
        )

        if asset is None:
            raise TypeError(
                "match nesnesi asset alanı içermelidir."
            )

        return cls(
            path=Path(asset.image_path),
            visual_id=str(asset.id),
            page_number=asset.page_number,
            caption=asset.caption,
            nearby_text=asset.nearby_text,
            region_type=asset.region_type.value,
            topic=asset.topic,
            category=asset.category,
            question_ids=tuple(
                getattr(
                    match,
                    "question_ids",
                    (),
                )
            ),
            detail=ImageDetail.normalize(
                detail
            ),
            metadata={
                "retrieval_rank": getattr(
                    match,
                    "rank",
                    None,
                ),
                "retrieval_score": getattr(
                    match,
                    "score",
                    None,
                ),
                "matched_terms": list(
                    getattr(
                        match,
                        "matched_terms",
                        (),
                    )
                ),
                **dict(
                    getattr(
                        match,
                        "metadata",
                        {},
                    )
                    or {}
                ),
            },
        )

    @property
    def exists(self) -> bool:
        try:
            return self.path.is_file()
        except OSError:
            return False

    @property
    def size_bytes(self) -> int:
        if not self.exists:
            return 0

        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    @property
    def suffix(self) -> str:
        return self.path.suffix.casefold()

    @property
    def stable_key(self) -> str:
        if self.visual_id:
            return self.visual_id

        return hashlib.sha256(
            str(self.path).encode("utf-8")
        ).hexdigest()

    def validate_file(
        self,
        *,
        maximum_size_bytes: int | None = None,
        allowed_mime_types: Sequence[str] | None = None,
    ) -> None:
        if not self.exists:
            raise MultimodalImageError(
                f"Görsel dosyası bulunamadı: {self.path}"
            )

        size = self.size_bytes

        if size <= 0:
            raise MultimodalImageError(
                f"Görsel dosyası boş: {self.path}"
            )

        if (
            maximum_size_bytes is not None
            and size > maximum_size_bytes
        ):
            raise MultimodalImageError(
                "Görsel dosyası izin verilen boyutu aşıyor: "
                f"{size} > {maximum_size_bytes} bayt, "
                f"path={self.path}"
            )

        if allowed_mime_types is not None:
            normalized_allowed = {
                str(value).strip().casefold()
                for value in allowed_mime_types
            }

            if self.mime_type not in normalized_allowed:
                raise MultimodalImageError(
                    "Görsel MIME türü desteklenmiyor: "
                    f"{self.mime_type}. "
                    f"Desteklenenler: "
                    f"{sorted(normalized_allowed)}"
                )

    def read_bytes(self) -> bytes:
        self.validate_file()

        try:
            return self.path.read_bytes()
        except OSError as exc:
            raise MultimodalImageError(
                f"Görsel dosyası okunamadı: {self.path}"
            ) from exc

    def to_base64(self) -> str:
        return base64.b64encode(
            self.read_bytes()
        ).decode("ascii")

    def to_data_url(self) -> str:
        return (
            f"data:{self.mime_type};base64,"
            f"{self.to_base64()}"
        )

    def describe(self) -> str:
        """
        Prompt içine eklenebilecek kısa, deterministik görsel açıklaması.
        """

        parts: list[str] = []

        if self.visual_id:
            parts.append(
                f"visual_id={self.visual_id}"
            )

        if self.page_number is not None:
            parts.append(
                f"page={self.page_number}"
            )

        if self.region_type:
            parts.append(
                f"type={self.region_type}"
            )

        if self.topic:
            parts.append(
                f"topic={self.topic}"
            )

        if self.question_ids:
            parts.append(
                "question_ids="
                + ",".join(
                    str(value)
                    for value in self.question_ids
                )
            )

        if self.caption:
            parts.append(
                f"caption={self.caption.strip()}"
            )

        return " | ".join(parts)

    @staticmethod
    def _guess_mime_type(
        path: Path,
    ) -> str:
        mime_type, _ = mimetypes.guess_type(
            str(path)
        )

        if (
            mime_type is not None
            and mime_type.startswith("image/")
        ):
            return mime_type.casefold()

        suffix_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }

        fallback = suffix_map.get(
            path.suffix.casefold()
        )

        if fallback is None:
            raise ValueError(
                "Görsel MIME türü dosya uzantısından "
                f"belirlenemedi: {path}"
            )

        return fallback


@dataclass(slots=True, frozen=True)
class MultimodalRequest:
    """Tek bir provider bağımsız multimodal model isteği."""

    prompt: str
    images: tuple[MultimodalImage, ...]

    system_instruction: str | None = None

    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | None = None
    model: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError(
                "MultimodalRequest.prompt boş olamaz."
            )

        if not self.images:
            raise ValueError(
                "MultimodalRequest en az bir görsel içermelidir."
            )

        if (
            self.temperature is not None
            and self.temperature < 0
        ):
            raise ValueError(
                "temperature negatif olamaz."
            )

        if (
            self.max_tokens is not None
            and self.max_tokens < 1
        ):
            raise ValueError(
                "max_tokens pozitif olmalıdır."
            )


@dataclass(slots=True, frozen=True)
class MultimodalResponse:
    """
    Provider bağımsız multimodal model cevabı.

    Metin modeli ile aynı `LLMResponse` yapısını korur; buna ek olarak
    hangi görsellerin çağrıya dâhil edildiğini kaydeder.
    """

    response: LLMResponse
    image_ids: tuple[str, ...]
    image_count: int

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.image_count < 1:
            raise ValueError(
                "image_count pozitif olmalıdır."
            )

        if len(self.image_ids) > self.image_count:
            raise ValueError(
                "image_ids sayısı image_count değerini aşamaz."
            )

    @property
    def content(self) -> str:
        return self.response.content

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def model(self) -> str:
        return self.response.model

    @property
    def provider(self) -> str | None:
        return self.response.provider

    @property
    def usage(self) -> Any:
        return self.response.usage


@runtime_checkable
class MultimodalGenerationClient(Protocol):
    """Görsel ve metin birlikte işleyebilen istemci sözleşmesi."""

    def generate_multimodal(
        self,
        prompt: str,
        images: Sequence[MultimodalImage],
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> MultimodalResponse:
        ...

    def generate_multimodal_json(
        self,
        prompt: str,
        images: Sequence[MultimodalImage],
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> MultimodalResponse:
        ...

    def close(self) -> None:
        ...


class BaseMultimodalClient(BaseLLMClient):
    """
    Provider'a özgü multimodal istemciler için ortak taban sınıfı.

    Alt sınıfın yalnızca `_generate_multimodal_request()` metodunu
    uygulaması yeterlidir.
    """

    provider_name: str = "unknown_multimodal"

    supported_mime_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
    )

    maximum_images_per_request: int = 4
    maximum_image_size_bytes: int = 20 * 1024 * 1024
    maximum_total_image_size_bytes: int = 40 * 1024 * 1024

    def generate_multimodal(
        self,
        prompt: str,
        images: Sequence[MultimodalImage],
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | None = None,
        model: str | None = None,
    ) -> MultimodalResponse:
        request = MultimodalRequest(
            prompt=self._clean_text(prompt),
            images=tuple(images),
            system_instruction=(
                self._clean_text(
                    system_instruction
                )
                if system_instruction
                else None
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
        )

        self._validate_request(
            request
        )

        return self._generate_multimodal_request(
            request
        )

    def generate_multimodal_json(
        self,
        prompt: str,
        images: Sequence[MultimodalImage],
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> MultimodalResponse:
        return self.generate_multimodal(
            prompt,
            images,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
            model=model,
        )

    def _generate_multimodal_request(
        self,
        request: MultimodalRequest,
    ) -> MultimodalResponse:
        raise NotImplementedError

    def _validate_request(
        self,
        request: MultimodalRequest,
    ) -> None:
        if (
            len(request.images)
            > self.maximum_images_per_request
        ):
            raise LLMConfigurationError(
                "Tek multimodal çağrıda izin verilen görsel "
                "sayısı aşıldı: "
                f"{len(request.images)} > "
                f"{self.maximum_images_per_request}.",
                provider=self.provider_name,
                model=request.model,
            )

        total_size = 0

        seen_keys: set[str] = set()

        for image in request.images:
            image.validate_file(
                maximum_size_bytes=(
                    self.maximum_image_size_bytes
                ),
                allowed_mime_types=(
                    self.supported_mime_types
                ),
            )

            total_size += image.size_bytes

            if image.stable_key in seen_keys:
                raise LLMConfigurationError(
                    "Aynı görsel bir multimodal çağrıya birden "
                    f"fazla kez eklendi: {image.path}",
                    provider=self.provider_name,
                    model=request.model,
                )

            seen_keys.add(
                image.stable_key
            )

        if (
            total_size
            > self.maximum_total_image_size_bytes
        ):
            raise LLMConfigurationError(
                "Multimodal çağrıdaki toplam görsel boyutu "
                "izin verilen sınırı aşıyor: "
                f"{total_size} > "
                f"{self.maximum_total_image_size_bytes} bayt.",
                provider=self.provider_name,
                model=request.model,
            )

    @staticmethod
    def build_image_context(
        images: Sequence[MultimodalImage],
    ) -> str:
        """
        Görsel metadata'sını prompt'a eklemek için güvenli metin oluşturur.

        Provider mesajındaki görsel sırası ile bu listedeki sıra aynıdır.
        """

        lines: list[str] = []

        for index, image in enumerate(
            images,
            start=1,
        ):
            description = image.describe()

            if not description:
                description = (
                    f"path={image.path.name}"
                )

            lines.append(
                f"[Görsel {index}] {description}"
            )

            if image.nearby_text:
                nearby = " ".join(
                    image.nearby_text.split()
                )[:500]

                lines.append(
                    f"[Görsel {index} yakın metin] "
                    f"{nearby}"
                )

        return "\n".join(lines)

    @staticmethod
    def image_ids(
        images: Sequence[MultimodalImage],
    ) -> tuple[str, ...]:
        return tuple(
            image.visual_id
            or image.stable_key
            for image in images
        )

    @staticmethod
    def text_messages(
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> tuple[LLMMessage, ...]:
        """
        Multimodal provider'ın düz metin fallback'i için mesaj üretir.
        """

        messages: list[LLMMessage] = []

        if system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        system_instruction
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return tuple(messages)


__all__ = [
    "BaseMultimodalClient",
    "ImageDetail",
    "MultimodalClientError",
    "MultimodalContentType",
    "MultimodalGenerationClient",
    "MultimodalImage",
    "MultimodalImageError",
    "MultimodalRequest",
    "MultimodalResponse",
]
