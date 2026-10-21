from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .llm_client import (
    LLMProvider,
    LLMResponse,
)
from .multimodal_client import (
    BaseMultimodalClient,
    ImageDetail,
    MultimodalImage,
    MultimodalRequest,
    MultimodalResponse,
)
from .openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleClientConfig,
)


@dataclass(slots=True, frozen=True)
class OpenAICompatibleMultimodalConfig:
    """
    OpenAI Chat Completions uyumlu vision modellerinin ek ayarları.

    Metin istemcisinin API URL, retry, authentication ve JSON mode ayarları
    `client_config` üzerinden korunur.
    """

    client_config: OpenAICompatibleClientConfig

    maximum_images_per_request: int = 4
    maximum_image_size_bytes: int = 20 * 1024 * 1024
    maximum_total_image_size_bytes: int = 40 * 1024 * 1024

    supported_mime_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
    )

    prepend_image_context: bool = True
    default_image_detail: ImageDetail = ImageDetail.HIGH

    def __post_init__(self) -> None:
        if self.maximum_images_per_request < 1:
            raise ValueError(
                "maximum_images_per_request en az 1 olmalıdır."
            )

        if self.maximum_image_size_bytes < 1:
            raise ValueError(
                "maximum_image_size_bytes pozitif olmalıdır."
            )

        if (
            self.maximum_total_image_size_bytes
            < self.maximum_image_size_bytes
        ):
            raise ValueError(
                "maximum_total_image_size_bytes tek görsel sınırından "
                "küçük olamaz."
            )

        if not self.supported_mime_types:
            raise ValueError(
                "En az bir supported_mime_types değeri gereklidir."
            )

        normalized_mime_types = tuple(
            dict.fromkeys(
                str(value).strip().casefold()
                for value in self.supported_mime_types
                if str(value).strip()
            )
        )

        invalid = [
            value
            for value in normalized_mime_types
            if not value.startswith("image/")
        ]

        if invalid:
            raise ValueError(
                f"Geçersiz MIME türleri: {invalid}"
            )

        object.__setattr__(
            self,
            "supported_mime_types",
            normalized_mime_types,
        )
        object.__setattr__(
            self,
            "default_image_detail",
            ImageDetail.normalize(
                self.default_image_detail
            ),
        )


class OpenAICompatibleMultimodalClient(
    OpenAICompatibleClient,
    BaseMultimodalClient,
):
    """
    Groq, OpenRouter ve diğer OpenAI Chat Completions uyumlu vision
    sağlayıcıları için multimodal istemci.

    Görseller OpenAI uyumlu message content dizisine çevrilir:

        {
            "role": "user",
            "content": [
                {"type": "text", "text": "..."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,...",
                        "detail": "high"
                    }
                }
            ]
        }

    HTTP, retry, rate-limit ve hata eşleme davranışı mevcut
    `OpenAICompatibleClient` sınıfından yeniden kullanılır.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: OpenAICompatibleMultimodalConfig,
        http_client: Any | None = None,
    ) -> None:
        self.multimodal_config = config

        super().__init__(
            api_key,
            config=config.client_config,
            http_client=http_client,
        )

        self.maximum_images_per_request = (
            config.maximum_images_per_request
        )
        self.maximum_image_size_bytes = (
            config.maximum_image_size_bytes
        )
        self.maximum_total_image_size_bytes = (
            config.maximum_total_image_size_bytes
        )
        self.supported_mime_types = (
            config.supported_mime_types
        )

    @classmethod
    def for_groq(
        cls,
        model: str,
        *,
        api_key: str | None = None,
        maximum_images_per_request: int = 4,
        maximum_image_size_bytes: int = 20 * 1024 * 1024,
        maximum_total_image_size_bytes: int = 40 * 1024 * 1024,
        supported_mime_types: Sequence[str] = (
            "image/png",
            "image/jpeg",
            "image/webp",
        ),
        prepend_image_context: bool = True,
        default_image_detail: str | ImageDetail = ImageDetail.HIGH,
        **client_config_overrides: Any,
    ) -> OpenAICompatibleMultimodalClient:
        client_config = OpenAICompatibleClientConfig(
            provider=LLMProvider.GROQ,
            model=model,
            **client_config_overrides,
        )

        return cls(
            api_key,
            config=OpenAICompatibleMultimodalConfig(
                client_config=client_config,
                maximum_images_per_request=(
                    maximum_images_per_request
                ),
                maximum_image_size_bytes=(
                    maximum_image_size_bytes
                ),
                maximum_total_image_size_bytes=(
                    maximum_total_image_size_bytes
                ),
                supported_mime_types=tuple(
                    supported_mime_types
                ),
                prepend_image_context=(
                    prepend_image_context
                ),
                default_image_detail=(
                    ImageDetail.normalize(
                        default_image_detail
                    )
                ),
            ),
        )

    @classmethod
    def for_openrouter(
        cls,
        model: str,
        *,
        api_key: str | None = None,
        maximum_images_per_request: int = 4,
        maximum_image_size_bytes: int = 20 * 1024 * 1024,
        maximum_total_image_size_bytes: int = 40 * 1024 * 1024,
        supported_mime_types: Sequence[str] = (
            "image/png",
            "image/jpeg",
            "image/webp",
        ),
        prepend_image_context: bool = True,
        default_image_detail: str | ImageDetail = ImageDetail.HIGH,
        **client_config_overrides: Any,
    ) -> OpenAICompatibleMultimodalClient:
        client_config = OpenAICompatibleClientConfig(
            provider=LLMProvider.OPENROUTER,
            model=model,
            **client_config_overrides,
        )

        return cls(
            api_key,
            config=OpenAICompatibleMultimodalConfig(
                client_config=client_config,
                maximum_images_per_request=(
                    maximum_images_per_request
                ),
                maximum_image_size_bytes=(
                    maximum_image_size_bytes
                ),
                maximum_total_image_size_bytes=(
                    maximum_total_image_size_bytes
                ),
                supported_mime_types=tuple(
                    supported_mime_types
                ),
                prepend_image_context=(
                    prepend_image_context
                ),
                default_image_detail=(
                    ImageDetail.normalize(
                        default_image_detail
                    )
                ),
            ),
        )

    @classmethod
    def for_openai_compatible(
        cls,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        maximum_images_per_request: int = 4,
        maximum_image_size_bytes: int = 20 * 1024 * 1024,
        maximum_total_image_size_bytes: int = 40 * 1024 * 1024,
        supported_mime_types: Sequence[str] = (
            "image/png",
            "image/jpeg",
            "image/webp",
        ),
        prepend_image_context: bool = True,
        default_image_detail: str | ImageDetail = ImageDetail.HIGH,
        **client_config_overrides: Any,
    ) -> OpenAICompatibleMultimodalClient:
        client_config = OpenAICompatibleClientConfig(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            model=model,
            base_url=base_url,
            **client_config_overrides,
        )

        return cls(
            api_key,
            config=OpenAICompatibleMultimodalConfig(
                client_config=client_config,
                maximum_images_per_request=(
                    maximum_images_per_request
                ),
                maximum_image_size_bytes=(
                    maximum_image_size_bytes
                ),
                maximum_total_image_size_bytes=(
                    maximum_total_image_size_bytes
                ),
                supported_mime_types=tuple(
                    supported_mime_types
                ),
                prepend_image_context=(
                    prepend_image_context
                ),
                default_image_detail=(
                    ImageDetail.normalize(
                        default_image_detail
                    )
                ),
            ),
        )

    def _generate_multimodal_request(
        self,
        request: MultimodalRequest,
    ) -> MultimodalResponse:
        resolved_model = (
            str(request.model).strip()
            if request.model
            else self.config.model
        )

        resolved_temperature = (
            self.config.temperature
            if request.temperature is None
            else request.temperature
        )

        resolved_max_tokens = (
            self.config.maximum_output_tokens
            if request.max_tokens is None
            else request.max_tokens
        )

        self.validate_generation_parameters(
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        )

        images = tuple(
            self._apply_default_detail(
                image
            )
            for image in request.images
        )

        prompt = request.prompt

        if (
            self.multimodal_config
            .prepend_image_context
        ):
            image_context = self.build_image_context(
                images
            )

            if image_context:
                prompt = (
                    f"{prompt}\n\n"
                    "GÖRSEL KAYNAK EŞLEMESİ:\n"
                    f"{image_context}"
                )

        messages = self._build_multimodal_messages(
            prompt=prompt,
            images=images,
            system_instruction=(
                request.system_instruction
            ),
        )

        body: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
            "temperature": resolved_temperature,
        }

        if self.provider_name in {
            LLMProvider.GROQ.value,
            LLMProvider.OPENROUTER.value,
        }:
            body["max_completion_tokens"] = (
                resolved_max_tokens
            )
        else:
            body["max_tokens"] = (
                resolved_max_tokens
            )

        json_mode_requested = self._is_json_request(
            request.response_format
        )
        json_mode_applied = (
            json_mode_requested
            and self.config.json_mode_policy
            != "disabled"
        )

        if json_mode_applied:
            body["response_format"] = {
                "type": "json_object"
            }

        body.update(
            dict(self.config.extra_body)
        )

        headers = self._build_headers()
        started_at = time.perf_counter()

        payload, attempts, fallback_used = (
            self._post_with_retries(
                body=body,
                headers=headers,
                resolved_model=resolved_model,
                json_mode_applied=(
                    json_mode_applied
                ),
            )
        )

        latency_seconds = (
            time.perf_counter()
            - started_at
        )

        response = self._parse_response(
            payload,
            requested_model=resolved_model,
            attempts=attempts,
            latency_seconds=latency_seconds,
            json_mode_requested=(
                json_mode_requested
            ),
            json_mode_applied=(
                json_mode_applied
                and not fallback_used
            ),
            json_mode_fallback_used=(
                fallback_used
            ),
            ignored_parameters={},
        )

        if json_mode_requested:
            self._validate_json_content(
                response
            )

        multimodal_metadata = {
            "image_count": len(images),
            "image_ids": list(
                self.image_ids(images)
            ),
            "image_sizes_bytes": [
                image.size_bytes
                for image in images
            ],
            "image_mime_types": [
                image.mime_type
                for image in images
            ],
            "image_details": [
                image.detail.value
                for image in images
            ],
            "json_mode_requested": (
                json_mode_requested
            ),
            "json_mode_fallback_used": (
                fallback_used
            ),
        }

        return MultimodalResponse(
            response=self._merge_response_metadata(
                response,
                multimodal_metadata,
            ),
            image_ids=self.image_ids(
                images
            ),
            image_count=len(images),
            metadata=multimodal_metadata,
        )

    def _build_multimodal_messages(
        self,
        *,
        prompt: str,
        images: Sequence[MultimodalImage],
        system_instruction: str | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        system_instruction
                    ),
                }
            )

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image.to_data_url(),
                        "detail": (
                            image.detail.value
                        ),
                    },
                }
            )

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        return messages

    def _apply_default_detail(
        self,
        image: MultimodalImage,
    ) -> MultimodalImage:
        if image.detail != ImageDetail.AUTO:
            return image

        return MultimodalImage(
            path=image.path,
            visual_id=image.visual_id,
            page_number=image.page_number,
            caption=image.caption,
            nearby_text=image.nearby_text,
            region_type=image.region_type,
            topic=image.topic,
            category=image.category,
            question_ids=image.question_ids,
            mime_type=image.mime_type,
            detail=(
                self.multimodal_config
                .default_image_detail
            ),
            metadata=image.metadata,
        )

    @staticmethod
    def _is_json_request(
        response_format: str | None,
    ) -> bool:
        if response_format is None:
            return False

        normalized = (
            str(response_format)
            .strip()
            .casefold()
            .replace("-", "_")
        )

        return normalized in {
            "json",
            "json_object",
            "application/json",
        }

    @staticmethod
    def _merge_response_metadata(
        response: LLMResponse,
        extra_metadata: Mapping[str, Any],
    ) -> LLMResponse:
        return LLMResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            finish_reason=response.finish_reason,
            usage=response.usage,
            response_id=response.response_id,
            raw_response=response.raw_response,
            metadata={
                **dict(response.metadata),
                **dict(extra_metadata),
            },
        )


__all__ = [
    "OpenAICompatibleMultimodalClient",
    "OpenAICompatibleMultimodalConfig",
]
