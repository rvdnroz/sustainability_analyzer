from __future__ import annotations

from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .page import BoundingBox
from .table import ScalarValue


class VisualRegionType(str, Enum):
    """
    PDF sayfasından çıkarılan görsel bölgenin içerik türünü belirtir.

    Bu sınıflandırma, bölgenin hangi çıkarım yoluna yönlendirileceğini
    belirlemek için kullanılır.
    """

    TABLE = "table"
    CHART = "chart"
    METRIC_CARD = "metric_card"
    INFOGRAPHIC = "infographic"
    FIGURE = "figure"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"


class VisualProcessingStatus(str, Enum):
    """
    Görsel varlığın işleme yaşam döngüsündeki mevcut durumunu belirtir.
    """

    DETECTED = "detected"
    IMAGE_EXTRACTED = "image_extracted"
    OCR_COMPLETED = "ocr_completed"
    OCR_PARTIAL = "ocr_partial"
    VISION_REQUIRED = "vision_required"
    VISION_COMPLETED = "vision_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VisualExtractionMethod(str, Enum):
    """
    Görsel içerikten bilginin hangi yöntemle çıkarıldığını belirtir.
    """

    NATIVE = "native"
    CELL_OCR = "cell_ocr"
    WHOLE_IMAGE_OCR = "whole_image_ocr"
    VISION_TABLE = "vision_table"
    VISION_CHART = "vision_chart"
    VISION_INFOGRAPHIC = "vision_infographic"
    RULE_BASED = "rule_based"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class OCRWord(BaseModel):
    """
    Bütün-görüntü OCR işlemi sonucunda bulunan tek bir sözcüğü temsil eder.

    Sözcük koordinatlarının korunması, metnin daha sonra satır ve sütunlara
    kümelenebilmesini sağlar.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    text: str = Field(
        min_length=1,
    )

    bounding_box: BoundingBox

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    line_index: int | None = Field(
        default=None,
        ge=0,
    )

    block_index: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("text")
    @classmethod
    def normalize_text(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(value.strip().split())

        if not normalized:
            raise ValueError(
                "OCR sözcüğü boş olamaz."
            )

        return normalized


class OCRLine(BaseModel):
    """
    OCR sözcüklerinin koordinat bilgileriyle oluşturulmuş tek bir metin satırı.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    line_index: int = Field(
        ge=0,
    )

    text: str = Field(
        min_length=1,
    )

    bounding_box: BoundingBox

    words: list[OCRWord] = Field(
        default_factory=list,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("text")
    @classmethod
    def normalize_text(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(value.strip().split())

        if not normalized:
            raise ValueError(
                "OCR satır metni boş olamaz."
            )

        return normalized

    @model_validator(mode="after")
    def validate_word_order(
        self,
    ) -> OCRLine:
        """
        Satır içindeki sözcükleri soldan sağa sıralar.
        """

        if self.words:
            self.words.sort(
                key=lambda item: (
                    item.bounding_box.x1,
                    item.bounding_box.y1,
                )
            )

        return self


class WholeImageOCRResult(BaseModel):
    """
    Hücre yapısı bulunamayan bir görsel bölgenin bütününe uygulanan OCR
    işleminin çıktısını temsil eder.

    Tam bir tablo matrisi üretilemese bile raw_text ve lines alanları retrieval
    için kullanılabilir.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    visual_id: UUID

    report_id: UUID

    page_number: int = Field(
        ge=1,
    )

    image_path: Path

    raw_text: str = ""

    words: list[OCRWord] = Field(
        default_factory=list,
    )

    lines: list[OCRLine] = Field(
        default_factory=list,
    )

    reconstructed_rows: list[list[str]] = Field(
        default_factory=list,
    )

    average_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    preprocessing_variant: str | None = None

    psm_mode: int | None = Field(
        default=None,
        ge=0,
        le=13,
    )

    language: str | None = None

    is_structured: bool = False

    extraction_method: VisualExtractionMethod = (
        VisualExtractionMethod.WHOLE_IMAGE_OCR
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "raw_text",
        mode="before",
    )
    @classmethod
    def normalize_raw_text(
        cls,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        normalized_lines = [
            " ".join(line.strip().split())
            for line in str(value).replace("\r", "\n").split("\n")
        ]

        return "\n".join(
            line
            for line in normalized_lines
            if line
        )

    @field_validator(
        "preprocessing_variant",
        "language",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @field_validator("reconstructed_rows")
    @classmethod
    def normalize_reconstructed_rows(
        cls,
        rows: list[list[str]],
    ) -> list[list[str]]:
        normalized_rows: list[list[str]] = []

        for row in rows:
            normalized_row = [
                " ".join(str(value).strip().split())
                for value in row
            ]

            if any(normalized_row):
                normalized_rows.append(normalized_row)

        return normalized_rows

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text.strip())

    @property
    def is_successful(self) -> bool:
        return self.has_text and (
            self.average_confidence is None
            or self.average_confidence > 0.0
        )

    @property
    def searchable_text(self) -> str:
        if self.raw_text:
            return self.raw_text

        return "\n".join(
            line.text
            for line in self.lines
        )


class VisualAsset(BaseModel):
    """
    PDF sayfasından kırpılmış ve daha sonra OCR veya görsel model ile
    işlenebilecek tek bir görsel varlığı temsil eder.

    Bu model şu modüller arasında ortak sözleşme olacaktır:

    - visual_region_classifier.py
    - visual_table_extractor.py
    - whole_image_ocr.py
    - chart_extractor.py
    - visual_asset_repository.py
    - visual_retriever.py
    - visual_evidence_pipeline.py
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    report_id: UUID

    page_id: UUID | None = None

    page_number: int = Field(
        ge=1,
    )

    region_type: VisualRegionType = VisualRegionType.UNKNOWN

    bounding_box: BoundingBox

    image_path: Path

    source_page_image_path: Path | None = None

    image_width: int | None = Field(
        default=None,
        gt=0,
    )

    image_height: int | None = Field(
        default=None,
        gt=0,
    )

    image_hash: str | None = None

    caption: str | None = None

    nearby_text: str | None = None

    section_title: str | None = None

    subsection_title: str | None = None

    category: str | None = None

    topic: str | None = None

    keywords: list[str] = Field(
        default_factory=list,
    )

    detection_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    processing_status: VisualProcessingStatus = (
        VisualProcessingStatus.DETECTED
    )

    extraction_methods: list[VisualExtractionMethod] = Field(
        default_factory=list,
    )

    ocr_text: str | None = None

    ocr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    requires_whole_image_ocr: bool = False

    requires_vision: bool = False

    error_messages: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "caption",
        "nearby_text",
        "section_title",
        "subsection_title",
        "category",
        "topic",
        "ocr_text",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = " ".join(value.strip().split())
        return normalized or None

    @field_validator("image_hash")
    @classmethod
    def validate_image_hash(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip().casefold()

        if not normalized:
            return None

        if len(normalized) != 64:
            raise ValueError(
                "image_hash SHA-256 biçiminde 64 karakter olmalıdır."
            )

        try:
            int(normalized, 16)
        except ValueError as exc:
            raise ValueError(
                "image_hash yalnızca onaltılık karakterler içermelidir."
            ) from exc

        return normalized

    @field_validator(
        "keywords",
        "error_messages",
    )
    @classmethod
    def normalize_unique_strings(
        cls,
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(str(value).strip().split())

            if not normalized:
                continue

            comparison = normalized.casefold()

            if comparison in seen:
                continue

            seen.add(comparison)
            result.append(normalized)

        return result

    @field_validator("extraction_methods")
    @classmethod
    def normalize_extraction_methods(
        cls,
        values: list[VisualExtractionMethod],
    ) -> list[VisualExtractionMethod]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_processing_flags(
        self,
    ) -> VisualAsset:
        """
        İşleme durumu ile yardımcı bayrakları tutarlı hale getirir.

        Modelde ``validate_assignment=True`` kullanıldığı için validator
        içinde normal ``self.field = value`` ataması yeni bir assignment
        doğrulaması başlatır ve sonsuz özyinelemeye yol açar. Bu nedenle
        yalnızca validator içindeki senkronizasyonlarda ``object.__setattr__``
        kullanılır.
        """

        if (
            self.processing_status
            == VisualProcessingStatus.VISION_REQUIRED
            and not self.requires_vision
        ):
            object.__setattr__(
                self,
                "requires_vision",
                True,
            )

        if (
            self.processing_status
            in {
                VisualProcessingStatus.OCR_COMPLETED,
                VisualProcessingStatus.OCR_PARTIAL,
            }
            and VisualExtractionMethod.WHOLE_IMAGE_OCR
            not in self.extraction_methods
            and VisualExtractionMethod.CELL_OCR
            not in self.extraction_methods
        ):
            object.__setattr__(
                self,
                "extraction_methods",
                [
                    *self.extraction_methods,
                    VisualExtractionMethod.WHOLE_IMAGE_OCR,
                ],
            )

        return self

    @property
    def stable_key(self) -> str:
        """
        Repository ve cache işlemlerinde kullanılabilecek kararlı anahtar.
        """

        if self.image_hash:
            return self.image_hash

        raw_key = (
            f"{self.report_id}|{self.page_number}|"
            f"{self.bounding_box.x1:.3f}|{self.bounding_box.y1:.3f}|"
            f"{self.bounding_box.x2:.3f}|{self.bounding_box.y2:.3f}"
        )

        return sha256(
            raw_key.encode("utf-8")
        ).hexdigest()

    @property
    def searchable_text(self) -> str:
        """
        Görsel retrieval ve chunk üretimi için birleştirilmiş metin.
        """

        parts = (
            self.section_title,
            self.subsection_title,
            self.caption,
            self.nearby_text,
            self.ocr_text,
            " ".join(self.keywords) if self.keywords else None,
        )

        normalized_parts: list[str] = []
        seen: set[str] = set()

        for part in parts:
            if not part:
                continue

            normalized = " ".join(part.strip().split())
            comparison = normalized.casefold()

            if comparison in seen:
                continue

            seen.add(comparison)
            normalized_parts.append(normalized)

        return "\n".join(normalized_parts)

    def add_error(
        self,
        message: str,
    ) -> None:
        normalized = " ".join(message.strip().split())

        if not normalized:
            return

        if normalized.casefold() not in {
            item.casefold()
            for item in self.error_messages
        }:
            self.error_messages.append(normalized)

    def mark_ocr_result(
        self,
        result: WholeImageOCRResult,
        *,
        minimum_confidence: float = 0.45,
    ) -> None:
        """
        WholeImageOCRResult sonucunu görsel varlığa işler.
        """

        if result.visual_id != self.id:
            raise ValueError(
                "OCR sonucu farklı bir visual_id değerine ait."
            )

        self.ocr_text = result.searchable_text or None
        self.ocr_confidence = result.average_confidence

        if (
            VisualExtractionMethod.WHOLE_IMAGE_OCR
            not in self.extraction_methods
        ):
            self.extraction_methods.append(
                VisualExtractionMethod.WHOLE_IMAGE_OCR
            )

        is_confident = (
            result.is_successful
            and (
                result.average_confidence is None
                or result.average_confidence >= minimum_confidence
            )
        )

        if is_confident:
            self.processing_status = (
                VisualProcessingStatus.OCR_COMPLETED
            )
            self.requires_vision = False
        elif result.has_text:
            self.processing_status = (
                VisualProcessingStatus.OCR_PARTIAL
            )
            self.requires_vision = True
        else:
            self.processing_status = (
                VisualProcessingStatus.VISION_REQUIRED
            )
            self.requires_vision = True


class VisualMetric(BaseModel):
    """
    OCR veya görsel model tarafından bir görselden çıkarılan tek bir sayısal
    ya da metinsel sürdürülebilirlik metriğini temsil eder.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    report_id: UUID

    visual_id: UUID

    page_number: int = Field(
        ge=1,
    )

    metric_key: str = Field(
        min_length=1,
    )

    metric_label: str = Field(
        min_length=1,
    )

    value: ScalarValue

    unit: str | None = None

    year: int | None = Field(
        default=None,
        ge=1990,
        le=2100,
    )

    scope: str | None = None

    category: str | None = None

    topic: str | None = None

    source_text: str | None = None

    source_bounding_box: BoundingBox | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    extraction_method: VisualExtractionMethod

    question_ids: list[int] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "metric_key",
        mode="before",
    )
    @classmethod
    def normalize_metric_key(
        cls,
        value: Any,
    ) -> str:
        normalized = "_".join(
            str(value).strip().casefold().replace("-", " ").split()
        )

        if not normalized:
            raise ValueError(
                "metric_key boş olamaz."
            )

        return normalized

    @field_validator(
        "metric_label",
        "unit",
        "scope",
        "category",
        "topic",
        "source_text",
    )
    @classmethod
    def normalize_optional_strings(
        cls,
        value: str | None,
        info: Any,
    ) -> str | None:
        if value is None:
            if info.field_name == "metric_label":
                raise ValueError(
                    "metric_label boş olamaz."
                )
            return None

        normalized = " ".join(value.strip().split())

        if not normalized:
            if info.field_name == "metric_label":
                raise ValueError(
                    "metric_label boş olamaz."
                )
            return None

        return normalized

    @field_validator("question_ids")
    @classmethod
    def normalize_question_ids(
        cls,
        values: list[int],
    ) -> list[int]:
        normalized = sorted(
            {
                int(value)
                for value in values
                if int(value) > 0
            }
        )

        return normalized

    @model_validator(mode="after")
    def validate_value(
        self,
    ) -> VisualMetric:
        if self.value is None:
            raise ValueError(
                "VisualMetric value alanı None olamaz."
            )

        if isinstance(self.value, str):
            normalized = " ".join(self.value.strip().split())

            if not normalized:
                raise ValueError(
                    "VisualMetric value alanı boş metin olamaz."
                )

            self.value = normalized

        return self

    @property
    def evidence_text(self) -> str:
        """
        Retrieval ve cevaplama katmanında kullanılabilecek kanıt cümlesi.
        """

        parts = [
            self.metric_label,
            str(self.value),
        ]

        if self.unit:
            parts.append(self.unit)

        if self.year is not None:
            parts.append(str(self.year))

        if self.scope:
            parts.append(self.scope)

        return " | ".join(parts)