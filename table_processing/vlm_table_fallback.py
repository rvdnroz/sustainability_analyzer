from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from io import BytesIO
from statistics import mean
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError as exc:
    raise ImportError(
        "Grafik kırpma için Pillow gereklidir. "
        "'python -m pip install pillow' komutunu çalıştırın."
    ) from exc

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "vlm_table_fallback.py için requests paketi gereklidir. "
        "'python -m pip install requests' komutunu çalıştırın."
    ) from exc

from .table_ocr import OCRExtractedTable
from .visual_table_extractor import VisualExtractedTable


logger = logging.getLogger(__name__)


class VLMTableFallbackError(RuntimeError):
    """VLM tablo çıkarımı tamamlanamadığında yükseltilir."""


class CloudflareQuotaExceededError(VLMTableFallbackError):
    """Groq 429 yanıtını taşıyan geriye dönük uyumluluk hatası."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        limit_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type


class VLMTableCellSchema(BaseModel):
    """Vision sağlayıcısı tarafından döndürülecek tek hücre yapısı."""

    row_index: int = Field(
        ge=0,
        description="Sıfır tabanlı satır indeksi.",
    )

    column_index: int = Field(
        ge=0,
        description="Sıfır tabanlı sütun indeksi.",
    )

    text: str = Field(
        default="",
        description="Hücrede görülen metin.",
    )

    row_span: int = Field(
        default=1,
        ge=1,
        description="Hücrenin kapladığı satır sayısı.",
    )

    column_span: int = Field(
        default=1,
        ge=1,
        description="Hücrenin kapladığı sütun sayısı.",
    )

    is_header: bool = Field(
        default=False,
        description="Hücrenin başlık hücresi olup olmadığı.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Hücre okumasına ilişkin güven skoru.",
    )


class VLMTableSchema(BaseModel):
    """Yapılandırılmış tablo çıktısı."""

    title: str | None = Field(
        default=None,
        description="Tablonun varsa başlığı.",
    )

    caption: str | None = Field(
        default=None,
        description="Tablonun varsa açıklaması.",
    )

    row_count: int = Field(
        ge=0,
        description="Toplam satır sayısı.",
    )

    column_count: int = Field(
        ge=0,
        description="Toplam sütun sayısı.",
    )

    headers: list[str] = Field(
        default_factory=list,
        description="Normalize edilmemiş sütun başlıkları.",
    )

    cells: list[VLMTableCellSchema] = Field(
        default_factory=list,
        description="Tablodaki hücreler.",
    )

    footnotes: list[str] = Field(
        default_factory=list,
        description="Tablo altında yer alan dipnotlar.",
    )

    units: list[str] = Field(
        default_factory=list,
        description="Tabloda belirtilen ölçü birimleri.",
    )

    years: list[int] = Field(
        default_factory=list,
        description="Tabloda açıkça görülen yıllar.",
    )

    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Tablo çıkarımının genel güven skoru.",
    )

    uncertain_regions: list[str] = Field(
        default_factory=list,
        description="Okunamayan veya belirsiz bölgeler.",
    )



class VLMChartPointSchema(BaseModel):
    """Grafikteki tek veri noktası."""

    series_name: str = Field(min_length=1)
    year: int | None = Field(default=None, ge=1900, le=2100)
    category_label: str | None = None
    value: str = Field(min_length=1)
    unit: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VLMChartSchema(BaseModel):
    """Groq Qwen Vision yapılandırılmış grafik çıktısı."""

    title: str = Field(min_length=1)
    chart_type: str | None = None
    unit: str | None = None
    years: list[int] = Field(default_factory=list)
    series_names: list[str] = Field(default_factory=list)
    points: list[VLMChartPointSchema] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertain_regions: list[str] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class VLMExtractedChart:
    """
    VLM ile çıkarılmış grafik.

    `matrix` alanı tablo normalizer ile geriye dönük uyumluluk sağlar.
    """

    table_id: str
    page_index: int
    page_number: int
    table_index: int
    extraction_method: str
    model_name: str
    title: str
    caption: str | None
    chart_type: str | None
    unit: str | None
    years: tuple[int, ...]
    series_names: tuple[str, ...]
    points: tuple[dict[str, Any], ...]
    headers: tuple[str, ...]
    matrix: tuple[tuple[str, ...], ...]
    footnotes: tuple[str, ...]
    overall_confidence: float
    uncertain_regions: tuple[str, ...]
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    content_hash: str
    is_valid: bool
    warnings: tuple[str, ...]
    source_image_path: Path
    raw_response: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def row_count(self) -> int:
        return len(self.matrix)

    @property
    def column_count(self) -> int:
        return len(self.headers)

    @property
    def rows(self) -> tuple[tuple[str, ...], ...]:
        return self.matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "extraction_method": self.extraction_method,
            "model_name": self.model_name,
            "title": self.title,
            "caption": self.caption,
            "chart_type": self.chart_type,
            "unit": self.unit,
            "years": list(self.years),
            "series_names": list(self.series_names),
            "points": list(self.points),
            "headers": list(self.headers),
            "matrix": [list(row) for row in self.matrix],
            "footnotes": list(self.footnotes),
            "overall_confidence": self.overall_confidence,
            "uncertain_regions": list(self.uncertain_regions),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "content_hash": self.content_hash,
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
            "source_image_path": str(self.source_image_path),
            "raw_response": self.raw_response,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class VLMExtractedCell:
    """VLM ile çıkarılmış tek tablo hücresi."""

    row_index: int
    column_index: int

    text: str
    normalized_text: str

    row_span: int
    column_span: int

    is_header: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class VLMExtractedRow:
    """VLM ile oluşturulmuş tablo satırı."""

    row_index: int
    cells: tuple[VLMExtractedCell, ...]

    is_header: bool
    is_empty: bool

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "is_header": self.is_header,
            "is_empty": self.is_empty,
            "text": self.text,
        }


@dataclass(slots=True, frozen=True)
class VLMExtractedTable:
    """VLM fallback sonucunda oluşturulan tablo."""

    table_id: str
    page_index: int
    page_number: int
    table_index: int

    extraction_method: str
    model_name: str

    title: str | None
    caption: str | None

    row_count: int
    column_count: int

    headers: tuple[str, ...]
    rows: tuple[VLMExtractedRow, ...]
    cells: tuple[VLMExtractedCell, ...]

    matrix: tuple[tuple[str, ...], ...]

    footnotes: tuple[str, ...]
    units: tuple[str, ...]
    years: tuple[int, ...]

    average_cell_confidence: float
    overall_confidence: float

    uncertain_regions: tuple[str, ...]

    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    content_hash: str

    is_valid: bool
    warnings: tuple[str, ...]

    source_image_path: Path
    raw_response: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "extraction_method": self.extraction_method,
            "model_name": self.model_name,
            "title": self.title,
            "caption": self.caption,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "headers": list(self.headers),
            "rows": [
                row.to_dict()
                for row in self.rows
            ],
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "matrix": [
                list(row)
                for row in self.matrix
            ],
            "footnotes": list(self.footnotes),
            "units": list(self.units),
            "years": list(self.years),
            "average_cell_confidence": (
                self.average_cell_confidence
            ),
            "overall_confidence": self.overall_confidence,
            "uncertain_regions": list(
                self.uncertain_regions
            ),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "content_hash": self.content_hash,
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
            "source_image_path": str(
                self.source_image_path
            ),
            "raw_response": self.raw_response,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class VLMFallbackDecision:
    """Bir tablonun VLM fallback gerektirip gerektirmediği."""

    should_use_vlm: bool
    reasons: tuple[str, ...]

    ocr_confidence: float | None
    empty_cell_ratio: float | None

    visual_table_valid: bool
    ocr_table_valid: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLMTableFallback:
    """
    Klasik extraction yöntemleri yetersiz olduğunda Groq Vision ile tablo veya grafik çıkarır.

    Bu sınıf yalnızca fallback amacıyla kullanılmalıdır. Önerilen sıra:

    1. NativeTableExtractor
    2. VisualTableExtractor
    3. TableOCR
    4. VLMTableFallback

    VLM çağrısı yalnızca şu durumlarda yapılır:

    - Native tablo yapısı bulunamamışsa,
    - Görsel tablo hücre sınırları güvenilir değilse,
    - OCR ortalama güveni eşik altında kalmışsa,
    - OCR boş hücre oranı yüksekse,
    - OCR sonucu açık biçimde VLM fallback istemişse.
    """

    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        account_id: str | None = None,
        model_name: str = "qwen/qwen3.6-27b",
        minimum_ocr_confidence: float = 45.0,
        maximum_empty_cell_ratio: float = 0.55,
        minimum_vlm_confidence: float = 0.55,
        minimum_rows: int = 2,
        minimum_columns: int = 2,
        max_retries: int = 3,
        initial_retry_delay: float = 2.0,
        retry_backoff: float = 2.0,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "model_name boş olamaz."
            )

        if not 0.0 <= minimum_ocr_confidence <= 100.0:
            raise ValueError(
                "minimum_ocr_confidence 0 ile 100 arasında olmalıdır."
            )

        if not 0.0 <= maximum_empty_cell_ratio <= 1.0:
            raise ValueError(
                "maximum_empty_cell_ratio 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= minimum_vlm_confidence <= 1.0:
            raise ValueError(
                "minimum_vlm_confidence 0 ile 1 arasında olmalıdır."
            )

        if minimum_rows < 1:
            raise ValueError(
                "minimum_rows en az 1 olmalıdır."
            )

        if minimum_columns < 1:
            raise ValueError(
                "minimum_columns en az 1 olmalıdır."
            )

        if max_retries < 1:
            raise ValueError(
                "max_retries en az 1 olmalıdır."
            )

        if initial_retry_delay < 0:
            raise ValueError(
                "initial_retry_delay negatif olamaz."
            )

        if retry_backoff < 1:
            raise ValueError(
                "retry_backoff en az 1 olmalıdır."
            )

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens pozitif olmalıdır."
            )

        resolved_api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
        )
        resolved_model_name = (
            os.getenv("GROQ_VISION_MODEL")
            or model_name
        )

        if not resolved_api_key:
            raise VLMTableFallbackError(
                "Groq API anahtarı bulunamadı. "
                "GROQ_API_KEY ortam değişkenini tanımlayın."
            )

        self.model_name = resolved_model_name.strip()
        self.account_id = None
        self.api_token = resolved_api_key.strip()
        self.endpoint = (
            os.getenv("GROQ_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/") + "/chat/completions"
        self.request_timeout = float(
            os.getenv("GROQ_VISION_TIMEOUT_SECONDS", "120")
        )

        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.maximum_empty_cell_ratio = maximum_empty_cell_ratio
        self.minimum_vlm_confidence = minimum_vlm_confidence
        self.minimum_rows = minimum_rows
        self.minimum_columns = minimum_columns

        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.retry_backoff = retry_backoff

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self.groq_call_count = 0
        self.groq_successful_call_count = 0
        self.groq_failed_call_count = 0
        self.groq_rate_limited = False
        self.processed_chart_count = 0
        self.skipped_visual_count = 0

        debug_directory = os.getenv("GROQ_VISION_DEBUG_DIR")
        self.debug_directory = (
            Path(debug_directory).expanduser().resolve()
            if debug_directory
            else None
        )
        if self.debug_directory is not None:
            self.debug_directory.mkdir(parents=True, exist_ok=True)

    def evaluate_fallback_need(
        self,
        *,
        visual_table: VisualExtractedTable,
        ocr_table: OCRExtractedTable | None = None,
    ) -> VLMFallbackDecision:
        """Tablo için VLM çağrısı gerekip gerekmediğini belirler."""

        reasons: list[str] = []

        if not visual_table.is_valid:
            reasons.append(
                "Görsel tablo yapısı geçerli değil."
            )

        if visual_table.row_count < self.minimum_rows:
            reasons.append(
                "Görsel tabloda yeterli satır bulunamadı."
            )

        if visual_table.column_count < self.minimum_columns:
            reasons.append(
                "Görsel tabloda yeterli sütun bulunamadı."
            )

        ocr_confidence: float | None = None
        empty_cell_ratio: float | None = None
        ocr_table_valid: bool | None = None

        if ocr_table is None:
            reasons.append(
                "OCR sonucu bulunmuyor."
            )

        else:
            ocr_confidence = (
                ocr_table.average_confidence
            )

            ocr_table_valid = ocr_table.is_valid

            total_cell_count = max(
                ocr_table.row_count
                * ocr_table.column_count,
                1,
            )

            empty_cell_ratio = (
                ocr_table.empty_cell_count
                / total_cell_count
            )

            if not ocr_table.is_valid:
                reasons.append(
                    "OCR tablo sonucu geçerli değil."
                )

            if (
                ocr_table.average_confidence
                < self.minimum_ocr_confidence
            ):
                reasons.append(
                    "OCR ortalama güven skoru eşik altında."
                )

            if (
                empty_cell_ratio
                > self.maximum_empty_cell_ratio
            ):
                reasons.append(
                    "OCR boş hücre oranı eşik üzerinde."
                )

            if ocr_table.requires_vlm_fallback:
                reasons.append(
                    "OCR modülü tabloyu VLM fallback için işaretledi."
                )

        return VLMFallbackDecision(
            should_use_vlm=bool(reasons),
            reasons=tuple(
                dict.fromkeys(reasons)
            ),
            ocr_confidence=ocr_confidence,
            empty_cell_ratio=(
                round(empty_cell_ratio, 4)
                if empty_cell_ratio is not None
                else None
            ),
            visual_table_valid=visual_table.is_valid,
            ocr_table_valid=ocr_table_valid,
        )

    def extract_table(
        self,
        *,
        visual_table: VisualExtractedTable,
        ocr_table: OCRExtractedTable | None = None,
        force: bool = False,
        contextual_text: str | None = None,
    ) -> VLMExtractedTable:
        """
        Tek bir tablo görüntüsünü Groq Qwen Vision ile yapılandırılmış biçimde çıkarır.

        force=False olduğunda fallback gerekmiyorsa çağrı yapılmaz.
        """

        decision = self.evaluate_fallback_need(
            visual_table=visual_table,
            ocr_table=ocr_table,
        )

        if not force and not decision.should_use_vlm:
            raise VLMTableFallbackError(
                "Tablo VLM fallback gerektirmiyor. "
                "Çağrıyı zorlamak için force=True kullanın."
            )

        image_path = self._resolve_table_image(
            visual_table
        )

        image_bytes = self._read_image(
            image_path
        )

        mime_type = self._detect_mime_type(
            image_path
        )

        prompt = self._build_prompt(
            visual_table=visual_table,
            ocr_table=ocr_table,
            fallback_decision=decision,
            contextual_text=contextual_text,
        )

        schema_result, response_metadata = (
            self._generate_structured_response(
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )
        )

        result = self._build_result(
            visual_table=visual_table,
            schema_result=schema_result,
            source_image_path=image_path,
            fallback_decision=decision,
            response_metadata=response_metadata,
        )

        logger.info(
            "VLM tablo çıkarımı tamamlandı: %s, güven=%.3f",
            result.table_id,
            result.overall_confidence,
        )

        return result

    def extract_chart(
        self,
        *,
        visual_table: VisualExtractedTable,
        contextual_text: str | None = None,
    ) -> VLMExtractedChart:
        """Grafik/çizelge görselini seri-yıl-değer noktalarına dönüştürür."""

        image_path = self._resolve_table_image(visual_table)
        image_bytes = self._read_image(image_path)
        mime_type = self._detect_mime_type(image_path)

        prompt = self._build_chart_prompt(
            visual_table=visual_table,
            contextual_text=contextual_text,
        )

        schema_result, response_metadata = (
            self._generate_chart_response(
                prompt=prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
            )
        )

        points = tuple(
            {
                "series_name": point.series_name.strip(),
                "year": point.year,
                "category_label": (
                    point.category_label.strip()
                    if point.category_label
                    else None
                ),
                "value": point.value.strip(),
                "unit": point.unit.strip() if point.unit else schema_result.unit,
                "confidence": round(point.confidence, 4),
            }
            for point in schema_result.points
            if point.value.strip()
        )

        headers = (
            "metric",
            "series",
            "year_or_category",
            "value",
            "unit",
        )
        matrix = tuple(
            (
                schema_result.title.strip(),
                str(point["series_name"]),
                str(point["year"] or point["category_label"] or ""),
                str(point["value"]),
                str(point["unit"] or ""),
            )
            for point in points
        )

        warnings: list[str] = []
        if not points:
            warnings.append("Grafikten veri noktası çıkarılamadı.")
        if schema_result.uncertain_regions:
            warnings.append(
                "VLM bazı grafik bölgelerini belirsiz olarak işaretledi."
            )

        is_valid = bool(points) and schema_result.overall_confidence >= (
            self.minimum_vlm_confidence
        )

        raw_response = schema_result.model_dump()
        content_hash = hashlib.sha256(
            json.dumps(
                raw_response,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        self.processed_chart_count += 1

        return VLMExtractedChart(
            table_id=visual_table.table_id,
            page_index=visual_table.page_index,
            page_number=visual_table.page_number,
            table_index=visual_table.table_index,
            extraction_method="groq_qwen_chart",
            model_name=self.model_name,
            title=schema_result.title.strip(),
            caption=None,
            chart_type=schema_result.chart_type,
            unit=schema_result.unit,
            years=tuple(sorted(set(schema_result.years))),
            series_names=tuple(dict.fromkeys(schema_result.series_names)),
            points=points,
            headers=headers,
            matrix=matrix,
            footnotes=tuple(schema_result.footnotes),
            overall_confidence=round(
                schema_result.overall_confidence,
                4,
            ),
            uncertain_regions=tuple(
                schema_result.uncertain_regions
            ),
            prompt_tokens=self._safe_int(
                response_metadata.get("prompt_tokens")
            ),
            output_tokens=self._safe_int(
                response_metadata.get("output_tokens")
            ),
            total_tokens=self._safe_int(
                response_metadata.get("total_tokens")
            ),
            content_hash=content_hash,
            is_valid=is_valid,
            warnings=tuple(warnings),
            source_image_path=image_path,
            raw_response=raw_response,
            metadata={
                "visual_region_type": "chart",
                "response_metadata": dict(response_metadata),
                "point_count": len(points),
                "minimum_vlm_confidence": self.minimum_vlm_confidence,
                "validation_reason": (
                    "valid"
                    if is_valid
                    else (
                        "no_points"
                        if not points
                        else "confidence_below_threshold"
                    )
                ),
            },
        )

    def _generate_chart_response(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> tuple[VLMChartSchema, dict[str, Any]]:
        """Grafiğin tamamını tek Groq Vision çağrısıyla çıkarır."""

        answer, response_metadata = self._run_cloudflare_query(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            max_tokens=min(self.max_output_tokens, 1800),
        )

        payload = self._extract_json_payload(answer)
        normalized_payload = self._normalize_chart_payload(payload)

        try:
            schema = VLMChartSchema.model_validate(normalized_payload)
        except ValidationError as exc:
            payload_preview = json.dumps(
                payload,
                ensure_ascii=False,
            )[:1200]
            raise VLMTableFallbackError(
                "Groq Qwen grafik JSON yanıtı beklenen şemaya uymuyor. "
                f"Doğrulama: {exc.errors(include_url=False)}. "
                f"Yanıt: {payload_preview}"
            ) from exc

        quality_errors = self._validate_chart_quality(schema)
        if quality_errors:
            # Ek API çağrısı yapılmaz. Modelin ürettiği görünür noktalar
            # korunur ve tutarsızlıklar belirsiz bölge olarak işaretlenir.
            merged_uncertain = list(schema.uncertain_regions)
            merged_uncertain.extend(quality_errors)
            repaired_payload = schema.model_dump()
            repaired_payload["uncertain_regions"] = list(
                dict.fromkeys(merged_uncertain)
            )
            repaired_payload["overall_confidence"] = min(
                schema.overall_confidence,
                0.70,
            )
            schema = VLMChartSchema.model_validate(repaired_payload)

        return schema, {
            **response_metadata,
            "provider": "groq",
            "model_name": self.model_name,
            "chart_extraction_mode": "single_full_chart_call",
            "quality_errors": quality_errors,
        }

    @staticmethod
    def _build_single_year_prompt(
        *,
        title: str,
        year: int,
        unit: str,
    ) -> str:
        return (
            f"This image shows only the {year} column of the chart "
            f"'{title}'. Read exactly three printed values: "
            "Female is the green lower bar label, Male is the gray upper "
            "bar label, and Total is the bold black line label above the bar. "
            "Return JSON only: "
            f'{{"year":{year},"unit":"{unit}",'
            '"female":"value","male":"value","total":"value",'
            '"confidence":0.95}}. '
            "Do not calculate or infer values; copy only the visible labels."
        )

    @staticmethod
    def _prepare_year_image(
        *,
        image_bytes: bytes,
        mime_type: str,
        year_index: int,
        year_count: int,
        year: int,
        tight: bool = False,
    ) -> tuple[bytes, str]:
        """
        Grafiğin tek bir yıl sütununu, başlık ve legend gürültüsünü azaltarak
        kırpar. Yatay eksen etiketi ile Female/Male/Total etiketlerinin tamamı
        korunur.
        """

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise VLMTableFallbackError(
                "Grafik görseli yıl bazlı kırpma için açılamadı."
            ) from exc

        width, height = image.size
        if year_count < 1 or width < 100 or height < 100:
            return image_bytes, mime_type

        # Grafik alanı yaklaşık olarak sol %8 ile sağ %95 arasındadır.
        chart_left = int(width * 0.08)
        chart_right = int(width * 0.95)
        chart_width = max(chart_right - chart_left, 1)
        slot_width = chart_width / year_count

        center_x = chart_left + (year_index + 0.5) * slot_width
        half_width = slot_width * (0.48 if tight else 0.62)

        x0 = max(0, int(center_x - half_width))
        x1 = min(width, int(center_x + half_width))

        # Tight repair crop removes more neighbouring labels.
        y0 = max(0, int(height * (0.26 if tight else 0.20)))
        y1 = min(height, int(height * (0.81 if tight else 0.84)))

        crop = image.crop((x0, y0, x1, y1))

        # Küçük etiketleri model için büyüt.
        scale = 2
        crop = crop.resize(
            (crop.width * scale, crop.height * scale),
            Image.Resampling.LANCZOS,
        )
        crop = ImageEnhance.Contrast(crop).enhance(1.10)
        crop = ImageEnhance.Sharpness(crop).enhance(1.30)

        output = BytesIO()
        crop.save(output, format="PNG", optimize=True)
        return output.getvalue(), "image/png"

    @staticmethod
    def _normalize_year_payload(
        *,
        payload: Mapping[str, Any],
        year: int,
        unit: str,
    ) -> list[dict[str, Any]]:
        # Nested compact format support.
        source = payload.get("values")
        if isinstance(source, Mapping):
            data = dict(source)
        else:
            data = dict(payload)

        confidence = data.get(
            "confidence",
            payload.get("confidence", 0.90),
        )
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            numeric_confidence = 0.90
        if numeric_confidence > 1.0:
            numeric_confidence /= 100.0
        numeric_confidence = max(0.0, min(numeric_confidence, 1.0))

        points: list[dict[str, Any]] = []
        aliases = {
            "Female": ("female", "women", "woman"),
            "Male": ("male", "men", "man"),
            "Total": ("total", "overall"),
        }

        lowered = {
            str(key).strip().casefold(): value
            for key, value in data.items()
        }

        for series_name, keys in aliases.items():
            raw_value = None
            for key in keys:
                if key in lowered:
                    raw_value = lowered[key]
                    break

            if raw_value is None:
                continue

            value = str(raw_value).strip()
            if not value or value.casefold() in {
                "null", "none", "unknown", "n/a", "na",
            }:
                continue

            points.append(
                {
                    "series_name": series_name,
                    "year": year,
                    "category_label": None,
                    "value": value,
                    "unit": unit,
                    "confidence": numeric_confidence,
                }
            )

        return points


    def _repair_inconsistent_years(
        self,
        *,
        schema: VLMChartSchema,
        quality_errors: Sequence[str],
        image_bytes: bytes,
        mime_type: str,
        unit: str,
        years: Sequence[int],
    ) -> tuple[VLMChartSchema, dict[str, Any]]:
        """
        Tutarsız yılları daha sıkı kırpımla yeniden okur.

        Yeniden okuma da tek bir değeri eksik veya açıkça kırpılmış bırakırsa,
        diğer iki görünür seri aritmetik olarak tutarlıysa üçüncü değer
        doğrulama amaçlı türetilir ve düşük güvenle işaretlenir.
        """

        affected_years = sorted(
            {
                int(match.group(1))
                for error in quality_errors
                for match in [re.search(r"\b((?:19|20)\d{2})\b", error)]
                if match
            }
        )

        points_by_key = {
            (point.year, point.series_name.casefold()): point.model_dump()
            for point in schema.points
            if point.year is not None
        }

        metadata: dict[str, Any] = {
            "affected_years": affected_years,
            "calls": {},
            "derived_values": [],
        }

        for year in affected_years:
            try:
                year_index = list(years).index(year)
            except ValueError:
                continue

            crop_bytes, crop_mime = self._prepare_year_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                year_index=year_index,
                year_count=len(years),
                year=year,
                tight=True,
            )
            self._save_debug_image(
                image_bytes=crop_bytes,
                filename=f"chart_year_{year}_repair.png",
            )

            repair_prompt = (
                f"Read the three printed values for year {year}. "
                "Female is the green lower label, Male is the black number "
                "inside the gray bar, Total is the bold number above the "
                "black line. Return JSON only: "
                f'{{"year":{year},"female":"value","male":"value",'
                f'"total":"value","unit":"{unit}","confidence":0.95}}. '
                "Copy every digit exactly. Do not shorten 30,275 to 30 or 29."
            )

            payload, call_metadata = self._query_chart_json(
                prompt=repair_prompt,
                image_bytes=crop_bytes,
                mime_type=crop_mime,
                max_tokens=min(self.max_output_tokens, 500),
            )
            metadata["calls"][str(year)] = call_metadata

            repaired_points = self._normalize_year_payload(
                payload=payload,
                year=year,
                unit=unit,
            )

            for point in repaired_points:
                points_by_key[
                    (year, point["series_name"].casefold())
                ] = point

            # Deterministic consistency recovery.
            values: dict[str, float] = {}
            for series in ("female", "male", "total"):
                point = points_by_key.get((year, series))
                if point is None:
                    continue
                parsed = self._parse_chart_number(str(point.get("value", "")))
                if parsed is not None:
                    values[series] = parsed

            # Treat implausibly short employee count as unread.
            for series in ("female", "male", "total"):
                value = values.get(series)
                if value is not None and value < 1000:
                    values.pop(series, None)
                    points_by_key.pop((year, series), None)

            missing = [
                series
                for series in ("female", "male", "total")
                if series not in values
            ]

            if len(missing) == 1:
                missing_series = missing[0]
                derived: float | None = None

                if missing_series == "male" and {
                    "female", "total"
                }.issubset(values):
                    derived = values["total"] - values["female"]
                elif missing_series == "female" and {
                    "male", "total"
                }.issubset(values):
                    derived = values["total"] - values["male"]
                elif missing_series == "total" and {
                    "female", "male"
                }.issubset(values):
                    derived = values["female"] + values["male"]

                if derived is not None and derived >= 0:
                    formatted = str(int(round(derived)))
                    points_by_key[(year, missing_series)] = {
                        "series_name": missing_series.title(),
                        "year": year,
                        "category_label": None,
                        "value": formatted,
                        "unit": unit,
                        "confidence": 0.75,
                    }
                    metadata["derived_values"].append(
                        {
                            "year": year,
                            "series": missing_series,
                            "value": formatted,
                            "reason": "arithmetic_consistency_recovery",
                        }
                    )

        repaired_payload = {
            "title": schema.title,
            "chart_type": schema.chart_type,
            "unit": schema.unit,
            "years": list(schema.years),
            "series_names": list(schema.series_names),
            "points": list(points_by_key.values()),
            "footnotes": list(schema.footnotes),
            "overall_confidence": schema.overall_confidence,
            "uncertain_regions": list(schema.uncertain_regions),
        }

        repaired_schema = VLMChartSchema.model_validate(
            self._normalize_chart_payload(repaired_payload)
        )
        return repaired_schema, metadata

    def _query_chart_json(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        answer, metadata = self._run_cloudflare_query(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            max_tokens=max_tokens,
        )

        try:
            payload = dict(self._extract_json_payload(answer))
        except VLMTableFallbackError:
            payload = self._parse_series_text_fallback(answer)

        # Compact series format:
        # {"series":"Female","unit":"employees","values":{"2019":"7420",...}}
        values = payload.get("values")
        series_name = payload.get("series")
        if isinstance(values, Mapping) and series_name:
            points: list[dict[str, Any]] = []
            for raw_year, raw_value in values.items():
                try:
                    year = int(raw_year)
                except (TypeError, ValueError):
                    continue

                value = str(raw_value).strip()
                if not value or value.casefold() in {
                    "null", "none", "unknown", "n/a", "na",
                }:
                    continue

                points.append(
                    {
                        "series_name": str(series_name),
                        "year": year,
                        "category_label": None,
                        "value": value,
                        "unit": payload.get("unit"),
                        "confidence": 0.90,
                    }
                )

            payload = {
                "title": payload.get("title") or "Chart",
                "chart_type": payload.get("chart_type"),
                "unit": payload.get("unit"),
                "years": sorted(
                    point["year"] for point in points
                ),
                "series_names": [str(series_name)],
                "points": points,
                "footnotes": [],
                "overall_confidence": 0.90,
                "uncertain_regions": [],
            }

        return payload, metadata

    @staticmethod
    def _parse_series_text_fallback(
        answer: str,
    ) -> dict[str, Any]:
        """
        Model JSON yerine kısa metin döndürürse yıl-değer çiftlerini ayıklar.

        Prompt tekrarından gelen numaraları almamak için yalnızca 2019-2023
        yıllarına açıkça bağlı değerler kabul edilir.
        """

        cleaned = str(answer or "").strip()
        series_match = re.search(
            r"\b(Female|Male|Total)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        series_name = (
            series_match.group(1).title()
            if series_match
            else "Unknown"
        )

        values: dict[str, str] = {}
        for year in range(2019, 2024):
            patterns = (
                rf"\b{year}\b\s*[:=\-]\s*([0-9][0-9., ]*)",
                rf'"{year}"\s*:\s*"?(\d[0-9., ]*)"?',
            )
            for pattern in patterns:
                match = re.search(pattern, cleaned)
                if match:
                    values[str(year)] = match.group(1).strip()
                    break

        if not values:
            response_preview = re.sub(r"\s+", " ", cleaned)[:500]

            raise VLMTableFallbackError(
                "Groq Qwen yanıtında JSON veya yıl-değer çiftleri bulunamadı. "
                f"Yanıt başlangıcı: {response_preview}"
            )

        return {
            "series": series_name,
            "unit": "employees",
            "values": values,
        }

    @staticmethod
    def _build_chart_structure_prompt(
        base_prompt: str,
    ) -> str:
        return (
            "Identify this chart structure. Return JSON only: "
            '{"title":"...","chart_type":"...",'
            '"unit":"employees","years":[2019,2020,2021,2022,2023],'
            '"series":[{"name":"Female","appearance":"green lower bars"},'
            '{"name":"Male","appearance":"gray upper bars"},'
            '{"name":"Total","appearance":"black line labels"}]}. '
            "Do not extract numeric values yet."
        )


    @staticmethod
    def _build_single_series_prompt(
        *,
        base_prompt: str,
        title: str,
        series_name: str,
        appearance: str,
        years: Sequence[int],
        unit: str | None,
    ) -> str:
        """
        Groq Qwen için kısa ve tek görevli seri promptu.

        Uzun talimat listeleri modelin promptu tekrar etmesine neden olduğu
        için yalnızca hedef seri, görsel konum ve çıktı biçimi verilir.
        """

        year_text = ", ".join(str(year) for year in years)
        target_unit = unit or "employees"

        return (
            f"Read only the {series_name} series in this chart. "
            f"It is the {appearance}. "
            f"Years: {year_text}. "
            f"Return exactly one compact JSON object, no explanation: "
            f'{{"series":"{series_name}","unit":"{target_unit}",'
            f'"values":{{"2019":"value","2020":"value",'
            f'"2021":"value","2022":"value","2023":"value"}}}}. '
            f"Use only numbers visibly printed for this series. "
            f"Do not copy values from another color or line."
        )


    def _repair_chart_series(
        self,
        *,
        schema: VLMChartSchema,
        quality_errors: Sequence[str],
        base_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        series_descriptions: Mapping[str, str],
        appearance_defaults: Mapping[str, str],
    ) -> tuple[VLMChartSchema, dict[str, Any]]:
        affected_series: set[str] = set()

        for error in quality_errors:
            lowered = error.casefold()
            if "female" in lowered:
                affected_series.add("Female")
            if "male" in lowered:
                affected_series.add("Male")
            if "total" in lowered or "!=" in lowered:
                affected_series.update({"Female", "Male", "Total"})

        if not affected_series:
            affected_series.update({"Female", "Male", "Total"})

        retained_points = [
            point.model_dump()
            for point in schema.points
            if point.series_name not in affected_series
        ]

        repair_metadata: dict[str, Any] = {
            "quality_errors": list(quality_errors),
            "requeried_series": sorted(affected_series),
        }

        for series_name in sorted(affected_series):
            appearance = (
                series_descriptions.get(series_name.casefold())
                or appearance_defaults[series_name]
            )

            error_context = "\n- ".join(quality_errors)
            repair_prompt = self._build_single_series_prompt(
                base_prompt=(
                    base_prompt
                    + "\n\nÖnceki okuma şu kalite hatalarını üretti:\n- "
                    + error_context
                    + "\nBu kez yalnızca hedef serinin gerçek etiketlerini oku."
                ),
                title=schema.title,
                series_name=series_name,
                appearance=appearance,
                years=schema.years,
                unit=schema.unit,
            )

            focused_image_bytes, focused_mime_type = (
                self._prepare_series_image(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    series_name=series_name,
                )
            )

            payload, metadata = self._query_chart_json(
                prompt=repair_prompt,
                image_bytes=focused_image_bytes,
                mime_type=focused_mime_type,
                max_tokens=min(self.max_output_tokens, 1000),
            )
            normalized = self._normalize_chart_payload(payload)

            for point in normalized.get("points", []):
                if not isinstance(point, Mapping):
                    continue
                fixed = dict(point)
                fixed["series_name"] = series_name
                if fixed.get("year") is None or fixed.get("value") is None:
                    continue
                fixed["value"] = str(fixed["value"]).strip()
                fixed["unit"] = (
                    self._optional_string(fixed.get("unit"))
                    or schema.unit
                    or "employees"
                )
                fixed["category_label"] = None
                retained_points.append(fixed)

            repair_metadata[series_name] = metadata

        repaired_payload = {
            "title": schema.title,
            "chart_type": schema.chart_type,
            "unit": schema.unit,
            "years": list(schema.years),
            "series_names": list(schema.series_names),
            "points": retained_points,
            "footnotes": list(schema.footnotes),
            "overall_confidence": schema.overall_confidence,
            "uncertain_regions": list(schema.uncertain_regions),
        }

        repaired = VLMChartSchema.model_validate(
            self._normalize_chart_payload(repaired_payload)
        )
        return repaired, repair_metadata

    @staticmethod
    def _normalize_year_list(
        values: Any,
    ) -> list[int]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return []

        years: list[int] = []
        for value in values:
            try:
                year = int(value)
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100:
                years.append(year)

        return sorted(set(years))

    @classmethod
    def _extract_years_from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> list[int]:
        text = json.dumps(payload, ensure_ascii=False)
        return sorted(
            {
                int(value)
                for value in re.findall(r"\b(?:19|20)\d{2}\b", text)
            }
        )

    @staticmethod
    def _normalize_series_descriptions(
        values: Any,
    ) -> dict[str, str]:
        result: dict[str, str] = {}

        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return result

        for value in values:
            if not isinstance(value, Mapping):
                continue
            name = str(value.get("name") or "").strip()
            appearance = str(value.get("appearance") or "").strip()
            if name and appearance:
                result[name.casefold()] = appearance

        return result

    @staticmethod
    def _deduplicate_series_points(
        points: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        by_year: dict[int, dict[str, Any]] = {}

        for point in points:
            try:
                year = int(point.get("year"))
            except (TypeError, ValueError):
                continue

            candidate = dict(point)
            existing = by_year.get(year)

            if existing is None:
                by_year[year] = candidate
                continue

            current_confidence = float(
                candidate.get("confidence", 0.0) or 0.0
            )
            existing_confidence = float(
                existing.get("confidence", 0.0) or 0.0
            )

            if current_confidence > existing_confidence:
                by_year[year] = candidate

        return [by_year[year] for year in sorted(by_year)]

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _sum_metadata_number(
        records: Sequence[Any],
        key: str,
    ) -> float | int | None:
        total = 0.0
        found = False

        def visit(record: Any) -> None:
            nonlocal total, found

            if not isinstance(record, Mapping):
                return

            value = record.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
                found = True

            for nested in record.values():
                if isinstance(nested, Mapping):
                    visit(nested)

        for record in records:
            visit(record)

        if not found:
            return None

        return int(total) if total.is_integer() else round(total, 6)


    @staticmethod
    def _prepare_series_image(
        *,
        image_bytes: bytes,
        mime_type: str,
        series_name: str,
    ) -> tuple[bytes, str]:
        """
        Aynı grafiğin hedef seriye odaklanmış bir kopyasını üretir.

        Groq Qwen tüm grafiği gördüğünde alt bardaki sayıyı diğer serilere
        kopyalayabildiği için hedef olmayan dikey bölgeler bulanıklaştırılır.
        Yıl etiketlerinin bulunduğu alt şerit her seride korunur.
        """

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise VLMTableFallbackError(
                "Grafik görseli seri odaklı kırpma için açılamadı."
            ) from exc

        width, height = image.size
        if width < 50 or height < 50:
            return image_bytes, mime_type

        target = series_name.strip().casefold()

        # Grafik alanının yaklaşık dikey bölgeleri:
        # Total: üst yarı, Male: orta-üst, Female: orta-alt.
        bands = {
            "female": (0.40, 0.92),
            "male": (0.18, 0.78),
            "total": (0.02, 0.58),
        }
        top_ratio, bottom_ratio = bands.get(
            target,
            (0.0, 1.0),
        )

        top = max(0, min(int(height * top_ratio), height - 1))
        bottom = max(top + 1, min(int(height * bottom_ratio), height))

        # Alt yüzde 15'lik şerit yıl etiketleri için korunur.
        year_strip_top = int(height * 0.85)

        blurred = image.filter(ImageFilter.GaussianBlur(radius=10))
        focused = blurred.copy()

        focused.paste(
            image.crop((0, top, width, bottom)),
            (0, top),
        )
        focused.paste(
            image.crop((0, year_strip_top, width, height)),
            (0, year_strip_top),
        )

        # Hedef bölgede sayı etiketlerini okunur tutmak için hafif kontrast.
        focused = ImageEnhance.Contrast(focused).enhance(1.15)
        focused = ImageEnhance.Sharpness(focused).enhance(1.25)

        output = BytesIO()
        focused.save(output, format="PNG", optimize=True)
        return output.getvalue(), "image/png"

    def _save_debug_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
    ) -> None:
        if self.debug_directory is None:
            return

        try:
            (self.debug_directory / filename).write_bytes(image_bytes)
        except OSError:
            logger.warning(
                "Vision debug görseli kaydedilemedi: %s",
                filename,
            )

    @staticmethod
    def _format_chart_points(
        schema: VLMChartSchema,
    ) -> str:
        lines: list[str] = []

        for point in sorted(
            schema.points,
            key=lambda item: (
                item.year or 0,
                item.series_name.casefold(),
            ),
        ):
            lines.append(
                f"{point.year or '-'} | "
                f"{point.series_name} | "
                f"{point.value} | "
                f"confidence={point.confidence:.3f}"
            )

        return "\\n".join(lines) or "(nokta yok)"

    @staticmethod
    def _normalize_chart_payload(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Groq'un aynı anlamsal içeriği farklı JSON biçimlerinde döndürmesini
        tek VLMChartSchema yapısına normalize eder.
        """

        normalized = dict(payload)

        title = (
            normalized.get("title")
            or normalized.get("chart_title")
            or normalized.get("name")
            or "Untitled chart"
        )
        normalized["title"] = str(title).strip() or "Untitled chart"

        if "chart_type" not in normalized:
            normalized["chart_type"] = (
                normalized.get("type")
                or normalized.get("chartType")
            )

        if "overall_confidence" not in normalized:
            # Alan hiç yoksa aşağıda nokta güvenlerinden türetilecek.
            normalized["overall_confidence"] = normalized.get("confidence")

        if "series_names" not in normalized:
            raw_names = (
                normalized.get("seriesNames")
                or normalized.get("legend")
                or []
            )
            normalized["series_names"] = raw_names

        points = normalized.get("points")
        normalized_points: list[dict[str, Any]] = []

        # Doğrudan points listesi.
        if isinstance(points, list):
            for raw_point in points:
                if not isinstance(raw_point, Mapping):
                    continue
                point = dict(raw_point)
                point.setdefault(
                    "series_name",
                    point.get("series")
                    or point.get("seriesName")
                    or point.get("name")
                    or "Unknown",
                )
                if "category_label" not in point:
                    point["category_label"] = (
                        point.get("category")
                        or point.get("label")
                    )
                if "value" not in point:
                    point["value"] = (
                        point.get("amount")
                        or point.get("y")
                    )
                normalized_points.append(point)

        # Nested series format:
        # {"series":[{"name":"Female","points":[...]}]}
        raw_series = normalized.get("series")
        if isinstance(raw_series, list):
            for series_item in raw_series:
                if not isinstance(series_item, Mapping):
                    continue

                series_name = str(
                    series_item.get("name")
                    or series_item.get("series_name")
                    or series_item.get("label")
                    or "Unknown"
                ).strip()

                nested_points = (
                    series_item.get("points")
                    or series_item.get("data")
                )
                if isinstance(nested_points, list):
                    for raw_point in nested_points:
                        if not isinstance(raw_point, Mapping):
                            continue
                        point = dict(raw_point)
                        point["series_name"] = series_name
                        point.setdefault(
                            "year",
                            point.get("x")
                            if isinstance(point.get("x"), int)
                            else None,
                        )
                        point.setdefault(
                            "category_label",
                            point.get("label")
                            or (
                                str(point.get("x"))
                                if point.get("x") is not None
                                and not isinstance(point.get("x"), int)
                                else None
                            ),
                        )
                        point.setdefault(
                            "value",
                            point.get("y")
                            or point.get("amount"),
                        )
                        point.setdefault(
                            "unit",
                            series_item.get("unit")
                            or normalized.get("unit"),
                        )
                        point.setdefault(
                            "confidence",
                            series_item.get("confidence")
                            or normalized.get("overall_confidence")
                            or 0.85,
                        )
                        normalized_points.append(point)

                values = series_item.get("values")
                if isinstance(values, Mapping):
                    for raw_label, raw_value in values.items():
                        year = None
                        try:
                            parsed_year = int(raw_label)
                            if 1900 <= parsed_year <= 2100:
                                year = parsed_year
                        except (TypeError, ValueError):
                            pass

                        normalized_points.append(
                            {
                                "series_name": series_name,
                                "year": year,
                                "category_label": (
                                    None if year is not None
                                    else str(raw_label)
                                ),
                                "value": raw_value,
                                "unit": (
                                    series_item.get("unit")
                                    or normalized.get("unit")
                                ),
                                "confidence": (
                                    series_item.get("confidence")
                                    or normalized.get("overall_confidence")
                                    or 0.85
                                ),
                            }
                        )

        clean_points: list[dict[str, Any]] = []
        for raw_point in normalized_points:
            point = dict(raw_point)

            value = point.get("value")
            if value is None:
                continue
            value_text = str(value).strip()
            if not value_text or value_text.casefold() in {
                "null", "none", "unknown", "n/a", "na",
            }:
                continue
            point["value"] = value_text

            raw_year = point.get("year")
            if raw_year is not None:
                try:
                    year = int(raw_year)
                except (TypeError, ValueError):
                    year = None
                point["year"] = (
                    year if year is not None and 1900 <= year <= 2100
                    else None
                )

            series_name = str(
                point.get("series_name") or "Unknown"
            ).strip()
            point["series_name"] = series_name or "Unknown"

            confidence = point.get("confidence")
            if confidence is None:
                numeric_confidence = 0.85
            else:
                try:
                    numeric_confidence = float(confidence)
                except (TypeError, ValueError):
                    numeric_confidence = 0.85
            if numeric_confidence > 1.0:
                numeric_confidence /= 100.0
            point["confidence"] = max(
                0.0,
                min(numeric_confidence, 1.0),
            )
            clean_points.append(point)

        normalized["points"] = clean_points

        overall_confidence = normalized.get("overall_confidence")
        if overall_confidence is None:
            point_confidences = [
                float(point.get("confidence", 0.85))
                for point in clean_points
            ]
            numeric_overall = (
                sum(point_confidences) / len(point_confidences)
                if point_confidences
                else 0.0
            )
        else:
            try:
                numeric_overall = float(overall_confidence)
            except (TypeError, ValueError):
                point_confidences = [
                    float(point.get("confidence", 0.85))
                    for point in clean_points
                ]
                numeric_overall = (
                    sum(point_confidences) / len(point_confidences)
                    if point_confidences
                    else 0.0
                )

        if numeric_overall > 1.0:
            numeric_overall /= 100.0

        normalized["overall_confidence"] = max(
            0.0,
            min(numeric_overall, 1.0),
        )

        inferred_series = [
            point["series_name"]
            for point in clean_points
            if point.get("series_name")
        ]
        supplied_series = normalized.get("series_names")
        if not isinstance(supplied_series, list):
            supplied_series = []
        normalized["series_names"] = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in supplied_series
                    if str(item).strip()
                ]
                + inferred_series
            )
        )

        inferred_years = [
            int(point["year"])
            for point in clean_points
            if point.get("year") is not None
        ]
        supplied_years = normalized.get("years")
        if not isinstance(supplied_years, list):
            supplied_years = []
        valid_years: list[int] = []
        for item in [*supplied_years, *inferred_years]:
            try:
                year = int(item)
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100:
                valid_years.append(year)
        normalized["years"] = sorted(set(valid_years))

        for field_name in ("footnotes", "uncertain_regions"):
            value = normalized.get(field_name)
            if not isinstance(value, list):
                normalized[field_name] = (
                    [str(value)] if value else []
                )

        return normalized

    @classmethod
    def _validate_chart_quality(
        cls,
        schema: VLMChartSchema,
    ) -> list[str]:
        errors: list[str] = []

        series_lookup = {
            str(name).strip().casefold()
            for name in schema.series_names
        }

        expected_series = {"female", "male", "total"}
        if expected_series.issubset(series_lookup):
            point_map: dict[tuple[int, str], float] = {}

            for point in schema.points:
                if point.year is None:
                    continue

                series_name = point.series_name.strip().casefold()
                parsed_value = cls._parse_chart_number(point.value)

                if parsed_value is not None:
                    point_map[(point.year, series_name)] = parsed_value

            for year in schema.years:
                missing = [
                    series
                    for series in ("female", "male", "total")
                    if (year, series) not in point_map
                ]
                if missing:
                    errors.append(
                        f"{year} yılı için eksik seriler: {', '.join(missing)}"
                    )
                    continue

                female = point_map[(year, "female")]
                male = point_map[(year, "male")]
                total = point_map[(year, "total")]

                if female == male:
                    errors.append(
                        f"{year} yılında Female ve Male aynı okunmuş"
                    )

                tolerance = max(abs(total) * 0.005, 1.0)
                if abs((female + male) - total) > tolerance:
                    errors.append(
                        f"{year} yılında Female + Male != Total"
                    )

        if not schema.points:
            errors.append("Hiç grafik veri noktası çıkarılmadı")

        return errors

    @staticmethod
    def _parse_chart_number(
        value: str,
    ) -> float | None:
        cleaned = str(value).strip().replace(" ", "")

        if not cleaned:
            return None

        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")

        try:
            return float(cleaned)
        except ValueError:
            return None


    def _build_chart_prompt(
        self,
        *,
        visual_table: VisualExtractedTable,
        contextual_text: str | None,
    ) -> str:
        context = (
            contextual_text.strip()
            if contextual_text
            else "Ek bağlam bulunmuyor."
        )

        return f"""
Sen sürdürülebilirlik raporlarındaki grafiklerden yapılandırılmış veri çıkaran
bir belge analiz uzmanısın.

Görev:
Görüntüdeki grafiğin başlığını, grafik türünü, legend/seri adlarını, yılları
ve her seri-yıl/kategori için görünen sayısal değeri çıkar.

Kurallar:
1. Yalnızca görüntüde açıkça görülen değerleri kullan.
2. Bar, çizgi ve yığılmış bar grafiklerinde legend renklerini doğru seriye
   bağla.
3. Yığılmış çalışan grafiğinde yeşil alt bölüm Female, gri üst bölüm Male,
   siyah çizgi ve üstteki kalın sayı Total serisidir.
4. Her yıl için Female, Male ve Total olmak üzere üç ayrı veri noktası çıkar.
5. Female değerini Male alanına kopyalama. Gri bölümün içinde yazan sayı
   Male değeridir.
6. Toplam değer siyah çizgi üzerindeki üst etikettir ve ayrı Total serisi
   olarak yazılmalıdır.
7. Her yıl için Female + Male = Total eşitliğini kontrol et. Eşitlik
   sağlanmıyorsa grafiği yeniden oku; değer türetme veya tahmin yapma.
8. Her veri noktası için series_name, year veya category_label, value, unit
   ve confidence alanlarını doldur.
9. Sayıları görüntüdeki biçimiyle koru; binlik ayraçları silme.
10. Bir değer okunamıyorsa tahmin etme ve points listesine ekleme.
8. Yalnızca JSON nesnesi döndür; açıklama, Markdown veya kod bloğu ekleme.
9. Çıktıyı aşağıdaki JSON şemasına tam olarak uygun üret.

Zorunlu JSON biçimi:
{{
  "title": "string",
  "chart_type": "string veya null",
  "unit": "string veya null",
  "years": [2023],
  "series_names": ["Female", "Male", "Total"],
  "points": [
    {{
      "series_name": "Female",
      "year": 2023,
      "category_label": null,
      "value": "11,136",
      "unit": "employees",
      "confidence": 0.95
    }}
  ],
  "footnotes": ["string"],
  "overall_confidence": 0.0,
  "uncertain_regions": ["string"]
}}

Görsel bilgileri:
- Kimlik: {visual_table.table_id}
- Sayfa: {visual_table.page_number}
- Bölge türü: {visual_table.visual_region_type}
- Çevresel metin: {context}
""".strip()

    def extract_tables(
        self,
        *,
        visual_tables: Sequence[VisualExtractedTable],
        ocr_tables_by_id: Mapping[str, OCRExtractedTable] | None = None,
        force: bool = False,
        contextual_text_by_id: Mapping[str, str] | None = None,
    ) -> list[VLMExtractedTable]:
        """Birden fazla görsel tabloyu VLM fallback ile işler."""

        ocr_tables_by_id = (
            ocr_tables_by_id or {}
        )

        contextual_text_by_id = (
            contextual_text_by_id or {}
        )

        results: list[VLMExtractedTable] = []

        for visual_table in visual_tables:
            ocr_table = ocr_tables_by_id.get(
                visual_table.table_id
            )

            decision = self.evaluate_fallback_need(
                visual_table=visual_table,
                ocr_table=ocr_table,
            )

            if not force and not decision.should_use_vlm:
                continue

            try:
                result = self.extract_table(
                    visual_table=visual_table,
                    ocr_table=ocr_table,
                    force=force,
                    contextual_text=(
                        contextual_text_by_id.get(
                            visual_table.table_id
                        )
                    ),
                )

            except Exception as exc:
                logger.warning(
                    "Tablo %s VLM fallback başarısız: %s",
                    visual_table.table_id,
                    exc,
                )

                continue

            results.append(result)

        return results

    def _generate_structured_response(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> tuple[
        VLMTableSchema,
        dict[str, Any],
    ]:
        answer, response_metadata = self._run_cloudflare_query(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            max_tokens=min(self.max_output_tokens, 4096),
        )

        payload = self._extract_json_payload(answer)

        try:
            return (
                VLMTableSchema.model_validate(payload),
                response_metadata,
            )
        except ValidationError as exc:
            raise VLMTableFallbackError(
                "Groq Qwen Vision JSON yanıtı tablo şemasına uymuyor."
            ) from exc


    def _run_cloudflare_query(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        """Geriye dönük metod adını koruyarak Groq Vision çağrısı yapar."""

        image_data_uri = (
            f"data:{mime_type};base64,"
            f"{base64.b64encode(image_bytes).decode('ascii')}"
        )

        request_payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }

        last_exception: Exception | None = None
        retry_delay = self.initial_retry_delay

        for attempt in range(1, self.max_retries + 1):
            started_at = time.perf_counter()
            try:
                self.groq_call_count += 1
                response = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.request_timeout,
                )
                elapsed = time.perf_counter() - started_at

                try:
                    envelope = response.json()
                except ValueError as exc:
                    raise VLMTableFallbackError(
                        "Groq geçerli JSON olmayan bir HTTP yanıtı döndürdü."
                    ) from exc

                if response.status_code >= 400:
                    error_obj = envelope.get("error") if isinstance(envelope, Mapping) else envelope
                    if response.status_code == 429:
                        self.groq_rate_limited = True
                        self.groq_failed_call_count += 1

                        retry_after_seconds = None
                        raw_retry_after = response.headers.get("retry-after")
                        if raw_retry_after:
                            try:
                                retry_after_seconds = float(raw_retry_after)
                            except (TypeError, ValueError):
                                retry_after_seconds = None

                        error_message = ""
                        if isinstance(error_obj, Mapping):
                            error_message = str(
                                error_obj.get("message")
                                or error_obj.get("type")
                                or ""
                            )
                        else:
                            error_message = str(error_obj or "")

                        limit_type = None
                        lowered = error_message.casefold()
                        for name in ("itpm", "otpm", "tpm", "rpm", "rpd", "tpd"):
                            if name in lowered:
                                limit_type = name.upper()
                                break

                        raise CloudflareQuotaExceededError(
                            "Groq vision rate limitine ulaşıldı. "
                            f"{error_message or 'HTTP 429'}",
                            retry_after_seconds=retry_after_seconds,
                            limit_type=limit_type,
                        )
                    raise VLMTableFallbackError(
                        "Groq Vision çağrısı başarısız "
                        f"(HTTP {response.status_code}): {error_obj or envelope}"
                    )

                choices = envelope.get("choices") if isinstance(envelope, Mapping) else None
                if not isinstance(choices, list) or not choices:
                    raise VLMTableFallbackError("Groq yanıtında choices listesi bulunamadı.")
                message = choices[0].get("message")
                if not isinstance(message, Mapping):
                    raise VLMTableFallbackError("Groq yanıtında message nesnesi bulunamadı.")
                answer = message.get("content")
                if not isinstance(answer, str) or not answer.strip():
                    raise VLMTableFallbackError("Groq Vision boş içerik döndürdü.")

                usage = envelope.get("usage")
                usage = dict(usage) if isinstance(usage, Mapping) else {}
                self.groq_successful_call_count += 1
                return answer.strip(), {
                    "attempt_count": attempt,
                    "elapsed_seconds": round(elapsed, 4),
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "provider": "groq",
                    "model_name": self.model_name,
                }
            except CloudflareQuotaExceededError:
                raise
            except Exception as exc:
                last_exception = exc
                self.groq_failed_call_count += 1
                logger.warning(
                    "Groq Vision çağrısı başarısız (deneme %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt >= self.max_retries:
                    break
                if retry_delay > 0:
                    time.sleep(retry_delay)
                retry_delay *= self.retry_backoff

        raise VLMTableFallbackError(
            "Groq Vision çağrısı bütün denemelerde başarısız oldu."
        ) from last_exception


    @classmethod
    def _extract_json_payload(
        cls,
        answer: str,
    ) -> Mapping[str, Any]:
        """
        Groq Qwen yanıtından ilk geçerli JSON nesnesini ayıklar.

        Model bazen JSON öncesine açıklama, sıra numarası veya birden fazla
        JSON parçası ekleyebilir. Bu nedenle yalnızca json.loads(answer)
        kullanılmaz; ilk açılış süslü parantezinden itibaren raw_decode ile
        ilk geçerli nesne okunur.
        """

        cleaned = str(answer or "").strip()

        if not cleaned:
            raise VLMTableFallbackError(
                "Groq Qwen boş yanıt döndürdü."
            )

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        decoder = json.JSONDecoder()

        # Önce tüm metni doğrudan dene.
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, Mapping):
                return payload
        except json.JSONDecodeError:
            pass

        # Metin içinde bulunan her "{" konumundan itibaren ilk geçerli
        # JSON nesnesini dene. raw_decode sondaki açıklamaları tolere eder.
        candidate_positions = [
            match.start()
            for match in re.finditer(r"\{", cleaned)
        ]

        for position in candidate_positions:
            candidate = cleaned[position:].lstrip()

            try:
                payload, _end_index = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, Mapping):
                return payload

        # Bazı yanıtlar JSON'u kaçışlı metin olarak döndürür:
        # "{\"title\": ...}"
        quoted_start = cleaned.find('"{"')
        if quoted_start >= 0:
            try:
                decoded_string, _ = decoder.raw_decode(
                    cleaned[quoted_start:]
                )
                if isinstance(decoded_string, str):
                    nested = json.loads(decoded_string)
                    if isinstance(nested, Mapping):
                        return nested
            except (json.JSONDecodeError, TypeError):
                pass

        preview = re.sub(
            r"\s+",
            " ",
            cleaned,
        )[:500]

        raise VLMTableFallbackError(
            "Groq Qwen yanıtında ayrıştırılabilir JSON nesnesi bulunamadı. "
            f"Yanıt başlangıcı: {preview}"
        )


    def _build_result(
        self,
        *,
        visual_table: VisualExtractedTable,
        schema_result: VLMTableSchema,
        source_image_path: Path,
        fallback_decision: VLMFallbackDecision,
        response_metadata: Mapping[str, Any],
    ) -> VLMExtractedTable:
        warnings: list[str] = []

        cells = [
            VLMExtractedCell(
                row_index=cell.row_index,
                column_index=cell.column_index,
                text=cell.text.strip(),
                normalized_text=self._normalize_text(
                    cell.text
                ),
                row_span=cell.row_span,
                column_span=cell.column_span,
                is_header=cell.is_header,
                confidence=round(
                    cell.confidence,
                    4,
                ),
            )
            for cell in schema_result.cells
            if (
                cell.row_index
                < schema_result.row_count
                and cell.column_index
                < schema_result.column_count
            )
        ]

        if len(cells) < len(
            schema_result.cells
        ):
            warnings.append(
                "Tablo boyutlarının dışındaki bazı hücreler kaldırıldı."
            )

        matrix = self._build_matrix(
            row_count=schema_result.row_count,
            column_count=schema_result.column_count,
            cells=cells,
        )

        rows = self._build_rows(
            matrix=matrix,
            cells=cells,
        )

        headers = self._build_headers(
            supplied_headers=schema_result.headers,
            matrix=matrix,
            column_count=schema_result.column_count,
        )

        cell_confidences = [
            cell.confidence
            for cell in cells
        ]

        average_cell_confidence = (
            mean(cell_confidences)
            if cell_confidences
            else 0.0
        )

        non_empty_cell_count = sum(
            bool(cell.normalized_text)
            for cell in cells
        )

        expected_cell_count = max(
            schema_result.row_count
            * schema_result.column_count,
            1,
        )

        coverage_ratio = min(
            non_empty_cell_count
            / expected_cell_count,
            1.0,
        )

        is_valid = (
            schema_result.row_count
            >= self.minimum_rows
            and schema_result.column_count
            >= self.minimum_columns
            and schema_result.overall_confidence
            >= self.minimum_vlm_confidence
            and bool(cells)
        )

        if (
            schema_result.row_count
            < self.minimum_rows
        ):
            warnings.append(
                "VLM sonucu yeterli satır içermiyor."
            )

        if (
            schema_result.column_count
            < self.minimum_columns
        ):
            warnings.append(
                "VLM sonucu yeterli sütun içermiyor."
            )

        if (
            schema_result.overall_confidence
            < self.minimum_vlm_confidence
        ):
            warnings.append(
                "VLM genel güven skoru eşik altında."
            )

        if coverage_ratio < 0.40:
            warnings.append(
                "VLM hücre doluluk oranı düşük."
            )

        if schema_result.uncertain_regions:
            warnings.append(
                "VLM bazı tablo bölgelerini belirsiz olarak işaretledi."
            )

        content_hash = self._calculate_content_hash(
            matrix
        )

        raw_response = {
            "title": schema_result.title,
            "caption": schema_result.caption,
            "row_count": schema_result.row_count,
            "column_count": schema_result.column_count,
            "headers": schema_result.headers,
            "cells": [
                cell.model_dump()
                for cell in schema_result.cells
            ],
            "footnotes": schema_result.footnotes,
            "units": schema_result.units,
            "years": schema_result.years,
            "overall_confidence": (
                schema_result.overall_confidence
            ),
            "uncertain_regions": (
                schema_result.uncertain_regions
            ),
        }

        return VLMExtractedTable(
            table_id=visual_table.table_id,
            page_index=visual_table.page_index,
            page_number=visual_table.page_number,
            table_index=visual_table.table_index,
            extraction_method=(
                "groq_qwen_table_fallback"
            ),
            model_name=self.model_name,
            title=schema_result.title,
            caption=schema_result.caption,
            row_count=schema_result.row_count,
            column_count=schema_result.column_count,
            headers=tuple(headers),
            rows=tuple(rows),
            cells=tuple(cells),
            matrix=tuple(
                tuple(row)
                for row in matrix
            ),
            footnotes=tuple(
                schema_result.footnotes
            ),
            units=tuple(
                dict.fromkeys(
                    schema_result.units
                )
            ),
            years=tuple(
                sorted(
                    set(
                        schema_result.years
                    )
                )
            ),
            average_cell_confidence=round(
                average_cell_confidence,
                4,
            ),
            overall_confidence=round(
                schema_result.overall_confidence,
                4,
            ),
            uncertain_regions=tuple(
                schema_result.uncertain_regions
            ),
            prompt_tokens=self._safe_int(
                response_metadata.get(
                    "prompt_tokens"
                )
            ),
            output_tokens=self._safe_int(
                response_metadata.get(
                    "output_tokens"
                )
            ),
            total_tokens=self._safe_int(
                response_metadata.get(
                    "total_tokens"
                )
            ),
            content_hash=content_hash,
            is_valid=is_valid,
            warnings=tuple(warnings),
            source_image_path=source_image_path,
            raw_response=raw_response,
            metadata={
                "fallback_decision": (
                    fallback_decision.to_dict()
                ),
                "coverage_ratio": round(
                    coverage_ratio,
                    4,
                ),
                "source_visual_confidence": (
                    visual_table.extraction_confidence
                ),
                "response_metadata": dict(
                    response_metadata
                ),
            },
        )

    def _build_prompt(
        self,
        *,
        visual_table: VisualExtractedTable,
        ocr_table: OCRExtractedTable | None,
        fallback_decision: VLMFallbackDecision,
        contextual_text: str | None,
    ) -> str:
        ocr_context = (
            self._format_ocr_context(
                ocr_table
            )
            if ocr_table is not None
            else "OCR sonucu bulunmuyor."
        )

        context_section = (
            contextual_text.strip()
            if contextual_text
            else "Ek bağlam bulunmuyor."
        )

        reasons = "\n".join(
            f"- {reason}"
            for reason in fallback_decision.reasons
        )

        return f"""
Sen bir sürdürülebilirlik raporundaki tabloyu yapılandırılmış veriye
dönüştüren belge analiz uzmanısın.

Görev:
Görüntüdeki tabloyu yalnızca görünür içeriğe dayanarak çıkar.

Kurallar:
1. Görüntüde bulunmayan hiçbir değer, yıl, birim veya başlığı üretme.
2. Okunamayan hücreleri tahmin etme; boş metin kullan ve confidence
   değerini düşük ver.
3. Satır ve sütun indekslerini sıfırdan başlat.
4. Birleşik hücreleri row_span ve column_span ile belirt.
5. Tablo başlığını, açıklamasını, birimleri ve dipnotları ayrı alanlara yaz.
6. Sayıları görüntüde yazıldığı biçimde koru.
7. Yüzde, para birimi, ton, tCO2e, MWh ve benzeri birimleri silme.
8. OCR metnini yardımcı sinyal olarak kullan; görüntüyle çelişirse görüntüyü
   esas al.
9. Çıktıyı verilen yapılandırılmış şemaya tam olarak uygun üret.
10. Belirsiz bölgeleri uncertain_regions alanında açıkça belirt.
11. Yalnızca JSON nesnesi döndür; açıklama, Markdown veya kod bloğu ekleme.

Zorunlu JSON biçimi:
{{
  "title": "string veya null",
  "caption": "string veya null",
  "row_count": 0,
  "column_count": 0,
  "headers": ["string"],
  "cells": [
    {{
      "row_index": 0,
      "column_index": 0,
      "text": "string",
      "row_span": 1,
      "column_span": 1,
      "is_header": false,
      "confidence": 0.0
    }}
  ],
  "footnotes": ["string"],
  "units": ["string"],
  "years": [2023],
  "overall_confidence": 0.0,
  "uncertain_regions": ["string"]
}}

Fallback nedenleri:
{reasons or "- Zorunlu VLM çağrısı"}

Görsel çıkarım bilgileri:
- Tablo kimliği: {visual_table.table_id}
- Sayfa: {visual_table.page_number}
- Görsel satır tahmini: {visual_table.row_count}
- Görsel sütun tahmini: {visual_table.column_count}
- Görsel extraction güveni: {visual_table.extraction_confidence:.4f}

Mevcut OCR sonucu:
{ocr_context}

Sayfa veya tablo çevresindeki ek metin:
{context_section}
""".strip()

    @staticmethod
    def _format_ocr_context(
        ocr_table: OCRExtractedTable,
    ) -> str:
        matrix_text = "\n".join(
            " | ".join(row)
            for row in ocr_table.matrix
        )

        return (
            f"OCR ortalama güveni: "
            f"{ocr_table.average_confidence:.2f}\n"
            f"OCR geçerli: {ocr_table.is_valid}\n"
            f"OCR matrisi:\n{matrix_text}"
        )

    @staticmethod
    def _resolve_table_image(
        visual_table: VisualExtractedTable,
    ) -> Path:
        if visual_table.table_image_path is None:
            raise VLMTableFallbackError(
                "VLM fallback için tablo görüntü yolu bulunmuyor."
            )

        image_path = Path(
            visual_table.table_image_path
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Tablo görüntüsü bulunamadı: {image_path}"
            )

        return image_path

    @staticmethod
    def _read_image(
        image_path: Path,
    ) -> bytes:
        try:
            image_bytes = image_path.read_bytes()

        except OSError as exc:
            raise VLMTableFallbackError(
                f"Tablo görüntüsü okunamadı: {image_path}"
            ) from exc

        if not image_bytes:
            raise VLMTableFallbackError(
                "Tablo görüntüsü boş."
            )

        return image_bytes

    @staticmethod
    def _detect_mime_type(
        image_path: Path,
    ) -> str:
        mime_type, _ = mimetypes.guess_type(
            image_path.name
        )

        if mime_type not in {
            "image/png",
            "image/jpeg",
            "image/webp",
        }:
            raise VLMTableFallbackError(
                "Desteklenmeyen tablo görüntü formatı: "
                f"{image_path.suffix}"
            )

        return mime_type

    @classmethod
    def _build_matrix(
        cls,
        *,
        row_count: int,
        column_count: int,
        cells: Sequence[VLMExtractedCell],
    ) -> list[list[str]]:
        matrix = [
            [
                ""
                for _ in range(
                    column_count
                )
            ]
            for _ in range(
                row_count
            )
        ]

        for cell in cells:
            if not (
                0 <= cell.row_index < row_count
            ):
                continue

            if not (
                0 <= cell.column_index < column_count
            ):
                continue

            existing = matrix[
                cell.row_index
            ][
                cell.column_index
            ]

            if existing and cell.normalized_text:
                matrix[
                    cell.row_index
                ][
                    cell.column_index
                ] = (
                    f"{existing} | "
                    f"{cell.normalized_text}"
                )
            else:
                matrix[
                    cell.row_index
                ][
                    cell.column_index
                ] = cell.normalized_text

        return matrix

    @staticmethod
    def _build_rows(
        *,
        matrix: Sequence[Sequence[str]],
        cells: Sequence[VLMExtractedCell],
    ) -> list[VLMExtractedRow]:
        cells_by_row: dict[
            int,
            list[VLMExtractedCell],
        ] = {}

        for cell in cells:
            cells_by_row.setdefault(
                cell.row_index,
                [],
            ).append(cell)

        rows: list[VLMExtractedRow] = []

        for row_index, row_values in enumerate(
            matrix
        ):
            row_cells = sorted(
                cells_by_row.get(
                    row_index,
                    [],
                ),
                key=lambda cell: (
                    cell.column_index
                ),
            )

            rows.append(
                VLMExtractedRow(
                    row_index=row_index,
                    cells=tuple(row_cells),
                    is_header=any(
                        cell.is_header
                        for cell in row_cells
                    ),
                    is_empty=not any(
                        value.strip()
                        for value in row_values
                    ),
                    text=" | ".join(
                        row_values
                    ),
                )
            )

        return rows

    @staticmethod
    def _build_headers(
        *,
        supplied_headers: Sequence[str],
        matrix: Sequence[Sequence[str]],
        column_count: int,
    ) -> list[str]:
        headers = [
            header.strip()
            for header in supplied_headers[
                :column_count
            ]
        ]

        if len(headers) < column_count:
            first_row = (
                list(matrix[0])
                if matrix
                else []
            )

            for column_index in range(
                len(headers),
                column_count,
            ):
                inferred = (
                    first_row[column_index]
                    if column_index
                    < len(first_row)
                    else ""
                )

                headers.append(
                    inferred.strip()
                    or f"column_{column_index + 1}"
                )

        counts: dict[str, int] = {}
        result: list[str] = []

        for column_index, header in enumerate(
            headers
        ):
            base = (
                header
                or f"column_{column_index + 1}"
            )

            normalized = base.casefold()

            counts[normalized] = (
                counts.get(
                    normalized,
                    0,
                )
                + 1
            )

            count = counts[normalized]

            result.append(
                base
                if count == 1
                else f"{base}_{count}"
            )

        return result

    @classmethod
    def _normalize_text(
        cls,
        text: str,
    ) -> str:
        normalized = text.replace(
            "\u00ad",
            ""
        )

        normalized = normalized.replace(
            "\r",
            " "
        )

        normalized = normalized.replace(
            "\n",
            " "
        )

        return cls._WHITESPACE_RE.sub(
            " ",
            normalized,
        ).strip()

    @classmethod
    def _calculate_content_hash(
        cls,
        matrix: Sequence[Sequence[str]],
    ) -> str:
        canonical = "\n".join(
            "\t".join(
                cls._normalize_text(
                    value
                ).casefold()
                for value in row
            )
            for row in matrix
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _extract_finish_reason(
        response: Any,
    ) -> str | None:
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if not candidates:
            return None

        finish_reason = getattr(
            candidates[0],
            "finish_reason",
            None,
        )

        if finish_reason is None:
            return None

        return str(
            getattr(
                finish_reason,
                "value",
                finish_reason,
            )
        )

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None


    def get_usage_counters(self) -> dict[str, int | bool]:
        """Groq vision sayaçlarını ve geriye dönük uyumluluk anahtarlarını döndürür."""
        return {
            "groq_call_count": self.groq_call_count,
            "groq_successful_call_count": self.groq_successful_call_count,
            "groq_failed_call_count": self.groq_failed_call_count,
            "groq_rate_limited": self.groq_rate_limited,
            "cloudflare_call_count": self.groq_call_count,
            "cloudflare_successful_call_count": self.groq_successful_call_count,
            "cloudflare_failed_call_count": self.groq_failed_call_count,
            "cloudflare_quota_exceeded": self.groq_rate_limited,
            "cloudflare_processed_chart_count": self.processed_chart_count,
            "cloudflare_skipped_visual_count": self.skipped_visual_count,
        }


def extract_table_with_vlm(
    *,
    visual_table: VisualExtractedTable,
    ocr_table: OCRExtractedTable | None = None,
    api_key: str | None = None,
    model_name: str = "qwen/qwen3.6-27b",
    force: bool = False,
    contextual_text: str | None = None,
) -> VLMExtractedTable:
    """VLMTableFallback için kısa kullanım arayüzü."""

    fallback = VLMTableFallback(
        api_key=api_key,
        model_name=model_name,
    )

    return fallback.extract_table(
        visual_table=visual_table,
        ocr_table=ocr_table,
        force=force,
        contextual_text=contextual_text,
    )



__all__ = [
    "CloudflareQuotaExceededError",
    "VLMChartPointSchema",
    "VLMChartSchema",
    "VLMExtractedChart",
    "VLMExtractedTable",
    "VLMFallbackDecision",
    "VLMTableFallback",
    "VLMTableFallbackError",
]