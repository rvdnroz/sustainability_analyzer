from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence
from uuid import UUID

from .multimodal_client import (
    MultimodalGenerationClient,
    MultimodalImage,
    MultimodalResponse,
)
from .question_catalog import (
    QuestionDefinition,
    QuestionGroup,
)
from models.visual_evidence import (
    VisualExtractionMethod,
    VisualMetric,
    VisualRegionType,
)
from retrieval.visual_retriever import (
    VisualGroupRetrievalResult,
    VisualRetrievalMatch,
)


class VisualEvidenceGenerationError(RuntimeError):
    """Görsel kanıt üretimi tamamlanamadığında yükseltilir."""


class VisualEvidenceResponseError(
    VisualEvidenceGenerationError,
):
    """Multimodal model cevabı beklenen şemaya uymadığında yükseltilir."""


class VisualEvidenceStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NO_VISUALS = "no_visuals"
    NO_METRICS = "no_metrics"
    GENERATION_FAILED = "generation_failed"
    INVALID_RESPONSE = "invalid_response"


@dataclass(slots=True, frozen=True)
class VisualEvidenceGeneratorConfig:
    """Görsel kanıt çağrısı ve doğrulaması için sınırlar."""

    maximum_images_per_request: int = 4
    maximum_metrics_per_image: int = 24
    maximum_total_metrics: int = 64

    minimum_metric_confidence: float = 0.45
    minimum_source_text_characters: int = 2
    maximum_source_text_characters: int = 800
    maximum_metric_label_characters: int = 200
    maximum_scope_characters: int = 160

    require_visual_id: bool = True
    require_question_ids: bool = True
    require_source_text: bool = True

    reject_unknown_visual_ids: bool = True
    reject_unknown_question_ids: bool = True
    reject_unrequested_years: bool = False

    keep_low_confidence_metrics: bool = False
    deduplicate_metrics: bool = True

    answer_language: str = "tr"
    temperature: float = 0.0
    maximum_output_tokens: int = 3200
    image_detail: str = "high"

    schema_version: str = "visual-evidence-v1"

    def __post_init__(self) -> None:
        positive_values = {
            "maximum_images_per_request": (
                self.maximum_images_per_request
            ),
            "maximum_metrics_per_image": (
                self.maximum_metrics_per_image
            ),
            "maximum_total_metrics": (
                self.maximum_total_metrics
            ),
            "minimum_source_text_characters": (
                self.minimum_source_text_characters
            ),
            "maximum_source_text_characters": (
                self.maximum_source_text_characters
            ),
            "maximum_metric_label_characters": (
                self.maximum_metric_label_characters
            ),
            "maximum_scope_characters": (
                self.maximum_scope_characters
            ),
            "maximum_output_tokens": (
                self.maximum_output_tokens
            ),
        }

        for name, value in positive_values.items():
            if int(value) < 1:
                raise ValueError(
                    f"{name} pozitif olmalıdır."
                )

        if (
            self.maximum_metrics_per_image
            > self.maximum_total_metrics
        ):
            raise ValueError(
                "maximum_metrics_per_image, maximum_total_metrics "
                "değerini aşamaz."
            )

        if not 0.0 <= self.minimum_metric_confidence <= 1.0:
            raise ValueError(
                "minimum_metric_confidence 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature 0 ile 2 arasında olmalıdır."
            )

        if not self.answer_language.strip():
            raise ValueError(
                "answer_language boş olamaz."
            )

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version boş olamaz."
            )


@dataclass(slots=True, frozen=True)
class VisualEvidenceGenerationRequest:
    """Bir kategori için görsel kanıt üretme isteği."""

    group: QuestionGroup
    retrieval_result: VisualGroupRetrievalResult

    report_id: UUID
    report_year: int
    company_name: str

    model: str | None = None
    additional_instructions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.company_name.strip():
            raise ValueError(
                "company_name boş olamaz."
            )

        if not 1990 <= self.report_year <= 2100:
            raise ValueError(
                "report_year geçersiz."
            )

        if (
            self.group.name
            != self.retrieval_result.group_name
        ):
            raise ValueError(
                "QuestionGroup ile VisualGroupRetrievalResult "
                "aynı gruba ait olmalıdır."
            )

        if len(
            self.retrieval_result.unique_matches
        ) == 0:
            return

        allowed_ids = set(
            self.group.question_ids
        )

        for match in (
            self.retrieval_result.unique_matches
        ):
            invalid = set(
                match.question_ids
            ) - allowed_ids

            if invalid:
                raise ValueError(
                    "Görsel eşleşmesi kategori dışı soru kimliği "
                    f"içeriyor: {sorted(invalid)}"
                )


@dataclass(slots=True, frozen=True)
class RejectedVisualMetric:
    """Model cevabından reddedilen metrik kaydı."""

    index: int
    reason: str
    payload: Mapping[str, Any]


@dataclass(slots=True, frozen=True)
class VisualEvidenceGenerationResult:
    """Kategori bazlı görsel kanıt üretim sonucu."""

    group_name: str
    status: VisualEvidenceStatus

    metrics: tuple[VisualMetric, ...]
    rejected_metrics: tuple[RejectedVisualMetric, ...]

    image_ids: tuple[str, ...]
    question_metric_ids: Mapping[int, tuple[str, ...]]

    model_call_count: int
    raw_response: str | None = None

    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    error: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def metric_count(self) -> int:
        return len(self.metrics)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_metrics)

    @property
    def has_metrics(self) -> bool:
        return bool(self.metrics)

    @property
    def metrics_by_question(
        self,
    ) -> Mapping[int, tuple[VisualMetric, ...]]:
        grouped: dict[int, list[VisualMetric]] = {}

        for metric in self.metrics:
            for question_id in metric.question_ids:
                grouped.setdefault(
                    question_id,
                    [],
                ).append(metric)

        return {
            question_id: tuple(values)
            for question_id, values in grouped.items()
        }


class VisualEvidenceGenerator:
    """
    Seçilmiş görselleri kategori sorularıyla birlikte multimodal modele
    gönderir ve cevabı doğrulanmış `VisualMetric` kayıtlarına dönüştürür.

    Güvenlik ve doğruluk ilkeleri:

    - Model yalnızca istekle gönderilen visual_id değerlerini kullanabilir.
    - Model yalnızca kategori içindeki question_id değerlerine bağlanabilir.
    - Her metrik kaynak metin ve güven değeri taşımalıdır.
    - Görselde açıkça bulunmayan veya tahmine dayalı kayıtlar reddedilir.
    - Düşük güvenli kayıtlar varsayılan olarak saklanmaz.
    - Aynı görsel/metrik/yıl/değer kaydı tekilleştirilir.
    """

    _CODE_BLOCK_RE = re.compile(
        r"^```(?:json)?\s*(.*?)\s*```$",
        re.IGNORECASE | re.DOTALL,
    )

    _JSON_OBJECT_RE = re.compile(
        r"\{.*\}",
        re.DOTALL,
    )

    _METRIC_KEY_RE = re.compile(
        r"^[a-z][a-z0-9_]*$"
    )

    def __init__(
        self,
        *,
        client: MultimodalGenerationClient,
        config: VisualEvidenceGeneratorConfig | None = None,
    ) -> None:
        self.client = client
        self.config = (
            config
            or VisualEvidenceGeneratorConfig()
        )

    def generate(
        self,
        request: VisualEvidenceGenerationRequest,
    ) -> VisualEvidenceGenerationResult:
        matches = tuple(
            request.retrieval_result.unique_matches[
                : self.config.maximum_images_per_request
            ]
        )

        if not matches:
            return VisualEvidenceGenerationResult(
                group_name=request.group.name,
                status=VisualEvidenceStatus.NO_VISUALS,
                metrics=(),
                rejected_metrics=(),
                image_ids=(),
                question_metric_ids={},
                model_call_count=0,
                metadata={
                    "schema_version": (
                        self.config.schema_version
                    ),
                    "reason": "no_retrieved_visuals",
                },
            )

        images = tuple(
            MultimodalImage.from_visual_match(
                match,
                detail=self.config.image_detail,
            )
            for match in matches
        )

        system_instruction = (
            self._build_system_instruction()
        )
        prompt = self._build_prompt(
            request=request,
            matches=matches,
        )

        try:
            response = (
                self.client.generate_multimodal_json(
                    prompt,
                    images,
                    system_instruction=(
                        system_instruction
                    ),
                    temperature=(
                        self.config.temperature
                    ),
                    max_tokens=(
                        self.config
                        .maximum_output_tokens
                    ),
                    model=request.model,
                )
            )
        except Exception as exc:
            return VisualEvidenceGenerationResult(
                group_name=request.group.name,
                status=(
                    VisualEvidenceStatus
                    .GENERATION_FAILED
                ),
                metrics=(),
                rejected_metrics=(),
                image_ids=tuple(
                    image.visual_id
                    or image.stable_key
                    for image in images
                ),
                question_metric_ids={},
                model_call_count=1,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                metadata={
                    "schema_version": (
                        self.config.schema_version
                    ),
                },
            )

        try:
            payload = self._parse_response(
                response.content
            )
            metrics, rejected = self._validate_metrics(
                payload=payload,
                request=request,
                matches=matches,
            )
        except Exception as exc:
            return self._invalid_response_result(
                request=request,
                response=response,
                images=images,
                error=exc,
            )

        question_metric_ids = (
            self._question_metric_ids(metrics)
        )

        if not metrics:
            status = (
                VisualEvidenceStatus.NO_METRICS
            )
        elif rejected:
            status = VisualEvidenceStatus.PARTIAL
        else:
            status = (
                VisualEvidenceStatus.COMPLETED
            )

        usage = response.usage

        return VisualEvidenceGenerationResult(
            group_name=request.group.name,
            status=status,
            metrics=tuple(metrics),
            rejected_metrics=tuple(rejected),
            image_ids=response.image_ids,
            question_metric_ids=(
                question_metric_ids
            ),
            model_call_count=1,
            raw_response=response.content,
            model=response.model,
            provider=response.provider,
            finish_reason=(
                response.response.finish_reason
            ),
            input_tokens=getattr(
                usage,
                "input_tokens",
                None,
            ),
            output_tokens=getattr(
                usage,
                "output_tokens",
                None,
            ),
            total_tokens=getattr(
                usage,
                "total_tokens",
                None,
            ),
            metadata={
                "schema_version": (
                    self.config.schema_version
                ),
                "requested_visual_count": len(
                    images
                ),
                "accepted_metric_count": len(
                    metrics
                ),
                "rejected_metric_count": len(
                    rejected
                ),
                "retrieval_group": (
                    request.retrieval_result
                    .group_name
                ),
                **dict(response.metadata),
            },
        )

    def _build_system_instruction(self) -> str:
        return (
            "Sen sürdürülebilirlik raporlarındaki tablo, grafik, "
            "metrik kartı ve infografikleri analiz eden bir veri çıkarım "
            "uzmanısın. Yalnızca gönderilen görsellerde açıkça görülen "
            "bilgileri kullan. Tahmin, interpolasyon, çubuk yüksekliğinden "
            "yaklaşık değer üretme veya görselde yazmayan birimi ekleme. "
            "Bir değer okunamıyorsa metrik üretme. Yılları, birimleri, "
            "kapsamları ve görsel kimliklerini karıştırma. Çıktıyı yalnızca "
            "geçerli JSON olarak üret."
        )

    def _build_prompt(
        self,
        *,
        request: VisualEvidenceGenerationRequest,
        matches: Sequence[VisualRetrievalMatch],
    ) -> str:
        question_lines = []

        for definition in request.group.questions:
            units = (
                ", ".join(
                    definition.expected_units
                )
                if definition.expected_units
                else "belirtilmemiş"
            )

            aliases = ", ".join(
                definition.aliases[:10]
            )

            question_lines.append(
                f"- question_id={definition.id}; "
                f"key={definition.key}; "
                f"soru={definition.question}; "
                f"topic={definition.topic}; "
                f"beklenen_birimler={units}; "
                f"aliases={aliases}"
            )

        visual_lines = []

        for index, match in enumerate(
            matches,
            start=1,
        ):
            asset = match.asset
            allowed_question_ids = (
                ",".join(
                    str(value)
                    for value in match.question_ids
                )
                or "yok"
            )

            visual_lines.append(
                f"- Görsel {index}: "
                f"visual_id={asset.id}; "
                f"sayfa={asset.page_number}; "
                f"tür={asset.region_type.value}; "
                f"topic={asset.topic or 'belirsiz'}; "
                f"izinli_question_ids={allowed_question_ids}; "
                f"caption={asset.caption or 'yok'}; "
                f"OCR={self._truncate(asset.ocr_text or '', 700)}"
            )

        extra = "\n".join(
            f"- {item.strip()}"
            for item in request.additional_instructions
            if item.strip()
        )

        schema = """
{
  "schema_version": "visual-evidence-v1",
  "group": "water",
  "metrics": [
    {
      "visual_id": "UUID",
      "question_ids": [30],
      "metric_key": "water_consumption",
      "metric_label": "Water consumption",
      "value": 6.2,
      "unit": "million m3",
      "year": 2023,
      "scope": null,
      "source_text": "Water consumption 6.2 million m3",
      "confidence": 0.93,
      "is_explicit": true,
      "notes": null
    }
  ]
}
""".strip()

        return (
            f"Şirket: {request.company_name}\n"
            f"Rapor yılı: {request.report_year}\n"
            f"Soru grubu: {request.group.name}\n\n"
            "SORULAR:\n"
            + "\n".join(question_lines)
            + "\n\nGÖRSELLER VE İZİNLİ SORU EŞLEMESİ:\n"
            + "\n".join(visual_lines)
            + (
                "\n\nEK TALİMATLAR:\n"
                + extra
                if extra
                else ""
            )
            + "\n\nKURALLAR:\n"
            "1. metrics dizisine yalnızca görselde açıkça yazılı değerleri ekle.\n"
            "2. visual_id tam olarak yukarıda verilen UUID değerlerinden biri olmalı.\n"
            "3. Her question_id, ilgili görsel için izinli_question_ids içinde olmalı.\n"
            "4. Bir değer birden fazla soruyu doğrudan cevaplıyorsa question_ids içine tümünü ekle.\n"
            "5. source_text görselde okuduğun kısa ifadeyi içermeli.\n"
            "6. value sayısal ise JSON number kullan; yüzde işaretini unit alanına yaz.\n"
            "7. Karşılaştırma veya metin değeri açıkça yazıyorsa kısa string kullanılabilir.\n"
            "8. Rapor yılı görselde veri yılı olarak görünmüyorsa year=null kullan.\n"
            "9. Güven 0 ile 1 arasında olmalı.\n"
            "10. Okunamayan, örtülü veya tahmini değerleri ekleme.\n\n"
            "ÇIKTI ŞEMASI:\n"
            f"{schema}"
        )

    def _parse_response(
        self,
        content: str,
    ) -> Mapping[str, Any]:
        text = str(content).strip()

        if not text:
            raise VisualEvidenceResponseError(
                "Model boş cevap döndürdü."
            )

        code_match = self._CODE_BLOCK_RE.match(
            text
        )

        if code_match:
            text = code_match.group(1).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            object_match = (
                self._JSON_OBJECT_RE.search(text)
            )

            if object_match is None:
                raise VisualEvidenceResponseError(
                    "Cevap içinde JSON nesnesi bulunamadı."
                )

            try:
                payload = json.loads(
                    object_match.group(0)
                )
            except json.JSONDecodeError as exc:
                raise VisualEvidenceResponseError(
                    "Model cevabındaki JSON çözümlenemedi."
                ) from exc

        if not isinstance(payload, Mapping):
            raise VisualEvidenceResponseError(
                "Model cevabının kökü JSON object olmalıdır."
            )

        metrics = payload.get("metrics")

        if metrics is None:
            raise VisualEvidenceResponseError(
                "Model cevabında metrics alanı bulunamadı."
            )

        if not isinstance(metrics, list):
            raise VisualEvidenceResponseError(
                "metrics alanı JSON array olmalıdır."
            )

        return payload

    def _validate_metrics(
        self,
        *,
        payload: Mapping[str, Any],
        request: VisualEvidenceGenerationRequest,
        matches: Sequence[VisualRetrievalMatch],
    ) -> tuple[
        list[VisualMetric],
        list[RejectedVisualMetric],
    ]:
        raw_metrics = payload.get("metrics", [])

        if len(raw_metrics) > self.config.maximum_total_metrics:
            raw_metrics = raw_metrics[
                : self.config.maximum_total_metrics
            ]

        match_by_visual_id = {
            str(match.asset.id): match
            for match in matches
        }

        allowed_question_ids = set(
            request.group.question_ids
        )

        accepted: list[VisualMetric] = []
        rejected: list[RejectedVisualMetric] = []

        per_visual_count: dict[str, int] = {}

        for index, raw_metric in enumerate(
            raw_metrics
        ):
            if not isinstance(raw_metric, Mapping):
                rejected.append(
                    RejectedVisualMetric(
                        index=index,
                        reason=(
                            "metric kaydı JSON object değil"
                        ),
                        payload={
                            "value": raw_metric
                        },
                    )
                )
                continue

            try:
                metric = self._build_metric(
                    raw=raw_metric,
                    request=request,
                    match_by_visual_id=(
                        match_by_visual_id
                    ),
                    allowed_question_ids=(
                        allowed_question_ids
                    ),
                )
            except Exception as exc:
                rejected.append(
                    RejectedVisualMetric(
                        index=index,
                        reason=str(exc),
                        payload=dict(raw_metric),
                    )
                )
                continue

            visual_uid = str(
                metric.visual_id
            )
            count = per_visual_count.get(
                visual_uid,
                0,
            )

            if (
                count
                >= self.config
                .maximum_metrics_per_image
            ):
                rejected.append(
                    RejectedVisualMetric(
                        index=index,
                        reason=(
                            "görsel başına metrik sınırı aşıldı"
                        ),
                        payload=dict(raw_metric),
                    )
                )
                continue

            per_visual_count[
                visual_uid
            ] = count + 1
            accepted.append(metric)

        if self.config.deduplicate_metrics:
            accepted = self._deduplicate(
                accepted
            )

        return accepted, rejected

    def _build_metric(
        self,
        *,
        raw: Mapping[str, Any],
        request: VisualEvidenceGenerationRequest,
        match_by_visual_id: Mapping[
            str,
            VisualRetrievalMatch,
        ],
        allowed_question_ids: set[int],
    ) -> VisualMetric:
        visual_id_text = self._clean_optional_text(
            raw.get("visual_id")
        )

        if (
            self.config.require_visual_id
            and not visual_id_text
        ):
            raise ValueError(
                "visual_id zorunludur"
            )

        match = (
            match_by_visual_id.get(
                visual_id_text or ""
            )
        )

        if (
            match is None
            and self.config
            .reject_unknown_visual_ids
        ):
            raise ValueError(
                f"izin verilmeyen visual_id: {visual_id_text!r}"
            )

        if match is None:
            raise ValueError(
                "visual_id için eşleşen görsel bulunamadı"
            )

        question_ids = self._parse_question_ids(
            raw.get("question_ids")
        )

        if (
            self.config.require_question_ids
            and not question_ids
        ):
            raise ValueError(
                "question_ids zorunludur"
            )

        unknown_questions = (
            set(question_ids)
            - allowed_question_ids
        )

        if (
            unknown_questions
            and self.config
            .reject_unknown_question_ids
        ):
            raise ValueError(
                "kategori dışı question_ids: "
                f"{sorted(unknown_questions)}"
            )

        visual_allowed = set(
            match.question_ids
        )

        if visual_allowed:
            invalid_for_visual = (
                set(question_ids)
                - visual_allowed
            )

            if invalid_for_visual:
                raise ValueError(
                    "question_ids ilgili görsel için izinli değil: "
                    f"{sorted(invalid_for_visual)}"
                )

        metric_key = self._normalize_metric_key(
            raw.get("metric_key")
        )

        metric_label = self._required_text(
            raw.get("metric_label"),
            "metric_label",
            maximum=(
                self.config
                .maximum_metric_label_characters
            ),
        )

        value = raw.get("value")

        if value is None:
            raise ValueError(
                "value zorunludur"
            )

        if isinstance(
            value,
            (dict, list),
        ):
            raise ValueError(
                "value yalnızca number, string veya boolean olabilir"
            )

        if isinstance(value, str):
            value = " ".join(
                value.strip().split()
            )
            if not value:
                raise ValueError(
                    "value boş metin olamaz"
                )

        unit = self._clean_optional_text(
            raw.get("unit")
        )

        year = self._parse_year(
            raw.get("year")
        )

        if (
            year is not None
            and self.config.reject_unrequested_years
            and year != request.report_year
        ):
            raise ValueError(
                f"beklenmeyen veri yılı: {year}"
            )

        scope = self._clean_optional_text(
            raw.get("scope"),
            maximum=(
                self.config
                .maximum_scope_characters
            ),
        )

        source_text = (
            self._clean_optional_text(
                raw.get("source_text"),
                maximum=(
                    self.config
                    .maximum_source_text_characters
                ),
            )
        )

        if (
            self.config.require_source_text
            and (
                source_text is None
                or len(source_text)
                < self.config
                .minimum_source_text_characters
            )
        ):
            raise ValueError(
                "source_text zorunludur veya çok kısadır"
            )

        confidence = self._parse_confidence(
            raw.get("confidence")
        )

        if (
            confidence
            < self.config
            .minimum_metric_confidence
            and not self.config
            .keep_low_confidence_metrics
        ):
            raise ValueError(
                "confidence eşik altında: "
                f"{confidence:.3f}"
            )

        is_explicit = raw.get(
            "is_explicit",
            True,
        )

        if is_explicit is not True:
            raise ValueError(
                "metrik görselde açıkça belirtilmemiş"
            )

        asset = match.asset

        extraction_method = (
            self._extraction_method(
                asset.region_type
            )
        )

        notes = self._clean_optional_text(
            raw.get("notes"),
            maximum=500,
        )

        return VisualMetric(
            report_id=request.report_id,
            visual_id=asset.id,
            page_number=asset.page_number,
            metric_key=metric_key,
            metric_label=metric_label,
            value=value,
            unit=unit,
            year=year,
            scope=scope,
            category=(
                asset.category
                or next(
                    iter(
                        request.group
                        .esg_categories
                    ),
                    None,
                )
            ),
            topic=(
                asset.topic
                or next(
                    iter(
                        request.group.topics
                    ),
                    None,
                )
            ),
            source_text=source_text,
            source_bounding_box=(
                asset.bounding_box
            ),
            confidence=confidence,
            extraction_method=(
                extraction_method
            ),
            question_ids=list(
                question_ids
            ),
            metadata={
                "schema_version": (
                    self.config.schema_version
                ),
                "group": request.group.name,
                "company_name": (
                    request.company_name
                ),
                "report_year": (
                    request.report_year
                ),
                "visual_retrieval_rank": (
                    match.rank
                ),
                "visual_retrieval_score": (
                    match.score
                ),
                "matched_terms": list(
                    match.matched_terms
                ),
                "notes": notes,
                "model_payload": dict(raw),
            },
        )

    def _deduplicate(
        self,
        metrics: Sequence[VisualMetric],
    ) -> list[VisualMetric]:
        selected: dict[
            tuple[Any, ...],
            VisualMetric,
        ] = {}

        for metric in metrics:
            key = (
                str(metric.visual_id),
                metric.metric_key,
                self._stable_value(
                    metric.value
                ),
                self._normalize_for_key(
                    metric.unit
                ),
                metric.year,
                self._normalize_for_key(
                    metric.scope
                ),
            )

            previous = selected.get(key)

            if (
                previous is None
                or metric.confidence
                > previous.confidence
            ):
                selected[key] = metric
                continue

            merged_ids = sorted(
                {
                    *previous.question_ids,
                    *metric.question_ids,
                }
            )

            if (
                merged_ids
                != previous.question_ids
            ):
                selected[key] = previous.model_copy(
                    update={
                        "question_ids": merged_ids,
                    }
                )

        return sorted(
            selected.values(),
            key=lambda item: (
                item.page_number,
                item.metric_key,
                -item.confidence,
                str(item.id),
            ),
        )

    def _invalid_response_result(
        self,
        *,
        request: VisualEvidenceGenerationRequest,
        response: MultimodalResponse,
        images: Sequence[MultimodalImage],
        error: Exception,
    ) -> VisualEvidenceGenerationResult:
        usage = response.usage

        return VisualEvidenceGenerationResult(
            group_name=request.group.name,
            status=(
                VisualEvidenceStatus
                .INVALID_RESPONSE
            ),
            metrics=(),
            rejected_metrics=(),
            image_ids=tuple(
                image.visual_id
                or image.stable_key
                for image in images
            ),
            question_metric_ids={},
            model_call_count=1,
            raw_response=response.content,
            model=response.model,
            provider=response.provider,
            finish_reason=(
                response.response.finish_reason
            ),
            input_tokens=getattr(
                usage,
                "input_tokens",
                None,
            ),
            output_tokens=getattr(
                usage,
                "output_tokens",
                None,
            ),
            total_tokens=getattr(
                usage,
                "total_tokens",
                None,
            ),
            error=(
                f"{type(error).__name__}: {error}"
            ),
            metadata={
                "schema_version": (
                    self.config.schema_version
                ),
                **dict(response.metadata),
            },
        )

    @staticmethod
    def _question_metric_ids(
        metrics: Sequence[VisualMetric],
    ) -> Mapping[int, tuple[str, ...]]:
        grouped: dict[int, list[str]] = {}

        for metric in metrics:
            for question_id in metric.question_ids:
                grouped.setdefault(
                    question_id,
                    [],
                ).append(
                    str(metric.id)
                )

        return {
            question_id: tuple(
                dict.fromkeys(metric_ids)
            )
            for question_id, metric_ids
            in grouped.items()
        }

    @classmethod
    def _normalize_metric_key(
        cls,
        value: Any,
    ) -> str:
        text = cls._required_text(
            value,
            "metric_key",
            maximum=160,
        )

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            cls._normalize_ascii(text),
        ).strip("_")

        if not normalized:
            raise ValueError(
                "metric_key normalize edilemedi"
            )

        if not normalized[0].isalpha():
            normalized = (
                "metric_" + normalized
            )

        if not cls._METRIC_KEY_RE.fullmatch(
            normalized
        ):
            raise ValueError(
                f"geçersiz metric_key: {normalized!r}"
            )

        return normalized

    @staticmethod
    def _parse_question_ids(
        value: Any,
    ) -> tuple[int, ...]:
        if value is None:
            return ()

        if isinstance(
            value,
            (str, int),
        ):
            value = [value]

        if not isinstance(value, Sequence):
            raise ValueError(
                "question_ids array olmalıdır"
            )

        result: set[int] = set()

        for item in value:
            try:
                question_id = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"geçersiz question_id: {item!r}"
                ) from exc

            if question_id < 1:
                raise ValueError(
                    "question_id pozitif olmalıdır"
                )

            result.add(question_id)

        return tuple(
            sorted(result)
        )

    @staticmethod
    def _parse_confidence(
        value: Any,
    ) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confidence sayısal olmalıdır"
            ) from exc

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence 0 ile 1 arasında olmalıdır"
            )

        return confidence

    @staticmethod
    def _parse_year(
        value: Any,
    ) -> int | None:
        if value in (
            None,
            "",
            "null",
            "None",
        ):
            return None

        try:
            year = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"geçersiz year: {value!r}"
            ) from exc

        if not 1990 <= year <= 2100:
            raise ValueError(
                f"year desteklenen aralık dışında: {year}"
            )

        return year

    @staticmethod
    def _extraction_method(
        region_type: VisualRegionType,
    ) -> VisualExtractionMethod:
        if region_type == VisualRegionType.TABLE:
            return (
                VisualExtractionMethod
                .VISION_TABLE
            )

        if region_type == VisualRegionType.CHART:
            return (
                VisualExtractionMethod
                .VISION_CHART
            )

        return (
            VisualExtractionMethod
            .VISION_INFOGRAPHIC
        )

    @staticmethod
    def _stable_value(
        value: Any,
    ) -> str:
        if isinstance(value, float):
            return format(
                value,
                ".12g",
            )

        return str(value).strip().casefold()

    @classmethod
    def _normalize_for_key(
        cls,
        value: str | None,
    ) -> str:
        return cls._normalize_ascii(
            value or ""
        )

    @staticmethod
    def _normalize_ascii(
        value: str,
    ) -> str:
        import unicodedata

        normalized = unicodedata.normalize(
            "NFKD",
            str(value).strip().casefold(),
        )

        return "".join(
            character
            for character in normalized
            if not unicodedata.combining(
                character
            )
        ).replace(
            "ı",
            "i",
        )

    @staticmethod
    def _required_text(
        value: Any,
        field_name: str,
        *,
        maximum: int,
    ) -> str:
        text = " ".join(
            str(value or "").strip().split()
        )

        if not text:
            raise ValueError(
                f"{field_name} zorunludur"
            )

        if len(text) > maximum:
            text = text[:maximum].rstrip()

        return text

    @staticmethod
    def _clean_optional_text(
        value: Any,
        *,
        maximum: int | None = None,
    ) -> str | None:
        if value is None:
            return None

        text = " ".join(
            str(value).strip().split()
        )

        if not text:
            return None

        if (
            maximum is not None
            and len(text) > maximum
        ):
            text = text[:maximum].rstrip()

        return text

    @staticmethod
    def _truncate(
        value: str,
        maximum: int,
    ) -> str:
        text = " ".join(
            str(value).split()
        )

        if len(text) <= maximum:
            return text

        return (
            text[: maximum - 1].rstrip()
            + "…"
        )


__all__ = [
    "RejectedVisualMetric",
    "VisualEvidenceGenerationError",
    "VisualEvidenceGenerationRequest",
    "VisualEvidenceGenerationResult",
    "VisualEvidenceGenerator",
    "VisualEvidenceGeneratorConfig",
    "VisualEvidenceResponseError",
    "VisualEvidenceStatus",
]
