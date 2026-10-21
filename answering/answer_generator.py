from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .llm_client import (
    LLMClientError,
    LLMResponse,
    TextGenerationClient,
)
from .question_analyzer import (
    ExpectedAnswerType,
    QuestionAnalysis,
    QuestionIntent,
)
from retrieval.evidence_validator import (
    EvidenceValidationResult,
    ValidatedEvidence,
)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERATION_FAILED = "generation_failed"
    INVALID_RESPONSE = "invalid_response"


@dataclass(slots=True, frozen=True)
class GeneratedAnswer:
    question: str
    answer: str | None
    status: AnswerStatus

    value: Any | None = None
    unit: str | None = None
    year: int | None = None
    metric: str | None = None

    confidence: float = 0.0
    evidence_ids: tuple[str, ...] = ()

    raw_response: str | None = None
    structured_data: Mapping[str, Any] = field(default_factory=dict)

    model: str | None = None
    finish_reason: str | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_answered(self) -> bool:
        return self.status in {
            AnswerStatus.ANSWERED,
            AnswerStatus.PARTIAL,
        }


@dataclass(slots=True, frozen=True)
class AnswerGenerationRequest:
    question_analysis: QuestionAnalysis
    evidence_validation: EvidenceValidationResult

    answer_language: str = "tr"
    temperature: float = 0.0
    maximum_output_tokens: int = 700

    include_evidence_ids: bool = True
    include_confidence: bool = True

    additional_instructions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.answer_language.strip():
            raise ValueError(
                "answer_language boş olamaz.",
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature 0 ile 2 arasında olmalıdır.",
            )

        if self.maximum_output_tokens <= 0:
            raise ValueError(
                "maximum_output_tokens pozitif olmalıdır.",
            )


@dataclass(slots=True, frozen=True)
class AnswerGeneratorConfig:
    maximum_evidence_count: int = 8
    maximum_evidence_characters: int = 5000
    maximum_total_context_characters: int = 24000
    maximum_answer_characters: int = 4000

    minimum_evidence_score: float = 0.0
    minimum_confidence: float = 0.2

    return_without_llm_when_insufficient: bool = True
    require_validated_evidence: bool = True

    include_page_numbers: bool = True
    include_section_titles: bool = True
    include_evidence_scores: bool = False

    def __post_init__(self) -> None:
        if self.maximum_evidence_count <= 0:
            raise ValueError(
                "maximum_evidence_count pozitif olmalıdır.",
            )

        if self.maximum_evidence_characters <= 0:
            raise ValueError(
                "maximum_evidence_characters pozitif olmalıdır.",
            )

        if self.maximum_total_context_characters <= 0:
            raise ValueError(
                "maximum_total_context_characters pozitif olmalıdır.",
            )

        if self.maximum_answer_characters <= 0:
            raise ValueError(
                "maximum_answer_characters pozitif olmalıdır.",
            )

        if not 0.0 <= self.minimum_evidence_score <= 1.0:
            raise ValueError(
                "minimum_evidence_score 0 ile 1 arasında olmalıdır.",
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence 0 ile 1 arasında olmalıdır.",
            )


class AnswerGeneratorError(RuntimeError):
    """Cevap üretimi sırasında oluşan temel hata."""


class AnswerGenerator:
    """
    Doğrulanmış kanıtları yapılandırılmış çıktı üretebilen bir LLM istemcisine göndererek nihai cevabı üretir.

    Bu sınıfın sorumlulukları:
    - Soru analizine göre uygun prompt oluşturmak
    - Doğrulanmış kanıtları kontrollü biçimde bağlama eklemek
    - Provider bağımsız LLM istemcisi üzerinden yapılandırılmış JSON cevap istemek
    - Model çıktısını GeneratedAnswer nesnesine dönüştürmek
    - Yetersiz kanıt durumunda tahmini cevap üretmemek
    """

    _CODE_BLOCK_PATTERN = re.compile(
        r"^```(?:json)?\s*(?P<content>.*?)\s*```$",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        llm_client: TextGenerationClient,
        *,
        config: AnswerGeneratorConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or AnswerGeneratorConfig()

    def generate(
        self,
        request: AnswerGenerationRequest,
    ) -> GeneratedAnswer:
        analysis = request.question_analysis
        validation = request.evidence_validation

        accepted_evidences = tuple(
            evidence
            for evidence in validation.accepted_evidences
            if evidence.score >= self.config.minimum_evidence_score
        )

        if (
            self.config.require_validated_evidence
            and not accepted_evidences
        ):
            if self.config.return_without_llm_when_insufficient:
                return self._build_insufficient_answer(
                    analysis=analysis,
                    validation=validation,
                    metadata=request.metadata,
                )

            raise AnswerGeneratorError(
                "Cevap üretimi için doğrulanmış kanıt bulunamadı.",
            )

        selected_evidences = self._select_evidences(
            accepted_evidences,
        )

        messages = self._build_messages(
            analysis=analysis,
            evidences=selected_evidences,
            answer_language=request.answer_language,
            include_evidence_ids=request.include_evidence_ids,
            include_confidence=request.include_confidence,
            additional_instructions=request.additional_instructions,
        )

        try:
            response = self.llm_client.generate_json(
                messages,
                temperature=request.temperature,
                max_tokens=request.maximum_output_tokens,
            )
        except LLMClientError as exc:
            return self._build_generation_failed_answer(
                analysis=analysis,
                validation=validation,
                metadata=request.metadata,
                error=exc,
            )
        except Exception as exc:
            # Geriye dönük uyumluluk: ortak LLM hata sınıflarına henüz
            # uyarlanmamış istemcilerin hatalarını da pipeline sınırında tut.
            return self._build_generation_failed_answer(
                analysis=analysis,
                validation=validation,
                metadata=request.metadata,
                error=exc,
            )

        return self._parse_response(
            analysis=analysis,
            validation=validation,
            selected_evidences=selected_evidences,
            response=response,
            metadata=request.metadata,
        )

    def generate_answer(
        self,
        question_analysis: QuestionAnalysis,
        evidence_validation: EvidenceValidationResult,
        *,
        answer_language: str = "tr",
        temperature: float = 0.0,
        maximum_output_tokens: int = 700,
        additional_instructions: Sequence[str] = (),
    ) -> GeneratedAnswer:
        return self.generate(
            AnswerGenerationRequest(
                question_analysis=question_analysis,
                evidence_validation=evidence_validation,
                answer_language=answer_language,
                temperature=temperature,
                maximum_output_tokens=maximum_output_tokens,
                additional_instructions=tuple(
                    additional_instructions,
                ),
            ),
        )

    def generate_many(
        self,
        requests: Sequence[AnswerGenerationRequest],
    ) -> list[GeneratedAnswer]:
        answers: list[GeneratedAnswer] = []

        for request in requests:
            try:
                answers.append(
                    self.generate(request),
                )
            except Exception as exc:
                answers.append(
                    GeneratedAnswer(
                        question=(
                            request
                            .question_analysis
                            .original_question
                        ),
                        answer=None,
                        status=AnswerStatus.GENERATION_FAILED,
                        confidence=0.0,
                        metadata={
                            **dict(request.metadata),
                            "error": str(exc),
                        },
                    ),
                )

        return answers

    def _build_messages(
        self,
        *,
        analysis: QuestionAnalysis,
        evidences: Sequence[ValidatedEvidence],
        answer_language: str,
        include_evidence_ids: bool,
        include_confidence: bool,
        additional_instructions: Sequence[str],
    ) -> tuple[dict[str, str], ...]:
        system_prompt = self._build_system_prompt(
            analysis=analysis,
            answer_language=answer_language,
            include_evidence_ids=include_evidence_ids,
            include_confidence=include_confidence,
            additional_instructions=additional_instructions,
        )

        user_prompt = self._build_user_prompt(
            analysis=analysis,
            evidences=evidences,
            include_evidence_ids=include_evidence_ids,
            include_confidence=include_confidence,
        )

        return (
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        )

    def _build_system_prompt(
        self,
        *,
        analysis: QuestionAnalysis,
        answer_language: str,
        include_evidence_ids: bool,
        include_confidence: bool,
        additional_instructions: Sequence[str],
    ) -> str:
        instructions = [
            (
                "Sen sürdürülebilirlik ve ESG raporlarını analiz eden "
                "kanıt-temelli bir cevap üretim bileşenisin."
            ),
            (
                "Yalnızca kullanıcı mesajında verilen doğrulanmış "
                "kanıtlara dayan."
            ),
            (
                "Kanıtlarda yer almayan bilgi, sayı, yıl, oran veya "
                "birim üretme."
            ),
            (
                "Bir sayısal değer verirken ilgili birimi ve yılı "
                "kanıttaki biçimiyle koru."
            ),
            (
                "Kanıtlar soruyu yanıtlamak için yeterli değilse "
                "'insufficient_evidence' durumu döndür."
            ),
            (
                f"Cevabı '{answer_language}' dilinde üret."
            ),
            "Yalnızca geçerli JSON döndür.",
            "JSON dışında açıklama veya Markdown kod bloğu üretme.",
        ]

        instructions.extend(
            self._intent_specific_instructions(analysis),
        )

        if include_evidence_ids:
            instructions.append(
                "Kullandığın kanıt kimliklerini evidence_ids "
                "alanında listele.",
            )

        if include_confidence:
            instructions.append(
                "confidence alanını 0 ile 1 arasında üret.",
            )

        for instruction in additional_instructions:
            cleaned = self._clean_text(instruction)

            if cleaned:
                instructions.append(cleaned)

        return "\n".join(
            f"{index}. {instruction}"
            for index, instruction in enumerate(
                instructions,
                start=1,
            )
        )

    def _intent_specific_instructions(
        self,
        analysis: QuestionAnalysis,
    ) -> list[str]:
        if analysis.intent == QuestionIntent.COMPARISON:
            return [
                "Karşılaştırılan değerleri ayrı ayrı belirt.",
                (
                    "Kanıtlarda açıkça hesaplanmamışsa yeni fark veya "
                    "oran hesabı yapma."
                ),
            ]

        if analysis.intent == QuestionIntent.TREND:
            return [
                "Yıllara ait değerleri kronolojik sırada ver.",
                (
                    "Eğilim yorumunu yalnızca verilen değerlerin açık "
                    "yönüne göre yap."
                ),
            ]

        if analysis.intent == QuestionIntent.TARGET:
            return [
                (
                    "Hedef değer, hedef yıl, baz yıl ve kapsam bilgisini "
                    "ayrı alanlarda belirt."
                ),
                "Gerçekleşen değer ile hedef değeri karıştırma.",
            ]

        if analysis.intent == QuestionIntent.BOOLEAN:
            return [
                (
                    "decision alanında yalnızca yes, no veya "
                    "insufficient_evidence değerlerinden birini kullan."
                ),
            ]

        if analysis.intent == QuestionIntent.LIST:
            return [
                "Yalnızca kanıtlarda açıkça bulunan öğeleri listele.",
                "Tekrarlanan öğeleri birleştir.",
            ]

        if analysis.expected_answer_type in {
            ExpectedAnswerType.NUMBER,
            ExpectedAnswerType.PERCENTAGE,
            ExpectedAnswerType.CURRENCY,
        }:
            return [
                (
                    "Sayısal cevabı value, unit, year ve metric "
                    "alanlarıyla birlikte döndür."
                ),
            ]

        return [
            "Sorunun doğrudan cevabını kısa ve açık biçimde ver.",
        ]

    def _build_user_prompt(
        self,
        *,
        analysis: QuestionAnalysis,
        evidences: Sequence[ValidatedEvidence],
        include_evidence_ids: bool,
        include_confidence: bool,
    ) -> str:
        context = self._build_evidence_context(
            evidences,
        )

        analysis_summary = {
            "intent": analysis.intent.value,
            "expected_answer_type": (
                analysis.expected_answer_type.value
            ),
            "retrieval_strategy": (
                analysis.retrieval_strategy.value
            ),
            "company_name": analysis.company_name,
            "report_id": analysis.report_id,
            "years": list(analysis.years),
            "metrics": [
                metric.name
                for metric in analysis.metrics
            ],
            "units": list(analysis.units),
            "standard_codes": list(
                analysis.standard_codes,
            ),
            "category": analysis.category,
            "expects_numeric_answer": (
                analysis.expects_numeric_answer
            ),
        }

        output_schema = self._build_output_schema(
            analysis=analysis,
            include_evidence_ids=include_evidence_ids,
            include_confidence=include_confidence,
        )

        return (
            "SORU:\n"
            f"{analysis.original_question}\n\n"
            "SORU ANALİZİ:\n"
            f"{json.dumps(analysis_summary, ensure_ascii=False, indent=2)}"
            "\n\n"
            "DOĞRULANMIŞ KANITLAR:\n"
            f"{context}\n\n"
            "ÜRETİLMESİ GEREKEN JSON ŞEMASI:\n"
            f"{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
        )

    def _build_output_schema(
        self,
        *,
        analysis: QuestionAnalysis,
        include_evidence_ids: bool,
        include_confidence: bool,
    ) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "answer": None,
            "status": (
                "answered | partial | insufficient_evidence"
            ),
        }

        if analysis.intent == QuestionIntent.COMPARISON:
            schema["items"] = [
                {
                    "label": None,
                    "value": None,
                    "unit": None,
                    "year": None,
                },
            ]
            schema["comparison_summary"] = None

        elif analysis.intent == QuestionIntent.TREND:
            schema["values"] = [
                {
                    "year": None,
                    "value": None,
                    "unit": None,
                },
            ]
            schema["trend"] = None

        elif analysis.intent == QuestionIntent.TARGET:
            schema.update(
                {
                    "target_value": None,
                    "unit": None,
                    "target_year": None,
                    "base_year": None,
                    "scope": None,
                },
            )

        elif analysis.intent == QuestionIntent.BOOLEAN:
            schema.update(
                {
                    "decision": (
                        "yes | no | insufficient_evidence"
                    ),
                    "reason": None,
                },
            )

        elif analysis.intent == QuestionIntent.LIST:
            schema["items"] = []

        elif analysis.expects_numeric_answer:
            schema.update(
                {
                    "value": None,
                    "unit": None,
                    "year": None,
                    "metric": None,
                },
            )

        if include_evidence_ids:
            schema["evidence_ids"] = []

        if include_confidence:
            schema["confidence"] = 0.0

        return schema

    def _build_evidence_context(
        self,
        evidences: Sequence[ValidatedEvidence],
    ) -> str:
        parts: list[str] = []
        current_length = 0

        for evidence in evidences:
            text = self._truncate_text(
                evidence.text,
                self.config.maximum_evidence_characters,
            )

            if not text:
                continue

            header_parts = [
                f"id={evidence.evidence_id}",
            ]

            if (
                self.config.include_page_numbers
                and evidence.page_number is not None
            ):
                header_parts.append(
                    f"page={evidence.page_number}",
                )

            if (
                self.config.include_section_titles
                and evidence.section_title
            ):
                header_parts.append(
                    f"section={evidence.section_title}",
                )

            if evidence.report_id:
                header_parts.append(
                    f"report_id={evidence.report_id}",
                )

            if evidence.row_index is not None:
                header_parts.append(
                    f"row={evidence.row_index}",
                )

            if self.config.include_evidence_scores:
                header_parts.append(
                    f"score={evidence.score:.4f}",
                )

            block = (
                f"[EVIDENCE {' | '.join(header_parts)}]\n"
                f"{text}"
            )

            separator_length = 2 if parts else 0
            remaining = (
                self.config.maximum_total_context_characters
                - current_length
                - separator_length
            )

            if remaining <= 0:
                break

            if len(block) > remaining:
                block = self._truncate_text(
                    block,
                    remaining,
                )

            if not block:
                break

            parts.append(block)
            current_length += (
                len(block)
                + separator_length
            )

        return "\n\n".join(parts)

    def _select_evidences(
        self,
        evidences: Sequence[ValidatedEvidence],
    ) -> tuple[ValidatedEvidence, ...]:
        deduplicated: dict[
            tuple[str, str],
            ValidatedEvidence,
        ] = {}

        for evidence in evidences:
            normalized_text = self._normalize_text(
                evidence.text,
            )

            if not normalized_text:
                continue

            key = (
                evidence.report_id or "",
                normalized_text,
            )

            existing = deduplicated.get(key)

            if (
                existing is None
                or evidence.score > existing.score
            ):
                deduplicated[key] = evidence

        ordered = sorted(
            deduplicated.values(),
            key=lambda evidence: (
                -evidence.score,
                evidence.page_number
                if evidence.page_number is not None
                else 10**9,
                evidence.evidence_id,
            ),
        )

        return tuple(
            ordered[: self.config.maximum_evidence_count],
        )

    def _build_generation_failed_answer(
        self,
        *,
        analysis: QuestionAnalysis,
        validation: EvidenceValidationResult,
        metadata: Mapping[str, Any],
        error: Exception,
    ) -> GeneratedAnswer:
        error_metadata: dict[str, Any] = {
            **dict(metadata),
            "error": str(error),
            "error_type": type(error).__name__,
            "validation_status": validation.status.value,
            "validation_score": validation.score,
        }

        if isinstance(error, LLMClientError):
            error_metadata.update(
                {
                    "llm_provider": error.provider,
                    "llm_model": error.model,
                    "http_status_code": error.status_code,
                    "retryable": error.retryable,
                    **dict(error.metadata),
                }
            )

        return GeneratedAnswer(
            question=analysis.original_question,
            answer=None,
            status=AnswerStatus.GENERATION_FAILED,
            confidence=0.0,
            evidence_ids=(),
            metadata=error_metadata,
        )

    def _parse_response(
        self,
        *,
        analysis: QuestionAnalysis,
        validation: EvidenceValidationResult,
        selected_evidences: Sequence[ValidatedEvidence],
        response: LLMResponse,
        metadata: Mapping[str, Any],
    ) -> GeneratedAnswer:
        cleaned_response = self._clean_model_response(
            response.content,
        )

        if not cleaned_response:
            return GeneratedAnswer(
                question=analysis.original_question,
                answer=None,
                status=AnswerStatus.INVALID_RESPONSE,
                confidence=0.0,
                raw_response=response.content,
                model=response.model,
                finish_reason=response.finish_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=(
                    response.usage.total_tokens
                    if response.usage is not None
                    else None
                ),
                metadata={
                    **dict(metadata),
                    "error": "Model boş cevap döndürdü.",
                },
            )

        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            extracted_json = self._extract_json_object(
                cleaned_response,
            )

            if extracted_json is None:
                return self._invalid_response(
                    analysis=analysis,
                    response=response,
                    raw_content=cleaned_response,
                    metadata=metadata,
                    error="Model çıktısı geçerli JSON değil.",
                )

            try:
                parsed = json.loads(extracted_json)
            except json.JSONDecodeError:
                return self._invalid_response(
                    analysis=analysis,
                    response=response,
                    raw_content=cleaned_response,
                    metadata=metadata,
                    error="Model JSON çıktısı ayrıştırılamadı.",
                )

        if not isinstance(parsed, Mapping):
            return self._invalid_response(
                analysis=analysis,
                response=response,
                raw_content=cleaned_response,
                metadata=metadata,
                error="Model JSON kök değeri nesne olmalıdır.",
            )

        answer_text = self._optional_string(
            parsed.get("answer"),
        )

        status = self._normalize_status(
            parsed.get("status"),
            answer=answer_text,
            validation=validation,
        )

        confidence = self._normalize_confidence(
            parsed.get("confidence"),
            fallback=validation.score,
        )

        evidence_ids = self._normalize_evidence_ids(
            parsed.get("evidence_ids"),
            selected_evidences=selected_evidences,
        )

        if (
            confidence < self.config.minimum_confidence
            and status == AnswerStatus.ANSWERED
        ):
            status = AnswerStatus.PARTIAL

        value = parsed.get("value")

        if value is None:
            value = (
                parsed.get("target_value")
                or parsed.get("decision")
            )

        unit = self._optional_string(
            parsed.get("unit"),
        )

        year = self._optional_int(
            parsed.get("year")
            or parsed.get("target_year"),
        )

        metric = self._optional_string(
            parsed.get("metric"),
        )

        return GeneratedAnswer(
            question=analysis.original_question,
            answer=self._truncate_text(
                answer_text,
                self.config.maximum_answer_characters,
            )
            if answer_text
            else None,
            status=status,
            value=value,
            unit=unit,
            year=year,
            metric=metric,
            confidence=confidence,
            evidence_ids=evidence_ids,
            raw_response=response.content,
            structured_data=dict(parsed),
            model=response.model,
            finish_reason=response.finish_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=(
                response.usage.total_tokens
                if response.usage is not None
                else None
            ),
            metadata={
                **dict(metadata),
                **dict(response.metadata),
                "llm_provider": getattr(response, "provider", None),
                "validation_status": validation.status.value,
                "validation_score": validation.score,
                "selected_evidence_count": len(
                    selected_evidences,
                ),
            },
        )

    def _invalid_response(
        self,
        *,
        analysis: QuestionAnalysis,
        response: LLMResponse,
        raw_content: str,
        metadata: Mapping[str, Any],
        error: str,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            question=analysis.original_question,
            answer=self._truncate_text(
                raw_content,
                self.config.maximum_answer_characters,
            ),
            status=AnswerStatus.INVALID_RESPONSE,
            confidence=0.0,
            raw_response=response.content,
            model=response.model,
            finish_reason=response.finish_reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=(
                response.usage.total_tokens
                if response.usage is not None
                else None
            ),
            metadata={
                **dict(metadata),
                **dict(response.metadata),
                "llm_provider": getattr(response, "provider", None),
                "error": error,
            },
        )

    def _build_insufficient_answer(
        self,
        *,
        analysis: QuestionAnalysis,
        validation: EvidenceValidationResult,
        metadata: Mapping[str, Any],
    ) -> GeneratedAnswer:
        answer_text = (
            "Bu soruyu yanıtlamak için raporda yeterli ve doğrulanmış "
            "kanıt bulunamadı."
        )

        return GeneratedAnswer(
            question=analysis.original_question,
            answer=answer_text,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            confidence=self._normalize_confidence(
                validation.score,
                fallback=0.0,
            ),
            evidence_ids=(),
            structured_data={
                "answer": answer_text,
                "status": "insufficient_evidence",
                "evidence_ids": [],
                "confidence": validation.score,
            },
            metadata={
                **dict(metadata),
                "validation_status": validation.status.value,
                "validation_score": validation.score,
                "valid_evidence_count": (
                    validation.valid_evidence_count
                ),
                "rejected_evidence_count": (
                    validation.rejected_evidence_count
                ),
            },
        )

    @staticmethod
    def _normalize_status(
        value: Any,
        *,
        answer: str | None,
        validation: EvidenceValidationResult,
    ) -> AnswerStatus:
        normalized = str(value or "").strip().casefold()

        mapping = {
            "answered": AnswerStatus.ANSWERED,
            "partial": AnswerStatus.PARTIAL,
            "insufficient_evidence": (
                AnswerStatus.INSUFFICIENT_EVIDENCE
            ),
            "insufficient": (
                AnswerStatus.INSUFFICIENT_EVIDENCE
            ),
            "not_found": (
                AnswerStatus.INSUFFICIENT_EVIDENCE
            ),
        }

        if normalized in mapping:
            return mapping[normalized]

        if not answer:
            return AnswerStatus.INSUFFICIENT_EVIDENCE

        if validation.is_sufficient:
            return AnswerStatus.ANSWERED

        return AnswerStatus.PARTIAL

    @staticmethod
    def _normalize_confidence(
        value: Any,
        *,
        fallback: float,
    ) -> float:
        try:
            confidence = (
                float(value)
                if value is not None
                else float(fallback)
            )
        except (TypeError, ValueError):
            confidence = float(fallback)

        return round(
            max(
                0.0,
                min(confidence, 1.0),
            ),
            6,
        )

    @staticmethod
    def _normalize_evidence_ids(
        value: Any,
        *,
        selected_evidences: Sequence[ValidatedEvidence],
    ) -> tuple[str, ...]:
        allowed_ids = {
            evidence.evidence_id
            for evidence in selected_evidences
        }

        if value is None:
            return tuple(sorted(allowed_ids))

        if isinstance(value, str):
            candidates = [value]
        else:
            try:
                candidates = list(value)
            except TypeError:
                candidates = [value]

        return tuple(
            sorted(
                {
                    str(candidate).strip()
                    for candidate in candidates
                    if candidate is not None
                    and str(candidate).strip() in allowed_ids
                },
            ),
        )

    def _clean_model_response(
        self,
        content: str,
    ) -> str:
        cleaned = str(content or "").strip()

        match = self._CODE_BLOCK_PATTERN.match(cleaned)

        if match:
            return match.group("content").strip()

        return cleaned

    @staticmethod
    def _extract_json_object(
        text: str,
    ) -> str | None:
        start_index = text.find("{")
        end_index = text.rfind("}")

        if (
            start_index == -1
            or end_index == -1
            or end_index <= start_index
        ):
            return None

        return text[start_index : end_index + 1]

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _truncate_text(
        text: str | None,
        maximum_length: int,
    ) -> str:
        cleaned = AnswerGenerator._clean_text(text)

        if not cleaned:
            return ""

        if len(cleaned) <= maximum_length:
            return cleaned

        if maximum_length <= 3:
            return cleaned[:maximum_length]

        return (
            cleaned[: maximum_length - 3].rstrip()
            + "..."
        )

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return " ".join(
            str(value).replace("\n", " ").split(),
        )

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        if not value:
            return ""

        translation_table = str.maketrans(
            {
                "ı": "i",
                "İ": "i",
                "ş": "s",
                "Ş": "s",
                "ğ": "g",
                "Ğ": "g",
                "ü": "u",
                "Ü": "u",
                "ö": "o",
                "Ö": "o",
                "ç": "c",
                "Ç": "c",
            },
        )

        normalized = str(value).translate(
            translation_table,
        ).casefold()

        return " ".join(normalized.split())


__all__ = [
    "AnswerGenerationRequest",
    "AnswerGenerator",
    "AnswerGeneratorConfig",
    "AnswerGeneratorError",
    "AnswerStatus",
    "GeneratedAnswer",
]