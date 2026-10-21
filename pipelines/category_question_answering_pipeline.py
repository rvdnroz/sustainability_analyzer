from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from answering.answer_generator import AnswerStatus, GeneratedAnswer
from answering.category_answer_generator import (
    CategoryAnswerGenerationResult,
    CategoryAnswerGenerator,
)
from answering.citation_builder import CitationStyle
from answering.question_catalog import QuestionDefinition, QuestionGroup
from models.visual_evidence import VisualMetric
from .visual_evidence_pipeline import (
    VisualEvidencePipeline,
    VisualEvidencePipelineRequest,
    VisualEvidencePipelineResult,
)
from .question_answering_pipeline import (
    PreparedQuestion,
    QuestionAnsweringPipeline,
    QuestionAnsweringResult,
    QuestionAnsweringStatus,
)


logger = logging.getLogger(__name__)


class CategoryQuestionAnsweringStatus(StrEnum):
    """Bir soru grubunun uçtan uca çalışma durumudur."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class CategoryQuestionAnsweringRequest:
    """Bir soru grubunun ayrı retrieval ve tek LLM çağrısıyla işlenmesi."""

    question_group: QuestionGroup
    report_id: str | None = None
    company_name: str | None = None
    report_year: int | None = None

    top_k: int = 12
    rerank_top_k: int = 6
    answer_language: str = "tr"
    citation_style: CitationStyle = CitationStyle.COMPACT
    strict_category_filter: bool = True
    allow_category_filter_fallback: bool = True

    temperature: float = 0.0
    maximum_output_tokens: int = 1200
    model: str | None = None
    additional_instructions: tuple[str, ...] = ()

    continue_on_preparation_error: bool = True

    enable_visual_fallback: bool = True
    report_database_id: int | None = None
    vision_model: str | None = None
    force_visual_refresh: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.question_group, QuestionGroup):
            raise TypeError("question_group bir QuestionGroup olmalıdır.")
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k pozitif olmalıdır.")
        if not self.answer_language.strip():
            raise ValueError("answer_language boş olamaz.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature 0 ile 2 arasında olmalıdır.")
        if self.maximum_output_tokens <= 0:
            raise ValueError("maximum_output_tokens pozitif olmalıdır.")

        if (
            self.enable_visual_fallback
            and self.report_database_id is not None
            and self.report_database_id < 1
        ):
            raise ValueError("report_database_id pozitif olmalıdır.")

        normalized_instructions = tuple(
            instruction.strip()
            for instruction in self.additional_instructions
            if instruction and instruction.strip()
        )
        object.__setattr__(
            self,
            "additional_instructions",
            normalized_instructions,
        )


@dataclass(slots=True, frozen=True)
class CategoryQuestionAnsweringResult:
    """Kategori pipeline'ının toplu ve soru bazlı sonuçlarını taşır."""

    group: str
    status: CategoryQuestionAnsweringStatus
    question_results: tuple[QuestionAnsweringResult, ...]

    prepared_questions: tuple[PreparedQuestion, ...] = ()
    generation_result: CategoryAnswerGenerationResult | None = None
    visual_evidence_result: VisualEvidencePipelineResult | None = None

    preparation_failed_question_ids: tuple[int, ...] = ()
    finalization_failed_question_ids: tuple[int, ...] = ()
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_call_count(self) -> int:
        text_calls = (
            self.generation_result.model_call_count
            if self.generation_result is not None
            else 0
        )
        vision_calls = (
            self.visual_evidence_result.model_call_count
            if self.visual_evidence_result is not None
            else 0
        )
        return text_calls + vision_calls

    @property
    def input_tokens(self) -> int | None:
        values = [
            value
            for value in (
                (
                    self.generation_result.input_tokens
                    if self.generation_result is not None
                    else None
                ),
                (
                    self.visual_evidence_result.metadata.get("input_tokens")
                    if self.visual_evidence_result is not None
                    else None
                ),
            )
            if isinstance(value, int)
        ]
        return sum(values) if values else None

    @property
    def output_tokens(self) -> int | None:
        values = [
            value
            for value in (
                (
                    self.generation_result.output_tokens
                    if self.generation_result is not None
                    else None
                ),
                (
                    self.visual_evidence_result.metadata.get("output_tokens")
                    if self.visual_evidence_result is not None
                    else None
                ),
            )
            if isinstance(value, int)
        ]
        return sum(values) if values else None

    @property
    def total_tokens(self) -> int | None:
        values = [
            value
            for value in (
                (
                    self.generation_result.total_tokens
                    if self.generation_result is not None
                    else None
                ),
                (
                    self.visual_evidence_result.metadata.get("total_tokens")
                    if self.visual_evidence_result is not None
                    else None
                ),
            )
            if isinstance(value, int)
        ]
        return sum(values) if values else None

    @property
    def question_result_by_id(self) -> Mapping[int, QuestionAnsweringResult]:
        result: dict[int, QuestionAnsweringResult] = {}
        for item in self.question_results:
            question_id = item.metadata.get("question_id")
            if isinstance(question_id, int):
                result[question_id] = item
        return result

    @property
    def completed_count(self) -> int:
        return sum(
            item.status == QuestionAnsweringStatus.COMPLETED
            for item in self.question_results
        )

    @property
    def insufficient_evidence_count(self) -> int:
        return sum(
            item.status == QuestionAnsweringStatus.INSUFFICIENT_EVIDENCE
            for item in self.question_results
        )

    @property
    def failed_count(self) -> int:
        return sum(
            item.status == QuestionAnsweringStatus.FAILED
            for item in self.question_results
        )


class CategoryQuestionAnsweringPipelineError(RuntimeError):
    """Kategori soru-cevap hattındaki temel hata."""


@dataclass(slots=True, frozen=True)
class _VisualMetricEvidence:
    id: str
    content: str
    report_id: str
    page_number: int
    score: float
    confidence: float
    source_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CategoryQuestionAnsweringPipeline:
    """
    Bir katalog grubunu ayrı retrieval ve tek model çağrısıyla işler.

    Her soru önce `QuestionAnsweringPipeline.prepare_definition()` ile ayrı
    hazırlanır. Başarılı hazırlıklar `CategoryAnswerGenerator` üzerinden tek
    model çağrısında cevaplanır. Her `GeneratedAnswer`, mevcut
    `finalize_prepared_question()` metoduyla bağımsız sonuca dönüştürülür.
    """

    def __init__(
        self,
        *,
        question_answering_pipeline: QuestionAnsweringPipeline,
        category_answer_generator: CategoryAnswerGenerator,
        visual_evidence_pipeline: VisualEvidencePipeline | None = None,
    ) -> None:
        self.question_answering_pipeline = question_answering_pipeline
        self.category_answer_generator = category_answer_generator
        self.visual_evidence_pipeline = visual_evidence_pipeline

    def run(
        self,
        request: CategoryQuestionAnsweringRequest,
    ) -> CategoryQuestionAnsweringResult:
        group = request.question_group
        run_id = self._resolve_run_id(request)

        prepared_by_id: dict[int, PreparedQuestion] = {}
        results_by_id: dict[int, QuestionAnsweringResult] = {}
        preparation_failed_ids: list[int] = []
        finalization_failed_ids: list[int] = []

        for definition in group.questions:
            try:
                prepared = self.question_answering_pipeline.prepare_definition(
                    definition,
                    report_id=request.report_id,
                    company_name=request.company_name,
                    report_year=request.report_year,
                    top_k=request.top_k,
                    rerank_top_k=request.rerank_top_k,
                    answer_language=request.answer_language,
                    citation_style=request.citation_style,
                    strict_category_filter=request.strict_category_filter,
                    allow_category_filter_fallback=(
                        request.allow_category_filter_fallback
                    ),
                    metadata={
                        **dict(request.metadata),
                        "category_run_id": run_id,
                        "category_group": group.name,
                        "category_group_order": group.order,
                    },
                )
                prepared_by_id[definition.id] = prepared
            except Exception as exc:
                preparation_failed_ids.append(definition.id)
                results_by_id[definition.id] = self._build_failed_result(
                    definition,
                    exc,
                    phase="preparation",
                    request=request,
                    run_id=run_id,
                )
                if not request.continue_on_preparation_error:
                    return self._build_aborted_result(
                        request=request,
                        run_id=run_id,
                        results_by_id=results_by_id,
                        prepared_by_id=prepared_by_id,
                        preparation_failed_ids=preparation_failed_ids,
                        error=(
                            f"Soru {definition.id} hazırlığı başarısız oldu: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )

        prepared_questions = tuple(
            prepared_by_id[definition.id]
            for definition in group.questions
            if definition.id in prepared_by_id
        )

        visual_evidence_result: VisualEvidencePipelineResult | None = None
        visual_fallback_error: str | None = None
        category_generation_error: str | None = None

        visual_fallback_question_ids = tuple(
            prepared.question_id
            for prepared in prepared_questions
            if (
                prepared.question_id is not None
                and not prepared.is_evidence_sufficient
            )
        )

        if (
            visual_fallback_question_ids
            and request.enable_visual_fallback
            and self.visual_evidence_pipeline is not None
            and request.report_database_id is not None
            and request.report_id is not None
            and request.company_name is not None
            and request.report_year is not None
        ):
            candidate_pages = self._candidate_pages_by_question(
                prepared_questions,
                allowed_question_ids=set(visual_fallback_question_ids),
            )

            try:
                visual_evidence_result = self.visual_evidence_pipeline.run(
                    VisualEvidencePipelineRequest(
                        group=group,
                        report_id=request.report_id,
                        report_database_id=request.report_database_id,
                        report_year=request.report_year,
                        company_name=request.company_name,
                        candidate_pages_by_question=candidate_pages,
                        model=request.vision_model,
                        force_refresh=request.force_visual_refresh,
                        metadata={
                            **dict(request.metadata),
                            "category_run_id": run_id,
                            "visual_fallback_question_ids": list(
                                visual_fallback_question_ids
                            ),
                        },
                    )
                )

                if visual_evidence_result.metrics:
                    prepared_by_id = self._augment_prepared_questions(
                        prepared_by_id=prepared_by_id,
                        metrics_by_question=(
                            visual_evidence_result.metrics_by_question
                        ),
                        request=request,
                        run_id=run_id,
                    )

                    prepared_questions = tuple(
                        prepared_by_id[definition.id]
                        for definition in group.questions
                        if definition.id in prepared_by_id
                    )
            except Exception as exc:
                visual_evidence_result = None
                visual_fallback_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception(
                    "Vision fallback başarısız oldu: group=%s",
                    group.name,
                )
                print(
                    "  Vision fallback hatası: "
                    f"{visual_fallback_error}",
                    flush=True,
                )

        for prepared in prepared_questions:
            accepted = tuple(prepared.accepted_evidences)
            print(
                "  Hazırlık özeti: "
                f"question_id={prepared.question_id} "
                f"retrieval={prepared.retrieval_result_count} "
                f"fused={prepared.fused_result_count} "
                f"reranked={prepared.reranked_result_count} "
                f"candidates={len(prepared.evidence_candidates)} "
                f"accepted={len(accepted)} "
                f"validation={prepared.evidence_validation.status.value} "
                f"score={prepared.evidence_validation.score:.3f}",
                flush=True,
            )

            for evidence_index, evidence in enumerate(
                accepted[:3],
                start=1,
            ):
                content = str(
                    getattr(evidence, "content", None)
                    or getattr(evidence, "text", None)
                    or getattr(evidence, "contextualized_text", None)
                    or ""
                ).replace("\n", " ").strip()

                page_number = getattr(
                    evidence,
                    "page_number",
                    None,
                )
                evidence_id = (
                    getattr(evidence, "evidence_id", None)
                    or getattr(evidence, "id", None)
                    or getattr(evidence, "identifier", None)
                )

                print(
                    "    Kanıt "
                    f"{evidence_index}: "
                    f"id={evidence_id} "
                    f"page={page_number} "
                    f"text={content[:350]}",
                    flush=True,
                )

        generation_result: CategoryAnswerGenerationResult | None = None
        if prepared_questions:
            try:
                generation_result = self.category_answer_generator.generate_group(
                    group=group.name,
                    prepared_questions=prepared_questions,
                    answer_language=request.answer_language,
                    temperature=request.temperature,
                    maximum_output_tokens=request.maximum_output_tokens,
                    model=request.model,
                    additional_instructions=request.additional_instructions,
                    metadata={
                        **dict(request.metadata),
                        "category_run_id": run_id,
                        "category_group": group.name,
                        "category_group_order": group.order,
                        "report_id": request.report_id,
                        "company_name": request.company_name,
                        "report_year": request.report_year,
                    },
                )

                if generation_result.error:
                    category_generation_error = generation_result.error
                    print(
                        "  Kategori üretim/JSON hatası: "
                        f"{category_generation_error}",
                        flush=True,
                    )

                if generation_result.missing_question_ids:
                    print(
                        "  Model cevabında eksik soru ID'leri: "
                        f"{generation_result.missing_question_ids}",
                        flush=True,
                    )

                if generation_result.unexpected_question_ids:
                    print(
                        "  Model cevabında beklenmeyen soru ID'leri: "
                        f"{generation_result.unexpected_question_ids}",
                        flush=True,
                    )

                if generation_result.raw_response and (
                    generation_result.error
                    or generation_result.status.value
                    == "insufficient_evidence"
                ):
                    preview = generation_result.raw_response[:2000]
                    print(
                        "  Ham model cevabı (ilk 2000 karakter):\n"
                        f"{preview}",
                        flush=True,
                    )
            except Exception as exc:
                generation_result = None
                category_generation_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception(
                    "Kategori model çağrısı başarısız oldu: group=%s",
                    group.name,
                )
                print(
                    "  Kategori model çağrısı hatası: "
                    f"{category_generation_error}",
                    flush=True,
                )

                for definition in group.questions:
                    prepared = prepared_by_id.get(definition.id)
                    if prepared is None:
                        continue
                    results_by_id[definition.id] = self._build_failed_result(
                        definition,
                        exc,
                        phase="category_generation",
                        request=request,
                        run_id=run_id,
                        prepared=prepared,
                    )
        
        if generation_result is not None:
            answers_by_id = generation_result.answer_by_question_id
            for definition in group.questions:
                prepared = prepared_by_id.get(definition.id)
                if prepared is None:
                    continue

                generated_answer = answers_by_id.get(definition.id)
                if generated_answer is None:
                    generated_answer = self._build_missing_generated_answer(
                        definition,
                        generation_result,
                        run_id=run_id,
                    )

                try:
                    results_by_id[definition.id] = (
                        self.question_answering_pipeline.
                        finalize_prepared_question(
                            prepared,
                            generated_answer,
                            citation_style=request.citation_style,
                            metadata={
                                "category_run_id": run_id,
                                "category_group": group.name,
                                "category_group_order": group.order,
                                "category_model_call_count": (
                                    generation_result.model_call_count
                                ),
                                "category_usage_scope": "group_result",
                            },
                        )
                    )
                except Exception as exc:
                    finalization_failed_ids.append(definition.id)
                    results_by_id[definition.id] = self._build_failed_result(
                        definition,
                        exc,
                        phase="finalization",
                        request=request,
                        run_id=run_id,
                        prepared=prepared,
                        generated_answer=generated_answer,
                    )

        ordered_results = tuple(
            results_by_id.get(definition.id)
            or self._build_failed_result(
                definition,
                CategoryQuestionAnsweringPipelineError(
                    "Soru için hiçbir sonuç üretilemedi."
                ),
                phase="result_assembly",
                request=request,
                run_id=run_id,
                prepared=prepared_by_id.get(definition.id),
            )
            for definition in group.questions
        )

        failed_details = tuple(
            (
                item.metadata.get("question_id"),
                item.metadata.get("failure_phase"),
                item.error,
            )
            for item in ordered_results
            if item.status == QuestionAnsweringStatus.FAILED
        )

        for question_id, phase, failure_error in failed_details:
            print(
                "  Soru hatası: "
                f"question_id={question_id} "
                f"phase={phase or 'answer_mapping'} "
                f"error={failure_error}",
                flush=True,
            )

        status = self._resolve_status(ordered_results)
        error = self._resolve_group_error(
            ordered_results,
            generation_result=generation_result,
        )

        return CategoryQuestionAnsweringResult(
            group=group.name,
            status=status,
            question_results=ordered_results,
            prepared_questions=prepared_questions,
            generation_result=generation_result,
            visual_evidence_result=visual_evidence_result,
            preparation_failed_question_ids=tuple(preparation_failed_ids),
            finalization_failed_question_ids=tuple(finalization_failed_ids),
            error=error,
            metadata={
                **dict(request.metadata),
                "category_run_id": run_id,
                "category_group": group.name,
                "category_group_order": group.order,
                "question_count": group.size,
                "prepared_question_count": len(prepared_questions),
                "preparation_failed_count": len(preparation_failed_ids),
                "finalization_failed_count": len(finalization_failed_ids),
                "completed_count": sum(
                    item.status == QuestionAnsweringStatus.COMPLETED
                    for item in ordered_results
                ),
                "insufficient_evidence_count": sum(
                    item.status
                    == QuestionAnsweringStatus.INSUFFICIENT_EVIDENCE
                    for item in ordered_results
                ),
                "failed_count": sum(
                    item.status == QuestionAnsweringStatus.FAILED
                    for item in ordered_results
                ),
                "text_model_call_count": (
                    generation_result.model_call_count
                    if generation_result is not None
                    else 0
                ),
                "vision_model_call_count": (
                    visual_evidence_result.model_call_count
                    if visual_evidence_result is not None
                    else 0
                ),
                "model_call_count": (
                    (
                        generation_result.model_call_count
                        if generation_result is not None
                        else 0
                    )
                    + (
                        visual_evidence_result.model_call_count
                        if visual_evidence_result is not None
                        else 0
                    )
                ),
                "visual_fallback_requested_question_ids": (
                    visual_fallback_question_ids
                ),
                "visual_metric_count": (
                    visual_evidence_result.metric_count
                    if visual_evidence_result is not None
                    else 0
                ),
                "visual_evidence_status": (
                    visual_evidence_result.status.value
                    if visual_evidence_result is not None
                    else None
                ),
                "visual_cache_hit": (
                    visual_evidence_result.cache_hit
                    if visual_evidence_result is not None
                    else False
                ),
                "visual_fallback_error": visual_fallback_error,
                "category_generation_error": category_generation_error,
                "failed_question_details": failed_details,
            },
        )

    def run_group(
        self,
        question_group: QuestionGroup,
        *,
        report_id: str | None = None,
        company_name: str | None = None,
        report_year: int | None = None,
        top_k: int = 12,
        rerank_top_k: int = 6,
        answer_language: str = "tr",
        citation_style: CitationStyle = CitationStyle.COMPACT,
        strict_category_filter: bool = True,
        allow_category_filter_fallback: bool = True,
        temperature: float = 0.0,
        maximum_output_tokens: int = 1200,
        model: str | None = None,
        additional_instructions: Sequence[str] = (),
        continue_on_preparation_error: bool = True,
        enable_visual_fallback: bool = True,
        report_database_id: int | None = None,
        vision_model: str | None = None,
        force_visual_refresh: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CategoryQuestionAnsweringResult:
        """Kolay kullanım için bir `QuestionGroup` nesnesini çalıştırır."""

        return self.run(
            CategoryQuestionAnsweringRequest(
                question_group=question_group,
                report_id=report_id,
                company_name=company_name,
                report_year=report_year,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                answer_language=answer_language,
                citation_style=citation_style,
                strict_category_filter=strict_category_filter,
                allow_category_filter_fallback=(
                    allow_category_filter_fallback
                ),
                temperature=temperature,
                maximum_output_tokens=maximum_output_tokens,
                model=model,
                additional_instructions=tuple(additional_instructions),
                continue_on_preparation_error=continue_on_preparation_error,
                enable_visual_fallback=enable_visual_fallback,
                report_database_id=report_database_id,
                vision_model=vision_model,
                force_visual_refresh=force_visual_refresh,
                metadata=dict(metadata or {}),
            )
        )

    def _augment_prepared_questions(
        self,
        *,
        prepared_by_id: Mapping[int, PreparedQuestion],
        metrics_by_question: Mapping[int, tuple[VisualMetric, ...]],
        request: CategoryQuestionAnsweringRequest,
        run_id: str,
    ) -> dict[int, PreparedQuestion]:
        updated = dict(prepared_by_id)

        for question_id, metrics in metrics_by_question.items():
            prepared = updated.get(question_id)

            if prepared is None or not metrics:
                continue

            visual_evidences = tuple(
                self._visual_metric_evidence(
                    metric,
                    report_id=request.report_id,
                )
                for metric in metrics
            )

            combined_candidates = (
                *prepared.evidence_candidates,
                *visual_evidences,
            )

            validation = (
                self.question_answering_pipeline
                ._validate_evidences(
                    prepared.analysis,
                    combined_candidates,
                )
            )

            updated[question_id] = replace(
                prepared,
                evidence_validation=validation,
                evidence_candidates=combined_candidates,
                metadata={
                    **dict(prepared.metadata),
                    "category_run_id": run_id,
                    "visual_fallback_applied": True,
                    "visual_metric_count": len(metrics),
                    "visual_metric_ids": [
                        str(metric.id)
                        for metric in metrics
                    ],
                    "accepted_evidence_count": len(
                        validation.accepted_evidences
                    ),
                    "validation_status": validation.status.value,
                    "validation_score": validation.score,
                },
            )

        return updated

    @staticmethod
    def _visual_metric_evidence(
        metric: VisualMetric,
        *,
        report_id: str | None,
    ) -> _VisualMetricEvidence:
        value_text = str(metric.value)
        unit_text = f" {metric.unit}" if metric.unit else ""
        year_text = (
            f" | year={metric.year}"
            if metric.year is not None
            else ""
        )
        scope_text = (
            f" | scope={metric.scope}"
            if metric.scope
            else ""
        )

        content = (
            f"{metric.metric_label}: {value_text}{unit_text}"
            f"{year_text}{scope_text}. "
            f"Visual source text: {metric.source_text or metric.metric_label}"
        )

        return _VisualMetricEvidence(
            id=str(metric.id),
            content=content,
            report_id=report_id or str(metric.report_id),
            page_number=metric.page_number,
            score=metric.confidence,
            confidence=metric.confidence,
            source_type="visual_metric",
            metadata={
                "visual_id": str(metric.visual_id),
                "metric_id": str(metric.id),
                "metric_key": metric.metric_key,
                "metric_label": metric.metric_label,
                "value": metric.value,
                "unit": metric.unit,
                "year": metric.year,
                "scope": metric.scope,
                "category": metric.category,
                "topic": metric.topic,
                "question_ids": list(metric.question_ids),
                "extraction_method": metric.extraction_method.value,
                "is_visual_evidence": True,
            },
        )

    @staticmethod
    def _candidate_pages_by_question(
        prepared_questions: Sequence[PreparedQuestion],
        *,
        allowed_question_ids: set[int],
    ) -> Mapping[int, tuple[int, ...]]:
        result: dict[int, tuple[int, ...]] = {}

        for prepared in prepared_questions:
            question_id = prepared.question_id

            if (
                question_id is None
                or question_id not in allowed_question_ids
            ):
                continue

            pages: set[int] = set()

            for candidate in (
                *prepared.reranked_results,
                *prepared.evidence_candidates,
            ):
                for attribute in (
                    "page_number",
                    "source_page",
                    "page",
                ):
                    value = getattr(candidate, attribute, None)
                    if isinstance(value, int) and value > 0:
                        pages.add(value)

                page_numbers = getattr(
                    candidate,
                    "page_numbers",
                    (),
                )
                if isinstance(page_numbers, Sequence):
                    pages.update(
                        int(value)
                        for value in page_numbers
                        if isinstance(value, int) and value > 0
                    )

                metadata = getattr(candidate, "metadata", {}) or {}
                if isinstance(metadata, Mapping):
                    for key in (
                        "page_number",
                        "page_start",
                        "page_end",
                    ):
                        value = metadata.get(key)
                        if isinstance(value, int) and value > 0:
                            pages.add(value)

            result[question_id] = tuple(sorted(pages))

        return result

    @staticmethod
    def _resolve_run_id(request: CategoryQuestionAnsweringRequest) -> str:
        candidate = request.metadata.get("category_run_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return f"{request.question_group.name}-{uuid4().hex[:12]}"

    @staticmethod
    def _build_missing_generated_answer(
        definition: QuestionDefinition,
        generation_result: CategoryAnswerGenerationResult,
        *,
        run_id: str,
    ) -> GeneratedAnswer:
        error = (
            "Kategori model cevabında bu soru için geçerli bir kayıt "
            "bulunamadı."
        )
        return GeneratedAnswer(
            question=definition.question,
            answer=None,
            status=AnswerStatus.INVALID_RESPONSE,
            confidence=0.0,
            evidence_ids=(),
            raw_response=generation_result.raw_response,
            model=generation_result.model,
            finish_reason=generation_result.finish_reason,
            metadata={
                "question_id": definition.id,
                "question_key": definition.key,
                "question_group": definition.group,
                "category_run_id": run_id,
                "category_model_call": (
                    generation_result.model_call_count > 0
                ),
                "usage_scope": "category_shared",
                "error": error,
            },
        )

    @staticmethod
    def _build_failed_result(
        definition: QuestionDefinition,
        exc: Exception,
        *,
        phase: str,
        request: CategoryQuestionAnsweringRequest,
        run_id: str,
        prepared: PreparedQuestion | None = None,
        generated_answer: GeneratedAnswer | None = None,
    ) -> QuestionAnsweringResult:
        error = f"{type(exc).__name__}: {exc}"
        return QuestionAnsweringResult(
            question=definition.question,
            status=QuestionAnsweringStatus.FAILED,
            analysis=prepared.analysis if prepared is not None else None,
            evidence_validation=(
                prepared.evidence_validation if prepared is not None else None
            ),
            generated_answer=generated_answer,
            retrieval_result_count=(
                prepared.retrieval_result_count if prepared is not None else 0
            ),
            fused_result_count=(
                prepared.fused_result_count if prepared is not None else 0
            ),
            reranked_result_count=(
                prepared.reranked_result_count if prepared is not None else 0
            ),
            error=error,
            metadata={
                **dict(request.metadata),
                "question_id": definition.id,
                "question_key": definition.key,
                "question_group": definition.group,
                "category_run_id": run_id,
                "category_group": request.question_group.name,
                "failure_phase": phase,
            },
        )

    def _build_aborted_result(
        self,
        *,
        request: CategoryQuestionAnsweringRequest,
        run_id: str,
        results_by_id: Mapping[int, QuestionAnsweringResult],
        prepared_by_id: Mapping[int, PreparedQuestion],
        preparation_failed_ids: Sequence[int],
        error: str,
    ) -> CategoryQuestionAnsweringResult:
        group = request.question_group
        ordered_results: list[QuestionAnsweringResult] = []

        for definition in group.questions:
            existing = results_by_id.get(definition.id)
            if existing is not None:
                ordered_results.append(existing)
                continue

            ordered_results.append(
                self._build_failed_result(
                    definition,
                    CategoryQuestionAnsweringPipelineError(
                        "Kategori hazırlığı önceki hata nedeniyle durduruldu."
                    ),
                    phase="preparation_aborted",
                    request=request,
                    run_id=run_id,
                    prepared=prepared_by_id.get(definition.id),
                )
            )

        return CategoryQuestionAnsweringResult(
            group=group.name,
            status=CategoryQuestionAnsweringStatus.FAILED,
            question_results=tuple(ordered_results),
            prepared_questions=tuple(prepared_by_id.values()),
            preparation_failed_question_ids=tuple(preparation_failed_ids),
            error=error,
            metadata={
                **dict(request.metadata),
                "category_run_id": run_id,
                "category_group": group.name,
                "category_group_order": group.order,
                "aborted": True,
                "model_call_count": 0,
            },
        )

    @staticmethod
    def _resolve_status(
        results: Sequence[QuestionAnsweringResult],
    ) -> CategoryQuestionAnsweringStatus:
        if not results:
            return CategoryQuestionAnsweringStatus.FAILED

        completed = sum(
            item.status == QuestionAnsweringStatus.COMPLETED
            for item in results
        )
        insufficient = sum(
            item.status == QuestionAnsweringStatus.INSUFFICIENT_EVIDENCE
            for item in results
        )
        failed = sum(
            item.status == QuestionAnsweringStatus.FAILED
            for item in results
        )

        if completed == len(results):
            return CategoryQuestionAnsweringStatus.COMPLETED
        if insufficient == len(results):
            return CategoryQuestionAnsweringStatus.INSUFFICIENT_EVIDENCE
        if failed == len(results):
            return CategoryQuestionAnsweringStatus.FAILED
        return CategoryQuestionAnsweringStatus.PARTIAL

    @staticmethod
    def _resolve_group_error(
        results: Sequence[QuestionAnsweringResult],
        *,
        generation_result: CategoryAnswerGenerationResult | None,
    ) -> str | None:
        if generation_result is not None and generation_result.error:
            return generation_result.error

        failed_errors = tuple(
            item.error
            for item in results
            if item.status == QuestionAnsweringStatus.FAILED and item.error
        )
        if not failed_errors:
            return None
        if len(failed_errors) == 1:
            return failed_errors[0]
        return f"{len(failed_errors)} soru başarısız oldu. İlk hata: {failed_errors[0]}"


__all__ = [
    "CategoryQuestionAnsweringPipeline",
    "CategoryQuestionAnsweringPipelineError",
    "CategoryQuestionAnsweringRequest",
    "CategoryQuestionAnsweringResult",
    "CategoryQuestionAnsweringStatus",
]