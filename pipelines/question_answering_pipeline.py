from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Mapping, Sequence

from answering.answer_generator import (
    AnswerGenerationRequest,
    AnswerGenerator,
    AnswerStatus,
    GeneratedAnswer,
)
from answering.citation_builder import (
    CitationBuildResult,
    CitationBuilder,
    CitationStyle,
)
from answering.question_analyzer import (
    QuestionAnalysis,
    QuestionAnalyzer,
    RetrievalStrategy,
)
from answering.question_catalog import QuestionDefinition
from retrieval.evidence_validator import (
    EvidenceValidationRequest,
    EvidenceValidationResult,
    EvidenceValidator,
)


class QuestionAnsweringStatus(StrEnum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class QuestionAnsweringRequest:
    question: str
    report_id: str | None = None
    company_name: str | None = None
    report_year: int | None = None
    category: str | None = None
    top_k: int = 15
    rerank_top_k: int = 10
    answer_language: str = "tr"
    citation_style: CitationStyle = CitationStyle.COMPACT
    question_definition: QuestionDefinition | None = None
    strict_category_filter: bool = False
    allow_category_filter_fallback: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned_question = self.question.strip()
        if not cleaned_question:
            raise ValueError("question boş olamaz.")
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k pozitif olmalıdır.")
        if not self.answer_language.strip():
            raise ValueError("answer_language boş olamaz.")

        definition = self.question_definition
        if definition is not None:
            if not isinstance(definition, QuestionDefinition):
                raise TypeError(
                    "question_definition bir QuestionDefinition olmalıdır."
                )
            if cleaned_question != definition.question.strip():
                raise ValueError(
                    "question ile question_definition.question aynı olmalıdır."
                )

    @classmethod
    def from_definition(
        cls,
        definition: QuestionDefinition,
        *,
        report_id: str | None = None,
        company_name: str | None = None,
        report_year: int | None = None,
        top_k: int = 15,
        rerank_top_k: int = 10,
        answer_language: str = "tr",
        citation_style: CitationStyle = CitationStyle.COMPACT,
        strict_category_filter: bool = False,
        allow_category_filter_fallback: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionAnsweringRequest:
        """Bir katalog sorusundan doğrulanmış pipeline isteği oluşturur."""

        return cls(
            question=definition.question,
            report_id=report_id,
            company_name=company_name,
            report_year=report_year,
            category=definition.esg_category,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            answer_language=answer_language,
            citation_style=citation_style,
            question_definition=definition,
            strict_category_filter=strict_category_filter,
            allow_category_filter_fallback=allow_category_filter_fallback,
            metadata={
                "question_id": definition.id,
                "question_key": definition.key,
                "question_group": definition.group,
                "question_topic": definition.topic,
                **dict(metadata or {}),
            },
        )


@dataclass(slots=True, frozen=True)
class PreparedQuestion:
    """
    Model çağrısından önce tamamlanan soru hazırlık sonucudur.

    Bu nesne; analiz, ayrı retrieval sonuçları, fusion, reranking, bağlam
    genişletme ve kanıt doğrulama çıktılarını taşır. Kategori pipeline'ı aynı
    nesneleri birleştirerek tek model çağrısı yapabilir.
    """

    request: QuestionAnsweringRequest
    analysis: QuestionAnalysis
    evidence_validation: EvidenceValidationResult

    retrieval_groups: Mapping[str, tuple[Any, ...]] = field(
        default_factory=dict
    )
    fused_results: tuple[Any, ...] = ()
    reranked_results: tuple[Any, ...] = ()
    evidence_candidates: tuple[Any, ...] = ()

    retrieval_result_count: int = 0
    fused_result_count: int = 0
    reranked_result_count: int = 0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def question(self) -> str:
        return self.analysis.original_question

    @property
    def definition(self) -> QuestionDefinition | None:
        return self.request.question_definition

    @property
    def question_id(self) -> int | None:
        return self.analysis.question_id

    @property
    def question_key(self) -> str | None:
        return self.analysis.question_key

    @property
    def question_group(self) -> str | None:
        return self.analysis.question_group

    @property
    def accepted_evidences(self) -> tuple[Any, ...]:
        return self.evidence_validation.accepted_evidences

    @property
    def is_evidence_sufficient(self) -> bool:
        return self.evidence_validation.is_sufficient


@dataclass(slots=True, frozen=True)
class QuestionAnsweringResult:
    question: str
    status: QuestionAnsweringStatus
    analysis: QuestionAnalysis | None = None
    evidence_validation: EvidenceValidationResult | None = None
    generated_answer: GeneratedAnswer | None = None
    citation_result: CitationBuildResult | None = None
    retrieval_result_count: int = 0
    fused_result_count: int = 0
    reranked_result_count: int = 0
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str | None:
        if self.citation_result is not None:
            return self.citation_result.answer_with_citations
        if self.generated_answer is not None:
            return self.generated_answer.answer
        return None


class QuestionAnsweringPipelineError(RuntimeError):
    """Soru cevaplama hattında oluşan temel hata."""


class QuestionPreparationError(QuestionAnsweringPipelineError):
    """Soru için retrieval ve kanıt hazırlama aşamasındaki temel hata."""


class QuestionAnsweringPipeline:
    """
    Analiz, retrieval, fusion, reranking, kanıt doğrulama ve cevap üretme hattı.

    Akış iki ayrı aşamaya bölünmüştür:
    1. `prepare_question`: Model çağrısı yapmadan kanıt paketini hazırlar.
    2. `finalize_prepared_question`: Üretilen cevabı atıflarla sonuç nesnesine
       dönüştürür.

    Mevcut `run` ve `answer` metotları geriye dönük uyumluluğu korur.
    """

    def __init__(
        self,
        *,
        question_analyzer: QuestionAnalyzer,
        table_retriever: Any,
        bm25_retriever: Any,
        vector_retriever: Any,
        standard_router: Any,
        result_fusion: Any,
        reranker: Any,
        context_expander: Any,
        evidence_validator: EvidenceValidator,
        answer_generator: AnswerGenerator,
        citation_builder: CitationBuilder,
    ) -> None:
        self.question_analyzer = question_analyzer
        self.table_retriever = table_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.standard_router = standard_router
        self.result_fusion = result_fusion
        self.reranker = reranker
        self.context_expander = context_expander
        self.evidence_validator = evidence_validator
        self.answer_generator = answer_generator
        self.citation_builder = citation_builder

    def run(self, request: QuestionAnsweringRequest) -> QuestionAnsweringResult:
        """Tek soru için hazırlık ve cevap üretme aşamalarını çalıştırır."""

        try:
            prepared = self.prepare_question(request)
            generated_answer = self.answer_generator.generate(
                AnswerGenerationRequest(
                    question_analysis=prepared.analysis,
                    evidence_validation=prepared.evidence_validation,
                    answer_language=request.answer_language,
                    metadata={
                        **dict(request.metadata),
                        "question_prepared": True,
                    },
                )
            )
            return self.finalize_prepared_question(
                prepared,
                generated_answer,
                citation_style=request.citation_style,
            )
        except Exception as exc:
            return self._failed_result(request, exc)

    def run_definition(
        self,
        definition: QuestionDefinition,
        *,
        report_id: str | None = None,
        company_name: str | None = None,
        report_year: int | None = None,
        top_k: int = 15,
        rerank_top_k: int = 10,
        answer_language: str = "tr",
        citation_style: CitationStyle = CitationStyle.COMPACT,
        strict_category_filter: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionAnsweringResult:
        """Katalogdaki tek bir soruyu mevcut tek-soru akışıyla yanıtlar."""

        return self.run(
            QuestionAnsweringRequest.from_definition(
                definition,
                report_id=report_id,
                company_name=company_name,
                report_year=report_year,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                answer_language=answer_language,
                citation_style=citation_style,
                strict_category_filter=strict_category_filter,
                metadata=metadata,
            )
        )

    def prepare_question(
        self,
        request: QuestionAnsweringRequest,
    ) -> PreparedQuestion:
        """
        Bir soru için model çağrısına kadar olan bütün adımları tamamlar.

        Bu metot LLM kullanmaz. Kategori tabanlı cevaplama pipeline'ı her soru
        için bu metodu ayrı ayrı çağırıp kanıtları daha sonra tek promptta
        birleştirebilir.
        """

        try:
            analysis = self._analyze_request(request)
            text_category_filter = (
                analysis.category if request.strict_category_filter else None
            )
            table_category_filter = text_category_filter
            table_topic_filter = (
                analysis.topic if request.strict_category_filter else None
            )

            retrieval_groups = self._retrieve(
                analysis,
                top_k=request.top_k,
                text_category_filter=text_category_filter,
                table_category_filter=table_category_filter,
                table_topic_filter=table_topic_filter,
            )

            table_filter_fallback_level = 0
            bm25_filter_fallback_used = False
            vector_filter_fallback_used = False

            if request.allow_category_filter_fallback:
                # Tablo kanalı ayrıntılı topic filtresiyle başlar. Topic
                # metadata'sı yoksa önce geniş ESG kategorisine, ardından
                # yalnızca bu kanal için filtresiz aramaya düşer.
                if (
                    not retrieval_groups["table"]
                    and table_topic_filter is not None
                ):
                    retry_groups = self._retrieve(
                        analysis,
                        top_k=request.top_k,
                        text_category_filter=text_category_filter,
                        table_category_filter=table_category_filter,
                        table_topic_filter=None,
                        include_text=False,
                        include_standard=False,
                    )
                    retrieval_groups["table"] = retry_groups["table"]
                    if retrieval_groups["table"]:
                        table_filter_fallback_level = 1

                if (
                    not retrieval_groups["table"]
                    and (
                        table_category_filter is not None
                        or table_topic_filter is not None
                    )
                ):
                    retry_groups = self._retrieve(
                        analysis,
                        top_k=request.top_k,
                        text_category_filter=text_category_filter,
                        table_category_filter=None,
                        table_topic_filter=None,
                        include_text=False,
                        include_standard=False,
                    )
                    retrieval_groups["table"] = retry_groups["table"]
                    if retrieval_groups["table"]:
                        table_filter_fallback_level = 2

                if (
                    not retrieval_groups["bm25"]
                    and text_category_filter is not None
                ):
                    retry_groups = self._retrieve(
                        analysis,
                        top_k=request.top_k,
                        text_category_filter=None,
                        table_category_filter=table_category_filter,
                        table_topic_filter=table_topic_filter,
                        include_table=False,
                        include_vector=False,
                        include_standard=False,
                    )
                    retrieval_groups["bm25"] = retry_groups["bm25"]
                    bm25_filter_fallback_used = bool(
                        retrieval_groups["bm25"]
                    )

                if (
                    not retrieval_groups["vector"]
                    and text_category_filter is not None
                ):
                    retry_groups = self._retrieve(
                        analysis,
                        top_k=request.top_k,
                        text_category_filter=None,
                        table_category_filter=table_category_filter,
                        table_topic_filter=table_topic_filter,
                        include_table=False,
                        include_bm25=False,
                        include_standard=False,
                    )
                    retrieval_groups["vector"] = retry_groups["vector"]
                    vector_filter_fallback_used = bool(
                        retrieval_groups["vector"]
                    )

            retrieval_count = sum(
                len(group) for group in retrieval_groups.values()
            )

            fused_results = tuple(
                self.result_fusion.fuse(
                    table_results=retrieval_groups["table"],
                    bm25_results=retrieval_groups["bm25"],
                    vector_results=retrieval_groups["vector"],
                    standard_results=retrieval_groups["standard"],
                    top_k=max(request.top_k, request.rerank_top_k * 2),
                )
                or ()
            )

            rerank_query = self._resolve_rerank_query(analysis)
            reranked_results = tuple(
                self.reranker.rerank(
                    rerank_query,
                    fused_results,
                    top_k=request.rerank_top_k,
                )
                or ()
            ) if fused_results else ()

            # Reranker doğru sayısal tablo satırlarını düşürebilir.
            # Aday sayısını artırmadan, en fazla iki güçlü gerçek veri
            # satırını düşük öncelikli rerank adaylarının yerine koru.
            reranked_results = self._preserve_quantitative_metric_rows(
                reranked_results,
                retrieval_groups["table"],
                analysis=analysis,
                maximum_candidates=request.rerank_top_k,
                maximum_preserved_rows=2,
            )

            expanded_contexts = tuple(
                self.context_expander.expand_many(
                    reranked_results,
                    maximum_candidates=request.rerank_top_k,
                )
                or ()
            )
            # Bağlam genişletme çıktıları yararlı olsa da özgün rerank
            # sonuçlarının yerini tamamen almamalıdır. Genişletilmiş bağlam
            # kimi retriever metadata alanlarını, tablo yapısını veya sayısal
            # hücre bilgisini taşımayabilir. Bu nedenle iki kaynağı birlikte
            # validator'a gönderiyoruz.
            evidence_candidates = self._merge_evidence_candidates(
                reranked_results,
                expanded_contexts,
                analysis=analysis,
                maximum_candidates=request.rerank_top_k,
            )

            validation = self._validate_evidences(
                analysis,
                evidence_candidates,
            )

            print(
                f"  Validator ayrıntısı: question_id={analysis.question_id}",
                flush=True,
            )

            for evidence in validation.evidences:
                issue_names = [
                    issue.issue_type.value
                    for issue in evidence.issues
                ]

                print(
                    "    "
                    f"id={evidence.evidence_id} "
                    f"status={evidence.status.value} "
                    f"score={evidence.score:.3f} "
                    f"row={evidence.row_index} "
                    f"structured={evidence.metadata.get('is_structured_evidence')} "
                    f"years={evidence.matched_years} "
                    f"metrics={evidence.matched_metrics} "
                    f"units={evidence.matched_units} "
                    f"values={evidence.numeric_values} "
                    f"issues={issue_names} "
                    f"text={evidence.text[:180]}",
                    flush=True,
                )

            frozen_groups = {
                name: tuple(values)
                for name, values in retrieval_groups.items()
            }

            return PreparedQuestion(
                request=request,
                analysis=analysis,
                evidence_validation=validation,
                retrieval_groups=frozen_groups,
                fused_results=fused_results,
                reranked_results=reranked_results,
                evidence_candidates=evidence_candidates,
                retrieval_result_count=retrieval_count,
                fused_result_count=len(fused_results),
                reranked_result_count=len(reranked_results),
                metadata={
                    **dict(request.metadata),
                    "retrieval_strategy": analysis.retrieval_strategy.value,
                    "retrieval_query": self._resolve_retrieval_query(
                        analysis
                    ),
                    "strict_category_filter": (
                        request.strict_category_filter
                    ),
                    "text_category_filter": text_category_filter,
                    "table_category_filter": table_category_filter,
                    "table_topic_filter": table_topic_filter,
                    "category_filter_fallback_allowed": (
                        request.allow_category_filter_fallback
                    ),
                    "table_filter_fallback_level": (
                        table_filter_fallback_level
                    ),
                    "bm25_filter_fallback_used": (
                        bm25_filter_fallback_used
                    ),
                    "vector_filter_fallback_used": (
                        vector_filter_fallback_used
                    ),
                    "evidence_candidate_count": len(
                        evidence_candidates
                    ),
                    "structured_evidence_candidate_count": sum(
                        1
                        for candidate in evidence_candidates
                        if self._is_table_candidate(candidate)
                    ),
                    "accepted_evidence_count": len(
                        validation.accepted_evidences
                    ),
                    "validation_status": validation.status.value,
                    "validation_score": validation.score,
                },
            )
        except Exception as exc:
            raise QuestionPreparationError(
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def prepare_definition(
        self,
        definition: QuestionDefinition,
        *,
        report_id: str | None = None,
        company_name: str | None = None,
        report_year: int | None = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        answer_language: str = "tr",
        citation_style: CitationStyle = CitationStyle.COMPACT,
        strict_category_filter: bool = False,
        allow_category_filter_fallback: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreparedQuestion:
        """Bir katalog sorusunu model çağrısı yapmadan hazırlar."""

        request = QuestionAnsweringRequest.from_definition(
            definition,
            report_id=report_id,
            company_name=company_name,
            report_year=report_year,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            answer_language=answer_language,
            citation_style=citation_style,
            strict_category_filter=strict_category_filter,
            allow_category_filter_fallback=allow_category_filter_fallback,
            metadata=metadata,
        )
        return self.prepare_question(request)

    def finalize_prepared_question(
        self,
        prepared: PreparedQuestion,
        generated_answer: GeneratedAnswer,
        *,
        citation_style: CitationStyle | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionAnsweringResult:
        """
        Hazırlanmış soruyu verilen model cevabıyla nihai sonuca dönüştürür.

        Yeni kategori cevap üreticisi her soru için bir `GeneratedAnswer`
        oluşturduğunda bu metodu kullanarak mevcut atıf ve sonuç yapısını
        yeniden kullanabilir.
        """

        if generated_answer.question.strip() != prepared.question.strip():
            raise QuestionAnsweringPipelineError(
                "GeneratedAnswer.question hazırlanmış soruyla eşleşmiyor."
            )

        citation_result = self.citation_builder.build(
            generated_answer,
            prepared.evidence_validation,
            style=citation_style or prepared.request.citation_style,
        )

        status, error = self._map_answer_status(generated_answer)

        return QuestionAnsweringResult(
            question=prepared.question,
            status=status,
            analysis=prepared.analysis,
            evidence_validation=prepared.evidence_validation,
            generated_answer=generated_answer,
            citation_result=citation_result,
            retrieval_result_count=prepared.retrieval_result_count,
            fused_result_count=prepared.fused_result_count,
            reranked_result_count=prepared.reranked_result_count,
            error=error,
            metadata={
                **dict(prepared.metadata),
                **dict(metadata or {}),
                "answer_status": generated_answer.status.value,
                "question_id": prepared.question_id,
                "question_key": prepared.question_key,
                "question_group": prepared.question_group,
            },
        )

    def answer(
        self,
        question: str,
        *,
        report_id: str | None = None,
        company_name: str | None = None,
        report_year: int | None = None,
    ) -> str:
        result = self.run(
            QuestionAnsweringRequest(
                question=question,
                report_id=report_id,
                company_name=company_name,
                report_year=report_year,
            )
        )
        if result.status == QuestionAnsweringStatus.FAILED:
            raise QuestionAnsweringPipelineError(
                result.error or "Soru cevaplama işlemi başarısız oldu."
            )
        return result.answer or "Bu soruyu yanıtlamak için yeterli kanıt bulunamadı."

    def _analyze_request(
        self,
        request: QuestionAnsweringRequest,
    ) -> QuestionAnalysis:
        definition = request.question_definition
        if definition is not None:
            return self.question_analyzer.analyze_definition(
                definition,
                company_name=request.company_name,
                report_id=request.report_id,
                report_year=request.report_year,
                category=request.category,
                metadata=request.metadata,
            )

        return self.question_analyzer.analyze(
            request.question,
            company_name=request.company_name,
            report_id=request.report_id,
            report_year=request.report_year,
            category=request.category,
            metadata=request.metadata,
        )

    def _retrieve(
        self,
        analysis: QuestionAnalysis,
        *,
        top_k: int,
        text_category_filter: str | None,
        table_category_filter: str | None,
        table_topic_filter: str | None,
        include_table: bool = True,
        include_bm25: bool = True,
        include_vector: bool = True,
        include_standard: bool = True,
        include_text: bool = True,
    ) -> dict[str, list[Any]]:
        query = self._resolve_retrieval_query(analysis)

        groups: dict[str, list[Any]] = {
            "table": [],
            "bm25": [],
            "vector": [],
            "standard": [],
        }
        strategy = analysis.retrieval_strategy

        preferred_sources = set(analysis.preferred_sources)
        allow_table = not preferred_sources or "table" in preferred_sources
        allow_text = not preferred_sources or "text" in preferred_sources
        allow_standard = (
            not preferred_sources or "standard" in preferred_sources
        )

        if include_table and allow_table and strategy in {
            RetrievalStrategy.TABLE_FIRST,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.STANDARD_FIRST,
        }:
            table_query = self._build_table_query(
                analysis,
                query=query,
                category_filter=table_category_filter,
                topic_filter=table_topic_filter,
            )
            groups["table"] = list(
                self.table_retriever.retrieve(
                    table_query,
                    top_k=top_k,
                    report_id=analysis.report_id,
                    category=table_category_filter,
                    topic=table_topic_filter,
                )
                or []
            )

        if include_text and allow_text and strategy in {
            RetrievalStrategy.TEXT_FIRST,
            RetrievalStrategy.TABLE_FIRST,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.STANDARD_FIRST,
        }:
            if include_bm25:
                self._ensure_bm25_index()
                groups["bm25"] = list(
                    self.bm25_retriever.retrieve(
                        query,
                        top_k=top_k,
                        report_id=analysis.report_id,
                        page_numbers=(),
                        category=text_category_filter,
                        standard_codes=analysis.standard_codes,
                    )
                    or []
                )
            if include_vector:
                groups["vector"] = list(
                    self.vector_retriever.retrieve(
                        query,
                        top_k=top_k,
                        report_id=analysis.report_id,
                        page_numbers=(),
                        category=text_category_filter,
                        standard_codes=analysis.standard_codes,
                    )
                    or []
                )

        if (
            include_standard
            and allow_standard
            and analysis.standard_codes
            and (
                strategy == RetrievalStrategy.STANDARD_FIRST
                or analysis.expects_standard_evidence
            )
        ):
            groups["standard"] = list(
                self.table_retriever.retrieve_by_standard(
                    analysis.standard_codes,
                    report_id=analysis.report_id,
                    top_k=top_k,
                )
                or []
            )

        print(
            "  Retrieval kaynak özeti: "
            f"query={query[:220]!r} "
            f"text_category_filter={text_category_filter!r} "
            f"table_category_filter={table_category_filter!r} "
            f"table_topic_filter={table_topic_filter!r} "
            f"table={len(groups['table'])} "
            f"bm25={len(groups['bm25'])} "
            f"vector={len(groups['vector'])} "
            f"standard={len(groups['standard'])}",
            flush=True,
        )

        for source_name, source_results in groups.items():
            for result_index, result in enumerate(
                source_results[:3],
                start=1,
            ):
                result_id = (
                    getattr(result, "table_id", None)
                    or getattr(result, "document_id", None)
                    or getattr(result, "id", None)
                )
                result_page = (
                    getattr(result, "source_page", None)
                    or getattr(result, "page_number", None)
                    or getattr(result, "page", None)
                )
                result_score = getattr(result, "score", None)
                nested_document = getattr(result, "document", None)
                result_text = str(
                    getattr(result, "matched_text", None)
                    or getattr(nested_document, "text", None)
                    or getattr(result, "text", None)
                    or ""
                ).replace("\n", " ").strip()

                print(
                    "    "
                    f"{source_name}[{result_index}] "
                    f"id={result_id} "
                    f"page={result_page} "
                    f"score={result_score} "
                    f"text={result_text[:260]}",
                    flush=True,
                )

        return groups

    def _build_table_query(
        self,
        analysis: QuestionAnalysis,
        *,
        query: str,
        category_filter: str | None,
        topic_filter: str | None,
    ) -> Any:
        """
        QuestionAnalysis içindeki metrik, yıl ve birim sinyallerini
        TableRetriever'a kayıpsız aktarır.
        """

        analyze_query = getattr(
            self.table_retriever,
            "analyze_query",
            None,
        )
        if not callable(analyze_query):
            return query

        metric_terms = self._select_table_metric_terms(analysis)

        try:
            return analyze_query(
                query,
                metric_terms=metric_terms,
                years=analysis.years,
                units=analysis.units,
                standard_codes=analysis.standard_codes,
                category=category_filter,
                topic=topic_filter,
                report_id=analysis.report_id,
                company_name=analysis.company_name,
                metadata={
                    **dict(analysis.metadata),
                    "question_id": analysis.question_id,
                    "question_key": analysis.question_key,
                    "question_group": analysis.question_group,
                    "structured_table_query": True,
                },
            )
        except TypeError:
            return query

    @staticmethod
    def _select_table_metric_terms(
        analysis: QuestionAnalysis,
        *,
        maximum_terms: int = 8,
    ) -> tuple[str, ...]:
        standard_pattern = re.compile(
            r"\\b(?:GRI|SASB|ESRS|CDP|TCFD|IFRS|ISSB)\\b"
            r"|\\b\\d{3}-\\d+\\b",
            re.IGNORECASE,
        )

        values = (
            *analysis.metric_names,
            *analysis.aliases,
        )

        selected: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = " ".join(str(value).split())
            normalized = cleaned.casefold()

            if (
                not cleaned
                or normalized in seen
                or standard_pattern.search(cleaned)
            ):
                continue

            word_count = len(cleaned.split())
            if word_count < 2 or word_count > 8:
                continue

            seen.add(normalized)
            selected.append(cleaned)

            if len(selected) >= maximum_terms:
                break

        return tuple(selected)

    @staticmethod
    def _resolve_retrieval_query(analysis: QuestionAnalysis) -> str:
        """
        Katalog sorularında QuestionAnalyzer tarafından üretilen kontrollü
        sorguyu önceliklendirir.

        QueryExpander çıktısı serbest sorularda yararlı olabilir; ancak katalog
        sorularında metrik alias'larını değiştirebilir veya sorguyu genel bir
        konu sorgusuna dönüştürebilir.
        """

        if analysis.retrieval_query.strip():
            return analysis.retrieval_query.strip()

        expanded = analysis.expanded_query
        if expanded is not None:
            primary_query = getattr(expanded, "primary_query", None)
            if isinstance(primary_query, str) and primary_query.strip():
                return primary_query.strip()

            expanded_queries = getattr(expanded, "expanded_queries", ())
            if expanded_queries:
                first_query = str(expanded_queries[0]).strip()
                if first_query:
                    return first_query

        return analysis.original_question.strip()

    @staticmethod
    def _resolve_rerank_query(analysis: QuestionAnalysis) -> str:
        """Reranker için soru niyetini ve katalog alias'larını korur."""

        if analysis.retrieval_query.strip():
            return analysis.retrieval_query.strip()
        return analysis.original_question.strip()

    @classmethod
    def _preserve_quantitative_metric_rows(
        cls,
        reranked_results: Sequence[Any],
        table_results: Sequence[Any],
        *,
        analysis: QuestionAnalysis,
        maximum_candidates: int,
        maximum_preserved_rows: int = 2,
    ) -> tuple[Any, ...]:
        """
        Doğru metrikle eşleşen gerçek sayısal tablo satırlarını korur.

        GRI içerik indeksi, sayfa yönlendirmesi ve genel tablo başlıkları
        korunmaz. Liste büyütülmez; düşük öncelikli rerank adayları güçlü
        sayısal satırlarla değiştirilir.
        """

        if maximum_candidates <= 0:
            return ()

        reranked = list(reranked_results[:maximum_candidates])

        direct_rows = [
            candidate
            for candidate in table_results
            if (
                cls._candidate_row_index(candidate) is not None
                and cls._candidate_is_quantitative_metric_row(
                    candidate,
                    analysis,
                )
            )
        ]
        direct_rows.sort(
            key=lambda candidate: (
                -cls._candidate_quantitative_value_count(candidate),
                -cls._candidate_score(candidate),
                cls._candidate_page(candidate)
                if cls._candidate_page(candidate) is not None
                else 10**9,
                cls._candidate_row_index(candidate)
                if cls._candidate_row_index(candidate) is not None
                else 10**9,
            )
        )

        preserved: list[Any] = []
        preserved_keys: set[str] = set()

        for candidate in direct_rows:
            key = cls._candidate_identity(candidate)
            if key in preserved_keys:
                continue

            preserved.append(candidate)
            preserved_keys.add(key)

            if len(preserved) >= maximum_preserved_rows:
                break

        if not preserved:
            return tuple(reranked)

        merged: list[Any] = []
        seen: set[str] = set()

        for candidate in preserved:
            key = cls._candidate_identity(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

        for candidate in reranked:
            key = cls._candidate_identity(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

            if len(merged) >= maximum_candidates:
                break

        return tuple(merged[:maximum_candidates])

    @classmethod
    def _candidate_is_quantitative_metric_row(
        cls,
        candidate: Any,
        analysis: QuestionAnalysis,
    ) -> bool:
        candidate_text = cls._candidate_text(candidate)
        normalized = " ".join(candidate_text.casefold().split())

        if not normalized:
            return False

        index_markers = (
            "gri content index",
            "page numbers",
            "annex 3",
            "disclosure",
            "environmental performance indicators, p.",
        )
        if any(marker in normalized for marker in index_markers):
            return False

        if not cls._candidate_matches_analysis_metric(
            normalized,
            analysis,
        ):
            return False

        return cls._candidate_quantitative_value_count(candidate) >= 2

    @classmethod
    def _candidate_matches_analysis_metric(
        cls,
        normalized_candidate_text: str,
        analysis: QuestionAnalysis,
    ) -> bool:
        query_text = " ".join(
            (
                analysis.original_question,
                analysis.retrieval_query,
                *analysis.metric_names,
                *analysis.aliases,
            )
        ).casefold()

        concept_aliases = {
            "nox": ("nox", "no x", "nitrogen oxides"),
            "sox": ("sox", "so x", "sulfur oxides", "sulphur oxides"),
            "voc": ("voc", "volatile organic compounds"),
            "pm": ("pm", "particulate matter"),
            "pop": ("pop", "persistent organic pollutants"),
            "hap": ("hap", "hazardous air pollutants"),
        }

        query_has_air_metric = False

        for aliases in concept_aliases.values():
            query_match = any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    query_text,
                )
                for alias in aliases
            )
            candidate_match = any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    normalized_candidate_text,
                )
                for alias in aliases
            )

            if query_match:
                query_has_air_metric = True

            if query_match and candidate_match:
                return True

        return not query_has_air_metric

    @classmethod
    def _candidate_quantitative_value_count(
        cls,
        candidate: Any,
    ) -> int:
        candidate_text = cls._candidate_text(candidate)
        values = re.findall(
            r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?",
            candidate_text,
        )

        count = 0

        for value in values:
            compact = value.replace(",", "").replace(".", "")

            if compact.isdigit() and len(compact) == 4:
                year = int(compact)
                if 1900 <= year <= 2100:
                    continue

            try:
                numeric = float(value.replace(",", "."))
            except ValueError:
                continue

            if numeric in {302.0, 305.0, 165.0}:
                continue

            count += 1

        return count

    @staticmethod
    def _candidate_text(candidate: Any) -> str:
        for attribute in (
            "matched_text",
            "text",
            "content",
        ):
            value = getattr(candidate, attribute, None)
            if value is not None and not callable(value):
                cleaned = str(value).strip()
                if cleaned:
                    return cleaned

        payload = getattr(candidate, "payload", None)
        if payload is not None and payload is not candidate:
            for attribute in (
                "matched_text",
                "text",
                "content",
            ):
                value = getattr(payload, attribute, None)
                if value is not None and not callable(value):
                    cleaned = str(value).strip()
                    if cleaned:
                        return cleaned

        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, Mapping):
            for key in (
                "matched_text",
                "text",
                "content",
            ):
                value = metadata.get(key)
                if value is not None:
                    cleaned = str(value).strip()
                    if cleaned:
                        return cleaned

        return ""

    @classmethod
    def _merge_evidence_candidates(
        cls,
        reranked_results: Sequence[Any],
        expanded_contexts: Sequence[Any],
        *,
        analysis: QuestionAnalysis,
        maximum_candidates: int,
    ) -> tuple[Any, ...]:
        """
        Rerank sonuçlarıyla genişletilmiş bağlamları kontrollü biçimde
        birleştirir.

        Sayısal veya tablo beklenen sorularda:
        - En güçlü tablo/tablo-satırı adayları önce korunur.
        - Rerank sonucu esas kaynak kabul edilir.
        - Genişletilmiş bağlam yalnızca boş kontenjan varsa eklenir.
        - Eski fusion adaylarının genişletme üzerinden listeyi yeniden
          doldurması engellenir.
        """

        if maximum_candidates <= 0:
            return ()

        reranked = tuple(reranked_results)
        expanded = tuple(expanded_contexts)

        expects_structured_evidence = bool(
            analysis.expects_numeric_answer
            or analysis.expects_table_evidence
        )

        selected: list[Any] = []
        seen: set[str] = set()

        def append_candidate(candidate: Any) -> None:
            if len(selected) >= maximum_candidates:
                return

            key = cls._candidate_identity(candidate)
            if key in seen:
                return

            seen.add(key)
            selected.append(candidate)

        if expects_structured_evidence:
            table_candidates = [
                candidate
                for candidate in reranked
                if cls._is_table_candidate(candidate)
            ]
            table_candidates.sort(
                key=lambda candidate: (
                    cls._candidate_row_index(candidate) is None,
                    -cls._candidate_score(candidate),
                    cls._candidate_page(candidate)
                    if cls._candidate_page(candidate) is not None
                    else 10**9,
                )
            )

            # Aynı tablonun tüm satırlarının bağlamı doldurmasını önle.
            table_counts: dict[str, int] = {}
            for candidate in table_candidates:
                base_id = cls._base_table_identity(candidate)
                count = table_counts.get(base_id, 0)
                if count >= 2:
                    continue

                append_candidate(candidate)
                table_counts[base_id] = count + 1

                # En fazla iki güçlü yapılandırılmış kanıtı önden koru.
                if sum(
                    1
                    for item in selected
                    if cls._is_table_candidate(item)
                ) >= 2:
                    break

        # Reranker çıktısı ana aday havuzudur.
        for candidate in reranked:
            append_candidate(candidate)

        # Genişletilmiş bağlam yalnızca rerank listesi yeterli değilse eklenir.
        # Böylece reranked=6 sonrasında candidates tekrar 12'ye çıkmaz.
        if len(selected) < maximum_candidates:
            for candidate in expanded:
                append_candidate(candidate)

        return tuple(selected[:maximum_candidates])

    @classmethod
    def _candidate_identity(cls, candidate: Any) -> str:
        candidate_id: Any = None

        for attribute in (
            "evidence_id",
            "candidate_id",
            "document_id",
            "chunk_id",
            "table_id",
            "context_id",
            "id",
            "identifier",
        ):
            value = getattr(candidate, attribute, None)
            if value is not None and not callable(value):
                candidate_id = value
                break

        metadata = getattr(candidate, "metadata", None)
        if candidate_id is None and isinstance(metadata, Mapping):
            for key in (
                "evidence_id",
                "candidate_id",
                "document_id",
                "chunk_id",
                "table_id",
                "context_id",
                "id",
                "identifier",
            ):
                value = metadata.get(key)
                if value is not None and not callable(value):
                    candidate_id = value
                    break

        if candidate_id is not None:
            row_index = cls._candidate_row_index(candidate)
            if row_index is not None:
                return f"value:{candidate_id}::row:{row_index}"
            return f"value:{candidate_id}"

        return f"object:{id(candidate)}"

    @classmethod
    def _is_table_candidate(cls, candidate: Any) -> bool:
        sources = getattr(candidate, "sources", ()) or ()
        normalized_sources = {
            str(getattr(source, "value", source)).casefold()
            for source in sources
        }
        if "table" in normalized_sources:
            return True

        if cls._candidate_row_index(candidate) is not None:
            return True

        candidate_id = cls._candidate_identity(candidate)
        if "page-" in candidate_id and "-table-" in candidate_id:
            return True

        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, Mapping):
            representative_source = metadata.get("representative_source")
            if (
                representative_source is not None
                and str(representative_source).casefold() == "table"
            ):
                return True

            source_type = metadata.get("source_type")
            if (
                source_type is not None
                and str(source_type).casefold() == "table"
            ):
                return True

        payload = getattr(candidate, "payload", None)
        if payload is not None and payload is not candidate:
            payload_id = cls._candidate_identity(payload)
            if "page-" in payload_id and "-table-" in payload_id:
                return True

        return False

    @staticmethod
    def _candidate_row_index(candidate: Any) -> int | None:
        value = getattr(candidate, "row_index", None)
        if value is None:
            metadata = getattr(candidate, "metadata", None)
            if isinstance(metadata, Mapping):
                value = metadata.get("row_index")

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _candidate_score(candidate: Any) -> float:
        for attribute in (
            "score",
            "rerank_score",
            "fusion_score",
            "retrieval_score",
            "original_score",
        ):
            value = getattr(candidate, attribute, None)
            if value is None or callable(value):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _candidate_page(candidate: Any) -> int | None:
        for attribute in (
            "page_number",
            "source_page",
            "page",
        ):
            value = getattr(candidate, attribute, None)
            if value is None or callable(value):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _base_table_identity(cls, candidate: Any) -> str:
        identity = cls._candidate_identity(candidate)
        return identity.split("::row:", 1)[0]


    def _ensure_bm25_index(self) -> None:
        count_method = getattr(self.bm25_retriever, "count", None)
        current_count = int(count_method()) if callable(count_method) else 0
        if current_count == 0:
            self.bm25_retriever.build_index()

    def _validate_evidences(
        self,
        analysis: QuestionAnalysis,
        evidences: Sequence[Any],
    ) -> EvidenceValidationResult:
        request = EvidenceValidationRequest(
            # Retrieval sorgusu alias ve genişletilmiş terimler içerebilir.
            # Validator kelime örtüşmesi de kullandığı için uzun retrieval
            # sorgusu doğru kanıtın relevance skorunu gereksiz düşürebilir.
            # Doğrulamada kullanıcının gerçek sorusunu temel al.
            query=analysis.original_question,
            report_id=analysis.report_id,
            company_name=analysis.company_name,
            years=analysis.years,
            metric_terms=analysis.metric_names,
            units=analysis.units,
            standard_codes=analysis.standard_codes,
            category=analysis.category,
            expects_numeric_answer=analysis.expects_numeric_answer,
            expects_table_evidence=analysis.expects_table_evidence,
            metadata={
                **dict(analysis.metadata),
                "question_id": analysis.question_id,
                "question_key": analysis.question_key,
                "question_group": analysis.question_group,
                "question_topic": analysis.topic,
            },
        )
        return self.evidence_validator.validate(request, evidences)

    @staticmethod
    def _map_answer_status(
        generated_answer: GeneratedAnswer,
    ) -> tuple[QuestionAnsweringStatus, str | None]:
        if generated_answer.is_answered:
            return QuestionAnsweringStatus.COMPLETED, None

        if generated_answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            return QuestionAnsweringStatus.INSUFFICIENT_EVIDENCE, None

        error = generated_answer.metadata.get("error")
        if not isinstance(error, str) or not error.strip():
            error = (
                "Cevap üretimi başarısız oldu: "
                f"{generated_answer.status.value}."
            )
        return QuestionAnsweringStatus.FAILED, error

    @staticmethod
    def _failed_result(
        request: QuestionAnsweringRequest,
        exc: Exception,
    ) -> QuestionAnsweringResult:
        definition = request.question_definition
        return QuestionAnsweringResult(
            question=request.question,
            status=QuestionAnsweringStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
            metadata={
                **dict(request.metadata),
                "question_id": definition.id if definition else None,
                "question_key": definition.key if definition else None,
                "question_group": definition.group if definition else None,
            },
        )


__all__ = [
    "PreparedQuestion",
    "QuestionAnsweringPipeline",
    "QuestionAnsweringPipelineError",
    "QuestionAnsweringRequest",
    "QuestionAnsweringResult",
    "QuestionAnsweringStatus",
    "QuestionPreparationError",
]