from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .answer_generator import (
    AnswerStatus,
    GeneratedAnswer,
)
from .llm_client import JsonGenerationClient
from retrieval.evidence_validator import ValidatedEvidence

if TYPE_CHECKING:
    from pipelines.question_answering_pipeline import PreparedQuestion


class CategoryAnswerStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERATION_FAILED = "generation_failed"
    INVALID_RESPONSE = "invalid_response"


@dataclass(slots=True, frozen=True)
class CategoryAnswerGeneratorConfig:
    """Kategori çağrısında kullanılacak bağlam ve doğrulama sınırları."""

    maximum_evidence_count_per_question: int = 3
    maximum_unique_evidence_count: int = 12
    maximum_evidence_characters: int = 850
    maximum_total_context_characters: int = 6500
    minimum_context_fragment_characters: int = 40

    minimum_evidence_score: float = 0.0
    minimum_confidence: float = 0.2
    maximum_answer_characters: int = 2500
    default_evidence_count_when_omitted: int = 2

    return_without_llm_when_all_insufficient: bool = True
    include_page_numbers: bool = True
    include_section_titles: bool = True
    include_evidence_scores: bool = False

    enable_json_repair_retry: bool = False
    maximum_json_repair_attempts: int = 1

    def __post_init__(self) -> None:
        positive_fields = {
            "maximum_evidence_count_per_question": (
                self.maximum_evidence_count_per_question
            ),
            "maximum_unique_evidence_count": self.maximum_unique_evidence_count,
            "maximum_evidence_characters": self.maximum_evidence_characters,
            "maximum_total_context_characters": (
                self.maximum_total_context_characters
            ),
            "minimum_context_fragment_characters": (
                self.minimum_context_fragment_characters
            ),
            "maximum_answer_characters": self.maximum_answer_characters,
            "default_evidence_count_when_omitted": (
                self.default_evidence_count_when_omitted
            ),
            "maximum_json_repair_attempts": (
                self.maximum_json_repair_attempts
            ),
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} pozitif olmalıdır.")

        if (
            self.minimum_context_fragment_characters
            > self.maximum_evidence_characters
        ):
            raise ValueError(
                "minimum_context_fragment_characters, "
                "maximum_evidence_characters değerini aşamaz."
            )

        for field_name, value in {
            "minimum_evidence_score": self.minimum_evidence_score,
            "minimum_confidence": self.minimum_confidence,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} 0 ile 1 arasında olmalıdır.")


@dataclass(slots=True, frozen=True)
class CategoryAnswerGenerationRequest:
    """Aynı soru grubunun tek model çağrısıyla cevaplanması için istek."""

    group: str
    prepared_questions: tuple[PreparedQuestion, ...]

    answer_language: str = "tr"
    temperature: float = 0.0
    maximum_output_tokens: int = 1200
    model: str | None = None

    include_evidence_ids: bool = True
    include_confidence: bool = True
    additional_instructions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned_group = self.group.strip()
        if not cleaned_group:
            raise ValueError("group boş olamaz.")
        if not self.prepared_questions:
            raise ValueError("prepared_questions boş olamaz.")
        if not self.answer_language.strip():
            raise ValueError("answer_language boş olamaz.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature 0 ile 2 arasında olmalıdır.")
        if self.maximum_output_tokens <= 0:
            raise ValueError("maximum_output_tokens pozitif olmalıdır.")

        seen_question_ids: set[int] = set()
        for prepared in self.prepared_questions:
            question_id = prepared.question_id
            if question_id is None:
                raise ValueError(
                    "Kategori üretimi yalnızca katalog sorularını kabul eder; "
                    "question_id bulunamadı."
                )
            if question_id in seen_question_ids:
                raise ValueError(
                    f"prepared_questions içinde tekrarlanan soru ID'si: "
                    f"{question_id}."
                )
            seen_question_ids.add(question_id)

            question_group = (prepared.question_group or "").strip()
            if question_group != cleaned_group:
                raise ValueError(
                    f"Soru {question_id} '{question_group}' grubunda; "
                    f"istek grubu '{cleaned_group}'."
                )


@dataclass(slots=True, frozen=True)
class CategoryAnswerGenerationResult:
    group: str
    status: CategoryAnswerStatus
    answers: tuple[GeneratedAnswer, ...]

    model_call_count: int
    prompted_question_ids: tuple[int, ...] = ()
    skipped_question_ids: tuple[int, ...] = ()
    missing_question_ids: tuple[int, ...] = ()
    unexpected_question_ids: tuple[int, ...] = ()

    raw_response: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def answer_by_question_id(self) -> Mapping[int, GeneratedAnswer]:
        result: dict[int, GeneratedAnswer] = {}
        for answer in self.answers:
            question_id = answer.metadata.get("question_id")
            if isinstance(question_id, int):
                result[question_id] = answer
        return result


@dataclass(slots=True, frozen=True)
class _CategoryEvidenceEntry:
    category_evidence_id: str
    evidence: ValidatedEvidence
    text: str


@dataclass(slots=True, frozen=True)
class _CategoryEvidenceBundle:
    entries: tuple[_CategoryEvidenceEntry, ...]
    allowed_ids_by_question: Mapping[int, tuple[str, ...]]
    original_id_by_question_and_category_id: Mapping[tuple[int, str], str]
    selected_evidences_by_question: Mapping[int, tuple[ValidatedEvidence, ...]]
    context_characters: int


class CategoryAnswerGeneratorError(RuntimeError):
    """Kategori cevap üretimi sırasında oluşan temel hata."""


class CategoryAnswerGenerator:
    """
    Bir soru kategorisini tek model çağrısıyla cevaplar.

    Her soru için retrieval ve kanıt doğrulama daha önce ayrı ayrı yapılmış
    olmalıdır. Bu sınıf yalnızca:
    - doğrulanmış kanıtları soru bazında seçer,
    - aynı kanıtı kategori bağlamında tekilleştirir,
    - her soruya kullanabileceği kanıt ID'lerini sınırlar,
    - kategori için tek JSON çağrısı yapar,
    - toplu cevabı ayrı `GeneratedAnswer` nesnelerine dönüştürür.
    """

    _CODE_BLOCK_PATTERN = re.compile(
        r"^```(?:json)?\s*(?P<content>.*?)\s*```$",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        llm_client: JsonGenerationClient,
        *,
        config: CategoryAnswerGeneratorConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or CategoryAnswerGeneratorConfig()

    def generate(
        self,
        request: CategoryAnswerGenerationRequest,
    ) -> CategoryAnswerGenerationResult:
        ordered_prepared = tuple(
            sorted(
                request.prepared_questions,
                key=lambda item: int(item.question_id or 0),
            )
        )

        bundle = self._build_evidence_bundle(ordered_prepared)
        prompted_questions = tuple(
            prepared
            for prepared in ordered_prepared
            if bundle.allowed_ids_by_question.get(
                int(prepared.question_id or 0)
            )
        )
        skipped_questions = tuple(
            prepared
            for prepared in ordered_prepared
            if not bundle.allowed_ids_by_question.get(
                int(prepared.question_id or 0)
            )
        )

        local_answers = {
            int(prepared.question_id): self._build_insufficient_answer(
                prepared,
                reason=(
                    "Doğrulanmış kanıt bulunamadı veya kategori bağlam "
                    "bütçesine kanıt eklenemedi."
                ),
                metadata=request.metadata,
            )
            for prepared in skipped_questions
        }

        if not prompted_questions:
            if not self.config.return_without_llm_when_all_insufficient:
                raise CategoryAnswerGeneratorError(
                    "Kategori içinde modele gönderilebilecek kanıtlı soru yok."
                )

            answers = tuple(
                local_answers[int(prepared.question_id)]
                for prepared in ordered_prepared
            )
            return CategoryAnswerGenerationResult(
                group=request.group,
                status=CategoryAnswerStatus.INSUFFICIENT_EVIDENCE,
                answers=answers,
                model_call_count=0,
                prompted_question_ids=(),
                skipped_question_ids=tuple(
                    int(item.question_id) for item in skipped_questions
                ),
                metadata={
                    **dict(request.metadata),
                    "unique_evidence_count": 0,
                    "context_characters": 0,
                },
            )

        messages = self._build_messages(
            request=request,
            prompted_questions=prompted_questions,
            bundle=bundle,
        )

        try:
            response = self.llm_client.generate_json(
                messages,
                temperature=request.temperature,
                max_tokens=request.maximum_output_tokens,
                model=request.model,
            )
        except Exception as exc:
            failure_answers = {
                int(prepared.question_id): self._build_generation_failed_answer(
                    prepared,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=request.metadata,
                )
                for prepared in prompted_questions
            }
            all_answers = {
                **local_answers,
                **failure_answers,
            }
            return CategoryAnswerGenerationResult(
                group=request.group,
                status=CategoryAnswerStatus.GENERATION_FAILED,
                answers=tuple(
                    all_answers[int(prepared.question_id)]
                    for prepared in ordered_prepared
                ),
                model_call_count=1,
                prompted_question_ids=tuple(
                    int(item.question_id) for item in prompted_questions
                ),
                skipped_question_ids=tuple(
                    int(item.question_id) for item in skipped_questions
                ),
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    **dict(request.metadata),
                    "unique_evidence_count": len(bundle.entries),
                    "context_characters": bundle.context_characters,
                },
            )

        parsed_result = self._parse_response(
            request=request,
            prompted_questions=prompted_questions,
            bundle=bundle,
            response=response,
        )

        model_call_count = 1

        if (
            parsed_result.status == CategoryAnswerStatus.INVALID_RESPONSE
            and self.config.enable_json_repair_retry
            and self.config.maximum_json_repair_attempts > 0
        ):
            repaired_result = self._retry_json_repair(
                request=request,
                prompted_questions=prompted_questions,
                bundle=bundle,
                invalid_response=parsed_result.raw_response or "",
                original_error=parsed_result.error,
            )

            if repaired_result is not None:
                parsed_result = repaired_result
                model_call_count += 1

        parsed_answers = {
            int(answer.metadata["question_id"]): answer
            for answer in parsed_result.answers
            if isinstance(answer.metadata.get("question_id"), int)
        }
        all_answers = {
            **local_answers,
            **parsed_answers,
        }
        ordered_answers = tuple(
            all_answers[int(prepared.question_id)]
            for prepared in ordered_prepared
        )

        final_status = parsed_result.status
        if skipped_questions and final_status == CategoryAnswerStatus.COMPLETED:
            final_status = CategoryAnswerStatus.PARTIAL

        return CategoryAnswerGenerationResult(
            group=request.group,
            status=final_status,
            answers=ordered_answers,
            model_call_count=model_call_count,
            prompted_question_ids=parsed_result.prompted_question_ids,
            skipped_question_ids=tuple(
                int(item.question_id) for item in skipped_questions
            ),
            missing_question_ids=parsed_result.missing_question_ids,
            unexpected_question_ids=parsed_result.unexpected_question_ids,
            raw_response=parsed_result.raw_response,
            model=parsed_result.model,
            finish_reason=parsed_result.finish_reason,
            input_tokens=parsed_result.input_tokens,
            output_tokens=parsed_result.output_tokens,
            total_tokens=parsed_result.total_tokens,
            error=parsed_result.error,
            metadata={
                **dict(parsed_result.metadata),
                "skipped_question_count": len(skipped_questions),
            },
        )

    def generate_group(
        self,
        group: str,
        prepared_questions: Sequence[PreparedQuestion],
        *,
        answer_language: str = "tr",
        temperature: float = 0.0,
        maximum_output_tokens: int = 1200,
        model: str | None = None,
        additional_instructions: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CategoryAnswerGenerationResult:
        return self.generate(
            CategoryAnswerGenerationRequest(
                group=group,
                prepared_questions=tuple(prepared_questions),
                answer_language=answer_language,
                temperature=temperature,
                maximum_output_tokens=maximum_output_tokens,
                model=model,
                additional_instructions=tuple(additional_instructions),
                metadata=dict(metadata or {}),
            )
        )

    def _build_messages(
        self,
        *,
        request: CategoryAnswerGenerationRequest,
        prompted_questions: Sequence[PreparedQuestion],
        bundle: _CategoryEvidenceBundle,
    ) -> tuple[dict[str, str], ...]:
        return (
            {
                "role": "system",
                "content": self._build_system_prompt(request),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(
                    request=request,
                    prompted_questions=prompted_questions,
                    bundle=bundle,
                ),
            },
        )

    def _build_system_prompt(
        self,
        request: CategoryAnswerGenerationRequest,
    ) -> str:
        instructions = [
            (
                "Sen sürdürülebilirlik ve ESG raporlarındaki sayısal "
                "göstergeleri kanıt-temelli olarak çıkaran bir bileşensin."
            ),
            (
                "Her soruyu yalnızca o sorunun allowed_evidence_ids "
                "alanında listelenen kanıtlarla cevapla."
            ),
            (
                "Bir başka soruya izin verilen kanıtı, mevcut soru için "
                "izin verilmemişse kullanma."
            ),
            (
                "Kanıtta açıkça bulunmayan sayı, oran, kapsam, yıl veya "
                "birim üretme ve tahmin yapma."
            ),
            (
                "Raporlama yılı ile baz yılı veya karşılaştırma yılını "
                "karıştırma; sorulan yılın değeri açık değilse "
                "insufficient_evidence döndür."
            ),
            (
                "Grup toplamı, ülke operasyonu, tesis kapsamı ve çalışan "
                "alt grupları gibi kapsamları birbirine karıştırma."
            ),
            (
                "Birim dönüştürme veya yeni hesaplama yapma; yalnızca "
                "kanıtta açıkça verilen değeri ve birimi kullan."
            ),
            (
                "Yapılandırılmış tablo kanıtında metrik adı, değer ve birim "
                "açıkça bulunuyorsa; kanıt kısmen geçerli olarak işaretlenmiş "
                "olsa bile soruyu answered veya partial olarak cevapla."
            ),
            (
                "insufficient_evidence yalnızca izin verilen kanıtlarda "
                "sorulan metriğin değeri gerçekten bulunmadığında kullanılmalıdır."
            ),
            (
                "Su tasarrufu, su geri dönüşümü/yeniden kullanımı, su çekimi, "
                "su tüketimi ve su yoğunluğu farklı metriklerdir; birbirlerinin "
                "yerine kullanma."
            ),
            (
                "Soru toplam miktar soruyorsa yalnızca yüzde/oran kanıtını "
                "toplam miktar cevabı olarak kullanma."
            ),
            (
                "Girdi içindeki her question_id için cevaplar dizisinde "
                "tam olarak bir nesne döndür."
            ),
            (
                "status alanında yalnızca answered, partial veya "
                "insufficient_evidence kullan."
            ),
            f"Cevap metinlerini '{request.answer_language}' dilinde üret.",
            "Yalnızca geçerli JSON döndür.",
            "JSON dışında açıklama veya Markdown kod bloğu üretme.",
        ]

        if request.include_evidence_ids:
            instructions.append(
                "Kullandığın kategori kanıt kimliklerini evidence_ids "
                "alanında listele."
            )
        if request.include_confidence:
            instructions.append(
                "confidence alanını 0 ile 1 arasında üret."
            )

        for instruction in request.additional_instructions:
            cleaned = self._clean_inline_text(instruction)
            if cleaned:
                instructions.append(cleaned)

        return "\n".join(
            f"{index}. {instruction}"
            for index, instruction in enumerate(instructions, start=1)
        )

    def _build_user_prompt(
        self,
        *,
        request: CategoryAnswerGenerationRequest,
        prompted_questions: Sequence[PreparedQuestion],
        bundle: _CategoryEvidenceBundle,
    ) -> str:
        questions_payload: list[dict[str, Any]] = []
        for prepared in prompted_questions:
            question_id = int(prepared.question_id)
            analysis = prepared.analysis
            questions_payload.append(
                {
                    "question_id": question_id,
                    "question_key": prepared.question_key,
                    "question": prepared.question,
                    "topic": analysis.topic,
                    "expected_answer_type": (
                        analysis.expected_answer_type.value
                    ),
                    "report_year": prepared.request.report_year,
                    "metric_names": list(analysis.metric_names),
                    "expected_units": list(analysis.units),
                    "allowed_evidence_ids": list(
                        bundle.allowed_ids_by_question.get(question_id, ())
                    ),
                }
            )

        evidences_payload = [
            self._evidence_payload(entry)
            for entry in bundle.entries
        ]

        output_schema = {
            "group": request.group,
            "answers": [
                {
                    "question_id": 0,
                    "status": (
                        "answered | partial | insufficient_evidence"
                    ),
                    "answer": None,
                    "value": None,
                    "unit": None,
                    "year": None,
                    "metric": None,
                    "scope": None,
                    "evidence_ids": [],
                    "confidence": 0.0,
                }
            ],
        }

        payload = {
            "group": request.group,
            "questions": questions_payload,
            "evidences": evidences_payload,
            "required_output_schema": output_schema,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _evidence_payload(
        self,
        entry: _CategoryEvidenceEntry,
    ) -> dict[str, Any]:
        evidence = entry.evidence
        payload: dict[str, Any] = {
            "evidence_id": entry.category_evidence_id,
            "text": entry.text,
        }
        if evidence.report_id:
            payload["report_id"] = evidence.report_id
        if self.config.include_page_numbers and evidence.page_number is not None:
            payload["page"] = evidence.page_number
        if (
            self.config.include_section_titles
            and evidence.section_title
        ):
            payload["section"] = evidence.section_title
        if evidence.row_index is not None:
            payload["row"] = evidence.row_index
        if self.config.include_evidence_scores:
            payload["score"] = round(evidence.score, 6)
        return payload

    def _build_evidence_bundle(
        self,
        prepared_questions: Sequence[PreparedQuestion],
    ) -> _CategoryEvidenceBundle:
        candidates_by_question: dict[int, tuple[ValidatedEvidence, ...]] = {}
        for prepared in prepared_questions:
            question_id = int(prepared.question_id)
            candidates_by_question[question_id] = self._select_question_evidences(
                prepared.accepted_evidences
            )

        global_entries: list[_CategoryEvidenceEntry] = []
        global_index_by_key: dict[tuple[Any, ...], int] = {}
        allowed_ids: dict[int, list[str]] = {
            int(item.question_id): [] for item in prepared_questions
        }
        original_ids: dict[tuple[int, str], str] = {}
        selected_by_question: dict[int, list[ValidatedEvidence]] = {
            int(item.question_id): [] for item in prepared_questions
        }
        context_characters = 0

        maximum_depth = max(
            (len(items) for items in candidates_by_question.values()),
            default=0,
        )

        for depth in range(maximum_depth):
            for prepared in prepared_questions:
                question_id = int(prepared.question_id)
                candidates = candidates_by_question[question_id]
                if depth >= len(candidates):
                    continue

                evidence = candidates[depth]
                key = self._evidence_deduplication_key(evidence)
                existing_index = global_index_by_key.get(key)

                if existing_index is not None:
                    entry = global_entries[existing_index]
                    category_id = entry.category_evidence_id
                    if category_id not in allowed_ids[question_id]:
                        allowed_ids[question_id].append(category_id)
                        original_ids[(question_id, category_id)] = (
                            evidence.evidence_id
                        )
                        selected_by_question[question_id].append(evidence)
                    continue

                if len(global_entries) >= self.config.maximum_unique_evidence_count:
                    continue

                remaining_context = (
                    self.config.maximum_total_context_characters
                    - context_characters
                )
                if remaining_context < self.config.minimum_context_fragment_characters:
                    continue

                maximum_text_length = min(
                    self.config.maximum_evidence_characters,
                    remaining_context,
                )
                text = self._truncate_context_text(
                    evidence.text,
                    maximum_text_length,
                )
                if len(text) < self.config.minimum_context_fragment_characters:
                    continue

                category_id = f"E{len(global_entries) + 1:03d}"
                entry = _CategoryEvidenceEntry(
                    category_evidence_id=category_id,
                    evidence=evidence,
                    text=text,
                )
                global_index_by_key[key] = len(global_entries)
                global_entries.append(entry)
                context_characters += len(text)

                allowed_ids[question_id].append(category_id)
                original_ids[(question_id, category_id)] = evidence.evidence_id
                selected_by_question[question_id].append(evidence)

        return _CategoryEvidenceBundle(
            entries=tuple(global_entries),
            allowed_ids_by_question={
                question_id: tuple(ids)
                for question_id, ids in allowed_ids.items()
            },
            original_id_by_question_and_category_id=dict(original_ids),
            selected_evidences_by_question={
                question_id: tuple(items)
                for question_id, items in selected_by_question.items()
            },
            context_characters=context_characters,
        )

    def _select_question_evidences(
        self,
        evidences: Sequence[ValidatedEvidence],
    ) -> tuple[ValidatedEvidence, ...]:
        deduplicated: dict[tuple[Any, ...], ValidatedEvidence] = {}
        for evidence in evidences:
            if evidence.score < self.config.minimum_evidence_score:
                continue
            if not self._clean_context_text(evidence.text):
                continue

            key = self._evidence_deduplication_key(evidence)
            existing = deduplicated.get(key)
            if existing is None or evidence.score > existing.score:
                deduplicated[key] = evidence

        ordered = sorted(
            deduplicated.values(),
            key=lambda evidence: (
                # Sayısal sorularda doğru tablo satırını, genel olarak yüksek
                # skorlu fakat ilgisiz metin parçalarının önüne al.
                not bool(evidence.metadata.get("is_structured_evidence")),
                not bool(evidence.matched_metrics),
                not bool(evidence.numeric_values),
                -len(evidence.matched_metrics),
                -evidence.score,
                evidence.page_number
                if evidence.page_number is not None
                else 10**9,
                evidence.evidence_id,
            ),
        )
        return tuple(
            ordered[: self.config.maximum_evidence_count_per_question]
        )

    def _retry_json_repair(
        self,
        *,
        request: CategoryAnswerGenerationRequest,
        prompted_questions: Sequence[PreparedQuestion],
        bundle: _CategoryEvidenceBundle,
        invalid_response: str,
        original_error: str | None,
    ) -> CategoryAnswerGenerationResult | None:
        """
        Geçersiz model çıktısını tek bir düşük maliyetli çağrıyla yalnızca
        JSON biçimine dönüştürmeyi dener.
        """

        repair_prompt = {
            "role": "user",
            "content": json.dumps(
                {
                    "task": (
                        "Aşağıdaki model çıktısını yalnızca geçerli JSON "
                        "biçimine dönüştür. Yeni bilgi ekleme, sayı üretme, "
                        "kanıt kimliklerini değiştirme ve açıklama yazma."
                    ),
                    "required_group": request.group,
                    "required_question_ids": [
                        int(item.question_id)
                        for item in prompted_questions
                    ],
                    "required_schema": {
                        "group": request.group,
                        "answers": [
                            {
                                "question_id": 0,
                                "status": (
                                    "answered | partial | "
                                    "insufficient_evidence"
                                ),
                                "answer": None,
                                "value": None,
                                "unit": None,
                                "year": None,
                                "metric": None,
                                "scope": None,
                                "evidence_ids": [],
                                "confidence": 0.0,
                            }
                        ],
                    },
                    "original_error": original_error,
                    "invalid_output": invalid_response,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }

        messages = (
            {
                "role": "system",
                "content": (
                    "Sen yalnızca bozuk JSON çıktısını geçerli JSON'a "
                    "dönüştüren bir biçim düzelticisin. Yalnızca JSON döndür."
                ),
            },
            repair_prompt,
        )

        try:
            response = self.llm_client.generate_json(
                messages,
                temperature=0.0,
                max_tokens=request.maximum_output_tokens,
                model=request.model,
            )
        except Exception:
            return None

        result = self._parse_response(
            request=request,
            prompted_questions=prompted_questions,
            bundle=bundle,
            response=response,
        )

        if result.status == CategoryAnswerStatus.INVALID_RESPONSE:
            return None

        return CategoryAnswerGenerationResult(
            group=result.group,
            status=result.status,
            answers=result.answers,
            model_call_count=2,
            prompted_question_ids=result.prompted_question_ids,
            skipped_question_ids=result.skipped_question_ids,
            missing_question_ids=result.missing_question_ids,
            unexpected_question_ids=result.unexpected_question_ids,
            raw_response=result.raw_response,
            model=result.model,
            finish_reason=result.finish_reason,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            error=result.error,
            metadata={
                **dict(result.metadata),
                "json_repair_applied": True,
                "original_parse_error": original_error,
            },
        )

    def _parse_response(
        self,
        *,
        request: CategoryAnswerGenerationRequest,
        prompted_questions: Sequence[PreparedQuestion],
        bundle: _CategoryEvidenceBundle,
        response: Any,
    ) -> CategoryAnswerGenerationResult:
        response_content = self._response_content(response)
        cleaned_response = self._clean_model_response(response_content)
        response_metadata = self._response_metadata(response)
        model = self._optional_string(getattr(response, "model", None))
        finish_reason = self._optional_string(
            getattr(response, "finish_reason", None)
        )
        input_tokens, output_tokens, total_tokens = self._response_usage(response)

        parsed, parse_error = self._parse_json_object(cleaned_response)
        if parse_error is not None:
            answers = tuple(
                self._build_invalid_response_answer(
                    prepared,
                    error=parse_error,
                    metadata={
                        **dict(request.metadata),
                        **response_metadata,
                    },
                    model=model,
                    finish_reason=finish_reason,
                )
                for prepared in prompted_questions
            )
            return CategoryAnswerGenerationResult(
                group=request.group,
                status=CategoryAnswerStatus.INVALID_RESPONSE,
                answers=answers,
                model_call_count=1,
                prompted_question_ids=tuple(
                    int(item.question_id) for item in prompted_questions
                ),
                missing_question_ids=tuple(
                    int(item.question_id) for item in prompted_questions
                ),
                raw_response=response_content,
                model=model,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                error=parse_error,
                metadata={
                    **dict(request.metadata),
                    **response_metadata,
                    "unique_evidence_count": len(bundle.entries),
                    "context_characters": bundle.context_characters,
                },
            )

        returned_group = self._optional_string(parsed.get("group"))
        if (
            returned_group is not None
            and self._normalize_text(returned_group)
            != self._normalize_text(request.group)
        ):
            error = (
                "Model cevabındaki grup istek grubuyla eşleşmiyor: "
                f"'{returned_group}' != '{request.group}'."
            )
            answers = tuple(
                self._build_invalid_response_answer(
                    prepared,
                    error=error,
                    metadata={
                        **dict(request.metadata),
                        **response_metadata,
                    },
                    model=model,
                    finish_reason=finish_reason,
                )
                for prepared in prompted_questions
            )
            return CategoryAnswerGenerationResult(
                group=request.group,
                status=CategoryAnswerStatus.INVALID_RESPONSE,
                answers=answers,
                model_call_count=1,
                prompted_question_ids=tuple(
                    int(item.question_id) for item in prompted_questions
                ),
                missing_question_ids=tuple(
                    int(item.question_id) for item in prompted_questions
                ),
                raw_response=response_content,
                model=model,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                error=error,
                metadata={
                    **dict(request.metadata),
                    **response_metadata,
                    "unique_evidence_count": len(bundle.entries),
                    "context_characters": bundle.context_characters,
                },
            )

        raw_answers = parsed.get("answers")
        if isinstance(raw_answers, Mapping):
            raw_answers = [
                {"question_id": key, **dict(value)}
                for key, value in raw_answers.items()
                if isinstance(value, Mapping)
            ]
        if not isinstance(raw_answers, list):
            raw_answers = []

        expected_by_id = {
            int(prepared.question_id): prepared
            for prepared in prompted_questions
        }
        items_by_id: dict[int, Mapping[str, Any]] = {}
        duplicate_ids: set[int] = set()
        unexpected_ids: set[int] = set()

        for item in raw_answers:
            if not isinstance(item, Mapping):
                continue
            question_id = self._optional_int(item.get("question_id"))
            if question_id is None:
                continue
            if question_id not in expected_by_id:
                unexpected_ids.add(question_id)
                continue
            if question_id in items_by_id:
                duplicate_ids.add(question_id)
                continue
            items_by_id[question_id] = item

        generated_answers: list[GeneratedAnswer] = []
        missing_ids: list[int] = []

        for question_id, prepared in expected_by_id.items():
            if question_id in duplicate_ids:
                generated_answers.append(
                    self._build_invalid_response_answer(
                        prepared,
                        error=(
                            "Kategori model cevabında aynı question_id birden "
                            "fazla kez döndürüldü."
                        ),
                        metadata={
                            **dict(request.metadata),
                            **response_metadata,
                        },
                        model=model,
                        finish_reason=finish_reason,
                    )
                )
                continue

            item = items_by_id.get(question_id)
            if item is None:
                missing_ids.append(question_id)
                deterministic = self._build_unambiguous_numeric_fallback(
                    prepared=prepared,
                    bundle=bundle,
                    request=request,
                    response_metadata=response_metadata,
                    model=model,
                    finish_reason=finish_reason,
                )
                if deterministic is not None:
                    generated_answers.append(deterministic)
                else:
                    generated_answers.append(
                        self._build_insufficient_answer(
                            prepared,
                            reason=(
                                "Kategori model cevabında bu soru için sonuç "
                                "bulunamadı; ek model çağrısı yapılmadı."
                            ),
                            metadata={
                                **dict(request.metadata),
                                **response_metadata,
                                "missing_from_category_response": True,
                            },
                        )
                    )
                continue

            generated_answers.append(
                self._parse_answer_item(
                    prepared=prepared,
                    item=item,
                    bundle=bundle,
                    request=request,
                    response_metadata=response_metadata,
                    model=model,
                    finish_reason=finish_reason,
                )
            )

        status = self._resolve_result_status(generated_answers)
        if missing_ids or duplicate_ids or unexpected_ids:
            if status == CategoryAnswerStatus.COMPLETED:
                status = CategoryAnswerStatus.PARTIAL

        return CategoryAnswerGenerationResult(
            group=request.group,
            status=status,
            answers=tuple(generated_answers),
            model_call_count=1,
            prompted_question_ids=tuple(expected_by_id),
            missing_question_ids=tuple(sorted(set(missing_ids) | duplicate_ids)),
            unexpected_question_ids=tuple(sorted(unexpected_ids)),
            raw_response=response_content,
            model=model,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            error=None,
            metadata={
                **dict(request.metadata),
                **response_metadata,
                "unique_evidence_count": len(bundle.entries),
                "context_characters": bundle.context_characters,
                "duplicate_question_ids": tuple(sorted(duplicate_ids)),
            },
        )

    def _build_unambiguous_numeric_fallback(
        self,
        *,
        prepared: PreparedQuestion,
        bundle: _CategoryEvidenceBundle,
        request: CategoryAnswerGenerationRequest,
        response_metadata: Mapping[str, Any],
        model: str | None,
        finish_reason: str | None,
    ) -> GeneratedAnswer | None:
        """
        Ek token harcamadan yalnızca tek ve açık sayısal değeri olan,
        metrikle eşleşmiş yapılandırılmış kanıttan cevap üretir.

        Birden fazla sayı bulunan satırlarda tahmin yapmaz.
        """
        question_id = int(prepared.question_id)
        candidates = tuple(
            evidence
            for evidence in bundle.selected_evidences_by_question.get(
                question_id,
                (),
            )
            if (
                evidence.metadata.get("is_structured_evidence")
                and evidence.matched_metrics
                and len(evidence.numeric_values) == 1
            )
        )
        if len(candidates) != 1:
            return None

        evidence = candidates[0]
        category_ids = tuple(
            bundle.allowed_ids_by_question.get(question_id, ())
        )
        category_id = next(
            (
                item
                for item in category_ids
                if bundle.original_id_by_question_and_category_id.get(
                    (question_id, item)
                ) == evidence.evidence_id
            ),
            None,
        )
        if category_id is None:
            return None

        raw_value = evidence.numeric_values[0]
        unit = (
            evidence.matched_units[0]
            if evidence.matched_units
            else None
        )
        year = (
            prepared.request.report_year
            if prepared.request.report_year is not None
            else None
        )
        metric = (
            evidence.matched_metrics[0]
            if evidence.matched_metrics
            else None
        )
        answer_text = " ".join(
            part
            for part in (
                metric,
                raw_value,
                unit,
            )
            if part
        ).strip()
        if not answer_text:
            return None

        return GeneratedAnswer(
            question=prepared.question,
            answer=answer_text,
            status=AnswerStatus.ANSWERED,
            value=raw_value,
            unit=unit,
            year=year,
            metric=metric,
            confidence=max(0.75, evidence.score),
            evidence_ids=(evidence.evidence_id,),
            structured_data={
                "question_id": question_id,
                "answer": answer_text,
                "status": AnswerStatus.ANSWERED.value,
                "value": raw_value,
                "unit": unit,
                "year": year,
                "metric": metric,
                "evidence_ids": [evidence.evidence_id],
                "category_evidence_ids": [category_id],
                "confidence": max(0.75, evidence.score),
            },
            model=model,
            finish_reason=finish_reason,
            metadata={
                **dict(request.metadata),
                **dict(response_metadata),
                "question_id": question_id,
                "question_key": prepared.question_key,
                "question_group": prepared.question_group,
                "category_model_call": True,
                "usage_scope": "category_shared",
                "deterministic_numeric_fallback": True,
                "missing_from_category_response": True,
                "category_evidence_ids": (category_id,),
                "validation_status": (
                    prepared.evidence_validation.status.value
                ),
                "validation_score": prepared.evidence_validation.score,
            },
        )

    def _parse_answer_item(
        self,
        *,
        prepared: PreparedQuestion,
        item: Mapping[str, Any],
        bundle: _CategoryEvidenceBundle,
        request: CategoryAnswerGenerationRequest,
        response_metadata: Mapping[str, Any],
        model: str | None,
        finish_reason: str | None,
    ) -> GeneratedAnswer:
        question_id = int(prepared.question_id)
        answer_text = self._optional_string(item.get("answer"))
        status = self._normalize_answer_status(
            item.get("status"),
            answer_text=answer_text,
            evidence_is_sufficient=prepared.is_evidence_sufficient,
        )
        confidence = self._normalize_confidence(
            item.get("confidence"),
            fallback=prepared.evidence_validation.score,
        )
        if (
            status == AnswerStatus.ANSWERED
            and confidence < self.config.minimum_confidence
        ):
            status = AnswerStatus.PARTIAL

        fallback_applied = False
        if status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            fallback = self._build_structured_numeric_fallback(
                prepared=prepared,
                bundle=bundle,
                request=request,
            )
            if fallback is not None:
                item = {**dict(item), **fallback}
                answer_text = self._optional_string(item.get("answer"))
                status = self._normalize_answer_status(
                    item.get("status"),
                    answer_text=answer_text,
                    evidence_is_sufficient=True,
                )
                confidence = self._normalize_confidence(
                    item.get("confidence"),
                    fallback=prepared.evidence_validation.score,
                )
                fallback_applied = True

        category_evidence_ids, original_evidence_ids, invalid_ids, defaulted = (
            self._normalize_answer_evidence_ids(
                question_id=question_id,
                value=item.get("evidence_ids"),
                bundle=bundle,
                answer_status=status,
            )
        )

        structured_data = dict(item)
        structured_data["evidence_ids"] = list(original_evidence_ids)
        structured_data["category_evidence_ids"] = list(category_evidence_ids)

        value = item.get("value")
        if value is None:
            value = item.get("target_value")
        if value is None:
            value = item.get("decision")

        return GeneratedAnswer(
            question=prepared.question,
            answer=(
                self._truncate_inline_text(
                    answer_text,
                    self.config.maximum_answer_characters,
                )
                if answer_text
                else None
            ),
            status=status,
            value=value,
            unit=self._optional_string(item.get("unit")),
            year=self._optional_int(item.get("year") or item.get("target_year")),
            metric=self._optional_string(item.get("metric")),
            confidence=confidence,
            evidence_ids=original_evidence_ids,
            raw_response=None,
            structured_data=structured_data,
            model=model,
            finish_reason=finish_reason,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            metadata={
                **dict(request.metadata),
                **dict(response_metadata),
                "question_id": question_id,
                "question_key": prepared.question_key,
                "question_group": prepared.question_group,
                "category_model_call": True,
                "usage_scope": "category_shared",
                "category_evidence_ids": category_evidence_ids,
                "invalid_category_evidence_ids": invalid_ids,
                "evidence_ids_defaulted": defaulted,
                "structured_numeric_fallback_applied": fallback_applied,
                "validation_status": (
                    prepared.evidence_validation.status.value
                ),
                "validation_score": prepared.evidence_validation.score,
            },
        )

    def _build_structured_numeric_fallback(
        self,
        *,
        prepared: PreparedQuestion,
        bundle: _CategoryEvidenceBundle,
        request: CategoryAnswerGenerationRequest,
    ) -> dict[str, Any] | None:
        """
        Model güçlü ve yapılandırılmış sayısal kanıta rağmen boş cevap
        döndürdüğünde sınırlı, deterministik bir geri dönüş üretir.

        Bu mekanizma hesaplama yapmaz; yalnızca açık tablo satırındaki
        metrik, değer ve birimi taşır. Yoğunluk, oran ve toplam miktar
        ayrımlarında muhafazakâr davranır.
        """
        question_id = int(prepared.question_id)
        question_text = self._normalize_text(prepared.question)
        report_year = prepared.request.report_year

        candidates = tuple(
            bundle.selected_evidences_by_question.get(question_id, ())
        )
        if not candidates:
            return None

        for evidence in candidates:
            metadata = dict(evidence.metadata or {})
            is_structured = bool(metadata.get("is_structured_evidence"))
            if not is_structured:
                continue
            if evidence.score < 0.55:
                continue
            if not evidence.numeric_values:
                continue
            if not evidence.matched_metrics:
                continue

            evidence_text = self._clean_inline_text(evidence.text)
            normalized_evidence = self._normalize_text(evidence_text)

            if not self._metric_family_matches(
                question_text=question_text,
                evidence_text=normalized_evidence,
            ):
                continue

            asks_amount = any(
                token in question_text
                for token in (
                    "toplam", "miktar", "amount", "total",
                )
            )
            asks_ratio = any(
                token in question_text
                for token in (
                    "oran", "yuzde", "percentage", "ratio", "rate",
                )
            )
            asks_intensity = any(
                token in question_text
                for token in (
                    "yogunluk", "intensity",
                )
            )

            percentage_only = self._evidence_is_percentage_only(
                evidence_text,
                evidence.numeric_values,
            )
            if asks_amount and not asks_ratio and percentage_only:
                continue
            if asks_intensity and not any(
                token in normalized_evidence
                for token in (
                    "intensity", "yogunluk", "per unit",
                    "per production", "per revenue",
                )
            ):
                continue

            extracted = self._extract_explicit_metric_value(
                evidence_text,
                report_year=report_year,
            )
            if extracted is None:
                continue

            value, unit = extracted
            metric = (
                evidence.matched_metrics[0]
                if evidence.matched_metrics
                else prepared.analysis.metric_names[0]
                if prepared.analysis.metric_names
                else prepared.question
            )

            category_id = None
            for candidate_id in bundle.allowed_ids_by_question.get(
                question_id, ()
            ):
                original_id = (
                    bundle.original_id_by_question_and_category_id.get(
                        (question_id, candidate_id)
                    )
                )
                if original_id == evidence.evidence_id:
                    category_id = candidate_id
                    break

            if category_id is None:
                continue

            year_text = f" ({report_year})" if report_year else ""
            answer = (
                f"Raporda {metric} değeri{year_text} "
                f"{value}{(' ' + unit) if unit else ''} olarak açıklanmıştır."
            )

            confidence = min(
                0.95,
                max(0.60, float(evidence.score)),
            )

            return {
                "status": "answered",
                "answer": answer,
                "value": value,
                "unit": unit,
                "year": report_year,
                "metric": metric,
                "evidence_ids": [category_id],
                "confidence": round(confidence, 6),
                "fallback_reason": (
                    "Model güçlü yapılandırılmış sayısal kanıta rağmen "
                    "insufficient_evidence döndürdü."
                ),
            }

        return None

    @classmethod
    def _metric_family_matches(
        cls,
        *,
        question_text: str,
        evidence_text: str,
    ) -> bool:
        families = (
            (
                ("water consumption", "su tuketim"),
                ("water consumption", "freshwater consumption", "su tuketim"),
            ),
            (
                ("water withdrawal", "su cekim"),
                ("water withdrawal", "su cekim"),
            ),
            (
                ("water recycled", "recycled water", "geri donustur", "yeniden kullan"),
                ("water recycled", "recycled and reused", "reused water",
                 "geri donustur", "yeniden kullan"),
            ),
            (
                ("water intensity", "su yogunluk"),
                ("water intensity", "su yogunluk", "per production", "per revenue"),
            ),
        )

        matched_family = False
        for question_terms, evidence_terms in families:
            if any(term in question_text for term in question_terms):
                matched_family = True
                return any(term in evidence_text for term in evidence_terms)

        return not matched_family

    @staticmethod
    def _evidence_is_percentage_only(
        text: str,
        numeric_values: Sequence[str],
    ) -> bool:
        has_percentage = "%" in text or "percent" in text.casefold()
        if not has_percentage:
            return False

        non_percentage_units = re.search(
            r"\b(?:m3|m³|litre|liter|ton|tonne|kg|mwh|gwh|kwh|gj)\b",
            text,
            re.IGNORECASE,
        )
        if non_percentage_units:
            return False

        return all("%" in value or value.strip().isdigit() for value in numeric_values)

    @classmethod
    def _extract_explicit_metric_value(
        cls,
        text: str,
        *,
        report_year: int | None,
    ) -> tuple[str, str | None] | None:
        cleaned = cls._clean_inline_text(text)

        unit_before_value = re.search(
            r"(?P<unit>m3|m³|litres?|liters?|tonnes?|tons?|kg|"
            r"MWh|GWh|kWh|GJ|%)"
            r"\s*(?:\||:|=|-)?\s*"
            r"(?P<value>[-+]?\d[\d.,\s]*%?)",
            cleaned,
            re.IGNORECASE,
        )
        if unit_before_value:
            value = unit_before_value.group("value").strip(" |")
            unit = unit_before_value.group("unit")
            if value.endswith("%"):
                unit = "%"
                value = value[:-1].strip()
            return value, unit

        value_before_unit = re.search(
            r"(?P<value>[-+]?\d[\d.,\s]*)\s*"
            r"(?P<unit>m3|m³|litres?|liters?|tonnes?|tons?|kg|"
            r"MWh|GWh|kWh|GJ|%)\b",
            cleaned,
            re.IGNORECASE,
        )
        if value_before_unit:
            return (
                value_before_unit.group("value").strip(" |"),
                value_before_unit.group("unit"),
            )

        percentages = re.findall(r"(\d+(?:[.,]\d+)?)\s*%", cleaned)
        if percentages:
            if report_year is not None and str(report_year) in cleaned:
                return percentages[-1], "%"
            if len(percentages) == 1:
                return percentages[0], "%"

        return None

    def _normalize_answer_evidence_ids(
        self,
        *,
        question_id: int,
        value: Any,
        bundle: _CategoryEvidenceBundle,
        answer_status: AnswerStatus,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool]:
        if answer_status not in {AnswerStatus.ANSWERED, AnswerStatus.PARTIAL}:
            return (), (), (), False

        allowed_category_ids = tuple(
            bundle.allowed_ids_by_question.get(question_id, ())
        )
        allowed_category_set = set(allowed_category_ids)
        original_to_category = {
            original_id: category_id
            for category_id in allowed_category_ids
            if (
                original_id := bundle.original_id_by_question_and_category_id.get(
                    (question_id, category_id)
                )
            )
        }

        provided = self._as_string_sequence(value)
        valid_category_ids: list[str] = []
        invalid_ids: list[str] = []

        for candidate in provided:
            if candidate in allowed_category_set:
                normalized = candidate
            elif candidate in original_to_category:
                normalized = original_to_category[candidate]
            else:
                invalid_ids.append(candidate)
                continue
            if normalized not in valid_category_ids:
                valid_category_ids.append(normalized)

        defaulted = False
        if (
            not valid_category_ids
            and answer_status in {AnswerStatus.ANSWERED, AnswerStatus.PARTIAL}
        ):
            valid_category_ids = list(
                allowed_category_ids[
                    : self.config.default_evidence_count_when_omitted
                ]
            )
            defaulted = True

        original_ids = tuple(
            bundle.original_id_by_question_and_category_id[
                (question_id, category_id)
            ]
            for category_id in valid_category_ids
            if (question_id, category_id)
            in bundle.original_id_by_question_and_category_id
        )

        return (
            tuple(valid_category_ids),
            tuple(dict.fromkeys(original_ids)),
            tuple(dict.fromkeys(invalid_ids)),
            defaulted,
        )

    def _build_insufficient_answer(
        self,
        prepared: PreparedQuestion,
        *,
        reason: str,
        metadata: Mapping[str, Any],
    ) -> GeneratedAnswer:
        answer = (
            "Bu soruyu yanıtlamak için raporda yeterli ve doğrulanmış "
            "kanıt bulunamadı."
        )
        return GeneratedAnswer(
            question=prepared.question,
            answer=answer,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            confidence=self._normalize_confidence(
                prepared.evidence_validation.score,
                fallback=0.0,
            ),
            evidence_ids=(),
            structured_data={
                "question_id": prepared.question_id,
                "answer": answer,
                "status": AnswerStatus.INSUFFICIENT_EVIDENCE.value,
                "evidence_ids": [],
                "confidence": prepared.evidence_validation.score,
            },
            metadata={
                **dict(metadata),
                "question_id": prepared.question_id,
                "question_key": prepared.question_key,
                "question_group": prepared.question_group,
                "category_model_call": False,
                "reason": reason,
                "validation_status": (
                    prepared.evidence_validation.status.value
                ),
                "validation_score": prepared.evidence_validation.score,
            },
        )

    def _build_generation_failed_answer(
        self,
        prepared: PreparedQuestion,
        *,
        error: str,
        metadata: Mapping[str, Any],
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            question=prepared.question,
            answer=None,
            status=AnswerStatus.GENERATION_FAILED,
            confidence=0.0,
            evidence_ids=(),
            metadata={
                **dict(metadata),
                "question_id": prepared.question_id,
                "question_key": prepared.question_key,
                "question_group": prepared.question_group,
                "category_model_call": True,
                "usage_scope": "category_shared",
                "error": error,
            },
        )

    def _build_invalid_response_answer(
        self,
        prepared: PreparedQuestion,
        *,
        error: str,
        metadata: Mapping[str, Any],
        model: str | None,
        finish_reason: str | None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            question=prepared.question,
            answer=None,
            status=AnswerStatus.INVALID_RESPONSE,
            confidence=0.0,
            evidence_ids=(),
            model=model,
            finish_reason=finish_reason,
            metadata={
                **dict(metadata),
                "question_id": prepared.question_id,
                "question_key": prepared.question_key,
                "question_group": prepared.question_group,
                "category_model_call": True,
                "usage_scope": "category_shared",
                "error": error,
            },
        )

    @staticmethod
    def _resolve_result_status(
        answers: Sequence[GeneratedAnswer],
    ) -> CategoryAnswerStatus:
        if not answers:
            return CategoryAnswerStatus.INVALID_RESPONSE

        statuses = {answer.status for answer in answers}
        if statuses <= {AnswerStatus.ANSWERED}:
            return CategoryAnswerStatus.COMPLETED
        if statuses <= {AnswerStatus.INSUFFICIENT_EVIDENCE}:
            return CategoryAnswerStatus.INSUFFICIENT_EVIDENCE
        if statuses <= {AnswerStatus.GENERATION_FAILED}:
            return CategoryAnswerStatus.GENERATION_FAILED
        if statuses <= {AnswerStatus.INVALID_RESPONSE}:
            return CategoryAnswerStatus.INVALID_RESPONSE
        return CategoryAnswerStatus.PARTIAL

    @staticmethod
    def _normalize_answer_status(
        value: Any,
        *,
        answer_text: str | None,
        evidence_is_sufficient: bool,
    ) -> AnswerStatus:
        normalized = str(value or "").strip().casefold()
        mapping = {
            "answered": AnswerStatus.ANSWERED,
            "partial": AnswerStatus.PARTIAL,
            "insufficient_evidence": AnswerStatus.INSUFFICIENT_EVIDENCE,
            "insufficient": AnswerStatus.INSUFFICIENT_EVIDENCE,
            "not_found": AnswerStatus.INSUFFICIENT_EVIDENCE,
        }
        if normalized in mapping:
            resolved = mapping[normalized]
            if (
                resolved in {AnswerStatus.ANSWERED, AnswerStatus.PARTIAL}
                and not answer_text
            ):
                return AnswerStatus.INSUFFICIENT_EVIDENCE
            return resolved
        if not answer_text:
            return AnswerStatus.INSUFFICIENT_EVIDENCE
        return (
            AnswerStatus.ANSWERED
            if evidence_is_sufficient
            else AnswerStatus.PARTIAL
        )

    @staticmethod
    def _normalize_confidence(value: Any, *, fallback: float) -> float:
        try:
            confidence = float(value) if value is not None else float(fallback)
        except (TypeError, ValueError):
            confidence = float(fallback)
        return round(max(0.0, min(confidence, 1.0)), 6)

    @classmethod
    def _parse_json_object(
        cls,
        text: str,
    ) -> tuple[Mapping[str, Any], str | None]:
        if not text:
            return {}, "Model boş cevap döndürdü."

        candidates = [text]

        extracted = cls._extract_json_object(text)
        if extracted is not None and extracted != text:
            candidates.append(extracted)

        last_error: Exception | None = None

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
            else:
                if isinstance(parsed, Mapping):
                    return parsed, None
                if isinstance(parsed, list):
                    return {"answers": parsed}, None

            # Bazı modeller Python sözlüğü biçiminde tek tırnaklı çıktı
            # üretebilir. Güvenli literal parser ile bunu da kabul et.
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError) as exc:
                last_error = exc
                continue

            if isinstance(parsed, Mapping):
                return dict(parsed), None
            if isinstance(parsed, list):
                return {"answers": parsed}, None

        detail = (
            f" ({type(last_error).__name__}: {last_error})"
            if last_error is not None
            else ""
        )
        return {}, f"Model JSON çıktısı ayrıştırılamadı{detail}."

    @classmethod
    def _clean_model_response(cls, content: str) -> str:
        cleaned = str(content or "").strip()
        match = cls._CODE_BLOCK_PATTERN.match(cleaned)
        if match:
            return match.group("content").strip()
        return cleaned

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return None
        return text[start_index : end_index + 1]

    @staticmethod
    def _response_content(response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, Mapping):
            return json.dumps(
                response,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        if isinstance(response, Sequence) and not isinstance(
            response,
            (str, bytes, bytearray),
        ):
            return json.dumps(
                list(response),
                ensure_ascii=False,
                separators=(",", ":"),
            )

        value = getattr(response, "content", None)
        if value is None:
            value = getattr(response, "text", None)

        if isinstance(value, Mapping):
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return json.dumps(
                list(value),
                ensure_ascii=False,
                separators=(",", ":"),
            )

        return str(value or "")

    @staticmethod
    def _response_metadata(response: Any) -> Mapping[str, Any]:
        value = getattr(response, "metadata", None)
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _response_usage(
        response: Any,
    ) -> tuple[int | None, int | None, int | None]:
        input_tokens = CategoryAnswerGenerator._optional_int(
            getattr(response, "input_tokens", None)
        )
        output_tokens = CategoryAnswerGenerator._optional_int(
            getattr(response, "output_tokens", None)
        )
        usage = getattr(response, "usage", None)
        total_tokens = CategoryAnswerGenerator._optional_int(
            getattr(usage, "total_tokens", None)
            if usage is not None
            else getattr(response, "total_tokens", None)
        )
        if total_tokens is None and (
            input_tokens is not None or output_tokens is not None
        ):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        return input_tokens, output_tokens, total_tokens

    @staticmethod
    def _evidence_deduplication_key(
        evidence: ValidatedEvidence,
    ) -> tuple[Any, ...]:
        return (
            evidence.report_id or "",
            evidence.page_number,
            evidence.row_index,
            CategoryAnswerGenerator._normalize_text(evidence.text),
        )

    @staticmethod
    def _clean_context_text(value: Any) -> str:
        if value is None:
            return ""
        raw = str(value).replace("\r\n", "\n").replace("\r", "\n")
        lines = [" ".join(line.split()) for line in raw.split("\n")]
        cleaned_lines: list[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line
            if is_blank and previous_blank:
                continue
            cleaned_lines.append(line)
            previous_blank = is_blank
        return "\n".join(cleaned_lines).strip()

    @classmethod
    def _truncate_context_text(cls, value: Any, maximum_length: int) -> str:
        cleaned = cls._clean_context_text(value)
        if len(cleaned) <= maximum_length:
            return cleaned
        if maximum_length <= 3:
            return cleaned[:maximum_length]
        return cleaned[: maximum_length - 3].rstrip() + "..."

    @staticmethod
    def _truncate_inline_text(value: str, maximum_length: int) -> str:
        cleaned = CategoryAnswerGenerator._clean_inline_text(value)
        if len(cleaned) <= maximum_length:
            return cleaned
        if maximum_length <= 3:
            return cleaned[:maximum_length]
        return cleaned[: maximum_length - 3].rstrip() + "..."

    @staticmethod
    def _clean_inline_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).replace("\n", " ").split())

    @staticmethod
    def _normalize_text(value: Any) -> str:
        cleaned = CategoryAnswerGenerator._clean_inline_text(value)
        if not cleaned:
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
            }
        )
        return cleaned.translate(translation_table).casefold()

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
            match = re.search(r"-?\d+", str(value))
            if match is None:
                return None
            try:
                return int(match.group(0))
            except ValueError:
                return None

    @staticmethod
    def _as_string_sequence(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            candidates = (value,)
        else:
            try:
                candidates = tuple(value)
            except TypeError:
                candidates = (value,)
        return tuple(
            dict.fromkeys(
                str(candidate).strip()
                for candidate in candidates
                if candidate is not None and str(candidate).strip()
            )
        )


__all__ = [
    "CategoryAnswerGenerationRequest",
    "CategoryAnswerGenerationResult",
    "CategoryAnswerGenerator",
    "CategoryAnswerGeneratorConfig",
    "CategoryAnswerGeneratorError",
    "CategoryAnswerStatus",
]