from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from .answer_generator import GeneratedAnswer
from retrieval.evidence_validator import (
    EvidenceValidationResult,
    ValidatedEvidence,
)


class CitationStyle(StrEnum):
    COMPACT = "compact"
    DETAILED = "detailed"
    INLINE = "inline"
    FOOTNOTE = "footnote"


@dataclass(slots=True, frozen=True)
class Citation:
    citation_id: str
    evidence_id: str
    label: str
    text: str

    report_id: str | None = None
    page_number: int | None = None
    row_index: int | None = None
    section_title: str | None = None

    source_type: str | None = None
    score: float = 0.0

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CitationBuildResult:
    answer: str | None
    answer_with_citations: str | None

    citations: tuple[Citation, ...]
    missing_evidence_ids: tuple[str, ...] = ()

    citation_style: CitationStyle = CitationStyle.COMPACT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)


@dataclass(slots=True, frozen=True)
class CitationBuilderConfig:
    citation_style: CitationStyle = CitationStyle.COMPACT

    maximum_citation_count: int = 8
    maximum_excerpt_characters: int = 500

    include_report_id: bool = False
    include_page_number: bool = True
    include_row_index: bool = True
    include_section_title: bool = True
    include_score: bool = False

    append_citations_to_answer: bool = True
    add_citation_list: bool = True

    citation_prefix: str = "Kaynak"
    missing_page_label: str = "sayfa bilgisi yok"

    def __post_init__(self) -> None:
        if self.maximum_citation_count <= 0:
            raise ValueError(
                "maximum_citation_count pozitif olmalıdır.",
            )

        if self.maximum_excerpt_characters <= 0:
            raise ValueError(
                "maximum_excerpt_characters pozitif olmalıdır.",
            )

        if not self.citation_prefix.strip():
            raise ValueError(
                "citation_prefix boş olamaz.",
            )


class CitationBuilderError(RuntimeError):
    """Atıf oluşturma sırasında meydana gelen temel hata."""


class CitationBuilder:
    """
    Üretilen cevaptaki evidence_ids alanını doğrulanmış kanıtlarla
    eşleştirerek kullanıcıya gösterilebilir atıflar üretir.

    Sorumlulukları:
    - Cevapta kullanılan kanıt kimliklerini çözümlemek
    - Sayfa, bölüm ve tablo satırı bilgilerini atıfa eklemek
    - Kanıt metninden kısa bir alıntı özeti üretmek
    - Cevaba satır içi veya liste biçiminde atıf eklemek
    - Geçersiz ya da bulunamayan kanıt kimliklerini raporlamak
    """

    def __init__(
        self,
        *,
        config: CitationBuilderConfig | None = None,
    ) -> None:
        self.config = config or CitationBuilderConfig()

    def build(
        self,
        answer: GeneratedAnswer,
        evidence_validation: EvidenceValidationResult,
        *,
        style: CitationStyle | None = None,
    ) -> CitationBuildResult:
        resolved_style = style or self.config.citation_style

        evidence_map = {
            evidence.evidence_id: evidence
            for evidence in evidence_validation.accepted_evidences
        }

        requested_ids = self._resolve_requested_evidence_ids(
            answer,
            evidence_validation,
        )

        citations: list[Citation] = []
        missing_ids: list[str] = []

        for evidence_id in requested_ids:
            evidence = evidence_map.get(evidence_id)

            if evidence is None:
                missing_ids.append(evidence_id)
                continue

            citations.append(
                self._build_citation(
                    evidence=evidence,
                    index=len(citations) + 1,
                    style=resolved_style,
                ),
            )

            if len(citations) >= self.config.maximum_citation_count:
                break

        answer_with_citations = self._attach_citations(
            answer.answer,
            citations,
            style=resolved_style,
        )

        return CitationBuildResult(
            answer=answer.answer,
            answer_with_citations=answer_with_citations,
            citations=tuple(citations),
            missing_evidence_ids=tuple(sorted(set(missing_ids))),
            citation_style=resolved_style,
            metadata={
                "requested_evidence_count": len(requested_ids),
                "resolved_citation_count": len(citations),
                "missing_evidence_count": len(missing_ids),
                "answer_status": answer.status.value,
                "validation_status": (
                    evidence_validation.status.value
                ),
            },
        )

    def build_many(
        self,
        items: Iterable[
            tuple[
                GeneratedAnswer,
                EvidenceValidationResult,
            ]
        ],
        *,
        style: CitationStyle | None = None,
    ) -> list[CitationBuildResult]:
        results: list[CitationBuildResult] = []

        for answer, validation in items:
            results.append(
                self.build(
                    answer,
                    validation,
                    style=style,
                ),
            )

        return results

    def format_citation_list(
        self,
        citations: Sequence[Citation],
        *,
        style: CitationStyle | None = None,
    ) -> str:
        resolved_style = style or self.config.citation_style

        if not citations:
            return ""

        lines: list[str] = []

        for citation in citations:
            if resolved_style == CitationStyle.FOOTNOTE:
                lines.append(
                    f"[^{citation.citation_id}]: {citation.text}",
                )

            elif resolved_style == CitationStyle.INLINE:
                lines.append(
                    f"[{citation.citation_id}] {citation.text}",
                )

            else:
                lines.append(
                    f"- {citation.label}: {citation.text}",
                )

        return "\n".join(lines)

    def _build_citation(
        self,
        *,
        evidence: ValidatedEvidence,
        index: int,
        style: CitationStyle,
    ) -> Citation:
        citation_id = str(index)

        label = self._build_label(
            evidence=evidence,
            citation_id=citation_id,
            style=style,
        )

        excerpt = self._truncate_text(
            evidence.text,
            self.config.maximum_excerpt_characters,
        )

        location = self._build_location_text(evidence)

        if location and excerpt:
            citation_text = f"{location} — {excerpt}"
        elif location:
            citation_text = location
        else:
            citation_text = excerpt

        return Citation(
            citation_id=citation_id,
            evidence_id=evidence.evidence_id,
            label=label,
            text=citation_text,
            report_id=evidence.report_id,
            page_number=evidence.page_number,
            row_index=evidence.row_index,
            section_title=evidence.section_title,
            source_type=self._resolve_source_type(evidence),
            score=evidence.score,
            metadata={
                **dict(evidence.metadata),
                "matched_years": evidence.matched_years,
                "matched_metrics": evidence.matched_metrics,
                "matched_units": evidence.matched_units,
                "matched_standard_codes": (
                    evidence.matched_standard_codes
                ),
            },
        )

    def _build_label(
        self,
        *,
        evidence: ValidatedEvidence,
        citation_id: str,
        style: CitationStyle,
    ) -> str:
        if style == CitationStyle.FOOTNOTE:
            return f"[^{citation_id}]"

        if style == CitationStyle.INLINE:
            return f"[{citation_id}]"

        parts = [
            f"{self.config.citation_prefix} {citation_id}",
        ]

        if (
            self.config.include_page_number
            and evidence.page_number is not None
        ):
            parts.append(
                f"s. {evidence.page_number}",
            )

        if (
            self.config.include_row_index
            and evidence.row_index is not None
        ):
            parts.append(
                f"satır {evidence.row_index}",
            )

        if (
            self.config.include_section_title
            and evidence.section_title
            and style == CitationStyle.DETAILED
        ):
            parts.append(evidence.section_title)

        if (
            self.config.include_report_id
            and evidence.report_id
        ):
            parts.append(evidence.report_id)

        if self.config.include_score:
            parts.append(
                f"skor={evidence.score:.3f}",
            )

        return " | ".join(parts)

    def _build_location_text(
        self,
        evidence: ValidatedEvidence,
    ) -> str:
        parts: list[str] = []

        if (
            self.config.include_report_id
            and evidence.report_id
        ):
            parts.append(
                f"Rapor: {evidence.report_id}",
            )

        if self.config.include_page_number:
            if evidence.page_number is not None:
                parts.append(
                    f"Sayfa {evidence.page_number}",
                )
            elif self.config.missing_page_label:
                parts.append(
                    self.config.missing_page_label,
                )

        if (
            self.config.include_row_index
            and evidence.row_index is not None
        ):
            parts.append(
                f"Tablo satırı {evidence.row_index}",
            )

        if (
            self.config.include_section_title
            and evidence.section_title
        ):
            parts.append(
                f"Bölüm: {evidence.section_title}",
            )

        return ", ".join(parts)

    def _attach_citations(
        self,
        answer_text: str | None,
        citations: Sequence[Citation],
        *,
        style: CitationStyle,
    ) -> str | None:
        cleaned_answer = self._clean_text(answer_text)

        if not cleaned_answer:
            return None

        if not citations:
            return cleaned_answer

        if not self.config.append_citations_to_answer:
            return cleaned_answer

        if style == CitationStyle.INLINE:
            markers = " ".join(
                f"[{citation.citation_id}]"
                for citation in citations
            )

            return f"{cleaned_answer} {markers}".strip()

        if style == CitationStyle.FOOTNOTE:
            markers = " ".join(
                f"[^{citation.citation_id}]"
                for citation in citations
            )

            answer_with_markers = (
                f"{cleaned_answer} {markers}".strip()
            )

            if not self.config.add_citation_list:
                return answer_with_markers

            citation_list = self.format_citation_list(
                citations,
                style=style,
            )

            return (
                f"{answer_with_markers}\n\n"
                f"{citation_list}"
            )

        if not self.config.add_citation_list:
            labels = " ".join(
                f"[{citation.label}]"
                for citation in citations
            )

            return f"{cleaned_answer} {labels}".strip()

        citation_list = self.format_citation_list(
            citations,
            style=style,
        )

        return (
            f"{cleaned_answer}\n\n"
            f"Kaynaklar:\n{citation_list}"
        )

    def _resolve_requested_evidence_ids(
        self,
        answer: GeneratedAnswer,
        validation: EvidenceValidationResult,
    ) -> tuple[str, ...]:
        accepted_ids = tuple(
            evidence.evidence_id
            for evidence in validation.accepted_evidences
        )

        if not answer.evidence_ids:
            return accepted_ids[
                : self.config.maximum_citation_count
            ]

        seen: set[str] = set()
        ordered_ids: list[str] = []

        for evidence_id in answer.evidence_ids:
            cleaned_id = self._clean_text(evidence_id)

            if not cleaned_id:
                continue

            if cleaned_id in seen:
                continue

            seen.add(cleaned_id)
            ordered_ids.append(cleaned_id)

        return tuple(
            ordered_ids[
                : self.config.maximum_citation_count
            ],
        )

    @staticmethod
    def _resolve_source_type(
        evidence: ValidatedEvidence,
    ) -> str:
        result_level = evidence.metadata.get(
            "result_level",
        )

        if result_level in {
            "table",
            "row",
        }:
            return str(result_level)

        if evidence.row_index is not None:
            return "table_row"

        source_type = (
            evidence.metadata.get("source_type")
            or evidence.metadata.get("type")
        )

        if source_type is not None:
            cleaned = str(source_type).strip()

            if cleaned:
                return cleaned

        return "text"

    @staticmethod
    def _truncate_text(
        text: str,
        maximum_length: int,
    ) -> str:
        cleaned = CitationBuilder._clean_text(text)

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


__all__ = [
    "Citation",
    "CitationBuildResult",
    "CitationBuilder",
    "CitationBuilderConfig",
    "CitationBuilderError",
    "CitationStyle",
]