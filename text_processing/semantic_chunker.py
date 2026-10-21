from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from .section_detector import (
    DetectedSection,
    SectionContentBlock,
    SectionDetectionResult,
)


logger = logging.getLogger(__name__)


class SemanticChunkingError(RuntimeError):
    """Semantik chunk üretimi tamamlanamadığında yükseltilir."""


class ChunkType(StrEnum):
    """Üretilen chunk'ın içerik türü."""

    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE_GROUP = "sentence_group"
    LIST = "list"
    HEADING_CONTEXT = "heading_context"
    OVERSIZED_SENTENCE = "oversized_sentence"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class ChunkSplitMethod(StrEnum):
    """Chunk sınırının nasıl oluşturulduğunu belirtir."""

    SECTION_BOUNDARY = "section_boundary"
    BLOCK_BOUNDARY = "block_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SENTENCE_BOUNDARY = "sentence_boundary"
    TOKEN_FALLBACK = "token_fallback"
    NO_SPLIT = "no_split"


@dataclass(slots=True, frozen=True)
class ChunkSourceReference:
    """Chunk içindeki kaynak bloklardan birine ait referans."""

    page_index: int
    page_number: int

    block_index: int
    source_block_index: int | None

    start_character: int
    end_character: int

    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SemanticChunk:
    """Retrieval katmanına aktarılacak tek bir semantik metin parçası."""

    chunk_id: str
    document_id: str

    section_id: str
    parent_section_id: str | None

    chunk_index: int
    section_chunk_index: int

    previous_chunk_id: str | None
    next_chunk_id: str | None

    chunk_type: ChunkType
    split_method: ChunkSplitMethod

    section_level: int
    section_title: str
    section_path: tuple[str, ...]

    text: str
    normalized_text: str
    searchable_text: str

    contextualized_text: str

    start_page: int
    end_page: int

    source_references: tuple[ChunkSourceReference, ...]

    character_count: int
    word_count: int
    estimated_token_count: int

    overlap_character_count: int
    overlap_word_count: int
    overlap_token_count: int

    semantic_density: float
    chunk_confidence: float

    content_hash: str

    is_oversized: bool
    is_synthetic: bool

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "parent_section_id": (
                self.parent_section_id
            ),
            "chunk_index": self.chunk_index,
            "section_chunk_index": (
                self.section_chunk_index
            ),
            "previous_chunk_id": (
                self.previous_chunk_id
            ),
            "next_chunk_id": self.next_chunk_id,
            "chunk_type": self.chunk_type.value,
            "split_method": self.split_method.value,
            "section_level": self.section_level,
            "section_title": self.section_title,
            "section_path": list(
                self.section_path
            ),
            "text": self.text,
            "normalized_text": (
                self.normalized_text
            ),
            "searchable_text": (
                self.searchable_text
            ),
            "contextualized_text": (
                self.contextualized_text
            ),
            "start_page": self.start_page,
            "end_page": self.end_page,
            "source_references": [
                reference.to_dict()
                for reference
                in self.source_references
            ],
            "character_count": (
                self.character_count
            ),
            "word_count": self.word_count,
            "estimated_token_count": (
                self.estimated_token_count
            ),
            "overlap_character_count": (
                self.overlap_character_count
            ),
            "overlap_word_count": (
                self.overlap_word_count
            ),
            "overlap_token_count": (
                self.overlap_token_count
            ),
            "semantic_density": (
                self.semantic_density
            ),
            "chunk_confidence": (
                self.chunk_confidence
            ),
            "content_hash": self.content_hash,
            "is_oversized": self.is_oversized,
            "is_synthetic": self.is_synthetic,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class SectionChunkingResult:
    """Tek bir bölümün chunking sonucu."""

    section_id: str
    section_title: str

    input_character_count: int
    input_word_count: int
    input_estimated_token_count: int

    chunk_count: int
    oversized_chunk_count: int

    chunks: tuple[SemanticChunk, ...]

    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "input_character_count": (
                self.input_character_count
            ),
            "input_word_count": (
                self.input_word_count
            ),
            "input_estimated_token_count": (
                self.input_estimated_token_count
            ),
            "chunk_count": self.chunk_count,
            "oversized_chunk_count": (
                self.oversized_chunk_count
            ),
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True, frozen=True)
class SemanticChunkingResult:
    """Belgenin bütün semantik chunking sonucu."""

    document_id: str

    section_count: int
    processed_section_count: int
    skipped_section_count: int

    chunk_count: int
    oversized_chunk_count: int

    total_character_count: int
    total_word_count: int
    total_estimated_token_count: int

    chunks: tuple[SemanticChunk, ...]
    section_results: tuple[
        SectionChunkingResult,
        ...
    ]

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "section_count": self.section_count,
            "processed_section_count": (
                self.processed_section_count
            ),
            "skipped_section_count": (
                self.skipped_section_count
            ),
            "chunk_count": self.chunk_count,
            "oversized_chunk_count": (
                self.oversized_chunk_count
            ),
            "total_character_count": (
                self.total_character_count
            ),
            "total_word_count": (
                self.total_word_count
            ),
            "total_estimated_token_count": (
                self.total_estimated_token_count
            ),
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "section_results": [
                result.to_dict()
                for result in self.section_results
            ],
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class _TextUnit:
    """Chunk oluşturulurken kullanılan iç metin birimi."""

    text: str
    normalized_text: str

    page_number: int
    page_index: int

    block_index: int
    source_block_index: int | None

    unit_type: str

    source_hash: str

    estimated_token_count: int
    word_count: int
    character_count: int


@dataclass(slots=True)
class _ChunkDraft:
    """SemanticChunk oluşturulmadan önce kullanılan geçici temsil."""

    section: DetectedSection
    section_path: tuple[str, ...]

    text_units: list[_TextUnit]

    text: str
    normalized_text: str

    chunk_type: ChunkType
    split_method: ChunkSplitMethod

    overlap_text: str
    overlap_token_count: int

    warnings: list[str]


class SemanticChunker:
    """
    Bölüm içeriklerini anlam ve token sınırlarını koruyarak parçalar.

    Chunking öncelik sırası:

    1. Bölüm sınırları korunur.
    2. Blok ve paragraf sınırları korunur.
    3. Uzun paragraflar cümle sınırlarından ayrılır.
    4. Tek bir cümle token sınırını aşıyorsa sözcük tabanlı fallback
       uygulanır.
    5. Ardışık chunk'lar arasında kontrollü overlap eklenir.

    Üretilen chunk'lar daha sonra:

    - category_classifier.py
    - metadata_enricher.py
    - chunk_repository.py
    - BM25 ve embedding indeksleri

    tarafından kullanılır.
    """

    _WHITESPACE_RE = re.compile(
        r"[ \t\f\v]+"
    )

    _MULTIPLE_NEWLINE_RE = re.compile(
        r"\n{3,}"
    )

    _PARAGRAPH_SPLIT_RE = re.compile(
        r"\n{2,}"
    )

    _SENTENCE_SPLIT_RE = re.compile(
        r"""
        (?<=[.!?…])
        (?:
            ["')\]}»”’]*
        )
        \s+
        (?=
            ["'(\[{«“‘]*
            [A-ZÇĞİÖŞÜ0-9]
        )
        """,
        re.VERBOSE,
    )

    _LIST_ITEM_RE = re.compile(
        r"""
        ^\s*
        (?:
            -
            |
            •
            |
            \d+(?:\.\d+)*[.)]
            |
            [A-Za-zÇĞİÖŞÜçğıöşü][.)]
        )
        \s+
        """,
        re.VERBOSE,
    )

    _WORD_RE = re.compile(
        r"""
        \d+(?:[.,]\d+)*
        |
        [^\W\d_]+(?:[-'][^\W\d_]+)*
        |
        [^\s]
        """,
        re.UNICODE | re.VERBOSE,
    )

    _ABBREVIATIONS = {
        "dr.",
        "prof.",
        "doç.",
        "mr.",
        "mrs.",
        "ms.",
        "inc.",
        "ltd.",
        "co.",
        "no.",
        "fig.",
        "tab.",
        "örn.",
        "vb.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    }

    def __init__(
        self,
        *,
        target_tokens: int = 420,
        maximum_tokens: int = 560,
        minimum_tokens: int = 80,
        overlap_tokens: int = 60,
        hard_maximum_tokens: int = 750,
        characters_per_token: float = 3.8,
        include_section_title: bool = True,
        include_section_path: bool = True,
        include_parent_titles: bool = True,
        preserve_lists: bool = True,
        preserve_paragraphs: bool = True,
        include_empty_sections: bool = False,
        merge_small_final_chunk: bool = True,
        maximum_section_path_depth: int = 6,
    ) -> None:
        if target_tokens < 1:
            raise ValueError(
                "target_tokens pozitif olmalıdır."
            )

        if maximum_tokens < target_tokens:
            raise ValueError(
                "maximum_tokens, target_tokens değerinden "
                "küçük olamaz."
            )

        if minimum_tokens < 0:
            raise ValueError(
                "minimum_tokens negatif olamaz."
            )

        if overlap_tokens < 0:
            raise ValueError(
                "overlap_tokens negatif olamaz."
            )

        if overlap_tokens >= maximum_tokens:
            raise ValueError(
                "overlap_tokens, maximum_tokens değerinden "
                "küçük olmalıdır."
            )

        if hard_maximum_tokens < maximum_tokens:
            raise ValueError(
                "hard_maximum_tokens, maximum_tokens değerinden "
                "küçük olamaz."
            )

        if characters_per_token <= 0:
            raise ValueError(
                "characters_per_token pozitif olmalıdır."
            )

        if maximum_section_path_depth < 1:
            raise ValueError(
                "maximum_section_path_depth pozitif olmalıdır."
            )

        self.target_tokens = target_tokens
        self.maximum_tokens = maximum_tokens
        self.minimum_tokens = minimum_tokens
        self.overlap_tokens = overlap_tokens
        self.hard_maximum_tokens = (
            hard_maximum_tokens
        )

        self.characters_per_token = (
            characters_per_token
        )

        self.include_section_title = (
            include_section_title
        )

        self.include_section_path = (
            include_section_path
        )

        self.include_parent_titles = (
            include_parent_titles
        )

        self.preserve_lists = preserve_lists

        self.preserve_paragraphs = (
            preserve_paragraphs
        )

        self.include_empty_sections = (
            include_empty_sections
        )

        self.merge_small_final_chunk = (
            merge_small_final_chunk
        )

        self.maximum_section_path_depth = (
            maximum_section_path_depth
        )

    def chunk_document(
        self,
        section_result: SectionDetectionResult,
    ) -> SemanticChunkingResult:
        """Belgedeki bütün bölümleri semantik chunk'lara ayırır."""

        if not isinstance(
            section_result,
            SectionDetectionResult,
        ):
            raise TypeError(
                "section_result SectionDetectionResult olmalıdır."
            )

        sections_by_id = {
            section.section_id: section
            for section in section_result.sections
        }

        section_results: list[
            SectionChunkingResult
        ] = []

        all_chunks: list[
            SemanticChunk
        ] = []

        warnings: list[str] = []
        skipped_section_count = 0

        for section in section_result.sections:
            if (
                section.is_empty
                and not self.include_empty_sections
            ):
                skipped_section_count += 1
                continue

            try:
                section_path = self._build_section_path(
                    section=section,
                    sections_by_id=sections_by_id,
                )

                section_result_item = (
                    self.chunk_section(
                        section=section,
                        document_id=(
                            section_result.document_id
                        ),
                        section_path=section_path,
                        global_chunk_offset=len(
                            all_chunks
                        ),
                    )
                )

            except Exception as exc:
                skipped_section_count += 1

                message = (
                    f"Bölüm chunking işlemi başarısız: "
                    f"{section.title}: {exc}"
                )

                warnings.append(message)
                logger.warning(message)

                continue

            section_results.append(
                section_result_item
            )

            all_chunks.extend(
                section_result_item.chunks
            )

        all_chunks = self._refresh_global_neighbors(
            all_chunks
        )

        refreshed_section_results = (
            self._refresh_section_results(
                section_results=section_results,
                chunks=all_chunks,
            )
        )

        content_hash = self._calculate_result_hash(
            all_chunks
        )

        result = SemanticChunkingResult(
            document_id=section_result.document_id,
            section_count=section_result.section_count,
            processed_section_count=len(
                refreshed_section_results
            ),
            skipped_section_count=(
                skipped_section_count
            ),
            chunk_count=len(all_chunks),
            oversized_chunk_count=sum(
                chunk.is_oversized
                for chunk in all_chunks
            ),
            total_character_count=sum(
                chunk.character_count
                for chunk in all_chunks
            ),
            total_word_count=sum(
                chunk.word_count
                for chunk in all_chunks
            ),
            total_estimated_token_count=sum(
                chunk.estimated_token_count
                for chunk in all_chunks
            ),
            chunks=tuple(all_chunks),
            section_results=tuple(
                refreshed_section_results
            ),
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "source_section_hash": (
                    section_result.content_hash
                ),
                "target_tokens": self.target_tokens,
                "maximum_tokens": (
                    self.maximum_tokens
                ),
                "minimum_tokens": (
                    self.minimum_tokens
                ),
                "overlap_tokens": (
                    self.overlap_tokens
                ),
                "hard_maximum_tokens": (
                    self.hard_maximum_tokens
                ),
                "characters_per_token": (
                    self.characters_per_token
                ),
            },
        )

        logger.info(
            "%d bölümden %d semantik chunk üretildi.",
            result.processed_section_count,
            result.chunk_count,
        )

        return result

    def chunk_section(
        self,
        *,
        section: DetectedSection,
        document_id: str,
        section_path: Sequence[str] | None = None,
        global_chunk_offset: int = 0,
    ) -> SectionChunkingResult:
        """Tek bir bölümü semantik chunk'lara ayırır."""

        resolved_section_path = tuple(
            section_path
            or (
                section.title,
            )
        )

        units = self._build_text_units(
            section
        )

        input_text = (
            section.normalized_content
            or section.content
        )

        input_token_count = (
            self.estimate_token_count(
                input_text
            )
        )

        if not units:
            return SectionChunkingResult(
                section_id=section.section_id,
                section_title=section.title,
                input_character_count=len(
                    input_text
                ),
                input_word_count=len(
                    input_text.split()
                ),
                input_estimated_token_count=(
                    input_token_count
                ),
                chunk_count=0,
                oversized_chunk_count=0,
                chunks=(),
                warnings=(
                    "Bölümde chunk üretilecek metin birimi bulunamadı.",
                ),
            )

        drafts = self._build_chunk_drafts(
            section=section,
            section_path=(
                resolved_section_path
            ),
            units=units,
        )

        if self.merge_small_final_chunk:
            drafts = self._merge_small_final_draft(
                drafts
            )

        chunks = self._freeze_drafts(
            drafts=drafts,
            document_id=document_id,
            global_chunk_offset=(
                global_chunk_offset
            ),
        )

        chunks = self._assign_local_neighbors(
            chunks
        )

        warnings: list[str] = []

        oversized_count = sum(
            chunk.is_oversized
            for chunk in chunks
        )

        if oversized_count:
            warnings.append(
                f"{oversized_count} chunk maksimum token "
                "hedefini aştı."
            )

        return SectionChunkingResult(
            section_id=section.section_id,
            section_title=section.title,
            input_character_count=len(
                input_text
            ),
            input_word_count=len(
                input_text.split()
            ),
            input_estimated_token_count=(
                input_token_count
            ),
            chunk_count=len(chunks),
            oversized_chunk_count=(
                oversized_count
            ),
            chunks=tuple(chunks),
            warnings=tuple(warnings),
        )

    def _build_text_units(
        self,
        section: DetectedSection,
    ) -> list[_TextUnit]:
        units: list[_TextUnit] = []

        for block in section.blocks:
            block_text = (
                block.normalized_text
                or block.text
            ).strip()

            if not block_text:
                continue

            paragraphs = self._split_paragraphs(
                block_text
            )

            for paragraph in paragraphs:
                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                paragraph_tokens = (
                    self.estimate_token_count(
                        paragraph
                    )
                )

                if paragraph_tokens <= self.maximum_tokens:
                    units.append(
                        self._make_text_unit(
                            block=block,
                            text=paragraph,
                            unit_type=(
                                "list"
                                if self._is_list_text(
                                    paragraph
                                )
                                else "paragraph"
                            ),
                        )
                    )
                    continue

                sentence_units = (
                    self._split_long_paragraph(
                        block=block,
                        paragraph=paragraph,
                    )
                )

                units.extend(
                    sentence_units
                )

        return units

    def _split_long_paragraph(
        self,
        *,
        block: SectionContentBlock,
        paragraph: str,
    ) -> list[_TextUnit]:
        sentences = self._split_sentences(
            paragraph
        )

        if len(sentences) <= 1:
            return self._split_oversized_text(
                block=block,
                text=paragraph,
            )

        units: list[_TextUnit] = []

        for sentence in sentences:
            token_count = self.estimate_token_count(
                sentence
            )

            if token_count <= self.maximum_tokens:
                units.append(
                    self._make_text_unit(
                        block=block,
                        text=sentence,
                        unit_type="sentence",
                    )
                )
            else:
                units.extend(
                    self._split_oversized_text(
                        block=block,
                        text=sentence,
                    )
                )

        return units

    def _split_oversized_text(
        self,
        *,
        block: SectionContentBlock,
        text: str,
    ) -> list[_TextUnit]:
        words = text.split()

        if not words:
            return []

        units: list[_TextUnit] = []
        current_words: list[str] = []

        for word in words:
            candidate_words = [
                *current_words,
                word,
            ]

            candidate = " ".join(
                candidate_words
            )

            candidate_tokens = (
                self.estimate_token_count(
                    candidate
                )
            )

            if (
                current_words
                and candidate_tokens
                > self.hard_maximum_tokens
            ):
                units.append(
                    self._make_text_unit(
                        block=block,
                        text=" ".join(
                            current_words
                        ),
                        unit_type=(
                            "token_fallback"
                        ),
                    )
                )

                current_words = [
                    word
                ]

            else:
                current_words.append(
                    word
                )

        if current_words:
            units.append(
                self._make_text_unit(
                    block=block,
                    text=" ".join(
                        current_words
                    ),
                    unit_type="token_fallback",
                )
            )

        return units

    def _build_chunk_drafts(
        self,
        *,
        section: DetectedSection,
        section_path: tuple[str, ...],
        units: Sequence[_TextUnit],
    ) -> list[_ChunkDraft]:
        drafts: list[_ChunkDraft] = []

        current_units: list[_TextUnit] = []
        previous_units: list[_TextUnit] = []

        for unit in units:
            candidate_units = [
                *current_units,
                unit,
            ]

            candidate_text = self._join_units(
                candidate_units
            )

            candidate_tokens = (
                self.estimate_token_count(
                    candidate_text
                )
            )

            should_split = (
                bool(current_units)
                and candidate_tokens
                > self.maximum_tokens
            )

            if should_split:
                draft = self._create_draft(
                    section=section,
                    section_path=section_path,
                    text_units=current_units,
                    previous_units=previous_units,
                )

                drafts.append(draft)

                previous_units = list(
                    current_units
                )

                overlap_units = self._select_overlap_units(
                    previous_units
                )

                current_units = [
                    *overlap_units,
                    unit,
                ]

                if (
                    self.estimate_token_count(
                        self._join_units(
                            current_units
                        )
                    )
                    > self.hard_maximum_tokens
                ):
                    current_units = [
                        unit
                    ]

                continue

            current_units.append(
                unit
            )

            current_token_count = (
                self.estimate_token_count(
                    self._join_units(
                        current_units
                    )
                )
            )

            if (
                current_token_count
                >= self.target_tokens
            ):
                next_unit_exists = (
                    unit is not units[-1]
                )

                if next_unit_exists:
                    draft = self._create_draft(
                        section=section,
                        section_path=section_path,
                        text_units=(
                            current_units
                        ),
                        previous_units=(
                            previous_units
                        ),
                    )

                    drafts.append(draft)

                    previous_units = list(
                        current_units
                    )

                    current_units = (
                        self._select_overlap_units(
                            current_units
                        )
                    )

        if current_units:
            drafts.append(
                self._create_draft(
                    section=section,
                    section_path=section_path,
                    text_units=current_units,
                    previous_units=previous_units,
                )
            )

        return drafts

    def _create_draft(
        self,
        *,
        section: DetectedSection,
        section_path: tuple[str, ...],
        text_units: Sequence[_TextUnit],
        previous_units: Sequence[_TextUnit],
    ) -> _ChunkDraft:
        text = self._join_units(
            text_units
        )

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        overlap_units = self._detect_overlap_units(
            current_units=text_units,
            previous_units=previous_units,
        )

        overlap_text = self._join_units(
            overlap_units
        )

        chunk_type = self._detect_chunk_type(
            text_units
        )

        split_method = self._detect_split_method(
            text_units
        )

        warnings: list[str] = []

        token_count = self.estimate_token_count(
            normalized_text
        )

        if token_count > self.maximum_tokens:
            warnings.append(
                "Chunk hedef maksimum token sınırını aşıyor."
            )

        if token_count > self.hard_maximum_tokens:
            warnings.append(
                "Chunk kesin maksimum token sınırını aşıyor."
            )

        return _ChunkDraft(
            section=section,
            section_path=section_path,
            text_units=list(
                text_units
            ),
            text=text,
            normalized_text=(
                normalized_text
            ),
            chunk_type=chunk_type,
            split_method=split_method,
            overlap_text=overlap_text,
            overlap_token_count=(
                self.estimate_token_count(
                    overlap_text
                )
            ),
            warnings=warnings,
        )

    def _freeze_drafts(
        self,
        *,
        drafts: Sequence[_ChunkDraft],
        document_id: str,
        global_chunk_offset: int,
    ) -> list[SemanticChunk]:
        chunks: list[SemanticChunk] = []

        for section_chunk_index, draft in enumerate(
            drafts
        ):
            global_chunk_index = (
                global_chunk_offset
                + section_chunk_index
            )

            normalized_text = (
                draft.normalized_text
            )

            content_hash = hashlib.sha256(
                normalized_text.encode(
                    "utf-8"
                )
            ).hexdigest()

            chunk_key = (
                f"{document_id}:"
                f"{draft.section.section_id}:"
                f"{section_chunk_index}:"
                f"{content_hash}"
            )

            chunk_id = hashlib.sha256(
                chunk_key.encode(
                    "utf-8"
                )
            ).hexdigest()

            token_count = (
                self.estimate_token_count(
                    normalized_text
                )
            )

            source_references = (
                self._build_source_references(
                    draft.text_units
                )
            )

            start_page = min(
                (
                    unit.page_number
                    for unit in draft.text_units
                ),
                default=draft.section.start_page,
            )

            end_page = max(
                (
                    unit.page_number
                    for unit in draft.text_units
                ),
                default=draft.section.end_page,
            )

            contextualized_text = (
                self._build_contextualized_text(
                    section=draft.section,
                    section_path=(
                        draft.section_path
                    ),
                    text=normalized_text,
                )
            )

            chunk_confidence = (
                self._calculate_chunk_confidence(
                    section_confidence=(
                        draft.section.section_confidence
                    ),
                    token_count=token_count,
                    source_unit_count=len(
                        draft.text_units
                    ),
                    is_oversized=(
                        token_count
                        > self.maximum_tokens
                    ),
                    warning_count=len(
                        draft.warnings
                    ),
                )
            )

            semantic_density = (
                self._calculate_semantic_density(
                    normalized_text
                )
            )

            chunks.append(
                SemanticChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    section_id=(
                        draft.section.section_id
                    ),
                    parent_section_id=(
                        draft.section.parent_section_id
                    ),
                    chunk_index=(
                        global_chunk_index
                    ),
                    section_chunk_index=(
                        section_chunk_index
                    ),
                    previous_chunk_id=None,
                    next_chunk_id=None,
                    chunk_type=draft.chunk_type,
                    split_method=(
                        draft.split_method
                    ),
                    section_level=(
                        draft.section.level
                    ),
                    section_title=(
                        draft.section.title
                    ),
                    section_path=(
                        draft.section_path
                    ),
                    text=draft.text,
                    normalized_text=(
                        normalized_text
                    ),
                    searchable_text=(
                        self._normalize_for_search(
                            normalized_text
                        )
                    ),
                    contextualized_text=(
                        contextualized_text
                    ),
                    start_page=start_page,
                    end_page=end_page,
                    source_references=(
                        source_references
                    ),
                    character_count=len(
                        normalized_text
                    ),
                    word_count=len(
                        normalized_text.split()
                    ),
                    estimated_token_count=(
                        token_count
                    ),
                    overlap_character_count=len(
                        draft.overlap_text
                    ),
                    overlap_word_count=len(
                        draft.overlap_text.split()
                    ),
                    overlap_token_count=(
                        draft.overlap_token_count
                    ),
                    semantic_density=(
                        semantic_density
                    ),
                    chunk_confidence=(
                        chunk_confidence
                    ),
                    content_hash=content_hash,
                    is_oversized=(
                        token_count
                        > self.maximum_tokens
                    ),
                    is_synthetic=(
                        draft.section.is_synthetic
                    ),
                    warnings=tuple(
                        draft.warnings
                    ),
                    metadata={
                        "source_section_type": (
                            draft.section.section_type.value
                        ),
                        "source_unit_count": len(
                            draft.text_units
                        ),
                        "target_tokens": (
                            self.target_tokens
                        ),
                        "maximum_tokens": (
                            self.maximum_tokens
                        ),
                        "hard_maximum_tokens": (
                            self.hard_maximum_tokens
                        ),
                        "section_start_page": (
                            draft.section.start_page
                        ),
                        "section_end_page": (
                            draft.section.end_page
                        ),
                    },
                )
            )

        return chunks

    def _merge_small_final_draft(
        self,
        drafts: list[_ChunkDraft],
    ) -> list[_ChunkDraft]:
        if len(drafts) < 2:
            return drafts

        final_draft = drafts[-1]

        final_tokens = self.estimate_token_count(
            final_draft.normalized_text
        )

        if final_tokens >= self.minimum_tokens:
            return drafts

        previous_draft = drafts[-2]

        merged_units = [
            *previous_draft.text_units,
            *[
                unit
                for unit in final_draft.text_units
                if unit not in previous_draft.text_units
            ],
        ]

        merged_text = self._join_units(
            merged_units
        )

        merged_tokens = self.estimate_token_count(
            merged_text
        )

        if merged_tokens > self.hard_maximum_tokens:
            return drafts

        merged_draft = self._create_draft(
            section=previous_draft.section,
            section_path=(
                previous_draft.section_path
            ),
            text_units=merged_units,
            previous_units=(),
        )

        merged_draft.warnings.append(
            "Küçük son chunk önceki chunk ile birleştirildi."
        )

        return [
            *drafts[:-2],
            merged_draft,
        ]

    def _select_overlap_units(
        self,
        units: Sequence[_TextUnit],
    ) -> list[_TextUnit]:
        if self.overlap_tokens <= 0:
            return []

        selected: list[_TextUnit] = []
        token_total = 0

        for unit in reversed(
            units
        ):
            if (
                selected
                and token_total
                + unit.estimated_token_count
                > self.overlap_tokens
            ):
                break

            selected.append(
                unit
            )

            token_total += (
                unit.estimated_token_count
            )

            if token_total >= self.overlap_tokens:
                break

        selected.reverse()

        return selected

    @staticmethod
    def _detect_overlap_units(
        *,
        current_units: Sequence[_TextUnit],
        previous_units: Sequence[_TextUnit],
    ) -> list[_TextUnit]:
        if not current_units or not previous_units:
            return []

        previous_hashes = {
            unit.source_hash
            for unit in previous_units
        }

        overlap_units: list[_TextUnit] = []

        for unit in current_units:
            if unit.source_hash in previous_hashes:
                overlap_units.append(
                    unit
                )
            else:
                break

        return overlap_units

    def _build_contextualized_text(
        self,
        *,
        section: DetectedSection,
        section_path: Sequence[str],
        text: str,
    ) -> str:
        context_parts: list[str] = []

        if self.include_section_path:
            path = list(
                section_path[
                    -self.maximum_section_path_depth:
                ]
            )

            if not self.include_parent_titles:
                path = [
                    section.title
                ]

            if path:
                context_parts.append(
                    "Section path: "
                    + " > ".join(path)
                )

        elif self.include_section_title:
            context_parts.append(
                f"Section: {section.title}"
            )

        context_parts.append(
            text
        )

        return "\n\n".join(
            part
            for part in context_parts
            if part.strip()
        )

    def _build_section_path(
        self,
        *,
        section: DetectedSection,
        sections_by_id: Mapping[
            str,
            DetectedSection,
        ],
    ) -> tuple[str, ...]:
        titles: list[str] = [
            section.title
        ]

        parent_id = (
            section.parent_section_id
        )

        visited_ids = {
            section.section_id
        }

        while parent_id:
            if parent_id in visited_ids:
                break

            visited_ids.add(
                parent_id
            )

            parent = sections_by_id.get(
                parent_id
            )

            if parent is None:
                break

            titles.append(
                parent.title
            )

            parent_id = (
                parent.parent_section_id
            )

        titles.reverse()

        return tuple(
            title
            for title in titles
            if title.strip()
        )

    def _build_source_references(
        self,
        units: Sequence[_TextUnit],
    ) -> tuple[ChunkSourceReference, ...]:
        references: list[
            ChunkSourceReference
        ] = []

        character_cursor = 0

        for unit in units:
            start_character = (
                character_cursor
            )

            end_character = (
                start_character
                + len(unit.normalized_text)
            )

            references.append(
                ChunkSourceReference(
                    page_index=unit.page_index,
                    page_number=unit.page_number,
                    block_index=unit.block_index,
                    source_block_index=(
                        unit.source_block_index
                    ),
                    start_character=(
                        start_character
                    ),
                    end_character=end_character,
                    content_hash=unit.source_hash,
                )
            )

            character_cursor = (
                end_character + 2
            )

        return tuple(
            references
        )

    def _make_text_unit(
        self,
        *,
        block: SectionContentBlock,
        text: str,
        unit_type: str,
    ) -> _TextUnit:
        normalized_text = (
            self._normalize_text(
                text
            )
        )

        source_hash = hashlib.sha256(
            (
                f"{block.content_hash}:"
                f"{unit_type}:"
                f"{normalized_text}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return _TextUnit(
            text=text,
            normalized_text=(
                normalized_text
            ),
            page_number=block.page_number,
            page_index=block.page_index,
            block_index=block.block_index,
            source_block_index=(
                block.source_block_index
            ),
            unit_type=unit_type,
            source_hash=source_hash,
            estimated_token_count=(
                self.estimate_token_count(
                    normalized_text
                )
            ),
            word_count=len(
                normalized_text.split()
            ),
            character_count=len(
                normalized_text
            ),
        )

    def _split_paragraphs(
        self,
        text: str,
    ) -> list[str]:
        normalized = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        if self.preserve_lists:
            return self._split_with_lists(
                normalized
            )

        if self.preserve_paragraphs:
            return [
                paragraph.strip()
                for paragraph
                in self._PARAGRAPH_SPLIT_RE.split(
                    normalized
                )
                if paragraph.strip()
            ]

        return [
            self._normalize_text(
                normalized
            )
        ]

    def _split_with_lists(
        self,
        text: str,
    ) -> list[str]:
        paragraphs: list[str] = []
        current_lines: list[str] = []
        list_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                if current_lines:
                    paragraphs.append(
                        " ".join(
                            current_lines
                        )
                    )

                    current_lines = []

                if list_lines:
                    paragraphs.append(
                        "\n".join(
                            list_lines
                        )
                    )

                    list_lines = []

                continue

            if self._LIST_ITEM_RE.match(
                stripped
            ):
                if current_lines:
                    paragraphs.append(
                        " ".join(
                            current_lines
                        )
                    )

                    current_lines = []

                list_lines.append(
                    stripped
                )

            else:
                if list_lines:
                    paragraphs.append(
                        "\n".join(
                            list_lines
                        )
                    )

                    list_lines = []

                current_lines.append(
                    stripped
                )

        if current_lines:
            paragraphs.append(
                " ".join(
                    current_lines
                )
            )

        if list_lines:
            paragraphs.append(
                "\n".join(
                    list_lines
                )
            )

        return [
            paragraph
            for paragraph in paragraphs
            if paragraph.strip()
        ]

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        candidates = self._SENTENCE_SPLIT_RE.split(
            text.strip()
        )

        sentences: list[str] = []
        buffer = ""

        for candidate in candidates:
            sentence = candidate.strip()

            if not sentence:
                continue

            if buffer:
                sentence = (
                    f"{buffer} {sentence}"
                ).strip()

                buffer = ""

            lowered = sentence.casefold()

            last_token = (
                lowered.split()[-1]
                if lowered.split()
                else ""
            )

            if last_token in self._ABBREVIATIONS:
                buffer = sentence
                continue

            sentences.append(
                sentence
            )

        if buffer:
            sentences.append(
                buffer
            )

        return sentences or [
            text.strip()
        ]

    @staticmethod
    def _join_units(
        units: Sequence[_TextUnit],
    ) -> str:
        return "\n\n".join(
            unit.normalized_text.strip()
            for unit in units
            if unit.normalized_text.strip()
        )

    @staticmethod
    def _detect_chunk_type(
        units: Sequence[_TextUnit],
    ) -> ChunkType:
        unit_types = {
            unit.unit_type
            for unit in units
        }

        if not unit_types:
            return ChunkType.UNKNOWN

        if "token_fallback" in unit_types:
            return ChunkType.OVERSIZED_SENTENCE

        if unit_types == {
            "list"
        }:
            return ChunkType.LIST

        if unit_types <= {
            "sentence",
        }:
            return ChunkType.SENTENCE_GROUP

        if unit_types <= {
            "paragraph",
            "list",
        }:
            return ChunkType.PARAGRAPH

        return ChunkType.SECTION

    @staticmethod
    def _detect_split_method(
        units: Sequence[_TextUnit],
    ) -> ChunkSplitMethod:
        unit_types = {
            unit.unit_type
            for unit in units
        }

        if "token_fallback" in unit_types:
            return ChunkSplitMethod.TOKEN_FALLBACK

        if unit_types <= {
            "sentence",
        }:
            return ChunkSplitMethod.SENTENCE_BOUNDARY

        if unit_types <= {
            "paragraph",
            "list",
        }:
            return ChunkSplitMethod.PARAGRAPH_BOUNDARY

        if len(units) == 1:
            return ChunkSplitMethod.NO_SPLIT

        return ChunkSplitMethod.BLOCK_BOUNDARY

    def estimate_token_count(
        self,
        text: str,
    ) -> int:
        """
        Harici tokenizer gerektirmeyen yaklaşık token sayımı yapar.

        Bu değer model sağlayıcısının gerçek tokenizer sonucu değildir.
        Retrieval ön işleme ve chunk sınırı oluşturma amacıyla kullanılır.
        """

        normalized = self._normalize_text(
            text
        )

        if not normalized:
            return 0

        lexical_items = self._WORD_RE.findall(
            normalized
        )

        character_estimate = math.ceil(
            len(normalized)
            / self.characters_per_token
        )

        lexical_estimate = math.ceil(
            len(lexical_items)
            * 1.12
        )

        return max(
            1,
            round(
                (
                    0.60
                    * character_estimate
                    + 0.40
                    * lexical_estimate
                )
            ),
        )

    @classmethod
    def _normalize_text(
        cls,
        text: str,
    ) -> str:
        normalized = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        lines = [
            cls._WHITESPACE_RE.sub(
                " ",
                line,
            ).strip()
            for line in normalized.splitlines()
        ]

        normalized = "\n".join(
            lines
        )

        normalized = (
            cls._MULTIPLE_NEWLINE_RE.sub(
                "\n\n",
                normalized,
            )
        )

        return normalized.strip()

    @classmethod
    def _normalize_for_search(
        cls,
        text: str,
    ) -> str:
        translation_table = str.maketrans(
            {
                "İ": "i",
                "I": "i",
                "ı": "i",
                "Ş": "s",
                "ş": "s",
                "Ğ": "g",
                "ğ": "g",
                "Ü": "u",
                "ü": "u",
                "Ö": "o",
                "ö": "o",
                "Ç": "c",
                "ç": "c",
            }
        )

        normalized = cls._normalize_text(
            text
        ).translate(
            translation_table
        ).casefold()

        normalized = re.sub(
            r"[^\w%€$₺£.+\- ]",
            " ",
            normalized,
        )

        return cls._WHITESPACE_RE.sub(
            " ",
            normalized,
        ).strip()

    @classmethod
    def _calculate_semantic_density(
        cls,
        text: str,
    ) -> float:
        words = [
            word
            for word in cls._normalize_for_search(
                text
            ).split()
            if word
        ]

        if not words:
            return 0.0

        unique_ratio = (
            len(set(words))
            / len(words)
        )

        numeric_ratio = (
            sum(
                any(
                    character.isdigit()
                    for character in word
                )
                for word in words
            )
            / len(words)
        )

        density = (
            0.80 * unique_ratio
            + 0.20 * min(
                numeric_ratio * 4,
                1.0,
            )
        )

        return round(
            min(
                max(
                    density,
                    0.0,
                ),
                1.0,
            ),
            4,
        )

    def _calculate_chunk_confidence(
        self,
        *,
        section_confidence: float,
        token_count: int,
        source_unit_count: int,
        is_oversized: bool,
        warning_count: int,
    ) -> float:
        if token_count <= 0:
            return 0.0

        token_distance = abs(
            token_count
            - self.target_tokens
        )

        token_fit_score = max(
            0.0,
            1.0
            - token_distance
            / max(
                self.target_tokens,
                1,
            ),
        )

        structure_score = min(
            source_unit_count / 4,
            1.0,
        )

        confidence = (
            0.50 * section_confidence
            + 0.35 * token_fit_score
            + 0.15 * structure_score
        )

        if is_oversized:
            confidence -= 0.12

        confidence -= min(
            warning_count * 0.04,
            0.16,
        )

        return round(
            min(
                max(
                    confidence,
                    0.0,
                ),
                0.99,
            ),
            4,
        )

    @staticmethod
    def _is_list_text(
        text: str,
    ) -> bool:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        list_line_count = sum(
            bool(
                SemanticChunker._LIST_ITEM_RE.match(
                    line
                )
            )
            for line in lines
        )

        return (
            list_line_count
            / len(lines)
            >= 0.50
        )

    @staticmethod
    def _assign_local_neighbors(
        chunks: Sequence[SemanticChunk],
    ) -> list[SemanticChunk]:
        result: list[SemanticChunk] = []

        for index, chunk in enumerate(
            chunks
        ):
            previous_chunk_id = (
                chunks[index - 1].chunk_id
                if index > 0
                else None
            )

            next_chunk_id = (
                chunks[index + 1].chunk_id
                if index < len(chunks) - 1
                else None
            )

            result.append(
                SemanticChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    section_id=chunk.section_id,
                    parent_section_id=(
                        chunk.parent_section_id
                    ),
                    chunk_index=chunk.chunk_index,
                    section_chunk_index=(
                        chunk.section_chunk_index
                    ),
                    previous_chunk_id=(
                        previous_chunk_id
                    ),
                    next_chunk_id=next_chunk_id,
                    chunk_type=chunk.chunk_type,
                    split_method=chunk.split_method,
                    section_level=chunk.section_level,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    text=chunk.text,
                    normalized_text=(
                        chunk.normalized_text
                    ),
                    searchable_text=(
                        chunk.searchable_text
                    ),
                    contextualized_text=(
                        chunk.contextualized_text
                    ),
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    source_references=(
                        chunk.source_references
                    ),
                    character_count=(
                        chunk.character_count
                    ),
                    word_count=chunk.word_count,
                    estimated_token_count=(
                        chunk.estimated_token_count
                    ),
                    overlap_character_count=(
                        chunk.overlap_character_count
                    ),
                    overlap_word_count=(
                        chunk.overlap_word_count
                    ),
                    overlap_token_count=(
                        chunk.overlap_token_count
                    ),
                    semantic_density=(
                        chunk.semantic_density
                    ),
                    chunk_confidence=(
                        chunk.chunk_confidence
                    ),
                    content_hash=chunk.content_hash,
                    is_oversized=chunk.is_oversized,
                    is_synthetic=chunk.is_synthetic,
                    warnings=chunk.warnings,
                    metadata=chunk.metadata,
                )
            )

        return result

    @staticmethod
    def _refresh_global_neighbors(
        chunks: Sequence[SemanticChunk],
    ) -> list[SemanticChunk]:
        result: list[SemanticChunk] = []

        for index, chunk in enumerate(
            chunks
        ):
            previous_chunk_id = (
                chunks[index - 1].chunk_id
                if index > 0
                else None
            )

            next_chunk_id = (
                chunks[index + 1].chunk_id
                if index
                < len(chunks) - 1
                else None
            )

            result.append(
                SemanticChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    section_id=chunk.section_id,
                    parent_section_id=(
                        chunk.parent_section_id
                    ),
                    chunk_index=index,
                    section_chunk_index=(
                        chunk.section_chunk_index
                    ),
                    previous_chunk_id=(
                        previous_chunk_id
                    ),
                    next_chunk_id=next_chunk_id,
                    chunk_type=chunk.chunk_type,
                    split_method=chunk.split_method,
                    section_level=chunk.section_level,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    text=chunk.text,
                    normalized_text=(
                        chunk.normalized_text
                    ),
                    searchable_text=(
                        chunk.searchable_text
                    ),
                    contextualized_text=(
                        chunk.contextualized_text
                    ),
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    source_references=(
                        chunk.source_references
                    ),
                    character_count=(
                        chunk.character_count
                    ),
                    word_count=chunk.word_count,
                    estimated_token_count=(
                        chunk.estimated_token_count
                    ),
                    overlap_character_count=(
                        chunk.overlap_character_count
                    ),
                    overlap_word_count=(
                        chunk.overlap_word_count
                    ),
                    overlap_token_count=(
                        chunk.overlap_token_count
                    ),
                    semantic_density=(
                        chunk.semantic_density
                    ),
                    chunk_confidence=(
                        chunk.chunk_confidence
                    ),
                    content_hash=chunk.content_hash,
                    is_oversized=chunk.is_oversized,
                    is_synthetic=chunk.is_synthetic,
                    warnings=chunk.warnings,
                    metadata=chunk.metadata,
                )
            )

        return result

    @staticmethod
    def _refresh_section_results(
        *,
        section_results: Sequence[
            SectionChunkingResult
        ],
        chunks: Sequence[SemanticChunk],
    ) -> list[SectionChunkingResult]:
        chunks_by_section: dict[
            str,
            list[SemanticChunk],
        ] = {}

        for chunk in chunks:
            chunks_by_section.setdefault(
                chunk.section_id,
                [],
            ).append(chunk)

        refreshed: list[
            SectionChunkingResult
        ] = []

        for result in section_results:
            section_chunks = tuple(
                chunks_by_section.get(
                    result.section_id,
                    (),
                )
            )

            refreshed.append(
                SectionChunkingResult(
                    section_id=result.section_id,
                    section_title=(
                        result.section_title
                    ),
                    input_character_count=(
                        result.input_character_count
                    ),
                    input_word_count=(
                        result.input_word_count
                    ),
                    input_estimated_token_count=(
                        result.input_estimated_token_count
                    ),
                    chunk_count=len(
                        section_chunks
                    ),
                    oversized_chunk_count=sum(
                        chunk.is_oversized
                        for chunk in section_chunks
                    ),
                    chunks=section_chunks,
                    warnings=result.warnings,
                )
            )

        return refreshed

    @staticmethod
    def _calculate_result_hash(
        chunks: Sequence[SemanticChunk],
    ) -> str:
        canonical = "\n".join(
            (
                f"{chunk.chunk_id}:"
                f"{chunk.section_id}:"
                f"{chunk.chunk_index}:"
                f"{chunk.content_hash}"
            )
            for chunk in chunks
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()


def create_semantic_chunks(
    section_result: SectionDetectionResult,
    *,
    target_tokens: int = 420,
    maximum_tokens: int = 560,
    minimum_tokens: int = 80,
    overlap_tokens: int = 60,
) -> SemanticChunkingResult:
    """SemanticChunker için kısa kullanım arayüzü."""

    chunker = SemanticChunker(
        target_tokens=target_tokens,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        overlap_tokens=overlap_tokens,
    )

    return chunker.chunk_document(
        section_result
    )