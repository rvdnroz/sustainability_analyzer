from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import median
from typing import Any, Mapping, Sequence

from .text_cleaner import (
    CleanedDocumentText,
    CleanedPageText,
    CleanedTextBlock,
)


logger = logging.getLogger(__name__)


class SectionDetectionError(RuntimeError):
    """Belge bölümleri tespit edilemediğinde yükseltilir."""


class SectionType(StrEnum):
    """Tespit edilen bölümün yapısal türü."""

    DOCUMENT_ROOT = "document_root"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    SUBSUBSECTION = "subsubsection"
    APPENDIX = "appendix"
    INDEX = "index"
    TABLE_OF_CONTENTS = "table_of_contents"
    UNKNOWN = "unknown"


class HeadingDetectionMethod(StrEnum):
    """Bir başlığın hangi sinyalle tespit edildiğini belirtir."""

    SOURCE_METADATA = "source_metadata"
    NUMBERING_PATTERN = "numbering_pattern"
    TYPOGRAPHIC_PATTERN = "typographic_pattern"
    LEXICAL_PATTERN = "lexical_pattern"
    UPPERCASE_PATTERN = "uppercase_pattern"
    SHORT_LINE_PATTERN = "short_line_pattern"
    COMBINED = "combined"
    SYNTHETIC = "synthetic"


@dataclass(slots=True, frozen=True)
class HeadingCandidate:
    """Başlık olma ihtimali bulunan tek bir metin bloğu."""

    candidate_id: str

    page_index: int
    page_number: int

    block_index: int
    source_block_index: int | None

    text: str
    normalized_text: str
    searchable_text: str

    detected_level: int
    section_type: SectionType

    detection_method: HeadingDetectionMethod
    confidence: float

    numbering_prefix: str | None
    numbering_depth: int | None

    is_source_heading: bool
    is_uppercase: bool
    is_title_case: bool

    word_count: int
    character_count: int

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["section_type"] = (
            self.section_type.value
        )

        data["detection_method"] = (
            self.detection_method.value
        )

        return data


@dataclass(slots=True, frozen=True)
class SectionContentBlock:
    """Bir bölüme atanmış tek bir temiz metin bloğu."""

    page_index: int
    page_number: int

    block_index: int
    source_block_index: int | None

    text: str
    normalized_text: str
    searchable_text: str

    block_type: str

    is_heading: bool
    word_count: int
    character_count: int

    content_hash: str

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DetectedSection:
    """Belgede tespit edilen tek bir bölüm veya alt bölüm."""

    section_id: str
    parent_section_id: str | None

    section_index: int
    sibling_index: int

    level: int
    section_type: SectionType

    title: str
    normalized_title: str
    title_search_text: str

    numbering_prefix: str | None

    start_page: int
    end_page: int

    start_block_index: int
    end_block_index: int

    content: str
    normalized_content: str
    searchable_content: str

    blocks: tuple[SectionContentBlock, ...]
    child_section_ids: tuple[str, ...]

    word_count: int
    character_count: int
    block_count: int

    heading_confidence: float
    section_confidence: float

    content_hash: str

    is_empty: bool
    is_synthetic: bool

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "parent_section_id": (
                self.parent_section_id
            ),
            "section_index": self.section_index,
            "sibling_index": self.sibling_index,
            "level": self.level,
            "section_type": self.section_type.value,
            "title": self.title,
            "normalized_title": self.normalized_title,
            "title_search_text": (
                self.title_search_text
            ),
            "numbering_prefix": (
                self.numbering_prefix
            ),
            "start_page": self.start_page,
            "end_page": self.end_page,
            "start_block_index": (
                self.start_block_index
            ),
            "end_block_index": (
                self.end_block_index
            ),
            "content": self.content,
            "normalized_content": (
                self.normalized_content
            ),
            "searchable_content": (
                self.searchable_content
            ),
            "blocks": [
                block.to_dict()
                for block in self.blocks
            ],
            "child_section_ids": list(
                self.child_section_ids
            ),
            "word_count": self.word_count,
            "character_count": (
                self.character_count
            ),
            "block_count": self.block_count,
            "heading_confidence": (
                self.heading_confidence
            ),
            "section_confidence": (
                self.section_confidence
            ),
            "content_hash": self.content_hash,
            "is_empty": self.is_empty,
            "is_synthetic": self.is_synthetic,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class SectionHierarchyNode:
    """Bölüm ağacındaki hafif düğüm temsili."""

    section_id: str
    parent_section_id: str | None

    level: int
    title: str

    child_section_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SectionDetectionResult:
    """Belge seviyesinde bölüm tespit sonucu."""

    document_id: str

    section_count: int
    root_section_count: int
    maximum_depth: int

    heading_candidate_count: int
    rejected_heading_count: int

    sections: tuple[DetectedSection, ...]
    hierarchy: tuple[SectionHierarchyNode, ...]

    unassigned_block_count: int

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "section_count": self.section_count,
            "root_section_count": (
                self.root_section_count
            ),
            "maximum_depth": self.maximum_depth,
            "heading_candidate_count": (
                self.heading_candidate_count
            ),
            "rejected_heading_count": (
                self.rejected_heading_count
            ),
            "sections": [
                section.to_dict()
                for section in self.sections
            ],
            "hierarchy": [
                node.to_dict()
                for node in self.hierarchy
            ],
            "unassigned_block_count": (
                self.unassigned_block_count
            ),
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class _MutableSection:
    """Bölüm ağacı kurulurken kullanılan iç temsil."""

    section_id: str
    parent_section_id: str | None

    section_index: int
    sibling_index: int

    level: int
    section_type: SectionType

    title: str
    normalized_title: str
    title_search_text: str

    numbering_prefix: str | None

    start_page: int
    end_page: int

    start_block_index: int
    end_block_index: int

    blocks: list[SectionContentBlock]
    child_section_ids: list[str]

    heading_confidence: float

    is_synthetic: bool
    warnings: list[str]

    metadata: dict[str, Any]


class SectionDetector:
    """
    Temizlenmiş belge metninde bölüm ve alt bölüm sınırlarını belirler.

    Kullanılan sinyaller:

    - CleanedTextBlock.is_heading_candidate
    - Kaynak blok tipi
    - Sayısal başlık örüntüleri
    - Roma rakamları ve harfli başlıklar
    - Büyük harf oranı
    - Başlık uzunluğu
    - Noktalama işaretleri
    - Sürdürülebilirlik raporlarında sık kullanılan bölüm isimleri
    - Sayfa başlangıcındaki kısa ve belirgin metinler

    Çıktı:

    - Hiyerarşik bölüm ağacı
    - Bölüm başlangıç ve bitiş sayfaları
    - Bölüme bağlı metin blokları
    - Başlık güven skoru
    - Parent-child ilişkileri

    Bu modül token bazlı chunk üretmez. Bölüm sonuçları
    semantic_chunker.py tarafından anlam bütünlüğü korunarak parçalanır.
    """

    _WHITESPACE_RE = re.compile(
        r"\s+"
    )

    _LEADING_NUMBERING_RE = re.compile(
        r"""
        ^\s*
        (?P<prefix>
            (?:
                \d+
                (?:\.\d+){0,5}
                \.?
            )
            |
            (?:
                [IVXLCDM]+
                [.)]
            )
            |
            (?:
                [A-ZÇĞİÖŞÜ]
                [.)]
            )
        )
        \s+
        (?P<title>.+?)
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _CHAPTER_RE = re.compile(
        r"""
        ^\s*
        (?:
            chapter
            |
            bölüm
            |
            part
            |
            kısım
        )
        \s+
        (?:
            \d+
            |
            [IVXLCDM]+
        )
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _APPENDIX_RE = re.compile(
        r"""
        ^\s*
        (?:
            appendix
            |
            annex
            |
            ek
        )
        (?:\s+[A-Z0-9ÇĞİÖŞÜ]+)?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _TABLE_OF_CONTENTS_RE = re.compile(
        r"""
        ^\s*
        (?:
            table\s+of\s+contents
            |
            contents
            |
            içindekiler
        )
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _INDEX_RE = re.compile(
        r"""
        ^\s*
        (?:
            gri\s+(?:content\s+)?index
            |
            esrs\s+(?:content\s+)?index
            |
            sustainability\s+index
            |
            sürdürülebilirlik\s+indeksi
            |
            standart(?:lar)?\s+indeksi
        )
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _SENTENCE_END_RE = re.compile(
        r"""[.!?;:]$"""
    )

    _COMMON_SECTION_TITLES = {
        "about this report",
        "rapor hakkında",
        "about us",
        "hakkımızda",
        "company profile",
        "şirket profili",
        "message from the ceo",
        "ceo mesajı",
        "chairman message",
        "yönetim kurulu başkanı mesajı",
        "sustainability approach",
        "sürdürülebilirlik yaklaşımı",
        "sustainability strategy",
        "sürdürülebilirlik stratejisi",
        "materiality assessment",
        "önceliklendirme analizi",
        "material topics",
        "öncelikli konular",
        "stakeholder engagement",
        "paydaş katılımı",
        "environment",
        "environmental performance",
        "çevre",
        "çevresel performans",
        "climate change",
        "iklim değişikliği",
        "energy",
        "enerji",
        "emissions",
        "emisyonlar",
        "water",
        "su",
        "waste",
        "atık",
        "biodiversity",
        "biyoçeşitlilik",
        "social",
        "social performance",
        "sosyal performans",
        "employees",
        "çalışanlar",
        "occupational health and safety",
        "iş sağlığı ve güvenliği",
        "human rights",
        "insan hakları",
        "governance",
        "yönetişim",
        "corporate governance",
        "kurumsal yönetişim",
        "ethics and compliance",
        "etik ve uyum",
        "risk management",
        "risk yönetimi",
        "supply chain",
        "tedarik zinciri",
        "innovation",
        "inovasyon",
        "digitalization",
        "dijitalleşme",
        "economic performance",
        "ekonomik performans",
        "performance indicators",
        "performans göstergeleri",
        "appendices",
        "ekler",
        "gri content index",
        "gri içerik indeksi",
        "esrs index",
        "esrs indeksi",
        "assurance statement",
        "güvence beyanı",
    }

    def __init__(
        self,
        *,
        minimum_heading_confidence: float = 0.55,
        minimum_section_words: int = 5,
        maximum_heading_words: int = 18,
        maximum_heading_characters: int = 180,
        maximum_heading_lines: int = 3,
        uppercase_heading_ratio: float = 0.65,
        title_case_heading_ratio: float = 0.60,
        create_document_root: bool = True,
        create_preamble_section: bool = True,
        include_heading_in_content: bool = False,
        merge_empty_sections: bool = True,
        allow_unnumbered_headings: bool = True,
        maximum_hierarchy_depth: int = 6,
    ) -> None:
        if not 0.0 <= minimum_heading_confidence <= 1.0:
            raise ValueError(
                "minimum_heading_confidence "
                "0 ile 1 arasında olmalıdır."
            )

        if minimum_section_words < 0:
            raise ValueError(
                "minimum_section_words negatif olamaz."
            )

        if maximum_heading_words < 1:
            raise ValueError(
                "maximum_heading_words pozitif olmalıdır."
            )

        if maximum_heading_characters < 1:
            raise ValueError(
                "maximum_heading_characters pozitif olmalıdır."
            )

        if maximum_heading_lines < 1:
            raise ValueError(
                "maximum_heading_lines pozitif olmalıdır."
            )

        for name, value in {
            "uppercase_heading_ratio": (
                uppercase_heading_ratio
            ),
            "title_case_heading_ratio": (
                title_case_heading_ratio
            ),
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        if maximum_hierarchy_depth < 1:
            raise ValueError(
                "maximum_hierarchy_depth pozitif olmalıdır."
            )

        self.minimum_heading_confidence = (
            minimum_heading_confidence
        )

        self.minimum_section_words = (
            minimum_section_words
        )

        self.maximum_heading_words = (
            maximum_heading_words
        )

        self.maximum_heading_characters = (
            maximum_heading_characters
        )

        self.maximum_heading_lines = (
            maximum_heading_lines
        )

        self.uppercase_heading_ratio = (
            uppercase_heading_ratio
        )

        self.title_case_heading_ratio = (
            title_case_heading_ratio
        )

        self.create_document_root = (
            create_document_root
        )

        self.create_preamble_section = (
            create_preamble_section
        )

        self.include_heading_in_content = (
            include_heading_in_content
        )

        self.merge_empty_sections = (
            merge_empty_sections
        )

        self.allow_unnumbered_headings = (
            allow_unnumbered_headings
        )

        self.maximum_hierarchy_depth = (
            maximum_hierarchy_depth
        )

    def detect_sections(
        self,
        document: CleanedDocumentText,
        *,
        document_id: str | None = None,
        document_title: str | None = None,
    ) -> SectionDetectionResult:
        """Temizlenmiş belgedeki bölüm hiyerarşisini çıkarır."""

        if not isinstance(
            document,
            CleanedDocumentText,
        ):
            raise TypeError(
                "document CleanedDocumentText olmalıdır."
            )

        resolved_document_id = (
            document_id
            or self._build_document_id(
                document
            )
        )

        ordered_blocks = self._flatten_blocks(
            document.pages
        )

        if not ordered_blocks:
            return self._empty_result(
                document_id=resolved_document_id,
                document=document,
            )

        heading_candidates: list[
            HeadingCandidate
        ] = []

        rejected_heading_count = 0

        for page, block in ordered_blocks:
            candidate = self.evaluate_heading(
                page=page,
                block=block,
            )

            if candidate is None:
                if (
                    block.is_heading_candidate
                    or block.block_type == "heading"
                ):
                    rejected_heading_count += 1

                continue

            heading_candidates.append(
                candidate
            )

        mutable_sections = self._build_sections(
            document_id=resolved_document_id,
            ordered_blocks=ordered_blocks,
            heading_candidates=heading_candidates,
            document_title=document_title,
        )

        if self.merge_empty_sections:
            mutable_sections = (
                self._merge_or_remove_empty_sections(
                    mutable_sections
                )
            )

        self._refresh_section_relationships(
            mutable_sections
        )

        sections = tuple(
            self._freeze_section(section)
            for section in mutable_sections
        )

        hierarchy = tuple(
            SectionHierarchyNode(
                section_id=section.section_id,
                parent_section_id=(
                    section.parent_section_id
                ),
                level=section.level,
                title=section.title,
                child_section_ids=(
                    section.child_section_ids
                ),
            )
            for section in sections
        )

        root_section_count = sum(
            section.parent_section_id is None
            for section in sections
        )

        maximum_depth = max(
            (
                section.level
                for section in sections
            ),
            default=0,
        )

        assigned_block_hashes = {
            block.content_hash
            for section in sections
            for block in section.blocks
        }

        unassigned_block_count = sum(
            block.content_hash
            not in assigned_block_hashes
            for _, block in ordered_blocks
        )

        warnings: list[str] = []

        if not heading_candidates:
            warnings.append(
                "Belgede güvenilir başlık adayı bulunamadı; "
                "sentetik bölüm yapısı oluşturuldu."
            )

        if maximum_depth >= self.maximum_hierarchy_depth:
            warnings.append(
                "Bölüm hiyerarşisi yapılandırılan maksimum "
                "derinliğe ulaştı."
            )

        content_hash = self._calculate_sections_hash(
            sections
        )

        result = SectionDetectionResult(
            document_id=resolved_document_id,
            section_count=len(sections),
            root_section_count=(
                root_section_count
            ),
            maximum_depth=maximum_depth,
            heading_candidate_count=len(
                heading_candidates
            ),
            rejected_heading_count=(
                rejected_heading_count
            ),
            sections=sections,
            hierarchy=hierarchy,
            unassigned_block_count=(
                unassigned_block_count
            ),
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "source_document_hash": (
                    document.content_hash
                ),
                "source_page_count": (
                    document.processed_page_count
                ),
                "minimum_heading_confidence": (
                    self.minimum_heading_confidence
                ),
                "create_document_root": (
                    self.create_document_root
                ),
                "create_preamble_section": (
                    self.create_preamble_section
                ),
            },
        )

        logger.info(
            "%d başlık adayından %d bölüm oluşturuldu.",
            len(heading_candidates),
            len(sections),
        )

        return result

    def evaluate_heading(
        self,
        *,
        page: CleanedPageText,
        block: CleanedTextBlock,
    ) -> HeadingCandidate | None:
        """Tek bir temiz bloğun başlık olup olmadığını değerlendirir."""

        normalized_text = self._normalize_heading_text(
            block.normalized_text
            or block.text
        )

        if not normalized_text:
            return None

        word_count = len(
            normalized_text.split()
        )

        character_count = len(
            normalized_text
        )

        line_count = max(
            block.line_count,
            1,
        )

        if (
            word_count
            > self.maximum_heading_words
        ):
            return None

        if (
            character_count
            > self.maximum_heading_characters
        ):
            return None

        if (
            line_count
            > self.maximum_heading_lines
        ):
            return None

        score = 0.0
        methods: list[
            HeadingDetectionMethod
        ] = []

        is_source_heading = bool(
            block.is_heading_candidate
            or block.block_type == "heading"
        )

        if is_source_heading:
            score += 0.42
            methods.append(
                HeadingDetectionMethod.SOURCE_METADATA
            )

        numbering_match = (
            self._LEADING_NUMBERING_RE.match(
                normalized_text
            )
        )

        numbering_prefix: str | None = None
        numbering_depth: int | None = None

        if numbering_match:
            numbering_prefix = (
                numbering_match.group(
                    "prefix"
                ).strip()
            )

            numbering_depth = (
                self._numbering_depth(
                    numbering_prefix
                )
            )

            score += 0.38
            methods.append(
                HeadingDetectionMethod.NUMBERING_PATTERN
            )

        search_text = (
            self._normalize_for_search(
                normalized_text
            )
        )

        if search_text in self._COMMON_SECTION_TITLES:
            score += 0.35
            methods.append(
                HeadingDetectionMethod.LEXICAL_PATTERN
            )

        uppercase_ratio = (
            self._uppercase_ratio(
                normalized_text
            )
        )

        is_uppercase = (
            uppercase_ratio
            >= self.uppercase_heading_ratio
        )

        if is_uppercase:
            score += 0.18
            methods.append(
                HeadingDetectionMethod.UPPERCASE_PATTERN
            )

        title_case_ratio = (
            self._title_case_ratio(
                normalized_text
            )
        )

        is_title_case = (
            title_case_ratio
            >= self.title_case_heading_ratio
        )

        if is_title_case:
            score += 0.12
            methods.append(
                HeadingDetectionMethod.TYPOGRAPHIC_PATTERN
            )

        if (
            word_count <= 10
            and character_count <= 100
        ):
            score += 0.10
            methods.append(
                HeadingDetectionMethod.SHORT_LINE_PATTERN
            )

        if self._SENTENCE_END_RE.search(
            normalized_text
        ):
            score -= 0.18

        if (
            character_count > 130
            or word_count > 14
        ):
            score -= 0.12

        if (
            not self.allow_unnumbered_headings
            and numbering_match is None
            and not is_source_heading
        ):
            return None

        score = min(
            max(
                score,
                0.0,
            ),
            0.99,
        )

        if score < self.minimum_heading_confidence:
            return None

        section_type, level = (
            self._classify_heading(
                text=normalized_text,
                numbering_prefix=numbering_prefix,
                numbering_depth=numbering_depth,
            )
        )

        if len(methods) > 1:
            detection_method = (
                HeadingDetectionMethod.COMBINED
            )
        elif methods:
            detection_method = methods[0]
        else:
            detection_method = (
                HeadingDetectionMethod.SHORT_LINE_PATTERN
            )

        candidate_id = hashlib.sha256(
            (
                f"{page.page_number}:"
                f"{block.block_index}:"
                f"{normalized_text}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return HeadingCandidate(
            candidate_id=candidate_id,
            page_index=page.page_index,
            page_number=page.page_number,
            block_index=block.block_index,
            source_block_index=(
                block.source_block_index
            ),
            text=block.text,
            normalized_text=normalized_text,
            searchable_text=search_text,
            detected_level=level,
            section_type=section_type,
            detection_method=(
                detection_method
            ),
            confidence=round(
                score,
                4,
            ),
            numbering_prefix=(
                numbering_prefix
            ),
            numbering_depth=(
                numbering_depth
            ),
            is_source_heading=(
                is_source_heading
            ),
            is_uppercase=is_uppercase,
            is_title_case=is_title_case,
            word_count=word_count,
            character_count=character_count,
            metadata={
                "source_block_type": (
                    block.block_type
                ),
                "source_cleaning_confidence": (
                    page.cleaning_confidence
                ),
                "uppercase_ratio": round(
                    uppercase_ratio,
                    4,
                ),
                "title_case_ratio": round(
                    title_case_ratio,
                    4,
                ),
            },
        )

    def _build_sections(
        self,
        *,
        document_id: str,
        ordered_blocks: Sequence[
            tuple[
                CleanedPageText,
                CleanedTextBlock,
            ]
        ],
        heading_candidates: Sequence[
            HeadingCandidate
        ],
        document_title: str | None,
    ) -> list[_MutableSection]:
        candidate_map = {
            (
                candidate.page_number,
                candidate.block_index,
            ): candidate
            for candidate in heading_candidates
        }

        sections: list[
            _MutableSection
        ] = []

        stack: list[
            _MutableSection
        ] = []

        sibling_counts: dict[
            str | None,
            int,
        ] = {}

        current_section: (
            _MutableSection | None
        ) = None

        if self.create_document_root:
            root_title = (
                document_title
                or "Document"
            )

            root_section = self._create_mutable_section(
                document_id=document_id,
                section_index=0,
                parent_section_id=None,
                sibling_index=0,
                level=0,
                section_type=(
                    SectionType.DOCUMENT_ROOT
                ),
                title=root_title,
                numbering_prefix=None,
                page_number=(
                    ordered_blocks[0][0].page_number
                ),
                block_index=-1,
                heading_confidence=1.0,
                is_synthetic=True,
            )

            sections.append(
                root_section
            )

            stack.append(
                root_section
            )

        preamble_section: (
            _MutableSection | None
        ) = None

        for page, block in ordered_blocks:
            candidate = candidate_map.get(
                (
                    page.page_number,
                    block.block_index,
                )
            )

            if candidate is not None:
                level = candidate.detected_level

                if self.create_document_root:
                    level = max(
                        level,
                        1,
                    )

                level = min(
                    level,
                    self.maximum_hierarchy_depth,
                )

                while (
                    stack
                    and stack[-1].level >= level
                ):
                    stack.pop()

                parent = (
                    stack[-1]
                    if stack
                    else None
                )

                parent_id = (
                    parent.section_id
                    if parent is not None
                    else None
                )

                sibling_index = (
                    sibling_counts.get(
                        parent_id,
                        0,
                    )
                )

                sibling_counts[
                    parent_id
                ] = sibling_index + 1

                section = self._create_mutable_section(
                    document_id=document_id,
                    section_index=len(
                        sections
                    ),
                    parent_section_id=(
                        parent_id
                    ),
                    sibling_index=(
                        sibling_index
                    ),
                    level=level,
                    section_type=(
                        candidate.section_type
                    ),
                    title=(
                        candidate.normalized_text
                    ),
                    numbering_prefix=(
                        candidate.numbering_prefix
                    ),
                    page_number=page.page_number,
                    block_index=block.block_index,
                    heading_confidence=(
                        candidate.confidence
                    ),
                    is_synthetic=False,
                )

                section.metadata.update(
                    {
                        "heading_candidate_id": (
                            candidate.candidate_id
                        ),
                        "heading_detection_method": (
                            candidate.detection_method.value
                        ),
                        "source_heading_text": (
                            candidate.text
                        ),
                    }
                )

                sections.append(
                    section
                )

                if parent is not None:
                    parent.child_section_ids.append(
                        section.section_id
                    )

                stack.append(
                    section
                )

                current_section = section

                if self.include_heading_in_content:
                    section.blocks.append(
                        self._convert_block(
                            page=page,
                            block=block,
                            is_heading=True,
                        )
                    )

                continue

            if current_section is None:
                if self.create_preamble_section:
                    if preamble_section is None:
                        parent = (
                            stack[0]
                            if (
                                self.create_document_root
                                and stack
                            )
                            else None
                        )

                        parent_id = (
                            parent.section_id
                            if parent is not None
                            else None
                        )

                        sibling_index = (
                            sibling_counts.get(
                                parent_id,
                                0,
                            )
                        )

                        sibling_counts[
                            parent_id
                        ] = sibling_index + 1

                        preamble_section = (
                            self._create_mutable_section(
                                document_id=document_id,
                                section_index=len(
                                    sections
                                ),
                                parent_section_id=(
                                    parent_id
                                ),
                                sibling_index=(
                                    sibling_index
                                ),
                                level=(
                                    1
                                    if self.create_document_root
                                    else 0
                                ),
                                section_type=(
                                    SectionType.SECTION
                                ),
                                title="Preamble",
                                numbering_prefix=None,
                                page_number=(
                                    page.page_number
                                ),
                                block_index=(
                                    block.block_index
                                ),
                                heading_confidence=0.50,
                                is_synthetic=True,
                            )
                        )

                        sections.append(
                            preamble_section
                        )

                        if parent is not None:
                            parent.child_section_ids.append(
                                preamble_section.section_id
                            )

                    current_section = (
                        preamble_section
                    )

                elif (
                    self.create_document_root
                    and sections
                ):
                    current_section = sections[0]

            if current_section is not None:
                current_section.blocks.append(
                    self._convert_block(
                        page=page,
                        block=block,
                        is_heading=False,
                    )
                )

                current_section.end_page = (
                    page.page_number
                )

                current_section.end_block_index = (
                    block.block_index
                )

        if (
            not sections
            or (
                self.create_document_root
                and len(sections) == 1
            )
        ):
            synthetic = self._create_synthetic_content_section(
                document_id=document_id,
                ordered_blocks=ordered_blocks,
                parent_section=(
                    sections[0]
                    if sections
                    else None
                ),
                section_index=len(
                    sections
                ),
            )

            if synthetic is not None:
                sections.append(synthetic)

                if sections[0] is not synthetic:
                    sections[0].child_section_ids.append(
                        synthetic.section_id
                    )

        return sections

    def _create_synthetic_content_section(
        self,
        *,
        document_id: str,
        ordered_blocks: Sequence[
            tuple[
                CleanedPageText,
                CleanedTextBlock,
            ]
        ],
        parent_section: _MutableSection | None,
        section_index: int,
    ) -> _MutableSection | None:
        if not ordered_blocks:
            return None

        first_page, first_block = (
            ordered_blocks[0]
        )

        section = self._create_mutable_section(
            document_id=document_id,
            section_index=section_index,
            parent_section_id=(
                parent_section.section_id
                if parent_section is not None
                else None
            ),
            sibling_index=0,
            level=(
                parent_section.level + 1
                if parent_section is not None
                else 0
            ),
            section_type=SectionType.SECTION,
            title="Document Content",
            numbering_prefix=None,
            page_number=first_page.page_number,
            block_index=first_block.block_index,
            heading_confidence=0.50,
            is_synthetic=True,
        )

        for page, block in ordered_blocks:
            section.blocks.append(
                self._convert_block(
                    page=page,
                    block=block,
                    is_heading=False,
                )
            )

            section.end_page = page.page_number
            section.end_block_index = (
                block.block_index
            )

        section.warnings.append(
            "Güvenilir başlık bulunamadığı için "
            "sentetik içerik bölümü oluşturuldu."
        )

        return section

    def _create_mutable_section(
        self,
        *,
        document_id: str,
        section_index: int,
        parent_section_id: str | None,
        sibling_index: int,
        level: int,
        section_type: SectionType,
        title: str,
        numbering_prefix: str | None,
        page_number: int,
        block_index: int,
        heading_confidence: float,
        is_synthetic: bool,
    ) -> _MutableSection:
        normalized_title = (
            self._normalize_heading_text(
                title
            )
        )

        section_key = (
            f"{document_id}:"
            f"{section_index}:"
            f"{parent_section_id}:"
            f"{level}:"
            f"{normalized_title}:"
            f"{page_number}:"
            f"{block_index}"
        )

        section_id = hashlib.sha256(
            section_key.encode(
                "utf-8"
            )
        ).hexdigest()

        return _MutableSection(
            section_id=section_id,
            parent_section_id=(
                parent_section_id
            ),
            section_index=section_index,
            sibling_index=sibling_index,
            level=level,
            section_type=section_type,
            title=normalized_title,
            normalized_title=(
                normalized_title
            ),
            title_search_text=(
                self._normalize_for_search(
                    normalized_title
                )
            ),
            numbering_prefix=(
                numbering_prefix
            ),
            start_page=page_number,
            end_page=page_number,
            start_block_index=block_index,
            end_block_index=block_index,
            blocks=[],
            child_section_ids=[],
            heading_confidence=(
                heading_confidence
            ),
            is_synthetic=is_synthetic,
            warnings=[],
            metadata={},
        )

    def _merge_or_remove_empty_sections(
        self,
        sections: list[_MutableSection],
    ) -> list[_MutableSection]:
        if not sections:
            return sections

        sections_by_id = {
            section.section_id: section
            for section in sections
        }

        removable_ids: set[str] = set()

        for section in sections:
            if (
                section.section_type
                == SectionType.DOCUMENT_ROOT
            ):
                continue

            own_word_count = sum(
                block.word_count
                for block in section.blocks
                if not block.is_heading
            )

            if own_word_count >= self.minimum_section_words:
                continue

            if section.child_section_ids:
                section.warnings.append(
                    "Bölüm kendi içeriği kısa ancak alt "
                    "bölümleri bulunduğu için korundu."
                )
                continue

            parent = (
                sections_by_id.get(
                    section.parent_section_id
                )
                if section.parent_section_id
                else None
            )

            if parent is None:
                continue

            parent.blocks.extend(
                section.blocks
            )

            parent.end_page = max(
                parent.end_page,
                section.end_page,
            )

            parent.end_block_index = max(
                parent.end_block_index,
                section.end_block_index,
            )

            parent.warnings.append(
                f"Kısa bölüm üst bölüme birleştirildi: "
                f"{section.title}"
            )

            removable_ids.add(
                section.section_id
            )

        retained_sections = [
            section
            for section in sections
            if section.section_id
            not in removable_ids
        ]

        for section in retained_sections:
            section.child_section_ids = [
                child_id
                for child_id
                in section.child_section_ids
                if child_id
                not in removable_ids
            ]

        return retained_sections

    def _refresh_section_relationships(
        self,
        sections: list[_MutableSection],
    ) -> None:
        existing_ids = {
            section.section_id
            for section in sections
        }

        for new_index, section in enumerate(
            sections
        ):
            section.section_index = new_index

            if (
                section.parent_section_id
                not in existing_ids
            ):
                section.parent_section_id = None

            section.child_section_ids = [
                child_id
                for child_id
                in section.child_section_ids
                if child_id in existing_ids
            ]

        sibling_counters: dict[
            str | None,
            int,
        ] = {}

        for section in sections:
            parent_id = (
                section.parent_section_id
            )

            section.sibling_index = (
                sibling_counters.get(
                    parent_id,
                    0,
                )
            )

            sibling_counters[
                parent_id
            ] = (
                section.sibling_index
                + 1
            )

    def _freeze_section(
        self,
        section: _MutableSection,
    ) -> DetectedSection:
        ordered_blocks = sorted(
            section.blocks,
            key=lambda block: (
                block.page_number,
                block.block_index,
            ),
        )

        content = "\n\n".join(
            block.text.strip()
            for block in ordered_blocks
            if block.text.strip()
        )

        normalized_content = (
            self._normalize_content(
                content
            )
        )

        searchable_content = (
            self._normalize_for_search(
                normalized_content
            )
        )

        word_count = len(
            normalized_content.split()
        )

        character_count = len(
            normalized_content
        )

        is_empty = not bool(
            normalized_content
        )

        warnings = list(
            section.warnings
        )

        if is_empty:
            warnings.append(
                "Bölüm doğrudan metin içeriği içermiyor."
            )

        section_confidence = (
            self._calculate_section_confidence(
                heading_confidence=(
                    section.heading_confidence
                ),
                word_count=word_count,
                block_count=len(
                    ordered_blocks
                ),
                is_synthetic=(
                    section.is_synthetic
                ),
            )
        )

        content_hash = hashlib.sha256(
            (
                f"{section.normalized_title}\n"
                f"{normalized_content}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return DetectedSection(
            section_id=section.section_id,
            parent_section_id=(
                section.parent_section_id
            ),
            section_index=(
                section.section_index
            ),
            sibling_index=(
                section.sibling_index
            ),
            level=section.level,
            section_type=(
                section.section_type
            ),
            title=section.title,
            normalized_title=(
                section.normalized_title
            ),
            title_search_text=(
                section.title_search_text
            ),
            numbering_prefix=(
                section.numbering_prefix
            ),
            start_page=section.start_page,
            end_page=section.end_page,
            start_block_index=(
                section.start_block_index
            ),
            end_block_index=(
                section.end_block_index
            ),
            content=content,
            normalized_content=(
                normalized_content
            ),
            searchable_content=(
                searchable_content
            ),
            blocks=tuple(
                ordered_blocks
            ),
            child_section_ids=tuple(
                section.child_section_ids
            ),
            word_count=word_count,
            character_count=(
                character_count
            ),
            block_count=len(
                ordered_blocks
            ),
            heading_confidence=round(
                section.heading_confidence,
                4,
            ),
            section_confidence=(
                section_confidence
            ),
            content_hash=content_hash,
            is_empty=is_empty,
            is_synthetic=(
                section.is_synthetic
            ),
            warnings=tuple(warnings),
            metadata=dict(
                section.metadata
            ),
        )

    def _classify_heading(
        self,
        *,
        text: str,
        numbering_prefix: str | None,
        numbering_depth: int | None,
    ) -> tuple[
        SectionType,
        int,
    ]:
        if self._TABLE_OF_CONTENTS_RE.match(
            text
        ):
            return (
                SectionType.TABLE_OF_CONTENTS,
                1,
            )

        if self._INDEX_RE.match(
            text
        ):
            return (
                SectionType.INDEX,
                1,
            )

        if self._APPENDIX_RE.match(
            text
        ):
            return (
                SectionType.APPENDIX,
                1,
            )

        if self._CHAPTER_RE.match(
            text
        ):
            return (
                SectionType.CHAPTER,
                1,
            )

        if numbering_depth is not None:
            level = min(
                max(
                    numbering_depth,
                    1,
                ),
                self.maximum_hierarchy_depth,
            )

            return (
                self._section_type_for_level(
                    level
                ),
                level,
            )

        return (
            SectionType.SECTION,
            1,
        )

    @staticmethod
    def _section_type_for_level(
        level: int,
    ) -> SectionType:
        if level <= 0:
            return SectionType.DOCUMENT_ROOT

        if level == 1:
            return SectionType.SECTION

        if level == 2:
            return SectionType.SUBSECTION

        return SectionType.SUBSUBSECTION

    @staticmethod
    def _numbering_depth(
        prefix: str,
    ) -> int:
        cleaned = prefix.strip(
            ".() "
        )

        if re.fullmatch(
            r"\d+(?:\.\d+)*",
            cleaned,
        ):
            return len(
                cleaned.split(".")
            )

        if re.fullmatch(
            r"[IVXLCDM]+",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return 1

        if re.fullmatch(
            r"[A-ZÇĞİÖŞÜ]",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return 1

        return 1

    @staticmethod
    def _convert_block(
        *,
        page: CleanedPageText,
        block: CleanedTextBlock,
        is_heading: bool,
    ) -> SectionContentBlock:
        return SectionContentBlock(
            page_index=page.page_index,
            page_number=page.page_number,
            block_index=block.block_index,
            source_block_index=(
                block.source_block_index
            ),
            text=block.text,
            normalized_text=(
                block.normalized_text
            ),
            searchable_text=(
                block.searchable_text
            ),
            block_type=block.block_type,
            is_heading=is_heading,
            word_count=block.word_count,
            character_count=(
                block.character_count
            ),
            content_hash=block.content_hash,
            metadata={
                **block.metadata,
                "page_cleaning_confidence": (
                    page.cleaning_confidence
                ),
            },
        )

    @staticmethod
    def _flatten_blocks(
        pages: Sequence[CleanedPageText],
    ) -> list[
        tuple[
            CleanedPageText,
            CleanedTextBlock,
        ]
    ]:
        flattened: list[
            tuple[
                CleanedPageText,
                CleanedTextBlock,
            ]
        ] = []

        for page in sorted(
            pages,
            key=lambda item: (
                item.page_number
            ),
        ):
            for block in sorted(
                page.blocks,
                key=lambda item: (
                    item.block_index
                ),
            ):
                flattened.append(
                    (
                        page,
                        block,
                    )
                )

        return flattened

    @classmethod
    def _normalize_heading_text(
        cls,
        text: str,
    ) -> str:
        normalized = text.replace(
            "\r",
            " "
        )

        normalized = normalized.replace(
            "\n",
            " "
        )

        normalized = cls._WHITESPACE_RE.sub(
            " ",
            normalized,
        ).strip()

        return normalized.strip(
            "•●▪◦‣·∙-–— "
        )

    @classmethod
    def _normalize_content(
        cls,
        text: str,
    ) -> str:
        paragraphs = [
            cls._WHITESPACE_RE.sub(
                " ",
                paragraph,
            ).strip()
            for paragraph in re.split(
                r"\n{2,}",
                text,
            )
        ]

        return "\n\n".join(
            paragraph
            for paragraph in paragraphs
            if paragraph
        )

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

        normalized = text.translate(
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

    @staticmethod
    def _uppercase_ratio(
        text: str,
    ) -> float:
        letters = [
            character
            for character in text
            if character.isalpha()
        ]

        if not letters:
            return 0.0

        return sum(
            character.isupper()
            for character in letters
        ) / len(letters)

    @staticmethod
    def _title_case_ratio(
        text: str,
    ) -> float:
        words = [
            word.strip(
                ".,;:!?()[]{}\"'"
            )
            for word in text.split()
        ]

        words = [
            word
            for word in words
            if any(
                character.isalpha()
                for character in word
            )
        ]

        if not words:
            return 0.0

        title_case_count = sum(
            word[0].isupper()
            for word in words
            if word
        )

        return (
            title_case_count
            / len(words)
        )

    @staticmethod
    def _calculate_section_confidence(
        *,
        heading_confidence: float,
        word_count: int,
        block_count: int,
        is_synthetic: bool,
    ) -> float:
        content_score = min(
            word_count / 100,
            1.0,
        )

        structure_score = min(
            block_count / 5,
            1.0,
        )

        confidence = (
            0.60 * heading_confidence
            + 0.25 * content_score
            + 0.15 * structure_score
        )

        if is_synthetic:
            confidence *= 0.75

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
    def _build_document_id(
        document: CleanedDocumentText,
    ) -> str:
        return hashlib.sha256(
            (
                f"cleaned-document:"
                f"{document.content_hash}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _calculate_sections_hash(
        sections: Sequence[DetectedSection],
    ) -> str:
        canonical = "\n".join(
            (
                f"{section.section_id}:"
                f"{section.parent_section_id}:"
                f"{section.level}:"
                f"{section.normalized_title}:"
                f"{section.content_hash}"
            )
            for section in sections
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

    def _empty_result(
        self,
        *,
        document_id: str,
        document: CleanedDocumentText,
    ) -> SectionDetectionResult:
        return SectionDetectionResult(
            document_id=document_id,
            section_count=0,
            root_section_count=0,
            maximum_depth=0,
            heading_candidate_count=0,
            rejected_heading_count=0,
            sections=(),
            hierarchy=(),
            unassigned_block_count=0,
            content_hash=hashlib.sha256(
                b""
            ).hexdigest(),
            warnings=(
                "Belgede bölüm tespiti için kullanılabilecek "
                "temiz metin bloğu bulunamadı.",
            ),
            metadata={
                "source_document_hash": (
                    document.content_hash
                ),
            },
        )


def detect_document_sections(
    document: CleanedDocumentText,
    *,
    document_id: str | None = None,
    document_title: str | None = None,
) -> SectionDetectionResult:
    """SectionDetector için kısa kullanım arayüzü."""

    detector = SectionDetector()

    return detector.detect_sections(
        document=document,
        document_id=document_id,
        document_title=document_title,
    )