from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)


class TextCleaningError(RuntimeError):
    """Metin temizleme işlemi tamamlanamadığında yükseltilir."""


class RemovedTextType(StrEnum):
    """Temizleme sırasında kaldırılan metin türü."""

    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    EMPTY = "empty"
    DUPLICATE = "duplicate"
    NOISE = "noise"
    ARTIFACT = "artifact"
    OTHER = "other"


@dataclass(slots=True, frozen=True)
class RemovedTextFragment:
    """Temizleme sırasında kaldırılan tek bir metin parçası."""

    page_number: int
    block_index: int | None
    line_index: int | None

    text: str
    normalized_text: str

    fragment_type: RemovedTextType
    reason: str

    confidence: float

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fragment_type"] = self.fragment_type.value
        return data


@dataclass(slots=True, frozen=True)
class CleanedTextBlock:
    """Temizleme işleminden geçmiş tek bir metin bloğu."""

    block_index: int
    source_block_index: int | None

    text: str
    normalized_text: str
    searchable_text: str

    block_type: str

    is_heading_candidate: bool
    is_header_candidate: bool
    is_footer_candidate: bool

    line_count: int
    word_count: int
    character_count: int

    content_hash: str

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class CleanedPageText:
    """Tek bir sayfanın temizlenmiş metin sonucu."""

    page_index: int
    page_number: int

    source_type: str

    text: str
    normalized_text: str
    searchable_text: str

    blocks: tuple[CleanedTextBlock, ...]

    character_count: int
    word_count: int
    line_count: int
    block_count: int

    removed_fragment_count: int
    removed_fragments: tuple[RemovedTextFragment, ...]

    source_confidence: float
    cleaning_confidence: float

    is_empty: bool
    is_valid: bool

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "source_type": self.source_type,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "searchable_text": self.searchable_text,
            "blocks": [
                block.to_dict()
                for block in self.blocks
            ],
            "character_count": self.character_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "block_count": self.block_count,
            "removed_fragment_count": (
                self.removed_fragment_count
            ),
            "removed_fragments": [
                fragment.to_dict()
                for fragment in self.removed_fragments
            ],
            "source_confidence": self.source_confidence,
            "cleaning_confidence": (
                self.cleaning_confidence
            ),
            "is_empty": self.is_empty,
            "is_valid": self.is_valid,
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class CleanedDocumentText:
    """Belge seviyesinde temizlenmiş metin sonucu."""

    page_count: int
    processed_page_count: int
    valid_page_count: int
    empty_page_count: int

    text: str
    normalized_text: str
    searchable_text: str

    pages: tuple[CleanedPageText, ...]

    repeated_header_candidates: tuple[str, ...]
    repeated_footer_candidates: tuple[str, ...]

    total_character_count: int
    total_word_count: int
    total_block_count: int

    removed_fragment_count: int

    content_hash: str
    warnings: tuple[str, ...]

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "processed_page_count": (
                self.processed_page_count
            ),
            "valid_page_count": self.valid_page_count,
            "empty_page_count": self.empty_page_count,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "searchable_text": self.searchable_text,
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
            "repeated_header_candidates": list(
                self.repeated_header_candidates
            ),
            "repeated_footer_candidates": list(
                self.repeated_footer_candidates
            ),
            "total_character_count": (
                self.total_character_count
            ),
            "total_word_count": self.total_word_count,
            "total_block_count": self.total_block_count,
            "removed_fragment_count": (
                self.removed_fragment_count
            ),
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class _SourceBlock:
    """Temizleme öncesi ortak iç blok temsili."""

    source_block_index: int | None
    text: str
    block_type: str

    is_heading_candidate: bool
    is_header_candidate: bool
    is_footer_candidate: bool

    metadata: dict[str, Any]


class TextCleaner:
    """
    Native ve OCR kaynaklı sayfa metinlerini ortak biçimde temizler.

    Temel görevler:

    - Unicode karakterlerini standardize etmek
    - Fazla boşlukları ve boş satırları temizlemek
    - Satır sonunda bölünmüş sözcükleri birleştirmek
    - Tekrarlanan header ve footer metinlerini kaldırmak
    - Tek başına bulunan sayfa numaralarını kaldırmak
    - OCR kaynaklı görünmez karakterleri temizlemek
    - Bozuk madde işaretlerini standardize etmek
    - Yinelenen bitişik satırları ve blokları azaltmak
    - Native ve OCR çıktılarını ortak veri yapısına dönüştürmek
    - Bölüm tespiti ve chunking için aranabilir metin üretmek

    Bu modül bölüm sınırı tespit etmez ve chunk üretmez. Çıktısı
    section_detector.py modülüne aktarılır.
    """

    _HORIZONTAL_WHITESPACE_RE = re.compile(
        r"[ \t\f\v]+"
    )

    _MULTIPLE_NEWLINE_RE = re.compile(
        r"\n{3,}"
    )

    _PAGE_NUMBER_RE = re.compile(
        r"""
        ^\s*
        (?:
            page|sayfa
        )?
        \s*
        (?:\d{1,4}|[ivxlcdm]{1,10})
        \s*
        (?:/|\sof\s|\s/\s)?
        \s*
        \d{0,4}
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _BULLET_RE = re.compile(
        r"""
        ^\s*
        (?:
            [•●▪◦‣·∙]
            |
            [*+]
            |
            [–—]
        )
        \s+
        """,
        re.VERBOSE,
    )

    _NUMBERED_LIST_RE = re.compile(
        r"""
        ^\s*
        (?P<prefix>
            (?:
                \d+(?:\.\d+)*
                |
                [a-zA-Z]
                |
                [ivxlcdmIVXLCDM]+
            )
            [.)]
        )
        \s+
        """,
        re.VERBOSE,
    )

    _LINE_END_HYPHEN_RE = re.compile(
        r"(?P<left>[A-Za-zÀ-ÖØ-öø-ÿÇĞİÖŞÜçğıöşü])"
        r"[-‐-‒–]\s*\n\s*"
        r"(?P<right>[a-zà-öø-ÿçğıöşü])"
    )

    _SOFT_HYPHEN_RE = re.compile(
        "\u00ad"
    )

    _ZERO_WIDTH_RE = re.compile(
        r"[\u200b\u200c\u200d\u2060\ufeff]"
    )

    _CONTROL_CHARACTER_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    _SEPARATOR_LINE_RE = re.compile(
        r"""
        ^\s*
        (?:
            [-_=*•·.]{4,}
            |
            [–—]{3,}
        )
        \s*$
        """,
        re.VERBOSE,
    )

    _OCR_GARBAGE_RE = re.compile(
        r"""
        ^\s*
        (?:
            [|Il1]{4,}
            |
            [^\w\s]{5,}
        )
        \s*$
        """,
        re.UNICODE | re.VERBOSE,
    )

    _SENTENCE_END_RE = re.compile(
        r"""[.!?…:;"')\]]$"""
    )

    def __init__(
        self,
        *,
        remove_repeated_headers: bool = True,
        remove_repeated_footers: bool = True,
        remove_page_numbers: bool = True,
        remove_separator_lines: bool = True,
        remove_ocr_garbage: bool = True,
        remove_duplicate_blocks: bool = True,
        remove_duplicate_lines: bool = True,
        join_hyphenated_words: bool = True,
        join_wrapped_lines: bool = True,
        preserve_paragraph_breaks: bool = True,
        preserve_headings: bool = True,
        normalize_bullets: bool = True,
        minimum_repeated_page_ratio: float = 0.30,
        minimum_repeated_page_count: int = 3,
        repeated_text_maximum_length: int = 180,
        minimum_page_characters: int = 20,
        minimum_page_words: int = 4,
        duplicate_similarity_threshold: float = 0.96,
    ) -> None:
        if not 0.0 <= minimum_repeated_page_ratio <= 1.0:
            raise ValueError(
                "minimum_repeated_page_ratio "
                "0 ile 1 arasında olmalıdır."
            )

        if minimum_repeated_page_count < 2:
            raise ValueError(
                "minimum_repeated_page_count en az 2 olmalıdır."
            )

        if repeated_text_maximum_length < 1:
            raise ValueError(
                "repeated_text_maximum_length pozitif olmalıdır."
            )

        if minimum_page_characters < 0:
            raise ValueError(
                "minimum_page_characters negatif olamaz."
            )

        if minimum_page_words < 0:
            raise ValueError(
                "minimum_page_words negatif olamaz."
            )

        if not 0.0 <= duplicate_similarity_threshold <= 1.0:
            raise ValueError(
                "duplicate_similarity_threshold "
                "0 ile 1 arasında olmalıdır."
            )

        self.remove_repeated_headers = (
            remove_repeated_headers
        )
        self.remove_repeated_footers = (
            remove_repeated_footers
        )
        self.remove_page_numbers = (
            remove_page_numbers
        )
        self.remove_separator_lines = (
            remove_separator_lines
        )
        self.remove_ocr_garbage = (
            remove_ocr_garbage
        )
        self.remove_duplicate_blocks = (
            remove_duplicate_blocks
        )
        self.remove_duplicate_lines = (
            remove_duplicate_lines
        )
        self.join_hyphenated_words = (
            join_hyphenated_words
        )
        self.join_wrapped_lines = (
            join_wrapped_lines
        )
        self.preserve_paragraph_breaks = (
            preserve_paragraph_breaks
        )
        self.preserve_headings = (
            preserve_headings
        )
        self.normalize_bullets = (
            normalize_bullets
        )

        self.minimum_repeated_page_ratio = (
            minimum_repeated_page_ratio
        )
        self.minimum_repeated_page_count = (
            minimum_repeated_page_count
        )
        self.repeated_text_maximum_length = (
            repeated_text_maximum_length
        )

        self.minimum_page_characters = (
            minimum_page_characters
        )
        self.minimum_page_words = (
            minimum_page_words
        )

        self.duplicate_similarity_threshold = (
            duplicate_similarity_threshold
        )

    def clean_document(
        self,
        pages: Sequence[Any],
    ) -> CleanedDocumentText:
        """
        NativePageText veya OCRPageResult nesnelerini belge seviyesinde
        temizler.

        Belge seviyesinde çalışmanın temel nedeni, tekrar eden header ve
        footer metinlerinin sayfalar arasında karşılaştırılarak bulunmasıdır.
        """

        if not pages:
            return CleanedDocumentText(
                page_count=0,
                processed_page_count=0,
                valid_page_count=0,
                empty_page_count=0,
                text="",
                normalized_text="",
                searchable_text="",
                pages=(),
                repeated_header_candidates=(),
                repeated_footer_candidates=(),
                total_character_count=0,
                total_word_count=0,
                total_block_count=0,
                removed_fragment_count=0,
                content_hash=self._calculate_hash(""),
                warnings=(
                    "Temizlenecek sayfa bulunamadı.",
                ),
                metadata={},
            )

        ordered_pages = sorted(
            pages,
            key=lambda page: int(
                getattr(
                    page,
                    "page_number",
                    0,
                )
            ),
        )

        header_candidates = (
            self._find_repeated_candidates(
                ordered_pages,
                region="header",
            )
        )

        footer_candidates = (
            self._find_repeated_candidates(
                ordered_pages,
                region="footer",
            )
        )

        cleaned_pages: list[
            CleanedPageText
        ] = []

        warnings: list[str] = []

        for page in ordered_pages:
            try:
                cleaned_page = self.clean_page(
                    page,
                    repeated_headers=header_candidates,
                    repeated_footers=footer_candidates,
                )

            except Exception as exc:
                page_number = int(
                    getattr(
                        page,
                        "page_number",
                        0,
                    )
                )

                message = (
                    f"Sayfa {page_number} metin temizleme "
                    f"işlemi başarısız: {exc}"
                )

                warnings.append(message)
                logger.warning(message)

                continue

            cleaned_pages.append(
                cleaned_page
            )

        document_text = self._join_page_texts(
            cleaned_pages
        )

        normalized_text = self.normalize_text(
            document_text,
            preserve_newlines=True,
        )

        searchable_text = self.normalize_for_search(
            normalized_text
        )

        return CleanedDocumentText(
            page_count=len(ordered_pages),
            processed_page_count=len(
                cleaned_pages
            ),
            valid_page_count=sum(
                page.is_valid
                for page in cleaned_pages
            ),
            empty_page_count=sum(
                page.is_empty
                for page in cleaned_pages
            ),
            text=document_text,
            normalized_text=normalized_text,
            searchable_text=searchable_text,
            pages=tuple(cleaned_pages),
            repeated_header_candidates=tuple(
                sorted(header_candidates)
            ),
            repeated_footer_candidates=tuple(
                sorted(footer_candidates)
            ),
            total_character_count=sum(
                page.character_count
                for page in cleaned_pages
            ),
            total_word_count=sum(
                page.word_count
                for page in cleaned_pages
            ),
            total_block_count=sum(
                page.block_count
                for page in cleaned_pages
            ),
            removed_fragment_count=sum(
                page.removed_fragment_count
                for page in cleaned_pages
            ),
            content_hash=self._calculate_hash(
                normalized_text
            ),
            warnings=tuple(warnings),
            metadata={
                "header_candidate_count": len(
                    header_candidates
                ),
                "footer_candidate_count": len(
                    footer_candidates
                ),
                "minimum_repeated_page_ratio": (
                    self.minimum_repeated_page_ratio
                ),
                "minimum_repeated_page_count": (
                    self.minimum_repeated_page_count
                ),
            },
        )

    def clean_page(
        self,
        page: Any,
        *,
        repeated_headers: Iterable[str] | None = None,
        repeated_footers: Iterable[str] | None = None,
    ) -> CleanedPageText:
        """Tek bir native veya OCR sayfa sonucunu temizler."""

        page_index = int(
            getattr(
                page,
                "page_index",
                0,
            )
        )

        page_number = int(
            getattr(
                page,
                "page_number",
                page_index + 1,
            )
        )

        source_type = self._detect_source_type(
            page
        )

        source_confidence = (
            self._extract_source_confidence(
                page
            )
        )

        repeated_header_keys = {
            self._canonical_repeated_text(
                value
            )
            for value in (
                repeated_headers or ()
            )
            if value
        }

        repeated_footer_keys = {
            self._canonical_repeated_text(
                value
            )
            for value in (
                repeated_footers or ()
            )
            if value
        }

        source_blocks = self._extract_source_blocks(
            page
        )

        removed_fragments: list[
            RemovedTextFragment
        ] = []

        cleaned_blocks: list[
            CleanedTextBlock
        ] = []

        seen_block_keys: list[str] = []

        for source_block in source_blocks:
            cleaned_result = self._clean_source_block(
                source_block=source_block,
                page_number=page_number,
                repeated_header_keys=(
                    repeated_header_keys
                ),
                repeated_footer_keys=(
                    repeated_footer_keys
                ),
            )

            block_text = cleaned_result["text"]

            removed_fragments.extend(
                cleaned_result[
                    "removed_fragments"
                ]
            )

            if not block_text:
                continue

            normalized_text = self.normalize_text(
                block_text,
                preserve_newlines=True,
            )

            if not normalized_text:
                continue

            block_key = self._canonical_repeated_text(
                normalized_text
            )

            if (
                self.remove_duplicate_blocks
                and self._contains_similar_text(
                    block_key,
                    seen_block_keys,
                )
            ):
                removed_fragments.append(
                    RemovedTextFragment(
                        page_number=page_number,
                        block_index=(
                            source_block.source_block_index
                        ),
                        line_index=None,
                        text=block_text,
                        normalized_text=(
                            normalized_text
                        ),
                        fragment_type=(
                            RemovedTextType.DUPLICATE
                        ),
                        reason=(
                            "Aynı sayfada yinelenen veya "
                            "çok benzer blok."
                        ),
                        confidence=0.92,
                        metadata={},
                    )
                )

                continue

            seen_block_keys.append(
                block_key
            )

            block_index = len(
                cleaned_blocks
            )

            cleaned_blocks.append(
                CleanedTextBlock(
                    block_index=block_index,
                    source_block_index=(
                        source_block.source_block_index
                    ),
                    text=block_text,
                    normalized_text=(
                        normalized_text
                    ),
                    searchable_text=(
                        self.normalize_for_search(
                            normalized_text
                        )
                    ),
                    block_type=(
                        source_block.block_type
                    ),
                    is_heading_candidate=(
                        source_block.is_heading_candidate
                    ),
                    is_header_candidate=False,
                    is_footer_candidate=False,
                    line_count=len(
                        normalized_text.splitlines()
                    ),
                    word_count=len(
                        normalized_text.split()
                    ),
                    character_count=len(
                        normalized_text
                    ),
                    content_hash=(
                        self._calculate_hash(
                            normalized_text
                        )
                    ),
                    metadata={
                        **source_block.metadata,
                        "source_type": source_type,
                    },
                )
            )

        page_text = self._join_cleaned_blocks(
            cleaned_blocks
        )

        normalized_text = self.normalize_text(
            page_text,
            preserve_newlines=True,
        )

        searchable_text = self.normalize_for_search(
            normalized_text
        )

        character_count = len(
            normalized_text
        )

        word_count = len(
            normalized_text.split()
        )

        line_count = len(
            [
                line
                for line in normalized_text.splitlines()
                if line.strip()
            ]
        )

        is_empty = not bool(
            normalized_text
        )

        is_valid = (
            character_count
            >= self.minimum_page_characters
            and word_count
            >= self.minimum_page_words
        )

        warnings: list[str] = []

        if is_empty:
            warnings.append(
                "Temizleme sonrasında sayfa metni boş kaldı."
            )

        elif not is_valid:
            warnings.append(
                "Temizlenmiş sayfa metni minimum içerik "
                "eşiklerini karşılamıyor."
            )

        cleaning_confidence = (
            self._calculate_cleaning_confidence(
                source_confidence=source_confidence,
                source_block_count=len(
                    source_blocks
                ),
                cleaned_block_count=len(
                    cleaned_blocks
                ),
                removed_fragment_count=len(
                    removed_fragments
                ),
                is_valid=is_valid,
            )
        )

        return CleanedPageText(
            page_index=page_index,
            page_number=page_number,
            source_type=source_type,
            text=page_text,
            normalized_text=normalized_text,
            searchable_text=searchable_text,
            blocks=tuple(cleaned_blocks),
            character_count=character_count,
            word_count=word_count,
            line_count=line_count,
            block_count=len(
                cleaned_blocks
            ),
            removed_fragment_count=len(
                removed_fragments
            ),
            removed_fragments=tuple(
                removed_fragments
            ),
            source_confidence=source_confidence,
            cleaning_confidence=(
                cleaning_confidence
            ),
            is_empty=is_empty,
            is_valid=is_valid,
            content_hash=self._calculate_hash(
                normalized_text
            ),
            warnings=tuple(warnings),
            metadata={
                "source_block_count": len(
                    source_blocks
                ),
                "repeated_header_count": len(
                    repeated_header_keys
                ),
                "repeated_footer_count": len(
                    repeated_footer_keys
                ),
            },
        )

    def clean_raw_text(
        self,
        text: str,
    ) -> str:
        """Bağımsız bir metin değerine temel temizleme uygular."""

        cleaned = self._remove_invisible_characters(
            text
        )

        cleaned = unicodedata.normalize(
            "NFKC",
            cleaned,
        )

        if self.join_hyphenated_words:
            cleaned = self._join_hyphenated_words(
                cleaned
            )

        lines = self._clean_lines(
            cleaned.splitlines(),
            page_number=0,
            block_index=None,
            is_heading=False,
        )["lines"]

        if self.join_wrapped_lines:
            cleaned = self._join_wrapped_lines(
                lines
            )
        else:
            cleaned = "\n".join(
                lines
            )

        return self.normalize_text(
            cleaned,
            preserve_newlines=True,
        )

    def _clean_source_block(
        self,
        *,
        source_block: _SourceBlock,
        page_number: int,
        repeated_header_keys: set[str],
        repeated_footer_keys: set[str],
    ) -> dict[str, Any]:
        removed_fragments: list[
            RemovedTextFragment
        ] = []

        text = self._remove_invisible_characters(
            source_block.text
        )

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        canonical_text = (
            self._canonical_repeated_text(
                text
            )
        )

        if (
            self.remove_repeated_headers
            and canonical_text
            in repeated_header_keys
        ):
            removed_fragments.append(
                RemovedTextFragment(
                    page_number=page_number,
                    block_index=(
                        source_block.source_block_index
                    ),
                    line_index=None,
                    text=text,
                    normalized_text=(
                        self.normalize_text(text)
                    ),
                    fragment_type=(
                        RemovedTextType.HEADER
                    ),
                    reason=(
                        "Birden fazla sayfada tekrar eden header."
                    ),
                    confidence=0.96,
                    metadata={},
                )
            )

            return {
                "text": "",
                "removed_fragments": (
                    removed_fragments
                ),
            }

        if (
            self.remove_repeated_footers
            and canonical_text
            in repeated_footer_keys
        ):
            removed_fragments.append(
                RemovedTextFragment(
                    page_number=page_number,
                    block_index=(
                        source_block.source_block_index
                    ),
                    line_index=None,
                    text=text,
                    normalized_text=(
                        self.normalize_text(text)
                    ),
                    fragment_type=(
                        RemovedTextType.FOOTER
                    ),
                    reason=(
                        "Birden fazla sayfada tekrar eden footer."
                    ),
                    confidence=0.96,
                    metadata={},
                )
            )

            return {
                "text": "",
                "removed_fragments": (
                    removed_fragments
                ),
            }

        if (
            self.remove_page_numbers
            and self._is_page_number_text(
                text
            )
        ):
            removed_fragments.append(
                RemovedTextFragment(
                    page_number=page_number,
                    block_index=(
                        source_block.source_block_index
                    ),
                    line_index=None,
                    text=text,
                    normalized_text=(
                        self.normalize_text(text)
                    ),
                    fragment_type=(
                        RemovedTextType.PAGE_NUMBER
                    ),
                    reason=(
                        "Tek başına bulunan sayfa numarası."
                    ),
                    confidence=0.98,
                    metadata={},
                )
            )

            return {
                "text": "",
                "removed_fragments": (
                    removed_fragments
                ),
            }

        if self.join_hyphenated_words:
            text = self._join_hyphenated_words(
                text
            )

        cleaned_line_result = self._clean_lines(
            text.splitlines(),
            page_number=page_number,
            block_index=(
                source_block.source_block_index
            ),
            is_heading=(
                source_block.is_heading_candidate
            ),
        )

        lines = cleaned_line_result["lines"]

        removed_fragments.extend(
            cleaned_line_result[
                "removed_fragments"
            ]
        )

        if self.join_wrapped_lines:
            cleaned_text = self._join_wrapped_lines(
                lines,
                is_heading=(
                    source_block.is_heading_candidate
                ),
            )
        else:
            cleaned_text = "\n".join(
                lines
            )

        cleaned_text = self.normalize_text(
            cleaned_text,
            preserve_newlines=True,
        )

        return {
            "text": cleaned_text,
            "removed_fragments": removed_fragments,
        }

    def _clean_lines(
        self,
        lines: Sequence[str],
        *,
        page_number: int,
        block_index: int | None,
        is_heading: bool,
    ) -> dict[str, Any]:
        cleaned_lines: list[str] = []

        removed_fragments: list[
            RemovedTextFragment
        ] = []

        seen_line_keys: list[str] = []

        for line_index, raw_line in enumerate(
            lines
        ):
            line = self.normalize_text(
                raw_line,
                preserve_newlines=False,
            )

            if not line:
                continue

            if (
                self.remove_separator_lines
                and self._SEPARATOR_LINE_RE.fullmatch(
                    line
                )
            ):
                removed_fragments.append(
                    RemovedTextFragment(
                        page_number=page_number,
                        block_index=block_index,
                        line_index=line_index,
                        text=raw_line,
                        normalized_text=line,
                        fragment_type=(
                            RemovedTextType.ARTIFACT
                        ),
                        reason=(
                            "Dekoratif ayırıcı satır."
                        ),
                        confidence=0.97,
                        metadata={},
                    )
                )
                continue

            if (
                self.remove_ocr_garbage
                and self._OCR_GARBAGE_RE.fullmatch(
                    line
                )
            ):
                removed_fragments.append(
                    RemovedTextFragment(
                        page_number=page_number,
                        block_index=block_index,
                        line_index=line_index,
                        text=raw_line,
                        normalized_text=line,
                        fragment_type=(
                            RemovedTextType.NOISE
                        ),
                        reason=(
                            "OCR kaynaklı anlamsız karakter dizisi."
                        ),
                        confidence=0.86,
                        metadata={},
                    )
                )
                continue

            if (
                self.remove_page_numbers
                and self._is_page_number_text(
                    line
                )
            ):
                removed_fragments.append(
                    RemovedTextFragment(
                        page_number=page_number,
                        block_index=block_index,
                        line_index=line_index,
                        text=raw_line,
                        normalized_text=line,
                        fragment_type=(
                            RemovedTextType.PAGE_NUMBER
                        ),
                        reason=(
                            "Satır düzeyinde sayfa numarası."
                        ),
                        confidence=0.96,
                        metadata={},
                    )
                )
                continue

            if self.normalize_bullets:
                line = self._normalize_list_marker(
                    line
                )

            line_key = (
                self._canonical_repeated_text(
                    line
                )
            )

            if (
                self.remove_duplicate_lines
                and not is_heading
                and self._contains_similar_text(
                    line_key,
                    seen_line_keys,
                )
            ):
                removed_fragments.append(
                    RemovedTextFragment(
                        page_number=page_number,
                        block_index=block_index,
                        line_index=line_index,
                        text=raw_line,
                        normalized_text=line,
                        fragment_type=(
                            RemovedTextType.DUPLICATE
                        ),
                        reason=(
                            "Aynı blok içinde yinelenen satır."
                        ),
                        confidence=0.92,
                        metadata={},
                    )
                )
                continue

            seen_line_keys.append(
                line_key
            )

            cleaned_lines.append(
                line
            )

        return {
            "lines": cleaned_lines,
            "removed_fragments": removed_fragments,
        }

    def _join_wrapped_lines(
        self,
        lines: Sequence[str],
        *,
        is_heading: bool = False,
    ) -> str:
        if not lines:
            return ""

        if is_heading:
            return " ".join(
                line.strip()
                for line in lines
                if line.strip()
            )

        paragraphs: list[str] = []
        current_parts: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if current_parts:
                    paragraphs.append(
                        " ".join(
                            current_parts
                        )
                    )
                    current_parts = []
                continue

            is_list_item = self._is_list_item(
                stripped
            )

            if is_list_item:
                if current_parts:
                    paragraphs.append(
                        " ".join(
                            current_parts
                        )
                    )
                    current_parts = []

                paragraphs.append(
                    stripped
                )
                continue

            if not current_parts:
                current_parts.append(
                    stripped
                )
                continue

            previous = current_parts[-1]

            if self._should_continue_paragraph(
                previous_line=previous,
                current_line=stripped,
            ):
                current_parts.append(
                    stripped
                )
            else:
                paragraphs.append(
                    " ".join(
                        current_parts
                    )
                )

                current_parts = [
                    stripped
                ]

        if current_parts:
            paragraphs.append(
                " ".join(
                    current_parts
                )
            )

        separator = (
            "\n\n"
            if self.preserve_paragraph_breaks
            else "\n"
        )

        return separator.join(
            paragraphs
        )

    def _should_continue_paragraph(
        self,
        *,
        previous_line: str,
        current_line: str,
    ) -> bool:
        if self._is_list_item(
            current_line
        ):
            return False

        if self._looks_like_heading_line(
            current_line
        ):
            return False

        if previous_line.endswith(
            (
                ":",
                ";",
            )
        ):
            return False

        if self._SENTENCE_END_RE.search(
            previous_line
        ):
            return False

        if (
            current_line
            and current_line[0].islower()
        ):
            return True

        if (
            current_line
            and current_line[0].isdigit()
            and not self._is_list_item(
                current_line
            )
        ):
            return True

        return len(previous_line) >= 40

    def _find_repeated_candidates(
        self,
        pages: Sequence[Any],
        *,
        region: str,
    ) -> set[str]:
        if region not in {
            "header",
            "footer",
        }:
            raise ValueError(
                "region yalnızca 'header' veya 'footer' olabilir."
            )

        page_candidate_sets: list[
            set[str]
        ] = []

        display_values: dict[
            str,
            str,
        ] = {}

        for page in pages:
            candidates: set[str] = set()

            for block in self._extract_source_blocks(
                page
            ):
                is_candidate = (
                    block.is_header_candidate
                    if region == "header"
                    else block.is_footer_candidate
                )

                if not is_candidate:
                    continue

                normalized = self.normalize_text(
                    block.text,
                    preserve_newlines=False,
                )

                if not normalized:
                    continue

                if (
                    len(normalized)
                    > self.repeated_text_maximum_length
                ):
                    continue

                canonical = (
                    self._canonical_repeated_text(
                        normalized
                    )
                )

                if not canonical:
                    continue

                candidates.add(
                    canonical
                )

                display_values.setdefault(
                    canonical,
                    normalized,
                )

            page_candidate_sets.append(
                candidates
            )

        counter: Counter[str] = Counter()

        for page_candidates in page_candidate_sets:
            counter.update(
                page_candidates
            )

        page_count = len(pages)

        required_count = max(
            self.minimum_repeated_page_count,
            int(
                page_count
                * self.minimum_repeated_page_ratio
                + 0.999
            ),
        )

        repeated_keys = {
            key
            for key, count in counter.items()
            if count >= required_count
        }

        return {
            display_values[key]
            for key in repeated_keys
        }

    def _extract_source_blocks(
        self,
        page: Any,
    ) -> list[_SourceBlock]:
        raw_blocks = getattr(
            page,
            "blocks",
            None,
        )

        source_blocks: list[
            _SourceBlock
        ] = []

        if raw_blocks:
            for fallback_index, block in enumerate(
                raw_blocks
            ):
                text = str(
                    getattr(
                        block,
                        "text",
                        "",
                    )
                    or getattr(
                        block,
                        "normalized_text",
                        "",
                    )
                    or ""
                )

                if not text.strip():
                    continue

                source_block_index = getattr(
                    block,
                    "block_index",
                    fallback_index,
                )

                block_type = str(
                    getattr(
                        block,
                        "block_type",
                        "paragraph",
                    )
                )

                bounding_box = (
                    getattr(
                        block,
                        "bounding_box",
                        None,
                    )
                    or getattr(
                        block,
                        "bounding_box_pdf",
                        None,
                    )
                )

                metadata: dict[str, Any] = {}

                if bounding_box is not None:
                    if hasattr(
                        bounding_box,
                        "to_dict",
                    ):
                        metadata[
                            "bounding_box"
                        ] = bounding_box.to_dict()

                    elif isinstance(
                        bounding_box,
                        Mapping,
                    ):
                        metadata[
                            "bounding_box"
                        ] = dict(
                            bounding_box
                        )

                source_blocks.append(
                    _SourceBlock(
                        source_block_index=int(
                            source_block_index
                        ),
                        text=text,
                        block_type=block_type,
                        is_heading_candidate=bool(
                            getattr(
                                block,
                                "is_heading_candidate",
                                block_type == "heading",
                            )
                        ),
                        is_header_candidate=bool(
                            getattr(
                                block,
                                "is_header_candidate",
                                block_type == "header",
                            )
                        ),
                        is_footer_candidate=bool(
                            getattr(
                                block,
                                "is_footer_candidate",
                                block_type == "footer",
                            )
                        ),
                        metadata=metadata,
                    )
                )

            return source_blocks

        page_text = str(
            getattr(
                page,
                "text",
                "",
            )
            or getattr(
                page,
                "normalized_text",
                "",
            )
            or ""
        )

        if page_text.strip():
            source_blocks.append(
                _SourceBlock(
                    source_block_index=0,
                    text=page_text,
                    block_type="paragraph",
                    is_heading_candidate=False,
                    is_header_candidate=False,
                    is_footer_candidate=False,
                    metadata={
                        "synthetic_block": True,
                    },
                )
            )

        return source_blocks

    @classmethod
    def normalize_text(
        cls,
        value: Any,
        *,
        preserve_newlines: bool = True,
    ) -> str:
        if value is None:
            return ""

        text = str(value)

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = cls._SOFT_HYPHEN_RE.sub(
            "",
            text,
        )

        text = cls._ZERO_WIDTH_RE.sub(
            "",
            text,
        )

        text = cls._CONTROL_CHARACTER_RE.sub(
            "",
            text,
        )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        if preserve_newlines:
            lines = [
                cls._HORIZONTAL_WHITESPACE_RE.sub(
                    " ",
                    line,
                ).strip()
                for line in text.splitlines()
            ]

            text = "\n".join(
                lines
            )

            text = cls._MULTIPLE_NEWLINE_RE.sub(
                "\n\n",
                text,
            )

            return text.strip()

        return cls._HORIZONTAL_WHITESPACE_RE.sub(
            " ",
            text.replace(
                "\n",
                " ",
            ),
        ).strip()

    @classmethod
    def normalize_for_search(
        cls,
        value: Any,
    ) -> str:
        text = cls.normalize_text(
            value,
            preserve_newlines=False,
        )

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

        text = text.translate(
            translation_table
        ).casefold()

        text = re.sub(
            r"[^\w%€$₺£.+\- ]",
            " ",
            text,
        )

        return cls._HORIZONTAL_WHITESPACE_RE.sub(
            " ",
            text,
        ).strip()

    @classmethod
    def _remove_invisible_characters(
        cls,
        text: str,
    ) -> str:
        cleaned = cls._SOFT_HYPHEN_RE.sub(
            "",
            text,
        )

        cleaned = cls._ZERO_WIDTH_RE.sub(
            "",
            cleaned,
        )

        return cls._CONTROL_CHARACTER_RE.sub(
            "",
            cleaned,
        )

    @classmethod
    def _join_hyphenated_words(
        cls,
        text: str,
    ) -> str:
        previous = None
        current = text

        while previous != current:
            previous = current

            current = cls._LINE_END_HYPHEN_RE.sub(
                lambda match: (
                    f"{match.group('left')}"
                    f"{match.group('right')}"
                ),
                current,
            )

        return current

    @classmethod
    def _normalize_list_marker(
        cls,
        line: str,
    ) -> str:
        if cls._BULLET_RE.match(
            line
        ):
            return cls._BULLET_RE.sub(
                "- ",
                line,
                count=1,
            )

        numbered_match = (
            cls._NUMBERED_LIST_RE.match(
                line
            )
        )

        if numbered_match:
            prefix = numbered_match.group(
                "prefix"
            )

            remainder = line[
                numbered_match.end():
            ].strip()

            return f"{prefix} {remainder}"

        return line

    @classmethod
    def _is_page_number_text(
        cls,
        text: str,
    ) -> bool:
        normalized = cls.normalize_text(
            text,
            preserve_newlines=False,
        )

        if len(normalized) > 30:
            return False

        return bool(
            cls._PAGE_NUMBER_RE.fullmatch(
                normalized
            )
        )

    @classmethod
    def _is_list_item(
        cls,
        line: str,
    ) -> bool:
        return bool(
            line.startswith("- ")
            or cls._NUMBERED_LIST_RE.match(
                line
            )
        )

    @classmethod
    def _looks_like_heading_line(
        cls,
        line: str,
    ) -> bool:
        normalized = cls.normalize_text(
            line,
            preserve_newlines=False,
        )

        if not normalized:
            return False

        if len(normalized) > 160:
            return False

        if normalized.endswith(
            (
                ".",
                ";",
                ",",
            )
        ):
            return False

        alphabetic = [
            character
            for character in normalized
            if character.isalpha()
        ]

        if not alphabetic:
            return False

        uppercase_ratio = sum(
            character.isupper()
            for character in alphabetic
        ) / len(alphabetic)

        return (
            normalized.istitle()
            or uppercase_ratio >= 0.60
        )

    def _contains_similar_text(
        self,
        candidate: str,
        existing_values: Sequence[str],
    ) -> bool:
        if not candidate:
            return False

        for existing in existing_values:
            similarity = self._token_similarity(
                candidate,
                existing,
            )

            if (
                similarity
                >= self.duplicate_similarity_threshold
            ):
                return True

        return False

    @staticmethod
    def _token_similarity(
        first: str,
        second: str,
    ) -> float:
        if first == second:
            return 1.0

        first_tokens = set(
            first.split()
        )

        second_tokens = set(
            second.split()
        )

        if not first_tokens or not second_tokens:
            return 0.0

        intersection = len(
            first_tokens
            & second_tokens
        )

        union = len(
            first_tokens
            | second_tokens
        )

        return (
            intersection
            / union
            if union
            else 0.0
        )

    @classmethod
    def _canonical_repeated_text(
        cls,
        value: Any,
    ) -> str:
        normalized = cls.normalize_for_search(
            value
        )

        # Header/footer içindeki değişken sayfa numaraları kaldırılır.
        normalized = re.sub(
            r"\b\d{1,4}\b",
            "<number>",
            normalized,
        )

        return cls._HORIZONTAL_WHITESPACE_RE.sub(
            " ",
            normalized,
        ).strip()

    def _join_cleaned_blocks(
        self,
        blocks: Sequence[CleanedTextBlock],
    ) -> str:
        values: list[str] = []

        for block in blocks:
            text = block.text.strip()

            if not text:
                continue

            if (
                self.preserve_headings
                and block.is_heading_candidate
            ):
                values.append(
                    text
                )
                continue

            values.append(
                text
            )

        return "\n\n".join(
            values
        )

    @staticmethod
    def _join_page_texts(
        pages: Sequence[CleanedPageText],
    ) -> str:
        return "\n\n".join(
            (
                f"<!-- PAGE {page.page_number} -->\n"
                f"{page.text.strip()}"
            )
            for page in pages
            if page.text.strip()
        )

    @staticmethod
    def _detect_source_type(
        page: Any,
    ) -> str:
        class_name = (
            page.__class__.__name__.casefold()
        )

        if "ocr" in class_name:
            return "ocr"

        if "native" in class_name:
            return "native"

        if hasattr(
            page,
            "average_confidence",
        ):
            return "ocr"

        if hasattr(
            page,
            "has_native_text",
        ):
            return "native"

        return "unknown"

    @staticmethod
    def _extract_source_confidence(
        page: Any,
    ) -> float:
        for attribute_name in (
            "average_confidence",
            "cleaning_confidence",
            "confidence",
        ):
            value = getattr(
                page,
                attribute_name,
                None,
            )

            if value is None:
                continue

            try:
                numeric_value = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if numeric_value > 1.0:
                numeric_value /= 100.0

            return round(
                min(
                    max(
                        numeric_value,
                        0.0,
                    ),
                    1.0,
                ),
                4,
            )

        if bool(
            getattr(
                page,
                "has_native_text",
                False,
            )
        ):
            return 1.0

        return 0.50

    @staticmethod
    def _calculate_cleaning_confidence(
        *,
        source_confidence: float,
        source_block_count: int,
        cleaned_block_count: int,
        removed_fragment_count: int,
        is_valid: bool,
    ) -> float:
        if source_block_count == 0:
            return 0.0

        retained_ratio = min(
            cleaned_block_count
            / source_block_count,
            1.0,
        )

        removal_ratio = min(
            removed_fragment_count
            / max(
                source_block_count
                + removed_fragment_count,
                1,
            ),
            1.0,
        )

        confidence = (
            0.50 * source_confidence
            + 0.30 * retained_ratio
            + 0.15 * (
                1.0
                - min(
                    removal_ratio,
                    0.80,
                )
            )
            + 0.05 * float(
                is_valid
            )
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
    def _calculate_hash(
        text: str,
    ) -> str:
        return hashlib.sha256(
            text.encode(
                "utf-8"
            )
        ).hexdigest()


def clean_document_text(
    pages: Sequence[Any],
) -> CleanedDocumentText:
    """TextCleaner için kısa kullanım arayüzü."""

    cleaner = TextCleaner()

    return cleaner.clean_document(
        pages
    )