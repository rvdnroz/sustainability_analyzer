from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


class NativeTextExtractionError(RuntimeError):
    """Native PDF metni çıkarılamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class TextBoundingBox:
    """PDF koordinat sistemindeki metin bölgesi."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(
            self.x1 - self.x0,
            0.0,
        )

    @property
    def height(self) -> float:
        return max(
            self.y1 - self.y0,
            0.0,
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": round(
                self.width,
                3,
            ),
            "height": round(
                self.height,
                3,
            ),
            "area": round(
                self.area,
                3,
            ),
        }


@dataclass(slots=True, frozen=True)
class NativeTextSpan:
    """PDF içindeki tek bir biçimlendirilmiş metin span'i."""

    text: str
    normalized_text: str

    bounding_box: TextBoundingBox

    font_name: str | None
    font_size: float
    font_flags: int

    color: int | None

    is_bold: bool
    is_italic: bool
    is_superscript: bool

    confidence: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bounding_box"] = (
            self.bounding_box.to_dict()
        )
        return data


@dataclass(slots=True, frozen=True)
class NativeTextLine:
    """Bir PDF metin bloğundaki tek satır."""

    line_index: int

    text: str
    normalized_text: str

    bounding_box: TextBoundingBox
    spans: tuple[NativeTextSpan, ...]

    dominant_font_size: float
    is_heading_candidate: bool
    is_empty: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_index": self.line_index,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "bounding_box": self.bounding_box.to_dict(),
            "spans": [
                span.to_dict()
                for span in self.spans
            ],
            "dominant_font_size": (
                self.dominant_font_size
            ),
            "is_heading_candidate": (
                self.is_heading_candidate
            ),
            "is_empty": self.is_empty,
        }


@dataclass(slots=True, frozen=True)
class NativeTextBlock:
    """Bir PDF sayfasındaki tek metin bloğu."""

    block_index: int

    text: str
    normalized_text: str

    bounding_box: TextBoundingBox
    lines: tuple[NativeTextLine, ...]

    block_type: str

    average_font_size: float
    maximum_font_size: float

    is_heading_candidate: bool
    is_header_candidate: bool
    is_footer_candidate: bool

    reading_order: int

    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "bounding_box": self.bounding_box.to_dict(),
            "lines": [
                line.to_dict()
                for line in self.lines
            ],
            "block_type": self.block_type,
            "average_font_size": (
                self.average_font_size
            ),
            "maximum_font_size": (
                self.maximum_font_size
            ),
            "is_heading_candidate": (
                self.is_heading_candidate
            ),
            "is_header_candidate": (
                self.is_header_candidate
            ),
            "is_footer_candidate": (
                self.is_footer_candidate
            ),
            "reading_order": self.reading_order,
            "content_hash": self.content_hash,
        }


@dataclass(slots=True, frozen=True)
class NativePageText:
    """Tek bir PDF sayfasının native metin çıkarım sonucu."""

    page_index: int
    page_number: int

    page_width: float
    page_height: float

    text: str
    normalized_text: str

    blocks: tuple[NativeTextBlock, ...]

    character_count: int
    word_count: int
    line_count: int
    block_count: int

    has_native_text: bool
    needs_ocr: bool

    detected_language_hint: str | None

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "blocks": [
                block.to_dict()
                for block in self.blocks
            ],
            "character_count": self.character_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "block_count": self.block_count,
            "has_native_text": self.has_native_text,
            "needs_ocr": self.needs_ocr,
            "detected_language_hint": (
                self.detected_language_hint
            ),
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class NativeDocumentText:
    """PDF belgesinin native metin çıkarım özeti."""

    pdf_path: Path
    page_count: int

    processed_page_count: int
    native_text_page_count: int
    ocr_required_page_count: int

    total_character_count: int
    total_word_count: int

    pages: tuple[NativePageText, ...]

    content_hash: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_path": str(self.pdf_path),
            "page_count": self.page_count,
            "processed_page_count": (
                self.processed_page_count
            ),
            "native_text_page_count": (
                self.native_text_page_count
            ),
            "ocr_required_page_count": (
                self.ocr_required_page_count
            ),
            "total_character_count": (
                self.total_character_count
            ),
            "total_word_count": (
                self.total_word_count
            ),
            "pages": [
                page.to_dict()
                for page in self.pages
            ],
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
        }


class NativeTextExtractor:
    """
    PDF sayfalarındaki yerleşik metni yapısal biçimde çıkarır.

    Çıkarılan bilgiler:

    - Sayfa metni
    - Metin blokları
    - Satırlar
    - Font span'leri
    - Font büyüklükleri
    - Başlık adayları
    - Header ve footer adayları
    - Okuma sırası
    - Native text / OCR gereksinimi

    Bu modül metin temizleme veya chunk üretme işlemi yapmaz.
    Çıktı daha sonra text_cleaner.py ve section_chunker.py
    modüllerine aktarılır.
    """

    _WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
    _MULTIPLE_NEWLINE_RE = re.compile(r"\n{3,}")

    _TURKISH_CHARACTERS = set(
        "çğıöşüÇĞİÖŞÜ"
    )

    _ENGLISH_COMMON_WORDS = {
        "the",
        "and",
        "of",
        "to",
        "in",
        "for",
        "with",
        "our",
        "sustainability",
        "report",
    }

    _TURKISH_COMMON_WORDS = {
        "ve",
        "bir",
        "bu",
        "ile",
        "için",
        "olarak",
        "sürdürülebilirlik",
        "raporu",
        "şirket",
        "yıl",
    }

    def __init__(
        self,
        *,
        minimum_native_characters: int = 40,
        minimum_native_words: int = 8,
        heading_font_ratio: float = 1.18,
        header_region_ratio: float = 0.08,
        footer_region_ratio: float = 0.08,
        maximum_heading_length: int = 180,
        preserve_line_breaks: bool = True,
        include_empty_blocks: bool = False,
        remove_image_blocks: bool = True,
        sort_reading_order: bool = True,
    ) -> None:
        if minimum_native_characters < 0:
            raise ValueError(
                "minimum_native_characters negatif olamaz."
            )

        if minimum_native_words < 0:
            raise ValueError(
                "minimum_native_words negatif olamaz."
            )

        if heading_font_ratio <= 1.0:
            raise ValueError(
                "heading_font_ratio 1.0'dan büyük olmalıdır."
            )

        for name, value in {
            "header_region_ratio": header_region_ratio,
            "footer_region_ratio": footer_region_ratio,
        }.items():
            if not 0.0 <= value <= 0.5:
                raise ValueError(
                    f"{name} 0 ile 0.5 arasında olmalıdır."
                )

        if maximum_heading_length < 1:
            raise ValueError(
                "maximum_heading_length pozitif olmalıdır."
            )

        self.minimum_native_characters = (
            minimum_native_characters
        )

        self.minimum_native_words = (
            minimum_native_words
        )

        self.heading_font_ratio = (
            heading_font_ratio
        )

        self.header_region_ratio = (
            header_region_ratio
        )

        self.footer_region_ratio = (
            footer_region_ratio
        )

        self.maximum_heading_length = (
            maximum_heading_length
        )

        self.preserve_line_breaks = (
            preserve_line_breaks
        )

        self.include_empty_blocks = (
            include_empty_blocks
        )

        self.remove_image_blocks = (
            remove_image_blocks
        )

        self.sort_reading_order = (
            sort_reading_order
        )

    def extract_pdf(
        self,
        pdf_path: str | Path,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> NativeDocumentText:
        """PDF içindeki seçili veya bütün sayfalardan native metni çıkarır."""

        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(
            pdf_path
        )

        try:
            document = pymupdf.open(
                pdf_path
            )
        except Exception as exc:
            raise NativeTextExtractionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise NativeTextExtractionError(
                    "Parola korumalı PDF analiz edilemez."
                )

            if document.page_count == 0:
                raise NativeTextExtractionError(
                    "PDF içinde sayfa bulunamadı."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            pages: list[
                NativePageText
            ] = []

            warnings: list[str] = []

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                try:
                    page_result = self.extract_page(
                        page
                    )

                except Exception as exc:
                    message = (
                        f"Sayfa {page_number} native metin "
                        f"çıkarımı başarısız: {exc}"
                    )

                    warnings.append(message)
                    logger.warning(message)

                    continue

                pages.append(
                    page_result
                )

            document_hash = self._calculate_document_hash(
                pages
            )

            result = NativeDocumentText(
                pdf_path=pdf_path,
                page_count=document.page_count,
                processed_page_count=len(
                    pages
                ),
                native_text_page_count=sum(
                    page.has_native_text
                    for page in pages
                ),
                ocr_required_page_count=sum(
                    page.needs_ocr
                    for page in pages
                ),
                total_character_count=sum(
                    page.character_count
                    for page in pages
                ),
                total_word_count=sum(
                    page.word_count
                    for page in pages
                ),
                pages=tuple(pages),
                content_hash=document_hash,
                warnings=tuple(warnings),
            )

            logger.info(
                "%d sayfadan %d native metin sayfası çıkarıldı: %s",
                result.processed_page_count,
                result.native_text_page_count,
                pdf_path,
            )

            return result

        finally:
            document.close()

    def extract_page(
        self,
        page: pymupdf.Page,
    ) -> NativePageText:
        """Açık bir PyMuPDF sayfasından native metni çıkarır."""

        page_index = int(
            page.number
        )

        page_number = (
            page_index + 1
        )

        page_width = float(
            page.rect.width
        )

        page_height = float(
            page.rect.height
        )

        warnings: list[str] = []

        try:
            raw_dictionary = page.get_text(
                "dict",
                sort=self.sort_reading_order,
            )

        except Exception as exc:
            raise NativeTextExtractionError(
                f"Sayfa {page_number} metin sözlüğü alınamadı."
            ) from exc

        raw_blocks = raw_dictionary.get(
            "blocks",
            []
        )

        font_sizes = self._collect_font_sizes(
            raw_blocks
        )

        body_font_size = (
            median(font_sizes)
            if font_sizes
            else 0.0
        )

        blocks: list[
            NativeTextBlock
        ] = []

        for block_index, raw_block in enumerate(
            raw_blocks
        ):
            block_type_code = int(
                raw_block.get(
                    "type",
                    0,
                )
            )

            if (
                block_type_code == 1
                and self.remove_image_blocks
            ):
                continue

            if block_type_code != 0:
                continue

            block = self._parse_text_block(
                raw_block=raw_block,
                block_index=block_index,
                page_rect=page.rect,
                body_font_size=body_font_size,
                page_height=page_height,
            )

            if block is None:
                continue

            if (
                not block.normalized_text
                and not self.include_empty_blocks
            ):
                continue

            blocks.append(
                block
            )

        if self.sort_reading_order:
            blocks = self._sort_blocks(
                blocks
            )

        blocks = [
            NativeTextBlock(
                block_index=block.block_index,
                text=block.text,
                normalized_text=block.normalized_text,
                bounding_box=block.bounding_box,
                lines=block.lines,
                block_type=block.block_type,
                average_font_size=block.average_font_size,
                maximum_font_size=block.maximum_font_size,
                is_heading_candidate=(
                    block.is_heading_candidate
                ),
                is_header_candidate=(
                    block.is_header_candidate
                ),
                is_footer_candidate=(
                    block.is_footer_candidate
                ),
                reading_order=reading_order,
                content_hash=block.content_hash,
            )
            for reading_order, block in enumerate(
                blocks
            )
        ]

        page_text = self._join_blocks(
            blocks
        )

        normalized_text = self._normalize_page_text(
            page_text
        )

        character_count = len(
            normalized_text
        )

        word_count = len(
            normalized_text.split()
        )

        line_count = sum(
            len(block.lines)
            for block in blocks
        )

        has_native_text = (
            character_count
            >= self.minimum_native_characters
            and word_count
            >= self.minimum_native_words
        )

        needs_ocr = not has_native_text

        if needs_ocr:
            warnings.append(
                "Native metin miktarı eşik altında; "
                "sayfa OCR için işaretlendi."
            )

        if not blocks:
            warnings.append(
                "Sayfada yerleşik metin bloğu bulunamadı."
            )

        content_hash = self._calculate_text_hash(
            normalized_text
        )

        return NativePageText(
            page_index=page_index,
            page_number=page_number,
            page_width=round(
                page_width,
                3,
            ),
            page_height=round(
                page_height,
                3,
            ),
            text=page_text,
            normalized_text=normalized_text,
            blocks=tuple(blocks),
            character_count=character_count,
            word_count=word_count,
            line_count=line_count,
            block_count=len(blocks),
            has_native_text=has_native_text,
            needs_ocr=needs_ocr,
            detected_language_hint=(
                self._detect_language_hint(
                    normalized_text
                )
            ),
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "body_font_size": round(
                    body_font_size,
                    3,
                ),
                "font_size_count": len(
                    font_sizes
                ),
                "raw_block_count": len(
                    raw_blocks
                ),
                "native_character_threshold": (
                    self.minimum_native_characters
                ),
                "native_word_threshold": (
                    self.minimum_native_words
                ),
            },
        )

    def _parse_text_block(
        self,
        *,
        raw_block: Mapping[str, Any],
        block_index: int,
        page_rect: pymupdf.Rect,
        body_font_size: float,
        page_height: float,
    ) -> NativeTextBlock | None:
        bounding_box = self._to_bounding_box(
            raw_block.get(
                "bbox"
            ),
            page_rect,
        )

        if bounding_box is None:
            return None

        raw_lines = raw_block.get(
            "lines",
            []
        )

        lines: list[
            NativeTextLine
        ] = []

        for line_index, raw_line in enumerate(
            raw_lines
        ):
            line = self._parse_text_line(
                raw_line=raw_line,
                line_index=line_index,
                page_rect=page_rect,
                body_font_size=body_font_size,
            )

            if line is None:
                continue

            lines.append(
                line
            )

        block_text = self._join_lines(
            lines
        )

        normalized_text = self._normalize_text(
            block_text
        )

        font_sizes = [
            span.font_size
            for line in lines
            for span in line.spans
            if span.font_size > 0
        ]

        average_font_size = (
            sum(font_sizes)
            / len(font_sizes)
            if font_sizes
            else 0.0
        )

        maximum_font_size = max(
            font_sizes,
            default=0.0,
        )

        is_heading_candidate = (
            self._is_heading_candidate(
                text=normalized_text,
                average_font_size=average_font_size,
                maximum_font_size=maximum_font_size,
                body_font_size=body_font_size,
                lines=lines,
            )
        )

        header_limit = (
            page_height
            * self.header_region_ratio
        )

        footer_limit = (
            page_height
            * (
                1.0
                - self.footer_region_ratio
            )
        )

        is_header_candidate = (
            bounding_box.y1
            <= header_limit
        )

        is_footer_candidate = (
            bounding_box.y0
            >= footer_limit
        )

        block_type = self._classify_block(
            text=normalized_text,
            is_heading=is_heading_candidate,
            is_header=is_header_candidate,
            is_footer=is_footer_candidate,
        )

        return NativeTextBlock(
            block_index=block_index,
            text=block_text,
            normalized_text=normalized_text,
            bounding_box=bounding_box,
            lines=tuple(lines),
            block_type=block_type,
            average_font_size=round(
                average_font_size,
                3,
            ),
            maximum_font_size=round(
                maximum_font_size,
                3,
            ),
            is_heading_candidate=(
                is_heading_candidate
            ),
            is_header_candidate=(
                is_header_candidate
            ),
            is_footer_candidate=(
                is_footer_candidate
            ),
            reading_order=block_index,
            content_hash=self._calculate_text_hash(
                normalized_text
            ),
        )

    def _parse_text_line(
        self,
        *,
        raw_line: Mapping[str, Any],
        line_index: int,
        page_rect: pymupdf.Rect,
        body_font_size: float,
    ) -> NativeTextLine | None:
        bounding_box = self._to_bounding_box(
            raw_line.get(
                "bbox"
            ),
            page_rect,
        )

        if bounding_box is None:
            return None

        spans: list[
            NativeTextSpan
        ] = []

        for raw_span in raw_line.get(
            "spans",
            []
        ):
            span = self._parse_text_span(
                raw_span=raw_span,
                page_rect=page_rect,
            )

            if span is not None:
                spans.append(
                    span
                )

        line_text = "".join(
            span.text
            for span in spans
        )

        normalized_text = self._normalize_text(
            line_text
        )

        font_sizes = [
            span.font_size
            for span in spans
            if span.font_size > 0
        ]

        dominant_font_size = max(
            font_sizes,
            default=0.0,
        )

        is_heading_candidate = (
            bool(normalized_text)
            and body_font_size > 0
            and dominant_font_size
            >= body_font_size
            * self.heading_font_ratio
            and len(normalized_text)
            <= self.maximum_heading_length
        )

        return NativeTextLine(
            line_index=line_index,
            text=line_text,
            normalized_text=normalized_text,
            bounding_box=bounding_box,
            spans=tuple(spans),
            dominant_font_size=round(
                dominant_font_size,
                3,
            ),
            is_heading_candidate=(
                is_heading_candidate
            ),
            is_empty=not bool(
                normalized_text
            ),
        )

    def _parse_text_span(
        self,
        *,
        raw_span: Mapping[str, Any],
        page_rect: pymupdf.Rect,
    ) -> NativeTextSpan | None:
        text = str(
            raw_span.get(
                "text",
                ""
            )
        )

        bounding_box = self._to_bounding_box(
            raw_span.get(
                "bbox"
            ),
            page_rect,
        )

        if bounding_box is None:
            return None

        font_flags = self._safe_int(
            raw_span.get(
                "flags",
                0,
            )
        )

        font_name = raw_span.get(
            "font"
        )

        font_size = self._safe_float(
            raw_span.get(
                "size",
                0.0,
            )
        )

        color = raw_span.get(
            "color"
        )

        is_superscript = bool(
            font_flags & 1
        )

        is_italic = bool(
            font_flags & 2
        )

        is_bold = bool(
            font_flags & 16
        )

        return NativeTextSpan(
            text=text,
            normalized_text=self._normalize_text(
                text
            ),
            bounding_box=bounding_box,
            font_name=(
                str(font_name)
                if font_name is not None
                else None
            ),
            font_size=round(
                font_size,
                3,
            ),
            font_flags=font_flags,
            color=(
                self._safe_int(color)
                if color is not None
                else None
            ),
            is_bold=is_bold,
            is_italic=is_italic,
            is_superscript=is_superscript,
            confidence=1.0,
        )

    def _is_heading_candidate(
        self,
        *,
        text: str,
        average_font_size: float,
        maximum_font_size: float,
        body_font_size: float,
        lines: Sequence[NativeTextLine],
    ) -> bool:
        if not text:
            return False

        if len(text) > self.maximum_heading_length:
            return False

        if body_font_size <= 0:
            return False

        font_condition = (
            maximum_font_size
            >= body_font_size
            * self.heading_font_ratio
        )

        bold_condition = any(
            span.is_bold
            for line in lines
            for span in line.spans
        )

        line_count_condition = (
            len(lines) <= 3
        )

        punctuation_condition = not text.endswith(
            (
                ".",
                ";",
                ",",
            )
        )

        uppercase_ratio = self._uppercase_ratio(
            text
        )

        case_condition = (
            uppercase_ratio >= 0.45
            or text.istitle()
        )

        return (
            line_count_condition
            and punctuation_condition
            and (
                font_condition
                or (
                    bold_condition
                    and case_condition
                    and average_font_size
                    >= body_font_size
                )
            )
        )

    @staticmethod
    def _classify_block(
        *,
        text: str,
        is_heading: bool,
        is_header: bool,
        is_footer: bool,
    ) -> str:
        if is_header:
            return "header"

        if is_footer:
            return "footer"

        if is_heading:
            return "heading"

        if text.isdigit():
            return "page_number"

        return "paragraph"

    def _join_lines(
        self,
        lines: Sequence[NativeTextLine],
    ) -> str:
        values = [
            line.text.strip()
            for line in lines
            if line.text.strip()
        ]

        separator = (
            "\n"
            if self.preserve_line_breaks
            else " "
        )

        return separator.join(
            values
        )

    def _join_blocks(
        self,
        blocks: Sequence[NativeTextBlock],
    ) -> str:
        values = [
            block.text.strip()
            for block in blocks
            if block.text.strip()
        ]

        return "\n\n".join(
            values
        )

    @classmethod
    def _normalize_text(
        cls,
        text: str,
    ) -> str:
        text = text.replace(
            "\u00ad",
            ""
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            cls._WHITESPACE_RE.sub(
                " ",
                line,
            ).strip()
            for line in text.splitlines()
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        return "\n".join(
            lines
        )

    @classmethod
    def _normalize_page_text(
        cls,
        text: str,
    ) -> str:
        normalized = cls._normalize_text(
            text
        )

        return cls._MULTIPLE_NEWLINE_RE.sub(
            "\n\n",
            normalized,
        ).strip()

    @staticmethod
    def _sort_blocks(
        blocks: Sequence[NativeTextBlock],
    ) -> list[NativeTextBlock]:
        return sorted(
            blocks,
            key=lambda block: (
                round(
                    block.bounding_box.y0,
                    1,
                ),
                round(
                    block.bounding_box.x0,
                    1,
                ),
            ),
        )

    @staticmethod
    def _collect_font_sizes(
        raw_blocks: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[float]:
        font_sizes: list[float] = []

        for block in raw_blocks:
            if int(
                block.get(
                    "type",
                    0,
                )
            ) != 0:
                continue

            for line in block.get(
                "lines",
                []
            ):
                for span in line.get(
                    "spans",
                    []
                ):
                    try:
                        font_size = float(
                            span.get(
                                "size",
                                0.0,
                            )
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if font_size > 0:
                        font_sizes.append(
                            font_size
                        )

        return font_sizes

    @classmethod
    def _detect_language_hint(
        cls,
        text: str,
    ) -> str | None:
        if not text.strip():
            return None

        lowered = text.casefold()

        words = {
            word.strip(
                ".,;:!?()[]{}\"'"
            )
            for word in lowered.split()
        }

        turkish_score = sum(
            word in cls._TURKISH_COMMON_WORDS
            for word in words
        )

        turkish_score += sum(
            character in cls._TURKISH_CHARACTERS
            for character in text
        )

        english_score = sum(
            word in cls._ENGLISH_COMMON_WORDS
            for word in words
        )

        if turkish_score > english_score:
            return "tr"

        if english_score > turkish_score:
            return "en"

        return None

    @staticmethod
    def _uppercase_ratio(
        text: str,
    ) -> float:
        alphabetic = [
            character
            for character in text
            if character.isalpha()
        ]

        if not alphabetic:
            return 0.0

        uppercase = sum(
            character.isupper()
            for character in alphabetic
        )

        return uppercase / len(
            alphabetic
        )

    @staticmethod
    def _to_bounding_box(
        value: Any,
        page_rect: pymupdf.Rect,
    ) -> TextBoundingBox | None:
        if value is None:
            return None

        try:
            rect = (
                pymupdf.Rect(value)
                & page_rect
            )
        except Exception:
            return None

        if rect.is_empty:
            return None

        if rect.is_infinite:
            return None

        if rect.width <= 0:
            return None

        if rect.height <= 0:
            return None

        return TextBoundingBox(
            x0=round(
                float(rect.x0),
                3,
            ),
            y0=round(
                float(rect.y0),
                3,
            ),
            x1=round(
                float(rect.x1),
                3,
            ),
            y1=round(
                float(rect.y1),
                3,
            ),
        )

    @staticmethod
    def _calculate_text_hash(
        text: str,
    ) -> str:
        return hashlib.sha256(
            text.encode(
                "utf-8"
            )
        ).hexdigest()

    @classmethod
    def _calculate_document_hash(
        cls,
        pages: Sequence[NativePageText],
    ) -> str:
        canonical = "\n\n".join(
            (
                f"PAGE:{page.page_number}\n"
                f"{page.normalized_text}"
            )
            for page in pages
        )

        return cls._calculate_text_hash(
            canonical
        )

    @staticmethod
    def _normalize_pages(
        *,
        page_numbers: Iterable[int] | None,
        page_count: int,
    ) -> list[int]:
        if page_numbers is None:
            return list(
                range(
                    1,
                    page_count + 1,
                )
            )

        pages = list(
            dict.fromkeys(
                int(page_number)
                for page_number in page_numbers
            )
        )

        if not pages:
            raise ValueError(
                "En az bir sayfa numarası verilmelidir."
            )

        invalid_pages = [
            page_number
            for page_number in pages
            if not 1
            <= page_number
            <= page_count
        ]

        if invalid_pages:
            raise ValueError(
                "Geçersiz sayfa numaraları: "
                f"{invalid_pages}. "
                f"Geçerli aralık 1-{page_count}."
            )

        return pages

    @staticmethod
    def _validate_pdf_path(
        pdf_path: Path,
    ) -> None:
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF bulunamadı: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Yalnızca .pdf uzantılı "
                "dosyalar desteklenir."
            )

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        try:
            return float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:
        try:
            return int(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0


def extract_native_text(
    pdf_path: str | Path,
    *,
    page_numbers: Iterable[int] | None = None,
) -> NativeDocumentText:
    """NativeTextExtractor için kısa kullanım arayüzü."""

    extractor = NativeTextExtractor()

    return extractor.extract_pdf(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
    )