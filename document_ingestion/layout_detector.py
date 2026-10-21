from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


class LayoutDetectionError(RuntimeError):
    """Sayfa layout analizi tamamlanamadığında yükseltilir."""


class LayoutElementType(StrEnum):
    """Sayfa üzerinde tespit edilebilecek içerik türleri."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"

    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"
    CAPTION = "caption"

    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"

    FORMULA = "formula"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class BoundingBox:
    """
    PDF koordinat sistemindeki bir dikdörtgen bölge.

    Koordinatlar point cinsindedir.
    PDF sayfalarında başlangıç noktası sol üst köşedir.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x0 + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y0 + self.height / 2

    def to_rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(
            self.x0,
            self.y0,
            self.x1,
            self.y1,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "area": round(self.area, 3),
        }


@dataclass(slots=True, frozen=True)
class LayoutElement:
    """Bir sayfa üzerinde tespit edilen içerik bölgesi."""

    element_id: str
    page_index: int
    page_number: int

    element_type: LayoutElementType
    bounding_box: BoundingBox

    text: str
    confidence: float

    reading_order: int

    font_size: float | None = None
    font_name: str | None = None

    is_bold: bool = False
    is_italic: bool = False

    source: str = "text_block"

    metadata: dict[str, Any] | None = None

    def to_dict(
        self,
        *,
        include_text: bool = True,
    ) -> dict[str, Any]:
        data = asdict(self)

        data["element_type"] = self.element_type.value
        data["bounding_box"] = self.bounding_box.to_dict()

        if not include_text:
            data.pop("text")
            data["text_preview"] = self.text[:300]

        return data


@dataclass(slots=True, frozen=True)
class PageLayout:
    """Tek bir sayfanın bütün layout analiz sonucu."""

    page_index: int
    page_number: int

    width_points: float
    height_points: float
    rotation: int

    column_count: int

    elements: tuple[LayoutElement, ...]

    text_element_count: int
    table_count: int
    image_count: int

    # Pipeline boyunca kaynak belge ve render dosyası bağlamını korur.
    source_pdf: Path | None = None
    image_path: Path | None = None

    def to_dict(
        self,
        *,
        include_text: bool = True,
    ) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "width_points": self.width_points,
            "height_points": self.height_points,
            "rotation": self.rotation,
            "column_count": self.column_count,
            "text_element_count": self.text_element_count,
            "table_count": self.table_count,
            "image_count": self.image_count,
            "source_pdf": (
                str(self.source_pdf)
                if self.source_pdf is not None
                else None
            ),
            "image_path": (
                str(self.image_path)
                if self.image_path is not None
                else None
            ),
            "elements": [
                element.to_dict(
                    include_text=include_text
                )
                for element in self.elements
            ],
        }


@dataclass(slots=True)
class _TextBlock:
    """Sınıflandırma öncesi dahili metin bloğu."""

    bounding_box: BoundingBox
    text: str

    font_size: float
    font_name: str | None

    is_bold: bool
    is_italic: bool

    line_count: int
    span_count: int


class LayoutDetector:
    """
    PyMuPDF kullanarak PDF sayfalarındaki layout bölgelerini tespit eder.

    Tespit edilen başlıca öğeler:

    - Başlık ve alt başlık
    - Paragraf
    - Liste maddesi
    - Tablo
    - Görsel ve şekil
    - Şekil veya tablo açıklaması
    - Header ve footer
    - Sayfa numarası

    Tespit yaklaşımı kural tabanlıdır. Sonraki aşamalarda LayoutLM,
    DocLayout-YOLO veya benzeri modellerle genişletilebilir.
    """

    _LIST_ITEM_RE = re.compile(
        r"""
        ^\s*
        (?:
            [•●▪◦‣⁃*-]
            |
            \d{1,3}[.)]
            |
            [a-zA-ZçğıöşüÇĞİÖŞÜ][.)]
            |
            [ivxlcdmIVXLCDM]+[.)]
        )
        \s+
        """,
        re.VERBOSE,
    )

    _PAGE_NUMBER_RE = re.compile(
        r"""
        ^\s*
        (?:
            page\s+
            |
            sayfa\s+
        )?
        \d{1,4}
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _CAPTION_RE = re.compile(
        r"""
        ^\s*
        (?:
            figure
            |
            fig\.?
            |
            şekil
            |
            chart
            |
            graph
            |
            grafik
            |
            table
            |
            tablo
        )
        \s*
        \d+
        (?:[.:)\-–—]|\s)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _FORMULA_RE = re.compile(
        r"""
        (?:
            [=∑∫√±≈≤≥]
            |
            \b[a-zA-Z]\s*=\s*
            |
            \b\d+\s*[+\-*/]\s*\d+
        )
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        *,
        header_ratio: float = 0.08,
        footer_ratio: float = 0.08,
        title_font_multiplier: float = 1.65,
        heading_font_multiplier: float = 1.22,
        minimum_text_length: int = 1,
        table_detection: bool = True,
        merge_adjacent_text_blocks: bool = True,
        merge_vertical_gap: float = 7.0,
        column_gap_ratio: float = 0.08,
    ) -> None:
        for name, value in {
            "header_ratio": header_ratio,
            "footer_ratio": footer_ratio,
            "column_gap_ratio": column_gap_ratio,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        if title_font_multiplier <= 1.0:
            raise ValueError(
                "title_font_multiplier 1.0'dan büyük olmalıdır."
            )

        if heading_font_multiplier <= 1.0:
            raise ValueError(
                "heading_font_multiplier 1.0'dan büyük olmalıdır."
            )

        if minimum_text_length < 0:
            raise ValueError(
                "minimum_text_length negatif olamaz."
            )

        if merge_vertical_gap < 0:
            raise ValueError(
                "merge_vertical_gap negatif olamaz."
            )

        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio
        self.title_font_multiplier = title_font_multiplier
        self.heading_font_multiplier = heading_font_multiplier
        self.minimum_text_length = minimum_text_length
        self.table_detection = table_detection
        self.merge_adjacent_text_blocks = (
            merge_adjacent_text_blocks
        )
        self.merge_vertical_gap = merge_vertical_gap
        self.column_gap_ratio = column_gap_ratio

    def detect_many(
        self,
        pages: Sequence[Any],
    ) -> list[PageLayout]:
        """
        IngestionPipeline'dan gelen sınıflandırılmış sayfaların layout
        analizini toplu olarak gerçekleştirir.

        Girdi olarak açık ``pymupdf.Page`` nesneleri veya ``source_pdf``
        ve ``page_number`` alanlarını taşıyan PageClassification benzeri
        nesneler kabul edilir. Aynı PDF'e ait sayfalarda belge yalnızca
        bir kez açılır.
        """
        page_items = list(pages)

        if not page_items:
            return []

        opened_documents: dict[Path, Any] = {}
        results: list[PageLayout] = []

        try:
            for item in page_items:
                if self._is_pymupdf_page(item):
                    results.append(self.detect_page(item))
                    continue

                source_pdf = self._resolve_source_pdf(item)
                page_number = self._resolve_page_number(item)
                image_path = self._resolve_image_path(item)

                document = opened_documents.get(source_pdf)

                if document is None:
                    document = self._open_document(source_pdf)
                    opened_documents[source_pdf] = document

                if not 1 <= page_number <= document.page_count:
                    raise LayoutDetectionError(
                        f"Geçersiz sayfa numarası: {page_number}. "
                        f"PDF sayfa aralığı 1-{document.page_count}."
                    )

                results.append(
                    self.detect_page(
                        document.load_page(page_number - 1),
                        source_pdf=source_pdf,
                        image_path=image_path,
                    )
                )

            logger.info(
                "%d sınıflandırılmış sayfanın layout analizi tamamlandı.",
                len(results),
            )

            return results
        finally:
            for document in opened_documents.values():
                document.close()

    def detect_layouts(
        self,
        pages: Sequence[Any],
    ) -> list[PageLayout]:
        """``detect_many`` için pipeline uyumluluk takma adı."""
        return self.detect_many(pages)

    def detect(
        self,
        page: Any,
    ) -> PageLayout:
        """
        Tek bir açık PyMuPDF sayfasını veya PageClassification benzeri
        bir kaydı analiz eder. Pipeline'ın tekil fallback çağrısıyla
        uyumludur.
        """
        if self._is_pymupdf_page(page):
            return self.detect_page(page)

        source_pdf = self._resolve_source_pdf(page)
        page_number = self._resolve_page_number(page)
        image_path = self._resolve_image_path(page)
        document = self._open_document(source_pdf)

        try:
            if not 1 <= page_number <= document.page_count:
                raise LayoutDetectionError(
                    f"Geçersiz sayfa numarası: {page_number}. "
                    f"PDF sayfa aralığı 1-{document.page_count}."
                )

            return self.detect_page(
                document.load_page(page_number - 1),
                source_pdf=source_pdf,
                image_path=image_path,
            )
        finally:
            document.close()

    def detect_pdf(
        self,
        pdf_path: str | Path,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[PageLayout]:
        """
        Bir PDF içerisindeki seçili veya bütün sayfaların layout'unu çıkarır.

        page_numbers değerleri 1 tabanlıdır.
        """

        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(pdf_path)

        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise LayoutDetectionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise LayoutDetectionError(
                    "Parola korumalı PDF analiz edilemez."
                )

            if document.page_count == 0:
                raise LayoutDetectionError(
                    "PDF içerisinde sayfa bulunamadı."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            results: list[PageLayout] = []

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                results.append(
                    self.detect_page(
                        page,
                        source_pdf=pdf_path,
                    )
                )

            logger.info(
                "%d sayfanın layout analizi tamamlandı: %s",
                len(results),
                pdf_path,
            )

            return results

        finally:
            document.close()

    def detect_page(
        self,
        page: pymupdf.Page,
        *,
        source_pdf: str | Path | None = None,
        image_path: str | Path | None = None,
    ) -> PageLayout:
        """Açık bir PyMuPDF sayfasındaki layout bölgelerini tespit eder."""

        page_number = int(page.number) + 1
        page_index = int(page.number)

        try:
            text_blocks = self._extract_text_blocks(
                page
            )

            table_boxes = self._extract_table_boxes(
                page
            )

            image_boxes = self._extract_image_boxes(
                page
            )

        except Exception as exc:
            raise LayoutDetectionError(
                f"Sayfa {page_number} analiz edilemedi."
            ) from exc

        if self.merge_adjacent_text_blocks:
            text_blocks = self._merge_text_blocks(
                text_blocks
            )

        body_font_size = self._estimate_body_font_size(
            text_blocks
        )

        text_elements = self._classify_text_blocks(
            page=page,
            blocks=text_blocks,
            body_font_size=body_font_size,
            table_boxes=table_boxes,
        )

        table_elements = self._create_table_elements(
            page=page,
            table_boxes=table_boxes,
        )

        image_elements = self._create_image_elements(
            page=page,
            image_boxes=image_boxes,
            text_elements=text_elements,
        )

        all_elements = [
            *text_elements,
            *table_elements,
            *image_elements,
        ]

        all_elements = self._remove_overlapping_unknowns(
            all_elements
        )

        column_count = self._estimate_column_count(
            elements=all_elements,
            page_width=float(page.rect.width),
        )

        ordered_elements = self._assign_reading_order(
            elements=all_elements,
            column_count=column_count,
            page_width=float(page.rect.width),
        )

        return PageLayout(
            page_index=page_index,
            page_number=page_number,
            width_points=round(
                float(page.rect.width),
                3,
            ),
            height_points=round(
                float(page.rect.height),
                3,
            ),
            rotation=int(page.rotation),
            column_count=column_count,
            elements=tuple(ordered_elements),
            text_element_count=sum(
                element.source == "text_block"
                for element in ordered_elements
            ),
            table_count=sum(
                element.element_type
                == LayoutElementType.TABLE
                for element in ordered_elements
            ),
            image_count=sum(
                element.element_type
                in {
                    LayoutElementType.IMAGE,
                    LayoutElementType.FIGURE,
                }
                for element in ordered_elements
            ),
            source_pdf=(
                Path(source_pdf).expanduser().resolve()
                if source_pdf is not None
                else None
            ),
            image_path=(
                Path(image_path).expanduser().resolve()
                if image_path is not None
                else None
            ),
        )

    def _extract_text_blocks(
        self,
        page: pymupdf.Page,
    ) -> list[_TextBlock]:
        page_dict = page.get_text(
            "dict",
            sort=True,
        )

        blocks: list[_TextBlock] = []

        for block in page_dict.get(
            "blocks",
            [],
        ):
            if int(block.get("type", -1)) != 0:
                continue

            lines = block.get(
                "lines",
                [],
            )

            text_parts: list[str] = []
            font_sizes: list[float] = []
            font_names: list[str] = []

            is_bold = False
            is_italic = False
            span_count = 0

            for line in lines:
                line_parts: list[str] = []

                for span in line.get(
                    "spans",
                    [],
                ):
                    span_text = str(
                        span.get("text", "")
                    )

                    if not span_text:
                        continue

                    line_parts.append(span_text)

                    size = float(
                        span.get("size", 0.0)
                    )

                    if size > 0:
                        font_sizes.append(size)

                    font_name = str(
                        span.get("font", "")
                    ).strip()

                    if font_name:
                        font_names.append(font_name)

                    flags = int(
                        span.get("flags", 0)
                    )

                    font_name_lower = (
                        font_name.lower()
                    )

                    is_bold = (
                        is_bold
                        or "bold" in font_name_lower
                        or "black" in font_name_lower
                        or bool(flags & 16)
                    )

                    is_italic = (
                        is_italic
                        or "italic" in font_name_lower
                        or "oblique" in font_name_lower
                        or bool(flags & 2)
                    )

                    span_count += 1

                line_text = "".join(
                    line_parts
                ).strip()

                if line_text:
                    text_parts.append(line_text)

            text = "\n".join(
                text_parts
            ).strip()

            if len(text) < self.minimum_text_length:
                continue

            bounding_box = self._to_bounding_box(
                block.get("bbox"),
                page.rect,
            )

            if bounding_box is None:
                continue

            average_font_size = (
                sum(font_sizes) / len(font_sizes)
                if font_sizes
                else 0.0
            )

            most_common_font = (
                max(
                    set(font_names),
                    key=font_names.count,
                )
                if font_names
                else None
            )

            blocks.append(
                _TextBlock(
                    bounding_box=bounding_box,
                    text=text,
                    font_size=round(
                        average_font_size,
                        3,
                    ),
                    font_name=most_common_font,
                    is_bold=is_bold,
                    is_italic=is_italic,
                    line_count=len(text_parts),
                    span_count=span_count,
                )
            )

        return blocks

    def _extract_table_boxes(
        self,
        page: pymupdf.Page,
    ) -> list[BoundingBox]:
        if not self.table_detection:
            return []

        if not hasattr(
            page,
            "find_tables",
        ):
            logger.warning(
                "PyMuPDF sürümünde Page.find_tables bulunmuyor."
            )

            return []

        try:
            finder = page.find_tables()
        except Exception as exc:
            logger.debug(
                "Sayfa %d tablo tespiti başarısız: %s",
                int(page.number) + 1,
                exc,
            )

            return []

        boxes: list[BoundingBox] = []

        for table in getattr(
            finder,
            "tables",
            [],
        ):
            bounding_box = self._to_bounding_box(
                getattr(
                    table,
                    "bbox",
                    None,
                ),
                page.rect,
            )

            if bounding_box is not None:
                boxes.append(
                    bounding_box
                )

        return self._deduplicate_boxes(
            boxes
        )

    def _extract_image_boxes(
        self,
        page: pymupdf.Page,
    ) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []

        try:
            image_info = page.get_image_info(
                xrefs=False
            )
        except Exception:
            image_info = []

        for image in image_info:
            bounding_box = self._to_bounding_box(
                image.get("bbox"),
                page.rect,
            )

            if bounding_box is not None:
                boxes.append(
                    bounding_box
                )

        if not boxes:
            raw_blocks = page.get_text(
                "blocks",
                sort=False,
            )

            for block in raw_blocks:
                if len(block) < 7:
                    continue

                if int(block[6]) != 1:
                    continue

                bounding_box = self._to_bounding_box(
                    block[:4],
                    page.rect,
                )

                if bounding_box is not None:
                    boxes.append(
                        bounding_box
                    )

        return self._deduplicate_boxes(
            boxes
        )

    def _classify_text_blocks(
        self,
        *,
        page: pymupdf.Page,
        blocks: Sequence[_TextBlock],
        body_font_size: float,
        table_boxes: Sequence[BoundingBox],
    ) -> list[LayoutElement]:
        elements: list[LayoutElement] = []

        page_height = float(
            page.rect.height
        )

        header_limit = (
            page_height * self.header_ratio
        )

        footer_limit = (
            page_height
            * (1.0 - self.footer_ratio)
        )

        for index, block in enumerate(blocks):
            element_type, confidence = (
                self._classify_text_block(
                    block=block,
                    body_font_size=body_font_size,
                    header_limit=header_limit,
                    footer_limit=footer_limit,
                    table_boxes=table_boxes,
                )
            )

            elements.append(
                LayoutElement(
                    element_id=(
                        f"p{int(page.number) + 1}"
                        f"_text_{index + 1}"
                    ),
                    page_index=int(page.number),
                    page_number=int(page.number) + 1,
                    element_type=element_type,
                    bounding_box=block.bounding_box,
                    text=block.text,
                    confidence=confidence,
                    reading_order=-1,
                    font_size=block.font_size,
                    font_name=block.font_name,
                    is_bold=block.is_bold,
                    is_italic=block.is_italic,
                    source="text_block",
                    metadata={
                        "line_count": block.line_count,
                        "span_count": block.span_count,
                    },
                )
            )

        return elements

    def _classify_text_block(
        self,
        *,
        block: _TextBlock,
        body_font_size: float,
        header_limit: float,
        footer_limit: float,
        table_boxes: Sequence[BoundingBox],
    ) -> tuple[LayoutElementType, float]:
        text = block.text.strip()
        single_line = block.line_count == 1

        if self._PAGE_NUMBER_RE.fullmatch(text):
            if (
                block.bounding_box.y0 <= header_limit
                or block.bounding_box.y1 >= footer_limit
            ):
                return (
                    LayoutElementType.PAGE_NUMBER,
                    0.97,
                )

        if block.bounding_box.y1 <= header_limit:
            return (
                LayoutElementType.HEADER,
                0.82,
            )

        if block.bounding_box.y0 >= footer_limit:
            return (
                LayoutElementType.FOOTER,
                0.82,
            )

        if self._CAPTION_RE.match(text):
            return (
                LayoutElementType.CAPTION,
                0.95,
            )

        if self._LIST_ITEM_RE.match(text):
            return (
                LayoutElementType.LIST_ITEM,
                0.91,
            )

        if self._FORMULA_RE.search(text):
            symbol_ratio = self._formula_symbol_ratio(
                text
            )

            if symbol_ratio >= 0.08:
                return (
                    LayoutElementType.FORMULA,
                    0.75,
                )

        if self._is_inside_any_box(
            block.bounding_box,
            table_boxes,
            minimum_overlap=0.60,
        ):
            return (
                LayoutElementType.UNKNOWN,
                0.55,
            )

        if body_font_size > 0:
            font_ratio = (
                block.font_size
                / body_font_size
            )
        else:
            font_ratio = 1.0

        short_text = len(text) <= 180
        very_short_text = len(text) <= 90

        if (
            font_ratio
            >= self.title_font_multiplier
            and very_short_text
        ):
            return (
                LayoutElementType.TITLE,
                min(
                    0.99,
                    0.72
                    + 0.12
                    * min(
                        font_ratio - 1.0,
                        2.0,
                    ),
                ),
            )

        if (
            font_ratio
            >= self.heading_font_multiplier
            and short_text
        ):
            return (
                LayoutElementType.HEADING,
                min(
                    0.96,
                    0.70
                    + 0.12
                    * min(
                        font_ratio - 1.0,
                        1.5,
                    ),
                ),
            )

        if (
            block.is_bold
            and single_line
            and short_text
        ):
            return (
                LayoutElementType.HEADING,
                0.78,
            )

        if (
            single_line
            and very_short_text
            and text.isupper()
            and len(text.split()) <= 12
        ):
            return (
                LayoutElementType.HEADING,
                0.73,
            )

        return (
            LayoutElementType.PARAGRAPH,
            0.84,
        )

    def _create_table_elements(
        self,
        *,
        page: pymupdf.Page,
        table_boxes: Sequence[BoundingBox],
    ) -> list[LayoutElement]:
        elements: list[LayoutElement] = []

        for index, bounding_box in enumerate(
            table_boxes
        ):
            elements.append(
                LayoutElement(
                    element_id=(
                        f"p{int(page.number) + 1}"
                        f"_table_{index + 1}"
                    ),
                    page_index=int(page.number),
                    page_number=int(page.number) + 1,
                    element_type=LayoutElementType.TABLE,
                    bounding_box=bounding_box,
                    text="",
                    confidence=0.90,
                    reading_order=-1,
                    source="table_detector",
                    metadata=None,
                )
            )

        return elements

    def _create_image_elements(
        self,
        *,
        page: pymupdf.Page,
        image_boxes: Sequence[BoundingBox],
        text_elements: Sequence[LayoutElement],
    ) -> list[LayoutElement]:
        elements: list[LayoutElement] = []

        caption_elements = [
            element
            for element in text_elements
            if element.element_type
            == LayoutElementType.CAPTION
        ]

        for index, bounding_box in enumerate(
            image_boxes
        ):
            has_nearby_caption = any(
                self._is_caption_near_box(
                    caption.bounding_box,
                    bounding_box,
                )
                for caption in caption_elements
            )

            element_type = (
                LayoutElementType.FIGURE
                if has_nearby_caption
                else LayoutElementType.IMAGE
            )

            confidence = (
                0.90
                if has_nearby_caption
                else 0.82
            )

            elements.append(
                LayoutElement(
                    element_id=(
                        f"p{int(page.number) + 1}"
                        f"_image_{index + 1}"
                    ),
                    page_index=int(page.number),
                    page_number=int(page.number) + 1,
                    element_type=element_type,
                    bounding_box=bounding_box,
                    text="",
                    confidence=confidence,
                    reading_order=-1,
                    source="image_block",
                    metadata={
                        "has_nearby_caption": (
                            has_nearby_caption
                        )
                    },
                )
            )

        return elements

    def _estimate_body_font_size(
        self,
        blocks: Sequence[_TextBlock],
    ) -> float:
        candidate_sizes: list[float] = []

        for block in blocks:
            if block.font_size <= 0:
                continue

            repeat_count = max(
                len(block.text),
                1,
            )

            repeat_count = min(
                repeat_count,
                500,
            )

            candidate_sizes.extend(
                [block.font_size] * repeat_count
            )

        if not candidate_sizes:
            return 0.0

        return round(
            float(
                median(candidate_sizes)
            ),
            3,
        )

    def _merge_text_blocks(
        self,
        blocks: Sequence[_TextBlock],
    ) -> list[_TextBlock]:
        if len(blocks) < 2:
            return list(blocks)

        sorted_blocks = sorted(
            blocks,
            key=lambda block: (
                round(
                    block.bounding_box.x0,
                    1,
                ),
                block.bounding_box.y0,
            ),
        )

        merged: list[_TextBlock] = []

        for block in sorted_blocks:
            if not merged:
                merged.append(block)
                continue

            previous = merged[-1]

            vertical_gap = (
                block.bounding_box.y0
                - previous.bounding_box.y1
            )

            horizontal_overlap = (
                self._horizontal_overlap_ratio(
                    previous.bounding_box,
                    block.bounding_box,
                )
            )

            font_difference = abs(
                previous.font_size
                - block.font_size
            )

            can_merge = (
                0 <= vertical_gap
                <= self.merge_vertical_gap
                and horizontal_overlap >= 0.70
                and font_difference <= 0.8
                and previous.is_bold == block.is_bold
                and previous.is_italic == block.is_italic
                and not self._CAPTION_RE.match(
                    previous.text
                )
                and not self._CAPTION_RE.match(
                    block.text
                )
                and not self._LIST_ITEM_RE.match(
                    block.text
                )
            )

            if not can_merge:
                merged.append(block)
                continue

            merged_box = BoundingBox(
                x0=min(
                    previous.bounding_box.x0,
                    block.bounding_box.x0,
                ),
                y0=min(
                    previous.bounding_box.y0,
                    block.bounding_box.y0,
                ),
                x1=max(
                    previous.bounding_box.x1,
                    block.bounding_box.x1,
                ),
                y1=max(
                    previous.bounding_box.y1,
                    block.bounding_box.y1,
                ),
            )

            merged[-1] = _TextBlock(
                bounding_box=merged_box,
                text=(
                    f"{previous.text}\n"
                    f"{block.text}"
                ),
                font_size=round(
                    (
                        previous.font_size
                        + block.font_size
                    )
                    / 2,
                    3,
                ),
                font_name=(
                    previous.font_name
                    or block.font_name
                ),
                is_bold=previous.is_bold,
                is_italic=previous.is_italic,
                line_count=(
                    previous.line_count
                    + block.line_count
                ),
                span_count=(
                    previous.span_count
                    + block.span_count
                ),
            )

        return merged

    def _estimate_column_count(
        self,
        *,
        elements: Sequence[LayoutElement],
        page_width: float,
    ) -> int:
        text_elements = [
            element
            for element in elements
            if element.element_type
            in {
                LayoutElementType.PARAGRAPH,
                LayoutElementType.LIST_ITEM,
                LayoutElementType.HEADING,
            }
            and element.bounding_box.width
            < page_width * 0.80
        ]

        if len(text_elements) < 4:
            return 1

        center_values = sorted(
            element.bounding_box.center_x
            for element in text_elements
        )

        minimum_gap = (
            page_width
            * self.column_gap_ratio
        )

        significant_gaps = [
            right - left
            for left, right in zip(
                center_values,
                center_values[1:],
                strict=False,
            )
            if right - left >= minimum_gap
        ]

        if not significant_gaps:
            return 1

        midpoint = page_width / 2

        left_count = sum(
            center < midpoint
            for center in center_values
        )

        right_count = sum(
            center >= midpoint
            for center in center_values
        )

        if left_count >= 2 and right_count >= 2:
            return 2

        return 1

    def _assign_reading_order(
        self,
        *,
        elements: Sequence[LayoutElement],
        column_count: int,
        page_width: float,
    ) -> list[LayoutElement]:
        if column_count <= 1:
            sorted_elements = sorted(
                elements,
                key=lambda element: (
                    element.bounding_box.y0,
                    element.bounding_box.x0,
                ),
            )

        else:
            midpoint = page_width / 2

            full_width_elements = [
                element
                for element in elements
                if element.bounding_box.width
                >= page_width * 0.72
            ]

            column_elements = [
                element
                for element in elements
                if element not in full_width_elements
            ]

            left_column = sorted(
                (
                    element
                    for element in column_elements
                    if element.bounding_box.center_x
                    < midpoint
                ),
                key=lambda element: (
                    element.bounding_box.y0,
                    element.bounding_box.x0,
                ),
            )

            right_column = sorted(
                (
                    element
                    for element in column_elements
                    if element.bounding_box.center_x
                    >= midpoint
                ),
                key=lambda element: (
                    element.bounding_box.y0,
                    element.bounding_box.x0,
                ),
            )

            full_width_elements = sorted(
                full_width_elements,
                key=lambda element: (
                    element.bounding_box.y0,
                    element.bounding_box.x0,
                ),
            )

            sorted_elements = self._combine_column_order(
                full_width_elements=full_width_elements,
                left_column=left_column,
                right_column=right_column,
            )

        ordered: list[LayoutElement] = []

        for reading_order, element in enumerate(
            sorted_elements
        ):
            ordered.append(
                LayoutElement(
                    element_id=element.element_id,
                    page_index=element.page_index,
                    page_number=element.page_number,
                    element_type=element.element_type,
                    bounding_box=element.bounding_box,
                    text=element.text,
                    confidence=element.confidence,
                    reading_order=reading_order,
                    font_size=element.font_size,
                    font_name=element.font_name,
                    is_bold=element.is_bold,
                    is_italic=element.is_italic,
                    source=element.source,
                    metadata=element.metadata,
                )
            )

        return ordered

    @staticmethod
    def _combine_column_order(
        *,
        full_width_elements: Sequence[LayoutElement],
        left_column: Sequence[LayoutElement],
        right_column: Sequence[LayoutElement],
    ) -> list[LayoutElement]:
        if not full_width_elements:
            return [
                *left_column,
                *right_column,
            ]

        ordered: list[LayoutElement] = []

        remaining_left = list(left_column)
        remaining_right = list(right_column)

        for full_width in full_width_elements:
            before_left = [
                element
                for element in remaining_left
                if element.bounding_box.y0
                < full_width.bounding_box.y0
            ]

            before_right = [
                element
                for element in remaining_right
                if element.bounding_box.y0
                < full_width.bounding_box.y0
            ]

            ordered.extend(before_left)
            ordered.extend(before_right)
            ordered.append(full_width)

            remaining_left = [
                element
                for element in remaining_left
                if element not in before_left
            ]

            remaining_right = [
                element
                for element in remaining_right
                if element not in before_right
            ]

        ordered.extend(remaining_left)
        ordered.extend(remaining_right)

        return ordered

    def _remove_overlapping_unknowns(
        self,
        elements: Sequence[LayoutElement],
    ) -> list[LayoutElement]:
        result: list[LayoutElement] = []

        table_elements = [
            element
            for element in elements
            if element.element_type
            == LayoutElementType.TABLE
        ]

        for element in elements:
            if (
                element.element_type
                == LayoutElementType.UNKNOWN
                and self._is_inside_any_box(
                    element.bounding_box,
                    [
                        table.bounding_box
                        for table in table_elements
                    ],
                    minimum_overlap=0.60,
                )
            ):
                continue

            result.append(element)

        return result

    @staticmethod
    def _is_caption_near_box(
        caption_box: BoundingBox,
        content_box: BoundingBox,
    ) -> bool:
        vertical_distance = min(
            abs(
                caption_box.y0
                - content_box.y1
            ),
            abs(
                content_box.y0
                - caption_box.y1
            ),
        )

        horizontal_overlap = (
            LayoutDetector._horizontal_overlap_ratio(
                caption_box,
                content_box,
            )
        )

        return (
            vertical_distance <= 45
            and horizontal_overlap >= 0.25
        )

    @staticmethod
    def _is_inside_any_box(
        bounding_box: BoundingBox,
        containers: Sequence[BoundingBox],
        *,
        minimum_overlap: float,
    ) -> bool:
        return any(
            LayoutDetector._overlap_ratio(
                bounding_box,
                container,
            )
            >= minimum_overlap
            for container in containers
        )

    @staticmethod
    def _overlap_ratio(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:
        x0 = max(first.x0, second.x0)
        y0 = max(first.y0, second.y0)
        x1 = min(first.x1, second.x1)
        y1 = min(first.y1, second.y1)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        intersection_area = (
            (x1 - x0)
            * (y1 - y0)
        )

        return min(
            intersection_area
            / max(first.area, 1.0),
            1.0,
        )

    @staticmethod
    def _horizontal_overlap_ratio(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:
        x0 = max(first.x0, second.x0)
        x1 = min(first.x1, second.x1)

        if x1 <= x0:
            return 0.0

        overlap = x1 - x0

        minimum_width = max(
            min(
                first.width,
                second.width,
            ),
            1.0,
        )

        return min(
            overlap / minimum_width,
            1.0,
        )

    @staticmethod
    def _formula_symbol_ratio(
        text: str,
    ) -> float:
        if not text:
            return 0.0

        formula_symbols = set(
            "=∑∫√±≈≤≥+-*/^()[]{}"
        )

        symbol_count = sum(
            character in formula_symbols
            for character in text
        )

        return (
            symbol_count
            / max(len(text), 1)
        )

    @staticmethod
    def _to_bounding_box(
        value: Any,
        page_rect: pymupdf.Rect,
    ) -> BoundingBox | None:
        if value is None:
            return None

        try:
            rect = (
                pymupdf.Rect(value)
                & page_rect
            )
        except Exception:
            return None

        if rect.is_empty or rect.is_infinite:
            return None

        if rect.width <= 0 or rect.height <= 0:
            return None

        return BoundingBox(
            x0=round(float(rect.x0), 3),
            y0=round(float(rect.y0), 3),
            x1=round(float(rect.x1), 3),
            y1=round(float(rect.y1), 3),
        )

    @classmethod
    def _deduplicate_boxes(
        cls,
        boxes: Sequence[BoundingBox],
    ) -> list[BoundingBox]:
        result: list[BoundingBox] = []

        for candidate in boxes:
            is_duplicate = any(
                cls._intersection_over_union(
                    candidate,
                    existing,
                )
                >= 0.90
                for existing in result
            )

            if not is_duplicate:
                result.append(candidate)

        return result

    @staticmethod
    def _intersection_over_union(
        first: BoundingBox,
        second: BoundingBox,
    ) -> float:
        x0 = max(first.x0, second.x0)
        y0 = max(first.y0, second.y0)
        x1 = min(first.x1, second.x1)
        y1 = min(first.y1, second.y1)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        intersection = (
            (x1 - x0)
            * (y1 - y0)
        )

        union = (
            first.area
            + second.area
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _is_pymupdf_page(value: Any) -> bool:
        """Nesnenin açık bir PyMuPDF sayfası olup olmadığını belirler."""
        return (
            hasattr(value, "get_text")
            and hasattr(value, "rect")
            and hasattr(value, "number")
        )

    @staticmethod
    def _extract_value(
        value: Any,
        names: Sequence[str],
    ) -> Any | None:
        """Nesne, sözlük veya metadata içinden ilk uygun alanı alır."""
        for name in names:
            if isinstance(value, dict) and name in value:
                candidate = value[name]
                if candidate is not None:
                    return candidate

            candidate = getattr(value, name, None)
            if candidate is not None:
                return candidate

        metadata = (
            value.get("metadata")
            if isinstance(value, dict)
            else getattr(value, "metadata", None)
        )

        if isinstance(metadata, dict):
            for name in names:
                candidate = metadata.get(name)
                if candidate is not None:
                    return candidate

        return None

    @classmethod
    def _resolve_page_number(
        cls,
        page: Any,
    ) -> int:
        value = cls._extract_value(
            page,
            ("page_number", "page_no", "number"),
        )

        if value is None:
            page_index = cls._extract_value(
                page,
                ("page_index", "index"),
            )

            if page_index is not None:
                value = int(page_index) + 1

        if value is None:
            raise LayoutDetectionError(
                "Layout analizi için sayfa numarası bulunamadı."
            )

        try:
            page_number = int(value)
        except (TypeError, ValueError) as exc:
            raise LayoutDetectionError(
                f"Geçersiz sayfa numarası: {value!r}"
            ) from exc

        if page_number <= 0:
            raise LayoutDetectionError(
                f"Sayfa numarası pozitif olmalıdır: {page_number}"
            )

        return page_number

    @classmethod
    def _resolve_image_path(
        cls,
        page: Any,
    ) -> Path | None:
        value = cls._extract_value(
            page,
            ("image_path", "rendered_path"),
        )

        if value is None:
            return None

        return Path(value).expanduser().resolve()

    @classmethod
    def _resolve_source_pdf(
        cls,
        page: Any,
    ) -> Path:
        direct_value = cls._extract_value(
            page,
            ("source_pdf", "pdf_path", "source_path"),
        )

        if direct_value is not None:
            source_pdf = Path(direct_value).expanduser().resolve()
            cls._validate_pdf_path(source_pdf)
            return source_pdf

        image_path = cls._resolve_image_path(page)

        if image_path is None:
            raise LayoutDetectionError(
                "Layout analizi için kaynak PDF yolu bulunamadı. "
                "Girdi source_pdf veya image_path alanı taşımalıdır."
            )

        manifest_path = image_path.parent / "render_manifest.json"

        if not manifest_path.is_file():
            raise LayoutDetectionError(
                f"Render manifesti bulunamadı: {manifest_path}"
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LayoutDetectionError(
                f"Render manifesti okunamadı: {manifest_path}"
            ) from exc

        source_value = manifest.get("source_pdf")

        if not source_value:
            raise LayoutDetectionError(
                f"Render manifestinde source_pdf bulunamadı: "
                f"{manifest_path}"
            )

        source_pdf = Path(source_value).expanduser()

        if not source_pdf.is_absolute():
            source_pdf = manifest_path.parent / source_pdf

        source_pdf = source_pdf.resolve()
        cls._validate_pdf_path(source_pdf)
        return source_pdf

    @staticmethod
    def _open_document(
        pdf_path: Path,
    ) -> Any:
        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise LayoutDetectionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        if document.needs_pass:
            document.close()
            raise LayoutDetectionError(
                "Parola korumalı PDF analiz edilemez."
            )

        if document.page_count == 0:
            document.close()
            raise LayoutDetectionError(
                "PDF içerisinde sayfa bulunamadı."
            )

        return document

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
            if not 1 <= page_number <= page_count
        ]

        if invalid_pages:
            raise ValueError(
                f"Geçersiz sayfa numaraları: {invalid_pages}. "
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
                "Yalnızca .pdf uzantılı dosyalar desteklenir."
            )


def detect_pdf_layout(
    pdf_path: str | Path,
    *,
    page_numbers: Iterable[int] | None = None,
    table_detection: bool = True,
) -> list[PageLayout]:
    """LayoutDetector sınıfı için kısa kullanım arayüzü."""

    detector = LayoutDetector(
        table_detection=table_detection
    )

    return detector.detect_pdf(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
    )


__all__ = [
    "BoundingBox",
    "LayoutDetectionError",
    "LayoutDetector",
    "LayoutElement",
    "LayoutElementType",
    "PageLayout",
    "detect_pdf_layout",
]
