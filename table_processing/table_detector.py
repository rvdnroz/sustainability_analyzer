from __future__ import annotations


import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


class TableDetectionError(RuntimeError):
    """Tablo tespit işlemi tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class TableBoundingBox:
    """
    PDF koordinat sisteminde tespit edilen tablo bölgesi.

    Koordinat sistemi sol üst köşeden başlar ve değerler point
    cinsindedir.
    """

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
class DetectedTable:
    """Bir PDF sayfasında tespit edilen tek bir tablo bölgesi."""

    table_id: str
    page_index: int
    page_number: int
    table_index: int

    bounding_box: TableBoundingBox

    confidence: float
    detection_method: str

    estimated_row_count: int | None
    estimated_column_count: int | None

    has_caption: bool
    caption_text: str | None

    has_native_structure: bool
    requires_visual_extraction: bool
    requires_ocr: bool

    text_preview: str

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["bounding_box"] = (
            self.bounding_box.to_dict()
        )

        return data


@dataclass(slots=True, frozen=True)
class PageTableDetection:
    """Tek bir sayfanın tablo tespit sonucu."""

    page_index: int
    page_number: int

    page_width: float
    page_height: float

    table_count: int
    tables: tuple[DetectedTable, ...]

    detection_methods: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "table_count": self.table_count,
            "tables": [
                table.to_dict()
                for table in self.tables
            ],
            "detection_methods": list(
                self.detection_methods
            ),
            "warnings": list(
                self.warnings
            ),
        }


@dataclass(slots=True)
class _TableCandidate:
    """Birleştirme öncesindeki dahili tablo adayı."""

    bounding_box: TableBoundingBox
    method: str
    confidence: float

    estimated_row_count: int | None = None
    estimated_column_count: int | None = None

    has_native_structure: bool = False
    metadata: dict[str, Any] | None = None


class TableDetector:
    """
    PDF sayfalarındaki tablo bölgelerini tespit eder.

    Kullanılan tespit yolları:

    1. PyMuPDF Page.find_tables() tabanlı native tespit
    2. Layout detector tarafından üretilmiş tablo bölgeleri
    3. Çizgi ve dikdörtgen yoğunluğuna dayalı görsel aday tespiti
    4. Metin bloklarındaki satır-sütun hizasına dayalı sezgisel tespit
    5. Türkçe ve İngilizce tablo başlıklarına dayalı caption-anchor tespiti
    6. Veri yoğun KPI/gösterge sayfası fallback tespiti

    Tespit edilen bölgeler daha sonra:

    - native_table_extractor.py
    - visual_table_extractor.py
    - table_ocr.py

    modüllerinden uygun olana yönlendirilir.
    """

    _NUMBERED_CAPTION_RE = re.compile(
        r"""
        ^\s*
        (?:
            table
            |tablo
            |çizelge
        )
        \s*
        (?:
            (?:no|number|numara)\.?\s*
        )?
        \d+
        (?:
            \s*[.\-–—/]\s*\d+
        )*
        \s*
        (?:
            [.:;\-–—]\s*
            .+
        )?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _UNNUMBERED_CAPTION_RE = re.compile(
        r"""
        ^\s*
        (?:
            summary\s+table
            |performance\s+table
            |indicator\s+table
            |data\s+table
            |kpi\s+table
            |appendix\s+table
            |özet\s+tablo
            |performans\s+tablosu
            |gösterge\s+tablosu
            |veri\s+tablosu
            |ek\s+tablo
        )
        (?:\s*[:.\-–—]\s*.+)?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    _CONTINUED_CAPTION_RE = re.compile(
        r"""
        ^\s*
        (?:
            table
            |tablo
            |çizelge
        )
        \s*
        \d+
        (?:
            \s*[.\-–—/]\s*\d+
        )*
        \s*
        (?:
            continued
            |cont\.? 
            |devamı
            |devami
            |devam
        )
        .*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        *,
        enable_native_detection: bool = True,
        enable_layout_detection: bool = True,
        enable_drawing_detection: bool = True,
        enable_text_alignment_detection: bool = True,
        enable_caption_detection: bool = True,
        minimum_width_ratio: float = 0.18,
        minimum_height_ratio: float = 0.04,
        minimum_area_ratio: float = 0.015,
        deduplication_iou_threshold: float = 0.65,
        candidate_merge_gap: float = 12.0,
        caption_search_distance: float = 55.0,
        caption_region_max_height_ratio: float = 0.72,
        caption_region_min_numeric_tokens: int = 2,
        scanned_text_threshold: int = 40,
        enable_indicator_page_fallback: bool = True,
        indicator_min_metric_matches: int = 3,
        indicator_min_numeric_tokens: int = 3,
        indicator_crop_margin_ratio: float = 0.025,
    ) -> None:
        ratio_values = {
            "minimum_width_ratio": minimum_width_ratio,
            "minimum_height_ratio": minimum_height_ratio,
            "minimum_area_ratio": minimum_area_ratio,
            "deduplication_iou_threshold": (
                deduplication_iou_threshold
            ),
        }

        for name, value in ratio_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        if candidate_merge_gap < 0:
            raise ValueError(
                "candidate_merge_gap negatif olamaz."
            )

        if caption_search_distance < 0:
            raise ValueError(
                "caption_search_distance negatif olamaz."
            )

        if not 0.10 <= caption_region_max_height_ratio <= 0.95:
            raise ValueError(
                "caption_region_max_height_ratio 0.10 ile 0.95 arasında olmalıdır."
            )

        if caption_region_min_numeric_tokens < 0:
            raise ValueError(
                "caption_region_min_numeric_tokens negatif olamaz."
            )

        if scanned_text_threshold < 0:
            raise ValueError(
                "scanned_text_threshold negatif olamaz."
            )

        if indicator_min_metric_matches < 1:
            raise ValueError(
                "indicator_min_metric_matches en az 1 olmalıdır."
            )

        if indicator_min_numeric_tokens < 1:
            raise ValueError(
                "indicator_min_numeric_tokens en az 1 olmalıdır."
            )

        if not 0.0 <= indicator_crop_margin_ratio <= 0.20:
            raise ValueError(
                "indicator_crop_margin_ratio 0 ile 0.20 arasında olmalıdır."
            )

        self.enable_native_detection = (
            enable_native_detection
        )

        self.enable_layout_detection = (
            enable_layout_detection
        )

        self.enable_drawing_detection = (
            enable_drawing_detection
        )

        self.enable_text_alignment_detection = (
            enable_text_alignment_detection
        )

        self.enable_caption_detection = bool(
            enable_caption_detection
        )

        self.minimum_width_ratio = (
            minimum_width_ratio
        )

        self.minimum_height_ratio = (
            minimum_height_ratio
        )

        self.minimum_area_ratio = (
            minimum_area_ratio
        )

        self.deduplication_iou_threshold = (
            deduplication_iou_threshold
        )

        self.candidate_merge_gap = (
            candidate_merge_gap
        )

        self.caption_search_distance = (
            caption_search_distance
        )

        self.caption_region_max_height_ratio = float(
            caption_region_max_height_ratio
        )

        self.caption_region_min_numeric_tokens = int(
            caption_region_min_numeric_tokens
        )

        self.scanned_text_threshold = (
            scanned_text_threshold
        )

        self.enable_indicator_page_fallback = bool(
            enable_indicator_page_fallback
        )
        self.indicator_min_metric_matches = int(
            indicator_min_metric_matches
        )
        self.indicator_min_numeric_tokens = int(
            indicator_min_numeric_tokens
        )
        self.indicator_crop_margin_ratio = float(
            indicator_crop_margin_ratio
        )

    def detect_pdf(
        self,
        pdf_path: str | Path,
        *,
        page_numbers: Iterable[int] | None = None,
        layouts_by_page: Mapping[
            int,
            Any,
        ]
        | None = None,
    ) -> list[PageTableDetection]:
        """
        PDF içindeki seçili veya bütün sayfalarda tablo tespiti yapar.

        Parametreler
        ------------
        pdf_path:
            İncelenecek PDF dosyasının yolu.

        page_numbers:
            1 tabanlı sayfa numaraları. None verilirse bütün PDF işlenir.

        layouts_by_page:
            layout_detector.py tarafından üretilmiş sayfa layout sonuçları.
            Anahtarlar 1 tabanlı sayfa numaralarıdır.
        """

        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(
            pdf_path
        )

        layouts_by_page = (
            layouts_by_page or {}
        )

        try:
            document = pymupdf.open(
                pdf_path
            )
        except Exception as exc:
            raise TableDetectionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise TableDetectionError(
                    "Parola korumalı PDF analiz edilemez."
                )

            if document.page_count == 0:
                raise TableDetectionError(
                    "PDF içinde sayfa bulunamadı."
                )

            selected_pages = (
                self._normalize_pages(
                    page_numbers=page_numbers,
                    page_count=document.page_count,
                )
            )

            results: list[
                PageTableDetection
            ] = []

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                result = self.detect_page(
                    page,
                    layout=layouts_by_page.get(
                        page_number
                    ),
                )

                results.append(
                    result
                )

            logger.info(
                "%d sayfada toplam %d tablo tespit edildi: %s",
                len(results),
                sum(
                    result.table_count
                    for result in results
                ),
                pdf_path,
            )

            return results

        finally:
            document.close()

    def detect_page(
        self,
        page: pymupdf.Page,
        *,
        layout: Any | None = None,
    ) -> PageTableDetection:
        """
        Açık bir PyMuPDF sayfasında tablo bölgelerini tespit eder.
        """

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

        candidates: list[
            _TableCandidate
        ] = []

        warnings: list[str] = []

        if self.enable_native_detection:
            native_candidates = (
                self._detect_native_tables(
                    page
                )
            )

            candidates.extend(
                native_candidates
            )

            if not hasattr(
                page,
                "find_tables",
            ):
                warnings.append(
                    "PyMuPDF Page.find_tables "
                    "arayüzü mevcut değil."
                )

        if (
            self.enable_layout_detection
            and layout is not None
        ):
            candidates.extend(
                self._detect_from_layout(
                    page=page,
                    layout=layout,
                )
            )

        if self.enable_drawing_detection:
            candidates.extend(
                self._detect_from_drawings(
                    page
                )
            )

        if self.enable_text_alignment_detection:
            candidates.extend(
                self._detect_from_text_alignment(
                    page
                )
            )

        if self.enable_caption_detection:
            candidates.extend(
                self._detect_from_table_captions(
                    page
                )
            )

        candidates = [
            candidate
            for candidate in candidates
            if self._is_valid_candidate(
                candidate,
                page_rect=page.rect,
            )
        ]

        candidates = (
            self._merge_candidates(
                candidates
            )
        )

        candidates = (
            self._deduplicate_candidates(
                candidates
            )
        )

        candidates.sort(
            key=lambda candidate: (
                candidate.bounding_box.y0,
                candidate.bounding_box.x0,
            )
        )

        page_text = page.get_text(
            "text",
            sort=True,
        ).strip()

        native_text_is_sparse = (
            len(page_text)
            < self.scanned_text_threshold
        )

        # Native, layout, drawing ve text-alignment yolları hiç tablo
        # bulamazsa; metrik, birim, yıl ve sayısal değer yoğunluğu yüksek
        # gösterge sayfalarını görsel/OCR hattına zorunlu olarak yönlendir.
        if (
            not candidates
            and self.enable_indicator_page_fallback
        ):
            indicator_candidate = (
                self._detect_indicator_page_fallback(
                    page=page,
                    page_text=page_text,
                )
            )

            if indicator_candidate is not None:
                candidates.append(indicator_candidate)
                warnings.append(
                    "Standart tablo tespiti sonuç üretmedi; "
                    "sayfa gösterge-tablosu fallback adayı olarak eklendi."
                )

        detected_tables: list[
            DetectedTable
        ] = []

        for table_index, candidate in enumerate(
            candidates
        ):
            caption_text = (
                self._find_caption(
                    page=page,
                    table_box=candidate.bounding_box,
                )
            )

            text_preview = (
                self._extract_region_text(
                    page=page,
                    bounding_box=candidate.bounding_box,
                    max_length=500,
                )
            )

            has_native_structure = (
                candidate.has_native_structure
            )

            fallback_prefers_ocr = bool(
                (candidate.metadata or {}).get(
                    "requires_whole_image_ocr_preferred",
                    False,
                )
            )

            requires_ocr = (
                (
                    native_text_is_sparse
                    and not text_preview.strip()
                )
                or fallback_prefers_ocr
            )

            requires_visual_extraction = (
                not has_native_structure
            )

            if requires_ocr:
                requires_visual_extraction = True

            confidence = self._adjust_confidence(
                candidate=candidate,
                text_preview=text_preview,
                caption_text=caption_text,
            )

            detected_tables.append(
                DetectedTable(
                    table_id=(
                        f"page-{page_number}"
                        f"-table-{table_index + 1}"
                    ),
                    page_index=page_index,
                    page_number=page_number,
                    table_index=table_index,
                    bounding_box=(
                        candidate.bounding_box
                    ),
                    confidence=confidence,
                    detection_method=(
                        candidate.method
                    ),
                    estimated_row_count=(
                        candidate.estimated_row_count
                    ),
                    estimated_column_count=(
                        candidate.estimated_column_count
                    ),
                    has_caption=bool(
                        caption_text
                    ),
                    caption_text=caption_text,
                    has_native_structure=(
                        has_native_structure
                    ),
                    requires_visual_extraction=(
                        requires_visual_extraction
                    ),
                    requires_ocr=requires_ocr,
                    text_preview=text_preview,
                    metadata={
                        **(
                            candidate.metadata
                            or {}
                        ),
                        "native_text_is_sparse": (
                            native_text_is_sparse
                        ),
                        "table_type": self._infer_table_type(
                            candidate
                        ),
                        "visual_region_type": (
                            self._infer_visual_region_type(
                                candidate=candidate,
                                caption_text=caption_text,
                                text_preview=text_preview,
                            )
                        ),
                    },
                )
            )

        methods = tuple(
            dict.fromkeys(
                table.detection_method
                for table in detected_tables
            )
        )

        return PageTableDetection(
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
            table_count=len(
                detected_tables
            ),
            tables=tuple(
                detected_tables
            ),
            detection_methods=methods,
            warnings=tuple(
                warnings
            ),
        )

    def _detect_native_tables(
        self,
        page: pymupdf.Page,
    ) -> list[_TableCandidate]:
        """
        PyMuPDF Page.find_tables kullanarak native tablo tespiti yapar.
        """

        if not hasattr(
            page,
            "find_tables",
        ):
            return []

        try:
            finder = page.find_tables()
        except Exception as exc:
            logger.debug(
                "Sayfa %d native tablo tespiti başarısız: %s",
                int(page.number) + 1,
                exc,
            )

            return []

        candidates: list[
            _TableCandidate
        ] = []

        for table in getattr(
            finder,
            "tables",
            [],
        ):
            bounding_box = (
                self._to_bounding_box(
                    getattr(
                        table,
                        "bbox",
                        None,
                    ),
                    page.rect,
                )
            )

            if bounding_box is None:
                continue

            row_count = self._safe_int(
                getattr(
                    table,
                    "row_count",
                    None,
                )
            )

            column_count = (
                self._safe_int(
                    getattr(
                        table,
                        "col_count",
                        None,
                    )
                )
            )

            candidates.append(
                _TableCandidate(
                    bounding_box=bounding_box,
                    method="pymupdf_native",
                    confidence=0.94,
                    estimated_row_count=(
                        row_count
                    ),
                    estimated_column_count=(
                        column_count
                    ),
                    has_native_structure=True,
                    metadata={
                        "source": (
                            "page.find_tables"
                        )
                    },
                )
            )

        return candidates

    def _detect_from_layout(
        self,
        *,
        page: pymupdf.Page,
        layout: Any,
    ) -> list[_TableCandidate]:
        """
        layout_detector.py çıktısındaki TABLE öğelerini tablo adayı yapar.
        """

        elements: Sequence[Any]

        if isinstance(
            layout,
            Mapping,
        ):
            elements = layout.get(
                "elements",
                [],
            )
        else:
            elements = getattr(
                layout,
                "elements",
                [],
            )

        candidates: list[
            _TableCandidate
        ] = []

        for element in elements:
            if isinstance(
                element,
                Mapping,
            ):
                element_type = (
                    element.get(
                        "element_type"
                    )
                )

                bounding_value = (
                    element.get(
                        "bounding_box"
                    )
                )

                confidence = float(
                    element.get(
                        "confidence",
                        0.80,
                    )
                )

            else:
                element_type = getattr(
                    element,
                    "element_type",
                    None,
                )

                bounding_value = getattr(
                    element,
                    "bounding_box",
                    None,
                )

                confidence = float(
                    getattr(
                        element,
                        "confidence",
                        0.80,
                    )
                )

            normalized_type = str(
                getattr(
                    element_type,
                    "value",
                    element_type,
                )
            ).lower()

            if normalized_type != "table":
                continue

            bounding_box = (
                self._to_bounding_box(
                    bounding_value,
                    page.rect,
                )
            )

            if bounding_box is None:
                continue

            candidates.append(
                _TableCandidate(
                    bounding_box=bounding_box,
                    method="layout_detector",
                    confidence=min(
                        max(
                            confidence,
                            0.0,
                        ),
                        1.0,
                    ),
                    has_native_structure=False,
                    metadata={
                        "source": (
                            "layout_detector"
                        )
                    },
                )
            )

        return candidates

    def _detect_from_drawings(
        self,
        page: pymupdf.Page,
    ) -> list[_TableCandidate]:
        """
        Çizgi ve dikdörtgen kümelerinden tablo bölgeleri üretir.
        """

        try:
            drawings = page.get_drawings()
        except Exception as exc:
            logger.debug(
                "Sayfa %d çizim analizi başarısız: %s",
                int(page.number) + 1,
                exc,
            )

            return []

        line_boxes: list[
            TableBoundingBox
        ] = []

        horizontal_line_count = 0
        vertical_line_count = 0

        for drawing in drawings:
            rect = self._to_bounding_box(
                drawing.get("rect"),
                page.rect,
            )

            if rect is None:
                continue

            items = drawing.get(
                "items",
                [],
            )

            for item in items:
                if not item:
                    continue

                item_type = str(
                    item[0]
                ).lower()

                if item_type == "l" and len(
                    item
                ) >= 3:
                    start = item[1]
                    end = item[2]

                    x_difference = abs(
                        float(end.x)
                        - float(start.x)
                    )

                    y_difference = abs(
                        float(end.y)
                        - float(start.y)
                    )

                    if x_difference >= (
                        y_difference * 3
                    ):
                        horizontal_line_count += 1

                    elif y_difference >= (
                        x_difference * 3
                    ):
                        vertical_line_count += 1

            if (
                rect.width
                >= page.rect.width
                * self.minimum_width_ratio
                and rect.height >= 1.0
            ):
                line_boxes.append(
                    rect
                )

        if (
            horizontal_line_count < 3
            or vertical_line_count < 2
        ):
            return []

        clustered_boxes = (
            self._cluster_nearby_boxes(
                line_boxes
            )
        )

        candidates: list[
            _TableCandidate
        ] = []

        for box in clustered_boxes:
            confidence = min(
                0.88,
                0.55
                + 0.02
                * horizontal_line_count
                + 0.02
                * vertical_line_count,
            )

            candidates.append(
                _TableCandidate(
                    bounding_box=box,
                    method="drawing_lines",
                    confidence=confidence,
                    has_native_structure=False,
                    metadata={
                        "horizontal_line_count": (
                            horizontal_line_count
                        ),
                        "vertical_line_count": (
                            vertical_line_count
                        ),
                    },
                )
            )

        return candidates

    def _detect_from_text_alignment(
        self,
        page: pymupdf.Page,
    ) -> list[_TableCandidate]:
        """
        Metin satırlarının yatay hizalanmasına göre tablo adayları üretir.
        """

        words = page.get_text(
            "words",
            sort=True,
        )

        if not words:
            return []

        lines: dict[
            tuple[int, int],
            list[tuple[Any, ...]],
        ] = {}

        for word in words:
            if len(word) < 8:
                continue

            block_number = int(
                word[5]
            )

            line_number = int(
                word[6]
            )

            key = (
                block_number,
                line_number,
            )

            lines.setdefault(
                key,
                []
            ).append(
                word
            )

        tabular_lines: list[
            TableBoundingBox
        ] = []

        estimated_columns: list[int] = []

        for line_words in lines.values():
            if len(line_words) < 2:
                continue

            sorted_words = sorted(
                line_words,
                key=lambda item: float(
                    item[0]
                ),
            )

            large_gap_count = 0

            for left_word, right_word in zip(
                sorted_words,
                sorted_words[1:],
                strict=False,
            ):
                gap = (
                    float(right_word[0])
                    - float(left_word[2])
                )

                if gap >= 16.0:
                    large_gap_count += 1

            numeric_count = sum(
                self._looks_numeric(
                    str(word[4])
                )
                for word in sorted_words
            )

            has_label_and_value = (
                len(sorted_words) >= 2
                and numeric_count >= 1
                and large_gap_count >= 1
            )

            has_multi_value_row = (
                numeric_count >= 2
                and large_gap_count >= 1
            )

            is_tabular = (
                has_label_and_value
                or has_multi_value_row
                or (
                    len(sorted_words) >= 4
                    and large_gap_count >= 2
                )
            )

            if not is_tabular:
                continue

            x0 = min(
                float(word[0])
                for word in sorted_words
            )

            y0 = min(
                float(word[1])
                for word in sorted_words
            )

            x1 = max(
                float(word[2])
                for word in sorted_words
            )

            y1 = max(
                float(word[3])
                for word in sorted_words
            )

            bounding_box = (
                self._to_bounding_box(
                    (
                        x0,
                        y0,
                        x1,
                        y1,
                    ),
                    page.rect,
                )
            )

            if bounding_box is None:
                continue

            tabular_lines.append(
                bounding_box
            )

            estimated_columns.append(
                large_gap_count + 1
            )

        if len(
            tabular_lines
        ) < 3:
            return []

        clustered_boxes = (
            self._cluster_nearby_boxes(
                tabular_lines
            )
        )

        candidates: list[
            _TableCandidate
        ] = []

        estimated_column_count = (
            round(
                sum(
                    estimated_columns
                )
                / len(
                    estimated_columns
                )
            )
            if estimated_columns
            else None
        )

        for box in clustered_boxes:
            contained_line_count = sum(
                self._overlap_ratio(
                    line_box,
                    box,
                )
                >= 0.70
                for line_box in tabular_lines
            )

            if contained_line_count < 3:
                continue

            confidence = min(
                0.82,
                0.52
                + contained_line_count
                * 0.035,
            )

            candidates.append(
                _TableCandidate(
                    bounding_box=box,
                    method="text_alignment",
                    confidence=confidence,
                    estimated_row_count=(
                        contained_line_count
                    ),
                    estimated_column_count=(
                        estimated_column_count
                    ),
                    has_native_structure=False,
                    metadata={
                        "tabular_line_count": (
                            contained_line_count
                        )
                    },
                )
            )

        return candidates

    def _detect_from_table_captions(
        self,
        page: pymupdf.Page,
    ) -> list[_TableCandidate]:
        """
        Türkçe ve İngilizce tablo başlıklarını anchor olarak kullanır.

        Başlık üstteyse başlığın altındaki veri yoğun bölgeyi, başlık
        alttaysa üstündeki veri yoğun bölgeyi tablo adayı yapar.
        """

        raw_blocks = page.get_text(
            "blocks",
            sort=True,
        )

        blocks: list[tuple[TableBoundingBox, str]] = []

        for block in raw_blocks:
            if len(block) < 5:
                continue

            text = str(block[4] or "").strip()

            if not text:
                continue

            box = self._to_bounding_box(
                block[:4],
                page.rect,
            )

            if box is not None:
                blocks.append((box, text))

        candidates: list[_TableCandidate] = []

        for index, (caption_box, caption_text) in enumerate(blocks):
            caption_kind = self._caption_kind(
                caption_text
            )

            if caption_kind is None:
                continue

            below_blocks = [
                item
                for item in blocks[index + 1 :]
                if item[0].y0 >= caption_box.y1 - 2.0
            ]

            above_blocks = [
                item
                for item in blocks[:index]
                if item[0].y1 <= caption_box.y0 + 2.0
            ]

            below_region = self._caption_neighbor_region(
                page=page,
                caption_box=caption_box,
                blocks=below_blocks,
                direction="below",
            )

            above_region = self._caption_neighbor_region(
                page=page,
                caption_box=caption_box,
                blocks=above_blocks,
                direction="above",
            )

            choices = [
                choice
                for choice in (below_region, above_region)
                if choice is not None
            ]

            if not choices:
                continue

            region_box, region_stats = max(
                choices,
                key=lambda item: (
                    item[1]["score"],
                    item[0].area,
                ),
            )

            confidence = 0.78

            if caption_kind == "numbered":
                confidence += 0.09
            elif caption_kind == "continued":
                confidence += 0.07
            else:
                confidence += 0.04

            confidence += min(
                region_stats["numeric_token_count"] * 0.012,
                0.06,
            )

            if region_stats["year_token_count"] >= 2:
                confidence += 0.03

            confidence = round(
                min(confidence, 0.96),
                4,
            )

            candidates.append(
                _TableCandidate(
                    bounding_box=region_box,
                    method="caption_anchor",
                    confidence=confidence,
                    estimated_row_count=(
                        region_stats["line_count"]
                        if region_stats["line_count"] >= 2
                        else None
                    ),
                    estimated_column_count=None,
                    has_native_structure=False,
                    metadata={
                        "source": "table_caption",
                        "caption_text": caption_text,
                        "caption_kind": caption_kind,
                        "caption_box": caption_box.to_dict(),
                        "caption_direction": region_stats["direction"],
                        "numeric_token_count": (
                            region_stats["numeric_token_count"]
                        ),
                        "year_token_count": (
                            region_stats["year_token_count"]
                        ),
                        "line_count": region_stats["line_count"],
                        "requires_whole_image_ocr_preferred": True,
                    },
                )
            )

        return candidates

    def _caption_neighbor_region(
        self,
        *,
        page: pymupdf.Page,
        caption_box: TableBoundingBox,
        blocks: Sequence[tuple[TableBoundingBox, str]],
        direction: str,
    ) -> tuple[TableBoundingBox, dict[str, Any]] | None:
        if direction not in {"below", "above"}:
            raise ValueError(
                "direction 'below' veya 'above' olmalıdır."
            )

        if not blocks:
            return None

        page_height = float(page.rect.height)
        maximum_height = (
            page_height
            * self.caption_region_max_height_ratio
        )

        ordered = sorted(
            blocks,
            key=lambda item: item[0].y0,
            reverse=(direction == "above"),
        )

        selected: list[tuple[TableBoundingBox, str]] = []

        for box, text in ordered:
            distance = (
                box.y0 - caption_box.y1
                if direction == "below"
                else caption_box.y0 - box.y1
            )

            if distance < -2.0:
                continue

            if distance > maximum_height:
                break

            if selected:
                previous_box = selected[-1][0]
                gap = (
                    box.y0 - previous_box.y1
                    if direction == "below"
                    else previous_box.y0 - box.y1
                )

                if gap > max(
                    self.caption_search_distance * 1.8,
                    90.0,
                ):
                    break

            normalized = self._normalize_indicator_text(
                text
            )

            if (
                selected
                and self._looks_like_section_heading(
                    text,
                    box=box,
                    page_width=float(page.rect.width),
                )
            ):
                break

            if normalized:
                selected.append((box, text))

        if not selected:
            return None

        if direction == "above":
            selected.reverse()

        selected_text = "\n".join(
            text
            for _, text in selected
        )

        numeric_tokens = re.findall(
            r"(?<!\w)[+-]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
            r"(?:[.,]\d+)?(?:\s*%)?(?!\w)",
            selected_text,
        )

        year_tokens = re.findall(
            r"\b(?:19|20)\d{2}\b",
            selected_text,
        )

        line_count = len(
            [
                line
                for line in selected_text.splitlines()
                if line.strip()
            ]
        )

        if (
            len(numeric_tokens)
            < self.caption_region_min_numeric_tokens
            and len(year_tokens) < 2
        ):
            return None

        content_box = self._union_boxes(
            [box for box, _ in selected]
        )

        padding_x = max(
            float(page.rect.width) * 0.02,
            6.0,
        )
        padding_y = 5.0

        padded = self._to_bounding_box(
            (
                max(float(page.rect.x0), content_box.x0 - padding_x),
                max(float(page.rect.y0), content_box.y0 - padding_y),
                min(float(page.rect.x1), content_box.x1 + padding_x),
                min(float(page.rect.y1), content_box.y1 + padding_y),
            ),
            page.rect,
        )

        if padded is None:
            return None

        numeric_density = (
            len(numeric_tokens)
            / max(line_count, 1)
        )

        score = (
            len(numeric_tokens)
            + 1.5 * len(set(year_tokens))
            + min(numeric_density, 3.0)
        )

        return padded, {
            "direction": direction,
            "numeric_token_count": len(numeric_tokens),
            "year_token_count": len(set(year_tokens)),
            "line_count": line_count,
            "score": score,
        }

    @classmethod
    def _caption_kind(
        cls,
        text: str,
    ) -> str | None:
        normalized = " ".join(
            str(text or "").split()
        )

        if cls._CONTINUED_CAPTION_RE.match(normalized):
            return "continued"

        if cls._NUMBERED_CAPTION_RE.match(normalized):
            return "numbered"

        if cls._UNNUMBERED_CAPTION_RE.match(normalized):
            return "unnumbered"

        return None

    @staticmethod
    def _looks_like_section_heading(
        text: str,
        *,
        box: TableBoundingBox,
        page_width: float,
    ) -> bool:
        normalized = " ".join(
            str(text or "").split()
        )

        if not normalized:
            return False

        if re.search(r"\d", normalized):
            return False

        word_count = len(normalized.split())

        if word_count > 12:
            return False

        uppercase_ratio = (
            sum(character.isupper() for character in normalized)
            / max(
                sum(character.isalpha() for character in normalized),
                1,
            )
        )

        return (
            box.width >= page_width * 0.35
            and (
                uppercase_ratio >= 0.75
                or normalized.endswith(":")
            )
        )

    def _detect_indicator_page_fallback(
        self,
        *,
        page: pymupdf.Page,
        page_text: str,
    ) -> _TableCandidate | None:
        """
        Çizgisiz veya infografik biçimli KPI/gösterge sayfalarını yakalar.

        Bu yöntem yalnızca diğer tespit yolları hiç aday üretmediğinde
        çalışır. Şirket ya da sabit sayfa numarası kullanmaz.
        """

        normalized_text = self._normalize_indicator_text(
            page_text
        )

        if not normalized_text:
            return None

        title_signals = (
            "environmental performance indicators",
            "performance indicators",
            "key performance indicators",
            "sustainability performance",
            "esg performance",
            "environmental indicators",
            "social performance indicators",
            "operational performance indicators",
            "performance data",
            "kpi",
            "gostergeler",
            "performans gostergeleri",
            "cevre performans gostergeleri",
        )

        metric_signals = (
            "total energy consumption",
            "direct energy consumption",
            "indirect energy consumption",
            "renewable energy",
            "freshwater withdrawal",
            "water withdrawal",
            "water consumption",
            "water discharged",
            "wastewater",
            "greenhouse gas emissions",
            "scope 1",
            "scope 2",
            "scope 3",
            "hazardous waste",
            "non hazardous waste",
            "recycled waste",
            "employee",
            "injury rate",
            "lost time injury",
            "training hours",
            "turnover rate",
            "total water",
            "total waste",
            "total emissions",
        )

        unit_signals = (
            "tco2e",
            "co2e",
            "m3",
            "m³",
            "gj",
            "mwh",
            "kwh",
            "gwh",
            "ton",
            "tonne",
            "tons",
            "%",
        )

        title_matches = [
            signal
            for signal in title_signals
            if signal in normalized_text
        ]

        metric_matches = [
            signal
            for signal in metric_signals
            if signal in normalized_text
        ]

        unit_matches = [
            signal
            for signal in unit_signals
            if signal in normalized_text
        ]

        numeric_tokens = re.findall(
            r"(?<!\w)[+-]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
            r"(?:[.,]\d+)?(?:\s*%)?(?!\w)",
            page_text,
        )

        year_tokens = sorted(
            set(
                re.findall(
                    r"\b(?:19|20)\d{2}\b",
                    page_text,
                )
            )
        )

        has_strong_title = bool(title_matches)
        has_dense_metrics = (
            len(metric_matches)
            >= self.indicator_min_metric_matches
        )
        has_numeric_density = (
            len(numeric_tokens)
            >= self.indicator_min_numeric_tokens
        )
        has_units = bool(unit_matches)

        qualifies = (
            has_numeric_density
            and has_units
            and (
                has_strong_title
                or has_dense_metrics
            )
        )

        if not qualifies:
            return None

        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

        margin_x = (
            page_width
            * self.indicator_crop_margin_ratio
        )
        margin_y = (
            page_height
            * self.indicator_crop_margin_ratio
        )

        bounding_box = self._indicator_content_box(
            page=page,
            fallback_box=TableBoundingBox(
                x0=round(margin_x, 3),
                y0=round(margin_y, 3),
                x1=round(page_width - margin_x, 3),
                y1=round(page_height - margin_y, 3),
            ),
        )

        confidence = 0.68

        if has_strong_title:
            confidence += 0.10

        confidence += min(
            len(metric_matches) * 0.015,
            0.09,
        )

        confidence += min(
            len(unit_matches) * 0.01,
            0.04,
        )

        if len(year_tokens) >= 2:
            confidence += 0.03

        confidence = round(
            min(confidence, 0.92),
            4,
        )

        return _TableCandidate(
            bounding_box=bounding_box,
            method="indicator_page_fallback",
            confidence=confidence,
            estimated_row_count=None,
            estimated_column_count=None,
            has_native_structure=False,
            metadata={
                "source": "indicator_page_heuristic",
                "title_matches": title_matches,
                "metric_matches": metric_matches,
                "unit_matches": unit_matches,
                "numeric_token_count": len(numeric_tokens),
                "year_tokens": year_tokens,
                "full_page_visual_fallback": True,
                "requires_whole_image_ocr_preferred": True,
            },
        )

    def _indicator_content_box(
        self,
        *,
        page: pymupdf.Page,
        fallback_box: TableBoundingBox,
    ) -> TableBoundingBox:
        blocks = page.get_text(
            "blocks",
            sort=True,
        )

        useful_boxes: list[TableBoundingBox] = []

        page_height = float(page.rect.height)

        for block in blocks:
            if len(block) < 5:
                continue

            text = str(block[4] or "").strip()

            if not text:
                continue

            box = self._to_bounding_box(
                block[:4],
                page.rect,
            )

            if box is None:
                continue

            # Tipik üstbilgi/altbilgi ve yalnız sayfa numarası bloklarını ele.
            if box.y1 <= page_height * 0.035:
                continue

            if box.y0 >= page_height * 0.965:
                continue

            if re.fullmatch(r"\s*\d{1,4}\s*", text):
                continue

            useful_boxes.append(box)

        if not useful_boxes:
            return fallback_box

        union = self._union_boxes(useful_boxes)

        padding_x = max(
            float(page.rect.width) * 0.018,
            5.0,
        )
        padding_y = max(
            float(page.rect.height) * 0.012,
            5.0,
        )

        bounded = self._to_bounding_box(
            (
                max(fallback_box.x0, union.x0 - padding_x),
                max(fallback_box.y0, union.y0 - padding_y),
                min(fallback_box.x1, union.x1 + padding_x),
                min(fallback_box.y1, union.y1 + padding_y),
            ),
            page.rect,
        )

        return bounded or fallback_box

    @staticmethod
    def _normalize_indicator_text(
        value: str,
    ) -> str:
        normalized = str(value or "").casefold()

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
                "₂": "2",
                "³": "3",
            }
        )

        normalized = normalized.translate(
            translation_table
        )

        normalized = re.sub(
            r"[^\w\s%./+-]",
            " ",
            normalized,
        )

        return " ".join(
            normalized.split()
        )

    @classmethod
    def _infer_visual_region_type(
        cls,
        *,
        candidate: _TableCandidate,
        caption_text: str | None,
        text_preview: str,
    ) -> str:
        """Görsel bölge türünü açıklanabilir puanlama ile belirler."""

        scores = cls._score_visual_region(
            candidate=candidate,
            caption_text=caption_text,
            text_preview=text_preview,
        )
        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score < 0.52:
            return "unknown"

        if best_score - second_score < 0.08:
            return "unknown"

        return best_type

    @classmethod
    def _score_visual_region(
        cls,
        *,
        candidate: _TableCandidate,
        caption_text: str | None,
        text_preview: str,
    ) -> dict[str, float]:
        """Tablo, grafik, metrik kartı ve infografik puanlarını üretir."""

        metadata = dict(candidate.metadata or {})
        method = candidate.method.casefold()
        table_type = cls._infer_table_type(candidate)

        searchable = cls._normalize_indicator_text(
            " ".join(
                value
                for value in (
                    caption_text or "",
                    text_preview or "",
                    str(metadata.get("caption_text") or ""),
                    " ".join(str(item) for item in metadata.get("title_matches", []) or []),
                    " ".join(str(item) for item in metadata.get("metric_matches", []) or []),
                    " ".join(str(item) for item in metadata.get("unit_matches", []) or []),
                )
                if value
            )
        )

        scores = {
            "table": 0.0,
            "chart": 0.0,
            "metric_card": 0.0,
            "infographic": 0.0,
        }

        explicit_chart_terms = (
            "bar chart", "line chart", "pie chart", "stacked bar",
            "area chart", "scatter plot", "bubble chart",
            "waterfall chart", "column chart", "donut chart",
            "chart", "graph", "grafik",
        )
        chart_context_terms = (
            "trend", "distribution", "breakdown", "share of",
            "by year", "yearly", "annual change", "evolution",
            "composition", "female", "male",
            "scope 1", "scope 2", "scope 3",
            "target", "actual", "baseline",
            "dagilim", "egilim", "yillara gore",
        )
        table_terms = (
            "table", "tablo", "cizelge", "summary table",
            "performance table", "indicator table",
            "data table", "kpi table",
        )
        metric_card_terms = (
            "kpi", "key performance indicator", "metric card",
            "highlight", "at a glance", "in numbers",
            "rakamlarla", "one bakis",
        )
        infographic_terms = (
            "infographic", "infografik", "roadmap",
            "value chain", "process flow", "timeline",
        )

        if candidate.has_native_structure:
            scores["table"] += 0.95

        if table_type == "grid_table":
            scores["table"] += 0.70
        elif table_type == "borderless_table":
            scores["table"] += 0.72
        elif table_type == "caption_anchored_table":
            scores["table"] += 0.78
        elif table_type == "kpi_or_indicator_table":
            scores["metric_card"] += 0.48

        if any(term in searchable for term in table_terms):
            scores["table"] += 0.42

        if any(term in searchable for term in explicit_chart_terms):
            scores["chart"] += 0.68

        chart_context_count = sum(
            term in searchable
            for term in chart_context_terms
        )
        scores["chart"] += min(chart_context_count * 0.09, 0.36)

        if any(term in searchable for term in metric_card_terms):
            scores["metric_card"] += 0.48

        if any(term in searchable for term in infographic_terms):
            scores["infographic"] += 0.72

        years_in_text = set(
            re.findall(r"\b(?:19|20)\d{2}\b", searchable)
        )
        years_in_metadata = {
            str(value)
            for value in metadata.get("year_tokens", []) or []
        }
        year_count = len(years_in_text | years_in_metadata)

        numeric_token_count = int(
            metadata.get("numeric_token_count", 0) or 0
        )
        if numeric_token_count == 0:
            numeric_token_count = len(
                re.findall(
                    r"(?<!\w)[+-]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
                    r"(?:[.,]\d+)?(?:\s*%)?(?!\w)",
                    searchable,
                )
            )

        if year_count >= 3:
            scores["chart"] += 0.18
        elif year_count == 2:
            scores["chart"] += 0.08

        if numeric_token_count >= 8:
            scores["chart"] += 0.12
            scores["metric_card"] += 0.08
        elif numeric_token_count >= 4:
            scores["chart"] += 0.06
            scores["metric_card"] += 0.05

        horizontal_lines = int(
            metadata.get("horizontal_line_count", 0) or 0
        )
        vertical_lines = int(
            metadata.get("vertical_line_count", 0) or 0
        )

        if "drawing_lines" in method:
            if horizontal_lines >= 4 and vertical_lines <= 2:
                scores["chart"] += 0.18
            elif horizontal_lines >= 3 and vertical_lines >= 3:
                scores["table"] += 0.22

        if "text_alignment" in method:
            scores["table"] += 0.28

        if "caption_anchor" in method:
            scores["table"] += 0.34

        if metadata.get("full_page_visual_fallback"):
            scores["metric_card"] += 0.20
            if year_count >= 3 and chart_context_count >= 1:
                scores["chart"] += 0.16

        return {
            key: round(min(max(value, 0.0), 1.0), 4)
            for key, value in scores.items()
        }

    @classmethod
    def _estimate_visual_region_confidence(
        cls,
        *,
        candidate: _TableCandidate,
        caption_text: str | None,
        text_preview: str,
    ) -> float:
        """Seçilen görsel türü için güven puanı üretir."""

        scores = cls._score_visual_region(
            candidate=candidate,
            caption_text=caption_text,
            text_preview=text_preview,
        )
        ranked = sorted(scores.values(), reverse=True)
        best = ranked[0] if ranked else 0.0
        second = ranked[1] if len(ranked) > 1 else 0.0

        confidence = best * 0.80 + max(best - second, 0.0) * 0.20
        return round(min(max(confidence, 0.0), 0.99), 4)

    @staticmethod
    def _infer_table_type(
        candidate: _TableCandidate,
    ) -> str:
        method = candidate.method.casefold()
        metadata = dict(candidate.metadata or {})

        if candidate.has_native_structure:
            return "grid_table"

        if "caption_anchor" in method:
            return "caption_anchored_table"

        if "text_alignment" in method:
            return "borderless_table"

        if "drawing_lines" in method:
            return "grid_table"

        if "indicator_page_fallback" in method:
            return "kpi_or_indicator_table"

        if metadata.get("full_page_visual_fallback"):
            return "kpi_or_indicator_table"

        return "unknown"

    def _merge_candidates(
        self,
        candidates: Sequence[
            _TableCandidate
        ],
    ) -> list[_TableCandidate]:
        """
        Birbirine yakın veya üst üste gelen adayları tek bölgeye birleştirir.
        """

        if len(candidates) < 2:
            return list(
                candidates
            )

        remaining = list(
            candidates
        )

        merged: list[
            _TableCandidate
        ] = []

        while remaining:
            current = remaining.pop(
                0
            )

            related: list[
                _TableCandidate
            ] = [
                current
            ]

            changed = True

            while changed:
                changed = False

                current_union = (
                    self._union_boxes(
                        [
                            candidate.bounding_box
                            for candidate in related
                        ]
                    )
                )

                unrelated: list[
                    _TableCandidate
                ] = []

                for candidate in remaining:
                    if (
                        self._intersection_over_union(
                            current_union,
                            candidate.bounding_box,
                        )
                        >= 0.20
                        or self._boxes_are_near(
                            current_union,
                            candidate.bounding_box,
                            max_gap=(
                                self.candidate_merge_gap
                            ),
                        )
                    ):
                        related.append(
                            candidate
                        )

                        changed = True
                    else:
                        unrelated.append(
                            candidate
                        )

                remaining = unrelated

            merged.append(
                self._combine_candidates(
                    related
                )
            )

        return merged

    def _combine_candidates(
        self,
        candidates: Sequence[
            _TableCandidate
        ],
    ) -> _TableCandidate:
        bounding_box = self._union_boxes(
            [
                candidate.bounding_box
                for candidate in candidates
            ]
        )

        method_names = list(
            dict.fromkeys(
                candidate.method
                for candidate in candidates
            )
        )

        strongest_candidate = max(
            candidates,
            key=lambda candidate: (
                candidate.confidence
            ),
        )

        confidence_bonus = min(
            0.08
            * (
                len(
                    method_names
                )
                - 1
            ),
            0.16,
        )

        confidence = min(
            0.99,
            strongest_candidate.confidence
            + confidence_bonus,
        )

        row_counts = [
            candidate.estimated_row_count
            for candidate in candidates
            if candidate.estimated_row_count
            is not None
        ]

        column_counts = [
            candidate.estimated_column_count
            for candidate in candidates
            if candidate.estimated_column_count
            is not None
        ]

        metadata: dict[str, Any] = {
            **dict(
                strongest_candidate.metadata
                or {}
            ),
            "merged_candidate_count": len(
                candidates
            ),
            "merged_methods": method_names,
        }

        return _TableCandidate(
            bounding_box=bounding_box,
            method="+".join(
                method_names
            ),
            confidence=confidence,
            estimated_row_count=(
                max(row_counts)
                if row_counts
                else None
            ),
            estimated_column_count=(
                max(column_counts)
                if column_counts
                else None
            ),
            has_native_structure=any(
                candidate.has_native_structure
                for candidate in candidates
            ),
            metadata=metadata,
        )

    def _deduplicate_candidates(
        self,
        candidates: Sequence[
            _TableCandidate
        ],
    ) -> list[_TableCandidate]:
        """
        Aynı tabloyu temsil eden yüksek örtüşmeli adayları kaldırır.
        """

        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.confidence,
                candidate.bounding_box.area,
            ),
            reverse=True,
        )

        result: list[
            _TableCandidate
        ] = []

        for candidate in ordered:
            duplicate = any(
                self._intersection_over_union(
                    candidate.bounding_box,
                    existing.bounding_box,
                )
                >= self.deduplication_iou_threshold
                for existing in result
            )

            if duplicate:
                continue

            result.append(
                candidate
            )

        return result

    def _cluster_nearby_boxes(
        self,
        boxes: Sequence[
            TableBoundingBox
        ],
    ) -> list[TableBoundingBox]:
        """
        Dikey olarak yakın bölgeleri tek tablo alanı altında birleştirir.
        """

        if not boxes:
            return []

        ordered = sorted(
            boxes,
            key=lambda box: (
                box.y0,
                box.x0,
            ),
        )

        clusters: list[
            list[TableBoundingBox]
        ] = []

        for box in ordered:
            if not clusters:
                clusters.append(
                    [box]
                )

                continue

            current_cluster = (
                clusters[-1]
            )

            union_box = self._union_boxes(
                current_cluster
            )

            vertical_gap = (
                box.y0
                - union_box.y1
            )

            horizontal_overlap = (
                self._horizontal_overlap_ratio(
                    union_box,
                    box,
                )
            )

            if (
                vertical_gap
                <= self.candidate_merge_gap
                and horizontal_overlap >= 0.25
            ):
                current_cluster.append(
                    box
                )
            else:
                clusters.append(
                    [box]
                )

        return [
            self._union_boxes(
                cluster
            )
            for cluster in clusters
        ]

    def _find_caption(
        self,
        *,
        page: pymupdf.Page,
        table_box: TableBoundingBox,
    ) -> str | None:
        """
        Tablo bölgesinin yakınındaki olası tablo başlığını bulur.
        """

        text_blocks = page.get_text(
            "blocks",
            sort=True,
        )

        candidates: list[
            tuple[float, str]
        ] = []

        for block in text_blocks:
            if len(block) < 5:
                continue

            text = str(
                block[4]
            ).strip()

            if not text:
                continue

            if self._caption_kind(text) is None:
                continue

            block_box = (
                self._to_bounding_box(
                    block[:4],
                    page.rect,
                )
            )

            if block_box is None:
                continue

            vertical_distance = min(
                abs(
                    block_box.y1
                    - table_box.y0
                ),
                abs(
                    table_box.y1
                    - block_box.y0
                ),
            )

            horizontal_overlap = (
                self._horizontal_overlap_ratio(
                    block_box,
                    table_box,
                )
            )

            if (
                vertical_distance
                <= self.caption_search_distance
                and horizontal_overlap >= 0.20
            ):
                candidates.append(
                    (
                        vertical_distance,
                        text,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        return candidates[0][1]

    def _extract_region_text(
        self,
        *,
        page: pymupdf.Page,
        bounding_box: TableBoundingBox,
        max_length: int,
    ) -> str:
        try:
            text = page.get_text(
                "text",
                clip=bounding_box.to_rect(),
                sort=True,
            ).strip()
        except Exception:
            return ""

        if len(text) <= max_length:
            return text

        return (
            text[:max_length].rstrip()
            + "..."
        )

    def _is_valid_candidate(
        self,
        candidate: _TableCandidate,
        *,
        page_rect: pymupdf.Rect,
    ) -> bool:
        box = candidate.bounding_box

        page_width = max(
            float(
                page_rect.width
            ),
            1.0,
        )

        page_height = max(
            float(
                page_rect.height
            ),
            1.0,
        )

        page_area = (
            page_width
            * page_height
        )

        width_ratio = (
            box.width
            / page_width
        )

        height_ratio = (
            box.height
            / page_height
        )

        area_ratio = (
            box.area
            / page_area
        )

        return (
            width_ratio
            >= self.minimum_width_ratio
            and height_ratio
            >= self.minimum_height_ratio
            and area_ratio
            >= self.minimum_area_ratio
        )

    def _adjust_confidence(
        self,
        *,
        candidate: _TableCandidate,
        text_preview: str,
        caption_text: str | None,
    ) -> float:
        confidence = (
            candidate.confidence
        )

        if caption_text:
            confidence += 0.04

        if text_preview:
            line_count = len(
                [
                    line
                    for line in text_preview.splitlines()
                    if line.strip()
                ]
            )

            numeric_token_count = sum(
                self._looks_numeric(
                    token
                )
                for token in text_preview.split()
            )

            if line_count >= 3:
                confidence += 0.03

            if numeric_token_count >= 3:
                confidence += 0.03

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
    def _looks_numeric(
        value: str,
    ) -> bool:
        normalized = (
            value.strip()
            .replace("%", "")
            .replace("₺", "")
            .replace("$", "")
            .replace("€", "")
            .replace(",", ".")
        )

        normalized = (
            normalized.replace(
                " ",
                "",
            )
        )

        try:
            float(
                normalized
            )
        except ValueError:
            return False

        return True

    @staticmethod
    def _to_bounding_box(
        value: Any,
        page_rect: pymupdf.Rect,
    ) -> TableBoundingBox | None:
        if value is None:
            return None

        if isinstance(
            value,
            TableBoundingBox,
        ):
            rect = (
                value.to_rect()
                & page_rect
            )

        elif isinstance(
            value,
            Mapping,
        ):
            try:
                rect = pymupdf.Rect(
                    float(value["x0"]),
                    float(value["y0"]),
                    float(value["x1"]),
                    float(value["y1"]),
                ) & page_rect

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                return None

        else:
            try:
                if all(
                    hasattr(
                        value,
                        attribute,
                    )
                    for attribute in (
                        "x0",
                        "y0",
                        "x1",
                        "y1",
                    )
                ):
                    rect = pymupdf.Rect(
                        float(value.x0),
                        float(value.y0),
                        float(value.x1),
                        float(value.y1),
                    ) & page_rect
                else:
                    rect = (
                        pymupdf.Rect(
                            value
                        )
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

        return TableBoundingBox(
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
    def _union_boxes(
        boxes: Sequence[
            TableBoundingBox
        ],
    ) -> TableBoundingBox:
        if not boxes:
            raise ValueError(
                "Birleştirmek için en az bir kutu gerekir."
            )

        return TableBoundingBox(
            x0=min(
                box.x0
                for box in boxes
            ),
            y0=min(
                box.y0
                for box in boxes
            ),
            x1=max(
                box.x1
                for box in boxes
            ),
            y1=max(
                box.y1
                for box in boxes
            ),
        )

    @staticmethod
    def _intersection_over_union(
        first: TableBoundingBox,
        second: TableBoundingBox,
    ) -> float:
        x0 = max(
            first.x0,
            second.x0,
        )

        y0 = max(
            first.y0,
            second.y0,
        )

        x1 = min(
            first.x1,
            second.x1,
        )

        y1 = min(
            first.y1,
            second.y1,
        )

        if x1 <= x0:
            return 0.0

        if y1 <= y0:
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

        return (
            intersection
            / union
        )

    @staticmethod
    def _overlap_ratio(
        first: TableBoundingBox,
        second: TableBoundingBox,
    ) -> float:
        x0 = max(
            first.x0,
            second.x0,
        )

        y0 = max(
            first.y0,
            second.y0,
        )

        x1 = min(
            first.x1,
            second.x1,
        )

        y1 = min(
            first.y1,
            second.y1,
        )

        if x1 <= x0:
            return 0.0

        if y1 <= y0:
            return 0.0

        intersection = (
            (x1 - x0)
            * (y1 - y0)
        )

        return min(
            intersection
            / max(
                first.area,
                1.0,
            ),
            1.0,
        )

    @staticmethod
    def _horizontal_overlap_ratio(
        first: TableBoundingBox,
        second: TableBoundingBox,
    ) -> float:
        x0 = max(
            first.x0,
            second.x0,
        )

        x1 = min(
            first.x1,
            second.x1,
        )

        if x1 <= x0:
            return 0.0

        overlap = (
            x1 - x0
        )

        minimum_width = max(
            min(
                first.width,
                second.width,
            ),
            1.0,
        )

        return min(
            overlap
            / minimum_width,
            1.0,
        )

    @staticmethod
    def _boxes_are_near(
        first: TableBoundingBox,
        second: TableBoundingBox,
        *,
        max_gap: float,
    ) -> bool:
        horizontal_overlap = (
            TableDetector._horizontal_overlap_ratio(
                first,
                second,
            )
        )

        vertical_gap = max(
            second.y0 - first.y1,
            first.y0 - second.y1,
            0.0,
        )

        return (
            horizontal_overlap >= 0.25
            and vertical_gap <= max_gap
        )

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

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
                int(
                    page_number
                )
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


def detect_pdf_tables(
    pdf_path: str | Path,
    *,
    page_numbers: Iterable[int] | None = None,
    layouts_by_page: Mapping[
        int,
        Any,
    ]
    | None = None,
) -> list[PageTableDetection]:
    """TableDetector sınıfı için kısa kullanım arayüzü."""

    detector = TableDetector()

    return detector.detect_pdf(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
        layouts_by_page=layouts_by_page,
    )