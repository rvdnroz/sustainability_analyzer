from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence
from uuid import UUID

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]

from .table_detector import (
    DetectedTable,
    TableBoundingBox,
)

if TYPE_CHECKING:
    from models.visual_evidence import VisualAsset


logger = logging.getLogger(__name__)


class VisualTableExtractionError(RuntimeError):
    """Görsel tablo çıkarımı tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class PixelBoundingBox:
    """Görüntü koordinat sistemindeki dikdörtgen bölge."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(
            self.x1 - self.x0,
            0,
        )

    @property
    def height(self) -> int:
        return max(
            self.y1 - self.y0,
            0,
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x0 + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y0 + self.height / 2

    def to_dict(self) -> dict[str, int]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass(slots=True, frozen=True)
class VisualTableCell:
    """Görsel yöntemle tespit edilen tek bir tablo hücresi."""

    row_index: int
    column_index: int

    pixel_box: PixelBoundingBox
    pdf_box: TableBoundingBox

    image_path: Path | None

    width_pixels: int
    height_pixels: int

    confidence: float

    is_header_candidate: bool
    requires_ocr: bool

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "column_index": self.column_index,
            "pixel_box": self.pixel_box.to_dict(),
            "pdf_box": self.pdf_box.to_dict(),
            "image_path": (
                str(self.image_path)
                if self.image_path is not None
                else None
            ),
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "confidence": self.confidence,
            "is_header_candidate": self.is_header_candidate,
            "requires_ocr": self.requires_ocr,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class VisualExtractedTable:
    """Görsel tablo çıkarımının yapılandırılmış sonucu."""

    table_id: str
    page_index: int
    page_number: int
    table_index: int

    pdf_bounding_box: TableBoundingBox
    pixel_bounding_box: PixelBoundingBox

    extraction_method: str
    extraction_confidence: float

    row_count: int
    column_count: int

    horizontal_lines: tuple[int, ...]
    vertical_lines: tuple[int, ...]

    cells: tuple[VisualTableCell, ...]

    table_image_path: Path | None
    debug_image_path: Path | None

    requires_ocr: bool
    is_valid: bool

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    # Hücre tabanlı grid çıkarımı başarısız olsa bile kırpılmış görselin
    # sonraki whole_image_ocr.py aşamasında korunmasını sağlar.
    requires_whole_image_ocr: bool = False
    visual_region_type: str = "table"

    @property
    def has_detected_cells(self) -> bool:
        return bool(self.cells)

    @property
    def ocr_strategy(self) -> str:
        if self.requires_whole_image_ocr:
            return "whole_image"
        if self.cells:
            return "cell"
        return "none"

    @property
    def can_run_whole_image_ocr(self) -> bool:
        return (
            self.requires_whole_image_ocr
            and self.table_image_path is not None
            and Path(self.table_image_path).is_file()
        )

    def to_visual_asset(
        self,
        *,
        report_id: UUID,
        page_id: UUID | None = None,
        source_page_image_path: str | Path | None = None,
        category: str | None = None,
        topic: str | None = None,
        keywords: Sequence[str] = (),
    ) -> "VisualAsset":
        """
        Görsel tablo ara sonucunu ortak VisualAsset sözleşmesine dönüştürür.

        VisualTableExtractor rapor kimliğini bilmediği için bu dönüşüm,
        ingestion pipeline rapor bağlamını sağladığında yapılır.
        """

        if self.table_image_path is None:
            raise VisualTableExtractionError(
                "VisualAsset oluşturmak için tablo görüntü yolu bulunamadı."
            )

        image_path = Path(self.table_image_path).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Tablo görüntüsü bulunamadı: {image_path}"
            )

        from models.page import BoundingBox
        from models.visual_evidence import (
            VisualAsset,
            VisualProcessingStatus,
            VisualRegionType,
        )

        try:
            region_type = VisualRegionType(self.visual_region_type)
        except ValueError:
            region_type = VisualRegionType.UNKNOWN

        processing_status = (
            VisualProcessingStatus.IMAGE_EXTRACTED
            if not self.requires_whole_image_ocr
            else VisualProcessingStatus.DETECTED
        )

        metadata = {
            **dict(self.metadata),
            "table_id": self.table_id,
            "table_index": self.table_index,
            "structured_grid_valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "cell_count": len(self.cells),
            "ocr_strategy": self.ocr_strategy,
            "visual_extraction_method": self.extraction_method,
            "visual_warnings": list(self.warnings),
            "pixel_bounding_box": self.pixel_bounding_box.to_dict(),
        }

        return VisualAsset(
            report_id=report_id,
            page_id=page_id,
            page_number=self.page_number,
            region_type=region_type,
            bounding_box=BoundingBox(
                x1=self.pdf_bounding_box.x0,
                y1=self.pdf_bounding_box.y0,
                x2=self.pdf_bounding_box.x1,
                y2=self.pdf_bounding_box.y1,
            ),
            image_path=image_path,
            source_page_image_path=(
                Path(source_page_image_path).expanduser().resolve()
                if source_page_image_path is not None
                else None
            ),
            image_width=self.pixel_bounding_box.width,
            image_height=self.pixel_bounding_box.height,
            image_hash=self.content_hash,
            caption=self.metadata.get("caption_text"),
            nearby_text=self.metadata.get("text_preview"),
            category=category,
            topic=topic,
            keywords=list(keywords),
            detection_confidence=self.extraction_confidence,
            processing_status=processing_status,
            requires_whole_image_ocr=self.requires_whole_image_ocr,
            requires_vision=region_type in {
                VisualRegionType.CHART,
                VisualRegionType.METRIC_CARD,
                VisualRegionType.INFOGRAPHIC,
                VisualRegionType.FIGURE,
            },
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "pdf_bounding_box": (
                self.pdf_bounding_box.to_dict()
            ),
            "pixel_bounding_box": (
                self.pixel_bounding_box.to_dict()
            ),
            "extraction_method": self.extraction_method,
            "extraction_confidence": (
                self.extraction_confidence
            ),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "horizontal_lines": list(
                self.horizontal_lines
            ),
            "vertical_lines": list(
                self.vertical_lines
            ),
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "table_image_path": (
                str(self.table_image_path)
                if self.table_image_path is not None
                else None
            ),
            "debug_image_path": (
                str(self.debug_image_path)
                if self.debug_image_path is not None
                else None
            ),
            "requires_ocr": self.requires_ocr,
            "requires_whole_image_ocr": (
                self.requires_whole_image_ocr
            ),
            "visual_region_type": self.visual_region_type,
            "ocr_strategy": self.ocr_strategy,
            "has_detected_cells": self.has_detected_cells,
            "is_valid": self.is_valid,
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PageVisualTableExtraction:
    """Tek sayfadaki görsel tablo çıkarım sonuçları."""

    page_index: int
    page_number: int

    table_count: int
    successful_table_count: int
    failed_table_count: int

    tables: tuple[VisualExtractedTable, ...]
    warnings: tuple[str, ...]

    @property
    def structured_table_count(self) -> int:
        return sum(table.is_valid for table in self.tables)

    @property
    def whole_image_ocr_candidate_count(self) -> int:
        return sum(
            table.requires_whole_image_ocr
            for table in self.tables
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_count": self.table_count,
            "successful_table_count": (
                self.successful_table_count
            ),
            "failed_table_count": (
                self.failed_table_count
            ),
            "structured_table_count": (
                self.structured_table_count
            ),
            "whole_image_ocr_candidate_count": (
                self.whole_image_ocr_candidate_count
            ),
            "tables": [
                table.to_dict()
                for table in self.tables
            ],
            "warnings": list(self.warnings),
        }


class VisualTableExtractor:
    """
    Native yapısı bulunmayan tabloları görüntü işleme ile çıkarır.

    İşlem sırası:

    1. PDF sayfası belirlenen DPI ile görüntüye dönüştürülür.
    2. TableDetector tarafından verilen tablo bölgesi kırpılır.
    3. Görüntü gri tona ve ikili görüntüye çevrilir.
    4. Yatay ve dikey çizgiler morfolojik işlemlerle bulunur.
    5. Çizgi koordinatları kümelenir.
    6. Satır, sütun ve hücre sınırları oluşturulur.
    7. Her hücre ayrı görüntü olarak isteğe bağlı kaydedilir.
    8. Grid geçerliyse hücreler table_ocr.py aşamasına aktarılır.
    9. Grid veya hücre bulunamazsa kırpılmış görsel kaybedilmez;
       whole_image_ocr.py aşaması için işaretlenir.

    Bu modül hücre metinlerini okumaz. Hücre metinleri table_ocr.py,
    hücresiz görseller ise whole_image_ocr.py tarafından işlenir.
    """

    def __init__(
        self,
        *,
        dpi: int = 200,
        adaptive_block_size: int = 31,
        adaptive_constant: int = 15,
        horizontal_kernel_ratio: float = 0.035,
        vertical_kernel_ratio: float = 0.035,
        line_merge_tolerance: int = 8,
        minimum_cell_width: int = 18,
        minimum_cell_height: int = 12,
        minimum_rows: int = 2,
        minimum_columns: int = 2,
        minimum_region_width: int = 300,
        minimum_region_height: int = 120,
        minimum_region_area: int = 50_000,
        crop_padding: int = 4,
        save_table_images: bool = True,
        save_cell_images: bool = True,
        save_debug_images: bool = False,
        image_extension: str = ".png",
    ) -> None:
        if dpi < 72:
            raise ValueError(
                "dpi en az 72 olmalıdır."
            )

        if adaptive_block_size < 3:
            raise ValueError(
                "adaptive_block_size en az 3 olmalıdır."
            )

        if adaptive_block_size % 2 == 0:
            raise ValueError(
                "adaptive_block_size tek sayı olmalıdır."
            )

        for name, value in {
            "horizontal_kernel_ratio": (
                horizontal_kernel_ratio
            ),
            "vertical_kernel_ratio": (
                vertical_kernel_ratio
            ),
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        if line_merge_tolerance < 0:
            raise ValueError(
                "line_merge_tolerance negatif olamaz."
            )

        if minimum_cell_width < 1:
            raise ValueError(
                "minimum_cell_width en az 1 olmalıdır."
            )

        if minimum_cell_height < 1:
            raise ValueError(
                "minimum_cell_height en az 1 olmalıdır."
            )

        if minimum_rows < 1:
            raise ValueError(
                "minimum_rows en az 1 olmalıdır."
            )

        if minimum_columns < 1:
            raise ValueError(
                "minimum_columns en az 1 olmalıdır."
            )

        if minimum_region_width < 1:
            raise ValueError(
                "minimum_region_width en az 1 olmalıdır."
            )

        if minimum_region_height < 1:
            raise ValueError(
                "minimum_region_height en az 1 olmalıdır."
            )

        if minimum_region_area < 1:
            raise ValueError(
                "minimum_region_area en az 1 olmalıdır."
            )

        if crop_padding < 0:
            raise ValueError(
                "crop_padding negatif olamaz."
            )

        normalized_extension = (
            image_extension.lower()
        )

        if not normalized_extension.startswith("."):
            normalized_extension = (
                f".{normalized_extension}"
            )

        if normalized_extension not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            raise ValueError(
                "Desteklenmeyen görüntü uzantısı: "
                f"{normalized_extension}"
            )

        self.dpi = dpi
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_constant = adaptive_constant

        self.horizontal_kernel_ratio = (
            horizontal_kernel_ratio
        )

        self.vertical_kernel_ratio = (
            vertical_kernel_ratio
        )

        self.line_merge_tolerance = (
            line_merge_tolerance
        )

        self.minimum_cell_width = (
            minimum_cell_width
        )

        self.minimum_cell_height = (
            minimum_cell_height
        )

        self.minimum_rows = minimum_rows
        self.minimum_columns = minimum_columns
        self.minimum_region_width = minimum_region_width
        self.minimum_region_height = minimum_region_height
        self.minimum_region_area = minimum_region_area

        self.crop_padding = crop_padding

        self.save_table_images = (
            save_table_images
        )

        self.save_cell_images = (
            save_cell_images
        )

        self.save_debug_images = (
            save_debug_images
        )

        self.image_extension = (
            normalized_extension
        )

    def extract_pdf(
        self,
        pdf_path: str | Path,
        *,
        detected_tables_by_page: Mapping[
            int,
            Sequence[DetectedTable],
        ],
        output_dir: str | Path | None = None,
        page_numbers: Iterable[int] | None = None,
    ) -> list[PageVisualTableExtraction]:
        """
        PDF içindeki görsel tabloları çıkarır.

        detected_tables_by_page anahtarları 1 tabanlı sayfa
        numaralarıdır.
        """

        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(
            pdf_path
        )

        if output_dir is not None:
            output_root = Path(
                output_dir
            ).expanduser().resolve()

            output_root.mkdir(
                parents=True,
                exist_ok=True,
            )
        else:
            output_root = None

        try:
            document = pymupdf.open(
                pdf_path
            )
        except Exception as exc:
            raise VisualTableExtractionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise VisualTableExtractionError(
                    "Parola korumalı PDF analiz edilemez."
                )

            if document.page_count == 0:
                raise VisualTableExtractionError(
                    "PDF içinde sayfa bulunamadı."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            results: list[
                PageVisualTableExtraction
            ] = []

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                detected_tables = list(
                    detected_tables_by_page.get(
                        page_number,
                        (),
                    )
                )

                visual_tables = [
                    table
                    for table in detected_tables
                    if (
                        table.requires_visual_extraction
                        or not table.has_native_structure
                    )
                ]

                result = self.extract_page(
                    page,
                    detected_tables=visual_tables,
                    output_dir=output_root,
                )

                results.append(result)

            logger.info(
                "%d sayfada %d görsel bölge çıkarıldı; "
                "%d bölge bütün-görüntü OCR bekliyor: %s",
                len(results),
                sum(
                    result.successful_table_count
                    for result in results
                ),
                sum(
                    result.whole_image_ocr_candidate_count
                    for result in results
                ),
                pdf_path,
            )

            return results

        finally:
            document.close()

    def extract_page(
        self,
        page: pymupdf.Page,
        *,
        detected_tables: Sequence[DetectedTable],
        output_dir: str | Path | None = None,
    ) -> PageVisualTableExtraction:
        """Açık bir PyMuPDF sayfasındaki tablo bölgelerini işler."""

        page_index = int(
            page.number
        )

        page_number = (
            page_index + 1
        )

        output_root = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else None
        )

        page_image = self._render_page(
            page
        )

        page_height_pixels, page_width_pixels = (
            page_image.shape[:2]
        )

        extracted_tables: list[
            VisualExtractedTable
        ] = []

        warnings: list[str] = []
        failed_count = 0

        for output_index, detected_table in enumerate(
            detected_tables
        ):
            preview_box = self._pdf_box_to_pixel_box(
                pdf_box=detected_table.bounding_box,
                page_rect=page.rect,
                image_width=page_width_pixels,
                image_height=page_height_pixels,
            )

            if (
                preview_box.width < self.minimum_region_width
                or preview_box.height < self.minimum_region_height
                or preview_box.area < self.minimum_region_area
            ):
                warnings.append(
                    f"Sayfa {page_number}, tablo {output_index + 1} "
                    "OCR kuyruğuna alınmadı: görsel bölge çok küçük "
                    f"({preview_box.width}x{preview_box.height}, "
                    f"alan={preview_box.area})."
                )
                continue

            try:
                result = self._extract_single_table(
                    page=page,
                    page_image=page_image,
                    page_width_pixels=page_width_pixels,
                    page_height_pixels=page_height_pixels,
                    detected_table=detected_table,
                    output_index=output_index,
                    output_dir=output_root,
                )

            except Exception as exc:
                failed_count += 1

                message = (
                    f"Sayfa {page_number}, "
                    f"tablo {output_index + 1} "
                    f"görsel olarak çıkarılamadı: {exc}"
                )

                warnings.append(message)

                logger.warning(message)

                continue

            extracted_tables.append(result)

        return PageVisualTableExtraction(
            page_index=page_index,
            page_number=page_number,
            table_count=len(detected_tables),
            successful_table_count=len(
                extracted_tables
            ),
            failed_table_count=failed_count,
            tables=tuple(
                extracted_tables
            ),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _resolve_visual_region_type(
        detected_table: DetectedTable,
        *,
        is_valid_grid: bool,
        row_count: int,
        column_count: int,
        cell_coverage_ratio: float,
    ) -> str:
        """
        Detector sınıflandırmasını OpenCV grid sonucu ile uzlaştırır.

        Grafiklerde eksen ve bar çizgileri sahte hücre grid'i oluşturabilir.
        Bu nedenle yalnızca güçlü tablo grid'i bulunduğunda ``table`` kararı
        verilir; detector'ın yüksek güvenli chart/infographic/metric_card
        kararı korunur.
        """

        metadata = dict(
            getattr(detected_table, "metadata", {}) or {}
        )

        inferred = str(
            metadata.get("visual_region_type")
            or ""
        ).strip().casefold()

        confidence = float(
            metadata.get("visual_region_confidence")
            or 0.0
        )

        allowed = {
            "table",
            "chart",
            "metric_card",
            "infographic",
            "figure",
            "decorative",
            "unknown",
        }

        if (
            inferred in {
                "chart",
                "metric_card",
                "infographic",
                "figure",
            }
            and confidence >= 0.70
        ):
            return inferred

        strong_table_grid = (
            is_valid_grid
            and row_count >= 3
            and column_count >= 2
            and cell_coverage_ratio >= 0.80
        )

        if strong_table_grid:
            return "table"

        if inferred in allowed:
            return inferred

        table_type = str(
            metadata.get("table_type")
            or ""
        ).strip().casefold()

        if table_type == "kpi_or_indicator_table":
            return "metric_card"

        return "unknown"

    def _extract_single_table(
        self,
        *,
        page: pymupdf.Page,
        page_image: np.ndarray,
        page_width_pixels: int,
        page_height_pixels: int,
        detected_table: DetectedTable,
        output_index: int,
        output_dir: Path | None,
    ) -> VisualExtractedTable:
        page_number = int(
            page.number
        ) + 1

        pixel_box = self._pdf_box_to_pixel_box(
            pdf_box=detected_table.bounding_box,
            page_rect=page.rect,
            image_width=page_width_pixels,
            image_height=page_height_pixels,
        )

        padded_box = self._apply_padding(
            pixel_box,
            image_width=page_width_pixels,
            image_height=page_height_pixels,
        )

        table_image = page_image[
            padded_box.y0:padded_box.y1,
            padded_box.x0:padded_box.x1,
        ].copy()

        if table_image.size == 0:
            raise VisualTableExtractionError(
                "Tablo kırpma bölgesi boş."
            )

        binary_image = self._prepare_binary_image(
            table_image
        )

        horizontal_mask = (
            self._extract_horizontal_lines(
                binary_image
            )
        )

        vertical_mask = (
            self._extract_vertical_lines(
                binary_image
            )
        )

        grid_mask = cv2.bitwise_or(
            horizontal_mask,
            vertical_mask,
        )

        horizontal_lines = (
            self._detect_horizontal_line_positions(
                horizontal_mask
            )
        )

        vertical_lines = (
            self._detect_vertical_line_positions(
                vertical_mask
            )
        )

        warnings: list[str] = []

        if len(horizontal_lines) < 2:
            warnings.append(
                "Yeterli yatay çizgi tespit edilemedi."
            )

        if len(vertical_lines) < 2:
            warnings.append(
                "Yeterli dikey çizgi tespit edilemedi."
            )

        if (
            len(horizontal_lines) < 2
            or len(vertical_lines) < 2
        ):
            contour_cells = (
                self._detect_cells_from_contours(
                    grid_mask
                )
            )

            (
                horizontal_lines,
                vertical_lines,
            ) = self._infer_lines_from_cells(
                contour_cells,
                existing_horizontal=horizontal_lines,
                existing_vertical=vertical_lines,
            )

        cells = self._build_cells(
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            table_image=table_image,
            table_pdf_box=detected_table.bounding_box,
            table_pixel_box=padded_box,
            page_rect=page.rect,
            page_width_pixels=page_width_pixels,
            page_height_pixels=page_height_pixels,
            output_dir=output_dir,
            page_number=page_number,
            table_index=detected_table.table_index,
        )

        row_count = max(
            len(horizontal_lines) - 1,
            0,
        )

        column_count = max(
            len(vertical_lines) - 1,
            0,
        )

        expected_cell_count = row_count * column_count
        cell_coverage_ratio = (
            min(len(cells) / expected_cell_count, 1.0)
            if expected_cell_count > 0
            else 0.0
        )

        is_valid = (
            row_count >= self.minimum_rows
            and column_count >= self.minimum_columns
            and cell_coverage_ratio >= 0.60
        )

        visual_region_type = self._resolve_visual_region_type(
            detected_table,
            is_valid_grid=is_valid,
            row_count=row_count,
            column_count=column_count,
            cell_coverage_ratio=cell_coverage_ratio,
        )

        # Grid geçerli değilse hücre OCR'ı yerine bütün-görüntü OCR
        # kullanılmalıdır. Hücrelerin kısmen bulunması bu kararı bozmaz;
        # eksik grid, metriklerin kaybolmasına yol açabilir.
        requires_whole_image_ocr = not is_valid
        requires_cell_ocr = bool(cells) and is_valid

        if not cells:
            warnings.append(
                "Hücre yapısı tespit edilemedi; kırpılmış görsel "
                "bütün-görüntü OCR için korunacak."
            )
        elif requires_whole_image_ocr:
            warnings.append(
                "Hücre grid kapsamı yetersiz; bütün-görüntü OCR "
                "tercih edilecek."
            )

        if row_count < self.minimum_rows:
            warnings.append(
                f"Satır sayısı minimum eşikten düşük: {row_count}."
            )

        if column_count < self.minimum_columns:
            warnings.append(
                "Sütun sayısı minimum eşikten düşük: "
                f"{column_count}."
            )

        table_image_path: Path | None = None
        debug_image_path: Path | None = None

        # Whole-image OCR için tablo görselinin diskte bulunması zorunludur.
        # Bu nedenle save_table_images=False olsa bile fallback adayı olan
        # bölgeler kaydedilir.
        if (
            output_dir is not None
            and (
                self.save_table_images
                or requires_whole_image_ocr
            )
        ):
            table_image_path = self._save_table_image(
                table_image=table_image,
                output_dir=output_dir,
                page_number=page_number,
                table_index=detected_table.table_index,
            )

        if requires_whole_image_ocr and table_image_path is None:
            warnings.append(
                "Bütün-görüntü OCR gerekli ancak tablo görüntüsü "
                "kaydedilemedi; output_dir sağlanmalıdır."
            )

        if (
            output_dir is not None
            and self.save_debug_images
        ):
            debug_image = (
                self._create_debug_image(
                    table_image=table_image,
                    horizontal_lines=horizontal_lines,
                    vertical_lines=vertical_lines,
                )
            )

            debug_image_path = self._save_debug_image(
                debug_image=debug_image,
                output_dir=output_dir,
                page_number=page_number,
                table_index=detected_table.table_index,
            )

        confidence = self._calculate_confidence(
            detected_table=detected_table,
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            cells=cells,
            is_valid=is_valid,
            warnings=warnings,
        )

        content_hash = self._calculate_content_hash(
            table_image
        )

        return VisualExtractedTable(
            table_id=detected_table.table_id,
            page_index=int(page.number),
            page_number=page_number,
            table_index=detected_table.table_index,
            pdf_bounding_box=(
                detected_table.bounding_box
            ),
            pixel_bounding_box=padded_box,
            extraction_method=(
                "opencv_morphological_grid"
                if is_valid
                else "opencv_visual_region"
            ),
            extraction_confidence=confidence,
            row_count=row_count,
            column_count=column_count,
            horizontal_lines=tuple(
                horizontal_lines
            ),
            vertical_lines=tuple(
                vertical_lines
            ),
            cells=tuple(cells),
            table_image_path=table_image_path,
            debug_image_path=debug_image_path,
            requires_ocr=(
                requires_cell_ocr
                or requires_whole_image_ocr
            ),
            is_valid=is_valid,
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "dpi": self.dpi,
                "source_detection_method": (
                    detected_table.detection_method
                ),
                "source_detection_confidence": (
                    detected_table.confidence
                ),
                "original_pixel_box": (
                    pixel_box.to_dict()
                ),
                "grid_pixel_count": int(
                    np.count_nonzero(
                        grid_mask
                    )
                ),
                "horizontal_pixel_count": int(
                    np.count_nonzero(
                        horizontal_mask
                    )
                ),
                "vertical_pixel_count": int(
                    np.count_nonzero(
                        vertical_mask
                    )
                ),
                "has_detected_cells": bool(cells),
                "cell_count": len(cells),
                "expected_cell_count": expected_cell_count,
                "cell_coverage_ratio": round(
                    cell_coverage_ratio,
                    4,
                ),
                "structured_grid_valid": is_valid,
                "requires_cell_ocr": requires_cell_ocr,
                "requires_whole_image_ocr": (
                    requires_whole_image_ocr
                ),
                "ocr_strategy": (
                    "cell"
                    if requires_cell_ocr
                    else "whole_image"
                ),
                "visual_candidate_preserved": True,
                "visual_region_type": visual_region_type,
                "caption_text": detected_table.caption_text,
                "text_preview": detected_table.text_preview,
                "estimated_row_count": (
                    detected_table.estimated_row_count
                ),
                "estimated_column_count": (
                    detected_table.estimated_column_count
                ),
                "foreground_ratio": round(
                    self._foreground_ratio(table_image),
                    6,
                ),
                "aspect_ratio": round(
                    table_image.shape[1]
                    / max(table_image.shape[0], 1),
                    6,
                ),
            },
            requires_whole_image_ocr=requires_whole_image_ocr,
            visual_region_type=visual_region_type,
        )

    def _render_page(
        self,
        page: pymupdf.Page,
    ) -> np.ndarray:
        try:
            pixmap = page.get_pixmap(
                dpi=self.dpi,
                alpha=False,
                colorspace=pymupdf.csRGB,
            )
        except Exception as exc:
            raise VisualTableExtractionError(
                f"Sayfa {int(page.number) + 1} render edilemedi."
            ) from exc

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        )

        expected_size = (
            pixmap.height
            * pixmap.width
            * pixmap.n
        )

        if image.size != expected_size:
            raise VisualTableExtractionError(
                "Render görüntüsü beklenen boyutta değil."
            )

        image = image.reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        if pixmap.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR,
            )

        elif pixmap.n == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        elif pixmap.n == 1:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR,
            )

        else:
            raise VisualTableExtractionError(
                f"Desteklenmeyen kanal sayısı: {pixmap.n}"
            )

        return image

    def _prepare_binary_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_constant,
        )

        return binary

    def _extract_horizontal_lines(
        self,
        binary_image: np.ndarray,
    ) -> np.ndarray:
        image_width = binary_image.shape[1]

        kernel_width = max(
            int(
                image_width
                * self.horizontal_kernel_ratio
            ),
            8,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_width,
                1,
            ),
        )

        horizontal = cv2.morphologyEx(
            binary_image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        horizontal = cv2.dilate(
            horizontal,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 1),
            ),
            iterations=1,
        )

        return horizontal

    def _extract_vertical_lines(
        self,
        binary_image: np.ndarray,
    ) -> np.ndarray:
        image_height = binary_image.shape[0]

        kernel_height = max(
            int(
                image_height
                * self.vertical_kernel_ratio
            ),
            8,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                kernel_height,
            ),
        )

        vertical = cv2.morphologyEx(
            binary_image,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        vertical = cv2.dilate(
            vertical,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, 3),
            ),
            iterations=1,
        )

        return vertical

    def _detect_horizontal_line_positions(
        self,
        mask: np.ndarray,
    ) -> list[int]:
        projection = np.sum(
            mask > 0,
            axis=1,
        )

        threshold = max(
            int(mask.shape[1] * 0.15),
            5,
        )

        positions = np.where(
            projection >= threshold
        )[0].tolist()

        return self._cluster_positions(
            positions
        )

    def _detect_vertical_line_positions(
        self,
        mask: np.ndarray,
    ) -> list[int]:
        projection = np.sum(
            mask > 0,
            axis=0,
        )

        threshold = max(
            int(mask.shape[0] * 0.15),
            5,
        )

        positions = np.where(
            projection >= threshold
        )[0].tolist()

        return self._cluster_positions(
            positions
        )

    def _cluster_positions(
        self,
        positions: Sequence[int],
    ) -> list[int]:
        if not positions:
            return []

        ordered = sorted(
            set(
                int(position)
                for position in positions
            )
        )

        clusters: list[list[int]] = [
            [ordered[0]]
        ]

        for position in ordered[1:]:
            if (
                position
                - clusters[-1][-1]
                <= self.line_merge_tolerance
            ):
                clusters[-1].append(
                    position
                )
            else:
                clusters.append(
                    [position]
                )

        return [
            int(
                round(
                    sum(cluster)
                    / len(cluster)
                )
            )
            for cluster in clusters
        ]

    def _detect_cells_from_contours(
        self,
        grid_mask: np.ndarray,
    ) -> list[PixelBoundingBox]:
        contours, _ = cv2.findContours(
            grid_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes: list[
            PixelBoundingBox
        ] = []

        image_area = (
            grid_mask.shape[0]
            * grid_mask.shape[1]
        )

        for contour in contours:
            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            if width < self.minimum_cell_width:
                continue

            if height < self.minimum_cell_height:
                continue

            area = width * height

            if area >= image_area * 0.95:
                continue

            boxes.append(
                PixelBoundingBox(
                    x0=int(x),
                    y0=int(y),
                    x1=int(x + width),
                    y1=int(y + height),
                )
            )

        return self._deduplicate_pixel_boxes(
            boxes
        )

    def _infer_lines_from_cells(
        self,
        cells: Sequence[PixelBoundingBox],
        *,
        existing_horizontal: Sequence[int],
        existing_vertical: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        horizontal_positions = list(
            existing_horizontal
        )

        vertical_positions = list(
            existing_vertical
        )

        for cell in cells:
            horizontal_positions.extend(
                [
                    cell.y0,
                    cell.y1,
                ]
            )

            vertical_positions.extend(
                [
                    cell.x0,
                    cell.x1,
                ]
            )

        return (
            self._cluster_positions(
                horizontal_positions
            ),
            self._cluster_positions(
                vertical_positions
            ),
        )

    def _build_cells(
        self,
        *,
        horizontal_lines: Sequence[int],
        vertical_lines: Sequence[int],
        table_image: np.ndarray,
        table_pdf_box: TableBoundingBox,
        table_pixel_box: PixelBoundingBox,
        page_rect: pymupdf.Rect,
        page_width_pixels: int,
        page_height_pixels: int,
        output_dir: Path | None,
        page_number: int,
        table_index: int,
    ) -> list[VisualTableCell]:
        if len(horizontal_lines) < 2:
            return []

        if len(vertical_lines) < 2:
            return []

        cells: list[
            VisualTableCell
        ] = []

        table_height, table_width = (
            table_image.shape[:2]
        )

        for row_index in range(
            len(horizontal_lines) - 1
        ):
            local_y0 = int(
                horizontal_lines[
                    row_index
                ]
            )

            local_y1 = int(
                horizontal_lines[
                    row_index + 1
                ]
            )

            for column_index in range(
                len(vertical_lines) - 1
            ):
                local_x0 = int(
                    vertical_lines[
                        column_index
                    ]
                )

                local_x1 = int(
                    vertical_lines[
                        column_index + 1
                    ]
                )

                local_x0 = max(
                    local_x0,
                    0,
                )

                local_y0 = max(
                    local_y0,
                    0,
                )

                local_x1 = min(
                    local_x1,
                    table_width,
                )

                local_y1 = min(
                    local_y1,
                    table_height,
                )

                width = (
                    local_x1
                    - local_x0
                )

                height = (
                    local_y1
                    - local_y0
                )

                if width < self.minimum_cell_width:
                    continue

                if height < self.minimum_cell_height:
                    continue

                global_x0 = (
                    table_pixel_box.x0
                    + local_x0
                )

                global_y0 = (
                    table_pixel_box.y0
                    + local_y0
                )

                global_x1 = (
                    table_pixel_box.x0
                    + local_x1
                )

                global_y1 = (
                    table_pixel_box.y0
                    + local_y1
                )

                global_pixel_box = (
                    PixelBoundingBox(
                        x0=global_x0,
                        y0=global_y0,
                        x1=global_x1,
                        y1=global_y1,
                    )
                )

                pdf_box = (
                    self._pixel_box_to_pdf_box(
                        pixel_box=global_pixel_box,
                        page_rect=page_rect,
                        image_width=page_width_pixels,
                        image_height=page_height_pixels,
                    )
                )

                cell_image = table_image[
                    local_y0:local_y1,
                    local_x0:local_x1,
                ]

                cell_image_path: Path | None = None

                if (
                    output_dir is not None
                    and self.save_cell_images
                ):
                    cell_image_path = (
                        self._save_cell_image(
                            cell_image=cell_image,
                            output_dir=output_dir,
                            page_number=page_number,
                            table_index=table_index,
                            row_index=row_index,
                            column_index=column_index,
                        )
                    )

                foreground_ratio = (
                    self._foreground_ratio(
                        cell_image
                    )
                )

                confidence = (
                    self._calculate_cell_confidence(
                        width=width,
                        height=height,
                        foreground_ratio=(
                            foreground_ratio
                        ),
                    )
                )

                cells.append(
                    VisualTableCell(
                        row_index=row_index,
                        column_index=(
                            column_index
                        ),
                        pixel_box=(
                            global_pixel_box
                        ),
                        pdf_box=pdf_box,
                        image_path=(
                            cell_image_path
                        ),
                        width_pixels=width,
                        height_pixels=height,
                        confidence=confidence,
                        is_header_candidate=(
                            row_index == 0
                        ),
                        requires_ocr=True,
                        metadata={
                            "local_pixel_box": {
                                "x0": local_x0,
                                "y0": local_y0,
                                "x1": local_x1,
                                "y1": local_y1,
                            },
                            "foreground_ratio": (
                                foreground_ratio
                            ),
                            "source_table_pdf_box": (
                                table_pdf_box.to_dict()
                            ),
                        },
                    )
                )

        return cells

    def _save_table_image(
        self,
        *,
        table_image: np.ndarray,
        output_dir: Path,
        page_number: int,
        table_index: int,
    ) -> Path:
        table_dir = (
            output_dir
            / f"page_{page_number:04d}"
            / f"table_{table_index + 1:03d}"
        )

        table_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_path = (
            table_dir
            / f"table{self.image_extension}"
        )

        self._write_image(
            target_path,
            table_image,
        )

        return target_path

    def _save_cell_image(
        self,
        *,
        cell_image: np.ndarray,
        output_dir: Path,
        page_number: int,
        table_index: int,
        row_index: int,
        column_index: int,
    ) -> Path:
        cell_dir = (
            output_dir
            / f"page_{page_number:04d}"
            / f"table_{table_index + 1:03d}"
            / "cells"
        )

        cell_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_path = (
            cell_dir
            / (
                f"row_{row_index + 1:03d}"
                f"_col_{column_index + 1:03d}"
                f"{self.image_extension}"
            )
        )

        self._write_image(
            target_path,
            cell_image,
        )

        return target_path

    def _save_debug_image(
        self,
        *,
        debug_image: np.ndarray,
        output_dir: Path,
        page_number: int,
        table_index: int,
    ) -> Path:
        debug_dir = (
            output_dir
            / f"page_{page_number:04d}"
            / f"table_{table_index + 1:03d}"
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_path = (
            debug_dir
            / f"debug_grid{self.image_extension}"
        )

        self._write_image(
            target_path,
            debug_image,
        )

        return target_path

    def _write_image(
        self,
        target_path: Path,
        image: np.ndarray,
    ) -> None:
        if image.size == 0:
            raise VisualTableExtractionError(
                "Boş görüntü dosyaya yazılamaz."
            )

        success = cv2.imwrite(
            str(target_path),
            image,
        )

        if not success:
            raise VisualTableExtractionError(
                f"Görüntü yazılamadı: {target_path}"
            )

    def _create_debug_image(
        self,
        *,
        table_image: np.ndarray,
        horizontal_lines: Sequence[int],
        vertical_lines: Sequence[int],
    ) -> np.ndarray:
        debug_image = table_image.copy()

        image_height, image_width = (
            debug_image.shape[:2]
        )

        for y_position in horizontal_lines:
            cv2.line(
                debug_image,
                (0, int(y_position)),
                (
                    image_width - 1,
                    int(y_position),
                ),
                (0, 0, 255),
                1,
            )

        for x_position in vertical_lines:
            cv2.line(
                debug_image,
                (int(x_position), 0),
                (
                    int(x_position),
                    image_height - 1,
                ),
                (255, 0, 0),
                1,
            )

        return debug_image

    def _calculate_confidence(
        self,
        *,
        detected_table: DetectedTable,
        horizontal_lines: Sequence[int],
        vertical_lines: Sequence[int],
        cells: Sequence[VisualTableCell],
        is_valid: bool,
        warnings: Sequence[str],
    ) -> float:
        confidence = (
            0.45
            + 0.25
            * detected_table.confidence
        )

        row_count = max(
            len(horizontal_lines) - 1,
            0,
        )

        column_count = max(
            len(vertical_lines) - 1,
            0,
        )

        if row_count >= self.minimum_rows:
            confidence += 0.08

        if column_count >= self.minimum_columns:
            confidence += 0.08

        expected_cell_count = (
            row_count
            * column_count
        )

        if expected_cell_count > 0:
            coverage_ratio = min(
                len(cells)
                / expected_cell_count,
                1.0,
            )

            confidence += (
                0.10
                * coverage_ratio
            )

        if is_valid:
            confidence += 0.04

        confidence -= min(
            len(warnings) * 0.05,
            0.20,
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
    def _calculate_cell_confidence(
        *,
        width: int,
        height: int,
        foreground_ratio: float,
    ) -> float:
        confidence = 0.65

        if width >= 30:
            confidence += 0.08

        if height >= 20:
            confidence += 0.08

        if 0.005 <= foreground_ratio <= 0.75:
            confidence += 0.12

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
    def _foreground_ratio(
        image: np.ndarray,
    ) -> float:
        if image.size == 0:
            return 0.0

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV
            | cv2.THRESH_OTSU,
        )

        foreground_pixels = int(
            np.count_nonzero(
                binary
            )
        )

        return round(
            foreground_pixels
            / max(
                binary.size,
                1,
            ),
            6,
        )

    def _apply_padding(
        self,
        box: PixelBoundingBox,
        *,
        image_width: int,
        image_height: int,
    ) -> PixelBoundingBox:
        return PixelBoundingBox(
            x0=max(
                box.x0 - self.crop_padding,
                0,
            ),
            y0=max(
                box.y0 - self.crop_padding,
                0,
            ),
            x1=min(
                box.x1 + self.crop_padding,
                image_width,
            ),
            y1=min(
                box.y1 + self.crop_padding,
                image_height,
            ),
        )

    @staticmethod
    def _pdf_box_to_pixel_box(
        *,
        pdf_box: TableBoundingBox,
        page_rect: pymupdf.Rect,
        image_width: int,
        image_height: int,
    ) -> PixelBoundingBox:
        scale_x = (
            image_width
            / max(
                float(page_rect.width),
                1.0,
            )
        )

        scale_y = (
            image_height
            / max(
                float(page_rect.height),
                1.0,
            )
        )

        return PixelBoundingBox(
            x0=max(
                int(
                    round(
                        (
                            pdf_box.x0
                            - float(
                                page_rect.x0
                            )
                        )
                        * scale_x
                    )
                ),
                0,
            ),
            y0=max(
                int(
                    round(
                        (
                            pdf_box.y0
                            - float(
                                page_rect.y0
                            )
                        )
                        * scale_y
                    )
                ),
                0,
            ),
            x1=min(
                int(
                    round(
                        (
                            pdf_box.x1
                            - float(
                                page_rect.x0
                            )
                        )
                        * scale_x
                    )
                ),
                image_width,
            ),
            y1=min(
                int(
                    round(
                        (
                            pdf_box.y1
                            - float(
                                page_rect.y0
                            )
                        )
                        * scale_y
                    )
                ),
                image_height,
            ),
        )

    @staticmethod
    def _pixel_box_to_pdf_box(
        *,
        pixel_box: PixelBoundingBox,
        page_rect: pymupdf.Rect,
        image_width: int,
        image_height: int,
    ) -> TableBoundingBox:
        scale_x = (
            float(page_rect.width)
            / max(
                image_width,
                1,
            )
        )

        scale_y = (
            float(page_rect.height)
            / max(
                image_height,
                1,
            )
        )

        return TableBoundingBox(
            x0=round(
                float(page_rect.x0)
                + pixel_box.x0
                * scale_x,
                3,
            ),
            y0=round(
                float(page_rect.y0)
                + pixel_box.y0
                * scale_y,
                3,
            ),
            x1=round(
                float(page_rect.x0)
                + pixel_box.x1
                * scale_x,
                3,
            ),
            y1=round(
                float(page_rect.y0)
                + pixel_box.y1
                * scale_y,
                3,
            ),
        )

    @staticmethod
    def _deduplicate_pixel_boxes(
        boxes: Sequence[PixelBoundingBox],
    ) -> list[PixelBoundingBox]:
        ordered = sorted(
            boxes,
            key=lambda box: box.area,
            reverse=True,
        )

        result: list[
            PixelBoundingBox
        ] = []

        for candidate in ordered:
            duplicate = any(
                VisualTableExtractor._pixel_iou(
                    candidate,
                    existing,
                )
                >= 0.85
                for existing in result
            )

            if duplicate:
                continue

            result.append(candidate)

        return result

    @staticmethod
    def _pixel_iou(
        first: PixelBoundingBox,
        second: PixelBoundingBox,
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

        return (
            intersection
            / union
        )

    @staticmethod
    def _calculate_content_hash(
        image: np.ndarray,
    ) -> str:
        success, encoded = cv2.imencode(
            ".png",
            image,
        )

        if not success:
            raise VisualTableExtractionError(
                "Tablo görüntüsü hash için kodlanamadı."
            )

        return hashlib.sha256(
            encoded.tobytes()
        ).hexdigest()

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


def extract_visual_tables(
    pdf_path: str | Path,
    *,
    detected_tables_by_page: Mapping[
        int,
        Sequence[DetectedTable],
    ],
    output_dir: str | Path | None = None,
    page_numbers: Iterable[int] | None = None,
) -> list[PageVisualTableExtraction]:
    """VisualTableExtractor için kısa kullanım arayüzü."""

    extractor = VisualTableExtractor()

    return extractor.extract_pdf(
        pdf_path=pdf_path,
        detected_tables_by_page=(
            detected_tables_by_page
        ),
        output_dir=output_dir,
        page_numbers=page_numbers,
    )