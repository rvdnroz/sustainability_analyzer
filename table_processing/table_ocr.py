from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
except ImportError as exc:
    raise ImportError(
        "table_ocr.py için pytesseract paketi gereklidir. "
        "'python -m pip install pytesseract' komutunu çalıştırın."
    ) from exc

from .visual_table_extractor import (
    VisualExtractedTable,
    VisualTableCell,
)
from .whole_image_ocr import (
    WholeImageOCRConfig,
    WholeImageOCRError,
    WholeImageOCRExtractor,
)


logger = logging.getLogger(__name__)


class TableOCRError(RuntimeError):
    """Tablo OCR işlemi tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class OCRWord:
    """OCR motoru tarafından tespit edilen tek bir sözcük."""

    text: str
    confidence: float

    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class OCRCellResult:
    """Tek bir tablo hücresinin OCR sonucu."""

    row_index: int
    column_index: int

    raw_text: str
    normalized_text: str

    confidence: float
    words: tuple[OCRWord, ...]

    is_empty: bool
    is_numeric: bool
    numeric_value: float | None

    language: str
    page_segmentation_mode: int

    source_image_path: Path | None
    image_hash: str | None

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "column_index": self.column_index,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "words": [
                word.to_dict()
                for word in self.words
            ],
            "is_empty": self.is_empty,
            "is_numeric": self.is_numeric,
            "numeric_value": self.numeric_value,
            "language": self.language,
            "page_segmentation_mode": (
                self.page_segmentation_mode
            ),
            "source_image_path": (
                str(self.source_image_path)
                if self.source_image_path is not None
                else None
            ),
            "image_hash": self.image_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class OCRTableRow:
    """OCR uygulanmış tablo satırı."""

    row_index: int
    cells: tuple[OCRCellResult, ...]

    is_header_candidate: bool
    is_empty: bool

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "is_header_candidate": (
                self.is_header_candidate
            ),
            "is_empty": self.is_empty,
            "text": self.text,
        }


@dataclass(slots=True, frozen=True)
class OCRExtractedTable:
    """Bir görsel tablonun OCR ile metne dönüştürülmüş sonucu."""

    table_id: str
    page_index: int
    page_number: int
    table_index: int

    extraction_method: str
    ocr_engine: str
    language: str

    row_count: int
    column_count: int

    headers: tuple[str, ...]
    rows: tuple[OCRTableRow, ...]

    matrix: tuple[tuple[str, ...], ...]

    average_confidence: float
    minimum_confidence: float
    maximum_confidence: float

    empty_cell_count: int
    numeric_cell_count: int

    requires_vlm_fallback: bool
    is_valid: bool

    content_hash: str
    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "extraction_method": self.extraction_method,
            "ocr_engine": self.ocr_engine,
            "language": self.language,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "headers": list(self.headers),
            "rows": [
                row.to_dict()
                for row in self.rows
            ],
            "matrix": [
                list(row)
                for row in self.matrix
            ],
            "average_confidence": (
                self.average_confidence
            ),
            "minimum_confidence": (
                self.minimum_confidence
            ),
            "maximum_confidence": (
                self.maximum_confidence
            ),
            "empty_cell_count": (
                self.empty_cell_count
            ),
            "numeric_cell_count": (
                self.numeric_cell_count
            ),
            "requires_vlm_fallback": (
                self.requires_vlm_fallback
            ),
            "is_valid": self.is_valid,
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PageTableOCRResult:
    """Tek bir sayfadaki tablo OCR sonuçları."""

    page_index: int
    page_number: int

    table_count: int
    successful_table_count: int
    failed_table_count: int

    tables: tuple[OCRExtractedTable, ...]
    warnings: tuple[str, ...]

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
            "tables": [
                table.to_dict()
                for table in self.tables
            ],
            "warnings": list(self.warnings),
        }


class TableOCR:
    """
    Görsel tablo hücrelerini Tesseract OCR ile metne dönüştürür.

    Temel akış:

    1. VisualTableExtractor hücre görüntülerini yükler.
    2. Görüntü gri tona dönüştürülür.
    3. Ölçekleme, gürültü azaltma ve threshold işlemleri uygulanır.
    4. Varsayılan olarak tablo görseline tek Tesseract OCR çağrısı yapılır.
    5. Sözcük kutuları hücre koordinatlarına dağıtılır ve güven skorları hesaplanır.
    6. Metin matrisi, satırlar ve başlık adayları oluşturulur.
    7. Hücre bulunmazsa kırpılmış görselin tamamına whole-image OCR uygulanır.
    8. OCR kalitesi yetersizse VLM fallback gereksinimi işaretlenir.

    Bu modül Tesseract uygulamasının işletim sisteminde kurulu olmasını
    gerektirir. pytesseract yalnızca Python istemcisidir.
    """

    _WHITESPACE_RE = re.compile(r"\s+")

    _NUMERIC_RE = re.compile(
        r"""
        ^\s*
        [+-]?
        (?:
            \d{1,3}(?:[.\s,]\d{3})+
            |
            \d+
        )
        (?:[.,]\d+)?
        \s*
        (?:%|₺|\$|€|kg|t|ton|mwh|kwh|gj|co2e?)?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        *,
        language: str = "tur+eng",
        tesseract_command: str | Path | None = None,
        default_psm: int = 7,
        header_psm: int = 6,
        numeric_psm: int = 7,
        oem: int = 3,
        minimum_word_confidence: float = 25.0,
        minimum_cell_confidence: float = 40.0,
        minimum_table_confidence: float = 45.0,
        vlm_fallback_confidence: float = 40.0,
        maximum_empty_cell_ratio: float = 0.60,
        upscale_factor: float = 1.5,
        apply_threshold: bool = True,
        apply_denoising: bool = False,
        apply_sharpening: bool = True,
        remove_cell_borders: bool = True,
        border_crop_pixels: int = 3,
        enable_whole_image_ocr: bool = True,
        whole_image_ocr_extractor: WholeImageOCRExtractor | None = None,
        whole_image_psm_modes: tuple[int, ...] = (6, 11),
        whole_image_minimum_confidence: float = 0.45,
        whole_image_preprocessing_variants: tuple[str, ...] = (
            "grayscale",
            "otsu",
        ),
        whole_image_maximum_dimension: int = 3000,
        whole_image_apply_deskew: bool = False,
        whole_image_enable_early_stop: bool = True,
        prefer_single_pass_cell_ocr: bool = True,
        single_pass_psm: int = 6,
    ) -> None:
        if not language.strip():
            raise ValueError(
                "language boş olamaz."
            )

        for name, value in {
            "minimum_word_confidence": (
                minimum_word_confidence
            ),
            "minimum_cell_confidence": (
                minimum_cell_confidence
            ),
            "minimum_table_confidence": (
                minimum_table_confidence
            ),
            "vlm_fallback_confidence": (
                vlm_fallback_confidence
            ),
        }.items():
            if not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"{name} 0 ile 100 arasında olmalıdır."
                )

        if not 0.0 <= maximum_empty_cell_ratio <= 1.0:
            raise ValueError(
                "maximum_empty_cell_ratio 0 ile 1 arasında olmalıdır."
            )

        if upscale_factor < 1.0:
            raise ValueError(
                "upscale_factor en az 1.0 olmalıdır."
            )

        if border_crop_pixels < 0:
            raise ValueError(
                "border_crop_pixels negatif olamaz."
            )

        if not 0.0 <= whole_image_minimum_confidence <= 1.0:
            raise ValueError(
                "whole_image_minimum_confidence 0 ile 1 arasında olmalıdır."
            )

        if enable_whole_image_ocr and not whole_image_psm_modes:
            raise ValueError(
                "Whole-image OCR açıkken en az bir PSM modu tanımlanmalıdır."
            )

        if whole_image_maximum_dimension < 512:
            raise ValueError(
                "whole_image_maximum_dimension en az 512 olmalıdır."
            )

        for name, value in {
            "default_psm": default_psm,
            "header_psm": header_psm,
            "numeric_psm": numeric_psm,
        }.items():
            if not 0 <= value <= 13:
                raise ValueError(
                    f"{name} 0 ile 13 arasında olmalıdır."
                )

        if not 0 <= oem <= 3:
            raise ValueError(
                "oem 0 ile 3 arasında olmalıdır."
            )

        if not 0 <= single_pass_psm <= 13:
            raise ValueError(
                "single_pass_psm 0 ile 13 arasında olmalıdır."
            )

        self.language = language

        self.default_psm = default_psm
        self.header_psm = header_psm
        self.numeric_psm = numeric_psm
        self.oem = oem

        self.minimum_word_confidence = (
            minimum_word_confidence
        )

        self.minimum_cell_confidence = (
            minimum_cell_confidence
        )

        self.minimum_table_confidence = (
            minimum_table_confidence
        )

        self.vlm_fallback_confidence = (
            vlm_fallback_confidence
        )

        self.maximum_empty_cell_ratio = (
            maximum_empty_cell_ratio
        )

        self.upscale_factor = upscale_factor

        self.apply_threshold = apply_threshold
        self.apply_denoising = apply_denoising
        self.apply_sharpening = apply_sharpening
        self.remove_cell_borders = remove_cell_borders

        self.border_crop_pixels = border_crop_pixels
        self.prefer_single_pass_cell_ocr = bool(
            prefer_single_pass_cell_ocr
        )
        self.single_pass_psm = int(single_pass_psm)

        self.enable_whole_image_ocr = bool(
            enable_whole_image_ocr
        )
        self.whole_image_minimum_confidence = (
            whole_image_minimum_confidence
        )

        self.whole_image_ocr_extractor: (
            WholeImageOCRExtractor | None
        ) = whole_image_ocr_extractor

        if (
            self.enable_whole_image_ocr
            and self.whole_image_ocr_extractor is None
        ):
            self.whole_image_ocr_extractor = WholeImageOCRExtractor(
                config=WholeImageOCRConfig(
                    language=language,
                    psm_modes=tuple(whole_image_psm_modes),
                    oem=oem,
                    minimum_word_confidence=(
                        minimum_word_confidence
                    ),
                    minimum_result_confidence=(
                        whole_image_minimum_confidence
                    ),
                    upscale_factor=upscale_factor,
                    maximum_image_dimension=(
                        whole_image_maximum_dimension
                    ),
                    apply_deskew=whole_image_apply_deskew,
                    apply_denoising=apply_denoising,
                    apply_sharpening=apply_sharpening,
                    preprocessing_variants=tuple(
                        whole_image_preprocessing_variants
                    ),
                    enable_early_stop=(
                        whole_image_enable_early_stop
                    ),
                    tesseract_command=tesseract_command,
                )
            )

        if tesseract_command is not None:
            pytesseract.pytesseract.tesseract_cmd = str(
                Path(
                    tesseract_command
                ).expanduser()
            )

        self._validate_tesseract()

    def process_pages(
        self,
        page_tables: Sequence[Any],
    ) -> list[PageTableOCRResult]:
        """
        PageVisualTableExtraction nesnelerini OCR ile işler.
        """

        results: list[PageTableOCRResult] = []

        for page_result in page_tables:
            tables = tuple(
                getattr(
                    page_result,
                    "tables",
                    (),
                )
            )

            results.append(
                self.process_page(
                    page_index=int(
                        getattr(
                            page_result,
                            "page_index",
                            0,
                        )
                    ),
                    page_number=int(
                        getattr(
                            page_result,
                            "page_number",
                            1,
                        )
                    ),
                    tables=tables,
                )
            )

        return results

    def process_page(
        self,
        *,
        page_index: int,
        page_number: int,
        tables: Sequence[VisualExtractedTable],
    ) -> PageTableOCRResult:
        """Tek sayfadaki bütün görsel tabloları OCR ile işler."""

        extracted_tables: list[
            OCRExtractedTable
        ] = []

        warnings: list[str] = []
        failed_count = 0

        for table in tables:
            try:
                result = self.process_table(
                    table
                )

            except Exception as exc:
                failed_count += 1

                message = (
                    f"Sayfa {page_number}, "
                    f"tablo {table.table_index + 1} "
                    f"OCR işlemi başarısız: {exc}"
                )

                warnings.append(message)
                logger.warning(message)

                continue

            extracted_tables.append(
                result
            )

        return PageTableOCRResult(
            page_index=page_index,
            page_number=page_number,
            table_count=len(tables),
            successful_table_count=len(
                extracted_tables
            ),
            failed_table_count=failed_count,
            tables=tuple(
                extracted_tables
            ),
            warnings=tuple(warnings),
        )

    def process_table(
        self,
        table: VisualExtractedTable,
    ) -> OCRExtractedTable:
        """
        Görsel tabloyu uygun OCR stratejisiyle işler.

        Öncelik sırası:
        1. Grid yoksa whole-image OCR.
        2. Grid ve tablo görseli varsa tablo başına tek Tesseract çağrısı;
           kelimeler hücre koordinatlarına dağıtılır.
        3. Tek-geçişli yöntem kullanılamazsa eski hücre-bazlı OCR fallback'i.
        """

        requires_whole_image = bool(
            getattr(
                table,
                "requires_whole_image_ocr",
                False,
            )
            or not table.cells
        )

        if requires_whole_image:
            return self.process_whole_image_table(table)

        if self.prefer_single_pass_cell_ocr:
            try:
                return self.process_structured_table_single_pass(
                    table
                )
            except Exception as exc:
                logger.warning(
                    "Tablo %s tek-geçişli OCR kullanılamadı; "
                    "hücre-bazlı fallback uygulanıyor: %s",
                    table.table_id,
                    exc,
                )

        return self.process_structured_table_per_cell(
            table
        )

    def process_structured_table_single_pass(
        self,
        table: VisualExtractedTable,
    ) -> OCRExtractedTable:
        """
        Tablo görseline yalnızca bir Tesseract çağrısı yapar.

        OCR sözcük kutuları, VisualTableExtractor tarafından oluşturulan
        yerel hücre koordinatlarıyla eşleştirilir. Böylece N hücre için N
        ayrı tesseract.exe süreci başlatılmaz.
        """

        image_path_value = getattr(
            table,
            "table_image_path",
            None,
        )

        if image_path_value is None:
            raise TableOCRError(
                "Tek-geçişli OCR için tablo görüntüsü bulunmuyor."
            )

        image_path = Path(
            image_path_value
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Tablo görüntüsü bulunamadı: {image_path}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise TableOCRError(
                f"Tablo görüntüsü okunamadı: {image_path}"
            )

        image_hash = self._calculate_image_hash(
            image
        )

        raw_text, words, _ = self._run_ocr(
            image,
            psm=self.single_pass_psm,
        )

        cell_results = self._assign_words_to_visual_cells(
            table=table,
            words=words,
            source_image_path=image_path,
            image_hash=image_hash,
        )

        if not cell_results:
            raise TableOCRError(
                "Tek-geçişli OCR hücre sonucu üretemedi."
            )

        return self._build_structured_ocr_table(
            table=table,
            cell_results=cell_results,
            extraction_method=(
                "opencv_cells+tesseract_single_pass"
            ),
            metadata={
                "ocr_strategy": "single_pass_word_assignment",
                "single_pass_psm": self.single_pass_psm,
                "single_pass_word_count": len(words),
                "single_pass_raw_text": self._normalize_text(
                    raw_text
                ),
                "tesseract_call_count": 1,
                "source_table_image": str(image_path),
            },
        )

    def process_structured_table_per_cell(
        self,
        table: VisualExtractedTable,
    ) -> OCRExtractedTable:
        """Eski hücre-bazlı OCR yöntemini güvenli fallback olarak çalıştırır."""

        ordered_cells = sorted(
            table.cells,
            key=lambda cell: (
                cell.row_index,
                cell.column_index,
            ),
        )

        cell_results: list[OCRCellResult] = []

        for cell in ordered_cells:
            try:
                cell_result = self.process_cell(
                    cell
                )
            except Exception as exc:
                logger.warning(
                    "Tablo %s hücre (%d, %d) OCR başarısız: %s",
                    table.table_id,
                    cell.row_index,
                    cell.column_index,
                    exc,
                )

                cell_result = OCRCellResult(
                    row_index=cell.row_index,
                    column_index=cell.column_index,
                    raw_text="",
                    normalized_text="",
                    confidence=0.0,
                    words=(),
                    is_empty=True,
                    is_numeric=False,
                    numeric_value=None,
                    language=self.language,
                    page_segmentation_mode=(
                        self._select_psm(cell)
                    ),
                    source_image_path=cell.image_path,
                    image_hash=None,
                    warnings=(
                        f"OCR işlemi başarısız: {exc}",
                    ),
                    metadata={
                        "visual_cell_confidence": (
                            cell.confidence
                        ),
                    },
                )

            cell_results.append(cell_result)

        return self._build_structured_ocr_table(
            table=table,
            cell_results=cell_results,
            extraction_method=(
                "opencv_cells+tesseract_per_cell"
            ),
            metadata={
                "ocr_strategy": "per_cell_fallback",
                "tesseract_call_count": len(ordered_cells),
                "source_table_image": (
                    str(table.table_image_path)
                    if table.table_image_path is not None
                    else None
                ),
            },
        )

    def _assign_words_to_visual_cells(
        self,
        *,
        table: VisualExtractedTable,
        words: Sequence[OCRWord],
        source_image_path: Path,
        image_hash: str,
    ) -> list[OCRCellResult]:
        buckets: dict[
            tuple[int, int],
            list[OCRWord],
        ] = {
            (
                int(cell.row_index),
                int(cell.column_index),
            ): []
            for cell in table.cells
        }

        cell_boxes: dict[
            tuple[int, int],
            tuple[float, float, float, float],
        ] = {}

        for cell in table.cells:
            metadata = dict(
                getattr(cell, "metadata", {}) or {}
            )
            local_box = metadata.get(
                "local_pixel_box"
            )

            if not isinstance(local_box, Mapping):
                continue

            try:
                cell_boxes[
                    (
                        int(cell.row_index),
                        int(cell.column_index),
                    )
                ] = (
                    float(local_box["x0"]),
                    float(local_box["y0"]),
                    float(local_box["x1"]),
                    float(local_box["y1"]),
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        if not cell_boxes:
            raise TableOCRError(
                "Hücrelerin yerel piksel koordinatları bulunamadı."
            )

        for word in words:
            center_x = (
                float(word.left)
                + float(word.width) / 2.0
            )
            center_y = (
                float(word.top)
                + float(word.height) / 2.0
            )

            matching_key: tuple[int, int] | None = None
            best_overlap = 0.0

            for key, box in cell_boxes.items():
                x0, y0, x1, y1 = box

                if (
                    x0 <= center_x <= x1
                    and y0 <= center_y <= y1
                ):
                    matching_key = key
                    break

                overlap = self._word_cell_overlap_ratio(
                    word=word,
                    cell_box=box,
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    matching_key = key

            if (
                matching_key is not None
                and (
                    best_overlap >= 0.20
                    or self._word_center_in_box(
                        word,
                        cell_boxes[matching_key],
                    )
                )
            ):
                buckets[matching_key].append(word)

        results: list[OCRCellResult] = []

        for cell in sorted(
            table.cells,
            key=lambda item: (
                item.row_index,
                item.column_index,
            ),
        ):
            key = (
                int(cell.row_index),
                int(cell.column_index),
            )
            assigned = sorted(
                buckets.get(key, ()),
                key=lambda item: (
                    item.top,
                    item.left,
                ),
            )

            accepted = [
                word
                for word in assigned
                if word.confidence
                >= self.minimum_word_confidence
            ]
            used_words = accepted or assigned

            raw_text = " ".join(
                word.text
                for word in used_words
            )
            normalized_text = self._normalize_text(
                raw_text
            )
            confidence = (
                mean(
                    word.confidence
                    for word in used_words
                )
                if used_words
                else 0.0
            )
            numeric_value = self._parse_numeric_value(
                normalized_text
            )
            is_numeric = bool(
                numeric_value is not None
                or self._NUMERIC_RE.fullmatch(
                    normalized_text
                )
            )

            cell_warnings: list[str] = []

            if (
                normalized_text
                and confidence
                < self.minimum_cell_confidence
            ):
                cell_warnings.append(
                    "Hücre OCR güven skoru düşük: "
                    f"{confidence:.2f}."
                )

            results.append(
                OCRCellResult(
                    row_index=int(cell.row_index),
                    column_index=int(
                        cell.column_index
                    ),
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    confidence=round(
                        confidence,
                        4,
                    ),
                    words=tuple(assigned),
                    is_empty=not bool(
                        normalized_text
                    ),
                    is_numeric=is_numeric,
                    numeric_value=numeric_value,
                    language=self.language,
                    page_segmentation_mode=(
                        self.single_pass_psm
                    ),
                    source_image_path=(
                        source_image_path
                    ),
                    image_hash=image_hash,
                    warnings=tuple(
                        cell_warnings
                    ),
                    metadata={
                        "visual_cell_confidence": (
                            cell.confidence
                        ),
                        "is_header_candidate": (
                            cell.is_header_candidate
                        ),
                        "assignment_method": (
                            "word_center_or_overlap"
                        ),
                        "assigned_word_count": len(
                            assigned
                        ),
                    },
                )
            )

        return results

    def _build_structured_ocr_table(
        self,
        *,
        table: VisualExtractedTable,
        cell_results: Sequence[OCRCellResult],
        extraction_method: str,
        metadata: Mapping[str, Any],
    ) -> OCRExtractedTable:
        warnings: list[str] = []

        matrix = self._build_matrix(
            cells=cell_results,
            row_count=table.row_count,
            column_count=table.column_count,
        )
        rows = self._build_rows(
            cell_results=cell_results,
            matrix=matrix,
        )
        headers = self._build_headers(
            matrix
        )

        confidences = [
            cell.confidence
            for cell in cell_results
        ]
        average_confidence = (
            mean(confidences)
            if confidences
            else 0.0
        )
        minimum_confidence = (
            min(confidences)
            if confidences
            else 0.0
        )
        maximum_confidence = (
            max(confidences)
            if confidences
            else 0.0
        )
        empty_cell_count = sum(
            cell.is_empty
            for cell in cell_results
        )
        numeric_cell_count = sum(
            cell.is_numeric
            for cell in cell_results
        )
        empty_cell_ratio = (
            empty_cell_count
            / max(
                len(cell_results),
                1,
            )
        )

        requires_vlm_fallback = (
            average_confidence
            < self.vlm_fallback_confidence
            or empty_cell_ratio
            > self.maximum_empty_cell_ratio
            or not table.is_valid
        )
        is_valid = (
            table.row_count >= 2
            and table.column_count >= 2
            and average_confidence
            >= self.minimum_table_confidence
            and empty_cell_ratio
            <= self.maximum_empty_cell_ratio
        )

        if (
            average_confidence
            < self.minimum_table_confidence
        ):
            warnings.append(
                "Tablo ortalama OCR güven skoru "
                f"düşük: {average_confidence:.2f}."
            )

        if (
            empty_cell_ratio
            > self.maximum_empty_cell_ratio
        ):
            warnings.append(
                "Boş hücre oranı eşik üzerinde: "
                f"{empty_cell_ratio:.2%}."
            )

        if requires_vlm_fallback:
            warnings.append(
                "Tablo VLM fallback için işaretlendi."
            )

        content_hash = self._calculate_content_hash(
            matrix
        )

        return OCRExtractedTable(
            table_id=table.table_id,
            page_index=table.page_index,
            page_number=table.page_number,
            table_index=table.table_index,
            extraction_method=extraction_method,
            ocr_engine="tesseract",
            language=self.language,
            row_count=table.row_count,
            column_count=table.column_count,
            headers=tuple(headers),
            rows=tuple(rows),
            matrix=tuple(
                tuple(row)
                for row in matrix
            ),
            average_confidence=round(
                average_confidence,
                4,
            ),
            minimum_confidence=round(
                minimum_confidence,
                4,
            ),
            maximum_confidence=round(
                maximum_confidence,
                4,
            ),
            empty_cell_count=empty_cell_count,
            numeric_cell_count=numeric_cell_count,
            requires_vlm_fallback=(
                requires_vlm_fallback
            ),
            is_valid=is_valid,
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "visual_extraction_confidence": (
                    table.extraction_confidence
                ),
                "empty_cell_ratio": round(
                    empty_cell_ratio,
                    4,
                ),
                "cell_count": len(
                    cell_results
                ),
                **dict(metadata),
            },
        )

    @staticmethod
    def _word_center_in_box(
        word: OCRWord,
        box: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> bool:
        center_x = word.left + word.width / 2.0
        center_y = word.top + word.height / 2.0
        x0, y0, x1, y1 = box
        return (
            x0 <= center_x <= x1
            and y0 <= center_y <= y1
        )

    @staticmethod
    def _word_cell_overlap_ratio(
        *,
        word: OCRWord,
        cell_box: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> float:
        word_x0 = float(word.left)
        word_y0 = float(word.top)
        word_x1 = word_x0 + float(word.width)
        word_y1 = word_y0 + float(word.height)

        cell_x0, cell_y0, cell_x1, cell_y1 = (
            cell_box
        )

        intersection_width = max(
            0.0,
            min(word_x1, cell_x1)
            - max(word_x0, cell_x0),
        )
        intersection_height = max(
            0.0,
            min(word_y1, cell_y1)
            - max(word_y0, cell_y0),
        )
        intersection = (
            intersection_width
            * intersection_height
        )
        word_area = max(
            float(word.width)
            * float(word.height),
            1.0,
        )
        return min(
            intersection / word_area,
            1.0,
        )

    def process_whole_image_table(
        self,
        table: VisualExtractedTable,
    ) -> OCRExtractedTable:
        """
        Hücre yapısı bulunamayan kırpılmış tablo/grafik görselinin tamamına
        OCR uygular ve sonucu mevcut `OCRExtractedTable` sözleşmesine çevirir.

        Bu dönüşüm tam tablo grid'i üretilemese bile `matrix` alanını korur.
        Böylece TableNormalizer, BM25 ve vector indexing ham OCR metnini
        kullanabilir.
        """

        if not self.enable_whole_image_ocr:
            raise TableOCRError(
                f"Tablo hücre içermiyor ve whole-image OCR kapalı: "
                f"{table.table_id}"
            )

        if self.whole_image_ocr_extractor is None:
            raise TableOCRError(
                "Whole-image OCR extractor yapılandırılmadı."
            )

        image_path_value = getattr(
            table,
            "table_image_path",
            None,
        )

        if image_path_value is None:
            raise TableOCRError(
                f"Whole-image OCR için tablo görüntüsü bulunmuyor: "
                f"{table.table_id}"
            )

        image_path = Path(
            image_path_value
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Whole-image OCR tablo görseli bulunamadı: {image_path}"
            )

        visual_id = uuid5(
            NAMESPACE_URL,
            f"visual-table:{table.table_id}",
        )
        report_id = self._resolve_report_uuid(table)

        try:
            whole_result = (
                self.whole_image_ocr_extractor.extract_from_path(
                    image_path=image_path,
                    visual_id=visual_id,
                    report_id=report_id,
                    page_number=table.page_number,
                )
            )
        except WholeImageOCRError as exc:
            raise TableOCRError(
                f"Whole-image OCR başarısız: {exc}"
            ) from exc

        matrix = self._whole_image_matrix(whole_result)

        if not matrix:
            raise TableOCRError(
                f"Whole-image OCR metin üretmedi: {table.table_id}"
            )

        row_count = len(matrix)
        column_count = max(
            (len(row) for row in matrix),
            default=0,
        )

        matrix = [
            [
                *row,
                *(
                    ""
                    for _ in range(
                        column_count - len(row)
                    )
                ),
            ]
            for row in matrix
        ]

        headers = self._build_headers(matrix)
        rows = self._build_rows(
            cell_results=(),
            matrix=matrix,
        )

        average_confidence = float(
            whole_result.average_confidence or 0.0
        ) * 100.0

        numeric_cell_count = sum(
            bool(
                self._NUMERIC_RE.fullmatch(
                    self._normalize_text(cell)
                )
                or re.search(r"\d", str(cell))
            )
            for row in matrix
            for cell in row
            if self._normalize_text(cell)
        )

        empty_cell_count = sum(
            not self._normalize_text(cell)
            for row in matrix
            for cell in row
        )

        total_cell_count = max(
            row_count * max(column_count, 1),
            1,
        )
        empty_cell_ratio = (
            empty_cell_count / total_cell_count
        )

        has_text = bool(
            self._normalize_text(whole_result.raw_text)
        )
        is_valid = bool(
            has_text
            and average_confidence
            >= self.minimum_table_confidence
        )

        requires_vlm_fallback = bool(
            not has_text
            or average_confidence
            < self.vlm_fallback_confidence
            or not whole_result.is_structured
            or numeric_cell_count == 0
        )

        warnings = list(
            getattr(table, "warnings", ()) or ()
        )

        if average_confidence < self.minimum_table_confidence:
            warnings.append(
                "Whole-image OCR ortalama güven skoru düşük: "
                f"{average_confidence:.2f}."
            )

        if not whole_result.is_structured:
            warnings.append(
                "Whole-image OCR tam bir satır-sütun yapısı kuramadı; "
                "ham metin retrieval için korunacaktır."
            )

        if numeric_cell_count == 0:
            warnings.append(
                "Whole-image OCR sonucunda açık sayısal hücre bulunamadı."
            )

        if requires_vlm_fallback:
            warnings.append(
                "Görsel VLM/vision fallback için işaretlendi."
            )

        content_hash = self._calculate_content_hash(
            matrix
        )

        raw_text = self._normalize_multiline_text(
            whole_result.raw_text
        )

        return OCRExtractedTable(
            table_id=table.table_id,
            page_index=table.page_index,
            page_number=table.page_number,
            table_index=table.table_index,
            extraction_method=(
                "whole_image+tesseract_ocr"
            ),
            ocr_engine="tesseract",
            language=(
                whole_result.language
                or self.language
            ),
            row_count=row_count,
            column_count=column_count,
            headers=tuple(headers),
            rows=tuple(rows),
            matrix=tuple(
                tuple(row)
                for row in matrix
            ),
            average_confidence=round(
                average_confidence,
                4,
            ),
            minimum_confidence=round(
                average_confidence,
                4,
            ),
            maximum_confidence=round(
                average_confidence,
                4,
            ),
            empty_cell_count=empty_cell_count,
            numeric_cell_count=numeric_cell_count,
            requires_vlm_fallback=(
                requires_vlm_fallback
            ),
            is_valid=is_valid,
            content_hash=content_hash,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                **dict(getattr(table, "metadata", {}) or {}),
                "visual_extraction_confidence": (
                    table.extraction_confidence
                ),
                "source_table_image": str(image_path),
                "ocr_strategy": "whole_image",
                "whole_image_ocr": True,
                "whole_image_visual_id": str(visual_id),
                "whole_image_report_id": str(report_id),
                "whole_image_raw_text": raw_text,
                "whole_image_psm_mode": (
                    whole_result.psm_mode
                ),
                "whole_image_preprocessing_variant": (
                    whole_result.preprocessing_variant
                ),
                "whole_image_is_structured": (
                    whole_result.is_structured
                ),
                "whole_image_word_count": len(
                    whole_result.words
                ),
                "whole_image_line_count": len(
                    whole_result.lines
                ),
                "empty_cell_ratio": round(
                    empty_cell_ratio,
                    4,
                ),
                "whole_image_ocr_metadata": dict(
                    whole_result.metadata
                ),
            },
        )

    def _whole_image_matrix(
        self,
        result: Any,
    ) -> list[list[str]]:
        """WholeImageOCRResult çıktısını normalize edilebilir matrise çevirir."""

        reconstructed_rows = list(
            getattr(
                result,
                "reconstructed_rows",
                (),
            )
            or ()
        )

        matrix: list[list[str]] = []

        for row in reconstructed_rows:
            normalized_row = [
                self._normalize_text(cell)
                for cell in row
            ]

            if any(normalized_row):
                matrix.append(normalized_row)

        if matrix:
            return matrix

        lines = list(
            getattr(result, "lines", ()) or ()
        )

        for line in lines:
            text = self._normalize_text(
                getattr(line, "text", "")
            )
            if text:
                matrix.append([text])

        if matrix:
            return matrix

        raw_text = self._normalize_multiline_text(
            getattr(result, "raw_text", "")
        )

        return [
            [line]
            for line in raw_text.splitlines()
            if self._normalize_text(line)
        ]

    @staticmethod
    def _resolve_report_uuid(
        table: VisualExtractedTable,
    ) -> UUID:
        """
        WholeImageOCRResult sözleşmesinin istediği UUID'yi oluşturur.

        Ingestion pipeline bu aşamada gerçek rapor UUID'sini aktarmıyorsa
        metadata veya table_id üzerinden kararlı bir UUID üretilir. Bu UUID
        yalnızca ara OCR kaydı içindir; kalıcı repository rapor kimliği daha
        sonra ingestion katmanında atanacaktır.
        """

        metadata = dict(
            getattr(table, "metadata", {}) or {}
        )

        for key in (
            "report_uuid",
            "report_id",
        ):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return UUID(str(value))
            except (TypeError, ValueError, AttributeError):
                continue

        return uuid5(
            NAMESPACE_URL,
            f"ocr-report:{table.table_id}",
        )

    @classmethod
    def _normalize_multiline_text(
        cls,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        return "\n".join(
            cls._normalize_text(line)
            for line in str(value)
            .replace("\r", "\n")
            .split("\n")
            if cls._normalize_text(line)
        )

    def process_cell(
        self,
        cell: VisualTableCell,
    ) -> OCRCellResult:
        """Tek bir tablo hücresine OCR uygular."""

        if cell.image_path is None:
            raise TableOCRError(
                "Hücre görüntü yolu bulunmuyor."
            )

        image_path = Path(
            cell.image_path
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Hücre görüntüsü bulunamadı: {image_path}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise TableOCRError(
                f"Hücre görüntüsü okunamadı: {image_path}"
            )

        image_hash = self._calculate_image_hash(
            image
        )

        processed_image = self._preprocess_image(
            image
        )

        psm = self._select_psm(
            cell
        )

        raw_text, words, confidence = (
            self._run_ocr(
                processed_image,
                psm=psm,
            )
        )

        normalized_text = (
            self._normalize_text(
                raw_text
            )
        )

        is_empty = not bool(
            normalized_text
        )

        numeric_value = self._parse_numeric_value(
            normalized_text
        )

        is_numeric = (
            numeric_value is not None
            or bool(
                self._NUMERIC_RE.fullmatch(
                    normalized_text
                )
            )
        )

        warnings: list[str] = []

        if (
            confidence
            < self.minimum_cell_confidence
            and not is_empty
        ):
            warnings.append(
                "Hücre OCR güven skoru düşük: "
                f"{confidence:.2f}."
            )

        if is_empty and self._has_visual_content(
            processed_image
        ):
            warnings.append(
                "Görsel içerik bulunmasına rağmen OCR metni üretilemedi."
            )

        return OCRCellResult(
            row_index=cell.row_index,
            column_index=cell.column_index,
            raw_text=raw_text,
            normalized_text=normalized_text,
            confidence=round(
                confidence,
                4,
            ),
            words=tuple(words),
            is_empty=is_empty,
            is_numeric=is_numeric,
            numeric_value=numeric_value,
            language=self.language,
            page_segmentation_mode=psm,
            source_image_path=image_path,
            image_hash=image_hash,
            warnings=tuple(warnings),
            metadata={
                "visual_cell_confidence": (
                    cell.confidence
                ),
                "is_header_candidate": (
                    cell.is_header_candidate
                ),
                "width_pixels": (
                    cell.width_pixels
                ),
                "height_pixels": (
                    cell.height_pixels
                ),
            },
        )

    def _run_ocr(
        self,
        image: np.ndarray,
        *,
        psm: int,
    ) -> tuple[
        str,
        list[OCRWord],
        float,
    ]:
        config = (
            f"--oem {self.oem} "
            f"--psm {psm} "
            "-c preserve_interword_spaces=1"
        )

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config,
                output_type=Output.DICT,
            )

        except pytesseract.TesseractNotFoundError as exc:
            raise TableOCRError(
                "Tesseract çalıştırılabilir dosyası bulunamadı."
            ) from exc

        except pytesseract.TesseractError as exc:
            raise TableOCRError(
                f"Tesseract OCR hatası: {exc}"
            ) from exc

        words: list[OCRWord] = []
        accepted_texts: list[str] = []
        accepted_confidences: list[float] = []

        text_values = data.get(
            "text",
            []
        )

        confidence_values = data.get(
            "conf",
            []
        )

        for index, text_value in enumerate(
            text_values
        ):
            text = self._normalize_text(
                str(text_value)
            )

            if not text:
                continue

            confidence = self._safe_float(
                confidence_values[index]
                if index
                < len(confidence_values)
                else -1
            )

            if confidence < 0:
                continue

            word = OCRWord(
                text=text,
                confidence=round(
                    confidence,
                    4,
                ),
                left=self._safe_int(
                    self._get_data_item(
                        data,
                        "left",
                        index,
                    )
                ),
                top=self._safe_int(
                    self._get_data_item(
                        data,
                        "top",
                        index,
                    )
                ),
                width=self._safe_int(
                    self._get_data_item(
                        data,
                        "width",
                        index,
                    )
                ),
                height=self._safe_int(
                    self._get_data_item(
                        data,
                        "height",
                        index,
                    )
                ),
            )

            words.append(word)

            if confidence >= self.minimum_word_confidence:
                accepted_texts.append(
                    text
                )

                accepted_confidences.append(
                    confidence
                )

        if accepted_texts:
            raw_text = " ".join(
                accepted_texts
            )

            cell_confidence = mean(
                accepted_confidences
            )

        elif words:
            raw_text = " ".join(
                word.text
                for word in words
            )

            cell_confidence = mean(
                word.confidence
                for word in words
            )

        else:
            raw_text = ""
            cell_confidence = 0.0

        return (
            raw_text,
            words,
            cell_confidence,
        )

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if image.size == 0:
            raise TableOCRError(
                "Boş hücre görüntüsü işlenemez."
            )

        processed = image.copy()

        if (
            self.remove_cell_borders
            and self.border_crop_pixels > 0
        ):
            processed = self._crop_borders(
                processed,
                self.border_crop_pixels,
            )

        gray = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2GRAY,
        )

        if self.upscale_factor > 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=self.upscale_factor,
                fy=self.upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        if self.apply_denoising:
            gray = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        if self.apply_sharpening:
            sharpening_kernel = np.array(
                [
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0],
                ],
                dtype=np.float32,
            )

            gray = cv2.filter2D(
                gray,
                -1,
                sharpening_kernel,
            )

        if self.apply_threshold:
            processed = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15,
            )
        else:
            processed = gray

        processed = cv2.copyMakeBorder(
            processed,
            12,
            12,
            12,
            12,
            cv2.BORDER_CONSTANT,
            value=255,
        )

        return processed

    def _build_matrix(
        self,
        *,
        cells: Sequence[OCRCellResult],
        row_count: int,
        column_count: int,
    ) -> list[list[str]]:
        matrix = [
            [
                ""
                for _ in range(
                    column_count
                )
            ]
            for _ in range(
                row_count
            )
        ]

        for cell in cells:
            if not (
                0
                <= cell.row_index
                < row_count
            ):
                continue

            if not (
                0
                <= cell.column_index
                < column_count
            ):
                continue

            matrix[
                cell.row_index
            ][
                cell.column_index
            ] = cell.normalized_text

        return matrix

    def _build_rows(
        self,
        *,
        cell_results: Sequence[OCRCellResult],
        matrix: Sequence[Sequence[str]],
    ) -> list[OCRTableRow]:
        cells_by_row: dict[
            int,
            list[OCRCellResult],
        ] = {}

        for cell in cell_results:
            cells_by_row.setdefault(
                cell.row_index,
                [],
            ).append(
                cell
            )

        rows: list[
            OCRTableRow
        ] = []

        for row_index, row_values in enumerate(
            matrix
        ):
            row_cells = sorted(
                cells_by_row.get(
                    row_index,
                    [],
                ),
                key=lambda cell: (
                    cell.column_index
                ),
            )

            rows.append(
                OCRTableRow(
                    row_index=row_index,
                    cells=tuple(row_cells),
                    is_header_candidate=(
                        row_index == 0
                    ),
                    is_empty=not any(
                        value.strip()
                        for value in row_values
                    ),
                    text=" | ".join(
                        row_values
                    ),
                )
            )

        return rows

    def _build_headers(
        self,
        matrix: Sequence[
            Sequence[str]
        ],
    ) -> list[str]:
        if not matrix:
            return []

        first_row = list(
            matrix[0]
        )

        headers: list[str] = []
        counts: dict[str, int] = {}

        for column_index, value in enumerate(
            first_row
        ):
            header = (
                value.strip()
                or f"column_{column_index + 1}"
            )

            normalized = header.casefold()

            counts[normalized] = (
                counts.get(
                    normalized,
                    0,
                )
                + 1
            )

            count = counts[normalized]

            if count > 1:
                header = (
                    f"{header}_{count}"
                )

            headers.append(
                header
            )

        return headers

    def _select_psm(
        self,
        cell: VisualTableCell,
    ) -> int:
        if cell.is_header_candidate:
            return self.header_psm

        aspect_ratio = (
            cell.width_pixels
            / max(
                cell.height_pixels,
                1,
            )
        )

        if aspect_ratio >= 4.0:
            return 7

        if cell.height_pixels >= 70:
            return 6

        return self.default_psm

    def _validate_tesseract(
        self,
    ) -> None:
        try:
            version = pytesseract.get_tesseract_version()

        except pytesseract.TesseractNotFoundError as exc:
            raise TableOCRError(
                "Tesseract OCR sistemde bulunamadı. "
                "Önce Tesseract uygulamasını kurmalısınız."
            ) from exc

        logger.info(
            "Tesseract OCR hazır. Sürüm: %s",
            version,
        )

        try:
            installed_languages = set(
                pytesseract.get_languages(
                    config=""
                )
            )

        except pytesseract.TesseractError:
            installed_languages = set()

        requested_languages = set(
            self.language.split("+")
        )

        missing_languages = (
            requested_languages
            - installed_languages
        )

        if missing_languages:
            logger.warning(
                "Tesseract dil paketleri eksik olabilir: %s",
                sorted(
                    missing_languages
                ),
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
            "\n",
            " "
        )

        text = cls._WHITESPACE_RE.sub(
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _parse_numeric_value(
        cls,
        text: str,
    ) -> float | None:
        if not text:
            return None

        if not cls._NUMERIC_RE.fullmatch(
            text
        ):
            return None

        normalized = text.casefold()

        for unit in (
            "%",
            "₺",
            "$",
            "€",
            "kg",
            "ton",
            "mwh",
            "kwh",
            "gj",
            "co2e",
            "co2",
            "t",
        ):
            normalized = normalized.replace(
                unit,
                "",
            )

        normalized = normalized.strip()
        normalized = normalized.replace(
            " ",
            "",
        )

        if (
            "," in normalized
            and "." in normalized
        ):
            if (
                normalized.rfind(",")
                > normalized.rfind(".")
            ):
                normalized = normalized.replace(
                    ".",
                    "",
                )

                normalized = normalized.replace(
                    ",",
                    ".",
                )
            else:
                normalized = normalized.replace(
                    ",",
                    "",
                )

        elif "," in normalized:
            parts = normalized.split(",")

            if (
                len(parts) == 2
                and len(parts[1]) <= 3
            ):
                normalized = normalized.replace(
                    ",",
                    ".",
                )
            else:
                normalized = normalized.replace(
                    ",",
                    "",
                )

        try:
            return float(
                normalized
            )

        except ValueError:
            return None

    @staticmethod
    def _crop_borders(
        image: np.ndarray,
        border_pixels: int,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        if (
            height <= border_pixels * 2
            or width <= border_pixels * 2
        ):
            return image

        return image[
            border_pixels:height - border_pixels,
            border_pixels:width - border_pixels,
        ]

    @staticmethod
    def _has_visual_content(
        image: np.ndarray,
    ) -> bool:
        if image.size == 0:
            return False

        if len(
            image.shape
        ) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = image

        dark_pixel_ratio = (
            np.count_nonzero(
                gray < 220
            )
            / max(
                gray.size,
                1,
            )
        )

        return dark_pixel_ratio >= 0.01

    @staticmethod
    def _calculate_image_hash(
        image: np.ndarray,
    ) -> str:
        success, encoded = cv2.imencode(
            ".png",
            image,
        )

        if not success:
            raise TableOCRError(
                "Hücre görüntüsü hash için kodlanamadı."
            )

        return hashlib.sha256(
            encoded.tobytes()
        ).hexdigest()

    @classmethod
    def _calculate_content_hash(
        cls,
        matrix: Sequence[
            Sequence[str]
        ],
    ) -> str:
        canonical = "\n".join(
            "\t".join(
                cls._normalize_text(
                    cell
                ).casefold()
                for cell in row
            )
            for row in matrix
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _get_data_item(
        data: Mapping[str, Sequence[Any]],
        key: str,
        index: int,
    ) -> Any:
        values = data.get(
            key,
            []
        )

        if index >= len(values):
            return 0

        return values[index]

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
            return -1.0

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:
        try:
            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0


def run_table_ocr(
    visual_tables: Sequence[
        VisualExtractedTable
    ],
    *,
    page_index: int,
    page_number: int,
    language: str = "tur+eng",
    tesseract_command: str | Path | None = None,
) -> PageTableOCRResult:
    """TableOCR sınıfı için kısa kullanım arayüzü."""

    table_ocr = TableOCR(
        language=language,
        tesseract_command=tesseract_command,
    )

    return table_ocr.process_page(
        page_index=page_index,
        page_number=page_number,
        tables=visual_tables,
    )