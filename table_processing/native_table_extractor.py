from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]

from .table_detector import (
    DetectedTable,
    TableBoundingBox,
)


logger = logging.getLogger(__name__)


class NativeTableExtractionError(RuntimeError):
    """Native tablo çıkarımı tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class NativeTableCell:
    """Native yöntemle çıkarılan tek bir tablo hücresi."""

    row_index: int
    column_index: int

    text: str
    normalized_text: str

    bounding_box: TableBoundingBox | None = None

    is_header: bool = False
    is_empty: bool = False

    row_span: int = 1
    column_span: int = 1

    confidence: float = 1.0

    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["bounding_box"] = (
            self.bounding_box.to_dict()
            if self.bounding_box is not None
            else None
        )

        return data


@dataclass(slots=True, frozen=True)
class NativeTableRow:
    """Native tablo içerisindeki tek bir satır."""

    row_index: int
    cells: tuple[NativeTableCell, ...]

    is_header: bool
    is_empty: bool

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "is_header": self.is_header,
            "is_empty": self.is_empty,
            "text": self.text,
        }


@dataclass(slots=True, frozen=True)
class NativeExtractedTable:
    """Native tablo çıkarımının yapılandırılmış sonucu."""

    table_id: str
    page_index: int
    page_number: int
    table_index: int

    bounding_box: TableBoundingBox

    extraction_method: str
    extraction_confidence: float

    row_count: int
    column_count: int

    headers: tuple[str, ...]
    rows: tuple[NativeTableRow, ...]

    matrix: tuple[tuple[str, ...], ...]

    caption_text: str | None
    title_text: str | None

    content_hash: str

    is_empty: bool
    is_valid: bool

    warnings: tuple[str, ...]

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "bounding_box": self.bounding_box.to_dict(),
            "extraction_method": self.extraction_method,
            "extraction_confidence": self.extraction_confidence,
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
            "caption_text": self.caption_text,
            "title_text": self.title_text,
            "content_hash": self.content_hash,
            "is_empty": self.is_empty,
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PageNativeTableExtraction:
    """Tek bir sayfada çıkarılan native tabloların sonucu."""

    page_index: int
    page_number: int

    table_count: int
    successful_table_count: int
    failed_table_count: int

    tables: tuple[NativeExtractedTable, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_count": self.table_count,
            "successful_table_count": self.successful_table_count,
            "failed_table_count": self.failed_table_count,
            "tables": [
                table.to_dict()
                for table in self.tables
            ],
            "warnings": list(self.warnings),
        }


class NativeTableExtractor:
    """
    PDF içerisindeki native tabloları yapılandırılmış biçimde çıkarır.

    Çalışma sırası:

    1. Page.find_tables() ile sayfadaki native tablolar bulunur.
    2. TableDetector çıktısındaki bounding box ile en yakın tablo eşleştirilir.
    3. Tablo matrisi Table.extract() üzerinden alınır.
    4. Boş satır ve sütunlar isteğe bağlı temizlenir.
    5. Header satırı otomatik veya yapılandırmaya göre belirlenir.
    6. Hücre ve satır nesneleri oluşturulur.
    7. Tablo bütünlüğü için temel kontroller uygulanır.

    Bu modül OCR veya görsel tablo analizi yapmaz. Native yapı
    bulunamadığında visual_table_extractor.py aşamasına geçilmelidir.
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
        %?
        \s*$
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        *,
        minimum_rows: int = 2,
        minimum_columns: int = 2,
        remove_empty_rows: bool = True,
        remove_empty_columns: bool = True,
        infer_header: bool = True,
        header_row_index: int | None = None,
        maximum_header_rows: int = 2,
        minimum_match_iou: float = 0.25,
        minimum_match_overlap: float = 0.55,
        empty_cell_value: str = "",
    ) -> None:
        if minimum_rows < 1:
            raise ValueError(
                "minimum_rows 1 veya daha büyük olmalıdır."
            )

        if minimum_columns < 1:
            raise ValueError(
                "minimum_columns 1 veya daha büyük olmalıdır."
            )

        if header_row_index is not None and header_row_index < 0:
            raise ValueError(
                "header_row_index negatif olamaz."
            )

        if maximum_header_rows < 1:
            raise ValueError(
                "maximum_header_rows 1 veya daha büyük olmalıdır."
            )

        for name, value in {
            "minimum_match_iou": minimum_match_iou,
            "minimum_match_overlap": minimum_match_overlap,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        self.minimum_rows = minimum_rows
        self.minimum_columns = minimum_columns

        self.remove_empty_rows = remove_empty_rows
        self.remove_empty_columns = remove_empty_columns

        self.infer_header = infer_header
        self.header_row_index = header_row_index
        self.maximum_header_rows = maximum_header_rows

        self.minimum_match_iou = minimum_match_iou
        self.minimum_match_overlap = minimum_match_overlap

        self.empty_cell_value = empty_cell_value

    def extract_pdf(
        self,
        pdf_path: str | Path,
        *,
        detected_tables_by_page: Mapping[
            int,
            Sequence[DetectedTable],
        ]
        | None = None,
        page_numbers: Iterable[int] | None = None,
    ) -> list[PageNativeTableExtraction]:
        """
        PDF içindeki native tabloları çıkarır.

        detected_tables_by_page verilirse yalnızca bu tablo bölgeleriyle
        eşleşen native tablolar işlenir. Sözlük anahtarları 1 tabanlı sayfa
        numaralarıdır.
        """

        pdf_path = Path(
            pdf_path
        ).expanduser().resolve()

        self._validate_pdf_path(
            pdf_path
        )

        detected_tables_by_page = (
            detected_tables_by_page or {}
        )

        try:
            document = pymupdf.open(
                pdf_path
            )
        except Exception as exc:
            raise NativeTableExtractionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise NativeTableExtractionError(
                    "Parola korumalı PDF analiz edilemez."
                )

            if document.page_count == 0:
                raise NativeTableExtractionError(
                    "PDF içinde sayfa bulunamadı."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            results: list[PageNativeTableExtraction] = []

            for page_number in selected_pages:
                page = document.load_page(
                    page_number - 1
                )

                result = self.extract_page(
                    page,
                    detected_tables=(
                        detected_tables_by_page.get(
                            page_number
                        )
                    ),
                )

                results.append(result)

            logger.info(
                "%d sayfada %d native tablo çıkarıldı: %s",
                len(results),
                sum(
                    result.successful_table_count
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
        detected_tables: Sequence[DetectedTable] | None = None,
    ) -> PageNativeTableExtraction:
        """Açık bir PyMuPDF sayfasındaki native tabloları çıkarır."""

        page_index = int(page.number)
        page_number = page_index + 1

        warnings: list[str] = []

        if not hasattr(
            page,
            "find_tables",
        ):
            raise NativeTableExtractionError(
                "Kullanılan PyMuPDF sürümünde "
                "Page.find_tables arayüzü bulunmuyor."
            )

        try:
            finder = page.find_tables()
        except Exception as exc:
            raise NativeTableExtractionError(
                f"Sayfa {page_number} native tablo analizi başarısız."
            ) from exc

        native_tables = list(
            getattr(
                finder,
                "tables",
                [],
            )
        )

        if not native_tables:
            return PageNativeTableExtraction(
                page_index=page_index,
                page_number=page_number,
                table_count=0,
                successful_table_count=0,
                failed_table_count=0,
                tables=(),
                warnings=(),
            )

        matches = self._match_tables(
            page=page,
            native_tables=native_tables,
            detected_tables=detected_tables,
        )

        extracted_tables: list[NativeExtractedTable] = []
        failed_count = 0

        for output_index, match in enumerate(matches):
            native_table = match["native_table"]
            detected_table = match["detected_table"]
            bounding_box = match["bounding_box"]

            try:
                extracted = self._extract_single_table(
                    page=page,
                    native_table=native_table,
                    detected_table=detected_table,
                    bounding_box=bounding_box,
                    output_index=output_index,
                    match_metadata=match["metadata"],
                )

            except Exception as exc:
                failed_count += 1

                logger.warning(
                    "Sayfa %d tablo %d çıkarılamadı: %s",
                    page_number,
                    output_index + 1,
                    exc,
                )

                warnings.append(
                    f"Tablo {output_index + 1} çıkarılamadı: {exc}"
                )

                continue

            extracted_tables.append(extracted)

        return PageNativeTableExtraction(
            page_index=page_index,
            page_number=page_number,
            table_count=len(matches),
            successful_table_count=len(extracted_tables),
            failed_table_count=failed_count,
            tables=tuple(extracted_tables),
            warnings=tuple(warnings),
        )

    def _extract_single_table(
        self,
        *,
        page: pymupdf.Page,
        native_table: Any,
        detected_table: DetectedTable | None,
        bounding_box: TableBoundingBox,
        output_index: int,
        match_metadata: Mapping[str, Any],
    ) -> NativeExtractedTable:
        page_number = int(page.number) + 1

        try:
            raw_matrix = native_table.extract()
        except Exception as exc:
            raise NativeTableExtractionError(
                "Table.extract() çağrısı başarısız."
            ) from exc

        matrix = self._normalize_matrix(
            raw_matrix
        )

        original_row_count = len(matrix)

        original_column_count = max(
            (
                len(row)
                for row in matrix
            ),
            default=0,
        )

        warnings: list[str] = []

        if self.remove_empty_rows:
            matrix = self._remove_empty_rows(
                matrix
            )

        matrix = self._rectangularize_matrix(
            matrix
        )

        if self.remove_empty_columns:
            matrix = self._remove_empty_columns(
                matrix
            )

        matrix = self._rectangularize_matrix(
            matrix
        )

        row_count = len(matrix)

        column_count = max(
            (
                len(row)
                for row in matrix
            ),
            default=0,
        )

        if row_count < self.minimum_rows:
            warnings.append(
                f"Tablo satır sayısı minimum eşikten düşük: {row_count}."
            )

        if column_count < self.minimum_columns:
            warnings.append(
                f"Tablo sütun sayısı minimum eşikten düşük: {column_count}."
            )

        header_indices = self._detect_header_rows(
            matrix
        )

        headers = self._build_headers(
            matrix=matrix,
            header_indices=header_indices,
        )

        cell_boxes = self._extract_cell_boxes(
            native_table=native_table,
            page_rect=page.rect,
            row_count=row_count,
            column_count=column_count,
        )

        rows = self._build_rows(
            matrix=matrix,
            header_indices=header_indices,
            cell_boxes=cell_boxes,
        )

        caption_text = (
            detected_table.caption_text
            if detected_table is not None
            else self._find_caption(
                page=page,
                table_box=bounding_box,
            )
        )

        title_text = self._find_title(
            page=page,
            table_box=bounding_box,
            caption_text=caption_text,
        )

        content_hash = self._calculate_content_hash(
            matrix
        )

        is_empty = not any(
            cell.strip()
            for row in matrix
            for cell in row
        )

        is_valid = (
            not is_empty
            and row_count >= self.minimum_rows
            and column_count >= self.minimum_columns
        )

        confidence = self._calculate_confidence(
            detected_table=detected_table,
            row_count=row_count,
            column_count=column_count,
            original_row_count=original_row_count,
            original_column_count=original_column_count,
            warnings=warnings,
        )

        table_id = (
            detected_table.table_id
            if detected_table is not None
            else (
                f"page-{page_number}"
                f"-native-table-{output_index + 1}"
            )
        )

        table_index = (
            detected_table.table_index
            if detected_table is not None
            else output_index
        )

        return NativeExtractedTable(
            table_id=table_id,
            page_index=int(page.number),
            page_number=page_number,
            table_index=table_index,
            bounding_box=bounding_box,
            extraction_method="pymupdf_native",
            extraction_confidence=confidence,
            row_count=row_count,
            column_count=column_count,
            headers=tuple(headers),
            rows=tuple(rows),
            matrix=tuple(
                tuple(row)
                for row in matrix
            ),
            caption_text=caption_text,
            title_text=title_text,
            content_hash=content_hash,
            is_empty=is_empty,
            is_valid=is_valid,
            warnings=tuple(warnings),
            metadata={
                "original_row_count": original_row_count,
                "original_column_count": original_column_count,
                "header_row_indices": header_indices,
                "matched_detected_table": (
                    detected_table is not None
                ),
                **dict(match_metadata),
            },
        )

    def _match_tables(
        self,
        *,
        page: pymupdf.Page,
        native_tables: Sequence[Any],
        detected_tables: Sequence[DetectedTable] | None,
    ) -> list[dict[str, Any]]:
        native_entries: list[dict[str, Any]] = []

        for native_index, native_table in enumerate(
            native_tables
        ):
            bounding_box = self._to_bounding_box(
                getattr(
                    native_table,
                    "bbox",
                    None,
                ),
                page.rect,
            )

            if bounding_box is None:
                continue

            native_entries.append(
                {
                    "native_index": native_index,
                    "native_table": native_table,
                    "bounding_box": bounding_box,
                }
            )

        if not detected_tables:
            return [
                {
                    **entry,
                    "detected_table": None,
                    "metadata": {
                        "native_index": entry["native_index"],
                        "match_iou": None,
                        "match_overlap": None,
                    },
                }
                for entry in native_entries
            ]

        matches: list[dict[str, Any]] = []
        used_native_indices: set[int] = set()

        eligible_detected_tables = [
            table
            for table in detected_tables
            if table.has_native_structure
        ]

        for detected_table in eligible_detected_tables:
            best_entry: dict[str, Any] | None = None
            best_score = -1.0
            best_iou = 0.0
            best_overlap = 0.0

            for entry in native_entries:
                native_index = int(
                    entry["native_index"]
                )

                if native_index in used_native_indices:
                    continue

                native_box = entry["bounding_box"]

                iou = self._intersection_over_union(
                    detected_table.bounding_box,
                    native_box,
                )

                overlap = max(
                    self._overlap_ratio(
                        detected_table.bounding_box,
                        native_box,
                    ),
                    self._overlap_ratio(
                        native_box,
                        detected_table.bounding_box,
                    ),
                )

                score = (
                    0.55 * iou
                    + 0.45 * overlap
                )

                if score > best_score:
                    best_entry = entry
                    best_score = score
                    best_iou = iou
                    best_overlap = overlap

            if best_entry is None:
                continue

            if (
                best_iou < self.minimum_match_iou
                and best_overlap < self.minimum_match_overlap
            ):
                continue

            native_index = int(
                best_entry["native_index"]
            )

            used_native_indices.add(
                native_index
            )

            matches.append(
                {
                    **best_entry,
                    "detected_table": detected_table,
                    "metadata": {
                        "native_index": native_index,
                        "match_iou": round(best_iou, 4),
                        "match_overlap": round(
                            best_overlap,
                            4,
                        ),
                        "match_score": round(
                            best_score,
                            4,
                        ),
                    },
                }
            )

        # Native tablo detector tarafından yakalanmadıysa kaybolmaması için
        # kalan native tablolar da sonuçlara eklenir.
        for entry in native_entries:
            native_index = int(
                entry["native_index"]
            )

            if native_index in used_native_indices:
                continue

            matches.append(
                {
                    **entry,
                    "detected_table": None,
                    "metadata": {
                        "native_index": native_index,
                        "match_iou": None,
                        "match_overlap": None,
                        "native_only": True,
                    },
                }
            )

        matches.sort(
            key=lambda item: (
                item["bounding_box"].y0,
                item["bounding_box"].x0,
            )
        )

        return matches

    def _normalize_matrix(
        self,
        matrix: Any,
    ) -> list[list[str]]:
        if matrix is None:
            return []

        normalized_rows: list[list[str]] = []

        for raw_row in matrix:
            if raw_row is None:
                continue

            if isinstance(
                raw_row,
                (
                    str,
                    bytes,
                    bytearray,
                ),
            ):
                raw_cells = [raw_row]
            else:
                try:
                    raw_cells = list(raw_row)
                except TypeError:
                    raw_cells = [raw_row]

            normalized_rows.append(
                [
                    self._normalize_cell_text(
                        cell
                    )
                    for cell in raw_cells
                ]
            )

        return normalized_rows

    def _normalize_cell_text(
        self,
        value: Any,
    ) -> str:
        if value is None:
            return self.empty_cell_value

        text = str(value)

        text = text.replace(
            "\u00ad",
            "",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            self._WHITESPACE_RE.sub(
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

        return " ".join(lines)

    def _remove_empty_rows(
        self,
        matrix: Sequence[Sequence[str]],
    ) -> list[list[str]]:
        return [
            list(row)
            for row in matrix
            if any(
                str(cell).strip()
                for cell in row
            )
        ]

    def _remove_empty_columns(
        self,
        matrix: Sequence[Sequence[str]],
    ) -> list[list[str]]:
        if not matrix:
            return []

        rectangular = self._rectangularize_matrix(
            matrix
        )

        column_count = len(
            rectangular[0]
        )

        kept_indices = [
            column_index
            for column_index in range(
                column_count
            )
            if any(
                row[column_index].strip()
                for row in rectangular
            )
        ]

        return [
            [
                row[column_index]
                for column_index in kept_indices
            ]
            for row in rectangular
        ]

    def _rectangularize_matrix(
        self,
        matrix: Sequence[Sequence[str]],
    ) -> list[list[str]]:
        if not matrix:
            return []

        column_count = max(
            len(row)
            for row in matrix
        )

        return [
            [
                *list(row),
                *[
                    self.empty_cell_value
                    for _ in range(
                        column_count - len(row)
                    )
                ],
            ]
            for row in matrix
        ]

    def _detect_header_rows(
        self,
        matrix: Sequence[Sequence[str]],
    ) -> list[int]:
        if not matrix:
            return []

        if self.header_row_index is not None:
            if self.header_row_index >= len(matrix):
                return []

            return [
                self.header_row_index
            ]

        if not self.infer_header:
            return []

        header_indices: list[int] = []

        maximum_rows = min(
            self.maximum_header_rows,
            len(matrix),
        )

        for row_index in range(
            maximum_rows
        ):
            row = matrix[row_index]

            non_empty_cells = [
                cell
                for cell in row
                if cell.strip()
            ]

            if not non_empty_cells:
                continue

            numeric_count = sum(
                self._looks_numeric(cell)
                for cell in non_empty_cells
            )

            text_count = (
                len(non_empty_cells)
                - numeric_count
            )

            average_length = sum(
                len(cell)
                for cell in non_empty_cells
            ) / len(non_empty_cells)

            unique_ratio = (
                len(
                    set(
                        cell.casefold()
                        for cell in non_empty_cells
                    )
                )
                / len(non_empty_cells)
            )

            looks_like_header = (
                text_count >= numeric_count
                and unique_ratio >= 0.60
                and average_length <= 80
            )

            if row_index == 0 and looks_like_header:
                header_indices.append(
                    row_index
                )

            elif (
                header_indices
                and looks_like_header
                and row_index
                == header_indices[-1] + 1
            ):
                header_indices.append(
                    row_index
                )

            else:
                break

        return header_indices

    def _build_headers(
        self,
        *,
        matrix: Sequence[Sequence[str]],
        header_indices: Sequence[int],
    ) -> list[str]:
        if not matrix:
            return []

        column_count = len(
            matrix[0]
        )

        if not header_indices:
            return [
                f"column_{index + 1}"
                for index in range(
                    column_count
                )
            ]

        headers: list[str] = []

        for column_index in range(
            column_count
        ):
            parts = [
                matrix[row_index][column_index]
                for row_index in header_indices
                if matrix[row_index][column_index].strip()
            ]

            header = " | ".join(
                dict.fromkeys(parts)
            ).strip()

            if not header:
                header = (
                    f"column_{column_index + 1}"
                )

            headers.append(header)

        return self._deduplicate_headers(
            headers
        )

    @staticmethod
    def _deduplicate_headers(
        headers: Sequence[str],
    ) -> list[str]:
        counts: dict[str, int] = {}
        result: list[str] = []

        for header in headers:
            base = (
                header.strip()
                or "column"
            )

            normalized = base.casefold()

            counts[normalized] = (
                counts.get(
                    normalized,
                    0,
                )
                + 1
            )

            count = counts[normalized]

            result.append(
                base
                if count == 1
                else f"{base}_{count}"
            )

        return result

    def _build_rows(
        self,
        *,
        matrix: Sequence[Sequence[str]],
        header_indices: Sequence[int],
        cell_boxes: Mapping[
            tuple[int, int],
            TableBoundingBox,
        ],
    ) -> list[NativeTableRow]:
        rows: list[NativeTableRow] = []

        header_index_set = set(
            header_indices
        )

        for row_index, row in enumerate(
            matrix
        ):
            is_header = (
                row_index in header_index_set
            )

            cells: list[NativeTableCell] = []

            for column_index, text in enumerate(
                row
            ):
                cells.append(
                    NativeTableCell(
                        row_index=row_index,
                        column_index=column_index,
                        text=text,
                        normalized_text=(
                            self._normalize_for_search(
                                text
                            )
                        ),
                        bounding_box=cell_boxes.get(
                            (
                                row_index,
                                column_index,
                            )
                        ),
                        is_header=is_header,
                        is_empty=not bool(
                            text.strip()
                        ),
                        confidence=1.0,
                        metadata=None,
                    )
                )

            rows.append(
                NativeTableRow(
                    row_index=row_index,
                    cells=tuple(cells),
                    is_header=is_header,
                    is_empty=not any(
                        cell.text.strip()
                        for cell in cells
                    ),
                    text=" | ".join(
                        cell.text
                        for cell in cells
                    ),
                )
            )

        return rows

    def _extract_cell_boxes(
        self,
        *,
        native_table: Any,
        page_rect: pymupdf.Rect,
        row_count: int,
        column_count: int,
    ) -> dict[
        tuple[int, int],
        TableBoundingBox,
    ]:
        result: dict[
            tuple[int, int],
            TableBoundingBox,
        ] = {}

        raw_cells = getattr(
            native_table,
            "cells",
            None,
        )

        if not raw_cells:
            return result

        cell_list = list(
            raw_cells
        )

        expected_count = (
            row_count
            * column_count
        )

        if expected_count <= 0:
            return result

        # PyMuPDF sürümüne göre cells düz bir liste olarak dönebilir.
        for flat_index, raw_cell in enumerate(
            cell_list[:expected_count]
        ):
            if raw_cell is None:
                continue

            bounding_box = self._to_bounding_box(
                raw_cell,
                page_rect,
            )

            if bounding_box is None:
                continue

            row_index = (
                flat_index
                // column_count
            )

            column_index = (
                flat_index
                % column_count
            )

            result[
                (
                    row_index,
                    column_index,
                )
            ] = bounding_box

        return result

    def _find_caption(
        self,
        *,
        page: pymupdf.Page,
        table_box: TableBoundingBox,
    ) -> str | None:
        candidates: list[
            tuple[float, str]
        ] = []

        for block in page.get_text(
            "blocks",
            sort=True,
        ):
            if len(block) < 5:
                continue

            text = str(
                block[4]
            ).strip()

            if not text:
                continue

            normalized = (
                text.casefold()
            )

            if not (
                normalized.startswith(
                    "table"
                )
                or normalized.startswith(
                    "tablo"
                )
            ):
                continue

            block_box = self._to_bounding_box(
                block[:4],
                page.rect,
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
                vertical_distance <= 60
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

    def _find_title(
        self,
        *,
        page: pymupdf.Page,
        table_box: TableBoundingBox,
        caption_text: str | None,
    ) -> str | None:
        candidates: list[
            tuple[float, str]
        ] = []

        for block in page.get_text(
            "blocks",
            sort=True,
        ):
            if len(block) < 5:
                continue

            text = str(
                block[4]
            ).strip()

            if not text:
                continue

            if (
                caption_text
                and text == caption_text
            ):
                continue

            block_box = self._to_bounding_box(
                block[:4],
                page.rect,
            )

            if block_box is None:
                continue

            # Tablo üstündeki en yakın kısa metni olası başlık kabul eder.
            vertical_distance = (
                table_box.y0
                - block_box.y1
            )

            if not 0 <= vertical_distance <= 90:
                continue

            if len(text) > 250:
                continue

            horizontal_overlap = (
                self._horizontal_overlap_ratio(
                    block_box,
                    table_box,
                )
            )

            if horizontal_overlap < 0.20:
                continue

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

    def _calculate_confidence(
        self,
        *,
        detected_table: DetectedTable | None,
        row_count: int,
        column_count: int,
        original_row_count: int,
        original_column_count: int,
        warnings: Sequence[str],
    ) -> float:
        confidence = 0.86

        if detected_table is not None:
            confidence = (
                0.55 * confidence
                + 0.45 * detected_table.confidence
            )

        if row_count >= 3:
            confidence += 0.03

        if column_count >= 2:
            confidence += 0.03

        if (
            row_count == original_row_count
            and column_count
            == original_column_count
        ):
            confidence += 0.02

        confidence -= min(
            len(warnings) * 0.06,
            0.24,
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

    @classmethod
    def _calculate_content_hash(
        cls,
        matrix: Sequence[
            Sequence[str]
        ],
    ) -> str:
        canonical = "\n".join(
            "\t".join(
                cls._normalize_for_search(
                    cell
                )
                for cell in row
            )
            for row in matrix
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

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
                "Ş": "ş",
                "Ğ": "ğ",
                "Ü": "ü",
                "Ö": "ö",
                "Ç": "ç",
            }
        )

        normalized = text.translate(
            translation_table
        ).casefold()

        return cls._WHITESPACE_RE.sub(
            " ",
            normalized,
        ).strip()

    @classmethod
    def _looks_numeric(
        cls,
        value: str,
    ) -> bool:
        return bool(
            cls._NUMERIC_RE.fullmatch(
                value
            )
        )

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

        if x1 <= x0 or y1 <= y0:
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

        overlap = x1 - x0

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


def extract_native_tables(
    pdf_path: str | Path,
    *,
    detected_tables_by_page: Mapping[
        int,
        Sequence[DetectedTable],
    ]
    | None = None,
    page_numbers: Iterable[int] | None = None,
) -> list[PageNativeTableExtraction]:
    """NativeTableExtractor için kısa kullanım arayüzü."""

    extractor = NativeTableExtractor()

    return extractor.extract_pdf(
        pdf_path=pdf_path,
        detected_tables_by_page=(
            detected_tables_by_page
        ),
        page_numbers=page_numbers,
    )