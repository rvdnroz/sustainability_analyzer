from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf  # type: ignore[no-redef]

from .table_detector import DetectedTable, TableBoundingBox


logger = logging.getLogger(__name__)


class BorderlessTableExtractionError(RuntimeError):
    """Çizgisiz tablo çıkarımı tamamlanamadığında yükseltilir."""


@dataclass(slots=True, frozen=True)
class BorderlessTableCell:
    """Koordinat tabanlı yöntemle oluşturulan tek hücre."""

    row_index: int
    column_index: int

    text: str
    normalized_text: str

    bounding_box: TableBoundingBox | None

    is_header: bool
    is_empty: bool
    is_numeric: bool

    confidence: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bounding_box"] = (
            self.bounding_box.to_dict()
            if self.bounding_box is not None
            else None
        )
        return data


@dataclass(slots=True, frozen=True)
class BorderlessTableRow:
    """Çizgisiz tabloda oluşturulan tek satır."""

    row_index: int
    cells: tuple[BorderlessTableCell, ...]

    is_header: bool
    is_empty: bool
    text: str

    confidence: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [cell.to_dict() for cell in self.cells],
            "is_header": self.is_header,
            "is_empty": self.is_empty,
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class BorderlessExtractedTable:
    """
    Çizgisiz tablo çıkarım sonucu.

    Alan adları TableNormalizer'ın ortak giriş sözleşmesiyle uyumludur.
    """

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
    rows: tuple[BorderlessTableRow, ...]
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
            "rows": [row.to_dict() for row in self.rows],
            "matrix": [list(row) for row in self.matrix],
            "caption_text": self.caption_text,
            "title_text": self.title_text,
            "content_hash": self.content_hash,
            "is_empty": self.is_empty,
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PageBorderlessTableExtraction:
    """Tek sayfadaki çizgisiz tablo çıkarım sonuçları."""

    page_index: int
    page_number: int

    table_count: int
    successful_table_count: int
    failed_table_count: int

    tables: tuple[BorderlessExtractedTable, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_count": self.table_count,
            "successful_table_count": self.successful_table_count,
            "failed_table_count": self.failed_table_count,
            "tables": [table.to_dict() for table in self.tables],
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class _WordToken:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_number: int
    line_number: int
    word_number: int

    @property
    def width(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def center_y(self) -> float:
        return self.y0 + self.height / 2.0

    @property
    def box(self) -> TableBoundingBox:
        return TableBoundingBox(
            x0=round(self.x0, 3),
            y0=round(self.y0, 3),
            x1=round(self.x1, 3),
            y1=round(self.y1, 3),
        )


@dataclass(slots=True)
class _TextSegment:
    text: str
    words: list[_WordToken]
    box: TableBoundingBox
    is_numeric: bool

    @property
    def x0(self) -> float:
        return self.box.x0

    @property
    def x1(self) -> float:
        return self.box.x1

    @property
    def center_x(self) -> float:
        return self.box.center_x


class BorderlessTableExtractor:
    """
    Çizgisiz PDF tablolarını kelime koordinatlarından yeniden kurar.

    İşlem sırası:

    1. DetectedTable bölgesindeki PDF kelimeleri alınır.
    2. Kelimeler y koordinatlarına göre satırlara gruplanır.
    3. Satırlardaki büyük yatay boşluklar hücre sınırı kabul edilir.
    4. Segment başlangıçları sayfa boyunca kümelenerek sütun çapaları kurulur.
    5. Segmentler en yakın sütuna yerleştirilir.
    6. Matris, hücreler, satırlar, başlıklar ve güven skorları üretilir.

    OCR kullanmaz. Native metin katmanı bulunan çizgisiz tablolar için
    visual OCR'dan önce çalıştırılması amaçlanmıştır.
    """

    _WHITESPACE_RE = re.compile(r"\s+")
    _YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
    _NUMERIC_RE = re.compile(
        r"""
        ^\s*
        [~≈<>≤≥]?\s*
        [+-]?
        (?:
            \d{1,3}(?:[.\s,]\d{3})+
            |
            \d+
        )
        (?:[.,]\d+)?
        \s*
        (?:
            %|₺|\$|€|
            kg|g|t|ton|tons|tonne|tonnes|
            m3|m³|l|lt|
            kwh|mwh|gwh|gj|mj|
            co2e|tco2e|kgco2e
        )?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(
        self,
        *,
        minimum_rows: int = 2,
        minimum_columns: int = 2,
        maximum_columns: int = 12,
        line_y_tolerance_ratio: float = 0.60,
        minimum_line_y_tolerance: float = 2.5,
        segment_gap_multiplier: float = 2.4,
        minimum_segment_gap: float = 14.0,
        column_cluster_tolerance: float = 24.0,
        minimum_column_support: int = 2,
        maximum_empty_cell_ratio: float = 0.72,
        minimum_numeric_row_ratio: float = 0.18,
        infer_header: bool = True,
        remove_empty_rows: bool = True,
        remove_empty_columns: bool = True,
        exclude_caption_from_region: bool = True,
    ) -> None:
        if minimum_rows < 1:
            raise ValueError("minimum_rows en az 1 olmalıdır.")
        if minimum_columns < 1:
            raise ValueError("minimum_columns en az 1 olmalıdır.")
        if maximum_columns < minimum_columns:
            raise ValueError(
                "maximum_columns minimum_columns değerinden küçük olamaz."
            )
        if line_y_tolerance_ratio <= 0:
            raise ValueError("line_y_tolerance_ratio pozitif olmalıdır.")
        if minimum_line_y_tolerance <= 0:
            raise ValueError("minimum_line_y_tolerance pozitif olmalıdır.")
        if segment_gap_multiplier <= 0:
            raise ValueError("segment_gap_multiplier pozitif olmalıdır.")
        if minimum_segment_gap < 0:
            raise ValueError("minimum_segment_gap negatif olamaz.")
        if column_cluster_tolerance <= 0:
            raise ValueError("column_cluster_tolerance pozitif olmalıdır.")
        if minimum_column_support < 1:
            raise ValueError("minimum_column_support en az 1 olmalıdır.")
        if not 0.0 <= maximum_empty_cell_ratio <= 1.0:
            raise ValueError(
                "maximum_empty_cell_ratio 0 ile 1 arasında olmalıdır."
            )
        if not 0.0 <= minimum_numeric_row_ratio <= 1.0:
            raise ValueError(
                "minimum_numeric_row_ratio 0 ile 1 arasında olmalıdır."
            )

        self.minimum_rows = minimum_rows
        self.minimum_columns = minimum_columns
        self.maximum_columns = maximum_columns
        self.line_y_tolerance_ratio = line_y_tolerance_ratio
        self.minimum_line_y_tolerance = minimum_line_y_tolerance
        self.segment_gap_multiplier = segment_gap_multiplier
        self.minimum_segment_gap = minimum_segment_gap
        self.column_cluster_tolerance = column_cluster_tolerance
        self.minimum_column_support = minimum_column_support
        self.maximum_empty_cell_ratio = maximum_empty_cell_ratio
        self.minimum_numeric_row_ratio = minimum_numeric_row_ratio
        self.infer_header = infer_header
        self.remove_empty_rows = remove_empty_rows
        self.remove_empty_columns = remove_empty_columns
        self.exclude_caption_from_region = exclude_caption_from_region

    def extract_pdf(
        self,
        pdf_path: str | Path,
        *,
        detected_tables_by_page: Mapping[
            int,
            Sequence[DetectedTable],
        ],
        page_numbers: Iterable[int] | None = None,
    ) -> list[PageBorderlessTableExtraction]:
        pdf_path = Path(pdf_path).expanduser().resolve()
        self._validate_pdf_path(pdf_path)

        try:
            document = pymupdf.open(pdf_path)
        except Exception as exc:
            raise BorderlessTableExtractionError(
                f"PDF açılamadı: {pdf_path}"
            ) from exc

        try:
            if document.needs_pass:
                raise BorderlessTableExtractionError(
                    "Parola korumalı PDF analiz edilemez."
                )
            if document.page_count == 0:
                raise BorderlessTableExtractionError(
                    "PDF içinde sayfa bulunamadı."
                )

            selected_pages = self._normalize_pages(
                page_numbers=page_numbers,
                page_count=document.page_count,
            )

            results: list[PageBorderlessTableExtraction] = []

            for page_number in selected_pages:
                page = document.load_page(page_number - 1)

                detected_tables = tuple(
                    detected_tables_by_page.get(page_number, ())
                )

                results.append(
                    self.extract_page(
                        page,
                        detected_tables=detected_tables,
                    )
                )

            return results
        finally:
            document.close()

    def extract_page(
        self,
        page: pymupdf.Page,
        *,
        detected_tables: Sequence[DetectedTable],
    ) -> PageBorderlessTableExtraction:
        page_index = int(page.number)
        page_number = page_index + 1

        tables: list[BorderlessExtractedTable] = []
        warnings: list[str] = []
        failed_count = 0

        for table in detected_tables:
            try:
                result = self.extract_table(
                    page,
                    detected_table=table,
                )
            except Exception as exc:
                failed_count += 1
                message = (
                    f"Sayfa {page_number}, tablo {table.table_index + 1} "
                    f"çizgisiz çıkarım başarısız: {exc}"
                )
                warnings.append(message)
                logger.warning(message)
                continue

            tables.append(result)

        return PageBorderlessTableExtraction(
            page_index=page_index,
            page_number=page_number,
            table_count=len(detected_tables),
            successful_table_count=len(tables),
            failed_table_count=failed_count,
            tables=tuple(tables),
            warnings=tuple(warnings),
        )

    def extract_table(
        self,
        page: pymupdf.Page,
        *,
        detected_table: DetectedTable,
    ) -> BorderlessExtractedTable:
        region = self._effective_region(
            detected_table=detected_table,
            page_rect=page.rect,
        )

        words = self._extract_words(
            page,
            region=region,
        )

        warnings: list[str] = []

        if not words:
            return self._empty_result(
                detected_table=detected_table,
                region=region,
                warnings=(
                    "Tablo bölgesinde native metin kelimesi bulunamadı.",
                ),
            )

        lines = self._group_words_into_lines(words)

        if len(lines) < self.minimum_rows:
            warnings.append(
                f"Satır sayısı minimum eşikten düşük: {len(lines)}."
            )

        segments_by_row = [
            self._segment_line(line_words)
            for line_words in lines
        ]
        segments_by_row = [
            segments
            for segments in segments_by_row
            if segments
        ]

        if not segments_by_row:
            return self._empty_result(
                detected_table=detected_table,
                region=region,
                warnings=(
                    "Kelime koordinatlarından hücre segmenti oluşturulamadı.",
                ),
            )

        anchors = self._infer_column_anchors(
            segments_by_row
        )

        if len(anchors) < self.minimum_columns:
            anchors = self._fallback_two_column_anchors(
                segments_by_row
            )

        if len(anchors) > self.maximum_columns:
            warnings.append(
                "Algılanan sütun sayısı üst sınırı aştı; "
                f"{len(anchors)} sütun {self.maximum_columns} sütuna indirildi."
            )
            anchors = self._reduce_anchors(
                anchors,
                maximum=self.maximum_columns,
            )

        matrix, cell_boxes, placement_scores = self._build_matrix(
            segments_by_row=segments_by_row,
            anchors=anchors,
        )

        if self.remove_empty_rows:
            matrix, cell_boxes, placement_scores = self._remove_empty_rows(
                matrix,
                cell_boxes,
                placement_scores,
            )

        if self.remove_empty_columns:
            matrix, cell_boxes, anchors = self._remove_empty_columns(
                matrix,
                cell_boxes,
                anchors,
            )

        row_count = len(matrix)
        column_count = max(
            (len(row) for row in matrix),
            default=0,
        )

        matrix = [
            row + [""] * (column_count - len(row))
            for row in matrix
        ]

        header_row_index = (
            self._infer_header_row(matrix)
            if self.infer_header
            else None
        )

        headers = self._build_headers(
            matrix=matrix,
            header_row_index=header_row_index,
        )

        rows = self._build_rows(
            matrix=matrix,
            cell_boxes=cell_boxes,
            placement_scores=placement_scores,
            header_row_index=header_row_index,
        )

        non_empty_cell_count = sum(
            bool(self._normalize_text(value))
            for row in matrix
            for value in row
        )
        total_cell_count = max(
            row_count * max(column_count, 1),
            1,
        )
        empty_cell_ratio = (
            1.0 - non_empty_cell_count / total_cell_count
        )

        numeric_rows = sum(
            any(self._looks_numeric(value) for value in row)
            for row in matrix
        )
        numeric_row_ratio = (
            numeric_rows / max(row_count, 1)
        )

        repeated_structure_ratio = (
            self._repeated_structure_ratio(matrix)
        )

        average_placement_score = (
            sum(placement_scores) / len(placement_scores)
            if placement_scores
            else 0.0
        )

        is_empty = non_empty_cell_count == 0
        is_valid = (
            not is_empty
            and row_count >= self.minimum_rows
            and column_count >= self.minimum_columns
            and empty_cell_ratio <= self.maximum_empty_cell_ratio
            and (
                numeric_row_ratio >= self.minimum_numeric_row_ratio
                or repeated_structure_ratio >= 0.55
            )
        )

        if empty_cell_ratio > self.maximum_empty_cell_ratio:
            warnings.append(
                "Boş hücre oranı eşik üzerinde: "
                f"{empty_cell_ratio:.2%}."
            )

        if numeric_row_ratio < self.minimum_numeric_row_ratio:
            warnings.append(
                "Sayısal veri içeren satır oranı düşük: "
                f"{numeric_row_ratio:.2%}."
            )

        if repeated_structure_ratio < 0.40:
            warnings.append(
                "Satır yapıları arasında sütun tutarlılığı düşük."
            )

        confidence = self._calculate_confidence(
            detection_confidence=detected_table.confidence,
            row_count=row_count,
            column_count=column_count,
            empty_cell_ratio=empty_cell_ratio,
            numeric_row_ratio=numeric_row_ratio,
            repeated_structure_ratio=repeated_structure_ratio,
            average_placement_score=average_placement_score,
            is_valid=is_valid,
        )

        content_hash = self._calculate_content_hash(
            matrix
        )

        return BorderlessExtractedTable(
            table_id=detected_table.table_id,
            page_index=int(page.number),
            page_number=int(page.number) + 1,
            table_index=detected_table.table_index,
            bounding_box=region,
            extraction_method="pdf_word_coordinates",
            extraction_confidence=confidence,
            row_count=row_count,
            column_count=column_count,
            headers=tuple(headers),
            rows=tuple(rows),
            matrix=tuple(
                tuple(row)
                for row in matrix
            ),
            caption_text=detected_table.caption_text,
            title_text=detected_table.caption_text,
            content_hash=content_hash,
            is_empty=is_empty,
            is_valid=is_valid,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                **dict(detected_table.metadata or {}),
                "source_detection_method": (
                    detected_table.detection_method
                ),
                "source_detection_confidence": (
                    detected_table.confidence
                ),
                "source_word_count": len(words),
                "source_line_count": len(lines),
                "segment_row_count": len(segments_by_row),
                "column_anchors": [
                    round(anchor, 3)
                    for anchor in anchors
                ],
                "empty_cell_ratio": round(
                    empty_cell_ratio,
                    4,
                ),
                "numeric_row_ratio": round(
                    numeric_row_ratio,
                    4,
                ),
                "repeated_structure_ratio": round(
                    repeated_structure_ratio,
                    4,
                ),
                "average_placement_score": round(
                    average_placement_score,
                    4,
                ),
                "header_row_index": header_row_index,
                "table_type": (
                    detected_table.metadata.get("table_type")
                    if detected_table.metadata
                    else "borderless_table"
                ),
                "requires_visual_fallback": not is_valid,
            },
        )

    def _extract_words(
        self,
        page: pymupdf.Page,
        *,
        region: TableBoundingBox,
    ) -> list[_WordToken]:
        try:
            raw_words = page.get_text(
                "words",
                clip=region.to_rect(),
                sort=True,
            )
        except Exception as exc:
            raise BorderlessTableExtractionError(
                "PDF kelime koordinatları okunamadı."
            ) from exc

        words: list[_WordToken] = []

        for item in raw_words:
            if len(item) < 5:
                continue

            text = self._normalize_text(item[4])
            if not text:
                continue

            words.append(
                _WordToken(
                    text=text,
                    x0=float(item[0]),
                    y0=float(item[1]),
                    x1=float(item[2]),
                    y1=float(item[3]),
                    block_number=(
                        int(item[5])
                        if len(item) > 5
                        else 0
                    ),
                    line_number=(
                        int(item[6])
                        if len(item) > 6
                        else 0
                    ),
                    word_number=(
                        int(item[7])
                        if len(item) > 7
                        else 0
                    ),
                )
            )

        return words

    def _group_words_into_lines(
        self,
        words: Sequence[_WordToken],
    ) -> list[list[_WordToken]]:
        if not words:
            return []

        typical_height = median(
            word.height
            for word in words
            if word.height > 0
        )
        tolerance = max(
            self.minimum_line_y_tolerance,
            typical_height * self.line_y_tolerance_ratio,
        )

        ordered = sorted(
            words,
            key=lambda word: (
                word.center_y,
                word.x0,
            ),
        )

        lines: list[list[_WordToken]] = []

        for word in ordered:
            best_index: int | None = None
            best_distance = math.inf

            for index, line in enumerate(lines):
                line_center = median(
                    item.center_y
                    for item in line
                )
                distance = abs(
                    word.center_y - line_center
                )

                if distance <= tolerance and distance < best_distance:
                    best_index = index
                    best_distance = distance

            if best_index is None:
                lines.append([word])
            else:
                lines[best_index].append(word)

        lines = [
            sorted(line, key=lambda word: word.x0)
            for line in lines
        ]

        lines.sort(
            key=lambda line: median(
                word.center_y
                for word in line
            )
        )

        return lines

    def _segment_line(
        self,
        words: Sequence[_WordToken],
    ) -> list[_TextSegment]:
        if not words:
            return []

        ordered = sorted(
            words,
            key=lambda word: word.x0,
        )

        char_widths = [
            word.width / max(len(word.text), 1)
            for word in ordered
            if word.width > 0
        ]
        typical_char_width = (
            median(char_widths)
            if char_widths
            else 4.0
        )
        gap_threshold = max(
            self.minimum_segment_gap,
            typical_char_width * self.segment_gap_multiplier,
        )

        groups: list[list[_WordToken]] = [
            [ordered[0]]
        ]

        for previous, current in zip(
            ordered,
            ordered[1:],
            strict=False,
        ):
            gap = current.x0 - previous.x1

            if gap >= gap_threshold:
                groups.append([current])
            else:
                groups[-1].append(current)

        segments: list[_TextSegment] = []

        for group in groups:
            text = self._normalize_text(
                " ".join(word.text for word in group)
            )
            if not text:
                continue

            box = TableBoundingBox(
                x0=round(min(word.x0 for word in group), 3),
                y0=round(min(word.y0 for word in group), 3),
                x1=round(max(word.x1 for word in group), 3),
                y1=round(max(word.y1 for word in group), 3),
            )

            segments.append(
                _TextSegment(
                    text=text,
                    words=list(group),
                    box=box,
                    is_numeric=self._looks_numeric(text),
                )
            )

        return segments

    def _infer_column_anchors(
        self,
        rows: Sequence[Sequence[_TextSegment]],
    ) -> list[float]:
        observations: list[float] = []

        for row in rows:
            if len(row) < 2:
                continue
            observations.extend(
                segment.x0
                for segment in row
            )

        if not observations:
            return []

        clusters: list[list[float]] = []

        for value in sorted(observations):
            if not clusters:
                clusters.append([value])
                continue

            cluster_center = median(clusters[-1])

            if abs(value - cluster_center) <= self.column_cluster_tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])

        supported = [
            cluster
            for cluster in clusters
            if len(cluster) >= self.minimum_column_support
        ]

        if len(supported) < self.minimum_columns:
            supported = clusters

        anchors = sorted(
            median(cluster)
            for cluster in supported
        )

        return self._deduplicate_anchors(anchors)

    def _fallback_two_column_anchors(
        self,
        rows: Sequence[Sequence[_TextSegment]],
    ) -> list[float]:
        candidate_rows = [
            row
            for row in rows
            if len(row) >= 2
        ]

        if not candidate_rows:
            return []

        left_values = [
            row[0].x0
            for row in candidate_rows
        ]

        right_values = [
            row[-1].x0
            for row in candidate_rows
            if row[-1].is_numeric
            or len(row) == 2
        ]

        if not right_values:
            right_values = [
                row[-1].x0
                for row in candidate_rows
            ]

        left_anchor = median(left_values)
        right_anchor = median(right_values)

        if abs(right_anchor - left_anchor) < self.column_cluster_tolerance:
            return []

        return [left_anchor, right_anchor]

    def _build_matrix(
        self,
        *,
        segments_by_row: Sequence[Sequence[_TextSegment]],
        anchors: Sequence[float],
    ) -> tuple[
        list[list[str]],
        list[list[TableBoundingBox | None]],
        list[float],
    ]:
        if not anchors:
            return [], [], []

        matrix: list[list[str]] = []
        cell_boxes: list[list[TableBoundingBox | None]] = []
        placement_scores: list[float] = []

        for segments in segments_by_row:
            row_values = ["" for _ in anchors]
            row_boxes: list[TableBoundingBox | None] = [
                None for _ in anchors
            ]

            for segment in segments:
                column_index, score = self._nearest_anchor(
                    segment=segment,
                    anchors=anchors,
                )

                existing = row_values[column_index]

                if existing:
                    row_values[column_index] = self._normalize_text(
                        f"{existing} {segment.text}"
                    )
                    row_boxes[column_index] = self._union_optional_boxes(
                        row_boxes[column_index],
                        segment.box,
                    )
                else:
                    row_values[column_index] = segment.text
                    row_boxes[column_index] = segment.box

                placement_scores.append(score)

            matrix.append(row_values)
            cell_boxes.append(row_boxes)

        return matrix, cell_boxes, placement_scores

    def _nearest_anchor(
        self,
        *,
        segment: _TextSegment,
        anchors: Sequence[float],
    ) -> tuple[int, float]:
        distances = [
            abs(segment.x0 - anchor)
            for anchor in anchors
        ]
        index = min(
            range(len(anchors)),
            key=lambda position: distances[position],
        )
        distance = distances[index]
        score = max(
            0.0,
            1.0 - distance / max(
                self.column_cluster_tolerance * 2.0,
                1.0,
            ),
        )
        return index, score

    def _remove_empty_rows(
        self,
        matrix: list[list[str]],
        boxes: list[list[TableBoundingBox | None]],
        placement_scores: list[float],
    ) -> tuple[
        list[list[str]],
        list[list[TableBoundingBox | None]],
        list[float],
    ]:
        kept_matrix: list[list[str]] = []
        kept_boxes: list[list[TableBoundingBox | None]] = []

        for row, row_boxes in zip(
            matrix,
            boxes,
            strict=False,
        ):
            if any(self._normalize_text(value) for value in row):
                kept_matrix.append(row)
                kept_boxes.append(row_boxes)

        return kept_matrix, kept_boxes, placement_scores

    def _remove_empty_columns(
        self,
        matrix: list[list[str]],
        boxes: list[list[TableBoundingBox | None]],
        anchors: list[float],
    ) -> tuple[
        list[list[str]],
        list[list[TableBoundingBox | None]],
        list[float],
    ]:
        if not matrix:
            return matrix, boxes, anchors

        column_count = max(
            len(row)
            for row in matrix
        )

        keep_indices = [
            column_index
            for column_index in range(column_count)
            if any(
                column_index < len(row)
                and self._normalize_text(row[column_index])
                for row in matrix
            )
        ]

        return (
            [
                [
                    row[index] if index < len(row) else ""
                    for index in keep_indices
                ]
                for row in matrix
            ],
            [
                [
                    row_boxes[index]
                    if index < len(row_boxes)
                    else None
                    for index in keep_indices
                ]
                for row_boxes in boxes
            ],
            [
                anchors[index]
                for index in keep_indices
                if index < len(anchors)
            ],
        )

    def _infer_header_row(
        self,
        matrix: Sequence[Sequence[str]],
    ) -> int | None:
        if not matrix:
            return None

        first_row = list(matrix[0])
        first_non_empty = [
            value
            for value in first_row
            if self._normalize_text(value)
        ]

        if not first_non_empty:
            return None

        year_count = sum(
            bool(self._YEAR_RE.fullmatch(self._normalize_text(value)))
            for value in first_non_empty
        )
        numeric_count = sum(
            self._looks_numeric(value)
            for value in first_non_empty
        )

        if year_count >= 1:
            return 0

        if numeric_count == 0 and len(matrix) >= 2:
            second_numeric_count = sum(
                self._looks_numeric(value)
                for value in matrix[1]
            )
            if second_numeric_count >= 1:
                return 0

        return None

    def _build_headers(
        self,
        *,
        matrix: Sequence[Sequence[str]],
        header_row_index: int | None,
    ) -> list[str]:
        column_count = max(
            (len(row) for row in matrix),
            default=0,
        )

        headers: list[str] = []

        for column_index in range(column_count):
            if (
                header_row_index is not None
                and header_row_index < len(matrix)
                and column_index < len(matrix[header_row_index])
            ):
                header = self._normalize_text(
                    matrix[header_row_index][column_index]
                )
            else:
                header = ""

            headers.append(
                header or f"column_{column_index + 1}"
            )

        return self._make_headers_unique(headers)

    def _build_rows(
        self,
        *,
        matrix: Sequence[Sequence[str]],
        cell_boxes: Sequence[
            Sequence[TableBoundingBox | None]
        ],
        placement_scores: Sequence[float],
        header_row_index: int | None,
    ) -> list[BorderlessTableRow]:
        rows: list[BorderlessTableRow] = []
        score_cursor = 0

        for row_index, row_values in enumerate(matrix):
            cells: list[BorderlessTableCell] = []
            cell_scores: list[float] = []

            for column_index, value in enumerate(row_values):
                normalized = self._normalize_text(value)
                box = (
                    cell_boxes[row_index][column_index]
                    if row_index < len(cell_boxes)
                    and column_index < len(cell_boxes[row_index])
                    else None
                )

                score = (
                    placement_scores[score_cursor]
                    if score_cursor < len(placement_scores)
                    else 0.75
                )
                score_cursor += 1
                cell_scores.append(score)

                cells.append(
                    BorderlessTableCell(
                        row_index=row_index,
                        column_index=column_index,
                        text=value,
                        normalized_text=normalized,
                        bounding_box=box,
                        is_header=(
                            header_row_index == row_index
                        ),
                        is_empty=not bool(normalized),
                        is_numeric=self._looks_numeric(normalized),
                        confidence=round(score, 4),
                        metadata={
                            "source": "pdf_word_coordinates",
                        },
                    )
                )

            row_confidence = (
                sum(cell_scores) / len(cell_scores)
                if cell_scores
                else 0.0
            )

            rows.append(
                BorderlessTableRow(
                    row_index=row_index,
                    cells=tuple(cells),
                    is_header=(
                        header_row_index == row_index
                    ),
                    is_empty=not any(
                        self._normalize_text(value)
                        for value in row_values
                    ),
                    text=" | ".join(row_values),
                    confidence=round(row_confidence, 4),
                    metadata={
                        "non_empty_cell_count": sum(
                            bool(self._normalize_text(value))
                            for value in row_values
                        ),
                    },
                )
            )

        return rows

    def _effective_region(
        self,
        *,
        detected_table: DetectedTable,
        page_rect: pymupdf.Rect,
    ) -> TableBoundingBox:
        box = detected_table.bounding_box

        if not (
            self.exclude_caption_from_region
            and detected_table.caption_text
        ):
            return box

        # Caption koordinatları DetectedTable sözleşmesinde tutulmadığı için
        # burada yalnızca caption-anchor metadata sınırı varsa kullanılır.
        metadata = dict(detected_table.metadata or {})
        caption_box_value = metadata.get("caption_box")

        if isinstance(caption_box_value, Mapping):
            try:
                caption_box = TableBoundingBox(
                    x0=float(caption_box_value["x0"]),
                    y0=float(caption_box_value["y0"]),
                    x1=float(caption_box_value["x1"]),
                    y1=float(caption_box_value["y1"]),
                )
            except (KeyError, TypeError, ValueError):
                caption_box = None
        else:
            caption_box = None

        if caption_box is None:
            return box

        if caption_box.y1 <= box.center_y:
            y0 = min(
                max(caption_box.y1 + 2.0, box.y0),
                box.y1,
            )
            if box.y1 - y0 >= 12.0:
                return TableBoundingBox(
                    x0=box.x0,
                    y0=round(y0, 3),
                    x1=box.x1,
                    y1=box.y1,
                )

        return box

    def _empty_result(
        self,
        *,
        detected_table: DetectedTable,
        region: TableBoundingBox,
        warnings: Sequence[str],
    ) -> BorderlessExtractedTable:
        empty_hash = hashlib.sha256(b"").hexdigest()

        return BorderlessExtractedTable(
            table_id=detected_table.table_id,
            page_index=detected_table.page_index,
            page_number=detected_table.page_number,
            table_index=detected_table.table_index,
            bounding_box=region,
            extraction_method="pdf_word_coordinates",
            extraction_confidence=0.0,
            row_count=0,
            column_count=0,
            headers=(),
            rows=(),
            matrix=(),
            caption_text=detected_table.caption_text,
            title_text=detected_table.caption_text,
            content_hash=empty_hash,
            is_empty=True,
            is_valid=False,
            warnings=tuple(warnings),
            metadata={
                **dict(detected_table.metadata or {}),
                "source_detection_method": (
                    detected_table.detection_method
                ),
                "requires_visual_fallback": True,
            },
        )

    def _calculate_confidence(
        self,
        *,
        detection_confidence: float,
        row_count: int,
        column_count: int,
        empty_cell_ratio: float,
        numeric_row_ratio: float,
        repeated_structure_ratio: float,
        average_placement_score: float,
        is_valid: bool,
    ) -> float:
        confidence = 0.18
        confidence += min(max(detection_confidence, 0.0), 1.0) * 0.22
        confidence += min(row_count / 12.0, 1.0) * 0.12
        confidence += min(column_count / 6.0, 1.0) * 0.10
        confidence += max(0.0, 1.0 - empty_cell_ratio) * 0.12
        confidence += numeric_row_ratio * 0.10
        confidence += repeated_structure_ratio * 0.12
        confidence += average_placement_score * 0.10

        if is_valid:
            confidence += 0.04

        return round(
            min(max(confidence, 0.0), 0.99),
            4,
        )

    def _repeated_structure_ratio(
        self,
        matrix: Sequence[Sequence[str]],
    ) -> float:
        if not matrix:
            return 0.0

        signatures = [
            tuple(
                index
                for index, value in enumerate(row)
                if self._normalize_text(value)
            )
            for row in matrix
        ]

        if not signatures:
            return 0.0

        counts: dict[tuple[int, ...], int] = {}

        for signature in signatures:
            counts[signature] = counts.get(signature, 0) + 1

        strongest = max(counts.values(), default=0)

        return strongest / max(len(signatures), 1)

    def _reduce_anchors(
        self,
        anchors: Sequence[float],
        *,
        maximum: int,
    ) -> list[float]:
        reduced = list(anchors)

        while len(reduced) > maximum:
            gaps = [
                reduced[index + 1] - reduced[index]
                for index in range(len(reduced) - 1)
            ]
            merge_index = min(
                range(len(gaps)),
                key=lambda index: gaps[index],
            )
            merged = (
                reduced[merge_index]
                + reduced[merge_index + 1]
            ) / 2.0
            reduced = (
                reduced[:merge_index]
                + [merged]
                + reduced[merge_index + 2:]
            )

        return reduced

    def _deduplicate_anchors(
        self,
        anchors: Sequence[float],
    ) -> list[float]:
        result: list[float] = []

        for anchor in sorted(anchors):
            if (
                result
                and abs(anchor - result[-1])
                <= self.column_cluster_tolerance * 0.5
            ):
                result[-1] = (
                    result[-1] + anchor
                ) / 2.0
            else:
                result.append(anchor)

        return result

    @classmethod
    def _normalize_text(
        cls,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        text = str(value).replace("\u00ad", "")
        text = text.replace("\n", " ")
        text = cls._WHITESPACE_RE.sub(" ", text)

        return text.strip()

    @classmethod
    def _looks_numeric(
        cls,
        value: Any,
    ) -> bool:
        text = cls._normalize_text(value)

        if not text:
            return False

        if cls._YEAR_RE.fullmatch(text):
            return True

        if cls._NUMERIC_RE.fullmatch(text):
            return True

        # Birim ayrı yazılmış veya dipnot işareti eklenmiş olabilir.
        normalized = re.sub(
            r"(?<=\d)[¹²³⁴⁵⁶⁷⁸⁹*]+$",
            "",
            text,
        )

        return bool(cls._NUMERIC_RE.fullmatch(normalized))

    @classmethod
    def _make_headers_unique(
        cls,
        headers: Sequence[str],
    ) -> list[str]:
        result: list[str] = []
        counts: dict[str, int] = {}

        for index, value in enumerate(headers):
            header = cls._normalize_text(value)
            header = header or f"column_{index + 1}"

            key = header.casefold()
            counts[key] = counts.get(key, 0) + 1

            if counts[key] > 1:
                header = f"{header}_{counts[key]}"

            result.append(header)

        return result

    @classmethod
    def _calculate_content_hash(
        cls,
        matrix: Sequence[Sequence[str]],
    ) -> str:
        canonical = "\n".join(
            "\t".join(
                cls._normalize_text(cell).casefold()
                for cell in row
            )
            for row in matrix
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _union_optional_boxes(
        first: TableBoundingBox | None,
        second: TableBoundingBox,
    ) -> TableBoundingBox:
        if first is None:
            return second

        return TableBoundingBox(
            x0=min(first.x0, second.x0),
            y0=min(first.y0, second.y0),
            x1=max(first.x1, second.x1),
            y1=max(first.y1, second.y1),
        )

    @staticmethod
    def _normalize_pages(
        *,
        page_numbers: Iterable[int] | None,
        page_count: int,
    ) -> list[int]:
        if page_numbers is None:
            return list(range(1, page_count + 1))

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

        invalid = [
            page_number
            for page_number in pages
            if not 1 <= page_number <= page_count
        ]

        if invalid:
            raise ValueError(
                f"Geçersiz sayfalar: {invalid}. "
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


def extract_borderless_tables(
    pdf_path: str | Path,
    *,
    detected_tables_by_page: Mapping[
        int,
        Sequence[DetectedTable],
    ],
    page_numbers: Iterable[int] | None = None,
) -> list[PageBorderlessTableExtraction]:
    """BorderlessTableExtractor için kısa kullanım arayüzü."""

    extractor = BorderlessTableExtractor()

    return extractor.extract_pdf(
        pdf_path=pdf_path,
        detected_tables_by_page=detected_tables_by_page,
        page_numbers=page_numbers,
    )