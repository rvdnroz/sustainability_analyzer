from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)


class TableNormalizationError(RuntimeError):
    """Tablo normalizasyonu tamamlanamadığında yükseltilir."""


class NormalizedValueType(StrEnum):
    """Normalize edilmiş hücrenin veri türü."""

    EMPTY = "empty"
    TEXT = "text"
    INTEGER = "integer"
    DECIMAL = "decimal"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    YEAR = "year"
    BOOLEAN = "boolean"
    RANGE = "range"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class ParsedNumericValue:
    """Metinsel bir hücreden çıkarılan sayısal değer."""

    raw_text: str
    normalized_text: str

    value: float | None
    value_type: NormalizedValueType

    unit: str | None
    currency: str | None

    lower_bound: float | None = None
    upper_bound: float | None = None

    is_percentage: bool = False
    is_negative: bool = False
    is_approximate: bool = False
    is_missing_value: bool = False

    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["value_type"] = self.value_type.value
        return data


@dataclass(slots=True, frozen=True)
class NormalizedTableCell:
    """Normalize edilmiş tek bir tablo hücresi."""

    row_index: int
    column_index: int

    raw_text: str
    normalized_text: str
    search_text: str

    value_type: NormalizedValueType

    numeric_value: float | None
    lower_bound: float | None
    upper_bound: float | None

    unit: str | None
    currency: str | None
    year: int | None

    is_header: bool
    is_empty: bool
    is_numeric: bool
    is_percentage: bool
    is_missing_value: bool

    confidence: float

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["value_type"] = self.value_type.value
        return data


@dataclass(slots=True, frozen=True)
class NormalizedTableRow:
    """Normalize edilmiş tablo satırı."""

    row_index: int
    cells: tuple[NormalizedTableCell, ...]

    row_label: str | None
    metric_name: str | None
    metric_slug: str | None

    detected_years: tuple[int, ...]
    detected_units: tuple[str, ...]

    numeric_value_count: int
    empty_cell_count: int

    is_header: bool
    is_empty: bool

    searchable_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [
                cell.to_dict()
                for cell in self.cells
            ],
            "row_label": self.row_label,
            "metric_name": self.metric_name,
            "metric_slug": self.metric_slug,
            "detected_years": list(self.detected_years),
            "detected_units": list(self.detected_units),
            "numeric_value_count": self.numeric_value_count,
            "empty_cell_count": self.empty_cell_count,
            "is_header": self.is_header,
            "is_empty": self.is_empty,
            "searchable_text": self.searchable_text,
        }


@dataclass(slots=True, frozen=True)
class NormalizedMetricRecord:
    """
    Tablo içinden retrieval ve veri tabanı için çıkarılmış tek metrik kaydı.

    Örnek:

    metric_name:
        Scope 1 greenhouse gas emissions

    year:
        2023

    numeric_value:
        125340.0

    unit:
        tCO2e
    """

    record_id: str
    table_id: str

    page_number: int
    table_index: int

    row_index: int
    column_index: int

    metric_name: str
    metric_slug: str

    raw_value: str
    normalized_value: str

    numeric_value: float | None
    value_type: NormalizedValueType

    year: int | None
    unit: str | None
    currency: str | None

    is_percentage: bool
    confidence: float

    context: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["value_type"] = self.value_type.value
        return data


@dataclass(slots=True, frozen=True)
class NormalizedTable:
    """Farklı extraction yollarından gelen ortak tablo yapısı."""

    table_id: str

    page_index: int
    page_number: int
    table_index: int

    source_extraction_method: str

    title: str | None
    caption: str | None

    row_count: int
    column_count: int

    headers: tuple[str, ...]
    normalized_headers: tuple[str, ...]

    rows: tuple[NormalizedTableRow, ...]
    matrix: tuple[tuple[str, ...], ...]

    metric_records: tuple[NormalizedMetricRecord, ...]

    detected_years: tuple[int, ...]
    detected_units: tuple[str, ...]
    detected_currencies: tuple[str, ...]

    default_unit: str | None
    default_year: int | None

    numeric_cell_count: int
    empty_cell_count: int

    normalization_confidence: float
    source_confidence: float

    content_hash: str

    is_valid: bool
    warnings: tuple[str, ...]

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_index": self.page_index,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "source_extraction_method": (
                self.source_extraction_method
            ),
            "title": self.title,
            "caption": self.caption,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "headers": list(self.headers),
            "normalized_headers": list(
                self.normalized_headers
            ),
            "rows": [
                row.to_dict()
                for row in self.rows
            ],
            "matrix": [
                list(row)
                for row in self.matrix
            ],
            "metric_records": [
                record.to_dict()
                for record in self.metric_records
            ],
            "detected_years": list(
                self.detected_years
            ),
            "detected_units": list(
                self.detected_units
            ),
            "detected_currencies": list(
                self.detected_currencies
            ),
            "default_unit": self.default_unit,
            "default_year": self.default_year,
            "numeric_cell_count": (
                self.numeric_cell_count
            ),
            "empty_cell_count": self.empty_cell_count,
            "normalization_confidence": (
                self.normalization_confidence
            ),
            "source_confidence": self.source_confidence,
            "content_hash": self.content_hash,
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


class TableNormalizer:
    """
    Native, OCR ve VLM tablo sonuçlarını ortak yapıya dönüştürür.

    Temel görevleri:

    - Hücre metinlerini temizlemek
    - Ondalık ve binlik ayraçlarını yorumlamak
    - Yüzde ve para değerlerini belirlemek
    - Birimleri standardize etmek
    - Yılları tespit etmek
    - Başlıkları benzersiz ve aranabilir hâle getirmek
    - Satır bazında metrik-değer-yıl-birim kayıtları üretmek
    - Farklı extraction yollarını tek NormalizedTable tipinde birleştirmek

    Girdi nesnesinin aşağıdaki alanları taşıması beklenir:

    - table_id
    - page_index
    - page_number
    - table_index
    - matrix
    - headers
    - extraction_method
    """

    _WHITESPACE_RE = re.compile(r"\s+")
    _MULTIPLE_SEPARATOR_RE = re.compile(r"[-_/|]{2,}")

    _YEAR_RE = re.compile(
        r"(?<!\d)"
        r"(?:19|20)\d{2}"
        r"(?!\d)"
    )

    _PERCENTAGE_RE = re.compile(
        r"^\s*"
        r"[~≈<>≤≥]?\s*"
        r"[+-]?"
        r"[\d\s.,]+"
        r"\s*%"
        r"\s*$"
    )

    _RANGE_RE = re.compile(
        r"^\s*"
        r"(?P<lower>[+-]?[\d\s.,]+)"
        r"\s*(?:-|–|—|to|ile)\s*"
        r"(?P<upper>[+-]?[\d\s.,]+)"
        r"\s*(?P<unit>[^\d\s]+)?"
        r"\s*$",
        re.IGNORECASE,
    )

    _APPROXIMATE_RE = re.compile(
        r"^\s*(?:~|≈|yaklaşık|approx\.?|approximately)\s*",
        re.IGNORECASE,
    )

    _COMPARISON_PREFIX_RE = re.compile(
        r"^\s*(?:<|>|≤|≥|en az|en fazla|at least|up to)\s*",
        re.IGNORECASE,
    )

    _CURRENCY_SYMBOLS = {
        "₺": "TRY",
        "tl": "TRY",
        "try": "TRY",
        "$": "USD",
        "usd": "USD",
        "us$": "USD",
        "€": "EUR",
        "eur": "EUR",
        "£": "GBP",
        "gbp": "GBP",
    }

    _MISSING_VALUES = {
        "",
        "-",
        "–",
        "—",
        "n/a",
        "na",
        "n.a.",
        "n.a",
        "not available",
        "not applicable",
        "veri yok",
        "uygulanamaz",
        "mevcut değil",
        "belirtilmemiş",
        "nil",
        "none",
        "null",
    }

    _BOOLEAN_TRUE = {
        "yes",
        "evet",
        "true",
        "var",
        "uygulanıyor",
        "mevcut",
    }

    _BOOLEAN_FALSE = {
        "no",
        "hayır",
        "false",
        "yok",
        "uygulanmıyor",
        "mevcut değil",
    }

    _UNIT_PATTERNS: tuple[
        tuple[re.Pattern[str], str],
        ...
    ] = (
        (
            re.compile(
                r"\b(?:t\s*co2e|tco2e|tonnes?\s+co2e|"
                r"metric\s+tons?\s+co2e|ton\s+co2e)\b",
                re.IGNORECASE,
            ),
            "tCO2e",
        ),
        (
            re.compile(
                r"\b(?:kg\s*co2e|kgco2e)\b",
                re.IGNORECASE,
            ),
            "kgCO2e",
        ),
        (
            re.compile(
                r"\b(?:co2e|co₂e)\b",
                re.IGNORECASE,
            ),
            "CO2e",
        ),
        (
            re.compile(
                r"\b(?:mwh)\b",
                re.IGNORECASE,
            ),
            "MWh",
        ),
        (
            re.compile(
                r"\b(?:kwh)\b",
                re.IGNORECASE,
            ),
            "kWh",
        ),
        (
            re.compile(
                r"\b(?:gwh)\b",
                re.IGNORECASE,
            ),
            "GWh",
        ),
        (
            re.compile(
                r"\b(?:gj)\b",
                re.IGNORECASE,
            ),
            "GJ",
        ),
        (
            re.compile(
                r"\b(?:mj)\b",
                re.IGNORECASE,
            ),
            "MJ",
        ),
        (
            re.compile(
                r"\b(?:m3|m³|cubic\s+met(?:er|re)s?)\b",
                re.IGNORECASE,
            ),
            "m³",
        ),
        (
            re.compile(
                r"\b(?:lit(?:er|re)s?|lt|l)\b",
                re.IGNORECASE,
            ),
            "L",
        ),
        (
            re.compile(
                r"\b(?:tonnes?|tons?|ton|metric\s+tons?|t)\b",
                re.IGNORECASE,
            ),
            "tonne",
        ),
        (
            re.compile(
                r"\b(?:kilograms?|kg)\b",
                re.IGNORECASE,
            ),
            "kg",
        ),
        (
            re.compile(
                r"\b(?:grams?|gr|g)\b",
                re.IGNORECASE,
            ),
            "g",
        ),
        (
            re.compile(
                r"\b(?:hectares?|ha)\b",
                re.IGNORECASE,
            ),
            "ha",
        ),
        (
            re.compile(
                r"\b(?:kilomet(?:er|re)s?|km)\b",
                re.IGNORECASE,
            ),
            "km",
        ),
        (
            re.compile(
                r"\b(?:hours?|saat)\b",
                re.IGNORECASE,
            ),
            "hour",
        ),
        (
            re.compile(
                r"\b(?:days?|gün)\b",
                re.IGNORECASE,
            ),
            "day",
        ),
        (
            re.compile(
                r"\b(?:employees?|çalışan(?:lar)?)\b",
                re.IGNORECASE,
            ),
            "employee",
        ),
        (
            re.compile(
                r"\b(?:people|persons?|kişi)\b",
                re.IGNORECASE,
            ),
            "person",
        ),
        (
            re.compile(
                r"\b(?:incidents?|vakalar?|olaylar?)\b",
                re.IGNORECASE,
            ),
            "incident",
        ),
    )

    def __init__(
        self,
        *,
        minimum_rows: int = 2,
        minimum_columns: int = 2,
        minimum_numeric_confidence: float = 0.50,
        infer_metric_records: bool = True,
        header_row_count: int | None = None,
        default_report_year: int | None = None,
        preserve_raw_matrix: bool = True,
    ) -> None:
        if minimum_rows < 1:
            raise ValueError(
                "minimum_rows en az 1 olmalıdır."
            )

        if minimum_columns < 1:
            raise ValueError(
                "minimum_columns en az 1 olmalıdır."
            )

        if not 0.0 <= minimum_numeric_confidence <= 1.0:
            raise ValueError(
                "minimum_numeric_confidence "
                "0 ile 1 arasında olmalıdır."
            )

        if (
            header_row_count is not None
            and header_row_count < 0
        ):
            raise ValueError(
                "header_row_count negatif olamaz."
            )

        if (
            default_report_year is not None
            and not 1900
            <= default_report_year
            <= 2100
        ):
            raise ValueError(
                "default_report_year geçersiz."
            )

        self.minimum_rows = minimum_rows
        self.minimum_columns = minimum_columns

        self.minimum_numeric_confidence = (
            minimum_numeric_confidence
        )

        self.infer_metric_records = (
            infer_metric_records
        )

        self.header_row_count = header_row_count
        self.default_report_year = (
            default_report_year
        )

        self.preserve_raw_matrix = (
            preserve_raw_matrix
        )

    def normalize_table(
        self,
        table: Any,
    ) -> NormalizedTable:
        """Tek bir extraction sonucunu normalize eder."""

        table_id = str(
            getattr(
                table,
                "table_id",
                "",
            )
        ).strip()

        if not table_id:
            raise TableNormalizationError(
                "Tablo nesnesinde table_id bulunmuyor."
            )

        raw_matrix = getattr(
            table,
            "matrix",
            None,
        )

        if raw_matrix is None:
            raise TableNormalizationError(
                f"Tablo matrisi bulunmuyor: {table_id}"
            )

        matrix = self._normalize_matrix_shape(
            raw_matrix
        )

        row_count = len(matrix)

        column_count = max(
            (
                len(row)
                for row in matrix
            ),
            default=0,
        )

        source_headers = list(
            getattr(
                table,
                "headers",
                (),
            )
            or ()
        )

        header_indices = self._infer_header_indices(
            matrix=matrix,
            supplied_headers=source_headers,
        )

        headers = self._build_headers(
            matrix=matrix,
            supplied_headers=source_headers,
            header_indices=header_indices,
            column_count=column_count,
        )

        normalized_headers = self._normalize_headers(
            headers
        )

        table_context = " ".join(
            filter(
                None,
                (
                    str(
                        getattr(
                            table,
                            "title",
                            "",
                        )
                        or ""
                    ),
                    str(
                        getattr(
                            table,
                            "title_text",
                            "",
                        )
                        or ""
                    ),
                    str(
                        getattr(
                            table,
                            "caption",
                            "",
                        )
                        or ""
                    ),
                    str(
                        getattr(
                            table,
                            "caption_text",
                            "",
                        )
                        or ""
                    ),
                    " ".join(headers),
                ),
            )
        )

        context_years = self.extract_years(
            table_context
        )

        context_units = self.extract_units(
            table_context
        )

        context_currencies = (
            self.extract_currencies(
                table_context
            )
        )

        default_year = self._select_default_year(
            context_years
        )

        default_unit = (
            context_units[0]
            if len(context_units) == 1
            else None
        )

        normalized_rows: list[
            NormalizedTableRow
        ] = []

        all_cells: list[
            NormalizedTableCell
        ] = []

        for row_index, row in enumerate(matrix):
            row_cells = self._normalize_row_cells(
                row=row,
                row_index=row_index,
                headers=headers,
                header_indices=header_indices,
                default_unit=default_unit,
                default_year=default_year,
                source_table=table,
            )

            all_cells.extend(row_cells)

            normalized_rows.append(
                self._build_normalized_row(
                    row_index=row_index,
                    cells=row_cells,
                    is_header=(
                        row_index in header_indices
                    ),
                )
            )

        detected_years = sorted(
            set(
                context_years
                + [
                    cell.year
                    for cell in all_cells
                    if cell.year is not None
                ]
            )
        )

        detected_units = sorted(
            set(
                context_units
                + [
                    cell.unit
                    for cell in all_cells
                    if cell.unit
                ]
            )
        )

        detected_currencies = sorted(
            set(
                context_currencies
                + [
                    cell.currency
                    for cell in all_cells
                    if cell.currency
                ]
            )
        )

        if default_year is None:
            default_year = (
                self._select_default_year(
                    detected_years
                )
            )

        if default_unit is None and len(
            detected_units
        ) == 1:
            default_unit = detected_units[0]

        metric_records: list[
            NormalizedMetricRecord
        ] = []

        if self.infer_metric_records:
            metric_records = (
                self._build_metric_records(
                    table_id=table_id,
                    page_number=int(
                        getattr(
                            table,
                            "page_number",
                            0,
                        )
                    ),
                    table_index=int(
                        getattr(
                            table,
                            "table_index",
                            0,
                        )
                    ),
                    rows=normalized_rows,
                    headers=headers,
                    default_year=default_year,
                    default_unit=default_unit,
                )
            )

        source_confidence = (
            self._extract_source_confidence(
                table
            )
        )

        numeric_cell_count = sum(
            cell.is_numeric
            for cell in all_cells
        )

        empty_cell_count = sum(
            cell.is_empty
            for cell in all_cells
        )

        warnings: list[str] = []

        if row_count < self.minimum_rows:
            warnings.append(
                "Tablo satır sayısı minimum eşikten düşük: "
                f"{row_count}."
            )

        if column_count < self.minimum_columns:
            warnings.append(
                "Tablo sütun sayısı minimum eşikten düşük: "
                f"{column_count}."
            )

        if not metric_records:
            warnings.append(
                "Tablodan yapılandırılmış metrik kaydı üretilemedi."
            )

        normalization_confidence = (
            self._calculate_normalization_confidence(
                row_count=row_count,
                column_count=column_count,
                cells=all_cells,
                source_confidence=source_confidence,
                metric_record_count=len(
                    metric_records
                ),
            )
        )

        is_valid = (
            row_count >= self.minimum_rows
            and column_count >= self.minimum_columns
            and any(
                not row.is_empty
                for row in normalized_rows
            )
        )

        title = (
            getattr(
                table,
                "title",
                None,
            )
            or getattr(
                table,
                "title_text",
                None,
            )
        )

        caption = (
            getattr(
                table,
                "caption",
                None,
            )
            or getattr(
                table,
                "caption_text",
                None,
            )
        )

        normalized_matrix = tuple(
            tuple(
                cell.normalized_text
                for cell in row.cells
            )
            for row in normalized_rows
        )

        return NormalizedTable(
            table_id=table_id,
            page_index=int(
                getattr(
                    table,
                    "page_index",
                    0,
                )
            ),
            page_number=int(
                getattr(
                    table,
                    "page_number",
                    0,
                )
            ),
            table_index=int(
                getattr(
                    table,
                    "table_index",
                    0,
                )
            ),
            source_extraction_method=str(
                getattr(
                    table,
                    "extraction_method",
                    table.__class__.__name__,
                )
            ),
            title=(
                self.normalize_text(title)
                if title
                else None
            ),
            caption=(
                self.normalize_text(caption)
                if caption
                else None
            ),
            row_count=row_count,
            column_count=column_count,
            headers=tuple(headers),
            normalized_headers=tuple(
                normalized_headers
            ),
            rows=tuple(normalized_rows),
            matrix=normalized_matrix,
            metric_records=tuple(
                metric_records
            ),
            detected_years=tuple(
                detected_years
            ),
            detected_units=tuple(
                detected_units
            ),
            detected_currencies=tuple(
                detected_currencies
            ),
            default_unit=default_unit,
            default_year=default_year,
            numeric_cell_count=(
                numeric_cell_count
            ),
            empty_cell_count=empty_cell_count,
            normalization_confidence=(
                normalization_confidence
            ),
            source_confidence=source_confidence,
            content_hash=self._calculate_hash(
                normalized_matrix
            ),
            is_valid=is_valid,
            warnings=tuple(warnings),
            metadata={
                "header_row_indices": (
                    header_indices
                ),
                "original_row_count": len(
                    raw_matrix
                ),
                "preserve_raw_matrix": (
                    self.preserve_raw_matrix
                ),
                "source_content_hash": getattr(
                    table,
                    "content_hash",
                    None,
                ),
            },
        )

    def normalize_tables(
        self,
        tables: Sequence[Any],
    ) -> list[NormalizedTable]:
        """Birden fazla tabloyu normalize eder."""

        results: list[NormalizedTable] = []

        for table in tables:
            try:
                results.append(
                    self.normalize_table(table)
                )

            except Exception as exc:
                logger.warning(
                    "Tablo normalizasyonu başarısız: %s",
                    exc,
                )

        return results

    def _normalize_row_cells(
        self,
        *,
        row: Sequence[str],
        row_index: int,
        headers: Sequence[str],
        header_indices: Sequence[int],
        default_unit: str | None,
        default_year: int | None,
        source_table: Any,
    ) -> list[NormalizedTableCell]:
        cells: list[NormalizedTableCell] = []

        for column_index, raw_value in enumerate(
            row
        ):
            raw_text = (
                ""
                if raw_value is None
                else str(raw_value)
            )

            normalized_text = self.normalize_text(
                raw_text
            )

            parsed_value = self.parse_numeric_value(
                normalized_text
            )

            detected_years = self.extract_years(
                normalized_text
            )

            year = (
                detected_years[0]
                if len(detected_years) == 1
                else None
            )

            if year is None:
                header_text = (
                    headers[column_index]
                    if column_index < len(headers)
                    else ""
                )

                header_years = self.extract_years(
                    header_text
                )

                if len(header_years) == 1:
                    year = header_years[0]

            if year is None:
                year = default_year

            unit = (
                parsed_value.unit
                or self._unit_from_header(
                    headers=headers,
                    column_index=column_index,
                )
                or default_unit
            )

            confidence = (
                parsed_value.confidence
            )

            source_cell_confidence = (
                self._get_source_cell_confidence(
                    source_table=source_table,
                    row_index=row_index,
                    column_index=column_index,
                )
            )

            if source_cell_confidence is not None:
                confidence = round(
                    (
                        0.60 * confidence
                        + 0.40
                        * source_cell_confidence
                    ),
                    4,
                )

            is_header = (
                row_index in header_indices
            )

            cells.append(
                NormalizedTableCell(
                    row_index=row_index,
                    column_index=column_index,
                    raw_text=raw_text,
                    normalized_text=(
                        normalized_text
                    ),
                    search_text=(
                        self.normalize_for_search(
                            normalized_text
                        )
                    ),
                    value_type=(
                        parsed_value.value_type
                    ),
                    numeric_value=(
                        parsed_value.value
                    ),
                    lower_bound=(
                        parsed_value.lower_bound
                    ),
                    upper_bound=(
                        parsed_value.upper_bound
                    ),
                    unit=unit,
                    currency=(
                        parsed_value.currency
                    ),
                    year=year,
                    is_header=is_header,
                    is_empty=(
                        parsed_value.value_type
                        == NormalizedValueType.EMPTY
                    ),
                    is_numeric=(
                        parsed_value.value
                        is not None
                        or parsed_value.value_type
                        == NormalizedValueType.RANGE
                    ),
                    is_percentage=(
                        parsed_value.is_percentage
                    ),
                    is_missing_value=(
                        parsed_value.is_missing_value
                    ),
                    confidence=confidence,
                    metadata={
                        "header": (
                            headers[column_index]
                            if column_index
                            < len(headers)
                            else None
                        ),
                        "source_cell_confidence": (
                            source_cell_confidence
                        ),
                    },
                )
            )

        return cells

    def parse_numeric_value(
        self,
        text: str,
    ) -> ParsedNumericValue:
        """Metinsel değeri sayısal veya kategorik tipe dönüştürür."""

        raw_text = text
        normalized_text = self.normalize_text(
            text
        )

        search_text = (
            normalized_text.casefold()
        )

        if search_text in self._MISSING_VALUES:
            return ParsedNumericValue(
                raw_text=raw_text,
                normalized_text=normalized_text,
                value=None,
                value_type=(
                    NormalizedValueType.EMPTY
                ),
                unit=None,
                currency=None,
                is_missing_value=True,
                confidence=1.0,
            )

        if search_text in self._BOOLEAN_TRUE:
            return ParsedNumericValue(
                raw_text=raw_text,
                normalized_text=normalized_text,
                value=1.0,
                value_type=(
                    NormalizedValueType.BOOLEAN
                ),
                unit=None,
                currency=None,
                confidence=0.95,
            )

        if search_text in self._BOOLEAN_FALSE:
            return ParsedNumericValue(
                raw_text=raw_text,
                normalized_text=normalized_text,
                value=0.0,
                value_type=(
                    NormalizedValueType.BOOLEAN
                ),
                unit=None,
                currency=None,
                confidence=0.95,
            )

        year_values = self.extract_years(
            normalized_text
        )

        if (
            len(year_values) == 1
            and normalized_text.strip()
            == str(year_values[0])
        ):
            return ParsedNumericValue(
                raw_text=raw_text,
                normalized_text=normalized_text,
                value=float(
                    year_values[0]
                ),
                value_type=(
                    NormalizedValueType.YEAR
                ),
                unit=None,
                currency=None,
                confidence=1.0,
            )

        is_approximate = bool(
            self._APPROXIMATE_RE.match(
                normalized_text
            )
        )

        cleaned_text = (
            self._APPROXIMATE_RE.sub(
                "",
                normalized_text,
            )
        )

        cleaned_text = (
            self._COMPARISON_PREFIX_RE.sub(
                "",
                cleaned_text,
            )
        )

        range_match = self._RANGE_RE.match(
            cleaned_text
        )

        if range_match:
            lower = self._parse_decimal_string(
                range_match.group("lower")
            )

            upper = self._parse_decimal_string(
                range_match.group("upper")
            )

            unit = self._normalize_unit(
                range_match.group("unit")
                or ""
            )

            if (
                lower is not None
                and upper is not None
            ):
                return ParsedNumericValue(
                    raw_text=raw_text,
                    normalized_text=(
                        normalized_text
                    ),
                    value=(
                        lower + upper
                    )
                    / 2,
                    value_type=(
                        NormalizedValueType.RANGE
                    ),
                    unit=unit,
                    currency=(
                        self._detect_currency(
                            normalized_text
                        )
                    ),
                    lower_bound=lower,
                    upper_bound=upper,
                    is_negative=(
                        lower < 0
                        or upper < 0
                    ),
                    is_approximate=(
                        is_approximate
                    ),
                    confidence=0.85,
                )

        is_percentage = bool(
            self._PERCENTAGE_RE.match(
                normalized_text
            )
        )

        currency = self._detect_currency(
            normalized_text
        )

        unit = self._normalize_unit(
            normalized_text
        )

        numeric_text = (
            self._remove_non_numeric_tokens(
                cleaned_text
            )
        )

        numeric_value = (
            self._parse_decimal_string(
                numeric_text
            )
        )

        if numeric_value is None:
            return ParsedNumericValue(
                raw_text=raw_text,
                normalized_text=normalized_text,
                value=None,
                value_type=(
                    NormalizedValueType.TEXT
                    if normalized_text
                    else NormalizedValueType.EMPTY
                ),
                unit=unit,
                currency=currency,
                is_missing_value=(
                    not normalized_text
                ),
                confidence=(
                    0.95
                    if normalized_text
                    else 1.0
                ),
            )

        if is_percentage:
            value_type = (
                NormalizedValueType.PERCENTAGE
            )

        elif currency:
            value_type = (
                NormalizedValueType.CURRENCY
            )

        elif math.isclose(
            numeric_value,
            round(numeric_value),
            abs_tol=1e-12,
        ):
            value_type = (
                NormalizedValueType.INTEGER
            )

        else:
            value_type = (
                NormalizedValueType.DECIMAL
            )

        confidence = 0.92

        if unit:
            confidence += 0.02

        if currency:
            confidence += 0.02

        return ParsedNumericValue(
            raw_text=raw_text,
            normalized_text=normalized_text,
            value=numeric_value,
            value_type=value_type,
            unit=(
                "%"
                if is_percentage
                else unit
            ),
            currency=currency,
            is_percentage=is_percentage,
            is_negative=numeric_value < 0,
            is_approximate=is_approximate,
            confidence=round(
                min(confidence, 0.99),
                4,
            ),
        )

    def _build_normalized_row(
        self,
        *,
        row_index: int,
        cells: Sequence[NormalizedTableCell],
        is_header: bool,
    ) -> NormalizedTableRow:
        non_empty_cells = [
            cell
            for cell in cells
            if not cell.is_empty
        ]

        row_label = self._detect_row_label(
            cells
        )

        metric_name = (
            self.normalize_text(row_label)
            if row_label
            else None
        )

        metric_slug = (
            self.slugify_metric(
                metric_name
            )
            if metric_name
            else None
        )

        return NormalizedTableRow(
            row_index=row_index,
            cells=tuple(cells),
            row_label=row_label,
            metric_name=metric_name,
            metric_slug=metric_slug,
            detected_years=tuple(
                sorted(
                    {
                        cell.year
                        for cell in cells
                        if cell.year is not None
                    }
                )
            ),
            detected_units=tuple(
                sorted(
                    {
                        cell.unit
                        for cell in cells
                        if cell.unit
                    }
                )
            ),
            numeric_value_count=sum(
                cell.is_numeric
                for cell in cells
            ),
            empty_cell_count=sum(
                cell.is_empty
                for cell in cells
            ),
            is_header=is_header,
            is_empty=not non_empty_cells,
            searchable_text=" | ".join(
                cell.search_text
                for cell in cells
                if cell.search_text
            ),
        )

    def _build_metric_records(
        self,
        *,
        table_id: str,
        page_number: int,
        table_index: int,
        rows: Sequence[NormalizedTableRow],
        headers: Sequence[str],
        default_year: int | None,
        default_unit: str | None,
    ) -> list[NormalizedMetricRecord]:
        records: list[
            NormalizedMetricRecord
        ] = []

        for row in rows:
            if row.is_header or row.is_empty:
                continue

            metric_name = row.metric_name

            if not metric_name:
                continue

            metric_slug = (
                row.metric_slug
                or self.slugify_metric(
                    metric_name
                )
            )

            for cell in row.cells:
                if not cell.is_numeric:
                    continue

                if (
                    cell.value_type
                    == NormalizedValueType.YEAR
                ):
                    continue

                header = (
                    headers[cell.column_index]
                    if cell.column_index
                    < len(headers)
                    else ""
                )

                year = (
                    self._year_from_header(header)
                    or cell.year
                    or default_year
                )

                unit = (
                    cell.unit
                    or self._normalize_unit(
                        header
                    )
                    or default_unit
                )

                context = " | ".join(
                    filter(
                        None,
                        (
                            metric_name,
                            header,
                            (
                                str(year)
                                if year
                                else None
                            ),
                            unit,
                        ),
                    )
                )

                record_key = (
                    f"{table_id}:"
                    f"{row.row_index}:"
                    f"{cell.column_index}:"
                    f"{metric_slug}:"
                    f"{year}:"
                    f"{unit}"
                )

                record_id = hashlib.sha256(
                    record_key.encode(
                        "utf-8"
                    )
                ).hexdigest()

                records.append(
                    NormalizedMetricRecord(
                        record_id=record_id,
                        table_id=table_id,
                        page_number=page_number,
                        table_index=table_index,
                        row_index=row.row_index,
                        column_index=(
                            cell.column_index
                        ),
                        metric_name=metric_name,
                        metric_slug=metric_slug,
                        raw_value=cell.raw_text,
                        normalized_value=(
                            cell.normalized_text
                        ),
                        numeric_value=(
                            cell.numeric_value
                        ),
                        value_type=(
                            cell.value_type
                        ),
                        year=year,
                        unit=unit,
                        currency=cell.currency,
                        is_percentage=(
                            cell.is_percentage
                        ),
                        confidence=(
                            cell.confidence
                        ),
                        context=context,
                        metadata={
                            "header": header,
                            "lower_bound": (
                                cell.lower_bound
                            ),
                            "upper_bound": (
                                cell.upper_bound
                            ),
                        },
                    )
                )

        return records

    def _infer_header_indices(
        self,
        *,
        matrix: Sequence[Sequence[str]],
        supplied_headers: Sequence[str],
    ) -> list[int]:
        if self.header_row_count is not None:
            return list(
                range(
                    min(
                        self.header_row_count,
                        len(matrix),
                    )
                )
            )

        if supplied_headers:
            return []

        if not matrix:
            return []

        first_row = matrix[0]

        non_empty = [
            self.normalize_text(cell)
            for cell in first_row
            if self.normalize_text(cell)
        ]

        if not non_empty:
            return []

        parsed = [
            self.parse_numeric_value(
                cell
            )
            for cell in non_empty
        ]

        numeric_count = sum(
            item.value is not None
            for item in parsed
        )

        text_count = (
            len(non_empty)
            - numeric_count
        )

        if text_count >= numeric_count:
            return [0]

        return []

    def _build_headers(
        self,
        *,
        matrix: Sequence[Sequence[str]],
        supplied_headers: Sequence[str],
        header_indices: Sequence[int],
        column_count: int,
    ) -> list[str]:
        headers: list[str] = []

        if supplied_headers:
            headers = [
                self.normalize_text(header)
                for header in supplied_headers[
                    :column_count
                ]
            ]

        elif header_indices:
            for column_index in range(
                column_count
            ):
                parts = [
                    self.normalize_text(
                        matrix[row_index][
                            column_index
                        ]
                    )
                    for row_index in header_indices
                    if (
                        row_index < len(matrix)
                        and column_index
                        < len(matrix[row_index])
                        and self.normalize_text(
                            matrix[row_index][
                                column_index
                            ]
                        )
                    )
                ]

                headers.append(
                    " | ".join(
                        dict.fromkeys(parts)
                    )
                )

        while len(headers) < column_count:
            headers.append(
                f"column_{len(headers) + 1}"
            )

        headers = [
            header
            or f"column_{index + 1}"
            for index, header in enumerate(
                headers
            )
        ]

        return self._deduplicate_headers(
            headers
        )

    def _normalize_headers(
        self,
        headers: Sequence[str],
    ) -> list[str]:
        return [
            self.slugify_metric(
                header
            )
            or f"column_{index + 1}"
            for index, header in enumerate(
                headers
            )
        ]

    def _normalize_matrix_shape(
        self,
        matrix: Sequence[Sequence[Any]],
    ) -> list[list[str]]:
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
                row_values = [raw_row]
            else:
                try:
                    row_values = list(raw_row)
                except TypeError:
                    row_values = [raw_row]

            normalized_rows.append(
                [
                    ""
                    if value is None
                    else str(value)
                    for value in row_values
                ]
            )

        if not normalized_rows:
            return []

        column_count = max(
            len(row)
            for row in normalized_rows
        )

        return [
            [
                *row,
                *[
                    ""
                    for _ in range(
                        column_count - len(row)
                    )
                ],
            ]
            for row in normalized_rows
        ]

    def _detect_row_label(
        self,
        cells: Sequence[NormalizedTableCell],
    ) -> str | None:
        for cell in cells:
            if cell.is_empty:
                continue

            if cell.is_numeric:
                continue

            if (
                cell.value_type
                == NormalizedValueType.YEAR
            ):
                continue

            return cell.normalized_text

        return None

    def extract_years(
        self,
        text: str,
    ) -> list[int]:
        """Metin içindeki dört basamaklı yılları döndürür."""

        years = [
            int(value)
            for value in self._YEAR_RE.findall(
                text or ""
            )
        ]

        return sorted(
            set(years)
        )

    def extract_units(
        self,
        text: str,
    ) -> list[str]:
        """Metin içindeki bilinen birimleri döndürür."""

        units: list[str] = []

        for pattern, unit in self._UNIT_PATTERNS:
            if pattern.search(
                text or ""
            ):
                units.append(unit)

        if "%" in (text or ""):
            units.append("%")

        return list(
            dict.fromkeys(units)
        )

    def extract_currencies(
        self,
        text: str,
    ) -> list[str]:
        currencies: list[str] = []

        search_text = (
            text or ""
        ).casefold()

        for token, currency in (
            self._CURRENCY_SYMBOLS.items()
        ):
            if token.casefold() in search_text:
                currencies.append(currency)

        return list(
            dict.fromkeys(currencies)
        )

    def _normalize_unit(
        self,
        text: str,
    ) -> str | None:
        if not text:
            return None

        units = self.extract_units(
            text
        )

        if units:
            return units[0]

        return None

    def _detect_currency(
        self,
        text: str,
    ) -> str | None:
        currencies = self.extract_currencies(
            text
        )

        if currencies:
            return currencies[0]

        return None

    def _remove_non_numeric_tokens(
        self,
        text: str,
    ) -> str:
        result = text

        for token in sorted(
            self._CURRENCY_SYMBOLS,
            key=len,
            reverse=True,
        ):
            result = re.sub(
                re.escape(token),
                "",
                result,
                flags=re.IGNORECASE,
            )

        for pattern, _ in self._UNIT_PATTERNS:
            result = pattern.sub(
                "",
                result,
            )

        result = result.replace(
            "%",
            ""
        )

        result = re.sub(
            r"[^\d.,+\-\s()]",
            "",
            result,
        )

        result = result.strip()

        if (
            result.startswith("(")
            and result.endswith(")")
        ):
            result = (
                "-"
                + result[1:-1]
            )

        return result

    @staticmethod
    def _parse_decimal_string(
        value: str,
    ) -> float | None:
        text = value.strip()

        if not text:
            return None

        text = text.replace(
            "\u00a0",
            " "
        )

        text = text.replace(
            " ",
            ""
        )

        negative = False

        if (
            text.startswith("(")
            and text.endswith(")")
        ):
            negative = True
            text = text[1:-1]

        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(
                    ".",
                    ""
                )

                text = text.replace(
                    ",",
                    ".",
                )

            else:
                text = text.replace(
                    ",",
                    ""
                )

        elif "," in text:
            parts = text.split(",")

            if (
                len(parts) == 2
                and 1 <= len(parts[1]) <= 3
            ):
                text = text.replace(
                    ",",
                    ".",
                )
            else:
                text = text.replace(
                    ",",
                    ""
                )

        elif "." in text:
            parts = text.split(".")

            if (
                len(parts) > 2
                and all(
                    len(part) == 3
                    for part in parts[1:]
                )
            ):
                text = text.replace(
                    ".",
                    ""
                )

            elif (
                len(parts) == 2
                and len(parts[1]) == 3
                and len(parts[0]) >= 1
            ):
                # 12.345 ifadesi çoğunlukla binlik ayırıcıdır.
                text = text.replace(
                    ".",
                    ""
                )

        try:
            decimal_value = Decimal(
                text
            )

        except InvalidOperation:
            return None

        numeric_value = float(
            decimal_value
        )

        if negative:
            numeric_value *= -1

        if not math.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    def _unit_from_header(
        self,
        *,
        headers: Sequence[str],
        column_index: int,
    ) -> str | None:
        if column_index >= len(headers):
            return None

        return self._normalize_unit(
            headers[column_index]
        )

    def _year_from_header(
        self,
        header: str,
    ) -> int | None:
        years = self.extract_years(
            header
        )

        if len(years) == 1:
            return years[0]

        return None

    def _select_default_year(
        self,
        years: Sequence[int],
    ) -> int | None:
        if len(years) == 1:
            return years[0]

        if self.default_report_year is not None:
            return self.default_report_year

        return None

    @staticmethod
    def _deduplicate_headers(
        headers: Sequence[str],
    ) -> list[str]:
        result: list[str] = []
        counts: dict[str, int] = {}

        for index, header in enumerate(headers):
            base = (
                header.strip()
                or f"column_{index + 1}"
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

    @classmethod
    def normalize_text(
        cls,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        text = str(value)

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = text.replace(
            "\u00ad",
            ""
        )

        text = text.replace(
            "\r",
            " "
        )

        text = text.replace(
            "\n",
            " "
        )

        text = cls._MULTIPLE_SEPARATOR_RE.sub(
            " - ",
            text,
        )

        return cls._WHITESPACE_RE.sub(
            " ",
            text,
        ).strip()

    @classmethod
    def normalize_for_search(
        cls,
        value: Any,
    ) -> str:
        text = cls.normalize_text(
            value
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

        return cls._WHITESPACE_RE.sub(
            " ",
            text,
        ).strip()

    @classmethod
    def slugify_metric(
        cls,
        value: str | None,
    ) -> str | None:
        if not value:
            return None

        normalized = cls.normalize_for_search(
            value
        )

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized,
        )

        normalized = normalized.strip(
            "_"
        )

        return normalized or None

    @staticmethod
    def _extract_source_confidence(
        table: Any,
    ) -> float:
        for attribute_name in (
            "extraction_confidence",
            "overall_confidence",
            "average_confidence",
            "confidence",
        ):
            value = getattr(
                table,
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

        return 0.50

    @staticmethod
    def _get_source_cell_confidence(
        *,
        source_table: Any,
        row_index: int,
        column_index: int,
    ) -> float | None:
        rows = getattr(
            source_table,
            "rows",
            (),
        )

        for row in rows:
            if int(
                getattr(
                    row,
                    "row_index",
                    -1,
                )
            ) != row_index:
                continue

            cells = getattr(
                row,
                "cells",
                (),
            )

            for cell in cells:
                if int(
                    getattr(
                        cell,
                        "column_index",
                        -1,
                    )
                ) != column_index:
                    continue

                confidence = getattr(
                    cell,
                    "confidence",
                    None,
                )

                if confidence is None:
                    return None

                try:
                    numeric_confidence = float(
                        confidence
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    return None

                if numeric_confidence > 1.0:
                    numeric_confidence /= 100.0

                return min(
                    max(
                        numeric_confidence,
                        0.0,
                    ),
                    1.0,
                )

        cells = getattr(
            source_table,
            "cells",
            (),
        )

        for cell in cells:
            if (
                int(
                    getattr(
                        cell,
                        "row_index",
                        -1,
                    )
                )
                == row_index
                and int(
                    getattr(
                        cell,
                        "column_index",
                        -1,
                    )
                )
                == column_index
            ):
                confidence = getattr(
                    cell,
                    "confidence",
                    None,
                )

                if confidence is None:
                    return None

                try:
                    numeric_confidence = float(
                        confidence
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    return None

                if numeric_confidence > 1.0:
                    numeric_confidence /= 100.0

                return min(
                    max(
                        numeric_confidence,
                        0.0,
                    ),
                    1.0,
                )

        return None

    @staticmethod
    def _calculate_normalization_confidence(
        *,
        row_count: int,
        column_count: int,
        cells: Sequence[NormalizedTableCell],
        source_confidence: float,
        metric_record_count: int,
    ) -> float:
        if not cells:
            return 0.0

        parsed_confidence = sum(
            cell.confidence
            for cell in cells
        ) / len(cells)

        structure_score = 0.0

        if row_count >= 2:
            structure_score += 0.5

        if column_count >= 2:
            structure_score += 0.5

        metric_score = min(
            metric_record_count / 5,
            1.0,
        )

        confidence = (
            0.35 * source_confidence
            + 0.35 * parsed_confidence
            + 0.20 * structure_score
            + 0.10 * metric_score
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
    def _calculate_hash(
        cls,
        matrix: Sequence[Sequence[str]],
    ) -> str:
        canonical = "\n".join(
            "\t".join(
                cls.normalize_for_search(
                    value
                )
                for value in row
            )
            for row in matrix
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()


def normalize_table(
    table: Any,
    *,
    default_report_year: int | None = None,
) -> NormalizedTable:
    """TableNormalizer için kısa kullanım arayüzü."""

    normalizer = TableNormalizer(
        default_report_year=default_report_year
    )

    return normalizer.normalize_table(
        table
    )