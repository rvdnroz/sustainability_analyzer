from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import median
from typing import Any, Mapping, Sequence

from .table_normalizer import (
    NormalizedMetricRecord,
    NormalizedTable,
    NormalizedValueType,
)


logger = logging.getLogger(__name__)


class NumericValidationError(RuntimeError):
    """Sayısal doğrulama işlemi tamamlanamadığında yükseltilir."""


class ValidationSeverity(StrEnum):
    """Doğrulama sorununun önem seviyesi."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssueType(StrEnum):
    """Tespit edilebilecek doğrulama problemi türleri."""

    MISSING_YEAR = "missing_year"
    INVALID_YEAR = "invalid_year"
    YEAR_MISMATCH = "year_mismatch"

    MISSING_UNIT = "missing_unit"
    UNIT_MISMATCH = "unit_mismatch"
    CURRENCY_MISMATCH = "currency_mismatch"

    LOW_CONFIDENCE = "low_confidence"
    INVALID_NUMBER = "invalid_number"
    NON_FINITE_NUMBER = "non_finite_number"

    NEGATIVE_VALUE = "negative_value"
    IMPOSSIBLE_PERCENTAGE = "impossible_percentage"
    PERCENTAGE_SCALE_SUSPECT = "percentage_scale_suspect"

    OUTLIER_VALUE = "outlier_value"
    ABRUPT_CHANGE = "abrupt_change"
    DUPLICATE_RECORD = "duplicate_record"
    CONFLICTING_VALUE = "conflicting_value"

    RANGE_ORDER_ERROR = "range_order_error"
    EMPTY_METRIC_NAME = "empty_metric_name"
    STRUCTURE_ERROR = "structure_error"

    OCR_CONFUSION_SUSPECTED = "ocr_confusion_suspected"
    UNIT_SCALE_SUSPECTED = "unit_scale_suspected"


@dataclass(slots=True, frozen=True)
class NumericValidationIssue:
    """Tek bir sayısal doğrulama problemi."""

    issue_id: str
    issue_type: ValidationIssueType
    severity: ValidationSeverity

    message: str

    table_id: str
    record_id: str | None

    page_number: int
    table_index: int

    row_index: int | None
    column_index: int | None

    metric_name: str | None
    year: int | None
    unit: str | None

    observed_value: float | None
    expected_value: float | None

    confidence: float

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issue_type"] = self.issue_type.value
        data["severity"] = self.severity.value
        return data


@dataclass(slots=True, frozen=True)
class RecordValidationResult:
    """Tek bir metrik kaydına ait doğrulama sonucu."""

    record_id: str
    table_id: str

    is_valid: bool
    validation_score: float

    issue_count: int
    issues: tuple[NumericValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "table_id": self.table_id,
            "is_valid": self.is_valid,
            "validation_score": self.validation_score,
            "issue_count": self.issue_count,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


@dataclass(slots=True, frozen=True)
class TableNumericValidationResult:
    """Normalize edilmiş bir tablonun sayısal doğrulama sonucu."""

    table_id: str
    page_number: int
    table_index: int

    record_count: int
    valid_record_count: int
    invalid_record_count: int

    issue_count: int
    error_count: int
    warning_count: int
    info_count: int

    validation_score: float
    is_valid: bool

    record_results: tuple[RecordValidationResult, ...]
    issues: tuple[NumericValidationIssue, ...]

    detected_units: tuple[str, ...]
    detected_years: tuple[int, ...]

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "record_count": self.record_count,
            "valid_record_count": self.valid_record_count,
            "invalid_record_count": self.invalid_record_count,
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "validation_score": self.validation_score,
            "is_valid": self.is_valid,
            "record_results": [
                result.to_dict()
                for result in self.record_results
            ],
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "detected_units": list(
                self.detected_units
            ),
            "detected_years": list(
                self.detected_years
            ),
            "metadata": self.metadata,
        }


class NumericValidator:
    """
    Normalize edilmiş tablo ve metrik kayıtlarını doğrular.

    Uygulanan kontroller:

    - Sayısal değerin sonlu olup olmadığı
    - Yıl değerinin kabul edilebilir aralıkta olması
    - Hedef rapor yılıyla uyum
    - Birim eksikliği ve satır içi birim tutarsızlığı
    - Yüzde değerlerinin mantıksal aralığı
    - Negatif değerlerin uygunluğu
    - Alt ve üst sınır sıralaması
    - Düşük extraction güveni
    - Aynı metrik-yıl-birim için çakışan değerler
    - Aynı seride güçlü aykırı değerler
    - Ardışık yıllar arasında aşırı değişim
    - OCR kaynaklı karakter karışıklığı olasılığı
    """

    _OCR_CONFUSION_RE = re.compile(
        r"""
        (?:
            (?<=\d)[Oo](?=\d)
            |
            (?<=\d)[Il](?=\d)
            |
            (?<=\d)[Ss](?=\d)
            |
            (?<=\d)[Bb](?=\d)
        )
        """,
        re.VERBOSE,
    )

    _NON_NEGATIVE_METRIC_TERMS = (
        "emission",
        "emissions",
        "emisyon",
        "energy",
        "enerji",
        "water",
        "su tüketimi",
        "waste",
        "atık",
        "employee",
        "çalışan",
        "training",
        "eğitim",
        "incident",
        "olay",
        "accident",
        "kaza",
        "injury",
        "yaralanma",
        "revenue",
        "gelir",
        "consumption",
        "tüketim",
        "production",
        "üretim",
    )

    _PERCENTAGE_UNIT_VALUES = {
        "%",
        "percent",
        "percentage",
        "yüzde",
    }

    _UNIT_SCALE_GROUPS = {
        "energy": {
            "kWh": 1.0,
            "MWh": 1_000.0,
            "GWh": 1_000_000.0,
            "MJ": 1.0,
            "GJ": 1_000.0,
        },
        "mass": {
            "g": 1.0,
            "kg": 1_000.0,
            "tonne": 1_000_000.0,
        },
        "emission": {
            "kgCO2e": 1.0,
            "tCO2e": 1_000.0,
        },
    }

    def __init__(
        self,
        *,
        expected_report_year: int | None = None,
        minimum_confidence: float = 0.50,
        minimum_validation_score: float = 0.60,
        allowed_year_difference: int = 5,
        percentage_minimum: float = 0.0,
        percentage_maximum: float = 100.0,
        percentage_fraction_tolerance: bool = True,
        detect_negative_values: bool = True,
        detect_outliers: bool = True,
        outlier_mad_threshold: float = 6.0,
        detect_abrupt_changes: bool = True,
        abrupt_change_ratio: float = 10.0,
        conflict_relative_tolerance: float = 0.01,
        duplicate_relative_tolerance: float = 0.0001,
    ) -> None:
        if (
            expected_report_year is not None
            and not 1900 <= expected_report_year <= 2100
        ):
            raise ValueError(
                "expected_report_year geçersiz."
            )

        for name, value in {
            "minimum_confidence": minimum_confidence,
            "minimum_validation_score": (
                minimum_validation_score
            ),
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        if allowed_year_difference < 0:
            raise ValueError(
                "allowed_year_difference negatif olamaz."
            )

        if percentage_minimum > percentage_maximum:
            raise ValueError(
                "percentage_minimum, percentage_maximum değerinden "
                "büyük olamaz."
            )

        if outlier_mad_threshold <= 0:
            raise ValueError(
                "outlier_mad_threshold pozitif olmalıdır."
            )

        if abrupt_change_ratio <= 1:
            raise ValueError(
                "abrupt_change_ratio 1'den büyük olmalıdır."
            )

        if conflict_relative_tolerance < 0:
            raise ValueError(
                "conflict_relative_tolerance negatif olamaz."
            )

        if duplicate_relative_tolerance < 0:
            raise ValueError(
                "duplicate_relative_tolerance negatif olamaz."
            )

        self.expected_report_year = expected_report_year
        self.minimum_confidence = minimum_confidence
        self.minimum_validation_score = minimum_validation_score

        self.allowed_year_difference = allowed_year_difference

        self.percentage_minimum = percentage_minimum
        self.percentage_maximum = percentage_maximum
        self.percentage_fraction_tolerance = (
            percentage_fraction_tolerance
        )

        self.detect_negative_values = detect_negative_values
        self.detect_outliers = detect_outliers
        self.outlier_mad_threshold = outlier_mad_threshold

        self.detect_abrupt_changes = detect_abrupt_changes
        self.abrupt_change_ratio = abrupt_change_ratio

        self.conflict_relative_tolerance = (
            conflict_relative_tolerance
        )

        self.duplicate_relative_tolerance = (
            duplicate_relative_tolerance
        )

    def validate_table(
        self,
        table: NormalizedTable,
    ) -> TableNumericValidationResult:
        """Normalize edilmiş tablodaki bütün metrik kayıtlarını doğrular."""

        issues: list[NumericValidationIssue] = []

        issues.extend(
            self._validate_table_structure(
                table
            )
        )

        record_issues: dict[
            str,
            list[NumericValidationIssue],
        ] = {
            record.record_id: []
            for record in table.metric_records
        }

        for record in table.metric_records:
            current_issues = self.validate_record(
                record=record,
                table=table,
            )

            record_issues[
                record.record_id
            ].extend(current_issues)

            issues.extend(current_issues)

        cross_record_issues = (
            self._validate_cross_record_consistency(
                table
            )
        )

        issues.extend(
            cross_record_issues
        )

        for issue in cross_record_issues:
            if (
                issue.record_id is not None
                and issue.record_id
                in record_issues
            ):
                record_issues[
                    issue.record_id
                ].append(issue)

        record_results: list[
            RecordValidationResult
        ] = []

        for record in table.metric_records:
            current_issues = record_issues[
                record.record_id
            ]

            score = self._calculate_record_score(
                record=record,
                issues=current_issues,
            )

            is_valid = (
                score >= self.minimum_validation_score
                and not any(
                    issue.severity
                    in {
                        ValidationSeverity.ERROR,
                        ValidationSeverity.CRITICAL,
                    }
                    for issue in current_issues
                )
            )

            record_results.append(
                RecordValidationResult(
                    record_id=record.record_id,
                    table_id=record.table_id,
                    is_valid=is_valid,
                    validation_score=score,
                    issue_count=len(
                        current_issues
                    ),
                    issues=tuple(
                        current_issues
                    ),
                )
            )

        error_count = sum(
            issue.severity
            in {
                ValidationSeverity.ERROR,
                ValidationSeverity.CRITICAL,
            }
            for issue in issues
        )

        warning_count = sum(
            issue.severity
            == ValidationSeverity.WARNING
            for issue in issues
        )

        info_count = sum(
            issue.severity
            == ValidationSeverity.INFO
            for issue in issues
        )

        table_score = self._calculate_table_score(
            table=table,
            record_results=record_results,
            issues=issues,
        )

        is_valid = (
            table.is_valid
            and table_score
            >= self.minimum_validation_score
            and error_count == 0
        )

        return TableNumericValidationResult(
            table_id=table.table_id,
            page_number=table.page_number,
            table_index=table.table_index,
            record_count=len(
                table.metric_records
            ),
            valid_record_count=sum(
                result.is_valid
                for result in record_results
            ),
            invalid_record_count=sum(
                not result.is_valid
                for result in record_results
            ),
            issue_count=len(issues),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            validation_score=table_score,
            is_valid=is_valid,
            record_results=tuple(
                record_results
            ),
            issues=tuple(issues),
            detected_units=table.detected_units,
            detected_years=table.detected_years,
            metadata={
                "expected_report_year": (
                    self.expected_report_year
                ),
                "source_normalization_confidence": (
                    table.normalization_confidence
                ),
                "source_extraction_confidence": (
                    table.source_confidence
                ),
            },
        )

    def validate_tables(
        self,
        tables: Sequence[NormalizedTable],
    ) -> list[TableNumericValidationResult]:
        """Birden fazla normalize edilmiş tabloyu doğrular."""

        results: list[
            TableNumericValidationResult
        ] = []

        for table in tables:
            try:
                results.append(
                    self.validate_table(table)
                )

            except Exception as exc:
                logger.warning(
                    "Tablo sayısal doğrulaması başarısız: %s",
                    exc,
                )

        return results

    def validate_record(
        self,
        *,
        record: NormalizedMetricRecord,
        table: NormalizedTable,
    ) -> list[NumericValidationIssue]:
        """Tek bir normalize edilmiş metrik kaydını doğrular."""

        issues: list[NumericValidationIssue] = []

        if not record.metric_name.strip():
            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.EMPTY_METRIC_NAME
                    ),
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "Metrik adı boş veya okunamaz durumda."
                    ),
                    record=record,
                    confidence=0.98,
                )
            )

        if record.numeric_value is None:
            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.INVALID_NUMBER
                    ),
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "Metrik kaydı sayısal bir değer içermiyor."
                    ),
                    record=record,
                    confidence=0.95,
                )
            )

        elif not math.isfinite(
            record.numeric_value
        ):
            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.NON_FINITE_NUMBER
                    ),
                    severity=ValidationSeverity.CRITICAL,
                    message=(
                        "Sayısal değer sonlu değil."
                    ),
                    record=record,
                    confidence=1.0,
                )
            )

        if record.confidence < self.minimum_confidence:
            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.LOW_CONFIDENCE
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Sayısal extraction güven skoru düşük: "
                        f"{record.confidence:.3f}."
                    ),
                    record=record,
                    confidence=0.90,
                    metadata={
                        "minimum_confidence": (
                            self.minimum_confidence
                        )
                    },
                )
            )

        issues.extend(
            self._validate_year(
                record
            )
        )

        issues.extend(
            self._validate_unit(
                record
            )
        )

        issues.extend(
            self._validate_percentage(
                record
            )
        )

        issues.extend(
            self._validate_negative_value(
                record
            )
        )

        issues.extend(
            self._validate_range(
                record
            )
        )

        issues.extend(
            self._detect_ocr_confusion(
                record
            )
        )

        return issues

    def _validate_table_structure(
        self,
        table: NormalizedTable,
    ) -> list[NumericValidationIssue]:
        issues: list[
            NumericValidationIssue
        ] = []

        if table.row_count < 2:
            issues.append(
                self._table_issue(
                    table=table,
                    issue_type=(
                        ValidationIssueType.STRUCTURE_ERROR
                    ),
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "Tablo en az iki satır içermelidir."
                    ),
                    confidence=1.0,
                    metadata={
                        "row_count": table.row_count
                    },
                )
            )

        if table.column_count < 2:
            issues.append(
                self._table_issue(
                    table=table,
                    issue_type=(
                        ValidationIssueType.STRUCTURE_ERROR
                    ),
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "Tablo en az iki sütun içermelidir."
                    ),
                    confidence=1.0,
                    metadata={
                        "column_count": (
                            table.column_count
                        )
                    },
                )
            )

        if not table.metric_records:
            issues.append(
                self._table_issue(
                    table=table,
                    issue_type=(
                        ValidationIssueType.STRUCTURE_ERROR
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Tablodan doğrulanabilir metrik kaydı üretilemedi."
                    ),
                    confidence=0.95,
                )
            )

        return issues

    def _validate_year(
        self,
        record: NormalizedMetricRecord,
    ) -> list[NumericValidationIssue]:
        issues: list[
            NumericValidationIssue
        ] = []

        if record.year is None:
            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.MISSING_YEAR
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Sayısal değer için yıl bilgisi bulunamadı."
                    ),
                    record=record,
                    confidence=0.90,
                )
            )

            return issues

        if not 1900 <= record.year <= 2100:
            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.INVALID_YEAR
                    ),
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Geçersiz yıl değeri: {record.year}."
                    ),
                    record=record,
                    confidence=1.0,
                )
            )

        if self.expected_report_year is not None:
            difference = abs(
                record.year
                - self.expected_report_year
            )

            if difference > self.allowed_year_difference:
                issues.append(
                    self._issue(
                        issue_type=(
                            ValidationIssueType.YEAR_MISMATCH
                        ),
                        severity=ValidationSeverity.WARNING,
                        message=(
                            "Kayıt yılı rapor yılıyla beklenenden fazla "
                            f"farklı: kayıt={record.year}, "
                            f"rapor={self.expected_report_year}."
                        ),
                        record=record,
                        confidence=0.85,
                        metadata={
                            "year_difference": difference,
                            "allowed_year_difference": (
                                self.allowed_year_difference
                            ),
                        },
                    )
                )

        return issues

    def _validate_unit(
        self,
        record: NormalizedMetricRecord,
    ) -> list[NumericValidationIssue]:
        if record.numeric_value is None:
            return []

        metric_name = (
            record.metric_name.casefold()
        )

        likely_requires_unit = any(
            term in metric_name
            for term in (
                "emission",
                "emisyon",
                "energy",
                "enerji",
                "water",
                "su ",
                "waste",
                "atık",
                "consumption",
                "tüketim",
                "production",
                "üretim",
                "revenue",
                "gelir",
            )
        )

        if likely_requires_unit and not record.unit:
            return [
                self._issue(
                    issue_type=(
                        ValidationIssueType.MISSING_UNIT
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Sayısal metrik için ölçü birimi bulunamadı."
                    ),
                    record=record,
                    confidence=0.82,
                )
            ]

        return []

    def _validate_percentage(
        self,
        record: NormalizedMetricRecord,
    ) -> list[NumericValidationIssue]:
        if (
            record.value_type
            != NormalizedValueType.PERCENTAGE
            and not record.is_percentage
            and record.unit not in self._PERCENTAGE_UNIT_VALUES
        ):
            return []

        if record.numeric_value is None:
            return []

        value = record.numeric_value

        if (
            self.percentage_minimum
            <= value
            <= self.percentage_maximum
        ):
            return []

        if (
            self.percentage_fraction_tolerance
            and 0.0 <= value <= 1.0
        ):
            return [
                self._issue(
                    issue_type=(
                        ValidationIssueType.PERCENTAGE_SCALE_SUSPECT
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Yüzde değeri 0–1 ölçeğinde olabilir; "
                        f"gözlenen değer: {value}."
                    ),
                    record=record,
                    confidence=0.78,
                    metadata={
                        "suggested_percentage": (
                            value * 100
                        )
                    },
                )
            ]

        return [
            self._issue(
                issue_type=(
                    ValidationIssueType.IMPOSSIBLE_PERCENTAGE
                ),
                severity=ValidationSeverity.ERROR,
                message=(
                    "Yüzde değeri kabul edilen aralığın dışında: "
                    f"{value}."
                ),
                record=record,
                confidence=0.95,
                metadata={
                    "minimum": (
                        self.percentage_minimum
                    ),
                    "maximum": (
                        self.percentage_maximum
                    ),
                },
            )
        ]

    def _validate_negative_value(
        self,
        record: NormalizedMetricRecord,
    ) -> list[NumericValidationIssue]:
        if not self.detect_negative_values:
            return []

        if record.numeric_value is None:
            return []

        if record.numeric_value >= 0:
            return []

        metric_name = (
            record.metric_name.casefold()
        )

        likely_non_negative = any(
            term in metric_name
            for term in self._NON_NEGATIVE_METRIC_TERMS
        )

        if not likely_non_negative:
            return []

        return [
            self._issue(
                issue_type=(
                    ValidationIssueType.NEGATIVE_VALUE
                ),
                severity=ValidationSeverity.WARNING,
                message=(
                    "Normalde negatif olması beklenmeyen metrikte "
                    f"negatif değer bulundu: {record.numeric_value}."
                ),
                record=record,
                confidence=0.80,
            )
        ]

    def _validate_range(
        self,
        record: NormalizedMetricRecord,
    ) -> list[NumericValidationIssue]:
        lower = record.metadata.get(
            "lower_bound"
        )

        upper = record.metadata.get(
            "upper_bound"
        )

        if lower is None or upper is None:
            return []

        try:
            lower_value = float(lower)
            upper_value = float(upper)

        except (
            TypeError,
            ValueError,
        ):
            return []

        if lower_value <= upper_value:
            return []

        return [
            self._issue(
                issue_type=(
                    ValidationIssueType.RANGE_ORDER_ERROR
                ),
                severity=ValidationSeverity.ERROR,
                message=(
                    "Aralık alt sınırı üst sınırdan büyük."
                ),
                record=record,
                confidence=0.98,
                metadata={
                    "lower_bound": lower_value,
                    "upper_bound": upper_value,
                },
            )
        ]

    def _detect_ocr_confusion(
        self,
        record: NormalizedMetricRecord,
    ) -> list[NumericValidationIssue]:
        raw_value = record.raw_value or ""

        if not self._OCR_CONFUSION_RE.search(
            raw_value
        ):
            return []

        return [
            self._issue(
                issue_type=(
                    ValidationIssueType.OCR_CONFUSION_SUSPECTED
                ),
                severity=ValidationSeverity.WARNING,
                message=(
                    "Sayısal değer içerisinde OCR karakter karışıklığı "
                    "olasılığı bulundu."
                ),
                record=record,
                confidence=0.75,
                metadata={
                    "raw_value": raw_value
                },
            )
        ]

    def _validate_cross_record_consistency(
        self,
        table: NormalizedTable,
    ) -> list[NumericValidationIssue]:
        issues: list[
            NumericValidationIssue
        ] = []

        issues.extend(
            self._detect_duplicate_and_conflicting_records(
                table.metric_records
            )
        )

        grouped_records = self._group_records_by_metric(
            table.metric_records
        )

        for records in grouped_records.values():
            if self.detect_outliers:
                issues.extend(
                    self._detect_outliers(
                        records
                    )
                )

            if self.detect_abrupt_changes:
                issues.extend(
                    self._detect_abrupt_changes(
                        records
                    )
                )

            issues.extend(
                self._detect_unit_mismatches(
                    records
                )
            )

        return issues

    def _detect_duplicate_and_conflicting_records(
        self,
        records: Sequence[NormalizedMetricRecord],
    ) -> list[NumericValidationIssue]:
        groups: dict[
            tuple[str, int | None, str | None],
            list[NormalizedMetricRecord],
        ] = {}

        for record in records:
            key = (
                record.metric_slug,
                record.year,
                record.unit,
            )

            groups.setdefault(
                key,
                [],
            ).append(record)

        issues: list[
            NumericValidationIssue
        ] = []

        for group_records in groups.values():
            if len(group_records) < 2:
                continue

            numeric_records = [
                record
                for record in group_records
                if record.numeric_value
                is not None
            ]

            if len(numeric_records) < 2:
                continue

            reference = numeric_records[0]

            for candidate in numeric_records[1:]:
                relative_difference = (
                    self._relative_difference(
                        reference.numeric_value,
                        candidate.numeric_value,
                    )
                )

                if (
                    relative_difference
                    <= self.duplicate_relative_tolerance
                ):
                    issues.append(
                        self._issue(
                            issue_type=(
                                ValidationIssueType.DUPLICATE_RECORD
                            ),
                            severity=ValidationSeverity.INFO,
                            message=(
                                "Aynı metrik, yıl ve birim için "
                                "yinelenen kayıt bulundu."
                            ),
                            record=candidate,
                            expected_value=(
                                reference.numeric_value
                            ),
                            confidence=0.90,
                            metadata={
                                "reference_record_id": (
                                    reference.record_id
                                ),
                                "relative_difference": (
                                    relative_difference
                                ),
                            },
                        )
                    )

                elif (
                    relative_difference
                    > self.conflict_relative_tolerance
                ):
                    issues.append(
                        self._issue(
                            issue_type=(
                                ValidationIssueType.CONFLICTING_VALUE
                            ),
                            severity=ValidationSeverity.ERROR,
                            message=(
                                "Aynı metrik, yıl ve birim için "
                                "birbiriyle çelişen değerler bulundu."
                            ),
                            record=candidate,
                            expected_value=(
                                reference.numeric_value
                            ),
                            confidence=0.95,
                            metadata={
                                "reference_record_id": (
                                    reference.record_id
                                ),
                                "relative_difference": (
                                    relative_difference
                                ),
                            },
                        )
                    )

        return issues

    def _detect_outliers(
        self,
        records: Sequence[NormalizedMetricRecord],
    ) -> list[NumericValidationIssue]:
        numeric_records = [
            record
            for record in records
            if record.numeric_value is not None
            and math.isfinite(
                record.numeric_value
            )
        ]

        if len(numeric_records) < 4:
            return []

        values = [
            record.numeric_value
            for record in numeric_records
        ]

        center = median(values)

        deviations = [
            abs(value - center)
            for value in values
        ]

        mad = median(deviations)

        if math.isclose(
            mad,
            0.0,
            abs_tol=1e-12,
        ):
            return []

        issues: list[
            NumericValidationIssue
        ] = []

        for record in numeric_records:
            modified_z_score = (
                0.6745
                * abs(
                    record.numeric_value
                    - center
                )
                / mad
            )

            if (
                modified_z_score
                <= self.outlier_mad_threshold
            ):
                continue

            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.OUTLIER_VALUE
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Aynı metrik serisinde güçlü aykırı değer "
                        f"tespit edildi: {record.numeric_value}."
                    ),
                    record=record,
                    expected_value=center,
                    confidence=min(
                        modified_z_score / 10,
                        0.99,
                    ),
                    metadata={
                        "median": center,
                        "mad": mad,
                        "modified_z_score": (
                            modified_z_score
                        ),
                    },
                )
            )

        return issues

    def _detect_abrupt_changes(
        self,
        records: Sequence[NormalizedMetricRecord],
    ) -> list[NumericValidationIssue]:
        yearly_records = sorted(
            (
                record
                for record in records
                if (
                    record.year is not None
                    and record.numeric_value
                    is not None
                    and math.isfinite(
                        record.numeric_value
                    )
                )
            ),
            key=lambda record: record.year,
        )

        if len(yearly_records) < 2:
            return []

        issues: list[
            NumericValidationIssue
        ] = []

        for previous, current in zip(
            yearly_records,
            yearly_records[1:],
            strict=False,
        ):
            if previous.numeric_value is None:
                continue

            if current.numeric_value is None:
                continue

            denominator = max(
                abs(
                    previous.numeric_value
                ),
                1e-12,
            )

            ratio = (
                abs(
                    current.numeric_value
                )
                / denominator
            )

            inverse_ratio = (
                abs(
                    previous.numeric_value
                )
                / max(
                    abs(
                        current.numeric_value
                    ),
                    1e-12,
                )
            )

            change_ratio = max(
                ratio,
                inverse_ratio,
            )

            if (
                change_ratio
                < self.abrupt_change_ratio
            ):
                continue

            issues.append(
                self._issue(
                    issue_type=(
                        ValidationIssueType.ABRUPT_CHANGE
                    ),
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Ardışık yıllar arasında olağan dışı büyüklükte "
                        f"değişim bulundu: {previous.year} → "
                        f"{current.year}."
                    ),
                    record=current,
                    expected_value=(
                        previous.numeric_value
                    ),
                    confidence=min(
                        change_ratio
                        / (
                            self.abrupt_change_ratio
                            * 2
                        ),
                        0.99,
                    ),
                    metadata={
                        "previous_record_id": (
                            previous.record_id
                        ),
                        "previous_year": (
                            previous.year
                        ),
                        "previous_value": (
                            previous.numeric_value
                        ),
                        "change_ratio": (
                            change_ratio
                        ),
                    },
                )
            )

        return issues

    def _detect_unit_mismatches(
        self,
        records: Sequence[NormalizedMetricRecord],
    ) -> list[NumericValidationIssue]:
        records_with_unit = [
            record
            for record in records
            if record.unit
        ]

        units = {
            record.unit
            for record in records_with_unit
        }

        if len(units) <= 1:
            return []

        issues: list[
            NumericValidationIssue
        ] = []

        normalized_groups = {
            self._unit_scale_group(
                record.unit
            )
            for record in records_with_unit
        }

        normalized_groups.discard(
            None
        )

        if len(normalized_groups) == 1:
            for record in records_with_unit:
                issues.append(
                    self._issue(
                        issue_type=(
                            ValidationIssueType.UNIT_SCALE_SUSPECTED
                        ),
                        severity=ValidationSeverity.INFO,
                        message=(
                            "Aynı metrik farklı ölçeklerdeki uyumlu "
                            "birimlerle raporlanmış olabilir."
                        ),
                        record=record,
                        confidence=0.75,
                        metadata={
                            "units_in_series": (
                                sorted(units)
                            )
                        },
                    )
                )

        else:
            for record in records_with_unit:
                issues.append(
                    self._issue(
                        issue_type=(
                            ValidationIssueType.UNIT_MISMATCH
                        ),
                        severity=ValidationSeverity.WARNING,
                        message=(
                            "Aynı metrik serisinde uyumsuz birimler "
                            "bulundu."
                        ),
                        record=record,
                        confidence=0.85,
                        metadata={
                            "units_in_series": (
                                sorted(units)
                            )
                        },
                    )
                )

        return issues

    @staticmethod
    def _group_records_by_metric(
        records: Sequence[NormalizedMetricRecord],
    ) -> dict[
        str,
        list[NormalizedMetricRecord],
    ]:
        groups: dict[
            str,
            list[NormalizedMetricRecord],
        ] = {}

        for record in records:
            groups.setdefault(
                record.metric_slug,
                [],
            ).append(record)

        return groups

    def _calculate_record_score(
        self,
        *,
        record: NormalizedMetricRecord,
        issues: Sequence[NumericValidationIssue],
    ) -> float:
        score = min(
            max(
                record.confidence,
                0.0,
            ),
            1.0,
        )

        penalties = {
            ValidationSeverity.INFO: 0.02,
            ValidationSeverity.WARNING: 0.10,
            ValidationSeverity.ERROR: 0.30,
            ValidationSeverity.CRITICAL: 0.60,
        }

        for issue in issues:
            score -= penalties[
                issue.severity
            ]

        return round(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            ),
            4,
        )

    @staticmethod
    def _calculate_table_score(
        *,
        table: NormalizedTable,
        record_results: Sequence[RecordValidationResult],
        issues: Sequence[NumericValidationIssue],
    ) -> float:
        if record_results:
            record_score = sum(
                result.validation_score
                for result in record_results
            ) / len(record_results)

        else:
            record_score = 0.0

        structural_score = (
            1.0
            if table.is_valid
            else 0.25
        )

        critical_count = sum(
            issue.severity
            == ValidationSeverity.CRITICAL
            for issue in issues
        )

        error_count = sum(
            issue.severity
            == ValidationSeverity.ERROR
            for issue in issues
        )

        score = (
            0.45
            * table.normalization_confidence
            + 0.40
            * record_score
            + 0.15
            * structural_score
        )

        score -= min(
            critical_count * 0.30,
            0.60,
        )

        score -= min(
            error_count * 0.10,
            0.30,
        )

        return round(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            ),
            4,
        )

    def _issue(
        self,
        *,
        issue_type: ValidationIssueType,
        severity: ValidationSeverity,
        message: str,
        record: NormalizedMetricRecord,
        confidence: float,
        expected_value: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NumericValidationIssue:
        issue_id = (
            f"{record.record_id}:"
            f"{issue_type.value}"
        )

        return NumericValidationIssue(
            issue_id=issue_id,
            issue_type=issue_type,
            severity=severity,
            message=message,
            table_id=record.table_id,
            record_id=record.record_id,
            page_number=record.page_number,
            table_index=record.table_index,
            row_index=record.row_index,
            column_index=record.column_index,
            metric_name=record.metric_name,
            year=record.year,
            unit=record.unit,
            observed_value=(
                record.numeric_value
            ),
            expected_value=expected_value,
            confidence=round(
                min(
                    max(
                        confidence,
                        0.0,
                    ),
                    1.0,
                ),
                4,
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    @staticmethod
    def _table_issue(
        *,
        table: NormalizedTable,
        issue_type: ValidationIssueType,
        severity: ValidationSeverity,
        message: str,
        confidence: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> NumericValidationIssue:
        issue_id = (
            f"{table.table_id}:"
            f"{issue_type.value}"
        )

        return NumericValidationIssue(
            issue_id=issue_id,
            issue_type=issue_type,
            severity=severity,
            message=message,
            table_id=table.table_id,
            record_id=None,
            page_number=table.page_number,
            table_index=table.table_index,
            row_index=None,
            column_index=None,
            metric_name=None,
            year=None,
            unit=None,
            observed_value=None,
            expected_value=None,
            confidence=round(
                min(
                    max(
                        confidence,
                        0.0,
                    ),
                    1.0,
                ),
                4,
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    @staticmethod
    def _relative_difference(
        first: float | None,
        second: float | None,
    ) -> float:
        if first is None or second is None:
            return math.inf

        denominator = max(
            abs(first),
            abs(second),
            1e-12,
        )

        return abs(
            first - second
        ) / denominator

    @classmethod
    def _unit_scale_group(
        cls,
        unit: str | None,
    ) -> str | None:
        if unit is None:
            return None

        for group_name, units in (
            cls._UNIT_SCALE_GROUPS.items()
        ):
            if unit in units:
                return group_name

        return None


def validate_numeric_table(
    table: NormalizedTable,
    *,
    expected_report_year: int | None = None,
) -> TableNumericValidationResult:
    """NumericValidator için kısa kullanım arayüzü."""

    validator = NumericValidator(
        expected_report_year=(
            expected_report_year
        )
    )

    return validator.validate_table(
        table
    )