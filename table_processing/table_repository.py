from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from storage import (
    ArtifactType,
    Database,
    FileStore,
    StoredArtifact,
)

from .numeric_validator import (
    NumericValidationIssue,
    TableNumericValidationResult,
)
from .table_normalizer import (
    NormalizedMetricRecord,
    NormalizedTable,
    NormalizedTableCell,
    NormalizedTableRow,
    NormalizedValueType,
)


logger = logging.getLogger(__name__)


class TableRepositoryError(RuntimeError):
    """Tablo repository işlemi başarısız olduğunda yükseltilir."""


@dataclass(slots=True, frozen=True)
class StoredTableResult:
    """Depolama katmanına kaydedilmiş tablo sonucu."""

    table_database_id: int
    table_uid: str

    report_id: int
    page_number: int
    table_index: int

    normalized_artifact: StoredArtifact | None
    validation_artifact: StoredArtifact | None

    metric_record_count: int
    validation_issue_count: int

    is_valid: bool
    stored_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_database_id": self.table_database_id,
            "table_uid": self.table_uid,
            "report_id": self.report_id,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "normalized_artifact": (
                self.normalized_artifact.to_dict()
                if self.normalized_artifact is not None
                else None
            ),
            "validation_artifact": (
                self.validation_artifact.to_dict()
                if self.validation_artifact is not None
                else None
            ),
            "metric_record_count": self.metric_record_count,
            "validation_issue_count": self.validation_issue_count,
            "is_valid": self.is_valid,
            "stored_at": self.stored_at.isoformat(),
        }


@dataclass(slots=True, frozen=True)
class TableRepositorySummary:
    """Birden fazla tablo saklama işleminin özeti."""

    report_id: int

    requested_table_count: int
    stored_table_count: int
    failed_table_count: int

    valid_table_count: int
    invalid_table_count: int

    metric_record_count: int
    validation_issue_count: int

    results: tuple[StoredTableResult, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "requested_table_count": (
                self.requested_table_count
            ),
            "stored_table_count": (
                self.stored_table_count
            ),
            "failed_table_count": (
                self.failed_table_count
            ),
            "valid_table_count": (
                self.valid_table_count
            ),
            "invalid_table_count": (
                self.invalid_table_count
            ),
            "metric_record_count": (
                self.metric_record_count
            ),
            "validation_issue_count": (
                self.validation_issue_count
            ),
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "errors": list(self.errors),
        }


class TableRepository:
    """
    Normalize edilmiş tabloları kalıcı depolama katmanına aktarır.

    Temel görevleri:

    - NormalizedTable nesnesini SQLite tablosuna kaydetmek
    - Tablo matrisi, başlıkları ve metrik kayıtlarını JSON olarak saklamak
    - NumericValidator çıktısını tablo metadata alanına eklemek
    - Normalize edilmiş tabloyu artifact dosyası olarak yazmak
    - Doğrulama raporunu ayrı JSON artifact olarak saklamak
    - Veri tabanı ve dosya sistemi arasında izlenebilirlik sağlamak

    Bu sınıf extraction veya normalizasyon yapmaz. Girdi olarak hazır
    NormalizedTable ve isteğe bağlı TableNumericValidationResult alır.
    """

    def __init__(
        self,
        *,
        database: Database,
        file_store: FileStore,
        save_json_artifacts: bool = True,
        overwrite_artifacts: bool = True,
    ) -> None:
        self.database = database
        self.file_store = file_store

        self.save_json_artifacts = (
            save_json_artifacts
        )

        self.overwrite_artifacts = (
            overwrite_artifacts
        )

        # Aynı çalışma sürecindeki retrieval katmanı için hafif bellek içi
        # indeks. Kalıcı kayıtlar yine Database ve FileStore üzerinden yapılır.
        self._tables_by_id: dict[str, NormalizedTable] = {}
        self._report_uid_by_table_id: dict[str, str] = {}

    def cache_tables(
        self,
        report_uid: str,
        tables: Sequence[NormalizedTable],
    ) -> int:
        normalized_report_uid = str(report_uid or "").strip()
        if not normalized_report_uid:
            raise ValueError("report_uid boş olamaz.")

        for table in tables:
            table_id = str(table.table_id).strip()
            if not table_id:
                continue
            self._tables_by_id[table_id] = table
            self._report_uid_by_table_id[table_id] = normalized_report_uid
        return len(tables)

    def all(self) -> list[NormalizedTable]:
        return sorted(
            self._tables_by_id.values(),
            key=lambda table: (
                self._report_uid_by_table_id.get(table.table_id, ""),
                table.page_number,
                table.table_index,
                table.table_id,
            ),
        )

    def get_optional(self, table_id: str) -> NormalizedTable | None:
        return self._tables_by_id.get(str(table_id).strip())

    def find(
        self,
        *,
        report_id: str | None = None,
        page_number: int | None = None,
        category: str | None = None,
        topic: str | None = None,
    ) -> list[NormalizedTable]:
        normalized_report_id = (
            str(report_id).strip() if report_id is not None else None
        )
        normalized_category = (
            str(category).strip().casefold() if category is not None else None
        )
        normalized_topic = (
            str(topic).strip().casefold() if topic is not None else None
        )

        results: list[NormalizedTable] = []
        for table in self._tables_by_id.values():
            if normalized_report_id is not None and (
                self._report_uid_by_table_id.get(table.table_id)
                != normalized_report_id
            ):
                continue
            if page_number is not None and table.page_number != int(page_number):
                continue
            metadata = dict(table.metadata or {})
            if normalized_category is not None:
                table_category = str(
                    metadata.get("category", "")
                ).strip().casefold()
                if table_category != normalized_category:
                    continue
            if normalized_topic is not None:
                topic_values = {
                    str(metadata.get("topic", "")).strip().casefold(),
                    *(
                        str(item).strip().casefold()
                        for item in metadata.get("topics", ()) or ()
                    ),
                }
                topic_values.discard("")
                if normalized_topic not in topic_values:
                    continue
            results.append(table)

        return sorted(
            results,
            key=lambda table: (
                table.page_number,
                table.table_index,
                table.table_id,
            ),
        )

    def load_report(
        self,
        report_uid: str,
        *,
        report_database_id: int,
        clear_existing: bool = False,
    ) -> int:
        """
        SQLite tablo kayıtlarını retrieval için bellek içi indekse yükler.
        """

        normalized_report_uid = str(report_uid or "").strip()
        if not normalized_report_uid:
            raise ValueError("report_uid boş olamaz.")
        if report_database_id < 1:
            raise ValueError("report_database_id pozitif olmalıdır.")

        records = list(
            self.database.get_tables(
                report_database_id
            )
            or []
        )

        if clear_existing:
            self._tables_by_id.clear()
            self._report_uid_by_table_id.clear()

        loaded = 0

        for record in records:
            table = self._table_from_database_record(
                record,
                report_uid=normalized_report_uid,
            )
            table_id = str(table.table_id).strip()
            if not table_id:
                continue

            self._tables_by_id[table_id] = table
            self._report_uid_by_table_id[
                table_id
            ] = normalized_report_uid
            loaded += 1

        return loaded

    @classmethod
    def _table_from_database_record(
        cls,
        record: Mapping[str, Any],
        *,
        report_uid: str,
    ) -> NormalizedTable:
        metadata = dict(record.get("metadata") or {})
        normalized_data = dict(
            record.get("normalized_data") or {}
        )

        table_id = str(
            metadata.get("source_table_id")
            or record.get("table_uid")
            or f"table-{record.get('id')}"
        ).strip()

        page_number = int(
            record.get("page_number")
            or 1
        )
        table_index = int(
            record.get("table_index")
            or 0
        )

        row_payloads = list(record.get("rows") or [])
        rows = tuple(
            cls._normalized_row_from_dict(row_payload)
            for row_payload in row_payloads
            if isinstance(row_payload, Mapping)
        )

        matrix_payload = normalized_data.get("matrix") or []
        matrix = tuple(
            tuple(str(cell) for cell in row)
            for row in matrix_payload
            if isinstance(row, (list, tuple))
        )

        if not matrix and rows:
            matrix = tuple(
                tuple(cell.normalized_text for cell in row.cells)
                for row in rows
            )

        metric_payloads = normalized_data.get(
            "metric_records"
        ) or []
        metric_records = tuple(
            cls._metric_record_from_dict(
                payload,
                fallback_table_id=table_id,
                fallback_page_number=page_number,
                fallback_table_index=table_index,
            )
            for payload in metric_payloads
            if isinstance(payload, Mapping)
        )

        headers = tuple(
            str(item)
            for item in (record.get("headers") or [])
        )
        normalized_headers = tuple(
            str(item)
            for item in (
                normalized_data.get("normalized_headers")
                or []
            )
        )

        metadata.update(
            {
                "report_id": report_uid,
                "original_report_uid": report_uid,
                "database_table_id": record.get("id"),
                "table_uid": record.get("table_uid"),
                "page_number": page_number,
                "table_index": table_index,
            }
        )

        numeric_cell_count = sum(
            cell.is_numeric
            for row in rows
            for cell in row.cells
        )
        empty_cell_count = sum(
            cell.is_empty
            for row in rows
            for cell in row.cells
        )

        return NormalizedTable(
            table_id=table_id,
            page_index=max(page_number - 1, 0),
            page_number=page_number,
            table_index=table_index,
            source_extraction_method=str(
                record.get("extraction_method")
                or "database_hydration"
            ),
            title=cls._optional_text(
                record.get("title")
            ),
            caption=cls._optional_text(
                record.get("caption")
            ),
            row_count=int(
                record.get("row_count")
                or len(rows)
            ),
            column_count=int(
                record.get("column_count")
                or max(
                    (len(row.cells) for row in rows),
                    default=0,
                )
            ),
            headers=headers,
            normalized_headers=normalized_headers,
            rows=rows,
            matrix=matrix,
            metric_records=metric_records,
            detected_years=tuple(
                int(value)
                for value in (
                    normalized_data.get("detected_years")
                    or []
                )
            ),
            detected_units=tuple(
                str(value)
                for value in (
                    normalized_data.get("detected_units")
                    or []
                )
            ),
            detected_currencies=tuple(
                str(value)
                for value in (
                    normalized_data.get("detected_currencies")
                    or []
                )
            ),
            default_unit=cls._optional_text(
                normalized_data.get("default_unit")
            ),
            default_year=cls._optional_int(
                normalized_data.get("default_year")
            ),
            numeric_cell_count=numeric_cell_count,
            empty_cell_count=empty_cell_count,
            normalization_confidence=float(
                record.get("confidence")
                or 0.0
            ),
            source_confidence=float(
                metadata.get("source_confidence")
                or record.get("confidence")
                or 0.0
            ),
            content_hash=str(
                normalized_data.get("content_hash")
                or hashlib.sha256(
                    repr(matrix).encode("utf-8")
                ).hexdigest()
            ),
            is_valid=bool(
                normalized_data.get("is_valid", True)
            ),
            warnings=tuple(
                str(item)
                for item in (
                    record.get("validation_issues")
                    or []
                )
            ),
            metadata=metadata,
        )

    @classmethod
    def _normalized_row_from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> NormalizedTableRow:
        cells = tuple(
            cls._normalized_cell_from_dict(cell)
            for cell in (payload.get("cells") or [])
            if isinstance(cell, Mapping)
        )

        return NormalizedTableRow(
            row_index=int(payload.get("row_index") or 0),
            cells=cells,
            row_label=cls._optional_text(
                payload.get("row_label")
            ),
            metric_name=cls._optional_text(
                payload.get("metric_name")
            ),
            metric_slug=cls._optional_text(
                payload.get("metric_slug")
            ),
            detected_years=tuple(
                int(value)
                for value in (
                    payload.get("detected_years")
                    or []
                )
            ),
            detected_units=tuple(
                str(value)
                for value in (
                    payload.get("detected_units")
                    or []
                )
            ),
            numeric_value_count=int(
                payload.get("numeric_value_count")
                or sum(cell.is_numeric for cell in cells)
            ),
            empty_cell_count=int(
                payload.get("empty_cell_count")
                or sum(cell.is_empty for cell in cells)
            ),
            is_header=bool(payload.get("is_header", False)),
            is_empty=bool(payload.get("is_empty", False)),
            searchable_text=str(
                payload.get("searchable_text")
                or " | ".join(
                    cell.search_text
                    for cell in cells
                    if cell.search_text
                )
            ),
        )

    @classmethod
    def _normalized_cell_from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> NormalizedTableCell:
        raw_value_type = str(
            payload.get("value_type")
            or "unknown"
        )

        try:
            value_type = NormalizedValueType(
                raw_value_type
            )
        except ValueError:
            value_type = NormalizedValueType.UNKNOWN

        return NormalizedTableCell(
            row_index=int(payload.get("row_index") or 0),
            column_index=int(
                payload.get("column_index")
                or 0
            ),
            raw_text=str(payload.get("raw_text") or ""),
            normalized_text=str(
                payload.get("normalized_text")
                or payload.get("raw_text")
                or ""
            ),
            search_text=str(
                payload.get("search_text")
                or payload.get("normalized_text")
                or payload.get("raw_text")
                or ""
            ),
            value_type=value_type,
            numeric_value=cls._optional_float(
                payload.get("numeric_value")
            ),
            lower_bound=cls._optional_float(
                payload.get("lower_bound")
            ),
            upper_bound=cls._optional_float(
                payload.get("upper_bound")
            ),
            unit=cls._optional_text(payload.get("unit")),
            currency=cls._optional_text(
                payload.get("currency")
            ),
            year=cls._optional_int(payload.get("year")),
            is_header=bool(payload.get("is_header", False)),
            is_empty=bool(payload.get("is_empty", False)),
            is_numeric=bool(
                payload.get("is_numeric", False)
            ),
            is_percentage=bool(
                payload.get("is_percentage", False)
            ),
            is_missing_value=bool(
                payload.get("is_missing_value", False)
            ),
            confidence=float(
                payload.get("confidence")
                or 0.0
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def _metric_record_from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        fallback_table_id: str,
        fallback_page_number: int,
        fallback_table_index: int,
    ) -> NormalizedMetricRecord:
        raw_value_type = str(
            payload.get("value_type")
            or "unknown"
        )
        try:
            value_type = NormalizedValueType(
                raw_value_type
            )
        except ValueError:
            value_type = NormalizedValueType.UNKNOWN

        return NormalizedMetricRecord(
            record_id=str(
                payload.get("record_id")
                or hashlib.sha256(
                    repr(sorted(payload.items())).encode(
                        "utf-8"
                    )
                ).hexdigest()
            ),
            table_id=str(
                payload.get("table_id")
                or fallback_table_id
            ),
            page_number=int(
                payload.get("page_number")
                or fallback_page_number
            ),
            table_index=int(
                payload.get("table_index")
                or fallback_table_index
            ),
            row_index=int(payload.get("row_index") or 0),
            column_index=int(
                payload.get("column_index")
                or 0
            ),
            metric_name=str(
                payload.get("metric_name")
                or ""
            ),
            metric_slug=str(
                payload.get("metric_slug")
                or ""
            ),
            raw_value=str(payload.get("raw_value") or ""),
            normalized_value=str(
                payload.get("normalized_value")
                or payload.get("raw_value")
                or ""
            ),
            numeric_value=cls._optional_float(
                payload.get("numeric_value")
            ),
            value_type=value_type,
            year=cls._optional_int(payload.get("year")),
            unit=cls._optional_text(payload.get("unit")),
            currency=cls._optional_text(
                payload.get("currency")
            ),
            is_percentage=bool(
                payload.get("is_percentage", False)
            ),
            confidence=float(
                payload.get("confidence")
                or 0.0
            ),
            context=str(payload.get("context") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def store_table(
        self,
        *,
        report_id: int,
        company_name: str,
        report_year: int,
        table: NormalizedTable,
        validation: TableNumericValidationResult | None = None,
        page_id: int | None = None,
    ) -> StoredTableResult:
        """
        Tek bir normalize edilmiş tabloyu veri tabanına ve dosya sistemine
        kaydeder.
        """

        if report_id < 1:
            raise ValueError(
                "report_id pozitif olmalıdır."
            )

        if not company_name.strip():
            raise ValueError(
                "company_name boş olamaz."
            )

        if table.page_number < 1:
            raise ValueError(
                "Tablo page_number değeri geçersiz."
            )

        if table.table_index < 0:
            raise ValueError(
                "Tablo table_index değeri negatif olamaz."
            )

        table_uid = self._build_table_uid(
            report_id=report_id,
            table=table,
        )

        validation_payload = (
            validation.to_dict()
            if validation is not None
            else None
        )

        normalized_payload = table.to_dict()

        metadata = self._build_database_metadata(
            table=table,
            validation=validation,
        )

        validation_issues = (
            [
                issue.to_dict()
                for issue in validation.issues
            ]
            if validation is not None
            else []
        )

        table_database_id = (
            self.database.upsert_table(
                table_uid=table_uid,
                report_id=report_id,
                page_id=page_id,
                page_number=table.page_number,
                table_index=table.table_index,
                title=table.title,
                caption=table.caption,
                extraction_method=(
                    table.source_extraction_method
                ),
                confidence=(
                    table.normalization_confidence
                ),
                row_count=table.row_count,
                column_count=table.column_count,
                bounding_box=self._extract_bounding_box(
                    table
                ),
                headers=list(table.headers),
                rows=[
                    row.to_dict()
                    for row in table.rows
                ],
                normalized_data={
                    "matrix": [
                        list(row)
                        for row in table.matrix
                    ],
                    "normalized_headers": list(
                        table.normalized_headers
                    ),
                    "metric_records": [
                        record.to_dict()
                        for record in table.metric_records
                    ],
                    "detected_years": list(
                        table.detected_years
                    ),
                    "detected_units": list(
                        table.detected_units
                    ),
                    "detected_currencies": list(
                        table.detected_currencies
                    ),
                    "default_year": table.default_year,
                    "default_unit": table.default_unit,
                    "content_hash": table.content_hash,
                    "is_valid": table.is_valid,
                },
                validation_issues=(
                    validation_issues
                ),
                metadata=metadata,
            )
        )

        normalized_artifact: StoredArtifact | None = None
        validation_artifact: StoredArtifact | None = None

        if self.save_json_artifacts:
            normalized_artifact = (
                self._save_normalized_artifact(
                    company_name=company_name,
                    report_year=report_year,
                    table_uid=table_uid,
                    table=table,
                )
            )

            if validation_payload is not None:
                validation_artifact = (
                    self._save_validation_artifact(
                        company_name=company_name,
                        report_year=report_year,
                        table_uid=table_uid,
                        validation=validation,
                    )
                )

        is_valid = (
            validation.is_valid
            if validation is not None
            else table.is_valid
        )

        result = StoredTableResult(
            table_database_id=(
                table_database_id
            ),
            table_uid=table_uid,
            report_id=report_id,
            page_number=table.page_number,
            table_index=table.table_index,
            normalized_artifact=(
                normalized_artifact
            ),
            validation_artifact=(
                validation_artifact
            ),
            metric_record_count=len(
                table.metric_records
            ),
            validation_issue_count=(
                len(validation.issues)
                if validation is not None
                else 0
            ),
            is_valid=is_valid,
            stored_at=datetime.now(
                UTC
            ),
        )

        logger.info(
            "Tablo repository kaydı tamamlandı: "
            "table_uid=%s, database_id=%d",
            table_uid,
            table_database_id,
        )

        return result

    def store_tables(
        self,
        *,
        report_id: int,
        company_name: str,
        report_year: int,
        tables: Sequence[NormalizedTable],
        validations_by_table_id: Mapping[
            str,
            TableNumericValidationResult,
        ]
        | None = None,
        page_ids_by_number: Mapping[
            int,
            int,
        ]
        | None = None,
        continue_on_error: bool = True,
    ) -> TableRepositorySummary:
        """
        Birden fazla normalize edilmiş tabloyu depolar.

        validations_by_table_id sözlüğünün anahtarı NormalizedTable.table_id
        değeridir.
        """

        validations_by_table_id = (
            validations_by_table_id or {}
        )

        page_ids_by_number = (
            page_ids_by_number or {}
        )

        results: list[
            StoredTableResult
        ] = []

        errors: list[str] = []

        for table in tables:
            validation = (
                validations_by_table_id.get(
                    table.table_id
                )
            )

            page_id = page_ids_by_number.get(
                table.page_number
            )

            try:
                result = self.store_table(
                    report_id=report_id,
                    company_name=company_name,
                    report_year=report_year,
                    table=table,
                    validation=validation,
                    page_id=page_id,
                )

            except Exception as exc:
                message = (
                    f"Tablo {table.table_id} "
                    f"depolanamadı: {exc}"
                )

                errors.append(message)

                logger.exception(
                    message
                )

                if not continue_on_error:
                    raise TableRepositoryError(
                        message
                    ) from exc

                continue

            results.append(result)

        return TableRepositorySummary(
            report_id=report_id,
            requested_table_count=len(
                tables
            ),
            stored_table_count=len(
                results
            ),
            failed_table_count=(
                len(tables)
                - len(results)
            ),
            valid_table_count=sum(
                result.is_valid
                for result in results
            ),
            invalid_table_count=sum(
                not result.is_valid
                for result in results
            ),
            metric_record_count=sum(
                result.metric_record_count
                for result in results
            ),
            validation_issue_count=sum(
                result.validation_issue_count
                for result in results
            ),
            results=tuple(results),
            errors=tuple(errors),
        )

    def get_report_tables(
        self,
        report_id: int,
        *,
        page_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Veri tabanındaki tablo kayıtlarını getirir.
        """

        return self.database.get_tables(
            report_id=report_id,
            page_number=page_number,
        )

    def get_metric_records(
        self,
        report_id: int,
        *,
        page_number: int | None = None,
        only_valid_tables: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Veri tabanındaki normalized_data alanlarından metrik kayıtlarını
        toplar.
        """

        table_records = self.get_report_tables(
            report_id=report_id,
            page_number=page_number,
        )

        metric_records: list[
            dict[str, Any]
        ] = []

        for table_record in table_records:
            normalized_data = table_record.get(
                "normalized_data",
                {},
            )

            if not isinstance(
                normalized_data,
                Mapping,
            ):
                continue

            if (
                only_valid_tables
                and not bool(
                    normalized_data.get(
                        "is_valid",
                        False,
                    )
                )
            ):
                continue

            raw_metric_records = (
                normalized_data.get(
                    "metric_records",
                    [],
                )
            )

            if not isinstance(
                raw_metric_records,
                Sequence,
            ):
                continue

            for metric_record in raw_metric_records:
                if not isinstance(
                    metric_record,
                    Mapping,
                ):
                    continue

                metric_records.append(
                    {
                        **dict(metric_record),
                        "database_table_id": (
                            table_record.get("id")
                        ),
                        "table_uid": (
                            table_record.get(
                                "table_uid"
                            )
                        ),
                        "report_id": report_id,
                    }
                )

        return metric_records

    def find_metrics(
        self,
        report_id: int,
        query: str,
        *,
        year: int | None = None,
        unit: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Normalize edilmiş metrik kayıtları üzerinde basit metinsel arama
        yapar.

        Bu metot retrieval katmanının yerine geçmez. Yalnızca repository
        seviyesinde hızlı ve deterministik arama sağlar.
        """

        if not query.strip():
            raise ValueError(
                "query boş olamaz."
            )

        if limit < 1:
            raise ValueError(
                "limit pozitif olmalıdır."
            )

        normalized_query = (
            self._normalize_search_text(
                query
            )
        )

        records = self.get_metric_records(
            report_id=report_id,
        )

        matches: list[
            tuple[float, dict[str, Any]]
        ] = []

        query_tokens = set(
            normalized_query.split()
        )

        for record in records:
            if (
                year is not None
                and record.get("year") != year
            ):
                continue

            if (
                unit is not None
                and record.get("unit") != unit
            ):
                continue

            metric_name = str(
                record.get(
                    "metric_name",
                    "",
                )
            )

            context = str(
                record.get(
                    "context",
                    "",
                )
            )

            searchable_text = (
                self._normalize_search_text(
                    f"{metric_name} {context}"
                )
            )

            if not searchable_text:
                continue

            searchable_tokens = set(
                searchable_text.split()
            )

            overlap_count = len(
                query_tokens
                & searchable_tokens
            )

            exact_bonus = (
                1.0
                if normalized_query
                in searchable_text
                else 0.0
            )

            score = (
                overlap_count
                / max(
                    len(query_tokens),
                    1,
                )
                + exact_bonus
            )

            if score <= 0:
                continue

            matches.append(
                (
                    score,
                    record,
                )
            )

        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            {
                **record,
                "repository_score": round(
                    score,
                    4,
                ),
            }
            for score, record in matches[
                :limit
            ]
        ]

    def load_normalized_artifact(
        self,
        artifact_path: str | Path,
    ) -> dict[str, Any]:
        """Daha önce saklanmış normalize tablo JSON'unu okur."""

        artifact_path = Path(
            artifact_path
        ).expanduser().resolve()

        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Tablo artifact dosyası bulunamadı: {artifact_path}"
            )

        try:
            with artifact_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

        except json.JSONDecodeError as exc:
            raise TableRepositoryError(
                "Tablo artifact dosyası geçerli JSON değil."
            ) from exc

        except OSError as exc:
            raise TableRepositoryError(
                "Tablo artifact dosyası okunamadı."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise TableRepositoryError(
                "Tablo artifact kök nesnesi dictionary olmalıdır."
            )

        return payload

    def _save_normalized_artifact(
        self,
        *,
        company_name: str,
        report_year: int,
        table_uid: str,
        table: NormalizedTable,
    ) -> StoredArtifact:
        payload = {
            "artifact_type": (
                "normalized_table"
            ),
            "schema_version": "1.0",
            "created_at": datetime.now(
                UTC
            ).isoformat(),
            "table": table.to_dict(),
        }

        relative_path = (
            Path("tables")
            / "normalized"
            / (
                f"page_{table.page_number:04d}"
                f"_table_{table.table_index + 1:03d}"
                f"_{table_uid[:12]}.json"
            )
        )

        return self._write_json_artifact(
            company_name=company_name,
            report_year=report_year,
            relative_path=relative_path,
            payload=payload,
            page_number=table.page_number,
            metadata={
                "table_id": table.table_id,
                "table_uid": table_uid,
                "content_hash": (
                    table.content_hash
                ),
                "normalization_confidence": (
                    table.normalization_confidence
                ),
                "metric_record_count": len(
                    table.metric_records
                ),
            },
        )

    def _save_validation_artifact(
        self,
        *,
        company_name: str,
        report_year: int,
        table_uid: str,
        validation: TableNumericValidationResult,
    ) -> StoredArtifact:
        payload = {
            "artifact_type": (
                "table_numeric_validation"
            ),
            "schema_version": "1.0",
            "created_at": datetime.now(
                UTC
            ).isoformat(),
            "validation": validation.to_dict(),
        }

        relative_path = (
            Path("tables")
            / "validation"
            / (
                f"page_{validation.page_number:04d}"
                f"_table_{validation.table_index + 1:03d}"
                f"_{table_uid[:12]}.json"
            )
        )

        return self._write_json_artifact(
            company_name=company_name,
            report_year=report_year,
            relative_path=relative_path,
            payload=payload,
            page_number=validation.page_number,
            metadata={
                "table_uid": table_uid,
                "validation_score": (
                    validation.validation_score
                ),
                "is_valid": validation.is_valid,
                "issue_count": (
                    validation.issue_count
                ),
            },
        )

    def _write_json_artifact(
        self,
        *,
        company_name: str,
        report_year: int,
        relative_path: Path,
        payload: Mapping[str, Any],
        page_number: int,
        metadata: Mapping[str, Any],
    ) -> StoredArtifact:
        """JSON tablo artifact'ını FileStore ortak API'siyle yazar."""

        report_paths = self.file_store.get_report_paths(
            company_name=company_name,
            report_year=report_year,
        )
        target_path = report_paths.report_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        return self.file_store.save_json(
            self._serialize_value(payload),
            target_path,
            overwrite=self.overwrite_artifacts,
            artifact_type=ArtifactType.TABLE,
            company_slug=self.file_store.slugify(company_name),
            report_year=report_year,
            page_number=page_number,
            metadata=dict(metadata),
        )

    @staticmethod
    def _write_text_atomic(
        *,
        target_path: Path,
        content: str,
        overwrite: bool,
    ) -> None:
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Artifact zaten mevcut: {target_path}"
            )

        temporary_path = target_path.with_suffix(
            target_path.suffix + ".part"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                file.write(content)
                file.flush()

            temporary_path.replace(
                target_path
            )

        except OSError as exc:
            raise TableRepositoryError(
                f"Artifact yazılamadı: {target_path}"
            ) from exc

        finally:
            temporary_path.unlink(
                missing_ok=True
            )

    @staticmethod
    def _build_table_uid(
        *,
        report_id: int,
        table: NormalizedTable,
    ) -> str:
        # Tablo kimliği içerikten bağımsız ve yeniden çalıştırmalarda
        # kararlı olmalıdır. OCR veya normalizasyon sonucu değişse bile aynı
        # rapor/sayfa/tablo konumu aynı UID altında güncellenir.
        canonical = (
            f"{report_id}:"
            f"{table.page_number}:"
            f"{table.table_index}"
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _build_database_metadata(
        *,
        table: NormalizedTable,
        validation: TableNumericValidationResult | None,
    ) -> dict[str, Any]:
        # Ingestion sırasında üretilen category/topic/topics gibi
        # sınıflandırma alanlarını kaybetme.
        metadata = dict(table.metadata or {})

        topics = metadata.get("topics")
        if isinstance(topics, tuple):
            metadata["topics"] = list(topics)

        metadata.update({
            "source_table_id": table.table_id,
            "source_extraction_method": (
                table.source_extraction_method
            ),
            "normalization_confidence": (
                table.normalization_confidence
            ),
            "source_confidence": (
                table.source_confidence
            ),
            "numeric_cell_count": (
                table.numeric_cell_count
            ),
            "empty_cell_count": (
                table.empty_cell_count
            ),
            "metric_record_count": len(
                table.metric_records
            ),
            "normalization_warnings": list(
                table.warnings
            ),
            "is_valid": table.is_valid,
        })

        if validation is not None:
            metadata["numeric_validation"] = {
                "validation_score": (
                    validation.validation_score
                ),
                "is_valid": validation.is_valid,
                "issue_count": (
                    validation.issue_count
                ),
                "error_count": (
                    validation.error_count
                ),
                "warning_count": (
                    validation.warning_count
                ),
                "info_count": (
                    validation.info_count
                ),
            }

        return metadata

    @staticmethod
    def _extract_bounding_box(
        table: NormalizedTable,
    ) -> dict[str, Any]:
        metadata = table.metadata

        if not isinstance(
            metadata,
            Mapping,
        ):
            return {}

        for key in (
            "bounding_box",
            "pdf_bounding_box",
            "source_bounding_box",
        ):
            value = metadata.get(key)

            if isinstance(
                value,
                Mapping,
            ):
                return dict(value)

        return {}

    @classmethod
    def _serialize_value(
        cls,
        value: Any,
    ) -> Any:
        if is_dataclass(value):
            return cls._serialize_value(
                asdict(value)
            )

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Mapping):
            return {
                str(key): cls._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        ):
            return [
                cls._serialize_value(item)
                for item in value
            ]

        return value

    @staticmethod
    def _normalize_search_text(
        value: str,
    ) -> str:
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

        normalized = value.translate(
            translation_table
        ).casefold()

        normalized = "".join(
            character
            if character.isalnum()
            or character.isspace()
            else " "
            for character in normalized
        )

        return " ".join(
            normalized.split()
        )


def store_normalized_table(
    *,
    database: Database,
    file_store: FileStore,
    report_id: int,
    company_name: str,
    report_year: int,
    table: NormalizedTable,
    validation: TableNumericValidationResult | None = None,
    page_id: int | None = None,
) -> StoredTableResult:
    """TableRepository için kısa kullanım arayüzü."""

    repository = TableRepository(
        database=database,
        file_store=file_store,
    )

    return repository.store_table(
        report_id=report_id,
        company_name=company_name,
        report_year=report_year,
        table=table,
        validation=validation,
        page_id=page_id,
    )