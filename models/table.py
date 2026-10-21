from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .page import BoundingBox


# Tablo hücrelerinde tutulabilecek temel veri tipleri
ScalarValue = str | int | float | bool | None


class TableExtractionMethod(str, Enum):
    """
    Tablonun hangi yöntemle çıkarıldığını belirtir.
    """

    NATIVE = "native"
    VISUAL = "visual"
    OCR = "ocr"
    VLM = "vlm"


class TableCellDataType(str, Enum):
    """
    Bir tablo hücresindeki değerin tahmin edilen veri türü.
    """

    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    DATE = "date"
    BOOLEAN = "boolean"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """
    Tablo doğrulama probleminin önem seviyesini belirtir.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TableCell(BaseModel):
    """
    Bir tablo içerisindeki tek bir hücreyi temsil eder.

    Bu model özellikle görsel tablo çıkarımı, OCR ve VLM çıktılarında
    hücre konumlarını ve güven skorlarını saklamak için kullanılacaktır.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    row_index: int = Field(
        ge=0,
    )

    column_index: int = Field(
        ge=0,
    )

    raw_text: str = ""

    normalized_value: ScalarValue = None

    data_type: TableCellDataType = TableCellDataType.UNKNOWN

    bounding_box: BoundingBox | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    is_header: bool = False

    row_span: int = Field(
        default=1,
        ge=1,
    )

    column_span: int = Field(
        default=1,
        ge=1,
    )

    unit: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("raw_text")
    @classmethod
    def normalize_raw_text(
        cls,
        value: str,
    ) -> str:
        """
        Hücre metninin başındaki ve sonundaki boşlukları temizler.
        """

        return value.strip()

    @field_validator("unit")
    @classmethod
    def normalize_unit(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş birim değerlerini None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_unit = value.strip()

        if not normalized_unit:
            return None

        return normalized_unit


class TableValidationIssue(BaseModel):
    """
    numeric_validator.py veya table_normalizer.py tarafından tespit edilen
    tablo problemlerini temsil eder.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    code: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
    )

    severity: ValidationSeverity = ValidationSeverity.WARNING

    row_index: int | None = Field(
        default=None,
        ge=0,
    )

    column_index: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ExtractedTable(BaseModel):
    """
    PDF içerisinden çıkarılan ve normalize edilen bir tabloyu temsil eder.

    Bu model aşağıdaki modüller arasında ortak veri yapısı olacaktır:

    - native_table_extractor.py
    - visual_table_extractor.py
    - table_ocr.py
    - vlm_table_fallback.py
    - table_normalizer.py
    - numeric_validator.py
    - table_repository.py
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    report_id: UUID

    page_id: UUID | None = None

    page_number: int = Field(
        ge=1,
    )

    title: str | None = None

    caption: str | None = None

    extraction_method: TableExtractionMethod

    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    bounding_box: BoundingBox | None = None

    # Normalize edilmiş sütun başlıkları
    headers: list[str] = Field(
        default_factory=list,
    )

    # Normalize edilmiş tablo satırları
    rows: list[list[ScalarValue]] = Field(
        default_factory=list,
    )

    # Hücre bazlı detaylı çıktı
    cells: list[TableCell] = Field(
        default_factory=list,
    )

    section_title: str | None = None

    subsection_title: str | None = None

    category: str | None = None

    units: dict[str, str] = Field(
        default_factory=dict,
    )

    standard_references: list[str] = Field(
        default_factory=list,
    )

    raw_text: str | None = None

    is_normalized: bool = False

    validation_issues: list[TableValidationIssue] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "title",
        "caption",
        "section_title",
        "subsection_title",
        "category",
        "raw_text",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş metin değerlerini None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_text = value.strip()

        if not normalized_text:
            return None

        return normalized_text

    @field_validator("headers")
    @classmethod
    def normalize_headers(
        cls,
        value: list[str],
    ) -> list[str]:
        """
        Sütun başlıklarını temizler.

        Boş sütun başlıklarına otomatik bir isim verir.
        """

        normalized_headers: list[str] = []

        for index, header in enumerate(value):
            normalized_header = header.strip()

            if not normalized_header:
                normalized_header = f"column_{index + 1}"

            normalized_headers.append(normalized_header)

        return normalized_headers

    @field_validator("standard_references")
    @classmethod
    def normalize_standard_references(
        cls,
        value: list[str],
    ) -> list[str]:
        """
        GRI veya ESRS referanslarındaki tekrarları kaldırır.
        """

        unique_references: list[str] = []

        for reference in value:
            normalized_reference = reference.strip()

            if (
                normalized_reference
                and normalized_reference not in unique_references
            ):
                unique_references.append(normalized_reference)

        return unique_references

    @model_validator(mode="after")
    def validate_table_dimensions(
        self,
    ) -> ExtractedTable:
        """
        Normalize edilmiş tablo satırlarının sütun sayısını kontrol eder.

        Başlıklar mevcutsa her satırın başlık sayısı kadar hücre
        içermesi gerekir.
        """

        if not self.headers:
            return self

        expected_column_count = len(self.headers)

        invalid_row_indexes = [
            row_index
            for row_index, row in enumerate(self.rows)
            if len(row) != expected_column_count
        ]

        if invalid_row_indexes:
            raise ValueError(
                "Tablo satırlarının sütun sayısı başlık sayısıyla "
                "uyuşmuyor. Hatalı satır indeksleri: "
                f"{invalid_row_indexes}"
            )

        return self

    @property
    def row_count(self) -> int:
        """
        Normalize edilmiş veri satırı sayısını döndürür.
        """

        return len(self.rows)

    @property
    def column_count(self) -> int:
        """
        Tablo sütun sayısını döndürür.
        """

        if self.headers:
            return len(self.headers)

        if self.rows:
            return max(
                len(row)
                for row in self.rows
            )

        if self.cells:
            return (
                max(
                    cell.column_index
                    for cell in self.cells
                )
                + 1
            )

        return 0

    @property
    def has_errors(self) -> bool:
        """
        Tabloda ERROR seviyesinde doğrulama problemi olup olmadığını belirtir.
        """

        return any(
            issue.severity == ValidationSeverity.ERROR
            for issue in self.validation_issues
        )

    @property
    def has_warnings(self) -> bool:
        """
        Tabloda WARNING seviyesinde problem olup olmadığını belirtir.
        """

        return any(
            issue.severity == ValidationSeverity.WARNING
            for issue in self.validation_issues
        )

    @property
    def is_empty(self) -> bool:
        """
        Tabloda herhangi bir kullanılabilir veri bulunup bulunmadığını belirtir.
        """

        return not self.rows and not self.cells

    def add_cell(
        self,
        cell: TableCell,
    ) -> None:
        """
        Tabloya yeni bir hücre ekler.

        Aynı satır ve sütun indeksine sahip ikinci bir hücrenin
        eklenmesine izin verilmez.
        """

        duplicate_exists = any(
            existing_cell.row_index == cell.row_index
            and existing_cell.column_index == cell.column_index
            for existing_cell in self.cells
        )

        if duplicate_exists:
            raise ValueError(
                "Aynı satır ve sütun konumunda zaten bir hücre bulunuyor. "
                f"Konum: ({cell.row_index}, {cell.column_index})"
            )

        self.cells.append(cell)

        self.cells.sort(
            key=lambda item: (
                item.row_index,
                item.column_index,
            )
        )

    def add_validation_issue(
        self,
        code: str,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.WARNING,
        row_index: int | None = None,
        column_index: int | None = None,
    ) -> None:
        """
        Tabloya yeni bir doğrulama problemi ekler.
        """

        issue = TableValidationIssue(
            code=code,
            message=message,
            severity=severity,
            row_index=row_index,
            column_index=column_index,
        )

        self.validation_issues.append(issue)

    def mark_normalized(
        self,
        headers: list[str],
        rows: list[list[ScalarValue]],
    ) -> None:
        """
        table_normalizer.py sonucunda normalize edilmiş tabloyu kaydeder.
        """

        if not headers:
            raise ValueError(
                "Normalize edilmiş tablo en az bir sütun başlığı içermelidir."
            )

        expected_column_count = len(headers)

        invalid_row_indexes = [
            row_index
            for row_index, row in enumerate(rows)
            if len(row) != expected_column_count
        ]

        if invalid_row_indexes:
            raise ValueError(
                "Normalize edilen satırların sütun sayısı başlıklarla "
                "uyuşmuyor. Hatalı satırlar: "
                f"{invalid_row_indexes}"
            )

        self.headers = headers
        self.rows = rows
        self.is_normalized = True

    def get_column(
        self,
        column_name: str,
    ) -> list[ScalarValue]:
        """
        Başlık adına göre bir tablo sütununu döndürür.
        """

        normalized_name = column_name.strip()

        if normalized_name not in self.headers:
            raise KeyError(
                f"'{normalized_name}' isimli sütun tabloda bulunamadı."
            )

        column_index = self.headers.index(normalized_name)

        return [
            row[column_index]
            for row in self.rows
        ]

    def to_records(
        self,
    ) -> list[dict[str, ScalarValue]]:
        """
        Tabloyu sözlük listesine dönüştürür.

        Örnek çıktı:

        [
            {
                "Metric": "Scope 1",
                "2023": 1250
            }
        ]
        """

        if not self.headers:
            raise ValueError(
                "Tabloyu kayıtlara dönüştürmek için sütun başlıkları gereklidir."
            )

        return [
            dict(
                zip(
                    self.headers,
                    row,
                    strict=True,
                )
            )
            for row in self.rows
        ]

    def get_cells_by_row(
        self,
        row_index: int,
    ) -> list[TableCell]:
        """
        Belirli bir satırdaki hücreleri sütun sırasıyla döndürür.
        """

        return sorted(
            [
                cell
                for cell in self.cells
                if cell.row_index == row_index
            ],
            key=lambda cell: cell.column_index,
        )