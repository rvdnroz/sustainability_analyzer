"""
Sustainability Analyzer tablo işleme katmanı.

Bu paket, sürdürülebilirlik raporlarındaki tabloların tespit edilmesi,
çıkarılması, OCR ile okunması, normalize edilmesi, doğrulanması ve
kalıcı depolama katmanına aktarılmasını sağlar.

Tablo işleme hattı
------------------

1. table_detector
    PDF sayfalarındaki tablo bölgelerini tespit eder.

2. native_table_extractor
    Metin tabanlı ve yapısal PDF tablolarını PyMuPDF üzerinden çıkarır.

3. visual_table_extractor
    Native yapısı bulunmayan tabloları OpenCV ile hücrelere ayırır.

4. table_ocr
    Görsel tablo hücrelerini Tesseract OCR ile metne dönüştürür.

5. vlm_table_fallback
    Native ve OCR yöntemleri yetersiz kaldığında Gemini tabanlı
    görsel tablo çıkarımı gerçekleştirir.

6. table_normalizer
    Farklı extraction yollarından gelen tabloları ortak veri modeline
    dönüştürür; sayı, yıl ve birimleri normalize eder.

7. numeric_validator
    Sayısal değerlerin yıl, birim, aralık ve seri tutarlılığını denetler.

8. table_repository
    Normalize edilmiş tabloları SQLite ve dosya sistemi üzerinde saklar.

Örnek kullanım
--------------

    from table_processing import (
        NativeTableExtractor,
        NumericValidator,
        TableDetector,
        TableNormalizer,
        TableRepository,
        VisualTableExtractor,
    )

    detector = TableDetector()
    native_extractor = NativeTableExtractor()
    visual_extractor = VisualTableExtractor()
    normalizer = TableNormalizer()
    validator = NumericValidator()
"""

from __future__ import annotations

# Alt modüller paket düzeyinde erişilebilir tutulur.
from . import (
    borderless_table_extractor,
    native_table_extractor,
    numeric_validator,
    table_detector,
    table_normalizer,
    table_ocr,
    table_repository,
    visual_table_extractor,
)

from .native_table_extractor import (
    NativeExtractedTable,
    NativeTableCell,
    NativeTableExtractionError,
    NativeTableExtractor,
    NativeTableRow,
    PageNativeTableExtraction,
    extract_native_tables,
)
from .borderless_table_extractor import (
    BorderlessExtractedTable,
    BorderlessTableCell,
    BorderlessTableExtractionError,
    BorderlessTableExtractor,
    BorderlessTableRow,
    PageBorderlessTableExtraction,
    extract_borderless_tables,
)
from .numeric_validator import (
    NumericValidationError,
    NumericValidationIssue,
    NumericValidator,
    RecordValidationResult,
    TableNumericValidationResult,
    ValidationIssueType,
    ValidationSeverity,
    validate_numeric_table,
)
from .table_detector import (
    DetectedTable,
    PageTableDetection,
    TableBoundingBox,
    TableDetectionError,
    TableDetector,
    detect_pdf_tables,
)
from .table_normalizer import (
    NormalizedMetricRecord,
    NormalizedTable,
    NormalizedTableCell,
    NormalizedTableRow,
    NormalizedValueType,
    ParsedNumericValue,
    TableNormalizationError,
    TableNormalizer,
    normalize_table,
)
from .table_ocr import (
    OCRCellResult,
    OCRExtractedTable,
    OCRTableRow,
    OCRWord,
    PageTableOCRResult,
    TableOCR,
    TableOCRError,
    run_table_ocr,
)
from .table_repository import (
    StoredTableResult,
    TableRepository,
    TableRepositoryError,
    TableRepositorySummary,
    store_normalized_table,
)
from .visual_table_extractor import (
    PageVisualTableExtraction,
    PixelBoundingBox,
    VisualExtractedTable,
    VisualTableCell,
    VisualTableExtractionError,
    VisualTableExtractor,
    extract_visual_tables,
)
# Gemini VLM fallback isteğe bağlıdır. google-genai kurulu olmadığında paket
# düzeyindeki diğer tablo bileşenleri kullanılmaya devam edebilmelidir.
VLM_TABLE_FALLBACK_AVAILABLE = False
VLM_TABLE_FALLBACK_IMPORT_ERROR: ImportError | None = None

try:
    from . import vlm_table_fallback
    from .vlm_table_fallback import (
        VLMExtractedCell,
        VLMExtractedRow,
        VLMExtractedTable,
        VLMFallbackDecision,
        VLMTableCellSchema,
        VLMTableFallback,
        VLMTableFallbackError,
        VLMTableSchema,
        extract_table_with_vlm,
    )
    VLM_TABLE_FALLBACK_AVAILABLE = True
except ImportError as exc:
    vlm_table_fallback = None  # type: ignore[assignment]
    VLM_TABLE_FALLBACK_IMPORT_ERROR = exc

    VLMExtractedCell = None  # type: ignore[assignment,misc]
    VLMExtractedRow = None  # type: ignore[assignment,misc]
    VLMExtractedTable = None  # type: ignore[assignment,misc]
    VLMFallbackDecision = None  # type: ignore[assignment,misc]
    VLMTableCellSchema = None  # type: ignore[assignment,misc]
    VLMTableFallback = None  # type: ignore[assignment,misc]
    VLMTableSchema = None  # type: ignore[assignment,misc]

    class VLMTableFallbackError(RuntimeError):
        """Opsiyonel Gemini VLM bağımlılığı kullanılamadığında oluşur."""

    def extract_table_with_vlm(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise VLMTableFallbackError(
            "Gemini VLM fallback kullanılamıyor. google-genai paketini "
            "kurun veya VLM fallback'i kapalı çalıştırın."
        ) from VLM_TABLE_FALLBACK_IMPORT_ERROR


__all__ = [
    # Alt modüller
    "table_detector",
    "native_table_extractor",
    "borderless_table_extractor",
    "visual_table_extractor",
    "table_ocr",
    "vlm_table_fallback",
    "VLM_TABLE_FALLBACK_AVAILABLE",
    "VLM_TABLE_FALLBACK_IMPORT_ERROR",
    "table_normalizer",
    "numeric_validator",
    "table_repository",

    # Table detector
    "TableDetectionError",
    "TableBoundingBox",
    "DetectedTable",
    "PageTableDetection",
    "TableDetector",
    "detect_pdf_tables",

    # Native table extractor
    "NativeTableExtractionError",
    "NativeTableCell",
    "NativeTableRow",
    "NativeExtractedTable",
    "PageNativeTableExtraction",
    "NativeTableExtractor",
    "extract_native_tables",

    # Borderless table extractor
    "BorderlessTableExtractionError",
    "BorderlessTableCell",
    "BorderlessTableRow",
    "BorderlessExtractedTable",
    "PageBorderlessTableExtraction",
    "BorderlessTableExtractor",
    "extract_borderless_tables",

    # Visual table extractor
    "VisualTableExtractionError",
    "PixelBoundingBox",
    "VisualTableCell",
    "VisualExtractedTable",
    "PageVisualTableExtraction",
    "VisualTableExtractor",
    "extract_visual_tables",

    # Table OCR
    "TableOCRError",
    "OCRWord",
    "OCRCellResult",
    "OCRTableRow",
    "OCRExtractedTable",
    "PageTableOCRResult",
    "TableOCR",
    "run_table_ocr",

    # VLM fallback
    "VLMTableFallbackError",
    "VLMTableCellSchema",
    "VLMTableSchema",
    "VLMExtractedCell",
    "VLMExtractedRow",
    "VLMExtractedTable",
    "VLMFallbackDecision",
    "VLMTableFallback",
    "extract_table_with_vlm",

    # Table normalizer
    "TableNormalizationError",
    "NormalizedValueType",
    "ParsedNumericValue",
    "NormalizedTableCell",
    "NormalizedTableRow",
    "NormalizedMetricRecord",
    "NormalizedTable",
    "TableNormalizer",
    "normalize_table",

    # Numeric validator
    "NumericValidationError",
    "ValidationSeverity",
    "ValidationIssueType",
    "NumericValidationIssue",
    "RecordValidationResult",
    "TableNumericValidationResult",
    "NumericValidator",
    "validate_numeric_table",

    # Table repository
    "TableRepositoryError",
    "StoredTableResult",
    "TableRepositorySummary",
    "TableRepository",
    "store_normalized_table",
]