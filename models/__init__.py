"""
Sustainability Analyzer veri modelleri.

Bu modül, proje genelinde kullanılan veri modellerini tek bir import
noktası üzerinden erişilebilir hale getirir.

Örnek:

    from models import Report, Page, Chunk

yerine her modeli kendi dosyasından ayrı ayrı import etmek zorunda
kalınmaz.
"""

from .chunk import (
    Chunk,
    ChunkCategory,
    ChunkSourceType,
)
from .evidence import (
    AnswerResult,
    AnswerStatus,
    Citation,
    EvidenceBundle,
    EvidenceItem,
    EvidenceSourceType,
    EvidenceValidationStatus,
    RetrievalChannel,
    RetrievalScore,
)
from .page import (
    BoundingBox,
    LayoutElement,
    LayoutElementType,
    Page,
    PageType,
)
from .query import (
    AnswerType,
    QueryAnalysis,
    QueryComplexity,
    QueryFilter,
    QueryIntent,
    RetrievalRoute,
)
from .report import (
    Report,
    ReportCandidate,
    ReportStatus,
)
from .visual_evidence import (
    OCRLine,
    OCRWord,
    VisualAsset,
    VisualExtractionMethod,
    VisualMetric,
    VisualProcessingStatus,
    VisualRegionType,
    WholeImageOCRResult,
)
from .table import (
    ExtractedTable,
    ScalarValue,
    TableCell,
    TableCellDataType,
    TableExtractionMethod,
    TableValidationIssue,
    ValidationSeverity,
)


__all__ = [
    # Report modelleri
    "Report",
    "ReportCandidate",
    "ReportStatus",

    # Page modelleri
    "BoundingBox",
    "LayoutElement",
    "LayoutElementType",
    "Page",
    "PageType",


    # Görsel kanıt modelleri
    "VisualRegionType",
    "VisualProcessingStatus",
    "VisualExtractionMethod",
    "OCRWord",
    "OCRLine",
    "WholeImageOCRResult",
    "VisualAsset",
    "VisualMetric",

    # Table modelleri
    "ExtractedTable",
    "ScalarValue",
    "TableCell",
    "TableCellDataType",
    "TableExtractionMethod",
    "TableValidationIssue",
    "ValidationSeverity",

    # Chunk modelleri
    "Chunk",
    "ChunkCategory",
    "ChunkSourceType",

    # Query modelleri
    "AnswerType",
    "QueryAnalysis",
    "QueryComplexity",
    "QueryFilter",
    "QueryIntent",
    "RetrievalRoute",

    # Evidence modelleri
    "AnswerResult",
    "AnswerStatus",
    "Citation",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceSourceType",
    "EvidenceValidationStatus",
    "RetrievalChannel",
    "RetrievalScore",
]