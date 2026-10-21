"""Public API for storage, persistence, cache and export components."""

from .cache import (
    CacheEntry,
    CacheError,
    CacheNamespace,
    CacheStats,
    FileCache,
    create_cache,
)
from .database import (
    Database,
    DatabaseError,
    RecordNotFoundError,
    create_database,
)
from .excel_exporter import (
    ExcelAnswerRow,
    ExcelExporter,
)
from .file_store import (
    ArtifactType,
    FileStore,
    FileStoreError,
    ReportPaths,
    StoredArtifact,
    create_file_store,
)
from .vector_store import (
    EmbeddingProviderProtocol,
    HashingEmbeddingProvider,
    VectorDimensionError,
    VectorDocument,
    VectorDocumentNotFoundError,
    VectorSearchResult,
    VectorStore,
    VectorStoreConfig,
    VectorStoreError,
    VectorStoreStats,
)
from .visual_asset_repository import (
    StoredVisualAssetResult,
    StoredVisualMetricResult,
    VisualAssetRepository,
    VisualAssetRepositoryError,
    VisualAssetRepositorySummary,
)

__all__ = [
    "ArtifactType",
    "CacheEntry",
    "CacheError",
    "CacheNamespace",
    "CacheStats",
    "Database",
    "DatabaseError",
    "EmbeddingProviderProtocol",
    "ExcelAnswerRow",
    "ExcelExporter",
    "FileCache",
    "FileStore",
    "FileStoreError",
    "HashingEmbeddingProvider",
    "RecordNotFoundError",
    "ReportPaths",
    "StoredArtifact",
    "StoredVisualAssetResult",
    "StoredVisualMetricResult",
    "VectorDimensionError",
    "VectorDocument",
    "VectorDocumentNotFoundError",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreError",
    "VectorStoreStats",
    "VisualAssetRepository",
    "VisualAssetRepositoryError",
    "VisualAssetRepositorySummary",
    "create_cache",
    "create_database",
    "create_file_store",
]
