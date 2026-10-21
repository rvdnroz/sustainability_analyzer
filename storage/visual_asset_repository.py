from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID

from models.visual_evidence import (
    VisualAsset,
    VisualMetric,
    VisualProcessingStatus,
    VisualRegionType,
    WholeImageOCRResult,
)
from .database import Database
from storage.file_store import (
    ArtifactType,
    FileStore,
    StoredArtifact,
)


logger = logging.getLogger(__name__)


class VisualAssetRepositoryError(RuntimeError):
    """Görsel varlık repository işlemi başarısız olduğunda yükseltilir."""


@dataclass(slots=True, frozen=True)
class StoredVisualAssetResult:
    """Kalıcı depolamaya yazılan tek bir görsel varlık sonucu."""

    visual_database_id: int
    visual_uid: str

    report_database_id: int
    report_uid: str
    page_number: int

    processing_status: str
    requires_vision: bool

    artifact: StoredArtifact | None
    stored_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_database_id": self.visual_database_id,
            "visual_uid": self.visual_uid,
            "report_database_id": self.report_database_id,
            "report_uid": self.report_uid,
            "page_number": self.page_number,
            "processing_status": self.processing_status,
            "requires_vision": self.requires_vision,
            "artifact": (
                self.artifact.to_dict()
                if self.artifact is not None
                else None
            ),
            "stored_at": self.stored_at.isoformat(),
        }


@dataclass(slots=True, frozen=True)
class StoredVisualMetricResult:
    """Kalıcı depolamaya yazılan tek bir görsel metrik sonucu."""

    metric_database_id: int
    metric_uid: str

    visual_uid: str
    report_database_id: int
    report_uid: str
    page_number: int

    metric_key: str
    confidence: float

    artifact: StoredArtifact | None
    stored_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_database_id": self.metric_database_id,
            "metric_uid": self.metric_uid,
            "visual_uid": self.visual_uid,
            "report_database_id": self.report_database_id,
            "report_uid": self.report_uid,
            "page_number": self.page_number,
            "metric_key": self.metric_key,
            "confidence": self.confidence,
            "artifact": (
                self.artifact.to_dict()
                if self.artifact is not None
                else None
            ),
            "stored_at": self.stored_at.isoformat(),
        }


@dataclass(slots=True, frozen=True)
class VisualAssetRepositorySummary:
    """Birden fazla görsel varlık saklama işleminin özeti."""

    requested_count: int
    stored_count: int
    failed_count: int

    vision_required_count: int
    ocr_completed_count: int
    ocr_partial_count: int

    results: tuple[StoredVisualAssetResult, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_count": self.requested_count,
            "stored_count": self.stored_count,
            "failed_count": self.failed_count,
            "vision_required_count": self.vision_required_count,
            "ocr_completed_count": self.ocr_completed_count,
            "ocr_partial_count": self.ocr_partial_count,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "errors": list(self.errors),
        }


class VisualAssetRepository:
    """
    PDF sayfalarından çıkarılan tablo, grafik, metrik kartı ve infografikleri
    kalıcı olarak saklar.

    Repository iki ayrı varlığı yönetir:

    - VisualAsset:
      Görsel dosyası, OCR metni, işleme durumu ve retrieval metadata'sı.
    - VisualMetric:
      OCR veya multimodal model tarafından görselden çıkarılmış yapılandırılmış
      sürdürülebilirlik metriği.

    Tasarım tercihleri:

    - Mevcut `Database` sınıfını değiştirmeden kendi SQLite tablolarını oluşturur.
    - Modelin UUID rapor kimliği ile SQLite'ın integer rapor kimliğini birlikte
      saklar.
    - Aynı çalışma süreci için bellek içi cache sağlar.
    - İsteğe bağlı JSON artifact üretir.
    - Aynı görselin `stable_key` değeri üzerinden tekrar kaydedilmesini engeller.
    - Görsel model gerektiren kayıtları hızlıca sorgular.
    """

    def __init__(
        self,
        *,
        database: Database,
        file_store: FileStore,
        save_json_artifacts: bool = True,
        overwrite_artifacts: bool = True,
        initialize_schema: bool = True,
    ) -> None:
        self.database = database
        self.file_store = file_store

        self.save_json_artifacts = bool(
            save_json_artifacts
        )
        self.overwrite_artifacts = bool(
            overwrite_artifacts
        )

        self._assets_by_id: dict[str, VisualAsset] = {}
        self._asset_id_by_stable_key: dict[str, str] = {}
        self._metrics_by_id: dict[str, VisualMetric] = {}
        self._metric_ids_by_visual_id: dict[str, list[str]] = {}

        if initialize_schema:
            self.initialize_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Görsel varlık ve görsel metrik tablolarını oluşturur."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS visual_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                visual_uid TEXT NOT NULL UNIQUE,
                stable_key TEXT NOT NULL,

                report_database_id INTEGER NOT NULL,
                report_uid TEXT NOT NULL,

                page_database_id INTEGER,
                page_uid TEXT,
                page_number INTEGER NOT NULL,

                region_type TEXT NOT NULL,
                processing_status TEXT NOT NULL,

                image_path TEXT NOT NULL,
                source_page_image_path TEXT,
                image_hash TEXT,

                image_width INTEGER,
                image_height INTEGER,

                bounding_box_json TEXT NOT NULL DEFAULT '{}',

                caption TEXT,
                nearby_text TEXT,
                section_title TEXT,
                subsection_title TEXT,

                category TEXT,
                topic TEXT,
                keywords_json TEXT NOT NULL DEFAULT '[]',

                detection_confidence REAL,

                extraction_methods_json TEXT NOT NULL DEFAULT '[]',

                ocr_text TEXT,
                ocr_confidence REAL,

                requires_whole_image_ocr INTEGER NOT NULL DEFAULT 0,
                requires_vision INTEGER NOT NULL DEFAULT 0,

                error_messages_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                model_json TEXT NOT NULL DEFAULT '{}',

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY(report_database_id)
                    REFERENCES reports(id)
                    ON DELETE CASCADE,

                FOREIGN KEY(page_database_id)
                    REFERENCES pages(id)
                    ON DELETE SET NULL,

                UNIQUE(report_database_id, stable_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS visual_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                metric_uid TEXT NOT NULL UNIQUE,
                visual_uid TEXT NOT NULL,

                report_database_id INTEGER NOT NULL,
                report_uid TEXT NOT NULL,
                page_number INTEGER NOT NULL,

                metric_key TEXT NOT NULL,
                metric_label TEXT NOT NULL,

                value_json TEXT NOT NULL,
                unit TEXT,
                metric_year INTEGER,
                scope TEXT,

                category TEXT,
                topic TEXT,

                source_text TEXT,
                source_bounding_box_json TEXT,

                confidence REAL NOT NULL,
                extraction_method TEXT NOT NULL,

                question_ids_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                model_json TEXT NOT NULL DEFAULT '{}',

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY(report_database_id)
                    REFERENCES reports(id)
                    ON DELETE CASCADE,

                FOREIGN KEY(visual_uid)
                    REFERENCES visual_assets(visual_uid)
                    ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_assets_report
                ON visual_assets(report_database_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_assets_report_page
                ON visual_assets(report_database_id, page_number)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_assets_status
                ON visual_assets(processing_status)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_assets_requires_vision
                ON visual_assets(report_database_id, requires_vision)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_assets_category_topic
                ON visual_assets(category, topic)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_assets_image_hash
                ON visual_assets(image_hash)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_metrics_report
                ON visual_metrics(report_database_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_metrics_visual
                ON visual_metrics(visual_uid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_metrics_key
                ON visual_metrics(metric_key)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_visual_metrics_year
                ON visual_metrics(metric_year)
            """,
        )

        for statement in statements:
            self.database.execute(statement)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def cache_asset(
        self,
        asset: VisualAsset,
    ) -> None:
        visual_uid = str(asset.id)

        previous_uid = self._asset_id_by_stable_key.get(
            asset.stable_key
        )

        if (
            previous_uid is not None
            and previous_uid != visual_uid
        ):
            self._assets_by_id.pop(
                previous_uid,
                None,
            )

        self._assets_by_id[visual_uid] = asset
        self._asset_id_by_stable_key[
            asset.stable_key
        ] = visual_uid

    def cache_assets(
        self,
        assets: Sequence[VisualAsset],
    ) -> int:
        for asset in assets:
            self.cache_asset(asset)

        return len(assets)

    def cache_metric(
        self,
        metric: VisualMetric,
    ) -> None:
        metric_uid = str(metric.id)
        visual_uid = str(metric.visual_id)

        self._metrics_by_id[metric_uid] = metric

        ids = self._metric_ids_by_visual_id.setdefault(
            visual_uid,
            [],
        )

        if metric_uid not in ids:
            ids.append(metric_uid)

    def cache_metrics(
        self,
        metrics: Sequence[VisualMetric],
    ) -> int:
        for metric in metrics:
            self.cache_metric(metric)

        return len(metrics)

    def all_assets(self) -> list[VisualAsset]:
        return sorted(
            self._assets_by_id.values(),
            key=lambda asset: (
                str(asset.report_id),
                asset.page_number,
                asset.bounding_box.y1,
                asset.bounding_box.x1,
                str(asset.id),
            ),
        )

    def all_metrics(self) -> list[VisualMetric]:
        return sorted(
            self._metrics_by_id.values(),
            key=lambda metric: (
                str(metric.report_id),
                metric.page_number,
                metric.metric_key,
                str(metric.id),
            ),
        )

    # ------------------------------------------------------------------
    # VisualAsset storage
    # ------------------------------------------------------------------

    def store_asset(
        self,
        *,
        report_database_id: int,
        company_name: str,
        report_year: int,
        asset: VisualAsset,
        page_database_id: int | None = None,
    ) -> StoredVisualAssetResult:
        """Tek bir görsel varlığı SQLite ve isteğe bağlı JSON'a kaydeder."""

        if report_database_id < 1:
            raise ValueError(
                "report_database_id pozitif olmalıdır."
            )

        if not company_name.strip():
            raise ValueError(
                "company_name boş olamaz."
            )

        if not asset.image_path:
            raise ValueError(
                "VisualAsset image_path alanı zorunludur."
            )

        # Aynı rapor içinde aynı stable_key daha önce kaydedilmişse mevcut
        # UUID korunur. Böylece overwrite çalıştırmalarında UNIQUE hatası
        # oluşmaz ve görsel metrik ilişkileri kararlı kalır.
        existing_stable_row = self.database.fetch_one(
            """
            SELECT visual_uid
            FROM visual_assets
            WHERE report_database_id = ?
              AND stable_key = ?
            LIMIT 1
            """,
            (
                report_database_id,
                asset.stable_key,
            ),
        )

        if existing_stable_row is not None:
            existing_uid = str(
                existing_stable_row["visual_uid"]
            )

            if existing_uid != str(asset.id):
                asset.id = UUID(existing_uid)

        now = self._utc_now()
        visual_uid = str(asset.id)
        report_uid = str(asset.report_id)
        page_uid = (
            str(asset.page_id)
            if asset.page_id is not None
            else None
        )

        payload = asset.model_dump(
            mode="json",
        )

        bounding_box = asset.bounding_box.model_dump(
            mode="json",
        )

        sql = """
        INSERT INTO visual_assets (
            visual_uid,
            stable_key,
            report_database_id,
            report_uid,
            page_database_id,
            page_uid,
            page_number,
            region_type,
            processing_status,
            image_path,
            source_page_image_path,
            image_hash,
            image_width,
            image_height,
            bounding_box_json,
            caption,
            nearby_text,
            section_title,
            subsection_title,
            category,
            topic,
            keywords_json,
            detection_confidence,
            extraction_methods_json,
            ocr_text,
            ocr_confidence,
            requires_whole_image_ocr,
            requires_vision,
            error_messages_json,
            metadata_json,
            model_json,
            created_at,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )

        ON CONFLICT(visual_uid)
        DO UPDATE SET
            stable_key = excluded.stable_key,
            report_database_id = excluded.report_database_id,
            report_uid = excluded.report_uid,
            page_database_id = excluded.page_database_id,
            page_uid = excluded.page_uid,
            page_number = excluded.page_number,
            region_type = excluded.region_type,
            processing_status = excluded.processing_status,
            image_path = excluded.image_path,
            source_page_image_path = excluded.source_page_image_path,
            image_hash = excluded.image_hash,
            image_width = excluded.image_width,
            image_height = excluded.image_height,
            bounding_box_json = excluded.bounding_box_json,
            caption = excluded.caption,
            nearby_text = excluded.nearby_text,
            section_title = excluded.section_title,
            subsection_title = excluded.subsection_title,
            category = excluded.category,
            topic = excluded.topic,
            keywords_json = excluded.keywords_json,
            detection_confidence = excluded.detection_confidence,
            extraction_methods_json = excluded.extraction_methods_json,
            ocr_text = excluded.ocr_text,
            ocr_confidence = excluded.ocr_confidence,
            requires_whole_image_ocr =
                excluded.requires_whole_image_ocr,
            requires_vision = excluded.requires_vision,
            error_messages_json = excluded.error_messages_json,
            metadata_json = excluded.metadata_json,
            model_json = excluded.model_json,
            updated_at = excluded.updated_at
        """

        parameters = (
            visual_uid,
            asset.stable_key,
            report_database_id,
            report_uid,
            page_database_id,
            page_uid,
            asset.page_number,
            asset.region_type.value,
            asset.processing_status.value,
            str(asset.image_path),
            (
                str(asset.source_page_image_path)
                if asset.source_page_image_path is not None
                else None
            ),
            asset.image_hash,
            asset.image_width,
            asset.image_height,
            self._json_dumps(bounding_box),
            asset.caption,
            asset.nearby_text,
            asset.section_title,
            asset.subsection_title,
            asset.category,
            asset.topic,
            self._json_dumps(asset.keywords),
            asset.detection_confidence,
            self._json_dumps(
                [
                    method.value
                    for method in asset.extraction_methods
                ]
            ),
            asset.ocr_text,
            asset.ocr_confidence,
            int(asset.requires_whole_image_ocr),
            int(asset.requires_vision),
            self._json_dumps(asset.error_messages),
            self._json_dumps(asset.metadata),
            self._json_dumps(payload),
            now,
            now,
        )

        self.database.execute(
            sql,
            parameters,
        )

        row = self.database.fetch_one(
            """
            SELECT id
            FROM visual_assets
            WHERE visual_uid = ?
            """,
            (visual_uid,),
        )

        if row is None:
            raise VisualAssetRepositoryError(
                f"Görsel kayıt kimliği okunamadı: {visual_uid}"
            )

        artifact = self._save_asset_artifact(
            company_name=company_name,
            report_year=report_year,
            asset=asset,
        )

        self.cache_asset(asset)

        result = StoredVisualAssetResult(
            visual_database_id=int(row["id"]),
            visual_uid=visual_uid,
            report_database_id=report_database_id,
            report_uid=report_uid,
            page_number=asset.page_number,
            processing_status=(
                asset.processing_status.value
            ),
            requires_vision=asset.requires_vision,
            artifact=artifact,
            stored_at=datetime.now(UTC),
        )

        logger.info(
            "Görsel varlık kaydedildi: visual_uid=%s, page=%d, status=%s",
            visual_uid,
            asset.page_number,
            asset.processing_status.value,
        )

        return result

    def store_assets(
        self,
        *,
        report_database_id: int,
        company_name: str,
        report_year: int,
        assets: Sequence[VisualAsset],
        page_database_ids: Mapping[int, int] | None = None,
        continue_on_error: bool = True,
    ) -> VisualAssetRepositorySummary:
        """Birden fazla görsel varlığı kaydeder."""

        page_database_ids = page_database_ids or {}

        results: list[StoredVisualAssetResult] = []
        errors: list[str] = []

        for asset in assets:
            try:
                result = self.store_asset(
                    report_database_id=report_database_id,
                    company_name=company_name,
                    report_year=report_year,
                    asset=asset,
                    page_database_id=page_database_ids.get(
                        asset.page_number
                    ),
                )
            except Exception as exc:
                message = (
                    f"Görsel {asset.id} depolanamadı: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(message)

                logger.exception(message)

                if not continue_on_error:
                    raise VisualAssetRepositoryError(
                        message
                    ) from exc

                continue

            results.append(result)

        return VisualAssetRepositorySummary(
            requested_count=len(assets),
            stored_count=len(results),
            failed_count=len(assets) - len(results),
            vision_required_count=sum(
                result.requires_vision
                for result in results
            ),
            ocr_completed_count=sum(
                result.processing_status
                == VisualProcessingStatus.OCR_COMPLETED.value
                for result in results
            ),
            ocr_partial_count=sum(
                result.processing_status
                == VisualProcessingStatus.OCR_PARTIAL.value
                for result in results
            ),
            results=tuple(results),
            errors=tuple(errors),
        )

    def apply_ocr_result(
        self,
        *,
        report_database_id: int,
        company_name: str,
        report_year: int,
        asset: VisualAsset,
        result: WholeImageOCRResult,
        page_database_id: int | None = None,
        minimum_confidence: float = 0.45,
    ) -> StoredVisualAssetResult:
        """
        OCR sonucunu VisualAsset'e uygular ve güncellenmiş kaydı saklar.
        """

        asset.mark_ocr_result(
            result,
            minimum_confidence=minimum_confidence,
        )

        self._save_ocr_result_artifact(
            company_name=company_name,
            report_year=report_year,
            asset=asset,
            result=result,
        )

        return self.store_asset(
            report_database_id=report_database_id,
            company_name=company_name,
            report_year=report_year,
            asset=asset,
            page_database_id=page_database_id,
        )

    # ------------------------------------------------------------------
    # VisualMetric storage
    # ------------------------------------------------------------------

    def store_metric(
        self,
        *,
        report_database_id: int,
        company_name: str,
        report_year: int,
        metric: VisualMetric,
    ) -> StoredVisualMetricResult:
        """Tek bir görsel metriği saklar."""

        if report_database_id < 1:
            raise ValueError(
                "report_database_id pozitif olmalıdır."
            )

        visual_uid = str(metric.visual_id)

        visual_row = self.database.fetch_one(
            """
            SELECT visual_uid
            FROM visual_assets
            WHERE visual_uid = ?
            """,
            (visual_uid,),
        )

        if visual_row is None:
            raise VisualAssetRepositoryError(
                "Metrik kaydedilmeden önce ilgili VisualAsset "
                f"kaydedilmelidir: {visual_uid}"
            )

        now = self._utc_now()
        metric_uid = str(metric.id)
        report_uid = str(metric.report_id)

        payload = metric.model_dump(
            mode="json",
        )

        source_box = (
            metric.source_bounding_box.model_dump(
                mode="json",
            )
            if metric.source_bounding_box is not None
            else None
        )

        sql = """
        INSERT INTO visual_metrics (
            metric_uid,
            visual_uid,
            report_database_id,
            report_uid,
            page_number,
            metric_key,
            metric_label,
            value_json,
            unit,
            metric_year,
            scope,
            category,
            topic,
            source_text,
            source_bounding_box_json,
            confidence,
            extraction_method,
            question_ids_json,
            metadata_json,
            model_json,
            created_at,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?
        )

        ON CONFLICT(metric_uid)
        DO UPDATE SET
            visual_uid = excluded.visual_uid,
            report_database_id = excluded.report_database_id,
            report_uid = excluded.report_uid,
            page_number = excluded.page_number,
            metric_key = excluded.metric_key,
            metric_label = excluded.metric_label,
            value_json = excluded.value_json,
            unit = excluded.unit,
            metric_year = excluded.metric_year,
            scope = excluded.scope,
            category = excluded.category,
            topic = excluded.topic,
            source_text = excluded.source_text,
            source_bounding_box_json =
                excluded.source_bounding_box_json,
            confidence = excluded.confidence,
            extraction_method = excluded.extraction_method,
            question_ids_json = excluded.question_ids_json,
            metadata_json = excluded.metadata_json,
            model_json = excluded.model_json,
            updated_at = excluded.updated_at
        """

        self.database.execute(
            sql,
            (
                metric_uid,
                visual_uid,
                report_database_id,
                report_uid,
                metric.page_number,
                metric.metric_key,
                metric.metric_label,
                self._json_dumps(metric.value),
                metric.unit,
                metric.year,
                metric.scope,
                metric.category,
                metric.topic,
                metric.source_text,
                (
                    self._json_dumps(source_box)
                    if source_box is not None
                    else None
                ),
                metric.confidence,
                metric.extraction_method.value,
                self._json_dumps(metric.question_ids),
                self._json_dumps(metric.metadata),
                self._json_dumps(payload),
                now,
                now,
            ),
        )

        row = self.database.fetch_one(
            """
            SELECT id
            FROM visual_metrics
            WHERE metric_uid = ?
            """,
            (metric_uid,),
        )

        if row is None:
            raise VisualAssetRepositoryError(
                f"Görsel metrik kayıt kimliği okunamadı: {metric_uid}"
            )

        artifact = self._save_metric_artifact(
            company_name=company_name,
            report_year=report_year,
            metric=metric,
        )

        self.cache_metric(metric)

        return StoredVisualMetricResult(
            metric_database_id=int(row["id"]),
            metric_uid=metric_uid,
            visual_uid=visual_uid,
            report_database_id=report_database_id,
            report_uid=report_uid,
            page_number=metric.page_number,
            metric_key=metric.metric_key,
            confidence=metric.confidence,
            artifact=artifact,
            stored_at=datetime.now(UTC),
        )

    def store_metrics(
        self,
        *,
        report_database_id: int,
        company_name: str,
        report_year: int,
        metrics: Sequence[VisualMetric],
        continue_on_error: bool = True,
    ) -> tuple[
        tuple[StoredVisualMetricResult, ...],
        tuple[str, ...],
    ]:
        """Birden fazla görsel metriği kaydeder."""

        results: list[StoredVisualMetricResult] = []
        errors: list[str] = []

        for metric in metrics:
            try:
                result = self.store_metric(
                    report_database_id=report_database_id,
                    company_name=company_name,
                    report_year=report_year,
                    metric=metric,
                )
            except Exception as exc:
                message = (
                    f"Görsel metrik {metric.id} depolanamadı: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(message)
                logger.exception(message)

                if not continue_on_error:
                    raise VisualAssetRepositoryError(
                        message
                    ) from exc

                continue

            results.append(result)

        return tuple(results), tuple(errors)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_asset(
        self,
        visual_id: str | UUID,
    ) -> VisualAsset:
        asset = self.get_asset_optional(
            visual_id
        )

        if asset is None:
            raise KeyError(
                f"Görsel varlık bulunamadı: {visual_id}"
            )

        return asset

    def get_asset_optional(
        self,
        visual_id: str | UUID,
    ) -> VisualAsset | None:
        visual_uid = str(visual_id)

        cached = self._assets_by_id.get(
            visual_uid
        )

        if cached is not None:
            return cached

        row = self.database.fetch_one(
            """
            SELECT model_json
            FROM visual_assets
            WHERE visual_uid = ?
            """,
            (visual_uid,),
        )

        if row is None:
            return None

        asset = VisualAsset.model_validate(
            self._json_loads(
                row["model_json"],
                default={},
            )
        )

        self.cache_asset(asset)
        return asset

    def get_by_stable_key(
        self,
        stable_key: str,
        *,
        report_database_id: int | None = None,
    ) -> VisualAsset | None:
        normalized_key = stable_key.strip()

        if not normalized_key:
            return None

        cached_uid = self._asset_id_by_stable_key.get(
            normalized_key
        )

        if cached_uid is not None:
            return self._assets_by_id.get(
                cached_uid
            )

        if report_database_id is None:
            row = self.database.fetch_one(
                """
                SELECT model_json
                FROM visual_assets
                WHERE stable_key = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (normalized_key,),
            )
        else:
            row = self.database.fetch_one(
                """
                SELECT model_json
                FROM visual_assets
                WHERE stable_key = ?
                  AND report_database_id = ?
                LIMIT 1
                """,
                (
                    normalized_key,
                    report_database_id,
                ),
            )

        if row is None:
            return None

        asset = VisualAsset.model_validate(
            self._json_loads(
                row["model_json"],
                default={},
            )
        )

        self.cache_asset(asset)
        return asset

    def find_assets(
        self,
        *,
        report_database_id: int | None = None,
        report_uid: str | UUID | None = None,
        page_number: int | None = None,
        region_type: VisualRegionType | str | None = None,
        processing_status: VisualProcessingStatus | str | None = None,
        requires_vision: bool | None = None,
        category: str | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[VisualAsset]:
        clauses: list[str] = []
        parameters: list[Any] = []

        if report_database_id is not None:
            clauses.append(
                "report_database_id = ?"
            )
            parameters.append(
                int(report_database_id)
            )

        if report_uid is not None:
            clauses.append(
                "report_uid = ?"
            )
            parameters.append(
                str(report_uid)
            )

        if page_number is not None:
            clauses.append(
                "page_number = ?"
            )
            parameters.append(
                int(page_number)
            )

        if region_type is not None:
            clauses.append(
                "region_type = ?"
            )
            parameters.append(
                self._enum_value(region_type)
            )

        if processing_status is not None:
            clauses.append(
                "processing_status = ?"
            )
            parameters.append(
                self._enum_value(processing_status)
            )

        if requires_vision is not None:
            clauses.append(
                "requires_vision = ?"
            )
            parameters.append(
                int(requires_vision)
            )

        if category is not None:
            clauses.append(
                "LOWER(category) = ?"
            )
            parameters.append(
                category.strip().casefold()
            )

        if topic is not None:
            clauses.append(
                "LOWER(topic) = ?"
            )
            parameters.append(
                topic.strip().casefold()
            )

        where_sql = (
            " WHERE " + " AND ".join(clauses)
            if clauses
            else ""
        )

        limit_sql = ""
        if limit is not None:
            if limit < 1:
                return []
            limit_sql = " LIMIT ?"
            parameters.append(int(limit))

        rows = self.database.fetch_all(
            f"""
            SELECT model_json
            FROM visual_assets
            {where_sql}
            ORDER BY
                page_number ASC,
                id ASC
            {limit_sql}
            """,
            tuple(parameters),
        )

        assets = [
            VisualAsset.model_validate(
                self._json_loads(
                    row["model_json"],
                    default={},
                )
            )
            for row in rows
        ]

        self.cache_assets(assets)
        return assets

    def find_pending_vision(
        self,
        *,
        report_database_id: int | None = None,
        report_uid: str | UUID | None = None,
        category: str | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[VisualAsset]:
        """Multimodal model ile işlenmesi gereken görselleri getirir."""

        return self.find_assets(
            report_database_id=report_database_id,
            report_uid=report_uid,
            requires_vision=True,
            category=category,
            topic=topic,
            limit=limit,
        )

    def get_metric(
        self,
        metric_id: str | UUID,
    ) -> VisualMetric:
        metric_uid = str(metric_id)

        cached = self._metrics_by_id.get(
            metric_uid
        )

        if cached is not None:
            return cached

        row = self.database.fetch_one(
            """
            SELECT model_json
            FROM visual_metrics
            WHERE metric_uid = ?
            """,
            (metric_uid,),
        )

        if row is None:
            raise KeyError(
                f"Görsel metrik bulunamadı: {metric_uid}"
            )

        metric = VisualMetric.model_validate(
            self._json_loads(
                row["model_json"],
                default={},
            )
        )

        self.cache_metric(metric)
        return metric

    def find_metrics(
        self,
        *,
        report_database_id: int | None = None,
        report_uid: str | UUID | None = None,
        visual_id: str | UUID | None = None,
        metric_key: str | None = None,
        year: int | None = None,
        category: str | None = None,
        topic: str | None = None,
        minimum_confidence: float | None = None,
        limit: int | None = None,
    ) -> list[VisualMetric]:
        clauses: list[str] = []
        parameters: list[Any] = []

        if report_database_id is not None:
            clauses.append(
                "report_database_id = ?"
            )
            parameters.append(
                int(report_database_id)
            )

        if report_uid is not None:
            clauses.append(
                "report_uid = ?"
            )
            parameters.append(
                str(report_uid)
            )

        if visual_id is not None:
            clauses.append(
                "visual_uid = ?"
            )
            parameters.append(
                str(visual_id)
            )

        if metric_key is not None:
            clauses.append(
                "LOWER(metric_key) = ?"
            )
            parameters.append(
                metric_key.strip().casefold()
            )

        if year is not None:
            clauses.append(
                "metric_year = ?"
            )
            parameters.append(
                int(year)
            )

        if category is not None:
            clauses.append(
                "LOWER(category) = ?"
            )
            parameters.append(
                category.strip().casefold()
            )

        if topic is not None:
            clauses.append(
                "LOWER(topic) = ?"
            )
            parameters.append(
                topic.strip().casefold()
            )

        if minimum_confidence is not None:
            if not 0.0 <= minimum_confidence <= 1.0:
                raise ValueError(
                    "minimum_confidence 0 ile 1 arasında olmalıdır."
                )

            clauses.append(
                "confidence >= ?"
            )
            parameters.append(
                float(minimum_confidence)
            )

        where_sql = (
            " WHERE " + " AND ".join(clauses)
            if clauses
            else ""
        )

        limit_sql = ""
        if limit is not None:
            if limit < 1:
                return []
            limit_sql = " LIMIT ?"
            parameters.append(int(limit))

        rows = self.database.fetch_all(
            f"""
            SELECT model_json
            FROM visual_metrics
            {where_sql}
            ORDER BY
                confidence DESC,
                page_number ASC,
                id ASC
            {limit_sql}
            """,
            tuple(parameters),
        )

        metrics = [
            VisualMetric.model_validate(
                self._json_loads(
                    row["model_json"],
                    default={},
                )
            )
            for row in rows
        ]

        self.cache_metrics(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_report_records(
        self,
        *,
        report_database_id: int,
        confirm: bool = False,
    ) -> None:
        """Bir rapora ait görsel varlık ve metrik kayıtlarını siler."""

        if not confirm:
            raise VisualAssetRepositoryError(
                "Silme işlemi için confirm=True verilmelidir."
            )

        affected_assets = self.database.fetch_all(
            """
            SELECT visual_uid
            FROM visual_assets
            WHERE report_database_id = ?
            """,
            (report_database_id,),
        )

        affected_metrics = self.database.fetch_all(
            """
            SELECT metric_uid, visual_uid
            FROM visual_metrics
            WHERE report_database_id = ?
            """,
            (report_database_id,),
        )

        self.database.execute(
            """
            DELETE FROM visual_metrics
            WHERE report_database_id = ?
            """,
            (report_database_id,),
        )

        self.database.execute(
            """
            DELETE FROM visual_assets
            WHERE report_database_id = ?
            """,
            (report_database_id,),
        )

        affected_asset_ids = {
            str(row["visual_uid"])
            for row in affected_assets
        }
        affected_metric_ids = {
            str(row["metric_uid"])
            for row in affected_metrics
        }

        for visual_uid in affected_asset_ids:
            asset = self._assets_by_id.pop(
                visual_uid,
                None,
            )
            if asset is not None:
                self._asset_id_by_stable_key.pop(
                    asset.stable_key,
                    None,
                )
            self._metric_ids_by_visual_id.pop(
                visual_uid,
                None,
            )

        for metric_uid in affected_metric_ids:
            self._metrics_by_id.pop(
                metric_uid,
                None,
            )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _save_asset_artifact(
        self,
        *,
        company_name: str,
        report_year: int,
        asset: VisualAsset,
    ) -> StoredArtifact | None:
        if not self.save_json_artifacts:
            return None

        paths = self.file_store.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=str(asset.report_id),
        )

        target = (
            paths.ocr_dir
            / "visual_assets"
            / f"{asset.id}.json"
        )

        return self.file_store.save_json(
            asset.model_dump(mode="json"),
            target,
            overwrite=self.overwrite_artifacts,
            artifact_type=ArtifactType.OCR,
            company_slug=self.file_store.slugify(
                company_name
            ),
            report_year=report_year,
            report_id=str(asset.report_id),
            page_number=asset.page_number,
            metadata={
                "visual_uid": str(asset.id),
                "region_type": asset.region_type.value,
                "processing_status": (
                    asset.processing_status.value
                ),
                "requires_vision": asset.requires_vision,
                "image_path": str(asset.image_path),
            },
        )

    def _save_ocr_result_artifact(
        self,
        *,
        company_name: str,
        report_year: int,
        asset: VisualAsset,
        result: WholeImageOCRResult,
    ) -> StoredArtifact | None:
        if not self.save_json_artifacts:
            return None

        paths = self.file_store.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=str(asset.report_id),
        )

        target = (
            paths.ocr_dir
            / "whole_image_results"
            / f"{result.id}.json"
        )

        return self.file_store.save_json(
            result.model_dump(mode="json"),
            target,
            overwrite=self.overwrite_artifacts,
            artifact_type=ArtifactType.OCR,
            company_slug=self.file_store.slugify(
                company_name
            ),
            report_year=report_year,
            report_id=str(asset.report_id),
            page_number=asset.page_number,
            metadata={
                "visual_uid": str(asset.id),
                "ocr_result_uid": str(result.id),
                "average_confidence": (
                    result.average_confidence
                ),
                "is_structured": result.is_structured,
                "psm_mode": result.psm_mode,
            },
        )

    def _save_metric_artifact(
        self,
        *,
        company_name: str,
        report_year: int,
        metric: VisualMetric,
    ) -> StoredArtifact | None:
        if not self.save_json_artifacts:
            return None

        paths = self.file_store.get_report_paths(
            company_name=company_name,
            report_year=report_year,
            report_id=str(metric.report_id),
        )

        target = (
            paths.evidence_dir
            / "visual_metrics"
            / f"{metric.id}.json"
        )

        return self.file_store.save_json(
            metric.model_dump(mode="json"),
            target,
            overwrite=self.overwrite_artifacts,
            artifact_type=ArtifactType.EVIDENCE,
            company_slug=self.file_store.slugify(
                company_name
            ),
            report_year=report_year,
            report_id=str(metric.report_id),
            page_number=metric.page_number,
            metadata={
                "metric_uid": str(metric.id),
                "visual_uid": str(metric.visual_id),
                "metric_key": metric.metric_key,
                "confidence": metric.confidence,
                "extraction_method": (
                    metric.extraction_method.value
                ),
            },
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def _json_dumps(
        cls,
        value: Any,
    ) -> str:
        return json.dumps(
            cls._serialize(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _json_loads(
        value: Any,
        *,
        default: Any,
    ) -> Any:
        if value is None:
            return default

        if isinstance(
            value,
            (dict, list, tuple),
        ):
            return value

        text = str(value).strip()

        if not text:
            return default

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise VisualAssetRepositoryError(
                "Repository JSON alanı çözümlenemedi."
            ) from exc

    @classmethod
    def _serialize(
        cls,
        value: Any,
    ) -> Any:
        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, UUID):
            return str(value)

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Mapping):
            return {
                str(key): cls._serialize(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize(item)
                for item in value
            ]

        model_dump = getattr(
            value,
            "model_dump",
            None,
        )

        if callable(model_dump):
            return cls._serialize(
                model_dump(mode="json")
            )

        return str(value)

    @staticmethod
    def _enum_value(
        value: Enum | str,
    ) -> str:
        if isinstance(value, Enum):
            return str(value.value)

        return str(value).strip()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            UTC
        ).isoformat()


__all__ = [
    "StoredVisualAssetResult",
    "StoredVisualMetricResult",
    "VisualAssetRepository",
    "VisualAssetRepositoryError",
    "VisualAssetRepositorySummary",
]
