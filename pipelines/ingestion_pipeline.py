from __future__ import annotations

import hashlib
import json
import inspect
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from models.page import BoundingBox
from models.visual_evidence import (
    VisualAsset,
    VisualExtractionMethod,
    VisualProcessingStatus,
    VisualRegionType,
)

try:
    from table_processing.vlm_table_fallback import (
        CloudflareQuotaExceededError,
    )
except ImportError:
    class CloudflareQuotaExceededError(RuntimeError):
        """Fallback import; gerçek sınıf VLM modülünden gelir."""
from text_processing.metadata_enricher import MetadataContext


class IngestionStatus(StrEnum):
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class IngestionRequest:
    company_name: str
    report_year: int
    report_url: str | None = None
    local_pdf_path: str | Path | None = None
    report_id: str | None = None
    force_download: bool = False
    overwrite_existing: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.company_name.strip():
            raise ValueError("company_name boş olamaz.")
        if not 1900 <= self.report_year <= 2100:
            raise ValueError("report_year 1900 ile 2100 arasında olmalıdır.")

        if self.report_url is not None and self.local_pdf_path is not None:
            raise ValueError(
                "report_url ve local_pdf_path aynı anda verilemez."
            )

        if self.local_pdf_path is not None:
            local_path = Path(self.local_pdf_path).expanduser().resolve()

            if local_path.suffix.casefold() != ".pdf":
                raise ValueError(
                    f"Yerel rapor dosyası PDF olmalıdır: {local_path}"
                )

            if not local_path.is_file():
                raise ValueError(
                    f"Yerel PDF dosyası bulunamadı: {local_path}"
                )

            object.__setattr__(
                self,
                "local_pdf_path",
                local_path,
            )


@dataclass(slots=True, frozen=True)
class IngestionResult:
    report_id: str
    company_name: str
    report_year: int
    report_url: str | None
    pdf_path: Path | None
    status: IngestionStatus
    page_count: int = 0
    table_count: int = 0
    chunk_count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return self.status in {
            IngestionStatus.COMPLETED,
            IngestionStatus.PARTIALLY_COMPLETED,
        }


@dataclass(slots=True, frozen=True)
class _VisualOCRChunk:
    """Görsel tablo/grafik OCR metnini retrieval katmanına taşıyan kayıt."""

    chunk_id: str
    report_id: str
    chunk_index: int

    page_number: int
    start_page: int
    end_page: int

    text: str
    normalized_text: str
    searchable_text: str
    contextualized_text: str

    section_title: str
    category: str
    source_type: str

    content_hash: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class _TableProcessingResult:
    tables: tuple[Any, ...]
    validations_by_table_id: Mapping[str, Any]
    visual_ocr_chunks: tuple[Any, ...] = ()
    visual_chart_chunks: tuple[Any, ...] = ()
    visual_assets: tuple[VisualAsset, ...] = ()
    visual_metrics: Mapping[str, int | bool] = field(default_factory=dict)
    standard_index_rows: tuple[Any, ...] = ()


class IngestionPipelineError(RuntimeError):
    """Belge işleme hattında oluşan temel hata."""


class IngestionPipeline:
    """Sürdürülebilirlik raporunu uçtan uca işleyen ana ingestion hattı.

    Pipeline, bileşenlerin gerçek ve tipli API'lerini kullanır. Tablo hattında
    sıra Native -> Borderless -> Visual -> OCR -> VLM fallback; metin hattında ise
    TextCleaner -> SectionDetector -> SemanticChunker -> CategoryClassifier ->
    MetadataEnricher şeklindedir.
    """

    def __init__(
        self,
        *,
        report_search: Any,
        candidate_ranker: Any,
        url_validator: Any,
        pdf_downloader: Any,
        pdf_validator: Any,
        page_renderer: Any,
        page_classifier: Any,
        layout_detector: Any,
        table_detector: Any,
        native_table_extractor: Any,
        visual_table_extractor: Any,
        table_ocr: Any | None,
        vlm_table_fallback: Any | None,
        table_normalizer: Any,
        numeric_validator: Any,
        table_repository: Any,
        visual_asset_repository: Any | None = None,
        text_cleaner: Any,
        section_detector: Any,
        semantic_chunker: Any,
        category_classifier: Any,
        metadata_enricher: Any,
        chunk_repository: Any,
        borderless_table_extractor: Any | None = None,
        standard_index_parser: Any | None = None,
        vector_store: Any | None = None,
        database: Any | None = None,
        file_store: Any | None = None,
        cache: Any | None = None,
        table_image_directory: str | Path | None = None,
    ) -> None:
        self.report_search = report_search
        self.candidate_ranker = candidate_ranker
        self.url_validator = url_validator
        self.pdf_downloader = pdf_downloader
        self.pdf_validator = pdf_validator
        self.page_renderer = page_renderer
        self.page_classifier = page_classifier
        self.layout_detector = layout_detector

        self.table_detector = table_detector
        self.native_table_extractor = native_table_extractor
        self.visual_table_extractor = visual_table_extractor
        self.table_ocr = table_ocr
        self.vlm_table_fallback = vlm_table_fallback
        self.table_normalizer = table_normalizer
        self.numeric_validator = numeric_validator
        self.table_repository = table_repository
        self.visual_asset_repository = visual_asset_repository

        self.text_cleaner = text_cleaner
        self.section_detector = section_detector
        self.semantic_chunker = semantic_chunker
        self.category_classifier = category_classifier
        self.metadata_enricher = metadata_enricher
        self.chunk_repository = chunk_repository

        if borderless_table_extractor is None:
            try:
                from table_processing.borderless_table_extractor import (
                    BorderlessTableExtractor,
                )

                borderless_table_extractor = BorderlessTableExtractor()
            except Exception:
                borderless_table_extractor = None

        self.borderless_table_extractor = borderless_table_extractor

        self.standard_index_parser = standard_index_parser
        self.vector_store = vector_store
        self.database = database
        self.file_store = file_store
        self.cache = cache
        self.table_image_directory = (
            Path(table_image_directory).expanduser().resolve()
            if table_image_directory is not None
            else None
        )

    def run(self, request: IngestionRequest) -> IngestionResult:
        errors: list[str] = []
        warnings: list[str] = []
        report_id = request.report_id or self._create_report_id(
            request.company_name,
            request.report_year,
        )
        report_url = request.report_url
        local_pdf_path = (
            Path(request.local_pdf_path).expanduser().resolve()
            if request.local_pdf_path is not None
            else None
        )
        pdf_path: Path | None = None
        database_report_id: int | None = None

        try:
            if local_pdf_path is not None:
                pdf_path = self._resolve_local_pdf(local_pdf_path)
            else:
                if report_url is None:
                    report_url = self._discover_report_url(
                        request.company_name,
                        request.report_year,
                    )

                pdf_path = self._download_pdf(
                    report_url,
                    report_id=report_id,
                    overwrite=request.overwrite_existing,
                )

            validation = self._validate_pdf(pdf_path)
            rendered_pages = self._render_pages(pdf_path)
            classified_pages = self._classify_pages(rendered_pages)
            layouts = self._detect_layouts(classified_pages)

            database_report_id = self._upsert_report(
                report_uid=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                report_url=report_url,
                pdf_path=pdf_path,
                validation=validation,
                page_count=len(rendered_pages),
                status="processing",
                metadata=request.metadata,
            )

            if request.overwrite_existing:
                self._prepare_overwrite(
                    database_report_id=database_report_id,
                    report_uid=report_id,
                    warnings=warnings,
                )

            page_ids = self._store_pages(
                report_uid=report_id,
                database_report_id=database_report_id,
                rendered_pages=rendered_pages,
                classified_pages=classified_pages,
                layouts=layouts,
            )

            table_result = self._process_tables(
                pdf_path=pdf_path,
                layouts=layouts,
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                warnings=warnings,
            )

            (
                layout_visual_assets,
                layout_visual_chunks,
                layout_visual_metrics,
            ) = self._process_layout_visuals(
                layouts=layouts,
                existing_assets=table_result.visual_assets,
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                warnings=warnings,
            )

            (
                page_chart_assets,
                page_chart_chunks,
                page_chart_metrics,
            ) = self._process_page_chart_candidates(
                layouts=layouts,
                existing_assets=(
                    *table_result.visual_assets,
                    *layout_visual_assets,
                ),
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                warnings=warnings,
            )

            table_result = _TableProcessingResult(
                tables=table_result.tables,
                validations_by_table_id=table_result.validations_by_table_id,
                visual_ocr_chunks=table_result.visual_ocr_chunks,
                visual_chart_chunks=(
                    *table_result.visual_chart_chunks,
                    *layout_visual_chunks,
                    *page_chart_chunks,
                ),
                visual_assets=(
                    *table_result.visual_assets,
                    *layout_visual_assets,
                    *page_chart_assets,
                ),
                visual_metrics={
                    **dict(table_result.visual_metrics),
                    **dict(layout_visual_metrics),
                    **dict(page_chart_metrics),
                },
                standard_index_rows=table_result.standard_index_rows,
            )

            cleaned_document, section_result, chunks = self._process_text(
                classified_pages=classified_pages,
                layouts=layouts,
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                report_url=report_url,
                pdf_path=pdf_path,
                warnings=warnings,
            )

            all_chunks, duplicate_chunk_count = self._deduplicate_chunks(
                [
                    *chunks,
                    *table_result.visual_ocr_chunks,
                    *table_result.visual_chart_chunks,
                ]
            )
            if duplicate_chunk_count:
                warnings.append(
                    f"{duplicate_chunk_count} yinelenen chunk_uid bellek "
                    "içinde tekilleştirildi."
                )

            self._store_tables(
                database_report_id=database_report_id,
                report_uid=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                page_ids=page_ids,
                result=table_result,
                warnings=warnings,
            )

            visual_store_summary = self._store_visual_assets(
                database_report_id=database_report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                page_ids=page_ids,
                assets=table_result.visual_assets,
                warnings=warnings,
            )

            self._store_chunks(
                all_chunks,
                database_report_id=database_report_id,
                report_uid=report_id,
                page_ids=page_ids,
                company_name=request.company_name,
                report_year=request.report_year,
                warnings=warnings,
            )
            self._index_vectors(
                all_chunks,
                table_result.tables,
                warnings,
            )

            standard_rows = self._parse_standard_index(
                tables=table_result.tables,
                warnings=warnings,
            )

            self._store_report_metadata(
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                report_url=report_url,
                pdf_path=pdf_path,
                page_count=len(rendered_pages),
                table_count=len(table_result.tables),
                chunk_count=len(all_chunks),
                section_count=getattr(section_result, "section_count", 0),
                cleaned_character_count=getattr(
                    cleaned_document,
                    "total_character_count",
                    0,
                ),
                standard_index_row_count=len(standard_rows),
                visual_ocr_chunk_count=len(
                    table_result.visual_ocr_chunks
                ),
                visual_chart_chunk_count=len(
                    table_result.visual_chart_chunks
                ),
                visual_asset_count=len(
                    table_result.visual_assets
                ),
                visual_asset_stored_count=(
                    int(getattr(visual_store_summary, "stored_count", 0))
                    if visual_store_summary is not None
                    else 0
                ),
                visual_asset_vision_required_count=sum(
                    asset.requires_vision
                    for asset in table_result.visual_assets
                ),
                **dict(table_result.visual_metrics),
            )

            if database_report_id is not None and self.database is not None:
                self.database.update_report_status(
                    database_report_id,
                    "partially_completed" if warnings else "completed",
                )

            status = (
                IngestionStatus.PARTIALLY_COMPLETED
                if warnings
                else IngestionStatus.COMPLETED
            )
            return IngestionResult(
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                report_url=report_url,
                pdf_path=pdf_path,
                status=status,
                page_count=len(rendered_pages),
                table_count=len(table_result.tables),
                chunk_count=len(all_chunks),
                warnings=tuple(warnings),
                metadata={
                    **dict(request.metadata),
                    "source_mode": (
                        "local_pdf"
                        if local_pdf_path is not None
                        else "url"
                    ),
                    "local_pdf_path": (
                        str(local_pdf_path)
                        if local_pdf_path is not None
                        else None
                    ),
                    "section_count": getattr(section_result, "section_count", 0),
                    "standard_index_row_count": len(standard_rows),
                    "native_text_chunk_count": len(chunks),
                    "visual_ocr_chunk_count": len(
                        table_result.visual_ocr_chunks
                    ),
                    "visual_chart_chunk_count": len(
                        table_result.visual_chart_chunks
                    ),
                    "visual_asset_count": len(
                        table_result.visual_assets
                    ),
                    "visual_asset_stored_count": (
                        int(
                            getattr(
                                visual_store_summary,
                                "stored_count",
                                0,
                            )
                        )
                        if visual_store_summary is not None
                        else 0
                    ),
                    "visual_asset_vision_required_count": sum(
                        asset.requires_vision
                        for asset in table_result.visual_assets
                    ),
                    **dict(table_result.visual_metrics),
                },
            )

        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if database_report_id is not None and self.database is not None:
                try:
                    self.database.update_report_status(
                        database_report_id,
                        "failed",
                        validation_message=str(exc),
                    )
                except Exception:
                    pass
            return IngestionResult(
                report_id=report_id,
                company_name=request.company_name,
                report_year=request.report_year,
                report_url=report_url,
                pdf_path=pdf_path,
                status=IngestionStatus.FAILED,
                errors=tuple(errors),
                warnings=tuple(warnings),
                metadata={
                    **dict(request.metadata),
                    "source_mode": (
                        "local_pdf"
                        if local_pdf_path is not None
                        else "url"
                    ),
                    "local_pdf_path": (
                        str(local_pdf_path)
                        if local_pdf_path is not None
                        else None
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Discovery / download / document stages
    # ------------------------------------------------------------------
    def _discover_report_url(self, company_name: str, report_year: int) -> str:
        candidates = list(
            self._call_compatible(
                self.report_search,
                ("search", "find", "discover"),
                company_name,
                report_year,
            )
            or []
        )
        if not candidates:
            raise IngestionPipelineError(
                "Sürdürülebilirlik raporu adayı bulunamadı."
            )

        ranked = list(
            self._call_compatible(
                self.candidate_ranker,
                ("rank", "rank_candidates"),
                candidates,
                company_name=company_name,
                report_year=report_year,
            )
            or candidates
        )

        for candidate in ranked:
            if not self._extract_value(candidate, ("url", "pdf_url", "link")):
                continue
            validation = self._call_compatible(
                self.url_validator,
                ("validate", "is_valid"),
                candidate,
            )
            if not self._is_successful_validation(validation):
                continue
            validated_url = self._extract_value(
                validation,
                ("pdf_url", "resolved_url", "url", "final_url"),
            ) or self._extract_value(candidate, ("url", "pdf_url", "link"))
            if validated_url:
                return str(validated_url)

        raise IngestionPipelineError(
            "Doğrulanabilir rapor URL'si bulunamadı."
        )

    @staticmethod
    def _resolve_local_pdf(pdf_path: str | Path) -> Path:
        """Yerel PDF yolunu doğrular ve indirme yapmadan kullanıma hazırlar."""

        resolved = Path(pdf_path).expanduser().resolve()

        if resolved.suffix.casefold() != ".pdf":
            raise IngestionPipelineError(
                f"Yerel rapor dosyası PDF olmalıdır: {resolved}"
            )

        if not resolved.is_file():
            raise IngestionPipelineError(
                f"Yerel PDF dosyası bulunamadı: {resolved}"
            )

        return resolved

    def _download_pdf(
        self,
        url: str,
        *,
        report_id: str,
        overwrite: bool,
    ) -> Path:
        result = self._call_compatible(
            self.pdf_downloader,
            ("download",),
            url,
            filename=f"{report_id}.pdf",
            overwrite=overwrite,
        )
        path_value = self._extract_value(
            result,
            ("path", "file_path", "pdf_path", "local_path"),
        ) or result
        pdf_path = Path(path_value).expanduser().resolve()
        if not pdf_path.is_file():
            raise IngestionPipelineError(
                f"İndirilen PDF bulunamadı: {pdf_path}"
            )
        return pdf_path

    def _validate_pdf(self, pdf_path: Path) -> Any:
        result = self._call_compatible(
            self.pdf_validator,
            ("validate",),
            pdf_path,
        )
        if not self._is_successful_validation(result):
            message = self._extract_value(
                result,
                ("message", "error", "validation_message"),
            )
            raise IngestionPipelineError(
                f"PDF doğrulaması başarısız oldu{': ' + str(message) if message else '.'}"
            )
        return result

    def _render_pages(self, pdf_path: Path) -> list[Any]:
        pages = list(
            self._call_compatible(
                self.page_renderer,
                ("render", "render_pages"),
                pdf_path,
            )
            or []
        )
        if not pages:
            raise IngestionPipelineError("PDF sayfaları render edilemedi.")
        return pages

    def _classify_pages(self, pages: Sequence[Any]) -> list[Any]:
        result = self._call_compatible(
            self.page_classifier,
            ("classify_many", "classify_pages"),
            pages,
            required=False,
        )
        if result is not None:
            return list(result)
        return [
            self._call_compatible(
                self.page_classifier,
                ("classify",),
                page,
            )
            for page in pages
        ]

    def _detect_layouts(self, pages: Sequence[Any]) -> list[Any]:
        result = self._call_compatible(
            self.layout_detector,
            ("detect_many", "detect_layouts"),
            pages,
            required=False,
        )
        if result is not None:
            return list(result)
        return [
            self._call_compatible(
                self.layout_detector,
                ("detect",),
                page,
            )
            for page in pages
        ]

    # ------------------------------------------------------------------
    # Table processing
    # ------------------------------------------------------------------
    def _process_tables(
        self,
        *,
        pdf_path: Path,
        layouts: Sequence[Any],
        report_id: str,
        company_name: str,
        report_year: int,
        warnings: list[str],
    ) -> _TableProcessingResult:
        layouts_by_page = {
            int(getattr(layout, "page_number", index + 1)): layout
            for index, layout in enumerate(layouts)
        }

        detections = list(
            self.table_detector.detect_pdf(
                pdf_path,
                layouts_by_page=layouts_by_page,
            )
            or []
        )
        detected_by_page: dict[int, tuple[Any, ...]] = {}
        detected_by_id: dict[str, Any] = {}
        all_detected: list[Any] = []
        for page_result in detections:
            page_number = int(getattr(page_result, "page_number", 0))
            page_tables = tuple(getattr(page_result, "tables", ()) or ())
            detected_by_page[page_number] = page_tables
            all_detected.extend(page_tables)
            for detected_table in page_tables:
                detected_by_id[str(detected_table.table_id)] = (
                    detected_table
                )
            warnings.extend(str(item) for item in getattr(page_result, "warnings", ()) or ())

        if not all_detected:
            return _TableProcessingResult(
                tables=(),
                validations_by_table_id={},
                visual_ocr_chunks=(),
                visual_chart_chunks=(),
                visual_assets=(),
                visual_metrics={
                    "detected_visual_count": 0,
                    "classified_chart_count": 0,
                    "cloudflare_call_count": 0,
                    "cloudflare_processed_chart_count": 0,
                    "cloudflare_skipped_visual_count": 0,
                },
            )

        native_page_results = list(
            self.native_table_extractor.extract_pdf(
                pdf_path,
                detected_tables_by_page=detected_by_page,
            )
            or []
        )
        native_by_id: dict[str, Any] = {}
        for page_result in native_page_results:
            warnings.extend(
                str(item)
                for item in getattr(page_result, "warnings", ()) or ()
                if str(item).strip()
                != "Sayfada native tablo bulunamadı."
            )
            for table in getattr(page_result, "tables", ()) or ():
                native_by_id[str(table.table_id)] = table

        # Native extraction geçerli sonuç üretmeyen bölgeleri önce PDF kelime
        # koordinatlarından çizgisiz tablo olarak yeniden kurmayı dener.
        borderless_by_id: dict[str, Any] = {}
        borderless_detection_map: dict[int, list[Any]] = {}

        if self.borderless_table_extractor is not None:
            for detected in all_detected:
                table_id = str(detected.table_id)
                native = native_by_id.get(table_id)
                native_is_valid = bool(
                    native is not None
                    and getattr(native, "is_valid", False)
                )

                if native_is_valid:
                    continue

                borderless_detection_map.setdefault(
                    int(detected.page_number),
                    [],
                ).append(detected)

        if borderless_detection_map:
            try:
                borderless_page_results = list(
                    self.borderless_table_extractor.extract_pdf(
                        pdf_path,
                        detected_tables_by_page=borderless_detection_map,
                        page_numbers=sorted(borderless_detection_map),
                    )
                    or []
                )

                for page_result in borderless_page_results:
                    warnings.extend(
                        str(item)
                        for item in getattr(
                            page_result,
                            "warnings",
                            (),
                        )
                        or ()
                    )

                    for table in getattr(
                        page_result,
                        "tables",
                        (),
                    ) or ():
                        borderless_by_id[
                            str(table.table_id)
                        ] = table

            except Exception as exc:
                warnings.append(
                    "Çizgisiz tablo çıkarımı başarısız: "
                    f"{exc}"
                )

        # Görsel/OCR hattı yalnızca native ve borderless yöntemlerinin ikisi de
        # geçerli sonuç üretmediğinde çalışır. Böylece koordinat tabanlı,
        # güvenilir bir tablo için gereksiz OCR veya harici VLM çağrısı yapılmaz.
        visual_detection_map: dict[int, list[Any]] = {}

        for detected in all_detected:
            table_id = str(detected.table_id)
            native = native_by_id.get(table_id)
            borderless = borderless_by_id.get(table_id)

            native_is_valid = bool(
                native is not None
                and getattr(native, "is_valid", False)
            )
            borderless_is_valid = bool(
                borderless is not None
                and getattr(borderless, "is_valid", False)
            )

            needs_visual = not (
                native_is_valid
                or borderless_is_valid
            )

            if needs_visual:
                visual_detection_map.setdefault(
                    int(detected.page_number),
                    [],
                ).append(detected)

        visual_page_results: list[Any] = []
        visual_by_id: dict[str, Any] = {}
        if visual_detection_map:
            output_dir = (
                self.table_image_directory / report_id
                if self.table_image_directory is not None
                else None
            )
            try:
                visual_page_results = list(
                    self.visual_table_extractor.extract_pdf(
                        pdf_path,
                        detected_tables_by_page=visual_detection_map,
                        output_dir=output_dir,
                        page_numbers=sorted(visual_detection_map),
                    )
                    or []
                )
                for page_result in visual_page_results:
                    warnings.extend(
                        str(item)
                        for item in getattr(page_result, "warnings", ()) or ()
                    )
                    for table in getattr(page_result, "tables", ()) or ():
                        visual_by_id[str(table.table_id)] = table
            except Exception as exc:
                warnings.append(f"Görsel tablo çıkarımı başarısız: {exc}")

        ocr_by_id: dict[str, Any] = {}
        if visual_page_results and self.table_ocr is not None:
            try:
                ocr_results = list(
                    self.table_ocr.process_pages(visual_page_results) or []
                )
                for page_result in ocr_results:
                    warnings.extend(
                        str(item)
                        for item in getattr(page_result, "warnings", ()) or ()
                    )
                    for table in getattr(page_result, "tables", ()) or ():
                        ocr_by_id[str(table.table_id)] = table
            except Exception as exc:
                warnings.append(f"Tablo OCR işlemi başarısız: {exc}")

        visual_ocr_chunks = self._build_visual_ocr_chunks(
            ocr_by_id=ocr_by_id,
            visual_by_id=visual_by_id,
            detected_by_id=detected_by_id,
            report_id=report_id,
            company_name=company_name,
            report_year=report_year,
        )

        visual_assets = self._build_visual_assets(
            visual_by_id=visual_by_id,
            ocr_by_id=ocr_by_id,
            detected_by_id=detected_by_id,
            report_id=report_id,
        )

        normalized_tables: list[Any] = []
        validations: dict[str, Any] = {}
        chart_by_id: dict[str, Any] = {}

        table_vlm_enabled = str(
            os.getenv(
                "GROQ_ENABLE_TABLE_FALLBACK",
                "false",
            )
        ).strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }

        print(
            "Groq tablo VLM fallback'i "
            + ("açık." if table_vlm_enabled else "kapalı; yalnız grafikler işlenecek."),
            flush=True,
        )

        for detected in all_detected:
            table_id = str(detected.table_id)
            native = native_by_id.get(table_id)
            borderless = borderless_by_id.get(table_id)
            visual = visual_by_id.get(table_id)
            ocr = ocr_by_id.get(table_id)
            candidate: Any | None = None

            if native is not None and bool(
                getattr(native, "is_valid", False)
            ):
                candidate = native

            if (
                candidate is None
                and borderless is not None
                and bool(
                    getattr(borderless, "is_valid", False)
                )
            ):
                candidate = borderless

            if ocr is not None and bool(
                getattr(ocr, "is_valid", False)
            ):
                # OCR yalnızca önceki yöntemler geçerli sonuç üretmediyse
                # seçilir. Borderless geçerliyken detected.requires_ocr
                # bayrağının OCR'ı gereksiz yere zorlamasına izin verilmez.
                if candidate is None:
                    candidate = ocr

            if visual is not None and self.vlm_table_fallback is not None:
                # Cloudflare kotasını yalnızca gerçek grafikler için kullan.
                # Tablo OCR/VLM fallback'i varsayılan olarak kapalıdır.
                enable_table_vlm = str(
                    os.getenv(
                        "GROQ_ENABLE_TABLE_FALLBACK",
                        "false",
                    )
                ).strip().casefold() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }

                try:
                    region_type = str(
                        getattr(visual, "visual_region_type", "table")
                        or "table"
                    ).casefold()

                    if region_type == "chart" and callable(
                        getattr(
                            self.vlm_table_fallback,
                            "extract_chart",
                            None,
                        )
                    ):
                        chart = self.vlm_table_fallback.extract_chart(
                            visual_table=visual,
                            contextual_text=getattr(
                                detected,
                                "text_preview",
                                None,
                            ),
                        )
                        if bool(getattr(chart, "is_valid", False)):
                            chart_by_id[table_id] = chart
                            candidate = chart

                    elif enable_table_vlm:
                        decision = (
                            self.vlm_table_fallback
                            .evaluate_fallback_need(
                                visual_table=visual,
                                ocr_table=ocr,
                            )
                        )
                        if bool(
                            getattr(
                                decision,
                                "should_use_vlm",
                                False,
                            )
                        ):
                            vlm_table = (
                                self.vlm_table_fallback.extract_table(
                                    visual_table=visual,
                                    ocr_table=ocr,
                                    contextual_text=getattr(
                                        detected,
                                        "text_preview",
                                        None,
                                    ),
                                )
                            )
                            if bool(
                                getattr(
                                    vlm_table,
                                    "is_valid",
                                    False,
                                )
                            ):
                                candidate = vlm_table

                except CloudflareQuotaExceededError:
                    raise

                except Exception as exc:
                    warnings.append(
                        f"Görsel {table_id} VLM fallback başarısız: {exc}"
                    )

            if candidate is None:
                # Geçersiz sonuçlar da ham matrix taşıyabilir. Normalizer'a
                # son çare olarak en bilgi zengini sırayla aktarılır.
                candidate = (
                    borderless
                    or native
                    or ocr
                )

            if candidate is None or getattr(
                candidate,
                "matrix",
                None,
            ) is None:
                warnings.append(
                    f"Tablo {table_id} için normalize edilebilir içerik bulunamadı."
                )
                continue

            try:
                normalized = self.table_normalizer.normalize_table(candidate)
                # Topic sınıflandırması yalnızca tablonun kendi içeriğine
                # dayanır. Sayfanın geniş text_preview alanı kullanılırsa,
                # aynı sayfadaki ilgisiz bir "energy", "water" veya "waste"
                # kelimesi tabloyu yanlış konuya taşıyabilir.
                table_classification_text = "\n".join(
                    value
                    for value in (
                        str(getattr(candidate, "caption", "") or "").strip(),
                        self._table_text_for_classification(normalized),
                    )
                    if value
                )
                table_category, table_topics = (
                    self._infer_table_category_topics(
                        table_classification_text
                    )
                )
                table_topic = (
                    table_topics[0] if table_topics else None
                )
                normalized = self._with_metadata(
                    normalized,
                    {
                        "report_id": report_id,
                        "company_name": company_name,
                        "report_year": report_year,
                        "page_number": int(
                            getattr(normalized, "page_number", 0)
                        ),
                        "source_pdf": str(pdf_path),
                        "selected_extraction_method": str(
                            getattr(
                                candidate,
                                "extraction_method",
                                getattr(
                                    detected,
                                    "detection_method",
                                    "unknown",
                                ),
                            )
                        ),
                        "borderless_attempted": (
                            borderless is not None
                        ),
                        "borderless_valid": bool(
                            borderless is not None
                            and getattr(
                                borderless,
                                "is_valid",
                                False,
                            )
                        ),
                        "visual_fallback_used": (
                            visual is not None
                        ),
                        "visual_region_type": str(
                            getattr(
                                visual,
                                "visual_region_type",
                                "table",
                            )
                            if visual is not None
                            else "table"
                        ),
                        "chart_fallback_used": (
                            table_id in chart_by_id
                        ),
                        "ocr_fallback_used": (
                            candidate is ocr
                        ),
                        "category": table_category,
                        "topic": table_topic,
                        "topics": table_topics,
                    },
                )
                validation = self.numeric_validator.validate_table(normalized)
                validations[str(normalized.table_id)] = validation
                normalized_tables.append(normalized)
            except Exception as exc:
                warnings.append(f"Tablo {table_id} normalizasyonu başarısız: {exc}")

        visual_chart_chunks = self._build_visual_chart_chunks(
            chart_by_id=chart_by_id,
            detected_by_id=detected_by_id,
            report_id=report_id,
            company_name=company_name,
            report_year=report_year,
        )

        detected_visual_count = len(visual_assets)

        visual_type_counts: dict[str, int] = {}
        visual_type_sample_pages: dict[str, list[int]] = {}

        for asset in visual_assets:
            region_type = self._enum_value(
                asset.region_type
            ).casefold()

            visual_type_counts[region_type] = (
                visual_type_counts.get(region_type, 0) + 1
            )

            pages = visual_type_sample_pages.setdefault(
                region_type,
                [],
            )
            page_number = int(asset.page_number)

            if (
                page_number not in pages
                and len(pages) < 10
            ):
                pages.append(page_number)

        classified_chart_count = int(
            visual_type_counts.get("chart", 0)
        )

        vlm_usage = {}
        get_usage_counters = getattr(
            self.vlm_table_fallback,
            "get_usage_counters",
            None,
        )
        if callable(get_usage_counters):
            vlm_usage = dict(get_usage_counters())

        cloudflare_processed_chart_count = int(
            vlm_usage.get(
                "cloudflare_processed_chart_count",
                len(chart_by_id),
            )
            or 0
        )
        cloudflare_call_count = int(
            vlm_usage.get("cloudflare_call_count", 0) or 0
        )
        cloudflare_skipped_visual_count = max(
            detected_visual_count - cloudflare_processed_chart_count,
            0,
        )

        visual_metrics = {
            "detected_visual_count": detected_visual_count,
            "classified_chart_count": classified_chart_count,
            "classified_table_visual_count": int(
                visual_type_counts.get("table", 0)
            ),
            "classified_metric_card_count": int(
                visual_type_counts.get("metric_card", 0)
            ),
            "classified_infographic_count": int(
                visual_type_counts.get("infographic", 0)
            ),
            "classified_figure_count": int(
                visual_type_counts.get("figure", 0)
            ),
            "classified_unknown_visual_count": int(
                visual_type_counts.get("unknown", 0)
            ),
            "cloudflare_call_count": cloudflare_call_count,
            "cloudflare_processed_chart_count": (
                cloudflare_processed_chart_count
            ),
            "cloudflare_skipped_visual_count": (
                cloudflare_skipped_visual_count
            ),
        }

        type_summary = ", ".join(
            f"{region_type}={count}"
            for region_type, count in sorted(
                visual_type_counts.items()
            )
        )

        sample_summary = "; ".join(
            f"{region_type}:pages={pages}"
            for region_type, pages in sorted(
                visual_type_sample_pages.items()
            )
        )

        print(
            "Visual ingestion özeti: "
            f"detected={detected_visual_count}, "
            f"charts={classified_chart_count}, "
            f"cloudflare_calls={cloudflare_call_count}, "
            f"processed_charts={cloudflare_processed_chart_count}, "
            f"skipped_visuals={cloudflare_skipped_visual_count}",
            flush=True,
        )
        print(
            "Visual tür dağılımı: "
            f"{type_summary or 'kayıt yok'}",
            flush=True,
        )
        print(
            "Visual örnek sayfalar: "
            f"{sample_summary or 'kayıt yok'}",
            flush=True,
        )

        return _TableProcessingResult(
            tables=tuple(normalized_tables),
            validations_by_table_id=validations,
            visual_ocr_chunks=tuple(visual_ocr_chunks),
            visual_chart_chunks=tuple(visual_chart_chunks),
            visual_assets=tuple(visual_assets),
            visual_metrics=visual_metrics,
        )

    @staticmethod
    def _table_text_for_classification(table: Any) -> str:
        """Normalize edilmiş tabloyu düşük maliyetli kategori metnine çevirir."""

        parts: list[str] = []
        for attribute_name in ("title", "caption"):
            value = getattr(table, attribute_name, None)
            if value is not None and not callable(value):
                cleaned = str(value).strip()
                if cleaned:
                    parts.append(cleaned)

        headers = getattr(table, "headers", ()) or ()
        parts.extend(str(item).strip() for item in headers if str(item).strip())

        for row in getattr(table, "rows", ()) or ():
            cells = getattr(row, "cells", row)
            try:
                iterable = list(cells)
            except TypeError:
                iterable = [cells]
            row_values: list[str] = []
            for cell in iterable:
                value = (
                    getattr(cell, "normalized_text", None)
                    or getattr(cell, "raw_text", None)
                    or cell
                )
                if value is not None and not callable(value):
                    cleaned = str(value).strip()
                    if cleaned:
                        row_values.append(cleaned)
            if row_values:
                parts.append(" | ".join(row_values))

        return "\n".join(parts)

    def _build_visual_assets(
        self,
        *,
        visual_by_id: Mapping[str, Any],
        ocr_by_id: Mapping[str, Any],
        detected_by_id: Mapping[str, Any],
        report_id: str,
    ) -> list[VisualAsset]:
        """
        Görsel çıkarım sonuçlarını kalıcı `VisualAsset` sözleşmesine dönüştürür.

        Rapor ve görsel UUID değerleri string rapor/table kimliklerinden
        deterministik olarak üretilir. Böylece yeniden çalıştırmalarda aynı
        görsel aynı kayıtla eşleşir.
        """

        report_uuid = uuid5(
            NAMESPACE_URL,
            f"sustainability-report:{report_id}",
        )

        assets: list[VisualAsset] = []

        for table_id, visual in sorted(
            visual_by_id.items(),
            key=lambda item: (
                int(getattr(item[1], "page_number", 0) or 0),
                int(getattr(item[1], "table_index", 0) or 0),
                item[0],
            ),
        ):
            detected = detected_by_id.get(table_id)
            ocr = ocr_by_id.get(table_id)

            ocr_metadata = dict(
                getattr(ocr, "metadata", {}) or {}
            ) if ocr is not None else {}

            raw_ocr_text = str(
                ocr_metadata.get("whole_image_raw_text")
                or ""
            ).strip()

            if not raw_ocr_text and ocr is not None:
                matrix = getattr(ocr, "matrix", ()) or ()
                raw_ocr_text = "\n".join(
                    " | ".join(
                        str(cell).strip()
                        for cell in row
                        if str(cell).strip()
                    )
                    for row in matrix
                    if any(str(cell).strip() for cell in row)
                ).strip()

            caption = str(
                getattr(visual, "caption_text", "")
                or getattr(visual, "metadata", {}).get(
                    "caption_text",
                    "",
                )
                or ""
            ).strip()

            nearby_text = str(
                getattr(detected, "text_preview", "")
                or getattr(visual, "metadata", {}).get(
                    "text_preview",
                    "",
                )
                or ""
            ).strip()

            category, topic = self._infer_visual_category_topic(
                "\n".join(
                    value
                    for value in (
                        caption,
                        nearby_text,
                        raw_ocr_text,
                    )
                    if value
                )
            )

            page_number = int(
                getattr(visual, "page_number", 0)
                or getattr(detected, "page_number", 0)
                or 1
            )
            page_uuid = uuid5(
                NAMESPACE_URL,
                f"sustainability-report:{report_id}:page:{page_number}",
            )

            try:
                asset = visual.to_visual_asset(
                    report_id=report_uuid,
                    page_id=page_uuid,
                    category=category,
                    topic=topic,
                    keywords=(topic,) if topic else (),
                )
            except Exception as exc:
                # Görsel dosyası olmayan kayıt repository'ye uygun değildir.
                # Bu durum tablo uyarılarında zaten görünür; kalan görsellerin
                # saklanmasını engellemez.
                continue

            deterministic_visual_id = uuid5(
                NAMESPACE_URL,
                f"visual-table:{table_id}",
            )
            asset.id = deterministic_visual_id
            asset.caption = caption or asset.caption
            asset.nearby_text = nearby_text or asset.nearby_text
            asset.category = category
            asset.topic = topic
            asset.metadata.update(
                {
                    "report_uid": report_id,
                    "source_table_id": table_id,
                    "database_page_number": page_number,
                }
            )

            if ocr is None:
                asset.requires_vision = True
                asset.processing_status = (
                    VisualProcessingStatus.VISION_REQUIRED
                )
                asset.add_error(
                    "Görsel için OCR sonucu üretilemedi."
                )
                assets.append(asset)
                continue

            confidence = float(
                getattr(ocr, "average_confidence", 0.0) or 0.0
            )
            if confidence > 1.0:
                confidence /= 100.0
            confidence = max(0.0, min(confidence, 1.0))

            strategy = str(
                ocr_metadata.get("ocr_strategy")
                or "cell"
            ).strip().casefold()
            extraction_method = (
                VisualExtractionMethod.WHOLE_IMAGE_OCR
                if strategy == "whole_image"
                else VisualExtractionMethod.CELL_OCR
            )
            if extraction_method not in asset.extraction_methods:
                asset.extraction_methods.append(extraction_method)

            asset.ocr_text = raw_ocr_text or None
            asset.ocr_confidence = confidence
            asset.requires_vision = bool(
                getattr(ocr, "requires_vlm_fallback", False)
            )

            if raw_ocr_text and not asset.requires_vision:
                asset.processing_status = (
                    VisualProcessingStatus.OCR_COMPLETED
                )
            elif raw_ocr_text:
                asset.processing_status = (
                    VisualProcessingStatus.OCR_PARTIAL
                )
            else:
                asset.processing_status = (
                    VisualProcessingStatus.VISION_REQUIRED
                )
                asset.requires_vision = True

            for warning in getattr(ocr, "warnings", ()) or ():
                asset.add_error(str(warning))

            asset.metadata.update(
                {
                    "ocr_strategy": strategy,
                    "ocr_extraction_method": str(
                        getattr(ocr, "extraction_method", "ocr")
                    ),
                    "ocr_is_valid": bool(
                        getattr(ocr, "is_valid", False)
                    ),
                    "ocr_requires_vision": asset.requires_vision,
                    "ocr_row_count": int(
                        getattr(ocr, "row_count", 0) or 0
                    ),
                    "ocr_column_count": int(
                        getattr(ocr, "column_count", 0) or 0
                    ),
                }
            )

            assets.append(asset)

        return assets

    def _build_visual_chart_chunks(
        self,
        *,
        chart_by_id: Mapping[str, Any],
        detected_by_id: Mapping[str, Any],
        report_id: str,
        company_name: str,
        report_year: int,
    ) -> list[_VisualOCRChunk]:
        """VLM grafik noktalarını retrieval için satır bazlı chunk'lara çevirir."""

        chunks: list[_VisualOCRChunk] = []

        for position, (table_id, chart) in enumerate(
            sorted(
                chart_by_id.items(),
                key=lambda item: (
                    int(getattr(item[1], "page_number", 0) or 0),
                    item[0],
                ),
            )
        ):
            page_number = int(
                getattr(chart, "page_number", 0) or 1
            )
            title = str(
                getattr(chart, "title", "") or "Visual chart"
            ).strip()
            detected = detected_by_id.get(table_id)
            nearby_text = str(
                getattr(detected, "text_preview", "") or ""
            ).strip()

            matrix = getattr(chart, "matrix", ()) or ()
            rows = [
                " | ".join(str(cell).strip() for cell in row)
                for row in matrix
                if any(str(cell).strip() for cell in row)
            ]
            if not rows:
                continue

            raw_text = "\n".join(
                [
                    title,
                    *rows,
                ]
            )
            category, topic = self._infer_visual_category_topic(
                "\n".join(
                    value
                    for value in (
                        title,
                        nearby_text,
                        raw_text,
                    )
                    if value
                )
            )
            normalized_text = re.sub(
                r"\s+",
                " ",
                raw_text,
            ).strip()
            content_hash = hashlib.sha256(
                normalized_text.encode("utf-8")
            ).hexdigest()
            chunk_id = (
                f"{report_id}_visual_chart_"
                f"{self._slugify(table_id)}"
            )

            chunks.append(
                _VisualOCRChunk(
                    chunk_id=chunk_id,
                    report_id=report_id,
                    chunk_index=1_100_000 + position,
                    page_number=page_number,
                    start_page=page_number,
                    end_page=page_number,
                    text=raw_text,
                    normalized_text=normalized_text,
                    searchable_text="\n".join(
                        value
                        for value in (
                            title,
                            nearby_text,
                            raw_text,
                        )
                        if value
                    ),
                    contextualized_text=(
                        f"{company_name} {report_year} "
                        f"sürdürülebilirlik raporu. Sayfa {page_number}. "
                        f"Yapılandırılmış grafik kaynağı: {table_id}.\n"
                        f"{raw_text}"
                    ),
                    section_title=title,
                    category=category,
                    source_type="visual_chart",
                    content_hash=content_hash,
                    metadata={
                        "report_id": report_id,
                        "company_name": company_name,
                        "report_year": report_year,
                        "page_number": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                        "table_id": table_id,
                        "source_type": "visual_chart",
                        "extraction_method": getattr(
                            chart,
                            "extraction_method",
                            "vision_chart",
                        ),
                        "chart_type": getattr(
                            chart,
                            "chart_type",
                            None,
                        ),
                        "chart_title": title,
                        "chart_unit": getattr(chart, "unit", None),
                        "chart_years": tuple(
                            getattr(chart, "years", ()) or ()
                        ),
                        "chart_series": tuple(
                            getattr(
                                chart,
                                "series_names",
                                (),
                            )
                            or ()
                        ),
                        "chart_confidence": float(
                            getattr(
                                chart,
                                "overall_confidence",
                                0.0,
                            )
                            or 0.0
                        ),
                        "requires_vision": False,
                        "category": category,
                        "topic": topic,
                        "topics": (topic,) if topic else (),
                        "is_visual_evidence": True,
                        "is_chart_evidence": True,
                        "content_hash": content_hash,
                    },
                )
            )

        return chunks

    def _build_visual_ocr_chunks(
        self,
        *,
        ocr_by_id: Mapping[str, Any],
        visual_by_id: Mapping[str, Any],
        detected_by_id: Mapping[str, Any],
        report_id: str,
        company_name: str,
        report_year: int,
    ) -> list[_VisualOCRChunk]:
        """
        OCR ile okunan görsel tablo/grafik metinlerini retrieval chunk'larına
        dönüştürür.

        Normalizasyon başarısız olsa bile ham OCR metni korunur. Böylece
        BM25 ve vector retrieval görseldeki başlık, yıl, birim ve sayıları
        arayabilir.
        """

        chunks: list[_VisualOCRChunk] = []

        sorted_items = sorted(
            ocr_by_id.items(),
            key=lambda item: (
                int(getattr(item[1], "page_number", 0) or 0),
                int(getattr(item[1], "table_index", 0) or 0),
                item[0],
            ),
        )

        for position, (table_id, ocr_table) in enumerate(sorted_items):
            metadata = dict(
                getattr(ocr_table, "metadata", {}) or {}
            )

            raw_text = str(
                metadata.get("whole_image_raw_text")
                or ""
            ).strip()

            if not raw_text:
                matrix = getattr(ocr_table, "matrix", ()) or ()
                raw_text = "\n".join(
                    " | ".join(
                        str(cell).strip()
                        for cell in row
                        if str(cell).strip()
                    )
                    for row in matrix
                    if any(str(cell).strip() for cell in row)
                ).strip()

            if not raw_text:
                continue

            visual = visual_by_id.get(table_id)
            detected = detected_by_id.get(table_id)

            page_number = int(
                getattr(ocr_table, "page_number", 0)
                or getattr(visual, "page_number", 0)
                or getattr(detected, "page_number", 0)
                or 1
            )

            caption = str(
                getattr(visual, "caption_text", "")
                or metadata.get("caption_text")
                or ""
            ).strip()

            nearby_text = str(
                getattr(detected, "text_preview", "")
                or metadata.get("text_preview")
                or ""
            ).strip()

            combined_for_classification = "\n".join(
                item
                for item in (
                    caption,
                    nearby_text,
                    raw_text,
                )
                if item
            )

            category, topic = self._infer_visual_category_topic(
                combined_for_classification
            )

            source_image = (
                metadata.get("source_table_image")
                or getattr(visual, "table_image_path", None)
            )

            confidence = float(
                getattr(ocr_table, "average_confidence", 0.0)
                or 0.0
            )

            if confidence > 1.0:
                confidence /= 100.0

            contextualized_text = (
                f"{company_name} {report_year} sürdürülebilirlik raporu. "
                f"Sayfa {page_number}. "
                f"Görsel OCR kaynağı: {table_id}.\n"
                f"{raw_text}"
            )

            normalized_text = re.sub(
                r"\s+",
                " ",
                raw_text,
            ).strip()

            content_hash = hashlib.sha256(
                normalized_text.encode("utf-8")
            ).hexdigest()

            chunk_id = (
                f"{report_id}_visual_ocr_"
                f"{self._slugify(table_id)}"
            )

            chunks.append(
                _VisualOCRChunk(
                    chunk_id=chunk_id,
                    report_id=report_id,
                    chunk_index=1_000_000 + position,
                    page_number=page_number,
                    start_page=page_number,
                    end_page=page_number,
                    text=raw_text,
                    normalized_text=normalized_text,
                    searchable_text="\n".join(
                        item
                        for item in (
                            caption,
                            nearby_text,
                            raw_text,
                        )
                        if item
                    ),
                    contextualized_text=contextualized_text,
                    section_title=(
                        caption
                        or nearby_text[:160]
                        or "Visual OCR"
                    ),
                    category=category,
                    source_type="visual_ocr",
                    content_hash=content_hash,
                    metadata={
                        "report_id": report_id,
                        "company_name": company_name,
                        "report_year": report_year,
                        "page_number": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                        "table_id": table_id,
                        "visual_id": metadata.get(
                            "whole_image_visual_id"
                        ),
                        "source_type": "visual_ocr",
                        "extraction_method": getattr(
                            ocr_table,
                            "extraction_method",
                            "ocr",
                        ),
                        "ocr_strategy": metadata.get(
                            "ocr_strategy",
                            "cell",
                        ),
                        "ocr_confidence": confidence,
                        "ocr_is_structured": metadata.get(
                            "whole_image_is_structured"
                        ),
                        "requires_vision": bool(
                            getattr(
                                ocr_table,
                                "requires_vlm_fallback",
                                False,
                            )
                        ),
                        "source_image_path": (
                            str(source_image)
                            if source_image
                            else None
                        ),
                        "caption": caption or None,
                        "nearby_text": nearby_text or None,
                        "category": category,
                        "topic": topic,
                        "topics": (
                            (topic,)
                            if topic
                            else ()
                        ),
                        "is_visual_evidence": True,
                        "content_hash": content_hash,
                    },
                )
            )

        return chunks

    @staticmethod
    def _infer_table_category_topics(
        text: str,
    ) -> tuple[str, tuple[str, ...]]:
        """
        Tablo metninden geniş ESG kategorisi ve alt konuları çıkarır.

        Tek bir genel kelime topic atamak için yeterli değildir. Öncelik,
        metrik adı + tipik birim/yapı taşıyan güçlü ifadelerdedir.
        """

        normalized = IngestionPipeline._slugify_for_matching(text)

        topic_rules: tuple[
            tuple[
                str,
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ],
            ...
        ] = (
            (
                "environmental",
                "energy",
                (
                    "total energy consumption",
                    "direct energy consumption",
                    "indirect energy consumption",
                    "electricity consumption",
                    "natural gas consumption",
                    "renewable energy generated",
                    "renewable electricity generated",
                    "energy intensity",
                    "fuel oil",
                    "coal consumption",
                ),
                (
                    "energy", "electricity", "natural gas",
                    "renewable", "coal", "fuel oil",
                ),
                ("gj", "mwh", "kwh", "sm3", "m3"),
            ),
            (
                "environmental",
                "water",
                (
                    "total water consumption",
                    "total water withdrawal",
                    "freshwater withdrawal",
                    "water discharged",
                    "water intensity",
                    "recycled water",
                    "reused water",
                    "wastewater",
                ),
                (
                    "water", "freshwater", "withdrawal",
                    "discharge", "wastewater",
                ),
                ("m3", "cubic meter", "tons"),
            ),
            (
                "environmental",
                "waste",
                (
                    "hazardous waste",
                    "non hazardous waste",
                    "waste to landfill",
                    "waste sent to landfill",
                    "recycled waste",
                    "recovered waste",
                    "total waste generated",
                    "waste diverted from disposal",
                ),
                (
                    "waste", "hazardous", "landfill",
                    "recycled", "recovered",
                ),
                ("ton", "tons", "kg"),
            ),
            (
                "environmental",
                "greenhouse_gas_emissions",
                (
                    "scope 1 greenhouse gas emissions",
                    "scope 2 greenhouse gas emissions",
                    "scope 3 greenhouse gas emissions",
                    "greenhouse gas emission intensity",
                    "direct greenhouse gas emissions",
                    "indirect greenhouse gas emissions",
                ),
                (
                    "scope 1", "scope 2", "scope 3",
                    "greenhouse gas", "carbon emission", "co2e",
                ),
                ("tco2e", "co2e", "ton co2", "tons co2"),
            ),
            (
                "environmental",
                "air_pollutant_emissions",
                (
                    "air emissions",
                    "nox emissions",
                    "sox emissions",
                    "so2 emissions",
                    "voc emissions",
                    "particulate matter emissions",
                    "total organic carbon",
                ),
                (
                    "nox", "sox", "so2", "voc", "toc",
                    "particulate matter", "air pollutant",
                ),
                ("ton", "tons", "kg"),
            ),
            (
                "environmental",
                "resource_use_and_materials",
                (
                    "raw material consumption",
                    "material consumption",
                    "recycled material content",
                ),
                (
                    "raw material", "material consumption",
                    "recycled material",
                ),
                ("ton", "tons", "kg", "%"),
            ),
            (
                "social",
                "employees",
                (
                    "total employees",
                    "total workforce",
                    "female employees",
                    "male employees",
                    "employee turnover",
                ),
                (
                    "employee", "employees", "workforce",
                    "female", "male", "manager",
                ),
                ("person", "people", "%"),
            ),
            (
                "social",
                "occupational_health_and_safety",
                (
                    "lost time injury frequency rate",
                    "total recordable incident rate",
                    "work accidents",
                    "occupational disease",
                ),
                (
                    "ltifr", "trir", "injury",
                    "work accident", "occupational",
                ),
                ("rate", "%", "hours"),
            ),
            (
                "economic",
                "economic_performance",
                (
                    "net sales",
                    "net profit",
                    "economic value generated",
                    "revenue",
                    "profit attributable",
                ),
                (
                    "sales", "revenue", "profit",
                    "economic value",
                ),
                ("try", "tl", "eur", "usd", "million", "thousand"),
            ),
        )

        matched: list[tuple[str, str, float]] = []

        for category, topic, strong_phrases, weak_terms, units in topic_rules:
            strong_hits = sum(
                1
                for phrase in strong_phrases
                if IngestionPipeline._slugify_for_matching(phrase)
                in normalized
            )
            weak_hits = sum(
                1
                for term in weak_terms
                if IngestionPipeline._slugify_for_matching(term)
                in normalized
            )
            unit_hits = sum(
                1
                for unit in units
                if IngestionPipeline._slugify_for_matching(unit)
                in normalized
            )

            # Güçlü metrik ifadesi tek başına yeterlidir. Yalnızca genel
            # kelimeler varsa en az iki farklı sinyal ve bir birim aranır.
            if strong_hits > 0:
                score = strong_hits * 3.0 + weak_hits * 0.5 + unit_hits
                matched.append((category, topic, score))
            elif weak_hits >= 2 and unit_hits >= 1:
                score = weak_hits + unit_hits * 0.75
                matched.append((category, topic, score))

        if not matched:
            return "general", ()

        category_scores: dict[str, float] = {}
        for category, _topic, score in matched:
            category_scores[category] = category_scores.get(category, 0.0) + score

        primary_category = max(category_scores, key=category_scores.get)

        topics = tuple(
            topic
            for category, topic, _score in sorted(
                matched,
                key=lambda item: (-item[2], item[1]),
            )
            if category == primary_category
        )
        return primary_category, topics

    @staticmethod
    def _infer_visual_category_topic(
        text: str,
    ) -> tuple[str, str | None]:
        """Görsel OCR metni için düşük maliyetli ESG kategori/topic tahmini."""

        normalized = IngestionPipeline._slugify_for_matching(text)

        topic_patterns: tuple[
            tuple[str, str, tuple[str, ...]],
            ...
        ] = (
            (
                "environmental",
                "water",
                (
                    "water",
                    "su tuketimi",
                    "su cekimi",
                    "water withdrawal",
                    "water consumption",
                    "recycled water",
                    "water intensity",
                ),
            ),
            (
                "environmental",
                "waste",
                (
                    "waste",
                    "atik",
                    "recycling",
                    "geri donusum",
                    "hazardous waste",
                ),
            ),
            (
                "environmental",
                "energy",
                (
                    "energy",
                    "enerji",
                    "electricity",
                    "elektrik",
                    "natural gas",
                    "dogal gaz",
                    "renewable",
                    "yenilenebilir",
                    "coal",
                    "komur",
                ),
            ),
            (
                "environmental",
                "greenhouse_gas_emissions",
                (
                    "scope 1",
                    "scope 2",
                    "greenhouse gas",
                    "sera gazi",
                    "carbon emission",
                    "co2e",
                    "co₂e",
                ),
            ),
            (
                "environmental",
                "air_pollutants",
                (
                    "nox",
                    "sox",
                    "voc",
                    "particulate matter",
                    "air pollut",
                    "hava emisyon",
                ),
            ),
            (
                "environmental",
                "resource_use_and_materials",
                (
                    "raw material",
                    "hammadde",
                    "material consumption",
                    "recycled material",
                ),
            ),
            (
                "social",
                "employees",
                (
                    "employee",
                    "employees",
                    "workforce",
                    "calisan",
                    "kadin calisan",
                    "male employee",
                    "female employee",
                    "manager",
                    "yonetici",
                ),
            ),
            (
                "social",
                "occupational_health_and_safety",
                (
                    "occupational",
                    "work accident",
                    "injury",
                    "ltifr",
                    "trir",
                    "is kazasi",
                    "meslek hastaligi",
                ),
            ),
            (
                "economic",
                "economic_performance",
                (
                    "revenue",
                    "net sales",
                    "net profit",
                    "economic value",
                    "satis geliri",
                    "net kar",
                    "ekonomik deger",
                ),
            ),
        )

        for category, topic, patterns in topic_patterns:
            if any(
                IngestionPipeline._slugify_for_matching(pattern)
                in normalized
                for pattern in patterns
            ):
                return category, topic

        return "general", None

    @staticmethod
    def _slugify_for_matching(value: str) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            str(value).strip().casefold(),
        )
        ascii_text = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        ascii_text = ascii_text.replace("ı", "i")
        return re.sub(
            r"[^a-z0-9%²³]+",
            " ",
            ascii_text,
        ).strip()


    def _process_layout_visuals(
        self,
        *,
        layouts: Sequence[Any],
        existing_assets: Sequence[VisualAsset],
        report_id: str,
        company_name: str,
        report_year: int,
        warnings: list[str],
    ) -> tuple[list[VisualAsset], list[_VisualOCRChunk], dict[str, int]]:
        """Tablo dışındaki layout IMAGE/FIGURE bölgelerini işler."""

        try:
            from document_ingestion.visual_region_classifier import (
                VisualRegionClassifier,
            )
        except Exception as exc:
            warnings.append(
                "Bağımsız görsel bölge sınıflandırıcısı yüklenemedi: "
                f"{exc}"
            )
            return [], [], {
                "layout_visual_candidate_count": 0,
                "layout_chart_candidate_count": 0,
                "layout_chart_processed_count": 0,
                "layout_chart_skipped_count": 0,
            }

        candidates = VisualRegionClassifier().classify_layouts(layouts)
        if not candidates:
            return [], [], {
                "layout_visual_candidate_count": 0,
                "layout_chart_candidate_count": 0,
                "layout_chart_processed_count": 0,
                "layout_chart_skipped_count": 0,
            }

        try:
            max_charts = max(
                int(os.getenv("GROQ_MAX_CHARTS_PER_RUN", "20")),
                0,
            )
        except ValueError:
            max_charts = 20

        try:
            minimum_chart_confidence = float(
                os.getenv("GROQ_MIN_CHART_CONFIDENCE", "0.72")
            )
        except ValueError:
            minimum_chart_confidence = 0.72

        chart_candidates = sorted(
            [
                candidate
                for candidate in candidates
                if candidate.region_type == "chart"
                and candidate.confidence >= minimum_chart_confidence
            ],
            key=lambda item: item.confidence,
            reverse=True,
        )
        allowed_chart_ids = {
            candidate.source_element_id
            for candidate in chart_candidates[:max_charts]
        }

        report_uuid = uuid5(
            NAMESPACE_URL,
            f"sustainability-report:{report_id}",
        )
        layout_by_page = {
            int(getattr(layout, "page_number", 0) or 0): layout
            for layout in layouts
        }
        existing_by_page: dict[int, list[VisualAsset]] = {}
        for asset in existing_assets:
            existing_by_page.setdefault(int(asset.page_number), []).append(asset)

        assets: list[VisualAsset] = []
        chunks: list[_VisualOCRChunk] = []
        processed = 0
        skipped = 0
        quota_exceeded = False

        for candidate in candidates:
            page_number = int(candidate.page_number)
            layout = layout_by_page.get(page_number)
            if layout is None:
                continue
            if self._overlaps_existing_visual_asset(
                candidate.bounding_box,
                existing_by_page.get(page_number, ()),
            ):
                continue

            page_image = getattr(layout, "image_path", None)
            if page_image is None:
                continue

            try:
                cropped_path, width, height, image_hash = self._crop_layout_visual(
                    page_image_path=Path(page_image),
                    page_width_points=float(getattr(layout, "width_points", 0.0) or 0.0),
                    page_height_points=float(getattr(layout, "height_points", 0.0) or 0.0),
                    bounding_box=candidate.bounding_box,
                    report_id=report_id,
                    page_number=page_number,
                    source_element_id=candidate.source_element_id,
                )
            except Exception as exc:
                warnings.append(
                    f"Layout görseli kırpılamadı (sayfa {page_number}): {exc}"
                )
                continue

            region_type = VisualRegionType(candidate.region_type)
            asset = VisualAsset(
                id=uuid5(
                    NAMESPACE_URL,
                    f"layout-visual:{report_id}:{candidate.source_element_id}",
                ),
                report_id=report_uuid,
                page_id=uuid5(
                    NAMESPACE_URL,
                    f"sustainability-report:{report_id}:page:{page_number}",
                ),
                page_number=page_number,
                region_type=region_type,
                bounding_box=BoundingBox(
                    x1=float(candidate.bounding_box.x0),
                    y1=float(candidate.bounding_box.y0),
                    x2=float(candidate.bounding_box.x1),
                    y2=float(candidate.bounding_box.y1),
                ),
                image_path=cropped_path,
                source_page_image_path=Path(page_image),
                image_width=width,
                image_height=height,
                image_hash=image_hash,
                caption=candidate.caption,
                nearby_text=candidate.nearby_text,
                detection_confidence=candidate.confidence,
                processing_status=(
                    VisualProcessingStatus.VISION_REQUIRED
                    if region_type in {
                        VisualRegionType.CHART,
                        VisualRegionType.METRIC_CARD,
                        VisualRegionType.INFOGRAPHIC,
                    }
                    else VisualProcessingStatus.IMAGE_EXTRACTED
                ),
                requires_vision=region_type in {
                    VisualRegionType.CHART,
                    VisualRegionType.METRIC_CARD,
                    VisualRegionType.INFOGRAPHIC,
                },
                metadata={
                    **dict(candidate.metadata),
                    "report_uid": report_id,
                    "source_element_id": candidate.source_element_id,
                    "layout_visual_region": True,
                },
            )
            assets.append(asset)

            if region_type != VisualRegionType.CHART:
                continue

            # Kota daha önce dolduysa yeni Cloudflare çağrısı yapma; ancak
            # kalan layout görsellerini toplamaya ve kaydetmeye devam et.
            if quota_exceeded:
                skipped += 1
                asset.processing_status = (
                    VisualProcessingStatus.VISION_REQUIRED
                )
                asset.requires_vision = True
                asset.add_error(
                    "Groq vision rate limitine ulaşıldı; "
                    "görsel daha sonra işlenmek üzere bekletildi."
                )
                continue

            if candidate.source_element_id not in allowed_chart_ids:
                skipped += 1
                continue
            extract_chart = getattr(
                self.vlm_table_fallback,
                "extract_chart",
                None,
            ) if self.vlm_table_fallback is not None else None
            if not callable(extract_chart):
                skipped += 1
                continue

            pseudo_visual = SimpleNamespace(
                table_id=candidate.source_element_id,
                page_index=max(page_number - 1, 0),
                page_number=page_number,
                table_index=0,
                table_image_path=cropped_path,
                image_path=cropped_path,
                visual_region_type="chart",
                metadata={
                    **dict(candidate.metadata),
                    "caption_text": candidate.caption,
                    "text_preview": candidate.nearby_text,
                },
            )

            try:
                chart = extract_chart(
                    visual_table=pseudo_visual,
                    contextual_text=candidate.nearby_text,
                )
            except CloudflareQuotaExceededError as exc:
                quota_exceeded = True
                skipped += 1
                warnings.append(
                    "Groq vision rate limitine ulaşıldığı için "
                    "bağımsız grafik çıkarımı durduruldu. Yerel ingestion "
                    "sonuçları korunarak işlem devam ediyor."
                )
                asset.processing_status = (
                    VisualProcessingStatus.VISION_REQUIRED
                )
                asset.requires_vision = True
                asset.add_error(str(exc))
                continue
            except Exception as exc:
                warnings.append(
                    f"Bağımsız grafik VLM çıkarımı başarısız "
                    f"(sayfa {page_number}): {exc}"
                )
                skipped += 1
                continue

            if not bool(getattr(chart, "is_valid", False)):
                skipped += 1

                chart_metadata = getattr(chart, "metadata", {}) or {}
                validation_reason = (
                    chart_metadata.get("validation_reason")
                    if isinstance(chart_metadata, dict)
                    else None
                )
                point_count = len(
                    getattr(chart, "points", ()) or ()
                )
                confidence = getattr(
                    chart,
                    "overall_confidence",
                    None,
                )
                chart_warnings = getattr(
                    chart,
                    "warnings",
                    (),
                ) or ()

                warnings.append(
                    "Bağımsız grafik sonucu geçersiz "
                    f"(sayfa {page_number}): "
                    f"reason={validation_reason or 'unknown'}, "
                    f"points={point_count}, "
                    f"confidence={confidence}, "
                    f"warnings={list(chart_warnings)}"
                )

                asset.processing_status = (
                    VisualProcessingStatus.VISION_REQUIRED
                )
                asset.requires_vision = True
                asset.add_error(
                    "Groq grafik sonucu doğrulamayı geçemedi: "
                    f"{validation_reason or 'unknown'}"
                )
                continue

            processed += 1
            asset.processing_status = VisualProcessingStatus.VISION_COMPLETED
            asset.requires_vision = False
            if VisualExtractionMethod.VISION_CHART not in asset.extraction_methods:
                asset.extraction_methods.append(
                    VisualExtractionMethod.VISION_CHART
                )

            chunk = self._build_layout_chart_chunk(
                chart=chart,
                asset=asset,
                report_id=report_id,
                company_name=company_name,
                report_year=report_year,
                source_element_id=candidate.source_element_id,
                position=len(chunks),
            )
            if chunk is not None:
                chunks.append(chunk)

        metrics = {
            "layout_visual_candidate_count": len(assets),
            "layout_chart_candidate_count": sum(
                asset.region_type == VisualRegionType.CHART
                for asset in assets
            ),
            "layout_chart_processed_count": processed,
            "layout_chart_skipped_count": skipped,
            "layout_chart_quota_exceeded": quota_exceeded,
        }

        print(
            "Bağımsız layout görsel özeti: "
            f"visuals={metrics['layout_visual_candidate_count']}, "
            f"charts={metrics['layout_chart_candidate_count']}, "
            f"processed={processed}, skipped={skipped}, "
            f"quota_exceeded={quota_exceeded}",
            flush=True,
        )
        return assets, chunks, metrics

    def _process_page_chart_candidates(
        self,
        *,
        layouts: Sequence[Any],
        existing_assets: Sequence[VisualAsset],
        report_id: str,
        company_name: str,
        report_year: int,
        warnings: list[str],
    ) -> tuple[
        list[VisualAsset],
        list[_VisualOCRChunk],
        dict[str, int | bool],
    ]:
        """
        Render edilmiş tam sayfalarda vektörel/bileşik grafik bölgelerini arar
        ve güçlü adayları Groq Vision ile yapılandırılmış veriye dönüştürür.
        """

        try:
            from document_ingestion.page_chart_detector import (
                PageChartDetector,
            )
        except Exception as exc:
            warnings.append(
                "Tam sayfa grafik dedektörü yüklenemedi: "
                f"{exc}"
            )
            return [], [], {
                "page_chart_scanned_page_count": 0,
                "page_chart_candidate_count": 0,
                "page_chart_selected_count": 0,
                "page_chart_processed_count": 0,
                "page_chart_skipped_count": 0,
                "page_chart_rate_limited": False,
            }

        detector_enabled = str(
            os.getenv("PAGE_CHART_DETECTOR_ENABLED", "true")
        ).strip().casefold() in {
            "1", "true", "yes", "on",
        }
        if not detector_enabled:
            return [], [], {
                "page_chart_scanned_page_count": 0,
                "page_chart_candidate_count": 0,
                "page_chart_selected_count": 0,
                "page_chart_processed_count": 0,
                "page_chart_skipped_count": 0,
                "page_chart_rate_limited": False,
            }

        try:
            minimum_confidence = float(
                os.getenv("PAGE_CHART_MIN_CONFIDENCE", "0.70")
            )
        except ValueError:
            minimum_confidence = 0.70

        try:
            max_candidates_per_page = max(
                int(
                    os.getenv(
                        "PAGE_CHART_MAX_CANDIDATES_PER_PAGE",
                        "2",
                    )
                ),
                1,
            )
        except ValueError:
            max_candidates_per_page = 2

        try:
            max_groq_calls = max(
                int(
                    os.getenv(
                        "PAGE_CHART_MAX_GROQ_CALLS_PER_RUN",
                        "1",
                    )
                ),
                0,
            )
        except ValueError:
            max_groq_calls = 1

        try:
            page_scan_limit = max(
                int(
                    os.getenv(
                        "PAGE_CHART_MAX_PAGES_PER_RUN",
                        "225",
                    )
                ),
                0,
            )
        except ValueError:
            page_scan_limit = 225

        try:
            target_success_count = max(
                int(os.getenv("PAGE_CHART_TARGET_SUCCESS_COUNT", "1")),
                1,
            )
        except ValueError:
            target_success_count = 1

        try:
            groq_delay_seconds = max(
                float(os.getenv("PAGE_CHART_GROQ_DELAY_SECONDS", "4")),
                0.0,
            )
        except ValueError:
            groq_delay_seconds = 4.0

        retry_on_rate_limit = str(
            os.getenv("PAGE_CHART_RETRY_ON_RATE_LIMIT", "false")
        ).strip().casefold() in {"1", "true", "yes", "on"}

        try:
            rate_limit_fallback_wait = max(
                float(
                    os.getenv(
                        "PAGE_CHART_RATE_LIMIT_FALLBACK_WAIT_SECONDS",
                        "65",
                    )
                ),
                1.0,
            )
        except ValueError:
            rate_limit_fallback_wait = 65.0

        detector = PageChartDetector(
            minimum_confidence=minimum_confidence,
            maximum_candidates_per_page=max_candidates_per_page,
            save_crops=True,
        )

        output_directory = (
            self.table_image_directory
            / report_id
            / "page_chart_candidates"
            if self.table_image_directory is not None
            else None
        )

        layouts_to_scan = list(layouts)
        if page_scan_limit > 0:
            layouts_to_scan = layouts_to_scan[:page_scan_limit]

        detection_results = detector.detect_pages(
            layouts_to_scan,
            output_directory=output_directory,
        )

        all_candidates: list[Any] = []
        for result in detection_results:
            warnings.extend(
                str(item)
                for item in getattr(result, "warnings", ()) or ()
                if str(item).strip()
            )
            all_candidates.extend(
                list(getattr(result, "candidates", ()) or ())
            )

        # Daha önce tablo/layout görseli olarak kaydedilen bölgelerle çakışan
        # adayları ele. PageChartBoundingBox piksel tabanlı olduğu için burada
        # sayfa bazında yalnız tam sayfa veya aşırı büyük varlıkları dışlarız.
        existing_pages = {
            int(asset.page_number)
            for asset in existing_assets
        }

        geometry_signals = {
            "axis_like_geometry",
            "repeated_bar_geometry",
        }
        semantic_signals = {
            "multi_year_numeric_sequence",
            "explicit_chart_language",
            "axis_or_series_language",
        }

        def candidate_signal_profile(
            item: Any,
        ) -> tuple[set[str], set[str], set[str]]:
            signals = set(
                getattr(item, "chart_signals", ()) or ()
            )
            geometry_hits = signals.intersection(
                geometry_signals
            )
            semantic_hits = signals.intersection(
                semantic_signals
            )
            return signals, geometry_hits, semantic_hits

        def is_high_precision_chart_candidate(
            item: Any,
        ) -> bool:
            signals, geometry_hits, semantic_hits = (
                candidate_signal_profile(item)
            )

            confidence = float(
                getattr(item, "confidence", 0.0) or 0.0
            )
            area_ratio = float(
                getattr(item, "area_ratio", 0.0) or 0.0
            )

            # Tek başına axis_like_geometry çok sayıda tabloyu ve metin
            # bloğunu yanlış pozitif yapar. En az bir semantik grafik sinyali
            # veya güçlü tekrar eden bar geometrisi gerekir.
            repeated_bar = (
                "repeated_bar_geometry" in geometry_hits
            )
            axis_plus_semantic = (
                "axis_like_geometry" in geometry_hits
                and bool(semantic_hits)
            )
            multi_semantic = len(semantic_hits) >= 2

            has_required_evidence = (
                repeated_bar
                or axis_plus_semantic
                or multi_semantic
            )

            # Çok küçük ve tam sayfaya yakın bölgeler de yanlış pozitif üretir.
            geometry_ok = 0.05 <= area_ratio <= 0.65

            return (
                has_required_evidence
                and geometry_ok
                and confidence >= 0.76
            )

        def candidate_rank_key(
            item: Any,
        ) -> tuple[int, int, float, float]:
            _, geometry_hits, semantic_hits = (
                candidate_signal_profile(item)
            )
            return (
                len(semantic_hits),
                len(geometry_hits),
                float(getattr(item, "confidence", 0.0) or 0.0),
                float(getattr(item, "area_ratio", 0.0) or 0.0),
            )

        strong_candidates = [
            item
            for item in all_candidates
            if is_high_precision_chart_candidate(item)
        ]

        state_directory = (
            self.table_image_directory
            / report_id
            / "page_chart_state"
            if self.table_image_directory is not None
            else Path("data")
            / "table_images"
            / report_id
            / "page_chart_state"
        )
        state_directory.mkdir(parents=True, exist_ok=True)
        state_path = state_directory / "candidate_state.json"

        candidate_state: dict[str, dict[str, Any]] = {}
        if state_path.is_file():
            try:
                loaded_state = json.loads(
                    state_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded_state, dict):
                    candidate_state = {
                        str(key): dict(value)
                        for key, value in loaded_state.items()
                        if isinstance(value, Mapping)
                    }
            except Exception as exc:
                warnings.append(
                    "Grafik aday durum dosyası okunamadı: "
                    f"{exc}"
                )

        previously_completed_ids = {
            candidate_id
            for candidate_id, state in candidate_state.items()
            if state.get("status") == "completed"
        }
        previously_rejected_ids = {
            candidate_id
            for candidate_id, state in candidate_state.items()
            if state.get("status") in {
                "no_points",
                "invalid",
                "failed",
            }
        }

        pending_strong_candidates = [
            item
            for item in strong_candidates
            if str(getattr(item, "candidate_id", ""))
            not in previously_completed_ids
            and str(getattr(item, "candidate_id", ""))
            not in previously_rejected_ids
        ]

        ranked_candidates = sorted(
            pending_strong_candidates,
            key=candidate_rank_key,
            reverse=True,
        )

        selected_candidates = ranked_candidates[:max_groq_calls]

        if selected_candidates:
            print(
                "Tam sayfa grafik dedektörü: "
                f"scanned_pages={len(layouts_to_scan)}, "
                f"raw_candidates={len(all_candidates)}, "
                f"strong_candidates={len(strong_candidates)}, "
                f"pending_candidates={len(pending_strong_candidates)}, "
                f"previously_rejected={len(previously_rejected_ids)}, "
                f"selected={len(selected_candidates)}, "
                f"pages={[int(item.page_number) for item in selected_candidates]}, "
                f"signals={[list(getattr(item, 'chart_signals', ()) or ()) for item in selected_candidates]}",
                flush=True,
            )
        else:
            print(
                "Tam sayfa grafik dedektörü: "
                f"scanned_pages={len(layouts_to_scan)}, "
                f"raw_candidates={len(all_candidates)}, "
                f"strong_candidates={len(strong_candidates)}, "
                f"pending_candidates={len(pending_strong_candidates)}, "
                f"previously_rejected={len(previously_rejected_ids)}, "
                "selected=0",
                flush=True,
            )

        report_uuid = uuid5(
            NAMESPACE_URL,
            f"sustainability-report:{report_id}",
        )

        assets: list[VisualAsset] = []
        chunks: list[_VisualOCRChunk] = []
        processed = 0
        skipped = 0
        rate_limited = False
        groq_calls_attempted = 0
        last_groq_call_finished_at: float | None = None

        def save_candidate_state() -> None:
            try:
                state_path.write_text(
                    json.dumps(
                        candidate_state,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                warnings.append(
                    "Grafik aday durum dosyası yazılamadı: "
                    f"{exc}"
                )

        extract_chart = (
            getattr(
                self.vlm_table_fallback,
                "extract_chart",
                None,
            )
            if self.vlm_table_fallback is not None
            else None
        )

        for candidate in selected_candidates:
            page_number = int(candidate.page_number)
            crop_path = getattr(candidate, "crop_path", None)
            if crop_path is None:
                skipped += 1
                continue

            crop_path = Path(crop_path).expanduser().resolve()
            if not crop_path.is_file():
                skipped += 1
                continue

            try:
                from PIL import Image
                with Image.open(crop_path) as crop_image:
                    image_width, image_height = crop_image.size
            except Exception as exc:
                warnings.append(
                    "Tam sayfa grafik kırpımı açılamadı "
                    f"(sayfa {page_number}): {exc}"
                )
                skipped += 1
                continue

            candidate_id = str(
                getattr(candidate, "candidate_id", "")
                or (
                    f"page-{page_number}-chart-"
                    f"{len(assets) + 1}"
                )
            )

            image_hash = hashlib.sha256(
                crop_path.read_bytes()
            ).hexdigest()

            asset = VisualAsset(
                id=uuid5(
                    NAMESPACE_URL,
                    f"page-chart:{report_id}:{candidate_id}",
                ),
                report_id=report_uuid,
                page_id=uuid5(
                    NAMESPACE_URL,
                    (
                        f"sustainability-report:{report_id}:"
                        f"page:{page_number}"
                    ),
                ),
                page_number=page_number,
                region_type=VisualRegionType.CHART,
                bounding_box=BoundingBox(
                    x1=float(candidate.bounding_box.x0),
                    y1=float(candidate.bounding_box.y0),
                    x2=float(candidate.bounding_box.x1),
                    y2=float(candidate.bounding_box.y1),
                ),
                image_path=crop_path,
                source_page_image_path=Path(
                    candidate.source_image_path
                ),
                image_width=int(image_width),
                image_height=int(image_height),
                image_hash=image_hash,
                caption=getattr(candidate, "title_hint", None),
                nearby_text=getattr(candidate, "nearby_text", None),
                detection_confidence=float(
                    getattr(candidate, "confidence", 0.0)
                    or 0.0
                ),
                processing_status=(
                    VisualProcessingStatus.VISION_REQUIRED
                ),
                requires_vision=True,
                metadata={
                    **dict(getattr(candidate, "metadata", {}) or {}),
                    "report_uid": report_id,
                    "page_chart_candidate": True,
                    "page_chart_candidate_id": candidate_id,
                    "page_chart_signals": tuple(
                        getattr(candidate, "chart_signals", ()) or ()
                    ),
                    "detection_method": getattr(
                        candidate,
                        "detection_method",
                        "page_chart_detector",
                    ),
                    "page_already_has_visual_assets": (
                        page_number in existing_pages
                    ),
                },
            )
            assets.append(asset)

            if rate_limited:
                skipped += 1
                asset.add_error(
                    "Groq vision rate limitine ulaşıldı; "
                    "grafik daha sonra işlenecek."
                )
                continue

            if not callable(extract_chart):
                skipped += 1
                asset.add_error(
                    "Groq extract_chart metodu kullanılamıyor."
                )
                continue

            if processed >= target_success_count:
                skipped += 1
                asset.add_error(
                    "Bu çalıştırma için hedeflenen başarılı grafik "
                    "sayısına ulaşıldı; aday daha sonra işlenecek."
                )
                continue

            if last_groq_call_finished_at is not None and groq_delay_seconds > 0:
                elapsed_since_last_call = time.monotonic() - last_groq_call_finished_at
                remaining_delay = groq_delay_seconds - elapsed_since_last_call
                if remaining_delay > 0:
                    time.sleep(remaining_delay)

            pseudo_visual = SimpleNamespace(
                table_id=candidate_id,
                page_index=max(page_number - 1, 0),
                page_number=page_number,
                table_index=0,
                table_image_path=crop_path,
                image_path=crop_path,
                visual_region_type="chart",
                metadata={
                    "caption_text": getattr(
                        candidate,
                        "title_hint",
                        None,
                    ),
                    "text_preview": getattr(
                        candidate,
                        "nearby_text",
                        None,
                    ),
                    "page_chart_candidate": True,
                    "page_chart_signals": tuple(
                        getattr(candidate, "chart_signals", ()) or ()
                    ),
                },
            )

            try:
                groq_calls_attempted += 1
                chart = extract_chart(
                    visual_table=pseudo_visual,
                    contextual_text=getattr(
                        candidate,
                        "nearby_text",
                        None,
                    ),
                )
                last_groq_call_finished_at = time.monotonic()
            except CloudflareQuotaExceededError as exc:
                last_groq_call_finished_at = time.monotonic()

                wait_seconds = getattr(
                    exc,
                    "retry_after_seconds",
                    None,
                )
                if wait_seconds is None:
                    wait_seconds = rate_limit_fallback_wait
                wait_seconds = max(float(wait_seconds), 1.0)

                if retry_on_rate_limit:
                    warnings.append(
                        "Groq rate limitine ulaşıldı; aynı grafik "
                        f"{wait_seconds:.1f} saniye sonra yeniden denenecek."
                    )
                    time.sleep(wait_seconds)
                    try:
                        groq_calls_attempted += 1
                        chart = extract_chart(
                            visual_table=pseudo_visual,
                            contextual_text=getattr(
                                candidate,
                                "nearby_text",
                                None,
                            ),
                        )
                        last_groq_call_finished_at = time.monotonic()
                    except CloudflareQuotaExceededError as retry_exc:
                        last_groq_call_finished_at = time.monotonic()
                        rate_limited = True
                        skipped += 1
                        asset.add_error(str(retry_exc))
                        warnings.append(
                            "Groq rate limit sonrası yeniden deneme de "
                            "başarısız oldu; kalan grafik çağrıları durduruldu."
                        )
                        continue
                else:
                    rate_limited = True
                    skipped += 1
                    asset.add_error(str(exc))
                    warnings.append(
                        "Groq vision rate limitine ulaşıldığı için "
                        "tam sayfa grafik çıkarımı durduruldu."
                    )
                    continue
            except Exception as exc:
                last_groq_call_finished_at = time.monotonic()
                skipped += 1
                asset.add_error(str(exc))
                candidate_state[candidate_id] = {
                    "status": "failed",
                    "page_number": page_number,
                    "reason": str(exc),
                }
                save_candidate_state()
                warnings.append(
                    "Tam sayfa grafik VLM çıkarımı başarısız "
                    f"(sayfa {page_number}): {exc}"
                )
                continue

            if not bool(getattr(chart, "is_valid", False)):
                skipped += 1
                chart_metadata = (
                    getattr(chart, "metadata", {}) or {}
                )
                validation_reason = (
                    chart_metadata.get("validation_reason")
                    if isinstance(chart_metadata, dict)
                    else None
                )
                point_count = len(
                    getattr(chart, "points", ()) or ()
                )
                confidence = getattr(
                    chart,
                    "overall_confidence",
                    None,
                )
                asset.add_error(
                    "Groq grafik sonucu doğrulamayı geçemedi: "
                    f"{validation_reason or 'unknown'}"
                )
                candidate_state[candidate_id] = {
                    "status": (
                        "no_points"
                        if validation_reason == "no_points"
                        else "invalid"
                    ),
                    "page_number": page_number,
                    "reason": validation_reason or "unknown",
                    "point_count": point_count,
                    "confidence": confidence,
                }
                save_candidate_state()
                warnings.append(
                    "Tam sayfa grafik sonucu geçersiz "
                    f"(sayfa {page_number}): "
                    f"reason={validation_reason or 'unknown'}, "
                    f"points={point_count}, "
                    f"confidence={confidence}"
                )
                continue

            asset.processing_status = (
                VisualProcessingStatus.VISION_COMPLETED
            )
            asset.requires_vision = False
            if (
                VisualExtractionMethod.VISION_CHART
                not in asset.extraction_methods
            ):
                asset.extraction_methods.append(
                    VisualExtractionMethod.VISION_CHART
                )

            chunk = self._build_layout_chart_chunk(
                chart=chart,
                asset=asset,
                report_id=report_id,
                company_name=company_name,
                report_year=report_year,
                source_element_id=candidate_id,
                position=len(chunks),
            )
            if chunk is None:
                skipped += 1
                asset.processing_status = (
                    VisualProcessingStatus.VISION_REQUIRED
                )
                asset.requires_vision = True
                asset.add_error(
                    "Geçerli Groq sonucu retrieval chunk'ına "
                    "dönüştürülemedi."
                )
                continue

            processed += 1
            chunks.append(chunk)
            candidate_state[candidate_id] = {
                "status": "completed",
                "page_number": page_number,
                "chunk_id": chunk.chunk_id,
                "confidence": float(
                    getattr(chart, "overall_confidence", 0.0)
                    or 0.0
                ),
            }
            save_candidate_state()

        metrics: dict[str, int | bool] = {
            "page_chart_scanned_page_count": len(layouts_to_scan),
            "page_chart_candidate_count": len(all_candidates),
            "page_chart_strong_candidate_count": len(strong_candidates),
            "page_chart_pending_candidate_count": len(
                pending_strong_candidates
            ),
            "page_chart_previously_rejected_count": len(
                previously_rejected_ids
            ),
            "page_chart_previously_completed_count": len(
                previously_completed_ids
            ),
            "page_chart_rejected_candidate_count": (
                len(all_candidates) - len(strong_candidates)
            ),
            "page_chart_selected_count": len(selected_candidates),
            "page_chart_processed_count": processed,
            "page_chart_skipped_count": skipped,
            "page_chart_groq_calls_attempted": groq_calls_attempted,
            "page_chart_target_success_count": target_success_count,
            "page_chart_groq_delay_seconds": groq_delay_seconds,
            "page_chart_retry_on_rate_limit": retry_on_rate_limit,
            "page_chart_rate_limit_fallback_wait_seconds": (
                rate_limit_fallback_wait
            ),
            "page_chart_rate_limited": rate_limited,
        }

        print(
            "Tam sayfa grafik işleme özeti: "
            f"candidates={metrics['page_chart_candidate_count']}, "
            f"selected={metrics['page_chart_selected_count']}, "
            f"processed={processed}, "
            f"skipped={skipped}, "
            f"groq_calls={groq_calls_attempted}, "
            f"target_success={target_success_count}, "
            f"rate_limited={rate_limited}",
            flush=True,
        )

        return assets, chunks, metrics

    def _crop_layout_visual(
        self,
        *,
        page_image_path: Path,
        page_width_points: float,
        page_height_points: float,
        bounding_box: Any,
        report_id: str,
        page_number: int,
        source_element_id: str,
    ) -> tuple[Path, int, int, str]:
        from PIL import Image

        page_image_path = page_image_path.expanduser().resolve()
        if not page_image_path.is_file():
            raise FileNotFoundError(
                f"Render sayfa görüntüsü bulunamadı: {page_image_path}"
            )

        output_root = (
            self.table_image_directory / report_id / "layout_visuals"
            if self.table_image_directory is not None
            else page_image_path.parent / "layout_visuals"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        target_path = output_root / (
            f"page_{page_number:04d}_"
            f"{self._slugify(source_element_id)}.png"
        )

        with Image.open(page_image_path) as image:
            image = image.convert("RGB")
            width_pixels, height_pixels = image.size
            scale_x = width_pixels / max(page_width_points, 1.0)
            scale_y = height_pixels / max(page_height_points, 1.0)

            x0 = max(int(round(float(bounding_box.x0) * scale_x)), 0)
            y0 = max(int(round(float(bounding_box.y0) * scale_y)), 0)
            x1 = min(int(round(float(bounding_box.x1) * scale_x)), width_pixels)
            y1 = min(int(round(float(bounding_box.y1) * scale_y)), height_pixels)

            if x1 <= x0 or y1 <= y0:
                raise ValueError("Görsel kırpma kutusu geçersiz.")

            image.crop((x0, y0, x1, y1)).save(
                target_path,
                format="PNG",
                optimize=True,
            )

        return (
            target_path,
            x1 - x0,
            y1 - y0,
            hashlib.sha256(target_path.read_bytes()).hexdigest(),
        )

    @staticmethod
    def _overlaps_existing_visual_asset(
        bounding_box: Any,
        assets: Sequence[VisualAsset],
        *,
        threshold: float = 0.78,
    ) -> bool:
        for asset in assets:
            other = asset.bounding_box
            x0 = max(float(bounding_box.x0), float(other.x1))
            y0 = max(float(bounding_box.y0), float(other.y1))
            x1 = min(float(bounding_box.x1), float(other.x2))
            y1 = min(float(bounding_box.y1), float(other.y2))
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = (x1 - x0) * (y1 - y0)
            first_area = max(
                (float(bounding_box.x1) - float(bounding_box.x0))
                * (float(bounding_box.y1) - float(bounding_box.y0)),
                1.0,
            )
            second_area = max(
                (float(other.x2) - float(other.x1))
                * (float(other.y2) - float(other.y1)),
                1.0,
            )
            union = first_area + second_area - intersection
            if union > 0 and intersection / union >= threshold:
                return True
        return False

    def _build_layout_chart_chunk(
        self,
        *,
        chart: Any,
        asset: VisualAsset,
        report_id: str,
        company_name: str,
        report_year: int,
        source_element_id: str,
        position: int,
    ) -> _VisualOCRChunk | None:
        matrix = getattr(chart, "matrix", ()) or ()
        rows = [
            " | ".join(str(cell).strip() for cell in row)
            for row in matrix
            if any(str(cell).strip() for cell in row)
        ]
        if not rows:
            return None

        title = str(
            getattr(chart, "title", "")
            or asset.caption
            or "Visual chart"
        ).strip()
        raw_text = "\n".join([title, *rows])
        normalized_text = re.sub(r"\s+", " ", raw_text).strip()
        content_hash = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()
        category, topic = self._infer_visual_category_topic(
            "\n".join(
                value
                for value in (
                    title,
                    asset.nearby_text,
                    raw_text,
                )
                if value
            )
        )

        return _VisualOCRChunk(
            chunk_id=(
                f"{report_id}_layout_chart_"
                f"{self._slugify(source_element_id)}"
            ),
            report_id=report_id,
            chunk_index=1_200_000 + position,
            page_number=asset.page_number,
            start_page=asset.page_number,
            end_page=asset.page_number,
            text=raw_text,
            normalized_text=normalized_text,
            searchable_text="\n".join(
                value
                for value in (
                    title,
                    asset.nearby_text,
                    raw_text,
                )
                if value
            ),
            contextualized_text=(
                f"{company_name} {report_year} sürdürülebilirlik raporu. "
                f"Sayfa {asset.page_number}. Bağımsız layout grafik kaynağı: "
                f"{source_element_id}.\n{raw_text}"
            ),
            section_title=title,
            category=category,
            source_type="layout_visual_chart",
            content_hash=content_hash,
            metadata={
                "report_id": report_id,
                "company_name": company_name,
                "report_year": report_year,
                "page_number": asset.page_number,
                "source_element_id": source_element_id,
                "visual_id": str(asset.id),
                "source_type": "layout_visual_chart",
                "chart_type": getattr(chart, "chart_type", None),
                "chart_title": title,
                "chart_unit": getattr(chart, "unit", None),
                "chart_years": tuple(
                    getattr(chart, "years", ()) or ()
                ),
                "chart_series": tuple(
                    getattr(chart, "series_names", ()) or ()
                ),
                "chart_confidence": float(
                    getattr(chart, "overall_confidence", 0.0) or 0.0
                ),
                "category": category,
                "topic": topic,
                "topics": (topic,) if topic else (),
                "is_visual_evidence": True,
                "is_chart_evidence": True,
                "content_hash": content_hash,
            },
        )
    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------
    def _process_text(
        self,
        *,
        classified_pages: Sequence[Any],
        layouts: Sequence[Any],
        report_id: str,
        company_name: str,
        report_year: int,
        report_url: str | None,
        pdf_path: Path,
        warnings: list[str],
    ) -> tuple[Any, Any, list[Any]]:
        text_pages = self._build_text_pages(classified_pages, layouts)
        cleaned_document = self.text_cleaner.clean_document(text_pages)
        warnings.extend(
            str(item)
            for item in getattr(cleaned_document, "warnings", ()) or ()
        )

        section_result = self.section_detector.detect_sections(
            cleaned_document,
            document_id=report_id,
            document_title=f"{company_name} {report_year} Sustainability Report",
        )
        warnings.extend(
            str(item) for item in getattr(section_result, "warnings", ()) or ()
        )

        chunking_result = self.semantic_chunker.chunk_document(section_result)
        warnings.extend(
            str(item) for item in getattr(chunking_result, "warnings", ()) or ()
        )
        chunks = list(getattr(chunking_result, "chunks", ()) or ())

        classification_result = self.category_classifier.classify_document(
            chunking_result
        )
        classifications = {
            str(item.chunk_id): item
            for item in getattr(classification_result, "classifications", ()) or ()
        }

        enriched_chunks: list[Any] = []
        for chunk in chunks:
            classification = classifications.get(str(chunk.chunk_id))
            category = self._enum_value(
                getattr(classification, "primary_category", None)
            )
            standards = tuple(
                getattr(classification, "detected_standard_families", ()) or ()
            )
            topics = tuple(
                getattr(classification, "detected_topics", ()) or ()
            )
            context = MetadataContext(
                report_id=report_id,
                company_name=company_name,
                report_year=report_year,
                page_number=int(getattr(chunk, "start_page", 0) or 0) or None,
                page_index=(
                    int(getattr(chunk, "start_page", 1) or 1) - 1
                ),
                section_title=getattr(chunk, "section_title", None),
                source_path=pdf_path,
                source_url=report_url,
                standards=standards,
            )
            enriched = self.metadata_enricher.enrich(
                chunk,
                context=context,
                extra_metadata={
                    "category": category,
                    "category_confidence": getattr(
                        classification,
                        "confidence",
                        None,
                    ),
                    "topics": topics,
                    "standards": standards,
                    "page_end": getattr(chunk, "end_page", None),
                },
            )
            enriched_chunks.append(enriched)

        return cleaned_document, section_result, enriched_chunks

    def _build_text_pages(
        self,
        classified_pages: Sequence[Any],
        layouts: Sequence[Any],
    ) -> list[Any]:
        pages: list[Any] = []
        for index, classified in enumerate(classified_pages):
            layout = layouts[index] if index < len(layouts) else None
            elements = tuple(getattr(layout, "elements", ()) or ())
            blocks: list[Any] = []
            for block_index, element in enumerate(elements):
                text = str(getattr(element, "text", "") or "").strip()
                if not text:
                    continue
                element_type = self._enum_value(
                    getattr(element, "element_type", "paragraph")
                ) or "paragraph"
                blocks.append(
                    SimpleNamespace(
                        block_index=block_index,
                        text=text,
                        normalized_text=text,
                        block_type=element_type,
                        is_heading_candidate=element_type in {"title", "heading"},
                        is_header_candidate=element_type == "header",
                        is_footer_candidate=element_type in {"footer", "page_number"},
                        bounding_box=getattr(element, "bounding_box", None),
                    )
                )

            page_number = int(
                getattr(classified, "page_number", index + 1) or index + 1
            )
            page_text = str(getattr(classified, "text", "") or "")
            pages.append(
                SimpleNamespace(
                    page_index=page_number - 1,
                    page_number=page_number,
                    text=page_text,
                    normalized_text=page_text,
                    blocks=tuple(blocks),
                    extraction_method="layout_and_native_text",
                    confidence=float(getattr(classified, "confidence", 0.0) or 0.0),
                    metadata={
                        "source_pdf": str(getattr(classified, "source_pdf", "") or ""),
                        "image_path": str(getattr(classified, "image_path", "") or ""),
                    },
                )
            )
        return pages

    # ------------------------------------------------------------------
    # Persistence / indexing
    # ------------------------------------------------------------------
    def _upsert_report(
        self,
        *,
        report_uid: str,
        company_name: str,
        report_year: int,
        report_url: str | None,
        pdf_path: Path,
        validation: Any,
        page_count: int,
        status: str,
        metadata: Mapping[str, Any],
    ) -> int | None:
        if self.database is None:
            return None
        file_size = pdf_path.stat().st_size if pdf_path.exists() else None
        sha256 = self._extract_value(validation, ("sha256", "checksum"))
        validation_status = self._extract_value(
            validation,
            ("status", "validation_status"),
        )
        validation_message = self._extract_value(
            validation,
            ("message", "validation_message"),
        )
        return int(
            self.database.upsert_report(
                report_uid=report_uid,
                company_name=company_name,
                company_slug=self._slugify(company_name),
                report_year=report_year,
                report_type="sustainability",
                source_url=report_url,
                final_url=report_url,
                local_pdf_path=pdf_path,
                sha256=str(sha256) if sha256 else None,
                file_size_bytes=file_size,
                page_count=page_count,
                status=status,
                validation_status=(
                    str(self._enum_value(validation_status))
                    if validation_status is not None
                    else "valid"
                ),
                validation_message=(
                    str(validation_message) if validation_message else None
                ),
                metadata=dict(metadata),
            )
        )

    def _prepare_overwrite(
        self,
        *,
        database_report_id: int | None,
        report_uid: str,
        warnings: list[str],
    ) -> None:
        """Overwrite öncesinde eski kalıcı ve bellek içi kayıtları temizler."""

        if self.database is None or database_report_id is None:
            return

        clear_method = getattr(
            self.database,
            "clear_report_content",
            None,
        )
        if not callable(clear_method):
            raise IngestionPipelineError(
                "Database.clear_report_content bulunamadı; güvenli overwrite "
                "gerçekleştirilemez."
            )

        deleted = clear_method(
            database_report_id,
            confirm=True,
            preserve_queries=False,
        )

        # Bu çalışma sürecinde repository'ler çoğunlukla boştur; yine de eski
        # nesnelerin yeni ingestion'a karışmasını kesin olarak engelle.
        clear_chunks = getattr(self.chunk_repository, "clear", None)
        if callable(clear_chunks):
            clear_chunks()

        table_cache = getattr(self.table_repository, "_tables_by_id", None)
        if isinstance(table_cache, dict):
            table_cache.clear()
        table_report_cache = getattr(
            self.table_repository,
            "_report_uid_by_table_id",
            None,
        )
        if isinstance(table_report_cache, dict):
            table_report_cache.clear()

        if self.visual_asset_repository is not None:
            for attribute_name in (
                "_assets_by_id",
                "_asset_id_by_stable_key",
                "_metrics_by_id",
                "_metric_ids_by_visual_id",
            ):
                cache = getattr(
                    self.visual_asset_repository,
                    attribute_name,
                    None,
                )
                if isinstance(cache, dict):
                    cache.clear()

        print(
            "Overwrite temizliği: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(deleted.items())
            ),
            flush=True,
        )

    @classmethod
    def _deduplicate_chunks(
        cls,
        chunks: Sequence[Any],
    ) -> tuple[list[Any], int]:
        """
        Aynı chunk_uid ile üretilen kayıtları SQLite ve vector store öncesinde
        tekilleştirir. Son üretilen, bilgi bakımından daha yeni kayıt korunur.
        """

        by_uid: dict[str, Any] = {}
        anonymous: list[Any] = []
        duplicate_count = 0

        for chunk in chunks:
            metadata = cls._extract_metadata(chunk)
            chunk_uid = str(
                cls._extract_value(
                    chunk,
                    ("chunk_id", "identifier", "id"),
                )
                or metadata.get("chunk_id")
                or metadata.get("original_chunk_uid")
                or ""
            ).strip()

            if not chunk_uid:
                anonymous.append(chunk)
                continue

            if chunk_uid in by_uid:
                duplicate_count += 1
            by_uid[chunk_uid] = chunk

        deduplicated = [
            *by_uid.values(),
            *anonymous,
        ]
        deduplicated.sort(
            key=lambda item: (
                int(
                    cls._extract_value(
                        item,
                        ("page_number", "start_page"),
                    )
                    or cls._extract_metadata(item).get("page_number")
                    or 0
                ),
                int(
                    cls._extract_value(
                        item,
                        ("chunk_index", "chunk_position"),
                    )
                    or cls._extract_metadata(item).get("chunk_index")
                    or 0
                ),
                str(
                    cls._extract_value(
                        item,
                        ("chunk_id", "identifier", "id"),
                    )
                    or ""
                ),
            )
        )
        return deduplicated, duplicate_count

    def _store_pages(
        self,
        *,
        report_uid: str,
        database_report_id: int | None,
        rendered_pages: Sequence[Any],
        classified_pages: Sequence[Any],
        layouts: Sequence[Any],
    ) -> dict[int, int]:
        if self.database is None or database_report_id is None:
            return {}
        page_ids: dict[int, int] = {}
        for index, classified in enumerate(classified_pages):
            rendered = rendered_pages[index] if index < len(rendered_pages) else None
            layout = layouts[index] if index < len(layouts) else None
            page_number = int(getattr(classified, "page_number", index + 1))
            primary_type = self._enum_value(
                getattr(classified, "primary_type", None)
            )
            classification_dict = self._to_dict(classified)
            layout_dict = self._to_dict(layout)
            page_id = self.database.upsert_page(
                page_uid=f"{report_uid}_page_{page_number}",
                report_id=database_report_id,
                page_index=int(getattr(classified, "page_index", page_number - 1)),
                page_number=page_number,
                page_type=str(primary_type) if primary_type else None,
                classification_confidence=float(
                    getattr(classified, "confidence", 0.0) or 0.0
                ),
                width_points=float(getattr(classified, "width_points", 0.0) or 0.0),
                height_points=float(getattr(classified, "height_points", 0.0) or 0.0),
                rotation=int(getattr(classified, "rotation", 0) or 0),
                has_native_text=bool(getattr(classified, "has_native_text", False)),
                needs_ocr=bool(getattr(classified, "needs_ocr", False)),
                is_scanned=bool(getattr(classified, "is_scanned", False)),
                text_content=str(getattr(classified, "text", "") or ""),
                rendered_image_path=getattr(rendered, "image_path", None),
                classification=classification_dict,
                layout=layout_dict,
                metadata={"report_uid": report_uid},
            )
            page_ids[page_number] = int(page_id)
        return page_ids

    def _store_tables(
        self,
        *,
        database_report_id: int | None,
        report_uid: str,
        company_name: str,
        report_year: int,
        page_ids: Mapping[int, int],
        result: _TableProcessingResult,
        warnings: list[str],
    ) -> None:
        tables = list(result.tables)
        if not tables:
            return

        cache_method = getattr(self.table_repository, "cache_tables", None)
        if callable(cache_method):
            cache_method(report_uid, tables)

        if database_report_id is None:
            return
        try:
            summary = self.table_repository.store_tables(
                report_id=database_report_id,
                company_name=company_name,
                report_year=report_year,
                tables=tables,
                validations_by_table_id=result.validations_by_table_id,
                page_ids_by_number=page_ids,
                continue_on_error=True,
            )
            for item in getattr(summary, "errors", ()) or ():
                warnings.append(f"Tablo kaydı: {item}")
        except Exception as exc:
            warnings.append(f"Tablolar kalıcı depoya kaydedilemedi: {exc}")

    def _store_visual_assets(
        self,
        *,
        database_report_id: int | None,
        company_name: str,
        report_year: int,
        page_ids: Mapping[int, int],
        assets: Sequence[VisualAsset],
        warnings: list[str],
    ) -> Any | None:
        """Görsel varlıkları cache ve kalıcı repository'ye yazar."""

        if not assets or self.visual_asset_repository is None:
            return None

        cache_method = getattr(
            self.visual_asset_repository,
            "cache_assets",
            None,
        )
        if callable(cache_method):
            cache_method(list(assets))

        if database_report_id is None:
            return None

        try:
            summary = self.visual_asset_repository.store_assets(
                report_database_id=database_report_id,
                company_name=company_name,
                report_year=report_year,
                assets=list(assets),
                page_database_ids=page_ids,
                continue_on_error=True,
            )
        except Exception as exc:
            warnings.append(
                "Görsel varlıklar kalıcı depoya kaydedilemedi: "
                f"{exc}"
            )
            return None

        for item in getattr(summary, "errors", ()) or ():
            warnings.append(f"Görsel varlık kaydı: {item}")

        return summary

    def _store_chunks(
        self,
        chunks: Sequence[Any],
        *,
        database_report_id: int | None,
        report_uid: str,
        page_ids: Mapping[int, int],
        company_name: str,
        report_year: int,
        warnings: list[str],
    ) -> None:
        if not chunks:
            return

        stored_chunks = self.chunk_repository.add_many(
            chunks,
            overwrite=True,
            infer_neighbors=True,
        )

        if self.database is not None and database_report_id is not None:
            for fallback_index, chunk in enumerate(stored_chunks):
                metadata = self._extract_metadata(chunk)
                chunk_uid = str(
                    self._extract_value(
                        chunk,
                        ("chunk_id", "id", "identifier"),
                    )
                    or ""
                ).strip()
                text_content = self._record_text(chunk).strip()
                page_number = int(
                    self._extract_value(
                        chunk,
                        ("page_number", "start_page"),
                    )
                    or metadata.get("page_number")
                    or 1
                )
                chunk_index = int(
                    self._extract_value(
                        chunk,
                        ("chunk_index", "chunk_position"),
                    )
                    or metadata.get("chunk_index")
                    or fallback_index
                )

                if not chunk_uid or not text_content:
                    continue

                try:
                    self.database.upsert_chunk(
                        chunk_uid=chunk_uid,
                        report_id=database_report_id,
                        page_id=page_ids.get(page_number),
                        page_number=page_number,
                        chunk_index=chunk_index,
                        section_title=(
                            self._extract_value(
                                chunk,
                                ("section_title",),
                            )
                            or metadata.get("section_title")
                        ),
                        subsection_title=(
                            self._extract_value(
                                chunk,
                                ("subsection_title",),
                            )
                            or metadata.get("subsection_title")
                        ),
                        text_content=text_content,
                        token_count=metadata.get("token_count"),
                        category=(
                            self._extract_value(
                                chunk,
                                ("category",),
                            )
                            or metadata.get("category")
                        ),
                        language=(
                            self._extract_value(
                                chunk,
                                ("language",),
                            )
                            or metadata.get("language")
                        ),
                        previous_chunk_uid=metadata.get(
                            "previous_chunk_id"
                        ),
                        next_chunk_uid=metadata.get(
                            "next_chunk_id"
                        ),
                        content_hash=(
                            self._extract_value(
                                chunk,
                                ("content_hash",),
                            )
                            or metadata.get("content_hash")
                        ),
                        metadata={
                            **metadata,
                            "report_id": report_uid,
                            "company_name": company_name,
                            "report_year": report_year,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        },
                    )
                except Exception as exc:
                    warnings.append(
                        "Chunk SQLite kaydı başarısız "
                        f"({chunk_uid}): {exc}"
                    )

        if self.file_store is not None:
            try:
                report_paths = self.file_store.get_report_paths(
                    company_name=company_name,
                    report_year=report_year,
                )
                repository_path = (
                    report_paths.report_root
                    / "chunks"
                    / "chunk_repository.json"
                )
                self.chunk_repository.save_json(
                    repository_path
                )
            except Exception as exc:
                warnings.append(
                    "Chunk repository JSON kaydı başarısız: "
                    f"{exc}"
                )

    def _index_vectors(
        self,
        chunks: Sequence[Any],
        tables: Sequence[Any],
        warnings: list[str],
    ) -> None:
        if self.vector_store is None:
            return

        document_ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for record in [*chunks, *tables]:
            text = self._record_text(record).strip()
            identifier = self._extract_value(
                record,
                ("chunk_id", "table_id", "document_id", "id"),
            )

            if not text or not identifier:
                continue

            if not self._is_meaningful_vector_text(text):
                continue

            matrix = getattr(record, "matrix", None)

            if matrix is not None and not any(
                str(cell).strip()
                for row in matrix
                for cell in row
            ):
                continue

            metadata = self._extract_metadata(record)
            metadata.setdefault(
                "page_number",
                self._extract_value(
                    record,
                    ("page_number", "start_page"),
                ),
            )

            document_ids.append(str(identifier))
            texts.append(text)
            metadatas.append(metadata)

        if not texts:
            return

        add_texts = getattr(
            self.vector_store,
            "add_texts",
            None,
        )

        if callable(add_texts):
            try:
                add_texts(
                    texts,
                    document_ids=document_ids,
                    metadatas=metadatas,
                    overwrite=True,
                )
                return
            except Exception as exc:
                warnings.append(
                    "Toplu vektör indeksleme başarısız; "
                    f"tekil yazıma geçiliyor: {exc}"
                )

        for identifier, text, metadata in zip(
            document_ids,
            texts,
            metadatas,
            strict=True,
        ):
            try:
                self.vector_store.add_text(
                    identifier,
                    text,
                    metadata=metadata,
                    overwrite=True,
                )
            except Exception as exc:
                message = str(exc).casefold()

                if (
                    "sıfır normlu" in message
                    or "zero norm" in message
                ):
                    continue

                warnings.append(
                    "Vektör indeksleme başarısız "
                    f"({identifier}): {exc}"
                )

    @staticmethod
    def _is_meaningful_vector_text(
        text: str,
    ) -> bool:
        normalized = re.sub(
            r"\s+",
            " ",
            str(text or ""),
        ).strip()

        if not normalized:
            return False

        alphanumeric = re.sub(
            r"[^0-9A-Za-zÀ-ž]+",
            "",
            normalized,
        )

        if len(alphanumeric) < 3:
            return False

        return bool(
            re.search(
                r"[0-9A-Za-zÀ-ž]",
                normalized,
            )
        )

    def _parse_standard_index(
        self,
        *,
        tables: Sequence[Any],
        warnings: list[str],
    ) -> list[Any]:
        if self.standard_index_parser is None:
            return []
        parsed_rows: list[Any] = []
        for table in tables:
            matrix = getattr(table, "matrix", ()) or ()
            if not matrix:
                continue
            try:
                result = self.standard_index_parser.parse(
                    matrix,
                    source_page=getattr(table, "page_number", None),
                )
                parsed_rows.extend(list(getattr(result, "rows", ()) or ()))
                warnings.extend(str(item) for item in getattr(result, "warnings", ()) or ())
            except Exception as exc:
                warnings.append(
                    f"Standart indeks tablosu {getattr(table, 'table_id', '?')} ayrıştırılamadı: {exc}"
                )
        return parsed_rows

    def _store_report_metadata(self, **metadata: Any) -> None:
        if self.file_store is None:
            return
        relative_path = f"{metadata['report_id']}/report_metadata.json"
        save_json = getattr(self.file_store, "save_json", None)
        if callable(save_json):
            save_json(
                metadata,
                relative_path,
                overwrite=True,
                report_id=str(metadata["report_id"]),
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _call_compatible(
        target: Any,
        method_names: Sequence[str],
        *args: Any,
        required: bool = True,
        **kwargs: Any,
    ) -> Any:
        if target is None:
            if required:
                raise IngestionPipelineError("Gerekli pipeline bileşeni None.")
            return None

        mismatches: list[str] = []
        for method_name in method_names:
            method = getattr(target, method_name, None)
            if not callable(method):
                continue
            try:
                signature = inspect.signature(method)
                signature.bind(*args, **kwargs)
            except TypeError as exc:
                mismatches.append(f"{method_name}{signature}: {exc}")
                continue
            except ValueError:
                pass
            # Metodun kendi TypeError hatası burada artık maskelenmez.
            return method(*args, **kwargs)

        if required:
            detail = f" İmza uyuşmazlıkları: {' | '.join(mismatches)}" if mismatches else ""
            raise IngestionPipelineError(
                f"{type(target).__name__} üzerinde uyumlu metod bulunamadı: "
                f"{', '.join(method_names)}.{detail}"
            )
        return None

    @staticmethod
    def _create_report_id(company_name: str, report_year: int) -> str:
        return f"{IngestionPipeline._slugify(company_name)}_{report_year}"

    @staticmethod
    def _slugify(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.strip().casefold())
        ascii_text = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        ascii_text = ascii_text.replace("ı", "i")
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
        return slug or "report"

    @staticmethod
    def _extract_value(source: Any, names: Sequence[str]) -> Any:
        if source is None:
            return None
        for name in names:
            value = source.get(name) if isinstance(source, Mapping) else getattr(source, name, None)
            if value is not None:
                return getattr(value, "value", value)
        return None

    @staticmethod
    def _extract_metadata(source: Any) -> dict[str, Any]:
        metadata = source.get("metadata") if isinstance(source, Mapping) else getattr(source, "metadata", None)
        return dict(metadata) if isinstance(metadata, Mapping) else {}

    @staticmethod
    def _with_metadata(source: Any, metadata: Mapping[str, Any]) -> Any:
        current = IngestionPipeline._extract_metadata(source)
        current.update({key: value for key, value in metadata.items() if value is not None})
        if isinstance(source, dict):
            result = dict(source)
            result["metadata"] = current
            return result
        if is_dataclass(source):
            if "metadata" in {item.name for item in fields(source)}:
                return replace(source, metadata=current)
        source_metadata = getattr(source, "metadata", None)
        if isinstance(source_metadata, dict):
            source_metadata.update(current)
        return source

    @staticmethod
    def _record_text(record: Any) -> str:
        text = IngestionPipeline._extract_value(
            record,
            ("contextualized_text", "searchable_text", "text", "content", "normalized_text"),
        )
        if text:
            return str(text)
        matrix = getattr(record, "matrix", None)
        if matrix:
            return "\n".join(" | ".join(str(cell) for cell in row) for row in matrix)
        return ""

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict"):
            try:
                return dict(value.to_dict(include_text=True))
            except TypeError:
                return dict(value.to_dict())
        if hasattr(value, "model_dump"):
            return dict(value.model_dump(mode="json", exclude_none=True))
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _is_successful_validation(result: Any) -> bool:
        if isinstance(result, bool):
            return result
        if result is None:
            return False
        for name in ("is_valid", "valid", "success", "is_successful"):
            value = IngestionPipeline._extract_value(result, (name,))
            if value is not None:
                return bool(value)
        status = IngestionPipeline._extract_value(result, ("status",))
        if status is not None:
            return str(IngestionPipeline._enum_value(status)).casefold() in {
                "valid", "success", "successful", "completed", "ok",
            }
        return True


__all__ = [
    "IngestionPipeline",
    "IngestionPipelineError",
    "IngestionRequest",
    "IngestionResult",
    "IngestionStatus",
]