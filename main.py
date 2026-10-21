# PROVIDER_ENABLED_MAIN_VERSION = "2026-07-28-1-GROQ-QWEN-VISION-VLM"
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from cloud_sync import print_sync_error, print_sync_result, sync_report_outputs
from cloud_sync.cli_support import (
    print_restore_error,
    print_restore_result,
    restore_report_inputs,
)

from answering.answer_generator import AnswerGenerator
from answering.category_answer_generator import CategoryAnswerGenerator
from answering.citation_builder import CitationBuilder
from answering.gemini_client import GeminiClient, GeminiClientConfig
from answering.llm_client import LLMProvider, TextGenerationClient
from answering.openai_compatible_multimodal_client import (
    OpenAICompatibleMultimodalClient,
)
from answering.visual_evidence_generator import (
    VisualEvidenceGenerator,
    VisualEvidenceGeneratorConfig,
)
from answering.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleClientConfig,
)
from config import Settings, get_settings
from answering.question_analyzer import QuestionAnalyzer
from answering.question_catalog import QuestionCatalog, QuestionGroup
from document_ingestion.layout_detector import LayoutDetector
from document_ingestion.page_classifier import PageClassifier
from document_ingestion.page_renderer import PageRenderer
from document_ingestion.pdf_downloader import PdfDownloader
from document_ingestion.pdf_validator import PdfValidator
from pipelines.category_question_answering_pipeline import (
    CategoryQuestionAnsweringPipeline,
    CategoryQuestionAnsweringResult,
)
from pipelines.ingestion_pipeline import IngestionPipeline, IngestionRequest
from pipelines.visual_evidence_pipeline import (
    VisualEvidencePipeline,
    VisualEvidencePipelineConfig,
)
from pipelines.question_answering_pipeline import (
    QuestionAnsweringPipeline,
    QuestionAnsweringRequest,
    QuestionAnsweringResult,
    QuestionAnsweringStatus,
)
from report_discovery.candidate_ranker import CandidateRanker
from report_discovery.url_validator import UrlValidator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.context_expander import ContextExpander
from retrieval.evidence_validator import EvidenceValidator
from retrieval.reranker import Reranker
from retrieval.result_fusion import ResultFusion
from retrieval.standard_router import StandardRouter
from retrieval.table_retriever import TableRetriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.visual_retriever import VisualRetriever, VisualRetrieverConfig
from standards.standard_index_parser import StandardIndexParser
from storage.database import Database
from storage.excel_exporter import ExcelAnswerRow, ExcelExporter
from storage.file_store import FileStore
from storage.visual_asset_repository import VisualAssetRepository
from storage.vector_store import (
    HashingEmbeddingProvider,
    VectorStore,
    VectorStoreConfig,
)
from table_processing.native_table_extractor import NativeTableExtractor
from table_processing.numeric_validator import NumericValidator
from table_processing.table_detector import TableDetector
from table_processing.table_normalizer import TableNormalizer
from table_processing.table_ocr import TableOCR
from table_processing.table_repository import TableRepository
from table_processing.visual_table_extractor import VisualTableExtractor
from text_processing.category_classifier import CategoryClassifier
from text_processing.chunk_repository import ChunkRepository
from text_processing.metadata_enricher import MetadataEnricher
from text_processing.section_detector import SectionDetector
from text_processing.semantic_chunker import SemanticChunker
from text_processing.text_cleaner import TextCleaner

try:
    from table_processing.vlm_table_fallback import VLMTableFallback
except ImportError as exc:
    VLMTableFallback = None  # type: ignore[assignment,misc]
    _VLM_IMPORT_ERROR: Exception | None = exc
else:
    _VLM_IMPORT_ERROR = None


load_dotenv()


@dataclass(slots=True)
class RuntimeResources:
    """Ingestion ve retrieval pipeline'larının paylaştığı çalışma kaynakları."""

    data_path: Path
    downloads_path: Path
    rendered_pages_path: Path
    vector_path: Path
    table_images_path: Path

    database: Database
    file_store: FileStore
    table_repository: TableRepository
    visual_asset_repository: VisualAssetRepository
    chunk_repository: ChunkRepository
    embedding_provider: HashingEmbeddingProvider
    vector_store: VectorStore


@dataclass(slots=True, frozen=True)
class LLMRuntimeOptions:
    """Komut satırı ve .env değerlerinden çözümlenmiş LLM ayarları."""

    provider: LLMProvider
    model: str
    api_key: str | None = None
    base_url: str | None = None


@dataclass(slots=True, frozen=True)
class VisionRuntimeOptions:
    provider: LLMProvider
    model: str
    api_key: str | None = None
    base_url: str | None = None
    enabled: bool = True


@dataclass(slots=True)
class AnsweringPipelines:
    """Tek soru ve kategori soru-cevap pipeline'larını birlikte taşır."""

    question_answering: QuestionAnsweringPipeline
    category_question_answering: CategoryQuestionAnsweringPipeline
    llm_client: TextGenerationClient
    llm_provider: str
    llm_model: str
    vision_client: Any | None = None
    vision_provider: str | None = None
    vision_model: str | None = None

    def close(self) -> None:
        self.llm_client.close()
        if self.vision_client is not None and self.vision_client is not self.llm_client:
            close = getattr(self.vision_client, "close", None)
            if callable(close):
                close()


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_llm_options(
    args: argparse.Namespace | None = None,
    *,
    settings: Settings | None = None,
    provider: str | LLMProvider | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMRuntimeOptions:
    """CLI değerlerini .env ayarlarının üzerine uygulayarak LLM'i çözümler."""

    resolved_settings = settings or get_settings()

    argument_provider = (
        _clean_optional_text(getattr(args, "llm_provider", None))
        if args is not None
        else None
    )
    argument_model = (
        _clean_optional_text(getattr(args, "llm_model", None))
        if args is not None
        else None
    )
    argument_api_key = (
        _clean_optional_text(getattr(args, "llm_api_key", None))
        if args is not None
        else None
    )
    argument_base_url = (
        _clean_optional_text(getattr(args, "llm_base_url", None))
        if args is not None
        else None
    )

    legacy_gemini_model = (
        _clean_optional_text(getattr(args, "gemini_model", None))
        if args is not None
        else None
    )
    legacy_gemini_api_key = (
        _clean_optional_text(getattr(args, "gemini_api_key", None))
        if args is not None
        else None
    )

    explicit_provider = _clean_optional_text(provider) or argument_provider
    if explicit_provider is None and (legacy_gemini_model or legacy_gemini_api_key):
        explicit_provider = LLMProvider.GEMINI.value

    provider_value = explicit_provider or resolved_settings.llm_provider
    resolved_provider = LLMProvider.normalize(provider_value)

    resolved_model = (
        _clean_optional_text(model)
        or argument_model
        or (legacy_gemini_model if resolved_provider == LLMProvider.GEMINI else None)
    )

    if resolved_model is None:
        if resolved_provider.value == resolved_settings.llm_provider:
            resolved_model = resolved_settings.resolved_llm_model
        elif resolved_provider == LLMProvider.GEMINI:
            resolved_model = resolved_settings.gemini_model
        else:
            provider_defaults = {
                LLMProvider.GROQ: "qwen/qwen3.6-27b",
                LLMProvider.OPENROUTER: "openrouter/free",
                LLMProvider.OLLAMA: "qwen3.5:9b",
            }
            resolved_model = provider_defaults.get(resolved_provider)

    if not resolved_model:
        raise ValueError(
            f"{resolved_provider.value} sağlayıcısı için model adı "
            "tanımlanmalıdır (--llm-model veya LLM_MODEL)."
        )

    resolved_api_key = _clean_optional_text(api_key) or argument_api_key
    if resolved_api_key is None and resolved_provider == LLMProvider.GEMINI:
        resolved_api_key = legacy_gemini_api_key

    if resolved_api_key is None:
        provider_keys = {
            LLMProvider.GEMINI: resolved_settings.gemini_api_key,
            LLMProvider.GROQ: resolved_settings.groq_api_key,
            LLMProvider.OPENROUTER: resolved_settings.openrouter_api_key,
            LLMProvider.OPENAI_COMPATIBLE: resolved_settings.openai_api_key,
            LLMProvider.OLLAMA: None,
        }
        resolved_api_key = (
            resolved_settings.llm_api_key
            or provider_keys.get(resolved_provider)
        )

    resolved_base_url = _clean_optional_text(base_url) or argument_base_url
    if resolved_base_url is None:
        if resolved_provider.value == resolved_settings.llm_provider:
            resolved_base_url = resolved_settings.resolved_llm_base_url
        elif resolved_provider == LLMProvider.GROQ:
            resolved_base_url = "https://api.groq.com/openai/v1"
        elif resolved_provider == LLMProvider.OPENROUTER:
            resolved_base_url = "https://openrouter.ai/api/v1"
        elif resolved_provider == LLMProvider.OLLAMA:
            resolved_base_url = resolved_settings.ollama_base_url

    return LLMRuntimeOptions(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=(resolved_base_url.rstrip("/") if resolved_base_url else None),
    )


def build_llm_client(
    options: LLMRuntimeOptions,
    *,
    settings: Settings | None = None,
) -> TextGenerationClient:
    """Seçilen sağlayıcı için ortak TextGenerationClient oluşturur."""

    resolved_settings = settings or get_settings()

    if options.provider == LLMProvider.GEMINI:
        return GeminiClient(
            api_key=options.api_key,
            config=GeminiClientConfig(
                model=options.model,
                temperature=resolved_settings.llm_temperature,
                maximum_output_tokens=(
                    resolved_settings.llm_maximum_output_tokens
                ),
                top_p=resolved_settings.llm_top_p,
                top_k=resolved_settings.gemini_top_k,
                thinking_level=resolved_settings.gemini_thinking_level,
                timeout_seconds=resolved_settings.llm_timeout_seconds,
                maximum_retries=resolved_settings.llm_maximum_retries,
                initial_retry_delay_seconds=(
                    resolved_settings.llm_initial_retry_delay_seconds
                ),
                maximum_retry_delay_seconds=(
                    resolved_settings.llm_maximum_retry_delay_seconds
                ),
            ),
        )

    if (
        options.provider == LLMProvider.OPENAI_COMPATIBLE
        and not options.base_url
    ):
        raise ValueError(
            "openai_compatible sağlayıcısı için --llm-base-url veya "
            "LLM_BASE_URL tanımlanmalıdır."
        )

    client_config = OpenAICompatibleClientConfig(
        provider=options.provider,
        model=options.model,
        base_url=options.base_url,
        temperature=resolved_settings.llm_temperature,
        maximum_output_tokens=resolved_settings.llm_maximum_output_tokens,
        top_p=resolved_settings.llm_top_p,
        timeout_seconds=resolved_settings.llm_timeout_seconds,
        connect_timeout_seconds=(
            resolved_settings.llm_connect_timeout_seconds
        ),
        maximum_retries=resolved_settings.llm_maximum_retries,
        initial_retry_delay_seconds=(
            resolved_settings.llm_initial_retry_delay_seconds
        ),
        maximum_retry_delay_seconds=(
            resolved_settings.llm_maximum_retry_delay_seconds
        ),
        retry_jitter_ratio=resolved_settings.llm_retry_jitter_ratio,
        json_mode_policy=resolved_settings.llm_json_mode_policy,
        openrouter_http_referer=resolved_settings.openrouter_http_referer,
        openrouter_app_title=resolved_settings.openrouter_app_title,
        verify_ssl=resolved_settings.llm_verify_ssl,
        trust_environment=resolved_settings.llm_trust_environment,
    )
    return OpenAICompatibleClient(
        api_key=options.api_key,
        config=client_config,
    )


def resolve_vision_options(
    args: argparse.Namespace | None = None,
    *,
    settings: Settings | None = None,
) -> VisionRuntimeOptions:
    resolved_settings = settings or get_settings()
    disabled = bool(getattr(args, "disable_vision_fallback", False)) if args else False
    explicitly_enabled = bool(getattr(args, "enable_vision_fallback", False)) if args else False

    provider_value = (
        _clean_optional_text(getattr(args, "vision_provider", None))
        if args is not None else None
    ) or resolved_settings.vision_provider
    provider = LLMProvider.normalize(provider_value)
    if provider not in {
        LLMProvider.GROQ,
        LLMProvider.OPENROUTER,
        LLMProvider.OPENAI_COMPATIBLE,
    }:
        raise ValueError(
            "Vision sağlayıcısı groq, openrouter veya openai_compatible olmalıdır."
        )

    model = (
        _clean_optional_text(getattr(args, "vision_model", None))
        if args is not None else None
    ) or resolved_settings.vision_model

    enabled = (
        explicitly_enabled
        or resolved_settings.vision_fallback_enabled
        or bool(model)
    ) and not disabled

    if enabled and not model:
        raise ValueError(
            "Vision fallback açık ancak model adı yok. "
            "--vision-model veya VISION_MODEL tanımlayın."
        )

    api_key = (
        _clean_optional_text(getattr(args, "vision_api_key", None))
        if args is not None else None
    ) or resolved_settings.resolved_vision_api_key

    base_url = (
        _clean_optional_text(getattr(args, "vision_base_url", None))
        if args is not None else None
    ) or resolved_settings.resolved_vision_base_url

    return VisionRuntimeOptions(
        provider=provider,
        model=model or "disabled",
        api_key=api_key,
        base_url=base_url,
        enabled=enabled,
    )


def build_vision_client(
    options: VisionRuntimeOptions,
    *,
    settings: Settings,
) -> OpenAICompatibleMultimodalClient | None:
    if not options.enabled:
        return None

    common = dict(
        api_key=options.api_key,
        maximum_images_per_request=settings.vision_maximum_images,
        default_image_detail=settings.vision_image_detail,
        temperature=settings.vision_temperature,
        maximum_output_tokens=settings.vision_maximum_output_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        connect_timeout_seconds=settings.llm_connect_timeout_seconds,
        maximum_retries=settings.llm_maximum_retries,
        initial_retry_delay_seconds=settings.llm_initial_retry_delay_seconds,
        maximum_retry_delay_seconds=settings.llm_maximum_retry_delay_seconds,
        retry_jitter_ratio=settings.llm_retry_jitter_ratio,
        json_mode_policy=settings.llm_json_mode_policy,
        verify_ssl=settings.llm_verify_ssl,
        trust_environment=settings.llm_trust_environment,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_app_title=settings.openrouter_app_title,
    )

    if options.provider == LLMProvider.GROQ:
        return OpenAICompatibleMultimodalClient.for_groq(
            options.model, **common
        )
    if options.provider == LLMProvider.OPENROUTER:
        return OpenAICompatibleMultimodalClient.for_openrouter(
            options.model, **common
        )
    if not options.base_url:
        raise ValueError(
            "openai_compatible vision sağlayıcısı için VISION_BASE_URL gereklidir."
        )
    return OpenAICompatibleMultimodalClient.for_openai_compatible(
        options.model,
        base_url=options.base_url,
        **common,
    )


def resolve_groq_vlm_enabled(
    args: argparse.Namespace | None = None,
    *,
    settings: Settings | None = None,
) -> bool:
    """Groq Qwen vision ingestion fallback durumunu çözümler."""
    resolved_settings = settings or get_settings()
    explicitly_enabled = bool(getattr(args, "enable_vlm_fallback", False)) if args is not None else False
    explicitly_disabled = bool(getattr(args, "disable_vlm_fallback", False)) if args is not None else False
    if explicitly_disabled:
        return False
    api_key = _clean_optional_text(os.getenv("GROQ_API_KEY"))
    configured_enabled = bool(getattr(resolved_settings, "vlm_fallback_enabled", False))
    enabled = explicitly_enabled or configured_enabled or bool(api_key)
    if enabled and not api_key:
        raise ValueError("Groq ingestion VLM fallback etkin ancak GROQ_API_KEY eksik.")
    return enabled



def build_runtime_resources(
    *,
    data_directory: str | Path = "data",
) -> RuntimeResources:
    """Ortak depolama ve retrieval bileşenlerini tek kez oluşturur."""

    data_path = Path(data_directory)
    downloads_path = data_path / "downloads"
    rendered_pages_path = data_path / "rendered_pages"
    vector_path = data_path / "vectors"
    table_images_path = data_path / "table_images"
    database_path = data_path / "sustainability.db"
    file_store_path = data_path / "files"

    for directory in (
        data_path,
        downloads_path,
        rendered_pages_path,
        vector_path,
        table_images_path,
        file_store_path,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    database = Database(database_path=database_path)
    file_store = FileStore(root_dir=file_store_path)

    table_repository = TableRepository(
        database=database,
        file_store=file_store,
    )
    visual_asset_repository = VisualAssetRepository(
        database=database,
        file_store=file_store,
    )
    chunk_repository = ChunkRepository()

    embedding_provider = HashingEmbeddingProvider(dimension=384)
    vector_store = VectorStore(
        config=VectorStoreConfig(
            collection_name="sustainability_documents",
            persist_directory=vector_path,
        ),
        embedding_provider=embedding_provider,
    )

    return RuntimeResources(
        data_path=data_path,
        downloads_path=downloads_path,
        rendered_pages_path=rendered_pages_path,
        vector_path=vector_path,
        table_images_path=table_images_path,
        database=database,
        file_store=file_store,
        table_repository=table_repository,
        visual_asset_repository=visual_asset_repository,
        chunk_repository=chunk_repository,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


def hydrate_runtime_resources(
    resources: RuntimeResources,
    *,
    report_uid: str,
    report_database_id: int,
) -> tuple[int, int]:
    """
    Yeni Python sürecinde bellek içi table/chunk repository'lerini
    SQLite kayıtlarından yeniden yükler.
    """

    table_count = resources.table_repository.load_report(
        report_uid,
        report_database_id=report_database_id,
        clear_existing=False,
    )

    chunk_count = resources.chunk_repository.load_from_database(
        resources.database,
        report_database_id=report_database_id,
        report_uid=report_uid,
        clear_existing=True,
    )

    print(
        "Repository hydration: "
        f"tables={table_count}, chunks={chunk_count}",
        flush=True,
    )

    return table_count, chunk_count


def build_ingestion_pipeline(
    resources: RuntimeResources,
    *,
    enable_vlm_fallback: bool = True,
) -> IngestionPipeline:
    """Rapor bulma ve işleme pipeline'ını oluşturur."""

    try:
        table_ocr = TableOCR()
    except Exception:
        table_ocr = None

    vlm_table_fallback: Any | None = None

    if enable_vlm_fallback and VLMTableFallback is None:
        raise RuntimeError(
            "Groq Qwen Vision VLM modülü içe aktarılamadı: "
            f"{_VLM_IMPORT_ERROR}"
        )

    if enable_vlm_fallback and VLMTableFallback is not None:
        try:
            # Cloudflare Account ID, API tokenı ve model adı sınıf
            # tarafından .env / ortam değişkenlerinden okunur.
            vlm_table_fallback = VLMTableFallback()
        except Exception as exc:
            raise RuntimeError(
                "Groq Qwen Vision VLM fallback oluşturulamadı: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    from report_discovery.exa_search import ExaReportSearcher

    return IngestionPipeline(
        report_search=ExaReportSearcher(),
        candidate_ranker=CandidateRanker(),
        url_validator=UrlValidator(),
        pdf_downloader=PdfDownloader(
            download_directory=resources.downloads_path,
        ),
        pdf_validator=PdfValidator(),
        page_renderer=PageRenderer(
            output_directory=resources.rendered_pages_path,
        ),
        page_classifier=PageClassifier(),
        layout_detector=LayoutDetector(),
        table_detector=TableDetector(),
        native_table_extractor=NativeTableExtractor(),
        visual_table_extractor=VisualTableExtractor(),
        table_ocr=table_ocr,
        vlm_table_fallback=vlm_table_fallback,
        table_normalizer=TableNormalizer(),
        numeric_validator=NumericValidator(),
        table_repository=resources.table_repository,
        visual_asset_repository=(
            resources.visual_asset_repository
        ),
        text_cleaner=TextCleaner(),
        section_detector=SectionDetector(),
        semantic_chunker=SemanticChunker(),
        category_classifier=CategoryClassifier(),
        metadata_enricher=MetadataEnricher(),
        chunk_repository=resources.chunk_repository,
        standard_index_parser=StandardIndexParser(),
        vector_store=resources.vector_store,
        database=resources.database,
        file_store=resources.file_store,
        table_image_directory=resources.table_images_path,
    )


def build_answering_pipelines(
    resources: RuntimeResources,
    *,
    llm_options: LLMRuntimeOptions | None = None,
    settings: Settings | None = None,
    llm_provider: str | LLMProvider | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    vision_options: VisionRuntimeOptions | None = None,
) -> AnsweringPipelines:
    """Tek soru ve kategori tabanlı cevaplama pipeline'larını oluşturur."""

    resolved_settings = settings or get_settings()
    options = llm_options or resolve_llm_options(
        settings=resolved_settings,
        provider=llm_provider,
        model=llm_model or gemini_model,
        api_key=llm_api_key or gemini_api_key,
        base_url=llm_base_url,
    )
    llm_client = build_llm_client(
        options,
        settings=resolved_settings,
    )

    standard_router = StandardRouter()
    question_analyzer = QuestionAnalyzer(
        standard_router=standard_router,
    )

    bm25_retriever = BM25Retriever(
        repository=resources.chunk_repository,
    )
    bm25_document_count = bm25_retriever.build_index()

    print(
        "BM25 index: "
        f"documents={bm25_document_count}",
        flush=True,
    )

    question_answering_pipeline = QuestionAnsweringPipeline(
        question_analyzer=question_analyzer,
        table_retriever=TableRetriever(
            repository=resources.table_repository,
        ),
        bm25_retriever=bm25_retriever,
        vector_retriever=VectorRetriever(
            embedding_provider=resources.embedding_provider,
            vector_store=resources.vector_store,
        ),
        standard_router=standard_router,
        result_fusion=ResultFusion(),
        reranker=Reranker(),
        context_expander=ContextExpander(
            chunk_repository=resources.chunk_repository,
            table_repository=resources.table_repository,
        ),
        evidence_validator=EvidenceValidator(),
        answer_generator=AnswerGenerator(llm_client),
        citation_builder=CitationBuilder(),
    )

    resolved_vision_options = vision_options or VisionRuntimeOptions(
        provider=LLMProvider.OPENROUTER,
        model="disabled",
        enabled=False,
    )
    vision_client = build_vision_client(
        resolved_vision_options,
        settings=resolved_settings,
    )

    visual_evidence_pipeline = None
    if vision_client is not None:
        visual_retriever = VisualRetriever(
            repository=resources.visual_asset_repository,
            config=VisualRetrieverConfig(
                top_k=resolved_settings.vision_maximum_images,
                maximum_candidate_count=300,
                minimum_score=0.12,
            ),
        )
        visual_generator = VisualEvidenceGenerator(
            client=vision_client,
            config=VisualEvidenceGeneratorConfig(
                maximum_images_per_request=(
                    resolved_settings.vision_maximum_images
                ),
                minimum_metric_confidence=(
                    resolved_settings.vision_minimum_metric_confidence
                ),
                temperature=resolved_settings.vision_temperature,
                maximum_output_tokens=(
                    resolved_settings.vision_maximum_output_tokens
                ),
                image_detail=resolved_settings.vision_image_detail,
            ),
        )
        visual_evidence_pipeline = VisualEvidencePipeline(
            visual_retriever=visual_retriever,
            evidence_generator=visual_generator,
            visual_asset_repository=resources.visual_asset_repository,
            config=VisualEvidencePipelineConfig(
                enabled=True,
                require_vision_only=(
                    resolved_settings.vision_require_pending_only
                ),
                top_k_per_question=(
                    resolved_settings.vision_top_k_per_question
                ),
                maximum_unique_visuals=(
                    resolved_settings.vision_maximum_images
                ),
                enable_cache=resolved_settings.vision_cache_enabled,
                cache_directory=(
                    resources.data_path / "cache" / "visual_evidence"
                ),
            ),
        )

    category_question_answering_pipeline = CategoryQuestionAnsweringPipeline(
        question_answering_pipeline=question_answering_pipeline,
        category_answer_generator=CategoryAnswerGenerator(llm_client),
        visual_evidence_pipeline=visual_evidence_pipeline,
    )

    return AnsweringPipelines(
        question_answering=question_answering_pipeline,
        category_question_answering=category_question_answering_pipeline,
        llm_client=llm_client,
        llm_provider=options.provider.value,
        llm_model=options.model,
        vision_client=vision_client,
        vision_provider=(
            resolved_vision_options.provider.value
            if resolved_vision_options.enabled else None
        ),
        vision_model=(
            resolved_vision_options.model
            if resolved_vision_options.enabled else None
        ),
    )


def build_pipelines(
    *,
    data_directory: str | Path = "data",
    llm_provider: str | LLMProvider | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
) -> tuple[
    IngestionPipeline,
    QuestionAnsweringPipeline,
    CategoryQuestionAnsweringPipeline,
]:
    """Geriye dönük kullanım için bütün pipeline'ları birlikte oluşturur."""

    settings = get_settings()
    options = resolve_llm_options(
        settings=settings,
        provider=llm_provider,
        model=llm_model or gemini_model,
        api_key=llm_api_key or gemini_api_key,
        base_url=llm_base_url,
    )
    resources = build_runtime_resources(data_directory=data_directory)
    ingestion_pipeline = build_ingestion_pipeline(
        resources,
        enable_vlm_fallback=resolve_groq_vlm_enabled(
            settings=settings,
        ),
    )
    answering_pipelines = build_answering_pipelines(
        resources,
        llm_options=options,
        settings=settings,
    )

    return (
        ingestion_pipeline,
        answering_pipelines.question_answering,
        answering_pipelines.category_question_answering,
    )


def _environment_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _should_sync_to_blob(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_sync_to_blob", False)):
        return False
    if bool(getattr(args, "sync_to_blob", False)):
        return True
    return _environment_flag("SYNC_TO_BLOB", default=False)


def _should_restore_from_blob(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "restore_from_blob", False)):
        return True
    return _environment_flag("RESTORE_FROM_BLOB", default=False)


def _restore_report_before_command(args: argparse.Namespace) -> bool:
    if not _should_restore_from_blob(args):
        return True

    report_id = _clean_optional_text(getattr(args, "report_id", None))
    if not report_id:
        print(
            "Cloud restore için --report-id zorunludur.",
            file=sys.stderr,
        )
        return False

    remote_prefix = (
        _clean_optional_text(getattr(args, "restore_prefix", None))
        or _clean_optional_text(getattr(args, "blob_prefix", None))
    )

    try:
        result = restore_report_inputs(
            report_id=report_id,
            data_directory=args.data_directory,
            remote_prefix=remote_prefix,
            overwrite=True,
            require_database=True,
            include_global_vectors=bool(
                getattr(args, "restore_global_vectors", False)
            ),
            include_report_assets=bool(
                getattr(args, "restore_with_assets", False)
            ),
        )
    except Exception as exc:
        print_restore_error(exc)
        return False

    print_restore_result(result)
    return True


def _sync_report_after_command(
    args: argparse.Namespace,
    *,
    report_id: str,
    additional_files: Mapping[str, Path | str] | None = None,
    include_sqlite_state: bool = True,
    include_global_vectors: bool = False,
    include_report_directories: bool = True,
) -> bool:
    """
    İstenen rapor çıktılarını cloud storage backend'e senkronize eder.

    additional_files, data klasörü dışında oluşturulan Excel gibi çıktıları
    uzak prefix altına eklemek için kullanılır.
    """
    if not _should_sync_to_blob(args):
        return True

    try:
        result = sync_report_outputs(
            report_id=report_id,
            data_directory=args.data_directory,
            remote_prefix=getattr(args, "blob_prefix", None),
            overwrite=True,
            additional_files=additional_files,
            include_sqlite_state=include_sqlite_state,
            include_global_vectors=include_global_vectors,
            include_report_directories=include_report_directories,
        )
    except Exception as exc:
        print_sync_error(exc)
        return not bool(getattr(args, "sync_required", False))

    print_sync_result(result)
    return True


def run_ingestion(args: argparse.Namespace) -> int:
    settings = get_settings()
    resources = build_runtime_resources(
        data_directory=args.data_directory,
    )
    vlm_enabled = resolve_groq_vlm_enabled(
        args,
        settings=settings,
    )
    ingestion_pipeline = build_ingestion_pipeline(
        resources,
        enable_vlm_fallback=vlm_enabled,
    )

    if vlm_enabled:
        print(
            "Groq Qwen tablo/grafik fallback'i açık.",
            flush=True,
        )
    else:
        print(
            "Groq Qwen tablo/grafik fallback'i kapalı.",
            flush=True,
        )

    started_at = time.monotonic()
    stop_indicator = threading.Event()

    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _show_elapsed_time() -> None:
        # İlk mesajı hemen, devamını 30 saniyede bir gösterir.
        while not stop_indicator.is_set():
            elapsed = time.monotonic() - started_at
            print(
                f"[Ingestion sürüyor] Geçen süre: "
                f"{_format_duration(elapsed)}",
                flush=True,
            )
            if stop_indicator.wait(30.0):
                break

    indicator_thread = threading.Thread(
        target=_show_elapsed_time,
        name="ingestion-elapsed-time",
        daemon=True,
    )
    indicator_thread.start()

    try:
        result = ingestion_pipeline.run(
            IngestionRequest(
                company_name=args.company,
                report_year=args.year,
                report_url=args.url,
                local_pdf_path=args.pdf_path,
                report_id=args.report_id,
                overwrite_existing=args.overwrite,
            ),
        )
    finally:
        stop_indicator.set()
        indicator_thread.join(timeout=1.0)

    elapsed = time.monotonic() - started_at
    print(
        f"Ingestion tamamlandı. Toplam süre: {_format_duration(elapsed)}",
        flush=True,
    )
    print(json.dumps(_serialize(result), ensure_ascii=False, indent=2))

    if result.is_successful and not _sync_report_after_command(
        args,
        report_id=result.report_id,
    ):
        return 1

    return 0 if result.is_successful else 1


def run_question_answering(args: argparse.Namespace) -> int:
    settings = get_settings()
    options = resolve_llm_options(args, settings=settings)
    resources = build_runtime_resources(
        data_directory=args.data_directory,
    )
    if args.report_id:
        report_record = resources.database.get_report_by_uid(
            args.report_id
        )
        if report_record is not None:
            hydrate_runtime_resources(
                resources,
                report_uid=args.report_id,
                report_database_id=int(report_record["id"]),
            )

    answering_pipelines = build_answering_pipelines(
        resources,
        llm_options=options,
        settings=settings,
    )

    try:
        result = answering_pipelines.question_answering.run(
            QuestionAnsweringRequest(
                question=args.question,
                report_id=args.report_id,
                company_name=args.company,
                report_year=args.year,
                top_k=args.top_k,
                rerank_top_k=args.rerank_top_k,
                answer_language=args.language,
            ),
        )

        if args.json:
            print(json.dumps(_serialize(result), ensure_ascii=False, indent=2))
        elif result.answer:
            print(result.answer)
        elif result.error:
            print(f"Hata: {result.error}", file=sys.stderr)
        else:
            print("Soruyu yanıtlamak için yeterli kanıt bulunamadı.")

        return 0 if result.error is None else 1
    finally:
        answering_pipelines.close()


def run_full_pipeline(args: argparse.Namespace) -> int:
    settings = get_settings()
    options = resolve_llm_options(args, settings=settings)
    resources = build_runtime_resources(
        data_directory=args.data_directory,
    )
    vlm_enabled = resolve_groq_vlm_enabled(
        args,
        settings=settings,
    )
    ingestion_pipeline = build_ingestion_pipeline(
        resources,
        enable_vlm_fallback=vlm_enabled,
    )

    ingestion_result = ingestion_pipeline.run(
        IngestionRequest(
            company_name=args.company,
            report_year=args.year,
            report_url=args.url,
            local_pdf_path=args.pdf_path,
            report_id=args.report_id,
            overwrite_existing=args.overwrite,
        ),
    )

    if not ingestion_result.is_successful:
        print(
            json.dumps(
                _serialize(ingestion_result),
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    answering_pipelines = build_answering_pipelines(
        resources,
        llm_options=options,
        settings=settings,
    )
    print(
        f"Cevap modeli: {answering_pipelines.llm_provider} / "
        f"{answering_pipelines.llm_model}"
    )

    try:
        for question in args.questions:
            result = answering_pipelines.question_answering.run(
                QuestionAnsweringRequest(
                    question=question,
                    report_id=ingestion_result.report_id,
                    company_name=args.company,
                    report_year=args.year,
                    top_k=args.top_k,
                    rerank_top_k=args.rerank_top_k,
                    answer_language=args.language,
                ),
            )

            print(f"\nSoru: {question}")
            print(
                f"Cevap: {result.answer or 'Yeterli kanıt bulunamadı.'}"
            )

        if not _sync_report_after_command(
            args,
            report_id=ingestion_result.report_id,
        ):
            return 1

        return 0
    finally:
        answering_pipelines.close()


def extract_result_pages(result: Any) -> tuple[int, ...]:
    validation = getattr(result, "evidence_validation", None)
    if validation is None:
        return ()

    accepted_evidences = getattr(validation, "accepted_evidences", ())
    pages = {
        evidence.page_number
        for evidence in accepted_evidences
        if getattr(evidence, "page_number", None) is not None
    }
    return tuple(sorted(pages))


def _select_groups(
    catalog: QuestionCatalog,
    requested_groups: Sequence[str] | None,
) -> tuple[QuestionGroup, ...]:
    if not requested_groups:
        return catalog.groups

    normalized_names = tuple(
        dict.fromkeys(
            name.strip().casefold()
            for name in requested_groups
            if name and name.strip()
        )
    )
    if not normalized_names:
        return catalog.groups

    unknown = [
        name
        for name in normalized_names
        if name not in catalog.group_names
    ]
    if unknown:
        raise ValueError(
            "Bilinmeyen soru kategorileri: "
            f"{', '.join(unknown)}. Geçerli kategoriler: "
            f"{', '.join(catalog.group_names)}."
        )

    return tuple(catalog.get_group(name) for name in normalized_names)


def _result_answer_text(result: QuestionAnsweringResult) -> str:
    if result.answer:
        return result.answer
    if result.status == QuestionAnsweringStatus.INSUFFICIENT_EVIDENCE:
        return "Yeterli kanıt bulunamadı."
    return "Cevap üretilemedi."


def _append_category_rows(
    *,
    rows: list[ExcelAnswerRow],
    group: QuestionGroup,
    result: CategoryQuestionAnsweringResult,
) -> None:
    if len(result.question_results) != len(group.questions):
        raise RuntimeError(
            f"{group.name} kategorisinde soru ve sonuç sayısı eşleşmiyor: "
            f"{len(group.questions)} soru, "
            f"{len(result.question_results)} sonuç."
        )

    for definition, question_result in zip(
        group.questions,
        result.question_results,
        strict=True,
    ):
        generated_answer = question_result.generated_answer

        rows.append(
            ExcelAnswerRow(
                question_number=definition.id,
                question=definition.question,
                answer=_result_answer_text(question_result),
                status=question_result.status.value,
                confidence=(
                    generated_answer.confidence
                    if generated_answer is not None
                    else None
                ),
                evidence_ids=(
                    generated_answer.evidence_ids
                    if generated_answer is not None
                    else ()
                ),
                pages=extract_result_pages(question_result),
                error=question_result.error,
            ),
        )


def _append_failed_group_rows(
    *,
    rows: list[ExcelAnswerRow],
    group: QuestionGroup,
    error: Exception,
) -> None:
    error_text = f"{type(error).__name__}: {error}"
    for definition in group.questions:
        rows.append(
            ExcelAnswerRow(
                question_number=definition.id,
                question=definition.question,
                answer="Cevap üretilemedi.",
                status=QuestionAnsweringStatus.FAILED.value,
                error=error_text,
            ),
        )


def _add_optional_total(current: int | None, value: int | None) -> int | None:
    if value is None:
        return current
    return (current or 0) + value


def run_batch(args: argparse.Namespace) -> int:
    if _should_restore_from_blob(args) and not args.skip_ingestion:
        print(
            "--restore-from-blob yalnızca --skip-ingestion ile kullanılmalıdır.",
            file=sys.stderr,
        )
        return 2

    if args.skip_ingestion and args.overwrite:
        print(
            "--skip-ingestion ile --overwrite birlikte kullanılamaz.",
            file=sys.stderr,
        )
        return 2

    if args.skip_ingestion and (args.url or args.pdf_path):
        print(
            "--skip-ingestion kullanılırken --url veya --pdf-path "
            "verilmemelidir.",
            file=sys.stderr,
        )
        return 2

    if not _restore_report_before_command(args):
        return 1

    settings = get_settings()
    options = resolve_llm_options(args, settings=settings)
    vision_options = resolve_vision_options(args, settings=settings)
    resources = build_runtime_resources(
        data_directory=args.data_directory,
    )
    vlm_enabled = resolve_groq_vlm_enabled(
        args,
        settings=settings,
    )
    ingestion_pipeline = build_ingestion_pipeline(
        resources,
        enable_vlm_fallback=vlm_enabled,
    )

    if args.skip_ingestion:
        if not args.report_id:
            print(
                "--skip-ingestion kullanılırken --report-id zorunludur.",
                file=sys.stderr,
            )
            return 2

        report_record = resources.database.get_report_by_uid(
            args.report_id
        )
        if report_record is None:
            print(
                "Mevcut rapor kaydı bulunamadı: "
                f"{args.report_id}",
                file=sys.stderr,
            )
            return 1

        def report_value(name: str, default: Any = None) -> Any:
            if isinstance(report_record, Mapping):
                return report_record.get(name, default)
            try:
                return report_record[name]
            except (KeyError, IndexError, TypeError):
                return default

        report_database_id = int(
            report_value("id")
        )
        hydrate_runtime_resources(
            resources,
            report_uid=args.report_id,
            report_database_id=report_database_id,
        )

        stored_company = str(
            report_value("company_name")
            or args.company
        )
        stored_year = int(
            report_value("report_year")
            or args.year
        )
        stored_pdf_path = report_value("local_pdf_path")
        stored_report_url = (
            report_value("final_url")
            or report_value("source_url")
        )

        if stored_company.strip().casefold() != args.company.strip().casefold():
            print(
                "Uyarı: Veritabanındaki şirket adı komut satırındaki "
                "şirket adıyla farklı. Kayıt kullanılmaya devam edilecek."
            )

        if stored_year != args.year:
            print(
                "Rapor yılı uyuşmuyor: "
                f"kayıt={stored_year}, istek={args.year}",
                file=sys.stderr,
            )
            return 1

        ingestion_result = SimpleNamespace(
            report_id=args.report_id,
            company_name=stored_company,
            report_year=stored_year,
            report_url=stored_report_url,
            pdf_path=(
                Path(stored_pdf_path)
                if stored_pdf_path
                else None
            ),
            status="existing",
            page_count=int(report_value("page_count") or 0),
            table_count=0,
            chunk_count=0,
            errors=(),
            warnings=(),
            metadata={
                "source_mode": "existing_report",
                "skip_ingestion": True,
            },
            is_successful=True,
        )

        print(
            f"{args.report_id} raporu mevcut kayıtlardan kullanılıyor; "
            "ingestion atlandı."
        )
    else:
        if args.pdf_path:
            print(
                f"{args.company} {args.year} yerel sürdürülebilirlik raporu "
                "işleniyor..."
            )
        else:
            print(
                f"{args.company} {args.year} sürdürülebilirlik raporu "
                "aranıyor ve işleniyor..."
            )

        ingestion_result = ingestion_pipeline.run(
            IngestionRequest(
                company_name=args.company,
                report_year=args.year,
                report_url=args.url,
                local_pdf_path=args.pdf_path,
                report_id=args.report_id,
                overwrite_existing=args.overwrite,
            ),
        )

        if not ingestion_result.is_successful:
            print("Rapor işleme başarısız oldu.", file=sys.stderr)
            print(
                json.dumps(
                    _serialize(ingestion_result),
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1

    answering_pipelines = build_answering_pipelines(
        resources,
        llm_options=options,
        settings=settings,
        vision_options=vision_options,
    )
    print(
        f"Cevap modeli: {answering_pipelines.llm_provider} / "
        f"{answering_pipelines.llm_model}"
    )
    if vlm_enabled:
        print(
            "Groq Qwen ingestion fallback'i açık: "
            f"{os.getenv('GROQ_VISION_MODEL', 'qwen/qwen3.6-27b')}"
        )
    else:
        print("Groq Qwen ingestion fallback'i kapalı.")
    if answering_pipelines.vision_client is not None:
        print(
            f"Vision modeli: {answering_pipelines.vision_provider} / "
            f"{answering_pipelines.vision_model}"
        )
    else:
        print("Kategori vision fallback'i kapalı.")

    report_record = resources.database.get_report_by_uid(
        ingestion_result.report_id
    )
    if report_record is None:
        raise RuntimeError(
            "İşlenen raporun veritabanı kaydı bulunamadı."
        )
    report_database_id = int(report_record["id"])

    catalog = QuestionCatalog.from_file(args.questions_file)
    groups = _select_groups(catalog, args.groups)

    print(
        f"\n{len(catalog)} soru, {len(groups)} kategori üzerinden "
        "işlenecek."
    )

    answer_rows: list[ExcelAnswerRow] = []
    category_results: list[CategoryQuestionAnsweringResult] = []

    total_model_calls = 0
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None

    for index, group in enumerate(groups, start=1):
        print(
            f"\n[{index}/{len(groups)}] {group.name} kategorisi "
            f"işleniyor: {group.size} soru"
        )

        try:
            category_result = (
                answering_pipelines.category_question_answering.run_group(
                    group,
                    report_id=ingestion_result.report_id,
                    company_name=args.company,
                    report_year=args.year,
                    top_k=args.top_k,
                    rerank_top_k=args.rerank_top_k,
                    answer_language=args.language,
                    strict_category_filter=True,
                    temperature=args.category_temperature,
                    maximum_output_tokens=(
                        args.category_max_output_tokens
                    ),
                    model=answering_pipelines.llm_model,
                    continue_on_preparation_error=True,
                    enable_visual_fallback=(
                        answering_pipelines.vision_client is not None
                    ),
                    report_database_id=report_database_id,
                    vision_model=answering_pipelines.vision_model,
                    force_visual_refresh=bool(
                        getattr(args, "force_vision_refresh", False)
                    ),
                    metadata={
                        "batch_company": args.company,
                        "batch_report_year": args.year,
                    },
                )
            )
        except Exception as exc:
            print(
                f"  Kategori çalıştırılamadı: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            _append_failed_group_rows(
                rows=answer_rows,
                group=group,
                error=exc,
            )
            if args.fail_fast:
                return 1
            continue

        category_results.append(category_result)
        _append_category_rows(
            rows=answer_rows,
            group=group,
            result=category_result,
        )

        total_model_calls += category_result.model_call_count
        total_input_tokens = _add_optional_total(
            total_input_tokens,
            category_result.input_tokens,
        )
        total_output_tokens = _add_optional_total(
            total_output_tokens,
            category_result.output_tokens,
        )
        total_tokens = _add_optional_total(
            total_tokens,
            category_result.total_tokens,
        )

        print(
            "  "
            f"Durum: {category_result.status.value} | "
            f"Tamamlanan: {category_result.completed_count} | "
            "Kanıt yetersiz: "
            f"{category_result.insufficient_evidence_count} | "
            f"Hata: {category_result.failed_count} | "
            f"Model çağrısı: {category_result.model_call_count}"
        )

        if category_result.total_tokens is not None:
            print(
                "  "
                f"Token: giriş={category_result.input_tokens or 0}, "
                f"çıkış={category_result.output_tokens or 0}, "
                f"toplam={category_result.total_tokens}"
            )

        if args.fail_fast and category_result.failed_count:
            print(
                "Kategori içerisinde hata oluştuğu için işlem durduruldu.",
                file=sys.stderr,
            )
            return 1

    answer_rows.sort(key=lambda item: item.question_number)

    exporter = ExcelExporter(
        output_directory=args.output_directory,
    )
    output_path = exporter.export_answers(
        company_name=args.company,
        report_year=args.year,
        rows=answer_rows,
        report_id=ingestion_result.report_id,
        report_url=ingestion_result.report_url,
    )

    completed_count = sum(
        row.status == QuestionAnsweringStatus.COMPLETED.value
        for row in answer_rows
    )
    insufficient_count = sum(
        row.status == QuestionAnsweringStatus.INSUFFICIENT_EVIDENCE.value
        for row in answer_rows
    )
    failed_count = sum(
        row.status == QuestionAnsweringStatus.FAILED.value
        for row in answer_rows
    )

    print(f"\nExcel dosyası oluşturuldu: {output_path}")
    print(
        "Batch özeti: "
        f"soru={len(answer_rows)}, "
        f"tamamlanan={completed_count}, "
        f"kanıt_yetersiz={insufficient_count}, "
        f"hata={failed_count}, "
        f"model_çağrısı={total_model_calls}"
    )

    if total_tokens is not None:
        print(
            "Token özeti: "
            f"giriş={total_input_tokens or 0}, "
            f"çıkış={total_output_tokens or 0}, "
            f"toplam={total_tokens}"
        )

    if not answer_rows:
        return 1
    if failed_count == len(answer_rows):
        return 1

    batch_additional_files = {
        f"outputs/{Path(output_path).name}": Path(output_path),
    }

    if not _sync_report_after_command(
        args,
        report_id=ingestion_result.report_id,
        additional_files=batch_additional_files,
        include_sqlite_state=True,
        include_global_vectors=False,
        include_report_directories=not bool(args.skip_ingestion),
    ):
        return 1

    return 0


def _argument_default(
    value: Any,
    *,
    suppress_defaults: bool,
) -> Any:
    return argparse.SUPPRESS if suppress_defaults else value


def _add_llm_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    parser.add_argument(
        "--llm-provider",
        choices=[item.value for item in LLMProvider],
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "Cevap modeli sağlayıcısı: gemini, groq, openrouter, "
            "ollama veya openai_compatible. Verilmezse LLM_PROVIDER "
            "ayarından okunur."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help="Cevap üretiminde kullanılacak model adı.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "Seçilen cevap modeli sağlayıcısının API anahtarı. "
            "Verilmezse .env içindeki sağlayıcıya özel anahtar kullanılır."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "OpenAI-compatible API temel adresi. Ollama veya özel "
            "sağlayıcılarda değiştirilebilir."
        ),
    )
    parser.add_argument(
        "--gemini-api-key",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "Geriye dönük alias. Yalnızca Gemini cevap sağlayıcısı "
            "için kullanılır."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "Geriye dönük alias. Kullanılırsa ve --llm-provider "
            "verilmemişse cevap sağlayıcısı Gemini seçilir."
        ),
    )


def _add_vlm_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    parser.add_argument(
        "--enable-vlm-fallback",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help=(
            "Ingestion sırasında Groq Qwen Vision ile görsel tablo ve "
            "grafik fallback işlemini açıkça etkinleştirir. Kimlik bilgileri "
            ".env içindeki GROQ_API_KEY alanından okunur."
        ),
    )
    parser.add_argument(
        "--disable-vlm-fallback",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help=(
            "Ingestion sırasında Groq Qwen görsel tablo/grafik "
            "fallback işlemini kapatır."
        ),
    )



def _add_vision_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    parser.add_argument(
        "--vision-provider",
        choices=["groq", "openrouter", "openai_compatible"],
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help="Multimodal vision sağlayıcısı.",
    )
    parser.add_argument(
        "--vision-model",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help="Görsel tablo/grafik analizinde kullanılacak multimodal model.",
    )
    parser.add_argument(
        "--vision-api-key",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help="Vision sağlayıcısının API anahtarı.",
    )
    parser.add_argument(
        "--vision-base-url",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help="Özel OpenAI-compatible vision API temel adresi.",
    )
    parser.add_argument(
        "--enable-vision-fallback",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help="Kategori aşamasındaki multimodal görsel fallback'i açar.",
    )
    parser.add_argument(
        "--disable-vision-fallback",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help="Kategori aşamasındaki multimodal görsel fallback'i kapatır.",
    )
    parser.add_argument(
        "--force-vision-refresh",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help="Görsel kanıt cache'i bulunsa bile vision analizini yeniden çalıştırır.",
    )


def _add_cloud_sync_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        "--sync-to-blob",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help=(
            "Komut başarıyla tamamlandığında rapor çıktılarını seçili "
            "storage backend'e yükler. STORAGE_BACKEND=azure_blob ise "
            "Azure Blob Storage kullanılır."
        ),
    )
    sync_group.add_argument(
        "--no-sync-to-blob",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help="SYNC_TO_BLOB ortam ayarı açık olsa bile senkronizasyonu kapatır.",
    )
    parser.add_argument(
        "--blob-prefix",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "Uzak hedef prefix'i. Verilmezse reports/<report_id> kullanılır."
        ),
    )
    parser.add_argument(
        "--sync-required",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help=(
            "Cloud senkronizasyonu başarısız olursa komutu hata koduyla "
            "sonlandırır."
        ),
    )
    parser.add_argument(
        "--restore-from-blob",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help=(
            "Blob'daki önceki rapor durumunu komut başlamadan önce yerel "
            "data klasörüne geri yükler. Batch komutunda --skip-ingestion "
            "ile birlikte kullanılmalıdır."
        ),
    )
    parser.add_argument(
        "--restore-prefix",
        default=_argument_default(None, suppress_defaults=suppress_defaults),
        help=(
            "Restore kaynağı Blob prefix'i. Verilmezse --blob-prefix, o da "
            "yoksa reports/<report_id> kullanılır."
        ),
    )
    parser.add_argument(
        "--restore-global-vectors",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help="state/vectors altındaki global vector store durumunu da indirir.",
    )
    parser.add_argument(
        "--restore-with-assets",
        action="store_true",
        default=_argument_default(False, suppress_defaults=suppress_defaults),
        help=(
            "Minimal restore yerine table_images, rendered_pages, pages, "
            "tables, chunks, vectors ve outputs klasörlerini de indirir. "
            "Vision veya görsel yeniden işleme gerektiğinde kullanılır."
        ),
    )


def _add_report_source_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """URL veya yerel PDF kaynaklarından yalnızca birini kabul eder."""

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--url",
        default=None,
        help="İsteğe bağlı doğrudan rapor URL'si.",
    )
    source_group.add_argument(
        "--pdf-path",
        default=None,
        help=(
            "Yerel PDF dosyasının yolu. Bu seçenek kullanıldığında "
            "rapor arama ve indirme adımları atlanır."
        ),
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sustainability-analyzer",
        description=(
            "Sürdürülebilirlik raporlarını bulan, işleyen "
            "ve soruları yanıtlayan analiz sistemi."
        ),
    )

    parser.add_argument(
        "--data-directory",
        default="data",
        help="Uygulama verilerinin saklanacağı dizin.",
    )
    _add_llm_arguments(parser)
    _add_vlm_arguments(parser)
    _add_vision_arguments(parser)
    _add_cloud_sync_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Bir sürdürülebilirlik raporunu işle.",
    )
    ingest_parser.add_argument("--company", required=True)
    ingest_parser.add_argument("--year", type=int, required=True)
    _add_report_source_arguments(ingest_parser)
    ingest_parser.add_argument("--report-id", default=None)
    ingest_parser.add_argument("--overwrite", action="store_true")
    # Global seçeneklerin komuttan sonra da yazılabilmesini sağlar.
    _add_vlm_arguments(ingest_parser, suppress_defaults=True)
    _add_cloud_sync_arguments(ingest_parser, suppress_defaults=True)
    ingest_parser.set_defaults(handler=run_ingestion)

    ask_parser = subparsers.add_parser(
        "ask",
        help="İşlenmiş bir rapora soru sor.",
    )
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--report-id", default=None)
    ask_parser.add_argument("--company", default=None)
    ask_parser.add_argument("--year", type=int, default=None)
    ask_parser.add_argument("--top-k", type=int, default=10)
    ask_parser.add_argument("--rerank-top-k", type=int, default=5)
    ask_parser.add_argument("--language", default="tr")
    ask_parser.add_argument("--json", action="store_true")
    _add_llm_arguments(ask_parser, suppress_defaults=True)
    ask_parser.set_defaults(handler=run_question_answering)

    batch_parser = subparsers.add_parser(
        "batch",
        help=(
            "Raporu işle, katalog sorularını kategori başına tek model "
            "çağrısıyla yanıtla ve Excel çıktısı oluştur."
        ),
    )
    batch_parser.add_argument("--company", required=True, help="Şirket adı.")
    batch_parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Rapor yılı.",
    )
    _add_report_source_arguments(batch_parser)
    batch_parser.add_argument("--report-id", default=None)
    batch_parser.add_argument(
        "--questions-file",
        default="config_data/questions.json",
        help="Metadata içeren JSON soru kataloğunun yolu.",
    )
    batch_parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Yalnızca belirtilen soru kategorilerini çalıştırır. "
            "Örnek: --groups energy water waste"
        ),
    )
    batch_parser.add_argument(
        "--output-directory",
        default="outputs",
        help="Excel çıktısının kaydedileceği klasör.",
    )
    batch_parser.add_argument("--top-k", type=int, default=12)
    batch_parser.add_argument("--rerank-top-k", type=int, default=6)
    batch_parser.add_argument("--language", default="tr")
    batch_parser.add_argument(
        "--category-temperature",
        type=float,
        default=0.0,
        help="Kategori model çağrısının temperature değeri.",
    )
    batch_parser.add_argument(
        "--category-max-output-tokens",
        type=int,
        default=1200,
        help="Tek kategori çağrısı için azami çıktı tokenı.",
    )
    batch_parser.add_argument(
        "--strict-category-filter",
        action="store_true",
        help=(
            "Retrieval sırasında ESG kategorisini katı filtre olarak uygular. "
            "Varsayılan olarak recall kaybını önlemek için kapalıdır."
        ),
    )
    batch_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="İlk kategori hatasında batch işlemini durdurur.",
    )
    batch_parser.add_argument("--overwrite", action="store_true")
    batch_parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help=(
            "Daha önce işlenmiş --report-id kaydını kullanır; "
            "rapor arama, indirme, render, OCR ve chunk üretimini atlar."
        ),
    )
    _add_llm_arguments(batch_parser, suppress_defaults=True)
    _add_vlm_arguments(batch_parser, suppress_defaults=True)
    _add_vision_arguments(batch_parser, suppress_defaults=True)
    _add_cloud_sync_arguments(batch_parser, suppress_defaults=True)
    batch_parser.set_defaults(handler=run_batch)

    full_parser = subparsers.add_parser(
        "run",
        help="Raporu işle ve komut satırında verilen serbest soruları yanıtla.",
    )
    full_parser.add_argument("--company", required=True)
    full_parser.add_argument("--year", type=int, required=True)
    _add_report_source_arguments(full_parser)
    full_parser.add_argument("--report-id", default=None)
    full_parser.add_argument("--questions", nargs="+", required=True)
    full_parser.add_argument("--top-k", type=int, default=10)
    full_parser.add_argument("--rerank-top-k", type=int, default=5)
    full_parser.add_argument("--language", default="tr")
    full_parser.add_argument("--overwrite", action="store_true")
    _add_llm_arguments(full_parser, suppress_defaults=True)
    _add_vlm_arguments(full_parser, suppress_defaults=True)
    _add_cloud_sync_arguments(full_parser, suppress_defaults=True)
    full_parser.set_defaults(handler=run_full_pipeline)

    return parser


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _serialize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]

    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _serialize(enum_value)

    try:
        return {
            key: _serialize(item)
            for key, item in asdict(value).items()
        }
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Beklenmeyen hata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())