from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from answering.question_catalog import QuestionGroup
from answering.visual_evidence_generator import (
    VisualEvidenceGenerationRequest,
    VisualEvidenceGenerationResult,
    VisualEvidenceGenerator,
    VisualEvidenceStatus,
)
from models.visual_evidence import VisualMetric, VisualProcessingStatus
from retrieval.visual_retriever import (
    VisualGroupRetrievalResult,
    VisualRetriever,
)
from storage.visual_asset_repository import VisualAssetRepository


logger = logging.getLogger(__name__)


class VisualEvidencePipelineError(RuntimeError):
    """Görsel kanıt pipeline işlemi başarısız olduğunda yükseltilir."""


class VisualEvidencePipelineStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NO_VISUALS = "no_visuals"
    NO_METRICS = "no_metrics"
    CACHED = "cached"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass(slots=True, frozen=True)
class VisualEvidencePipelineConfig:
    enabled: bool = True
    require_vision_only: bool = True

    top_k_per_question: int = 2
    maximum_unique_visuals: int = 2

    enable_cache: bool = True
    cache_directory: Path = Path("data/cache/visual_evidence")
    overwrite_cache: bool = False

    save_metrics: bool = True
    update_visual_status: bool = True

    continue_on_metric_store_error: bool = True
    continue_on_asset_update_error: bool = True

    schema_version: str = "visual-evidence-pipeline-v1"

    def __post_init__(self) -> None:
        if self.top_k_per_question < 1:
            raise ValueError("top_k_per_question en az 1 olmalıdır.")

        if self.maximum_unique_visuals < 1:
            raise ValueError("maximum_unique_visuals en az 1 olmalıdır.")

        if not self.schema_version.strip():
            raise ValueError("schema_version boş olamaz.")

        object.__setattr__(
            self,
            "cache_directory",
            Path(self.cache_directory).expanduser().resolve(),
        )


@dataclass(slots=True, frozen=True)
class VisualEvidencePipelineRequest:
    group: QuestionGroup

    report_id: str | UUID
    report_database_id: int
    report_year: int
    company_name: str

    candidate_pages_by_question: Mapping[int, Sequence[int]] = field(
        default_factory=dict
    )

    model: str | None = None
    force_refresh: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.report_database_id < 1:
            raise ValueError("report_database_id pozitif olmalıdır.")

        if not self.company_name.strip():
            raise ValueError("company_name boş olamaz.")

        if not 1990 <= self.report_year <= 2100:
            raise ValueError("report_year geçersiz.")


@dataclass(slots=True, frozen=True)
class VisualEvidencePipelineResult:
    group_name: str
    status: VisualEvidencePipelineStatus

    retrieval_result: VisualGroupRetrievalResult | None
    generation_result: VisualEvidenceGenerationResult | None

    metrics: tuple[VisualMetric, ...]
    stored_metric_ids: tuple[str, ...]

    visual_count: int
    metric_count: int
    model_call_count: int

    cache_key: str | None = None
    cache_hit: bool = False

    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_metrics(self) -> bool:
        return bool(self.metrics)

    @property
    def metrics_by_question(self) -> Mapping[int, tuple[VisualMetric, ...]]:
        grouped: dict[int, list[VisualMetric]] = {}

        for metric in self.metrics:
            for question_id in metric.question_ids:
                grouped.setdefault(question_id, []).append(metric)

        return {
            question_id: tuple(items)
            for question_id, items in grouped.items()
        }


class VisualEvidencePipeline:
    """
    Görsel retrieval, multimodal üretim, cache ve repository kayıtlarını
    tek akışta yöneten orkestrasyon katmanı.
    """

    def __init__(
        self,
        *,
        visual_retriever: VisualRetriever,
        evidence_generator: VisualEvidenceGenerator,
        visual_asset_repository: VisualAssetRepository,
        config: VisualEvidencePipelineConfig | None = None,
    ) -> None:
        self.visual_retriever = visual_retriever
        self.evidence_generator = evidence_generator
        self.visual_asset_repository = visual_asset_repository
        self.config = config or VisualEvidencePipelineConfig()

        if self.config.enable_cache:
            self.config.cache_directory.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        request: VisualEvidencePipelineRequest,
    ) -> VisualEvidencePipelineResult:
        if not self.config.enabled:
            return VisualEvidencePipelineResult(
                group_name=request.group.name,
                status=VisualEvidencePipelineStatus.DISABLED,
                retrieval_result=None,
                generation_result=None,
                metrics=(),
                stored_metric_ids=(),
                visual_count=0,
                metric_count=0,
                model_call_count=0,
                metadata={
                    "reason": "pipeline_disabled",
                    "schema_version": self.config.schema_version,
                },
            )

        errors: list[str] = []

        try:
            retrieval_result = self.visual_retriever.retrieve_group(
                request.group,
                report_id=request.report_id,
                report_year=request.report_year,
                candidate_pages_by_question=(
                    request.candidate_pages_by_question
                ),
                require_vision_only=self.config.require_vision_only,
                top_k_per_question=self.config.top_k_per_question,
                maximum_unique_visuals=(
                    self.config.maximum_unique_visuals
                ),
                metadata=request.metadata,
            )
        except Exception as exc:
            return self._failed_result(
                request,
                [
                    "Visual retrieval başarısız: "
                    f"{type(exc).__name__}: {exc}"
                ],
            )

        if not retrieval_result.unique_matches:
            return VisualEvidencePipelineResult(
                group_name=request.group.name,
                status=VisualEvidencePipelineStatus.NO_VISUALS,
                retrieval_result=retrieval_result,
                generation_result=None,
                metrics=(),
                stored_metric_ids=(),
                visual_count=0,
                metric_count=0,
                model_call_count=0,
                metadata={
                    "reason": "no_visual_matches",
                    "schema_version": self.config.schema_version,
                },
            )

        cache_key = self._build_cache_key(request, retrieval_result)

        if (
            self.config.enable_cache
            and not request.force_refresh
            and not self.config.overwrite_cache
        ):
            cached = self._load_cache(cache_key)

            if cached is not None:
                metrics = self._metrics_from_cache(cached)

                stored_metric_ids = (
                    self._store_metrics(request, metrics, errors)
                    if self.config.save_metrics
                    else ()
                )

                return VisualEvidencePipelineResult(
                    group_name=request.group.name,
                    status=VisualEvidencePipelineStatus.CACHED,
                    retrieval_result=retrieval_result,
                    generation_result=None,
                    metrics=metrics,
                    stored_metric_ids=stored_metric_ids,
                    visual_count=len(retrieval_result.unique_matches),
                    metric_count=len(metrics),
                    model_call_count=0,
                    cache_key=cache_key,
                    cache_hit=True,
                    errors=tuple(errors),
                    metadata={
                        "cache_path": str(self._cache_path(cache_key)),
                        "schema_version": self.config.schema_version,
                    },
                )

        generation_request = VisualEvidenceGenerationRequest(
            group=request.group,
            retrieval_result=retrieval_result,
            report_id=self._resolve_report_uuid(request.report_id),
            report_year=request.report_year,
            company_name=request.company_name,
            model=request.model,
            metadata=request.metadata,
        )

        generation_result = self.evidence_generator.generate(
            generation_request
        )

        metrics = tuple(generation_result.metrics)

        if generation_result.error:
            errors.append(generation_result.error)

        stored_metric_ids: tuple[str, ...] = ()

        if metrics and self.config.save_metrics:
            stored_metric_ids = self._store_metrics(
                request,
                metrics,
                errors,
            )

        if self.config.update_visual_status:
            self._update_visual_assets(
                request,
                retrieval_result,
                generation_result,
                errors,
            )

        if (
            self.config.enable_cache
            and generation_result.status
            in {
                VisualEvidenceStatus.COMPLETED,
                VisualEvidenceStatus.PARTIAL,
                VisualEvidenceStatus.NO_METRICS,
            }
        ):
            self._save_cache(
                cache_key,
                request,
                retrieval_result,
                generation_result,
            )

        status = self._map_status(generation_result.status)

        if errors and status == VisualEvidencePipelineStatus.COMPLETED:
            status = VisualEvidencePipelineStatus.PARTIAL

        return VisualEvidencePipelineResult(
            group_name=request.group.name,
            status=status,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            metrics=metrics,
            stored_metric_ids=stored_metric_ids,
            visual_count=len(retrieval_result.unique_matches),
            metric_count=len(metrics),
            model_call_count=generation_result.model_call_count,
            cache_key=cache_key,
            cache_hit=False,
            errors=tuple(errors),
            metadata={
                "schema_version": self.config.schema_version,
                "generation_status": generation_result.status.value,
                "provider": generation_result.provider,
                "model": generation_result.model,
                "input_tokens": generation_result.input_tokens,
                "output_tokens": generation_result.output_tokens,
                "total_tokens": generation_result.total_tokens,
            },
        )

    def _store_metrics(
        self,
        request: VisualEvidencePipelineRequest,
        metrics: Sequence[VisualMetric],
        errors: list[str],
    ) -> tuple[str, ...]:
        try:
            results, store_errors = (
                self.visual_asset_repository.store_metrics(
                    report_database_id=request.report_database_id,
                    company_name=request.company_name,
                    report_year=request.report_year,
                    metrics=metrics,
                    continue_on_error=(
                        self.config.continue_on_metric_store_error
                    ),
                )
            )
        except Exception as exc:
            errors.append(
                "VisualMetric kayıt işlemi başarısız: "
                f"{type(exc).__name__}: {exc}"
            )
            return ()

        errors.extend(store_errors)

        return tuple(result.metric_uid for result in results)

    def _update_visual_assets(
        self,
        request: VisualEvidencePipelineRequest,
        retrieval_result: VisualGroupRetrievalResult,
        generation_result: VisualEvidenceGenerationResult,
        errors: list[str],
    ) -> None:
        processed_visual_ids = {
            str(metric.visual_id)
            for metric in generation_result.metrics
        }

        successful_statuses = {
            VisualEvidenceStatus.COMPLETED,
            VisualEvidenceStatus.PARTIAL,
            VisualEvidenceStatus.NO_METRICS,
        }

        for match in retrieval_result.unique_matches:
            asset = match.asset

            if (
                str(asset.id) not in processed_visual_ids
                and generation_result.status not in successful_statuses
            ):
                continue

            asset.processing_status = (
                VisualProcessingStatus.VISION_COMPLETED
            )
            asset.requires_vision = False

            asset.metadata["last_visual_evidence_group"] = (
                request.group.name
            )
            asset.metadata["last_visual_evidence_status"] = (
                generation_result.status.value
            )

            try:
                self.visual_asset_repository.store_asset(
                    report_database_id=request.report_database_id,
                    company_name=request.company_name,
                    report_year=request.report_year,
                    asset=asset,
                )
            except Exception as exc:
                message = (
                    f"VisualAsset güncellenemedi ({asset.id}): "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(message)

                if not self.config.continue_on_asset_update_error:
                    raise

    def _build_cache_key(
        self,
        request: VisualEvidencePipelineRequest,
        retrieval_result: VisualGroupRetrievalResult,
    ) -> str:
        visuals = []

        for match in retrieval_result.unique_matches:
            asset = match.asset
            visuals.append(
                {
                    "visual_id": str(asset.id),
                    "image_hash": (
                        asset.image_hash
                        or self._file_hash(asset.image_path)
                    ),
                    "question_ids": list(match.question_ids),
                }
            )

        payload = {
            "pipeline_schema": self.config.schema_version,
            "generator_schema": (
                self.evidence_generator.config.schema_version
            ),
            "report_id": str(request.report_id),
            "report_year": request.report_year,
            "group": request.group.name,
            "question_ids": list(request.group.question_ids),
            "model": request.model,
            "visuals": visuals,
        }

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.config.cache_directory / f"{cache_key}.json"

    def _save_cache(
        self,
        cache_key: str,
        request: VisualEvidencePipelineRequest,
        retrieval_result: VisualGroupRetrievalResult,
        generation_result: VisualEvidenceGenerationResult,
    ) -> None:
        path = self._cache_path(cache_key)

        payload = {
            "schema_version": self.config.schema_version,
            "cache_key": cache_key,
            "group": request.group.name,
            "report_id": str(request.report_id),
            "report_year": request.report_year,
            "visual_ids": [
                match.visual_id
                for match in retrieval_result.unique_matches
            ],
            "generation_status": generation_result.status.value,
            "metrics": [
                metric.model_dump(mode="json")
                for metric in generation_result.metrics
            ],
            "model": generation_result.model,
            "provider": generation_result.provider,
        }

        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _load_cache(
        self,
        cache_key: str,
    ) -> Mapping[str, Any] | None:
        path = self._cache_path(cache_key)

        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Görsel kanıt cache dosyası okunamadı: %s",
                path,
            )
            return None

        if payload.get("cache_key") != cache_key:
            return None

        if payload.get("schema_version") != self.config.schema_version:
            return None

        return payload

    @staticmethod
    def _metrics_from_cache(
        payload: Mapping[str, Any],
    ) -> tuple[VisualMetric, ...]:
        raw_metrics = payload.get("metrics", [])

        if not isinstance(raw_metrics, list):
            return ()

        metrics: list[VisualMetric] = []

        for item in raw_metrics:
            try:
                metrics.append(VisualMetric.model_validate(item))
            except Exception:
                continue

        return tuple(metrics)

    @staticmethod
    def _map_status(
        status: VisualEvidenceStatus,
    ) -> VisualEvidencePipelineStatus:
        mapping = {
            VisualEvidenceStatus.COMPLETED: (
                VisualEvidencePipelineStatus.COMPLETED
            ),
            VisualEvidenceStatus.PARTIAL: (
                VisualEvidencePipelineStatus.PARTIAL
            ),
            VisualEvidenceStatus.NO_VISUALS: (
                VisualEvidencePipelineStatus.NO_VISUALS
            ),
            VisualEvidenceStatus.NO_METRICS: (
                VisualEvidencePipelineStatus.NO_METRICS
            ),
            VisualEvidenceStatus.GENERATION_FAILED: (
                VisualEvidencePipelineStatus.FAILED
            ),
            VisualEvidenceStatus.INVALID_RESPONSE: (
                VisualEvidencePipelineStatus.FAILED
            ),
        }

        return mapping[status]

    def _failed_result(
        self,
        request: VisualEvidencePipelineRequest,
        errors: Sequence[str],
    ) -> VisualEvidencePipelineResult:
        return VisualEvidencePipelineResult(
            group_name=request.group.name,
            status=VisualEvidencePipelineStatus.FAILED,
            retrieval_result=None,
            generation_result=None,
            metrics=(),
            stored_metric_ids=(),
            visual_count=0,
            metric_count=0,
            model_call_count=0,
            errors=tuple(errors),
            metadata={"schema_version": self.config.schema_version},
        )

    @staticmethod
    def _resolve_report_uuid(
        report_id: str | UUID,
    ) -> UUID:
        if isinstance(report_id, UUID):
            return report_id

        raw = str(report_id).strip()

        try:
            return UUID(raw)
        except ValueError:
            return uuid5(
                NAMESPACE_URL,
                f"sustainability-report:{raw}",
            )

    @staticmethod
    def _file_hash(path: str | Path) -> str | None:
        resolved = Path(path).expanduser()

        if not resolved.is_file():
            return None

        digest = hashlib.sha256()

        try:
            with resolved.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
        except OSError:
            return None

        return digest.hexdigest()


__all__ = [
    "VisualEvidencePipeline",
    "VisualEvidencePipelineConfig",
    "VisualEvidencePipelineError",
    "VisualEvidencePipelineRequest",
    "VisualEvidencePipelineResult",
    "VisualEvidencePipelineStatus",
]
