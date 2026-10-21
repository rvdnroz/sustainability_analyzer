from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from answering.question_analyzer import QuestionAnalysis
from answering.question_catalog import (
    QuestionDefinition,
    QuestionGroup,
)
from models.visual_evidence import (
    VisualAsset,
    VisualProcessingStatus,
    VisualRegionType,
)
from storage.visual_asset_repository import (
    VisualAssetRepository,
)


class VisualRetrieverError(RuntimeError):
    """Görsel retrieval işlemi tamamlanamadığında yükseltilir."""


class VisualAssetRepositoryProtocol(Protocol):
    """VisualRetriever tarafından ihtiyaç duyulan repository sözleşmesi."""

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
        ...

    def all_assets(self) -> list[VisualAsset]:
        ...


@dataclass(slots=True, frozen=True)
class VisualRetrieverConfig:
    """Görsel retrieval sıralama ve filtreleme ayarları."""

    top_k: int = 4
    maximum_candidate_count: int = 300
    minimum_score: float = 0.12

    require_existing_image: bool = True
    exclude_decorative_regions: bool = True
    include_completed_assets: bool = True

    prefer_vision_required: bool = True
    vision_required_boost: float = 0.08

    exact_topic_boost: float = 0.18
    exact_category_boost: float = 0.08
    candidate_page_boost: float = 0.12
    year_match_boost: float = 0.06
    unit_match_boost: float = 0.05

    caption_weight: float = 1.45
    section_weight: float = 1.30
    nearby_text_weight: float = 1.15
    ocr_text_weight: float = 1.00
    keyword_weight: float = 1.35

    phrase_score_weight: float = 0.36
    token_score_weight: float = 0.34
    fuzzy_score_weight: float = 0.10
    metadata_score_weight: float = 0.20

    minimum_fuzzy_term_length: int = 5
    fuzzy_match_threshold: float = 0.80
    maximum_matched_terms: int = 16

    allowed_region_types: tuple[VisualRegionType, ...] = (
        VisualRegionType.TABLE,
        VisualRegionType.CHART,
        VisualRegionType.METRIC_CARD,
        VisualRegionType.INFOGRAPHIC,
        VisualRegionType.FIGURE,
        VisualRegionType.UNKNOWN,
    )

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k en az 1 olmalıdır.")

        if self.maximum_candidate_count < self.top_k:
            raise ValueError(
                "maximum_candidate_count top_k değerinden küçük olamaz."
            )

        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError(
                "minimum_score 0 ile 1 arasında olmalıdır."
            )

        for field_name in (
            "vision_required_boost",
            "exact_topic_boost",
            "exact_category_boost",
            "candidate_page_boost",
            "year_match_boost",
            "unit_match_boost",
        ):
            value = float(getattr(self, field_name))
            if value < 0:
                raise ValueError(
                    f"{field_name} negatif olamaz."
                )

        if not 0.0 <= self.fuzzy_match_threshold <= 1.0:
            raise ValueError(
                "fuzzy_match_threshold 0 ile 1 arasında olmalıdır."
            )

        if self.minimum_fuzzy_term_length < 2:
            raise ValueError(
                "minimum_fuzzy_term_length en az 2 olmalıdır."
            )

        if self.maximum_matched_terms < 1:
            raise ValueError(
                "maximum_matched_terms en az 1 olmalıdır."
            )

        if not self.allowed_region_types:
            raise ValueError(
                "En az bir allowed_region_types değeri tanımlanmalıdır."
            )


@dataclass(slots=True, frozen=True)
class VisualRetrievalQuery:
    """Tek bir soru için görsel retrieval sorgusu."""

    query_text: str
    report_id: str | UUID

    question_id: int | None = None
    question_key: str | None = None
    group: str | None = None

    category: str | None = None
    topic: str | None = None

    aliases: tuple[str, ...] = ()
    metric_terms: tuple[str, ...] = ()
    expected_units: tuple[str, ...] = ()
    years: tuple[int, ...] = ()

    candidate_pages: tuple[int, ...] = ()
    preferred_region_types: tuple[VisualRegionType, ...] = ()

    require_vision_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.query_text).strip():
            raise ValueError("query_text boş olamaz.")

        if not str(self.report_id).strip():
            raise ValueError("report_id boş olamaz.")

        if self.question_id is not None and self.question_id < 1:
            raise ValueError(
                "question_id pozitif olmalıdır."
            )

        invalid_pages = [
            value
            for value in self.candidate_pages
            if int(value) < 1
        ]
        if invalid_pages:
            raise ValueError(
                f"candidate_pages pozitif olmalıdır: {invalid_pages}"
            )

    @classmethod
    def from_definition(
        cls,
        definition: QuestionDefinition,
        *,
        report_id: str | UUID,
        report_year: int | None = None,
        candidate_pages: Sequence[int] = (),
        require_vision_only: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualRetrievalQuery:
        metric_terms = (
            definition.key.replace("_", " "),
            definition.original_header,
        )

        years = (
            (int(report_year),)
            if report_year is not None
            else ()
        )

        return cls(
            query_text=definition.build_retrieval_query(),
            report_id=report_id,
            question_id=definition.id,
            question_key=definition.key,
            group=definition.group,
            category=definition.esg_category,
            topic=definition.topic,
            aliases=definition.aliases,
            metric_terms=metric_terms,
            expected_units=definition.expected_units,
            years=years,
            candidate_pages=tuple(
                dict.fromkeys(
                    int(value)
                    for value in candidate_pages
                )
            ),
            preferred_region_types=(
                VisualRegionType.TABLE,
                VisualRegionType.METRIC_CARD,
                VisualRegionType.CHART,
                VisualRegionType.INFOGRAPHIC,
            ),
            require_vision_only=require_vision_only,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_analysis(
        cls,
        analysis: QuestionAnalysis,
        *,
        report_id: str | UUID | None = None,
        candidate_pages: Sequence[int] = (),
        require_vision_only: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualRetrievalQuery:
        resolved_report_id = (
            report_id
            or analysis.report_id
        )

        if resolved_report_id is None:
            raise ValueError(
                "QuestionAnalysis için report_id gereklidir."
            )

        metric_terms: list[str] = []
        for metric in analysis.metrics:
            metric_terms.append(metric.name)
            metric_terms.extend(metric.aliases)

        preferred_types: tuple[VisualRegionType, ...]
        if analysis.expects_table_evidence:
            preferred_types = (
                VisualRegionType.TABLE,
                VisualRegionType.METRIC_CARD,
                VisualRegionType.CHART,
                VisualRegionType.INFOGRAPHIC,
            )
        else:
            preferred_types = (
                VisualRegionType.METRIC_CARD,
                VisualRegionType.TABLE,
                VisualRegionType.CHART,
                VisualRegionType.INFOGRAPHIC,
            )

        return cls(
            query_text=(
                analysis.retrieval_query
                or analysis.original_question
            ),
            report_id=resolved_report_id,
            question_id=analysis.question_id,
            question_key=analysis.question_key,
            group=analysis.question_group,
            category=analysis.category,
            topic=analysis.topic,
            aliases=analysis.aliases,
            metric_terms=tuple(metric_terms),
            expected_units=analysis.units,
            years=analysis.years,
            candidate_pages=tuple(
                dict.fromkeys(
                    int(value)
                    for value in candidate_pages
                )
            ),
            preferred_region_types=preferred_types,
            require_vision_only=require_vision_only,
            metadata=dict(metadata or {}),
        )

    @property
    def report_uuid(self) -> UUID:
        """
        String rapor kimliğini ingestion sırasında kullanılan deterministik
        UUID biçimine çevirir.
        """

        if isinstance(self.report_id, UUID):
            return self.report_id

        raw_value = str(self.report_id).strip()

        try:
            return UUID(raw_value)
        except ValueError:
            return uuid5(
                NAMESPACE_URL,
                f"sustainability-report:{raw_value}",
            )


@dataclass(slots=True, frozen=True)
class VisualRetrievalMatch:
    """Tek bir görsel retrieval eşleşmesi."""

    asset: VisualAsset
    score: float
    rank: int

    matched_terms: tuple[str, ...] = ()
    score_components: Mapping[str, float] = field(
        default_factory=dict
    )

    question_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def visual_id(self) -> str:
        return str(self.asset.id)

    @property
    def page_number(self) -> int:
        return self.asset.page_number

    @property
    def image_path(self) -> Path:
        return self.asset.image_path

    @property
    def requires_vision(self) -> bool:
        return self.asset.requires_vision

    @property
    def searchable_text(self) -> str:
        return self.asset.searchable_text

    def with_question_ids(
        self,
        question_ids: Sequence[int],
    ) -> VisualRetrievalMatch:
        merged = tuple(
            sorted(
                {
                    *self.question_ids,
                    *(
                        int(value)
                        for value in question_ids
                        if int(value) > 0
                    ),
                }
            )
        )

        return replace(
            self,
            question_ids=merged,
        )


@dataclass(slots=True, frozen=True)
class VisualRetrievalResult:
    """Tek soru için görsel retrieval sonucu."""

    query: VisualRetrievalQuery
    matches: tuple[VisualRetrievalMatch, ...]

    candidate_count: int
    scored_candidate_count: int
    rejected_candidate_count: int

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def has_matches(self) -> bool:
        return bool(self.matches)

    @property
    def visual_ids(self) -> tuple[str, ...]:
        return tuple(
            item.visual_id
            for item in self.matches
        )

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    item.page_number
                    for item in self.matches
                }
            )
        )


@dataclass(slots=True, frozen=True)
class VisualGroupRetrievalResult:
    """Bir soru grubunun birleştirilmiş görsel retrieval sonucu."""

    group_name: str
    question_results: tuple[VisualRetrievalResult, ...]
    unique_matches: tuple[VisualRetrievalMatch, ...]

    question_visual_ids: Mapping[int, tuple[str, ...]]
    visual_question_ids: Mapping[str, tuple[int, ...]]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def has_matches(self) -> bool:
        return bool(self.unique_matches)

    @property
    def visual_count(self) -> int:
        return len(self.unique_matches)

    @property
    def question_count(self) -> int:
        return len(self.question_results)


class VisualRetriever:
    """
    Görsel varlıkları soru metni, metric alias'ları, topic, kategori, yıl,
    birim ve yakın sayfa sinyallerine göre sıralar.

    Retriever herhangi bir model çağrısı yapmaz. Repository'deki görsel
    metadata'sı ve OCR metnini kullanarak düşük maliyetli aday seçimi yapar.

    Temel kullanım:

        result = retriever.retrieve_definition(
            definition,
            report_id="arçelik_2023",
            report_year=2023,
            require_vision_only=True,
        )

    Kategori kullanımı:

        result = retriever.retrieve_group(
            group,
            report_id="arçelik_2023",
            report_year=2023,
        )

    Grup sonucunda aynı görsel birden fazla soruyla eşleşse bile
    `unique_matches` içinde yalnızca bir kez bulunur.
    """

    _TOKEN_RE = re.compile(
        r"[a-z0-9]+(?:[.,][0-9]+)?|%|‰",
        re.IGNORECASE,
    )

    _STOP_WORDS = frozenset(
        {
            "ve",
            "ile",
            "icin",
            "olan",
            "olarak",
            "nedir",
            "nedir?",
            "ne",
            "kadar",
            "kac",
            "şirketin",
            "sirketin",
            "toplam",
            "the",
            "of",
            "and",
            "for",
            "to",
            "in",
            "is",
            "are",
            "what",
            "how",
            "much",
            "total",
            "company",
        }
    )

    _REGION_TYPE_PRIOR = {
        VisualRegionType.METRIC_CARD: 1.00,
        VisualRegionType.TABLE: 0.96,
        VisualRegionType.CHART: 0.90,
        VisualRegionType.INFOGRAPHIC: 0.80,
        VisualRegionType.FIGURE: 0.58,
        VisualRegionType.UNKNOWN: 0.45,
        VisualRegionType.DECORATIVE: 0.0,
    }

    _PROCESSING_STATUS_PRIOR = {
        VisualProcessingStatus.VISION_REQUIRED: 1.00,
        VisualProcessingStatus.OCR_PARTIAL: 0.95,
        VisualProcessingStatus.IMAGE_EXTRACTED: 0.85,
        VisualProcessingStatus.DETECTED: 0.75,
        VisualProcessingStatus.OCR_COMPLETED: 0.70,
        VisualProcessingStatus.VISION_COMPLETED: 0.55,
        VisualProcessingStatus.COMPLETED: 0.50,
        VisualProcessingStatus.FAILED: 0.20,
        VisualProcessingStatus.SKIPPED: 0.0,
    }

    def __init__(
        self,
        *,
        repository: VisualAssetRepositoryProtocol
        | VisualAssetRepository,
        config: VisualRetrieverConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or VisualRetrieverConfig()

    def retrieve_definition(
        self,
        definition: QuestionDefinition,
        *,
        report_id: str | UUID,
        report_year: int | None = None,
        candidate_pages: Sequence[int] = (),
        require_vision_only: bool = False,
        top_k: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualRetrievalResult:
        query = VisualRetrievalQuery.from_definition(
            definition,
            report_id=report_id,
            report_year=report_year,
            candidate_pages=candidate_pages,
            require_vision_only=require_vision_only,
            metadata=metadata,
        )

        return self.retrieve(
            query,
            top_k=top_k,
        )

    def retrieve_analysis(
        self,
        analysis: QuestionAnalysis,
        *,
        report_id: str | UUID | None = None,
        candidate_pages: Sequence[int] = (),
        require_vision_only: bool = False,
        top_k: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualRetrievalResult:
        query = VisualRetrievalQuery.from_analysis(
            analysis,
            report_id=report_id,
            candidate_pages=candidate_pages,
            require_vision_only=require_vision_only,
            metadata=metadata,
        )

        return self.retrieve(
            query,
            top_k=top_k,
        )

    def retrieve_group(
        self,
        group: QuestionGroup,
        *,
        report_id: str | UUID,
        report_year: int | None = None,
        candidate_pages_by_question: Mapping[
            int,
            Sequence[int],
        ] | None = None,
        require_vision_only: bool = False,
        top_k_per_question: int | None = None,
        maximum_unique_visuals: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VisualGroupRetrievalResult:
        candidate_pages_by_question = (
            candidate_pages_by_question or {}
        )

        question_results: list[
            VisualRetrievalResult
        ] = []

        for definition in group.questions:
            question_result = self.retrieve_definition(
                definition,
                report_id=report_id,
                report_year=report_year,
                candidate_pages=(
                    candidate_pages_by_question.get(
                        definition.id,
                        (),
                    )
                ),
                require_vision_only=require_vision_only,
                top_k=top_k_per_question,
                metadata=metadata,
            )
            question_results.append(
                question_result
            )

        maximum_visuals = (
            maximum_unique_visuals
            if maximum_unique_visuals is not None
            else self.config.top_k
        )

        if maximum_visuals < 1:
            raise ValueError(
                "maximum_unique_visuals en az 1 olmalıdır."
            )

        aggregated: dict[
            str,
            dict[str, Any],
        ] = {}

        question_visual_ids: dict[
            int,
            tuple[str, ...],
        ] = {}

        for result in question_results:
            question_id = result.query.question_id

            if question_id is not None:
                question_visual_ids[question_id] = (
                    result.visual_ids
                )

            for match in result.matches:
                visual_id = match.visual_id

                record = aggregated.setdefault(
                    visual_id,
                    {
                        "asset": match.asset,
                        "scores": [],
                        "matched_terms": set(),
                        "question_ids": set(),
                        "score_components": defaultdict(
                            float
                        ),
                    },
                )

                record["scores"].append(
                    match.score
                )
                record["matched_terms"].update(
                    match.matched_terms
                )

                if question_id is not None:
                    record["question_ids"].add(
                        question_id
                    )

                for name, value in (
                    match.score_components.items()
                ):
                    record[
                        "score_components"
                    ][name] = max(
                        record[
                            "score_components"
                        ][name],
                        float(value),
                    )

        merged_matches: list[
            VisualRetrievalMatch
        ] = []

        for visual_id, record in aggregated.items():
            scores = sorted(
                (
                    float(value)
                    for value in record["scores"]
                ),
                reverse=True,
            )

            best_score = scores[0]
            supporting_questions = len(
                record["question_ids"]
            )

            multi_question_bonus = min(
                0.04
                * max(
                    supporting_questions - 1,
                    0,
                ),
                0.12,
            )

            secondary_score_bonus = min(
                sum(scores[1:]) * 0.08,
                0.08,
            )

            final_score = min(
                best_score
                + multi_question_bonus
                + secondary_score_bonus,
                1.0,
            )

            merged_matches.append(
                VisualRetrievalMatch(
                    asset=record["asset"],
                    score=round(
                        final_score,
                        6,
                    ),
                    rank=0,
                    matched_terms=tuple(
                        sorted(
                            record["matched_terms"],
                            key=lambda item: (
                                -len(item),
                                item,
                            ),
                        )[
                            : self.config
                            .maximum_matched_terms
                        ]
                    ),
                    score_components={
                        **dict(
                            record[
                                "score_components"
                            ]
                        ),
                        "multi_question_bonus": round(
                            multi_question_bonus,
                            6,
                        ),
                        "secondary_score_bonus": round(
                            secondary_score_bonus,
                            6,
                        ),
                    },
                    question_ids=tuple(
                        sorted(
                            record["question_ids"]
                        )
                    ),
                    metadata={
                        "group_name": group.name,
                        "supporting_question_count": (
                            supporting_questions
                        ),
                        "source_score_count": len(
                            scores
                        ),
                    },
                )
            )

        merged_matches.sort(
            key=lambda item: (
                -item.score,
                -len(item.question_ids),
                item.page_number,
                item.visual_id,
            )
        )

        unique_matches = tuple(
            replace(
                match,
                rank=index,
            )
            for index, match in enumerate(
                merged_matches[:maximum_visuals],
                start=1,
            )
        )

        retained_visual_ids = {
            item.visual_id
            for item in unique_matches
        }

        filtered_question_visual_ids = {
            question_id: tuple(
                visual_id
                for visual_id in visual_ids
                if visual_id
                in retained_visual_ids
            )
            for question_id, visual_ids
            in question_visual_ids.items()
        }

        visual_question_ids = {
            item.visual_id: item.question_ids
            for item in unique_matches
        }

        return VisualGroupRetrievalResult(
            group_name=group.name,
            question_results=tuple(
                question_results
            ),
            unique_matches=unique_matches,
            question_visual_ids=(
                filtered_question_visual_ids
            ),
            visual_question_ids=visual_question_ids,
            metadata={
                "report_id": str(report_id),
                "report_uuid": str(
                    self._resolve_report_uuid(
                        report_id
                    )
                ),
                "requested_question_count": (
                    len(group.questions)
                ),
                "unique_candidate_count": len(
                    aggregated
                ),
                "retained_visual_count": len(
                    unique_matches
                ),
                "require_vision_only": (
                    require_vision_only
                ),
            },
        )

    def retrieve(
        self,
        query: VisualRetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> VisualRetrievalResult:
        resolved_top_k = (
            int(top_k)
            if top_k is not None
            else self.config.top_k
        )

        if resolved_top_k < 1:
            raise ValueError(
                "top_k en az 1 olmalıdır."
            )

        candidates = self._load_candidates(
            query
        )

        matches: list[
            VisualRetrievalMatch
        ] = []
        rejected_count = 0

        for asset in candidates:
            rejection_reason = (
                self._candidate_rejection_reason(
                    asset,
                    query,
                )
            )

            if rejection_reason is not None:
                rejected_count += 1
                continue

            score, matched_terms, components = (
                self._score_asset(
                    asset,
                    query,
                )
            )

            if score < self.config.minimum_score:
                rejected_count += 1
                continue

            question_ids = (
                (query.question_id,)
                if query.question_id is not None
                else ()
            )

            matches.append(
                VisualRetrievalMatch(
                    asset=asset,
                    score=round(
                        score,
                        6,
                    ),
                    rank=0,
                    matched_terms=matched_terms,
                    score_components=components,
                    question_ids=question_ids,
                    metadata={
                        "query_topic": query.topic,
                        "query_category": (
                            query.category
                        ),
                        "question_key": (
                            query.question_key
                        ),
                        "report_uuid": str(
                            query.report_uuid
                        ),
                    },
                )
            )

        matches.sort(
            key=lambda item: (
                -item.score,
                (
                    0
                    if item.asset.requires_vision
                    else 1
                ),
                item.page_number,
                item.visual_id,
            )
        )

        ranked_matches = tuple(
            replace(
                match,
                rank=index,
            )
            for index, match in enumerate(
                matches[:resolved_top_k],
                start=1,
            )
        )

        return VisualRetrievalResult(
            query=query,
            matches=ranked_matches,
            candidate_count=len(candidates),
            scored_candidate_count=len(matches),
            rejected_candidate_count=(
                rejected_count
            ),
            metadata={
                "resolved_top_k": resolved_top_k,
                "minimum_score": (
                    self.config.minimum_score
                ),
                "report_uuid": str(
                    query.report_uuid
                ),
                "require_vision_only": (
                    query.require_vision_only
                ),
            },
        )

    def _load_candidates(
        self,
        query: VisualRetrievalQuery,
    ) -> list[VisualAsset]:
        """
        Önce repository'nin kalıcı sorgusunu kullanır. Eski cache veya farklı
        rapor UID biçimleri için bellek içi fallback uygular.
        """

        try:
            candidates = self.repository.find_assets(
                report_uid=query.report_uuid,
                requires_vision=(
                    True
                    if query.require_vision_only
                    else None
                ),
                limit=(
                    self.config
                    .maximum_candidate_count
                ),
            )
        except Exception as exc:
            raise VisualRetrieverError(
                "Görsel repository sorgusu başarısız: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if candidates:
            return candidates

        all_assets_method = getattr(
            self.repository,
            "all_assets",
            None,
        )

        if not callable(all_assets_method):
            return []

        original_report_id = str(
            query.report_id
        ).strip()
        report_uuid = str(
            query.report_uuid
        )

        fallback_candidates: list[
            VisualAsset
        ] = []

        for asset in all_assets_method():
            metadata_report_uid = str(
                asset.metadata.get(
                    "report_uid",
                    "",
                )
            ).strip()

            if (
                str(asset.report_id) != report_uuid
                and metadata_report_uid
                != original_report_id
            ):
                continue

            if (
                query.require_vision_only
                and not asset.requires_vision
            ):
                continue

            fallback_candidates.append(
                asset
            )

            if (
                len(fallback_candidates)
                >= self.config
                .maximum_candidate_count
            ):
                break

        return fallback_candidates

    def _candidate_rejection_reason(
        self,
        asset: VisualAsset,
        query: VisualRetrievalQuery,
    ) -> str | None:
        if (
            asset.region_type
            not in self.config.allowed_region_types
        ):
            return "region_type_not_allowed"

        if (
            self.config.exclude_decorative_regions
            and asset.region_type
            == VisualRegionType.DECORATIVE
        ):
            return "decorative"

        if (
            not self.config.include_completed_assets
            and asset.processing_status
            in {
                VisualProcessingStatus.COMPLETED,
                VisualProcessingStatus
                .VISION_COMPLETED,
            }
        ):
            return "already_completed"

        if (
            query.require_vision_only
            and not asset.requires_vision
        ):
            return "vision_not_required"

        if self.config.require_existing_image:
            try:
                if not Path(
                    asset.image_path
                ).expanduser().is_file():
                    return "image_not_found"
            except (OSError, TypeError):
                return "image_not_found"

        return None

    def _score_asset(
        self,
        asset: VisualAsset,
        query: VisualRetrievalQuery,
    ) -> tuple[
        float,
        tuple[str, ...],
        Mapping[str, float],
    ]:
        fields = self._asset_text_fields(
            asset
        )

        query_terms = self._query_terms(
            query
        )

        matched_terms: set[str] = set()

        phrase_score = self._phrase_score(
            query_terms=query_terms,
            fields=fields,
            matched_terms=matched_terms,
        )

        token_score = self._token_overlap_score(
            query_terms=query_terms,
            fields=fields,
            matched_terms=matched_terms,
        )

        fuzzy_score = self._fuzzy_score(
            query_terms=query_terms,
            fields=fields,
            matched_terms=matched_terms,
        )

        metadata_score, metadata_components = (
            self._metadata_score(
                asset,
                query,
            )
        )

        weighted_core = (
            phrase_score
            * self.config.phrase_score_weight
            + token_score
            * self.config.token_score_weight
            + fuzzy_score
            * self.config.fuzzy_score_weight
            + metadata_score
            * self.config.metadata_score_weight
        )

        core_weight_total = (
            self.config.phrase_score_weight
            + self.config.token_score_weight
            + self.config.fuzzy_score_weight
            + self.config.metadata_score_weight
        )

        score = (
            weighted_core
            / max(
                core_weight_total,
                1e-9,
            )
        )

        if (
            self.config.prefer_vision_required
            and asset.requires_vision
        ):
            score += (
                self.config
                .vision_required_boost
            )

        score = max(
            0.0,
            min(score, 1.0),
        )

        components = {
            "phrase_score": round(
                phrase_score,
                6,
            ),
            "token_score": round(
                token_score,
                6,
            ),
            "fuzzy_score": round(
                fuzzy_score,
                6,
            ),
            "metadata_score": round(
                metadata_score,
                6,
            ),
            "vision_required_boost": (
                self.config
                .vision_required_boost
                if (
                    self.config
                    .prefer_vision_required
                    and asset.requires_vision
                )
                else 0.0
            ),
            **metadata_components,
        }

        return (
            score,
            tuple(
                sorted(
                    matched_terms,
                    key=lambda item: (
                        -len(item),
                        item,
                    ),
                )[
                    : self.config
                    .maximum_matched_terms
                ]
            ),
            components,
        )

    def _asset_text_fields(
        self,
        asset: VisualAsset,
    ) -> tuple[
        tuple[str, float],
        ...,
    ]:
        fields: list[
            tuple[str, float]
        ] = []

        def add(
            value: str | None,
            weight: float,
        ) -> None:
            normalized = self._normalize_text(
                value or ""
            )
            if normalized:
                fields.append(
                    (
                        normalized,
                        weight,
                    )
                )

        add(
            asset.caption,
            self.config.caption_weight,
        )
        add(
            asset.section_title,
            self.config.section_weight,
        )
        add(
            asset.subsection_title,
            self.config.section_weight,
        )
        add(
            asset.nearby_text,
            self.config.nearby_text_weight,
        )
        add(
            asset.ocr_text,
            self.config.ocr_text_weight,
        )
        add(
            " ".join(asset.keywords),
            self.config.keyword_weight,
        )

        metadata = asset.metadata or {}

        for key in (
            "source_table_id",
            "table_id",
            "caption_text",
            "text_preview",
            "section_title",
            "topic",
            "category",
        ):
            value = metadata.get(key)
            if value:
                add(
                    str(value),
                    0.75,
                )

        return tuple(fields)

    def _query_terms(
        self,
        query: VisualRetrievalQuery,
    ) -> tuple[
        tuple[str, float],
        ...,
    ]:
        weighted_terms: list[
            tuple[str, float]
        ] = []

        def add(
            value: Any,
            weight: float,
        ) -> None:
            normalized = self._normalize_text(
                str(value or "")
            )

            if not normalized:
                return

            weighted_terms.append(
                (
                    normalized,
                    weight,
                )
            )

        add(
            query.query_text,
            1.00,
        )
        add(
            query.question_key
            .replace("_", " ")
            if query.question_key
            else "",
            1.45,
        )
        add(
            query.topic,
            1.45,
        )
        add(
            query.group,
            1.10,
        )
        add(
            query.category,
            0.55,
        )

        for value in query.metric_terms:
            add(
                value,
                1.50,
            )

        for value in query.aliases:
            add(
                value,
                1.30,
            )

        for value in query.expected_units:
            add(
                value,
                0.72,
            )

        for value in query.years:
            add(
                str(value),
                0.70,
            )

        deduplicated: dict[
            str,
            float,
        ] = {}

        for term, weight in weighted_terms:
            deduplicated[term] = max(
                deduplicated.get(
                    term,
                    0.0,
                ),
                weight,
            )

        return tuple(
            sorted(
                deduplicated.items(),
                key=lambda item: (
                    -item[1],
                    -len(item[0]),
                    item[0],
                ),
            )
        )

    def _phrase_score(
        self,
        *,
        query_terms: Sequence[
            tuple[str, float]
        ],
        fields: Sequence[
            tuple[str, float]
        ],
        matched_terms: set[str],
    ) -> float:
        numerator = 0.0
        denominator = 0.0

        for term, term_weight in query_terms:
            if len(term) < 3:
                continue

            denominator += term_weight

            best_field_match = 0.0

            for field_text, field_weight in fields:
                if term in field_text:
                    best_field_match = max(
                        best_field_match,
                        min(
                            field_weight / 1.45,
                            1.0,
                        ),
                    )

            if best_field_match > 0:
                matched_terms.add(term)
                numerator += (
                    term_weight
                    * best_field_match
                )

        if denominator <= 0:
            return 0.0

        return min(
            numerator / denominator,
            1.0,
        )

    def _token_overlap_score(
        self,
        *,
        query_terms: Sequence[
            tuple[str, float]
        ],
        fields: Sequence[
            tuple[str, float]
        ],
        matched_terms: set[str],
    ) -> float:
        field_tokens: dict[
            str,
            float,
        ] = {}

        for field_text, field_weight in fields:
            for token in self._tokens(
                field_text
            ):
                field_tokens[token] = max(
                    field_tokens.get(
                        token,
                        0.0,
                    ),
                    field_weight,
                )

        query_tokens: dict[
            str,
            float,
        ] = {}

        for term, term_weight in query_terms:
            for token in self._tokens(
                term
            ):
                query_tokens[token] = max(
                    query_tokens.get(
                        token,
                        0.0,
                    ),
                    term_weight,
                )

        if not query_tokens:
            return 0.0

        numerator = 0.0
        denominator = sum(
            query_tokens.values()
        )

        for token, token_weight in (
            query_tokens.items()
        ):
            field_weight = field_tokens.get(
                token
            )

            if field_weight is None:
                continue

            matched_terms.add(token)

            numerator += (
                token_weight
                * min(
                    field_weight / 1.45,
                    1.0,
                )
            )

        return min(
            numerator
            / max(denominator, 1e-9),
            1.0,
        )

    def _fuzzy_score(
        self,
        *,
        query_terms: Sequence[
            tuple[str, float]
        ],
        fields: Sequence[
            tuple[str, float]
        ],
        matched_terms: set[str],
    ) -> float:
        query_tokens = {
            token
            for term, _ in query_terms
            for token in self._tokens(term)
            if len(token)
            >= self.config
            .minimum_fuzzy_term_length
        }

        field_tokens = {
            token
            for field, _ in fields
            for token in self._tokens(field)
            if len(token)
            >= self.config
            .minimum_fuzzy_term_length
        }

        if (
            not query_tokens
            or not field_tokens
        ):
            return 0.0

        matched_scores: list[
            float
        ] = []

        for query_token in query_tokens:
            best_ratio = 0.0
            best_field_token: str | None = None

            for field_token in field_tokens:
                if query_token == field_token:
                    best_ratio = 1.0
                    best_field_token = (
                        field_token
                    )
                    break

                if abs(
                    len(query_token)
                    - len(field_token)
                ) > 3:
                    continue

                ratio = SequenceMatcher(
                    None,
                    query_token,
                    field_token,
                ).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_field_token = (
                        field_token
                    )

            if (
                best_ratio
                >= self.config
                .fuzzy_match_threshold
            ):
                matched_scores.append(
                    best_ratio
                )
                matched_terms.add(
                    query_token
                )

                if best_field_token:
                    matched_terms.add(
                        best_field_token
                    )

        if not matched_scores:
            return 0.0

        return min(
            sum(matched_scores)
            / len(query_tokens),
            1.0,
        )

    def _metadata_score(
        self,
        asset: VisualAsset,
        query: VisualRetrievalQuery,
    ) -> tuple[
        float,
        Mapping[str, float],
    ]:
        components: dict[
            str,
            float,
        ] = {}

        score = 0.0
        maximum = 0.0

        maximum += (
            self.config.exact_topic_boost
        )
        if (
            query.topic
            and asset.topic
            and self._normalize_text(
                query.topic
            )
            == self._normalize_text(
                asset.topic
            )
        ):
            value = (
                self.config
                .exact_topic_boost
            )
            score += value
            components[
                "exact_topic_boost"
            ] = value
        else:
            components[
                "exact_topic_boost"
            ] = 0.0

        maximum += (
            self.config
            .exact_category_boost
        )
        if (
            query.category
            and asset.category
            and self._normalize_text(
                query.category
            )
            == self._normalize_text(
                asset.category
            )
        ):
            value = (
                self.config
                .exact_category_boost
            )
            score += value
            components[
                "exact_category_boost"
            ] = value
        else:
            components[
                "exact_category_boost"
            ] = 0.0

        maximum += (
            self.config
            .candidate_page_boost
        )
        if (
            query.candidate_pages
            and asset.page_number
            in query.candidate_pages
        ):
            value = (
                self.config
                .candidate_page_boost
            )
            score += value
            components[
                "candidate_page_boost"
            ] = value
        else:
            components[
                "candidate_page_boost"
            ] = 0.0

        normalized_asset_text = (
            self._normalize_text(
                asset.searchable_text
            )
        )

        maximum += (
            self.config.year_match_boost
        )
        year_match = any(
            str(year)
            in normalized_asset_text
            for year in query.years
        )
        if year_match:
            value = (
                self.config
                .year_match_boost
            )
            score += value
            components[
                "year_match_boost"
            ] = value
        else:
            components[
                "year_match_boost"
            ] = 0.0

        maximum += (
            self.config.unit_match_boost
        )
        unit_match = any(
            self._normalize_text(unit)
            in normalized_asset_text
            for unit in query.expected_units
            if self._normalize_text(unit)
        )
        if unit_match:
            value = (
                self.config
                .unit_match_boost
            )
            score += value
            components[
                "unit_match_boost"
            ] = value
        else:
            components[
                "unit_match_boost"
            ] = 0.0

        region_prior = (
            self._REGION_TYPE_PRIOR.get(
                asset.region_type,
                0.35,
            )
        )

        if (
            query.preferred_region_types
            and asset.region_type
            in query.preferred_region_types
        ):
            preferred_index = (
                query.preferred_region_types
                .index(asset.region_type)
            )

            preferred_factor = max(
                1.0
                - preferred_index * 0.08,
                0.68,
            )
            region_prior *= (
                preferred_factor
            )

        processing_prior = (
            self._PROCESSING_STATUS_PRIOR.get(
                asset.processing_status,
                0.40,
            )
        )

        confidence_values = [
            value
            for value in (
                asset.detection_confidence,
                asset.ocr_confidence,
            )
            if value is not None
        ]

        confidence_prior = (
            sum(confidence_values)
            / len(confidence_values)
            if confidence_values
            else 0.50
        )

        prior_score = (
            region_prior * 0.45
            + processing_prior * 0.35
            + confidence_prior * 0.20
        )

        maximum += 0.16
        score += prior_score * 0.16

        components[
            "region_prior"
        ] = round(
            region_prior,
            6,
        )
        components[
            "processing_prior"
        ] = round(
            processing_prior,
            6,
        )
        components[
            "confidence_prior"
        ] = round(
            confidence_prior,
            6,
        )
        components[
            "prior_contribution"
        ] = round(
            prior_score * 0.16,
            6,
        )

        if maximum <= 0:
            return 0.0, components

        return (
            min(
                score / maximum,
                1.0,
            ),
            components,
        )

    @classmethod
    def _tokens(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        tokens = []

        for token in cls._TOKEN_RE.findall(
            cls._normalize_text(text)
        ):
            normalized = token.strip()

            if (
                not normalized
                or normalized
                in cls._STOP_WORDS
            ):
                continue

            tokens.append(normalized)

        return tuple(
            dict.fromkeys(tokens)
        )

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        normalized = unicodedata.normalize(
            "NFKD",
            str(value).strip().casefold(),
        )

        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(
                character
            )
        )

        translation = str.maketrans(
            {
                "ı": "i",
                "ş": "s",
                "ğ": "g",
                "ü": "u",
                "ö": "o",
                "ç": "c",
                "²": "2",
                "³": "3",
                "₂": "2",
                "₃": "3",
                "₂": "2",
                "₺": " try ",
                "€": " eur ",
                "$": " usd ",
                "£": " gbp ",
                "₂": "2",
            }
        )

        normalized = normalized.translate(
            translation
        )

        normalized = re.sub(
            r"[^a-z0-9%‰.,]+",
            " ",
            normalized,
        )

        return re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

    @staticmethod
    def _resolve_report_uuid(
        report_id: str | UUID,
    ) -> UUID:
        if isinstance(report_id, UUID):
            return report_id

        raw_value = str(report_id).strip()

        try:
            return UUID(raw_value)
        except ValueError:
            return uuid5(
                NAMESPACE_URL,
                f"sustainability-report:{raw_value}",
            )


__all__ = [
    "VisualGroupRetrievalResult",
    "VisualRetrievalMatch",
    "VisualRetrievalQuery",
    "VisualRetrievalResult",
    "VisualRetriever",
    "VisualRetrieverConfig",
    "VisualRetrieverError",
]
