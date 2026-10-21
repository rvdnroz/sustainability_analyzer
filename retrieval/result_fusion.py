from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


class RetrievalSource(StrEnum):
    TABLE = "table"
    BM25 = "bm25"
    VECTOR = "vector"
    STANDARD = "standard"
    UNKNOWN = "unknown"


class FusionMethod(StrEnum):
    RRF = "rrf"
    WEIGHTED_SCORE = "weighted_score"
    HYBRID = "hybrid"


@dataclass(slots=True, frozen=True)
class FusionInput:
    document_id: str
    source: RetrievalSource
    rank: int
    score: float
    payload: Any
    report_id: str | None = None
    page_number: int | None = None
    row_index: int | None = None
    matched_terms: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FusedRetrievalResult:
    document_id: str
    score: float
    rank: int
    payload: Any
    sources: tuple[RetrievalSource, ...]
    source_scores: Mapping[str, float]
    source_ranks: Mapping[str, int]
    report_id: str | None = None
    page_number: int | None = None
    row_index: int | None = None
    matched_terms: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ResultFusionConfig:
    method: FusionMethod = FusionMethod.HYBRID
    top_k: int = 30
    rrf_k: int = 60

    rrf_weight: float = 0.6
    normalized_score_weight: float = 0.4

    table_weight: float = 1.25
    bm25_weight: float = 1.0
    vector_weight: float = 1.1
    standard_weight: float = 1.15

    multi_source_bonus: float = 0.08
    exact_id_match_bonus: float = 0.03

    table_rank_bonus: float = 0.10
    table_row_bonus: float = 0.05
    minimum_table_results: int = 2
    maximum_results_per_table: int = 2

    minimum_score: float = 0.0
    normalize_scores_per_source: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k pozitif olmalıdır.")

        if self.rrf_weight < 0:
            raise ValueError("rrf_weight negatif olamaz.")

        if self.normalized_score_weight < 0:
            raise ValueError(
                "normalized_score_weight negatif olamaz.",
            )

        if math.isclose(
            self.rrf_weight + self.normalized_score_weight,
            0.0,
        ):
            raise ValueError(
                "Fusion ağırlıklarının toplamı sıfır olamaz.",
            )

        source_weights = (
            self.table_weight,
            self.bm25_weight,
            self.vector_weight,
            self.standard_weight,
        )

        if any(weight < 0 for weight in source_weights):
            raise ValueError(
                "Kaynak ağırlıkları negatif olamaz.",
            )

        if self.multi_source_bonus < 0:
            raise ValueError(
                "multi_source_bonus negatif olamaz.",
            )

        if self.exact_id_match_bonus < 0:
            raise ValueError(
                "exact_id_match_bonus negatif olamaz.",
            )

        if self.table_rank_bonus < 0:
            raise ValueError(
                "table_rank_bonus negatif olamaz.",
            )

        if self.table_row_bonus < 0:
            raise ValueError(
                "table_row_bonus negatif olamaz.",
            )

        if self.minimum_table_results < 0:
            raise ValueError(
                "minimum_table_results negatif olamaz.",
            )

        if self.maximum_results_per_table <= 0:
            raise ValueError(
                "maximum_results_per_table pozitif olmalıdır.",
            )

        if self.minimum_score < 0:
            raise ValueError("minimum_score negatif olamaz.")


class ResultFusionError(RuntimeError):
    """Retrieval sonuçlarını birleştirme sırasında oluşan temel hata."""


class ResultFusion:
    """
    Tablo, BM25, embedding ve standart retrieval sonuçlarını tek sıralamada
    birleştirir.

    Desteklenen yöntemler:
    - Reciprocal Rank Fusion
    - Normalize edilmiş ağırlıklı skor
    - RRF ve ağırlıklı skorun birlikte kullanıldığı hibrit fusion

    Aynı chunk, tablo veya kanıt birden fazla retrieval kaynağında bulunursa
    sonuçlar document_id üzerinden birleştirilir.
    """

    def __init__(
        self,
        *,
        config: ResultFusionConfig | None = None,
    ) -> None:
        self.config = config or ResultFusionConfig()

    def fuse(
        self,
        *,
        table_results: Iterable[Any] = (),
        bm25_results: Iterable[Any] = (),
        vector_results: Iterable[Any] = (),
        standard_results: Iterable[Any] = (),
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[FusedRetrievalResult]:
        grouped_inputs = {
            RetrievalSource.TABLE: self._prepare_source_results(
                table_results,
                source=RetrievalSource.TABLE,
            ),
            RetrievalSource.BM25: self._prepare_source_results(
                bm25_results,
                source=RetrievalSource.BM25,
            ),
            RetrievalSource.VECTOR: self._prepare_source_results(
                vector_results,
                source=RetrievalSource.VECTOR,
            ),
            RetrievalSource.STANDARD: self._prepare_source_results(
                standard_results,
                source=RetrievalSource.STANDARD,
            ),
        }

        all_inputs = [
            result
            for results in grouped_inputs.values()
            for result in results
        ]

        if not all_inputs:
            return []

        normalized_scores = self._normalize_source_scores(
            grouped_inputs,
        )

        grouped_by_document: dict[str, list[FusionInput]] = {}

        for result in all_inputs:
            grouped_by_document.setdefault(
                result.document_id,
                [],
            ).append(result)

        fused_results: list[FusedRetrievalResult] = []

        for document_id, document_results in grouped_by_document.items():
            fused_result = self._fuse_document(
                document_id=document_id,
                inputs=document_results,
                normalized_scores=normalized_scores,
            )
            fused_results.append(fused_result)

        score_threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )

        if score_threshold < 0:
            raise ValueError("minimum_score negatif olamaz.")

        fused_results = [
            result
            for result in fused_results
            if result.score >= score_threshold
        ]

        fused_results.sort(
            key=lambda result: (
                -result.score,
                result.page_number
                if result.page_number is not None
                else 10**9,
                result.document_id,
            ),
        )

        result_limit = top_k or self.config.top_k

        if result_limit <= 0:
            return []

        selected_results = self._select_final_results(
            fused_results,
            result_limit=result_limit,
        )

        return [
            FusedRetrievalResult(
                document_id=result.document_id,
                score=result.score,
                rank=index + 1,
                payload=result.payload,
                sources=result.sources,
                source_scores=result.source_scores,
                source_ranks=result.source_ranks,
                report_id=result.report_id,
                page_number=result.page_number,
                row_index=result.row_index,
                matched_terms=result.matched_terms,
                metadata=result.metadata,
            )
            for index, result in enumerate(selected_results)
        ]

    def _select_final_results(
        self,
        fused_results: Sequence[FusedRetrievalResult],
        *,
        result_limit: int,
    ) -> list[FusedRetrievalResult]:
        """
        Fusion sonucunda güçlü tablo kanıtlarının kaybolmasını engeller.

        Kurallar:
        - İlk N sonuç içinde en az minimum_table_results tablo sonucu tutulur.
        - Aynı temel tablodan en fazla maximum_results_per_table sonuç alınır.
        - Satır sonucu, aynı tablonun genel sonucuna göre tercih edilir.
        """

        if result_limit <= 0:
            return []

        table_candidates = [
            result
            for result in fused_results
            if RetrievalSource.TABLE in result.sources
        ]

        table_candidates.sort(
            key=lambda result: (
                result.row_index is None,
                -result.score,
                result.page_number
                if result.page_number is not None
                else 10**9,
            ),
        )

        selected: list[FusedRetrievalResult] = []
        selected_ids: set[str] = set()
        table_counts: dict[str, int] = {}

        required_tables = min(
            self.config.minimum_table_results,
            result_limit,
            len(table_candidates),
        )

        for result in table_candidates:
            if len(selected) >= required_tables:
                break

            base_table_id = self._base_table_id(result.document_id)
            current_count = table_counts.get(base_table_id, 0)

            if current_count >= self.config.maximum_results_per_table:
                continue

            selected.append(result)
            selected_ids.add(result.document_id)
            table_counts[base_table_id] = current_count + 1

        for result in fused_results:
            if len(selected) >= result_limit:
                break

            if result.document_id in selected_ids:
                continue

            if RetrievalSource.TABLE in result.sources:
                base_table_id = self._base_table_id(result.document_id)
                current_count = table_counts.get(base_table_id, 0)

                if current_count >= self.config.maximum_results_per_table:
                    continue

                table_counts[base_table_id] = current_count + 1

            selected.append(result)
            selected_ids.add(result.document_id)

        selected.sort(
            key=lambda result: (
                -result.score,
                result.page_number
                if result.page_number is not None
                else 10**9,
                result.document_id,
            ),
        )

        return selected[:result_limit]

    @staticmethod
    def _base_table_id(document_id: str) -> str:
        return document_id.split("::row:", 1)[0]

    def fuse_lists(
        self,
        result_lists: Mapping[
            RetrievalSource | str,
            Iterable[Any],
        ],
        *,
        top_k: int | None = None,
    ) -> list[FusedRetrievalResult]:
        normalized_lists: dict[
            RetrievalSource,
            Iterable[Any],
        ] = {}

        for source, results in result_lists.items():
            normalized_source = self._normalize_source(source)
            normalized_lists[normalized_source] = results

        return self.fuse(
            table_results=normalized_lists.get(
                RetrievalSource.TABLE,
                (),
            ),
            bm25_results=normalized_lists.get(
                RetrievalSource.BM25,
                (),
            ),
            vector_results=normalized_lists.get(
                RetrievalSource.VECTOR,
                (),
            ),
            standard_results=normalized_lists.get(
                RetrievalSource.STANDARD,
                (),
            ),
            top_k=top_k,
        )

    def reciprocal_rank_score(
        self,
        rank: int,
        *,
        source: RetrievalSource | str,
    ) -> float:
        if rank <= 0:
            raise ValueError("rank pozitif olmalıdır.")

        normalized_source = self._normalize_source(source)

        return (
            self._source_weight(normalized_source)
            / (self.config.rrf_k + rank)
        )

    def _fuse_document(
        self,
        *,
        document_id: str,
        inputs: Sequence[FusionInput],
        normalized_scores: Mapping[
            tuple[RetrievalSource, str, int],
            float,
        ],
    ) -> FusedRetrievalResult:
        rrf_score = 0.0
        weighted_score = 0.0

        source_scores: dict[str, float] = {}
        source_ranks: dict[str, int] = {}
        source_set: set[RetrievalSource] = set()
        matched_terms: set[str] = set()

        for item in inputs:
            source_weight = self._source_weight(item.source)

            rrf_component = (
                source_weight
                / (self.config.rrf_k + item.rank)
            )
            rrf_score += rrf_component

            normalized_score = normalized_scores.get(
                (
                    item.source,
                    item.document_id,
                    item.rank,
                ),
                0.0,
            )
            weighted_score += (
                normalized_score
                * source_weight
            )

            source_name = item.source.value

            previous_score = source_scores.get(source_name)

            if (
                previous_score is None
                or item.score > previous_score
            ):
                source_scores[source_name] = item.score

            previous_rank = source_ranks.get(source_name)

            if (
                previous_rank is None
                or item.rank < previous_rank
            ):
                source_ranks[source_name] = item.rank

            source_set.add(item.source)
            matched_terms.update(item.matched_terms)

        if source_set:
            total_source_weight = sum(
                self._source_weight(source)
                for source in source_set
            )
            weighted_score = (
                weighted_score / total_source_weight
                if total_source_weight > 0
                else 0.0
            )

        normalized_rrf = self._normalize_rrf_score(
            rrf_score,
            source_set,
        )

        final_score = self._calculate_final_score(
            rrf_score=normalized_rrf,
            weighted_score=weighted_score,
        )

        if len(source_set) > 1:
            final_score += (
                self.config.multi_source_bonus
                * min(len(source_set) - 1, 3)
            )

        if len(inputs) > len(source_set):
            final_score += self.config.exact_id_match_bonus

        table_inputs = [
            item
            for item in inputs
            if item.source == RetrievalSource.TABLE
        ]

        if table_inputs:
            best_table_rank = min(item.rank for item in table_inputs)
            final_score += (
                self.config.table_rank_bonus
                / max(best_table_rank, 1)
            )

            if any(item.row_index is not None for item in table_inputs):
                final_score += self.config.table_row_bonus

        final_score = max(
            0.0,
            min(final_score, 1.0),
        )

        representative = self._select_representative(inputs)

        sources = tuple(
            sorted(
                source_set,
                key=self._source_sort_key,
            ),
        )

        return FusedRetrievalResult(
            document_id=document_id,
            score=round(final_score, 6),
            rank=0,
            payload=representative.payload,
            sources=sources,
            source_scores={
                key: round(value, 6)
                for key, value in source_scores.items()
            },
            source_ranks=dict(source_ranks),
            report_id=representative.report_id,
            page_number=representative.page_number,
            row_index=representative.row_index,
            matched_terms=tuple(sorted(matched_terms)),
            metadata={
                **dict(representative.metadata),
                "rrf_score": round(normalized_rrf, 6),
                "weighted_score": round(
                    weighted_score,
                    6,
                ),
                "source_count": len(source_set),
                "input_count": len(inputs),
                "representative_source": (
                    representative.source.value
                ),
            },
        )

    def _calculate_final_score(
        self,
        *,
        rrf_score: float,
        weighted_score: float,
    ) -> float:
        if self.config.method == FusionMethod.RRF:
            return rrf_score

        if self.config.method == FusionMethod.WEIGHTED_SCORE:
            return weighted_score

        if self.config.method == FusionMethod.HYBRID:
            total_weight = (
                self.config.rrf_weight
                + self.config.normalized_score_weight
            )

            return (
                rrf_score * self.config.rrf_weight
                + weighted_score
                * self.config.normalized_score_weight
            ) / total_weight

        raise ResultFusionError(
            f"Desteklenmeyen fusion yöntemi: {self.config.method}",
        )

    def _prepare_source_results(
        self,
        results: Iterable[Any],
        *,
        source: RetrievalSource,
    ) -> list[FusionInput]:
        prepared: list[FusionInput] = []

        for position, result in enumerate(results, start=1):
            try:
                prepared.append(
                    self._to_fusion_input(
                        result,
                        source=source,
                        fallback_rank=position,
                    ),
                )
            except ResultFusionError:
                continue

        prepared.sort(
            key=lambda item: (
                item.rank,
                -item.score,
                item.document_id,
            ),
        )

        return prepared

    def _to_fusion_input(
        self,
        result: Any,
        *,
        source: RetrievalSource,
        fallback_rank: int,
    ) -> FusionInput:
        document_id = self._extract_document_id(
            result,
            source=source,
        )

        score = self._extract_score(result)
        rank = self._extract_rank(
            result,
            fallback_rank=fallback_rank,
        )

        payload = self._extract_payload(result)

        report_id = self._extract_value(
            result,
            ("report_id",),
        )

        if report_id is None:
            report_id = self._extract_nested_value(
                payload,
                ("report_id",),
            )

        page_number = self._extract_page_number(
            result,
            payload,
        )

        row_index = self._optional_int(
            self._extract_value(
                result,
                ("row_index",),
            ),
        )

        matched_terms = self._extract_string_sequence(
            self._extract_value(
                result,
                (
                    "matched_terms",
                    "terms",
                    "matched_keywords",
                ),
            ),
        )

        metadata = self._extract_metadata(result)

        return FusionInput(
            document_id=document_id,
            source=source,
            rank=rank,
            score=score,
            payload=payload,
            report_id=self._optional_string(report_id),
            page_number=page_number,
            row_index=row_index,
            matched_terms=matched_terms,
            metadata=metadata,
        )

    def _normalize_source_scores(
        self,
        grouped_inputs: Mapping[
            RetrievalSource,
            Sequence[FusionInput],
        ],
    ) -> dict[
        tuple[RetrievalSource, str, int],
        float,
    ]:
        normalized: dict[
            tuple[RetrievalSource, str, int],
            float,
        ] = {}

        for source, inputs in grouped_inputs.items():
            if not inputs:
                continue

            scores = [item.score for item in inputs]

            if not self.config.normalize_scores_per_source:
                for item in inputs:
                    normalized[
                        (
                            source,
                            item.document_id,
                            item.rank,
                        )
                    ] = max(0.0, min(item.score, 1.0))

                continue

            minimum = min(scores)
            maximum = max(scores)

            if math.isclose(minimum, maximum):
                for item in inputs:
                    normalized[
                        (
                            source,
                            item.document_id,
                            item.rank,
                        )
                    ] = 1.0 if maximum > 0 else 0.0

                continue

            score_range = maximum - minimum

            for item in inputs:
                normalized_score = (
                    item.score - minimum
                ) / score_range

                normalized[
                    (
                        source,
                        item.document_id,
                        item.rank,
                    )
                ] = normalized_score

        return normalized

    def _normalize_rrf_score(
        self,
        score: float,
        sources: set[RetrievalSource],
    ) -> float:
        if not sources:
            return 0.0

        maximum_possible = sum(
            self._source_weight(source)
            / (self.config.rrf_k + 1)
            for source in sources
        )

        if maximum_possible <= 0:
            return 0.0

        return min(
            score / maximum_possible,
            1.0,
        )

    def _select_representative(
        self,
        inputs: Sequence[FusionInput],
    ) -> FusionInput:
        if not inputs:
            raise ResultFusionError(
                "Temsilci seçiminde sonuç listesi boş olamaz.",
            )

        table_inputs = [
            item
            for item in inputs
            if item.source == RetrievalSource.TABLE
        ]

        candidates = table_inputs or list(inputs)

        return min(
            candidates,
            key=lambda item: (
                item.rank,
                item.row_index is None,
                -self._source_weight(item.source),
                -item.score,
            ),
        )

    def _source_weight(
        self,
        source: RetrievalSource,
    ) -> float:
        weights = {
            RetrievalSource.TABLE: self.config.table_weight,
            RetrievalSource.BM25: self.config.bm25_weight,
            RetrievalSource.VECTOR: self.config.vector_weight,
            RetrievalSource.STANDARD: (
                self.config.standard_weight
            ),
            RetrievalSource.UNKNOWN: 1.0,
        }

        return weights[source]

    @staticmethod
    def _source_sort_key(
        source: RetrievalSource,
    ) -> int:
        order = {
            RetrievalSource.TABLE: 0,
            RetrievalSource.BM25: 1,
            RetrievalSource.VECTOR: 2,
            RetrievalSource.STANDARD: 3,
            RetrievalSource.UNKNOWN: 99,
        }

        return order[source]

    def _extract_document_id(
        self,
        result: Any,
        *,
        source: RetrievalSource,
    ) -> str:
        direct_candidates = (
            "document_id",
            "chunk_id",
            "table_id",
            "evidence_id",
            "id",
            "identifier",
        )

        value = self._extract_value(
            result,
            direct_candidates,
        )

        if value is None:
            payload = self._extract_payload(result)
            value = self._extract_nested_value(
                payload,
                direct_candidates,
            )

        if value is None:
            metadata = self._extract_metadata(result)
            value = next(
                (
                    metadata.get(key)
                    for key in direct_candidates
                    if metadata.get(key) is not None
                ),
                None,
            )

        if value is None or not str(value).strip():
            raise ResultFusionError(
                f"{source.value} sonucunda belge kimliği bulunamadı.",
            )

        document_id = str(value).strip()

        row_index = self._optional_int(
            self._extract_value(
                result,
                ("row_index",),
            ),
        )

        if (
            source == RetrievalSource.TABLE
            and row_index is not None
        ):
            return f"{document_id}::row:{row_index}"

        return document_id

    @staticmethod
    def _extract_score(result: Any) -> float:
        value = ResultFusion._extract_value(
            result,
            (
                "score",
                "similarity",
                "retrieval_score",
                "confidence",
            ),
        )

        if value is None:
            return 0.0

        try:
            numeric_score = float(value)
        except (TypeError, ValueError) as exc:
            raise ResultFusionError(
                "Retrieval skoru sayısal olmalıdır.",
            ) from exc

        if math.isnan(numeric_score) or math.isinf(numeric_score):
            raise ResultFusionError(
                "Retrieval skoru NaN veya sonsuz olamaz.",
            )

        return numeric_score

    @staticmethod
    def _extract_rank(
        result: Any,
        *,
        fallback_rank: int,
    ) -> int:
        value = ResultFusion._extract_value(
            result,
            ("rank", "position"),
        )

        if value is None:
            return fallback_rank

        try:
            rank = int(value)
        except (TypeError, ValueError):
            return fallback_rank

        return rank if rank > 0 else fallback_rank

    @staticmethod
    def _extract_payload(result: Any) -> Any:
        for name in (
            "document",
            "table",
            "payload",
            "item",
            "record",
            "disclosure",
        ):
            value = ResultFusion._extract_value(
                result,
                (name,),
            )

            if value is not None:
                return value

        return result

    @staticmethod
    def _extract_page_number(
        result: Any,
        payload: Any,
    ) -> int | None:
        value = ResultFusion._extract_value(
            result,
            (
                "page_number",
                "source_page",
                "page",
            ),
        )

        if value is None:
            value = ResultFusion._extract_nested_value(
                payload,
                (
                    "page_number",
                    "source_page",
                    "page",
                ),
            )

        return ResultFusion._optional_int(value)

    @staticmethod
    def _extract_metadata(result: Any) -> dict[str, Any]:
        metadata = ResultFusion._extract_value(
            result,
            ("metadata",),
        )

        if metadata is None:
            return {}

        if isinstance(metadata, Mapping):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(
                metadata.model_dump(exclude_none=True),
            )

        if hasattr(metadata, "dict"):
            return dict(
                metadata.dict(exclude_none=True),
            )

        return {}

    @staticmethod
    def _extract_value(
        source: Any,
        names: Sequence[str],
    ) -> Any:
        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
                value = getattr(source, name, None)

            if value is not None and not callable(value):
                return getattr(value, "value", value)

        return None

    @staticmethod
    def _extract_nested_value(
        source: Any,
        names: Sequence[str],
    ) -> Any:
        if source is None:
            return None

        value = ResultFusion._extract_value(
            source,
            names,
        )

        if value is not None:
            return value

        metadata = ResultFusion._extract_value(
            source,
            ("metadata",),
        )

        if isinstance(metadata, Mapping):
            for name in names:
                if metadata.get(name) is not None:
                    return metadata[name]

        return None

    @staticmethod
    def _extract_string_sequence(
        value: Any,
    ) -> tuple[str, ...]:
        if value is None:
            return ()

        if isinstance(value, str):
            cleaned = value.strip()
            return (cleaned,) if cleaned else ()

        if isinstance(value, Mapping):
            iterable = value.keys()
        else:
            try:
                iterable = list(value)
            except TypeError:
                iterable = [value]

        return tuple(
            sorted(
                {
                    str(item).strip()
                    for item in iterable
                    if item is not None
                    and str(item).strip()
                },
            ),
        )

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_source(
        source: RetrievalSource | str,
    ) -> RetrievalSource:
        if isinstance(source, RetrievalSource):
            return source

        normalized = str(source).strip().casefold()

        aliases = {
            "table": RetrievalSource.TABLE,
            "table_retriever": RetrievalSource.TABLE,
            "bm25": RetrievalSource.BM25,
            "lexical": RetrievalSource.BM25,
            "vector": RetrievalSource.VECTOR,
            "embedding": RetrievalSource.VECTOR,
            "semantic": RetrievalSource.VECTOR,
            "standard": RetrievalSource.STANDARD,
            "standards": RetrievalSource.STANDARD,
            "gri": RetrievalSource.STANDARD,
            "esrs": RetrievalSource.STANDARD,
        }

        return aliases.get(
            normalized,
            RetrievalSource.UNKNOWN,
        )


__all__ = [
    "FusedRetrievalResult",
    "FusionInput",
    "FusionMethod",
    "ResultFusion",
    "ResultFusionConfig",
    "ResultFusionError",
    "RetrievalSource",
]