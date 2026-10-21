from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


class CrossEncoderProtocol(Protocol):
    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[float]:
        ...


@dataclass(slots=True, frozen=True)
class RerankCandidate:
    candidate_id: str
    text: str
    original_score: float
    payload: Any

    report_id: str | None = None
    page_number: int | None = None
    row_index: int | None = None

    sources: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RerankedResult:
    candidate_id: str
    score: float
    rank: int
    payload: Any

    cross_encoder_score: float
    original_score: float

    report_id: str | None = None
    page_number: int | None = None
    row_index: int | None = None

    sources: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RerankerConfig:
    top_k: int = 15
    minimum_score: float = 0.0

    cross_encoder_weight: float = 0.75
    original_score_weight: float = 0.25

    normalize_cross_encoder_scores: bool = True
    normalize_original_scores: bool = True

    exact_phrase_bonus: float = 0.05
    matched_term_bonus: float = 0.02
    multi_source_bonus: float = 0.03

    metric_phrase_bonus: float = 0.12
    numeric_evidence_bonus: float = 0.08
    unit_evidence_bonus: float = 0.10
    year_evidence_bonus: float = 0.04
    unrelated_context_penalty: float = 0.16

    table_source_bonus: float = 0.14
    table_row_bonus: float = 0.08
    metric_numeric_pair_bonus: float = 0.12
    minimum_table_results: int = 1
    maximum_results_per_table: int = 2

    maximum_text_length: int = 5000
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError(
                "minimum_score 0 ile 1 arasında olmalıdır.",
            )

        if self.cross_encoder_weight < 0:
            raise ValueError(
                "cross_encoder_weight negatif olamaz.",
            )

        if self.original_score_weight < 0:
            raise ValueError(
                "original_score_weight negatif olamaz.",
            )

        if math.isclose(
            self.cross_encoder_weight
            + self.original_score_weight,
            0.0,
        ):
            raise ValueError(
                "Reranker ağırlıklarının toplamı sıfır olamaz.",
            )

        if self.exact_phrase_bonus < 0:
            raise ValueError(
                "exact_phrase_bonus negatif olamaz.",
            )

        if self.matched_term_bonus < 0:
            raise ValueError(
                "matched_term_bonus negatif olamaz.",
            )

        if self.multi_source_bonus < 0:
            raise ValueError(
                "multi_source_bonus negatif olamaz.",
            )

        for field_name, value in (
            ("metric_phrase_bonus", self.metric_phrase_bonus),
            ("numeric_evidence_bonus", self.numeric_evidence_bonus),
            ("unit_evidence_bonus", self.unit_evidence_bonus),
            ("year_evidence_bonus", self.year_evidence_bonus),
            ("unrelated_context_penalty", self.unrelated_context_penalty),
            ("table_source_bonus", self.table_source_bonus),
            ("table_row_bonus", self.table_row_bonus),
            ("metric_numeric_pair_bonus", self.metric_numeric_pair_bonus),
        ):
            if value < 0:
                raise ValueError(
                    f"{field_name} negatif olamaz.",
                )

        if self.minimum_table_results < 0:
            raise ValueError(
                "minimum_table_results negatif olamaz.",
            )

        if self.maximum_results_per_table <= 0:
            raise ValueError(
                "maximum_results_per_table pozitif olmalıdır.",
            )

        if self.maximum_text_length <= 0:
            raise ValueError(
                "maximum_text_length pozitif olmalıdır.",
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size pozitif olmalıdır.",
            )


class RerankerError(RuntimeError):
    """Reranking işlemleri sırasında oluşan temel hata."""


class CrossEncoderNotConfiguredError(RerankerError):
    """Cross-encoder modeli tanımlanmadığında yükseltilir."""


class Reranker:
    """
    Retrieval sonuçlarını cross-encoder tabanlı olarak yeniden sıralar.

    Nihai skor şu sinyallerden oluşturulur:
    - Cross-encoder sorgu–kanıt uygunluk skoru
    - Fusion veya retriever tarafından üretilen ilk skor
    - Tam sorgu ifadesi eşleşmesi
    - Eşleşen terim sayısı
    - Birden fazla retrieval kaynağında bulunma bonusu

    Sınıf; result_fusion.py çıktılarıyla doğrudan çalışabilecek şekilde
    tasarlanmıştır ancak ortak alanları sağlayan diğer sonuç modellerini
    de kabul eder.
    """

    _NUMBER_PATTERN = re.compile(
        r"(?<![A-Za-z])[-+]?\\d{1,3}(?:[.,]\\d{3})*(?:[.,]\\d+)?"
        r"|(?<![A-Za-z])[-+]?\\d+(?:[.,]\\d+)?",
    )

    _YEAR_PATTERN = re.compile(r"\\b(?:19|20)\\d{2}\\b")

    _UNIT_FAMILIES: tuple[
        tuple[tuple[str, ...], tuple[str, ...]],
        ...,
    ] = (
        (
            ("employee", "employees", "workforce", "headcount", "manager"),
            ("person", "people", "employee", "employees", "kişi"),
        ),
        (
            ("water", "withdrawal", "recycled water", "water intensity"),
            (
                "m3", "m³", "cubic meter", "cubic meters",
                "cubic metre", "cubic metres", "megalitre", "ml",
            ),
        ),
        (
            ("energy", "electricity", "natural gas", "coal", "petroleum"),
            ("gj", "tj", "mj", "mwh", "gwh", "kwh", "nm3", "sm3", "toe"),
        ),
        (
            ("scope 1", "scope 2", "ghg", "greenhouse gas", "carbon"),
            ("tco2e", "co2e", "kgco2e", "ton co2e", "tonnes co2e"),
        ),
        (
            ("waste", "hazardous waste", "landfill", "recycled waste"),
            ("ton", "tons", "tonne", "tonnes", "kg", "metric ton"),
        ),
        (
            ("nox", "sox", "voc", "particulate", "pm ", "hap", "pop"),
            ("ton", "tons", "tonne", "tonnes", "kg", "metric ton"),
        ),
        (
            ("revenue", "net sales", "profit", "economic value"),
            ("try", "tl", "usd", "eur", "gbp", "million", "billion"),
        ),
        (
            ("rate", "ratio", "intensity", "percentage"),
            ("%", "percent", "percentage", "per 200,000", "per 1,000,000"),
        ),
    )

    _UNRELATED_PATTERNS: tuple[
        tuple[tuple[str, ...], tuple[str, ...]],
        ...,
    ] = (
        (
            ("injury", "occupational", "severity rate", "illness rate"),
            (
                "goodwill", "discount rate", "interest rate",
                "financial risk", "impairment test",
            ),
        ),
        (
            ("water consumption", "water withdrawal", "water intensity"),
            (
                "dishwasher", "washing machine", "product rating",
                "water efficient product",
            ),
        ),
        (
            ("energy consumption", "electricity consumption", "natural gas"),
            (
                "energy rating", "energy efficient product",
                "dishwasher", "refrigerator", "washing machine",
            ),
        ),
        (
            ("hazardous waste", "non hazardous waste", "landfill"),
            (
                "food waste", "financial liabilities",
                "trade payables", "capital risk",
            ),
        ),
        (
            ("employee headcount", "total employees", "female employees",
             "male employees", "white collar", "blue collar"),
            (
                "training hours", "webinar", "language training",
                "student", "dealer",
            ),
        ),
        (
            ("scope 1", "scope 2", "ghg emission intensity"),
            (
                "emissions reduction target", "net zero target",
                "cost of response", "financial impact",
            ),
        ),
    )

    def __init__(
        self,
        cross_encoder: CrossEncoderProtocol | None = None,
        *,
        config: RerankerConfig | None = None,
    ) -> None:
        self.cross_encoder = cross_encoder
        self.config = config or RerankerConfig()

    def rerank(
        self,
        query: str,
        candidates: Iterable[Any],
        *,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[RerankedResult]:
        cleaned_query = self._clean_text(query)

        if not cleaned_query:
            raise RerankerError(
                "Reranking sorgusu boş olamaz.",
            )

        prepared_candidates = [
            self._to_candidate(candidate)
            for candidate in candidates
        ]

        prepared_candidates = [
            candidate
            for candidate in prepared_candidates
            if candidate.text
        ]

        if not prepared_candidates:
            return []

        cross_encoder_scores = self._predict_scores(
            cleaned_query,
            prepared_candidates,
        )

        if len(cross_encoder_scores) != len(prepared_candidates):
            raise RerankerError(
                "Cross-encoder skor sayısı aday sayısıyla eşleşmiyor.",
            )

        normalized_cross_scores = self._normalize_scores(
            cross_encoder_scores,
            enabled=self.config.normalize_cross_encoder_scores,
        )

        original_scores = [
            candidate.original_score
            for candidate in prepared_candidates
        ]

        normalized_original_scores = self._normalize_scores(
            original_scores,
            enabled=self.config.normalize_original_scores,
        )

        reranked_results: list[RerankedResult] = []

        normalized_query = self._normalize_text(cleaned_query)

        for index, candidate in enumerate(prepared_candidates):
            cross_score = normalized_cross_scores[index]
            original_score = normalized_original_scores[index]

            final_score = self._combine_scores(
                cross_encoder_score=cross_score,
                original_score=original_score,
            )

            normalized_candidate_text = self._normalize_text(
                candidate.text,
            )

            if (
                normalized_query
                and normalized_query in normalized_candidate_text
            ):
                final_score += self.config.exact_phrase_bonus

            if candidate.matched_terms:
                matched_term_bonus = (
                    len(candidate.matched_terms)
                    * self.config.matched_term_bonus
                )
                final_score += min(matched_term_bonus, 0.12)

            if len(candidate.sources) > 1:
                final_score += (
                    min(len(candidate.sources) - 1, 3)
                    * self.config.multi_source_bonus
                )

            metric_adjustment, metric_metadata = (
                self._metric_evidence_adjustment(
                    query=normalized_query,
                    candidate_text=normalized_candidate_text,
                )
            )
            final_score += metric_adjustment

            is_table_source = self._is_table_candidate(candidate)
            if is_table_source:
                final_score += self.config.table_source_bonus

            if candidate.row_index is not None:
                final_score += self.config.table_row_bonus

            if (
                metric_metadata.get("matched_metric_phrases")
                and (
                    metric_metadata.get("has_numeric_evidence")
                    or self._has_missing_value_marker(
                        normalized_candidate_text,
                    )
                )
            ):
                final_score += self.config.metric_numeric_pair_bonus

            final_score = max(
                0.0,
                min(final_score, 1.0),
            )

            reranked_results.append(
                RerankedResult(
                    candidate_id=candidate.candidate_id,
                    score=round(final_score, 6),
                    rank=0,
                    payload=candidate.payload,
                    cross_encoder_score=round(
                        cross_encoder_scores[index],
                        6,
                    ),
                    original_score=round(
                        candidate.original_score,
                        6,
                    ),
                    report_id=candidate.report_id,
                    page_number=candidate.page_number,
                    row_index=candidate.row_index,
                    sources=candidate.sources,
                    matched_terms=candidate.matched_terms,
                    metadata={
                        **dict(candidate.metadata),
                        "normalized_cross_encoder_score": round(
                            cross_score,
                            6,
                        ),
                        "normalized_original_score": round(
                            original_score,
                            6,
                        ),
                        **metric_metadata,
                    },
                ),
            )

        score_threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                "minimum_score 0 ile 1 arasında olmalıdır.",
            )

        reranked_results = [
            result
            for result in reranked_results
            if result.score >= score_threshold
        ]

        reranked_results.sort(
            key=lambda result: (
                -result.score,
                result.page_number
                if result.page_number is not None
                else 10**9,
                result.candidate_id,
            ),
        )

        result_limit = top_k or self.config.top_k

        if result_limit <= 0:
            return []

        selected_results = self._select_final_results(
            reranked_results,
            result_limit=result_limit,
        )

        return [
            RerankedResult(
                candidate_id=result.candidate_id,
                score=result.score,
                rank=index + 1,
                payload=result.payload,
                cross_encoder_score=result.cross_encoder_score,
                original_score=result.original_score,
                report_id=result.report_id,
                page_number=result.page_number,
                row_index=result.row_index,
                sources=result.sources,
                matched_terms=result.matched_terms,
                metadata=result.metadata,
            )
            for index, result in enumerate(selected_results)
        ]

    def _select_final_results(
        self,
        reranked_results: Sequence[RerankedResult],
        *,
        result_limit: int,
    ) -> list[RerankedResult]:
        """
        Nicel sorularda güçlü tablo kanıtlarının reranking sonrasında
        kaybolmasını önler.
        """

        if result_limit <= 0:
            return []

        table_candidates = [
            result
            for result in reranked_results
            if self._is_table_result(result)
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

        selected: list[RerankedResult] = []
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

            base_id = self._base_table_id(result.candidate_id)
            current_count = table_counts.get(base_id, 0)

            if current_count >= self.config.maximum_results_per_table:
                continue

            selected.append(result)
            selected_ids.add(result.candidate_id)
            table_counts[base_id] = current_count + 1

        for result in reranked_results:
            if len(selected) >= result_limit:
                break

            if result.candidate_id in selected_ids:
                continue

            if self._is_table_result(result):
                base_id = self._base_table_id(result.candidate_id)
                current_count = table_counts.get(base_id, 0)

                if current_count >= self.config.maximum_results_per_table:
                    continue

                table_counts[base_id] = current_count + 1

            selected.append(result)
            selected_ids.add(result.candidate_id)

        selected.sort(
            key=lambda result: (
                -result.score,
                result.page_number
                if result.page_number is not None
                else 10**9,
                result.candidate_id,
            ),
        )

        return selected[:result_limit]

    @staticmethod
    def _base_table_id(candidate_id: str) -> str:
        return candidate_id.split("::row:", 1)[0]

    @staticmethod
    def _is_table_result(result: RerankedResult) -> bool:
        return (
            "table" in {source.casefold() for source in result.sources}
            or result.row_index is not None
            or result.candidate_id.startswith("page-")
            and "-table-" in result.candidate_id
        )

    @staticmethod
    def _is_table_candidate(candidate: RerankCandidate) -> bool:
        return (
            "table" in {source.casefold() for source in candidate.sources}
            or candidate.row_index is not None
            or candidate.candidate_id.startswith("page-")
            and "-table-" in candidate.candidate_id
        )

    @staticmethod
    def _has_missing_value_marker(text: str) -> bool:
        return bool(
            re.search(
                r"(?:\*{2,}|\bn/?a\b|\bnot disclosed\b|\bnot available\b)",
                text,
            )
        )

    def rerank_texts(
        self,
        query: str,
        texts: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        candidates = [
            RerankCandidate(
                candidate_id=f"candidate-{index}",
                text=text,
                original_score=0.0,
                payload=text,
            )
            for index, text in enumerate(texts)
            if self._clean_text(text)
        ]

        return self.rerank(
            query,
            candidates,
            top_k=top_k,
        )

    def score_pair(
        self,
        query: str,
        candidate_text: str,
    ) -> float:
        cleaned_query = self._clean_text(query)
        cleaned_candidate = self._clean_text(candidate_text)

        if not cleaned_query:
            raise RerankerError(
                "Sorgu metni boş olamaz.",
            )

        if not cleaned_candidate:
            raise RerankerError(
                "Aday metni boş olamaz.",
            )

        scores = self._predict_pairs(
            [
                (
                    cleaned_query,
                    self._truncate_text(cleaned_candidate),
                ),
            ],
        )

        if len(scores) != 1:
            raise RerankerError(
                "Cross-encoder tek skor döndürmelidir.",
            )

        return float(scores[0])

    def _predict_scores(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> list[float]:
        pairs = [
            (
                query,
                self._truncate_text(candidate.text),
            )
            for candidate in candidates
        ]

        scores: list[float] = []

        for start_index in range(
            0,
            len(pairs),
            self.config.batch_size,
        ):
            batch = pairs[
                start_index : start_index + self.config.batch_size
            ]

            batch_scores = self._predict_pairs(batch)

            if len(batch_scores) != len(batch):
                raise RerankerError(
                    "Cross-encoder batch skor sayısı giriş sayısıyla "
                    "eşleşmiyor.",
                )

            scores.extend(
                float(score)
                for score in batch_scores
            )

        return scores

    def _predict_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> Sequence[float]:
        if self.cross_encoder is None:
            return self._lexical_fallback_scores(pairs)

        predict_method = getattr(
            self.cross_encoder,
            "predict",
            None,
        )

        if not callable(predict_method):
            raise CrossEncoderNotConfiguredError(
                "Cross-encoder predict metodu sağlamıyor.",
            )

        try:
            predictions = predict_method(pairs)
        except Exception as exc:
            raise RerankerError(
                "Cross-encoder skor üretimi başarısız oldu.",
            ) from exc

        return [
            self._to_float(score)
            for score in predictions
        ]

    def _lexical_fallback_scores(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        """
        Cross-encoder sağlanmadığında deterministik sözcüksel fallback
        skoru üretir.

        Bu yöntem gerçek cross-encoder yerine geçmez; geliştirme ve test
        aşamalarında pipeline'ın çalışmasını sürdürmek için kullanılır.
        """
        scores: list[float] = []

        for query, candidate in pairs:
            query_tokens = set(
                self._tokenize(query),
            )
            candidate_tokens = set(
                self._tokenize(candidate),
            )

            if not query_tokens or not candidate_tokens:
                scores.append(0.0)
                continue

            overlap = query_tokens.intersection(
                candidate_tokens,
            )

            recall = len(overlap) / len(query_tokens)
            precision = len(overlap) / len(candidate_tokens)

            if recall + precision == 0:
                f1_score = 0.0
            else:
                f1_score = (
                    2 * recall * precision
                    / (recall + precision)
                )

            normalized_query = self._normalize_text(query)
            normalized_candidate = self._normalize_text(candidate)

            phrase_bonus = (
                0.15
                if normalized_query in normalized_candidate
                else 0.0
            )

            scores.append(
                min(f1_score + phrase_bonus, 1.0),
            )

        return scores

    def _metric_evidence_adjustment(
        self,
        *,
        query: str,
        candidate_text: str,
    ) -> tuple[float, dict[str, Any]]:
        adjustment = 0.0
        matched_metric_phrases = self._matched_metric_phrases(
            query,
            candidate_text,
        )

        if matched_metric_phrases:
            adjustment += min(
                len(matched_metric_phrases)
                * self.config.metric_phrase_bonus,
                0.24,
            )

        has_number = self._NUMBER_PATTERN.search(candidate_text) is not None
        has_missing_marker = self._has_missing_value_marker(candidate_text)

        if has_number or has_missing_marker:
            adjustment += self.config.numeric_evidence_bonus

        matched_units = self._matched_expected_units(
            query,
            candidate_text,
        )
        if matched_units:
            adjustment += self.config.unit_evidence_bonus

        query_years = set(self._YEAR_PATTERN.findall(query))
        candidate_years = set(self._YEAR_PATTERN.findall(candidate_text))
        matched_years = sorted(query_years.intersection(candidate_years))
        if matched_years:
            adjustment += self.config.year_evidence_bonus

        unrelated_matches = self._unrelated_context_matches(
            query,
            candidate_text,
        )
        if unrelated_matches:
            adjustment -= min(
                len(unrelated_matches)
                * self.config.unrelated_context_penalty,
                0.32,
            )

        metadata = {
            "metric_evidence_adjustment": round(adjustment, 6),
            "matched_metric_phrases": matched_metric_phrases,
            "has_numeric_evidence": has_number,
            "has_missing_value_marker": has_missing_marker,
            "matched_expected_units": matched_units,
            "matched_years": tuple(matched_years),
            "unrelated_context_matches": unrelated_matches,
        }
        return adjustment, metadata

    @classmethod
    def _matched_metric_phrases(
        cls,
        query: str,
        candidate_text: str,
    ) -> tuple[str, ...]:
        query_tokens = [
            token
            for token in cls._tokenize(query)
            if token not in {
                "sirketin", "toplam", "nedir", "kadardir", "kactir",
                "total", "for", "the", "of", "in",
            }
        ]

        matches: set[str] = set()
        maximum_ngram = min(5, len(query_tokens))

        for size in range(maximum_ngram, 1, -1):
            for index in range(0, len(query_tokens) - size + 1):
                phrase = " ".join(
                    query_tokens[index : index + size]
                )
                if phrase in candidate_text:
                    matches.add(phrase)

            if len(matches) >= 2:
                break

        return tuple(
            sorted(
                matches,
                key=lambda value: (-len(value.split()), value),
            )[:3]
        )

    @classmethod
    def _matched_expected_units(
        cls,
        query: str,
        candidate_text: str,
    ) -> tuple[str, ...]:
        units: set[str] = set()

        for query_terms, family_units in cls._UNIT_FAMILIES:
            if not any(term in query for term in query_terms):
                continue

            for unit in family_units:
                normalized_unit = cls._normalize_text(unit)
                if normalized_unit and normalized_unit in candidate_text:
                    units.add(unit)

        return tuple(sorted(units))

    @classmethod
    def _unrelated_context_matches(
        cls,
        query: str,
        candidate_text: str,
    ) -> tuple[str, ...]:
        matches: set[str] = set()

        for query_terms, negative_terms in cls._UNRELATED_PATTERNS:
            if not any(term in query for term in query_terms):
                continue

            for term in negative_terms:
                if term in candidate_text:
                    matches.add(term)

        return tuple(sorted(matches))

    def _combine_scores(
        self,
        *,
        cross_encoder_score: float,
        original_score: float,
    ) -> float:
        total_weight = (
            self.config.cross_encoder_weight
            + self.config.original_score_weight
        )

        return (
            cross_encoder_score
            * self.config.cross_encoder_weight
            + original_score
            * self.config.original_score_weight
        ) / total_weight

    def _to_candidate(
        self,
        source: Any,
    ) -> RerankCandidate:
        if isinstance(source, RerankCandidate):
            return source

        candidate_id = self._extract_candidate_id(source)
        payload = self._extract_payload(source)
        text = self._extract_candidate_text(
            source,
            payload,
        )

        score = self._extract_score(source)

        report_id = self._extract_first_value(
            source,
            ("report_id",),
        )

        if report_id is None:
            report_id = self._extract_nested_value(
                payload,
                ("report_id",),
            )

        page_number = self._optional_int(
            self._extract_first_value(
                source,
                (
                    "page_number",
                    "source_page",
                    "page",
                ),
            ),
        )

        if page_number is None:
            page_number = self._optional_int(
                self._extract_nested_value(
                    payload,
                    (
                        "page_number",
                        "source_page",
                        "page",
                    ),
                ),
            )

        row_index = self._optional_int(
            self._extract_first_value(
                source,
                ("row_index",),
            ),
        )

        sources = self._extract_string_sequence(
            self._extract_first_value(
                source,
                ("sources", "source_types"),
            ),
        )

        matched_terms = self._extract_string_sequence(
            self._extract_first_value(
                source,
                (
                    "matched_terms",
                    "matched_keywords",
                    "terms",
                ),
            ),
        )

        metadata = self._extract_metadata(source)

        return RerankCandidate(
            candidate_id=candidate_id,
            text=text,
            original_score=score,
            payload=payload,
            report_id=self._optional_string(report_id),
            page_number=page_number,
            row_index=row_index,
            sources=sources,
            matched_terms=matched_terms,
            metadata=metadata,
        )

    def _extract_candidate_id(
        self,
        source: Any,
    ) -> str:
        value = self._extract_first_value(
            source,
            (
                "candidate_id",
                "document_id",
                "chunk_id",
                "table_id",
                "evidence_id",
                "id",
                "identifier",
            ),
        )

        if value is None:
            payload = self._extract_payload(source)
            value = self._extract_nested_value(
                payload,
                (
                    "candidate_id",
                    "document_id",
                    "chunk_id",
                    "table_id",
                    "evidence_id",
                    "id",
                    "identifier",
                ),
            )

        if value is None or not str(value).strip():
            raise RerankerError(
                "Reranking adayında kimlik alanı bulunamadı.",
            )

        return str(value).strip()

    def _extract_candidate_text(
        self,
        source: Any,
        payload: Any,
    ) -> str:
        direct_text = self._extract_first_value(
            source,
            (
                "text",
                "matched_text",
                "content",
                "chunk_text",
            ),
        )

        if direct_text is not None:
            return self._clean_text(direct_text)

        payload_text = self._extract_nested_value(
            payload,
            (
                "text",
                "matched_text",
                "content",
                "chunk_text",
            ),
        )

        if payload_text is not None:
            return self._clean_text(payload_text)

        table_text = self._build_table_text(payload)

        if table_text:
            return table_text

        return self._clean_text(payload)

    @staticmethod
    def _extract_payload(source: Any) -> Any:
        for name in (
            "payload",
            "document",
            "table",
            "item",
            "record",
            "disclosure",
        ):
            value = Reranker._extract_first_value(
                source,
                (name,),
            )

            if value is not None:
                return value

        return source

    @staticmethod
    def _extract_score(source: Any) -> float:
        value = Reranker._extract_first_value(
            source,
            (
                "score",
                "fusion_score",
                "retrieval_score",
                "similarity",
                "confidence",
            ),
        )

        if value is None:
            return 0.0

        return Reranker._to_float(value)

    @staticmethod
    def _extract_metadata(source: Any) -> dict[str, Any]:
        metadata = Reranker._extract_first_value(
            source,
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
    def _extract_first_value(
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

        value = Reranker._extract_first_value(
            source,
            names,
        )

        if value is not None:
            return value

        metadata = Reranker._extract_first_value(
            source,
            ("metadata",),
        )

        if isinstance(metadata, Mapping):
            for name in names:
                if metadata.get(name) is not None:
                    return metadata[name]

        return None

    @staticmethod
    def _normalize_scores(
        scores: Sequence[float],
        *,
        enabled: bool,
    ) -> list[float]:
        if not scores:
            return []

        numeric_scores = [
            Reranker._to_float(score)
            for score in scores
        ]

        if not enabled:
            return [
                max(0.0, min(score, 1.0))
                for score in numeric_scores
            ]

        minimum = min(numeric_scores)
        maximum = max(numeric_scores)

        if math.isclose(minimum, maximum):
            if maximum > 0:
                return [1.0] * len(numeric_scores)

            return [0.0] * len(numeric_scores)

        score_range = maximum - minimum

        return [
            (score - minimum) / score_range
            for score in numeric_scores
        ]

    @classmethod
    def _build_table_text(
        cls,
        source: Any,
    ) -> str:
        if source is None:
            return ""

        title = cls._extract_first_value(
            source,
            (
                "title",
                "caption",
                "table_title",
                "name",
            ),
        )

        headers = cls._extract_first_value(
            source,
            (
                "headers",
                "column_headers",
                "columns",
            ),
        )

        rows = cls._extract_first_value(
            source,
            (
                "rows",
                "data",
                "records",
                "table_rows",
            ),
        )

        parts: list[str] = []

        if title:
            parts.append(cls._clean_text(title))

        if headers:
            header_values = cls._extract_string_sequence(
                headers,
            )
            if header_values:
                parts.append(" | ".join(header_values))

        if rows and not isinstance(rows, (str, bytes)):
            try:
                iterable_rows = list(rows)
            except TypeError:
                iterable_rows = []

            for row in iterable_rows:
                row_values = cls._extract_row_values(row)

                if row_values:
                    parts.append(" | ".join(row_values))

        return "\n".join(
            part
            for part in parts
            if part
        )

    @classmethod
    def _extract_row_values(
        cls,
        row: Any,
    ) -> tuple[str, ...]:
        if row is None:
            return ()

        if isinstance(row, Mapping):
            values = row.values()
        elif isinstance(row, (str, bytes)):
            cleaned = cls._clean_text(row)
            return (cleaned,) if cleaned else ()
        else:
            cells = getattr(row, "cells", None)

            if cells is not None:
                values = cells
            else:
                try:
                    values = list(row)
                except TypeError:
                    values = [row]

        cleaned_values: list[str] = []

        for value in values:
            if isinstance(value, Mapping):
                resolved = (
                    value.get("normalized_value")
                    or value.get("value")
                    or value.get("text")
                    or value.get("content")
                )
            else:
                resolved = getattr(
                    value,
                    "normalized_value",
                    getattr(
                        value,
                        "value",
                        getattr(
                            value,
                            "text",
                            getattr(value, "content", value),
                        ),
                    ),
                )

            cleaned = cls._clean_text(resolved)

            if cleaned:
                cleaned_values.append(cleaned)

        return tuple(cleaned_values)

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

        normalized: set[str] = set()

        for item in iterable:
            enum_value = getattr(item, "value", item)

            if enum_value is None:
                continue

            cleaned = str(enum_value).strip()

            if cleaned:
                normalized.add(cleaned)

        return tuple(sorted(normalized))

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
    def _to_float(value: Any) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise RerankerError(
                "Reranker skoru sayısal olmalıdır.",
            ) from exc

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            raise RerankerError(
                "Reranker skoru NaN veya sonsuz olamaz.",
            )

        return numeric_value

    def _truncate_text(self, text: str) -> str:
        cleaned = self._clean_text(text)

        if len(cleaned) <= self.config.maximum_text_length:
            return cleaned

        return (
            cleaned[
                : self.config.maximum_text_length - 3
            ].rstrip()
            + "..."
        )

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        normalized = Reranker._normalize_text(text)

        return tuple(
            token
            for token in normalized.split()
            if len(token) >= 2
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        enum_value = getattr(value, "value", value)

        return " ".join(
            str(enum_value).replace("\n", " ").split(),
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        if not value:
            return ""

        translation_table = str.maketrans(
            {
                "ı": "i",
                "İ": "i",
                "ş": "s",
                "Ş": "s",
                "ğ": "g",
                "Ğ": "g",
                "ü": "u",
                "Ü": "u",
                "ö": "o",
                "Ö": "o",
                "ç": "c",
                "Ç": "c",
            },
        )

        normalized = str(value).translate(
            translation_table,
        ).casefold()

        cleaned_characters = [
            character
            if character.isalnum()
            or character.isspace()
            or character in "-./%"
            else " "
            for character in normalized
        ]

        return " ".join(
            "".join(cleaned_characters).split(),
        )


__all__ = [
    "CrossEncoderNotConfiguredError",
    "CrossEncoderProtocol",
    "RerankCandidate",
    "RerankedResult",
    "Reranker",
    "RerankerConfig",
    "RerankerError",
]