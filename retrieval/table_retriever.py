from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Protocol, Sequence


class TableQueryIntent(StrEnum):
    VALUE = "value"
    COMPARISON = "comparison"
    TREND = "trend"
    TARGET = "target"
    RATIO = "ratio"
    COUNT = "count"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class TableQuery:
    text: str
    normalized_text: str
    intent: TableQueryIntent = TableQueryIntent.UNKNOWN
    metric_terms: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    units: tuple[str, ...] = ()
    standard_codes: tuple[str, ...] = ()
    category: str | None = None
    topic: str | None = None
    report_id: str | None = None
    company_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TableSearchResult:
    table_id: str
    score: float
    table: Any
    matched_text: str
    matched_terms: tuple[str, ...] = ()
    matched_years: tuple[int, ...] = ()
    matched_units: tuple[str, ...] = ()
    matched_standard_codes: tuple[str, ...] = ()
    row_index: int | None = None
    column_index: int | None = None
    source_page: int | None = None
    report_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TableRetrieverConfig:
    top_k: int = 20
    minimum_score: float = 0.1

    lexical_weight: float = 0.25
    metric_weight: float = 0.35
    year_weight: float = 0.14
    unit_weight: float = 0.08
    standard_weight: float = 0.08
    category_weight: float = 0.05
    structure_weight: float = 0.05

    exact_phrase_bonus: float = 0.12
    title_match_bonus: float = 0.1
    header_match_bonus: float = 0.08
    numeric_content_bonus: float = 0.05
    metric_numeric_pair_bonus: float = 0.18
    requested_year_bonus: float = 0.08
    semantic_metric_bonus: float = 0.16
    weak_metric_penalty: float = 0.72
    generic_narrative_penalty: float = 0.55

    invalid_table_score_multiplier: float = 0.55
    minimum_table_quality_multiplier: float = 0.35

    search_rows: bool = True
    search_cells: bool = True
    maximum_row_results_per_table: int = 8

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalıdır.")

        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError(
                "minimum_score 0 ile 1 arasında olmalıdır.",
            )

        if not 0.0 <= self.invalid_table_score_multiplier <= 1.0:
            raise ValueError(
                "invalid_table_score_multiplier 0 ile 1 arasında olmalıdır.",
            )

        if not 0.0 <= self.minimum_table_quality_multiplier <= 1.0:
            raise ValueError(
                "minimum_table_quality_multiplier 0 ile 1 arasında olmalıdır.",
            )

        if self.maximum_row_results_per_table <= 0:
            raise ValueError(
                "maximum_row_results_per_table pozitif olmalıdır.",
            )

        weights = (
            self.lexical_weight,
            self.metric_weight,
            self.year_weight,
            self.unit_weight,
            self.standard_weight,
            self.category_weight,
            self.structure_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Retriever ağırlıkları negatif olamaz.",
            )

        if math.isclose(sum(weights), 0.0):
            raise ValueError(
                "Retriever ağırlıklarının toplamı sıfır olamaz.",
            )


class TableRepositoryProtocol(Protocol):
    def all(self) -> Iterable[Any]:
        ...

    def find(
        self,
        *,
        report_id: str | None = None,
        category: str | None = None,
        topic: str | None = None,
    ) -> Iterable[Any]:
        ...


class TableRetrieverError(RuntimeError):
    """Tablo retrieval işlemlerinde oluşan temel hata."""


class TableRetriever:
    """
    Sayısal ve tablo odaklı sorular için normalize edilmiş tabloları arar.

    Skorlama bileşenleri:
    - Sorgu ile tablo metni arasındaki sözcüksel benzerlik
    - Metrik ve gösterge terimi eşleşmesi
    - Yıl eşleşmesi
    - Birim eşleşmesi
    - GRI ve ESRS kodu eşleşmesi
    - ESG kategorisi eşleşmesi
    - Başlık, sütun başlığı ve sayısal yapı bonusları

    Retriever hem tablo düzeyinde hem de satır düzeyinde sonuç üretebilir.
    """

    _YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    _NUMBER_PATTERN = re.compile(
        r"(?<!\w)"
        r"[-+]?"
        r"(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
        r"(?:[.,]\d+)?"
        r"(?:\s*%)?"
        r"(?!\w)",
    )

    _STANDARD_PATTERN = re.compile(
        r"\b(?:"
        r"GRI\s+\d{1,3}(?:-\d{1,3})?"
        r"|ESRS\s+(?:1|2|E[1-5]|S[1-4]|G1)"
        r"|SASB\s+[A-Z0-9.-]+"
        r"|TCFD"
        r"|SDG\s+(?:[1-9]|1[0-7])"
        r")\b",
        re.IGNORECASE,
    )

    _COMMON_UNITS = (
        "tco2e",
        "co2e",
        "kg co2e",
        "ton",
        "tonne",
        "tonnes",
        "metric ton",
        "metric tons",
        "kg",
        "g",
        "mg",
        "mwh",
        "gwh",
        "kwh",
        "gj",
        "mj",
        "m3",
        "m³",
        "litre",
        "liter",
        "litres",
        "liters",
        "megalitre",
        "megaliter",
        "hectare",
        "hectares",
        "ha",
        "percent",
        "percentage",
        "%",
        "oran",
        "adet",
        "kişi",
        "employee",
        "employees",
        "hour",
        "hours",
        "saat",
        "try",
        "tl",
        "usd",
        "eur",
    )

    _METRIC_CONCEPTS: Mapping[str, tuple[str, ...]] = {
        "petroleum": (
            "petroleum", "petrol", "fuel oil", "fuel-oil",
            "diesel", "gasoline", "oil products",
        ),
        "natural_gas": (
            "natural gas", "doğal gaz", "dogal gaz", "sm3",
        ),
        "coal": (
            "coal", "kömür", "komur", "lignite", "hard coal",
        ),
        "electricity_consumption": (
            "electricity consumption", "electrical energy consumption",
            "purchased electricity", "electricity used",
            "indirect energy consumption",
        ),
        "renewable_generation": (
            "renewable energy generated",
            "renewable electricity generated",
            "renewable electricity production",
            "onsite renewable generation",
            "self generated renewable electricity",
        ),
        "energy_intensity": (
            "energy intensity", "specific energy consumption",
            "energy use per production", "energy consumption intensity",
        ),
        "total_energy": (
            "total energy consumption", "energy consumption",
        ),
        "scope_1": (
            "scope 1", "direct greenhouse gas emissions",
            "direct ghg emissions",
        ),
        "scope_2": (
            "scope 2", "indirect greenhouse gas emissions",
            "energy indirect greenhouse gas emissions",
        ),
        "scope_3": (
            "scope 3", "other indirect greenhouse gas emissions",
        ),
        "ghg_intensity": (
            "greenhouse gas emission intensity",
            "ghg emission intensity", "carbon intensity",
        ),
        "nox": (
            "nox", "no x", "nitrogen oxides",
            "nitrogen oxide emissions",
        ),
        "sox": (
            "sox", "so x", "sulfur oxides", "sulphur oxides",
            "sulfur oxide emissions", "sulphur oxide emissions",
        ),
        "voc": (
            "voc", "volatile organic compounds",
            "volatile organic compound emissions",
        ),
        "pm": (
            "pm", "particulate matter",
            "particulate matter emissions",
        ),
        "pop": (
            "pop", "persistent organic pollutants",
            "persistent organic pollutant emissions",
        ),
        "hap": (
            "hap", "hazardous air pollutants",
            "hazardous air pollutant emissions",
        ),
        "water_consumption": (
            "water consumption", "water consumed",
        ),
        "water_withdrawal": (
            "water withdrawal", "water withdrawn",
            "freshwater withdrawal",
        ),
        "water_intensity": (
            "water intensity", "water consumption intensity",
            "water use per production",
        ),
        "recycled_water": (
            "recycled water", "reused water",
            "water recycled", "water reused",
        ),
        "hazardous_waste": (
            "hazardous waste",
        ),
        "non_hazardous_waste": (
            "non hazardous waste", "non-hazardous waste",
        ),
        "landfill_waste": (
            "waste to landfill", "sent to landfill",
            "disposed to landfill",
        ),
        "recycled_waste": (
            "recycled waste", "waste recycled",
            "sent for recycling",
        ),
        "recovered_waste": (
            "recovered waste", "waste recovered",
            "diverted from disposal",
        ),
        "net_sales": (
            "net sales", "net sales revenue",
        ),
        "net_profit": (
            "net profit", "net income",
            "profit attributable to owners",
        ),
        "economic_value": (
            "economic value generated",
            "direct economic value generated",
        ),
    }

    _INTENT_TERMS: Mapping[TableQueryIntent, tuple[str, ...]] = {
        TableQueryIntent.COMPARISON: (
            "compare",
            "comparison",
            "compared",
            "versus",
            "vs",
            "difference",
            "karşılaştır",
            "karşılaştırma",
            "kıyasla",
            "fark",
        ),
        TableQueryIntent.TREND: (
            "trend",
            "change",
            "increase",
            "decrease",
            "growth",
            "reduction",
            "artış",
            "azalış",
            "değişim",
            "eğilim",
        ),
        TableQueryIntent.TARGET: (
            "target",
            "goal",
            "commitment",
            "baseline",
            "base year",
            "hedef",
            "taahhüt",
            "baz yıl",
        ),
        TableQueryIntent.RATIO: (
            "ratio",
            "rate",
            "percentage",
            "share",
            "proportion",
            "oran",
            "yüzde",
            "pay",
        ),
        TableQueryIntent.COUNT: (
            "count",
            "number of",
            "how many",
            "total number",
            "sayısı",
            "kaç",
            "adet",
        ),
        TableQueryIntent.VALUE: (
            "value",
            "amount",
            "total",
            "quantity",
            "değer",
            "miktar",
            "toplam",
        ),
    }

    def __init__(
        self,
        repository: TableRepositoryProtocol | None = None,
        *,
        config: TableRetrieverConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or TableRetrieverConfig()

    def retrieve(
        self,
        query: str | TableQuery,
        *,
        tables: Iterable[Any] | None = None,
        top_k: int | None = None,
        minimum_score: float | None = None,
        report_id: str | None = None,
        category: str | None = None,
        topic: str | None = None,
    ) -> list[TableSearchResult]:
        normalized_query = (
            query
            if isinstance(query, TableQuery)
            else self.analyze_query(
                query,
                report_id=report_id,
                category=category,
                topic=topic,
            )
        )

        candidate_tables = list(
            self._resolve_tables(
                tables=tables,
                report_id=normalized_query.report_id,
                category=normalized_query.category,
                topic=normalized_query.topic,
            ),
        )

        result_limit = top_k or self.config.top_k
        score_threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )

        if result_limit <= 0:
            return []

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                "minimum_score 0 ile 1 arasında olmalıdır.",
            )

        results: list[TableSearchResult] = []

        for table in candidate_tables:
            if not self._matches_hard_filters(
                table,
                normalized_query,
            ):
                continue

            table_result = self._score_table(
                table,
                normalized_query,
            )

            if (
                table_result is not None
                and table_result.score >= score_threshold
            ):
                results.append(table_result)

            if self.config.search_rows:
                row_results = self._score_rows(
                    table,
                    normalized_query,
                )

                valid_row_results = [
                    result
                    for result in row_results
                    if result.score >= score_threshold
                ]

                results.extend(
                    valid_row_results[
                        : self.config.maximum_row_results_per_table
                    ],
                )

        deduplicated = self._deduplicate_results(results)

        deduplicated.sort(
            key=lambda result: (
                not bool(
                    result.metadata.get("direct_metric_match")
                ),
                result.metadata.get("result_level") != "row",
                -result.score,
                result.source_page
                if result.source_page is not None
                else 10**9,
                result.table_id,
                result.row_index
                if result.row_index is not None
                else -1,
            ),
        )

        return deduplicated[:result_limit]

    def analyze_query(
        self,
        query: str,
        *,
        metric_terms: Sequence[str] = (),
        years: Sequence[int] = (),
        units: Sequence[str] = (),
        standard_codes: Sequence[str] = (),
        category: str | None = None,
        topic: str | None = None,
        report_id: str | None = None,
        company_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TableQuery:
        cleaned_query = self._clean_text(query)

        if not cleaned_query:
            raise TableRetrieverError(
                "Tablo retrieval sorgusu boş olamaz.",
            )

        normalized_query = self.normalize_text(cleaned_query)

        detected_years = {
            int(match.group(0))
            for match in self._YEAR_PATTERN.finditer(cleaned_query)
        }
        detected_years.update(
            int(year)
            for year in years
            if self._is_valid_year(year)
        )

        detected_units = set(
            self._extract_units(cleaned_query),
        )
        detected_units.update(
            self._clean_text(unit)
            for unit in units
            if self._clean_text(unit)
        )

        detected_standard_codes = {
            self._normalize_standard_code(match.group(0))
            for match in self._STANDARD_PATTERN.finditer(cleaned_query)
        }
        detected_standard_codes.update(
            self._normalize_standard_code(code)
            for code in standard_codes
            if self._clean_text(code)
        )

        cleaned_metrics = {
            self._clean_text(metric)
            for metric in metric_terms
            if self._clean_text(metric)
        }

        if not cleaned_metrics:
            cleaned_metrics.update(
                self._infer_metric_terms(
                    normalized_query,
                ),
            )

        return TableQuery(
            text=cleaned_query,
            normalized_text=normalized_query,
            intent=self._detect_intent(normalized_query),
            metric_terms=tuple(sorted(cleaned_metrics)),
            years=tuple(sorted(detected_years)),
            units=tuple(sorted(detected_units)),
            standard_codes=tuple(
                sorted(detected_standard_codes),
            ),
            category=category,
            topic=topic,
            report_id=report_id,
            company_name=company_name,
            metadata=dict(metadata or {}),
        )

    def retrieve_by_metric(
        self,
        metric: str,
        *,
        years: Sequence[int] = (),
        units: Sequence[str] = (),
        report_id: str | None = None,
        tables: Iterable[Any] | None = None,
        top_k: int | None = None,
    ) -> list[TableSearchResult]:
        query = self.analyze_query(
            metric,
            metric_terms=(metric,),
            years=years,
            units=units,
            report_id=report_id,
        )

        return self.retrieve(
            query,
            tables=tables,
            top_k=top_k,
        )

    def retrieve_by_standard(
        self,
        standard_codes: Sequence[str],
        *,
        report_id: str | None = None,
        tables: Iterable[Any] | None = None,
        top_k: int | None = None,
    ) -> list[TableSearchResult]:
        cleaned_codes = tuple(
            self._normalize_standard_code(code)
            for code in standard_codes
            if self._clean_text(code)
        )

        if not cleaned_codes:
            return []

        query = self.analyze_query(
            " ".join(cleaned_codes),
            standard_codes=cleaned_codes,
            report_id=report_id,
        )

        return self.retrieve(
            query,
            tables=tables,
            top_k=top_k,
        )

    def _score_table(
        self,
        table: Any,
        query: TableQuery,
    ) -> TableSearchResult | None:
        table_id = self._get_table_id(table)
        table_text = self._build_table_text(table)

        if not table_text:
            return None

        normalized_table_text = self.normalize_text(table_text)
        title = self._get_table_title(table)
        headers = self._get_table_headers(table)
        metadata = self._get_metadata(table)

        score_components = self._calculate_score_components(
            query=query,
            candidate_text=normalized_table_text,
            title=title,
            headers=headers,
            table=table,
        )

        quality_multiplier, quality_metadata = (
            self._table_quality_multiplier(table)
        )
        total_score = (
            self._combine_scores(score_components)
            * quality_multiplier
        )

        numeric_intent = query.intent in {
            TableQueryIntent.VALUE,
            TableQueryIntent.RATIO,
            TableQueryIntent.COUNT,
            TableQueryIntent.COMPARISON,
            TableQueryIntent.TREND,
        }
        has_number_or_placeholder = bool(
            self._NUMBER_PATTERN.search(normalized_table_text)
            or re.search(r"(?:\*{2,}|\bn/?a\b|\bnot disclosed\b)", normalized_table_text)
        )
        exact_metric_score = score_components.get("metric", 0.0)
        semantic_metric_score = self._semantic_metric_score(
            query.normalized_text,
            normalized_table_text,
        )
        metric_score = max(exact_metric_score, semantic_metric_score)
        score_components["semantic_metric"] = round(
            semantic_metric_score,
            4,
        )

        # Nicel sorularda sayı veya kaynakta eksik veri işareti bulunmalıdır.
        # Metrik eşleşmesi ise artık birebir ifade zorunluluğu değildir:
        # eş anlamlı metrik aileleri de kabul edilir.
        if numeric_intent and not has_number_or_placeholder:
            return None

        # Hem exact hem semantik metrik sinyali yoksa tablo tamamen
        # elenmez; fakat ciddi biçimde aşağı çekilir. Böylece farklı rapor
        # terminolojileri gözden kaçmazken genel anlatı tabloları üst sıraya
        # çıkamaz.
        if numeric_intent and query.metric_terms and metric_score == 0.0:
            total_score *= self.config.weak_metric_penalty

        total_score = self._apply_quantitative_adjustments(
            total_score,
            query=query,
            metric_score=metric_score,
            candidate_text=normalized_table_text,
            has_number_or_placeholder=has_number_or_placeholder,
        )

        matched_terms = self._matched_terms(
            query,
            normalized_table_text,
        )
        matched_years = self._matched_years(
            query.years,
            normalized_table_text,
        )
        matched_units = self._matched_units(
            query.units,
            normalized_table_text,
        )
        matched_standards = self._matched_standards(
            query.standard_codes,
            normalized_table_text,
        )

        return TableSearchResult(
            table_id=table_id,
            score=round(total_score, 4),
            table=table,
            matched_text=self._truncate_text(table_text),
            matched_terms=matched_terms,
            matched_years=matched_years,
            matched_units=matched_units,
            matched_standard_codes=matched_standards,
            source_page=self._get_source_page(table),
            report_id=self._get_report_id(table),
            metadata={
                **metadata,
                "result_level": "table",
                "score_components": score_components,
                "quality_multiplier": quality_multiplier,
                "quality_metadata": quality_metadata,
            },
        )

    def _score_rows(
        self,
        table: Any,
        query: TableQuery,
    ) -> list[TableSearchResult]:
        rows = self._get_table_rows(table)

        if not rows:
            return []

        table_id = self._get_table_id(table)
        title = self._get_table_title(table)
        headers = self._get_table_headers(table)
        metadata = self._get_metadata(table)
        results: list[TableSearchResult] = []

        for row_index, row in enumerate(rows):
            row_values = self._row_to_values(row)
            row_text = " | ".join(
                value
                for value in row_values
                if value
            )

            if not row_text:
                continue

            contextual_text = " | ".join(
                part
                for part in (
                    title,
                    " | ".join(headers),
                    row_text,
                )
                if part
            )

            normalized_text = self.normalize_text(
                contextual_text,
            )

            score_components = self._calculate_score_components(
                query=query,
                candidate_text=normalized_text,
                title=title,
                headers=headers,
                table=table,
            )

            row_density = self._numeric_density(row_text)

            if row_density > 0:
                score_components["structure"] = min(
                    1.0,
                    score_components["structure"] + row_density * 0.25,
                )

            quality_multiplier, quality_metadata = (
                self._table_quality_multiplier(table)
            )
            total_score = (
                self._combine_scores(score_components)
                * quality_multiplier
            )

            numeric_intent = query.intent in {
                TableQueryIntent.VALUE,
                TableQueryIntent.RATIO,
                TableQueryIntent.COUNT,
                TableQueryIntent.COMPARISON,
                TableQueryIntent.TREND,
            }
            exact_metric_score = score_components.get("metric", 0.0)
            semantic_metric_score = self._semantic_metric_score(
                query.normalized_text,
                normalized_text,
            )
            metric_score = max(exact_metric_score, semantic_metric_score)
            score_components["semantic_metric"] = round(
                semantic_metric_score,
                4,
            )
            has_number_or_placeholder = bool(
                self._NUMBER_PATTERN.search(row_text)
                or re.search(
                    r"(?:\*{2,}|\bn/?a\b|\bnot disclosed\b)",
                    self.normalize_text(row_text),
                )
            )

            if numeric_intent and not has_number_or_placeholder:
                continue

            if numeric_intent and query.metric_terms and metric_score == 0.0:
                total_score *= self.config.weak_metric_penalty

            total_score = self._apply_quantitative_adjustments(
                total_score,
                query=query,
                metric_score=metric_score,
                candidate_text=normalized_text,
                has_number_or_placeholder=has_number_or_placeholder,
            )

            direct_metric_match = self._direct_row_metric_match(
                query.normalized_text,
                row_text,
            )

            if direct_metric_match and has_number_or_placeholder:
                total_score = min(1.0, total_score + 0.45)

            contextual_row_text = " | ".join(
                part
                for part in (
                    title,
                    " | ".join(headers),
                    row_text,
                )
                if part
            )

            results.append(
                TableSearchResult(
                    table_id=table_id,
                    score=round(total_score, 4),
                    table=table,
                    matched_text=contextual_row_text,
                    matched_terms=self._matched_terms(
                        query,
                        normalized_text,
                    ),
                    matched_years=self._matched_years(
                        query.years,
                        normalized_text,
                    ),
                    matched_units=self._matched_units(
                        query.units,
                        normalized_text,
                    ),
                    matched_standard_codes=self._matched_standards(
                        query.standard_codes,
                        normalized_text,
                    ),
                    row_index=row_index,
                    source_page=self._get_source_page(table),
                    report_id=self._get_report_id(table),
                    metadata={
                        **metadata,
                        "result_level": "row",
                        "score_components": score_components,
                        "quality_multiplier": quality_multiplier,
                        "quality_metadata": quality_metadata,
                        "row_values": tuple(row_values),
                        "row_headers": tuple(headers),
                        "row_title": title,
                        "direct_metric_match": direct_metric_match,
                    },
                ),
            )

        results.sort(
            key=lambda result: (
                not bool(
                    result.metadata.get("direct_metric_match")
                ),
                -result.score,
            ),
        )

        return results

    def _calculate_score_components(
        self,
        *,
        query: TableQuery,
        candidate_text: str,
        title: str,
        headers: Sequence[str],
        table: Any,
    ) -> dict[str, float]:
        query_terms = set(
            self._tokenize(query.normalized_text),
        )
        candidate_terms = set(
            self._tokenize(candidate_text),
        )

        lexical_score = self._lexical_similarity(
            query_terms,
            candidate_terms,
        )

        normalized_title = self.normalize_text(title)
        normalized_headers = self.normalize_text(
            " ".join(headers),
        )

        metric_score = self._collection_match_score(
            query.metric_terms,
            candidate_text,
        )
        year_score = self._collection_match_score(
            tuple(str(year) for year in query.years),
            candidate_text,
        )
        unit_score = self._collection_match_score(
            query.units,
            candidate_text,
        )
        standard_score = self._collection_match_score(
            query.standard_codes,
            candidate_text,
        )

        table_category = self._get_table_category(table)
        category_score = 0.0

        if query.category:
            category_score = (
                1.0
                if self.normalize_text(query.category)
                == self.normalize_text(table_category or "")
                else 0.0
            )

        structure_score = self._structure_score(
            table,
            candidate_text,
        )

        if (
            query.normalized_text
            and query.normalized_text in candidate_text
        ):
            lexical_score = min(
                1.0,
                lexical_score + self.config.exact_phrase_bonus,
            )

        if (
            normalized_title
            and query_terms.intersection(
                self._tokenize(normalized_title),
            )
        ):
            lexical_score = min(
                1.0,
                lexical_score + self.config.title_match_bonus,
            )

        if (
            normalized_headers
            and query_terms.intersection(
                self._tokenize(normalized_headers),
            )
        ):
            lexical_score = min(
                1.0,
                lexical_score + self.config.header_match_bonus,
            )

        if self._NUMBER_PATTERN.search(candidate_text):
            structure_score = min(
                1.0,
                structure_score + self.config.numeric_content_bonus,
            )

        return {
            "lexical": round(lexical_score, 4),
            "metric": round(metric_score, 4),
            "year": round(year_score, 4),
            "unit": round(unit_score, 4),
            "standard": round(standard_score, 4),
            "category": round(category_score, 4),
            "structure": round(structure_score, 4),
        }

    @classmethod
    def _semantic_metric_score(
        cls,
        query_text: str,
        candidate_text: str,
    ) -> float:
        """
        Sorgu ile aday tablo/satırın aynı metrik ailesine ait olup
        olmadığını ölçer.

        Birebir cümle eşleşmesi gerekmez. Örneğin:
        - renewable electricity production
        - renewable energy generated
        aynı kavram olarak değerlendirilir.
        """

        normalized_query = cls.normalize_text(query_text)
        normalized_candidate = cls.normalize_text(candidate_text)

        query_concepts: set[str] = set()
        candidate_concepts: set[str] = set()

        for concept, aliases in cls._METRIC_CONCEPTS.items():
            normalized_aliases = tuple(
                cls.normalize_text(alias)
                for alias in aliases
            )

            if any(alias in normalized_query for alias in normalized_aliases):
                query_concepts.add(concept)

            if any(
                alias in normalized_candidate
                for alias in normalized_aliases
            ):
                candidate_concepts.add(concept)

        if not query_concepts or not candidate_concepts:
            return 0.0

        overlap = query_concepts.intersection(candidate_concepts)

        if not overlap:
            return 0.0

        return len(overlap) / len(query_concepts)

    def _apply_quantitative_adjustments(
        self,
        score: float,
        *,
        query: TableQuery,
        metric_score: float,
        candidate_text: str,
        has_number_or_placeholder: bool,
    ) -> float:
        adjusted = score
        numeric_intent = query.intent in {
            TableQueryIntent.VALUE,
            TableQueryIntent.RATIO,
            TableQueryIntent.COUNT,
            TableQueryIntent.COMPARISON,
            TableQueryIntent.TREND,
        }

        if numeric_intent and metric_score > 0.0 and has_number_or_placeholder:
            adjusted += self.config.metric_numeric_pair_bonus
            adjusted += (
                self.config.semantic_metric_bonus
                * min(metric_score, 1.0)
            )

        if query.years and any(
            str(year) in candidate_text for year in query.years
        ):
            adjusted += self.config.requested_year_bonus

        generic_terms = (
            "award", "awards", "project", "initiative",
            "recognition", "platform", "brand", "campaign",
        )
        if (
            any(term in candidate_text for term in generic_terms)
            and metric_score < 0.2
        ):
            adjusted *= self.config.generic_narrative_penalty

        return max(0.0, min(adjusted, 1.0))

    def _combine_scores(
        self,
        components: Mapping[str, float],
    ) -> float:
        weights = {
            "lexical": self.config.lexical_weight,
            "metric": self.config.metric_weight,
            "year": self.config.year_weight,
            "unit": self.config.unit_weight,
            "standard": self.config.standard_weight,
            "category": self.config.category_weight,
            "structure": self.config.structure_weight,
        }

        active_weights: dict[str, float] = {}

        # Metrik/yıl/birim eşleşmediğinde ağırlığı paydadan çıkarmak,
        # alakasız fakat sayısal tabloların puanını yapay olarak yükseltiyordu.
        # Bütün sinyaller sabit paydada tutulur.
        active_weights = dict(weights)

        if not active_weights:
            return 0.0

        total_weight = sum(active_weights.values())

        weighted_score = sum(
            components.get(name, 0.0) * weight
            for name, weight in active_weights.items()
        )

        return max(
            0.0,
            min(weighted_score / total_weight, 1.0),
        )

    def _resolve_tables(
        self,
        *,
        tables: Iterable[Any] | None,
        report_id: str | None,
        category: str | None,
        topic: str | None,
    ) -> Iterable[Any]:
        if tables is not None:
            return tables

        if self.repository is None:
            raise TableRetrieverError(
                "Tablo kaynağı belirtilmedi. tables veya repository "
                "sağlanmalıdır.",
            )

        find_method = getattr(self.repository, "find", None)

        if callable(find_method):
            try:
                return find_method(
                    report_id=report_id,
                    category=category,
                    topic=topic,
                )
            except TypeError:
                pass

        all_method = getattr(self.repository, "all", None)

        if callable(all_method):
            return all_method()

        raise TableRetrieverError(
            "Repository find veya all metodu sağlamıyor.",
        )

    def _matches_hard_filters(
        self,
        table: Any,
        query: TableQuery,
    ) -> bool:
        if query.report_id:
            table_report_id = self._get_report_id(table)

            if (
                table_report_id is not None
                and str(table_report_id).strip()
                != str(query.report_id).strip()
            ):
                return False

        if query.category:
            table_category = self._get_table_category(table)

            if (
                table_category is not None
                and self.normalize_text(table_category)
                != self.normalize_text(query.category)
            ):
                return False

        if query.topic:
            metadata = self._get_metadata(table)
            topic_values = {
                self.normalize_text(metadata.get("topic", "")),
                *(
                    self.normalize_text(item)
                    for item in metadata.get("topics", ()) or ()
                ),
            }
            topic_values.discard("")
            if topic_values and self.normalize_text(query.topic) not in topic_values:
                return False

        return True

    @staticmethod
    def _lexical_similarity(
        query_terms: set[str],
        candidate_terms: set[str],
    ) -> float:
        if not query_terms or not candidate_terms:
            return 0.0

        overlap = query_terms.intersection(candidate_terms)

        if not overlap:
            return 0.0

        recall = len(overlap) / len(query_terms)
        precision = len(overlap) / len(candidate_terms)

        if recall + precision == 0:
            return 0.0

        return 2 * recall * precision / (recall + precision)

    @classmethod
    def _direct_row_metric_match(
        cls,
        query_text: str,
        row_text: str,
    ) -> bool:
        normalized_query = cls.normalize_text(query_text)
        normalized_row = cls.normalize_text(row_text)

        if not normalized_query or not normalized_row:
            return False

        concept_aliases = {
            "nox": ("nox", "nitrogen oxides"),
            "sox": ("sox", "sulfur oxides", "sulphur oxides"),
            "voc": ("voc", "volatile organic compounds"),
            "pm": ("pm", "particulate matter"),
            "pop": ("pop", "persistent organic pollutants"),
            "hap": ("hap", "hazardous air pollutants"),
        }

        for aliases in concept_aliases.values():
            query_matches = any(alias in normalized_query for alias in aliases)
            row_matches = any(alias in normalized_row for alias in aliases)

            if query_matches and row_matches:
                return True

        return False

    @classmethod
    def _collection_match_score(
        cls,
        expected_values: Sequence[Any],
        candidate_text: str,
    ) -> float:
        if not expected_values:
            return 0.0

        normalized_candidate = cls.normalize_text(
            candidate_text,
        )

        normalized_expected = {
            cls.normalize_text(str(value))
            for value in expected_values
            if cls.normalize_text(str(value))
        }

        if not normalized_expected:
            return 0.0

        matched = {
            value
            for value in normalized_expected
            if value in normalized_candidate
        }

        return len(matched) / len(normalized_expected)

    def _structure_score(
        self,
        table: Any,
        candidate_text: str,
    ) -> float:
        score = 0.0

        headers = self._get_table_headers(table)
        rows = self._get_table_rows(table)

        if headers:
            score += 0.3

        if rows:
            score += 0.25

        if len(headers) >= 2:
            score += 0.15

        if len(rows) >= 2:
            score += 0.15

        if self._numeric_density(candidate_text) >= 0.1:
            score += 0.15

        return min(score, 1.0)

    def _infer_metric_terms(
        self,
        normalized_query: str,
    ) -> set[str]:
        stop_terms = {
            "nedir",
            "kaçtır",
            "kaç",
            "ne",
            "hangi",
            "what",
            "how",
            "which",
            "the",
            "bir",
            "bu",
            "şirketin",
            "company",
            "report",
            "rapor",
            "year",
            "yıl",
            "yılı",
            "değeri",
            "value",
            "amount",
            "miktarı",
        }

        terms = [
            term
            for term in self._tokenize(normalized_query)
            if len(term) > 2 and term not in stop_terms
        ]

        ngrams: set[str] = set(terms)

        for size in (2, 3):
            for index in range(len(terms) - size + 1):
                ngrams.add(
                    " ".join(terms[index : index + size]),
                )

        return ngrams

    def _detect_intent(
        self,
        normalized_query: str,
    ) -> TableQueryIntent:
        best_intent = TableQueryIntent.UNKNOWN
        best_score = 0

        for intent, terms in self._INTENT_TERMS.items():
            score = sum(
                1
                for term in terms
                if self.normalize_text(term) in normalized_query
            )

            if score > best_score:
                best_intent = intent
                best_score = score

        return best_intent

    def _extract_units(
        self,
        text: str,
    ) -> tuple[str, ...]:
        normalized_text = self.normalize_text(text)

        units = {
            unit
            for unit in self._COMMON_UNITS
            if self.normalize_text(unit) in normalized_text
        }

        return tuple(sorted(units))

    @staticmethod
    def _is_valid_year(value: Any) -> bool:
        try:
            year = int(value)
        except (TypeError, ValueError):
            return False

        return 1900 <= year <= 2100

    def _matched_terms(
        self,
        query: TableQuery,
        candidate_text: str,
    ) -> tuple[str, ...]:
        candidates = {
            query.text,
            *query.metric_terms,
        }

        matched = {
            term
            for term in candidates
            if self.normalize_text(term)
            and self.normalize_text(term) in candidate_text
        }

        return tuple(sorted(matched))

    @classmethod
    def _matched_years(
        cls,
        years: Sequence[int],
        candidate_text: str,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                year
                for year in years
                if str(year) in candidate_text
            ),
        )

    @classmethod
    def _matched_units(
        cls,
        units: Sequence[str],
        candidate_text: str,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                unit
                for unit in units
                if cls.normalize_text(unit) in candidate_text
            ),
        )

    @classmethod
    def _matched_standards(
        cls,
        standards: Sequence[str],
        candidate_text: str,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                standard
                for standard in standards
                if cls.normalize_text(standard) in candidate_text
            ),
        )

    @staticmethod
    def _deduplicate_results(
        results: Iterable[TableSearchResult],
    ) -> list[TableSearchResult]:
        deduplicated: dict[
            tuple[str, int | None, int | None],
            TableSearchResult,
        ] = {}

        for result in results:
            key = (
                result.table_id,
                result.row_index,
                result.column_index,
            )

            existing = deduplicated.get(key)

            if (
                existing is None
                or result.score > existing.score
            ):
                deduplicated[key] = result

        return list(deduplicated.values())

    @classmethod
    def _build_table_text(cls, table: Any) -> str:
        parts: list[str] = []

        title = cls._get_table_title(table)
        if title:
            parts.append(title)

        caption = cls._get_first_value(
            table,
            ("caption", "table_caption", "description"),
        )
        if caption:
            parts.append(str(caption))

        headers = cls._get_table_headers(table)
        if headers:
            parts.append(" | ".join(headers))

        for row in cls._get_table_rows(table):
            values = cls._row_to_values(row)

            if values:
                parts.append(" | ".join(values))

        metric_records = cls._get_first_value(
            table,
            ("metric_records",),
        ) or ()

        for record in metric_records:
            if isinstance(record, Mapping):
                metric_name = record.get("metric_name")
                raw_value = (
                    record.get("normalized_value")
                    or record.get("raw_value")
                    or record.get("numeric_value")
                )
                year = record.get("year")
                unit = record.get("unit")
            else:
                metric_name = getattr(record, "metric_name", None)
                raw_value = (
                    getattr(record, "normalized_value", None)
                    or getattr(record, "raw_value", None)
                    or getattr(record, "numeric_value", None)
                )
                year = getattr(record, "year", None)
                unit = getattr(record, "unit", None)

            metric_line = " | ".join(
                str(value)
                for value in (
                    metric_name,
                    raw_value,
                    year,
                    unit,
                )
                if value is not None and str(value).strip()
            )
            if metric_line:
                parts.append(metric_line)

        metadata = cls._get_metadata(table)

        for key in (
            "category",
            "topic",
            "topics",
            "section_title",
            "standards",
            "standard_codes",
            "metric",
            "unit",
            "years",
        ):
            value = metadata.get(key)

            if value is None:
                continue

            if isinstance(value, (list, tuple, set)):
                parts.append(
                    " ".join(str(item) for item in value),
                )
            else:
                parts.append(str(value))

        return "\n".join(
            part
            for part in parts
            if cls._clean_text(part)
        )

    @classmethod
    def _get_table_id(cls, table: Any) -> str:
        value = cls._get_first_value(
            table,
            ("table_id", "id", "identifier"),
        )

        if value is None:
            metadata = cls._get_metadata(table)
            value = (
                metadata.get("table_id")
                or metadata.get("id")
            )

        if value is None or not str(value).strip():
            raise TableRetrieverError(
                "Tablo üzerinde table_id veya id alanı bulunamadı.",
            )

        return str(value).strip()

    @classmethod
    def _get_table_title(cls, table: Any) -> str:
        value = cls._get_first_value(
            table,
            (
                "title",
                "table_title",
                "caption",
                "name",
            ),
        )

        if value is None:
            metadata = cls._get_metadata(table)
            value = (
                metadata.get("title")
                or metadata.get("table_title")
                or metadata.get("caption")
            )

        return cls._clean_text(value)

    @classmethod
    def _get_table_headers(
        cls,
        table: Any,
    ) -> tuple[str, ...]:
        value = cls._get_first_value(
            table,
            (
                "headers",
                "column_headers",
                "columns",
            ),
        )

        if value is None:
            metadata = cls._get_metadata(table)
            value = (
                metadata.get("headers")
                or metadata.get("column_headers")
            )

        if value is None:
            return ()

        if isinstance(value, Mapping):
            return tuple(
                cls._clean_text(item)
                for item in value.keys()
                if cls._clean_text(item)
            )

        if isinstance(value, (str, bytes)):
            cleaned = cls._clean_text(value)
            return (cleaned,) if cleaned else ()

        try:
            values = list(value)
        except TypeError:
            cleaned = cls._clean_text(value)
            return (cleaned,) if cleaned else ()

        headers: list[str] = []

        for item in values:
            if isinstance(item, Mapping):
                header = (
                    item.get("name")
                    or item.get("title")
                    or item.get("label")
                    or item.get("normalized_text")
                    or item.get("raw_text")
                    or item.get("text")
                )
            elif isinstance(item, (str, bytes)):
                header = item
            else:
                header = (
                    getattr(item, "name", None)
                    or getattr(item, "label", None)
                    or getattr(item, "normalized_text", None)
                    or getattr(item, "raw_text", None)
                    or getattr(item, "text", None)
                    or item
                )

            cleaned = cls._clean_text(header)

            if cleaned:
                headers.append(cleaned)

        return tuple(headers)

    @classmethod
    def _get_table_rows(
        cls,
        table: Any,
    ) -> list[Any]:
        value = cls._get_first_value(
            table,
            (
                "rows",
                "data",
                "records",
                "table_rows",
            ),
        )

        if value is None:
            return []

        if isinstance(value, Mapping):
            return list(value.values())

        if isinstance(value, (str, bytes)):
            return []

        try:
            return list(value)
        except TypeError:
            return []

    @classmethod
    def _row_to_values(
        cls,
        row: Any,
    ) -> list[str]:
        if row is None:
            return []

        if isinstance(row, Mapping):
            iterable = row.values()
        elif isinstance(row, (str, bytes)):
            cleaned = cls._clean_text(row)
            return [cleaned] if cleaned else []
        else:
            cells = getattr(row, "cells", None)

            if cells is not None:
                iterable = cells
            else:
                try:
                    iterable = list(row)
                except TypeError:
                    iterable = [row]

        values: list[str] = []

        for item in iterable:
            if isinstance(item, Mapping):
                value = (
                    item.get("normalized_text")
                    or item.get("raw_text")
                    or item.get("search_text")
                    or item.get("normalized_value")
                    or item.get("value")
                    or item.get("text")
                    or item.get("content")
                )
            elif isinstance(item, (str, bytes)):
                value = item
            else:
                value = (
                    getattr(item, "normalized_text", None)
                    or getattr(item, "raw_text", None)
                    or getattr(item, "search_text", None)
                    or getattr(item, "normalized_value", None)
                    or getattr(item, "value", None)
                    or getattr(item, "text", None)
                    or getattr(item, "content", None)
                )

                # Gerçek bir hücre modeli olup bütün metin alanları boşsa
                # nesnenin repr çıktısını retrieval metnine ekleme.
                if value is None and hasattr(item, "is_empty"):
                    continue

                if value is None:
                    value = item

            cleaned = cls._clean_text(value)

            if cleaned:
                values.append(cleaned)

        return values

    @classmethod
    def _get_source_page(cls, table: Any) -> int | None:
        value = cls._get_first_value(
            table,
            (
                "page_number",
                "source_page",
                "page",
            ),
        )

        if value is None:
            metadata = cls._get_metadata(table)
            value = (
                metadata.get("page_number")
                or metadata.get("source_page")
                or metadata.get("page")
            )

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _get_report_id(cls, table: Any) -> str | None:
        value = cls._get_first_value(
            table,
            ("report_id",),
        )

        if value is None:
            value = cls._get_metadata(table).get("report_id")

        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    def _table_quality_multiplier(
        self,
        table: Any,
    ) -> tuple[float, dict[str, Any]]:
        metadata = self._get_metadata(table)
        numeric_validation = metadata.get(
            "numeric_validation"
        )

        if not isinstance(numeric_validation, Mapping):
            numeric_validation = {}

        explicit_invalid = any(
            value is False
            for value in (
                metadata.get("is_valid"),
                numeric_validation.get("is_valid"),
            )
        )

        raw_score = (
            numeric_validation.get("validation_score")
            if numeric_validation
            else metadata.get("validation_score")
        )

        validation_score: float | None

        try:
            validation_score = (
                float(raw_score)
                if raw_score is not None
                else None
            )
        except (TypeError, ValueError):
            validation_score = None

        if validation_score is not None:
            validation_score = max(
                0.0,
                min(validation_score, 1.0),
            )

        multiplier = 1.0

        if explicit_invalid:
            multiplier *= (
                self.config.invalid_table_score_multiplier
            )

        if validation_score is not None:
            multiplier *= max(
                self.config.minimum_table_quality_multiplier,
                validation_score,
            )

        multiplier = max(
            self.config.minimum_table_quality_multiplier,
            min(multiplier, 1.0),
        )

        return (
            multiplier,
            {
                "explicitly_invalid": explicit_invalid,
                "validation_score": validation_score,
                "source_is_valid": metadata.get("is_valid"),
                "numeric_validation_is_valid": (
                    numeric_validation.get("is_valid")
                ),
            },
        )

    @classmethod
    def _get_table_category(cls, table: Any) -> str | None:
        value = cls._get_first_value(
            table,
            ("category", "esg_category"),
        )

        if value is None:
            metadata = cls._get_metadata(table)
            value = (
                metadata.get("category")
                or metadata.get("esg_category")
            )

        if value is None:
            return None

        enum_value = getattr(value, "value", value)
        cleaned = cls._clean_text(enum_value)

        return cleaned or None

    @staticmethod
    def _get_metadata(table: Any) -> dict[str, Any]:
        metadata = getattr(table, "metadata", None)

        if metadata is None and isinstance(table, Mapping):
            metadata = table.get("metadata")

        if metadata is None:
            return {}

        if isinstance(metadata, Mapping):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(
                metadata.model_dump(exclude_none=True),
            )

        if hasattr(metadata, "dict"):
            return dict(metadata.dict(exclude_none=True))

        return {}

    @staticmethod
    def _get_first_value(
        obj: Any,
        attribute_names: Sequence[str],
    ) -> Any:
        for attribute_name in attribute_names:
            if isinstance(obj, Mapping):
                value = obj.get(attribute_name)
            else:
                value = getattr(obj, attribute_name, None)

            if value is not None and not callable(value):
                return value

        return None

    @classmethod
    def _numeric_density(cls, text: str) -> float:
        cleaned = cls._clean_text(text)

        if not cleaned:
            return 0.0

        numeric_characters = sum(
            character.isdigit()
            for character in cleaned
        )

        non_space_characters = sum(
            not character.isspace()
            for character in cleaned
        )

        if non_space_characters == 0:
            return 0.0

        return numeric_characters / non_space_characters

    @staticmethod
    def _truncate_text(
        text: str,
        maximum_length: int = 1500,
    ) -> str:
        cleaned = TableRetriever._clean_text(text)

        if len(cleaned) <= maximum_length:
            return cleaned

        return f"{cleaned[: maximum_length - 3].rstrip()}..."

    @staticmethod
    def _normalize_standard_code(value: str) -> str:
        normalized = " ".join(
            str(value).strip().upper().split(),
        )

        normalized = re.sub(
            r"\s*[-:]\s*",
            " ",
            normalized,
            count=1,
        )

        return normalized

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        return tuple(
            token
            for token in re.findall(
                r"[a-z0-9]+(?:[.-][a-z0-9]+)*",
                text,
            )
            if token
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
    def normalize_text(value: str) -> str:
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

        normalized = re.sub(
            r"[^\w\s%./+-]",
            " ",
            normalized,
        )

        normalized = " ".join(normalized.split())

        abbreviation_patterns = {
            r"\bno\s+x\b": "nox",
            r"\bso\s+x\b": "sox",
            r"\bco\s+2\b": "co2",
        }

        for pattern, replacement in abbreviation_patterns.items():
            normalized = re.sub(
                pattern,
                replacement,
                normalized,
            )

        return normalized


__all__ = [
    "TableQuery",
    "TableQueryIntent",
    "TableRepositoryProtocol",
    "TableRetriever",
    "TableRetrieverConfig",
    "TableRetrieverError",
    "TableSearchResult",
]