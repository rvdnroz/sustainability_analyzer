from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .question_catalog import QuestionDefinition
from retrieval.query_expander import (
    QueryExpander,
    QueryExpansionResult,
)
from retrieval.standard_router import (
    StandardRoute,
    StandardRouter,
)
from standards.esg_ontology import ESGPillar


class QuestionIntent(StrEnum):
    VALUE_LOOKUP = "value_lookup"
    COMPARISON = "comparison"
    TREND = "trend"
    TARGET = "target"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    LIST = "list"
    BOOLEAN = "boolean"
    STANDARD_DISCLOSURE = "standard_disclosure"
    UNKNOWN = "unknown"


class ExpectedAnswerType(StrEnum):
    NUMBER = "number"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    BOOLEAN = "boolean"
    TEXT = "text"
    LIST = "list"
    COMPARISON = "comparison"
    TREND = "trend"
    TARGET = "target"
    UNKNOWN = "unknown"


class RetrievalStrategy(StrEnum):
    TABLE_FIRST = "table_first"
    TEXT_FIRST = "text_first"
    STANDARD_FIRST = "standard_first"
    HYBRID = "hybrid"


@dataclass(slots=True, frozen=True)
class QuestionMetric:
    name: str
    normalized_name: str
    aliases: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    standard_codes: tuple[str, ...] = ()
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class QuestionAnalysis:
    original_question: str
    normalized_question: str

    intent: QuestionIntent
    expected_answer_type: ExpectedAnswerType
    retrieval_strategy: RetrievalStrategy

    question_id: int | None = None
    question_key: str | None = None
    question_group: str | None = None
    topic: str | None = None
    retrieval_query: str = ""
    aliases: tuple[str, ...] = ()
    preferred_sources: tuple[str, ...] = ()
    preferred_document_types: tuple[str, ...] = ()

    years: tuple[int, ...] = ()
    metrics: tuple[QuestionMetric, ...] = ()
    units: tuple[str, ...] = ()
    standard_codes: tuple[str, ...] = ()
    pillars: tuple[ESGPillar, ...] = ()

    company_name: str | None = None
    report_id: str | None = None
    category: str | None = None

    expects_numeric_answer: bool = False
    expects_table_evidence: bool = False
    expects_standard_evidence: bool = False

    expanded_query: QueryExpansionResult | None = None
    standard_route: StandardRoute | None = None

    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(metric.name for metric in self.metrics)

    @property
    def is_catalog_question(self) -> bool:
        return self.question_id is not None and self.question_key is not None


@dataclass(slots=True, frozen=True)
class QuestionAnalyzerConfig:
    enable_query_expansion: bool = True
    enable_standard_routing: bool = True

    maximum_metrics: int = 8
    maximum_standard_codes: int = 12

    default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID

    def __post_init__(self) -> None:
        if self.maximum_metrics <= 0:
            raise ValueError("maximum_metrics pozitif olmalıdır.")

        if self.maximum_standard_codes <= 0:
            raise ValueError("maximum_standard_codes pozitif olmalıdır.")


class QuestionAnalyzerError(RuntimeError):
    """Soru analizi sırasında oluşan temel hata."""


class QuestionAnalyzer:
    """
    Kullanıcı sorusunu retrieval ve cevap üretme katmanlarının
    kullanabileceği yapılandırılmış forma dönüştürür.

    Serbest sorularda kural tabanlı analiz ve query expansion uygulanır.
    QuestionDefinition verildiğinde katalogdaki açık metadata önceliklidir;
    kural tabanlı çıkarım yalnızca destekleyici veya yedek sinyal olarak
    kullanılır.
    """

    _YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    _PERCENTAGE_PATTERN = re.compile(
        r"(?:%|\byüzde\b|\boran\b|\bpercent(?:age)?\b|\brate\b)",
        re.IGNORECASE,
    )

    _CURRENCY_PATTERN = re.compile(
        r"(?:₺|\$|€|\bTRY\b|\bTL\b|\bUSD\b|\bEUR\b|"
        r"\bgelir\b|\bmaliyet\b|\brevenue\b|\bcost\b|\bprofit\b|"
        r"\bkâr\b|\bkar\b)",
        re.IGNORECASE,
    )

    _BOOLEAN_PATTERN = re.compile(
        r"\b(?:"
        r"var mı|mevcut mu|bulunuyor mu|raporlandı mı|açıklandı mı|"
        r"is there|are there|does|did|has|have|whether"
        r")\b",
        re.IGNORECASE,
    )

    _COMPARISON_TERMS = (
        "karşılaştır",
        "karşılaştırma",
        "kıyasla",
        "fark",
        "daha yüksek",
        "daha düşük",
        "compare",
        "comparison",
        "versus",
        "difference",
        "higher",
        "lower",
    )

    _TREND_TERMS = (
        "trend",
        "eğilim",
        "değişim",
        "artış",
        "azalış",
        "yıllara göre",
        "yıllar içinde",
        "increase",
        "decrease",
        "change",
        "growth",
        "reduction",
        "over the years",
    )

    _TARGET_TERMS = (
        "hedef",
        "taahhüt",
        "baz yıl",
        "hedef yıl",
        "target",
        "goal",
        "commitment",
        "baseline",
        "base year",
    )

    _DEFINITION_TERMS = (
        "nedir",
        "ne demek",
        "tanımı",
        "anlamı",
        "what is",
        "what does",
        "define",
        "definition",
    )

    _EXPLANATION_TERMS = (
        "açıkla",
        "nasıl",
        "neden",
        "hangi yöntemle",
        "explain",
        "describe",
        "how",
        "why",
    )

    _LIST_TERMS = (
        "nelerdir",
        "hangileri",
        "listele",
        "sırala",
        "list",
        "which",
        "what are",
    )

    _VALUE_TERMS = (
        "kaç",
        "ne kadar",
        "değer",
        "miktar",
        "toplam",
        "sayısı",
        "value",
        "amount",
        "total",
        "how much",
        "how many",
    )

    _COMMON_UNITS = (
        "tCO2e",
        "CO2e",
        "kg CO2e",
        "ton",
        "tonne",
        "tonnes",
        "kg",
        "g",
        "mg",
        "MWh",
        "GWh",
        "kWh",
        "GJ",
        "TJ",
        "MJ",
        "m3",
        "m³",
        "litre",
        "liter",
        "litres",
        "liters",
        "hectare",
        "ha",
        "%",
        "percent",
        "percentage",
        "oran",
        "adet",
        "kişi",
        "person",
        "people",
        "employee",
        "employees",
        "hour",
        "hours",
        "saat",
        "TRY",
        "TL",
        "USD",
        "EUR",
    )

    def __init__(
        self,
        *,
        query_expander: QueryExpander | None = None,
        standard_router: StandardRouter | None = None,
        config: QuestionAnalyzerConfig | None = None,
    ) -> None:
        self.config = config or QuestionAnalyzerConfig()
        self.query_expander = query_expander or QueryExpander()
        self.standard_router = standard_router or StandardRouter()

    def analyze(
        self,
        question: str,
        *,
        company_name: str | None = None,
        report_id: str | None = None,
        report_year: int | None = None,
        category: str | None = None,
        definition: QuestionDefinition | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionAnalysis:
        """
        Bir soruyu analiz eder.

        `definition` verilmezse önceki serbest-soru davranışı korunur.
        Verilirse soru kataloğundaki metrik, alias, birim, kategori ve
        retrieval stratejisi tahmin edilen değerlerin önüne geçer.
        """

        cleaned_question = self._clean_text(question)
        if not cleaned_question:
            raise QuestionAnalyzerError("Analiz edilecek soru boş olamaz.")

        if definition is not None:
            self._validate_definition_question(cleaned_question, definition)

        normalized_question = self.normalize_text(cleaned_question)
        retrieval_query = self._build_retrieval_query(
            cleaned_question,
            definition,
        )

        years = self._extract_years(cleaned_question)
        if report_year is not None:
            if not self._is_valid_year(report_year):
                raise ValueError(f"Geçersiz rapor yılı: {report_year}")
            years.add(int(report_year))

        expanded_query = self._expand_query(retrieval_query)
        standard_route = self._route_standard(retrieval_query)

        collected_standard_codes = self._collect_standard_codes(
            expanded_query=expanded_query,
            standard_route=standard_route,
        )
        has_explicit_standard_signal = self._has_explicit_standard_signal(
            standard_route
        )
        standard_codes = (
            collected_standard_codes
            if definition is None or has_explicit_standard_signal
            else set()
        )

        pillars = self._collect_pillars(
            expanded_query=expanded_query,
            standard_route=standard_route,
        )

        resolved_category = (
            self._clean_text(category)
            or (definition.esg_category if definition is not None else "")
            or self._category_from_pillars(pillars)
        )

        catalog_pillar = self._pillar_from_category(resolved_category)
        if catalog_pillar is not None:
            pillars.add(catalog_pillar)

        metrics = self._extract_metrics(
            expanded_query,
            definition=definition,
        )

        units = self._extract_units(
            retrieval_query,
            metrics,
            expected_units=(
                definition.expected_units if definition is not None else ()
            ),
        )

        intent = self._detect_intent(
            normalized_question,
            standard_route,
            allow_inferred_standard_intent=definition is None,
        )
        if definition is not None:
            intent = self._resolve_catalog_intent(definition, intent)

        inferred_answer_type = self._detect_answer_type(
            cleaned_question,
            intent,
        )
        expected_answer_type = (
            self._coerce_answer_type(definition.expected_answer_type)
            if definition is not None
            else inferred_answer_type
        )

        expects_numeric_answer = self._expects_numeric_answer(
            intent,
            expected_answer_type,
        )
        expects_table_evidence = self._expects_table_evidence(
            intent,
            expected_answer_type,
        )
        expects_standard_evidence = bool(
            standard_route
            and (
                intent == QuestionIntent.STANDARD_DISCLOSURE
                or has_explicit_standard_signal
            )
        )

        inferred_strategy = self._select_retrieval_strategy(
            intent=intent,
            expects_table_evidence=expects_table_evidence,
            expects_standard_evidence=expects_standard_evidence,
        )
        retrieval_strategy = (
            self._coerce_retrieval_strategy(definition.retrieval_strategy)
            if definition is not None
            else inferred_strategy
        )

        confidence = self._calculate_confidence(
            intent=intent,
            expected_answer_type=expected_answer_type,
            metrics=metrics,
            years=years,
            standard_codes=standard_codes,
            pillars=pillars,
            catalog_backed=definition is not None,
        )

        catalog_metadata = self._definition_metadata(definition)
        analysis_metadata = {
            **catalog_metadata,
            **dict(metadata or {}),
            "analysis_source": (
                "question_catalog" if definition is not None else "inferred"
            ),
            "metric_count": len(metrics),
            "year_count": len(years),
            "unit_count": len(units),
            "standard_code_count": len(standard_codes),
            "inferred_standard_codes": sorted(collected_standard_codes),
            "has_explicit_standard_signal": has_explicit_standard_signal,
            "pillar_count": len(pillars),
            "inferred_answer_type": inferred_answer_type.value,
            "inferred_retrieval_strategy": inferred_strategy.value,
        }

        return QuestionAnalysis(
            original_question=cleaned_question,
            normalized_question=normalized_question,
            intent=intent,
            expected_answer_type=expected_answer_type,
            retrieval_strategy=retrieval_strategy,
            question_id=(definition.id if definition is not None else None),
            question_key=(definition.key if definition is not None else None),
            question_group=(
                definition.group if definition is not None else None
            ),
            topic=(definition.topic if definition is not None else None),
            retrieval_query=retrieval_query,
            aliases=(definition.aliases if definition is not None else ()),
            preferred_sources=(
                definition.preferred_sources if definition is not None else ()
            ),
            preferred_document_types=(
                definition.preferred_document_types
                if definition is not None
                else ()
            ),
            years=tuple(sorted(years)),
            metrics=tuple(metrics),
            units=tuple(sorted(units, key=self.normalize_text)),
            standard_codes=tuple(sorted(standard_codes)),
            pillars=tuple(sorted(pillars, key=lambda pillar: pillar.value)),
            company_name=self._optional_string(company_name),
            report_id=self._optional_string(report_id),
            category=self._optional_string(resolved_category),
            expects_numeric_answer=expects_numeric_answer,
            expects_table_evidence=expects_table_evidence,
            expects_standard_evidence=expects_standard_evidence,
            expanded_query=expanded_query,
            standard_route=standard_route,
            confidence=confidence,
            metadata=analysis_metadata,
        )

    def analyze_definition(
        self,
        definition: QuestionDefinition,
        *,
        company_name: str | None = None,
        report_id: str | None = None,
        report_year: int | None = None,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionAnalysis:
        """Katalogdaki sabit bir soruyu açık metadata ile analiz eder."""

        if not isinstance(definition, QuestionDefinition):
            raise TypeError("definition bir QuestionDefinition olmalıdır.")

        return self.analyze(
            definition.question,
            company_name=company_name,
            report_id=report_id,
            report_year=report_year,
            category=category,
            definition=definition,
            metadata=metadata,
        )

    def analyze_many(
        self,
        questions: Sequence[str],
        *,
        company_name: str | None = None,
        report_id: str | None = None,
        report_year: int | None = None,
    ) -> list[QuestionAnalysis]:
        analyses: list[QuestionAnalysis] = []

        for question in questions:
            if not isinstance(question, str) or not question.strip():
                continue
            analyses.append(
                self.analyze(
                    question,
                    company_name=company_name,
                    report_id=report_id,
                    report_year=report_year,
                )
            )

        return analyses

    def analyze_definitions(
        self,
        definitions: Sequence[QuestionDefinition],
        *,
        company_name: str | None = None,
        report_id: str | None = None,
        report_year: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[QuestionAnalysis]:
        """Bir soru grubundaki katalog tanımlarını sırayla analiz eder."""

        return [
            self.analyze_definition(
                definition,
                company_name=company_name,
                report_id=report_id,
                report_year=report_year,
                metadata=metadata,
            )
            for definition in definitions
        ]

    def _expand_query(
        self,
        question: str,
    ) -> QueryExpansionResult | None:
        if not self.config.enable_query_expansion:
            return None

        try:
            return self.query_expander.expand(question)
        except Exception:
            return None

    def _route_standard(
        self,
        question: str,
    ) -> StandardRoute | None:
        if not self.config.enable_standard_routing:
            return None

        try:
            return self.standard_router.route(question)
        except Exception:
            return None

    def _extract_metrics(
        self,
        expansion: QueryExpansionResult | None,
        *,
        definition: QuestionDefinition | None = None,
    ) -> list[QuestionMetric]:
        metrics: list[QuestionMetric] = []
        seen: set[str] = set()

        if definition is not None:
            catalog_name = definition.key.replace("_", " ")
            normalized_name = self.normalize_text(catalog_name)
            seen.add(normalized_name)
            metrics.append(
                QuestionMetric(
                    name=catalog_name,
                    normalized_name=normalized_name,
                    aliases=definition.aliases,
                    units=definition.expected_units,
                    confidence=1.0,
                    metadata={
                        "source": "question_catalog",
                        "question_id": definition.id,
                        "question_key": definition.key,
                        "group": definition.group,
                        "topic": definition.topic,
                    },
                )
            )

        if expansion is None:
            return metrics[: self.config.maximum_metrics]

        for metric_name in expansion.metrics:
            normalized_name = self.normalize_text(metric_name)
            if not normalized_name or normalized_name in seen:
                continue

            seen.add(normalized_name)
            metrics.append(
                QuestionMetric(
                    name=metric_name,
                    normalized_name=normalized_name,
                    standard_codes=tuple(expansion.standard_codes),
                    confidence=0.7,
                    metadata={"source": "query_expander"},
                )
            )

            if len(metrics) >= self.config.maximum_metrics:
                break

        return metrics

    def _collect_standard_codes(
        self,
        *,
        expanded_query: QueryExpansionResult | None,
        standard_route: StandardRoute | None,
    ) -> set[str]:
        codes: set[str] = set()

        if expanded_query is not None:
            codes.update(
                self._normalize_standard_code(code)
                for code in expanded_query.standard_codes
            )

        if standard_route is not None:
            codes.update(
                self._normalize_standard_code(code)
                for code in standard_route.standard_codes
            )

        codes.discard("")
        return set(
            sorted(codes)[: self.config.maximum_standard_codes]
        )

    @staticmethod
    def _collect_pillars(
        *,
        expanded_query: QueryExpansionResult | None,
        standard_route: StandardRoute | None,
    ) -> set[ESGPillar]:
        pillars: set[ESGPillar] = set()

        if expanded_query is not None:
            pillars.update(expanded_query.detected_pillars)

        if standard_route is not None:
            pillars.update(standard_route.pillars)

        pillars.discard(ESGPillar.UNKNOWN)
        return pillars

    def _extract_units(
        self,
        question: str,
        metrics: Sequence[QuestionMetric],
        *,
        expected_units: Sequence[str] = (),
    ) -> set[str]:
        normalized_question = self.normalize_text(question)
        units: set[str] = {
            self._clean_text(unit)
            for unit in expected_units
            if self._clean_text(unit)
        }

        for unit in self._COMMON_UNITS:
            if self._contains_normalized_term(
                normalized_question,
                self.normalize_text(unit),
            ):
                units.add(unit)

        expected_unit_set = {
            self._clean_text(unit)
            for unit in expected_units
            if self._clean_text(unit)
        }
        for metric in metrics:
            for unit in metric.units:
                if (
                    unit in expected_unit_set
                    or self._contains_normalized_term(
                        normalized_question,
                        self.normalize_text(unit),
                    )
                ):
                    units.add(unit)

        return units

    def _detect_intent(
        self,
        normalized_question: str,
        standard_route: StandardRoute | None,
        *,
        allow_inferred_standard_intent: bool = True,
    ) -> QuestionIntent:
        if (
            allow_inferred_standard_intent
            and standard_route is not None
            and standard_route.should_search_standard_index
        ):
            return QuestionIntent.STANDARD_DISCLOSURE

        intent_groups = (
            (QuestionIntent.COMPARISON, self._COMPARISON_TERMS),
            (QuestionIntent.TREND, self._TREND_TERMS),
            (QuestionIntent.TARGET, self._TARGET_TERMS),
            (QuestionIntent.DEFINITION, self._DEFINITION_TERMS),
            (QuestionIntent.LIST, self._LIST_TERMS),
            (QuestionIntent.EXPLANATION, self._EXPLANATION_TERMS),
            (QuestionIntent.VALUE_LOOKUP, self._VALUE_TERMS),
        )

        best_intent = QuestionIntent.UNKNOWN
        best_score = 0

        for intent, terms in intent_groups:
            score = sum(
                1
                for term in terms
                if self.normalize_text(term) in normalized_question
            )
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent != QuestionIntent.UNKNOWN:
            return best_intent

        if self._BOOLEAN_PATTERN.search(normalized_question):
            return QuestionIntent.BOOLEAN

        return QuestionIntent.UNKNOWN

    @classmethod
    def _resolve_catalog_intent(
        cls,
        definition: QuestionDefinition,
        inferred_intent: QuestionIntent,
    ) -> QuestionIntent:
        answer_type = cls._coerce_answer_type(
            definition.expected_answer_type
        )
        numeric_types = {
            ExpectedAnswerType.NUMBER,
            ExpectedAnswerType.PERCENTAGE,
            ExpectedAnswerType.CURRENCY,
        }

        if (
            answer_type in numeric_types
            and inferred_intent
            in {
                QuestionIntent.UNKNOWN,
                QuestionIntent.DEFINITION,
                QuestionIntent.EXPLANATION,
                QuestionIntent.STANDARD_DISCLOSURE,
            }
        ):
            return QuestionIntent.VALUE_LOOKUP

        return inferred_intent

    def _detect_answer_type(
        self,
        question: str,
        intent: QuestionIntent,
    ) -> ExpectedAnswerType:
        if intent == QuestionIntent.COMPARISON:
            return ExpectedAnswerType.COMPARISON
        if intent == QuestionIntent.TREND:
            return ExpectedAnswerType.TREND
        if intent == QuestionIntent.TARGET:
            return ExpectedAnswerType.TARGET
        if intent == QuestionIntent.LIST:
            return ExpectedAnswerType.LIST
        if intent == QuestionIntent.BOOLEAN:
            return ExpectedAnswerType.BOOLEAN
        if self._PERCENTAGE_PATTERN.search(question):
            return ExpectedAnswerType.PERCENTAGE
        if self._CURRENCY_PATTERN.search(question):
            return ExpectedAnswerType.CURRENCY
        if intent == QuestionIntent.VALUE_LOOKUP:
            return ExpectedAnswerType.NUMBER
        if intent in {
            QuestionIntent.DEFINITION,
            QuestionIntent.EXPLANATION,
            QuestionIntent.STANDARD_DISCLOSURE,
        }:
            return ExpectedAnswerType.TEXT
        return ExpectedAnswerType.TEXT

    @staticmethod
    def _expects_numeric_answer(
        intent: QuestionIntent,
        answer_type: ExpectedAnswerType,
    ) -> bool:
        return (
            intent
            in {
                QuestionIntent.VALUE_LOOKUP,
                QuestionIntent.COMPARISON,
                QuestionIntent.TREND,
                QuestionIntent.TARGET,
            }
            or answer_type
            in {
                ExpectedAnswerType.NUMBER,
                ExpectedAnswerType.PERCENTAGE,
                ExpectedAnswerType.CURRENCY,
                ExpectedAnswerType.COMPARISON,
                ExpectedAnswerType.TREND,
                ExpectedAnswerType.TARGET,
            }
        )

    @staticmethod
    def _expects_table_evidence(
        intent: QuestionIntent,
        answer_type: ExpectedAnswerType,
    ) -> bool:
        return (
            intent
            in {
                QuestionIntent.VALUE_LOOKUP,
                QuestionIntent.COMPARISON,
                QuestionIntent.TREND,
                QuestionIntent.TARGET,
            }
            or answer_type
            in {
                ExpectedAnswerType.NUMBER,
                ExpectedAnswerType.PERCENTAGE,
                ExpectedAnswerType.CURRENCY,
                ExpectedAnswerType.COMPARISON,
                ExpectedAnswerType.TREND,
                ExpectedAnswerType.TARGET,
            }
        )

    def _select_retrieval_strategy(
        self,
        *,
        intent: QuestionIntent,
        expects_table_evidence: bool,
        expects_standard_evidence: bool,
    ) -> RetrievalStrategy:
        if (
            intent == QuestionIntent.STANDARD_DISCLOSURE
            or expects_standard_evidence
        ):
            return RetrievalStrategy.STANDARD_FIRST

        if expects_table_evidence:
            return RetrievalStrategy.TABLE_FIRST

        if intent in {
            QuestionIntent.DEFINITION,
            QuestionIntent.EXPLANATION,
            QuestionIntent.LIST,
            QuestionIntent.BOOLEAN,
        }:
            return RetrievalStrategy.TEXT_FIRST

        return self.config.default_strategy

    @classmethod
    def _extract_years(cls, text: str) -> set[int]:
        return {
            int(match.group(0))
            for match in cls._YEAR_PATTERN.finditer(text)
        }

    @staticmethod
    def _category_from_pillars(
        pillars: set[ESGPillar],
    ) -> str | None:
        if len(pillars) != 1:
            return None

        pillar = next(iter(pillars))
        mapping = {
            ESGPillar.ENVIRONMENTAL: "environmental",
            ESGPillar.SOCIAL: "social",
            ESGPillar.GOVERNANCE: "governance",
            ESGPillar.ECONOMIC: "economic",
            ESGPillar.CROSS_CUTTING: "standards",
        }
        return mapping.get(pillar)

    @staticmethod
    def _pillar_from_category(
        category: str | None,
    ) -> ESGPillar | None:
        mapping = {
            "environmental": ESGPillar.ENVIRONMENTAL,
            "social": ESGPillar.SOCIAL,
            "governance": ESGPillar.GOVERNANCE,
            "economic": ESGPillar.ECONOMIC,
            "standards": ESGPillar.CROSS_CUTTING,
        }
        return mapping.get(str(category or "").strip().casefold())


    @staticmethod
    def _has_explicit_standard_signal(
        standard_route: StandardRoute | None,
    ) -> bool:
        if standard_route is None:
            return False

        metadata = dict(standard_route.metadata or {})
        try:
            explicit_code_count = int(
                metadata.get("explicit_code_count", 0)
            )
        except (TypeError, ValueError):
            explicit_code_count = 0

        return explicit_code_count > 0

    @staticmethod
    def _contains_normalized_term(
        normalized_text: str,
        normalized_term: str,
    ) -> bool:
        if not normalized_text or not normalized_term:
            return False

        pattern = re.compile(
            rf"(?<!\w){re.escape(normalized_term)}(?!\w)"
        )
        return pattern.search(normalized_text) is not None

    @staticmethod
    def _calculate_confidence(
        *,
        intent: QuestionIntent,
        expected_answer_type: ExpectedAnswerType,
        metrics: Sequence[QuestionMetric],
        years: set[int],
        standard_codes: set[str],
        pillars: set[ESGPillar],
        catalog_backed: bool,
    ) -> float:
        score = 0.2

        if intent != QuestionIntent.UNKNOWN:
            score += 0.2
        if expected_answer_type != ExpectedAnswerType.UNKNOWN:
            score += 0.15
        if metrics:
            score += 0.15
        if years:
            score += 0.08
        if standard_codes:
            score += 0.08
        if pillars:
            score += 0.06
        if catalog_backed:
            score += 0.18

        return round(min(score, 1.0), 4)

    @classmethod
    def _build_retrieval_query(
        cls,
        question: str,
        definition: QuestionDefinition | None,
    ) -> str:
        """
        Katalog soruları için kısa ve kontrollü bir retrieval sorgusu üretir.

        Bütün alias'ları tek sorguya eklemek vektör temsilini seyreltir.
        Ayrıca GRI/SASB gibi standart kodlarının sorguya eklenmesi,
        StandardRouter'ın zorunlu standard_codes filtresi oluşturmasına ve
        kodla etiketlenmemiş normal chunk'ların tamamen elenmesine yol açar.

        Bu nedenle:
        - Ana soru korunur.
        - Soru anahtarı doğal metne dönüştürülür.
        - original_header eklenir.
        - En fazla üç güçlü ve kısa alias seçilir.
        - Standart kodu içeren alias'lar retrieval sorgusundan çıkarılır.
        """

        if definition is None:
            return question

        selected_aliases = cls._select_retrieval_aliases(
            definition.aliases,
            maximum_aliases=3,
        )

        terms = cls._deduplicate_texts(
            (
                question,
                definition.key.replace("_", " "),
                definition.original_header,
                *selected_aliases,
            )
        )

        return " ".join(terms)

    @classmethod
    def _select_retrieval_aliases(
        cls,
        aliases: Sequence[str],
        *,
        maximum_aliases: int,
    ) -> tuple[str, ...]:
        if maximum_aliases <= 0:
            return ()

        standard_pattern = re.compile(
            r"\b(?:GRI|SASB|ESRS|CDP|TCFD|IFRS|ISSB)\b"
            r"|\b\d{3}-\d+\b",
            re.IGNORECASE,
        )

        candidates: list[tuple[tuple[int, int, int], str]] = []
        seen: set[str] = set()

        for alias in aliases:
            cleaned = cls._clean_text(alias)
            normalized = cls.normalize_text(cleaned)

            if not cleaned or not normalized or normalized in seen:
                continue

            seen.add(normalized)

            if standard_pattern.search(cleaned):
                continue

            words = normalized.split()

            # Tek kelimelik ve aşırı genel alias'lar sorguyu gürültülendirir.
            if len(words) < 2:
                continue

            # İngilizce ve tam metrik ifadeleri öne al.
            contains_ascii_metric = bool(
                re.search(
                    r"\b(?:total|number|consumption|withdrawal|"
                    r"emission|employees|waste|energy|water|profit|"
                    r"revenue|rate|intensity|recycled|generated)\b",
                    normalized,
                )
            )
            starts_with_total = normalized.startswith("total ")
            word_count_distance = abs(len(words) - 4)

            priority = (
                0 if starts_with_total else 1,
                0 if contains_ascii_metric else 1,
                word_count_distance,
            )
            candidates.append((priority, cleaned))

        candidates.sort(key=lambda item: item[0])

        return tuple(
            alias
            for _, alias in candidates[:maximum_aliases]
        )

    @classmethod
    def _validate_definition_question(
        cls,
        question: str,
        definition: QuestionDefinition,
    ) -> None:
        if cls.normalize_text(question) != cls.normalize_text(
            definition.question
        ):
            raise QuestionAnalyzerError(
                "QuestionDefinition ile analiz edilen soru metni uyuşmuyor: "
                f"id={definition.id}, key={definition.key!r}."
            )

    @staticmethod
    def _definition_metadata(
        definition: QuestionDefinition | None,
    ) -> dict[str, Any]:
        if definition is None:
            return {}

        return {
            "question_id": definition.id,
            "question_key": definition.key,
            "question_group": definition.group,
            "group_order": definition.group_order,
            "topic": definition.topic,
            "source_column": definition.source_column,
            "original_header": definition.original_header,
            "aliases": list(definition.aliases),
            "expected_units": list(definition.expected_units),
            "preferred_sources": list(definition.preferred_sources),
            "preferred_document_types": list(
                definition.preferred_document_types
            ),
            **dict(definition.metadata),
        }

    @staticmethod
    def _coerce_answer_type(
        value: ExpectedAnswerType | str,
    ) -> ExpectedAnswerType:
        try:
            return ExpectedAnswerType(str(getattr(value, "value", value)))
        except ValueError as exc:
            raise QuestionAnalyzerError(
                f"Geçersiz expected_answer_type: {value!r}."
            ) from exc

    @staticmethod
    def _coerce_retrieval_strategy(
        value: RetrievalStrategy | str,
    ) -> RetrievalStrategy:
        try:
            return RetrievalStrategy(str(getattr(value, "value", value)))
        except ValueError as exc:
            raise QuestionAnalyzerError(
                f"Geçersiz retrieval_strategy: {value!r}."
            ) from exc

    @classmethod
    def _deduplicate_texts(
        cls,
        values: Sequence[str],
    ) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = cls._clean_text(value)
            normalized = cls.normalize_text(cleaned)
            if not cleaned or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(cleaned)

        return tuple(result)

    @staticmethod
    def _normalize_standard_code(code: str) -> str:
        return " ".join(str(code).strip().upper().split())

    @staticmethod
    def _is_valid_year(year: Any) -> bool:
        try:
            numeric_year = int(year)
        except (TypeError, ValueError):
            return False
        return 1900 <= numeric_year <= 2100

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        enum_value = getattr(value, "value", value)
        return " ".join(str(enum_value).replace("\n", " ").split())

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
            }
        )

        normalized = str(value).translate(translation_table).casefold()
        normalized = re.sub(r"[^\w\s%./+-]", " ", normalized)
        return " ".join(normalized.split())


__all__ = [
    "ExpectedAnswerType",
    "QuestionAnalysis",
    "QuestionAnalyzer",
    "QuestionAnalyzerConfig",
    "QuestionAnalyzerError",
    "QuestionIntent",
    "QuestionMetric",
    "RetrievalStrategy",
]