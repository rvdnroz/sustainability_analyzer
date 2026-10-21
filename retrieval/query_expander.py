from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

from standards.esg_ontology import ESGOntology, ESGPillar
from standards.esrs_mapper import ESRSMapper
from standards.gri_mapper import GRIMapper


class QueryLanguage(StrEnum):
    TURKISH = "tr"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class QueryExpansionSource(StrEnum):
    ORIGINAL = "original"
    ONTOLOGY = "ontology"
    SYNONYM = "synonym"
    TRANSLATION = "translation"
    STANDARD = "standard"
    METRIC = "metric"
    RULE = "rule"


@dataclass(slots=True, frozen=True)
class ExpandedQueryTerm:
    term: str
    normalized_term: str
    source: QueryExpansionSource
    weight: float = 1.0
    concept_id: str | None = None
    standard_code: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class QueryExpansionResult:
    original_query: str
    normalized_query: str
    detected_language: QueryLanguage
    expanded_queries: tuple[str, ...]
    expanded_terms: tuple[ExpandedQueryTerm, ...]
    standard_codes: tuple[str, ...]
    metrics: tuple[str, ...]
    detected_pillars: tuple[ESGPillar, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def primary_query(self) -> str:
        if self.expanded_queries:
            return self.expanded_queries[0]

        return self.original_query

    @property
    def all_terms(self) -> tuple[str, ...]:
        return tuple(term.term for term in self.expanded_terms)


class QueryExpansionError(RuntimeError):
    """Sorgu genişletme işlemleri sırasında oluşan temel hata."""


class QueryExpander:
    """
    Kullanıcı sorularından retrieval sisteminde kullanılacak genişletilmiş
    sorgular üretir.

    Genişletme kaynakları:
    - ESG ontolojisindeki eş anlamlı ve ilişkili kavramlar
    - Türkçe ve İngilizce terim karşılıkları
    - GRI ve ESRS standart kodları
    - Metrik ve birim ifadeleri
    - ESG alanına özgü kural tabanlı terimler

    Üretilen sorgular BM25, embedding, tablo ve standart retrieval
    bileşenleri tarafından kullanılabilir.
    """

    _TURKISH_CHAR_PATTERN = re.compile(r"[çğıöşüÇĞİÖŞÜ]")

    _TURKISH_HINTS = {
        "ve",
        "ile",
        "için",
        "nedir",
        "kaç",
        "ne",
        "hangi",
        "şirket",
        "rapor",
        "yıl",
        "yılı",
        "toplam",
        "oran",
        "miktar",
        "emisyon",
        "enerji",
        "çalışan",
        "atık",
        "su",
        "hedef",
        "artış",
        "azalış",
        "azalma",
        "artma",
        "kapsam",
    }

    _ENGLISH_HINTS = {
        "and",
        "with",
        "for",
        "what",
        "which",
        "how",
        "company",
        "report",
        "year",
        "total",
        "ratio",
        "amount",
        "emissions",
        "energy",
        "employees",
        "waste",
        "water",
        "target",
        "increase",
        "decrease",
        "scope",
    }

    _TRANSLATION_PAIRS: tuple[tuple[str, str], ...] = (
        ("sera gazı emisyonları", "greenhouse gas emissions"),
        ("karbon emisyonları", "carbon emissions"),
        ("kapsam 1 emisyonları", "scope 1 emissions"),
        ("kapsam 2 emisyonları", "scope 2 emissions"),
        ("kapsam 3 emisyonları", "scope 3 emissions"),
        ("doğrudan emisyonlar", "direct emissions"),
        ("dolaylı emisyonlar", "indirect emissions"),
        ("enerji tüketimi", "energy consumption"),
        ("yenilenebilir enerji", "renewable energy"),
        ("enerji yoğunluğu", "energy intensity"),
        ("su tüketimi", "water consumption"),
        ("su çekimi", "water withdrawal"),
        ("su deşarjı", "water discharge"),
        ("atık yönetimi", "waste management"),
        ("toplam atık", "total waste"),
        ("tehlikeli atık", "hazardous waste"),
        ("geri dönüşüm", "recycling"),
        ("geri dönüştürülen atık", "recycled waste"),
        ("döngüsel ekonomi", "circular economy"),
        ("biyolojik çeşitlilik", "biodiversity"),
        ("çalışan sayısı", "employee count"),
        ("iş gücü", "workforce"),
        ("çalışan devir oranı", "employee turnover"),
        ("iş sağlığı ve güvenliği", "occupational health and safety"),
        ("iş kazası", "work-related injury"),
        ("eğitim saati", "training hours"),
        ("kadın çalışan oranı", "female employee ratio"),
        ("cinsiyet çeşitliliği", "gender diversity"),
        ("ücret farkı", "pay gap"),
        ("insan hakları", "human rights"),
        ("yolsuzlukla mücadele", "anti-corruption"),
        ("rüşvetle mücadele", "anti-bribery"),
        ("iş etiği", "business ethics"),
        ("veri gizliliği", "data privacy"),
        ("müşteri gizliliği", "customer privacy"),
        ("paydaş katılımı", "stakeholder engagement"),
        ("önemlilik değerlendirmesi", "materiality assessment"),
        ("çifte önemlilik", "double materiality"),
        ("ekonomik değer", "economic value"),
        ("gelir", "revenue"),
        ("hedef", "target"),
        ("baz yıl", "base year"),
        ("raporlama yılı", "reporting year"),
    )

    _RULE_EXPANSIONS: Mapping[str, tuple[str, ...]] = {
        "scope 1": (
            "direct greenhouse gas emissions",
            "direct emissions",
            "kapsam 1",
            "doğrudan sera gazı emisyonları",
            "GRI 305-1",
        ),
        "scope 2": (
            "energy indirect greenhouse gas emissions",
            "purchased electricity emissions",
            "kapsam 2",
            "enerji kaynaklı dolaylı emisyonlar",
            "GRI 305-2",
        ),
        "scope 3": (
            "value chain emissions",
            "other indirect greenhouse gas emissions",
            "kapsam 3",
            "değer zinciri emisyonları",
            "GRI 305-3",
        ),
        "emission intensity": (
            "carbon intensity",
            "GHG intensity",
            "emissions per production",
            "emisyon yoğunluğu",
            "karbon yoğunluğu",
            "GRI 305-4",
        ),
        "energy consumption": (
            "total energy consumed",
            "electricity consumption",
            "fuel consumption",
            "enerji tüketimi",
            "toplam enerji tüketimi",
            "GRI 302-1",
        ),
        "water consumption": (
            "water withdrawal",
            "freshwater consumption",
            "su tüketimi",
            "su çekimi",
            "GRI 303-3",
        ),
        "waste": (
            "waste generated",
            "waste recycled",
            "waste disposed",
            "oluşan atık",
            "geri dönüştürülen atık",
            "bertaraf edilen atık",
            "GRI 306-3",
            "GRI 306-4",
            "GRI 306-5",
        ),
        "employee turnover": (
            "staff turnover",
            "employee departures",
            "çalışan devir oranı",
            "personel devri",
            "GRI 401-1",
        ),
        "occupational health and safety": (
            "work-related injuries",
            "lost time injury frequency rate",
            "LTIFR",
            "TRIR",
            "iş sağlığı ve güvenliği",
            "iş kazaları",
            "GRI 403-9",
        ),
        "training": (
            "training hours",
            "average training hours per employee",
            "employee development",
            "eğitim saatleri",
            "çalışan başına eğitim saati",
            "GRI 404-1",
        ),
        "diversity": (
            "gender diversity",
            "women employees",
            "women in management",
            "çeşitlilik",
            "kadın çalışan oranı",
            "kadın yönetici oranı",
            "GRI 405-1",
        ),
        "anti-corruption": (
            "anti-bribery",
            "corruption incidents",
            "ethics training",
            "yolsuzlukla mücadele",
            "rüşvetle mücadele",
            "GRI 205-2",
            "GRI 205-3",
        ),
    }

    def __init__(
        self,
        *,
        ontology: ESGOntology | None = None,
        gri_mapper: GRIMapper | None = None,
        esrs_mapper: ESRSMapper | None = None,
        maximum_terms: int = 40,
        maximum_queries: int = 8,
        minimum_term_length: int = 2,
    ) -> None:
        if maximum_terms <= 0:
            raise ValueError("maximum_terms pozitif olmalıdır.")

        if maximum_queries <= 0:
            raise ValueError("maximum_queries pozitif olmalıdır.")

        if minimum_term_length <= 0:
            raise ValueError("minimum_term_length pozitif olmalıdır.")

        self.ontology = ontology or ESGOntology()
        self.gri_mapper = gri_mapper or GRIMapper()
        self.esrs_mapper = esrs_mapper or ESRSMapper()

        self.maximum_terms = maximum_terms
        self.maximum_queries = maximum_queries
        self.minimum_term_length = minimum_term_length

        self._translation_index = self._build_translation_index()

    def expand(
        self,
        query: str,
        *,
        include_ontology: bool = True,
        include_translations: bool = True,
        include_standards: bool = True,
        include_metrics: bool = True,
        include_rules: bool = True,
    ) -> QueryExpansionResult:
        cleaned_query = self._clean_query(query)

        if not cleaned_query:
            raise QueryExpansionError("Genişletilecek sorgu boş olamaz.")

        normalized_query = self.normalize_text(cleaned_query)
        detected_language = self.detect_language(cleaned_query)

        terms: list[ExpandedQueryTerm] = [
            ExpandedQueryTerm(
                term=cleaned_query,
                normalized_term=normalized_query,
                source=QueryExpansionSource.ORIGINAL,
                weight=1.0,
            ),
        ]

        standard_codes: set[str] = set()
        metrics: set[str] = set()
        detected_pillars: set[ESGPillar] = set()

        if include_ontology:
            ontology_terms, ontology_codes, ontology_metrics, pillars = (
                self._expand_from_ontology(
                    cleaned_query,
                    include_metrics=include_metrics,
                )
            )
            terms.extend(ontology_terms)
            standard_codes.update(ontology_codes)
            metrics.update(ontology_metrics)
            detected_pillars.update(pillars)

        if include_translations:
            terms.extend(
                self._expand_translations(
                    normalized_query,
                ),
            )

        if include_rules:
            terms.extend(
                self._expand_rules(
                    normalized_query,
                ),
            )

        if include_standards:
            gri_terms, gri_codes = self._expand_gri(cleaned_query)
            esrs_terms, esrs_codes = self._expand_esrs(cleaned_query)

            terms.extend(gri_terms)
            terms.extend(esrs_terms)

            standard_codes.update(gri_codes)
            standard_codes.update(esrs_codes)

        for term in terms:
            if term.standard_code:
                standard_codes.add(term.standard_code)

        deduplicated_terms = self._deduplicate_terms(terms)
        deduplicated_terms = deduplicated_terms[: self.maximum_terms]

        expanded_queries = self._build_query_variants(
            original_query=cleaned_query,
            terms=deduplicated_terms,
            standard_codes=standard_codes,
            detected_language=detected_language,
        )

        return QueryExpansionResult(
            original_query=cleaned_query,
            normalized_query=normalized_query,
            detected_language=detected_language,
            expanded_queries=tuple(expanded_queries),
            expanded_terms=tuple(deduplicated_terms),
            standard_codes=tuple(sorted(standard_codes)),
            metrics=tuple(sorted(metrics)),
            detected_pillars=tuple(
                sorted(
                    detected_pillars,
                    key=lambda pillar: pillar.value,
                ),
            ),
            metadata={
                "term_count": len(deduplicated_terms),
                "query_variant_count": len(expanded_queries),
                "standard_code_count": len(standard_codes),
                "metric_count": len(metrics),
            },
        )

    def expand_many(
        self,
        queries: Iterable[str],
    ) -> list[QueryExpansionResult]:
        results: list[QueryExpansionResult] = []

        for query in queries:
            if not isinstance(query, str) or not query.strip():
                continue

            results.append(self.expand(query))

        return results

    def build_bm25_query(
        self,
        result: QueryExpansionResult,
        *,
        include_standard_codes: bool = True,
        maximum_terms: int = 25,
    ) -> str:
        weighted_terms = sorted(
            result.expanded_terms,
            key=lambda item: (
                -item.weight,
                item.normalized_term,
            ),
        )

        terms: list[str] = [result.original_query]

        for expanded_term in weighted_terms:
            if expanded_term.source == QueryExpansionSource.ORIGINAL:
                continue

            if expanded_term.term not in terms:
                terms.append(expanded_term.term)

            if len(terms) >= maximum_terms:
                break

        if include_standard_codes:
            terms.extend(
                code
                for code in result.standard_codes
                if code not in terms
            )

        return " ".join(terms)

    def build_embedding_queries(
        self,
        result: QueryExpansionResult,
        *,
        maximum_queries: int | None = None,
    ) -> tuple[str, ...]:
        limit = maximum_queries or self.maximum_queries

        return result.expanded_queries[:limit]

    def build_table_query(
        self,
        result: QueryExpansionResult,
    ) -> str:
        table_terms: list[str] = [result.original_query]

        for metric in result.metrics:
            if metric not in table_terms:
                table_terms.append(metric)

        for term in result.expanded_terms:
            if term.source in {
                QueryExpansionSource.METRIC,
                QueryExpansionSource.STANDARD,
                QueryExpansionSource.RULE,
            }:
                if term.term not in table_terms:
                    table_terms.append(term.term)

        return " ".join(table_terms[:20])

    def detect_language(self, query: str) -> QueryLanguage:
        normalized = self.normalize_text(query)
        terms = set(normalized.split())

        turkish_score = len(terms.intersection(self._TURKISH_HINTS))
        english_score = len(terms.intersection(self._ENGLISH_HINTS))

        if self._TURKISH_CHAR_PATTERN.search(query):
            turkish_score += 2

        if turkish_score > 0 and english_score > 0:
            if abs(turkish_score - english_score) <= 1:
                return QueryLanguage.MIXED

        if turkish_score > english_score:
            return QueryLanguage.TURKISH

        if english_score > turkish_score:
            return QueryLanguage.ENGLISH

        if self._TURKISH_CHAR_PATTERN.search(query):
            return QueryLanguage.TURKISH

        if re.search(r"[a-zA-Z]", query):
            return QueryLanguage.UNKNOWN

        return QueryLanguage.UNKNOWN

    def _expand_from_ontology(
        self,
        query: str,
        *,
        include_metrics: bool,
    ) -> tuple[
        list[ExpandedQueryTerm],
        set[str],
        set[str],
        set[ESGPillar],
    ]:
        expansion = self.ontology.expand_query(
            query,
            limit_concepts=5,
            include_aliases=True,
            include_keywords=True,
            include_metrics=include_metrics,
            include_related=True,
            maximum_terms=self.maximum_terms,
        )

        terms: list[ExpandedQueryTerm] = []
        standard_codes = set(expansion.standard_codes)
        metrics = set(expansion.metrics)
        pillars: set[ESGPillar] = set()

        concept_by_term: dict[str, str] = {}

        for match in expansion.matched_concepts:
            pillars.add(match.concept.pillar)

            concept_terms = {
                match.concept.preferred_label,
                *match.concept.aliases,
                *match.concept.keywords,
                *match.concept.metrics,
            }

            for concept_term in concept_terms:
                concept_by_term[
                    self.normalize_text(concept_term)
                ] = match.concept.concept_id

        for term in expansion.expanded_terms:
            normalized_term = self.normalize_text(term)

            if not normalized_term:
                continue

            source = (
                QueryExpansionSource.METRIC
                if term in metrics
                else QueryExpansionSource.ONTOLOGY
            )

            terms.append(
                ExpandedQueryTerm(
                    term=term,
                    normalized_term=normalized_term,
                    source=source,
                    weight=0.8 if source == QueryExpansionSource.ONTOLOGY else 0.75,
                    concept_id=concept_by_term.get(normalized_term),
                ),
            )

        for code in expansion.standard_codes:
            terms.append(
                ExpandedQueryTerm(
                    term=code,
                    normalized_term=self.normalize_text(code),
                    source=QueryExpansionSource.STANDARD,
                    weight=0.9,
                    standard_code=code,
                ),
            )

        return terms, standard_codes, metrics, pillars

    def _expand_translations(
        self,
        normalized_query: str,
    ) -> list[ExpandedQueryTerm]:
        terms: list[ExpandedQueryTerm] = []

        for source_term, target_terms in self._translation_index.items():
            if source_term not in normalized_query:
                continue

            for target_term in target_terms:
                terms.append(
                    ExpandedQueryTerm(
                        term=target_term,
                        normalized_term=self.normalize_text(target_term),
                        source=QueryExpansionSource.TRANSLATION,
                        weight=0.72,
                        metadata={
                            "matched_source_term": source_term,
                        },
                    ),
                )

        return terms

    def _expand_rules(
        self,
        normalized_query: str,
    ) -> list[ExpandedQueryTerm]:
        terms: list[ExpandedQueryTerm] = []

        for trigger, expansions in self._RULE_EXPANSIONS.items():
            normalized_trigger = self.normalize_text(trigger)

            if normalized_trigger not in normalized_query:
                continue

            for expansion in expansions:
                is_standard = self._looks_like_standard_code(expansion)

                terms.append(
                    ExpandedQueryTerm(
                        term=expansion,
                        normalized_term=self.normalize_text(expansion),
                        source=(
                            QueryExpansionSource.STANDARD
                            if is_standard
                            else QueryExpansionSource.RULE
                        ),
                        weight=0.78 if is_standard else 0.68,
                        standard_code=expansion if is_standard else None,
                        metadata={
                            "rule_trigger": trigger,
                        },
                    ),
                )

        return terms

    def _expand_gri(
        self,
        query: str,
    ) -> tuple[list[ExpandedQueryTerm], set[str]]:
        matches = self.gri_mapper.map_text(
            query,
            limit=5,
            minimum_score=0.15,
        )

        terms: list[ExpandedQueryTerm] = []
        codes: set[str] = set()

        for match in matches:
            disclosure = match.disclosure
            codes.add(disclosure.code)

            terms.append(
                ExpandedQueryTerm(
                    term=disclosure.code,
                    normalized_term=self.normalize_text(disclosure.code),
                    source=QueryExpansionSource.STANDARD,
                    weight=max(0.7, match.score),
                    standard_code=disclosure.code,
                    metadata={
                        "standard": "GRI",
                        "title": disclosure.title,
                        "topic": disclosure.topic,
                    },
                ),
            )

            terms.append(
                ExpandedQueryTerm(
                    term=disclosure.title,
                    normalized_term=self.normalize_text(disclosure.title),
                    source=QueryExpansionSource.STANDARD,
                    weight=max(0.6, match.score * 0.85),
                    standard_code=disclosure.code,
                ),
            )

        return terms, codes

    def _expand_esrs(
        self,
        query: str,
    ) -> tuple[list[ExpandedQueryTerm], set[str]]:
        matches = self.esrs_mapper.map_text(
            query,
            limit=5,
            minimum_score=0.15,
        )

        terms: list[ExpandedQueryTerm] = []
        codes: set[str] = set()

        for match in matches:
            disclosure = match.disclosure
            codes.add(disclosure.code)

            terms.append(
                ExpandedQueryTerm(
                    term=disclosure.code,
                    normalized_term=self.normalize_text(disclosure.code),
                    source=QueryExpansionSource.STANDARD,
                    weight=max(0.7, match.score),
                    standard_code=disclosure.code,
                    metadata={
                        "standard": "ESRS",
                        "title": disclosure.title,
                        "topic": disclosure.topic,
                    },
                ),
            )

            terms.append(
                ExpandedQueryTerm(
                    term=disclosure.title,
                    normalized_term=self.normalize_text(disclosure.title),
                    source=QueryExpansionSource.STANDARD,
                    weight=max(0.6, match.score * 0.85),
                    standard_code=disclosure.code,
                ),
            )

        return terms, codes

    def _build_query_variants(
        self,
        *,
        original_query: str,
        terms: Sequence[ExpandedQueryTerm],
        standard_codes: set[str],
        detected_language: QueryLanguage,
    ) -> list[str]:
        variants: list[str] = [original_query]

        ontology_terms = [
            term.term
            for term in terms
            if term.source
            in {
                QueryExpansionSource.ONTOLOGY,
                QueryExpansionSource.SYNONYM,
            }
        ]

        translation_terms = [
            term.term
            for term in terms
            if term.source == QueryExpansionSource.TRANSLATION
        ]

        metric_terms = [
            term.term
            for term in terms
            if term.source == QueryExpansionSource.METRIC
        ]

        rule_terms = [
            term.term
            for term in terms
            if term.source == QueryExpansionSource.RULE
        ]

        if ontology_terms:
            variants.append(
                self._join_query(
                    original_query,
                    ontology_terms[:8],
                ),
            )

        if translation_terms:
            variants.append(
                self._join_query(
                    original_query,
                    translation_terms[:8],
                ),
            )

        if standard_codes:
            variants.append(
                self._join_query(
                    original_query,
                    sorted(standard_codes)[:8],
                ),
            )

        if metric_terms:
            variants.append(
                self._join_query(
                    original_query,
                    metric_terms[:8],
                ),
            )

        if rule_terms:
            variants.append(
                self._join_query(
                    original_query,
                    rule_terms[:8],
                ),
            )

        if (
            detected_language == QueryLanguage.TURKISH
            and translation_terms
        ):
            variants.append(" ".join(translation_terms[:12]))

        if (
            detected_language == QueryLanguage.ENGLISH
            and translation_terms
        ):
            variants.append(" ".join(translation_terms[:12]))

        combined_terms = [
            term.term
            for term in sorted(
                terms,
                key=lambda item: -item.weight,
            )
            if term.source != QueryExpansionSource.ORIGINAL
        ]

        if combined_terms:
            variants.append(
                self._join_query(
                    original_query,
                    combined_terms[:15],
                ),
            )

        return self._deduplicate_queries(
            variants,
        )[: self.maximum_queries]

    def _deduplicate_terms(
        self,
        terms: Sequence[ExpandedQueryTerm],
    ) -> list[ExpandedQueryTerm]:
        deduplicated: dict[str, ExpandedQueryTerm] = {}

        for term in terms:
            cleaned_term = self._clean_term(term.term)

            if len(cleaned_term) < self.minimum_term_length:
                continue

            normalized_term = self.normalize_text(cleaned_term)

            if not normalized_term:
                continue

            normalized_record = ExpandedQueryTerm(
                term=cleaned_term,
                normalized_term=normalized_term,
                source=term.source,
                weight=term.weight,
                concept_id=term.concept_id,
                standard_code=term.standard_code,
                metadata=dict(term.metadata),
            )

            existing = deduplicated.get(normalized_term)

            if existing is None or normalized_record.weight > existing.weight:
                deduplicated[normalized_term] = normalized_record

        return sorted(
            deduplicated.values(),
            key=lambda item: (
                -item.weight,
                item.normalized_term,
            ),
        )

    def _build_translation_index(self) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}

        for turkish_term, english_term in self._TRANSLATION_PAIRS:
            normalized_turkish = self.normalize_text(turkish_term)
            normalized_english = self.normalize_text(english_term)

            index.setdefault(normalized_turkish, set()).add(
                english_term,
            )
            index.setdefault(normalized_english, set()).add(
                turkish_term,
            )

        return index

    @staticmethod
    def _looks_like_standard_code(value: str) -> bool:
        normalized = value.strip().upper()

        return normalized.startswith(
            (
                "GRI ",
                "ESRS ",
                "SASB ",
                "TCFD ",
                "SDG ",
            ),
        )

    @staticmethod
    def _join_query(
        original_query: str,
        additional_terms: Sequence[str],
    ) -> str:
        parts = [original_query.strip()]

        seen = {QueryExpander.normalize_text(original_query)}

        for term in additional_terms:
            cleaned = QueryExpander._clean_term(term)
            normalized = QueryExpander.normalize_text(cleaned)

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            parts.append(cleaned)

        return " ".join(parts)

    @staticmethod
    def _deduplicate_queries(
        queries: Sequence[str],
    ) -> list[str]:
        deduplicated: list[str] = []
        seen: set[str] = set()

        for query in queries:
            cleaned = QueryExpander._clean_query(query)
            normalized = QueryExpander.normalize_text(cleaned)

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            deduplicated.append(cleaned)

        return deduplicated

    @staticmethod
    def _clean_query(query: str) -> str:
        if query is None:
            return ""

        return " ".join(str(query).strip().split())

    @staticmethod
    def _clean_term(term: str) -> str:
        return " ".join(str(term).strip().split())

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
            r"[^\w\s\-./]",
            " ",
            normalized,
        )

        return " ".join(normalized.split())


__all__ = [
    "ExpandedQueryTerm",
    "QueryExpander",
    "QueryExpansionError",
    "QueryExpansionResult",
    "QueryExpansionSource",
    "QueryLanguage",
]