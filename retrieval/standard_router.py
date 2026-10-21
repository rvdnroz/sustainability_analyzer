from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from standards.esg_ontology import ESGOntology, ESGPillar
from standards.esrs_mapper import ESRSMapper
from standards.gri_mapper import GRIMapper
from standards.standard_index_parser import StandardType


class StandardRouteType(StrEnum):
    GRI = "gri"
    ESRS = "esrs"
    SASB = "sasb"
    TCFD = "tcfd"
    SDG = "sdg"
    MULTI_STANDARD = "multi_standard"
    GENERAL_ESG = "general_esg"
    NONE = "none"


@dataclass(slots=True, frozen=True)
class StandardRoute:
    route_type: StandardRouteType
    confidence: float
    standard_types: tuple[StandardType, ...] = ()
    standard_codes: tuple[str, ...] = ()
    disclosure_titles: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    pillars: tuple[ESGPillar, ...] = ()
    matched_terms: tuple[str, ...] = ()
    should_search_standard_index: bool = False
    should_search_tables: bool = False
    should_search_text: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_standard_route(self) -> bool:
        return bool(self.standard_types or self.standard_codes)


@dataclass(slots=True, frozen=True)
class StandardRouterConfig:
    minimum_standard_score: float = 0.15
    maximum_matches_per_standard: int = 5
    explicit_code_confidence: float = 1.0
    inferred_code_confidence: float = 0.72
    general_esg_confidence: float = 0.45
    standard_index_bonus: float = 0.12

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_standard_score", self.minimum_standard_score),
            ("explicit_code_confidence", self.explicit_code_confidence),
            ("inferred_code_confidence", self.inferred_code_confidence),
            ("general_esg_confidence", self.general_esg_confidence),
            ("standard_index_bonus", self.standard_index_bonus),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} değeri 0 ile 1 arasında olmalıdır.",
                )

        if self.maximum_matches_per_standard <= 0:
            raise ValueError(
                "maximum_matches_per_standard pozitif olmalıdır.",
            )


class StandardRouterError(RuntimeError):
    """Standart yönlendirme işlemleri sırasında oluşan temel hata."""


class StandardRouter:
    """
    GRI, ESRS, SASB, TCFD ve SDG içeren soruları ilgili standart
    indekslerine ve retrieval yollarına yönlendirir.

    Router şu sinyalleri kullanır:
    - Açık standart kodları
    - GRI ve ESRS mapper sonuçları
    - ESG ontolojisi eşleşmeleri
    - Standart indeks, disclosure ve sayfa referansı ifadeleri
    - Sayısal metrik, hedef ve karşılaştırma ifadeleri
    """

    _GRI_PATTERN = re.compile(
        r"\bGRI\s*[-:]?\s*"
        r"(?P<code>\d{1,3}(?:-\d{1,3})?)\b",
        re.IGNORECASE,
    )

    _ESRS_PATTERN = re.compile(
        r"\bESRS\s*[-:]?\s*"
        r"(?P<code>1|2|E[1-5]|S[1-4]|G1)"
        r"(?:\s*[-:]\s*DR\s+[A-Z0-9-]+)?\b",
        re.IGNORECASE,
    )

    _SASB_PATTERN = re.compile(
        r"\b(?:SASB\s+)?"
        r"(?P<code>[A-Z]{2}-[A-Z]{2}-\d{3}[A-Z]?\.\d+)\b",
        re.IGNORECASE,
    )

    _SDG_PATTERN = re.compile(
        r"\b(?:SDG|SUSTAINABLE\s+DEVELOPMENT\s+GOAL|"
        r"SÜRDÜRÜLEBİLİR\s+KALKINMA\s+AMACI)"
        r"\s*(?P<number>1[0-7]|[1-9])\b",
        re.IGNORECASE,
    )

    _TCFD_PATTERN = re.compile(
        r"\b(?:TCFD|"
        r"TASK\s+FORCE\s+ON\s+CLIMATE[-\s]RELATED\s+"
        r"FINANCIAL\s+DISCLOSURES)\b",
        re.IGNORECASE,
    )

    _INDEX_TERMS = (
        "content index",
        "standard index",
        "disclosure index",
        "gri index",
        "esrs index",
        "sasb index",
        "reference table",
        "cross reference",
        "içerik indeksi",
        "standart indeksi",
        "açıklama indeksi",
        "gri indeksi",
        "esrs indeksi",
        "referans tablosu",
        "sayfa referansı",
    )

    _TABLE_TERMS = (
        "value",
        "amount",
        "total",
        "ratio",
        "rate",
        "percentage",
        "target",
        "baseline",
        "trend",
        "comparison",
        "increase",
        "decrease",
        "değer",
        "miktar",
        "toplam",
        "oran",
        "yüzde",
        "hedef",
        "baz yıl",
        "eğilim",
        "karşılaştırma",
        "artış",
        "azalış",
        "kaç",
        "ne kadar",
    )

    _STANDARD_NAMES: Mapping[str, StandardType] = {
        "gri": StandardType.GRI,
        "global reporting initiative": StandardType.GRI,
        "esrs": StandardType.ESRS,
        "european sustainability reporting standards": StandardType.ESRS,
        "sasb": StandardType.SASB,
        "sustainability accounting standards board": StandardType.SASB,
        "tcfd": StandardType.TCFD,
        "task force on climate-related financial disclosures": (
            StandardType.TCFD
        ),
        "sdg": StandardType.SDG,
        "sustainable development goals": StandardType.SDG,
        "sürdürülebilir kalkınma amaçları": StandardType.SDG,
    }

    def __init__(
        self,
        *,
        gri_mapper: GRIMapper | None = None,
        esrs_mapper: ESRSMapper | None = None,
        ontology: ESGOntology | None = None,
        config: StandardRouterConfig | None = None,
    ) -> None:
        self.gri_mapper = gri_mapper or GRIMapper()
        self.esrs_mapper = esrs_mapper or ESRSMapper()
        self.ontology = ontology or ESGOntology()
        self.config = config or StandardRouterConfig()

    def route(
        self,
        query: str,
        *,
        standard_hints: Sequence[StandardType | str] = (),
        standard_codes: Sequence[str] = (),
    ) -> StandardRoute:
        cleaned_query = self._clean_text(query)

        if not cleaned_query:
            raise StandardRouterError(
                "Yönlendirilecek sorgu boş olamaz.",
            )

        normalized_query = self.normalize_text(cleaned_query)

        detected_types: set[StandardType] = set()
        detected_codes: set[str] = set()
        disclosure_titles: set[str] = set()
        topics: set[str] = set()
        matched_terms: set[str] = set()
        pillars: set[ESGPillar] = set()

        explicit_code_count = 0

        explicit_results = self._extract_explicit_standards(
            cleaned_query,
        )

        for standard_type, codes, terms in explicit_results:
            detected_types.add(standard_type)
            detected_codes.update(codes)
            matched_terms.update(terms)
            explicit_code_count += len(codes)

        for hint in standard_hints:
            detected_types.add(
                self._normalize_standard_type(hint),
            )

        for code in standard_codes:
            normalized_code, standard_type = (
                self._normalize_external_standard_code(code)
            )
            detected_codes.add(normalized_code)

            if standard_type != StandardType.UNKNOWN:
                detected_types.add(standard_type)

        named_standard_types = self._detect_standard_names(
            normalized_query,
        )
        detected_types.update(named_standard_types)

        gri_matches = self.gri_mapper.map_text(
            cleaned_query,
            limit=self.config.maximum_matches_per_standard,
            minimum_score=self.config.minimum_standard_score,
        )

        for match in gri_matches:
            if match.score < self.config.minimum_standard_score:
                continue

            detected_types.add(StandardType.GRI)
            detected_codes.add(match.code)
            disclosure_titles.add(match.title)
            topics.add(match.disclosure.topic)
            matched_terms.update(match.matched_terms)

        esrs_matches = self.esrs_mapper.map_text(
            cleaned_query,
            limit=self.config.maximum_matches_per_standard,
            minimum_score=self.config.minimum_standard_score,
        )

        for match in esrs_matches:
            if match.score < self.config.minimum_standard_score:
                continue

            detected_types.add(StandardType.ESRS)
            detected_codes.add(match.code)
            disclosure_titles.add(match.title)
            topics.add(match.disclosure.topic)
            matched_terms.update(match.matched_terms)

        ontology_matches = self.ontology.search(
            cleaned_query,
            limit=8,
            minimum_score=0.12,
        )

        for match in ontology_matches:
            pillars.add(match.concept.pillar)
            detected_codes.update(
                match.concept.standard_codes,
            )
            matched_terms.update(match.matched_terms)

            for code in match.concept.standard_codes:
                standard_type = self._standard_type_from_code(code)

                if standard_type != StandardType.UNKNOWN:
                    detected_types.add(standard_type)

        should_search_standard_index = self._should_search_standard_index(
            normalized_query=normalized_query,
            standard_types=detected_types,
            standard_codes=detected_codes,
        )

        should_search_tables = self._should_search_tables(
            normalized_query,
        )

        route_type = self._resolve_route_type(
            detected_types,
            pillars,
        )

        confidence = self._calculate_confidence(
            explicit_code_count=explicit_code_count,
            standard_types=detected_types,
            standard_codes=detected_codes,
            gri_scores=tuple(match.score for match in gri_matches),
            esrs_scores=tuple(match.score for match in esrs_matches),
            ontology_scores=tuple(
                match.score for match in ontology_matches
            ),
            should_search_standard_index=should_search_standard_index,
        )

        return StandardRoute(
            route_type=route_type,
            confidence=confidence,
            standard_types=tuple(
                sorted(
                    detected_types,
                    key=lambda item: item.value,
                ),
            ),
            standard_codes=tuple(
                sorted(
                    detected_codes,
                    key=self._standard_code_sort_key,
                ),
            ),
            disclosure_titles=tuple(sorted(disclosure_titles)),
            topics=tuple(sorted(topics)),
            pillars=tuple(
                sorted(
                    pillars,
                    key=lambda item: item.value,
                ),
            ),
            matched_terms=tuple(sorted(matched_terms)),
            should_search_standard_index=should_search_standard_index,
            should_search_tables=should_search_tables,
            should_search_text=True,
            metadata={
                "explicit_code_count": explicit_code_count,
                "gri_match_count": len(gri_matches),
                "esrs_match_count": len(esrs_matches),
                "ontology_match_count": len(ontology_matches),
                "detected_standard_count": len(detected_types),
                "detected_code_count": len(detected_codes),
            },
        )

    def route_many(
        self,
        queries: Sequence[str],
    ) -> list[StandardRoute]:
        return [
            self.route(query)
            for query in queries
            if isinstance(query, str) and query.strip()
        ]

    def extract_standard_codes(
        self,
        text: str,
    ) -> tuple[str, ...]:
        routes = self._extract_explicit_standards(text)

        codes = {
            code
            for _, detected_codes, _ in routes
            for code in detected_codes
        }

        return tuple(
            sorted(
                codes,
                key=self._standard_code_sort_key,
            ),
        )

    def detect_standard_types(
        self,
        text: str,
    ) -> tuple[StandardType, ...]:
        route = self.route(text)
        return route.standard_types

    def should_use_standard_index(
        self,
        query: str,
    ) -> bool:
        return self.route(query).should_search_standard_index

    def _extract_explicit_standards(
        self,
        text: str,
    ) -> list[tuple[StandardType, set[str], set[str]]]:
        results: list[
            tuple[StandardType, set[str], set[str]]
        ] = []

        gri_codes = {
            self.gri_mapper.normalize_code(match.group("code"))
            for match in self._GRI_PATTERN.finditer(text)
        }

        if gri_codes:
            results.append(
                (
                    StandardType.GRI,
                    gri_codes,
                    set(gri_codes),
                ),
            )

        esrs_codes = {
            self.esrs_mapper.normalize_code(match.group("code"))
            for match in self._ESRS_PATTERN.finditer(text)
        }

        if esrs_codes:
            results.append(
                (
                    StandardType.ESRS,
                    esrs_codes,
                    set(esrs_codes),
                ),
            )

        sasb_codes = {
            match.group("code").upper()
            for match in self._SASB_PATTERN.finditer(text)
        }

        if sasb_codes:
            results.append(
                (
                    StandardType.SASB,
                    sasb_codes,
                    set(sasb_codes),
                ),
            )

        sdg_codes = {
            f"SDG {match.group('number')}"
            for match in self._SDG_PATTERN.finditer(text)
        }

        if sdg_codes:
            results.append(
                (
                    StandardType.SDG,
                    sdg_codes,
                    set(sdg_codes),
                ),
            )

        if self._TCFD_PATTERN.search(text):
            results.append(
                (
                    StandardType.TCFD,
                    {"TCFD"},
                    {"TCFD"},
                ),
            )

        return results

    def _detect_standard_names(
        self,
        normalized_query: str,
    ) -> set[StandardType]:
        detected: set[StandardType] = set()

        for name, standard_type in self._STANDARD_NAMES.items():
            if self.normalize_text(name) in normalized_query:
                detected.add(standard_type)

        return detected

    def _should_search_standard_index(
        self,
        *,
        normalized_query: str,
        standard_types: set[StandardType],
        standard_codes: set[str],
    ) -> bool:
        if standard_codes:
            return True

        if any(
            self.normalize_text(term) in normalized_query
            for term in self._INDEX_TERMS
        ):
            return True

        return bool(standard_types) and any(
            phrase in normalized_query
            for phrase in (
                "page",
                "reference",
                "location",
                "disclosure",
                "açıklama",
                "sayfa",
                "referans",
                "konum",
            )
        )

    def _should_search_tables(
        self,
        normalized_query: str,
    ) -> bool:
        if re.search(r"\b(?:19|20)\d{2}\b", normalized_query):
            return True

        if re.search(
            r"\b\d+(?:[.,]\d+)?\s*(?:%|tco2e|mwh|gwh|gj|m3|m³)\b",
            normalized_query,
        ):
            return True

        return any(
            self.normalize_text(term) in normalized_query
            for term in self._TABLE_TERMS
        )

    def _resolve_route_type(
        self,
        standard_types: set[StandardType],
        pillars: set[ESGPillar],
    ) -> StandardRouteType:
        known_types = {
            standard_type
            for standard_type in standard_types
            if standard_type != StandardType.UNKNOWN
        }

        if len(known_types) > 1:
            return StandardRouteType.MULTI_STANDARD

        if known_types:
            standard_type = next(iter(known_types))

            mapping = {
                StandardType.GRI: StandardRouteType.GRI,
                StandardType.ESRS: StandardRouteType.ESRS,
                StandardType.SASB: StandardRouteType.SASB,
                StandardType.TCFD: StandardRouteType.TCFD,
                StandardType.SDG: StandardRouteType.SDG,
            }

            return mapping.get(
                standard_type,
                StandardRouteType.GENERAL_ESG,
            )

        if pillars:
            return StandardRouteType.GENERAL_ESG

        return StandardRouteType.NONE

    def _calculate_confidence(
        self,
        *,
        explicit_code_count: int,
        standard_types: set[StandardType],
        standard_codes: set[str],
        gri_scores: Sequence[float],
        esrs_scores: Sequence[float],
        ontology_scores: Sequence[float],
        should_search_standard_index: bool,
    ) -> float:
        if explicit_code_count > 0:
            score = self.config.explicit_code_confidence
        elif standard_codes:
            score = self.config.inferred_code_confidence
        elif standard_types:
            scores = [
                *gri_scores,
                *esrs_scores,
            ]

            score = (
                max(scores)
                if scores
                else self.config.general_esg_confidence
            )
        elif ontology_scores:
            score = max(
                self.config.general_esg_confidence,
                max(ontology_scores),
            )
        else:
            score = 0.0

        if should_search_standard_index:
            score += self.config.standard_index_bonus

        if len(standard_types) > 1:
            score += 0.05

        return round(
            min(max(score, 0.0), 1.0),
            4,
        )

    def _normalize_external_standard_code(
        self,
        code: str,
    ) -> tuple[str, StandardType]:
        cleaned_code = self._clean_text(code)

        if not cleaned_code:
            raise ValueError("Standart kodu boş olamaz.")

        standard_type = self._standard_type_from_code(cleaned_code)

        if standard_type == StandardType.GRI:
            return (
                self.gri_mapper.normalize_code(cleaned_code),
                standard_type,
            )

        if standard_type == StandardType.ESRS:
            return (
                self.esrs_mapper.normalize_code(cleaned_code),
                standard_type,
            )

        return (
            " ".join(cleaned_code.upper().split()),
            standard_type,
        )

    @staticmethod
    def _standard_type_from_code(
        code: str,
    ) -> StandardType:
        normalized = " ".join(
            str(code).strip().upper().split(),
        )

        if normalized.startswith("GRI "):
            return StandardType.GRI

        if normalized.startswith("ESRS "):
            return StandardType.ESRS

        if normalized.startswith("SASB ") or re.fullmatch(
            r"[A-Z]{2}-[A-Z]{2}-\d{3}[A-Z]?\.\d+",
            normalized,
        ):
            return StandardType.SASB

        if normalized.startswith("TCFD"):
            return StandardType.TCFD

        if normalized.startswith("SDG "):
            return StandardType.SDG

        return StandardType.UNKNOWN

    @staticmethod
    def _normalize_standard_type(
        value: StandardType | str,
    ) -> StandardType:
        if isinstance(value, StandardType):
            return value

        normalized = str(value).strip().casefold()

        for member in StandardType:
            if normalized in {
                member.value.casefold(),
                member.name.casefold(),
            }:
                return member

        raise ValueError(
            f"Geçersiz standart türü: {value!r}",
        )

    @classmethod
    def _standard_code_sort_key(
        cls,
        code: str,
    ) -> tuple[int, str]:
        standard_type = cls._standard_type_from_code(code)

        order = {
            StandardType.GRI: 0,
            StandardType.ESRS: 1,
            StandardType.SASB: 2,
            StandardType.TCFD: 3,
            StandardType.SDG: 4,
            StandardType.UNKNOWN: 99,
        }

        return (
            order.get(standard_type, 99),
            code,
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        return " ".join(
            str(value).replace("\n", " ").split(),
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
            r"[^\w\s\-./]",
            " ",
            normalized,
        )

        return " ".join(normalized.split())


__all__ = [
    "StandardRoute",
    "StandardRouter",
    "StandardRouterConfig",
    "StandardRouterError",
    "StandardRouteType",
]