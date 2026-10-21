from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Mapping, Sequence


class ESGPillar(StrEnum):
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    ECONOMIC = "economic"
    CROSS_CUTTING = "cross_cutting"
    UNKNOWN = "unknown"


class ESGRelationType(StrEnum):
    SYNONYM = "synonym"
    BROADER = "broader"
    NARROWER = "narrower"
    RELATED = "related"
    METRIC = "metric"
    STANDARD = "standard"


@dataclass(slots=True, frozen=True)
class ESGConcept:
    concept_id: str
    preferred_label: str
    pillar: ESGPillar
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    standard_codes: tuple[str, ...] = ()
    description: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ESGRelation:
    source_concept_id: str
    target_concept_id: str
    relation_type: ESGRelationType
    weight: float = 1.0


@dataclass(slots=True, frozen=True)
class ESGConceptMatch:
    concept: ESGConcept
    score: float
    matched_terms: tuple[str, ...] = ()
    match_type: str = "term"

    @property
    def concept_id(self) -> str:
        return self.concept.concept_id

    @property
    def preferred_label(self) -> str:
        return self.concept.preferred_label


@dataclass(slots=True, frozen=True)
class QueryExpansion:
    original_query: str
    normalized_query: str
    expanded_terms: tuple[str, ...]
    matched_concepts: tuple[ESGConceptMatch, ...]
    standard_codes: tuple[str, ...]
    metrics: tuple[str, ...]


class ESGOntologyError(RuntimeError):
    """ESG ontoloji işlemleri sırasında oluşan temel hata."""


class ESGConceptNotFoundError(ESGOntologyError):
    """İstenen ESG kavramı bulunamadığında yükseltilir."""


class DuplicateESGConceptError(ESGOntologyError):
    """Aynı kimliğe sahip kavram yeniden eklendiğinde yükseltilir."""


class ESGOntology:
    """
    ESG terminolojisi, eş anlamlılar, metrikler ve standart kodları için
    merkezi ontoloji sağlar.

    Ontoloji şu amaçlarla kullanılır:
    - Türkçe ve İngilizce ESG terimlerini ortak kavramlarda birleştirme
    - Sorgu genişletme için eş anlamlı ve ilişkili terimler üretme
    - Kavramları GRI ve ESRS kodlarıyla ilişkilendirme
    - Metrik, kategori ve ESG sütunu bilgisini sağlama
    - Retrieval katmanında terminoloji farklılıklarını azaltma
    """

    def __init__(
        self,
        concepts: Iterable[ESGConcept] | None = None,
        relations: Iterable[ESGRelation] | None = None,
    ) -> None:
        self._concepts: dict[str, ESGConcept] = {}
        self._relations: dict[str, list[ESGRelation]] = {}
        self._term_index: dict[str, set[str]] = {}

        for concept in self._default_concepts():
            self.register_concept(concept)

        for relation in self._default_relations():
            self.register_relation(relation)

        if concepts is not None:
            for concept in concepts:
                self.register_concept(concept, overwrite=True)

        if relations is not None:
            for relation in relations:
                self.register_relation(relation)

    def register_concept(
        self,
        concept: ESGConcept,
        *,
        overwrite: bool = False,
    ) -> None:
        concept_id = self._normalize_identifier(concept.concept_id)

        if concept_id in self._concepts and not overwrite:
            raise DuplicateESGConceptError(
                f"'{concept_id}' kimliğine sahip ESG kavramı zaten kayıtlı.",
            )

        normalized_concept = ESGConcept(
            concept_id=concept_id,
            preferred_label=self._clean_label(concept.preferred_label),
            pillar=self._normalize_pillar(concept.pillar),
            aliases=self._normalize_terms(concept.aliases),
            keywords=self._normalize_terms(concept.keywords),
            metrics=self._normalize_terms(concept.metrics),
            standard_codes=self._normalize_standard_codes(
                concept.standard_codes,
            ),
            description=(
                self._clean_label(concept.description)
                if concept.description
                else None
            ),
            metadata=dict(concept.metadata),
        )

        if concept_id in self._concepts:
            self._remove_from_term_index(self._concepts[concept_id])

        self._concepts[concept_id] = normalized_concept
        self._add_to_term_index(normalized_concept)

    def register_relation(
        self,
        relation: ESGRelation,
    ) -> None:
        source_id = self._normalize_identifier(
            relation.source_concept_id,
        )
        target_id = self._normalize_identifier(
            relation.target_concept_id,
        )

        if source_id not in self._concepts:
            raise ESGConceptNotFoundError(
                f"Kaynak kavram bulunamadı: {source_id}",
            )

        if target_id not in self._concepts:
            raise ESGConceptNotFoundError(
                f"Hedef kavram bulunamadı: {target_id}",
            )

        if source_id == target_id:
            raise ESGOntologyError(
                "Bir kavram kendisiyle ilişkilendirilemez.",
            )

        if not 0.0 < relation.weight <= 1.0:
            raise ValueError(
                "İlişki ağırlığı 0 ile 1 arasında olmalıdır.",
            )

        normalized_relation = ESGRelation(
            source_concept_id=source_id,
            target_concept_id=target_id,
            relation_type=relation.relation_type,
            weight=relation.weight,
        )

        existing_relations = self._relations.setdefault(source_id, [])

        if normalized_relation not in existing_relations:
            existing_relations.append(normalized_relation)

    def get(self, concept_id: str) -> ESGConcept:
        normalized_id = self._normalize_identifier(concept_id)

        try:
            return self._concepts[normalized_id]
        except KeyError as exc:
            raise ESGConceptNotFoundError(
                f"'{normalized_id}' kimliğine sahip ESG kavramı bulunamadı.",
            ) from exc

    def get_optional(self, concept_id: str) -> ESGConcept | None:
        normalized_id = self._normalize_identifier(concept_id)
        return self._concepts.get(normalized_id)

    def all(self) -> list[ESGConcept]:
        return sorted(
            self._concepts.values(),
            key=lambda concept: (
                concept.pillar.value,
                concept.preferred_label.casefold(),
            ),
        )

    def search(
        self,
        query: str,
        *,
        pillar: ESGPillar | str | None = None,
        limit: int = 10,
        minimum_score: float = 0.1,
    ) -> list[ESGConceptMatch]:
        if limit <= 0:
            return []

        normalized_query = self.normalize_text(query)

        if not normalized_query:
            return []

        pillar_filter = (
            self._normalize_pillar(pillar)
            if pillar is not None
            else None
        )

        query_terms = set(normalized_query.split())
        matches: list[ESGConceptMatch] = []

        for concept in self._concepts.values():
            if (
                pillar_filter is not None
                and concept.pillar != pillar_filter
            ):
                continue

            score, matched_terms, match_type = self._score_concept(
                concept=concept,
                normalized_query=normalized_query,
                query_terms=query_terms,
            )

            if score < minimum_score:
                continue

            matches.append(
                ESGConceptMatch(
                    concept=concept,
                    score=round(min(score, 1.0), 4),
                    matched_terms=tuple(sorted(matched_terms)),
                    match_type=match_type,
                ),
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                match.concept.preferred_label.casefold(),
            ),
        )

        return matches[:limit]

    def find_by_term(self, term: str) -> list[ESGConcept]:
        normalized_term = self.normalize_text(term)

        concept_ids = self._term_index.get(
            normalized_term,
            set(),
        )

        return [
            self._concepts[concept_id]
            for concept_id in sorted(concept_ids)
        ]

    def find_by_pillar(
        self,
        pillar: ESGPillar | str,
    ) -> list[ESGConcept]:
        normalized_pillar = self._normalize_pillar(pillar)

        return [
            concept
            for concept in self.all()
            if concept.pillar == normalized_pillar
        ]

    def find_by_standard_code(
        self,
        standard_code: str,
    ) -> list[ESGConcept]:
        normalized_code = self._normalize_standard_code(
            standard_code,
        )

        return [
            concept
            for concept in self.all()
            if normalized_code in concept.standard_codes
        ]

    def get_related_concepts(
        self,
        concept_id: str,
        *,
        relation_types: Sequence[ESGRelationType] | None = None,
        minimum_weight: float = 0.0,
        include_reverse: bool = True,
    ) -> list[tuple[ESGConcept, ESGRelation]]:
        normalized_id = self._normalize_identifier(concept_id)

        if normalized_id not in self._concepts:
            raise ESGConceptNotFoundError(
                f"'{normalized_id}' kimliğine sahip ESG kavramı bulunamadı.",
            )

        allowed_types = (
            set(relation_types)
            if relation_types is not None
            else None
        )

        matched_relations: list[tuple[ESGConcept, ESGRelation]] = []

        for relation in self._relations.get(normalized_id, []):
            if relation.weight < minimum_weight:
                continue

            if (
                allowed_types is not None
                and relation.relation_type not in allowed_types
            ):
                continue

            matched_relations.append(
                (
                    self._concepts[relation.target_concept_id],
                    relation,
                ),
            )

        if include_reverse:
            for relations in self._relations.values():
                for relation in relations:
                    if relation.target_concept_id != normalized_id:
                        continue

                    if relation.weight < minimum_weight:
                        continue

                    if (
                        allowed_types is not None
                        and relation.relation_type not in allowed_types
                    ):
                        continue

                    reverse_relation = ESGRelation(
                        source_concept_id=normalized_id,
                        target_concept_id=relation.source_concept_id,
                        relation_type=relation.relation_type,
                        weight=relation.weight,
                    )

                    matched_relations.append(
                        (
                            self._concepts[relation.source_concept_id],
                            reverse_relation,
                        ),
                    )

        deduplicated: dict[
            tuple[str, ESGRelationType],
            tuple[ESGConcept, ESGRelation],
        ] = {}

        for concept, relation in matched_relations:
            key = (
                concept.concept_id,
                relation.relation_type,
            )

            existing = deduplicated.get(key)

            if (
                existing is None
                or relation.weight > existing[1].weight
            ):
                deduplicated[key] = (concept, relation)

        return sorted(
            deduplicated.values(),
            key=lambda item: (
                -item[1].weight,
                item[0].preferred_label.casefold(),
            ),
        )

    def expand_query(
        self,
        query: str,
        *,
        limit_concepts: int = 5,
        include_aliases: bool = True,
        include_keywords: bool = True,
        include_metrics: bool = True,
        include_related: bool = True,
        related_minimum_weight: float = 0.6,
        maximum_terms: int = 30,
    ) -> QueryExpansion:
        normalized_query = self.normalize_text(query)

        matches = self.search(
            query,
            limit=limit_concepts,
            minimum_score=0.1,
        )

        expanded_terms: list[str] = []
        standard_codes: set[str] = set()
        metrics: set[str] = set()

        for match in matches:
            concept = match.concept

            expanded_terms.append(concept.preferred_label)

            if include_aliases:
                expanded_terms.extend(concept.aliases)

            if include_keywords:
                expanded_terms.extend(concept.keywords)

            if include_metrics:
                expanded_terms.extend(concept.metrics)
                metrics.update(concept.metrics)

            standard_codes.update(concept.standard_codes)

            if include_related:
                related = self.get_related_concepts(
                    concept.concept_id,
                    minimum_weight=related_minimum_weight,
                )

                for related_concept, _ in related:
                    expanded_terms.append(
                        related_concept.preferred_label,
                    )

                    if include_aliases:
                        expanded_terms.extend(
                            related_concept.aliases,
                        )

                    standard_codes.update(
                        related_concept.standard_codes,
                    )
                    metrics.update(related_concept.metrics)

        normalized_original_terms = set(normalized_query.split())

        cleaned_terms: list[str] = []
        seen_terms: set[str] = set()

        for term in expanded_terms:
            cleaned_term = self._clean_label(term)
            normalized_term = self.normalize_text(cleaned_term)

            if not normalized_term:
                continue

            if normalized_term in normalized_original_terms:
                continue

            if normalized_term in seen_terms:
                continue

            seen_terms.add(normalized_term)
            cleaned_terms.append(cleaned_term)

            if len(cleaned_terms) >= maximum_terms:
                break

        return QueryExpansion(
            original_query=query,
            normalized_query=normalized_query,
            expanded_terms=tuple(cleaned_terms),
            matched_concepts=tuple(matches),
            standard_codes=tuple(sorted(standard_codes)),
            metrics=tuple(sorted(metrics)),
        )

    def build_expanded_query(
        self,
        query: str,
        *,
        separator: str = " ",
        maximum_terms: int = 30,
    ) -> str:
        expansion = self.expand_query(
            query,
            maximum_terms=maximum_terms,
        )

        parts = [
            query.strip(),
            *expansion.expanded_terms,
            *expansion.standard_codes,
        ]

        return separator.join(
            part
            for part in parts
            if part
        )

    def detect_pillars(
        self,
        text: str,
        *,
        minimum_score: float = 0.15,
    ) -> tuple[ESGPillar, ...]:
        matches = self.search(
            text,
            limit=len(self._concepts),
            minimum_score=minimum_score,
        )

        pillars = {
            match.concept.pillar
            for match in matches
            if match.concept.pillar != ESGPillar.UNKNOWN
        }

        return tuple(
            sorted(
                pillars,
                key=lambda pillar: pillar.value,
            ),
        )

    def _score_concept(
        self,
        *,
        concept: ESGConcept,
        normalized_query: str,
        query_terms: set[str],
    ) -> tuple[float, set[str], str]:
        preferred_label = self.normalize_text(
            concept.preferred_label,
        )

        aliases = {
            self.normalize_text(alias)
            for alias in concept.aliases
        }
        keywords = {
            self.normalize_text(keyword)
            for keyword in concept.keywords
        }
        metrics = {
            self.normalize_text(metric)
            for metric in concept.metrics
        }
        standard_codes = {
            self.normalize_text(code)
            for code in concept.standard_codes
        }

        score = 0.0
        matched_terms: set[str] = set()
        match_type = "term"

        if preferred_label == normalized_query:
            return 1.0, {concept.preferred_label}, "exact"

        if preferred_label in normalized_query:
            score += 0.6
            matched_terms.add(concept.preferred_label)
            match_type = "label"

        for alias in aliases:
            if alias == normalized_query:
                return 0.95, {alias}, "alias"

            if alias and alias in normalized_query:
                score += 0.45
                matched_terms.add(alias)
                match_type = "alias"

        for code in standard_codes:
            if code and code in normalized_query:
                score += 0.5
                matched_terms.add(code)
                match_type = "standard"

        for metric in metrics:
            if metric and metric in normalized_query:
                score += 0.35
                matched_terms.add(metric)
                match_type = "metric"

        for keyword in keywords:
            if keyword and keyword in normalized_query:
                score += 0.22
                matched_terms.add(keyword)
                continue

            keyword_terms = set(keyword.split())
            overlap = query_terms.intersection(keyword_terms)

            if overlap:
                score += 0.12 * (
                    len(overlap) / max(len(keyword_terms), 1)
                )
                matched_terms.update(overlap)

        label_terms = set(preferred_label.split())
        alias_terms = {
            term
            for alias in aliases
            for term in alias.split()
        }
        keyword_terms = {
            term
            for keyword in keywords
            for term in keyword.split()
        }

        score += 0.2 * self._overlap_ratio(
            query_terms,
            label_terms,
        )
        score += 0.12 * self._overlap_ratio(
            query_terms,
            alias_terms,
        )
        score += 0.08 * self._overlap_ratio(
            query_terms,
            keyword_terms,
        )

        return score, matched_terms, match_type

    def _add_to_term_index(
        self,
        concept: ESGConcept,
    ) -> None:
        terms = {
            concept.preferred_label,
            *concept.aliases,
            *concept.keywords,
            *concept.metrics,
            *concept.standard_codes,
        }

        for term in terms:
            normalized_term = self.normalize_text(term)

            if not normalized_term:
                continue

            self._term_index.setdefault(
                normalized_term,
                set(),
            ).add(concept.concept_id)

    def _remove_from_term_index(
        self,
        concept: ESGConcept,
    ) -> None:
        for concept_ids in self._term_index.values():
            concept_ids.discard(concept.concept_id)

        empty_terms = [
            term
            for term, concept_ids in self._term_index.items()
            if not concept_ids
        ]

        for term in empty_terms:
            del self._term_index[term]

    @staticmethod
    def _overlap_ratio(
        source_terms: set[str],
        target_terms: set[str],
    ) -> float:
        if not source_terms or not target_terms:
            return 0.0

        overlap = source_terms.intersection(target_terms)

        return len(overlap) / min(
            len(source_terms),
            len(target_terms),
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

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        if value is None or not str(value).strip():
            raise ValueError("Kavram kimliği boş olamaz.")

        normalized = re.sub(
            r"[^a-z0-9_]+",
            "_",
            str(value).strip().casefold(),
        )

        return normalized.strip("_")

    @staticmethod
    def _normalize_pillar(
        value: ESGPillar | str,
    ) -> ESGPillar:
        if isinstance(value, ESGPillar):
            return value

        normalized = str(value).strip().casefold()

        for member in ESGPillar:
            if normalized in {
                member.value.casefold(),
                member.name.casefold(),
            }:
                return member

        raise ValueError(
            f"Geçersiz ESG sütunu: {value!r}",
        )

    @classmethod
    def _normalize_terms(
        cls,
        values: Iterable[str],
    ) -> tuple[str, ...]:
        normalized: dict[str, str] = {}

        for value in values:
            if value is None:
                continue

            cleaned = cls._clean_label(str(value))

            if not cleaned:
                continue

            normalized_key = cls.normalize_text(cleaned)
            normalized.setdefault(normalized_key, cleaned)

        return tuple(
            normalized[key]
            for key in sorted(normalized)
        )

    @classmethod
    def _normalize_standard_codes(
        cls,
        values: Iterable[str],
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    cls._normalize_standard_code(value)
                    for value in values
                    if value is not None and str(value).strip()
                },
            ),
        )

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
    def _clean_label(value: str) -> str:
        return " ".join(str(value).strip().split())

    @staticmethod
    def _default_relations() -> tuple[ESGRelation, ...]:
        return (
            ESGRelation(
                source_concept_id="greenhouse_gas_emissions",
                target_concept_id="climate_change",
                relation_type=ESGRelationType.RELATED,
                weight=0.95,
            ),
            ESGRelation(
                source_concept_id="scope_1_emissions",
                target_concept_id="greenhouse_gas_emissions",
                relation_type=ESGRelationType.NARROWER,
                weight=1.0,
            ),
            ESGRelation(
                source_concept_id="scope_2_emissions",
                target_concept_id="greenhouse_gas_emissions",
                relation_type=ESGRelationType.NARROWER,
                weight=1.0,
            ),
            ESGRelation(
                source_concept_id="scope_3_emissions",
                target_concept_id="greenhouse_gas_emissions",
                relation_type=ESGRelationType.NARROWER,
                weight=1.0,
            ),
            ESGRelation(
                source_concept_id="energy_consumption",
                target_concept_id="climate_change",
                relation_type=ESGRelationType.RELATED,
                weight=0.8,
            ),
            ESGRelation(
                source_concept_id="renewable_energy",
                target_concept_id="energy_consumption",
                relation_type=ESGRelationType.RELATED,
                weight=0.9,
            ),
            ESGRelation(
                source_concept_id="water_consumption",
                target_concept_id="water_management",
                relation_type=ESGRelationType.NARROWER,
                weight=1.0,
            ),
            ESGRelation(
                source_concept_id="waste_generated",
                target_concept_id="waste_management",
                relation_type=ESGRelationType.NARROWER,
                weight=1.0,
            ),
            ESGRelation(
                source_concept_id="recycling",
                target_concept_id="circular_economy",
                relation_type=ESGRelationType.RELATED,
                weight=0.95,
            ),
            ESGRelation(
                source_concept_id="employee_health_safety",
                target_concept_id="own_workforce",
                relation_type=ESGRelationType.RELATED,
                weight=0.95,
            ),
            ESGRelation(
                source_concept_id="employee_training",
                target_concept_id="own_workforce",
                relation_type=ESGRelationType.RELATED,
                weight=0.85,
            ),
            ESGRelation(
                source_concept_id="diversity_inclusion",
                target_concept_id="own_workforce",
                relation_type=ESGRelationType.RELATED,
                weight=0.9,
            ),
            ESGRelation(
                source_concept_id="anti_corruption",
                target_concept_id="business_ethics",
                relation_type=ESGRelationType.RELATED,
                weight=0.95,
            ),
            ESGRelation(
                source_concept_id="data_privacy",
                target_concept_id="customer_responsibility",
                relation_type=ESGRelationType.RELATED,
                weight=0.85,
            ),
        )

    @staticmethod
    def _default_concepts() -> tuple[ESGConcept, ...]:
        return (
            ESGConcept(
                concept_id="climate_change",
                preferred_label="Climate change",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "İklim değişikliği",
                    "Climate impact",
                    "İklim etkisi",
                    "Climate risk",
                    "İklim riski",
                ),
                keywords=(
                    "climate transition",
                    "iklim geçişi",
                    "decarbonization",
                    "karbonsuzlaşma",
                    "net zero",
                    "net sıfır",
                ),
                standard_codes=(
                    "ESRS E1",
                    "GRI 305",
                ),
            ),
            ESGConcept(
                concept_id="greenhouse_gas_emissions",
                preferred_label="Greenhouse gas emissions",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "GHG emissions",
                    "Sera gazı emisyonları",
                    "Carbon emissions",
                    "Karbon emisyonları",
                    "Operational emissions",
                    "Operasyonel emisyonlar",
                ),
                keywords=(
                    "emissions",
                    "emisyonlar",
                    "carbon footprint",
                    "karbon ayak izi",
                    "tCO2e",
                    "CO2 equivalent",
                ),
                metrics=(
                    "tCO2e",
                    "kg CO2e",
                    "GHG intensity",
                    "Emissions intensity",
                ),
                standard_codes=(
                    "ESRS E1",
                    "GRI 305-1",
                    "GRI 305-2",
                    "GRI 305-3",
                    "GRI 305-4",
                ),
            ),
            ESGConcept(
                concept_id="scope_1_emissions",
                preferred_label="Scope 1 emissions",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Kapsam 1 emisyonları",
                    "Direct GHG emissions",
                    "Doğrudan sera gazı emisyonları",
                    "Direct emissions",
                    "Doğrudan emisyonlar",
                ),
                metrics=(
                    "Scope 1 tCO2e",
                    "Direct emissions tCO2e",
                ),
                standard_codes=(
                    "GRI 305-1",
                    "ESRS E1",
                ),
            ),
            ESGConcept(
                concept_id="scope_2_emissions",
                preferred_label="Scope 2 emissions",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Kapsam 2 emisyonları",
                    "Energy indirect emissions",
                    "Enerji kaynaklı dolaylı emisyonlar",
                    "Purchased electricity emissions",
                ),
                metrics=(
                    "Scope 2 location based",
                    "Scope 2 market based",
                ),
                standard_codes=(
                    "GRI 305-2",
                    "ESRS E1",
                ),
            ),
            ESGConcept(
                concept_id="scope_3_emissions",
                preferred_label="Scope 3 emissions",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Kapsam 3 emisyonları",
                    "Value chain emissions",
                    "Değer zinciri emisyonları",
                    "Other indirect emissions",
                    "Diğer dolaylı emisyonlar",
                ),
                metrics=(
                    "Scope 3 tCO2e",
                    "Value chain emissions tCO2e",
                ),
                standard_codes=(
                    "GRI 305-3",
                    "ESRS E1",
                ),
            ),
            ESGConcept(
                concept_id="energy_consumption",
                preferred_label="Energy consumption",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Enerji tüketimi",
                    "Total energy consumed",
                    "Toplam enerji tüketimi",
                    "Electricity consumption",
                    "Elektrik tüketimi",
                ),
                metrics=(
                    "MWh",
                    "GWh",
                    "GJ",
                    "Energy intensity",
                ),
                standard_codes=(
                    "GRI 302-1",
                    "GRI 302-3",
                    "ESRS E1",
                ),
            ),
            ESGConcept(
                concept_id="renewable_energy",
                preferred_label="Renewable energy",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Yenilenebilir enerji",
                    "Clean energy",
                    "Temiz enerji",
                    "Renewable electricity",
                    "Yenilenebilir elektrik",
                ),
                metrics=(
                    "Renewable energy percentage",
                    "Renewable electricity MWh",
                ),
                standard_codes=(
                    "GRI 302-1",
                    "ESRS E1",
                ),
            ),
            ESGConcept(
                concept_id="water_management",
                preferred_label="Water management",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Su yönetimi",
                    "Water resources",
                    "Su kaynakları",
                    "Water stewardship",
                ),
                keywords=(
                    "water stress",
                    "su stresi",
                    "freshwater",
                    "tatlı su",
                    "wastewater",
                    "atık su",
                ),
                standard_codes=(
                    "GRI 303",
                    "ESRS E3",
                ),
            ),
            ESGConcept(
                concept_id="water_consumption",
                preferred_label="Water consumption",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Su tüketimi",
                    "Water withdrawal",
                    "Su çekimi",
                    "Freshwater consumption",
                ),
                metrics=(
                    "m3",
                    "Megalitre",
                    "Water intensity",
                ),
                standard_codes=(
                    "GRI 303-3",
                    "ESRS E3",
                ),
            ),
            ESGConcept(
                concept_id="waste_management",
                preferred_label="Waste management",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Atık yönetimi",
                    "Waste disposal",
                    "Atık bertarafı",
                    "Hazardous waste management",
                ),
                standard_codes=(
                    "GRI 306",
                    "ESRS E5",
                ),
            ),
            ESGConcept(
                concept_id="waste_generated",
                preferred_label="Waste generated",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Oluşan atık",
                    "Total waste",
                    "Toplam atık",
                    "Generated waste",
                ),
                metrics=(
                    "tonnes",
                    "metric tons",
                    "waste intensity",
                ),
                standard_codes=(
                    "GRI 306-3",
                    "ESRS E5",
                ),
            ),
            ESGConcept(
                concept_id="recycling",
                preferred_label="Recycling",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Geri dönüşüm",
                    "Recycled waste",
                    "Geri dönüştürülen atık",
                    "Waste recovery",
                    "Atık geri kazanımı",
                ),
                metrics=(
                    "Recycling rate",
                    "Recycled waste tonnes",
                ),
                standard_codes=(
                    "GRI 306-4",
                    "ESRS E5",
                ),
            ),
            ESGConcept(
                concept_id="circular_economy",
                preferred_label="Circular economy",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Döngüsel ekonomi",
                    "Circularity",
                    "Döngüsellik",
                    "Resource efficiency",
                    "Kaynak verimliliği",
                ),
                standard_codes=(
                    "ESRS E5",
                    "GRI 301",
                    "GRI 306",
                ),
            ),
            ESGConcept(
                concept_id="biodiversity",
                preferred_label="Biodiversity",
                pillar=ESGPillar.ENVIRONMENTAL,
                aliases=(
                    "Biyolojik çeşitlilik",
                    "Ecosystems",
                    "Ekosistemler",
                    "Nature impact",
                    "Doğa etkisi",
                ),
                standard_codes=(
                    "ESRS E4",
                    "GRI 304",
                ),
            ),
            ESGConcept(
                concept_id="own_workforce",
                preferred_label="Own workforce",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "Kendi iş gücü",
                    "Employees",
                    "Çalışanlar",
                    "Workforce",
                    "İş gücü",
                ),
                standard_codes=(
                    "ESRS S1",
                    "GRI 2-7",
                    "GRI 401",
                ),
            ),
            ESGConcept(
                concept_id="employee_health_safety",
                preferred_label="Occupational health and safety",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "İş sağlığı ve güvenliği",
                    "Employee health and safety",
                    "Çalışan sağlığı ve güvenliği",
                    "Workplace safety",
                    "İş yeri güvenliği",
                ),
                keywords=(
                    "work accident",
                    "iş kazası",
                    "lost time injury",
                    "kayıp günlü yaralanma",
                    "fatality",
                    "ölümlü kaza",
                ),
                metrics=(
                    "LTIFR",
                    "TRIR",
                    "Fatalities",
                    "Lost time injuries",
                ),
                standard_codes=(
                    "GRI 403-9",
                    "ESRS S1",
                ),
            ),
            ESGConcept(
                concept_id="employee_training",
                preferred_label="Employee training",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "Çalışan eğitimi",
                    "Training and development",
                    "Eğitim ve gelişim",
                    "Learning and development",
                ),
                metrics=(
                    "Training hours per employee",
                    "Average training hours",
                ),
                standard_codes=(
                    "GRI 404-1",
                    "ESRS S1",
                ),
            ),
            ESGConcept(
                concept_id="diversity_inclusion",
                preferred_label="Diversity and inclusion",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "Çeşitlilik ve kapsayıcılık",
                    "Diversity and equal opportunity",
                    "Çeşitlilik ve fırsat eşitliği",
                    "Gender diversity",
                    "Cinsiyet çeşitliliği",
                ),
                metrics=(
                    "Female employee ratio",
                    "Women in management",
                    "Gender pay gap",
                ),
                standard_codes=(
                    "GRI 405-1",
                    "GRI 405-2",
                    "ESRS S1",
                ),
            ),
            ESGConcept(
                concept_id="human_rights",
                preferred_label="Human rights",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "İnsan hakları",
                    "Human rights due diligence",
                    "İnsan hakları durum tespiti",
                    "Forced labour",
                    "Zorla çalıştırma",
                    "Child labour",
                    "Çocuk işçiliği",
                ),
                standard_codes=(
                    "ESRS S1",
                    "ESRS S2",
                    "GRI 408",
                    "GRI 409",
                ),
            ),
            ESGConcept(
                concept_id="customer_responsibility",
                preferred_label="Customer responsibility",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "Müşteri sorumluluğu",
                    "Consumer responsibility",
                    "Tüketici sorumluluğu",
                    "Product responsibility",
                    "Ürün sorumluluğu",
                ),
                standard_codes=(
                    "ESRS S4",
                    "GRI 416",
                    "GRI 417",
                    "GRI 418",
                ),
            ),
            ESGConcept(
                concept_id="data_privacy",
                preferred_label="Data privacy",
                pillar=ESGPillar.SOCIAL,
                aliases=(
                    "Veri gizliliği",
                    "Customer privacy",
                    "Müşteri gizliliği",
                    "Personal data protection",
                    "Kişisel verilerin korunması",
                    "Data protection",
                    "Veri koruma",
                ),
                metrics=(
                    "Data breaches",
                    "Privacy complaints",
                ),
                standard_codes=(
                    "GRI 418-1",
                    "ESRS S4",
                ),
            ),
            ESGConcept(
                concept_id="business_ethics",
                preferred_label="Business ethics",
                pillar=ESGPillar.GOVERNANCE,
                aliases=(
                    "İş etiği",
                    "Corporate ethics",
                    "Kurumsal etik",
                    "Business conduct",
                    "İş davranışları",
                    "Code of conduct",
                    "Etik kurallar",
                ),
                standard_codes=(
                    "ESRS G1",
                    "GRI 2-23",
                ),
            ),
            ESGConcept(
                concept_id="anti_corruption",
                preferred_label="Anti-corruption",
                pillar=ESGPillar.GOVERNANCE,
                aliases=(
                    "Yolsuzlukla mücadele",
                    "Anti-bribery",
                    "Rüşvetle mücadele",
                    "Corruption prevention",
                    "Yolsuzluğun önlenmesi",
                ),
                metrics=(
                    "Corruption incidents",
                    "Anti-corruption training coverage",
                ),
                standard_codes=(
                    "GRI 205-2",
                    "GRI 205-3",
                    "ESRS G1",
                ),
            ),
            ESGConcept(
                concept_id="corporate_governance",
                preferred_label="Corporate governance",
                pillar=ESGPillar.GOVERNANCE,
                aliases=(
                    "Kurumsal yönetişim",
                    "Board governance",
                    "Yönetim kurulu yönetişimi",
                    "Governance structure",
                    "Yönetişim yapısı",
                ),
                standard_codes=(
                    "ESRS 2",
                    "GRI 2",
                ),
            ),
            ESGConcept(
                concept_id="risk_management",
                preferred_label="Risk management",
                pillar=ESGPillar.GOVERNANCE,
                aliases=(
                    "Risk yönetimi",
                    "Enterprise risk management",
                    "Kurumsal risk yönetimi",
                    "Sustainability risks",
                    "Sürdürülebilirlik riskleri",
                ),
                standard_codes=(
                    "ESRS 2",
                    "TCFD Risk Management",
                ),
            ),
            ESGConcept(
                concept_id="economic_value",
                preferred_label="Economic value generated and distributed",
                pillar=ESGPillar.ECONOMIC,
                aliases=(
                    "Üretilen ve dağıtılan ekonomik değer",
                    "Direct economic value",
                    "Doğrudan ekonomik değer",
                    "Economic contribution",
                    "Ekonomik katkı",
                ),
                metrics=(
                    "Revenue",
                    "Operating costs",
                    "Employee wages",
                    "Payments to governments",
                    "Community investments",
                ),
                standard_codes=(
                    "GRI 201-1",
                ),
            ),
            ESGConcept(
                concept_id="materiality",
                preferred_label="Materiality assessment",
                pillar=ESGPillar.CROSS_CUTTING,
                aliases=(
                    "Önemlilik değerlendirmesi",
                    "Material topics",
                    "Önemli konular",
                    "Double materiality",
                    "Çifte önemlilik",
                ),
                standard_codes=(
                    "GRI 3-1",
                    "ESRS 1",
                    "ESRS 2",
                ),
            ),
            ESGConcept(
                concept_id="stakeholder_engagement",
                preferred_label="Stakeholder engagement",
                pillar=ESGPillar.CROSS_CUTTING,
                aliases=(
                    "Paydaş katılımı",
                    "Stakeholder dialogue",
                    "Paydaş diyaloğu",
                    "Stakeholder consultation",
                    "Paydaş danışma süreci",
                ),
                standard_codes=(
                    "GRI 2-29",
                    "ESRS 2",
                ),
            ),
        )


__all__ = [
    "DuplicateESGConceptError",
    "ESGConcept",
    "ESGConceptMatch",
    "ESGConceptNotFoundError",
    "ESGOntology",
    "ESGOntologyError",
    "ESGPillar",
    "ESGRelation",
    "ESGRelationType",
    "QueryExpansion",
]