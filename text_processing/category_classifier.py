from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from .semantic_chunker import (
    SemanticChunk,
    SemanticChunkingResult,
)


logger = logging.getLogger(__name__)


class CategoryClassificationError(RuntimeError):
    """Chunk kategori sınıflandırması tamamlanamadığında yükseltilir."""


class ESGCategory(StrEnum):
    """Sürdürülebilirlik içeriğinin ana kategorileri."""

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    ECONOMIC = "economic"
    STANDARDS = "standards"
    GENERAL = "general"
    UNKNOWN = "unknown"


class ClassificationMethod(StrEnum):
    """Kategori sınıflandırmasında kullanılan yöntem."""

    RULE_BASED = "rule_based"
    KEYWORD_MATCH = "keyword_match"
    SECTION_PRIOR = "section_prior"
    COMBINED = "combined"
    FALLBACK = "fallback"


@dataclass(slots=True, frozen=True)
class CategoryScore:
    """Tek bir kategoriye ait sınıflandırma skoru."""

    category: ESGCategory

    raw_score: float
    normalized_score: float
    confidence: float

    matched_terms: tuple[str, ...]
    matched_phrases: tuple[str, ...]

    title_match_count: int
    path_match_count: int
    content_match_count: int

    negative_match_count: int

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["category"] = (
            self.category.value
        )

        return data


@dataclass(slots=True, frozen=True)
class TopicScore:
    """Kategori altında tespit edilen alt konu skoru."""

    topic: str
    category: ESGCategory

    score: float
    confidence: float

    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "category": self.category.value,
            "score": self.score,
            "confidence": self.confidence,
            "matched_terms": list(
                self.matched_terms
            ),
        }


@dataclass(slots=True, frozen=True)
class ChunkCategoryClassification:
    """Tek bir SemanticChunk için kategori sınıflandırma sonucu."""

    classification_id: str

    chunk_id: str
    document_id: str
    section_id: str

    primary_category: ESGCategory
    secondary_categories: tuple[ESGCategory, ...]

    category_scores: tuple[CategoryScore, ...]
    topic_scores: tuple[TopicScore, ...]

    detected_topics: tuple[str, ...]
    detected_standard_families: tuple[str, ...]

    confidence: float
    margin: float

    classification_method: ClassificationMethod

    is_multilabel: bool
    is_ambiguous: bool
    requires_review: bool

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification_id": (
                self.classification_id
            ),
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section_id": self.section_id,
            "primary_category": (
                self.primary_category.value
            ),
            "secondary_categories": [
                category.value
                for category
                in self.secondary_categories
            ],
            "category_scores": [
                score.to_dict()
                for score in self.category_scores
            ],
            "topic_scores": [
                score.to_dict()
                for score in self.topic_scores
            ],
            "detected_topics": list(
                self.detected_topics
            ),
            "detected_standard_families": list(
                self.detected_standard_families
            ),
            "confidence": self.confidence,
            "margin": self.margin,
            "classification_method": (
                self.classification_method.value
            ),
            "is_multilabel": self.is_multilabel,
            "is_ambiguous": self.is_ambiguous,
            "requires_review": (
                self.requires_review
            ),
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class CategoryClassificationResult:
    """Belgedeki bütün chunk sınıflandırmalarının özeti."""

    document_id: str

    chunk_count: int
    classified_chunk_count: int
    unknown_chunk_count: int
    ambiguous_chunk_count: int
    review_required_count: int

    category_distribution: dict[str, int]
    topic_distribution: dict[str, int]

    classifications: tuple[
        ChunkCategoryClassification,
        ...
    ]

    average_confidence: float

    content_hash: str

    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "classified_chunk_count": (
                self.classified_chunk_count
            ),
            "unknown_chunk_count": (
                self.unknown_chunk_count
            ),
            "ambiguous_chunk_count": (
                self.ambiguous_chunk_count
            ),
            "review_required_count": (
                self.review_required_count
            ),
            "category_distribution": dict(
                self.category_distribution
            ),
            "topic_distribution": dict(
                self.topic_distribution
            ),
            "classifications": [
                classification.to_dict()
                for classification
                in self.classifications
            ],
            "average_confidence": (
                self.average_confidence
            ),
            "content_hash": self.content_hash,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class _CategoryDefinition:
    """Kategori sözlüğünün iç temsili."""

    category: ESGCategory

    keywords: tuple[str, ...]
    phrases: tuple[str, ...]
    negative_terms: tuple[str, ...]

    section_terms: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _TopicDefinition:
    """Alt konu sözlüğünün iç temsili."""

    name: str
    category: ESGCategory

    keywords: tuple[str, ...]
    phrases: tuple[str, ...]


class CategoryClassifier:
    """
    SemanticChunk içeriklerini ESG ve standart kategorilerinde sınıflandırır.

    Sınıflandırma sinyalleri:

    - Bölüm başlığı
    - Bölüm hiyerarşi yolu
    - Chunk içeriği
    - Tek sözcüklü anahtar terimler
    - Çok sözcüklü ifadeler
    - Standart kodları
    - Negatif terimler
    - Kategori skorları arasındaki fark

    Ağırlıklandırma:

    - Bölüm başlığı eşleşmesi en güçlü sinyaldir.
    - Bölüm yolu, üst bölüm bağlamını taşır.
    - Chunk içeriği temel semantik sinyali sağlar.
    - Tam ifade eşleşmeleri tek sözcük eşleşmelerinden daha değerlidir.
    - Aynı terimin tekrar etmesi sınırlı bir katkı sağlar; skorun
      kontrolsüz yükselmesi engellenir.

    Bu modül çok etiketli çalışır. Örneğin çalışan eğitimi ve etik uyum
    içeren bir chunk hem social hem governance kategorisi alabilir.
    """

    _WHITESPACE_RE = re.compile(
        r"\s+"
    )

    _TOKEN_RE = re.compile(
        r"[a-z0-9]+(?:[-_.][a-z0-9]+)*",
        re.IGNORECASE,
    )

    _STANDARD_PATTERNS: tuple[
        tuple[re.Pattern[str], str],
        ...
    ] = (
        (
            re.compile(
                r"\bgri(?:\s+\d{1,3}(?:-\d)?)?\b",
                re.IGNORECASE,
            ),
            "GRI",
        ),
        (
            re.compile(
                r"\besrs(?:\s+[e-sg]\d)?\b",
                re.IGNORECASE,
            ),
            "ESRS",
        ),
        (
            re.compile(
                r"\bsasb\b",
                re.IGNORECASE,
            ),
            "SASB",
        ),
        (
            re.compile(
                r"\btcfd\b",
                re.IGNORECASE,
            ),
            "TCFD",
        ),
        (
            re.compile(
                r"\bissb\b",
                re.IGNORECASE,
            ),
            "ISSB",
        ),
        (
            re.compile(
                r"\bifrs\s+s[12]\b",
                re.IGNORECASE,
            ),
            "IFRS Sustainability",
        ),
        (
            re.compile(
                r"\bcdp\b",
                re.IGNORECASE,
            ),
            "CDP",
        ),
        (
            re.compile(
                r"\bun\s+global\s+compact\b",
                re.IGNORECASE,
            ),
            "UN Global Compact",
        ),
        (
            re.compile(
                r"\bsdgs?\b|\bsustainable development goals?\b",
                re.IGNORECASE,
            ),
            "SDG",
        ),
        (
            re.compile(
                r"\biso\s*(?:14001|45001|50001|26000|37001)\b",
                re.IGNORECASE,
            ),
            "ISO",
        ),
    )

    _CATEGORY_DEFINITIONS: tuple[
        _CategoryDefinition,
        ...
    ] = (
        _CategoryDefinition(
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "environment",
                "environmental",
                "cevre",
                "cevresel",
                "climate",
                "iklim",
                "emission",
                "emissions",
                "emisyon",
                "emisyonlar",
                "carbon",
                "karbon",
                "energy",
                "enerji",
                "water",
                "su",
                "waste",
                "atik",
                "biodiversity",
                "biyocesitlilik",
                "pollution",
                "kirlilik",
                "recycling",
                "geri donusum",
                "renewable",
                "yenilenebilir",
                "ecosystem",
                "ekosistem",
                "resource",
                "kaynak",
                "scope",
                "ghg",
                "co2",
                "co2e",
                "tco2e",
            ),
            phrases=(
                "climate change",
                "iklim degisikligi",
                "greenhouse gas",
                "sera gazi",
                "scope 1",
                "scope 2",
                "scope 3",
                "carbon footprint",
                "karbon ayak izi",
                "energy efficiency",
                "enerji verimliligi",
                "renewable energy",
                "yenilenebilir enerji",
                "water consumption",
                "su tuketimi",
                "water withdrawal",
                "su cekimi",
                "waste management",
                "atik yonetimi",
                "circular economy",
                "dongusel ekonomi",
                "nature positive",
                "net zero",
                "zero waste",
                "sifir atik",
                "environmental impact",
                "cevresel etki",
            ),
            negative_terms=(
                "work environment",
                "calisma ortami",
                "business environment",
                "is ortami",
            ),
            section_terms=(
                "environmental performance",
                "cevresel performans",
                "environment",
                "cevre",
                "climate",
                "iklim",
                "energy",
                "enerji",
                "water",
                "su",
                "waste",
                "atik",
            ),
        ),
        _CategoryDefinition(
            category=ESGCategory.SOCIAL,
            keywords=(
                "social",
                "sosyal",
                "employee",
                "employees",
                "calisan",
                "calisanlar",
                "workforce",
                "isgucu",
                "worker",
                "workers",
                "isci",
                "health",
                "saglik",
                "safety",
                "guvenlik",
                "injury",
                "yaralanma",
                "accident",
                "kaza",
                "training",
                "egitim",
                "diversity",
                "cesitlilik",
                "inclusion",
                "kapsayicilik",
                "equality",
                "esitlik",
                "gender",
                "cinsiyet",
                "community",
                "toplum",
                "human",
                "insan",
                "rights",
                "haklari",
                "customer",
                "musteri",
                "consumer",
                "tuketici",
                "volunteer",
                "gonullu",
                "wellbeing",
                "refah",
                "welfare",
                "union",
                "sendika",
                "talent",
                "yetenek",
            ),
            phrases=(
                "human rights",
                "insan haklari",
                "occupational health and safety",
                "is sagligi ve guvenligi",
                "employee engagement",
                "calisan bagliligi",
                "employee satisfaction",
                "calisan memnuniyeti",
                "diversity and inclusion",
                "cesitlilik ve kapsayicilik",
                "equal opportunity",
                "firsat esitligi",
                "gender equality",
                "toplumsal cinsiyet esitligi",
                "community investment",
                "toplumsal yatirim",
                "social impact",
                "sosyal etki",
                "customer satisfaction",
                "musteri memnuniyeti",
                "product safety",
                "urun guvenligi",
                "child labor",
                "cocuk isciligi",
                "forced labor",
                "zorla calistirma",
                "living wage",
                "adil ucret",
            ),
            negative_terms=(
                "social media",
                "sosyal medya",
                "social network",
            ),
            section_terms=(
                "social performance",
                "sosyal performans",
                "employees",
                "calisanlar",
                "human rights",
                "insan haklari",
                "health and safety",
                "saglik ve guvenlik",
                "community",
                "toplum",
            ),
        ),
        _CategoryDefinition(
            category=ESGCategory.GOVERNANCE,
            keywords=(
                "governance",
                "yonetisim",
                "board",
                "kurul",
                "committee",
                "komite",
                "ethics",
                "etik",
                "compliance",
                "uyum",
                "corruption",
                "yolsuzluk",
                "bribery",
                "rusvet",
                "risk",
                "denetim",
                "audit",
                "policy",
                "politika",
                "procedure",
                "prosedur",
                "control",
                "kontrol",
                "transparency",
                "seffaflik",
                "accountability",
                "hesap verebilirlik",
                "privacy",
                "gizlilik",
                "cybersecurity",
                "siber guvenlik",
                "whistleblowing",
                "ihbar",
                "integrity",
                "durustluk",
                "management",
                "yonetim",
                "stakeholder",
                "paydas",
                "materiality",
                "onceliklendirme",
            ),
            phrases=(
                "corporate governance",
                "kurumsal yonetisim",
                "board of directors",
                "yonetim kurulu",
                "business ethics",
                "is etigi",
                "ethics and compliance",
                "etik ve uyum",
                "anti corruption",
                "yolsuzlukla mucadele",
                "anti bribery",
                "rusvetle mucadele",
                "risk management",
                "risk yonetimi",
                "internal control",
                "ic kontrol",
                "internal audit",
                "ic denetim",
                "data privacy",
                "veri gizliligi",
                "information security",
                "bilgi guvenligi",
                "stakeholder engagement",
                "paydas katilimi",
                "materiality assessment",
                "onceliklendirme analizi",
                "sustainability governance",
                "surdurulebilirlik yonetisimi",
            ),
            negative_terms=(
                "water management",
                "su yonetimi",
                "waste management",
                "atik yonetimi",
                "energy management",
                "enerji yonetimi",
            ),
            section_terms=(
                "governance",
                "yonetisim",
                "corporate governance",
                "kurumsal yonetisim",
                "ethics",
                "etik",
                "risk management",
                "risk yonetimi",
                "stakeholder engagement",
                "paydas katilimi",
            ),
        ),
        _CategoryDefinition(
            category=ESGCategory.ECONOMIC,
            keywords=(
                "economic",
                "ekonomik",
                "financial",
                "finansal",
                "revenue",
                "gelir",
                "profit",
                "kar",
                "loss",
                "zarar",
                "investment",
                "yatirim",
                "cost",
                "maliyet",
                "tax",
                "vergi",
                "market",
                "pazar",
                "procurement",
                "satinalma",
                "supplier",
                "tedarikci",
                "supply",
                "tedarik",
                "innovation",
                "inovasyon",
                "research",
                "arastirma",
                "development",
                "gelistirme",
                "digitalization",
                "dijitallesme",
                "productivity",
                "verimlilik",
                "production",
                "uretim",
                "sales",
                "satis",
                "value",
                "deger",
                "capital",
                "sermaye",
                "asset",
                "varlik",
                "ebitda",
                "turnover",
                "ciro",
            ),
            phrases=(
                "economic performance",
                "ekonomik performans",
                "financial performance",
                "finansal performans",
                "economic value generated",
                "yaratilan ekonomik deger",
                "direct economic value",
                "dogrudan ekonomik deger",
                "local procurement",
                "yerel satin alma",
                "supply chain",
                "tedarik zinciri",
                "responsible procurement",
                "sorumlu tedarik",
                "research and development",
                "arastirma ve gelistirme",
                "sustainable finance",
                "surdurulebilir finans",
                "green finance",
                "yesil finans",
                "capital expenditure",
                "sermaye harcamasi",
                "operating expenditure",
                "faaliyet gideri",
            ),
            negative_terms=(
                "energy efficiency",
                "enerji verimliligi",
                "resource efficiency",
                "kaynak verimliligi",
            ),
            section_terms=(
                "economic performance",
                "ekonomik performans",
                "financial performance",
                "finansal performans",
                "innovation",
                "inovasyon",
                "supply chain",
                "tedarik zinciri",
            ),
        ),
        _CategoryDefinition(
            category=ESGCategory.STANDARDS,
            keywords=(
                "standard",
                "standards",
                "standart",
                "standartlar",
                "disclosure",
                "bildirim",
                "index",
                "indeks",
                "assurance",
                "guvence",
                "framework",
                "cerceve",
                "requirement",
                "gereklilik",
                "alignment",
                "uyum",
                "reference",
                "referans",
                "gri",
                "esrs",
                "sasb",
                "tcfd",
                "issb",
                "ifrs",
                "cdp",
                "sdg",
                "iso",
            ),
            phrases=(
                "content index",
                "icerik indeksi",
                "gri content index",
                "gri icerik indeksi",
                "esrs content index",
                "esrs icerik indeksi",
                "reporting standards",
                "raporlama standartlari",
                "reporting framework",
                "raporlama cercevesi",
                "limited assurance",
                "sinirli guvence",
                "reasonable assurance",
                "makul guvence",
                "external assurance",
                "bagimsiz guvence",
                "basis of preparation",
                "hazirlama esaslari",
                "disclosure requirement",
                "aciklama gerekliligi",
            ),
            negative_terms=(),
            section_terms=(
                "gri content index",
                "gri icerik indeksi",
                "esrs index",
                "esrs indeksi",
                "standards index",
                "standartlar indeksi",
                "assurance statement",
                "guvence beyani",
            ),
        ),
    )

    _TOPIC_DEFINITIONS: tuple[
        _TopicDefinition,
        ...
    ] = (
        _TopicDefinition(
            name="climate_change",
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "climate",
                "iklim",
                "decarbonization",
                "karbonsuzlasma",
            ),
            phrases=(
                "climate change",
                "iklim degisikligi",
                "climate risk",
                "iklim riski",
                "net zero",
            ),
        ),
        _TopicDefinition(
            name="greenhouse_gas_emissions",
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "emission",
                "emissions",
                "emisyon",
                "ghg",
                "co2e",
            ),
            phrases=(
                "greenhouse gas",
                "sera gazi",
                "scope 1",
                "scope 2",
                "scope 3",
            ),
        ),
        _TopicDefinition(
            name="energy",
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "energy",
                "enerji",
                "electricity",
                "elektrik",
            ),
            phrases=(
                "energy consumption",
                "enerji tuketimi",
                "renewable energy",
                "yenilenebilir enerji",
                "energy efficiency",
                "enerji verimliligi",
            ),
        ),
        _TopicDefinition(
            name="water",
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "water",
                "su",
            ),
            phrases=(
                "water consumption",
                "su tuketimi",
                "water withdrawal",
                "su cekimi",
                "water stress",
                "su stresi",
            ),
        ),
        _TopicDefinition(
            name="waste",
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "waste",
                "atik",
                "recycling",
            ),
            phrases=(
                "waste management",
                "atik yonetimi",
                "zero waste",
                "sifir atik",
            ),
        ),
        _TopicDefinition(
            name="biodiversity",
            category=ESGCategory.ENVIRONMENTAL,
            keywords=(
                "biodiversity",
                "biyocesitlilik",
                "ecosystem",
                "ekosistem",
            ),
            phrases=(
                "natural habitat",
                "dogal yasam alani",
            ),
        ),
        _TopicDefinition(
            name="occupational_health_and_safety",
            category=ESGCategory.SOCIAL,
            keywords=(
                "injury",
                "yaralanma",
                "accident",
                "kaza",
                "fatality",
            ),
            phrases=(
                "occupational health and safety",
                "is sagligi ve guvenligi",
                "lost time injury",
                "kayip gunlu kaza",
            ),
        ),
        _TopicDefinition(
            name="employees",
            category=ESGCategory.SOCIAL,
            keywords=(
                "employee",
                "employees",
                "calisan",
                "calisanlar",
                "workforce",
            ),
            phrases=(
                "employee engagement",
                "calisan bagliligi",
                "employee turnover",
                "calisan devir orani",
            ),
        ),
        _TopicDefinition(
            name="diversity_and_inclusion",
            category=ESGCategory.SOCIAL,
            keywords=(
                "diversity",
                "cesitlilik",
                "inclusion",
                "kapsayicilik",
                "gender",
                "cinsiyet",
            ),
            phrases=(
                "diversity and inclusion",
                "cesitlilik ve kapsayicilik",
                "gender equality",
                "toplumsal cinsiyet esitligi",
            ),
        ),
        _TopicDefinition(
            name="human_rights",
            category=ESGCategory.SOCIAL,
            keywords=(
                "human",
                "insan",
                "rights",
                "haklari",
            ),
            phrases=(
                "human rights",
                "insan haklari",
                "child labor",
                "cocuk isciligi",
                "forced labor",
                "zorla calistirma",
            ),
        ),
        _TopicDefinition(
            name="community",
            category=ESGCategory.SOCIAL,
            keywords=(
                "community",
                "toplum",
                "social",
                "sosyal",
            ),
            phrases=(
                "community investment",
                "toplumsal yatirim",
                "social impact",
                "sosyal etki",
            ),
        ),
        _TopicDefinition(
            name="corporate_governance",
            category=ESGCategory.GOVERNANCE,
            keywords=(
                "governance",
                "yonetisim",
                "board",
                "kurul",
            ),
            phrases=(
                "corporate governance",
                "kurumsal yonetisim",
                "board of directors",
                "yonetim kurulu",
            ),
        ),
        _TopicDefinition(
            name="ethics_and_compliance",
            category=ESGCategory.GOVERNANCE,
            keywords=(
                "ethics",
                "etik",
                "compliance",
                "uyum",
                "bribery",
                "rusvet",
            ),
            phrases=(
                "ethics and compliance",
                "etik ve uyum",
                "anti corruption",
                "yolsuzlukla mucadele",
            ),
        ),
        _TopicDefinition(
            name="risk_management",
            category=ESGCategory.GOVERNANCE,
            keywords=(
                "risk",
                "control",
                "kontrol",
                "audit",
                "denetim",
            ),
            phrases=(
                "risk management",
                "risk yonetimi",
                "internal control",
                "ic kontrol",
            ),
        ),
        _TopicDefinition(
            name="data_privacy_and_security",
            category=ESGCategory.GOVERNANCE,
            keywords=(
                "privacy",
                "gizlilik",
                "cybersecurity",
                "siber",
            ),
            phrases=(
                "data privacy",
                "veri gizliligi",
                "information security",
                "bilgi guvenligi",
            ),
        ),
        _TopicDefinition(
            name="economic_performance",
            category=ESGCategory.ECONOMIC,
            keywords=(
                "revenue",
                "gelir",
                "profit",
                "kar",
                "ebitda",
            ),
            phrases=(
                "economic performance",
                "ekonomik performans",
                "financial performance",
                "finansal performans",
            ),
        ),
        _TopicDefinition(
            name="supply_chain",
            category=ESGCategory.ECONOMIC,
            keywords=(
                "supplier",
                "tedarikci",
                "procurement",
                "satinalma",
            ),
            phrases=(
                "supply chain",
                "tedarik zinciri",
                "responsible procurement",
                "sorumlu tedarik",
            ),
        ),
        _TopicDefinition(
            name="innovation",
            category=ESGCategory.ECONOMIC,
            keywords=(
                "innovation",
                "inovasyon",
                "research",
                "arastirma",
            ),
            phrases=(
                "research and development",
                "arastirma ve gelistirme",
                "digital transformation",
                "dijital donusum",
            ),
        ),
        _TopicDefinition(
            name="reporting_standards",
            category=ESGCategory.STANDARDS,
            keywords=(
                "gri",
                "esrs",
                "sasb",
                "tcfd",
                "issb",
                "ifrs",
            ),
            phrases=(
                "content index",
                "icerik indeksi",
                "reporting standard",
                "raporlama standardi",
            ),
        ),
        _TopicDefinition(
            name="external_assurance",
            category=ESGCategory.STANDARDS,
            keywords=(
                "assurance",
                "guvence",
                "audit",
                "denetim",
            ),
            phrases=(
                "external assurance",
                "bagimsiz guvence",
                "limited assurance",
                "sinirli guvence",
            ),
        ),
    )

    def __init__(
        self,
        *,
        minimum_primary_score: float = 1.75,
        minimum_secondary_score: float = 1.25,
        minimum_secondary_ratio: float = 0.55,
        ambiguity_margin: float = 0.10,
        review_confidence_threshold: float = 0.50,
        maximum_secondary_categories: int = 2,
        maximum_topics: int = 5,
        title_weight: float = 3.0,
        path_weight: float = 2.0,
        content_weight: float = 1.0,
        phrase_multiplier: float = 2.25,
        section_prior_weight: float = 1.50,
        negative_penalty: float = 1.50,
        repeated_term_limit: int = 3,
        include_general_category: bool = True,
    ) -> None:
        for name, value in {
            "minimum_primary_score": minimum_primary_score,
            "minimum_secondary_score": minimum_secondary_score,
            "title_weight": title_weight,
            "path_weight": path_weight,
            "content_weight": content_weight,
            "phrase_multiplier": phrase_multiplier,
            "section_prior_weight": section_prior_weight,
            "negative_penalty": negative_penalty,
        }.items():
            if value < 0:
                raise ValueError(
                    f"{name} negatif olamaz."
                )

        for name, value in {
            "minimum_secondary_ratio": minimum_secondary_ratio,
            "ambiguity_margin": ambiguity_margin,
            "review_confidence_threshold": (
                review_confidence_threshold
            ),
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 0 ile 1 arasında olmalıdır."
                )

        if maximum_secondary_categories < 0:
            raise ValueError(
                "maximum_secondary_categories negatif olamaz."
            )

        if maximum_topics < 1:
            raise ValueError(
                "maximum_topics pozitif olmalıdır."
            )

        if repeated_term_limit < 1:
            raise ValueError(
                "repeated_term_limit pozitif olmalıdır."
            )

        self.minimum_primary_score = (
            minimum_primary_score
        )

        self.minimum_secondary_score = (
            minimum_secondary_score
        )

        self.minimum_secondary_ratio = (
            minimum_secondary_ratio
        )

        self.ambiguity_margin = ambiguity_margin

        self.review_confidence_threshold = (
            review_confidence_threshold
        )

        self.maximum_secondary_categories = (
            maximum_secondary_categories
        )

        self.maximum_topics = maximum_topics

        self.title_weight = title_weight
        self.path_weight = path_weight
        self.content_weight = content_weight

        self.phrase_multiplier = (
            phrase_multiplier
        )

        self.section_prior_weight = (
            section_prior_weight
        )

        self.negative_penalty = (
            negative_penalty
        )

        self.repeated_term_limit = (
            repeated_term_limit
        )

        self.include_general_category = (
            include_general_category
        )

    def classify_document(
        self,
        chunking_result: SemanticChunkingResult,
    ) -> CategoryClassificationResult:
        """Belgedeki bütün semantic chunk'ları sınıflandırır."""

        if not isinstance(
            chunking_result,
            SemanticChunkingResult,
        ):
            raise TypeError(
                "chunking_result SemanticChunkingResult olmalıdır."
            )

        classifications: list[
            ChunkCategoryClassification
        ] = []

        warnings: list[str] = []

        for chunk in chunking_result.chunks:
            try:
                classification = (
                    self.classify_chunk(
                        chunk
                    )
                )

            except Exception as exc:
                message = (
                    f"Chunk sınıflandırılamadı: "
                    f"{chunk.chunk_id}: {exc}"
                )

                warnings.append(message)
                logger.warning(message)

                continue

            classifications.append(
                classification
            )

        category_distribution = Counter(
            classification.primary_category.value
            for classification in classifications
        )

        topic_distribution = Counter(
            topic
            for classification in classifications
            for topic
            in classification.detected_topics
        )

        average_confidence = (
            sum(
                classification.confidence
                for classification
                in classifications
            )
            / len(classifications)
            if classifications
            else 0.0
        )

        content_hash = self._calculate_result_hash(
            classifications
        )

        result = CategoryClassificationResult(
            document_id=chunking_result.document_id,
            chunk_count=chunking_result.chunk_count,
            classified_chunk_count=len(
                classifications
            ),
            unknown_chunk_count=sum(
                classification.primary_category
                == ESGCategory.UNKNOWN
                for classification
                in classifications
            ),
            ambiguous_chunk_count=sum(
                classification.is_ambiguous
                for classification
                in classifications
            ),
            review_required_count=sum(
                classification.requires_review
                for classification
                in classifications
            ),
            category_distribution=dict(
                sorted(
                    category_distribution.items()
                )
            ),
            topic_distribution=dict(
                sorted(
                    topic_distribution.items()
                )
            ),
            classifications=tuple(
                classifications
            ),
            average_confidence=round(
                average_confidence,
                4,
            ),
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "source_chunking_hash": (
                    chunking_result.content_hash
                ),
                "minimum_primary_score": (
                    self.minimum_primary_score
                ),
                "minimum_secondary_score": (
                    self.minimum_secondary_score
                ),
                "minimum_secondary_ratio": (
                    self.minimum_secondary_ratio
                ),
                "classification_method": (
                    ClassificationMethod.RULE_BASED.value
                ),
            },
        )

        logger.info(
            "%d chunk sınıflandırıldı. Dağılım: %s",
            result.classified_chunk_count,
            result.category_distribution,
        )

        return result

    def classify_chunks(
        self,
        chunks: Sequence[SemanticChunk],
    ) -> list[ChunkCategoryClassification]:
        """Bir SemanticChunk listesini sınıflandırır."""

        results: list[
            ChunkCategoryClassification
        ] = []

        for chunk in chunks:
            try:
                results.append(
                    self.classify_chunk(chunk)
                )

            except Exception as exc:
                logger.warning(
                    "Chunk kategori sınıflandırması başarısız: "
                    "%s: %s",
                    chunk.chunk_id,
                    exc,
                )

        return results

    def classify_chunk(
        self,
        chunk: SemanticChunk,
    ) -> ChunkCategoryClassification:
        """Tek bir semantic chunk için çok etiketli kategori üretir."""

        if not isinstance(
            chunk,
            SemanticChunk,
        ):
            raise TypeError(
                "chunk SemanticChunk olmalıdır."
            )

        title_text = self.normalize_for_search(
            chunk.section_title
        )

        path_text = self.normalize_for_search(
            " ".join(
                chunk.section_path
            )
        )

        content_text = self.normalize_for_search(
            (
                chunk.normalized_text
                or chunk.text
                or chunk.contextualized_text
            )
        )

        if not any(
            (
                title_text,
                path_text,
                content_text,
            )
        ):
            raise CategoryClassificationError(
                "Chunk sınıflandırılabilir metin içermiyor."
            )

        category_scores = [
            self._score_category(
                definition=definition,
                title_text=title_text,
                path_text=path_text,
                content_text=content_text,
            )
            for definition
            in self._CATEGORY_DEFINITIONS
        ]

        category_scores = (
            self._normalize_category_scores(
                category_scores
            )
        )

        ordered_scores = sorted(
            category_scores,
            key=lambda score: (
                score.normalized_score,
                score.raw_score,
            ),
            reverse=True,
        )

        primary_score = ordered_scores[0]

        if (
            primary_score.raw_score
            < self.minimum_primary_score
        ):
            primary_category = (
                ESGCategory.GENERAL
                if self.include_general_category
                else ESGCategory.UNKNOWN
            )

            classification_method = (
                ClassificationMethod.FALLBACK
            )

        else:
            primary_category = (
                primary_score.category
            )

            classification_method = (
                ClassificationMethod.COMBINED
            )

        secondary_categories = (
            self._select_secondary_categories(
                ordered_scores=ordered_scores,
                primary_category=(
                    primary_category
                ),
            )
        )

        second_normalized_score = (
            ordered_scores[1].normalized_score
            if len(ordered_scores) > 1
            else 0.0
        )

        margin = max(
            primary_score.normalized_score
            - second_normalized_score,
            0.0,
        )

        is_ambiguous = (
            primary_category
            not in {
                ESGCategory.UNKNOWN,
                ESGCategory.GENERAL,
            }
            and margin
            <= self.ambiguity_margin
        )

        topic_scores = self._score_topics(
            title_text=title_text,
            path_text=path_text,
            content_text=content_text,
            allowed_categories={
                primary_category,
                *secondary_categories,
            },
        )

        detected_topics = tuple(
            score.topic
            for score in topic_scores
        )

        standard_families = (
            self._detect_standard_families(
                " ".join(
                    (
                        title_text,
                        path_text,
                        content_text,
                    )
                )
            )
        )

        confidence = (
            self._calculate_classification_confidence(
                primary_score=primary_score,
                margin=margin,
                matched_category_count=sum(
                    score.raw_score
                    >= self.minimum_secondary_score
                    for score in ordered_scores
                ),
                primary_category=(
                    primary_category
                ),
                chunk_confidence=(
                    chunk.chunk_confidence
                ),
            )
        )

        requires_review = (
            confidence
            < self.review_confidence_threshold
            or is_ambiguous
            or primary_category
            == ESGCategory.UNKNOWN
        )

        warnings: list[str] = []

        if primary_category == ESGCategory.UNKNOWN:
            warnings.append(
                "Chunk için güvenilir kategori bulunamadı."
            )

        elif primary_category == ESGCategory.GENERAL:
            warnings.append(
                "Chunk genel sürdürülebilirlik içeriği olarak "
                "sınıflandırıldı."
            )

        if is_ambiguous:
            warnings.append(
                "En yüksek iki kategori skoru birbirine yakın."
            )

        if requires_review:
            warnings.append(
                "Sınıflandırma manuel inceleme için işaretlendi."
            )

        content_hash = hashlib.sha256(
            (
                f"{chunk.content_hash}:"
                f"{primary_category.value}:"
                f"{','.join(detected_topics)}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        classification_id = hashlib.sha256(
            (
                f"{chunk.chunk_id}:"
                f"{content_hash}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return ChunkCategoryClassification(
            classification_id=(
                classification_id
            ),
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            section_id=chunk.section_id,
            primary_category=(
                primary_category
            ),
            secondary_categories=(
                secondary_categories
            ),
            category_scores=tuple(
                ordered_scores
            ),
            topic_scores=tuple(
                topic_scores
            ),
            detected_topics=(
                detected_topics
            ),
            detected_standard_families=(
                standard_families
            ),
            confidence=confidence,
            margin=round(
                margin,
                4,
            ),
            classification_method=(
                classification_method
            ),
            is_multilabel=bool(
                secondary_categories
            ),
            is_ambiguous=is_ambiguous,
            requires_review=(
                requires_review
            ),
            content_hash=content_hash,
            warnings=tuple(warnings),
            metadata={
                "chunk_start_page": (
                    chunk.start_page
                ),
                "chunk_end_page": (
                    chunk.end_page
                ),
                "section_title": (
                    chunk.section_title
                ),
                "section_path": list(
                    chunk.section_path
                ),
                "chunk_confidence": (
                    chunk.chunk_confidence
                ),
                "semantic_density": (
                    chunk.semantic_density
                ),
            },
        )

    def _score_category(
        self,
        *,
        definition: _CategoryDefinition,
        title_text: str,
        path_text: str,
        content_text: str,
    ) -> CategoryScore:
        title_matches = self._match_terms(
            text=title_text,
            keywords=definition.keywords,
            phrases=definition.phrases,
        )

        path_matches = self._match_terms(
            text=path_text,
            keywords=definition.keywords,
            phrases=definition.phrases,
        )

        content_matches = self._match_terms(
            text=content_text,
            keywords=definition.keywords,
            phrases=definition.phrases,
        )

        section_prior_matches = (
            self._match_section_terms(
                text=(
                    f"{title_text} {path_text}"
                ),
                section_terms=(
                    definition.section_terms
                ),
            )
        )

        negative_matches = (
            self._match_negative_terms(
                text=(
                    f"{title_text} "
                    f"{path_text} "
                    f"{content_text}"
                ),
                terms=definition.negative_terms,
            )
        )

        title_score = (
            title_matches["keyword_score"]
            + self.phrase_multiplier
            * title_matches["phrase_score"]
        ) * self.title_weight

        path_score = (
            path_matches["keyword_score"]
            + self.phrase_multiplier
            * path_matches["phrase_score"]
        ) * self.path_weight

        content_score = (
            content_matches["keyword_score"]
            + self.phrase_multiplier
            * content_matches["phrase_score"]
        ) * self.content_weight

        section_prior_score = (
            len(section_prior_matches)
            * self.section_prior_weight
        )

        negative_score = (
            len(negative_matches)
            * self.negative_penalty
        )

        raw_score = max(
            title_score
            + path_score
            + content_score
            + section_prior_score
            - negative_score,
            0.0,
        )

        matched_terms = tuple(
            dict.fromkeys(
                [
                    *title_matches["keywords"],
                    *path_matches["keywords"],
                    *content_matches["keywords"],
                ]
            )
        )

        matched_phrases = tuple(
            dict.fromkeys(
                [
                    *title_matches["phrases"],
                    *path_matches["phrases"],
                    *content_matches["phrases"],
                    *section_prior_matches,
                ]
            )
        )

        match_count = (
            len(matched_terms)
            + len(matched_phrases)
        )

        confidence = (
            1.0
            - math.exp(
                -raw_score / 7.0
            )
            if raw_score > 0
            else 0.0
        )

        if match_count == 1:
            confidence *= 0.78

        return CategoryScore(
            category=definition.category,
            raw_score=round(
                raw_score,
                4,
            ),
            normalized_score=0.0,
            confidence=round(
                min(
                    max(
                        confidence,
                        0.0,
                    ),
                    0.99,
                ),
                4,
            ),
            matched_terms=matched_terms,
            matched_phrases=(
                matched_phrases
            ),
            title_match_count=(
                len(title_matches["keywords"])
                + len(
                    title_matches["phrases"]
                )
            ),
            path_match_count=(
                len(path_matches["keywords"])
                + len(
                    path_matches["phrases"]
                )
            ),
            content_match_count=(
                len(
                    content_matches["keywords"]
                )
                + len(
                    content_matches["phrases"]
                )
            ),
            negative_match_count=len(
                negative_matches
            ),
            metadata={
                "title_score": round(
                    title_score,
                    4,
                ),
                "path_score": round(
                    path_score,
                    4,
                ),
                "content_score": round(
                    content_score,
                    4,
                ),
                "section_prior_score": round(
                    section_prior_score,
                    4,
                ),
                "negative_score": round(
                    negative_score,
                    4,
                ),
            },
        )

    def _normalize_category_scores(
        self,
        scores: Sequence[CategoryScore],
    ) -> list[CategoryScore]:
        if not scores:
            return []

        positive_values = [
            score.raw_score
            for score in scores
        ]

        maximum = max(
            positive_values,
            default=0.0,
        )

        if maximum <= 0:
            return list(scores)

        exponentials = [
            math.exp(
                min(
                    score.raw_score
                    / max(maximum, 1.0)
                    * 3.0,
                    20.0,
                )
            )
            for score in scores
        ]

        total = sum(
            exponentials
        )

        result: list[
            CategoryScore
        ] = []

        for score, exponential in zip(
            scores,
            exponentials,
            strict=True,
        ):
            normalized_score = (
                exponential / total
                if total > 0
                else 0.0
            )

            result.append(
                CategoryScore(
                    category=score.category,
                    raw_score=score.raw_score,
                    normalized_score=round(
                        normalized_score,
                        4,
                    ),
                    confidence=score.confidence,
                    matched_terms=(
                        score.matched_terms
                    ),
                    matched_phrases=(
                        score.matched_phrases
                    ),
                    title_match_count=(
                        score.title_match_count
                    ),
                    path_match_count=(
                        score.path_match_count
                    ),
                    content_match_count=(
                        score.content_match_count
                    ),
                    negative_match_count=(
                        score.negative_match_count
                    ),
                    metadata=score.metadata,
                )
            )

        return result

    def _select_secondary_categories(
        self,
        *,
        ordered_scores: Sequence[CategoryScore],
        primary_category: ESGCategory,
    ) -> tuple[ESGCategory, ...]:
        if primary_category in {
            ESGCategory.UNKNOWN,
            ESGCategory.GENERAL,
        }:
            return ()

        primary_score = next(
            (
                score
                for score in ordered_scores
                if score.category
                == primary_category
            ),
            None,
        )

        if primary_score is None:
            return ()

        selected: list[
            ESGCategory
        ] = []

        for score in ordered_scores:
            if score.category == primary_category:
                continue

            if (
                score.raw_score
                < self.minimum_secondary_score
            ):
                continue

            ratio = (
                score.raw_score
                / max(
                    primary_score.raw_score,
                    1e-12,
                )
            )

            if ratio < self.minimum_secondary_ratio:
                continue

            selected.append(
                score.category
            )

            if (
                len(selected)
                >= self.maximum_secondary_categories
            ):
                break

        return tuple(selected)

    def _score_topics(
        self,
        *,
        title_text: str,
        path_text: str,
        content_text: str,
        allowed_categories: set[ESGCategory],
    ) -> list[TopicScore]:
        topic_scores: list[
            TopicScore
        ] = []

        for definition in self._TOPIC_DEFINITIONS:
            if (
                definition.category
                not in allowed_categories
                and allowed_categories
            ):
                continue

            title_matches = self._match_terms(
                text=title_text,
                keywords=definition.keywords,
                phrases=definition.phrases,
            )

            path_matches = self._match_terms(
                text=path_text,
                keywords=definition.keywords,
                phrases=definition.phrases,
            )

            content_matches = self._match_terms(
                text=content_text,
                keywords=definition.keywords,
                phrases=definition.phrases,
            )

            score = (
                (
                    title_matches["keyword_score"]
                    + self.phrase_multiplier
                    * title_matches["phrase_score"]
                )
                * self.title_weight
                + (
                    path_matches["keyword_score"]
                    + self.phrase_multiplier
                    * path_matches["phrase_score"]
                )
                * self.path_weight
                + (
                    content_matches["keyword_score"]
                    + self.phrase_multiplier
                    * content_matches["phrase_score"]
                )
                * self.content_weight
            )

            if score < 1.0:
                continue

            matched_terms = tuple(
                dict.fromkeys(
                    [
                        *title_matches["keywords"],
                        *title_matches["phrases"],
                        *path_matches["keywords"],
                        *path_matches["phrases"],
                        *content_matches["keywords"],
                        *content_matches["phrases"],
                    ]
                )
            )

            confidence = (
                1.0
                - math.exp(
                    -score / 5.0
                )
            )

            topic_scores.append(
                TopicScore(
                    topic=definition.name,
                    category=definition.category,
                    score=round(
                        score,
                        4,
                    ),
                    confidence=round(
                        min(
                            confidence,
                            0.99,
                        ),
                        4,
                    ),
                    matched_terms=(
                        matched_terms
                    ),
                )
            )

        topic_scores.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return topic_scores[
            :self.maximum_topics
        ]

    def _match_terms(
        self,
        *,
        text: str,
        keywords: Sequence[str],
        phrases: Sequence[str],
    ) -> dict[str, Any]:
        if not text:
            return {
                "keywords": [],
                "phrases": [],
                "keyword_score": 0.0,
                "phrase_score": 0.0,
            }

        tokens = self._TOKEN_RE.findall(
            text
        )

        token_counts = Counter(
            tokens
        )

        matched_keywords: list[str] = []
        keyword_score = 0.0

        for keyword in keywords:
            normalized_keyword = (
                self.normalize_for_search(
                    keyword
                )
            )

            if " " in normalized_keyword:
                count = self._phrase_count(
                    text,
                    normalized_keyword,
                )
            else:
                count = token_counts.get(
                    normalized_keyword,
                    0,
                )

            if count <= 0:
                continue

            matched_keywords.append(
                keyword
            )

            keyword_score += min(
                count,
                self.repeated_term_limit,
            )

        matched_phrases: list[str] = []
        phrase_score = 0.0

        for phrase in phrases:
            normalized_phrase = (
                self.normalize_for_search(
                    phrase
                )
            )

            count = self._phrase_count(
                text,
                normalized_phrase,
            )

            if count <= 0:
                continue

            matched_phrases.append(
                phrase
            )

            phrase_score += min(
                count,
                self.repeated_term_limit,
            )

        return {
            "keywords": matched_keywords,
            "phrases": matched_phrases,
            "keyword_score": keyword_score,
            "phrase_score": phrase_score,
        }

    def _match_section_terms(
        self,
        *,
        text: str,
        section_terms: Sequence[str],
    ) -> list[str]:
        matched: list[str] = []

        for term in section_terms:
            normalized_term = (
                self.normalize_for_search(
                    term
                )
            )

            if self._phrase_count(
                text,
                normalized_term,
            ):
                matched.append(term)

        return matched

    def _match_negative_terms(
        self,
        *,
        text: str,
        terms: Sequence[str],
    ) -> list[str]:
        matched: list[str] = []

        for term in terms:
            normalized_term = (
                self.normalize_for_search(
                    term
                )
            )

            if self._phrase_count(
                text,
                normalized_term,
            ):
                matched.append(term)

        return matched

    @staticmethod
    def _phrase_count(
        text: str,
        phrase: str,
    ) -> int:
        if not text or not phrase:
            return 0

        pattern = re.compile(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            re.IGNORECASE,
        )

        return len(
            pattern.findall(text)
        )

    def _detect_standard_families(
        self,
        text: str,
    ) -> tuple[str, ...]:
        families: list[str] = []

        for pattern, family in (
            self._STANDARD_PATTERNS
        ):
            if pattern.search(text):
                families.append(family)

        return tuple(
            dict.fromkeys(
                families
            )
        )

    def _calculate_classification_confidence(
        self,
        *,
        primary_score: CategoryScore,
        margin: float,
        matched_category_count: int,
        primary_category: ESGCategory,
        chunk_confidence: float,
    ) -> float:
        if primary_category == ESGCategory.UNKNOWN:
            return 0.0

        if primary_category == ESGCategory.GENERAL:
            base = min(
                primary_score.confidence,
                0.55,
            )
        else:
            base = (
                primary_score.confidence
            )

        margin_score = min(
            margin
            / max(
                self.ambiguity_margin * 3,
                0.10,
            ),
            1.0,
        )

        specificity_score = (
            1.0
            if matched_category_count <= 2
            else max(
                0.40,
                1.0
                - 0.12
                * (
                    matched_category_count
                    - 2
                ),
            )
        )

        confidence = (
            0.55 * base
            + 0.20 * margin_score
            + 0.15 * specificity_score
            + 0.10 * min(
                max(
                    chunk_confidence,
                    0.0,
                ),
                1.0,
            )
        )

        return round(
            min(
                max(
                    confidence,
                    0.0,
                ),
                0.99,
            ),
            4,
        )

    @classmethod
    def normalize_for_search(
        cls,
        value: Any,
    ) -> str:
        """Türkçe karakterleri korumalı biçimde arama metnine çevirir."""

        if value is None:
            return ""

        text = unicodedata.normalize(
            "NFKC",
            str(value),
        )

        translation_table = str.maketrans(
            {
                "İ": "i",
                "I": "i",
                "ı": "i",
                "Ş": "s",
                "ş": "s",
                "Ğ": "g",
                "ğ": "g",
                "Ü": "u",
                "ü": "u",
                "Ö": "o",
                "ö": "o",
                "Ç": "c",
                "ç": "c",
            }
        )

        text = text.translate(
            translation_table
        ).casefold()

        text = re.sub(
            r"[^\w%€$₺£.+\- ]",
            " ",
            text,
        )

        return cls._WHITESPACE_RE.sub(
            " ",
            text,
        ).strip()

    @staticmethod
    def _calculate_result_hash(
        classifications: Sequence[
            ChunkCategoryClassification
        ],
    ) -> str:
        canonical = "\n".join(
            (
                f"{item.classification_id}:"
                f"{item.chunk_id}:"
                f"{item.primary_category.value}:"
                f"{','.join(item.detected_topics)}:"
                f"{item.content_hash}"
            )
            for item in classifications
        )

        return hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()


def classify_semantic_chunks(
    chunking_result: SemanticChunkingResult,
    *,
    minimum_primary_score: float = 1.75,
    minimum_secondary_score: float = 1.25,
) -> CategoryClassificationResult:
    """CategoryClassifier için kısa kullanım arayüzü."""

    classifier = CategoryClassifier(
        minimum_primary_score=(
            minimum_primary_score
        ),
        minimum_secondary_score=(
            minimum_secondary_score
        ),
    )

    return classifier.classify_document(
        chunking_result
    )