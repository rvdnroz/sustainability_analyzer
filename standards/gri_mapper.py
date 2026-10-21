from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


class GRICategory(StrEnum):
    UNIVERSAL = "universal"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    SECTOR = "sector"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class GRIDisclosure:
    code: str
    title: str
    category: GRICategory
    topic: str
    keywords: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GRIMatch:
    disclosure: GRIDisclosure
    score: float
    matched_terms: tuple[str, ...] = ()
    match_type: str = "semantic"

    @property
    def code(self) -> str:
        return self.disclosure.code

    @property
    def title(self) -> str:
        return self.disclosure.title


class GRIMapperError(RuntimeError):
    """GRI eşleştirme işlemlerindeki temel hata."""


class InvalidGRICodeError(GRIMapperError):
    """Geçersiz GRI disclosure kodu kullanıldığında yükseltilir."""


class GRIMapper:
    """
    GRI disclosure kodlarını sürdürülebilirlik konuları ve anahtar
    kavramlarla eşleştirir.

    Desteklenen işlemler:
    - Metin içerisindeki GRI kodlarını tespit etme
    - GRI kodundan disclosure bilgisine erişme
    - Soru veya metni ilgili GRI disclosure'larıyla eşleştirme
    - Kategori, konu ve anahtar sözcük bazlı arama
    - Projeye özel disclosure kayıtları ekleme
    """

    _CODE_PATTERN = re.compile(
        r"\bGRI\s*[-:]?\s*(?P<code>\d{1,3}(?:-\d{1,3})?)\b",
        re.IGNORECASE,
    )

    _BARE_CODE_PATTERN = re.compile(
        r"^(?P<code>\d{1,3}(?:-\d{1,3})?)$",
    )

    def __init__(
        self,
        disclosures: Iterable[GRIDisclosure] | None = None,
    ) -> None:
        self._disclosures: dict[str, GRIDisclosure] = {}

        for disclosure in self._default_disclosures():
            self.register(disclosure)

        if disclosures is not None:
            for disclosure in disclosures:
                self.register(disclosure, overwrite=True)

    def register(
        self,
        disclosure: GRIDisclosure,
        *,
        overwrite: bool = False,
    ) -> None:
        normalized_code = self.normalize_code(disclosure.code)

        if normalized_code in self._disclosures and not overwrite:
            raise GRIMapperError(
                f"{normalized_code} kodlu GRI disclosure zaten kayıtlı.",
            )

        normalized_disclosure = GRIDisclosure(
            code=normalized_code,
            title=disclosure.title.strip(),
            category=disclosure.category,
            topic=disclosure.topic.strip(),
            keywords=self._normalize_terms(disclosure.keywords),
            aliases=self._normalize_terms(disclosure.aliases),
            description=(
                disclosure.description.strip()
                if disclosure.description
                else None
            ),
            metadata=dict(disclosure.metadata),
        )

        self._disclosures[normalized_code] = normalized_disclosure

    def register_many(
        self,
        disclosures: Iterable[GRIDisclosure],
        *,
        overwrite: bool = False,
    ) -> None:
        for disclosure in disclosures:
            self.register(disclosure, overwrite=overwrite)

    def get(self, code: str) -> GRIDisclosure:
        normalized_code = self.normalize_code(code)

        try:
            return self._disclosures[normalized_code]
        except KeyError as exc:
            raise InvalidGRICodeError(
                f"{normalized_code} kodlu GRI disclosure bulunamadı.",
            ) from exc

    def get_optional(self, code: str) -> GRIDisclosure | None:
        try:
            normalized_code = self.normalize_code(code)
        except InvalidGRICodeError:
            return None

        return self._disclosures.get(normalized_code)

    def exists(self, code: str) -> bool:
        return self.get_optional(code) is not None

    def all(self) -> list[GRIDisclosure]:
        return sorted(
            self._disclosures.values(),
            key=self._disclosure_sort_key,
        )

    def find_codes(self, text: str) -> list[str]:
        if not text:
            return []

        found_codes = {
            self.normalize_code(match.group("code"))
            for match in self._CODE_PATTERN.finditer(text)
        }

        return sorted(found_codes, key=self._code_sort_key)

    def find_disclosures(self, text: str) -> list[GRIDisclosure]:
        return [
            disclosure
            for code in self.find_codes(text)
            if (disclosure := self.get_optional(code)) is not None
        ]

    def search(
        self,
        query: str,
        *,
        category: GRICategory | str | None = None,
        limit: int = 10,
        minimum_score: float = 0.1,
    ) -> list[GRIMatch]:
        if limit <= 0:
            return []

        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []

        category_filter = self._normalize_category(category)
        query_terms = set(normalized_query.split())
        explicit_codes = set(self.find_codes(query))

        matches: list[GRIMatch] = []

        for disclosure in self._disclosures.values():
            if (
                category_filter is not None
                and disclosure.category != category_filter
            ):
                continue

            score, matched_terms, match_type = self._score_disclosure(
                disclosure=disclosure,
                normalized_query=normalized_query,
                query_terms=query_terms,
                explicit_codes=explicit_codes,
            )

            if score < minimum_score:
                continue

            matches.append(
                GRIMatch(
                    disclosure=disclosure,
                    score=round(min(score, 1.0), 4),
                    matched_terms=tuple(sorted(matched_terms)),
                    match_type=match_type,
                ),
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                self._disclosure_sort_key(match.disclosure),
            ),
        )

        return matches[:limit]

    def map_text(
        self,
        text: str,
        *,
        limit: int = 10,
        minimum_score: float = 0.15,
    ) -> list[GRIMatch]:
        """
        Metni en ilgili GRI disclosure kayıtlarıyla eşleştirir.

        Açık GRI kodları en yüksek önceliğe sahiptir. Kod bulunmadığında
        başlık, konu, anahtar sözcük ve alias alanları kullanılır.
        """
        return self.search(
            text,
            limit=limit,
            minimum_score=minimum_score,
        )

    def map_keywords(
        self,
        keywords: Sequence[str],
        *,
        category: GRICategory | str | None = None,
        limit: int = 10,
    ) -> list[GRIMatch]:
        query = " ".join(
            keyword.strip()
            for keyword in keywords
            if isinstance(keyword, str) and keyword.strip()
        )

        return self.search(
            query,
            category=category,
            limit=limit,
        )

    def find_by_category(
        self,
        category: GRICategory | str,
    ) -> list[GRIDisclosure]:
        normalized_category = self._normalize_category(category)

        if normalized_category is None:
            return []

        return [
            disclosure
            for disclosure in self.all()
            if disclosure.category == normalized_category
        ]

    def find_by_topic(self, topic: str) -> list[GRIDisclosure]:
        normalized_topic = self._normalize_text(topic)

        if not normalized_topic:
            return []

        return [
            disclosure
            for disclosure in self.all()
            if normalized_topic
            in self._normalize_text(disclosure.topic)
        ]

    @classmethod
    def normalize_code(cls, code: str) -> str:
        if code is None:
            raise InvalidGRICodeError("GRI kodu boş olamaz.")

        normalized = str(code).strip().upper()
        normalized = re.sub(r"^GRI\s*[-:]?\s*", "", normalized)
        normalized = re.sub(r"\s+", "", normalized)

        match = cls._BARE_CODE_PATTERN.fullmatch(normalized)
        if match is None:
            raise InvalidGRICodeError(
                f"Geçersiz GRI disclosure kodu: {code!r}",
            )

        return f"GRI {match.group('code')}"

    def _score_disclosure(
        self,
        *,
        disclosure: GRIDisclosure,
        normalized_query: str,
        query_terms: set[str],
        explicit_codes: set[str],
    ) -> tuple[float, set[str], str]:
        matched_terms: set[str] = set()

        if disclosure.code in explicit_codes:
            return 1.0, {disclosure.code}, "code"

        title = self._normalize_text(disclosure.title)
        topic = self._normalize_text(disclosure.topic)
        description = self._normalize_text(disclosure.description or "")

        keywords = {
            self._normalize_text(keyword)
            for keyword in disclosure.keywords
        }
        aliases = {
            self._normalize_text(alias)
            for alias in disclosure.aliases
        }

        score = 0.0

        if title and title in normalized_query:
            score += 0.55
            matched_terms.add(disclosure.title)

        if topic and topic in normalized_query:
            score += 0.4
            matched_terms.add(disclosure.topic)

        for alias in aliases:
            if not alias:
                continue

            if alias in normalized_query:
                score += 0.32
                matched_terms.add(alias)

        for keyword in keywords:
            if not keyword:
                continue

            if keyword in normalized_query:
                score += 0.18
                matched_terms.add(keyword)
                continue

            keyword_terms = set(keyword.split())
            overlap = query_terms.intersection(keyword_terms)

            if overlap:
                overlap_ratio = len(overlap) / len(keyword_terms)
                score += 0.12 * overlap_ratio
                matched_terms.update(overlap)

        title_terms = set(title.split())
        topic_terms = set(topic.split())
        description_terms = set(description.split())

        title_overlap = self._overlap_ratio(query_terms, title_terms)
        topic_overlap = self._overlap_ratio(query_terms, topic_terms)
        description_overlap = self._overlap_ratio(
            query_terms,
            description_terms,
        )

        score += 0.25 * title_overlap
        score += 0.2 * topic_overlap
        score += 0.08 * description_overlap

        if not matched_terms:
            matched_terms.update(
                query_terms.intersection(
                    title_terms
                    | topic_terms
                    | description_terms,
                ),
            )

        return score, matched_terms, "term"

    @staticmethod
    def _overlap_ratio(
        query_terms: set[str],
        target_terms: set[str],
    ) -> float:
        if not query_terms or not target_terms:
            return 0.0

        overlap = query_terms.intersection(target_terms)
        return len(overlap) / min(len(query_terms), len(target_terms))

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

        normalized = str(value).translate(translation_table).casefold()
        normalized = re.sub(r"[^\w\s-]", " ", normalized)
        return " ".join(normalized.split())

    @classmethod
    def _normalize_terms(
        cls,
        terms: Iterable[str],
    ) -> tuple[str, ...]:
        normalized_terms = {
            " ".join(str(term).strip().split())
            for term in terms
            if term is not None and str(term).strip()
        }

        return tuple(sorted(normalized_terms))

    @staticmethod
    def _normalize_category(
        category: GRICategory | str | None,
    ) -> GRICategory | None:
        if category is None:
            return None

        if isinstance(category, GRICategory):
            return category

        normalized = str(category).strip().casefold()

        for member in GRICategory:
            if normalized in {
                member.value.casefold(),
                member.name.casefold(),
            }:
                return member

        raise ValueError(f"Geçersiz GRI kategorisi: {category!r}")

    @staticmethod
    def _code_sort_key(code: str) -> tuple[int, ...]:
        numeric_part = code.removeprefix("GRI ").strip()

        return tuple(
            int(part)
            for part in numeric_part.split("-")
        )

    @classmethod
    def _disclosure_sort_key(
        cls,
        disclosure: GRIDisclosure,
    ) -> tuple[int, ...]:
        return cls._code_sort_key(disclosure.code)

    @staticmethod
    def _default_disclosures() -> tuple[GRIDisclosure, ...]:
        return (
            GRIDisclosure(
                code="GRI 2-1",
                title="Organizational details",
                category=GRICategory.UNIVERSAL,
                topic="Organization profile",
                keywords=(
                    "organization",
                    "company profile",
                    "headquarters",
                    "ownership",
                    "legal form",
                    "kuruluş bilgileri",
                    "şirket profili",
                    "merkez",
                    "mülkiyet",
                ),
            ),
            GRIDisclosure(
                code="GRI 2-6",
                title="Activities, value chain and other business relationships",
                category=GRICategory.UNIVERSAL,
                topic="Activities and value chain",
                keywords=(
                    "value chain",
                    "supply chain",
                    "business activities",
                    "products",
                    "services",
                    "değer zinciri",
                    "tedarik zinciri",
                    "faaliyetler",
                    "ürünler",
                    "hizmetler",
                ),
            ),
            GRIDisclosure(
                code="GRI 2-7",
                title="Employees",
                category=GRICategory.UNIVERSAL,
                topic="Workforce",
                keywords=(
                    "employees",
                    "employee count",
                    "workforce",
                    "full time",
                    "part time",
                    "çalışanlar",
                    "çalışan sayısı",
                    "iş gücü",
                    "tam zamanlı",
                    "yarı zamanlı",
                ),
            ),
            GRIDisclosure(
                code="GRI 2-23",
                title="Policy commitments",
                category=GRICategory.UNIVERSAL,
                topic="Policies and commitments",
                keywords=(
                    "policy commitments",
                    "human rights policy",
                    "environment policy",
                    "code of conduct",
                    "politika taahhütleri",
                    "insan hakları politikası",
                    "çevre politikası",
                    "etik kurallar",
                ),
            ),
            GRIDisclosure(
                code="GRI 2-29",
                title="Approach to stakeholder engagement",
                category=GRICategory.UNIVERSAL,
                topic="Stakeholder engagement",
                keywords=(
                    "stakeholder engagement",
                    "stakeholder dialogue",
                    "materiality consultation",
                    "paydaş katılımı",
                    "paydaş diyaloğu",
                    "önemlilik analizi",
                ),
            ),
            GRIDisclosure(
                code="GRI 3-1",
                title="Process to determine material topics",
                category=GRICategory.UNIVERSAL,
                topic="Materiality assessment",
                keywords=(
                    "material topics",
                    "materiality process",
                    "double materiality",
                    "önemli konular",
                    "önemlilik süreci",
                    "çifte önemlilik",
                ),
            ),
            GRIDisclosure(
                code="GRI 201-1",
                title="Direct economic value generated and distributed",
                category=GRICategory.ECONOMIC,
                topic="Economic performance",
                keywords=(
                    "economic value",
                    "revenue",
                    "operating costs",
                    "employee wages",
                    "payments to government",
                    "ekonomik değer",
                    "gelir",
                    "operasyonel maliyet",
                    "çalışan ücretleri",
                    "devlete yapılan ödemeler",
                ),
            ),
            GRIDisclosure(
                code="GRI 202-1",
                title=(
                    "Ratios of standard entry level wage by gender "
                    "compared to local minimum wage"
                ),
                category=GRICategory.ECONOMIC,
                topic="Market presence",
                keywords=(
                    "entry level wage",
                    "minimum wage",
                    "gender pay",
                    "başlangıç ücreti",
                    "asgari ücret",
                    "cinsiyete göre ücret",
                ),
            ),
            GRIDisclosure(
                code="GRI 203-1",
                title="Infrastructure investments and services supported",
                category=GRICategory.ECONOMIC,
                topic="Indirect economic impacts",
                keywords=(
                    "infrastructure investment",
                    "community investment",
                    "public benefit",
                    "altyapı yatırımı",
                    "toplumsal yatırım",
                    "kamu yararı",
                ),
            ),
            GRIDisclosure(
                code="GRI 205-2",
                title=(
                    "Communication and training about anti-corruption "
                    "policies and procedures"
                ),
                category=GRICategory.ECONOMIC,
                topic="Anti-corruption",
                keywords=(
                    "anti-corruption training",
                    "corruption policy",
                    "bribery",
                    "yolsuzlukla mücadele eğitimi",
                    "yolsuzluk politikası",
                    "rüşvet",
                ),
            ),
            GRIDisclosure(
                code="GRI 205-3",
                title="Confirmed incidents of corruption and actions taken",
                category=GRICategory.ECONOMIC,
                topic="Anti-corruption",
                keywords=(
                    "corruption incidents",
                    "confirmed corruption",
                    "disciplinary action",
                    "yolsuzluk vakaları",
                    "doğrulanmış yolsuzluk",
                    "disiplin işlemi",
                ),
            ),
            GRIDisclosure(
                code="GRI 302-1",
                title="Energy consumption within the organization",
                category=GRICategory.ENVIRONMENTAL,
                topic="Energy",
                keywords=(
                    "energy consumption",
                    "electricity consumption",
                    "fuel consumption",
                    "renewable energy",
                    "enerji tüketimi",
                    "elektrik tüketimi",
                    "yakıt tüketimi",
                    "yenilenebilir enerji",
                ),
                aliases=(
                    "total energy consumed",
                    "toplam enerji tüketimi",
                ),
            ),
            GRIDisclosure(
                code="GRI 302-3",
                title="Energy intensity",
                category=GRICategory.ENVIRONMENTAL,
                topic="Energy",
                keywords=(
                    "energy intensity",
                    "energy per production",
                    "energy efficiency",
                    "enerji yoğunluğu",
                    "üretim başına enerji",
                    "enerji verimliliği",
                ),
            ),
            GRIDisclosure(
                code="GRI 303-3",
                title="Water withdrawal",
                category=GRICategory.ENVIRONMENTAL,
                topic="Water and effluents",
                keywords=(
                    "water withdrawal",
                    "water consumption",
                    "freshwater",
                    "groundwater",
                    "surface water",
                    "su çekimi",
                    "su tüketimi",
                    "tatlı su",
                    "yeraltı suyu",
                    "yüzey suyu",
                ),
            ),
            GRIDisclosure(
                code="GRI 303-4",
                title="Water discharge",
                category=GRICategory.ENVIRONMENTAL,
                topic="Water and effluents",
                keywords=(
                    "water discharge",
                    "wastewater",
                    "effluent",
                    "discharged water",
                    "su deşarjı",
                    "atık su",
                    "deşarj edilen su",
                ),
            ),
            GRIDisclosure(
                code="GRI 305-1",
                title="Direct greenhouse gas emissions",
                category=GRICategory.ENVIRONMENTAL,
                topic="Emissions",
                keywords=(
                    "scope 1",
                    "direct greenhouse gas emissions",
                    "direct emissions",
                    "ghg emissions",
                    "doğrudan sera gazı emisyonları",
                    "doğrudan emisyonlar",
                    "sera gazı emisyonları",
                ),
                aliases=(
                    "scope one",
                    "kapsam 1",
                    "kapsam bir",
                ),
            ),
            GRIDisclosure(
                code="GRI 305-2",
                title="Energy indirect greenhouse gas emissions",
                category=GRICategory.ENVIRONMENTAL,
                topic="Emissions",
                keywords=(
                    "scope 2",
                    "indirect greenhouse gas emissions",
                    "purchased electricity emissions",
                    "dolaylı sera gazı emisyonları",
                    "satın alınan enerji emisyonları",
                ),
                aliases=(
                    "scope two",
                    "kapsam 2",
                    "kapsam iki",
                ),
            ),
            GRIDisclosure(
                code="GRI 305-3",
                title="Other indirect greenhouse gas emissions",
                category=GRICategory.ENVIRONMENTAL,
                topic="Emissions",
                keywords=(
                    "scope 3",
                    "value chain emissions",
                    "other indirect emissions",
                    "değer zinciri emisyonları",
                    "diğer dolaylı emisyonlar",
                ),
                aliases=(
                    "scope three",
                    "kapsam 3",
                    "kapsam üç",
                ),
            ),
            GRIDisclosure(
                code="GRI 305-4",
                title="Greenhouse gas emissions intensity",
                category=GRICategory.ENVIRONMENTAL,
                topic="Emissions",
                keywords=(
                    "emission intensity",
                    "carbon intensity",
                    "emissions per production",
                    "emisyon yoğunluğu",
                    "karbon yoğunluğu",
                    "üretim başına emisyon",
                ),
            ),
            GRIDisclosure(
                code="GRI 306-3",
                title="Waste generated",
                category=GRICategory.ENVIRONMENTAL,
                topic="Waste",
                keywords=(
                    "waste generated",
                    "total waste",
                    "hazardous waste",
                    "non-hazardous waste",
                    "oluşan atık",
                    "toplam atık",
                    "tehlikeli atık",
                    "tehlikesiz atık",
                ),
            ),
            GRIDisclosure(
                code="GRI 306-4",
                title="Waste diverted from disposal",
                category=GRICategory.ENVIRONMENTAL,
                topic="Waste",
                keywords=(
                    "recycled waste",
                    "reused waste",
                    "recovered waste",
                    "waste diverted",
                    "geri dönüştürülen atık",
                    "yeniden kullanılan atık",
                    "geri kazanılan atık",
                ),
            ),
            GRIDisclosure(
                code="GRI 306-5",
                title="Waste directed to disposal",
                category=GRICategory.ENVIRONMENTAL,
                topic="Waste",
                keywords=(
                    "landfill",
                    "incineration",
                    "disposed waste",
                    "waste disposal",
                    "düzenli depolama",
                    "yakma",
                    "bertaraf edilen atık",
                    "atık bertarafı",
                ),
            ),
            GRIDisclosure(
                code="GRI 401-1",
                title="New employee hires and employee turnover",
                category=GRICategory.SOCIAL,
                topic="Employment",
                keywords=(
                    "new hires",
                    "employee turnover",
                    "staff turnover",
                    "employee departures",
                    "yeni işe alımlar",
                    "çalışan devir oranı",
                    "personel devri",
                    "işten ayrılan çalışanlar",
                ),
            ),
            GRIDisclosure(
                code="GRI 403-9",
                title="Work-related injuries",
                category=GRICategory.SOCIAL,
                topic="Occupational health and safety",
                keywords=(
                    "work related injuries",
                    "lost time injury",
                    "fatality",
                    "accident frequency rate",
                    "iş kazaları",
                    "kayıp günlü yaralanma",
                    "ölümlü kaza",
                    "kaza sıklık oranı",
                ),
                aliases=(
                    "ltifr",
                    "lti",
                ),
            ),
            GRIDisclosure(
                code="GRI 404-1",
                title="Average hours of training per year per employee",
                category=GRICategory.SOCIAL,
                topic="Training and education",
                keywords=(
                    "training hours",
                    "employee training",
                    "average training",
                    "çalışan başına eğitim saati",
                    "çalışan eğitimi",
                    "ortalama eğitim süresi",
                ),
            ),
            GRIDisclosure(
                code="GRI 405-1",
                title="Diversity of governance bodies and employees",
                category=GRICategory.SOCIAL,
                topic="Diversity and equal opportunity",
                keywords=(
                    "gender diversity",
                    "women employees",
                    "women managers",
                    "age diversity",
                    "cinsiyet çeşitliliği",
                    "kadın çalışanlar",
                    "kadın yöneticiler",
                    "yaş çeşitliliği",
                ),
            ),
            GRIDisclosure(
                code="GRI 405-2",
                title="Ratio of basic salary and remuneration of women to men",
                category=GRICategory.SOCIAL,
                topic="Diversity and equal opportunity",
                keywords=(
                    "gender pay ratio",
                    "equal pay",
                    "pay gap",
                    "salary ratio",
                    "cinsiyet ücret oranı",
                    "eşit ücret",
                    "ücret farkı",
                    "maaş oranı",
                ),
            ),
            GRIDisclosure(
                code="GRI 406-1",
                title="Incidents of discrimination and corrective actions taken",
                category=GRICategory.SOCIAL,
                topic="Non-discrimination",
                keywords=(
                    "discrimination incidents",
                    "harassment",
                    "corrective action",
                    "ayrımcılık vakaları",
                    "taciz",
                    "düzeltici faaliyet",
                ),
            ),
            GRIDisclosure(
                code="GRI 413-1",
                title=(
                    "Operations with local community engagement, impact "
                    "assessments and development programs"
                ),
                category=GRICategory.SOCIAL,
                topic="Local communities",
                keywords=(
                    "community engagement",
                    "social investment",
                    "community programs",
                    "impact assessment",
                    "toplum katılımı",
                    "sosyal yatırım",
                    "toplumsal programlar",
                    "etki değerlendirmesi",
                ),
            ),
            GRIDisclosure(
                code="GRI 418-1",
                title=(
                    "Substantiated complaints concerning breaches of "
                    "customer privacy and losses of customer data"
                ),
                category=GRICategory.SOCIAL,
                topic="Customer privacy",
                keywords=(
                    "customer privacy",
                    "data breach",
                    "customer data loss",
                    "privacy complaints",
                    "müşteri gizliliği",
                    "veri ihlali",
                    "müşteri verisi kaybı",
                    "gizlilik şikayetleri",
                ),
            ),
        )


__all__ = [
    "GRICategory",
    "GRIDisclosure",
    "GRIMatch",
    "GRIMapper",
    "GRIMapperError",
    "InvalidGRICodeError",
]