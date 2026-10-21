from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


class ESRSPillar(StrEnum):
    CROSS_CUTTING = "cross_cutting"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class ESRSDisclosure:
    code: str
    title: str
    pillar: ESRSPillar
    topic: str
    keywords: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ESRSMatch:
    disclosure: ESRSDisclosure
    score: float
    matched_terms: tuple[str, ...] = ()
    match_type: str = "term"

    @property
    def code(self) -> str:
        return self.disclosure.code

    @property
    def title(self) -> str:
        return self.disclosure.title


class ESRSMapperError(RuntimeError):
    """ESRS eşleştirme işlemleri sırasında oluşan temel hata."""


class InvalidESRSCodeError(ESRSMapperError):
    """Geçersiz ESRS kodu kullanıldığında yükseltilir."""


class ESRSMapper:
    """
    ESRS standartlarını sürdürülebilirlik konuları ve göstergelerle eşleştirir.

    Desteklenen işlemler:
    - Metindeki ESRS kodlarını tespit etme
    - ESRS kodundan standart bilgisine erişme
    - Soru veya metni ilgili ESRS standartlarıyla eşleştirme
    - Çevresel, sosyal, yönetişim ve genel standartlara göre filtreleme
    - Projeye özel ESRS kayıtları ekleme
    """

    _CODE_PATTERN = re.compile(
        r"\bESRS\s*[-:]?\s*"
        r"(?P<code>1|2|E[1-5]|S[1-4]|G1)"
        r"(?:\s*[-:]?\s*(?P<requirement>DR\s*[A-Z0-9-]+))?\b",
        re.IGNORECASE,
    )

    _BARE_CODE_PATTERN = re.compile(
        r"^(?P<code>1|2|E[1-5]|S[1-4]|G1)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        disclosures: Iterable[ESRSDisclosure] | None = None,
    ) -> None:
        self._disclosures: dict[str, ESRSDisclosure] = {}

        for disclosure in self._default_disclosures():
            self.register(disclosure)

        if disclosures is not None:
            self.register_many(disclosures, overwrite=True)

    def register(
        self,
        disclosure: ESRSDisclosure,
        *,
        overwrite: bool = False,
    ) -> None:
        normalized_code = self.normalize_code(disclosure.code)

        if normalized_code in self._disclosures and not overwrite:
            raise ESRSMapperError(
                f"{normalized_code} kodlu ESRS standardı zaten kayıtlı.",
            )

        normalized_disclosure = ESRSDisclosure(
            code=normalized_code,
            title=disclosure.title.strip(),
            pillar=disclosure.pillar,
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
        disclosures: Iterable[ESRSDisclosure],
        *,
        overwrite: bool = False,
    ) -> None:
        for disclosure in disclosures:
            self.register(disclosure, overwrite=overwrite)

    def get(self, code: str) -> ESRSDisclosure:
        normalized_code = self.normalize_code(code)

        try:
            return self._disclosures[normalized_code]
        except KeyError as exc:
            raise InvalidESRSCodeError(
                f"{normalized_code} kodlu ESRS standardı bulunamadı.",
            ) from exc

    def get_optional(self, code: str) -> ESRSDisclosure | None:
        try:
            normalized_code = self.normalize_code(code)
        except InvalidESRSCodeError:
            return None

        return self._disclosures.get(normalized_code)

    def exists(self, code: str) -> bool:
        return self.get_optional(code) is not None

    def all(self) -> list[ESRSDisclosure]:
        return sorted(
            self._disclosures.values(),
            key=self._disclosure_sort_key,
        )

    def find_codes(self, text: str) -> list[str]:
        if not text:
            return []

        codes = {
            self.normalize_code(match.group("code"))
            for match in self._CODE_PATTERN.finditer(text)
        }

        return sorted(codes, key=self._code_sort_key)

    def find_disclosures(self, text: str) -> list[ESRSDisclosure]:
        return [
            disclosure
            for code in self.find_codes(text)
            if (disclosure := self.get_optional(code)) is not None
        ]

    def search(
        self,
        query: str,
        *,
        pillar: ESRSPillar | str | None = None,
        limit: int = 10,
        minimum_score: float = 0.1,
    ) -> list[ESRSMatch]:
        if limit <= 0:
            return []

        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []

        pillar_filter = self._normalize_pillar(pillar)
        query_terms = set(normalized_query.split())
        explicit_codes = set(self.find_codes(query))

        matches: list[ESRSMatch] = []

        for disclosure in self._disclosures.values():
            if (
                pillar_filter is not None
                and disclosure.pillar != pillar_filter
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
                ESRSMatch(
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
    ) -> list[ESRSMatch]:
        return self.search(
            text,
            limit=limit,
            minimum_score=minimum_score,
        )

    def map_keywords(
        self,
        keywords: Sequence[str],
        *,
        pillar: ESRSPillar | str | None = None,
        limit: int = 10,
    ) -> list[ESRSMatch]:
        query = " ".join(
            keyword.strip()
            for keyword in keywords
            if isinstance(keyword, str) and keyword.strip()
        )

        return self.search(
            query,
            pillar=pillar,
            limit=limit,
        )

    def find_by_pillar(
        self,
        pillar: ESRSPillar | str,
    ) -> list[ESRSDisclosure]:
        normalized_pillar = self._normalize_pillar(pillar)

        if normalized_pillar is None:
            return []

        return [
            disclosure
            for disclosure in self.all()
            if disclosure.pillar == normalized_pillar
        ]

    def find_by_topic(self, topic: str) -> list[ESRSDisclosure]:
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
            raise InvalidESRSCodeError("ESRS kodu boş olamaz.")

        normalized = str(code).strip().upper()
        normalized = re.sub(r"^ESRS\s*[-:]?\s*", "", normalized)
        normalized = re.sub(r"\s+", "", normalized)

        match = cls._BARE_CODE_PATTERN.fullmatch(normalized)
        if match is None:
            raise InvalidESRSCodeError(
                f"Geçersiz ESRS kodu: {code!r}",
            )

        return f"ESRS {match.group('code').upper()}"

    def _score_disclosure(
        self,
        *,
        disclosure: ESRSDisclosure,
        normalized_query: str,
        query_terms: set[str],
        explicit_codes: set[str],
    ) -> tuple[float, set[str], str]:
        if disclosure.code in explicit_codes:
            return 1.0, {disclosure.code}, "code"

        matched_terms: set[str] = set()
        score = 0.0

        title = self._normalize_text(disclosure.title)
        topic = self._normalize_text(disclosure.topic)
        description = self._normalize_text(disclosure.description or "")

        aliases = {
            self._normalize_text(alias)
            for alias in disclosure.aliases
            if alias
        }
        keywords = {
            self._normalize_text(keyword)
            for keyword in disclosure.keywords
            if keyword
        }

        if title and title in normalized_query:
            score += 0.55
            matched_terms.add(disclosure.title)

        if topic and topic in normalized_query:
            score += 0.4
            matched_terms.add(disclosure.topic)

        for alias in aliases:
            if alias in normalized_query:
                score += 0.32
                matched_terms.add(alias)

        for keyword in keywords:
            if keyword in normalized_query:
                score += 0.18
                matched_terms.add(keyword)
                continue

            keyword_terms = set(keyword.split())
            overlap = query_terms.intersection(keyword_terms)

            if overlap:
                score += 0.12 * (
                    len(overlap) / max(len(keyword_terms), 1)
                )
                matched_terms.update(overlap)

        title_terms = set(title.split())
        topic_terms = set(topic.split())
        description_terms = set(description.split())

        score += 0.25 * self._overlap_ratio(
            query_terms,
            title_terms,
        )
        score += 0.2 * self._overlap_ratio(
            query_terms,
            topic_terms,
        )
        score += 0.08 * self._overlap_ratio(
            query_terms,
            description_terms,
        )

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

        return len(overlap) / min(
            len(query_terms),
            len(target_terms),
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

        normalized = str(value).translate(translation_table).casefold()
        normalized = re.sub(r"[^\w\s-]", " ", normalized)

        return " ".join(normalized.split())

    @classmethod
    def _normalize_terms(
        cls,
        terms: Iterable[str],
    ) -> tuple[str, ...]:
        normalized = {
            " ".join(str(term).strip().split())
            for term in terms
            if term is not None and str(term).strip()
        }

        return tuple(sorted(normalized))

    @staticmethod
    def _normalize_pillar(
        pillar: ESRSPillar | str | None,
    ) -> ESRSPillar | None:
        if pillar is None:
            return None

        if isinstance(pillar, ESRSPillar):
            return pillar

        normalized = str(pillar).strip().casefold()

        for member in ESRSPillar:
            if normalized in {
                member.value.casefold(),
                member.name.casefold(),
            }:
                return member

        raise ValueError(f"Geçersiz ESRS sütunu: {pillar!r}")

    @staticmethod
    def _code_sort_key(code: str) -> tuple[int, int]:
        normalized = code.removeprefix("ESRS ").strip()

        if normalized == "1":
            return 0, 1

        if normalized == "2":
            return 0, 2

        prefix_order = {
            "E": 1,
            "S": 2,
            "G": 3,
        }

        prefix = normalized[0]
        number = int(normalized[1:])

        return prefix_order.get(prefix, 99), number

    @classmethod
    def _disclosure_sort_key(
        cls,
        disclosure: ESRSDisclosure,
    ) -> tuple[int, int]:
        return cls._code_sort_key(disclosure.code)

    @staticmethod
    def _default_disclosures() -> tuple[ESRSDisclosure, ...]:
        return (
            ESRSDisclosure(
                code="ESRS 1",
                title="General Requirements",
                pillar=ESRSPillar.CROSS_CUTTING,
                topic="General sustainability reporting requirements",
                keywords=(
                    "general requirements",
                    "reporting principles",
                    "double materiality",
                    "value chain",
                    "due diligence",
                    "genel gereklilikler",
                    "raporlama ilkeleri",
                    "çifte önemlilik",
                    "değer zinciri",
                    "özen yükümlülüğü",
                ),
                aliases=(
                    "general reporting requirements",
                    "genel raporlama gereklilikleri",
                ),
            ),
            ESRSDisclosure(
                code="ESRS 2",
                title="General Disclosures",
                pillar=ESRSPillar.CROSS_CUTTING,
                topic="General corporate sustainability disclosures",
                keywords=(
                    "governance",
                    "strategy",
                    "business model",
                    "material impacts",
                    "risks",
                    "opportunities",
                    "metrics",
                    "targets",
                    "yönetişim",
                    "strateji",
                    "iş modeli",
                    "önemli etkiler",
                    "riskler",
                    "fırsatlar",
                    "metrikler",
                    "hedefler",
                ),
                aliases=(
                    "general disclosures",
                    "genel açıklamalar",
                ),
            ),
            ESRSDisclosure(
                code="ESRS E1",
                title="Climate Change",
                pillar=ESRSPillar.ENVIRONMENTAL,
                topic="Climate change mitigation and adaptation",
                keywords=(
                    "climate change",
                    "climate transition plan",
                    "greenhouse gas emissions",
                    "scope 1",
                    "scope 2",
                    "scope 3",
                    "carbon emissions",
                    "energy consumption",
                    "renewable energy",
                    "climate risks",
                    "iklim değişikliği",
                    "iklim geçiş planı",
                    "sera gazı emisyonları",
                    "kapsam 1",
                    "kapsam 2",
                    "kapsam 3",
                    "karbon emisyonları",
                    "enerji tüketimi",
                    "yenilenebilir enerji",
                    "iklim riskleri",
                ),
                aliases=(
                    "climate",
                    "iklim",
                    "carbon footprint",
                    "karbon ayak izi",
                ),
            ),
            ESRSDisclosure(
                code="ESRS E2",
                title="Pollution",
                pillar=ESRSPillar.ENVIRONMENTAL,
                topic="Pollution of air, water and soil",
                keywords=(
                    "air pollution",
                    "water pollution",
                    "soil pollution",
                    "substances of concern",
                    "hazardous substances",
                    "microplastics",
                    "emissions to air",
                    "hava kirliliği",
                    "su kirliliği",
                    "toprak kirliliği",
                    "zararlı maddeler",
                    "tehlikeli maddeler",
                    "mikroplastikler",
                    "havaya salımlar",
                ),
                aliases=(
                    "pollution",
                    "kirlilik",
                ),
            ),
            ESRSDisclosure(
                code="ESRS E3",
                title="Water and Marine Resources",
                pillar=ESRSPillar.ENVIRONMENTAL,
                topic="Water consumption and marine resources",
                keywords=(
                    "water withdrawal",
                    "water consumption",
                    "water discharge",
                    "water stress",
                    "marine resources",
                    "ocean impact",
                    "su çekimi",
                    "su tüketimi",
                    "su deşarjı",
                    "su stresi",
                    "deniz kaynakları",
                    "okyanus etkisi",
                ),
                aliases=(
                    "water resources",
                    "su kaynakları",
                    "marine ecosystems",
                    "deniz ekosistemleri",
                ),
            ),
            ESRSDisclosure(
                code="ESRS E4",
                title="Biodiversity and Ecosystems",
                pillar=ESRSPillar.ENVIRONMENTAL,
                topic="Biodiversity and ecosystem impacts",
                keywords=(
                    "biodiversity",
                    "ecosystems",
                    "habitat",
                    "protected areas",
                    "species",
                    "land use",
                    "deforestation",
                    "biyolojik çeşitlilik",
                    "ekosistemler",
                    "habitat",
                    "korunan alanlar",
                    "türler",
                    "arazi kullanımı",
                    "ormansızlaşma",
                ),
                aliases=(
                    "nature impact",
                    "doğa etkisi",
                    "natural capital",
                    "doğal sermaye",
                ),
            ),
            ESRSDisclosure(
                code="ESRS E5",
                title="Resource Use and Circular Economy",
                pillar=ESRSPillar.ENVIRONMENTAL,
                topic="Resource use, waste and circular economy",
                keywords=(
                    "resource use",
                    "circular economy",
                    "waste generated",
                    "waste recycled",
                    "material consumption",
                    "recycling",
                    "reuse",
                    "resource inflows",
                    "resource outflows",
                    "kaynak kullanımı",
                    "döngüsel ekonomi",
                    "oluşan atık",
                    "geri dönüştürülen atık",
                    "malzeme tüketimi",
                    "geri dönüşüm",
                    "yeniden kullanım",
                    "kaynak girdileri",
                    "kaynak çıktıları",
                ),
                aliases=(
                    "circularity",
                    "döngüsellik",
                    "waste management",
                    "atık yönetimi",
                ),
            ),
            ESRSDisclosure(
                code="ESRS S1",
                title="Own Workforce",
                pillar=ESRSPillar.SOCIAL,
                topic="Employees and non-employee workers",
                keywords=(
                    "employees",
                    "workforce",
                    "working conditions",
                    "wages",
                    "training",
                    "health and safety",
                    "diversity",
                    "gender equality",
                    "collective bargaining",
                    "work-life balance",
                    "çalışanlar",
                    "iş gücü",
                    "çalışma koşulları",
                    "ücretler",
                    "eğitim",
                    "iş sağlığı ve güvenliği",
                    "çeşitlilik",
                    "cinsiyet eşitliği",
                    "toplu sözleşme",
                    "iş yaşam dengesi",
                ),
                aliases=(
                    "own workforce",
                    "kendi iş gücü",
                    "employee matters",
                    "çalışan konuları",
                ),
            ),
            ESRSDisclosure(
                code="ESRS S2",
                title="Workers in the Value Chain",
                pillar=ESRSPillar.SOCIAL,
                topic="Workers throughout the value chain",
                keywords=(
                    "value chain workers",
                    "supplier workers",
                    "human rights",
                    "forced labour",
                    "child labour",
                    "occupational safety",
                    "living wage",
                    "değer zinciri çalışanları",
                    "tedarikçi çalışanları",
                    "insan hakları",
                    "zorla çalıştırma",
                    "çocuk işçiliği",
                    "iş güvenliği",
                    "yaşam ücreti",
                ),
                aliases=(
                    "supply chain workers",
                    "tedarik zinciri çalışanları",
                ),
            ),
            ESRSDisclosure(
                code="ESRS S3",
                title="Affected Communities",
                pillar=ESRSPillar.SOCIAL,
                topic="Communities affected by company activities",
                keywords=(
                    "affected communities",
                    "local communities",
                    "indigenous peoples",
                    "community rights",
                    "social impact",
                    "community engagement",
                    "land rights",
                    "etkilenen topluluklar",
                    "yerel topluluklar",
                    "yerli halklar",
                    "topluluk hakları",
                    "sosyal etki",
                    "toplum katılımı",
                    "arazi hakları",
                ),
                aliases=(
                    "community impact",
                    "toplumsal etki",
                ),
            ),
            ESRSDisclosure(
                code="ESRS S4",
                title="Consumers and End-users",
                pillar=ESRSPillar.SOCIAL,
                topic="Consumer and end-user impacts",
                keywords=(
                    "consumers",
                    "end users",
                    "product safety",
                    "customer privacy",
                    "data protection",
                    "responsible marketing",
                    "accessibility",
                    "customer health",
                    "tüketiciler",
                    "son kullanıcılar",
                    "ürün güvenliği",
                    "müşteri gizliliği",
                    "veri koruma",
                    "sorumlu pazarlama",
                    "erişilebilirlik",
                    "müşteri sağlığı",
                ),
                aliases=(
                    "consumer impact",
                    "tüketici etkisi",
                    "customer welfare",
                    "müşteri refahı",
                ),
            ),
            ESRSDisclosure(
                code="ESRS G1",
                title="Business Conduct",
                pillar=ESRSPillar.GOVERNANCE,
                topic="Ethics, compliance and business conduct",
                keywords=(
                    "business conduct",
                    "corporate culture",
                    "anti-corruption",
                    "anti-bribery",
                    "whistleblowing",
                    "political influence",
                    "supplier relationships",
                    "payment practices",
                    "ethics",
                    "compliance",
                    "iş etiği",
                    "kurum kültürü",
                    "yolsuzlukla mücadele",
                    "rüşvetle mücadele",
                    "ihbar mekanizması",
                    "siyasi etki",
                    "tedarikçi ilişkileri",
                    "ödeme uygulamaları",
                    "etik",
                    "uyum",
                ),
                aliases=(
                    "business ethics",
                    "iş ahlakı",
                    "corporate ethics",
                    "kurumsal etik",
                ),
            ),
        )


__all__ = [
    "ESRSDisclosure",
    "ESRSMapper",
    "ESRSMapperError",
    "ESRSMatch",
    "ESRSPillar",
    "InvalidESRSCodeError",
]