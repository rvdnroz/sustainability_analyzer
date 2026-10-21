from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class QueryIntent(str, Enum):
    """
    Sorunun temel bilgi ihtiyacını belirtir.

    Bu sınıflandırma question_analyzer.py tarafından yapılacaktır.
    """

    KPI = "kpi"
    TARGET = "target"
    POLICY = "policy"
    TREND = "trend"
    COMPARISON = "comparison"
    STANDARD_DISCLOSURE = "standard_disclosure"
    DEFINITION = "definition"
    EXISTENCE = "existence"
    GENERAL = "general"
    UNKNOWN = "unknown"


class AnswerType(str, Enum):
    """
    Sorudan beklenen cevap türünü belirtir.
    """

    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    """
    Sorunun cevaplanması için gereken işlem karmaşıklığını belirtir.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RetrievalRoute(str, Enum):
    """
    Sorunun hangi retrieval kanallarında aranması gerektiğini belirtir.
    """

    TABLE = "table"
    BM25 = "bm25"
    VECTOR = "vector"
    STANDARD = "standard"
    HYBRID = "hybrid"


class QueryFilter(BaseModel):
    """
    Retrieval sırasında uygulanabilecek filtreleri temsil eder.

    Örneğin yalnızca belirli sayfalarda, yıllarda, bölümlerde veya
    ESG kategorilerinde arama yapılmasını sağlar.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    page_numbers: list[int] = Field(
        default_factory=list,
    )

    years: list[int] = Field(
        default_factory=list,
    )

    categories: list[str] = Field(
        default_factory=list,
    )

    section_titles: list[str] = Field(
        default_factory=list,
    )

    standard_references: list[str] = Field(
        default_factory=list,
    )

    @field_validator("page_numbers")
    @classmethod
    def validate_page_numbers(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Geçersiz sayfa numaralarını engeller ve tekrarları kaldırır.
        """

        if any(page_number < 1 for page_number in values):
            raise ValueError(
                "Sayfa numaraları 1 veya daha büyük olmalıdır."
            )

        return sorted(set(values))

    @field_validator("years")
    @classmethod
    def validate_years(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Yıl değerlerini kontrol eder ve tekrarları kaldırır.
        """

        invalid_years = [
            year
            for year in values
            if year < 1990 or year > 2100
        ]

        if invalid_years:
            raise ValueError(
                "Geçersiz yıl değerleri bulundu: "
                f"{invalid_years}"
            )

        return sorted(set(values))

    @field_validator(
        "categories",
        "section_titles",
        "standard_references",
    )
    @classmethod
    def normalize_string_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Metin listelerini temizler ve büyük-küçük harf duyarsız
        şekilde tekrarları kaldırır.
        """

        normalized_values: list[str] = []
        seen_values: set[str] = set()

        for value in values:
            normalized_value = value.strip()

            if not normalized_value:
                continue

            comparison_value = normalized_value.casefold()

            if comparison_value in seen_values:
                continue

            seen_values.add(comparison_value)
            normalized_values.append(normalized_value)

        return normalized_values

    @property
    def is_empty(self) -> bool:
        """
        Herhangi bir retrieval filtresi tanımlanıp tanımlanmadığını belirtir.
        """

        return not any(
            [
                self.page_numbers,
                self.years,
                self.categories,
                self.section_titles,
                self.standard_references,
            ]
        )


class QueryAnalysis(BaseModel):
    """
    Kullanıcı sorusunun retrieval öncesindeki analiz sonucunu temsil eder.

    Bu model aşağıdaki modüller arasında ortak veri yapısı olacaktır:

    - question_analyzer.py
    - query_expander.py
    - standard_router.py
    - table_retriever.py
    - bm25_retriever.py
    - vector_retriever.py
    - result_fusion.py
    - reranker.py
    - answer_generator.py
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    report_id: UUID | None = None

    original_question: str = Field(
        min_length=1,
    )

    normalized_question: str = Field(
        min_length=1,
    )

    intent: QueryIntent = QueryIntent.UNKNOWN

    expected_answer_type: AnswerType = AnswerType.UNKNOWN

    complexity: QueryComplexity = QueryComplexity.SIMPLE

    # Question analyzer tarafından çıkarılacak temel terimler
    keywords: list[str] = Field(
        default_factory=list,
    )

    # Şirket adı, gösterge adı, tarih, standart kodu gibi varlıklar
    entities: dict[str, list[str]] = Field(
        default_factory=dict,
    )

    categories: list[str] = Field(
        default_factory=list,
    )

    standard_references: list[str] = Field(
        default_factory=list,
    )

    requested_years: list[int] = Field(
        default_factory=list,
    )

    # Query expander tarafından üretilen alternatif sorgular
    expanded_queries: list[str] = Field(
        default_factory=list,
    )

    retrieval_routes: list[RetrievalRoute] = Field(
        default_factory=lambda: [
            RetrievalRoute.HYBRID,
        ],
    )

    filters: QueryFilter = Field(
        default_factory=QueryFilter,
    )

    requires_numeric_evidence: bool = False

    table_preferred: bool = False

    requires_standard_mapping: bool = False

    requires_context_expansion: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "original_question",
        "normalized_question",
    )
    @classmethod
    def normalize_question_text(
        cls,
        value: str,
    ) -> str:
        """
        Soru metnindeki gereksiz boşlukları temizler.
        """

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            raise ValueError(
                "Soru metni boş olamaz."
            )

        return normalized_value

    @field_validator(
        "keywords",
        "categories",
        "standard_references",
        "expanded_queries",
    )
    @classmethod
    def normalize_unique_strings(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Metin listelerini temizler ve tekrarları kaldırır.
        """

        normalized_values: list[str] = []
        seen_values: set[str] = set()

        for value in values:
            normalized_value = " ".join(
                value.strip().split()
            )

            if not normalized_value:
                continue

            comparison_value = normalized_value.casefold()

            if comparison_value in seen_values:
                continue

            seen_values.add(comparison_value)
            normalized_values.append(normalized_value)

        return normalized_values

    @field_validator("requested_years")
    @classmethod
    def validate_requested_years(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Soruda geçen yılları doğrular ve tekrarları kaldırır.
        """

        invalid_years = [
            year
            for year in values
            if year < 1990 or year > 2100
        ]

        if invalid_years:
            raise ValueError(
                "Geçersiz sorgu yılı bulundu: "
                f"{invalid_years}"
            )

        return sorted(set(values))

    @field_validator("retrieval_routes")
    @classmethod
    def normalize_retrieval_routes(
        cls,
        values: list[RetrievalRoute],
    ) -> list[RetrievalRoute]:
        """
        Retrieval route listesinin boş olmasını engeller ve
        tekrar eden rotaları kaldırır.
        """

        if not values:
            return [RetrievalRoute.HYBRID]

        unique_routes: list[RetrievalRoute] = []

        for route in values:
            if route not in unique_routes:
                unique_routes.append(route)

        return unique_routes

    @field_validator("entities")
    @classmethod
    def normalize_entities(
        cls,
        values: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """
        Entity anahtarlarını ve değerlerini temizler.
        """

        normalized_entities: dict[str, list[str]] = {}

        for key, entity_values in values.items():
            normalized_key = key.strip()

            if not normalized_key:
                continue

            normalized_items: list[str] = []
            seen_items: set[str] = set()

            for entity_value in entity_values:
                normalized_value = " ".join(
                    entity_value.strip().split()
                )

                if not normalized_value:
                    continue

                comparison_value = normalized_value.casefold()

                if comparison_value in seen_items:
                    continue

                seen_items.add(comparison_value)
                normalized_items.append(normalized_value)

            if normalized_items:
                normalized_entities[normalized_key] = normalized_items

        return normalized_entities

    @model_validator(mode="after")
    def infer_retrieval_requirements(
        self,
    ) -> QueryAnalysis:
        """
        Soru türüne göre bazı retrieval tercihlerinin otomatik
        belirlenmesini sağlar.
        """

        numeric_answer_types = {
            AnswerType.NUMBER,
            AnswerType.PERCENTAGE,
            AnswerType.CURRENCY,
        }

        if self.expected_answer_type in numeric_answer_types:
            object.__setattr__(
                self,
                "requires_numeric_evidence",
                True,
            )

        table_preferred_intents = {
            QueryIntent.KPI,
            QueryIntent.TREND,
            QueryIntent.COMPARISON,
        }

        if (
            self.intent in table_preferred_intents
            or self.expected_answer_type == AnswerType.TABLE
        ):
            object.__setattr__(
                self,
                "table_preferred",
                True,
            )

        if self.intent == QueryIntent.STANDARD_DISCLOSURE:
            object.__setattr__(
                self,
                "requires_standard_mapping",
                True,
            )

        return self

    @property
    def primary_query(self) -> str:
        """
        Retrieval işlemlerinde kullanılacak ana sorguyu döndürür.
        """

        return self.normalized_question

    @property
    def all_queries(self) -> list[str]:
        """
        Ana sorgu ile genişletilmiş sorguları tek listede döndürür.
        """

        queries = [
            self.normalized_question,
            *self.expanded_queries,
        ]

        unique_queries: list[str] = []
        seen_queries: set[str] = set()

        for query in queries:
            comparison_value = query.casefold()

            if comparison_value in seen_queries:
                continue

            seen_queries.add(comparison_value)
            unique_queries.append(query)

        return unique_queries

    @property
    def is_numeric_query(self) -> bool:
        """
        Sorunun sayısal bir cevap gerektirip gerektirmediğini belirtir.
        """

        numeric_answer_types = {
            AnswerType.NUMBER,
            AnswerType.PERCENTAGE,
            AnswerType.CURRENCY,
        }

        return (
            self.requires_numeric_evidence
            or self.expected_answer_type in numeric_answer_types
        )

    @property
    def is_multi_year_query(self) -> bool:
        """
        Soruda birden fazla yıl bulunup bulunmadığını belirtir.
        """

        return len(self.requested_years) > 1

    @property
    def should_search_tables_first(self) -> bool:
        """
        Tablo repository'sinin ilk retrieval kanalı olup olmayacağını belirtir.
        """

        return (
            self.table_preferred
            or self.is_numeric_query
            or RetrievalRoute.TABLE in self.retrieval_routes
        )

    def add_expanded_query(
        self,
        query: str,
    ) -> None:
        """
        Query expander tarafından üretilen yeni bir sorguyu ekler.
        """

        normalized_query = " ".join(
            query.strip().split()
        )

        if not normalized_query:
            raise ValueError(
                "Genişletilmiş sorgu boş olamaz."
            )

        existing_queries = {
            existing_query.casefold()
            for existing_query in self.all_queries
        }

        if normalized_query.casefold() not in existing_queries:
            self.expanded_queries.append(
                normalized_query
            )

    def add_keyword(
        self,
        keyword: str,
    ) -> None:
        """
        Sorguya yeni bir anahtar kelime ekler.
        """

        normalized_keyword = " ".join(
            keyword.strip().split()
        )

        if not normalized_keyword:
            raise ValueError(
                "Anahtar kelime boş olamaz."
            )

        existing_keywords = {
            existing_keyword.casefold()
            for existing_keyword in self.keywords
        }

        if normalized_keyword.casefold() not in existing_keywords:
            self.keywords.append(
                normalized_keyword
            )

    def add_standard_reference(
        self,
        reference: str,
    ) -> None:
        """
        Sorguya GRI, ESRS veya başka bir standart referansı ekler.
        """

        normalized_reference = " ".join(
            reference.strip().split()
        )

        if not normalized_reference:
            raise ValueError(
                "Standart referansı boş olamaz."
            )

        existing_references = {
            existing_reference.casefold()
            for existing_reference in self.standard_references
        }

        if normalized_reference.casefold() not in existing_references:
            self.standard_references.append(
                normalized_reference
            )

    def add_retrieval_route(
        self,
        route: RetrievalRoute,
    ) -> None:
        """
        Sorguya yeni bir retrieval rotası ekler.
        """

        if route not in self.retrieval_routes:
            self.retrieval_routes.append(
                route
            )

    def add_entity(
        self,
        entity_type: str,
        value: str,
    ) -> None:
        """
        Sorguya yeni bir entity ekler.

        Örnek:

        entity_type = "metric"
        value = "Scope 1 emisyonları"
        """

        normalized_type = entity_type.strip()
        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_type:
            raise ValueError(
                "Entity türü boş olamaz."
            )

        if not normalized_value:
            raise ValueError(
                "Entity değeri boş olamaz."
            )

        current_values = self.entities.setdefault(
            normalized_type,
            [],
        )

        existing_values = {
            current_value.casefold()
            for current_value in current_values
        }

        if normalized_value.casefold() not in existing_values:
            current_values.append(
                normalized_value
            )

    def update_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Sorguya ek metadata bilgisi ekler.
        """

        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError(
                "Metadata anahtarı boş olamaz."
            )

        self.metadata[normalized_key] = value