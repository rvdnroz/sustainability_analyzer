from __future__ import annotations

from enum import Enum
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ChunkCategory(str, Enum):
    """
    Metin parçasının sürdürülebilirlik kategorisini belirtir.

    Kategori bilgisi category_classifier.py tarafından belirlenecektir.
    """

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    ECONOMIC = "economic"
    STANDARDS = "standards"
    GENERAL = "general"
    UNKNOWN = "unknown"


class ChunkSourceType(str, Enum):
    """
    Chunk içerisindeki metnin hangi kaynaktan üretildiğini belirtir.
    """

    NATIVE_TEXT = "native_text"
    OCR_TEXT = "ocr_text"
    TABLE_CONTEXT = "table_context"
    STANDARD_INDEX = "standard_index"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Chunk(BaseModel):
    """
    PDF içerisinden çıkarılan semantik bir metin parçasını temsil eder.

    Bu model aşağıdaki modüller arasında ortak veri yapısı olacaktır:

    - section_detector.py
    - semantic_chunker.py
    - category_classifier.py
    - metadata_enricher.py
    - chunk_repository.py
    - bm25_retriever.py
    - vector_retriever.py
    - reranker.py
    - context_expander.py
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    report_id: UUID

    # Chunk'ın rapor içerisindeki sırası
    chunk_index: int = Field(
        ge=0,
    )

    # Chunk'ın başladığı ve bittiği PDF sayfaları
    page_start: int = Field(
        ge=1,
    )

    page_end: int = Field(
        ge=1,
    )

    # Chunk'ın ana metni
    text: str = Field(
        min_length=1,
    )

    section_title: str | None = None

    subsection_title: str | None = None

    category: ChunkCategory = ChunkCategory.UNKNOWN

    category_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    source_type: ChunkSourceType = ChunkSourceType.UNKNOWN

    # Metnin üretildiği Page veya LayoutElement nesnelerinin kimlikleri
    source_element_ids: list[UUID] = Field(
        default_factory=list,
    )

    standard_references: list[str] = Field(
        default_factory=list,
    )

    keywords: list[str] = Field(
        default_factory=list,
    )

    # Tokenizer ile daha sonra hesaplanacaktır
    token_count: int | None = Field(
        default=None,
        ge=1,
    )

    # Aynı içeriğin tekrar indekslenmesini önlemek için kullanılacaktır
    content_hash: str | None = None

    # Vector store veya BM25 indeksine eklendiğini belirtir
    is_indexed: bool = False

    previous_chunk_id: UUID | None = None

    next_chunk_id: UUID | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("text")
    @classmethod
    def normalize_text(
        cls,
        value: str,
    ) -> str:
        """
        Metindeki satır sonlarını standartlaştırır ve gereksiz
        baştaki/sondaki boşlukları temizler.

        Paragraf yapısı korunur.
        """

        normalized_text = value.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        normalized_lines = [
            line.strip()
            for line in normalized_text.split("\n")
        ]

        cleaned_lines: list[str] = []
        previous_line_was_empty = False

        for line in normalized_lines:
            is_empty = not line

            if is_empty and previous_line_was_empty:
                continue

            cleaned_lines.append(line)
            previous_line_was_empty = is_empty

        normalized_text = "\n".join(cleaned_lines).strip()

        if not normalized_text:
            raise ValueError(
                "Chunk metni boş olamaz."
            )

        return normalized_text

    @field_validator(
        "section_title",
        "subsection_title",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş başlık değerlerini None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_value = value.strip()

        if not normalized_value:
            return None

        return normalized_value

    @field_validator(
        "standard_references",
        "keywords",
    )
    @classmethod
    def normalize_unique_strings(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Metin listelerini temizler ve büyük-küçük harf duyarsız
        biçimde tekrarları kaldırır.
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

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(
        cls,
        value: str | None,
    ) -> str | None:
        """
        İçerik hash değerinin SHA-256 formatında olduğunu kontrol eder.
        """

        if value is None:
            return None

        normalized_hash = value.strip().lower()

        if len(normalized_hash) != 64:
            raise ValueError(
                "content_hash değeri 64 karakter uzunluğunda olmalıdır."
            )

        valid_characters = set(
            "0123456789abcdef"
        )

        if any(
            character not in valid_characters
            for character in normalized_hash
        ):
            raise ValueError(
                "content_hash yalnızca hexadecimal karakterler içermelidir."
            )

        return normalized_hash

    @model_validator(mode="after")
    def validate_chunk(
        self,
    ) -> Chunk:
        """
        Sayfa aralığını kontrol eder ve içerik hash değerini hesaplar.
        """

        if self.page_end < self.page_start:
            raise ValueError(
                "page_end değeri page_start değerinden küçük olamaz."
            )

        calculated_hash = self.calculate_content_hash()

        object.__setattr__(
            self,
            "content_hash",
            calculated_hash,
        )

        return self

    def calculate_content_hash(
        self,
    ) -> str:
        """
        Rapor kimliği, sayfa aralığı ve metin üzerinden SHA-256 hash üretir.

        Böylece aynı rapordaki aynı metin parçalarının tekrar indekslenmesi
        engellenebilir.
        """

        hash_input = (
            f"{self.report_id}|"
            f"{self.page_start}|"
            f"{self.page_end}|"
            f"{self.text}"
        )

        return sha256(
            hash_input.encode("utf-8")
        ).hexdigest()

    @property
    def page_numbers(
        self,
    ) -> list[int]:
        """
        Chunk'ın kapsadığı bütün sayfa numaralarını döndürür.
        """

        return list(
            range(
                self.page_start,
                self.page_end + 1,
            )
        )

    @property
    def character_count(
        self,
    ) -> int:
        """
        Chunk metninin karakter sayısını döndürür.
        """

        return len(self.text)

    @property
    def word_count(
        self,
    ) -> int:
        """
        Chunk içerisindeki yaklaşık kelime sayısını döndürür.
        """

        return len(
            self.text.split()
        )

    @property
    def citation_label(
        self,
    ) -> str:
        """
        Chunk için okunabilir sayfa etiketi oluşturur.

        Örnekler:

        Sayfa 12
        Sayfalar 12-14
        """

        if self.page_start == self.page_end:
            return f"Sayfa {self.page_start}"

        return (
            f"Sayfalar "
            f"{self.page_start}-{self.page_end}"
        )

    @property
    def has_standard_references(
        self,
    ) -> bool:
        """
        Chunk içerisinde GRI, ESRS veya başka standart referansı
        bulunup bulunmadığını belirtir.
        """

        return bool(
            self.standard_references
        )

    def set_category(
        self,
        category: ChunkCategory,
        confidence: float,
    ) -> None:
        """
        category_classifier.py sonucunu chunk nesnesine kaydeder.
        """

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Kategori güven skoru 0 ile 1 arasında olmalıdır."
            )

        self.category = category
        self.category_confidence = confidence

    def add_keyword(
        self,
        keyword: str,
    ) -> None:
        """
        Chunk'a yeni bir anahtar kelime ekler.
        """

        normalized_keyword = keyword.strip()

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
        Chunk'a GRI, ESRS veya başka bir standart referansı ekler.
        """

        normalized_reference = reference.strip()

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

    def set_token_count(
        self,
        token_count: int,
    ) -> None:
        """
        Tokenizer tarafından hesaplanan token sayısını kaydeder.
        """

        if token_count < 1:
            raise ValueError(
                "Token sayısı en az 1 olmalıdır."
            )

        self.token_count = token_count

    def mark_indexed(
        self,
    ) -> None:
        """
        Chunk BM25 ve vector store indekslerine eklendiğinde çağrılır.
        """

        self.is_indexed = True

    def set_neighbors(
        self,
        previous_chunk_id: UUID | None,
        next_chunk_id: UUID | None,
    ) -> None:
        """
        Context expander tarafından kullanılacak komşu chunk
        bağlantılarını kaydeder.
        """

        if previous_chunk_id == self.id:
            raise ValueError(
                "Bir chunk kendisini önceki chunk olarak gösteremez."
            )

        if next_chunk_id == self.id:
            raise ValueError(
                "Bir chunk kendisini sonraki chunk olarak gösteremez."
            )

        if (
            previous_chunk_id is not None
            and previous_chunk_id == next_chunk_id
        ):
            raise ValueError(
                "Önceki ve sonraki chunk kimlikleri aynı olamaz."
            )

        self.previous_chunk_id = previous_chunk_id
        self.next_chunk_id = next_chunk_id

    def update_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        metadata_enricher.py tarafından yeni metadata alanı eklemek
        için kullanılır.
        """

        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError(
                "Metadata anahtarı boş olamaz."
            )

        self.metadata[normalized_key] = value