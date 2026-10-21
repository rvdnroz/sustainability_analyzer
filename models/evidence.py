from __future__ import annotations

from datetime import datetime, timezone
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


def utc_now() -> datetime:
    """
    UTC tabanlı zaman damgası oluşturur.
    """

    return datetime.now(timezone.utc)


class EvidenceSourceType(str, Enum):
    """
    Kanıtın hangi veri kaynağından geldiğini belirtir.
    """

    TEXT_CHUNK = "text_chunk"
    TABLE = "table"
    STANDARD_INDEX = "standard_index"
    PAGE = "page"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RetrievalChannel(str, Enum):
    """
    Kanıtın hangi retrieval kanalı tarafından bulunduğunu belirtir.
    """

    TABLE = "table"
    BM25 = "bm25"
    VECTOR = "vector"
    STANDARD = "standard"
    FUSION = "fusion"
    CONTEXT_EXPANSION = "context_expansion"


class EvidenceValidationStatus(str, Enum):
    """
    Kanıt doğrulama sonucunu belirtir.
    """

    UNVALIDATED = "unvalidated"
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INVALID = "invalid"


class AnswerStatus(str, Enum):
    """
    Soru cevaplama işleminin sonucunu belirtir.
    """

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class RetrievalScore(BaseModel):
    """
    Bir kanıtın belirli bir retrieval kanalındaki skorunu temsil eder.

    raw_score:
        Retrieval yönteminin ürettiği özgün skor.

    normalized_score:
        Farklı retrieval yöntemlerini karşılaştırabilmek için
        0-1 aralığına dönüştürülmüş skor.

    rank:
        İlgili retrieval kanalındaki sıralama konumu.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    channel: RetrievalChannel

    raw_score: float = Field(
        ge=0.0,
    )

    normalized_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    rank: int | None = Field(
        default=None,
        ge=1,
    )

    query_variant: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("query_variant")
    @classmethod
    def normalize_query_variant(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş sorgu varyantlarını None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            return None

        return normalized_value


class EvidenceItem(BaseModel):
    """
    Retrieval sistemi tarafından bulunan tek bir kanıt parçasını temsil eder.

    Kanıt şu kaynaklardan gelebilir:

    - Chunk
    - Tablo
    - Standart indeksi
    - PDF sayfası
    - Context expansion sonucu

    Bu model aşağıdaki modüller tarafından kullanılacaktır:

    - table_retriever.py
    - bm25_retriever.py
    - vector_retriever.py
    - standard_router.py
    - result_fusion.py
    - reranker.py
    - context_expander.py
    - evidence_validator.py
    - citation_builder.py
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    query_id: UUID

    report_id: UUID

    # Chunk, ExtractedTable veya başka bir kaynak nesnesinin kimliği
    source_id: UUID

    source_type: EvidenceSourceType

    content: str = Field(
        min_length=1,
    )

    page_numbers: list[int] = Field(
        min_length=1,
    )

    section_title: str | None = None

    subsection_title: str | None = None

    source_label: str | None = None

    retrieval_scores: list[RetrievalScore] = Field(
        default_factory=list,
    )

    # Reciprocal Rank Fusion veya başka bir fusion yöntemi sonucu
    fusion_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # Cross-encoder reranker sonucu
    rerank_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # evidence_validator.py tarafından hesaplanacak skor
    validation_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    # Bütün skorlar değerlendirildikten sonra kullanılacak son skor
    final_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    validation_status: EvidenceValidationStatus = (
        EvidenceValidationStatus.UNVALIDATED
    )

    validation_notes: list[str] = Field(
        default_factory=list,
    )

    standard_references: list[str] = Field(
        default_factory=list,
    )

    # Tablo kanıtlarında ilgili satır ve sütun bilgileri
    row_indexes: list[int] = Field(
        default_factory=list,
    )

    column_names: list[str] = Field(
        default_factory=list,
    )

    # Kanıt doğrudan cevap için değil, komşu bağlam olarak eklendiyse True
    is_context_only: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("content")
    @classmethod
    def normalize_content(
        cls,
        value: str,
    ) -> str:
        """
        Kanıt metnindeki gereksiz boşlukları temizler.
        """

        normalized_lines = [
            " ".join(line.strip().split())
            for line in value.replace("\r", "\n").split("\n")
            if line.strip()
        ]

        normalized_content = "\n".join(
            normalized_lines
        ).strip()

        if not normalized_content:
            raise ValueError(
                "Kanıt içeriği boş olamaz."
            )

        return normalized_content

    @field_validator(
        "section_title",
        "subsection_title",
        "source_label",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş metin alanlarını None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            return None

        return normalized_value

    @field_validator("page_numbers")
    @classmethod
    def validate_page_numbers(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Sayfa numaralarını doğrular, sıralar ve tekrarları kaldırır.
        """

        if not values:
            raise ValueError(
                "Kanıt en az bir sayfa numarası içermelidir."
            )

        invalid_page_numbers = [
            page_number
            for page_number in values
            if page_number < 1
        ]

        if invalid_page_numbers:
            raise ValueError(
                "Geçersiz sayfa numaraları bulundu: "
                f"{invalid_page_numbers}"
            )

        return sorted(set(values))

    @field_validator("row_indexes")
    @classmethod
    def validate_row_indexes(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Tablo satır indekslerini doğrular ve tekrarları kaldırır.
        """

        invalid_indexes = [
            row_index
            for row_index in values
            if row_index < 0
        ]

        if invalid_indexes:
            raise ValueError(
                "Tablo satır indeksleri negatif olamaz."
            )

        return sorted(set(values))

    @field_validator(
        "validation_notes",
        "standard_references",
        "column_names",
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

    @property
    def ranking_score(self) -> float:
        """
        Kanıt sıralamasında kullanılabilecek en gelişmiş mevcut skoru döndürür.

        Öncelik sırası:

        1. final_score
        2. rerank_score
        3. fusion_score
        4. Retrieval kanallarındaki en yüksek normalize skor
        """

        if self.final_score is not None:
            return self.final_score

        if self.rerank_score is not None:
            return self.rerank_score

        if self.fusion_score is not None:
            return self.fusion_score

        normalized_scores = [
            score.normalized_score
            for score in self.retrieval_scores
            if score.normalized_score is not None
        ]

        if normalized_scores:
            return max(normalized_scores)

        return 0.0

    @property
    def retrieval_channels(
        self,
    ) -> list[RetrievalChannel]:
        """
        Kanıtı bulan retrieval kanallarını döndürür.
        """

        unique_channels: list[RetrievalChannel] = []

        for score in self.retrieval_scores:
            if score.channel not in unique_channels:
                unique_channels.append(
                    score.channel
                )

        return unique_channels

    @property
    def is_valid(
        self,
    ) -> bool:
        """
        Kanıtın doğrulama sonucunun geçerli olup olmadığını belirtir.
        """

        return self.validation_status == EvidenceValidationStatus.VALID

    @property
    def citation_label(
        self,
    ) -> str:
        """
        Kanıt için okunabilir sayfa etiketi oluşturur.
        """

        if len(self.page_numbers) == 1:
            return f"Sayfa {self.page_numbers[0]}"

        consecutive_pages = self.page_numbers == list(
            range(
                self.page_numbers[0],
                self.page_numbers[-1] + 1,
            )
        )

        if consecutive_pages:
            return (
                f"Sayfalar "
                f"{self.page_numbers[0]}-"
                f"{self.page_numbers[-1]}"
            )

        pages = ", ".join(
            str(page_number)
            for page_number in self.page_numbers
        )

        return f"Sayfalar {pages}"

    def add_retrieval_score(
        self,
        channel: RetrievalChannel,
        raw_score: float,
        normalized_score: float | None = None,
        rank: int | None = None,
        query_variant: str | None = None,
    ) -> None:
        """
        Kanıta yeni bir retrieval skoru ekler.

        Aynı kanal ve sorgu varyantına ait skor daha önce eklenmişse
        mevcut kayıt güncellenir.
        """

        new_score = RetrievalScore(
            channel=channel,
            raw_score=raw_score,
            normalized_score=normalized_score,
            rank=rank,
            query_variant=query_variant,
        )

        for index, existing_score in enumerate(
            self.retrieval_scores
        ):
            same_channel = (
                existing_score.channel
                == new_score.channel
            )

            same_query_variant = (
                existing_score.query_variant
                == new_score.query_variant
            )

            if same_channel and same_query_variant:
                self.retrieval_scores[index] = new_score
                return

        self.retrieval_scores.append(
            new_score
        )

    def set_fusion_score(
        self,
        score: float,
    ) -> None:
        """
        Result fusion sonucunu kaydeder.
        """

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Fusion skoru 0 ile 1 arasında olmalıdır."
            )

        self.fusion_score = score

    def set_rerank_score(
        self,
        score: float,
    ) -> None:
        """
        Cross-encoder reranker sonucunu kaydeder.
        """

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Rerank skoru 0 ile 1 arasında olmalıdır."
            )

        self.rerank_score = score

    def set_final_score(
        self,
        score: float,
    ) -> None:
        """
        Nihai kanıt skorunu kaydeder.
        """

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Nihai kanıt skoru 0 ile 1 arasında olmalıdır."
            )

        self.final_score = score

    def mark_valid(
        self,
        validation_score: float,
        note: str | None = None,
    ) -> None:
        """
        Kanıtı geçerli olarak işaretler.
        """

        if not 0.0 <= validation_score <= 1.0:
            raise ValueError(
                "Doğrulama skoru 0 ile 1 arasında olmalıdır."
            )

        self.validation_status = EvidenceValidationStatus.VALID
        self.validation_score = validation_score

        if note is not None:
            self.add_validation_note(note)

    def mark_partially_valid(
        self,
        validation_score: float,
        note: str,
    ) -> None:
        """
        Kanıtı kısmen geçerli olarak işaretler.
        """

        if not 0.0 <= validation_score <= 1.0:
            raise ValueError(
                "Doğrulama skoru 0 ile 1 arasında olmalıdır."
            )

        self.validation_status = (
            EvidenceValidationStatus.PARTIALLY_VALID
        )
        self.validation_score = validation_score
        self.add_validation_note(note)

    def mark_invalid(
        self,
        note: str,
        validation_score: float = 0.0,
    ) -> None:
        """
        Kanıtı geçersiz olarak işaretler.
        """

        if not 0.0 <= validation_score <= 1.0:
            raise ValueError(
                "Doğrulama skoru 0 ile 1 arasında olmalıdır."
            )

        self.validation_status = EvidenceValidationStatus.INVALID
        self.validation_score = validation_score
        self.add_validation_note(note)

    def add_validation_note(
        self,
        note: str,
    ) -> None:
        """
        Kanıta doğrulama açıklaması ekler.
        """

        normalized_note = " ".join(
            note.strip().split()
        )

        if not normalized_note:
            raise ValueError(
                "Doğrulama notu boş olamaz."
            )

        existing_notes = {
            existing_note.casefold()
            for existing_note in self.validation_notes
        }

        if normalized_note.casefold() not in existing_notes:
            self.validation_notes.append(
                normalized_note
            )

    def update_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Kanıta ek metadata bilgisi ekler.
        """

        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError(
                "Metadata anahtarı boş olamaz."
            )

        self.metadata[normalized_key] = value


class EvidenceBundle(BaseModel):
    """
    Bir soru için seçilen bütün kanıtları tek yapıda toplar.

    Result fusion, reranking, context expansion ve evidence validation
    tamamlandıktan sonra Gemini'ye gönderilecek bağlam bu model üzerinden
    oluşturulacaktır.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    query_id: UUID

    report_id: UUID

    items: list[EvidenceItem] = Field(
        default_factory=list,
    )

    sufficient: bool = False

    sufficiency_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    insufficiency_reason: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("insufficiency_reason")
    @classmethod
    def normalize_insufficiency_reason(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş yetersizlik açıklamalarını None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            return None

        return normalized_value

    @model_validator(mode="after")
    def validate_items(
        self,
    ) -> EvidenceBundle:
        """
        Bundle içerisindeki kanıtların aynı sorgu ve rapora ait
        olmasını doğrular.
        """

        duplicate_ids: set[UUID] = set()

        for item in self.items:
            if item.query_id != self.query_id:
                raise ValueError(
                    "EvidenceItem query_id değeri bundle query_id "
                    "değeriyle uyuşmuyor."
                )

            if item.report_id != self.report_id:
                raise ValueError(
                    "EvidenceItem report_id değeri bundle report_id "
                    "değeriyle uyuşmuyor."
                )

            if item.id in duplicate_ids:
                raise ValueError(
                    "EvidenceBundle içerisinde tekrar eden kanıt "
                    f"kimliği bulundu: {item.id}"
                )

            duplicate_ids.add(item.id)

        return self

    @property
    def page_numbers(
        self,
    ) -> list[int]:
        """
        Bundle içerisindeki bütün kanıtların sayfa numaralarını döndürür.
        """

        page_numbers = {
            page_number
            for item in self.items
            for page_number in item.page_numbers
        }

        return sorted(page_numbers)

    @property
    def validated_items(
        self,
    ) -> list[EvidenceItem]:
        """
        Geçerli veya kısmen geçerli kanıtları döndürür.
        """

        accepted_statuses = {
            EvidenceValidationStatus.VALID,
            EvidenceValidationStatus.PARTIALLY_VALID,
        }

        return [
            item
            for item in self.items
            if item.validation_status in accepted_statuses
        ]

    @property
    def answer_evidence(
        self,
    ) -> list[EvidenceItem]:
        """
        Yalnızca doğrudan cevap üretiminde kullanılacak kanıtları döndürür.

        Context expansion ile eklenmiş yardımcı parçalar dışarıda bırakılır.
        """

        return [
            item
            for item in self.validated_items
            if not item.is_context_only
        ]

    @property
    def has_table_evidence(
        self,
    ) -> bool:
        """
        Bundle içerisinde tablo kanıtı olup olmadığını belirtir.
        """

        return any(
            item.source_type == EvidenceSourceType.TABLE
            for item in self.items
        )

    @property
    def has_valid_evidence(
        self,
    ) -> bool:
        """
        En az bir geçerli kanıt bulunup bulunmadığını belirtir.
        """

        return bool(
            self.validated_items
        )

    def add_item(
        self,
        item: EvidenceItem,
    ) -> None:
        """
        Bundle içerisine yeni kanıt ekler.
        """

        if item.query_id != self.query_id:
            raise ValueError(
                "Eklenen kanıt farklı bir sorguya ait."
            )

        if item.report_id != self.report_id:
            raise ValueError(
                "Eklenen kanıt farklı bir rapora ait."
            )

        if any(
            existing_item.id == item.id
            for existing_item in self.items
        ):
            raise ValueError(
                "Bu kanıt bundle içerisinde zaten bulunuyor."
            )

        self.items.append(
            item
        )

    def sort_by_score(
        self,
    ) -> None:
        """
        Kanıtları ranking_score değerine göre büyükten küçüğe sıralar.
        """

        self.items.sort(
            key=lambda item: item.ranking_score,
            reverse=True,
        )

    def top_items(
        self,
        limit: int,
        validated_only: bool = False,
    ) -> list[EvidenceItem]:
        """
        En yüksek skorlu ilk kanıtları döndürür.
        """

        if limit < 1:
            raise ValueError(
                "Kanıt limiti en az 1 olmalıdır."
            )

        candidate_items = (
            self.validated_items
            if validated_only
            else self.items
        )

        return sorted(
            candidate_items,
            key=lambda item: item.ranking_score,
            reverse=True,
        )[:limit]

    def mark_sufficient(
        self,
        sufficiency_score: float,
    ) -> None:
        """
        Kanıt paketini soru için yeterli olarak işaretler.
        """

        if not 0.0 <= sufficiency_score <= 1.0:
            raise ValueError(
                "Yeterlilik skoru 0 ile 1 arasında olmalıdır."
            )

        self.sufficient = True
        self.sufficiency_score = sufficiency_score
        self.insufficiency_reason = None

    def mark_insufficient(
        self,
        reason: str,
        sufficiency_score: float = 0.0,
    ) -> None:
        """
        Kanıt paketini yetersiz olarak işaretler.
        """

        normalized_reason = " ".join(
            reason.strip().split()
        )

        if not normalized_reason:
            raise ValueError(
                "Yetersizlik nedeni boş olamaz."
            )

        if not 0.0 <= sufficiency_score <= 1.0:
            raise ValueError(
                "Yeterlilik skoru 0 ile 1 arasında olmalıdır."
            )

        self.sufficient = False
        self.sufficiency_score = sufficiency_score
        self.insufficiency_reason = normalized_reason


class Citation(BaseModel):
    """
    Cevap içerisinde kullanılabilecek kaynak gösterimini temsil eder.

    Kullanıcıya kanıt gösterilmemesi tercih edilse bile bu yapı,
    sistemin iç denetimi ve cevap doğrulaması için korunacaktır.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    evidence_id: UUID

    report_id: UUID

    source_type: EvidenceSourceType

    page_numbers: list[int] = Field(
        min_length=1,
    )

    label: str = Field(
        min_length=1,
    )

    excerpt: str | None = None

    section_title: str | None = None

    @field_validator("page_numbers")
    @classmethod
    def validate_page_numbers(
        cls,
        values: list[int],
    ) -> list[int]:
        """
        Kaynak sayfalarını doğrular ve tekrarları kaldırır.
        """

        if any(page_number < 1 for page_number in values):
            raise ValueError(
                "Kaynak sayfa numaraları 1 veya daha büyük olmalıdır."
            )

        return sorted(set(values))

    @field_validator(
        "label",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: str,
    ) -> str:
        """
        Zorunlu metin alanlarını temizler.
        """

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            raise ValueError(
                "Kaynak etiketi boş olamaz."
            )

        return normalized_value

    @field_validator(
        "excerpt",
        "section_title",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş isteğe bağlı metinleri None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            return None

        return normalized_value


class AnswerResult(BaseModel):
    """
    Soru cevaplama pipeline'ının nihai çıktısını temsil eder.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    query_id: UUID

    report_id: UUID

    status: AnswerStatus

    answer: str | None = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    citations: list[Citation] = Field(
        default_factory=list,
    )

    evidence_ids: list[UUID] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

    model_name: str | None = None

    raw_model_response: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator(
        "answer",
        "model_name",
        "raw_model_response",
    )
    @classmethod
    def normalize_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Boş metin alanlarını None değerine dönüştürür.
        """

        if value is None:
            return None

        normalized_value = value.strip()

        if not normalized_value:
            return None

        return normalized_value

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Uyarıları temizler ve tekrarları kaldırır.
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

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(
        cls,
        values: list[UUID],
    ) -> list[UUID]:
        """
        Tekrar eden kanıt kimliklerini kaldırır.
        """

        unique_values: list[UUID] = []

        for value in values:
            if value not in unique_values:
                unique_values.append(value)

        return unique_values

    @model_validator(mode="after")
    def validate_answer_status(
        self,
    ) -> AnswerResult:
        """
        Başarılı cevap durumunda answer alanının dolu olmasını sağlar.
        """

        if (
            self.status == AnswerStatus.ANSWERED
            and self.answer is None
        ):
            raise ValueError(
                "ANSWERED durumundaki bir sonuç cevap metni içermelidir."
            )

        return self

    def add_warning(
        self,
        warning: str,
    ) -> None:
        """
        Cevap sonucuna yeni bir uyarı ekler.
        """

        normalized_warning = " ".join(
            warning.strip().split()
        )

        if not normalized_warning:
            raise ValueError(
                "Uyarı metni boş olamaz."
            )

        existing_warnings = {
            existing_warning.casefold()
            for existing_warning in self.warnings
        }

        if normalized_warning.casefold() not in existing_warnings:
            self.warnings.append(
                normalized_warning
            )

    def add_citation(
        self,
        citation: Citation,
    ) -> None:
        """
        Cevaba kaynak bilgisi ekler.
        """

        if citation.report_id != self.report_id:
            raise ValueError(
                "Citation farklı bir rapora ait."
            )

        if any(
            existing_citation.evidence_id
            == citation.evidence_id
            for existing_citation in self.citations
        ):
            return

        self.citations.append(
            citation
        )

        if citation.evidence_id not in self.evidence_ids:
            self.evidence_ids.append(
                citation.evidence_id
            )