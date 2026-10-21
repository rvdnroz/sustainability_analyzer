from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
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
    Zaman damgalarının UTC olarak oluşturulmasını sağlar.
    """

    return datetime.now(timezone.utc)


def normalize_http_url(value: str) -> str:
    """
    URL değerini temizler ve HTTP/HTTPS formatında olup olmadığını kontrol eder.
    """

    normalized_url = value.strip()

    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError(
            "URL yalnızca http:// veya https:// protokolünü kullanmalıdır."
        )

    if not parsed_url.netloc:
        raise ValueError(
            "URL geçerli bir alan adı içermelidir."
        )

    return normalized_url


def extract_domain(url: str) -> str:
    """
    URL içerisinden sadeleştirilmiş alan adını çıkarır.

    Örnek:
        https://www.example.com/report.pdf
        -> example.com
    """

    domain = urlparse(url).netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


class ReportStatus(str, Enum):
    """
    Bir sürdürülebilirlik raporunun sistem içerisindeki durumları.
    """

    DISCOVERED = "discovered"
    VALIDATED = "validated"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ReportCandidate(BaseModel):
    """
    Exa araması sonucunda bulunan bir rapor adayını temsil eder.

    Bu model henüz kesin olarak seçilmemiş rapor URL'leri için kullanılır.
    Candidate ranker daha sonra bu alanlardaki skorları kullanarak en uygun
    rapor adayını belirler.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    company_name: str = Field(
        min_length=1,
    )

    report_year: int = Field(
        ge=1990,
        le=2100,
    )

    url: str

    title: str = ""

    snippet: str | None = None

    source_domain: str | None = None

    # Exa tarafından döndürülen ham skor.
    # Exa skorunun mutlaka 0-1 aralığında olacağı varsayılmamıştır.
    exa_score: float | None = Field(
        default=None,
        ge=0.0,
    )

    # Aşağıdaki skorlar candidate_ranker.py tarafından hesaplanacaktır.
    company_match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    year_match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    report_type_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    pdf_likelihood_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    domain_authority_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    final_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    is_validated: bool = False

    validation_notes: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("url")
    @classmethod
    def validate_url(
        cls,
        value: str,
    ) -> str:
        return normalize_http_url(value)

    @model_validator(mode="after")
    def populate_source_domain(
        self,
    ) -> ReportCandidate:
        """
        source_domain verilmemişse URL üzerinden otomatik olarak üretir.
        """

        if self.source_domain is None:
            self.source_domain = extract_domain(self.url)

        return self


class Report(BaseModel):
    """
    Sistemde işlenmek üzere seçilmiş asıl raporu temsil eder.

    ReportCandidate geçici bir arama sonucu iken Report nesnesi,
    doğrulama, indirme, ayrıştırma ve retrieval aşamalarında kullanılan
    kalıcı ana veri modelidir.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
    )

    selected_candidate_id: UUID | None = None

    company_name: str = Field(
        min_length=1,
    )

    report_year: int = Field(
        ge=1990,
        le=2100,
    )

    title: str | None = None

    source_url: str

    source_domain: str | None = None

    local_pdf_path: Path | None = None

    sha256: str | None = None

    content_type: str | None = None

    status: ReportStatus = ReportStatus.DISCOVERED

    page_count: int | None = Field(
        default=None,
        ge=1,
    )

    file_size_bytes: int | None = Field(
        default=None,
        ge=0,
    )

    language: str | None = None

    discovered_at: datetime = Field(
        default_factory=utc_now,
    )

    validated_at: datetime | None = None

    downloaded_at: datetime | None = None

    processed_at: datetime | None = None

    error_message: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("source_url")
    @classmethod
    def validate_source_url(
        cls,
        value: str,
    ) -> str:
        return normalize_http_url(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(
        cls,
        value: str | None,
    ) -> str | None:
        """
        SHA-256 değerinin 64 karakterlik hexadecimal formatta
        olup olmadığını kontrol eder.
        """

        if value is None:
            return None

        normalized_hash = value.strip().lower()

        if len(normalized_hash) != 64:
            raise ValueError(
                "SHA-256 değeri tam olarak 64 karakter olmalıdır."
            )

        valid_characters = set("0123456789abcdef")

        if any(
            character not in valid_characters
            for character in normalized_hash
        ):
            raise ValueError(
                "SHA-256 değeri yalnızca hexadecimal karakterler içermelidir."
            )

        return normalized_hash

    @model_validator(mode="after")
    def populate_source_domain(
        self,
    ) -> Report:
        """
        source_domain verilmemişse kaynak URL üzerinden otomatik oluşturur.
        """

        if self.source_domain is None:
            self.source_domain = extract_domain(self.source_url)

        return self

    @classmethod
    def from_candidate(
        cls,
        candidate: ReportCandidate,
    ) -> Report:
        """
        Seçilmiş bir ReportCandidate nesnesini Report nesnesine dönüştürür.
        """

        return cls(
            selected_candidate_id=candidate.id,
            company_name=candidate.company_name,
            report_year=candidate.report_year,
            title=candidate.title or None,
            source_url=candidate.url,
            source_domain=candidate.source_domain,
            metadata={
                "candidate_final_score": candidate.final_score,
                "candidate_exa_score": candidate.exa_score,
            },
        )

    def mark_validated(
        self,
        content_type: str | None = None,
    ) -> None:
        """
        URL ve PDF doğrulaması başarılı olduğunda çağrılır.
        """

        self.status = ReportStatus.VALIDATED
        self.validated_at = utc_now()
        self.content_type = content_type
        self.error_message = None

    def mark_downloaded(
        self,
        local_pdf_path: Path,
        file_size_bytes: int,
        sha256: str,
    ) -> None:
        """
        PDF başarıyla indirildiğinde raporun indirme bilgilerini kaydeder.
        """

        self.local_pdf_path = local_pdf_path
        self.file_size_bytes = file_size_bytes
        self.sha256 = sha256
        self.downloaded_at = utc_now()
        self.status = ReportStatus.DOWNLOADED
        self.error_message = None

    def mark_processing(self) -> None:
        """
        PDF ayrıştırma ve indeksleme işlemleri başladığında çağrılır.
        """

        self.status = ReportStatus.PROCESSING
        self.error_message = None

    def mark_ready(
        self,
        page_count: int,
        language: str | None = None,
    ) -> None:
        """
        Rapor bütün ingestion aşamalarını tamamladığında çağrılır.
        """

        self.page_count = page_count
        self.language = language
        self.processed_at = utc_now()
        self.status = ReportStatus.READY
        self.error_message = None

    def mark_failed(
        self,
        error_message: str,
    ) -> None:
        """
        İşlem hattının herhangi bir aşamasında hata oluştuğunda çağrılır.
        """

        normalized_message = error_message.strip()

        if not normalized_message:
            normalized_message = "Bilinmeyen rapor işleme hatası."

        self.status = ReportStatus.FAILED
        self.error_message = normalized_message