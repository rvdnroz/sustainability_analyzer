from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import Settings, get_settings
from .pdf_downloader import has_pdf_signature
from models import Report


logger = logging.getLogger(__name__)


PDF_SIGNATURE_READ_SIZE = 1024
PDF_END_MARKER_READ_SIZE = 4096
DEFAULT_SAMPLE_PAGE_COUNT = 5
DEFAULT_MINIMUM_NATIVE_TEXT_CHARACTERS = 50


class PdfValidationStatus(str, Enum):
    """
    PDF doğrulama işleminin sonucunu belirtir.
    """

    VALID = "valid"
    FILE_NOT_FOUND = "file_not_found"
    NOT_A_FILE = "not_a_file"
    EMPTY_FILE = "empty_file"
    TOO_LARGE = "too_large"
    INVALID_SIGNATURE = "invalid_signature"
    HASH_MISMATCH = "hash_mismatch"
    ENCRYPTED = "encrypted"
    TOO_FEW_PAGES = "too_few_pages"
    INVALID_PAGE_STRUCTURE = "invalid_page_structure"
    PARSE_ERROR = "parse_error"
    FILE_SYSTEM_ERROR = "file_system_error"


@dataclass(frozen=True)
class PageInspection:
    """
    Örnek olarak incelenen tek bir PDF sayfasının sonucunu temsil eder.
    """

    page_number: int
    width: float | None
    height: float | None
    extracted_text_characters: int
    is_valid: bool
    error: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "extracted_text_characters": (
                self.extracted_text_characters
            ),
            "is_valid": self.is_valid,
            "error": self.error,
        }


@dataclass(frozen=True)
class PdfValidationResult:
    """
    İndirilen bir PDF için gerçekleştirilen doğrulama sonucudur.
    """

    status: PdfValidationStatus
    is_valid: bool

    local_path: Path
    file_size_bytes: int | None = None
    sha256: str | None = None

    pdf_version: str | None = None
    page_count: int | None = None

    is_encrypted: bool = False
    decrypted_with_empty_password: bool = False

    has_pdf_signature: bool = False
    has_eof_marker: bool = False

    native_text_detected: bool = False
    sampled_text_characters: int = 0

    metadata: dict[str, str | None] = field(
        default_factory=dict
    )

    inspected_pages: tuple[PageInspection, ...] = ()

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def file_size_mb(self) -> float | None:
        if self.file_size_bytes is None:
            return None

        return self.file_size_bytes / (
            1024 * 1024
        )

    def to_metadata(self) -> dict[str, object]:
        """
        Sonucu Report.metadata alanında saklanabilir biçime dönüştürür.
        """

        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "local_path": str(self.local_path),
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": self.file_size_mb,
            "sha256": self.sha256,
            "pdf_version": self.pdf_version,
            "page_count": self.page_count,
            "is_encrypted": self.is_encrypted,
            "decrypted_with_empty_password": (
                self.decrypted_with_empty_password
            ),
            "has_pdf_signature": self.has_pdf_signature,
            "has_eof_marker": self.has_eof_marker,
            "native_text_detected": self.native_text_detected,
            "sampled_text_characters": (
                self.sampled_text_characters
            ),
            "document_metadata": dict(self.metadata),
            "inspected_pages": [
                page.to_metadata()
                for page in self.inspected_pages
            ],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class PdfValidator:
    """
    İndirilen PDF dosyalarının yapısal olarak kullanılabilir olduğunu
    doğrular.

    Kontroller:

    1. Dosyanın mevcut olması
    2. Dosyanın boş olmaması
    3. Maksimum boyut sınırı
    4. PDF magic-byte imzası
    5. SHA-256 bütünlük kontrolü
    6. PDF'nin pypdf ile ayrıştırılabilmesi
    7. Şifreleme kontrolü
    8. Minimum sayfa sayısı
    9. Örnek sayfaların yapısal kontrolü
    10. Native metin bulunabilirliği
    11. PDF metadata çıkarımı
    """

    def __init__(
        self,
        settings: Settings | None = None,
        sample_page_count: int = DEFAULT_SAMPLE_PAGE_COUNT,
        minimum_native_text_characters: int = (
            DEFAULT_MINIMUM_NATIVE_TEXT_CHARACTERS
        ),
    ) -> None:
        self.settings = settings or get_settings()

        if sample_page_count < 1:
            raise ValueError(
                "sample_page_count en az 1 olmalıdır."
            )

        if minimum_native_text_characters < 0:
            raise ValueError(
                "minimum_native_text_characters negatif olamaz."
            )

        self.sample_page_count = sample_page_count

        self.minimum_native_text_characters = (
            minimum_native_text_characters
        )

        self.maximum_pdf_size_bytes = (
            self.settings.maximum_pdf_size_mb
            * 1024
            * 1024
        )

    def validate(
        self,
        source: Report | str | Path,
        *,
        password: str | None = None,
        expected_sha256: str | None = None,
    ) -> PdfValidationResult:
        """
        Bir Report nesnesini veya doğrudan yerel PDF yolunu doğrular.

        IngestionPipeline bu metoda ``Path`` gönderir. Eski kullanım
        biçimiyle ``Report`` gönderildiğinde ise doğrulama sonucu Report
        metadata alanına uygulanmaya devam eder.

        Args:
            source:
                Doğrulanacak ``Report``, ``str`` veya ``Path`` değeri.

            password:
                Şifreli PDF için isteğe bağlı parola.

            expected_sha256:
                Beklenen SHA-256 değeri. Report kullanıldığında bu değer
                verilmezse ``Report.sha256`` kullanılır.
        """

        if isinstance(
            source,
            (str, Path),
        ):
            return self.validate_path(
                path=source,
                expected_sha256=expected_sha256,
                password=password,
            )

        report = source

        if report.local_pdf_path is None:
            result = self._failure(
                path=Path("."),
                status=PdfValidationStatus.FILE_NOT_FOUND,
                error=(
                    "Report nesnesinde local_pdf_path "
                    "tanımlanmamış."
                ),
            )

            self._apply_result_to_report(
                report,
                result,
            )

            return result

        expected_hash = (
            expected_sha256
            if expected_sha256 is not None
            else report.sha256
        )

        result = self.validate_path(
            path=report.local_pdf_path,
            expected_sha256=expected_hash,
            password=password,
        )

        self._apply_result_to_report(
            report,
            result,
        )

        return result

    def validate_path(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        password: str | None = None,
    ) -> PdfValidationResult:
        """
        Verilen dosya yolundaki PDF'yi doğrular.
        """

        normalized_path = Path(
            path
        ).expanduser().resolve()

        if not normalized_path.exists():
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.FILE_NOT_FOUND,
                error="PDF dosyası bulunamadı.",
            )

        if not normalized_path.is_file():
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.NOT_A_FILE,
                error="Verilen yol bir dosyayı göstermiyor.",
            )

        try:
            file_size = normalized_path.stat().st_size
        except OSError as error:
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.FILE_SYSTEM_ERROR,
                error=(
                    "PDF dosya bilgileri okunamadı: "
                    f"{error}"
                ),
            )

        if file_size <= 0:
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.EMPTY_FILE,
                error="PDF dosyası boş.",
                file_size_bytes=file_size,
            )

        if file_size > self.maximum_pdf_size_bytes:
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.TOO_LARGE,
                error=(
                    "PDF izin verilen maksimum dosya "
                    "boyutunu aşıyor. "
                    f"Boyut: {file_size / (1024 * 1024):.2f} MB, "
                    f"sınır: {self.settings.maximum_pdf_size_mb} MB"
                ),
                file_size_bytes=file_size,
            )

        try:
            signature_sample = self._read_file_start(
                normalized_path
            )

            eof_sample = self._read_file_end(
                normalized_path
            )

            calculated_hash = self._calculate_sha256(
                normalized_path
            )
        except OSError as error:
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.FILE_SYSTEM_ERROR,
                error=(
                    "PDF dosyası okunurken dosya sistemi "
                    f"hatası oluştu: {error}"
                ),
                file_size_bytes=file_size,
            )

        signature_valid = has_pdf_signature(
            signature_sample
        )

        if not signature_valid:
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.INVALID_SIGNATURE,
                error=(
                    "Dosyanın başlangıcında geçerli "
                    "PDF imzası bulunamadı."
                ),
                file_size_bytes=file_size,
                sha256=calculated_hash,
            )

        normalized_expected_hash = self._normalize_hash(
            expected_sha256
        )

        if (
            normalized_expected_hash is not None
            and calculated_hash != normalized_expected_hash
        ):
            return self._failure(
                path=normalized_path,
                status=PdfValidationStatus.HASH_MISMATCH,
                error=(
                    "PDF'nin hesaplanan SHA-256 değeri, "
                    "beklenen değerle uyuşmuyor."
                ),
                file_size_bytes=file_size,
                sha256=calculated_hash,
                has_pdf_signature=True,
            )

        has_eof_marker = b"%%EOF" in eof_sample

        warnings: list[str] = []

        if not has_eof_marker:
            warnings.append(
                "PDF dosyasının sonunda %%EOF işareti "
                "bulunamadı; ayrıştırma sonucuyla doğrulamaya "
                "devam edildi."
            )

        return self._parse_pdf(
            path=normalized_path,
            file_size=file_size,
            calculated_hash=calculated_hash,
            has_eof_marker=has_eof_marker,
            password=password,
            warnings=warnings,
        )

    def _parse_pdf(
        self,
        *,
        path: Path,
        file_size: int,
        calculated_hash: str,
        has_eof_marker: bool,
        password: str | None,
        warnings: list[str],
    ) -> PdfValidationResult:
        """
        PDF dosyasını pypdf ile ayrıştırır ve sayfaları denetler.
        """

        reader: PdfReader | None = None

        try:
            reader = PdfReader(
                str(path),
                strict=False,
            )

            pdf_version = self._get_pdf_version(
                reader
            )

            was_encrypted = reader.is_encrypted
            empty_password_used = False

            if was_encrypted:
                password_to_try = (
                    password
                    if password is not None
                    else ""
                )

                try:
                    decryption_result = reader.decrypt(
                        password_to_try
                    )
                except Exception as error:
                    return self._failure(
                        path=path,
                        status=PdfValidationStatus.ENCRYPTED,
                        error=(
                            "Şifreli PDF çözülemedi: "
                            f"{error}"
                        ),
                        file_size_bytes=file_size,
                        sha256=calculated_hash,
                        has_pdf_signature=True,
                        has_eof_marker=has_eof_marker,
                        is_encrypted=True,
                    )

                if not decryption_result:
                    return self._failure(
                        path=path,
                        status=PdfValidationStatus.ENCRYPTED,
                        error=(
                            "PDF şifreli ve verilen parola "
                            "belgeyi açmadı."
                        ),
                        file_size_bytes=file_size,
                        sha256=calculated_hash,
                        has_pdf_signature=True,
                        has_eof_marker=has_eof_marker,
                        is_encrypted=True,
                    )

                empty_password_used = (
                    password is None
                    or password == ""
                )

                if empty_password_used:
                    warnings.append(
                        "PDF şifreleme işaretine sahipti ancak "
                        "boş parola ile açılabildi."
                    )
                else:
                    warnings.append(
                        "PDF verilen parola ile açıldı."
                    )

            page_count = len(
                reader.pages
            )

            if (
                page_count
                < self.settings.minimum_pdf_page_count
            ):
                return self._failure(
                    path=path,
                    status=PdfValidationStatus.TOO_FEW_PAGES,
                    error=(
                        "PDF beklenen minimum sayfa sayısını "
                        "karşılamıyor. "
                        f"Sayfa sayısı: {page_count}, "
                        f"minimum: "
                        f"{self.settings.minimum_pdf_page_count}"
                    ),
                    file_size_bytes=file_size,
                    sha256=calculated_hash,
                    pdf_version=pdf_version,
                    page_count=page_count,
                    has_pdf_signature=True,
                    has_eof_marker=has_eof_marker,
                    is_encrypted=was_encrypted,
                )

            inspected_pages = self._inspect_pages(
                reader=reader,
                page_count=page_count,
            )

            invalid_pages = [
                page
                for page in inspected_pages
                if not page.is_valid
            ]

            if invalid_pages:
                invalid_numbers = [
                    page.page_number
                    for page in invalid_pages
                ]

                return PdfValidationResult(
                    status=(
                        PdfValidationStatus.INVALID_PAGE_STRUCTURE
                    ),
                    is_valid=False,
                    local_path=path,
                    file_size_bytes=file_size,
                    sha256=calculated_hash,
                    pdf_version=pdf_version,
                    page_count=page_count,
                    is_encrypted=was_encrypted,
                    decrypted_with_empty_password=(
                        empty_password_used
                    ),
                    has_pdf_signature=True,
                    has_eof_marker=has_eof_marker,
                    inspected_pages=tuple(
                        inspected_pages
                    ),
                    errors=(
                        "Bazı örnek PDF sayfaları yapısal "
                        "olarak okunamadı. "
                        f"Sayfalar: {invalid_numbers}",
                    ),
                    warnings=tuple(warnings),
                )

            sampled_text_characters = sum(
                page.extracted_text_characters
                for page in inspected_pages
            )

            native_text_detected = (
                sampled_text_characters
                >= self.minimum_native_text_characters
            )

            if not native_text_detected:
                warnings.append(
                    "Örnek sayfalarda yeterli native metin "
                    "bulunamadı. Belge taranmış olabilir ve "
                    "OCR işlemine ihtiyaç duyabilir."
                )

            document_metadata = self._extract_metadata(
                reader
            )

            return PdfValidationResult(
                status=PdfValidationStatus.VALID,
                is_valid=True,
                local_path=path,
                file_size_bytes=file_size,
                sha256=calculated_hash,
                pdf_version=pdf_version,
                page_count=page_count,
                is_encrypted=was_encrypted,
                decrypted_with_empty_password=(
                    empty_password_used
                ),
                has_pdf_signature=True,
                has_eof_marker=has_eof_marker,
                native_text_detected=native_text_detected,
                sampled_text_characters=(
                    sampled_text_characters
                ),
                metadata=document_metadata,
                inspected_pages=tuple(
                    inspected_pages
                ),
                warnings=tuple(warnings),
            )

        except PdfReadError as error:
            return self._failure(
                path=path,
                status=PdfValidationStatus.PARSE_ERROR,
                error=(
                    "PDF yapısı pypdf tarafından "
                    f"ayrıştırılamadı: {error}"
                ),
                file_size_bytes=file_size,
                sha256=calculated_hash,
                has_pdf_signature=True,
                has_eof_marker=has_eof_marker,
            )

        except Exception as error:
            logger.exception(
                "Beklenmeyen PDF doğrulama hatası."
            )

            return self._failure(
                path=path,
                status=PdfValidationStatus.PARSE_ERROR,
                error=(
                    "PDF doğrulaması sırasında beklenmeyen "
                    f"bir hata oluştu: {error}"
                ),
                file_size_bytes=file_size,
                sha256=calculated_hash,
                has_pdf_signature=True,
                has_eof_marker=has_eof_marker,
            )

        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    logger.debug(
                        "PdfReader kapatılırken hata oluştu.",
                        exc_info=True,
                    )

    def _inspect_pages(
        self,
        reader: PdfReader,
        page_count: int,
    ) -> list[PageInspection]:
        """
        PDF içerisinden eşit aralıklı örnek sayfaları denetler.
        """

        page_indexes = self._select_page_indexes(
            page_count
        )

        inspections: list[PageInspection] = []

        for page_index in page_indexes:
            page_number = page_index + 1

            try:
                page = reader.pages[
                    page_index
                ]

                width = float(
                    page.mediabox.width
                )

                height = float(
                    page.mediabox.height
                )

                if width <= 0 or height <= 0:
                    raise ValueError(
                        "Sayfa boyutları pozitif değil."
                    )

                text_characters = 0

                try:
                    extracted_text = (
                        page.extract_text()
                        or ""
                    )

                    text_characters = len(
                        extracted_text.strip()
                    )

                except Exception as error:
                    logger.debug(
                        "Sayfa metni çıkarılamadı. "
                        "Sayfa=%s, hata=%s",
                        page_number,
                        error,
                    )

                inspections.append(
                    PageInspection(
                        page_number=page_number,
                        width=width,
                        height=height,
                        extracted_text_characters=(
                            text_characters
                        ),
                        is_valid=True,
                    )
                )

            except Exception as error:
                inspections.append(
                    PageInspection(
                        page_number=page_number,
                        width=None,
                        height=None,
                        extracted_text_characters=0,
                        is_valid=False,
                        error=str(error),
                    )
                )

        return inspections

    def _select_page_indexes(
        self,
        page_count: int,
    ) -> list[int]:
        """
        Belgenin başından, ortasından ve sonundan eşit aralıklı
        sayfa indeksleri seçer.
        """

        if page_count <= self.sample_page_count:
            return list(
                range(page_count)
            )

        if self.sample_page_count == 1:
            return [0]

        indexes = {
            round(
                position
                * (page_count - 1)
                / (self.sample_page_count - 1)
            )
            for position in range(
                self.sample_page_count
            )
        }

        return sorted(indexes)

    @staticmethod
    def _extract_metadata(
        reader: PdfReader,
    ) -> dict[str, str | None]:
        """
        PDF document information metadata alanlarını çıkarır.
        """

        try:
            metadata = reader.metadata
        except Exception:
            return {}

        if metadata is None:
            return {}

        fields: dict[str, Any] = {
            "title": getattr(
                metadata,
                "title",
                None,
            ),
            "author": getattr(
                metadata,
                "author",
                None,
            ),
            "subject": getattr(
                metadata,
                "subject",
                None,
            ),
            "creator": getattr(
                metadata,
                "creator",
                None,
            ),
            "producer": getattr(
                metadata,
                "producer",
                None,
            ),
            "creation_date": getattr(
                metadata,
                "creation_date",
                None,
            ),
            "modification_date": getattr(
                metadata,
                "modification_date",
                None,
            ),
        }

        normalized_metadata: dict[
            str,
            str | None,
        ] = {}

        for key, value in fields.items():
            if value is None:
                normalized_metadata[key] = None

            elif isinstance(
                value,
                datetime,
            ):
                normalized_metadata[key] = (
                    value.isoformat()
                )

            else:
                normalized_metadata[key] = str(
                    value
                ).strip() or None

        return normalized_metadata

    @staticmethod
    def _get_pdf_version(
        reader: PdfReader,
    ) -> str | None:
        """
        PDF başlığından sürüm bilgisini çıkarır.
        """

        try:
            pdf_header = reader.pdf_header
        except Exception:
            return None

        if not pdf_header:
            return None

        return str(
            pdf_header
        ).strip()

    @staticmethod
    def _read_file_start(
        path: Path,
    ) -> bytes:
        with path.open(
            mode="rb"
        ) as file:
            return file.read(
                PDF_SIGNATURE_READ_SIZE
            )

    @staticmethod
    def _read_file_end(
        path: Path,
    ) -> bytes:
        with path.open(
            mode="rb"
        ) as file:
            file.seek(
                0,
                2,
            )

            file_size = file.tell()

            read_size = min(
                PDF_END_MARKER_READ_SIZE,
                file_size,
            )

            file.seek(
                -read_size,
                2,
            )

            return file.read(
                read_size
            )

    @staticmethod
    def _calculate_sha256(
        path: Path,
    ) -> str:
        hasher = hashlib.sha256()

        with path.open(
            mode="rb"
        ) as file:
            while True:
                chunk = file.read(
                    64 * 1024
                )

                if not chunk:
                    break

                hasher.update(
                    chunk
                )

        return hasher.hexdigest()

    @staticmethod
    def _normalize_hash(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized_value = value.strip().casefold()

        if not normalized_value:
            return None

        if (
            len(normalized_value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in normalized_value
            )
        ):
            raise ValueError(
                "Beklenen SHA-256 değeri geçerli değil."
            )

        return normalized_value

    @staticmethod
    def _failure(
        *,
        path: Path,
        status: PdfValidationStatus,
        error: str,
        file_size_bytes: int | None = None,
        sha256: str | None = None,
        pdf_version: str | None = None,
        page_count: int | None = None,
        has_pdf_signature: bool = False,
        has_eof_marker: bool = False,
        is_encrypted: bool = False,
    ) -> PdfValidationResult:
        return PdfValidationResult(
            status=status,
            is_valid=False,
            local_path=path,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            pdf_version=pdf_version,
            page_count=page_count,
            is_encrypted=is_encrypted,
            has_pdf_signature=has_pdf_signature,
            has_eof_marker=has_eof_marker,
            errors=(
                error,
            ),
        )

    @staticmethod
    def _apply_result_to_report(
        report: Report,
        result: PdfValidationResult,
    ) -> None:
        """
        PDF doğrulama sonucunu Report nesnesine kaydeder.

        Başarılı doğrulamada Report durumu READY yapılmaz.
        Çünkü sayfa render, sınıflandırma, tablo çıkarımı ve chunking
        aşamaları henüz tamamlanmamıştır.
        """

        report.metadata[
            "pdf_validation"
        ] = result.to_metadata()

        if result.is_valid:
            report.page_count = (
                result.page_count
            )

            if result.sha256 is not None:
                report.sha256 = result.sha256

            report.error_message = None

        else:
            error_message = (
                "; ".join(result.errors)
                if result.errors
                else "PDF doğrulaması başarısız oldu."
            )

            report.mark_failed(
                error_message
            )


def print_validation_result(
    result: PdfValidationResult,
) -> None:
    """
    PDF doğrulama sonucunu konsola yazdırır.
    """

    print(
        "\nPDF doğrulama sonucu\n"
    )

    print(
        f"Durum: {result.status.value}"
    )

    print(
        f"Geçerli: {result.is_valid}"
    )

    print(
        f"Dosya: {result.local_path}"
    )

    if result.file_size_mb is not None:
        print(
            f"Boyut: {result.file_size_mb:.2f} MB"
        )

    print(
        f"SHA-256: {result.sha256 or 'hesaplanmadı'}"
    )

    print(
        f"PDF sürümü: {result.pdf_version or 'bilinmiyor'}"
    )

    print(
        f"Sayfa sayısı: {result.page_count or 'bilinmiyor'}"
    )

    print(
        f"Şifreli: {result.is_encrypted}"
    )

    print(
        f"Native metin tespit edildi: "
        f"{result.native_text_detected}"
    )

    print(
        f"Örnek metin karakteri: "
        f"{result.sampled_text_characters}"
    )

    if result.metadata:
        print(
            "Metadata:"
        )

        for key, value in result.metadata.items():
            if value is not None:
                print(
                    f"  {key}: {value}"
                )

    if result.inspected_pages:
        print(
            "İncelenen sayfalar:"
        )

        for page in result.inspected_pages:
            print(
                f"  Sayfa {page.page_number}: "
                f"geçerli={page.is_valid}, "
                f"boyut={page.width}x{page.height}, "
                f"metin={page.extracted_text_characters}"
            )

    if result.warnings:
        print(
            "Uyarılar:"
        )

        for warning in result.warnings:
            print(
                f"  - {warning}"
            )

    if result.errors:
        print(
            "Hatalar:"
        )

        for error in result.errors:
            print(
                f"  - {error}"
            )


def main() -> None:
    """
    Yerel bir PDF dosyasını doğrular.

    Örnek:

        python -m document_ingestion.pdf_validator \
            data/downloads/arcelik-2023-sustainability-report.pdf
    """

    parser = argparse.ArgumentParser(
        description=(
            "İndirilen sürdürülebilirlik raporu "
            "PDF'sini yapısal olarak doğrular."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Doğrulanacak yerel PDF dosyası.",
    )

    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Beklenen SHA-256 değeri.",
    )

    parser.add_argument(
        "--password",
        default=None,
        help="Şifreli PDF için parola.",
    )

    parser.add_argument(
        "--sample-pages",
        type=int,
        default=DEFAULT_SAMPLE_PAGE_COUNT,
        help="Yapısal olarak incelenecek örnek sayfa sayısı.",
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    validator = PdfValidator(
        sample_page_count=arguments.sample_pages
    )

    result = validator.validate_path(
        path=arguments.pdf_path,
        expected_sha256=arguments.expected_sha256,
        password=arguments.password,
    )

    print_validation_result(
        result
    )

    if not result.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "PageInspection",
    "PdfValidationResult",
    "PdfValidationStatus",
    "PdfValidator",
]
