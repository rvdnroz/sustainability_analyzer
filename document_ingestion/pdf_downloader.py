from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlsplit
from uuid import UUID

import httpx

from config import Settings, get_settings
from models import Report
from report_discovery.url_validator import (
    PDF_CONTENT_TYPES,
    PDF_MAGIC_SIGNATURE,
    get_clean_content_type,
    is_public_http_url,
    parse_content_length,
)


logger = logging.getLogger(__name__)


DEFAULT_DOWNLOAD_CHUNK_SIZE = 64 * 1024
PDF_SIGNATURE_SEARCH_LIMIT = 1024


class PdfDownloadStatus(str, Enum):
    """PDF indirme işleminin sonucunu belirtir."""

    DOWNLOADED = "downloaded"
    REUSED_EXISTING = "reused_existing"


class PdfDownloadError(RuntimeError):
    """PDF indirme veya dosya doğrulama işlemi başarısız olduğunda üretilir."""


@dataclass(frozen=True)
class PdfDownloadResult:
    """Başarılı PDF indirme işleminin sonucunu temsil eder."""

    report_id: str | UUID | None
    status: PdfDownloadStatus

    source_url: str
    resolved_url: str

    local_path: Path

    file_size_bytes: int
    sha256: str

    content_type: str | None
    elapsed_seconds: float

    warnings: tuple[str, ...] = ()

    @property
    def file_size_mb(self) -> float:
        """Dosya boyutunu megabayt olarak döndürür."""

        return self.file_size_bytes / (1024 * 1024)

    # IngestionPipeline şu alan adlarından birini arıyor:
    # path, file_path veya pdf_path. Üçünü de local_path'e yönlendiriyoruz.
    @property
    def path(self) -> Path:
        return self.local_path

    @property
    def file_path(self) -> Path:
        return self.local_path

    @property
    def pdf_path(self) -> Path:
        return self.local_path

    def to_metadata(self) -> dict[str, object]:
        """Sonucu Report.metadata içerisinde saklanabilir sözlüğe dönüştürür."""

        return {
            "report_id": (
                str(self.report_id)
                if self.report_id is not None
                else None
            ),
            "status": self.status.value,
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "local_path": str(self.local_path),
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": self.file_size_mb,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _DownloadedPayload:
    """Tek HTTP indirme girişiminin dahili sonucudur."""

    resolved_url: str
    content_type: str | None
    file_size_bytes: int
    sha256: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ExistingPdf:
    """Diskte mevcut olan geçerli bir PDF'nin bilgileri."""

    file_size_bytes: int
    sha256: str


WINDOWS_RESERVED_FILENAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}


def normalize_filename_text(value: str) -> str:
    """Metni güvenli bir dosya adı parçasına dönüştürür."""

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
        }
    )

    translated_value = value.translate(translation_table)

    normalized_unicode = unicodedata.normalize(
        "NFKD",
        translated_value,
    )

    without_accents = "".join(
        character
        for character in normalized_unicode
        if not unicodedata.combining(character)
    )

    lowercase_value = without_accents.casefold()

    safe_value = re.sub(
        r"[^a-z0-9]+",
        "-",
        lowercase_value,
    ).strip("-.")

    if not safe_value:
        safe_value = "report"

    if safe_value in WINDOWS_RESERVED_FILENAMES:
        safe_value = f"report-{safe_value}"

    return safe_value[:100]


def build_report_filename(report: Report) -> str:
    """Report nesnesi için tutarlı ve güvenli bir dosya adı oluşturur."""

    company_slug = normalize_filename_text(report.company_name)

    return (
        f"{company_slug}-"
        f"{report.report_year}-"
        "sustainability-report.pdf"
    )


def build_url_filename(url: str) -> str:
    """URL'den güvenli bir PDF dosya adı üretir."""

    parsed_url = urlsplit(url)
    raw_name = Path(unquote(parsed_url.path)).name

    if raw_name.casefold().endswith(".pdf"):
        stem = Path(raw_name).stem
    else:
        stem = "downloaded-report"

    return f"{normalize_filename_text(stem)}.pdf"


def has_pdf_signature(content: bytes) -> bool:
    """Verilen başlangıç baytlarında PDF magic-byte imzasını arar."""

    return PDF_MAGIC_SIGNATURE in content[:PDF_SIGNATURE_SEARCH_LIMIT]


class PdfDownloader:
    """
    Doğrulanmış sürdürülebilirlik raporlarını güvenli biçimde indirir.

    Hem IngestionPipeline tarafından gönderilen URL metnini hem de eski
    kullanım biçimindeki Report nesnesini kabul eder.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        download_directory: str | Path | None = None,
        chunk_size_bytes: int = DEFAULT_DOWNLOAD_CHUNK_SIZE,
        maximum_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.settings = settings or get_settings()

        if chunk_size_bytes < 1024:
            raise ValueError(
                "chunk_size_bytes en az 1024 olmalıdır."
            )

        if maximum_attempts < 1:
            raise ValueError(
                "maximum_attempts en az 1 olmalıdır."
            )

        if retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds negatif olamaz."
            )

        self.chunk_size_bytes = chunk_size_bytes
        self.maximum_attempts = maximum_attempts
        self.retry_delay_seconds = retry_delay_seconds

        configured_download_directory = (
            download_directory
            if download_directory is not None
            else self.settings.downloads_directory
        )

        self.download_directory = Path(
            configured_download_directory
        )

        self.maximum_pdf_size_bytes = (
            self.settings.maximum_pdf_size_mb
            * 1024
            * 1024
        )

        self._owns_client = client is None

        self.client = client or httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(
                self.settings.http_timeout_seconds
            ),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(compatible; SustainabilityAnalyzer/1.0)"
                ),
                "Accept": (
                    "application/pdf,"
                    "application/octet-stream;q=0.9,"
                    "*/*;q=0.5"
                ),
            },
        )

    def __enter__(self) -> PdfDownloader:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Downloader kendi HTTP istemcisini oluşturduysa istemciyi kapatır."""

        if self._owns_client:
            self.client.close()

    def download(
        self,
        url: str | Report | None = None,
        *,
        filename: str | None = None,
        destination_path: str | Path | None = None,
        overwrite: bool = False,
        report: Report | None = None,
    ) -> PdfDownloadResult:
        """
        PDF'yi indirir.

        Desteklenen kullanım biçimleri:

            downloader.download(
                "https://example.com/report.pdf",
                filename="company_2023.pdf",
            )

            downloader.download(
                report=my_report,
                destination_path="data/downloads/report.pdf",
            )
        """

        report_object, source_url = self._resolve_download_input(
            url=url,
            report=report,
        )

        is_safe, safety_error = is_public_http_url(source_url)

        if not is_safe:
            raise PdfDownloadError(
                safety_error
                or "Rapor URL'si güvenlik kontrolünü geçemedi."
            )

        final_path = self._resolve_destination_path(
            source_url=source_url,
            report=report_object,
            filename=filename,
            destination_path=destination_path,
        )

        final_path = final_path.expanduser().resolve()

        if final_path.suffix.casefold() != ".pdf":
            raise ValueError(
                "Hedef dosya yolu .pdf uzantısıyla bitmelidir."
            )

        final_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_id = (
            getattr(report_object, "id", None)
            if report_object is not None
            else final_path.stem
        )

        if final_path.exists() and not overwrite:
            existing_pdf = self._inspect_existing_pdf(final_path)

            if existing_pdf is not None:
                result = PdfDownloadResult(
                    report_id=report_id,
                    status=PdfDownloadStatus.REUSED_EXISTING,
                    source_url=source_url,
                    resolved_url=source_url,
                    local_path=final_path,
                    file_size_bytes=existing_pdf.file_size_bytes,
                    sha256=existing_pdf.sha256,
                    content_type="application/pdf",
                    elapsed_seconds=0.0,
                    warnings=(
                        "Diskte bulunan geçerli PDF yeniden kullanıldı.",
                    ),
                )

                if report_object is not None:
                    self._apply_result_to_report(
                        report=report_object,
                        result=result,
                    )

                return result

            logger.warning(
                "Mevcut dosya geçerli PDF değil; yeniden indirilecek. "
                "Dosya=%s",
                final_path,
            )

        started_at = time.perf_counter()
        last_error: Exception | None = None

        for attempt_number in range(
            1,
            self.maximum_attempts + 1,
        ):
            temporary_path = self._create_temporary_path(final_path)

            try:
                logger.info(
                    "PDF indiriliyor. URL=%s, deneme=%s/%s",
                    source_url,
                    attempt_number,
                    self.maximum_attempts,
                )

                payload = self._download_once(
                    source_url=source_url,
                    temporary_path=temporary_path,
                )

                os.replace(
                    temporary_path,
                    final_path,
                )

                elapsed_seconds = (
                    time.perf_counter()
                    - started_at
                )

                result = PdfDownloadResult(
                    report_id=report_id,
                    status=PdfDownloadStatus.DOWNLOADED,
                    source_url=source_url,
                    resolved_url=payload.resolved_url,
                    local_path=final_path,
                    file_size_bytes=payload.file_size_bytes,
                    sha256=payload.sha256,
                    content_type=payload.content_type,
                    elapsed_seconds=elapsed_seconds,
                    warnings=payload.warnings,
                )

                if report_object is not None:
                    self._apply_result_to_report(
                        report=report_object,
                        result=result,
                    )

                logger.info(
                    "PDF başarıyla indirildi. Dosya=%s, boyut=%.2f MB",
                    final_path,
                    result.file_size_mb,
                )

                return result

            except httpx.HTTPStatusError as error:
                last_error = error
                self._remove_file_if_exists(temporary_path)

                if (
                    attempt_number < self.maximum_attempts
                    and self._is_retryable_status(
                        error.response.status_code
                    )
                ):
                    self._wait_before_retry(attempt_number)
                    continue

                raise PdfDownloadError(
                    "PDF sunucusu başarısız HTTP durumu döndürdü. "
                    f"Durum: {error.response.status_code}, "
                    f"URL: {error.request.url}"
                ) from error

            except httpx.TransportError as error:
                last_error = error
                self._remove_file_if_exists(temporary_path)

                if attempt_number < self.maximum_attempts:
                    self._wait_before_retry(attempt_number)
                    continue

                raise PdfDownloadError(
                    "PDF indirilirken ağ bağlantısı hatası oluştu. "
                    f"URL: {source_url}"
                ) from error

            except PdfDownloadError:
                self._remove_file_if_exists(temporary_path)
                raise

            except OSError as error:
                self._remove_file_if_exists(temporary_path)

                raise PdfDownloadError(
                    "PDF diske yazılırken dosya sistemi hatası oluştu. "
                    f"Hedef: {final_path}"
                ) from error

        raise PdfDownloadError(
            "PDF belirlenen deneme sayısı içerisinde indirilemedi."
        ) from last_error

    @staticmethod
    def _resolve_download_input(
        *,
        url: str | Report | None,
        report: Report | None,
    ) -> tuple[Report | None, str]:
        """URL/Report girişlerini tek bir source_url değerine dönüştürür."""

        report_object = report

        if isinstance(url, Report):
            if report_object is not None and report_object is not url:
                raise ValueError(
                    "Aynı anda iki farklı Report nesnesi verilemez."
                )

            report_object = url
            url = None

        if report_object is not None:
            source_url = str(report_object.source_url).strip()
        elif isinstance(url, str):
            source_url = url.strip()
        else:
            raise TypeError(
                "download() bir URL metni veya Report nesnesi bekler."
            )

        if not source_url:
            raise ValueError(
                "PDF indirme URL'si boş olamaz."
            )

        return report_object, source_url

    def _resolve_destination_path(
        self,
        *,
        source_url: str,
        report: Report | None,
        filename: str | None,
        destination_path: str | Path | None,
    ) -> Path:
        """Nihai PDF dosya yolunu belirler."""

        if destination_path is not None:
            return Path(destination_path)

        if filename is not None:
            safe_filename = Path(filename).name.strip()

            if not safe_filename:
                raise ValueError(
                    "filename boş olamaz."
                )

            if not safe_filename.casefold().endswith(".pdf"):
                safe_filename = f"{safe_filename}.pdf"

            return self.download_directory / safe_filename

        if report is not None:
            return self.build_destination_path(report)

        return (
            self.download_directory
            / build_url_filename(source_url)
        )

    def build_destination_path(self, report: Report) -> Path:
        """Report nesnesi için varsayılan indirme yolunu oluşturur."""

        return (
            self.download_directory
            / build_report_filename(report)
        )

    def _download_once(
        self,
        source_url: str,
        temporary_path: Path,
    ) -> _DownloadedPayload:
        """PDF'yi tek HTTP girişiminde geçici dosyaya indirir."""

        with self.client.stream(
            method="GET",
            url=source_url,
        ) as response:
            response.raise_for_status()

            resolved_url = str(response.url)

            is_safe, safety_error = is_public_http_url(
                resolved_url
            )

            if not is_safe:
                raise PdfDownloadError(
                    "HTTP yönlendirmesi güvenli olmayan bir hedefe ulaştı. "
                    f"Neden: {safety_error}"
                )

            content_type = get_clean_content_type(
                response.headers.get("content-type")
            )

            declared_content_length = parse_content_length(
                response.headers.get("content-length")
            )

            if (
                declared_content_length is not None
                and declared_content_length
                > self.maximum_pdf_size_bytes
            ):
                raise PdfDownloadError(
                    "PDF sunucu tarafından bildirilen dosya boyutu "
                    "sınırını aşıyor. "
                    f"Boyut: "
                    f"{declared_content_length / (1024 * 1024):.2f} MB, "
                    f"Sınır: {self.settings.maximum_pdf_size_mb} MB"
                )

            hasher = hashlib.sha256()
            first_bytes = bytearray()
            total_written = 0

            with temporary_path.open(mode="wb") as temporary_file:
                for chunk in response.iter_bytes(
                    chunk_size=self.chunk_size_bytes
                ):
                    if not chunk:
                        continue

                    total_written += len(chunk)

                    if total_written > self.maximum_pdf_size_bytes:
                        raise PdfDownloadError(
                            "İndirilen PDF izin verilen maksimum "
                            "dosya boyutunu aştı. "
                            f"Sınır: "
                            f"{self.settings.maximum_pdf_size_mb} MB"
                        )

                    self._capture_signature_bytes(
                        first_bytes=first_bytes,
                        chunk=chunk,
                    )

                    temporary_file.write(chunk)
                    hasher.update(chunk)

                temporary_file.flush()
                os.fsync(temporary_file.fileno())

        if total_written == 0:
            raise PdfDownloadError(
                "Sunucu boş bir dosya döndürdü."
            )

        if not has_pdf_signature(bytes(first_bytes)):
            raise PdfDownloadError(
                "İndirilen dosyanın başlangıcında geçerli "
                "PDF imzası bulunamadı."
            )

        warnings: list[str] = []

        if (
            content_type is not None
            and content_type not in PDF_CONTENT_TYPES
        ):
            warnings.append(
                "Sunucu PDF içeriğini standart olmayan bir "
                "Content-Type ile döndürdü; PDF imzası doğrulandı."
            )

        if (
            declared_content_length is not None
            and declared_content_length != total_written
        ):
            warnings.append(
                "Sunucunun bildirdiği Content-Length değeri ile "
                "indirilen bayt sayısı farklı."
            )

        return _DownloadedPayload(
            resolved_url=resolved_url,
            content_type=content_type,
            file_size_bytes=total_written,
            sha256=hasher.hexdigest(),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _capture_signature_bytes(
        first_bytes: bytearray,
        chunk: bytes,
    ) -> None:
        """PDF imza kontrolü için dosyanın ilk 1024 baytını saklar."""

        remaining_size = (
            PDF_SIGNATURE_SEARCH_LIMIT
            - len(first_bytes)
        )

        if remaining_size <= 0:
            return

        first_bytes.extend(chunk[:remaining_size])

    def _inspect_existing_pdf(
        self,
        path: Path,
    ) -> _ExistingPdf | None:
        """Mevcut dosyanın geçerli bir PDF olup olmadığını kontrol eder."""

        try:
            file_size = path.stat().st_size
        except OSError:
            return None

        if file_size <= 0:
            return None

        if file_size > self.maximum_pdf_size_bytes:
            return None

        try:
            with path.open(mode="rb") as file:
                signature_sample = file.read(
                    PDF_SIGNATURE_SEARCH_LIMIT
                )
        except OSError:
            return None

        if not has_pdf_signature(signature_sample):
            return None

        try:
            file_hash = self._calculate_file_sha256(path)
        except OSError:
            return None

        return _ExistingPdf(
            file_size_bytes=file_size,
            sha256=file_hash,
        )

    def _calculate_file_sha256(self, path: Path) -> str:
        """Diskteki dosyanın SHA-256 değerini hesaplar."""

        hasher = hashlib.sha256()

        with path.open(mode="rb") as file:
            self._update_hash_from_file(
                file=file,
                hasher=hasher,
            )

        return hasher.hexdigest()

    def _update_hash_from_file(
        self,
        file: BinaryIO,
        hasher: object,
    ) -> None:
        """Dosya içeriğini parça parça hash nesnesine ekler."""

        while True:
            chunk = file.read(self.chunk_size_bytes)

            if not chunk:
                break

            hasher.update(chunk)

    @staticmethod
    def _create_temporary_path(final_path: Path) -> Path:
        """Nihai dosyayla aynı klasörde benzersiz geçici dosya oluşturur."""

        temporary_file = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{final_path.stem}-",
            suffix=".part",
            dir=final_path.parent,
            delete=False,
        )

        temporary_path = Path(temporary_file.name)
        temporary_file.close()

        return temporary_path

    @staticmethod
    def _remove_file_if_exists(path: Path) -> None:
        """Geçici dosyayı hata üretmeden siler."""

        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            logger.warning(
                "Geçici dosya silinemedi. Dosya=%s, hata=%s",
                path,
                error,
            )

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        """Yeniden denenmesi anlamlı HTTP durumlarını belirler."""

        return (
            status_code == 408
            or status_code == 425
            or status_code == 429
            or 500 <= status_code <= 599
        )

    def _wait_before_retry(self, attempt_number: int) -> None:
        """Üstel artan bekleme süresi uygular."""

        wait_seconds = (
            self.retry_delay_seconds
            * (2 ** (attempt_number - 1))
        )

        logger.warning(
            "İndirme yeniden denenecek. Bekleme: %.1f saniye",
            wait_seconds,
        )

        time.sleep(wait_seconds)

    @staticmethod
    def _apply_result_to_report(
        report: Report,
        result: PdfDownloadResult,
    ) -> None:
        """İndirme sonucunu Report nesnesine kaydeder."""

        report.mark_downloaded(
            local_pdf_path=result.local_path,
            file_size_bytes=result.file_size_bytes,
            sha256=result.sha256,
        )

        report.content_type = (
            result.content_type
            or "application/pdf"
        )

        report.metadata["download"] = result.to_metadata()


def print_download_result(result: PdfDownloadResult) -> None:
    """İndirme sonucunu konsola okunabilir biçimde yazdırır."""

    print("\nPDF işlemi başarıyla tamamlandı.\n")
    print(f"Durum: {result.status.value}")
    print(f"Kaynak URL: {result.source_url}")
    print(f"Çözümlenen URL: {result.resolved_url}")
    print(f"Dosya: {result.local_path}")
    print(f"Boyut: {result.file_size_mb:.2f} MB")
    print(f"SHA-256: {result.sha256}")
    print(
        "Content-Type: "
        f"{result.content_type or 'bilinmiyor'}"
    )
    print(f"Süre: {result.elapsed_seconds:.2f} saniye")

    if result.warnings:
        print("Uyarılar:")

        for warning in result.warnings:
            print(f"  - {warning}")


def main() -> None:
    """
    Dosyanın doğrudan test edilmesini sağlar.

    Örnek:

        python -m document_ingestion.pdf_downloader \
            "Arçelik" 2023 "https://example.com/report.pdf"
    """

    parser = argparse.ArgumentParser(
        description=(
            "Doğrudan bir URL üzerinden sürdürülebilirlik "
            "raporu PDF'sini indirir."
        )
    )

    parser.add_argument(
        "company_name",
        help="Raporun ait olduğu şirket adı.",
    )
    parser.add_argument(
        "report_year",
        type=int,
        help="Rapor yılı.",
    )
    parser.add_argument(
        "url",
        help="Doğrudan veya yönlendirmeli PDF URL'si.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Rapor başlığı.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Özel PDF çıktı yolu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut dosyayı yeniden indirir.",
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    report = Report(
        company_name=arguments.company_name,
        report_year=arguments.report_year,
        title=arguments.title,
        source_url=arguments.url,
    )

    with PdfDownloader() as downloader:
        result = downloader.download(
            report=report,
            destination_path=arguments.output,
            overwrite=arguments.overwrite,
        )

    print_download_result(result)
    print(f"\nReport durumu: {report.status}")


if __name__ == "__main__":
    main()


__all__ = [
    "PdfDownloadError",
    "PdfDownloadResult",
    "PdfDownloadStatus",
    "PdfDownloader",
    "build_report_filename",
    "build_url_filename",
    "has_pdf_signature",
    "normalize_filename_text",
]
