from __future__ import annotations

import argparse
import html
import ipaddress
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from html.parser import HTMLParser
from urllib.parse import (
    unquote,
    urldefrag,
    urljoin,
    urlsplit,
)

import httpx

from config import Settings, get_settings
from models import ReportCandidate
from .candidate_ranker import (
    CandidateRanker,
    candidate_is_summary_report,
    normalize_company_name,
    normalize_text,
)


logger = logging.getLogger(__name__)


PDF_MAGIC_SIGNATURE = b"%PDF-"

PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
}

HTML_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
}

ALLOWED_HTTP_STATUS_CODES = {
    200,
    206,
}

HEAD_FALLBACK_STATUS_CODES = {
    403,
    405,
}

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
}

DEFAULT_SAMPLE_SIZE_BYTES = 256 * 1024
DEFAULT_MAXIMUM_HTML_SIZE_BYTES = 5 * 1024 * 1024
DEFAULT_MAXIMUM_HTML_LINKS = 25


class UrlValidationStatus(str, Enum):
    """
    Bir URL doğrulama işleminin sonucunu belirtir.
    """

    VALID_PDF = "valid_pdf"
    HTML_WITH_PDF_LINK = "html_with_pdf_link"
    NOT_PDF = "not_pdf"
    INACCESSIBLE = "inaccessible"
    TOO_LARGE = "too_large"
    UNSAFE_URL = "unsafe_url"
    INVALID_URL = "invalid_url"
    REJECTED_SUMMARY = "rejected_summary"
    FAILED = "failed"


@dataclass
class UrlInspection:
    """
    Bir HTTP kaynağı üzerinde yapılan düşük seviyeli inceleme sonucu.

    Bu sınıf yalnızca url_validator.py içerisinde kullanılır.
    """

    original_url: str
    resolved_url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    content_length: int | None = None
    sample: bytes = b""
    headers: dict[str, str] = field(
        default_factory=dict
    )
    errors: list[str] = field(
        default_factory=list
    )
    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def is_http_success(self) -> bool:
        return self.status_code in ALLOWED_HTTP_STATUS_CODES

    @property
    def has_pdf_signature(self) -> bool:
        """
        PDF imzasını ilk 1024 bayt içerisinde arar.

        PDF standardında dosyanın başlangıcında %PDF- imzası bulunur.
        Bazı sistemler başlangıca küçük miktarda ek veri koyabildiği için
        yalnızca ilk beş bayt yerine ilk 1024 bayt kontrol edilir.
        """

        return PDF_MAGIC_SIGNATURE in self.sample[:1024]

    @property
    def looks_like_html(self) -> bool:
        if self.content_type in HTML_CONTENT_TYPES:
            return True

        beginning = self.sample[:2048].lstrip().lower()

        html_markers = (
            b"<!doctype html",
            b"<html",
            b"<head",
            b"<body",
        )

        return beginning.startswith(
            html_markers
        )


@dataclass
class UrlValidationResult:
    """
    Kullanıma hazır URL doğrulama sonucunu temsil eder.
    """

    original_url: str
    status: UrlValidationStatus
    is_valid: bool

    resolved_url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    content_length: int | None = None

    has_pdf_signature: bool = False
    discovered_from_html: bool = False
    source_page_url: str | None = None

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def file_size_mb(self) -> float | None:
        if self.content_length is None:
            return None

        return self.content_length / (
            1024 * 1024
        )

    def to_metadata(self) -> dict[str, object]:
        """
        Sonucu ReportCandidate.metadata içerisine yazılabilir sözlüğe
        dönüştürür.
        """

        return asdict(self)


class PdfLinkParser(HTMLParser):
    """
    HTML içerisindeki bağlantı benzeri öznitelikleri toplar.

    Bazı kurumsal siteler PDF adresini yalnızca ``href`` alanında değil,
    ``data-url``, ``data-href`` veya ``data-download`` gibi alanlarda
    saklayabilir. Bu nedenle yalnızca ``<a href>`` etiketleriyle sınırlı
    kalınmaz.
    """

    LINK_ATTRIBUTE_NAMES = {
        "href",
        "src",
        "data-href",
        "data-url",
        "data-download",
        "data-file",
    }

    def __init__(self) -> None:
        super().__init__(
            convert_charrefs=True
        )

        self.links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag

        for name, value in attrs:
            if name.casefold() not in self.LINK_ATTRIBUTE_NAMES:
                continue

            if value is None:
                continue

            normalized_value = html.unescape(
                value
            ).strip()

            if normalized_value:
                self.links.append(
                    normalized_value
                )


def get_clean_content_type(
    value: str | None,
) -> str | None:
    """
    Content-Type başlığındaki charset gibi ek bilgileri kaldırır.

    Örnek:

        application/pdf; charset=utf-8
        -> application/pdf
    """

    if value is None:
        return None

    content_type = value.split(
        ";",
        maxsplit=1,
    )[0].strip().casefold()

    if not content_type:
        return None

    return content_type


def parse_content_length(
    value: str | None,
) -> int | None:
    """
    Content-Length başlığını güvenli biçimde tam sayıya dönüştürür.
    """

    if value is None:
        return None

    try:
        content_length = int(
            value.strip()
        )
    except (TypeError, ValueError):
        return None

    if content_length < 0:
        return None

    return content_length


def get_domain(
    url: str,
) -> str:
    """
    URL içerisinden sade alan adını çıkarır.
    """

    domain = urlsplit(
        url
    ).netloc.casefold()

    if domain.startswith("www."):
        domain = domain[4:]

    if ":" in domain:
        domain = domain.split(
            ":",
            maxsplit=1,
        )[0]

    return domain


def is_probable_pdf_url(
    url: str,
) -> bool:
    """
    URL metninden bağlantının PDF olma ihtimalini kontrol eder.
    """

    parsed_url = urlsplit(
        url
    )

    decoded_text = unquote(
        f"{parsed_url.path} {parsed_url.query}"
    ).casefold()

    return (
        decoded_text.endswith(".pdf")
        or ".pdf?" in decoded_text
        or ".pdf&" in decoded_text
        or "format=pdf" in decoded_text
        or "filetype=pdf" in decoded_text
        or "download=pdf" in decoded_text
        or "/pdf/" in decoded_text
    )

def is_probable_report_download_url(
    url: str,
) -> bool:
    """
    Doğrudan PDF veya PDF'ye yönlenmesi muhtemel rapor
    indirme bağlantılarını tespit eder.

    URL'nin mutlaka .pdf ile bitmesi gerekmez.
    """

    decoded_url = unquote(
        url,
    ).casefold()

    if is_probable_pdf_url(url):
        return True

    report_link_terms = (
        "download",
        "sustainability",
        "surdurulebilirlik",
        "report",
        "rapor",
        "document",
        "documentfile",
        "attachment",
        "media",
        "file",
        "content",
    )

    return any(
        term in decoded_url
        for term in report_link_terms
    )

def detect_link_years(
    url: str,
    *,
    target_year: int,
) -> set[int]:
    """URL dosya adındaki dört ve iki basamaklı rapor yıllarını bulur."""

    decoded = unquote(url)
    years = {
        int(value)
        for value in re.findall(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            decoded,
        )
    }

    filename = urlsplit(decoded).path.rsplit("/", 1)[-1]
    century = (target_year // 100) * 100

    for value in re.findall(
        r"(?:^|[_\-\s])(\d{2})(?=[_\-\s.]|$)",
        filename,
    ):
        mapped = century + int(value)
        if mapped - target_year > 50:
            mapped -= 100
        elif target_year - mapped > 50:
            mapped += 100

        if 1990 <= mapped <= 2100:
            years.add(mapped)

    return years


def is_summary_report_url(url: str) -> bool:
    """
    URL'nin yönetici özeti/özet rapor bağlantısı olup olmadığını belirler.

    Sabit kalıp yerine token tabanlı kontrol de yapılır. Bu sayede URL'deki
    yazım hataları ve ayraç varyasyonları yakalanır.
    """

    normalized = normalize_text(unquote(url))
    tokens = set(normalized.split())

    if any(
        token.startswith("ozet")
        or token in {"summary", "summaries"}
        for token in tokens
    ):
        return True

    compact = normalized.replace(" ", "")

    return any(
        marker in compact
        for marker in (
            "executivesummary",
            "managementsummary",
            "summaryreport",
            "reportsummary",
            "sustainabilitysummary",
            "yoneticiozeti",
            "yonecticiozeti",
            "ozetrapor",
            "raporozeti",
            "surdurulebilirlikozeti",
        )
    )


def is_public_http_url(
    url: str,
) -> tuple[bool, str | None]:
    """
    URL'nin HTTP/HTTPS kullandığını ve açıkça yerel bir hedefe
    yönelmediğini kontrol eder.

    Bu kontrol, literal özel IP adreslerini ve localhost adreslerini
    engeller.
    """

    try:
        parsed_url = urlsplit(
            url
        )
    except ValueError:
        return False, "URL ayrıştırılamadı."

    if parsed_url.scheme.casefold() not in {
        "http",
        "https",
    }:
        return (
            False,
            "URL yalnızca HTTP veya HTTPS protokolü kullanmalıdır.",
        )

    hostname = parsed_url.hostname

    if hostname is None:
        return (
            False,
            "URL geçerli bir alan adı içermiyor.",
        )

    normalized_hostname = hostname.casefold()

    if normalized_hostname in BLOCKED_HOSTNAMES:
        return (
            False,
            "Yerel makine adreslerine istek gönderilemez.",
        )

    try:
        ip_address = ipaddress.ip_address(
            normalized_hostname
        )
    except ValueError:
        return True, None

    if (
        ip_address.is_private
        or ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_multicast
        or ip_address.is_reserved
        or ip_address.is_unspecified
    ):
        return (
            False,
            "Özel veya yerel IP adreslerine istek gönderilemez.",
        )

    return True, None


class UrlValidator:
    """
    Rapor adaylarının gerçekten erişilebilir PDF olup olmadığını doğrular.

    Doğrulama sırası:

    1. URL güvenlik ve biçim kontrolü
    2. HEAD isteği
    3. Dosya boyutu kontrolü
    4. Kısmi GET isteği
    5. PDF magic-byte kontrolü
    6. Gerekirse HTML içerisinden doğrudan PDF bağlantısı çıkarımı
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        maximum_attempts: int = 2,
        retry_delay_seconds: float = 1.0,
        sample_size_bytes: int = DEFAULT_SAMPLE_SIZE_BYTES,
        maximum_html_size_bytes: int = DEFAULT_MAXIMUM_HTML_SIZE_BYTES,
        maximum_html_links: int = DEFAULT_MAXIMUM_HTML_LINKS,
    ) -> None:
        self.settings = settings or get_settings()

        if maximum_attempts < 1:
            raise ValueError(
                "maximum_attempts en az 1 olmalıdır."
            )

        if retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds negatif olamaz."
            )

        if sample_size_bytes < 1024:
            raise ValueError(
                "sample_size_bytes en az 1024 olmalıdır."
            )

        if maximum_html_size_bytes < sample_size_bytes:
            raise ValueError(
                "maximum_html_size_bytes, sample_size_bytes değerinden "
                "küçük olamaz."
            )

        if maximum_html_links < 1:
            raise ValueError(
                "maximum_html_links en az 1 olmalıdır."
            )

        self.maximum_attempts = maximum_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.sample_size_bytes = sample_size_bytes
        self.maximum_html_size_bytes = maximum_html_size_bytes
        self.maximum_html_links = maximum_html_links

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
                    "(compatible; SustainabilityAnalyzer/1.0; "
                    "+https://example.invalid)"
                ),
                "Accept": (
                    "application/pdf,"
                    "text/html;q=0.9,"
                    "*/*;q=0.8"
                ),
            },
        )

    def __enter__(
        self,
    ) -> UrlValidator:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(self) -> None:
        """
        Sınıf kendi HTTP istemcisini oluşturduysa istemciyi kapatır.
        """

        if self._owns_client:
            self.client.close()

    def validate(
        self,
        candidate: ReportCandidate,
        *,
        discover_pdf_from_html: bool = True,
    ) -> UrlValidationResult:
        """
        Tek bir ReportCandidate nesnesini doğrular.

        Başarılı doğrulamada:

        - candidate.is_validated True yapılır.
        - Doğrudan PDF URL'si candidate.url alanına yazılır.
        - Eski URL metadata içerisinde korunur.
        - HTTP ve PDF bilgileri metadata içerisine eklenir.
        """

        original_url = candidate.url

        if (
            candidate_is_summary_report(candidate)
            or is_summary_report_url(original_url)
            or (
                candidate.title
                and is_summary_report_url(candidate.title)
            )
        ):
            result = UrlValidationResult(
                original_url=original_url,
                status=UrlValidationStatus.REJECTED_SUMMARY,
                is_valid=False,
                resolved_url=original_url,
                errors=[
                    "Kaynak tam sürdürülebilirlik raporu yerine yönetici "
                    "özeti veya özet rapor olduğu için reddedildi."
                ],
            )
            self._apply_result_to_candidate(
                candidate=candidate,
                result=result,
            )
            return result

        inspection = self._inspect_url(
            original_url
        )

        direct_result = self._inspection_to_result(
            inspection=inspection,
            original_url=original_url,
        )

        if direct_result.is_valid:
            self._apply_result_to_candidate(
                candidate=candidate,
                result=direct_result,
            )

            return direct_result

        if (
            discover_pdf_from_html
            and inspection.looks_like_html
            and inspection.is_http_success
        ):
            html_result = self._discover_pdf_from_html(
                inspection=inspection,
                candidate=candidate,
            )

            if html_result is not None:
                self._apply_result_to_candidate(
                    candidate=candidate,
                    result=html_result,
                )

                return html_result

        self._apply_result_to_candidate(
            candidate=candidate,
            result=direct_result,
        )

        logger.info(
            "URL doğrulanamadı. url=%s, durum=%s, hatalar=%s",
            original_url,
            direct_result.status.value,
            direct_result.errors,
        )

        return direct_result

    def validate_candidates(
        self,
        candidates: list[ReportCandidate],
        *,
        include_invalid: bool = False,
        stop_after_first_valid: bool = False,
        discover_pdf_from_html: bool = True,
    ) -> list[ReportCandidate]:
        """
        Birden fazla rapor adayını sırasıyla doğrular.

        Adayların mevcut sıralaması korunur. Bu nedenle önce
        candidate_ranker.py tarafından sıralanmış liste verilmelidir.
        """

        validated_candidates: list[ReportCandidate] = []

        for candidate in candidates:
            result = self.validate(
                candidate,
                discover_pdf_from_html=discover_pdf_from_html,
            )

            if result.is_valid or include_invalid:
                validated_candidates.append(
                    candidate
                )

            if (
                result.is_valid
                and stop_after_first_valid
            ):
                break

        return validated_candidates

    def _inspect_url(
        self,
        url: str,
    ) -> UrlInspection:
        """
        URL üzerinde HEAD ve kısmi GET isteği gerçekleştirir.
        """

        inspection = UrlInspection(
            original_url=url
        )

        is_safe, safety_error = is_public_http_url(
            url
        )

        if not is_safe:
            inspection.errors.append(
                safety_error
                or "URL güvenlik kontrolünü geçemedi."
            )

            return inspection

        head_response = self._request_head(
            url
        )

        if head_response is not None:
            inspection.status_code = (
                head_response.status_code
            )

            inspection.resolved_url = str(
                head_response.url
            )

            inspection.headers = {
                key.casefold(): value
                for key, value in head_response.headers.items()
            }

            inspection.content_type = get_clean_content_type(
                head_response.headers.get(
                    "content-type"
                )
            )

            inspection.content_length = parse_content_length(
                head_response.headers.get(
                    "content-length"
                )
            )

            if (
                inspection.content_length is not None
                and inspection.content_length
                > self.maximum_pdf_size_bytes
            ):
                inspection.errors.append(
                    "Kaynak izin verilen maksimum PDF "
                    "boyutundan daha büyük."
                )

                return inspection

            if (
                head_response.status_code
                not in ALLOWED_HTTP_STATUS_CODES
                and head_response.status_code
                not in HEAD_FALLBACK_STATUS_CODES
            ):
                inspection.errors.append(
                    "HEAD isteği başarısız oldu. "
                    f"HTTP durumu: {head_response.status_code}"
                )

                return inspection

            if (
                head_response.status_code
                in HEAD_FALLBACK_STATUS_CODES
            ):
                inspection.warnings.append(
                    "Sunucu HEAD isteğine izin vermedi; "
                    "GET isteğiyle doğrulamaya devam edildi."
                )

        get_response_data = self._request_sample(
            url
        )

        if get_response_data is None:
            inspection.errors.append(
                "Kaynağın başlangıç baytları alınamadı."
            )

            return inspection

        (
            get_status_code,
            get_resolved_url,
            get_headers,
            sample,
        ) = get_response_data

        inspection.status_code = get_status_code
        inspection.resolved_url = get_resolved_url
        inspection.sample = sample

        inspection.headers.update(
            {
                key.casefold(): value
                for key, value in get_headers.items()
            }
        )

        get_content_type = get_clean_content_type(
            get_headers.get(
                "content-type"
            )
        )

        if get_content_type is not None:
            inspection.content_type = (
                get_content_type
            )

        get_content_length = parse_content_length(
            get_headers.get(
                "content-length"
            )
        )

        if get_content_length is not None:
            inspection.content_length = (
                get_content_length
            )

        if (
            inspection.content_length is not None
            and inspection.content_length
            > self.maximum_pdf_size_bytes
        ):
            inspection.errors.append(
                "Kaynak izin verilen maksimum PDF "
                "boyutundan daha büyük."
            )

            return inspection

        if get_status_code not in ALLOWED_HTTP_STATUS_CODES:
            inspection.errors.append(
                "GET isteği başarısız oldu. "
                f"HTTP durumu: {get_status_code}"
            )

        return inspection

    def _inspection_to_result(
        self,
        inspection: UrlInspection,
        original_url: str,
    ) -> UrlValidationResult:
        """
        Düşük seviyeli inceleme sonucunu nihai doğrulama sonucuna çevirir.
        """

        if inspection.errors:
            status = UrlValidationStatus.FAILED

            if any(
                "güvenlik" in error.casefold()
                or "özel veya yerel" in error.casefold()
                or "yerel makine" in error.casefold()
                for error in inspection.errors
            ):
                status = UrlValidationStatus.UNSAFE_URL

            elif any(
                "maksimum pdf boyutundan" in error.casefold()
                for error in inspection.errors
            ):
                status = UrlValidationStatus.TOO_LARGE

            elif inspection.status_code is not None:
                status = UrlValidationStatus.INACCESSIBLE

            else:
                status = UrlValidationStatus.INVALID_URL

            return UrlValidationResult(
                original_url=original_url,
                status=status,
                is_valid=False,
                resolved_url=inspection.resolved_url,
                status_code=inspection.status_code,
                content_type=inspection.content_type,
                content_length=inspection.content_length,
                has_pdf_signature=inspection.has_pdf_signature,
                errors=list(
                    inspection.errors
                ),
                warnings=list(
                    inspection.warnings
                ),
            )

        if inspection.has_pdf_signature:
            warnings = list(
                inspection.warnings
            )

            if (
                inspection.content_type is not None
                and inspection.content_type
                not in PDF_CONTENT_TYPES
            ):
                warnings.append(
                    "Sunucu PDF dosyasını farklı bir Content-Type "
                    "değeriyle döndürdü; magic-byte kontrolü başarılı."
                )

            return UrlValidationResult(
                original_url=original_url,
                status=UrlValidationStatus.VALID_PDF,
                is_valid=True,
                resolved_url=(
                    inspection.resolved_url
                    or original_url
                ),
                status_code=inspection.status_code,
                content_type=inspection.content_type,
                content_length=inspection.content_length,
                has_pdf_signature=True,
                warnings=warnings,
            )

        if inspection.looks_like_html:
            return UrlValidationResult(
                original_url=original_url,
                status=UrlValidationStatus.NOT_PDF,
                is_valid=False,
                resolved_url=inspection.resolved_url,
                status_code=inspection.status_code,
                content_type=inspection.content_type,
                content_length=inspection.content_length,
                has_pdf_signature=False,
                errors=[
                    "URL doğrudan PDF yerine HTML sayfası döndürdü."
                ],
                warnings=list(
                    inspection.warnings
                ),
            )

        if inspection.content_type in PDF_CONTENT_TYPES:
            return UrlValidationResult(
                original_url=original_url,
                status=UrlValidationStatus.NOT_PDF,
                is_valid=False,
                resolved_url=inspection.resolved_url,
                status_code=inspection.status_code,
                content_type=inspection.content_type,
                content_length=inspection.content_length,
                has_pdf_signature=False,
                errors=[
                    "Content-Type PDF olmasına rağmen "
                    "dosya başlangıcında PDF imzası bulunamadı."
                ],
                warnings=list(
                    inspection.warnings
                ),
            )

        return UrlValidationResult(
            original_url=original_url,
            status=UrlValidationStatus.NOT_PDF,
            is_valid=False,
            resolved_url=inspection.resolved_url,
            status_code=inspection.status_code,
            content_type=inspection.content_type,
            content_length=inspection.content_length,
            has_pdf_signature=False,
            errors=[
                "Kaynağın PDF olduğu doğrulanamadı."
            ],
            warnings=list(
                inspection.warnings
            ),
        )

    def _discover_pdf_from_html(
        self,
        inspection: UrlInspection,
        candidate: ReportCandidate,
    ) -> UrlValidationResult | None:
        """
        HTML sayfasındaki olası PDF bağlantılarını çıkarır ve doğrular.
        """

        base_url = (
            inspection.resolved_url
            or inspection.original_url
        )

        links = self._extract_pdf_links(
            html_content=inspection.sample,
            base_url=base_url,
        )

        if not links:
            return None

        ranked_links = self._rank_pdf_links(
            links=links,
            candidate=candidate,
            source_page_url=base_url,
        )

        for link in ranked_links[
            : self.maximum_html_links
        ]:
            if is_summary_report_url(link):
                logger.info(
                    "HTML içindeki özet rapor bağlantısı atlandı: %s",
                    link,
                )
                continue

            detected_link_years = detect_link_years(
                link,
                target_year=candidate.report_year,
            )
            if (
                detected_link_years
                and candidate.report_year not in detected_link_years
            ):
                logger.info(
                    "HTML içindeki farklı yıl raporu atlandı. "
                    "hedef=%s, bulunan=%s, url=%s",
                    candidate.report_year,
                    sorted(detected_link_years),
                    link,
                )
                continue

            linked_inspection = self._inspect_url(
                link
            )

            linked_result = self._inspection_to_result(
                inspection=linked_inspection,
                original_url=link,
            )

            if not linked_result.is_valid:
                continue

            linked_result.status = (
                UrlValidationStatus.HTML_WITH_PDF_LINK
            )

            linked_result.discovered_from_html = True
            linked_result.source_page_url = base_url

            linked_result.warnings.insert(
                0,
                "Doğrudan PDF bağlantısı aday HTML sayfası "
                "içerisinden çıkarıldı.",
            )

            return linked_result

        return None

    def _extract_pdf_links(
        self,
        html_content: bytes,
        base_url: str,
    ) -> list[str]:
        """
        HTML, JavaScript ve gömülü JSON içerisinden doğrudan PDF
        bağlantılarıyla birlikte PDF'ye yönlenebilecek rapor
        indirme bağlantılarını çıkarır.
        """

        try:
            decoded_html = html_content.decode(
                "utf-8",
            )
        except UnicodeDecodeError:
            decoded_html = html_content.decode(
                "latin-1",
                errors="ignore",
            )

        decoded_html = html.unescape(
            decoded_html,
        )

        escape_replacements = {
            r"\/": "/",
            r"\u002F": "/",
            r"\u002f": "/",
            r"\u003A": ":",
            r"\u003a": ":",
            r"\u003F": "?",
            r"\u003f": "?",
            r"\u0026": "&",
            r"\u003D": "=",
            r"\u003d": "=",
        }

        for escaped_value, replacement in (
            escape_replacements.items()
        ):
            decoded_html = decoded_html.replace(
                escaped_value,
                replacement,
            )

        raw_links: list[str] = []

        # Standart <a href="..."> bağlantıları.
        parser = PdfLinkParser()

        try:
            parser.feed(
                decoded_html,
            )

            raw_links.extend(
                parser.links,
            )

        except Exception as error:
            logger.warning(
                "HTML bağlantıları ayrıştırılamadı: %s",
                error,
            )

        # href, data-url, onclick benzeri özniteliklerdeki
        # bütün URL değerlerini yakalar.
        attribute_pattern = re.compile(
            r"""
            (?:
                href
                |src
                |data-url
                |data-href
                |data-download
                |data-file
                |data-src
                |data-link
                |onclick
            )
            \s*=\s*
            ["']
            (?P<value>[^"']+)
            ["']
            """,
            flags=(
                re.IGNORECASE
                | re.VERBOSE
            ),
        )

        for match in attribute_pattern.finditer(
            decoded_html,
        ):
            raw_links.append(
                match.group("value"),
            )

        # Doğrudan PDF URL’leri.
        pdf_pattern = re.compile(
            r"""
            (?P<url>
                (?:
                    https?://
                    |
                    //
                    |
                    /
                    |
                    \.\./
                    |
                    \./
                )
                [^\s"'<>]+?
                \.pdf
                (?:
                    \?[^\s"'<>]*
                )?
            )
            """,
            flags=(
                re.IGNORECASE
                | re.VERBOSE
            ),
        )

        for match in pdf_pattern.finditer(
            decoded_html,
        ):
            raw_links.append(
                match.group("url"),
            )

        # .pdf içermeyen download/report/media bağlantıları.
        general_url_pattern = re.compile(
            r"""
            (?P<url>
                (?:
                    https?://
                    |
                    //
                    |
                    /
                    |
                    \.\./
                    |
                    \./
                )
                [^\s"'<>]+
            )
            """,
            flags=(
                re.IGNORECASE
                | re.VERBOSE
            ),
        )

        for match in general_url_pattern.finditer(
            decoded_html,
        ):
            possible_url = match.group(
                "url",
            )

            if is_probable_report_download_url(
                possible_url,
            ):
                raw_links.append(
                    possible_url,
                )

        resolved_links: list[str] = []
        seen_links: set[str] = set()

        for raw_link in raw_links:
            normalized_link = html.unescape(
                str(raw_link),
            ).strip()

            normalized_link = (
                normalized_link
                .replace(r"\/", "/")
                .replace("&amp;", "&")
            )

            # onclick="window.open('/media/...')" gibi
            # JavaScript ifadelerinden URL'yi ayır.
            javascript_url_match = re.search(
                r"""
                (?P<url>
                    (?:
                        https?://
                        |
                        //
                        |
                        /
                        |
                        \.\./
                        |
                        \./
                    )
                    [^"'()\s<>]+
                )
                """,
                normalized_link,
                flags=(
                    re.IGNORECASE
                    | re.VERBOSE
                ),
            )

            if javascript_url_match is not None:
                normalized_link = (
                    javascript_url_match.group(
                        "url",
                    )
                )

            normalized_link = normalized_link.rstrip(
                ".,;)]}",
            )

            if not normalized_link:
                continue

            lowered_link = normalized_link.casefold()

            if lowered_link.startswith(
                (
                    "javascript:",
                    "mailto:",
                    "tel:",
                    "data:",
                    "#",
                )
            ):
                continue

            absolute_link = urljoin(
                base_url,
                normalized_link,
            )

            absolute_link = urldefrag(
                absolute_link,
            )[0]

            if not is_probable_report_download_url(
                absolute_link,
            ):
                continue

            is_safe, _ = is_public_http_url(
                absolute_link,
            )

            if not is_safe:
                continue

            comparison_link = (
                absolute_link.casefold()
            )

            if comparison_link in seen_links:
                continue

            seen_links.add(
                comparison_link,
            )

            resolved_links.append(
                absolute_link,
            )

        logger.info(
            "%d muhtemel rapor indirme bağlantısı çıkarıldı. "
            "Kaynak=%s",
            len(resolved_links),
            base_url,
        )

        return resolved_links

    def _rank_pdf_links(
        self,
        links: list[str],
        candidate: ReportCandidate,
        source_page_url: str,
    ) -> list[str]:
        """
        HTML içerisinden çıkarılan rapor bağlantılarını hedef yıl,
        şirket adı, sürdürülebilirlik terimleri ve dosya türüne göre
        sıralar.
        """

        normalized_company = normalize_company_name(
            candidate.company_name
        )

        company_tokens = set(
            normalized_company.split()
        )

        target_year = str(
            candidate.report_year
        )

        source_domain = get_domain(
            source_page_url
        )

        sustainability_terms = {
            "sustainability",
            "sustainable",
            "surdurulebilirlik",
            "esg",
            "integrated",
            "entegre",
            "responsibility",
            "csr",
        }

        summary_terms = {
            "executive summary",
            "management summary",
            "summary report",
            "report summary",
            "yonetici ozeti",
            "yönetici özeti",
            "ozet rapor",
            "özet rapor",
            "rapor ozeti",
            "rapor özeti",
        }

        negative_terms = {
            "annual report",
            "financial report",
            "faaliyet raporu",
            "financial statement",
            "quarterly report",
            "interim report",
        }

        def score_link(
            link: str,
        ) -> float:
            normalized_link = normalize_text(
                unquote(link)
            )

            link_tokens = set(
                normalized_link.split()
            )

            score = 0.0

            # Doğrudan PDF bağlantısı en güçlü işaretlerden biridir.
            if is_probable_pdf_url(link):
                score += 6.0

            # Hedef yıl eşleşmesi güçlü biçimde önceliklendirilir.
            if target_year in normalized_link:
                score += 10.0

            company_overlap = len(
                company_tokens.intersection(
                    link_tokens
                )
            )

            score += company_overlap * 2.0

            sustainability_overlap = len(
                sustainability_terms.intersection(
                    link_tokens
                )
            )

            score += sustainability_overlap * 3.0

            if "download" in normalized_link:
                score += 1.5

            if "report" in normalized_link:
                score += 1.0

            if "rapor" in normalized_link:
                score += 1.0

            if get_domain(link) == source_domain:
                score += 1.5

            detected_years = detect_link_years(
                link,
                target_year=candidate.report_year,
            )

            # Bağlantı açıkça farklı bir yılı gösteriyorsa tamamen geriye at.
            if (
                detected_years
                and candidate.report_year
                not in detected_years
            ):
                score -= 1000.0

            if is_summary_report_url(link):
                score -= 1000.0

            if any(
                normalize_text(term) in normalized_link
                for term in negative_terms
            ):
                if not sustainability_overlap:
                    score -= 15.0

            return score

        # Bu return satırı score_link fonksiyonunun dışında olmalıdır.
        return sorted(
            links,
            key=score_link,
            reverse=True,
        )

    def _request_head(
        self,
        url: str,
    ) -> httpx.Response | None:
        """
        HEAD isteğini retry mekanizmasıyla çalıştırır.
        """

        last_error: Exception | None = None

        for attempt_number in range(
            1,
            self.maximum_attempts + 1,
        ):
            try:
                return self.client.head(
                    url
                )

            except httpx.HTTPError as error:
                last_error = error

                logger.warning(
                    "HEAD isteği başarısız oldu. "
                    "URL=%s, deneme=%s/%s, hata=%s",
                    url,
                    attempt_number,
                    self.maximum_attempts,
                    error,
                )

                if attempt_number < self.maximum_attempts:
                    self._wait_before_retry(
                        attempt_number
                    )

        if last_error is not None:
            logger.warning(
                "HEAD isteği tamamen başarısız oldu; "
                "GET isteği yine de denenecek. URL=%s",
                url,
            )

        return None

    def _request_sample(
        self,
        url: str,
    ) -> tuple[
        int,
        str,
        httpx.Headers,
        bytes,
    ] | None:
        """
        Kaynağın doğrulama için gerekli bölümünü indirir.

        PDF dosyalarında magic-byte kontrolü için küçük bir başlangıç örneği
        yeterlidir. HTML sayfalarında ise rapor bağlantıları sayfanın alt
        bölümünde bulunabildiğinden daha geniş bir içerik sınırı kullanılır.
        """

        last_error: Exception | None = None

        for attempt_number in range(
            1,
            self.maximum_attempts + 1,
        ):
            try:
                with self.client.stream(
                    method="GET",
                    url=url,
                ) as response:
                    response_content_type = get_clean_content_type(
                        response.headers.get(
                            "content-type"
                        )
                    )

                    if response_content_type in HTML_CONTENT_TYPES:
                        read_limit = self.maximum_html_size_bytes
                    else:
                        read_limit = self.sample_size_bytes

                    sample = bytearray()

                    for chunk in response.iter_bytes():
                        remaining_size = (
                            read_limit
                            - len(sample)
                        )

                        if remaining_size <= 0:
                            break

                        sample.extend(
                            chunk[:remaining_size]
                        )

                        if len(sample) >= read_limit:
                            break

                    return (
                        response.status_code,
                        str(response.url),
                        response.headers,
                        bytes(sample),
                    )

            except httpx.HTTPError as error:
                last_error = error

                logger.warning(
                    "GET örnek isteği başarısız oldu. "
                    "URL=%s, deneme=%s/%s, hata=%s",
                    url,
                    attempt_number,
                    self.maximum_attempts,
                    error,
                )

                if attempt_number < self.maximum_attempts:
                    self._wait_before_retry(
                        attempt_number
                    )

        if last_error is not None:
            logger.error(
                "URL örneği alınamadı. URL=%s, hata=%s",
                url,
                last_error,
            )

        return None

    def _wait_before_retry(
        self,
        attempt_number: int,
    ) -> None:
        """
        Üstel artan retry bekleme süresi uygular.
        """

        wait_seconds = (
            self.retry_delay_seconds
            * (2 ** (attempt_number - 1))
        )

        time.sleep(
            wait_seconds
        )

    def _apply_result_to_candidate(
        self,
        candidate: ReportCandidate,
        result: UrlValidationResult,
    ) -> None:
        """
        URL doğrulama sonucunu ReportCandidate nesnesine kaydeder.
        """

        candidate.is_validated = (
            result.is_valid
        )

        candidate.metadata[
            "url_validation"
        ] = result.to_metadata()

        notes = [
            *result.warnings,
            *result.errors,
        ]

        if result.is_valid:
            notes.insert(
                0,
                "PDF dosya imzası başarıyla doğrulandı.",
            )

        existing_notes = {
            note.casefold()
            for note in candidate.validation_notes
        }

        for note in notes:
            normalized_note = " ".join(
                note.strip().split()
            )

            if not normalized_note:
                continue

            if normalized_note.casefold() in existing_notes:
                continue

            candidate.validation_notes.append(
                normalized_note
            )

            existing_notes.add(
                normalized_note.casefold()
            )

        if (
            result.is_valid
            and result.resolved_url is not None
        ):
            if result.resolved_url != candidate.url:
                candidate.metadata.setdefault(
                    "original_candidate_url",
                    candidate.url,
                )

                candidate.url = (
                    result.resolved_url
                )

                candidate.source_domain = get_domain(
                    result.resolved_url
                )


def print_validation_results(
    candidates: list[ReportCandidate],
) -> None:
    """
    Doğrulanan adayları okunabilir biçimde yazdırır.
    """

    if not candidates:
        print(
            "Doğrulama sonucu gösterilecek aday bulunamadı."
        )

        return

    print(
        f"\nToplam {len(candidates)} aday doğrulandı.\n"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        validation_metadata = candidate.metadata.get(
            "url_validation",
            {},
        )

        status = validation_metadata.get(
            "status",
            "unknown",
        )

        is_valid = validation_metadata.get(
            "is_valid",
            False,
        )

        content_type = validation_metadata.get(
            "content_type",
        )

        content_length = validation_metadata.get(
            "content_length",
        )

        print(
            f"{index}. {candidate.title or 'Başlık bulunamadı'}"
        )

        print(
            f"   Final sıralama skoru: "
            f"{candidate.final_score:.4f}"
        )

        print(
            f"   Doğrulama durumu: {status}"
        )

        print(
            f"   Geçerli PDF: {is_valid}"
        )

        print(
            f"   URL: {candidate.url}"
        )

        print(
            f"   Content-Type: "
            f"{content_type or 'bilinmiyor'}"
        )

        if isinstance(
            content_length,
            int,
        ):
            file_size_mb = (
                content_length
                / (1024 * 1024)
            )

            print(
                f"   Tahmini dosya boyutu: "
                f"{file_size_mb:.2f} MB"
            )

        if candidate.validation_notes:
            print(
                "   Notlar:"
            )

            for note in candidate.validation_notes:
                print(
                    f"      - {note}"
                )

        print()


def main() -> None:
    """
    Exa araması, sıralama ve URL doğrulama adımlarını birlikte çalıştırır.

    Örnek:

        python -m report_discovery.url_validator "Arçelik" 2023
    """

    parser = argparse.ArgumentParser(
        description=(
            "Sürdürülebilirlik raporu adaylarını arar, "
            "sıralar ve PDF URL'lerini doğrular."
        )
    )

    parser.add_argument(
        "company_name",
        help="Raporu aranacak şirket adı.",
    )

    parser.add_argument(
        "report_year",
        type=int,
        help="Aranan rapor yılı.",
    )

    parser.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="Exa'dan alınacak maksimum benzersiz aday sayısı.",
    )

    parser.add_argument(
        "--validate-limit",
        type=int,
        default=5,
        help="Doğrulanacak en yüksek skorlu aday sayısı.",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help="Candidate ranker minimum kabul skoru.",
    )

    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Geçersiz adayları da konsol çıktısında gösterir.",
    )

    parser.add_argument(
        "--stop-after-first-valid",
        action="store_true",
        help="İlk geçerli PDF bulunduğunda doğrulamayı durdurur.",
    )

    parser.add_argument(
        "--disable-html-discovery",
        action="store_true",
        help=(
            "HTML sayfalarındaki doğrudan PDF bağlantılarını "
            "arama özelliğini kapatır."
        ),
    )

    arguments = parser.parse_args()

    if arguments.validate_limit < 1:
        parser.error(
            "--validate-limit en az 1 olmalıdır."
        )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    from report_discovery.exa_search import ExaReportSearcher

    searcher = ExaReportSearcher()

    candidates = searcher.search(
        company_name=arguments.company_name,
        report_year=arguments.report_year,
        max_candidates=arguments.search_limit,
        include_highlights=False,
    )

    ranker = CandidateRanker()

    ranked_candidates = ranker.rank_candidates(
        candidates,
        minimum_score=arguments.minimum_score,
        include_rejected=False,
    )

    candidates_to_validate = ranked_candidates[
        : arguments.validate_limit
    ]

    with UrlValidator() as validator:
        validated_candidates = validator.validate_candidates(
            candidates_to_validate,
            include_invalid=arguments.include_invalid,
            stop_after_first_valid=(
                arguments.stop_after_first_valid
            ),
            discover_pdf_from_html=(
                not arguments.disable_html_discovery
            ),
        )

    print_validation_results(
        validated_candidates
    )


if __name__ == "__main__":
    main()