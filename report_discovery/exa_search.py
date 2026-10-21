from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)

from exa_py import Exa
from pydantic import ValidationError

from config import Settings, get_settings
from models import ReportCandidate


logger = logging.getLogger(__name__)


class ExaSearchError(RuntimeError):
    """
    Exa araması tamamen başarısız olduğunda üretilen uygulama hatası.
    """


class ExaSearchResultProtocol(Protocol):
    """
    Exa sonucunda kullandığımız alanları tanımlayan protokol.

    Bu protokol sayesinde gerçek Exa istemcisinin yanı sıra testlerde
    sahte istemci de kullanılabilir.
    """

    url: str
    title: str | None
    score: float | None
    highlights: list[str] | None
    id: str | None
    published_date: str | None
    author: str | None
    crawl_date: str | None


class ExaSearchResponseProtocol(Protocol):
    """
    Exa arama cevabının bu modülde kullanılan kısmı.
    """

    results: Sequence[ExaSearchResultProtocol]


class ExaClientProtocol(Protocol):
    """
    Exa istemcisi için minimum arayüz.

    Dependency injection uygulanarak test sırasında gerçek API çağrısı
    yapmak zorunda kalmamamızı sağlar.
    """

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> ExaSearchResponseProtocol:
        ...


DEFAULT_QUERY_TEMPLATES: tuple[str, ...] = (
    # Doğrudan PDF bağlantılarını bulmaya yönelik sorgular.
    "{company_name} {report_year} sustainability report PDF",
    "{company_name} {report_year} ESG report PDF",
    "{company_name} {report_year} sürdürülebilirlik raporu PDF",
    "{company_name} {report_year} integrated report PDF",

    # PDF bağlantısını içeren kurumsal HTML/arşiv sayfalarını da bulur.
    "{company_name} {report_year} sustainability report",
    "{company_name} sustainability reports {report_year}",
    "{company_name} sustainability reports archive",
    "{company_name} sustainability report download {report_year}",
    "{company_name} {report_year} full sustainability report PDF",
    "{company_name} {report_year} complete sustainability report PDF",
    "{company_name} {report_year} tam sürdürülebilirlik raporu PDF",
)


TRACKING_QUERY_PARAMETERS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
}


def normalize_company_name(
    company_name: str,
) -> str:
    """
    Şirket adındaki gereksiz boşlukları temizler.
    """

    normalized_name = " ".join(
        company_name.strip().split()
    )

    if not normalized_name:
        raise ValueError(
            "Şirket adı boş olamaz."
        )

    return normalized_name


def validate_report_year(
    report_year: int,
) -> int:
    """
    Rapor yılının kabul edilen aralıkta olduğunu doğrular.
    """

    if report_year < 1990 or report_year > 2100:
        raise ValueError(
            "Rapor yılı 1990 ile 2100 arasında olmalıdır."
        )

    return report_year


def canonicalize_url(
    url: str,
) -> str:
    """
    URL'yi tekrar tespiti için standart bir biçime dönüştürür.

    Yapılan işlemler:

    - Şema ve alan adı küçük harfe çevrilir.
    - Fragment kısmı kaldırılır.
    - UTM ve benzeri takip parametreleri kaldırılır.
    - Kalan query parametreleri sıralanır.
    - Gereksiz sondaki eğik çizgi kaldırılır.

    Bu fonksiyon yalnızca tekrar tespiti amacıyla kullanılır.
    Asıl indirme URL'si değiştirilmeden ReportCandidate içerisinde saklanır.
    """

    normalized_url = url.strip()

    parsed_url = urlsplit(
        normalized_url
    )

    scheme = parsed_url.scheme.lower()
    netloc = parsed_url.netloc.lower()

    path = parsed_url.path

    if path != "/":
        path = path.rstrip("/")

    filtered_query_parameters: list[tuple[str, str]] = []

    for key, value in parse_qsl(
        parsed_url.query,
        keep_blank_values=True,
    ):
        normalized_key = key.lower()

        if normalized_key.startswith("utm_"):
            continue

        if normalized_key in TRACKING_QUERY_PARAMETERS:
            continue

        filtered_query_parameters.append(
            (
                key,
                value,
            )
        )

    filtered_query_parameters.sort()

    query = urlencode(
        filtered_query_parameters,
        doseq=True,
    )

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            query,
            "",
        )
    )


def safe_non_negative_float(
    value: Any,
) -> float | None:
    """
    Exa skorunu güvenli şekilde float değerine dönüştürür.

    None, negatif veya sonlu olmayan değerlerde None döndürür.
    """

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    if numeric_value < 0:
        return None

    return numeric_value


def build_snippet(
    highlights: Sequence[str] | None,
) -> str | None:
    """
    Exa highlight sonuçlarını tek bir kısa açıklamaya dönüştürür.
    """

    if not highlights:
        return None

    normalized_highlights: list[str] = []

    for highlight in highlights:
        normalized_highlight = " ".join(
            str(highlight).strip().split()
        )

        if normalized_highlight:
            normalized_highlights.append(
                normalized_highlight
            )

    if not normalized_highlights:
        return None

    return " ".join(
        normalized_highlights
    )


def is_probable_direct_pdf_url(
    url: str,
) -> bool:
    """
    URL metninden bağlantının doğrudan PDF olma ihtimalini belirler.

    Bu yalnızca arama sonucunu etiketlemek için kullanılır. HTML sayfaları
    elenmez; sonraki UrlValidator aşaması bu sayfalardaki PDF bağlantılarını
    keşfedebilir.
    """

    parsed_url = urlsplit(url)

    path = parsed_url.path.casefold()
    query = parsed_url.query.casefold()

    return (
        path.endswith(".pdf")
        or ".pdf" in path
        or "format=pdf" in query
        or "filetype=pdf" in query
        or "download=pdf" in query
    )


def get_candidate_resource_kind(
    url: str,
) -> str:
    """
    Arama sonucunu doğrudan PDF veya HTML/landing-page adayı olarak etiketler.
    """

    if is_probable_direct_pdf_url(url):
        return "direct_pdf_candidate"

    return "html_or_landing_page_candidate"


def candidate_search_sort_key(
    candidate: ReportCandidate,
) -> tuple[int, int, int, int, float]:
    """
    Exa sonuçlarını keşif aşamasında hafifçe önceliklendirir.

    Nihai uygunluk değerlendirmesi CandidateRanker tarafından yapılır. Burada
    amaç, düşük aday limitlerinde doğru yıl ve rapor terimleri içeren HTML veya
    PDF sonuçlarının listede kalma olasılığını artırmaktır.
    """

    searchable_text = " ".join(
        (
            candidate.title or "",
            candidate.url,
            candidate.snippet or "",
        )
    ).casefold()

    year_match = int(
        str(candidate.report_year) in searchable_text
    )

    report_term_match = int(
        any(
            term in searchable_text
            for term in (
                "sustainability",
                "sürdürülebilirlik",
                "esg",
                "integrated report",
                "entegre rapor",
            )
        )
    )

    summary_match = int(
        any(
            term in searchable_text
            for term in (
                "executive summary",
                "management summary",
                "summary report",
                "report summary",
                "yönetici özeti",
                "yonetici ozeti",
                "özet rapor",
                "ozet rapor",
                "rapor özeti",
                "rapor ozeti",
            )
        )
    )

    matched_queries = candidate.metadata.get(
        "matched_queries",
        [],
    )

    matched_query_count = (
        len(matched_queries)
        if isinstance(matched_queries, list)
        else 0
    )

    exa_score = (
        candidate.exa_score
        if candidate.exa_score is not None
        else -1.0
    )

    return (
        -summary_match,
        year_match,
        report_term_match,
        matched_query_count,
        exa_score,
    )


class ExaReportSearcher:
    """
    Şirket ve yıl bilgisine göre PDF ve HTML rapor adaylarını arar.

    Bu sınıf yalnızca keşif işlemini gerçekleştirir. Adayların gerçekten
    doğru rapor olup olmadığı sonraki modüllerde değerlendirilecektir:

    - candidate_ranker.py
    - url_validator.py
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: ExaClientProtocol | None = None,
        query_templates: Sequence[str] | None = None,
        maximum_attempts: int = 2,
        retry_delay_seconds: float = 1.0,
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

        self.maximum_attempts = maximum_attempts
        self.retry_delay_seconds = retry_delay_seconds

        self.query_templates = tuple(
            query_templates
            or DEFAULT_QUERY_TEMPLATES
        )

        if not self.query_templates:
            raise ValueError(
                "En az bir Exa sorgu şablonu tanımlanmalıdır."
            )

        if client is not None:
            self.client = client
        else:
            self.client = Exa(
                api_key=self.settings.require_exa_api_key()
            )

    def build_queries(
        self,
        company_name: str,
        report_year: int,
    ) -> list[str]:
        """
        Aynı raporu farklı terminolojilerle arayan sorgular üretir.

        Bir şirket raporu şu adlardan biriyle yayımlanmış olabilir:

        - Sustainability Report
        - ESG Report
        - Sürdürülebilirlik Raporu
        - Integrated Sustainability Report
        """

        normalized_company_name = normalize_company_name(
            company_name
        )

        validated_report_year = validate_report_year(
            report_year
        )

        queries: list[str] = []
        seen_queries: set[str] = set()

        for template in self.query_templates:
            query = template.format(
                company_name=normalized_company_name,
                report_year=validated_report_year,
            )

            normalized_query = " ".join(
                query.strip().split()
            )

            comparison_query = normalized_query.casefold()

            if comparison_query in seen_queries:
                continue

            seen_queries.add(
                comparison_query
            )

            queries.append(
                normalized_query
            )

        return queries

    def search(
        self,
        company_name: str,
        report_year: int,
        *,
        max_candidates: int | None = None,
        include_highlights: bool = False,
    ) -> list[ReportCandidate]:
        """
        Exa üzerinde rapor adaylarını arar.

        Args:
            company_name:
                Raporu aranacak şirket adı.

            report_year:
                Aranan rapor yılı.

            max_candidates:
                Tekrarlar kaldırıldıktan sonra döndürülecek maksimum
                aday sayısı. Verilmezse config.py içerisindeki
                exa_result_limit kullanılır.

            include_highlights:
                True olduğunda Exa'dan kısa içerik parçaları da istenir.
                Bu bilgi candidate ranker için yararlı olabilir ancak
                ek içerik çıkarımı maliyeti doğurabilir.

        Returns:
            PDF ve HTML sonuçlarını içerebilen, keşif önceliğine göre
            sıralanmış ReportCandidate listesi.

        Raises:
            ExaSearchError:
                Bütün Exa sorguları başarısız olursa.
        """

        normalized_company_name = normalize_company_name(
            company_name
        )

        validated_report_year = validate_report_year(
            report_year
        )

        candidate_limit = (
            max_candidates
            if max_candidates is not None
            else self.settings.exa_result_limit
        )

        if candidate_limit < 1:
            raise ValueError(
                "max_candidates en az 1 olmalıdır."
            )

        queries = self.build_queries(
            company_name=normalized_company_name,
            report_year=validated_report_year,
        )

        candidates_by_url: dict[str, ReportCandidate] = {}

        successful_query_count = 0
        search_errors: list[str] = []

        # Her sorguda yalnızca aday limitine eşit sayıda sonuç istemek,
        # özellikle düşük limitlerde kurumsal HTML rapor arşivlerinin gözden
        # kaçmasına neden olabilir. Exa'nın tek sorgu için güvenli üst sınırı
        # olan 10 korunurken en az 5 sonuç istenir.
        results_per_query = min(
            max(candidate_limit * 2, 5),
            10,
        )

        for query in queries:
            try:
                response = self._execute_search(
                    query=query,
                    results_per_query=results_per_query,
                    include_highlights=include_highlights,
                )
            except Exception as error:
                error_message = (
                    f"Sorgu başarısız oldu: {query}. "
                    f"Hata: {error}"
                )

                logger.warning(
                    error_message
                )

                search_errors.append(
                    error_message
                )

                continue

            successful_query_count += 1

            results = getattr(
                response,
                "results",
                [],
            )

            for result in results:
                candidate = self._result_to_candidate(
                    result=result,
                    company_name=normalized_company_name,
                    report_year=validated_report_year,
                    search_query=query,
                )

                if candidate is None:
                    continue

                canonical_url = canonicalize_url(
                    candidate.url
                )

                existing_candidate = candidates_by_url.get(
                    canonical_url
                )

                if existing_candidate is None:
                    candidates_by_url[canonical_url] = candidate
                    continue

                self._merge_candidate(
                    existing_candidate=existing_candidate,
                    new_candidate=candidate,
                    search_query=query,
                )

        if successful_query_count == 0 and search_errors:
            joined_errors = "\n".join(
                search_errors
            )

            raise ExaSearchError(
                "Exa üzerindeki bütün rapor aramaları başarısız oldu.\n"
                f"{joined_errors}"
            )

        candidates = list(
            candidates_by_url.values()
        )

        candidates.sort(
            key=candidate_search_sort_key,
            reverse=True,
        )

        selected_candidates = candidates[:candidate_limit]

        logger.info(
            "Exa araması tamamlandı. Toplam benzersiz aday=%s, "
            "döndürülen aday=%s",
            len(candidates),
            len(selected_candidates),
        )

        for index, candidate in enumerate(
            selected_candidates,
            start=1,
        ):
            logger.info(
                "Exa adayı %s | tür=%s | başlık=%s | URL=%s",
                index,
                candidate.metadata.get("resource_kind"),
                candidate.title or "Başlık bulunamadı",
                candidate.url,
            )

        return selected_candidates

    def _execute_search(
        self,
        query: str,
        results_per_query: int,
        include_highlights: bool,
    ) -> ExaSearchResponseProtocol:
        """
        Tek bir Exa sorgusunu retry mekanizmasıyla çalıştırır.
        """

        # category="pdf" kullanılmaz. Böylece doğrudan PDF dosyalarının
        # yanında, rapor bağlantısını içeren kurumsal HTML/arşiv sayfaları da
        # sonuçlara girebilir. HTML içinden PDF keşfi UrlValidator tarafından
        # gerçekleştirilir.
        search_options: dict[str, Any] = {
            "type": "auto",
            "num_results": results_per_query,
        }

        if include_highlights:
            search_options["contents"] = {
                "highlights": {
                    "query": query,
                    "max_characters": 700,
                },
                # Önceden taranmış içerik uygunsa yeniden canlı tarama
                # yapılmasını engeller.
                "max_age_hours": -1,
            }

        last_error: Exception | None = None

        for attempt_number in range(
            1,
            self.maximum_attempts + 1,
        ):
            try:
                logger.info(
                    "Exa araması çalıştırılıyor. "
                    "Sorgu=%s, deneme=%s/%s",
                    query,
                    attempt_number,
                    self.maximum_attempts,
                )

                return self.client.search(
                    query,
                    **search_options,
                )

            except Exception as error:
                last_error = error

                if attempt_number >= self.maximum_attempts:
                    break

                wait_seconds = (
                    self.retry_delay_seconds
                    * (2 ** (attempt_number - 1))
                )

                logger.warning(
                    "Exa araması başarısız oldu. "
                    "%.1f saniye sonra tekrar denenecek.",
                    wait_seconds,
                )

                time.sleep(
                    wait_seconds
                )

        raise ExaSearchError(
            f"Exa sorgusu çalıştırılamadı: {query}"
        ) from last_error

    def _result_to_candidate(
        self,
        result: ExaSearchResultProtocol,
        company_name: str,
        report_year: int,
        search_query: str,
    ) -> ReportCandidate | None:
        """
        Tek bir Exa sonucunu ReportCandidate nesnesine dönüştürür.

        Hatalı veya HTTP dışı URL içeren sonuçlar atlanır.
        """

        url = str(
            getattr(
                result,
                "url",
                "",
            )
        ).strip()

        if not url:
            logger.warning(
                "URL içermeyen Exa sonucu atlandı."
            )
            return None

        parsed_url = urlsplit(url)

        if parsed_url.scheme.casefold() not in {"http", "https"}:
            logger.warning(
                "HTTP/HTTPS dışı Exa sonucu atlandı. URL=%s",
                url,
            )
            return None

        if not parsed_url.netloc:
            logger.warning(
                "Alan adı içermeyen Exa sonucu atlandı. URL=%s",
                url,
            )
            return None

        title_value = getattr(
            result,
            "title",
            None,
        )

        title = (
            str(title_value).strip()
            if title_value is not None
            else ""
        )

        highlights = getattr(
            result,
            "highlights",
            None,
        )

        snippet = build_snippet(
            highlights
        )

        exa_score = safe_non_negative_float(
            getattr(
                result,
                "score",
                None,
            )
        )

        searchable_text = f"{title} {url}".casefold()
        is_summary_candidate = any(
            term in searchable_text
            for term in (
                "executive summary",
                "management summary",
                "summary report",
                "report summary",
                "yönetici özeti",
                "yonetici ozeti",
                "özet rapor",
                "ozet rapor",
                "rapor özeti",
                "rapor ozeti",
            )
        )

        metadata = {
            "is_summary_candidate": is_summary_candidate,
            "exa_result_id": getattr(
                result,
                "id",
                None,
            ),
            "published_date": getattr(
                result,
                "published_date",
                None,
            ),
            "author": getattr(
                result,
                "author",
                None,
            ),
            "crawl_date": getattr(
                result,
                "crawl_date",
                None,
            ),
            "matched_queries": [
                search_query,
            ],
            "resource_kind": get_candidate_resource_kind(
                url
            ),
        }

        try:
            return ReportCandidate(
                company_name=company_name,
                report_year=report_year,
                url=url,
                title=title,
                snippet=snippet,
                exa_score=exa_score,
                metadata=metadata,
            )

        except (ValidationError, ValueError) as error:
            logger.warning(
                "Geçersiz Exa sonucu atlandı. URL=%s, hata=%s",
                url,
                error,
            )

            return None

    @staticmethod
    def _merge_candidate(
        existing_candidate: ReportCandidate,
        new_candidate: ReportCandidate,
        search_query: str,
    ) -> None:
        """
        Aynı URL'nin farklı sorgulardan gelen sonuçlarını birleştirir.
        """

        if (
            new_candidate.exa_score is not None
            and (
                existing_candidate.exa_score is None
                or new_candidate.exa_score
                > existing_candidate.exa_score
            )
        ):
            existing_candidate.exa_score = (
                new_candidate.exa_score
            )

        if (
            not existing_candidate.title
            and new_candidate.title
        ):
            existing_candidate.title = (
                new_candidate.title
            )

        if (
            existing_candidate.snippet is None
            and new_candidate.snippet is not None
        ):
            existing_candidate.snippet = (
                new_candidate.snippet
            )

        matched_queries = existing_candidate.metadata.setdefault(
            "matched_queries",
            [],
        )

        if (
            isinstance(matched_queries, list)
            and search_query not in matched_queries
        ):
            matched_queries.append(
                search_query
            )


def print_candidates(
    candidates: Sequence[ReportCandidate],
) -> None:
    """
    Komut satırı testi için adayları okunabilir biçimde yazdırır.
    """

    if not candidates:
        print(
            "Herhangi bir rapor adayı bulunamadı."
        )
        return

    print(
        f"\nToplam {len(candidates)} benzersiz rapor adayı bulundu.\n"
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        score_text = (
            f"{candidate.exa_score:.4f}"
            if candidate.exa_score is not None
            else "yok"
        )

        print(
            f"{index}. {candidate.title or 'Başlık bulunamadı'}"
        )
        print(
            f"   URL: {candidate.url}"
        )
        print(
            f"   Alan adı: {candidate.source_domain}"
        )
        print(
            f"   Exa skoru: {score_text}"
        )

        matched_queries = candidate.metadata.get(
            "matched_queries",
            [],
        )

        print(
            f"   Eşleşen sorgu sayısı: {len(matched_queries)}"
        )

        if candidate.snippet:
            print(
                f"   Açıklama: {candidate.snippet}"
            )

        print()


def main() -> None:
    """
    Dosyanın komut satırından test edilmesini sağlar.

    Örnek:

        python -m report_discovery.exa_search "Arçelik" 2023
    """

    parser = argparse.ArgumentParser(
        description=(
            "Exa kullanarak sürdürülebilirlik raporu "
            "adaylarını arar."
        )
    )

    parser.add_argument(
        "company_name",
        help="Raporu aranacak şirket adı.",
    )

    parser.add_argument(
        "report_year",
        type=int,
        help="Aranacak sürdürülebilirlik raporunun yılı.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Döndürülecek maksimum benzersiz aday sayısı. "
            "Varsayılan değer config.py üzerinden alınır."
        ),
    )

    parser.add_argument(
        "--highlights",
        action="store_true",
        help=(
            "Arama sonuçlarıyla birlikte kısa içerik "
            "parçalarını da getirir."
        ),
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    searcher = ExaReportSearcher()

    candidates = searcher.search(
        company_name=arguments.company_name,
        report_year=arguments.report_year,
        max_candidates=arguments.limit,
        include_highlights=arguments.highlights,
    )

    print_candidates(
        candidates
    )


if __name__ == "__main__":
    main()