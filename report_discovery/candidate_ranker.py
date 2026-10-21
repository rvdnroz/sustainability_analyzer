from __future__ import annotations

import argparse
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from config import Settings, get_settings
from models import ReportCandidate


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Sabitler
# ---------------------------------------------------------

SUSTAINABILITY_TERMS = {
    "sustainability",
    "sustainable",
    "surdurulebilirlik",
    "esg",
    "environmental social governance",
    "entegre rapor",
    "integrated report",
    "integrated sustainability report",
    "sustainability performance",
    "surdurulebilirlik raporu",
    "corporate responsibility",
    "responsibility report",
    "corporate social responsibility",
    "csr report",
    "non financial report",
    "non-financial report",
    "impact report",
}

REPORT_TERMS = {
    "report",
    "rapor",
    "disclosure",
    "statement",
    "performance",
    "index",
}

SUMMARY_REPORT_TERMS = {
    "executive summary",
    "management summary",
    "summary report",
    "report summary",
    "sustainability summary",
    "sustainability report summary",
    "yonetici ozeti",
    "yönetici özeti",
    "yonectici ozeti",
    "yonectici ozet",
    "yonetici ozeti pdf",
    "yonetici ozet",
    "yönetici özet",
    "ozet rapor",
    "özet rapor",
    "rapor ozeti",
    "rapor özeti",
    "surdurulebilirlik ozeti",
    "sürdürülebilirlik özeti",
}

NEGATIVE_REPORT_TERMS = {
    "annual report",
    "financial report",
    "faaliyet raporu",
    "financial statements",
    "investor presentation",
    "earnings presentation",
    "quarterly report",
    "interim report",
    "activity report",
}

PDF_TERMS = {
    "pdf",
    "download pdf",
    "view pdf",
}

COMPANY_SUFFIXES = {
    "as",
    "a s",
    "anonim sirketi",
    "anonim ortakligi",
    "holding",
    "holdings",
    "group",
    "grubu",
    "company",
    "corporation",
    "corp",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "llc",
    "plc",
    "sa",
    "ag",
    "gmbh",
}

TRUSTED_REPORT_DOMAINS = {
    "sustainabilityreports.com",
    "responsibilityreports.com",
    "annualreports.com",
    "unglobalcompact.org",
    "globalreporting.org",
}

LOW_AUTHORITY_DOMAINS = {
    "scribd.com",
    "slideshare.net",
    "studocu.com",
    "coursehero.com",
    "academia.edu",
    "researchgate.net",
    "studiopathway.com",
}

BLOCKED_OR_IRRELEVANT_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "x.com",
    "twitter.com",
    "pinterest.com",
}


# ---------------------------------------------------------
# Yardımcı veri sınıfı
# ---------------------------------------------------------

@dataclass(frozen=True)
class RankingWeights:
    """
    Aday rapor puanlamasında kullanılan bileşen ağırlıkları.

    Bütün ağırlıkların toplamı 1.0 olmalıdır.
    """

    company_match: float = 0.28
    year_match: float = 0.22
    report_type: float = 0.20
    pdf_likelihood: float = 0.15
    domain_authority: float = 0.10
    exa_score: float = 0.05

    def validate(self) -> None:
        values = [
            self.company_match,
            self.year_match,
            self.report_type,
            self.pdf_likelihood,
            self.domain_authority,
            self.exa_score,
        ]

        if any(value < 0.0 for value in values):
            raise ValueError(
                "Sıralama ağırlıkları negatif olamaz."
            )

        total = sum(values)

        if not math.isclose(
            total,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Sıralama ağırlıklarının toplamı 1.0 olmalıdır. "
                f"Mevcut toplam: {total}"
            )


# ---------------------------------------------------------
# Metin normalizasyonu
# ---------------------------------------------------------

def normalize_text(
    value: str | None,
) -> str:
    """
    Metni karşılaştırma için normalize eder.

    Uygulanan işlemler:

    - Türkçe karakterleri sadeleştirme
    - Unicode aksanlarını kaldırma
    - Küçük harfe çevirme
    - Noktalama karakterlerini boşluğa dönüştürme
    - Tekrarlanan boşlukları temizleme
    """

    if value is None:
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
        }
    )

    translated_value = value.translate(
        translation_table
    )

    unicode_normalized = unicodedata.normalize(
        "NFKD",
        translated_value,
    )

    without_accents = "".join(
        character
        for character in unicode_normalized
        if not unicodedata.combining(character)
    )

    lowercase_value = without_accents.casefold()

    alphanumeric_value = re.sub(
        r"[^a-z0-9]+",
        " ",
        lowercase_value,
    )

    return " ".join(
        alphanumeric_value.split()
    )


def tokenize(
    value: str | None,
) -> set[str]:
    """
    Normalize edilmiş metni benzersiz kelimelere ayırır.
    """

    normalized_value = normalize_text(
        value
    )

    if not normalized_value:
        return set()

    return set(
        normalized_value.split()
    )


def normalize_company_name(
    company_name: str,
) -> str:
    """
    Şirket adını karşılaştırma için sadeleştirir.

    Şirket türü belirten A.Ş., Holding, Ltd. gibi ekleri kaldırır.
    """

    normalized_name = normalize_text(
        company_name
    )

    company_tokens = normalized_name.split()

    filtered_tokens = [
        token
        for token in company_tokens
        if token not in COMPANY_SUFFIXES
    ]

    if filtered_tokens:
        return " ".join(
            filtered_tokens
        )

    return normalized_name


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


def get_url_search_text(
    url: str,
) -> str:
    """
    URL yolunu ve query parametrelerini aranabilir metne dönüştürür.
    """

    parsed_url = urlsplit(
        url
    )

    url_text = unquote(
        f"{parsed_url.netloc} "
        f"{parsed_url.path} "
        f"{parsed_url.query}"
    )

    return normalize_text(
        url_text
    )


def clamp_score(
    value: float,
) -> float:
    """
    Skoru 0 ile 1 aralığında sınırlar.
    """

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def normalize_exa_score(
    score: float | None,
) -> float:
    """
    Sınırı bilinmeyen Exa skorunu 0-1 aralığına dönüştürür.

    Kullanılan dönüşüm:

        normalized = score / (score + 1)

    Örnek:

        1.0  -> 0.50
        4.0  -> 0.80
        9.0  -> 0.90
        28.0 -> 0.97
    """

    if score is None or score <= 0:
        return 0.0

    return clamp_score(
        score / (score + 1.0)
    )


def sequence_similarity(
    first_value: str,
    second_value: str,
) -> float:
    """
    İki normalize metin arasındaki karakter dizisi benzerliğini hesaplar.
    """

    if not first_value or not second_value:
        return 0.0

    return SequenceMatcher(
        None,
        first_value,
        second_value,
    ).ratio()


def token_overlap_score(
    expected_text: str,
    candidate_text: str,
) -> float:
    """
    Beklenen metindeki tokenların aday metinde bulunma oranını hesaplar.
    """

    expected_tokens = tokenize(
        expected_text
    )

    candidate_tokens = tokenize(
        candidate_text
    )

    if not expected_tokens:
        return 0.0

    intersection = expected_tokens.intersection(
        candidate_tokens
    )

    return len(intersection) / len(expected_tokens)


def contains_any_term(
    text: str,
    terms: set[str],
) -> bool:
    """
    Normalize edilmiş metinde terimlerden en az birinin bulunup
    bulunmadığını kontrol eder.
    """

    normalized_text = normalize_text(
        text
    )

    return any(
        normalize_text(term) in normalized_text
        for term in terms
    )


def detect_candidate_years(
    candidate: ReportCandidate,
) -> set[int]:
    """
    Adayın başlık, URL ve açıklamasında geçen rapor yıllarını bulur.

    Dört basamaklı yılların yanında URL dosya adındaki ``_22_`` gibi
    iki basamaklı yıl gösterimlerini de hedef yıla en yakın yüzyıla
    dönüştürür.
    """

    # Yıl çakışması için yalnızca başlık ve URL kullanılır.
    # Arama snippet'leri aynı sayfadaki başka rapor yıllarını
    # içerebildiği için burada yanlış negatif üretebilir.
    combined_text = " ".join(
        [
            candidate.title or "",
            candidate.url,
        ]
    )

    detected_years = {
        int(value)
        for value in re.findall(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            combined_text,
        )
    }

    filename = Path(
        unquote(urlsplit(candidate.url).path)
    ).name

    target_year = candidate.report_year
    target_century = (target_year // 100) * 100

    for value in re.findall(
        r"(?:^|[_\-\s])(\d{2})(?=[_\-\s.]|$)",
        filename,
    ):
        mapped_year = target_century + int(value)

        if mapped_year - target_year > 50:
            mapped_year -= 100
        elif target_year - mapped_year > 50:
            mapped_year += 100

        if 1990 <= mapped_year <= 2100:
            detected_years.add(mapped_year)

    return detected_years


def candidate_has_sustainability_signal(
    candidate: ReportCandidate,
) -> bool:
    """
    Adayın sürdürülebilirlik, ESG, entegre raporlama veya kurumsal
    sorumluluk raporu olduğuna dair en az bir açık sinyal taşıyıp
    taşımadığını belirler.

    HTML rapor arşivleri de kabul edilebilmesi için başlık, URL ve
    snippet birlikte değerlendirilir.
    """

    combined_text = " ".join(
        [
            candidate.title or "",
            candidate.url,
            candidate.snippet or "",
        ]
    )

    return contains_any_term(
        combined_text,
        SUSTAINABILITY_TERMS,
    )


def candidate_is_summary_report(
    candidate: ReportCandidate,
) -> bool:
    """
    Adayın tam rapor yerine yönetici özeti veya özet rapor olduğunu belirler.

    Yalnızca sabit ifadeler değil, normalize edilmiş tokenlar da kontrol edilir.
    Böylece "yonetici-ozeti", "yonectici_ozeti", "ozet", "ozeti" ve benzeri
    yazım/URL varyasyonları kaçmaz.
    """

    primary_text = " ".join(
        [
            candidate.title or "",
            candidate.url,
        ]
    )

    normalized = normalize_text(primary_text)
    tokens = set(normalized.split())

    if contains_any_term(
        primary_text,
        SUMMARY_REPORT_TERMS,
    ):
        return True

    if any(
        token.startswith("ozet")
        or token in {"summary", "summaries"}
        for token in tokens
    ):
        return True

    compact = normalized.replace(" ", "")
    return (
        "executivesummary" in compact
        or "managementsummary" in compact
        or "yoneticiozeti" in compact
        or "yonecticiozeti" in compact
    )


def has_report_type_conflict(
    candidate: ReportCandidate,
) -> bool:
    """
    Başlık veya URL açıkça faaliyet, finansal ya da yıllık raporu
    gösteriyorsa ve aynı ana alanlarda sürdürülebilirlik sinyali yoksa
    adayı kesin olarak reddeder.

    Snippet bu sert kontrolde kullanılmaz; snippet farklı raporları
    veya sayfa içindeki genel sürdürülebilirlik metnini içerebilir.
    """

    primary_text = " ".join(
        [
            candidate.title or "",
            candidate.url,
        ]
    )

    has_negative_signal = contains_any_term(
        primary_text,
        NEGATIVE_REPORT_TERMS,
    )

    has_primary_sustainability_signal = contains_any_term(
        primary_text,
        SUSTAINABILITY_TERMS,
    )

    return (
        has_negative_signal
        and not has_primary_sustainability_signal
    )


# ---------------------------------------------------------
# Candidate ranker
# ---------------------------------------------------------

class CandidateRanker:
    """
    Exa rapor adaylarını alan bazlı puanlayarak sıralar.

    Hesaplanan bileşenler:

    - Şirket adı eşleşmesi
    - Rapor yılı eşleşmesi
    - Sürdürülebilirlik raporu türü eşleşmesi
    - PDF olma ihtimali
    - Kaynak alan adı güvenilirliği
    - Normalize edilmiş Exa skoru
    """

    def __init__(
        self,
        settings: Settings | None = None,
        weights: RankingWeights | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.weights = weights or RankingWeights()

        self.weights.validate()

    def rank(
        self,
        candidates: list[ReportCandidate],
        *,
        company_name: str | None = None,
        report_year: int | None = None,
    ) -> list[ReportCandidate]:
        """IngestionPipeline ile uyumluluk sağlayan sıralama metodu."""

        del company_name, report_year

        return self.rank_candidates(
            list(candidates),
        )

    def rank_candidates(
        self,
        candidates: list[ReportCandidate],
        *,
        minimum_score: float | None = None,
        include_rejected: bool = False,
    ) -> list[ReportCandidate]:
        """
        Adayları puanlar ve final_score değerine göre sıralar.

        Args:
            candidates:
                ExaReportSearcher tarafından bulunan adaylar.

            minimum_score:
                Kabul edilmesi gereken minimum final skor.
                Verilmezse config.py içerisindeki
                minimum_candidate_score kullanılır.

            include_rejected:
                False ise eşik altındaki adaylar sonuçtan çıkarılır.
                True ise bütün adaylar sıralanarak döndürülür.
        """

        score_threshold = (
            minimum_score
            if minimum_score is not None
            else self.settings.minimum_candidate_score
        )

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                "Minimum aday skoru 0 ile 1 arasında olmalıdır."
            )

        ranked_candidates: list[ReportCandidate] = []

        for candidate in candidates:
            self.score_candidate(
                candidate
            )

            detected_years = detect_candidate_years(
                candidate
            )

            has_year_conflict = (
                bool(detected_years)
                and candidate.report_year
                not in detected_years
            )

            has_sustainability_signal = (
                candidate_has_sustainability_signal(
                    candidate
                )
            )

            report_type_conflict = (
                has_report_type_conflict(
                    candidate
                )
            )

            is_summary_report = candidate_is_summary_report(
                candidate
            )

            is_accepted = (
                candidate.final_score
                >= score_threshold
                and not has_year_conflict
                and has_sustainability_signal
                and not report_type_conflict
                and not is_summary_report
            )

            candidate.metadata[
                "detected_report_years"
            ] = sorted(detected_years)

            candidate.metadata[
                "year_conflict"
            ] = has_year_conflict

            candidate.metadata[
                "has_sustainability_signal"
            ] = has_sustainability_signal

            candidate.metadata[
                "report_type_conflict"
            ] = report_type_conflict

            candidate.metadata[
                "is_summary_report"
            ] = is_summary_report

            if has_year_conflict:
                conflict_note = (
                    "Aday açıkça farklı bir rapor yılı içerdiği "
                    "için reddedildi."
                )

                if conflict_note not in candidate.validation_notes:
                    candidate.validation_notes.append(
                        conflict_note
                    )

            if not has_sustainability_signal:
                conflict_note = (
                    "Adayda sürdürülebilirlik, ESG, entegre raporlama "
                    "veya kurumsal sorumluluk sinyali bulunmadığı "
                    "için reddedildi."
                )

                if conflict_note not in candidate.validation_notes:
                    candidate.validation_notes.append(
                        conflict_note
                    )

            if report_type_conflict:
                conflict_note = (
                    "Aday açıkça faaliyet, finansal veya yıllık rapor "
                    "olduğu için reddedildi."
                )

                if conflict_note not in candidate.validation_notes:
                    candidate.validation_notes.append(
                        conflict_note
                    )

            if is_summary_report:
                conflict_note = (
                    "Aday tam sürdürülebilirlik raporu yerine yönetici "
                    "özeti veya özet rapor olduğu için reddedildi."
                )

                if conflict_note not in candidate.validation_notes:
                    candidate.validation_notes.append(
                        conflict_note
                    )

            candidate.metadata[
                "ranking_accepted"
            ] = is_accepted

            candidate.metadata[
                "ranking_threshold"
            ] = score_threshold

            logger.info(
                "Aday değerlendirildi. kabul=%s, skor=%.4f, "
                "yıllar=%s, yıl_çakışması=%s, "
                "sürdürülebilirlik_sinyali=%s, "
                "rapor_türü_çakışması=%s, özet_rapor=%s, url=%s",
                is_accepted,
                candidate.final_score,
                sorted(detected_years),
                has_year_conflict,
                has_sustainability_signal,
                report_type_conflict,
                is_summary_report,
                candidate.url,
            )

            if is_accepted or include_rejected:
                ranked_candidates.append(
                    candidate
                )

        ranked_candidates.sort(
            key=lambda candidate: (
                candidate.final_score,
                candidate.domain_authority_score,
                candidate.pdf_likelihood_score,
                candidate.year_match_score,
                candidate.company_match_score,
            ),
            reverse=True,
        )

        for rank, candidate in enumerate(
            ranked_candidates,
            start=1,
        ):
            candidate.metadata[
                "ranking_position"
            ] = rank

        return ranked_candidates

    def score_candidate(
        self,
        candidate: ReportCandidate,
    ) -> ReportCandidate:
        """
        Tek bir adayın bütün alt skorlarını ve final skorunu hesaplar.
        """

        candidate.company_match_score = (
            self.calculate_company_match_score(
                candidate
            )
        )

        candidate.year_match_score = (
            self.calculate_year_match_score(
                candidate
            )
        )

        candidate.report_type_score = (
            self.calculate_report_type_score(
                candidate
            )
        )

        candidate.pdf_likelihood_score = (
            self.calculate_pdf_likelihood_score(
                candidate
            )
        )

        candidate.domain_authority_score = (
            self.calculate_domain_authority_score(
                candidate
            )
        )

        normalized_exa_score = normalize_exa_score(
            candidate.exa_score
        )

        raw_final_score = (
            candidate.company_match_score
            * self.weights.company_match
            + candidate.year_match_score
            * self.weights.year_match
            + candidate.report_type_score
            * self.weights.report_type
            + candidate.pdf_likelihood_score
            * self.weights.pdf_likelihood
            + candidate.domain_authority_score
            * self.weights.domain_authority
            + normalized_exa_score
            * self.weights.exa_score
        )

        penalty = self.calculate_penalty(
            candidate
        )

        candidate.final_score = round(
            clamp_score(
                raw_final_score - penalty
            ),
            4,
        )

        candidate.metadata[
            "ranking_breakdown"
        ] = {
            "company_match_score": (
                candidate.company_match_score
            ),
            "year_match_score": (
                candidate.year_match_score
            ),
            "report_type_score": (
                candidate.report_type_score
            ),
            "pdf_likelihood_score": (
                candidate.pdf_likelihood_score
            ),
            "domain_authority_score": (
                candidate.domain_authority_score
            ),
            "normalized_exa_score": (
                normalized_exa_score
            ),
            "raw_final_score": round(
                raw_final_score,
                4,
            ),
            "penalty": round(
                penalty,
                4,
            ),
            "final_score": candidate.final_score,
        }

        self.populate_ranking_notes(
            candidate
        )

        return candidate

    def calculate_company_match_score(
        self,
        candidate: ReportCandidate,
    ) -> float:
        """
        Şirket adının başlık, URL, açıklama ve alan adıyla eşleşmesini ölçer.
        """

        company_name = normalize_company_name(
            candidate.company_name
        )

        title_text = normalize_text(
            candidate.title
        )

        url_text = get_url_search_text(
            candidate.url
        )

        snippet_text = normalize_text(
            candidate.snippet
        )

        domain_text = normalize_text(
            candidate.source_domain
            or get_domain(candidate.url)
        )

        combined_text = " ".join(
            value
            for value in [
                title_text,
                url_text,
                snippet_text,
            ]
            if value
        )

        if not company_name:
            return 0.0

        # Tam şirket adı başlık veya URL içerisinde bulunuyorsa
        # en güçlü eşleşme olarak kabul edilir.
        exact_title_match = (
            company_name in title_text
        )

        exact_url_match = (
            company_name.replace(" ", "")
            in url_text.replace(" ", "")
        )

        exact_domain_match = (
            company_name.replace(" ", "")
            in domain_text.replace(" ", "")
        )

        token_overlap = token_overlap_score(
            company_name,
            combined_text,
        )

        title_similarity = sequence_similarity(
            company_name,
            title_text,
        )

        score = (
            token_overlap * 0.55
            + title_similarity * 0.15
        )

        if exact_title_match:
            score += 0.25

        if exact_url_match:
            score += 0.15

        if exact_domain_match:
            score += 0.20

        return round(
            clamp_score(score),
            4,
        )

    def calculate_year_match_score(
        self,
        candidate: ReportCandidate,
    ) -> float:
        """
        Aranan yılın başlık, URL ve açıklamada bulunmasını puanlar.
        """

        target_year = str(
            candidate.report_year
        )

        title_text = normalize_text(
            candidate.title
        )

        url_text = get_url_search_text(
            candidate.url
        )

        snippet_text = normalize_text(
            candidate.snippet
        )

        if target_year in title_text:
            return 1.0

        if target_year in url_text:
            return 0.95

        if target_year in snippet_text:
            return 0.80

        combined_text = " ".join(
            [
                title_text,
                url_text,
                snippet_text,
            ]
        )

        detected_years = {
            int(year)
            for year in re.findall(
                r"\b(?:19|20)\d{2}\b",
                combined_text,
            )
        }

        if not detected_years:
            return 0.35

        closest_difference = min(
            abs(
                detected_year
                - candidate.report_year
            )
            for detected_year in detected_years
        )

        if closest_difference == 1:
            return 0.0

        return 0.0

    def calculate_report_type_score(
        self,
        candidate: ReportCandidate,
    ) -> float:
        """
        Adayın sürdürülebilirlik veya ESG raporu olma ihtimalini hesaplar.
        """

        title_text = normalize_text(
            candidate.title
        )

        url_text = get_url_search_text(
            candidate.url
        )

        snippet_text = normalize_text(
            candidate.snippet
        )

        combined_text = " ".join(
            [
                title_text,
                url_text,
                snippet_text,
            ]
        )

        sustainability_matches = sum(
            1
            for term in SUSTAINABILITY_TERMS
            if normalize_text(term) in combined_text
        )

        report_matches = sum(
            1
            for term in REPORT_TERMS
            if normalize_text(term) in combined_text
        )

        negative_matches = sum(
            1
            for term in NEGATIVE_REPORT_TERMS
            if normalize_text(term) in combined_text
        )

        score = 0.0

        if sustainability_matches > 0:
            score += 0.70

        if sustainability_matches > 1:
            score += 0.10

        if report_matches > 0:
            score += 0.20

        if negative_matches > 0:
            score -= min(
                0.50,
                negative_matches * 0.25,
            )

        return round(
            clamp_score(score),
            4,
        )

    def calculate_pdf_likelihood_score(
        self,
        candidate: ReportCandidate,
    ) -> float:
        """
        URL ve başlık üzerinden kaynağın doğrudan PDF olma ihtimalini ölçer.
        """

        parsed_url = urlsplit(
            candidate.url
        )

        path = unquote(
            parsed_url.path
        ).casefold()

        query = unquote(
            parsed_url.query
        ).casefold()

        title_text = normalize_text(
            candidate.title
        )

        score = 0.0

        if path.endswith(".pdf"):
            score = 1.0

        elif ".pdf" in path:
            score = 0.90

        elif "pdf" in query:
            score = 0.75

        elif contains_any_term(
            title_text,
            PDF_TERMS,
        ):
            score = 0.65

        elif candidate.metadata.get(
            "exa_category"
        ) == "pdf":
            score = 0.60

        else:
            score = 0.25

        return round(
            clamp_score(score),
            4,
        )

    def calculate_domain_authority_score(
        self,
        candidate: ReportCandidate,
    ) -> float:
        """
        Alan adının güvenilirliğini ve şirketle ilişkisini puanlar.
        """

        domain = (
            candidate.source_domain
            or get_domain(candidate.url)
        ).casefold()

        normalized_domain = normalize_text(
            domain
        ).replace(
            " ",
            ""
        )

        normalized_company = normalize_company_name(
            candidate.company_name
        ).replace(
            " ",
            ""
        )

        if domain in BLOCKED_OR_IRRELEVANT_DOMAINS:
            return 0.0

        if domain in LOW_AUTHORITY_DOMAINS:
            return 0.20

        if domain in TRUSTED_REPORT_DOMAINS:
            return 0.70

        # Şirket adının kurumsal alan adı içinde bulunması,
        # kaynağın resmi şirket sitesi olma ihtimalini yükseltir.
        if (
            normalized_company
            and normalized_company in normalized_domain
        ):
            return 1.0

        company_tokens = tokenize(
            candidate.company_name
        )

        domain_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                normalize_text(domain),
            )
        )

        if company_tokens.intersection(
            domain_tokens
        ):
            return 0.90

        if domain.endswith(
            (
                ".gov",
                ".gov.tr",
                ".edu",
                ".edu.tr",
                ".org",
            )
        ):
            return 0.65

        return 0.45

    def calculate_penalty(
        self,
        candidate: ReportCandidate,
    ) -> float:
        """
        Belirgin yanlış eşleşmeler için final skordan düşülecek cezayı hesaplar.
        """

        penalty = 0.0

        combined_text = normalize_text(
            " ".join(
                [
                    candidate.title,
                    candidate.url,
                    candidate.snippet or "",
                ]
            )
        )

        detected_years = {
            int(year)
            for year in re.findall(
                r"\b(?:19|20)\d{2}\b",
                combined_text,
            )
        }

        if (
            detected_years
            and candidate.report_year
            not in detected_years
        ):
            penalty += 0.15

        if candidate.company_match_score < 0.25:
            penalty += 0.20

        if candidate.report_type_score < 0.30:
            penalty += 0.15

        if contains_any_term(
            " ".join([candidate.title or "", candidate.url]),
            SUMMARY_REPORT_TERMS,
        ):
            penalty += 0.60

        if contains_any_term(
            combined_text,
            NEGATIVE_REPORT_TERMS,
        ):
            if not contains_any_term(
                combined_text,
                SUSTAINABILITY_TERMS,
            ):
                penalty += 0.15

        domain = (
            candidate.source_domain
            or get_domain(candidate.url)
        ).casefold()

        if domain in BLOCKED_OR_IRRELEVANT_DOMAINS:
            penalty += 0.40

        return clamp_score(
            penalty
        )

    def populate_ranking_notes(
        self,
        candidate: ReportCandidate,
    ) -> None:
        """
        Puanlama sonucuna göre açıklayıcı notlar oluşturur.
        """

        notes: list[str] = []

        if candidate.company_match_score >= 0.80:
            notes.append(
                "Şirket adı güçlü biçimde eşleşiyor."
            )
        elif candidate.company_match_score < 0.40:
            notes.append(
                "Şirket adı eşleşmesi zayıf."
            )

        if candidate.year_match_score >= 0.90:
            notes.append(
                "Aranan rapor yılı açıkça eşleşiyor."
            )
        elif candidate.year_match_score < 0.50:
            notes.append(
                "Rapor yılı eşleşmesi belirsiz veya zayıf."
            )

        if candidate.report_type_score >= 0.80:
            notes.append(
                "Başlık veya URL sürdürülebilirlik raporunu güçlü biçimde işaret ediyor."
            )
        elif candidate.report_type_score < 0.40:
            notes.append(
                "Kaynağın sürdürülebilirlik raporu olduğu kesin değil."
            )

        if candidate.pdf_likelihood_score >= 0.90:
            notes.append(
                "URL doğrudan PDF dosyasını işaret ediyor."
            )
        elif candidate.pdf_likelihood_score < 0.50:
            notes.append(
                "URL doğrudan PDF olmayabilir."
            )

        if candidate_is_summary_report(candidate):
            notes.append(
                "Kaynak tam rapor değil; yönetici özeti veya özet rapor."
            )

        if candidate.domain_authority_score >= 0.90:
            notes.append(
                "Kaynak alan adı resmi şirket sitesi olma ihtimali taşıyor."
            )
        elif candidate.domain_authority_score <= 0.20:
            notes.append(
                "Kaynak alan adının güvenilirliği düşük."
            )

        existing_notes = {
            note.casefold()
            for note in candidate.validation_notes
        }

        for note in notes:
            if note.casefold() not in existing_notes:
                candidate.validation_notes.append(
                    note
                )
                existing_notes.add(
                    note.casefold()
                )


# ---------------------------------------------------------
# Konsol çıktısı
# ---------------------------------------------------------

def print_ranked_candidates(
    candidates: list[ReportCandidate],
) -> None:
    """
    Puanlanmış adayları okunabilir şekilde yazdırır.
    """

    if not candidates:
        print(
            "Minimum puanı geçen rapor adayı bulunamadı."
        )
        return

    print(
        f"\nToplam {len(candidates)} aday sıralandı.\n"
    )

    for position, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{position}. {candidate.title or 'Başlık bulunamadı'}"
        )

        print(
            f"   Final skor: "
            f"{candidate.final_score:.4f}"
        )

        print(
            f"   Şirket eşleşmesi: "
            f"{candidate.company_match_score:.4f}"
        )

        print(
            f"   Yıl eşleşmesi: "
            f"{candidate.year_match_score:.4f}"
        )

        print(
            f"   Rapor türü: "
            f"{candidate.report_type_score:.4f}"
        )

        print(
            f"   PDF ihtimali: "
            f"{candidate.pdf_likelihood_score:.4f}"
        )

        print(
            f"   Alan adı güveni: "
            f"{candidate.domain_authority_score:.4f}"
        )

        print(
            f"   URL: {candidate.url}"
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


# ---------------------------------------------------------
# Komut satırı testi
# ---------------------------------------------------------

def main() -> None:
    """
    Exa aramasını çalıştırır ve bulunan adayları puanlar.

    Örnek:

        python -m report_discovery.candidate_ranker "Arçelik" 2023
    """

    parser = argparse.ArgumentParser(
        description=(
            "Exa sonuçlarını şirket, yıl, rapor türü, PDF ve "
            "alan adı kriterlerine göre sıralar."
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
        "--limit",
        type=int,
        default=10,
        help="Exa'dan alınacak maksimum aday sayısı.",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help=(
            "Aday kabul eşiği. Verilmezse config.py "
            "içerisindeki değer kullanılır."
        ),
    )

    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Minimum puanın altındaki adayları da gösterir.",
    )

    parser.add_argument(
        "--highlights",
        action="store_true",
        help="Exa sonuçlarında kısa içerik parçalarını da getirir.",
    )

    arguments = parser.parse_args()

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
        max_candidates=arguments.limit,
        include_highlights=arguments.highlights,
    )

    ranker = CandidateRanker()

    ranked_candidates = ranker.rank_candidates(
        candidates,
        minimum_score=arguments.minimum_score,
        include_rejected=arguments.include_rejected,
    )

    print_ranked_candidates(
        ranked_candidates
    )


if __name__ == "__main__":
    main()