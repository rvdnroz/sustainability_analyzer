from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


class EvidenceValidationStatus(StrEnum):
    VALID = "valid"
    PARTIALLY_VALID = "partially_valid"
    INSUFFICIENT = "insufficient"
    INVALID = "invalid"


class EvidenceIssueType(StrEnum):
    QUERY_MISMATCH = "query_mismatch"
    REPORT_MISMATCH = "report_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    METRIC_MISMATCH = "metric_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    STANDARD_MISMATCH = "standard_mismatch"
    CATEGORY_MISMATCH = "category_mismatch"
    EMPTY_CONTENT = "empty_content"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    LOW_RERANK_SCORE = "low_rerank_score"
    MISSING_NUMERIC_VALUE = "missing_numeric_value"
    DISCLOSED_AS_MISSING = "disclosed_as_missing"
    NUMERIC_ARTIFACT = "numeric_artifact"
    LOW_SOURCE_QUALITY = "low_source_quality"
    CONFLICTING_VALUES = "conflicting_values"
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class EvidenceValidationIssue:
    issue_type: EvidenceIssueType
    message: str
    severity: float
    evidence_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EvidenceValidationRequest:
    query: str
    report_id: str | None = None
    company_name: str | None = None
    years: tuple[int, ...] = ()
    metric_terms: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    standard_codes: tuple[str, ...] = ()
    category: str | None = None
    expects_numeric_answer: bool = False
    expects_table_evidence: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ValidatedEvidence:
    evidence_id: str
    payload: Any
    text: str
    score: float
    status: EvidenceValidationStatus

    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    relevance_score: float = 0.0
    consistency_score: float = 0.0
    completeness_score: float = 0.0

    report_id: str | None = None
    page_number: int | None = None
    row_index: int | None = None
    section_title: str | None = None

    matched_years: tuple[int, ...] = ()
    matched_metrics: tuple[str, ...] = ()
    matched_units: tuple[str, ...] = ()
    matched_standard_codes: tuple[str, ...] = ()
    numeric_values: tuple[str, ...] = ()

    issues: tuple[EvidenceValidationIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EvidenceValidationResult:
    status: EvidenceValidationStatus
    score: float
    evidences: tuple[ValidatedEvidence, ...]
    valid_evidence_count: int
    rejected_evidence_count: int
    issues: tuple[EvidenceValidationIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_sufficient(self) -> bool:
        return self.status in {
            EvidenceValidationStatus.VALID,
            EvidenceValidationStatus.PARTIALLY_VALID,
        }

    @property
    def accepted_evidences(self) -> tuple[ValidatedEvidence, ...]:
        return tuple(
            evidence
            for evidence in self.evidences
            if evidence.status
            in {
                EvidenceValidationStatus.VALID,
                EvidenceValidationStatus.PARTIALLY_VALID,
            }
        )


@dataclass(slots=True, frozen=True)
class EvidenceValidatorConfig:
    minimum_evidence_score: float = 0.34
    minimum_relevance_score: float = 0.08
    minimum_retrieval_score: float = 0.0
    minimum_rerank_score: float = 0.0

    minimum_valid_evidence_count: int = 1
    maximum_evidence_count: int = 15

    query_weight: float = 0.18
    metric_weight: float = 0.26
    year_weight: float = 0.08
    unit_weight: float = 0.06
    standard_weight: float = 0.05
    report_weight: float = 0.07
    category_weight: float = 0.08
    source_score_weight: float = 0.22

    partial_valid_threshold: float = 0.20
    conflict_tolerance: float = 0.10

    require_report_match: bool = False
    reject_year_mismatch: bool = False
    reject_unit_mismatch: bool = False
    require_metric_match_for_numeric: bool = True
    require_year_match_for_numeric: bool = True
    reject_numeric_artifacts: bool = True
    reject_invalid_numeric_sources: bool = True
    minimum_numeric_source_validation_score: float = 0.45

    structured_evidence_bonus: float = 0.14
    missing_disclosure_bonus: float = 0.18
    preserve_structured_evidence: bool = True
    maximum_results_per_table: int = 2

    deduplicate_evidence: bool = True

    def __post_init__(self) -> None:
        score_fields = (
            self.minimum_evidence_score,
            self.minimum_relevance_score,
            self.minimum_retrieval_score,
            self.minimum_rerank_score,
            self.partial_valid_threshold,
            self.conflict_tolerance,
            self.minimum_numeric_source_validation_score,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in score_fields
        ):
            raise ValueError(
                "Skor eşikleri 0 ile 1 arasında olmalıdır.",
            )

        if self.minimum_valid_evidence_count <= 0:
            raise ValueError(
                "minimum_valid_evidence_count pozitif olmalıdır.",
            )

        if self.maximum_evidence_count <= 0:
            raise ValueError(
                "maximum_evidence_count pozitif olmalıdır.",
            )

        if self.structured_evidence_bonus < 0:
            raise ValueError(
                "structured_evidence_bonus negatif olamaz.",
            )

        if self.missing_disclosure_bonus < 0:
            raise ValueError(
                "missing_disclosure_bonus negatif olamaz.",
            )

        if self.maximum_results_per_table <= 0:
            raise ValueError(
                "maximum_results_per_table pozitif olmalıdır.",
            )

        weights = (
            self.query_weight,
            self.metric_weight,
            self.year_weight,
            self.unit_weight,
            self.standard_weight,
            self.report_weight,
            self.category_weight,
            self.source_score_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Doğrulama ağırlıkları negatif olamaz.",
            )

        if math.isclose(sum(weights), 0.0):
            raise ValueError(
                "Doğrulama ağırlıklarının toplamı sıfır olamaz.",
            )


class EvidenceValidatorError(RuntimeError):
    """Kanıt doğrulama işlemleri sırasında oluşan temel hata."""


class EvidenceValidator:
    """
    Retrieval sonrasında elde edilen kanıtların soru ile uyumunu denetler.

    Doğrulanan alanlar:
    - Soru ve kanıt arasındaki anlamsal/sözcüksel uygunluk
    - Rapor kimliği ve şirket uyumu
    - Yıl eşleşmesi
    - Metrik ve gösterge eşleşmesi
    - Birim eşleşmesi
    - GRI/ESRS standart kodu eşleşmesi
    - ESG kategori uyumu
    - Sayısal cevap gereken sorularda sayısal değer bulunması
    - Kanıtlar arasında olası sayı çelişkileri
    """

    _YEAR_PATTERN = re.compile(
        r"\b(?:19|20)\d{2}\b",
    )

    _NUMBER_PATTERN = re.compile(
        r"(?<!\w)"
        r"[-+]?"
        r"(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)"
        r"(?:[.,]\d+)?"
        r"(?:\s*%)?"
        r"(?!\w)",
    )

    # Eksik/verilmemiş değer işaretleri sayısal artifact değildir.
    # "***3" gibi ifadelerde yıldızlar eksik değeri, sondaki sayı ise
    # dipnot numarasını temsil edebilir.
    _MISSING_VALUE_PATTERN = re.compile(
        r"(?:\*{2,}(?:\s*\d+)?|\bn/?a\b|\bnot disclosed\b|"
        r"\bnot available\b|\bnot reported\b|\bdata unavailable\b)",
        re.IGNORECASE,
    )

    _NUMERIC_ARTIFACT_PATTERN = re.compile(
        r"(?:ocr\s*error|unreadable\s*number|illegible\s*value)",
        re.IGNORECASE,
    )

    _STANDARD_PATTERN = re.compile(
        r"\b(?:"
        r"GRI\s+\d{1,3}(?:-\d{1,3})?"
        r"|ESRS\s+(?:1|2|E[1-5]|S[1-4]|G1)"
        r"|SASB\s+[A-Z0-9.-]+"
        r"|TCFD"
        r"|SDG\s+(?:[1-9]|1[0-7])"
        r")\b",
        re.IGNORECASE,
    )

    _METRIC_CONCEPT_ALIASES: Mapping[str, tuple[str, ...]] = {
        "nox": (
            "nox",
            "no x",
            "nitrogen oxide",
            "nitrogen oxides",
            "nitrogen oxide emissions",
            "nitrogen oxides emissions",
        ),
        "sox": (
            "sox",
            "so x",
            "sulfur oxide",
            "sulfur oxides",
            "sulphur oxide",
            "sulphur oxides",
            "sulfur oxide emissions",
            "sulphur oxide emissions",
        ),
        "voc": (
            "voc",
            "volatile organic compound",
            "volatile organic compounds",
            "volatile organic compound emissions",
        ),
        "pm": (
            "pm",
            "particulate matter",
            "particulate matter emissions",
        ),
        "pop": (
            "pop",
            "persistent organic pollutant",
            "persistent organic pollutants",
        ),
        "hap": (
            "hap",
            "hazardous air pollutant",
            "hazardous air pollutants",
        ),
    }

    _COMMON_UNITS = (
        "tco2e",
        "co2e",
        "kg co2e",
        "ton",
        "tonne",
        "tonnes",
        "metric ton",
        "metric tons",
        "kg",
        "g",
        "mg",
        "mwh",
        "gwh",
        "kwh",
        "gj",
        "mj",
        "m3",
        "m³",
        "litre",
        "liter",
        "litres",
        "liters",
        "megalitre",
        "megaliter",
        "hectare",
        "hectares",
        "ha",
        "percent",
        "percentage",
        "%",
        "oran",
        "adet",
        "kişi",
        "employee",
        "employees",
        "hour",
        "hours",
        "saat",
        "try",
        "tl",
        "usd",
        "eur",
    )

    def __init__(
        self,
        *,
        config: EvidenceValidatorConfig | None = None,
    ) -> None:
        self.config = config or EvidenceValidatorConfig()

    def validate(
        self,
        request: EvidenceValidationRequest,
        evidences: Iterable[Any],
    ) -> EvidenceValidationResult:
        cleaned_query = self._clean_text(request.query)

        if not cleaned_query:
            raise EvidenceValidatorError(
                "Kanıt doğrulama sorgusu boş olamaz.",
            )

        prepared_evidences = [
            self._validate_single(
                request=request,
                source=evidence,
            )
            for evidence in evidences
        ]

        if self.config.deduplicate_evidence:
            prepared_evidences = self._deduplicate(
                prepared_evidences,
            )

        prepared_evidences.sort(
            key=lambda evidence: (
                -evidence.score,
                evidence.page_number
                if evidence.page_number is not None
                else 10**9,
                evidence.evidence_id,
            ),
        )

        prepared_evidences = self._select_final_evidences(
            prepared_evidences,
            maximum_count=self.config.maximum_evidence_count,
            expects_structured=(
                request.expects_numeric_answer
                or request.expects_table_evidence
            ),
        )

        conflict_issues = self._detect_numeric_conflicts(
            request,
            prepared_evidences,
        )

        if conflict_issues:
            prepared_evidences = self._attach_conflict_issues(
                prepared_evidences,
                conflict_issues,
            )

        accepted = [
            evidence
            for evidence in prepared_evidences
            if evidence.status
            in {
                EvidenceValidationStatus.VALID,
                EvidenceValidationStatus.PARTIALLY_VALID,
            }
        ]

        rejected_count = len(prepared_evidences) - len(accepted)

        overall_score = self._calculate_overall_score(
            accepted,
        )

        overall_status = self._resolve_overall_status(
            accepted=accepted,
            overall_score=overall_score,
            has_conflicts=bool(conflict_issues),
        )

        all_issues = [
            issue
            for evidence in prepared_evidences
            for issue in evidence.issues
        ]
        all_issues.extend(conflict_issues)

        return EvidenceValidationResult(
            status=overall_status,
            score=round(overall_score, 6),
            evidences=tuple(prepared_evidences),
            valid_evidence_count=len(accepted),
            rejected_evidence_count=rejected_count,
            issues=tuple(self._deduplicate_issues(all_issues)),
            metadata={
                "total_evidence_count": len(prepared_evidences),
                "accepted_evidence_count": len(accepted),
                "rejected_evidence_count": rejected_count,
                "minimum_required_count": (
                    self.config.minimum_valid_evidence_count
                ),
                "has_numeric_conflict": bool(conflict_issues),
            },
        )

    def validate_query(
        self,
        query: str,
        evidences: Iterable[Any],
        *,
        report_id: str | None = None,
        years: Sequence[int] = (),
        metric_terms: Sequence[str] = (),
        units: Sequence[str] = (),
        standard_codes: Sequence[str] = (),
        category: str | None = None,
        expects_numeric_answer: bool = False,
        expects_table_evidence: bool = False,
    ) -> EvidenceValidationResult:
        request = EvidenceValidationRequest(
            query=self._clean_text(query),
            report_id=self._optional_string(report_id),
            years=tuple(
                sorted(
                    {
                        int(year)
                        for year in years
                        if self._is_valid_year(year)
                    },
                ),
            ),
            metric_terms=self._normalize_string_sequence(
                metric_terms,
            ),
            units=self._normalize_string_sequence(units),
            standard_codes=tuple(
                sorted(
                    {
                        self._normalize_standard_code(code)
                        for code in standard_codes
                        if self._clean_text(code)
                    },
                ),
            ),
            category=self._optional_string(category),
            expects_numeric_answer=expects_numeric_answer,
            expects_table_evidence=expects_table_evidence,
        )

        return self.validate(request, evidences)

    def _validate_single(
        self,
        *,
        request: EvidenceValidationRequest,
        source: Any,
    ) -> ValidatedEvidence:
        evidence_id = self._extract_identifier(source)
        payload = self._extract_payload(source)
        text = self._extract_text(source, payload)

        issues: list[EvidenceValidationIssue] = []

        if not text:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.EMPTY_CONTENT,
                    message="Kanıt içeriği boş.",
                    severity=1.0,
                    evidence_id=evidence_id,
                ),
            )

            return ValidatedEvidence(
                evidence_id=evidence_id,
                payload=payload,
                text="",
                score=0.0,
                status=EvidenceValidationStatus.INVALID,
                issues=tuple(issues),
                metadata=self._extract_metadata(source),
            )

        report_id = (
            self._optional_string(
                self._extract_first_value(
                    source,
                    ("report_id",),
                ),
            )
            or self._extract_report_id(payload)
        )

        page_number = (
            self._optional_int(
                self._extract_first_value(
                    source,
                    (
                        "page_number",
                        "source_page",
                        "page",
                    ),
                ),
            )
            or self._extract_page_number(payload)
        )

        row_index = self._optional_int(
            self._extract_first_value(
                source,
                ("row_index",),
            ),
        )

        section_title = (
            self._optional_string(
                self._extract_first_value(
                    source,
                    (
                        "section_title",
                        "section",
                        "heading",
                    ),
                ),
            )
            or self._extract_section_title(payload)
        )

        retrieval_score = self._extract_numeric_score(
            source,
            (
                "retrieval_score",
                "original_score",
                "fusion_score",
            ),
        )

        rerank_score = self._extract_numeric_score(
            source,
            (
                "rerank_score",
                "cross_encoder_score",
            ),
        )

        general_score = self._extract_numeric_score(
            source,
            ("score", "confidence"),
        )

        if retrieval_score == 0.0:
            retrieval_score = general_score

        if rerank_score == 0.0:
            rerank_score = general_score

        if retrieval_score < self.config.minimum_retrieval_score:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.LOW_RETRIEVAL_SCORE,
                    message=(
                        "Kanıt retrieval skoru minimum eşiğin altında."
                    ),
                    severity=0.55,
                    evidence_id=evidence_id,
                    metadata={
                        "score": retrieval_score,
                        "threshold": (
                            self.config.minimum_retrieval_score
                        ),
                    },
                ),
            )

        if rerank_score < self.config.minimum_rerank_score:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.LOW_RERANK_SCORE,
                    message=(
                        "Kanıt rerank skoru minimum eşiğin altında."
                    ),
                    severity=0.55,
                    evidence_id=evidence_id,
                    metadata={
                        "score": rerank_score,
                        "threshold": (
                            self.config.minimum_rerank_score
                        ),
                    },
                ),
            )

        is_structured_evidence = self._is_structured_evidence(
            source,
            payload,
            evidence_id=evidence_id,
            row_index=row_index,
        )
        has_missing_value_marker = (
            self._MISSING_VALUE_PATTERN.search(text) is not None
        )

        detected_years = self._extract_years(text)
        detected_units = self._extract_units(text)
        detected_standards = self._extract_standard_codes(text)
        numeric_values = self._extract_numeric_values(text)

        matched_years = self._matched_years(
            request.years,
            detected_years,
        )
        matched_metrics = self._matched_terms(
            request.metric_terms,
            text,
        )
        matched_units = self._matched_terms(
            request.units,
            text,
        )
        matched_standards = self._matched_standard_codes(
            request.standard_codes,
            detected_standards,
        )

        relevance_score = self._query_relevance(
            request.query,
            text,
        )

        if matched_metrics:
            relevance_score = max(
                relevance_score,
                0.35,
            )

        if relevance_score < self.config.minimum_relevance_score:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.QUERY_MISMATCH,
                    message="Kanıt soru ile yeterince ilişkili değil.",
                    severity=0.75,
                    evidence_id=evidence_id,
                    metadata={
                        "relevance_score": relevance_score,
                        "threshold": (
                            self.config.minimum_relevance_score
                        ),
                    },
                ),
            )

        report_score = self._report_match_score(
            expected=request.report_id,
            actual=report_id,
        )

        if (
            request.report_id is not None
            and report_score == 0.0
        ):
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.REPORT_MISMATCH,
                    message=(
                        "Kanıt beklenen rapora ait görünmüyor."
                    ),
                    severity=1.0,
                    evidence_id=evidence_id,
                    metadata={
                        "expected_report_id": request.report_id,
                        "actual_report_id": report_id,
                    },
                ),
            )

        if not request.years:
            year_score = 1.0
        elif not detected_years:
            year_score = 0.65
        else:
            year_score = self._collection_score(
                request.years,
                detected_years,
            )

        if request.years and detected_years and not matched_years:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.YEAR_MISMATCH,
                    message=(
                        "Kanıtta istenen raporlama yılı bulunamadı."
                    ),
                    severity=0.55,
                    evidence_id=evidence_id,
                    metadata={
                        "expected_years": request.years,
                        "detected_years": detected_years,
                    },
                ),
            )

        metric_score = self._term_collection_score(
            request.metric_terms,
            text,
        )

        if request.metric_terms and not matched_metrics:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.METRIC_MISMATCH,
                    message=(
                        "Kanıtta beklenen metrik veya gösterge bulunamadı."
                    ),
                    severity=0.7,
                    evidence_id=evidence_id,
                    metadata={
                        "expected_metrics": request.metric_terms,
                    },
                ),
            )

        if not request.units:
            unit_score = 1.0
        elif not detected_units:
            unit_score = 0.50
        else:
            unit_score = self._term_collection_score(
                request.units,
                text,
            )

        if request.units and detected_units and not matched_units:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.UNIT_MISMATCH,
                    message=(
                        "Kanıtta beklenen ölçü birimi bulunamadı."
                    ),
                    severity=0.45,
                    evidence_id=evidence_id,
                    metadata={
                        "expected_units": request.units,
                        "detected_units": detected_units,
                    },
                ),
            )

        standard_score = self._collection_score(
            {
                self._normalize_standard_code(code)
                for code in request.standard_codes
            },
            {
                self._normalize_standard_code(code)
                for code in detected_standards
            },
        )

        if request.standard_codes and not matched_standards:
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.STANDARD_MISMATCH,
                    message=(
                        "Kanıtta beklenen standart kodu bulunamadı."
                    ),
                    severity=0.6,
                    evidence_id=evidence_id,
                    metadata={
                        "expected_standard_codes": (
                            request.standard_codes
                        ),
                        "detected_standard_codes": (
                            detected_standards
                        ),
                    },
                ),
            )

        evidence_category = (
            self._optional_string(
                self._extract_first_value(
                    source,
                    ("category", "esg_category"),
                ),
            )
            or self._extract_category(payload)
        )

        category_score = self._category_match_score(
            request.category,
            evidence_category,
        )

        if (
            request.category is not None
            and category_score == 0.0
        ):
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.CATEGORY_MISMATCH,
                    message=(
                        "Kanıt beklenen ESG kategorisiyle eşleşmiyor."
                    ),
                    severity=0.45,
                    evidence_id=evidence_id,
                    metadata={
                        "expected_category": request.category,
                        "actual_category": evidence_category,
                    },
                ),
            )

        source_metadata = self._extract_metadata(source)
        source_quality = self._extract_source_quality(
            source_metadata
        )

        # Bazı PDF tabloları, tablo genelindeki OCR/normalizasyon sorunları
        # nedeniyle düşük kaynak kalite etiketi alabilir. Buna karşın kısa,
        # doğrudan metrik eşleşen ve sayısal değer taşıyan gerçek bir tablo
        # satırı güvenilir olabilir. Bu dar istisna yalnızca temiz satır
        # kanıtlarına uygulanır; bütün tablo nesnesi repr'leri ve uzun OCR
        # parçaları kapsam dışıdır.
        trusted_direct_numeric_row = bool(
            request.expects_numeric_answer
            and is_structured_evidence
            and row_index is not None
            and matched_metrics
            and numeric_values
            and len(text) <= 240
            and "normalizedtablecell" not in text.casefold()
            and "gri content index" not in text.casefold()
            and "annex 3" not in text.casefold()
        )

        if (
            request.expects_numeric_answer
            and source_quality["explicitly_invalid"]
            and not trusted_direct_numeric_row
        ):
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.LOW_SOURCE_QUALITY,
                    message=(
                        "Sayısal kanıtın kaynak tablosu doğrulama "
                        "aşamasında geçersiz işaretlenmiş."
                    ),
                    severity=0.95,
                    evidence_id=evidence_id,
                    metadata=source_quality,
                ),
            )

        if (
            request.expects_numeric_answer
            and source_quality["validation_score"] is not None
            and source_quality["validation_score"]
            < self.config.minimum_numeric_source_validation_score
            and not trusted_direct_numeric_row
        ):
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.LOW_SOURCE_QUALITY,
                    message=(
                        "Sayısal kanıtın kaynak kalite skoru minimum "
                        "eşiğin altında."
                    ),
                    severity=0.85,
                    evidence_id=evidence_id,
                    metadata={
                        **source_quality,
                        "threshold": (
                            self.config
                            .minimum_numeric_source_validation_score
                        ),
                    },
                ),
            )

        numeric_artifacts = self._extract_numeric_artifacts(text)

        if (
            request.expects_numeric_answer
            and numeric_artifacts
            and not has_missing_value_marker
        ):
            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.NUMERIC_ARTIFACT,
                    message=(
                        "Kanıtta dipnot veya OCR artığına benzeyen "
                        "sayısal ifadeler bulundu."
                    ),
                    severity=0.8,
                    evidence_id=evidence_id,
                    metadata={
                        "artifacts": numeric_artifacts,
                    },
                ),
            )

        if request.expects_numeric_answer and not numeric_values:
            if has_missing_value_marker:
                issues.append(
                    EvidenceValidationIssue(
                        issue_type=EvidenceIssueType.DISCLOSED_AS_MISSING,
                        message=(
                            "Kaynak, istenen sayısal değeri yıldız/N/A "
                            "gibi bir eksik-açıklama işaretiyle göstermektedir."
                        ),
                        severity=0.10,
                        evidence_id=evidence_id,
                        metadata={
                            "supports_non_disclosure": True,
                        },
                    ),
                )
            else:
                issues.append(
                    EvidenceValidationIssue(
                        issue_type=EvidenceIssueType.MISSING_NUMERIC_VALUE,
                        message=(
                            "Sayısal cevap beklenmesine rağmen kanıtta "
                            "sayısal değer bulunamadı."
                        ),
                        severity=0.85,
                        evidence_id=evidence_id,
                    ),
                )

        source_score = max(
            general_score,
            retrieval_score,
            rerank_score,
        )

        completeness_score = self._calculate_completeness(
            request=request,
            matched_years=matched_years,
            matched_metrics=matched_metrics,
            matched_units=matched_units,
            matched_standards=matched_standards,
            numeric_values=numeric_values,
            has_missing_value_marker=has_missing_value_marker,
        )

        consistency_score = self._calculate_consistency(
            issues,
        )

        final_score = self._combine_scores(
            query_score=relevance_score,
            metric_score=metric_score,
            year_score=year_score,
            unit_score=unit_score,
            standard_score=standard_score,
            report_score=report_score,
            category_score=category_score,
            source_score=source_score,
        )

        final_score *= consistency_score
        final_score = (
            final_score * 0.85
            + completeness_score * 0.15
        )

        if is_structured_evidence:
            final_score += self.config.structured_evidence_bonus

        if (
            has_missing_value_marker
            and matched_years
            and (
                matched_metrics
                or metric_score >= 0.50
            )
        ):
            final_score += self.config.missing_disclosure_bonus

        final_score = max(
            0.0,
            min(final_score, 1.0),
        )

        status = self._resolve_evidence_status(
            score=final_score,
            issues=issues,
            request=request,
        )

        return ValidatedEvidence(
            evidence_id=evidence_id,
            payload=payload,
            text=text,
            score=round(final_score, 6),
            status=status,
            retrieval_score=round(retrieval_score, 6),
            rerank_score=round(rerank_score, 6),
            relevance_score=round(relevance_score, 6),
            consistency_score=round(consistency_score, 6),
            completeness_score=round(completeness_score, 6),
            report_id=report_id,
            page_number=page_number,
            row_index=row_index,
            section_title=section_title,
            matched_years=matched_years,
            matched_metrics=matched_metrics,
            matched_units=matched_units,
            matched_standard_codes=matched_standards,
            numeric_values=numeric_values,
            issues=tuple(issues),
            metadata={
                **source_metadata,
                "detected_years": detected_years,
                "detected_units": detected_units,
                "detected_standard_codes": detected_standards,
                "category": evidence_category,
                "is_structured_evidence": is_structured_evidence,
                "trusted_direct_numeric_row": trusted_direct_numeric_row,
                "has_missing_value_marker": has_missing_value_marker,
                "supports_non_disclosure": bool(
                    has_missing_value_marker and matched_years
                ),
            },
        )

    def _select_final_evidences(
        self,
        evidences: Sequence[ValidatedEvidence],
        *,
        maximum_count: int,
        expects_structured: bool,
    ) -> list[ValidatedEvidence]:
        if maximum_count <= 0:
            return []

        selected: list[ValidatedEvidence] = []
        selected_ids: set[str] = set()
        table_counts: dict[str, int] = {}

        if (
            expects_structured
            and self.config.preserve_structured_evidence
        ):
            structured = [
                evidence
                for evidence in evidences
                if evidence.metadata.get("is_structured_evidence")
            ]
            structured.sort(
                key=lambda evidence: (
                    not bool(
                        evidence.metadata.get(
                            "supports_non_disclosure"
                        )
                    ),
                    -evidence.score,
                    evidence.row_index is None,
                )
            )

            for evidence in structured:
                base_id = evidence.evidence_id.split(
                    "::row:",
                    1,
                )[0]
                count = table_counts.get(base_id, 0)

                if count >= self.config.maximum_results_per_table:
                    continue

                selected.append(evidence)
                selected_ids.add(evidence.evidence_id)
                table_counts[base_id] = count + 1
                break

        for evidence in evidences:
            if len(selected) >= maximum_count:
                break

            if evidence.evidence_id in selected_ids:
                continue

            if evidence.metadata.get("is_structured_evidence"):
                base_id = evidence.evidence_id.split(
                    "::row:",
                    1,
                )[0]
                count = table_counts.get(base_id, 0)

                if count >= self.config.maximum_results_per_table:
                    continue

                table_counts[base_id] = count + 1

            selected.append(evidence)
            selected_ids.add(evidence.evidence_id)

        selected.sort(
            key=lambda evidence: (
                -evidence.score,
                evidence.page_number
                if evidence.page_number is not None
                else 10**9,
                evidence.evidence_id,
            )
        )
        return selected[:maximum_count]

    @classmethod
    def _is_structured_evidence(
        cls,
        source: Any,
        payload: Any,
        *,
        evidence_id: str,
        row_index: int | None,
    ) -> bool:
        if row_index is not None:
            return True

        if evidence_id.startswith("page-") and "-table-" in evidence_id:
            return True

        sources = cls._extract_first_value(
            source,
            ("sources", "source_types"),
        )
        if sources is not None:
            try:
                iterable = (
                    [sources]
                    if isinstance(sources, str)
                    else list(sources)
                )
            except TypeError:
                iterable = [sources]

            normalized_sources = {
                str(getattr(item, "value", item)).casefold()
                for item in iterable
            }
            if "table" in normalized_sources:
                return True

        metadata = cls._extract_metadata(source)
        for key in (
            "representative_source",
            "source_type",
            "evidence_type",
        ):
            value = metadata.get(key)
            if (
                value is not None
                and str(value).casefold() in {
                    "table",
                    "table_row",
                    "structured_table",
                }
            ):
                return True

        payload_id = cls._extract_first_value(
            payload,
            ("table_id", "id", "identifier"),
        )
        if payload_id is not None:
            normalized_id = str(payload_id)
            if (
                normalized_id.startswith("page-")
                and "-table-" in normalized_id
            ):
                return True

        return False

    def _combine_scores(
        self,
        *,
        query_score: float,
        metric_score: float,
        year_score: float,
        unit_score: float,
        standard_score: float,
        report_score: float,
        category_score: float,
        source_score: float,
    ) -> float:
        components = {
            "query": (
                query_score,
                self.config.query_weight,
            ),
            "metric": (
                metric_score,
                self.config.metric_weight,
            ),
            "year": (
                year_score,
                self.config.year_weight,
            ),
            "unit": (
                unit_score,
                self.config.unit_weight,
            ),
            "standard": (
                standard_score,
                self.config.standard_weight,
            ),
            "report": (
                report_score,
                self.config.report_weight,
            ),
            "category": (
                category_score,
                self.config.category_weight,
            ),
            "source": (
                source_score,
                self.config.source_score_weight,
            ),
        }

        total_weight = sum(
            weight
            for _, weight in components.values()
        )

        if total_weight <= 0:
            return 0.0

        weighted_score = sum(
            score * weight
            for score, weight in components.values()
        )

        return weighted_score / total_weight

    def _resolve_evidence_status(
        self,
        *,
        score: float,
        issues: Sequence[EvidenceValidationIssue],
        request: EvidenceValidationRequest,
    ) -> EvidenceValidationStatus:
        issue_types = {
            issue.issue_type
            for issue in issues
        }

        if EvidenceIssueType.EMPTY_CONTENT in issue_types:
            return EvidenceValidationStatus.INVALID

        if (
            self.config.require_report_match
            and EvidenceIssueType.REPORT_MISMATCH in issue_types
        ):
            return EvidenceValidationStatus.INVALID

        if (
            self.config.reject_year_mismatch
            and EvidenceIssueType.YEAR_MISMATCH in issue_types
        ):
            return EvidenceValidationStatus.INVALID

        if (
            self.config.reject_unit_mismatch
            and EvidenceIssueType.UNIT_MISMATCH in issue_types
        ):
            return EvidenceValidationStatus.INVALID

        if request.expects_numeric_answer:
            if (
                EvidenceIssueType.MISSING_NUMERIC_VALUE
                in issue_types
            ):
                return EvidenceValidationStatus.INSUFFICIENT

            if (
                self.config.require_metric_match_for_numeric
                and request.metric_terms
                and EvidenceIssueType.METRIC_MISMATCH
                in issue_types
            ):
                return EvidenceValidationStatus.INSUFFICIENT

            if (
                self.config.require_year_match_for_numeric
                and request.years
                and EvidenceIssueType.YEAR_MISMATCH
                in issue_types
            ):
                return EvidenceValidationStatus.INSUFFICIENT

            if (
                self.config.reject_numeric_artifacts
                and EvidenceIssueType.NUMERIC_ARTIFACT
                in issue_types
            ):
                return EvidenceValidationStatus.INSUFFICIENT

            if (
                self.config.reject_invalid_numeric_sources
                and EvidenceIssueType.LOW_SOURCE_QUALITY
                in issue_types
            ):
                return EvidenceValidationStatus.INSUFFICIENT

        if score >= self.config.minimum_evidence_score:
            return EvidenceValidationStatus.VALID

        if score >= self.config.partial_valid_threshold:
            return EvidenceValidationStatus.PARTIALLY_VALID

        return EvidenceValidationStatus.INSUFFICIENT

    def _resolve_overall_status(
        self,
        *,
        accepted: Sequence[ValidatedEvidence],
        overall_score: float,
        has_conflicts: bool,
    ) -> EvidenceValidationStatus:
        if len(accepted) < self.config.minimum_valid_evidence_count:
            return EvidenceValidationStatus.INSUFFICIENT

        if has_conflicts:
            return EvidenceValidationStatus.PARTIALLY_VALID

        if overall_score >= self.config.minimum_evidence_score:
            return EvidenceValidationStatus.VALID

        if overall_score >= self.config.partial_valid_threshold:
            return EvidenceValidationStatus.PARTIALLY_VALID

        return EvidenceValidationStatus.INSUFFICIENT

    @staticmethod
    def _calculate_consistency(
        issues: Sequence[EvidenceValidationIssue],
    ) -> float:
        if not issues:
            return 1.0

        penalty = sum(
            issue.severity * 0.06
            for issue in issues
        )

        return max(
            0.0,
            1.0 - min(penalty, 1.0),
        )

    @staticmethod
    def _calculate_completeness(
        *,
        request: EvidenceValidationRequest,
        matched_years: Sequence[int],
        matched_metrics: Sequence[str],
        matched_units: Sequence[str],
        matched_standards: Sequence[str],
        numeric_values: Sequence[str],
        has_missing_value_marker: bool = False,
    ) -> float:
        checks: list[bool] = []

        if request.years:
            checks.append(bool(matched_years))

        if request.metric_terms:
            checks.append(bool(matched_metrics))

        if request.units:
            checks.append(bool(matched_units))

        if request.standard_codes:
            checks.append(bool(matched_standards))

        if request.expects_numeric_answer:
            checks.append(
                bool(numeric_values) or has_missing_value_marker
            )

        if not checks:
            return 1.0

        return sum(checks) / len(checks)

    @staticmethod
    def _calculate_overall_score(
        evidences: Sequence[ValidatedEvidence],
    ) -> float:
        if not evidences:
            return 0.0

        weights = [
            1.0 / (index + 1)
            for index in range(len(evidences))
        ]

        total_weight = sum(weights)

        return sum(
            evidence.score * weight
            for evidence, weight in zip(
                evidences,
                weights,
                strict=True,
            )
        ) / total_weight

    def _detect_numeric_conflicts(
        self,
        request: EvidenceValidationRequest,
        evidences: Sequence[ValidatedEvidence],
    ) -> list[EvidenceValidationIssue]:
        if not request.expects_numeric_answer:
            return []

        accepted = [
            evidence
            for evidence in evidences
            if evidence.status
            in {
                EvidenceValidationStatus.VALID,
                EvidenceValidationStatus.PARTIALLY_VALID,
            }
            and evidence.numeric_values
        ]

        if len(accepted) < 2:
            return []

        numeric_groups: dict[
            tuple[int | None, str | None],
            list[tuple[str, float]],
        ] = {}

        for evidence in accepted:
            year = (
                evidence.matched_years[0]
                if evidence.matched_years
                else None
            )
            unit = (
                evidence.matched_units[0]
                if evidence.matched_units
                else None
            )

            for raw_value in evidence.numeric_values:
                parsed_value = self._parse_number(raw_value)

                if parsed_value is None:
                    continue

                numeric_groups.setdefault(
                    (year, unit),
                    [],
                ).append(
                    (
                        evidence.evidence_id,
                        parsed_value,
                    ),
                )

        issues: list[EvidenceValidationIssue] = []

        for key, values in numeric_groups.items():
            if len(values) < 2:
                continue

            numeric_values = [
                value
                for _, value in values
            ]

            minimum = min(numeric_values)
            maximum = max(numeric_values)

            denominator = max(
                abs(maximum),
                abs(minimum),
                1.0,
            )

            difference_ratio = (
                abs(maximum - minimum)
                / denominator
            )

            if difference_ratio <= self.config.conflict_tolerance:
                continue

            issues.append(
                EvidenceValidationIssue(
                    issue_type=EvidenceIssueType.CONFLICTING_VALUES,
                    message=(
                        "Aynı yıl ve birim için farklı sayısal "
                        "değerler tespit edildi."
                    ),
                    severity=0.8,
                    metadata={
                        "year": key[0],
                        "unit": key[1],
                        "values": tuple(values),
                        "difference_ratio": round(
                            difference_ratio,
                            6,
                        ),
                    },
                ),
            )

        return issues

    @staticmethod
    def _attach_conflict_issues(
        evidences: Sequence[ValidatedEvidence],
        conflict_issues: Sequence[EvidenceValidationIssue],
    ) -> list[ValidatedEvidence]:
        if not conflict_issues:
            return list(evidences)

        updated: list[ValidatedEvidence] = []

        for evidence in evidences:
            related_issues = [
                issue
                for issue in conflict_issues
                if any(
                    evidence.evidence_id == evidence_id
                    for evidence_id, _ in issue.metadata.get(
                        "values",
                        (),
                    )
                )
            ]

            if not related_issues:
                updated.append(evidence)
                continue

            new_score = max(
                0.0,
                evidence.score - 0.08,
            )

            new_status = (
                EvidenceValidationStatus.PARTIALLY_VALID
                if evidence.status == EvidenceValidationStatus.VALID
                else evidence.status
            )

            updated.append(
                ValidatedEvidence(
                    evidence_id=evidence.evidence_id,
                    payload=evidence.payload,
                    text=evidence.text,
                    score=round(new_score, 6),
                    status=new_status,
                    retrieval_score=evidence.retrieval_score,
                    rerank_score=evidence.rerank_score,
                    relevance_score=evidence.relevance_score,
                    consistency_score=max(
                        0.0,
                        evidence.consistency_score - 0.1,
                    ),
                    completeness_score=evidence.completeness_score,
                    report_id=evidence.report_id,
                    page_number=evidence.page_number,
                    row_index=evidence.row_index,
                    section_title=evidence.section_title,
                    matched_years=evidence.matched_years,
                    matched_metrics=evidence.matched_metrics,
                    matched_units=evidence.matched_units,
                    matched_standard_codes=(
                        evidence.matched_standard_codes
                    ),
                    numeric_values=evidence.numeric_values,
                    issues=tuple(
                        [
                            *evidence.issues,
                            *related_issues,
                        ],
                    ),
                    metadata=evidence.metadata,
                ),
            )

        return updated

    @staticmethod
    def _deduplicate(
        evidences: Sequence[ValidatedEvidence],
    ) -> list[ValidatedEvidence]:
        deduplicated: dict[
            tuple[str, str],
            ValidatedEvidence,
        ] = {}

        for evidence in evidences:
            normalized_text = EvidenceValidator._normalize_text(
                evidence.text,
            )

            key = (
                evidence.report_id or "",
                normalized_text,
            )

            existing = deduplicated.get(key)

            if (
                existing is None
                or evidence.score > existing.score
            ):
                deduplicated[key] = evidence

        return list(deduplicated.values())

    @staticmethod
    def _deduplicate_issues(
        issues: Sequence[EvidenceValidationIssue],
    ) -> list[EvidenceValidationIssue]:
        deduplicated: dict[
            tuple[EvidenceIssueType, str, str | None],
            EvidenceValidationIssue,
        ] = {}

        for issue in issues:
            key = (
                issue.issue_type,
                issue.message,
                issue.evidence_id,
            )

            existing = deduplicated.get(key)

            if (
                existing is None
                or issue.severity > existing.severity
            ):
                deduplicated[key] = issue

        return list(deduplicated.values())

    @classmethod
    def _query_relevance(
        cls,
        query: str,
        evidence_text: str,
    ) -> float:
        query_terms = set(
            cls._tokenize(query),
        )
        evidence_terms = set(
            cls._tokenize(evidence_text),
        )

        if not query_terms or not evidence_terms:
            return 0.0

        overlap = query_terms.intersection(evidence_terms)

        if not overlap:
            return 0.0

        recall = len(overlap) / len(query_terms)
        precision = len(overlap) / len(evidence_terms)

        if recall + precision == 0:
            return 0.0

        f1_score = (
            2 * recall * precision
            / (recall + precision)
        )

        normalized_query = cls._normalize_text(query)
        normalized_evidence = cls._normalize_text(evidence_text)

        if normalized_query in normalized_evidence:
            f1_score += 0.2

        return min(f1_score, 1.0)

    @classmethod
    def _shared_metric_concepts(
        cls,
        first_text: str,
        second_text: str,
    ) -> set[str]:
        first = cls._normalize_text(first_text)
        second = cls._normalize_text(second_text)

        shared: set[str] = set()

        for concept, aliases in cls._METRIC_CONCEPT_ALIASES.items():
            normalized_aliases = tuple(
                cls._normalize_text(alias)
                for alias in aliases
            )

            first_match = any(
                cls._contains_metric_alias(first, alias)
                for alias in normalized_aliases
            )
            second_match = any(
                cls._contains_metric_alias(second, alias)
                for alias in normalized_aliases
            )

            if first_match and second_match:
                shared.add(concept)

        return shared

    @staticmethod
    def _contains_metric_alias(
        normalized_text: str,
        normalized_alias: str,
    ) -> bool:
        if not normalized_text or not normalized_alias:
            return False

        if " " in normalized_alias:
            return normalized_alias in normalized_text

        return re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
            normalized_text,
        ) is not None

    @classmethod
    def _term_collection_score(
        cls,
        expected_terms: Sequence[str],
        text: str,
    ) -> float:
        if not expected_terms:
            return 1.0

        normalized_text = cls._normalize_text(text)
        text_tokens = set(cls._tokenize(normalized_text))
        scores: list[float] = []

        for term in expected_terms:
            normalized_term = cls._normalize_text(term)
            if not normalized_term:
                continue

            if cls._shared_metric_concepts(
                normalized_term,
                normalized_text,
            ):
                scores.append(1.0)
                continue

            if normalized_term in normalized_text:
                scores.append(1.0)
                continue

            term_tokens = set(cls._tokenize(normalized_term))
            if not term_tokens:
                continue

            overlap = len(term_tokens & text_tokens) / len(term_tokens)
            scores.append(overlap)

        return max(scores) if scores else 1.0

    @staticmethod
    def _collection_score(
        expected: Iterable[Any],
        actual: Iterable[Any],
    ) -> float:
        expected_set = set(expected)

        if not expected_set:
            return 1.0

        actual_set = set(actual)

        return (
            len(expected_set.intersection(actual_set))
            / len(expected_set)
        )

    @classmethod
    def _matched_terms(
        cls,
        expected_terms: Sequence[str],
        text: str,
    ) -> tuple[str, ...]:
        normalized_text = cls._normalize_text(text)
        text_tokens = set(cls._tokenize(normalized_text))
        matched: set[str] = set()

        for term in expected_terms:
            normalized_term = cls._normalize_text(term)
            if not normalized_term:
                continue

            if cls._shared_metric_concepts(
                normalized_term,
                normalized_text,
            ):
                matched.add(term)
                continue

            if normalized_term in normalized_text:
                matched.add(term)
                continue

            term_tokens = set(cls._tokenize(normalized_term))
            if not term_tokens:
                continue

            overlap = len(term_tokens & text_tokens) / len(term_tokens)
            if overlap >= 0.50:
                matched.add(term)

        return tuple(sorted(matched))

    @staticmethod
    def _matched_years(
        expected_years: Sequence[int],
        detected_years: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(expected_years).intersection(
                    detected_years,
                ),
            ),
        )

    @classmethod
    def _matched_standard_codes(
        cls,
        expected_codes: Sequence[str],
        detected_codes: Sequence[str],
    ) -> tuple[str, ...]:
        detected_map = {
            cls._normalize_standard_code(code): code
            for code in detected_codes
        }

        matched: set[str] = set()

        for code in expected_codes:
            normalized = cls._normalize_standard_code(code)

            if normalized in detected_map:
                matched.add(code)

        return tuple(sorted(matched))

    @staticmethod
    def _report_match_score(
        expected: str | None,
        actual: str | None,
    ) -> float:
        if expected is None:
            return 1.0

        if actual is None:
            return 0.50

        expected_normalized = EvidenceValidator._normalize_text(expected)
        actual_normalized = EvidenceValidator._normalize_text(actual)

        return 1.0 if expected_normalized == actual_normalized else 0.0

    @classmethod
    def _category_match_score(
        cls,
        expected: str | None,
        actual: str | None,
    ) -> float:
        if expected is None:
            return 1.0

        if actual is None:
            return 0.0

        return (
            1.0
            if cls._normalize_text(expected)
            == cls._normalize_text(actual)
            else 0.0
        )

    @classmethod
    def _extract_text(
        cls,
        source: Any,
        payload: Any,
    ) -> str:
        value = cls._extract_first_value(
            source,
            (
                "text",
                "combined_text",
                "matched_text",
                "content",
                "chunk_text",
            ),
        )

        if value is not None:
            return cls._clean_text(value)

        value = cls._extract_first_value(
            payload,
            (
                "text",
                "combined_text",
                "matched_text",
                "content",
                "chunk_text",
            ),
        )

        if value is not None:
            return cls._clean_text(value)

        table_text = cls._build_table_text(payload)

        if table_text:
            return table_text

        return ""

    @staticmethod
    def _extract_payload(source: Any) -> Any:
        for name in (
            "payload",
            "document",
            "table",
            "item",
            "record",
        ):
            value = EvidenceValidator._extract_first_value(
                source,
                (name,),
            )

            if value is not None:
                return value

        return source

    @classmethod
    def _extract_identifier(
        cls,
        source: Any,
    ) -> str:
        value = cls._extract_first_value(
            source,
            (
                "evidence_id",
                "candidate_id",
                "document_id",
                "chunk_id",
                "table_id",
                "context_id",
                "target_id",
                "id",
                "identifier",
            ),
        )

        if value is None:
            payload = cls._extract_payload(source)
            value = cls._extract_first_value(
                payload,
                (
                    "evidence_id",
                    "candidate_id",
                    "document_id",
                    "chunk_id",
                    "table_id",
                    "id",
                    "identifier",
                ),
            )

        if value is None:
            raise EvidenceValidatorError(
                "Kanıt için kimlik alanı bulunamadı.",
            )

        cleaned = str(value).strip()

        if not cleaned:
            raise EvidenceValidatorError(
                "Kanıt kimliği boş olamaz.",
            )

        return cleaned

    @classmethod
    def _extract_report_id(
        cls,
        source: Any,
    ) -> str | None:
        value = cls._extract_first_value(
            source,
            ("report_id",),
        )

        if value is None:
            value = cls._extract_metadata(source).get(
                "report_id",
            )

        return cls._optional_string(value)

    @classmethod
    def _extract_page_number(
        cls,
        source: Any,
    ) -> int | None:
        value = cls._extract_first_value(
            source,
            (
                "page_number",
                "source_page",
                "page",
            ),
        )

        if value is None:
            metadata = cls._extract_metadata(source)
            value = (
                metadata.get("page_number")
                or metadata.get("source_page")
                or metadata.get("page")
            )

        return cls._optional_int(value)

    @classmethod
    def _extract_section_title(
        cls,
        source: Any,
    ) -> str | None:
        value = cls._extract_first_value(
            source,
            (
                "section_title",
                "section",
                "heading",
            ),
        )

        if value is None:
            metadata = cls._extract_metadata(source)
            value = (
                metadata.get("section_title")
                or metadata.get("section")
                or metadata.get("heading")
            )

        return cls._optional_string(value)

    @classmethod
    def _extract_category(
        cls,
        source: Any,
    ) -> str | None:
        value = cls._extract_first_value(
            source,
            (
                "category",
                "esg_category",
            ),
        )

        if value is None:
            metadata = cls._extract_metadata(source)
            value = (
                metadata.get("category")
                or metadata.get("esg_category")
            )

        return cls._optional_string(value)

    @staticmethod
    def _extract_numeric_score(
        source: Any,
        names: Sequence[str],
    ) -> float:
        value = EvidenceValidator._extract_first_value(
            source,
            names,
        )

        if value is None:
            return 0.0

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return 0.0

        return max(
            0.0,
            min(numeric_value, 1.0),
        )

    @classmethod
    def _extract_years(
        cls,
        text: str,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    int(match.group(0))
                    for match in cls._YEAR_PATTERN.finditer(text)
                },
            ),
        )

    @classmethod
    def _extract_numeric_values(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        values: set[str] = set()

        missing_spans = [
            marker.span()
            for marker in cls._MISSING_VALUE_PATTERN.finditer(text)
        ]

        for match in cls._NUMBER_PATTERN.finditer(text):
            if any(
                start <= match.start() < end
                for start, end in missing_spans
            ):
                continue

            value = match.group(0).strip()
            compact = value.replace(" ", "")

            if cls._YEAR_PATTERN.fullmatch(compact):
                continue

            left = text[max(0, match.start() - 4):match.start()]
            right = text[match.end():match.end() + 4]
            neighborhood = f"{left}{value}{right}"

            if cls._NUMERIC_ARTIFACT_PATTERN.search(
                neighborhood
            ):
                continue

            values.add(value)

        return tuple(sorted(values))

    @classmethod
    def _extract_numeric_artifacts(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    match.group(0).strip()
                    for match in cls._NUMERIC_ARTIFACT_PATTERN.finditer(
                        text
                    )
                }
            )
        )

    @staticmethod
    def _extract_source_quality(
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        numeric_validation = metadata.get(
            "numeric_validation"
        )

        if not isinstance(numeric_validation, Mapping):
            numeric_validation = {}

        explicit_values = (
            metadata.get("is_valid"),
            numeric_validation.get("is_valid"),
        )
        explicitly_invalid = any(
            value is False
            for value in explicit_values
        )

        raw_score = (
            numeric_validation.get("validation_score")
            if numeric_validation
            else metadata.get("validation_score")
        )

        validation_score: float | None

        try:
            validation_score = (
                float(raw_score)
                if raw_score is not None
                else None
            )
        except (TypeError, ValueError):
            validation_score = None

        if validation_score is not None:
            validation_score = max(
                0.0,
                min(validation_score, 1.0),
            )

        return {
            "explicitly_invalid": explicitly_invalid,
            "validation_score": validation_score,
            "source_is_valid": metadata.get("is_valid"),
            "numeric_validation_is_valid": (
                numeric_validation.get("is_valid")
            ),
        }

    @classmethod
    def _extract_standard_codes(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    cls._normalize_standard_code(
                        match.group(0),
                    )
                    for match in cls._STANDARD_PATTERN.finditer(
                        text,
                    )
                },
            ),
        )

    @classmethod
    def _extract_units(
        cls,
        text: str,
    ) -> tuple[str, ...]:
        normalized_text = cls._normalize_text(text)

        return tuple(
            sorted(
                {
                    unit
                    for unit in cls._COMMON_UNITS
                    if cls._normalize_text(unit)
                    in normalized_text
                },
            ),
        )

    @staticmethod
    def _parse_number(
        value: str,
    ) -> float | None:
        cleaned = (
            value.strip()
            .replace("%", "")
            .replace(" ", "")
        )

        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = (
                    cleaned.replace(".", "")
                    .replace(",", ".")
                )
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            comma_parts = cleaned.split(",")

            if (
                len(comma_parts) == 2
                and len(comma_parts[1]) <= 2
            ):
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "." in cleaned:
            dot_parts = cleaned.split(".")

            if (
                len(dot_parts) > 2
                or (
                    len(dot_parts) == 2
                    and len(dot_parts[1]) == 3
                )
            ):
                cleaned = cleaned.replace(".", "")

        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def _build_table_text(
        cls,
        table: Any,
    ) -> str:
        if table is None:
            return ""

        parts: list[str] = []

        title = cls._extract_first_value(
            table,
            (
                "title",
                "caption",
                "table_title",
                "name",
            ),
        )

        if title:
            parts.append(cls._clean_text(title))

        headers = cls._extract_first_value(
            table,
            (
                "headers",
                "column_headers",
                "columns",
            ),
        )

        if headers is not None:
            header_values = cls._to_string_sequence(headers)

            if header_values:
                parts.append(" | ".join(header_values))

        rows = cls._extract_first_value(
            table,
            (
                "rows",
                "data",
                "records",
                "table_rows",
            ),
        )

        if rows is not None and not isinstance(
            rows,
            (str, bytes),
        ):
            try:
                iterable_rows = list(rows)
            except TypeError:
                iterable_rows = []

            for row in iterable_rows:
                row_values = cls._extract_row_values(row)

                if row_values:
                    parts.append(
                        " | ".join(row_values),
                    )

        return "\n".join(
            part
            for part in parts
            if part
        )

    @classmethod
    def _extract_row_values(
        cls,
        row: Any,
    ) -> tuple[str, ...]:
        if row is None:
            return ()

        if isinstance(row, Mapping):
            values = row.values()
        elif isinstance(row, (str, bytes)):
            cleaned = cls._clean_text(row)
            return (cleaned,) if cleaned else ()
        else:
            cells = getattr(row, "cells", None)

            if cells is not None:
                values = cells
            else:
                try:
                    values = list(row)
                except TypeError:
                    values = [row]

        result: list[str] = []

        for value in values:
            if isinstance(value, Mapping):
                resolved = (
                    value.get("normalized_value")
                    or value.get("value")
                    or value.get("text")
                    or value.get("content")
                )
            else:
                resolved = getattr(
                    value,
                    "normalized_value",
                    getattr(
                        value,
                        "value",
                        getattr(
                            value,
                            "text",
                            getattr(value, "content", value),
                        ),
                    ),
                )

            cleaned = cls._clean_text(resolved)

            if cleaned:
                result.append(cleaned)

        return tuple(result)

    @staticmethod
    def _extract_metadata(
        source: Any,
    ) -> dict[str, Any]:
        if isinstance(source, Mapping):
            metadata = source.get("metadata")
        else:
            metadata = getattr(source, "metadata", None)

        if metadata is None:
            return {}

        if isinstance(metadata, Mapping):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(
                metadata.model_dump(exclude_none=True),
            )

        if hasattr(metadata, "dict"):
            return dict(
                metadata.dict(exclude_none=True),
            )

        return {}

    @staticmethod
    def _extract_first_value(
        source: Any,
        names: Sequence[str],
    ) -> Any:
        if source is None:
            return None

        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
                value = getattr(source, name, None)

            if value is not None and not callable(value):
                return getattr(value, "value", value)

        return None

    @classmethod
    def _to_string_sequence(
        cls,
        value: Any,
    ) -> tuple[str, ...]:
        if value is None:
            return ()

        if isinstance(value, str):
            cleaned = cls._clean_text(value)
            return (cleaned,) if cleaned else ()

        if isinstance(value, Mapping):
            iterable = value.keys()
        else:
            try:
                iterable = list(value)
            except TypeError:
                iterable = [value]

        result: list[str] = []

        for item in iterable:
            if isinstance(item, Mapping):
                resolved = (
                    item.get("name")
                    or item.get("title")
                    or item.get("label")
                    or item.get("text")
                    or item.get("value")
                )
            else:
                resolved = getattr(
                    item,
                    "name",
                    getattr(
                        item,
                        "title",
                        getattr(
                            item,
                            "label",
                            getattr(
                                item,
                                "text",
                                getattr(item, "value", item),
                            ),
                        ),
                    ),
                )

            cleaned = cls._clean_text(resolved)

            if cleaned:
                result.append(cleaned)

        return tuple(result)

    @staticmethod
    def _normalize_string_sequence(
        values: Iterable[Any],
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    EvidenceValidator._clean_text(value)
                    for value in values
                    if EvidenceValidator._clean_text(value)
                },
            ),
        )

    @staticmethod
    def _normalize_standard_code(
        value: str,
    ) -> str:
        return " ".join(
            str(value).strip().upper().split(),
        )

    @staticmethod
    def _is_valid_year(
        value: Any,
    ) -> bool:
        try:
            year = int(value)
        except (TypeError, ValueError):
            return False

        return 1900 <= year <= 2100

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tokenize(
        text: str,
    ) -> tuple[str, ...]:
        normalized = EvidenceValidator._normalize_text(text)

        return tuple(
            token
            for token in re.findall(
                r"[a-z0-9]+(?:[./-][a-z0-9]+)*",
                normalized,
            )
            if len(token) >= 2
        )

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        enum_value = getattr(value, "value", value)
        if callable(enum_value):
            return ""

        return " ".join(
            str(enum_value).replace("\n", " ").split(),
        )

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
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

        normalized = str(value).translate(
            translation_table,
        ).casefold()

        normalized = re.sub(
            r"[^\w\s%./+-]",
            " ",
            normalized,
        )

        normalized = " ".join(normalized.split())

        abbreviation_patterns = {
            r"\bno\s+x\b": "nox",
            r"\bso\s+x\b": "sox",
            r"\bco\s+2\b": "co2",
        }

        for pattern, replacement in abbreviation_patterns.items():
            normalized = re.sub(
                pattern,
                replacement,
                normalized,
            )

        return normalized


__all__ = [
    "EvidenceIssueType",
    "EvidenceValidationIssue",
    "EvidenceValidationRequest",
    "EvidenceValidationResult",
    "EvidenceValidationStatus",
    "EvidenceValidator",
    "EvidenceValidatorConfig",
    "EvidenceValidatorError",
    "ValidatedEvidence",
]