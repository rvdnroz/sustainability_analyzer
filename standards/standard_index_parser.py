from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


class StandardType(StrEnum):
    GRI = "gri"
    ESRS = "esrs"
    SASB = "sasb"
    TCFD = "tcfd"
    SDG = "sdg"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class StandardIndexRow:
    standard_type: StandardType
    disclosure_code: str
    disclosure_title: str | None = None
    page_references: tuple[int, ...] = ()
    section_references: tuple[str, ...] = ()
    omissions: str | None = None
    raw_cells: tuple[str, ...] = ()
    source_page: int | None = None
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class StandardIndexParseResult:
    rows: tuple[StandardIndexRow, ...]
    detected_standard_types: tuple[StandardType, ...]
    source_page: int | None = None
    total_input_rows: int = 0
    parsed_row_count: int = 0
    skipped_row_count: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.rows


class StandardIndexParserError(RuntimeError):
    """Standart indeks ayrıştırma işlemlerindeki temel hata."""


class StandardIndexParser:
    """
    Sürdürülebilirlik raporlarındaki standart indeks tablolarını ayrıştırır.

    Desteklenen yapılar:
    - GRI içerik indeksleri
    - ESRS açıklama gereklilikleri indeksleri
    - SASB standart tabloları
    - TCFD öneri indeksleri
    - SDG eşleştirme tabloları

    Parser; tablo satırlarını disclosure kodu, başlık, sayfa referansı,
    bölüm referansı ve omission bilgilerine dönüştürür.
    """

    _GRI_CODE_PATTERN = re.compile(
        r"\b(?:GRI\s*)?(?P<code>"
        r"(?:2|3)-\d{1,2}"
        r"|(?:20[1-7]|30[1-8]|40[1-9]|41[1-9])-?\d{1,2}"
        r")\b",
        re.IGNORECASE,
    )

    _ESRS_CODE_PATTERN = re.compile(
        r"\bESRS\s+"
        r"(?P<standard>1|2|E[1-5]|S[1-4]|G1)"
        r"(?:\s*[-–—:]\s*|\s+)"
        r"(?P<requirement>"
        r"DR\s+[A-Z0-9-]+"
        r"|[A-Z]{2,5}-\d+"
        r")\b",
        re.IGNORECASE,
    )

    _ESRS_STANDARD_PATTERN = re.compile(
        r"\bESRS\s+(?P<standard>1|2|E[1-5]|S[1-4]|G1)\b",
        re.IGNORECASE,
    )

    _SASB_CODE_PATTERN = re.compile(
        r"\b(?P<code>[A-Z]{2}-[A-Z]{2}-\d{3}[a-z]?\.\d+)\b",
        re.IGNORECASE,
    )

    _SDG_PATTERN = re.compile(
        r"\b(?:SDG|SUSTAINABLE\s+DEVELOPMENT\s+GOAL)"
        r"\s*(?P<number>1[0-7]|[1-9])\b",
        re.IGNORECASE,
    )

    _PAGE_RANGE_PATTERN = re.compile(
        r"(?<!\d)(?P<start>\d{1,4})"
        r"\s*(?:[-–—]\s*(?P<end>\d{1,4}))?(?!\d)",
    )

    _HEADER_TERMS = {
        "standard",
        "disclosure",
        "disclosure number",
        "disclosure title",
        "requirement",
        "reporting requirement",
        "page",
        "page number",
        "page reference",
        "location",
        "reference",
        "omission",
        "reason for omission",
        "gri standard",
        "gri disclosure",
        "esrs disclosure requirement",
        "sasb code",
        "açıklama",
        "açıklama kodu",
        "standart",
        "sayfa",
        "sayfa numarası",
        "referans",
        "konum",
        "hariç tutma",
        "eksiklik",
    }

    _OMISSION_TERMS = (
        "omitted",
        "omission",
        "not reported",
        "not applicable",
        "not available",
        "confidential",
        "proprietary",
        "uygulanamaz",
        "raporlanmamıştır",
        "raporlanmadı",
        "mevcut değil",
        "hariç tutulmuştur",
        "gizli bilgi",
    )

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.45,
        maximum_page_number: int = 5000,
        expand_page_ranges: bool = True,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence 0 ile 1 arasında olmalıdır.",
            )

        if maximum_page_number <= 0:
            raise ValueError(
                "maximum_page_number pozitif olmalıdır.",
            )

        self.minimum_confidence = minimum_confidence
        self.maximum_page_number = maximum_page_number
        self.expand_page_ranges = expand_page_ranges

    def parse(
        self,
        rows: Iterable[Sequence[Any] | Mapping[str, Any]],
        *,
        source_page: int | None = None,
        standard_hint: StandardType | str | None = None,
    ) -> StandardIndexParseResult:
        normalized_rows = [
            self._normalize_input_row(row)
            for row in rows
        ]

        normalized_rows = [
            row
            for row in normalized_rows
            if any(cell for cell in row)
        ]

        if not normalized_rows:
            return StandardIndexParseResult(
                rows=(),
                detected_standard_types=(),
                source_page=source_page,
            )

        hint = self._normalize_standard_type(standard_hint)
        header_map, data_start_index = self._detect_header(
            normalized_rows,
        )

        parsed_rows: list[StandardIndexRow] = []
        warnings: list[str] = []
        skipped_count = 0

        for row_index, cells in enumerate(
            normalized_rows[data_start_index:],
            start=data_start_index,
        ):
            try:
                parsed = self.parse_row(
                    cells,
                    source_page=source_page,
                    standard_hint=hint,
                    header_map=header_map,
                )
            except StandardIndexParserError as exc:
                skipped_count += 1
                warnings.append(
                    f"Satır {row_index + 1} ayrıştırılamadı: {exc}",
                )
                continue

            if parsed is None:
                skipped_count += 1
                continue

            if parsed.confidence < self.minimum_confidence:
                skipped_count += 1
                warnings.append(
                    f"Satır {row_index + 1} düşük güven nedeniyle "
                    f"atlandı: {parsed.confidence:.2f}",
                )
                continue

            parsed_rows.append(parsed)

        deduplicated_rows = self._deduplicate_rows(parsed_rows)

        detected_types = tuple(
            sorted(
                {
                    row.standard_type
                    for row in deduplicated_rows
                },
                key=lambda item: item.value,
            ),
        )

        return StandardIndexParseResult(
            rows=tuple(deduplicated_rows),
            detected_standard_types=detected_types,
            source_page=source_page,
            total_input_rows=len(normalized_rows),
            parsed_row_count=len(deduplicated_rows),
            skipped_row_count=skipped_count,
            warnings=tuple(warnings),
        )

    def parse_row(
        self,
        cells: Sequence[Any],
        *,
        source_page: int | None = None,
        standard_hint: StandardType | str | None = None,
        header_map: Mapping[str, int] | None = None,
    ) -> StandardIndexRow | None:
        normalized_cells = tuple(
            self._normalize_text(cell)
            for cell in cells
        )

        if not any(normalized_cells):
            return None

        if self._is_header_row(normalized_cells):
            return None

        combined_text = " | ".join(
            cell
            for cell in normalized_cells
            if cell
        )

        hint = self._normalize_standard_type(standard_hint)
        standard_type = self.detect_standard_type(
            combined_text,
            hint=hint,
        )

        code = self._extract_code(
            combined_text,
            standard_type,
        )

        if code is None and header_map:
            code_index = header_map.get("code")

            if (
                code_index is not None
                and code_index < len(normalized_cells)
            ):
                code_cell = normalized_cells[code_index]
                standard_type = self.detect_standard_type(
                    code_cell,
                    hint=standard_type,
                )
                code = self._extract_code(
                    code_cell,
                    standard_type,
                )

        if code is None:
            return None

        title = self._extract_title(
            normalized_cells,
            disclosure_code=code,
            header_map=header_map,
        )

        page_references = self._extract_page_references(
            normalized_cells,
            header_map=header_map,
            disclosure_code=code,
        )

        section_references = self._extract_section_references(
            normalized_cells,
            title=title,
            disclosure_code=code,
            header_map=header_map,
        )

        omission = self._extract_omission(
            normalized_cells,
            header_map=header_map,
        )

        confidence = self._calculate_confidence(
            standard_type=standard_type,
            disclosure_code=code,
            title=title,
            page_references=page_references,
            section_references=section_references,
            omission=omission,
        )

        return StandardIndexRow(
            standard_type=standard_type,
            disclosure_code=code,
            disclosure_title=title,
            page_references=page_references,
            section_references=section_references,
            omissions=omission,
            raw_cells=normalized_cells,
            source_page=source_page,
            confidence=confidence,
            metadata={
                "has_page_reference": bool(page_references),
                "has_section_reference": bool(section_references),
                "has_omission": omission is not None,
            },
        )

    def detect_standard_type(
        self,
        text: str,
        *,
        hint: StandardType | str | None = None,
    ) -> StandardType:
        normalized_hint = self._normalize_standard_type(hint)
        normalized_text = self._normalize_text(text)

        if self._ESRS_CODE_PATTERN.search(normalized_text):
            return StandardType.ESRS

        if self._ESRS_STANDARD_PATTERN.search(normalized_text):
            return StandardType.ESRS

        if self._SASB_CODE_PATTERN.search(normalized_text):
            return StandardType.SASB

        if self._SDG_PATTERN.search(normalized_text):
            return StandardType.SDG

        if (
            "tcfd" in normalized_text.casefold()
            or "task force on climate-related financial disclosures"
            in normalized_text.casefold()
        ):
            return StandardType.TCFD

        if (
            "gri" in normalized_text.casefold()
            or self._GRI_CODE_PATTERN.search(normalized_text)
        ):
            return StandardType.GRI

        return normalized_hint or StandardType.UNKNOWN

    def parse_text_lines(
        self,
        lines: Iterable[str],
        *,
        source_page: int | None = None,
        standard_hint: StandardType | str | None = None,
    ) -> StandardIndexParseResult:
        rows = [
            self._split_text_line(line)
            for line in lines
            if self._normalize_text(line)
        ]

        return self.parse(
            rows,
            source_page=source_page,
            standard_hint=standard_hint,
        )

    def group_by_standard(
        self,
        rows: Iterable[StandardIndexRow],
    ) -> dict[StandardType, list[StandardIndexRow]]:
        grouped: dict[StandardType, list[StandardIndexRow]] = {}

        for row in rows:
            grouped.setdefault(row.standard_type, []).append(row)

        return grouped

    def find_by_code(
        self,
        rows: Iterable[StandardIndexRow],
        disclosure_code: str,
    ) -> list[StandardIndexRow]:
        normalized_target = self._normalize_code_for_comparison(
            disclosure_code,
        )

        return [
            row
            for row in rows
            if self._normalize_code_for_comparison(
                row.disclosure_code,
            )
            == normalized_target
        ]

    def _extract_code(
        self,
        text: str,
        standard_type: StandardType,
    ) -> str | None:
        if standard_type == StandardType.ESRS:
            match = self._ESRS_CODE_PATTERN.search(text)

            if match:
                standard = match.group("standard").upper()
                requirement = re.sub(
                    r"\s+",
                    " ",
                    match.group("requirement").upper(),
                )
                return f"ESRS {standard} {requirement}"

            match = self._ESRS_STANDARD_PATTERN.search(text)

            if match:
                return f"ESRS {match.group('standard').upper()}"

        if standard_type == StandardType.SASB:
            match = self._SASB_CODE_PATTERN.search(text)

            if match:
                return match.group("code").upper()

        if standard_type == StandardType.SDG:
            match = self._SDG_PATTERN.search(text)

            if match:
                return f"SDG {match.group('number')}"

        if standard_type == StandardType.TCFD:
            return self._extract_tcfd_code(text)

        match = self._GRI_CODE_PATTERN.search(text)

        if match:
            code = match.group("code")
            code = re.sub(
                r"(?<=\d)(?=\d$)",
                "-",
                code,
            ) if "-" not in code else code

            return f"GRI {code}"

        if standard_type == StandardType.UNKNOWN:
            for candidate_type in (
                StandardType.ESRS,
                StandardType.SASB,
                StandardType.SDG,
                StandardType.TCFD,
                StandardType.GRI,
            ):
                code = self._extract_code(text, candidate_type)

                if code:
                    return code

        return None

    @staticmethod
    def _extract_tcfd_code(text: str) -> str | None:
        normalized = text.casefold()

        mapping = {
            "governance": "TCFD Governance",
            "strategy": "TCFD Strategy",
            "risk management": "TCFD Risk Management",
            "metrics and targets": "TCFD Metrics and Targets",
            "metrics & targets": "TCFD Metrics and Targets",
        }

        for phrase, code in mapping.items():
            if phrase in normalized:
                return code

        if "tcfd" in normalized:
            return "TCFD"

        return None

    def _extract_title(
        self,
        cells: Sequence[str],
        *,
        disclosure_code: str,
        header_map: Mapping[str, int] | None,
    ) -> str | None:
        if header_map:
            title_index = header_map.get("title")

            if (
                title_index is not None
                and title_index < len(cells)
            ):
                title = cells[title_index].strip()

                if title and disclosure_code.casefold() not in title.casefold():
                    return title

        candidates: list[str] = []

        for cell in cells:
            if not cell:
                continue

            cleaned = self._remove_disclosure_code(
                cell,
                disclosure_code,
            )

            if not cleaned:
                continue

            if self._looks_like_page_reference(cleaned):
                continue

            if self._contains_omission_term(cleaned):
                continue

            if len(cleaned) < 4:
                continue

            candidates.append(cleaned)

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda value: (
                self._alphabetic_ratio(value),
                len(value),
            ),
        )

    def _extract_page_references(
        self,
        cells: Sequence[str],
        *,
        header_map: Mapping[str, int] | None,
        disclosure_code: str,
    ) -> tuple[int, ...]:
        candidate_cells: list[str] = []

        if header_map:
            page_index = header_map.get("page")

            if (
                page_index is not None
                and page_index < len(cells)
            ):
                candidate_cells.append(cells[page_index])

        if not candidate_cells:
            candidate_cells.extend(
                cell
                for cell in cells
                if self._looks_like_page_reference(cell)
            )

        pages: set[int] = set()

        for cell in candidate_cells:
            cleaned = self._remove_disclosure_code(
                cell,
                disclosure_code,
            )

            for match in self._PAGE_RANGE_PATTERN.finditer(cleaned):
                start = int(match.group("start"))
                end_text = match.group("end")
                end = int(end_text) if end_text else start

                if (
                    start <= 0
                    or end <= 0
                    or start > self.maximum_page_number
                    or end > self.maximum_page_number
                ):
                    continue

                if end < start:
                    start, end = end, start

                if self.expand_page_ranges:
                    if end - start > 100:
                        continue

                    pages.update(range(start, end + 1))
                else:
                    pages.add(start)
                    pages.add(end)

        return tuple(sorted(pages))

    def _extract_section_references(
        self,
        cells: Sequence[str],
        *,
        title: str | None,
        disclosure_code: str,
        header_map: Mapping[str, int] | None,
    ) -> tuple[str, ...]:
        references: list[str] = []

        if header_map:
            location_index = header_map.get("location")

            if (
                location_index is not None
                and location_index < len(cells)
            ):
                value = cells[location_index].strip()

                if value and not self._looks_like_page_reference(value):
                    references.append(value)

        if not references:
            for cell in cells:
                cleaned = self._remove_disclosure_code(
                    cell,
                    disclosure_code,
                )

                if not cleaned:
                    continue

                if title and cleaned == title:
                    continue

                if self._looks_like_page_reference(cleaned):
                    continue

                if self._contains_omission_term(cleaned):
                    continue

                if self._alphabetic_ratio(cleaned) < 0.35:
                    continue

                if len(cleaned) < 5:
                    continue

                references.append(cleaned)

        return tuple(dict.fromkeys(references))

    def _extract_omission(
        self,
        cells: Sequence[str],
        *,
        header_map: Mapping[str, int] | None,
    ) -> str | None:
        if header_map:
            omission_index = header_map.get("omission")

            if (
                omission_index is not None
                and omission_index < len(cells)
            ):
                value = cells[omission_index].strip()

                if value:
                    return value

        omission_cells = [
            cell
            for cell in cells
            if self._contains_omission_term(cell)
        ]

        if not omission_cells:
            return None

        return " | ".join(dict.fromkeys(omission_cells))

    def _detect_header(
        self,
        rows: Sequence[Sequence[str]],
    ) -> tuple[dict[str, int], int]:
        search_limit = min(len(rows), 5)

        for index in range(search_limit):
            row = rows[index]

            if not self._is_header_row(row):
                continue

            header_map: dict[str, int] = {}

            for cell_index, cell in enumerate(row):
                normalized = self._normalize_header(cell)

                if self._is_code_header(normalized):
                    header_map.setdefault("code", cell_index)
                elif self._is_title_header(normalized):
                    header_map.setdefault("title", cell_index)
                elif self._is_page_header(normalized):
                    header_map.setdefault("page", cell_index)
                elif self._is_location_header(normalized):
                    header_map.setdefault("location", cell_index)
                elif self._is_omission_header(normalized):
                    header_map.setdefault("omission", cell_index)

            return header_map, index + 1

        return {}, 0

    def _is_header_row(
        self,
        cells: Sequence[str],
    ) -> bool:
        normalized_cells = {
            self._normalize_header(cell)
            for cell in cells
            if cell
        }

        matches = sum(
            any(
                header_term in cell
                for header_term in self._HEADER_TERMS
            )
            for cell in normalized_cells
        )

        has_code = any(
            self._extract_code(
                cell,
                self.detect_standard_type(cell),
            )
            is not None
            for cell in cells
        )

        return matches >= 2 and not has_code

    @staticmethod
    def _is_code_header(value: str) -> bool:
        return any(
            term in value
            for term in (
                "code",
                "disclosure number",
                "disclosure code",
                "gri disclosure",
                "gri standard",
                "requirement",
                "açıklama kodu",
                "standart kodu",
            )
        )

    @staticmethod
    def _is_title_header(value: str) -> bool:
        return any(
            term in value
            for term in (
                "title",
                "disclosure title",
                "description",
                "topic",
                "açıklama",
                "başlık",
                "konu",
            )
        )

    @staticmethod
    def _is_page_header(value: str) -> bool:
        return any(
            term in value
            for term in (
                "page",
                "page number",
                "page reference",
                "sayfa",
                "sayfa numarası",
            )
        )

    @staticmethod
    def _is_location_header(value: str) -> bool:
        return any(
            term in value
            for term in (
                "location",
                "reference",
                "section",
                "report section",
                "konum",
                "referans",
                "bölüm",
            )
        )

    @staticmethod
    def _is_omission_header(value: str) -> bool:
        return any(
            term in value
            for term in (
                "omission",
                "reason for omission",
                "hariç tutma",
                "eksiklik",
                "raporlanmama",
            )
        )

    def _looks_like_page_reference(self, value: str) -> bool:
        normalized = value.casefold()

        if any(
            token in normalized
            for token in ("page", "pages", "sayfa", "pp.")
        ):
            return bool(self._PAGE_RANGE_PATTERN.search(value))

        alphabetic_count = sum(
            character.isalpha()
            for character in value
        )

        numeric_count = sum(
            character.isdigit()
            for character in value
        )

        return (
            numeric_count > 0
            and alphabetic_count <= 3
            and bool(self._PAGE_RANGE_PATTERN.search(value))
        )

    def _contains_omission_term(self, value: str) -> bool:
        normalized = value.casefold()

        return any(
            term in normalized
            for term in self._OMISSION_TERMS
        )

    @staticmethod
    def _calculate_confidence(
        *,
        standard_type: StandardType,
        disclosure_code: str,
        title: str | None,
        page_references: Sequence[int],
        section_references: Sequence[str],
        omission: str | None,
    ) -> float:
        score = 0.0

        if standard_type != StandardType.UNKNOWN:
            score += 0.25

        if disclosure_code:
            score += 0.4

        if title:
            score += 0.15

        if page_references:
            score += 0.15

        if section_references:
            score += 0.05

        if omission:
            score += 0.05

        return round(min(score, 1.0), 4)

    @staticmethod
    def _deduplicate_rows(
        rows: Iterable[StandardIndexRow],
    ) -> list[StandardIndexRow]:
        deduplicated: dict[
            tuple[StandardType, str, tuple[int, ...]],
            StandardIndexRow,
        ] = {}

        for row in rows:
            key = (
                row.standard_type,
                row.disclosure_code,
                row.page_references,
            )

            existing = deduplicated.get(key)

            if (
                existing is None
                or row.confidence > existing.confidence
            ):
                deduplicated[key] = row

        return sorted(
            deduplicated.values(),
            key=lambda row: (
                row.standard_type.value,
                row.disclosure_code,
                row.page_references,
            ),
        )

    @staticmethod
    def _normalize_input_row(
        row: Sequence[Any] | Mapping[str, Any],
    ) -> tuple[str, ...]:
        if isinstance(row, Mapping):
            return tuple(
                StandardIndexParser._normalize_text(value)
                for value in row.values()
            )

        if isinstance(row, (str, bytes)):
            return (
                StandardIndexParser._normalize_text(row),
            )

        return tuple(
            StandardIndexParser._normalize_text(value)
            for value in row
        )

    @staticmethod
    def _split_text_line(line: str) -> tuple[str, ...]:
        normalized = StandardIndexParser._normalize_text(line)

        if "\t" in normalized:
            parts = normalized.split("\t")
        elif "|" in normalized:
            parts = normalized.split("|")
        elif re.search(r"\s{2,}", normalized):
            parts = re.split(r"\s{2,}", normalized)
        else:
            parts = (normalized,)

        return tuple(
            part.strip()
            for part in parts
            if part.strip()
        )

    @staticmethod
    def _remove_disclosure_code(
        value: str,
        disclosure_code: str,
    ) -> str:
        cleaned = value

        code_variants = {
            disclosure_code,
            disclosure_code.replace("GRI ", ""),
            disclosure_code.replace("ESRS ", ""),
        }

        for variant in sorted(
            code_variants,
            key=len,
            reverse=True,
        ):
            cleaned = re.sub(
                re.escape(variant),
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

        return " ".join(cleaned.split()).strip(" |,;:-")

    @staticmethod
    def _alphabetic_ratio(value: str) -> float:
        non_space_characters = [
            character
            for character in value
            if not character.isspace()
        ]

        if not non_space_characters:
            return 0.0

        alphabetic_count = sum(
            character.isalpha()
            for character in non_space_characters
        )

        return alphabetic_count / len(non_space_characters)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""

        return " ".join(str(value).replace("\n", " ").split())

    @staticmethod
    def _normalize_header(value: str) -> str:
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

        normalized = value.translate(
            translation_table,
        ).casefold()

        normalized = re.sub(
            r"[^\w\s]",
            " ",
            normalized,
        )

        return " ".join(normalized.split())

    @staticmethod
    def _normalize_standard_type(
        value: StandardType | str | None,
    ) -> StandardType | None:
        if value is None:
            return None

        if isinstance(value, StandardType):
            return value

        normalized = str(value).strip().casefold()

        for member in StandardType:
            if normalized in {
                member.value.casefold(),
                member.name.casefold(),
            }:
                return member

        raise ValueError(
            f"Geçersiz standart türü: {value!r}",
        )

    @staticmethod
    def _normalize_code_for_comparison(
        value: str,
    ) -> str:
        return re.sub(
            r"[\s\-–—_:]+",
            "",
            value,
        ).casefold()


__all__ = [
    "StandardIndexParseResult",
    "StandardIndexParser",
    "StandardIndexParserError",
    "StandardIndexRow",
    "StandardType",
]