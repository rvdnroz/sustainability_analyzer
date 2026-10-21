from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from models.chunk import Chunk


_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

_STANDARD_PATTERNS: dict[str, re.Pattern[str]] = {
    "GRI": re.compile(
        r"\bGRI(?:\s+(?:STANDARD|STANDARDS|INDEX))?"
        r"(?:\s*[-:]?\s*\d{1,3}(?:-\d{1,3})?)?\b",
        re.IGNORECASE,
    ),
    "ESRS": re.compile(
        r"\bESRS(?:\s+[ESG]\d{1,2})?\b",
        re.IGNORECASE,
    ),
    "SASB": re.compile(
        r"\bSASB\b",
        re.IGNORECASE,
    ),
    "TCFD": re.compile(
        r"\bTCFD\b",
        re.IGNORECASE,
    ),
    "CDP": re.compile(
        r"\bCDP\b",
        re.IGNORECASE,
    ),
    "UNGC": re.compile(
        r"\b(?:UNGC|UNITED\s+NATIONS\s+GLOBAL\s+COMPACT)\b",
        re.IGNORECASE,
    ),
    "SDG": re.compile(
        r"\b(?:SDG|SUSTAINABLE\s+DEVELOPMENT\s+GOAL)"
        r"(?:S)?(?:\s+\d{1,2})?\b",
        re.IGNORECASE,
    ),
}


@dataclass(slots=True, frozen=True)
class MetadataContext:
    report_id: str | None = None
    company_name: str | None = None
    report_year: int | None = None

    page_number: int | None = None
    page_index: int | None = None

    section_title: str | None = None
    subsection_title: str | None = None

    source_path: str | Path | None = None
    source_url: str | None = None
    language: str | None = None

    standards: Sequence[str] = ()
    extra: Mapping[str, Any] | None = None


class MetadataEnrichmentError(RuntimeError):
    """Chunk metadata zenginleştirme işlemi başarısız olduğunda yükseltilir."""


class MetadataEnricher:
    """
    Metin chunk'larına retrieval ve izlenebilirlik için gerekli metadata'yı ekler.

    Eklenen temel alanlar:
    - rapor, şirket ve rapor yılı
    - sayfa ve bölüm bilgileri
    - kategori
    - metin içerisinde geçen yıllar
    - GRI, ESRS ve diğer sürdürülebilirlik standartları
    - kaynak dosya/URL bilgileri
    - chunk komşuluk ve sıra bilgileri
    - oluşturulma zamanı
    """

    def __init__(
        self,
        *,
        detect_years: bool = True,
        detect_standards: bool = True,
        overwrite_existing: bool = False,
    ) -> None:
        self.detect_years = detect_years
        self.detect_standards = detect_standards
        self.overwrite_existing = overwrite_existing

    def enrich(
        self,
        chunk: Chunk,
        *,
        context: MetadataContext | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Chunk:
        if chunk is None:
            raise MetadataEnrichmentError("Zenginleştirilecek chunk None olamaz.")

        context = context or MetadataContext()
        text = self._get_chunk_text(chunk)

        current_metadata = self._get_current_metadata(chunk)
        generated_metadata = self._build_metadata(
            chunk=chunk,
            text=text,
            context=context,
        )

        if context.extra:
            generated_metadata.update(
                self._clean_mapping(context.extra),
            )

        if extra_metadata:
            generated_metadata.update(
                self._clean_mapping(extra_metadata),
            )

        merged_metadata = self._merge_metadata(
            current=current_metadata,
            generated=generated_metadata,
        )

        return self._update_chunk_metadata(chunk, merged_metadata)

    def enrich_many(
        self,
        chunks: Iterable[Chunk],
        *,
        context: MetadataContext | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        enriched_chunks: list[Chunk] = []
        total_chunks = len(chunk_list)

        for position, chunk in enumerate(chunk_list):
            sequence_metadata = {
                "chunk_position": position,
                "chunk_number": position + 1,
                "total_chunks": total_chunks,
                "is_first_chunk": position == 0,
                "is_last_chunk": position == total_chunks - 1,
            }

            if position > 0:
                previous_id = self._get_chunk_identifier(
                    chunk_list[position - 1],
                )
                if previous_id is not None:
                    sequence_metadata["previous_chunk_id"] = previous_id

            if position < total_chunks - 1:
                next_id = self._get_chunk_identifier(
                    chunk_list[position + 1],
                )
                if next_id is not None:
                    sequence_metadata["next_chunk_id"] = next_id

            if extra_metadata:
                sequence_metadata.update(extra_metadata)

            enriched_chunks.append(
                self.enrich(
                    chunk,
                    context=context,
                    extra_metadata=sequence_metadata,
                ),
            )

        return enriched_chunks

    def _build_metadata(
        self,
        *,
        chunk: Chunk,
        text: str,
        context: MetadataContext,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "enriched_at": datetime.now(UTC).isoformat(),
            "character_count": len(text),
            "word_count": len(text.split()),
        }

        self._add_if_present(metadata, "report_id", context.report_id)
        self._add_if_present(metadata, "company_name", context.company_name)
        self._add_if_present(metadata, "report_year", context.report_year)

        self._add_if_present(metadata, "page_number", context.page_number)
        self._add_if_present(metadata, "page_index", context.page_index)

        self._add_if_present(
            metadata,
            "section_title",
            context.section_title,
        )
        self._add_if_present(
            metadata,
            "subsection_title",
            context.subsection_title,
        )

        self._add_if_present(metadata, "source_url", context.source_url)
        self._add_if_present(metadata, "language", context.language)

        if context.source_path is not None:
            source_path = Path(context.source_path)
            metadata["source_path"] = str(source_path)
            metadata["source_file_name"] = source_path.name
            metadata["source_file_suffix"] = source_path.suffix.lower()

        chunk_id = self._get_chunk_identifier(chunk)
        self._add_if_present(metadata, "chunk_id", chunk_id)

        category = self._extract_attribute_value(chunk, "category")
        self._add_if_present(metadata, "category", category)

        section_title = self._extract_first_present_attribute(
            chunk,
            ("section_title", "section", "heading"),
        )
        self._add_if_present(
            metadata,
            "chunk_section_title",
            section_title,
        )

        page_number = self._extract_first_present_attribute(
            chunk,
            ("page_number", "page", "source_page"),
        )
        self._add_if_present(
            metadata,
            "chunk_page_number",
            page_number,
        )

        token_count = self._extract_first_present_attribute(
            chunk,
            ("token_count", "tokens"),
        )
        self._add_if_present(metadata, "token_count", token_count)

        content_hash = self._extract_first_present_attribute(
            chunk,
            ("content_hash", "text_hash", "hash"),
        )
        self._add_if_present(metadata, "content_hash", content_hash)

        detected_years: list[int] = []
        if self.detect_years:
            detected_years = self._extract_years(text)

        if context.report_year is not None:
            detected_years = sorted(
                set(detected_years) | {context.report_year},
            )

        if detected_years:
            metadata["mentioned_years"] = detected_years
            metadata["contains_report_year"] = (
                context.report_year in detected_years
                if context.report_year is not None
                else False
            )

        standards = {
            standard.strip()
            for standard in context.standards
            if isinstance(standard, str) and standard.strip()
        }

        if self.detect_standards:
            standards.update(self._extract_standards(text))

        chunk_standards = self._extract_first_present_attribute(
            chunk,
            ("standards", "standard_codes", "standard_references"),
        )
        standards.update(self._normalize_string_collection(chunk_standards))

        if standards:
            metadata["standards"] = sorted(standards)

        return self._clean_mapping(metadata)

    def _merge_metadata(
        self,
        *,
        current: Mapping[str, Any],
        generated: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.overwrite_existing:
            merged = dict(current)
            merged.update(generated)
            return self._clean_mapping(merged)

        merged = dict(generated)
        merged.update(current)
        return self._clean_mapping(merged)

    @staticmethod
    def _get_chunk_text(chunk: Chunk) -> str:
        for attribute_name in ("text", "content", "chunk_text"):
            value = getattr(chunk, attribute_name, None)
            if isinstance(value, str):
                return value.strip()

        raise MetadataEnrichmentError(
            "Chunk üzerinde text, content veya chunk_text alanı bulunamadı.",
        )

    @staticmethod
    def _get_current_metadata(chunk: Chunk) -> dict[str, Any]:
        metadata = getattr(chunk, "metadata", None)

        if metadata is None:
            return {}

        if isinstance(metadata, Mapping):
            return dict(metadata)

        if hasattr(metadata, "model_dump"):
            return dict(metadata.model_dump(exclude_none=True))

        if hasattr(metadata, "dict"):
            return dict(metadata.dict(exclude_none=True))

        raise MetadataEnrichmentError(
            "Chunk metadata alanı mapping veya model türünde olmalıdır.",
        )

    @staticmethod
    def _update_chunk_metadata(
        chunk: Chunk,
        metadata: Mapping[str, Any],
    ) -> Chunk:
        cleaned_metadata = dict(metadata)

        if hasattr(chunk, "model_copy"):
            try:
                return chunk.model_copy(
                    update={"metadata": cleaned_metadata},
                    deep=True,
                )
            except (TypeError, ValueError) as exc:
                raise MetadataEnrichmentError(
                    "Pydantic model_copy ile chunk güncellenemedi.",
                ) from exc

        if hasattr(chunk, "copy"):
            try:
                return chunk.copy(
                    update={"metadata": cleaned_metadata},
                    deep=True,
                )
            except (TypeError, ValueError):
                pass

        if is_dataclass(chunk):
            field_names = {field.name for field in fields(chunk)}
            if "metadata" not in field_names:
                raise MetadataEnrichmentError(
                    "Chunk dataclass modelinde metadata alanı bulunamadı.",
                )

            try:
                return replace(
                    chunk,
                    metadata=cleaned_metadata,
                )
            except (TypeError, ValueError) as exc:
                raise MetadataEnrichmentError(
                    "Dataclass chunk metadata alanı güncellenemedi.",
                ) from exc

        copied_chunk = deepcopy(chunk)

        try:
            setattr(copied_chunk, "metadata", cleaned_metadata)
        except (AttributeError, TypeError) as exc:
            raise MetadataEnrichmentError(
                "Chunk metadata alanı güncellenemedi.",
            ) from exc

        return copied_chunk

    @staticmethod
    def _get_chunk_identifier(chunk: Chunk) -> str | None:
        for attribute_name in ("chunk_id", "id", "identifier"):
            value = getattr(chunk, attribute_name, None)
            if value is not None:
                return str(value)

        return None

    @classmethod
    def _extract_first_present_attribute(
        cls,
        obj: Any,
        attribute_names: Sequence[str],
    ) -> Any:
        for attribute_name in attribute_names:
            value = getattr(obj, attribute_name, None)
            if value is not None:
                return cls._normalize_value(value)

        return None

    @classmethod
    def _extract_attribute_value(
        cls,
        obj: Any,
        attribute_name: str,
    ) -> Any:
        value = getattr(obj, attribute_name, None)
        return cls._normalize_value(value)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return enum_value

        if isinstance(value, Path):
            return str(value)

        return value

    @staticmethod
    def _extract_years(text: str) -> list[int]:
        years = {
            int(match.group(0))
            for match in _YEAR_PATTERN.finditer(text)
        }
        return sorted(years)

    @staticmethod
    def _extract_standards(text: str) -> set[str]:
        standards: set[str] = set()

        for standard_name, pattern in _STANDARD_PATTERNS.items():
            matches = pattern.finditer(text)

            for match in matches:
                normalized_match = " ".join(
                    match.group(0).upper().split(),
                )

                if standard_name in {"GRI", "ESRS", "SDG"}:
                    standards.add(normalized_match)
                else:
                    standards.add(standard_name)

        return standards

    @classmethod
    def _normalize_string_collection(
        cls,
        value: Any,
    ) -> set[str]:
        if value is None:
            return set()

        if isinstance(value, str):
            cleaned = value.strip()
            return {cleaned} if cleaned else set()

        if isinstance(value, Mapping):
            return {
                str(key).strip()
                for key in value
                if str(key).strip()
            }

        if isinstance(value, Sequence):
            normalized: set[str] = set()

            for item in value:
                item_value = cls._normalize_value(item)
                if item_value is None:
                    continue

                cleaned = str(item_value).strip()
                if cleaned:
                    normalized.add(cleaned)

            return normalized

        cleaned = str(cls._normalize_value(value)).strip()
        return {cleaned} if cleaned else set()

    @classmethod
    def _clean_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}

        for key, value in mapping.items():
            if value is None:
                continue

            normalized_value = cls._to_serializable(value)

            if normalized_value in ("", [], {}, ()):
                continue

            cleaned[str(key)] = normalized_value

        return cleaned

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return cls._to_serializable(enum_value)

        if isinstance(value, Mapping):
            return {
                str(key): cls._to_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                cls._to_serializable(item)
                for item in value
            ]

        if hasattr(value, "model_dump"):
            return cls._to_serializable(
                value.model_dump(exclude_none=True),
            )

        if hasattr(value, "dict"):
            return cls._to_serializable(
                value.dict(exclude_none=True),
            )

        return value

    @staticmethod
    def _add_if_present(
        target: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if value is None:
            return

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return

        target[key] = value


__all__ = [
    "MetadataContext",
    "MetadataEnricher",
    "MetadataEnrichmentError",
]
